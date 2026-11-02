import asyncio
import pickle
import weakref

from types import SimpleNamespace

import numpy as np
import pytest

from _fakes import FakeTokenizer

from nexus.gateway.backend import BackendResult
from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.gateway.processor import Processor
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore, build_trajectory_record_store
from nexus.gateway.types import PromptRender


class _TokenBackend:
    async def create_completion(self, *_args, **_kwargs):
        return BackendResult(
            token_ids=[20, 21],
            log_probs=[-0.1, -0.2],
            extra_fields={
                'resp_topk_logprob_indice': [[3, 4], [5, 6]],
                'resp_topk_logprob_value': [[-0.3, -0.4], [-0.5, -0.6]],
            },
        )


class _RoutingTokenBackend(_TokenBackend):
    """Return one routing row per requested stream position, holding that position."""

    async def create_completion(self, *args, **kwargs):
        result = await super().create_completion(*args, **kwargs)
        routing_end = len(kwargs['prompt_ids']) + len(result.token_ids) - 1
        result.routed_experts = np.arange(kwargs['routed_experts_prompt_start'], routing_end).reshape(-1, 1)
        return result


class _FailingStore(InMemoryTrajectoryRecordStore):
    async def put_many(self, records):
        del records
        raise RuntimeError('store unavailable')


class _DelayedStore(InMemoryTrajectoryRecordStore):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        self.published_refs = []
        self.release_count = 0

    async def put_many(self, records):
        self.started.set()
        await self.proceed.wait()
        refs = await super().put_many(records)
        self.published_refs.extend(refs)
        return refs

    async def release_many(self, refs):
        self.release_count += 1
        await super().release_many(refs)


async def test_put_wait_metric_survives_session_and_runner_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    import nexus.gateway.session as session_module

    store = _DelayedStore('wait-metric')
    session = GatewaySession(
        'wait-metric', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
    )
    await session.create_completion([{'role': 'user', 'content': 'question'}])
    await asyncio.wait_for(store.started.wait(), timeout=1)
    wait_started = asyncio.Event()
    clock_readings = iter([10.0, 10.25])

    def clock() -> float:
        wait_started.set()
        return next(clock_readings)

    monkeypatch.setattr(session_module, 'time', SimpleNamespace(perf_counter=clock, time=session_module.time.time))
    finalization = asyncio.create_task(session.finish_pending_publications())
    await asyncio.wait_for(wait_started.wait(), timeout=1)
    assert not finalization.done(), 'finalization must wait for the pending put'
    store.proceed.set()
    await asyncio.wait_for(finalization, timeout=1)
    await session.finish_pending_publications()
    trajectory = session.end(metrics={'caller_metric': 7.0})
    assert trajectory.metrics['trajectory_store_put_wait_time'] == 0.25
    finalized = NexusRunTracker().finalize_trajectory(trajectory, 'wait-metric', termination_reason='ENV_DONE')
    assert finalized is not None
    assert finalized.metrics['trajectory_store_put_wait_time'] == 0.25
    assert finalized.metrics['caller_metric'] == 7.0


class _FailAfterFirstPutStore(InMemoryTrajectoryRecordStore):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.put_count = 0
        self.published_refs = []

    async def put_many(self, records):
        self.put_count += 1
        if self.put_count == 2:
            raise RuntimeError('store unavailable')
        refs = await super().put_many(records)
        self.published_refs.extend(refs)
        return refs


class _SerializedDelayedStore(InMemoryTrajectoryRecordStore):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self.proceed = [asyncio.Event(), asyncio.Event()]
        self.record_refs = []
        self.payload_refs = []

    async def put_many(self, records):
        index = len(self.record_refs)
        self.record_refs.append([weakref.ref(record) for record in records])
        self.payload_refs.append(
            [weakref.ref(record.routed_experts) for record in records if record.routed_experts is not None]
        )
        await self.proceed[index].wait()
        return await super().put_many(pickle.loads(pickle.dumps(records)))


class _ConcurrentFailureStore(InMemoryTrajectoryRecordStore):
    def __init__(self, namespace):
        super().__init__(namespace)
        self._second_put_started = asyncio.Event()
        self._first_put_failed = asyncio.Event()
        self._put_count = 0
        self.published_refs = []

    async def put_many(self, records):
        self._put_count += 1
        if self._put_count == 1:
            await self._second_put_started.wait()
            self._first_put_failed.set()
            raise RuntimeError('store unavailable')
        self._second_put_started.set()
        await self._first_put_failed.wait()
        refs = await super().put_many(records)
        self.published_refs.extend(refs)
        return refs


class _SizedTokenBackend:
    def __init__(self, token_count):
        self._token_count = token_count

    async def create_completion(self, *_args, **_kwargs):
        return BackendResult(token_ids=[65] * self._token_count)


class _StartRecordingRoutingBackend(_RoutingTokenBackend):
    """Record each request's routing start so a test can assert where capture resumed."""

    def __init__(self) -> None:
        self.starts: list[int] = []

    async def create_completion(self, *args, **kwargs):
        self.starts.append(kwargs['routed_experts_prompt_start'])
        return await super().create_completion(*args, **kwargs)


class TestSegmentRoutingProvenance:
    """Every segment derives its whole routing, prompt span included, from its own inferences."""

    @staticmethod
    async def _segment_count(store: InMemoryTrajectoryRecordStore, trajectory) -> int:
        node_lists = trajectory.to_segment_node_lists()
        for node_list in node_lists:
            payloads = await store.get_many([node.token_account.payload_ref for node in node_list.response_nodes])
            routing = np.concatenate([payload['routed_experts'] for payload in payloads])
            last_response_index = node_list.nodes.index(node_list.last_response_node)
            token_count = sum(len(node.token_account.token_ids) for node in node_list.nodes[: last_response_index + 1])
            np.testing.assert_array_equal(
                routing[:, 0],
                np.arange(token_count - 1),
                err_msg=f'segment {node_list.segment_id} must hold its own contiguous routing',
            )
        return len(node_lists)

    @pytest.mark.parametrize('scaffold_policy', ['generic', 'nexus_main_agent', 'codex', 'hycode'])
    async def test_summary_inference_reusing_the_prefix_recaptures_its_prompt_routing(
        self, scaffold_policy: str
    ) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('summary-inference-routing')
        session = GatewaySession(
            'summary-inference-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy=scaffold_policy,
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'user', 'content': 'please summarize'}])
        await session.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
        await session.finish_pending_publications()

        assert backend.starts[1] > 0, 'A plain continuation captures only its delta'
        assert backend.starts[2] == 0, 'A summary inference opens a segment, so it recaptures the prefix'
        assert await self._segment_count(store, session.end()) == 2

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason='Resume markers on an unchanged prefix are deferred')
    async def test_compaction_marker_at_the_request_edge_recaptures_its_prompt_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('edge-compaction-routing')
        session = GatewaySession(
            'edge-compaction-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='nexus_main_agent',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        reply = await session.create_completion(messages)
        summary = {'role': 'user', 'content': 'summary so far', '_is_compacted_summary': True}
        messages.extend([reply.message, summary])
        await session.create_completion(messages, request_meta={'invocation_kind': 'compaction_resume'})
        await session.finish_pending_publications()

        assert backend.starts[2] == 0, 'A compaction anchored on the message just added opens a segment'
        assert await self._segment_count(store, session.end()) == 2

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason='Resume markers on an unchanged prefix are deferred')
    async def test_compaction_marker_behind_an_observation_recaptures_its_prompt_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('behind-compaction-routing')
        session = GatewaySession(
            'behind-compaction-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='nexus_main_agent',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        summary = {'role': 'user', 'content': 'summary so far', '_is_compacted_summary': True}
        messages.extend([reply.message, summary, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'next', 'tool_call_id': 'next'}])
        await session.create_completion(messages)
        await session.finish_pending_publications()

        assert backend.starts[1] == 0, 'A compaction behind a newer message still opens a segment'
        assert await self._segment_count(store, session.end()) == 2

    async def test_llm_sibling_on_a_shared_tool_node_recaptures_its_prompt_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('sibling-routing')
        session = GatewaySession(
            'sibling-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='generic',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        await session.create_completion(messages)
        retry = await session.create_completion(messages)
        messages.extend([retry.message, {'role': 'tool', 'content': 'next', 'tool_call_id': 'next'}])
        await session.create_completion(messages)
        await session.finish_pending_publications()

        assert backend.starts[1] > 0, 'The first response under a tool Node continues the recorded stream'
        assert backend.starts[2] == 0, 'A retry that keeps every response lands on its own segment'
        assert await self._segment_count(store, session.end()) == 2

    async def test_branch_leaving_a_node_that_already_has_an_observation_recaptures_its_prompt_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('divergent-branch-routing')
        session = GatewaySession(
            'divergent-branch-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='hycode',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        first_branch = [*messages, reply.message, {'role': 'tool', 'content': 'result A', 'tool_call_id': 'call'}]
        await session.create_completion(first_branch)
        second_branch = [*messages, reply.message, {'role': 'tool', 'content': 'result B', 'tool_call_id': 'call'}]
        retry = await session.create_completion(second_branch)
        second_branch.extend([retry.message, {'role': 'tool', 'content': 'next', 'tool_call_id': 'next'}])
        await session.create_completion(second_branch)
        await session.finish_pending_publications()

        assert backend.starts[2] == 0, 'A branch leaving a Node that already continued opens its own segment'
        assert await self._segment_count(store, session.end()) == 2

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason='Explicit boundary replay is deferred')
    async def test_explicitly_finalized_segment_recaptures_its_prompt_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('finalized-segment-routing')
        session = GatewaySession(
            'finalized-segment-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='generic',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        reply = await session.create_completion(messages)
        session.segment_assembler.finalize_segment()
        messages.extend([reply.message, {'role': 'tool', 'content': 'next', 'tool_call_id': 'next'}])
        await session.create_completion(messages)
        await session.finish_pending_publications()

        assert backend.starts[2] == 0, 'A scaffold that finalized the segment makes the next request recapture'
        assert await self._segment_count(store, session.end()) == 2

    async def test_abandoned_sibling_shares_its_segment_and_recaptures_prefix_routing(self) -> None:
        backend = _StartRecordingRoutingBackend()
        store = InMemoryTrajectoryRecordStore('abandoned-sibling-routing')
        session = GatewaySession(
            'abandoned-sibling-routing',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='hycode',
        )
        messages = [{'role': 'user', 'content': 'question'}]
        reply = await session.create_completion(messages)
        messages.extend([reply.message, {'role': 'tool', 'content': 'result', 'tool_call_id': 'first'}])
        await session.create_completion(messages)
        retry = await session.create_completion(messages)
        retry_id = max(session.segment_assembler.trajectory.nodes)
        messages.extend([retry.message, {'role': 'tool', 'content': 'next', 'tool_call_id': 'next'}])
        await session.create_completion(messages)
        await session.finish_pending_publications()

        assert backend.starts[2] == 0, 'A retry under an existing response recaptures its prompt routing'
        assert backend.starts[3] > 0, 'A linear continuation after the retry captures only its delta'
        (node_list,) = session.end().to_segment_node_lists()
        retry_node = next(node for node in node_list.response_nodes if node.node_id == retry_id)
        (payload,) = await store.get_many([retry_node.token_account.payload_ref])
        retry_index = node_list.nodes.index(retry_node)
        token_count = sum(len(node.token_account.token_ids) for node in node_list.nodes[: retry_index + 1])
        np.testing.assert_array_equal(
            payload['routed_experts'][:, 0],
            np.arange(token_count - 1),
            err_msg='The retry that replaces the abandoned sibling must hold its whole prompt routing',
        )


class TestSessionRecordPublication:
    @pytest.mark.parametrize('completed_put', [0, 1])
    async def test_successful_put_releases_payload_before_other_puts_and_finalization(self, completed_put: int) -> None:
        class MetadataBackend(_RoutingTokenBackend):
            async def create_completion(self, *args, **kwargs):
                result = await super().create_completion(*args, **kwargs)
                result.extra_fields['retained_metadata'] = 'kept'
                return result

        store = _SerializedDelayedStore('payload-lifetime')
        session = GatewaySession(
            'trajectory-1',
            Processor(MetadataBackend(), tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='generic',
        )
        try:
            completion = await session.create_completion([{'role': 'user', 'content': 'first'}])
            assert completion.token_ids == [20, 21]
            assert completion.log_probs is None
            assert completion.routed_experts is None
            assert completion.extra_fields == {'retained_metadata': 'kept'}
            await session.create_completion([{'role': 'user', 'content': 'second'}])
            await asyncio.sleep(0)
            assert len(store.record_refs) == 2
            assert all(ref() is not None for refs in store.record_refs for ref in refs)
            store.proceed[completed_put].set()
            await asyncio.sleep(0)

            # Completed tasks must not retain producer payloads until session finalization.
            assert all(ref() is None for ref in store.record_refs[completed_put])
            assert all(ref() is None for ref in store.payload_refs[completed_put])
            assert all(ref() is not None for ref in store.record_refs[1 - completed_put])
            assert all(ref() is not None for ref in store.payload_refs[1 - completed_put])
            accounts = [
                node.token_account
                for node in session.segment_assembler.trajectory.iter_nodes()
                if node.token_account is not None
            ]
            assert all(account.token_ids is not None for account in accounts)
        finally:
            for event in store.proceed:
                event.set()
            await session.finish_pending_publications()

        trajectory = session.end()
        refs = [node.token_account.payload_ref for node in trajectory.iter_nodes() if node.token_account is not None]
        payloads = await store.get_many(refs)
        assert [payload['token_ids'].tolist() for payload in payloads] == [
            account.token_ids.tolist() for account in accounts
        ]
        output_payload = next(payload for payload in payloads if payload.get('logprobs') is not None)
        np.testing.assert_array_equal(output_payload['logprobs'], np.array([-0.1, -0.2], dtype=np.float32))
        assert output_payload['topk_indices'].tolist() == [[3, 4], [5, 6]]
        np.testing.assert_array_equal(
            output_payload['topk_values'], np.array([[-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32)
        )
        first_output = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        path_token_count = len(first_output.token_account.token_ids) + len(
            trajectory.nodes[first_output.parent].token_account.token_ids
        )
        assert output_payload['routed_experts'].shape[0] == path_token_count - 1
        for account, payload in zip(accounts, payloads, strict=True):
            account.hydrate_store_payload(payload)
        trajectory.discard_inline_payload()
        assert all(
            account.token_ids is None
            and account.logprobs is None
            and account.topk_indices is None
            and account.topk_values is None
            and account.routed_experts is None
            and account.multi_modal_items is None
            for account in accounts
        )

    @pytest.mark.parametrize('cancel_finalization', [False, True])
    async def test_completion_returns_before_put_and_finalization_waits(self, cancel_finalization: bool) -> None:
        store = _DelayedStore('delayed')
        session = GatewaySession(
            'trajectory-1', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
        )
        await asyncio.wait_for(session.create_completion([{'role': 'user', 'content': 'question'}]), timeout=1)
        await asyncio.wait_for(store.started.wait(), timeout=1)
        with pytest.raises(RuntimeError, match='finish_pending_publications'):
            session.end()
        finalization = asyncio.create_task(session.finish_pending_publications())
        await asyncio.sleep(0)
        assert not finalization.done()
        with pytest.raises(RuntimeError, match='closing'):
            await session.create_completion([{'role': 'user', 'content': 'late'}])
        if cancel_finalization:
            finalization.cancel()
            await asyncio.sleep(0)
            assert not finalization.done()
        store.proceed.set()
        if cancel_finalization:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(finalization, timeout=1)
        else:
            await asyncio.wait_for(finalization, timeout=1)
        await session.finish_pending_publications()
        trajectory = session.end(termination_reason='TASK_TIMEOUT')
        refs = [node.token_account.payload_ref for node in trajectory.iter_nodes() if node.token_account is not None]
        assert await store.get_many(refs, fields=['token_ids'])
        assert store.release_count == 0

    @pytest.mark.parametrize('release_fails', [False, True])
    async def test_assembly_failure_waits_for_pending_put_before_releasing(
        self, monkeypatch, release_fails: bool
    ) -> None:
        store = _DelayedStore('discard')
        session = GatewaySession(
            'trajectory-1', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
        )
        await asyncio.wait_for(session.create_completion([{'role': 'user', 'content': 'first'}]), timeout=1)
        await asyncio.wait_for(store.started.wait(), timeout=1)

        def fail_assembly(*args, **kwargs):
            raise ValueError('invalid assembly')

        monkeypatch.setattr(session.segment_assembler, 'record_completion', fail_assembly)
        if release_fails:

            async def fail_release(refs):
                store.release_count += 1
                raise RuntimeError('release unavailable')

            monkeypatch.setattr(store, 'release_many', fail_release)
        with pytest.raises(ValueError, match='invalid assembly'):
            await session.create_completion([{'role': 'user', 'content': 'second'}])
        with pytest.raises(RuntimeError, match='failed trajectory publication'):
            await session.create_completion([{'role': 'user', 'content': 'third'}])
        finalization = asyncio.create_task(session.finish_pending_publications())
        await asyncio.sleep(0)
        assert not finalization.done()
        assert store.release_count == 0
        store.proceed.set()
        error_type = RuntimeError if release_fails else ValueError
        error_message = 'release unavailable' if release_fails else 'invalid assembly'
        for attempt in (finalization, session.finish_pending_publications()):
            with pytest.raises(error_type, match=error_message):
                await asyncio.wait_for(attempt, timeout=1)
        assert store.release_count == 1
        if release_fails:
            return
        assert await store.get_many(store.published_refs)
        assert all(
            node.token_account.payload_ref is None
            for node in session.segment_assembler.trajectory.iter_nodes()
            if node.token_account is not None
        )

    def test_assembler_returns_only_nodes_sealed_by_the_current_completion(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic', session_id='trajectory-1')
        messages = [{'role': 'user', 'content': 'question'}]
        render = PromptRender(
            text='UA',
            token_ids=np.array([10, 11], dtype=np.int32),
            templated_messages=['U'],
            message_token_ids=[np.array([10], dtype=np.int32)],
            generation_prompt='A',
            generation_prompt_ids=np.array([11], dtype=np.int32),
        )

        first = assembler.record_completion(
            messages,
            {'role': 'assistant', 'content': 'first'},
            token_training={'prompt_ids': [10, 11], 'raw_prompt_ids': [10, 11], 'response_ids': [20]},
            completion_text='first',
            prompt_render=render,
        )
        second = assembler.record_completion(
            messages,
            {'role': 'assistant', 'content': 'second'},
            token_training={'prompt_ids': [10, 11], 'raw_prompt_ids': [10, 11], 'response_ids': [21]},
            completion_text='second',
            prompt_render=render,
        )

        assert len(first) == 2
        assert len(second) == 1
        assert second[0].node_id not in {record.node_id for record in first}

    async def test_finalization_publishes_complete_node_records(self):
        store = InMemoryTrajectoryRecordStore('publication')
        session = GatewaySession(
            'trajectory-1',
            Processor(_RoutingTokenBackend(), tokenizer=FakeTokenizer(), enable_rollout_routing_replay=True),
            store,
            scaffold_policy='generic',
        )

        await session.create_completion([{'role': 'user', 'content': 'question'}])

        await session.finish_pending_publications()
        trajectory = session.end()
        accounts = [node.token_account for node in trajectory.iter_nodes() if node.token_account is not None]
        refs = [account.payload_ref for account in accounts]
        assert all(ref is not None for ref in refs)
        output = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        [record] = await store.get_many(
            [output.token_account.payload_ref],
            fields=['token_ids', 'logprobs', 'topk_indices', 'topk_values', 'routed_experts'],
        )
        routing = record.pop('routed_experts')
        np.testing.assert_equal(
            record,
            {
                'token_ids': output.token_account.token_ids,
                'logprobs': np.array([-0.1, -0.2], dtype=np.float32),
                'topk_indices': [[3, 4], [5, 6]],
                'topk_values': np.array([[-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32),
            },
        )
        input_token_count = len(trajectory.nodes[output.parent].token_account.token_ids)
        output_token_count = len(output.token_account.token_ids)
        np.testing.assert_array_equal(routing[:, 0], np.arange(input_token_count + output_token_count - 1))

    @pytest.mark.parametrize('training', [False, True])
    async def test_completed_trajectory_allows_inline_payload_release(self, training: bool) -> None:
        store = InMemoryTrajectoryRecordStore('hot-cache')
        session = GatewaySession(
            'trajectory-1', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
        )
        await session.create_completion([{'role': 'user', 'content': 'question'}])
        await session.finish_pending_publications()
        trajectory = session.end()
        if training:
            node_list = trajectory.to_segment_node_lists(chains='main')[0]
            trajectory.segment_training_data = trajectory.prepare_segment_training_data([(node_list, True)])
        refs = [node.token_account.payload_ref for node in trajectory.iter_nodes() if node.token_account is not None]

        trajectory.discard_inline_payload()

        assert all(
            node.token_account.token_ids is None for node in trajectory.iter_nodes() if node.token_account is not None
        )
        assert all(not node.original_message and node.templated_message is None for node in trajectory.iter_nodes())
        assert all(ref is not None for ref in refs)
        assert await store.get_many(refs, fields=['token_ids'])

    def test_running_trajectory_rejects_inline_payload_release(self) -> None:
        from nexus.gateway.trajectory import Trajectory

        with pytest.raises(RuntimeError, match='completed trajectory'):
            Trajectory(session_id='running', state='running').discard_inline_payload()

    @pytest.mark.parametrize('routing_enabled', [False, True])
    async def test_routing_metadata_matches_published_payload_and_survives_rpc(self, routing_enabled: bool) -> None:
        store = InMemoryTrajectoryRecordStore('routing-metadata')
        backend = _RoutingTokenBackend() if routing_enabled else _SizedTokenBackend(2)
        session = GatewaySession(
            'trajectory-1',
            Processor(backend, tokenizer=FakeTokenizer(), enable_rollout_routing_replay=routing_enabled),
            store,
            scaffold_policy='generic',
        )
        await session.create_completion([{'role': 'user', 'content': 'question'}])
        await session.finish_pending_publications()
        trajectory = session.end()
        for node in trajectory.iter_nodes():
            account = node.token_account
            if account is not None:
                payload = (await store.get_many([account.payload_ref]))[0]
                assert account.has_routed_experts == ('routed_experts' in payload)
        output = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        assert output.token_account.has_routed_experts is routing_enabled
        dump = trajectory.dump()
        restored = type(trajectory).from_dump(dump)
        assert restored.nodes[output.node_id].token_account.has_routed_experts is routing_enabled
        del dump['nodes'][0]['token_account']['has_routed_experts']
        with pytest.raises(KeyError, match='has_routed_experts'):
            type(trajectory).from_dump(dump)
        node_list = trajectory.to_segment_node_lists(chains='main')[0]
        trajectory.segment_training_data = trajectory.prepare_segment_training_data([(node_list, True)])
        trajectory.discard_inline_payload()
        restored = pickle.loads(pickle.dumps(trajectory))
        assert restored.nodes[output.node_id].token_account.has_routed_experts is routing_enabled
        assert restored.nodes[output.node_id].token_account.token_ids is None

    @pytest.mark.parametrize('backend', ['memory', 'ray', 'redis'])
    async def test_serialized_trajectory_retains_backend_payload_contract(self, backend: str, request) -> None:
        ray = None
        if backend == 'ray':
            ray = pytest.importorskip('ray')
            ray.init(num_cpus=2, ignore_reinit_error=True)

        async def compact_size(token_count: int) -> int:
            config = {'backend': backend, 'namespace': f'size-{token_count}'}
            if backend == 'redis':
                config.update(backend='redis', redis=request.getfixturevalue(f'{backend}_config'))
            store = build_trajectory_record_store(config)
            session = GatewaySession(
                'trajectory-1',
                Processor(_SizedTokenBackend(token_count), tokenizer=FakeTokenizer()),
                store,
                scaffold_policy='generic',
            )
            await session.create_completion([{'role': 'user', 'content': 'question'}])
            await session.finish_pending_publications()
            trajectory = session.end()
            node_list = trajectory.to_segment_node_lists(chains='main')[0]
            trajectory.segment_training_data = trajectory.prepare_segment_training_data([(node_list, True)])
            trajectory.discard_inline_payload()
            serialized = pickle.dumps(trajectory)
            restored = pickle.loads(serialized)
            refs = [
                node.token_account.payload_ref
                for node in restored.iter_nodes()
                if node.token_account is not None and node.token_account.payload_ref is not None
            ]
            output = next(node for node in restored.iter_nodes() if node.generated_by == 'llm')
            account = output.token_account
            [payload] = await store.get_many([account.payload_ref])
            assert (
                payload['token_ids'][account.gen_start : account.gen_start + account.gen_len].tolist()
                == [65] * token_count
            )
            await store.release_many(refs)
            return len(serialized)

        try:
            large, small = await compact_size(10_000), await compact_size(1)
            if backend != 'memory':
                assert large < small + 512
            else:
                assert large > small + 10_000
        finally:
            if ray is not None:
                ray.shutdown()

    async def test_store_failure_rejects_finalization(self):
        session = GatewaySession(
            'trajectory-1',
            Processor(_TokenBackend(), tokenizer=FakeTokenizer()),
            _FailingStore('publication-failure'),
            scaffold_policy='generic',
        )

        await session.create_completion([{'role': 'user', 'content': 'question'}])
        for _ in range(2):
            with pytest.raises(RuntimeError, match='store unavailable'):
                await session.finish_pending_publications()
        with pytest.raises(RuntimeError, match='failed trajectory publication'):
            session.end()

    async def test_later_store_failure_rejects_finalization(self):
        store = _FailAfterFirstPutStore('later-publication-failure')
        session = GatewaySession(
            'trajectory-1', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
        )
        await session.create_completion([{'role': 'user', 'content': 'first'}])
        await session.create_completion([{'role': 'user', 'content': 'second'}])

        with pytest.raises(RuntimeError, match='store unavailable'):
            await session.finish_pending_publications()

        with pytest.raises(RuntimeError, match='failed trajectory publication'):
            session.end()

    async def test_finalization_drains_concurrent_puts_before_raising_store_failure(self):
        store = _ConcurrentFailureStore('concurrent-publication-failure')
        session = GatewaySession(
            'trajectory-1', Processor(_TokenBackend(), tokenizer=FakeTokenizer()), store, scaffold_policy='generic'
        )

        results = await asyncio.gather(
            session.create_completion([{'role': 'user', 'content': 'first'}]),
            session.create_completion([{'role': 'user', 'content': 'second'}]),
            return_exceptions=True,
        )

        assert all(not isinstance(result, BaseException) for result in results)
        with pytest.raises(RuntimeError, match='store unavailable'):
            await asyncio.wait_for(session.finish_pending_publications(), timeout=1)
        assert store.published_refs
