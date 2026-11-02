import asyncio

from types import SimpleNamespace

import pytest

from _fakes import FakeTokenizer, memory_store

import nexus.gateway.nexus_run_tracker as nexus_run_tracker

from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.gateway.processor import Processor
from nexus.gateway.session import GatewaySession as _GatewaySession
from nexus.gateway.session_manager import GatewaySessionManager
from nexus.gateway.trajectory import Chain, ProviderFacts, Trajectory, TrajectoryNode
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import BackendResult


def _gateway_session(*args, **kwargs):
    kwargs.setdefault('scaffold_policy', 'generic')
    session_id = args[0] if args else kwargs['session_id']
    kwargs.setdefault('record_store', memory_store(str(session_id)))
    return _GatewaySession(*args, **kwargs)


class StubProcessor:
    is_token_level = False

    def __init__(self, messages):
        self._messages = iter(messages)

    async def create_completion(self, **_kwargs):
        return SimpleNamespace(
            message=next(self._messages),
            prompt_ids=None,
            raw_prompt_ids=None,
            token_ids=None,
            log_probs=None,
            routed_experts=None,
            stop_reason='stop',
            usage=None,
            id=None,
            created=None,
            model=None,
            object=None,
            system_fingerprint=None,
            service_tier=None,
            choice_logprobs=None,
            choice_provider_specific_fields=None,
            extra_fields={},
            completion_text=None,
            prompt_render=None,
        )


class TestSessionFinalization:
    def test_tool_stats_ignore_rejected_branch_descendants(self):
        nodes = {
            0: TrajectoryNode(0, 0, None, 'client', children=[1], selection='rejected_sibling'),
            1: TrajectoryNode(
                1,
                1,
                0,
                'llm',
                children=[],
                original_message={
                    'role': 'assistant',
                    'tool_calls': [{'id': 'discarded', 'function': {'name': 'search', 'arguments': '{}'}}],
                },
            ),
        }
        trajectory = Trajectory(session_id='rejected', nodes=nodes, roots=[0], chains={'main': Chain('main', [0])})
        assert NexusRunTracker()._aggregate_tool_stats(trajectory) == {}

    async def test_tool_stats_count_reused_ids_after_compaction(self):
        def call(name):
            return {
                'role': 'assistant',
                'tool_calls': [{'id': 'reused', 'function': {'name': name, 'arguments': '{}'}}],
            }

        search_call, open_call = call('search'), call('open')
        final = {'role': 'assistant', 'content': 'done'}
        session = _gateway_session('reused', processor=StubProcessor([search_call, final, open_call, final]))
        user = {'role': 'user', 'content': 'question'}
        compacted = {'role': 'user', 'content': 'summary'}
        observation = {'role': 'tool', 'tool_call_id': 'reused', 'content': 'evidence'}
        await session.create_completion([user])
        await session.create_completion([user, search_call, observation])
        await session.create_completion([compacted])
        await session.create_completion([compacted, open_call, observation])
        stats = NexusRunTracker()._aggregate_tool_stats(session.end())
        for name in ('search', 'open'):
            assert stats[name]['total'] == 1
            assert stats[name]['success'] == 1

    @pytest.mark.parametrize('replay_old_history', [False, True])
    @pytest.mark.parametrize(('result_content', 'successes'), [('evidence', 1), ('Error: unavailable', 0)])
    async def test_tool_stats_include_calls_before_a_prompt_rewrite(
        self, replay_old_history, result_content, successes
    ):
        call = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': 'before-compact', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}
            ],
        }
        session = _gateway_session(
            'tool-history',
            processor=StubProcessor(
                [
                    call,
                    {'role': 'assistant', 'content': 'working'},
                    {'role': 'assistant', 'content': 'final'},
                ]
            ),
        )
        user = {'role': 'user', 'content': 'question'}
        await session.create_completion([user])
        observation = {'role': 'tool', 'tool_call_id': 'before-compact', 'content': result_content}
        await session.create_completion([user, call, observation])
        rewritten = [{'role': 'user', 'content': 'compacted context'}]
        if replay_old_history:
            rewritten.extend([call, observation])
        await session.create_completion(rewritten)
        trajectory = session.end()
        tracker = NexusRunTracker(trajectory_id='tool-history')

        finalized = tracker.finalize_trajectory(trajectory, 'tool-history', termination_reason='ENV_DONE')

        assert finalized.extra['tool_stats']['search']['total'] == 1
        assert finalized.extra['tool_stats']['search']['success'] == successes
        assert finalized.extra['tool_stats']['search']['failure'] == 1 - successes
        assert call['tool_calls'][0]['id'] == 'before-compact'

    async def test_tool_stats_scope_call_ids_to_each_chain(self):
        def call(name):
            return {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': 'shared-id', 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}],
            }

        root_call, child_call = call('search'), call('open')
        final = {'role': 'assistant', 'content': 'done'}
        session = _gateway_session(
            'tool-chains',
            scaffold_policy='nexus_main_agent',
            processor=StubProcessor([root_call, final, child_call, final]),
        )
        root_user = {'role': 'user', 'content': 'root question'}
        child_user = {'role': 'user', 'content': 'child question'}
        observation = {'role': 'tool', 'tool_call_id': 'shared-id', 'content': 'evidence'}
        child_meta = {'declared_chain': {'chain_id': 'child', 'parent_chain_id': 'main', 'kind': 'subagent'}}
        await session.create_completion([root_user])
        await session.create_completion([root_user, root_call, observation])
        await session.create_completion([child_user], request_meta=child_meta)
        await session.create_completion([child_user, child_call, observation], request_meta=child_meta)

        tracker = NexusRunTracker(trajectory_id='tool-chains')
        trajectory = session.end()
        finalized = tracker.finalize_trajectory(trajectory, 'tool-chains', termination_reason='ENV_DONE')

        for name in ('search', 'open'):
            assert finalized.extra['tool_stats'][name]['total'] == 1
            assert finalized.extra['tool_stats'][name]['success'] == 1

    def test_finalize_exposes_wall_and_true_inference_times(self) -> None:
        trajectory = Trajectory(
            session_id='session-1',
            nodes={
                0: TrajectoryNode(
                    node_id=0,
                    message_id=0,
                    parent=None,
                    generated_by='llm',
                    children=[],
                    provider_facts=ProviderFacts(
                        inference_time=1.25,
                        wall_inference_time=2.0,
                    ),
                ),
                1: TrajectoryNode(
                    node_id=1,
                    message_id=1,
                    parent=0,
                    generated_by='llm',
                    children=[],
                    # Compatibility with trajectories recorded before wall time
                    # was split from the reported inference time.
                    provider_facts=ProviderFacts(inference_time=0.5),
                ),
            },
            roots=[0],
            metrics={},
            extra={},
        )

        finalized = NexusRunTracker(trajectory_id='session-1').finalize_trajectory(
            trajectory,
            'session-1',
            termination_reason='ENV_DONE',
        )

        assert finalized is not None
        assert finalized.metrics['llm_inference_time'] == pytest.approx(2.5)
        assert finalized.metrics['true_llm_inference_time'] == pytest.approx(1.75)

    def test_finalize_exposes_harbor_subphase_durations(self) -> None:
        tracker = NexusRunTracker(trajectory_id='session-1')
        for name, seconds in (
            ('task.handoff.artifact_collect_time', 1),
            ('task.handoff.env_setup_time', 2),
            ('task.handoff.release_main_time', 3),
            ('task.scoring.check_score_time', 5),
        ):
            span = SimpleNamespace(
                name=name,
                attributes={'type': 'task_phase', 'phase': name.split('.')[1]},
                start_time_unix_nano=1_000_000_000,
                end_time_unix_nano=1_000_000_000 + seconds * 1_000_000_000,
            )
            tracker.on_end(span, SimpleNamespace())

        finalized = tracker.finalize_trajectory(None, 'session-1', termination_reason='ENV_DONE', allow_empty=True)

        assert finalized is not None
        assert finalized.metrics['handoff_artifact_collect_time'] == pytest.approx(1.0)
        assert finalized.metrics['handoff_env_setup_time'] == pytest.approx(2.0)
        assert finalized.metrics['handoff_release_main_time'] == pytest.approx(3.0)
        assert finalized.metrics['scoring_check_score_time'] == pytest.approx(5.0)
        # Shared verification never acquires a verifier sandbox.
        assert finalized.metrics['handoff_sandbox_init_time'] == 0.0

    async def test_run_output_adds_terminal_gui_observations_without_changing_tito_or_segments(self, monkeypatch):
        class TokenBackend:
            async def create_completion(self, *_args, **_kwargs):
                return BackendResult(token_ids=tokenizer.encode('answer'), stop_reason='stop')

        class ToolParser:
            def extract_tool_calls(self, _token_ids):
                return (
                    None,
                    '',
                    [
                        {
                            'id': 'call-1',
                            'type': 'function',
                            'function': {'name': 'search', 'arguments': '{}'},
                        }
                    ],
                )

        manager = GatewaySessionManager()
        tokenizer = FakeTokenizer()
        session = manager.get_or_create(
            'session-1',
            processor=Processor(TokenBackend(), tokenizer=tokenizer, tool_parser=ToolParser()),
            record_store=memory_store('session-1'),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}
        completion = await session.create_completion([user])
        before = [
            (node.node_id, node.token_account.token_ids.tolist(), node.token_account.payload_ref)
            for node in session.get_trajectory().iter_nodes()
            if node.generated_by == 'llm' and node.token_account is not None
        ]
        observation = {'role': 'tool', 'content': 'result', 'tool_call_id': 'call-1'}
        screenshot = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Current desktop screenshot.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ],
        }
        reported_completion = {
            **completion.message,
            'provider_specific_fields': {'reasoning_effort': 'no_think'},
        }
        output = SimpleNamespace(
            agent_output=SimpleNamespace(main_trajectory=[user, reported_completion, observation, screenshot]),
        )
        context = SimpleNamespace(get_runtime_id=lambda _name: None)
        monkeypatch.setattr(nexus_run_tracker, 'session_manager', manager)
        tracker = NexusRunTracker(trajectory_id='session-1')

        await tracker.capture_run_output(context, output)

        trajectory = session.get_trajectory()
        after = [
            (node.node_id, node.token_account.token_ids.tolist(), node.token_account.payload_ref)
            for node in trajectory.iter_nodes()
            if node.generated_by == 'llm' and node.token_account is not None
        ]
        observation_node = next(node for node in trajectory.iter_nodes() if node.original_message == observation)
        screenshot_node = next(node for node in trajectory.iter_nodes() if node.original_message == screenshot)
        assert trajectory.to_messages() == [user, completion.message, observation, screenshot]
        assert observation_node.generated_by == 'observation'
        assert observation_node.token_account is None
        assert screenshot_node.generated_by == 'observation'
        assert screenshot_node.token_account is None
        assert after == before

        await session.finish_pending_publications()
        finalized = tracker.finalize_trajectory(
            session.end(),
            'session-1',
            termination_reason='ENV_DONE',
        )
        assert finalized is not None
        assert finalized.extra['tool_stats']['search'] == {
            'total': 1,
            'success': 1,
            'failure': 0,
            'failure_readpage': 0,
            'failure_summary_llm': 0,
        }
        expected_messages = [user, completion.message, observation, screenshot]
        assert finalized.to_messages() == expected_messages

    async def test_run_output_does_not_import_observation_after_a_divergent_transcript(self, monkeypatch, caplog):
        manager = GatewaySessionManager()
        session = manager.get_or_create(
            'session-1',
            processor=StubProcessor([{'role': 'assistant', 'content': 'ok'}]),
            record_store=memory_store('session-1'),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}
        completion = await session.create_completion([user])
        output = SimpleNamespace(
            agent_output=SimpleNamespace(
                main_trajectory=[
                    {'role': 'user', 'content': 'different'},
                    completion.message,
                    {'role': 'tool', 'content': 'result'},
                ],
            ),
        )
        context = SimpleNamespace(get_runtime_id=lambda _name: None)
        monkeypatch.setattr(nexus_run_tracker, 'session_manager', manager)

        with caplog.at_level('WARNING', logger='nexus.gateway.nexus_run_tracker'):
            await NexusRunTracker(trajectory_id='session-1').capture_run_output(context, output)

        trajectory = session.get_trajectory()
        assert trajectory.to_messages() == [user, completion.message]
        assert all(node.generated_by != 'observation' for node in trajectory.iter_nodes())
        assert 'does not extend the recorded transcript' in caplog.text

    async def test_run_output_imports_root_terminal_observation_when_a_child_chain_is_latest(self, monkeypatch):
        root_reply = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'root-call',
                    'type': 'function',
                    'function': {'name': 'wait_agent', 'arguments': '{"agent_ids":["n1"]}'},
                }
            ],
        }
        child_reply = {'role': 'assistant', 'content': '42'}
        manager = GatewaySessionManager()
        session = manager.get_or_create(
            'session-1',
            processor=StubProcessor([root_reply, child_reply]),
            record_store=memory_store('session-1'),
            scaffold_policy='nexus_main_agent',
        )
        root_user = {'role': 'user', 'content': 'solve the task'}
        root_completion = await session.create_completion([root_user])
        child_messages = [
            {'role': 'system', 'content': 'child system'},
            {'role': 'user', 'content': 'child task'},
        ]
        await session.create_completion(
            child_messages,
            request_meta={
                'declared_chain': {
                    'chain_id': 'agent-1',
                    'parent_chain_id': 'main',
                    'kind': 'subagent',
                }
            },
        )
        terminal_observation = {
            'role': 'tool',
            'tool_call_id': 'root-call',
            'content': "KeyError: 'unknown agent_id: n1'",
        }
        output = SimpleNamespace(
            agent_output=SimpleNamespace(
                main_trajectory=[root_user, root_completion.message, terminal_observation],
            ),
        )
        context = SimpleNamespace(get_runtime_id=lambda _name: None)
        monkeypatch.setattr(nexus_run_tracker, 'session_manager', manager)

        await NexusRunTracker(trajectory_id='session-1').capture_run_output(context, output)

        trajectory = session.get_trajectory()
        assert trajectory.to_messages() == [root_user, root_completion.message, terminal_observation]
        assert trajectory.to_messages() == [root_user, root_completion.message, terminal_observation]

    async def test_run_output_does_not_import_an_unrecorded_terminal_completion(self, monkeypatch, caplog):
        manager = GatewaySessionManager()
        session = manager.get_or_create(
            'session-1',
            processor=StubProcessor([{'role': 'assistant', 'content': 'recorded'}]),
            record_store=memory_store('session-1'),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}
        completion = await session.create_completion([user])
        unrecorded = {'role': 'assistant', 'content': 'not recorded by Gateway'}
        output = SimpleNamespace(
            agent_output=SimpleNamespace(main_trajectory=[user, completion.message, unrecorded]),
        )
        context = SimpleNamespace(get_runtime_id=lambda _name: None)
        monkeypatch.setattr(nexus_run_tracker, 'session_manager', manager)

        with caplog.at_level('WARNING', logger='nexus.gateway.nexus_run_tracker'):
            await NexusRunTracker(trajectory_id='session-1').capture_run_output(context, output)

        trajectory = session.get_trajectory()
        assert trajectory.to_messages() == [user, completion.message]
        assert all(node.original_message != unrecorded for node in trajectory.iter_nodes())
        assert 'has an unsupported terminal input suffix' in caplog.text

    @pytest.mark.parametrize(
        'invalid',
        [
            {'role': 'assistant', 'content': 'no tool'},
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'bad',
                        'type': 'function',
                        'function': {'name': 'search', 'arguments': '{'},
                    }
                ],
            },
        ],
    )
    async def test_retry_generations_are_retained_as_siblings(self, invalid):
        valid = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'good',
                    'type': 'function',
                    'function': {'name': 'search', 'arguments': '{}'},
                }
            ],
        }
        session = _gateway_session('session-1', StubProcessor([invalid, valid]))
        messages = [{'role': 'user', 'content': 'question'}]

        rejected = await session.create_completion(messages)
        accepted = await session.create_completion(messages)
        trajectory = session.end()

        assert rejected.message == invalid
        assert rejected.recorded
        assert accepted.recorded
        llm = [node for node in trajectory.iter_nodes() if node.generated_by == 'llm']
        assert [node.original_message for node in llm] == [invalid, valid]
        assert llm[0].parent == llm[1].parent

    def test_final_facts_are_frozen_into_trajectory(self):
        metadata = {'run_id': 'run-1', 'nested': {'value': 1}}
        metrics = {'llm_inference_time': {'value': 2.0}}
        final_extra = {'runtime_id': 'runtime-1', 'nested_extra': {'value': 2}}
        session = _gateway_session(
            session_id='session-1',
            processor=object(),
            metadata=metadata,
        )
        metadata['nested']['value'] = 2

        trajectory = session.end(
            termination_reason='ENV_DONE',
            final_reward=1.0,
            metrics=metrics,
            extra=final_extra,
        )
        metrics['llm_inference_time']['value'] = 99.0
        final_extra['nested_extra']['value'] = 99

        assert trajectory.termination_reason == 'ENV_DONE'
        assert trajectory.final_reward == 1.0
        assert trajectory.metrics == {'llm_inference_time': {'value': 2.0}}
        assert trajectory.extra == {
            'run_id': 'run-1',
            'nested': {'value': 1},
            'runtime_id': 'runtime-1',
            'nested_extra': {'value': 2},
        }
        assert trajectory is session.segment_assembler.trajectory
        assert session.get_trajectory() is not trajectory

    async def test_latest_messages_preserve_the_last_wire_variant(self):
        decorated = {'role': 'assistant', 'content': 'ok', 'provider_specific_fields': {'cached': True}}
        resent = {'role': 'assistant', 'content': 'ok'}
        done = {'role': 'assistant', 'content': 'done'}
        session = _gateway_session('session-1', StubProcessor([decorated, done]))
        system = {'role': 'system', 'content': 'system'}
        user = {'role': 'user', 'content': 'question'}
        next_user = {'role': 'user', 'content': 'next'}

        await session.create_completion([system, user])
        await session.create_completion([system, user, resent, next_user])
        trajectory = session.end()

        assert trajectory.to_messages() == [system, user, resent, next_user, done]

    async def test_backend_mutation_cannot_desynchronize_wire_identity_or_scaffold_context(self):
        class MutatingBackend:
            def __init__(self):
                self.calls = 0

            async def create_completion(self, _request_id, *, messages, request_meta, **_kwargs):
                self.calls += 1
                messages[0]['content'] = 'backend-mutated'
                request_meta['declared_chain']['chain_id'] = 'B'
                return BackendResult(
                    message={'role': 'assistant', 'content': f'answer {self.calls}'},
                    stop_reason='stop',
                )

        backend = MutatingBackend()
        session = _gateway_session('session-1', Processor(backend), scaffold_policy='nexus_main_agent')
        messages = [{'role': 'user', 'content': 'caller-owned'}]
        request_meta = {'declared_chain': {'chain_id': 'A', 'kind': 'worker'}}

        await session.create_completion(messages, request_meta=request_meta)
        await session.create_completion(messages, request_meta=request_meta)
        trajectory = session.end()

        assert messages == [{'role': 'user', 'content': 'caller-owned'}]
        assert request_meta == {'declared_chain': {'chain_id': 'A', 'kind': 'worker'}}
        assert trajectory.roots == [0]
        assert trajectory.nodes[0].original_message == messages[0]
        assert trajectory.nodes[0].children == [1, 2]
        assert list(trajectory.chains) == ['main', 'A']
        assert trajectory.chains['A'].roots == [0]
        assert trajectory.to_messages('A') == [messages[0], {'role': 'assistant', 'content': 'answer 2'}]

    async def test_tool_call_finish_reason_matches_the_scaffold_response(self):
        tool_call = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}],
        }
        session = _gateway_session('session-1', StubProcessor([tool_call]))

        await session.create_completion([{'role': 'user', 'content': 'question'}])

        llm = next(node for node in session.end().iter_nodes() if node.generated_by == 'llm')
        assert llm.provider_facts.finish_reason == 'tool_calls'

    async def test_explicit_observation_parent_replaces_the_wire_snapshot(self):
        session = _gateway_session('session-1', StubProcessor([{'role': 'assistant', 'content': 'reply'}]))
        parent = [{'role': 'user', 'content': 'first'}]
        await session.create_completion(parent)
        parent.append({'role': 'assistant', 'content': 'reply'})

        session.record_observation('result', None, 'tool', parent)

        assert session.end().to_messages() == [*parent, {'role': 'tool', 'content': 'result'}]

    async def test_ambiguous_observation_parent_fails_loudly(self):
        user = {'role': 'user', 'content': 'question'}
        reply = {'role': 'assistant', 'content': 'same'}
        session = _gateway_session('session-1', StubProcessor([reply, reply]))
        await session.create_completion([user])
        await session.create_completion([user])

        with pytest.raises(ValueError, match='multiple physical trajectory paths'):
            session.record_observation('result', parent_messages=[user, reply])

    def test_missing_observation_parent_fails_loudly(self):
        session = _gateway_session('session-1', StubProcessor([]))

        with pytest.raises(ValueError, match='do not match this trajectory'):
            session.record_observation('result', parent_messages=[{'role': 'user', 'content': 'missing'}])

    async def test_explicit_tool_message_preserves_wire_identity(self):
        user = {'role': 'user', 'content': 'question'}
        assistant = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}],
        }
        tool = {'role': 'tool', 'tool_call_id': 'call-1', 'name': 'search', 'content': 'result'}
        session = _gateway_session(
            'session-1',
            StubProcessor([assistant, {'role': 'assistant', 'content': 'done'}]),
        )

        await session.create_completion([user])
        session.record_observation(message=tool, parent_messages=[user, assistant])
        await session.create_completion([user, assistant, tool])
        trajectory = session.end()

        tool_nodes = [node for node in trajectory.iter_nodes() if node.original_message.get('role') == 'tool']
        assert len(tool_nodes) == 1
        assert tool_nodes[0].original_message == tool

    async def test_observation_switches_latest_messages_to_its_declared_chain(self):
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]
        reply_a = {'role': 'assistant', 'content': 'A'}
        reply_b = {'role': 'assistant', 'content': 'B'}
        observation = {'role': 'tool', 'content': 'A result'}
        meta_a = {'declared_chain': {'chain_id': 'A', 'kind': 'worker'}}
        meta_b = {'declared_chain': {'chain_id': 'B', 'kind': 'worker'}}
        session = _gateway_session('session-1', StubProcessor([reply_a, reply_b]), scaffold_policy='nexus_main_agent')
        await session.create_completion(messages, request_meta=meta_a)
        await session.create_completion(messages, request_meta=meta_b)

        session.record_observation(message=observation, parent_messages=[*messages, reply_a], request_meta=meta_a)
        trajectory = session.end()

        assert trajectory.to_messages('A') == [reply_a, observation]

    def test_end_all_removes_every_session_before_raising(self):
        calls = []

        class FakeSession:
            def __init__(self, session_id, error=None):
                self.session_id = session_id
                self.error = error

            def end(self, **_kwargs):
                calls.append(self.session_id)
                if self.error is not None:
                    raise self.error
                return object()

        manager = GatewaySessionManager()
        first_error = RuntimeError('first')
        manager._sessions = {
            'first': FakeSession('first', first_error),
            'second': FakeSession('second'),
        }

        with pytest.raises(RuntimeError) as caught:
            manager.end_all('cleanup')

        assert caught.value is first_error
        assert calls == ['first', 'second']
        assert manager.active_count == 0
        assert not manager.has_session('first')
        assert not manager.has_session('second')

    def test_manager_rejects_an_active_id_owned_by_another_processor(self):
        manager = GatewaySessionManager()
        store = memory_store('trajectory-1')
        manager.get_or_create('trajectory-1', processor=object(), record_store=store, scaffold_policy='generic')

        with pytest.raises(RuntimeError, match='trajectory-1'):
            manager.get_or_create('trajectory-1', processor=object(), record_store=store, scaffold_policy='generic')

    def test_ending_one_trajectory_does_not_affect_another(self):
        manager = GatewaySessionManager()
        manager.get_or_create(
            'trajectory-1', processor=object(), record_store=memory_store('trajectory-1'), scaffold_policy='generic'
        )
        manager.get_or_create(
            'trajectory-2', processor=object(), record_store=memory_store('trajectory-2'), scaffold_policy='generic'
        )

        manager.end_session('trajectory-1')

        assert not manager.has_session('trajectory-1')
        assert manager.has_session('trajectory-2')


class TestObservationLineage:
    def test_declared_chain_selects_its_own_latest_parent(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]
        meta_a = {'declared_chain': {'chain_id': 'A', 'kind': 'worker'}}
        meta_b = {'declared_chain': {'chain_id': 'B', 'kind': 'worker'}}
        reply_a = {'role': 'assistant', 'content': 'A'}
        reply_b = {'role': 'assistant', 'content': 'B'}
        assembler.record_completion(messages, reply_a, request_meta=meta_a)
        assembler.record_completion(messages, reply_b, request_meta=meta_b)
        llm_a, llm_b = assembler.trajectory.nodes[2], assembler.trajectory.nodes[3]

        assembler.record_observation(
            message={'role': 'tool', 'content': 'A result'},
            parent_messages=[*messages, reply_a],
            request_meta=meta_a,
        )

        observation = assembler.trajectory.nodes[llm_a.children[0]]
        assert observation.parent == llm_a.node_id
        assert not llm_b.children

    def test_explicit_parent_must_match_the_declared_chain(self):
        assembler = TrajectoryAssembler(scaffold_policy='nexus_main_agent')
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]
        reply = {'role': 'assistant', 'content': 'A'}
        assembler.record_completion(
            messages,
            reply,
            request_meta={'declared_chain': {'chain_id': 'A', 'kind': 'worker'}},
        )

        with pytest.raises(ValueError, match='do not match this trajectory'):
            assembler.record_observation(
                message={'role': 'tool', 'content': 'wrong lane'},
                parent_messages=[*messages, reply],
                request_meta={'declared_chain': {'chain_id': 'B', 'kind': 'worker'}},
            )


class HangingProcessor(StubProcessor):
    """Serves scripted messages, then hangs forever on the next call."""

    def __init__(self, messages):
        super().__init__(messages)
        self._remaining = len(messages)
        self.started = asyncio.Event()

    async def create_completion(self, **kwargs):
        if self._remaining > 0:
            self._remaining -= 1
            return await super().create_completion(**kwargs)
        self.started.set()
        await asyncio.Event().wait()
        return None


class TestSessionAbortInFlight:
    async def test_abort_cancels_pending_and_end_annotates(self, caplog):
        processor = HangingProcessor([])
        session = _gateway_session('session-1', processor)
        pending = asyncio.ensure_future(session.create_completion([{'role': 'user', 'content': 'q'}]))
        await processor.started.wait()

        with caplog.at_level('WARNING'):
            labels = await session.abort_in_flight()
        trajectory = session.end(termination_reason='TASK_TIMEOUT')

        assert pending.cancelled()
        assert len(labels) == 1
        assert 'aborted 1 in-flight request' in caplog.text
        assert trajectory.extra['aborted_in_flight_requests'] == labels
        assert session._in_flight == 0
        assert session._pending_completions == {}

    async def test_abort_seals_the_session_against_late_requests(self):
        session = _gateway_session('session-1', StubProcessor([{'role': 'assistant', 'content': 'a'}]))
        await session.create_completion([{'role': 'user', 'content': 'q'}])

        assert await session.abort_in_flight() == []

        with pytest.raises(RuntimeError, match='already ended'):
            await session.create_completion([{'role': 'user', 'content': 'late'}])

        trajectory = session.end(termination_reason='ENV_DONE')
        assert [node.original_message['content'] for node in trajectory.iter_nodes()] == ['q', 'a']

    async def test_end_without_abort_still_rejects_in_flight(self):
        processor = HangingProcessor([])
        session = _gateway_session('session-1', processor)
        pending = asyncio.ensure_future(session.create_completion([{'role': 'user', 'content': 'q'}]))
        await processor.started.wait()

        with pytest.raises(RuntimeError, match='in-flight request'):
            session.end(termination_reason='ENV_DONE')

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    async def test_abort_without_pending_is_a_noop(self, caplog):
        session = _gateway_session('session-1', StubProcessor([{'role': 'assistant', 'content': 'a'}]))
        await session.create_completion([{'role': 'user', 'content': 'q'}])

        with caplog.at_level('WARNING'):
            labels = await session.abort_in_flight()
        trajectory = session.end(termination_reason='ENV_DONE')

        assert labels == []
        assert 'aborted' not in caplog.text
        assert 'aborted_in_flight_requests' not in trajectory.extra
        assert [m['content'] for m in trajectory.to_messages()] == ['q', 'a']

    async def test_abort_keeps_recorded_completion_and_drops_only_the_orphan(self):
        processor = HangingProcessor([{'role': 'assistant', 'content': 'a'}])
        session = _gateway_session('session-1', processor)
        await session.create_completion([{'role': 'user', 'content': 'q'}])
        history = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'q2'},
        ]
        orphan = asyncio.ensure_future(session.create_completion(history))
        await processor.started.wait()

        await session.abort_in_flight()
        trajectory = session.end(termination_reason='UNHANDLED_ERROR')

        assert orphan.cancelled()
        assert [m['content'] for m in trajectory.to_messages()] == ['q', 'a']
        assert session._in_flight == 0

    async def test_abort_cancels_every_pending_request(self):
        processor = HangingProcessor([])
        session = _gateway_session('session-1', processor)
        first = asyncio.ensure_future(session.create_completion([{'role': 'user', 'content': 'q1'}]))
        await processor.started.wait()
        processor.started.clear()
        second = asyncio.ensure_future(session.create_completion([{'role': 'user', 'content': 'q2'}]))
        await processor.started.wait()

        labels = await session.abort_in_flight()

        assert first.cancelled()
        assert second.cancelled()
        assert len(labels) == 2
        assert session._in_flight == 0
        assert session._pending_completions == {}

    async def test_manager_abort_passthrough_and_missing_session_noop(self):
        manager = GatewaySessionManager()
        assert await manager.abort_in_flight('missing') == []

        processor = HangingProcessor([])
        manager.get_or_create(
            'session-1', processor=processor, record_store=memory_store('session-1'), scaffold_policy='generic'
        )
        pending = asyncio.ensure_future(
            manager._sessions['session-1'].create_completion([{'role': 'user', 'content': 'q'}])
        )
        await processor.started.wait()

        labels = await manager.abort_in_flight('session-1')
        trajectory = manager.end_session('session-1', termination_reason='TASK_TIMEOUT')

        assert pending.cancelled()
        assert trajectory.extra['aborted_in_flight_requests'] == labels


class _StopReasonProcessor(StubProcessor):
    def __init__(self, outcomes):
        super().__init__([{'role': 'assistant', 'content': 'x'}] * len(outcomes))
        self._outcomes = iter(outcomes)

    async def create_completion(self, **kwargs):
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        result = await super().create_completion(**kwargs)
        result.stop_reason = outcome
        return result


async def test_session_tracks_whether_the_last_main_call_was_length_limited():
    from nexus.core import ContextLengthExceededError

    processor = _StopReasonProcessor(['length', 'stop', ContextLengthExceededError('too long')])
    session = _gateway_session('truncation-tracking', processor=processor)
    messages = [{'role': 'user', 'content': 'q'}]

    await session.create_completion(messages)
    assert session.last_main_call_truncated
    await session.create_completion(messages)
    assert not session.last_main_call_truncated
    with pytest.raises(ContextLengthExceededError):
        await session.create_completion(messages)
    assert session.last_main_call_truncated


async def test_ordinary_failure_after_length_limited_call_clears_truncation():
    processor = _StopReasonProcessor(['length', RuntimeError('backend unavailable')])
    session = _gateway_session('truncation-reset', processor=processor)
    messages = [{'role': 'user', 'content': 'q'}]

    await session.create_completion(messages)
    assert session.last_main_call_truncated
    with pytest.raises(RuntimeError):
        await session.create_completion(messages)
    assert not session.last_main_call_truncated
