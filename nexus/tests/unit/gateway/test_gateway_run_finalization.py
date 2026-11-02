import asyncio
import json

from types import SimpleNamespace

import numpy as np
import pytest

from _fakes import FakeTokenizer

from nexus.gateway.backend import BackendResult, OpenAIAPIBackend
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import load_trajectory_with_payloads
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore, RedisTrajectoryRecordStore
from nexus.runner import single
from nexus.runner.single import ProcessorConfig, RunHooks, RunInput


class MessageBackend:
    async def create_completion(self, **kwargs):
        message = SimpleNamespace(
            content='done', reasoning_content=None, provider_specific_fields=None, tool_calls=None
        )
        choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
        return SimpleNamespace(choices=[choice], usage=None, model='fake')


class TokenBackend:
    async def create_completion(self, *args, **kwargs):
        return BackendResult(
            token_ids=[65, 66],
            log_probs=[-0.1, -0.2],
            extra_fields={'resp_topk_logprob_indice': [[3], [4]], 'resp_topk_logprob_value': [[-0.3], [-0.4]]},
        )


def run_input(run_id, token_level=False):
    return RunInput(
        run_id=run_id,
        task_name='identity',
        agent_name='main_agent',
        agent_config={'user_prompt': 'say done', 'tools': [], 'max_iterations': 1},
        openai_clients={
            'main': {
                'name': 'gateway',
                'config': {
                    'client_args': {
                        'trajectory_id': run_id,
                        'scaffold_policy': 'nexus_main_agent',
                        **({'tokenizer': FakeTokenizer()} if token_level else {}),
                    }
                },
            }
        },
        processors={'gateway': ProcessorConfig(name='gateway_trajectory', args={'trajectory_id': run_id})},
    )


@pytest.fixture(params=['memory', 'redis'])
def stores(monkeypatch, request):
    """Record store traffic for stores the Gateway client builds from config."""

    class StoreList(list):
        pass

    created = StoreList()
    real_redis = request.param != 'memory'
    base = RedisTrajectoryRecordStore if real_redis else InMemoryTrajectoryRecordStore

    class RecordingStore(base):
        def __init__(self, namespace, *args, **kwargs):
            super().__init__(namespace, *args, **kwargs)
            self.refs = []
            self.events = []
            created.append(self)

        async def put_many(self, records):
            await asyncio.sleep(0)
            refs = await super().put_many(records)
            self.refs.extend(refs)
            self.events.append('put')
            return refs

        async def get_many(self, refs, **kwargs):
            self.events.append('get')
            return await super().get_many(refs, **kwargs)

        async def release_many(self, refs):
            self.events.append('release')
            await super().release_many(refs)

    created.store_type = RecordingStore
    created.real_redis = real_redis
    if real_redis:
        from nexus.gateway import client as gateway_client

        config = request.getfixturevalue(f'{request.param}_config')
        build = gateway_client.build_trajectory_record_store
        monkeypatch.setattr(
            gateway_client,
            'build_trajectory_record_store',
            lambda options: build({**options, 'backend': 'redis', 'redis': config}),
        )
        monkeypatch.setattr('nexus.gateway.trajectory_store.RedisTrajectoryRecordStore', RecordingStore)
    else:
        monkeypatch.setattr('nexus.gateway.trajectory_store.InMemoryTrajectoryRecordStore', RecordingStore)
    monkeypatch.setattr(
        GatewayOpenAIClient,
        '_build_backend',
        lambda self, args: TokenBackend() if args.get('tokenizer') else OpenAIAPIBackend(MessageBackend()),
    )
    return created


@pytest.mark.parametrize('token_level', [False, True])
async def test_processor_exports_all_payloads_and_releases_store(stores, token_level):
    config = run_input('processor-export', token_level)
    processor_key = 'token_trajectory' if token_level else 'gateway'
    config.processors = {processor_key: config.processors['gateway']}
    callbacks = []

    def capture_output(context, output):
        assert 'node_payloads' in output.processor_results[processor_key]
        callbacks.append('output')

    hooks = RunHooks(
        on_context_ready=lambda context: callbacks.append('ready'),
        on_run_output=capture_output,
        on_context_closing=lambda context: callbacks.append('closing'),
    )
    result = await single.run(config, hooks=hooks)
    assert result.success
    assert callbacks == ['ready', 'output', 'closing']
    # The runner stays Gateway-agnostic: no injected store, no rewritten hooks.
    assert hooks.on_run_output is capture_output
    assert hooks.span_processors == []
    assert config.processors[processor_key].name == 'gateway_trajectory'
    assert 'record_store' not in config.openai_clients['main']['config']['client_args']
    trajectory, payloads = load_trajectory_with_payloads(
        json.loads(result.model_dump_json())['processor_results'][processor_key]
    )
    assert trajectory.nodes
    assert all(node.token_account is None or node.token_account.payload_ref is None for node in trajectory.iter_nodes())
    if token_level:
        output = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        np.testing.assert_array_equal(payloads[output.node_id]['token_ids'], output.token_account.token_ids)
        np.testing.assert_array_equal(payloads[output.node_id]['logprobs'], np.array([-0.1, -0.2], dtype=np.float32))
        assert payloads[output.node_id]['topk_indices'].tolist() == [[3], [4]]
        np.testing.assert_array_equal(
            payloads[output.node_id]['topk_values'], np.array([[-0.3], [-0.4]], dtype=np.float32)
        )
        assert stores[0].events == ['put', 'get', 'release']
        if stores.real_redis:
            with pytest.raises(KeyError):
                await stores[0].get_many(stores[0].refs)
        else:
            assert await stores[0].get_many(stores[0].refs)
    else:
        assert payloads == {}
        assert stores[0].events == []
    assert not session_manager.has_session(config.run_id)


async def test_concurrent_runs_own_separate_stores(stores):
    results = await asyncio.gather(*(single.run(run_input(f'concurrent-{index}', True)) for index in range(2)))
    assert all(result.success for result in results)
    assert len({store.namespace for store in stores}) == 2
    assert all(store.events == ['put', 'get', 'release'] for store in stores)


async def test_failed_task_keeps_completed_payloads(stores, request):
    from nexus.core import TaskRunOutput, task_registry
    from nexus.extensions.tasks.identity.task import IdentityTaskInput

    task_name = f'test_processor_task_failure_{request.node.callspec.id}'

    @task_registry.register(task_name)
    async def failed_task(task_input: IdentityTaskInput) -> TaskRunOutput:
        await task_input.run_agent()
        raise RuntimeError('task failed after completion')

    config = run_input('failed-task', True)
    config.task_name = task_name
    result = await single.run(config)
    assert not result.success
    assert result.error.message == 'task failed after completion'
    trajectory, payloads = load_trajectory_with_payloads(result.processor_results['gateway'])
    assert trajectory.nodes
    assert payloads
    assert stores[0].events == ['put', 'get', 'release']


@pytest.mark.parametrize('failure', ['read', 'dump'])
async def test_finalization_failure_releases_payloads(stores, monkeypatch, failure):
    config = run_input('finalize-failure', True)
    if failure == 'read':

        async def fail_read(self, refs, **kwargs):
            raise RuntimeError('read failed')

        monkeypatch.setattr(stores.store_type, 'get_many', fail_read)
    elif failure == 'dump':

        def fail_dump(*args):
            raise RuntimeError('dump failed')

        monkeypatch.setattr('nexus.gateway.trajectory_processor.dump_trajectory_with_payloads', fail_dump)
    # Finalization must fail closed rather than yield a result with missing payloads.
    with pytest.raises(RuntimeError, match=f'{failure} failed'):
        await single.run(config)
    assert stores[0].events[-1] == 'release'
    assert not session_manager.has_session(config.run_id)


async def test_cancellation_waits_for_put_before_releasing(stores, monkeypatch):
    started = asyncio.Event()
    proceed = asyncio.Event()
    original_put = stores.store_type.put_many

    async def delayed_put(self, records):
        started.set()
        await proceed.wait()
        return await original_put(self, records)

    monkeypatch.setattr(stores.store_type, 'put_many', delayed_put)
    task = asyncio.create_task(single.run(run_input('cancelled', True)))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert stores[0].events[-1] == 'release'
    assert not session_manager.has_session('cancelled')


async def test_batch_exports_gateway_without_store_config(stores):
    from nexus.runner.batch import BatchConfig, BatchItem, run_batch

    summary = await run_batch(
        generator_path=None,
        db_path=':memory:',
        batch_config=BatchConfig(max_concurrent=2, enable_trace_db=False, enable_log_capture=False),
        batch_items=[BatchItem(run_id=f'batch-{index}', config=run_input(f'batch-{index}')) for index in range(2)],
    )
    assert summary.success == 2
    assert summary.failed == 0
    assert len(stores) == 2


async def test_non_gateway_run_creates_no_store(stores):
    config = run_input('plain')
    config.openai_clients = {}
    config.processors = {}
    # The identity task needs a client, so the run fails; what matters is that a run
    # without a Gateway client never builds a record store.
    await single.run(config)
    assert stores == []


@pytest.mark.parametrize(('finish_reason', 'expected'), [('length', 'TRUNCATION'), ('stop', 'ENV_DONE')])
async def test_gateway_marks_length_limited_run_as_truncation_without_agent_report(
    monkeypatch, finish_reason, expected
):
    class LengthBackend:
        async def create_completion(self, **kwargs):
            message = SimpleNamespace(
                content='cut', reasoning_content=None, provider_specific_fields=None, tool_calls=None
            )
            choice = SimpleNamespace(message=message, finish_reason=finish_reason, logprobs=None)
            return SimpleNamespace(choices=[choice], usage=None, model='fake')

    monkeypatch.setattr(GatewayOpenAIClient, '_build_backend', lambda self, args: OpenAIAPIBackend(LengthBackend()))
    run_id = f'gateway-truncation-{finish_reason}'
    config = RunInput(
        run_id=run_id,
        task_name='identity',
        agent_name='single_turn',
        agent_config={'prompt': [{'role': 'user', 'content': 'q'}]},
        openai_clients={'main': {'name': 'gateway', 'config': {'client_args': {'trajectory_id': run_id}}}},
        processors={'gateway': ProcessorConfig(name='gateway_trajectory', args={'trajectory_id': run_id})},
    )
    result = await single.run(config)
    assert result.success
    trajectory, _ = load_trajectory_with_payloads(json.loads(result.model_dump_json())['processor_results']['gateway'])
    assert trajectory.termination_reason == expected
