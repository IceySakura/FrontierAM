"""Both installed OpenCC adapters keep the same host-side RL contract."""

import asyncio
import json

from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.agents.hycode import agent as hycode
from nexus.extensions.agents.hyness import agent as hyness
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import RemoteTraceContext


@pytest.fixture(params=['hycode', 'hyness'])
def adapter(request):
    if request.param == 'hycode':
        return SimpleNamespace(module=hycode, input=hycode.HyCodeInput, run=hycode.run_hycode, defaults={})
    return SimpleNamespace(
        module=hyness,
        input=hyness.HynessInput,
        run=hyness.run_hyness,
        defaults={'packages': ['@tencent/hyness@1.0.0']},
    )


def input_for(adapter, **kwargs):
    return adapter.input(user_prompt='test', working_dir='/workspace', **adapter.defaults, **kwargs)


def completed(**updates):
    return {
        'success': True,
        'exit_code': 0,
        'process_exit_code': 0,
        'result_count': 1,
        'result_subtype': 'success',
        'result_is_error': False,
        **updates,
    }


def capture_diagnostics(monkeypatch, attributes, *, fail=False):
    def record(_span, *, attributes: dict):
        if fail:
            raise RuntimeError('trace sink unavailable')
        captured.update(attributes)

    captured = attributes
    for module in (hycode, hyness):
        monkeypatch.setattr(module, 'get_current_span_id', lambda: 'agent-span')
        monkeypatch.setattr(module, 'update_span', record, raising=False)


@pytest.mark.parametrize(
    ('facts', 'expected'),
    [
        (
            completed(
                success=False,
                exit_code=64,
                process_exit_code=1,
                result_subtype='error_max_turns',
                result_is_error=True,
                num_turns=600,
            ),
            {
                'hytuner.termination_reason': 'MAX_STEPS',
                'hycode.exit_code': 64,
                'hycode.termination_source': 'opencc_exit_protocol',
                'hycode.num_turns': 600,
                'hycode.result_subtype': 'error_max_turns',
            },
        ),
        (
            completed(
                success=False,
                exit_code=1,
                process_exit_code=1,
                result_is_error=True,
                last_llm_error_code='context_length_exceeded',
            ),
            {
                'hytuner.termination_reason': 'TRUNCATION',
                'hycode.exit_code': 1,
                'hycode.termination_source': 'host_llm_error_code',
            },
        ),
        (
            {'success': False, 'exit_code': -9, 'result_count': 0},
            {'hytuner.termination_reason': 'ENV_DONE', 'hycode.exit_code': -9, 'hycode.scoreable_process_exit': True},
        ),
        (
            completed(success=False, exit_code=1, result_subtype='error_during_execution', result_is_error=True),
            {
                'hytuner.termination_reason': 'ENV_ERROR',
                'hycode.exit_code': 1,
                'hycode.result_subtype': 'error_during_execution',
            },
        ),
    ],
)
async def test_adapters_keep_shared_exit_diagnostics(adapter, monkeypatch, facts, expected):
    attributes = {}
    capture_diagnostics(monkeypatch, attributes)

    async def invoke(*_args, **_kwargs):
        return dict(facts)

    monkeypatch.setattr(adapter.module, 'call_runtime_agent', invoke)
    result = await adapter.run(input_for(adapter))
    assert result.termination_reason == expected['hytuner.termination_reason']
    assert expected.items() <= attributes.items()
    if adapter.module is hyness:
        assert attributes['hyness.process_exit_code'] == facts.get('process_exit_code', facts['exit_code'])


@pytest.mark.parametrize('sink_fails', [False, True])
@pytest.mark.parametrize('unhealthy', [False, True])
async def test_diagnostic_failure_never_replaces_the_runtime_result(adapter, monkeypatch, sink_fails, unhealthy):
    attributes = {}
    capture_diagnostics(monkeypatch, attributes, fail=sink_fails)

    async def invoke(*_args, **_kwargs):
        if unhealthy:
            raise RuntimeUnhealthyError('sandbox OOMKilled')
        return completed()

    monkeypatch.setattr(adapter.module, 'call_runtime_agent', invoke)
    result = await adapter.run(input_for(adapter))
    assert result.termination_reason == ('ENV_ERROR' if unhealthy else 'ENV_DONE')
    assert result.success is not unhealthy
    if unhealthy:
        assert result.failure_kind == 'infrastructure'
        assert result.failure_stage == 'runtime_unhealthy'
        assert result.exit_status == 'env_error'
        assert 'sandbox OOMKilled' in result.stderr
        if not sink_fails:
            assert attributes['hycode.termination_source'] == 'runtime_unhealthy'


@pytest.mark.parametrize('task_config', [None, {}, {'is_validation': False}, {'is_validation': True}])
@pytest.mark.parametrize('explicit', [None, False, True])
async def test_invocation_scope_keeps_training_defaults_and_explicit_override(
    adapter, monkeypatch, task_config, explicit
):
    trace = RemoteTraceContext(trace_id='trace', parent_span_id='runtime-budget')
    monkeypatch.setattr(RemoteTraceContext, 'from_current_span', lambda: trace)
    context = None
    captured = {}

    async def invoke(*_args, **kwargs):
        captured.update(kwargs)
        if context is not None:
            assert context.runtime_command_timeout_id == 'runtime-budget'
        return completed()

    monkeypatch.setattr(adapter.module, 'call_runtime_agent', invoke)
    async with AsyncExitStack() as stack:
        if task_config is not None:
            context = await stack.enter_async_context(ExecutionContext(task_config=task_config))
        await adapter.run(input_for(adapter, disable_invoke_wait_timeout=explicit))
        if context is not None:
            assert context.runtime_command_timeout_id is None
    expected = (
        explicit if explicit is not None else task_config is not None and task_config.get('is_validation') is False
    )
    assert captured['disable_invoke_wait_timeout'] is expected
    assert captured['runtime_trace_context'] is trace


@pytest.mark.parametrize('error_type', [asyncio.CancelledError, ValueError, RuntimeUnhealthyError])
async def test_invocation_scope_is_released_on_failure(adapter, monkeypatch, error_type):
    async def invoke(*_args, **_kwargs):
        raise error_type('stopped')

    monkeypatch.setattr(adapter.module, 'call_runtime_agent', invoke)
    async with ExecutionContext(task_config={'is_validation': False}) as context:
        if error_type is RuntimeUnhealthyError:
            result = await adapter.run(input_for(adapter))
            assert result.termination_reason == 'ENV_ERROR'
        else:
            with pytest.raises(error_type, match='stopped'):
                await adapter.run(input_for(adapter))
        assert context.runtime_command_timeout_id is None


@pytest.mark.parametrize('missing_trace', [False, True])
async def test_invalid_invocation_scope_does_not_run_or_clear_another_owner(adapter, monkeypatch, missing_trace):
    async def must_not_run(*_args, **_kwargs):
        raise AssertionError('invalid scope reached the runtime')

    monkeypatch.setattr(adapter.module, 'call_runtime_agent', must_not_run)
    if missing_trace:
        monkeypatch.setattr(RemoteTraceContext, 'from_current_span', lambda: None)
    async with ExecutionContext() as context:
        previous = None if missing_trace else 'another-invocation'
        context.runtime_command_timeout_id = previous
        with pytest.raises(RuntimeError, match='trace' if missing_trace else 'timeout'):
            await adapter.run(input_for(adapter))
        assert context.runtime_command_timeout_id == previous


@pytest.mark.parametrize('flag', ['--mcp-config', '--mcp-config=file.json', '--strict-mcp-config'])
async def test_mcp_injection_rejects_conflicting_cli_before_runtime(adapter, flag):
    with pytest.raises(ValueError, match='explicit MCP'):
        await adapter.run(
            input_for(adapter, mcp_config_servers={'tools': {'command': 'server'}}, extra_cli_args=[flag])
        )


async def test_hycode_keeps_legacy_strict_mcp_equals_handling(monkeypatch):
    uploaded = []

    async def upload(path, content, _operation, *, mode):
        uploaded.append((path, json.loads(content)))

    monkeypatch.setattr(hycode, '_upload_runtime_text', upload)
    item = hycode.HyCodeInput(
        user_prompt='test', working_dir='/workspace', mcp_config_servers={'tools': {'command': 'server'}}
    )
    args = ['--strict-mcp-config=true']
    await hycode._prepare_strict_mcp_file(item, args)
    assert len(uploaded) == 1
    path, content = uploaded[0]
    assert content == {'mcpServers': item.mcp_config_servers}
    assert args == ['--strict-mcp-config=true', '--mcp-config', path, '--strict-mcp-config']


async def test_hyness_rejects_strict_mcp_equals_with_task_servers():
    with pytest.raises(ValueError, match='explicit MCP'):
        await hyness.run_hyness(
            hyness.HynessInput(
                user_prompt='test',
                working_dir='/workspace',
                packages=['@tencent/hyness@1.0.0'],
                mcp_config_servers={'tools': {'command': 'server'}},
                extra_cli_args=['--strict-mcp-config=true'],
            )
        )


async def test_mcp_input_merge_preserves_both_sources_without_mutating_input(adapter, monkeypatch):
    uploaded = {}
    captured = {}

    async def upload(_path, content, _operation, *, mode):
        uploaded.update(json.loads(content)['mcpServers'])

    async def invoke(*_args, **kwargs):
        captured.update(kwargs)
        return completed()

    monkeypatch.setattr(hycode, '_upload_runtime_text', upload)
    monkeypatch.setattr(adapter.module, 'call_runtime_agent', invoke)
    item = input_for(
        adapter,
        mcp_servers=[{'name': 'task', 'client': {'transport': 'streamable_http', 'url': 'http://task/mcp'}}],
        mcp_config_servers={'native': {'command': 'server', 'args': ['--stdio']}},
    )
    before = item.model_dump()
    await adapter.run(item)
    servers = uploaded if adapter.module is hycode else captured['mcp_config_servers']
    assert servers == {
        'task': {'type': 'http', 'url': 'http://task/mcp'},
        'native': {'command': 'server', 'args': ['--stdio']},
    }
    assert item.model_dump() == before


async def test_mcp_rejects_duplicate_names_across_sources(adapter):
    with pytest.raises(ValueError, match='Duplicate'):
        await adapter.run(
            input_for(
                adapter,
                mcp_servers=[{'name': 'same', 'client': {'transport': 'streamable_http', 'url': 'http://task/mcp'}}],
                mcp_config_servers={'same': {'command': 'server'}},
            )
        )
