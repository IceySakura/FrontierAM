"""Contracts shared with HyCode; model calls and process execution remain real at their boundaries."""

import asyncio
import hashlib
import inspect
import os
import sys

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.agents.hycode.agent import HyCodeInput
from nexus.extensions.agents.hyness import agent, installer, runtime_setup
from nexus.runtime.executors import command as executor
from nexus.trace import RemoteTraceContext


@pytest.mark.parametrize(
    ('code', 'subtype', 'error', 'expected'),
    [
        (0, 'success', False, 'ENV_DONE'),
        (64, 'error_max_turns', True, 'MAX_STEPS'),
        (65, 'error_tool_use', True, 'FORMAT_ERROR'),
        (66, 'error_context_length', True, 'TRUNCATION'),
        (1, 'error_during_execution', True, 'ENV_ERROR'),
    ],
)
def test_decoded_controls_match_hycode(code, subtype, error, expected):
    output = agent.classify_hyness_result(
        {
            'success': code == 0,
            'exit_code': code,
            'process_exit_code': int(error),
            'result_count': 1,
            'result_subtype': subtype,
            'result_is_error': error,
        }
    )
    assert output.termination_reason == expected
    assert output.process_exit_code == int(error)


def test_host_context_refusal_is_retained_and_startup_is_not_trainable():
    output = agent.classify_hyness_result(
        {
            'success': False,
            'exit_code': 1,
            'result_count': 1,
            'result_subtype': 'success',
            'result_is_error': True,
            'last_llm_error_code': 'context_length_exceeded',
        }
    )
    assert output.termination_reason == 'TRUNCATION'
    startup = agent.classify_hyness_result({'success': False, 'exit_code': 1, 'result_count': 0})
    assert startup.termination_reason == 'UNHANDLED_ERROR'
    assert startup.failure_kind == 'infrastructure'
    assert not startup.success


@pytest.mark.parametrize(
    ('training', 'explicit', 'expected'),
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    ],
)
async def test_invoke_deadline_matches_hycode_precedence(monkeypatch, training, explicit, expected):
    calls = []

    async def invoke(_target, **kwargs):
        calls.append(kwargs)
        assert 'version' not in kwargs
        return {
            'success': True,
            'exit_code': 0,
            'result_count': 1,
            'result_subtype': 'success',
            'result_is_error': False,
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', invoke)
    monkeypatch.setattr(
        agent.RemoteTraceContext, 'from_current_span', lambda: RemoteTraceContext(trace_id='t', parent_span_id='p')
    )
    context = ExecutionContext(task_config={'is_validation': not training})
    async with context:
        output = await agent.run_hyness(
            agent.HynessInput(
                user_prompt='test',
                working_dir='/workspace',
                packages=['@tencent/hyness@1.0.0'],
                disable_invoke_wait_timeout=explicit,
                stop_on_tool_error=True,
                pre_start_scripts=[{'code': 'echo ready'}],
            )
        )
        assert context.runtime_command_timeout_id is None
    assert calls[0]['disable_invoke_wait_timeout'] is expected
    assert calls[0]['stop_on_tool_error'] is True
    assert calls[0]['pre_start_scripts'][0]['code'] == 'echo ready'
    assert output.termination_reason == 'ENV_DONE'


def test_inputs_keep_hycode_task_contract_and_require_explicit_packages():
    assert set(HyCodeInput.model_fields) <= set(agent.HynessInput.model_fields)
    input_data = agent.HynessInput(user_prompt='test', working_dir='/workspace', max_turns=0)
    assert input_data.max_turns == 0
    with pytest.raises(ValueError, match='Specify pinned'):
        installer.normalize_package_specs([])
    with pytest.raises(ValueError, match='Remove the HyCode version'):
        agent.HynessInput(user_prompt='test', working_dir='/workspace', version='0.1.14')


@pytest.mark.parametrize('options', [{}, {'emit_spans': False}])
async def test_hycode_host_fields_do_not_leak_into_hyness_runtime(monkeypatch, options):
    captured = {}

    async def invoke(
        target,
        *,
        no_thinking_request_args=None,
        no_thinking_request_args_mode='replace',
        haiku_model=None,
        runtime_trace_context=None,
        disable_invoke_wait_timeout=False,
        **kwargs,
    ):
        # Bind the actual runtime signature so newly inherited fields cannot
        # silently pass a permissive mock and break sandbox startup.
        inspect.signature(target).bind(**kwargs)
        captured.update(kwargs)
        return {'success': True, 'exit_code': 0, 'result_count': 1}

    monkeypatch.setattr(agent, 'call_runtime_agent', invoke)
    item = agent.HynessInput(
        user_prompt='test', working_dir='/workspace', packages=['@tencent/hyness@1.0.0'], **options
    )
    result = await agent.run_hyness(item)
    assert result.success
    assert item.emit_spans is False
    assert 'emit_spans' not in item.model_dump()
    assert 'emit_spans' not in captured


def test_hyness_rejects_unsupported_hycode_span_file_capture():
    with pytest.raises(ValueError, match='emit_spans'):
        agent.HynessInput(user_prompt='test', working_dir='/workspace', emit_spans=True)


def test_persistent_session_rejects_conflicting_cli_flag():
    with pytest.raises(ValueError, match='session_persistence cannot be combined'):
        agent.HynessInput(
            user_prompt='test',
            working_dir='/workspace',
            session_persistence=True,
            extra_cli_args=['--no-session-persistence'],
        )


def test_legacy_opencc_defaults_survive_serialization():
    legacy = agent.HynessInput(user_prompt='test', working_dir='/workspace', packages=['@tencent/hyness@1.0.0'])
    assert legacy.app_config == '@tencent/hyness-opencc/apps/opencc/app.yaml'
    assert legacy.app_config_overlay == '@tencent/hyness-opencc/apps/opencc/overlays/training.yaml'
    assert legacy.harness_id == 'opencc'
    restored = agent.HynessInput.model_validate(legacy.model_dump())
    assert restored.model_dump() == legacy.model_dump()


@pytest.mark.parametrize('harness', [None, 'mini-swe-agent', 'swe-agent', 'workbuddy-cli', 'custom-agent'])
@pytest.mark.parametrize('app_config', ['@tencent/hyness/app-definitions/hyness/app.yaml', '/opt/custom/app.yaml'])
def test_custom_app_does_not_inherit_opencc_defaults(app_config, harness):
    config = {'app_config': app_config}
    if harness is not None:
        config['harness_id'] = harness
    selected = agent.HynessInput(user_prompt='test', working_dir='/workspace', **config)
    assert selected.harness_id == harness
    assert selected.app_config_overlay is None
    restored = agent.HynessInput.model_validate(selected.model_dump())
    assert restored.harness_id == harness
    assert restored.app_config_overlay is None


@pytest.mark.parametrize('overlay', [None, '/opt/policies/rl.yaml'])
def test_explicit_overlay_and_app_default_harness_are_preserved(overlay):
    selected = agent.HynessInput(
        user_prompt='test', working_dir='/workspace', app_config_overlay=overlay, harness_id=None
    )
    assert selected.app_config_overlay == overlay
    assert selected.harness_id is None


def test_other_harness_in_opencc_app_does_not_inherit_training_overlay():
    selected = agent.HynessInput(user_prompt='test', working_dir='/workspace', harness_id='custom-agent')
    assert selected.harness_id == 'custom-agent'
    assert selected.app_config_overlay is None


@pytest.mark.parametrize('field', ['app_config', 'app_config_overlay', 'harness_id'])
def test_empty_launch_selection_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        agent.HynessInput.model_validate({'user_prompt': 'test', 'working_dir': '/workspace', field: ''})


@pytest.mark.parametrize('spec', ['@tencent/hyness@latest', '@tencent/hyness@^1.0.0', '--flag', ''])
def test_install_requires_exact_version(spec):
    with pytest.raises(ValueError, match='exact npm'):
        installer.normalize_package_specs([spec])


@pytest.mark.parametrize('name', ['@tencent/hyness', '@example/agent-app', 'extra-plugin'])
def test_conflicting_versions_of_one_package_fail_before_runtime(name):
    packages = [f'{name}@1.0.0', f'{name}@2.0.0']
    with pytest.raises(ValueError, match='Conflicting versions'):
        installer.normalize_package_specs(packages)
    with pytest.raises(ValueError, match='Conflicting versions'):
        agent.HynessInput(user_prompt='test', working_dir='/workspace', packages=packages)


def test_input_canonicalizes_packages_without_equating_engine_and_app_versions():
    selected = agent.HynessInput(
        user_prompt='test',
        working_dir='/workspace',
        packages=[' @tencent/hyness@1.0.0 ', '@example/agent-app@2.0.0', '@tencent/hyness@1.0.0'],
    )
    assert selected.packages == ['@tencent/hyness@1.0.0', '@example/agent-app@2.0.0']
    assert selected.model_dump()['packages'] == selected.packages


def test_legacy_headless_flags_keep_one_protocol():
    args, turns, stop = runtime_setup.normalize_hycode_cli_args(
        [
            '--print',
            '--input-format',
            'text',
            '--output-format=stream-json',
            '--verbose',
            '--max-turns',
            '8',
            '--stop-on-tool-error',
            '--disallowedTools',
            'WebSearch,WebFetch',
        ],
        0,
        False,
    )
    assert args == ['--disallowedTools', 'WebSearch,WebFetch']
    assert turns == 0
    assert stop is True
    with pytest.raises(ValueError, match='Nexus controls'):
        runtime_setup.normalize_hycode_cli_args(['--result-protocol=sdk'], None, False)


async def test_task_mcp_conflict_is_rejected_before_runtime(monkeypatch):
    with pytest.raises(ValueError, match='explicit MCP'):
        await agent.run_hyness(
            agent.HynessInput(
                user_prompt='test',
                working_dir='/workspace',
                packages=['@tencent/hyness@1.0.0'],
                mcp_config_servers={'test': {'command': 'server'}},
                extra_cli_args=['--mcp-config', 'x.json'],
            )
        )


async def test_cancelled_process_is_reaped(tmp_path):
    pid_file = tmp_path / 'pid'
    task = asyncio.create_task(
        runtime_setup._capture_process(
            [
                sys.executable,
                '-c',
                f'import os,time;open({str(pid_file)!r},"w").write(str(os.getpid()));time.sleep(60)',
            ],
            str(tmp_path),
            {},
            60,
            None,
        )
    )
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_large_unicode_task_is_delivered_through_stdin(tmp_path):
    task = '修复文件\n' + 'x' * 200_000 + '\n$(do-not-execute) --app-config "quoted"\n'
    command = [sys.executable, '-c', 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())']
    code, stdout, stderr, timed_out = await runtime_setup._capture_process(
        command, str(tmp_path), {}, 5, None, stdin_input=task
    )
    assert code == 0
    assert not timed_out
    assert not stderr
    assert stdout.strip() == hashlib.sha256(task.encode()).hexdigest()


async def test_active_timeout_pauses_and_resumes(tmp_path):
    release = tmp_path / 'release'
    command = asyncio.create_task(
        runtime_setup._capture_process(
            [
                sys.executable,
                '-c',
                f'import pathlib,time\nwhile not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)',
            ],
            str(tmp_path),
            {},
            0.3,
            'hyness-pause',
        )
    )
    for _ in range(100):
        if await executor.pause_command_timeout('hyness-pause'):
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError('active timeout was not registered')
    await asyncio.sleep(0.4)
    assert not command.done()
    assert await executor.resume_command_timeout('hyness-pause')
    release.touch()
    code, _, _, timed_out = await asyncio.wait_for(command, 2)
    assert code == 0
    assert not timed_out


async def test_local_artifact_upload_is_hashed_and_checked(monkeypatch, tmp_path):
    import base64
    import hashlib

    from types import SimpleNamespace

    from nexus import core
    from nexus.extensions.agents.hyness.artifacts import runtime_package_spec

    path = tmp_path / 'engine.tgz'
    path.write_bytes(b'fixture-package')
    uploads = []

    class Runtime:
        async def upload_file(self, files, overwrite):
            uploads.extend(files)
            return SimpleNamespace(results=[SimpleNamespace(path=files[0].path, success=True)])

    class Context:
        async def get_runtime(self):
            return Runtime()

    monkeypatch.setattr(core, 'require_context', lambda: Context())
    remote = await runtime_package_spec(str(path))
    assert remote.endswith(hashlib.sha256(path.read_bytes()).hexdigest() + '.tgz')
    assert base64.b64decode(uploads[0].content) == path.read_bytes()
    assert await runtime_package_spec('@tencent/hyness@1.0.0') == '@tencent/hyness@1.0.0'


async def test_setup_retains_raw_protocol_failure_and_cleans_proxy(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nexus.runtime import context
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    paths = {'node': '/node', 'entry': '/hyness.js', 'prefix': str(tmp_path)}
    calls = []

    async def install(*_):
        return paths

    async def resolve(_transport, _installed, config):
        return str(tmp_path / 'app.yaml') if config else None

    async def verify(*_):
        return None

    async def start(self):
        calls.append('start')

    async def stop(self):
        calls.append('stop')

    async def capture(*_, **__):
        return 0, '{"type":"result","subtype":"success"}', 'original CLI stderr', False

    monkeypatch.setattr(runtime_setup, 'install_hyness', install)
    monkeypatch.setattr(runtime_setup, 'resolve_app_config', resolve)
    monkeypatch.setattr(runtime_setup, 'verify_app_engine', verify)
    monkeypatch.setattr(runtime_setup, '_capture_process', capture)
    monkeypatch.setattr(context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr(AnthropicProxy, 'start', start)
    monkeypatch.setattr(AnthropicProxy, 'stop', stop)
    monkeypatch.setenv('HOME', str(tmp_path))
    result = await runtime_setup.setup_and_run_hyness(
        task='test',
        packages=['@tencent/hyness@1.0.0'],
        app_config='app.yaml',
        app_config_overlay=None,
        harness_id='opencc',
        working_dir=str(tmp_path),
    )
    assert result['failure_stage'] == 'runtime_protocol'
    assert result['success'] is False
    assert result['process_exit_code'] == 0
    assert result['process_output'] == '{"type":"result","subtype":"success"}'
    assert result['process_stderr'] == 'original CLI stderr'
    assert 'original CLI stderr' in result['stderr']
    assert 'execution-result' in result['stderr']
    assert calls == ['start', 'stop']
    assert list((tmp_path / '.hyness/nexus').iterdir()) == []


async def test_signalled_result_keeps_hycode_scoreable_exit_policy():
    result = agent.classify_hyness_result(
        {
            'success': False,
            'exit_code': -9,
            'process_exit_code': -9,
            'result_count': 0,
        }
    )
    assert result.success is True
    assert result.termination_reason == 'ENV_DONE'
    assert result.scoreable_process_exit is True
    assert result.hycode_exit_name == 'SIGKILL'


def test_hyness_selects_existing_hycode_gateway_policy():
    from nexus.core.agent import get_agent_gateway_scaffold_policy

    assert get_agent_gateway_scaffold_policy('hyness') == 'hycode'
    assert get_agent_gateway_scaffold_policy('hyness', 'generic') == 'generic'
