import asyncio
import json

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.pi_agent import installer, runtime_setup
from nexus.extensions.agents.pi_agent.runtime_setup import (
    AGENT_DIR_ENV,
    SESSION_DIR_ENV,
    _build_pi_env,
    build_models_config,
    build_pi_argv,
    summarize_json_trace,
)


def test_models_config_routes_to_the_local_proxy() -> None:
    config = build_models_config(proxy_port=4321)
    provider = config['providers']['nexus-proxy']

    assert provider['baseUrl'] == 'http://127.0.0.1:4321/v1'
    assert provider['api'] == 'openai-completions'
    # A literal key marks the provider authenticated without a /login round-trip.
    assert provider['apiKey'] == 'dummy-key-for-proxy'
    assert provider['models'][0]['id'] == 'nexus-proxy-model'


def test_models_config_pins_openai_compat_knobs() -> None:
    """pi's detectCompat treats an unknown 127.0.0.1 host as stock OpenAI.

    The Nexus proxy forwards bodies to the host client untouched, so leaving
    these unset sends developer-role messages, max_completion_tokens, store and
    reasoning_effort upstream.  Verified against pi 0.84.1 on the wire.
    """
    compat = build_models_config(proxy_port=4321)['providers']['nexus-proxy']['compat']

    assert compat['supportsDeveloperRole'] is False
    assert compat['supportsReasoningEffort'] is False
    assert compat['supportsStore'] is False
    assert compat['maxTokensField'] == 'max_tokens'


def test_models_config_honors_model_limits() -> None:
    config = build_models_config(
        proxy_port=1,
        model_id='custom-model',
        context_window=131_072,
        max_tokens=8_192,
        reasoning=True,
    )
    model = config['providers']['nexus-proxy']['models'][0]

    assert model['id'] == 'custom-model'
    assert model['contextWindow'] == 131_072
    assert model['maxTokens'] == 8_192
    assert model['reasoning'] is True


def test_argv_defaults_to_an_isolated_run() -> None:
    argv = build_pi_argv('/opt/pi/pi', 'fix the bug')

    assert argv[0] == '/opt/pi/pi'
    assert argv[1:3] == ['--model', 'nexus-proxy/nexus-proxy-model']
    assert '--no-context-files' in argv
    assert '--no-approve' in argv
    assert '--no-extensions' in argv
    assert '--no-session' in argv
    # The prompt stays last so a task starting with '-' is not parsed as a flag.
    assert argv[-2:] == ['-p', 'fix the bug']


def test_argv_context_and_trust_are_independent_switches() -> None:
    """In pi, --no-approve does not suppress AGENTS.md and vice versa."""
    trusted_only = build_pi_argv('/opt/pi/pi', 't', trust_project_files=True, load_context_files=False)
    assert '--approve' in trusted_only
    assert '--no-context-files' in trusted_only

    context_only = build_pi_argv('/opt/pi/pi', 't', trust_project_files=False, load_context_files=True)
    assert '--no-approve' in context_only
    assert '--no-context-files' not in context_only


def test_argv_text_mode_omits_mode_flag() -> None:
    assert '--mode' not in build_pi_argv('/opt/pi/pi', 't', output_format='text')
    assert build_pi_argv('/opt/pi/pi', 't', output_format='json')[3:5] == ['--mode', 'json']


def test_argv_disables_tools_explicitly_when_the_allowlist_is_empty() -> None:
    """pi has no "zero tools" default: given neither --tools nor --no-tools it
    enables read/bash/edit/write, so omitting both would silently grant shell
    and write access to a caller that asked for none.  Verified against pi
    0.84.1: no tool flag advertises ['read','bash','edit','write'] on the wire,
    while --no-tools advertises none."""
    disabled = build_pi_argv('/opt/pi/pi', 't', tools=[])
    assert '--tools' not in disabled
    assert '--no-tools' in disabled

    disabled_none = build_pi_argv('/opt/pi/pi', 't', tools=None)
    assert '--no-tools' in disabled_none

    # A non-empty allowlist must not carry the kill switch alongside it.
    allowed = build_pi_argv('/opt/pi/pi', 't', tools=['read'])
    assert '--no-tools' not in allowed
    assert allowed[allowed.index('--tools') : allowed.index('--tools') + 2] == ['--tools', 'read']


def test_argv_supports_extra_args() -> None:
    extra = build_pi_argv('/opt/pi/pi', 't', extra_cli_args=['--verbose'])
    assert '--verbose' in extra
    assert extra.index('--verbose') < extra.index('-p')


def test_argv_thinking_is_forwarded() -> None:
    argv = build_pi_argv('/opt/pi/pi', 't', thinking='high')
    assert argv[argv.index('--thinking') : argv.index('--thinking') + 2] == ['--thinking', 'high']


def test_build_pi_env_keeps_nexus_isolation_authoritative(tmp_path: Path) -> None:
    env = _build_pi_env(
        working_dir='/workspace',
        session_root=tmp_path,
        extra_env={
            AGENT_DIR_ENV: '/outside/agent',
            SESSION_DIR_ENV: '/outside/sessions',
            'HOME': '/outside',
            'PI_OFFLINE': '0',
            'MY_TOKEN': 'keep-me',
        },
    )

    assert env[AGENT_DIR_ENV] == str(tmp_path / 'agent')
    assert env[SESSION_DIR_ENV] == str(tmp_path / 'sessions')
    assert env['HOME'] == str(tmp_path / 'home')
    assert env['PI_OFFLINE'] == '1'
    assert env['PI_SKIP_VERSION_CHECK'] == '1'
    # Non-protected user configuration still passes through.
    assert env['MY_TOKEN'] == 'keep-me'


def test_build_pi_env_creates_config_dirs(tmp_path: Path) -> None:
    env = _build_pi_env(working_dir='/workspace', session_root=tmp_path, extra_env=None)
    assert Path(env[AGENT_DIR_ENV]).is_dir()
    assert Path(env[SESSION_DIR_ENV]).is_dir()
    assert Path(env['HOME']).is_dir()


def test_summarize_json_trace_uses_runtime_event_names() -> None:
    """Event names are taken from a real pi 0.84.1 --mode json run.

    The exported TypeScript union suggests tool_call/tool_result; the runtime
    actually emits tool_execution_start/tool_execution_end, and one turn can
    span several message_start/message_end pairs.
    """
    stdout = '\n'.join(
        json.dumps(event)
        for event in [
            {'type': 'session'},
            {'type': 'agent_start'},
            {'type': 'turn_start'},
            {'type': 'message_start'},
            {'type': 'message_update'},
            {'type': 'message_end'},
            {'type': 'tool_execution_start'},
            {'type': 'tool_execution_end'},
            {'type': 'turn_end'},
            {'type': 'turn_start'},
            {'type': 'message_start'},
            {'type': 'message_end'},
            {'type': 'turn_end'},
            {
                'type': 'agent_end',
                'messages': [
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'ignored'}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'the answer is 42'}]},
                ],
            },
            {'type': 'agent_settled'},
        ]
    )

    trace = summarize_json_trace(stdout)

    assert trace['turns'] == 2
    assert trace['tool_calls'] == 1
    assert trace['final_text'] == 'the answer is 42'
    assert trace['malformed_lines'] == 0
    assert trace['event_counts']['message_start'] == 2


def test_summarize_json_trace_tolerates_noise() -> None:
    """pi can interleave non-JSON warnings; they must not sink the whole trace."""
    stdout = '\n'.join([
        'Warning: something human-readable',
        json.dumps({'type': 'turn_start'}),
        '{"broken": ',
        '[]',
        '',
        json.dumps({'type': 'agent_end', 'messages': []}),
    ])

    trace = summarize_json_trace(stdout)

    assert trace['turns'] == 1
    assert trace['malformed_lines'] == 3
    assert trace['final_text'] == ''


def test_summarize_json_trace_handles_empty_output() -> None:
    trace = summarize_json_trace('')
    assert trace['turns'] == 0
    assert trace['tool_calls'] == 0
    assert trace['event_counts'] == {}


class _FakeProxy:
    instances: list[dict[str, Any]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        _FakeProxy.instances.append({'kwargs': kwargs, 'proxy': self})

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def _reset_proxy_instances():
    _FakeProxy.instances = []
    yield
    _FakeProxy.instances = []


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binary: Path) -> None:
    async def fake_install_pi_via_npm(**kwargs):
        return binary

    monkeypatch.setattr(installer, 'install_pi_via_npm', fake_install_pi_via_npm)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio='socket'))
    monkeypatch.setattr('nexus.runtime.server.services.openai_proxy.OpenAIProxy', _FakeProxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 4321)
    monkeypatch.setattr(runtime_setup.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path / 'session'))


@pytest.mark.asyncio
async def test_setup_and_run_pi_writes_config_runs_and_stops_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    binary = tmp_path / 'installed' / 'pi'
    binary.parent.mkdir()
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        captured['cmd'] = cmd
        captured['run'] = kwargs
        # models.json must already exist by the time pi starts.
        agent_dir = Path(kwargs['env'][AGENT_DIR_ENV])
        captured['models_json'] = json.loads((agent_dir / 'models.json').read_text())
        return 0, json.dumps({'type': 'agent_end', 'messages': []}), ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi('do the task', working_dir=str(tmp_path / 'workspace'))

    assert result['success'] is True
    assert result['exit_code'] == 0
    proxy = _FakeProxy.instances[0]
    assert proxy['proxy'].started is True
    assert proxy['proxy'].stopped is True
    assert proxy['kwargs']['listen_port'] == 4321
    assert proxy['kwargs']['models'] == ['nexus-proxy-model']

    # The generated config must point at the port the proxy actually bound.
    assert captured['models_json']['providers']['nexus-proxy']['baseUrl'] == 'http://127.0.0.1:4321/v1'
    assert captured['cmd'][0] == str(binary)
    assert captured['cmd'][-2:] == ['-p', 'do the task']
    # The per-run session root is removed once the run finishes.
    assert not (tmp_path / 'session').exists()


@pytest.mark.asyncio
async def test_setup_and_run_pi_summarizes_json_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        stdout = '\n'.join([
            json.dumps({'type': 'turn_start'}),
            json.dumps({'type': 'tool_execution_start'}),
            json.dumps({'type': 'tool_execution_end'}),
            json.dumps({
                'type': 'agent_end',
                'messages': [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'done'}]}],
            }),
        ])
        return 0, stdout, ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert result['trace']['turns'] == 1
    assert result['trace']['tool_calls'] == 1
    assert result['trace']['final_text'] == 'done'


@pytest.mark.asyncio
async def test_setup_and_run_pi_skips_trace_in_text_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        return 0, 'just the final answer', ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi(
        't',
        working_dir=str(tmp_path / 'workspace'),
        output_format='text',
    )

    assert 'trace' not in result
    assert result['output'] == 'just the final answer'


@pytest.mark.asyncio
async def test_setup_and_run_pi_reports_install_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def failing_install(**kwargs):
        raise RuntimeError('registry unreachable')

    monkeypatch.setattr(installer, 'install_pi_via_npm', failing_install)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path))

    assert result['success'] is False
    assert result['exit_code'] == -1
    assert 'registry unreachable' in result['stderr']


@pytest.mark.asyncio
async def test_setup_and_run_pi_stops_proxy_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A crash mid-run must not leak the proxy or the session dir."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def exploding_run(cmd, **kwargs):
        raise RuntimeError('sandbox died')

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', exploding_run)

    with pytest.raises(RuntimeError, match='sandbox died'):
        await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert _FakeProxy.instances[0]['proxy'].stopped is True
    assert not (tmp_path / 'session').exists()


@pytest.mark.asyncio
async def test_setup_and_run_pi_stops_proxy_when_start_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OpenAIProxy.start() spawns a detached server task before it returns, so a
    failure partway through must still release the port."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    class ExplodingProxy(_FakeProxy):
        async def start(self):
            raise RuntimeError('port already bound')

    async def fake_install_pi_via_npm(**kwargs):
        return binary

    monkeypatch.setattr(installer, 'install_pi_via_npm', fake_install_pi_via_npm)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio='socket'))
    monkeypatch.setattr('nexus.runtime.server.services.openai_proxy.OpenAIProxy', ExplodingProxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 4321)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert result['success'] is False
    assert result['exit_code'] == -1
    assert 'port already bound' in result['stderr']
    assert 'trace' not in result
    # The half-started proxy must have been torn down, not left holding the port.
    assert _FakeProxy.instances[0]['proxy'].stopped is True


@pytest.mark.asyncio
async def test_setup_and_run_pi_propagates_cancellation_after_stopping_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation must not be converted into a normal failure result."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    class CancellingProxy(_FakeProxy):
        async def start(self):
            raise asyncio.CancelledError

    async def fake_install_pi_via_npm(**kwargs):
        return binary

    monkeypatch.setattr(installer, 'install_pi_via_npm', fake_install_pi_via_npm)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio='socket'))
    monkeypatch.setattr('nexus.runtime.server.services.openai_proxy.OpenAIProxy', CancellingProxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 4321)

    with pytest.raises(asyncio.CancelledError):
        await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert _FakeProxy.instances[0]['proxy'].stopped is True


@pytest.mark.asyncio
async def test_run_pi_streaming_reports_timeout(tmp_path: Path) -> None:
    """A hung pi must be killed, reported as exit -1, and annotated in stderr."""
    script = tmp_path / 'sleeper.sh'
    script.write_text('#!/bin/sh\necho started\nsleep 30\n')
    script.chmod(0o755)

    exit_code, stdout, stderr = await runtime_setup._run_pi_streaming(
        ['/bin/sh', str(script)],
        cwd=str(tmp_path),
        env={},
        timeout=1.0,
    )

    assert exit_code == -1
    assert 'started' in stdout
    assert '[timeout after 1s]' in stderr


@pytest.mark.asyncio
async def test_run_pi_streaming_captures_exit_code_and_streams(tmp_path: Path) -> None:
    script = tmp_path / 'noisy.sh'
    script.write_text('#!/bin/sh\necho to-stdout\necho to-stderr >&2\nexit 3\n')
    script.chmod(0o755)

    exit_code, stdout, stderr = await runtime_setup._run_pi_streaming(
        ['/bin/sh', str(script)],
        cwd=str(tmp_path),
        env={},
        timeout=30.0,
    )

    assert exit_code == 3
    assert 'to-stdout' in stdout
    assert 'to-stderr' in stderr


@pytest.mark.asyncio
async def test_pipe_to_logger_recovers_from_oversized_line() -> None:
    """StreamReader.readline() clears the buffer before raising ValueError, so a
    single huge JSONL event must not stall or drop the lines after it."""
    reader = asyncio.StreamReader(limit=64)
    reader.feed_data(b'X' * 500 + b'\n')
    reader.feed_data(b'after-the-big-one\n')
    reader.feed_data(b'and-another\n')
    reader.feed_eof()

    buffer: list[str] = []
    await asyncio.wait_for(runtime_setup._pipe_to_logger(reader, '[pi] ', 20, buffer), timeout=5)

    assert buffer == ['after-the-big-one', 'and-another']


def test_build_pi_env_filters_every_protected_var(tmp_path: Path) -> None:
    """An adversarial extra_env naming every protected key must not break routing."""
    hostile = dict.fromkeys(runtime_setup._PROTECTED_ENV, '/hijacked')
    hostile['KEEP_ME'] = 'kept'

    env = _build_pi_env(working_dir='/workspace', session_root=tmp_path, extra_env=hostile)

    assert '/hijacked' not in env.values()
    assert env['KEEP_ME'] == 'kept'
    for key in runtime_setup._PROTECTED_ENV:
        assert env[key] != '/hijacked'


@pytest.mark.asyncio
async def test_setup_and_run_pi_summarizes_the_real_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """trace must be derived from the captured stdout, not fabricated."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')
    seen: dict[str, Any] = {}

    real_summarize = runtime_setup.summarize_json_trace

    def spying_summarize(stdout: str):
        seen['stdout'] = stdout
        return real_summarize(stdout)

    async def fake_run_pi_streaming(cmd, **kwargs):
        return 0, json.dumps({'type': 'turn_start'}), ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)
    monkeypatch.setattr(runtime_setup, 'summarize_json_trace', spying_summarize)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert seen['stdout'] == json.dumps({'type': 'turn_start'})
    assert result['trace']['turns'] == 1


def _error_agent_end(stop_reason: str, error_message: str | None = None) -> str:
    """The exact shape pi 0.84.1 emits when a turn fails (captured from a real run
    against a 500-ing endpoint): content is empty, stopReason carries the failure."""
    message: dict[str, Any] = {
        'role': 'assistant',
        'content': [],
        'api': 'openai-completions',
        'provider': 'nexus-proxy',
        'model': 'nexus-proxy-model',
        'stopReason': stop_reason,
    }
    if error_message is not None:
        message['errorMessage'] = error_message
    return '\n'.join([
        json.dumps({'type': 'turn_start'}),
        json.dumps({'type': 'agent_end', 'messages': [{'role': 'user', 'content': []}, message], 'willRetry': False}),
    ])


def test_summarize_json_trace_reports_stop_reason_and_error() -> None:
    trace = summarize_json_trace(
        _error_agent_end('error', '500: {"message":"upstream model exploded","type":"server_error"}')
    )

    assert trace['stop_reason'] == 'error'
    assert 'upstream model exploded' in trace['error_message']


def test_summarize_json_trace_reports_success_stop_reason() -> None:
    stdout = '\n'.join([
        json.dumps({'type': 'turn_start'}),
        json.dumps({
            'type': 'agent_end',
            'messages': [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'done'}], 'stopReason': 'stop'},
            ],
        }),
    ])

    trace = summarize_json_trace(stdout)

    assert trace['stop_reason'] == 'stop'
    assert trace['error_message'] is None
    assert trace['final_text'] == 'done'


def test_summarize_json_trace_stop_reason_defaults_to_none() -> None:
    trace = summarize_json_trace(json.dumps({'type': 'turn_start'}))
    assert trace['stop_reason'] is None
    assert trace['error_message'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('stop_reason', sorted(runtime_setup.FAILED_STOP_REASONS))
async def test_setup_and_run_pi_fails_on_error_stop_reason_despite_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop_reason: str,
) -> None:
    """pi only maps a failed turn to exit 1 in text mode (print-mode.ts guards
    that branch with mode === 'text'), so in JSON mode a provider or model error
    exits 0. Verified against pi 0.84.1: the same upstream 500 exits 0 under
    --mode json and 1 under text mode."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        return 0, _error_agent_end(stop_reason, 'upstream model exploded'), ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert result['success'] is False, 'a failed turn must not be reported as success'
    # The process really did exit 0; only `success` is corrected.
    assert result['exit_code'] == 0
    assert 'upstream model exploded' in result['stderr']
    assert result['trace']['stop_reason'] == stop_reason


@pytest.mark.asyncio
async def test_setup_and_run_pi_reports_stop_reason_without_error_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An aborted turn may carry no errorMessage; stderr must still say why."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        return 0, _error_agent_end('aborted'), ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert result['success'] is False
    assert 'aborted' in result['stderr']


@pytest.mark.asyncio
async def test_setup_and_run_pi_keeps_success_on_normal_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The failure check must not turn healthy runs into failures."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        stdout = '\n'.join([
            json.dumps({'type': 'turn_start'}),
            json.dumps({
                'type': 'agent_end',
                'messages': [
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}], 'stopReason': 'stop'},
                ],
            }),
        ])
        return 0, stdout, ''

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi('t', working_dir=str(tmp_path / 'workspace'))

    assert result['success'] is True
    assert result['stderr'] == ''
    assert result['trace']['final_text'] == 'ok'


@pytest.mark.asyncio
async def test_setup_and_run_pi_text_mode_still_trusts_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Text mode needs no correction: pi maps a failed turn to exit 1 itself."""
    binary = tmp_path / 'pi'
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_run_pi_streaming(cmd, **kwargs):
        return 1, '', 'Request error'

    _patch_runtime(monkeypatch, tmp_path, binary)
    monkeypatch.setattr(runtime_setup, '_run_pi_streaming', fake_run_pi_streaming)

    result = await runtime_setup.setup_and_run_pi(
        't',
        working_dir=str(tmp_path / 'workspace'),
        output_format='text',
    )

    assert result['success'] is False
    assert 'trace' not in result
