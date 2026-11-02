import asyncio
import json
import os
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.core import UserInputRequest, agent_registry
from nexus.extensions.agents import installed_common
from nexus.extensions.agents.claude_code import agent as claude_agent
from nexus.extensions.agents.claude_code import runtime_setup as claude_runtime_setup
from nexus.extensions.agents.claude_code import session_runtime as claude_session_runtime
from nexus.extensions.agents.claude_code.agent import ClaudeCodeMultiturnInput
from nexus.extensions.agents.claude_code.session_runtime import (
    _ClaudeCodeSession,
    _build_hook_output,
    _build_interaction_hint,
    _build_session_command,
    _read_stderr,
    _read_stdout,
    _record_trace_event,
)
from nexus.extensions.agents.claude_code.stream_json import encode_user_message, summarize_turn_events
from nexus.extensions.agents.installed_common import multiturn
from nexus.extensions.agents.installed_common.multiturn import (
    build_user_input_request,
    normalize_stream_json_session_id,
)
from nexus.extensions.agents.installed_common.stream_trace import StreamTraceBuffer


async def _end_input(_hint: str | None) -> str | None:
    return None


class _FakeStdin:
    def __init__(self):
        self.closed = False
        self.writes: list[bytes] = []

    def write(self, data):
        self.writes.append(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _BlockingStream:
    async def readline(self):
        await asyncio.Event().wait()
        return b''


class _EofStream:
    async def readline(self):
        return b''


class _ErrorStream:
    def __init__(self, message: str):
        self.message = message

    async def readline(self):
        raise ValueError(self.message)


class _FakeProc:
    def __init__(self, *, stdout=None, stderr=None, wait_blocks: bool = False):
        self.stdin = _FakeStdin()
        self.stdout = stdout or _BlockingStream()
        self.stderr = stderr or _BlockingStream()
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_blocks = wait_blocks

    async def wait(self):
        if self.wait_blocks:
            await asyncio.Event().wait()
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeProxy:
    instances: list['_FakeProxy'] = []

    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def test_session_input_accepts_nullable_user_input_provider():
    input_data = ClaudeCodeMultiturnInput(
        user_prompt='fix the bug',
        working_dir='/workspace',
        get_user_input=_end_input,
    )

    assert input_data.get_user_input is _end_input


def test_claude_code_multiturn_agent_is_registered():
    assert agent_registry.get('claude_code_multiturn')


def test_normalize_claude_session_id_returns_uuid():
    generated = normalize_stream_json_session_id(None, namespace='claude-code-session')
    stable = normalize_stream_json_session_id('nexus-session-1', namespace='claude-code-session')

    assert generated.count('-') == 4
    assert stable.count('-') == 4
    assert stable == normalize_stream_json_session_id('nexus-session-1', namespace='claude-code-session')


def test_encode_user_message_matches_stream_json_shape():
    payload = json.loads(encode_user_message('next step', session_id='session-1').decode())

    assert payload == {
        'type': 'user',
        'message': {'role': 'user', 'content': 'next step'},
        'parent_tool_use_id': None,
        'session_id': 'session-1',
    }


def test_encode_user_message_can_send_deferred_tool_result():
    payload = json.loads(
        encode_user_message('Alpha', session_id='session-1', parent_tool_use_id='tooluse_ask_1').decode()
    )

    assert payload == {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'tooluse_ask_1',
                    'content': 'Alpha',
                    'is_error': False,
                }
            ],
        },
        'parent_tool_use_id': 'tooluse_ask_1',
        'session_id': 'session-1',
    }


def test_record_stream_event_uses_generic_agent_span(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'span-1')

    def fake_update_span(span_id, *, events):
        captured['span_id'] = span_id
        captured['events'] = events

    monkeypatch.setattr('nexus.trace.update_span', fake_update_span)

    session = SimpleNamespace(session_id='session-1', cwd='/workspace')
    _record_trace_event(session, {'type': 'assistant', 'message': {'content': 'hello'}})

    assert captured['span_id'] == 'span-1'
    event = captured['events'][0]
    assert event.name == 'agent.stream'
    assert event.attributes['agent'] == 'claude_code'
    assert event.attributes['session_id'] == 'session-1'
    assert event.attributes['cwd'] == '/workspace'
    assert event.attributes['event_type'] == 'assistant'
    assert event.attributes['event']['message']['content'] == 'hello'


def test_record_stream_event_merges_adjacent_partial_text(monkeypatch):
    captured: list[list] = []

    monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr('nexus.trace.update_span', lambda _span_id, *, events: captured.append(events))
    session = SimpleNamespace(session_id='session-1', cwd='/workspace')

    for text in ('hello', ' ', 'world'):
        _record_trace_event(
            session,
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'text_delta', 'text': text},
                },
            },
        )
    _record_trace_event(session, {'type': 'result'})

    assert len(captured) == 2
    assert captured[0][0].attributes['event']['event']['delta']['text'] == 'hello world'
    assert captured[1][0].attributes['event_type'] == 'result'


def test_record_stream_event_can_be_disabled(monkeypatch):
    monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr('nexus.trace.update_span', lambda *_args, **_kwargs: pytest.fail('trace update disabled'))

    session = SimpleNamespace(session_id='session-1', cwd='/workspace')
    _record_trace_event(session, {'type': 'assistant'}, enabled=False)


def test_summarize_turn_events_prefers_assistant_text():
    summary = summarize_turn_events(
        [
            {'type': 'system', 'session_id': 'session-1'},
            {
                'type': 'assistant',
                'message': {
                    'content': [
                        {'type': 'text', 'text': 'hello'},
                        {'type': 'tool_use', 'name': 'Bash'},
                        {'type': 'text', 'text': ' world'},
                    ],
                },
            },
            {'type': 'result', 'session_id': 'session-1', 'is_error': False, 'result': 'fallback'},
        ]
    )

    assert summary['session_id'] == 'session-1'
    assert summary['assistant_text'] == 'hello world'
    assert summary['result_text'] == 'fallback'
    assert summary['is_error'] is False


def test_stream_json_command_uses_stdin_protocol_without_print_prompt():
    cmd = _build_session_command(
        '/opt/claude',
        session_id='session-1',
        extra_cli_args=['--model', 'sonnet'],
        include_partial_messages=True,
        include_hook_events=True,
    )

    assert '-p' not in cmd
    assert '--print' not in cmd
    assert cmd[:5] == ['/opt/claude', '--output-format', 'stream-json', '--verbose', '--session-id']
    assert '--dangerously-skip-permissions' in cmd
    assert cmd[-2:] == ['--input-format', 'stream-json']
    assert '--include-partial-messages' in cmd
    assert '--include-hook-events' in cmd


def test_single_run_command_adds_print_by_default():
    cmd = claude_runtime_setup._build_single_run_command(
        '/opt/claude',
        extra_cli_args=['--output-format', 'json', '--verbose'],
    )

    # The prompt is fed via stdin, so it must NOT appear in argv (no trailing `-- <task>`).
    assert cmd == [
        '/opt/claude',
        '--dangerously-skip-permissions',
        '--print',
        '--output-format',
        'json',
        '--verbose',
    ]
    assert '--' not in cmd


def test_single_run_command_does_not_duplicate_print():
    cmd = claude_runtime_setup._build_single_run_command(
        '/opt/claude',
        extra_cli_args=['--print', '--output-format', 'json'],
    )
    short_cmd = claude_runtime_setup._build_single_run_command(
        '/opt/claude',
        extra_cli_args=['-p', '--output-format', 'json'],
    )

    assert cmd.count('--print') == 1
    assert short_cmd.count('--print') == 0
    assert short_cmd.count('-p') == 1


@pytest.mark.asyncio
async def test_claude_single_run_feeds_prompt_via_stdin(monkeypatch, tmp_path):
    # The prompt must reach claude through the child's stdin (not argv), so an
    # agent-issued `pkill -f <substring-of-prompt>` can't match claude's cmdline.
    proc = _FakeProc(stdout=_EofStream(), stderr=_EofStream())
    subprocess_kwargs: dict = {}

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        subprocess_kwargs.update(_kwargs)
        return proc

    monkeypatch.setattr(claude_runtime_setup.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})

    prompt = 'do the task; ignore pkill -9 -f ./executable'
    exit_code, _stdout, _stderr = await claude_runtime_setup._run_claude_streaming(
        ['/tmp/claude', '--print'],
        str(tmp_path),
        {},
        timeout=5,
        stdin_input=prompt,
    )

    assert exit_code == 0
    # stdin opened as a pipe, whole prompt written, then closed.
    assert subprocess_kwargs['stdin'] is claude_runtime_setup.subprocess.PIPE
    assert b''.join(proc.stdin.writes) == prompt.encode('utf-8')
    assert proc.stdin.closed


@pytest.mark.asyncio
async def test_claude_single_run_uses_devnull_stdin_by_default(monkeypatch, tmp_path):
    # Without stdin_input, stdin stays DEVNULL and nothing is written to the child.
    proc = _FakeProc(stdout=_EofStream(), stderr=_EofStream())
    subprocess_kwargs: dict = {}

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        subprocess_kwargs.update(_kwargs)
        return proc

    monkeypatch.setattr(claude_runtime_setup.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})

    exit_code, _stdout, _stderr = await claude_runtime_setup._run_claude_streaming(
        ['/tmp/claude', '--print'],
        str(tmp_path),
        {},
        timeout=5,
    )

    assert exit_code == 0
    assert subprocess_kwargs['stdin'] is claude_runtime_setup.subprocess.DEVNULL
    assert proc.stdin.writes == []
    assert not proc.stdin.closed


def test_stream_json_command_does_not_pass_session_id_with_resume():
    cmd = _build_session_command(
        '/opt/claude',
        session_id='session-1',
        extra_cli_args=['--resume', 'session-1'],
        include_partial_messages=True,
        include_hook_events=True,
    )

    assert '--session-id' not in cmd
    assert cmd[-4:-2] == ['--resume', 'session-1']
    assert cmd[-2:] == ['--input-format', 'stream-json']


def test_stream_json_command_keeps_session_id_for_resume_fork():
    cmd = _build_session_command(
        '/opt/claude',
        session_id='session-2',
        extra_cli_args=['--resume', 'session-1', '--fork-session'],
        include_partial_messages=False,
        include_hook_events=False,
    )

    assert cmd[cmd.index('--session-id') + 1] == 'session-2'
    assert '--fork-session' in cmd


@pytest.mark.asyncio
async def test_claude_single_run_reports_stdout_reader_error(monkeypatch, tmp_path):
    proc = _FakeProc(stdout=_ErrorStream('line too long'), stderr=_EofStream())
    proc.pid = 123456789

    async def kill_children(pid):
        assert pid == proc.pid

    monkeypatch.setattr('nexus.utils.shell.kill_all_child_processes', kill_children)
    subprocess_kwargs: dict = {}

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        subprocess_kwargs.update(_kwargs)
        return proc

    monkeypatch.setattr(claude_runtime_setup.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})

    exit_code, stdout, stderr = await claude_runtime_setup._run_claude_streaming(
        ['/tmp/claude'],
        str(tmp_path),
        {},
        timeout=1,
    )

    assert exit_code == -1
    assert stdout == ''
    assert 'stdout reader failed: ValueError: line too long' in stderr
    assert subprocess_kwargs['limit'] == installed_common.DEFAULT_CLI_STREAM_READER_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [None, 'latest'])
async def test_claude_session_uses_expanded_subprocess_stream_limit(monkeypatch, tmp_path, version):
    proc = _FakeProc()
    subprocess_kwargs: dict = {}

    async def fake_install(**_kwargs):
        assert _kwargs['version'] == version
        return Path('/tmp/claude')

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        subprocess_kwargs.update(_kwargs)
        return proc

    async def fake_initialize(self, **_kwargs):
        return {}

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.installer.install_claude_code_via_npm',
        fake_install,
    )
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setitem(
        sys.modules,
        'nexus.runtime.server.services.anthropic_proxy',
        SimpleNamespace(AnthropicProxy=_FakeProxy),
    )
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})
    monkeypatch.setattr(claude_session_runtime.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr(_ClaudeCodeSession, 'initialize', fake_initialize)
    _FakeProxy.instances.clear()

    session = await claude_session_runtime._start_session(
        session_id='session-1',
        working_dir=str(tmp_path),
        timeout=10,
        version=version,
        npm_registry=None,
        binary_url=None,
        binary_checksum=None,
        mcp_servers=None,
        extra_env=None,
        extra_cli_args=None,
        config_file_patches=None,
        pre_start_scripts=None,
        conversion_log_level='INFO',
        thinking_preset='hunyuan',
        include_partial_messages=False,
        include_hook_events=False,
        enable_user_input_hooks=False,
        record_trace_events=True,
    )

    try:
        assert subprocess_kwargs['env']['PATH'].split(os.pathsep)[0] == str(Path('/tmp/claude').parent)
        assert subprocess_kwargs['limit'] == installed_common.DEFAULT_CLI_STREAM_READER_LIMIT
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_claude_stdout_reader_error_is_reported_without_waiting_for_process():
    proc = _FakeProc(stdout=_ErrorStream('line too long'), wait_blocks=True)
    session = _ClaudeCodeSession(
        session_id='session-1',
        proc=proc,
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    await asyncio.wait_for(_read_stdout(session), timeout=0.2)

    event = await asyncio.wait_for(session.events.get(), timeout=0.2)
    assert event['type'] == '_stdout_read_error'
    assert 'line too long' in event['error']
    assert proc.terminated is True


@pytest.mark.asyncio
async def test_claude_send_user_message_fails_fast_on_stdout_reader_error():
    session = _ClaudeCodeSession(
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    await session.events.put({'type': '_stdout_read_error', 'error': 'line too long'})

    with pytest.raises(RuntimeError, match='stdout reader failed: line too long'):
        await session.send_user_message('hello', timeout=1)


@pytest.mark.asyncio
async def test_claude_session_turn_flushes_pending_stream_trace_when_cancelled(monkeypatch):
    session = _ClaudeCodeSession(
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    session._stream_trace_buffer = StreamTraceBuffer(
        pending_span_id='span-1',
        pending_event={'type': 'stream_event', 'event': {'delta': {'text': 'final'}}},
        pending_key=('session-1', None, 0),
        pending_text_bytes=5,
    )
    captured: list[str] = []

    async def cancel_turn(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(session, 'send_user_message', cancel_turn)
    monkeypatch.setattr('nexus.trace.update_span', lambda span_id, *, events: captured.append(span_id))
    claude_session_runtime._SESSIONS['session-1'] = session
    try:
        with pytest.raises(asyncio.CancelledError):
            await claude_session_runtime.run_claude_code_session_turn(
                session_id='session-1',
                prompt='hello',
            )
    finally:
        claude_session_runtime._SESSIONS.pop('session-1', None)

    assert captured == ['span-1']


@pytest.mark.asyncio
async def test_claude_stderr_reader_error_is_recorded_and_reported():
    session = _ClaudeCodeSession(
        session_id='session-1',
        proc=_FakeProc(stderr=_ErrorStream('stderr exploded')),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    await _read_stderr(session)

    assert session.stderr_lines[-1] == 'stderr reader failed: ValueError: stderr exploded'
    event = await asyncio.wait_for(session.events.get(), timeout=0.2)
    assert event['type'] == '_stderr_read_error'


def test_ask_user_question_hint_includes_options():
    hint = _build_interaction_hint(
        'AskUserQuestion',
        {
            'questions': [
                {
                    'header': 'Choice',
                    'question': 'Which option?',
                    'options': [{'label': 'Alpha'}, {'label': 'Beta'}],
                }
            ]
        },
    )

    assert 'Choice: Which option?' in hint
    assert 'Options: Alpha, Beta' in hint


def test_exit_plan_mode_hook_output_defers_for_user_input():
    output = _build_hook_output('ExitPlanMode', 'approve')

    assert output['hookSpecificOutput']['permissionDecision'] == 'defer'


def test_ask_user_question_hook_output_defers_for_user_input():
    output = _build_hook_output('AskUserQuestion', 'Use Alpha')

    hook_output = output['hookSpecificOutput']
    assert hook_output['permissionDecision'] == 'defer'


async def test_user_input_request_includes_deferred_tool_metadata():
    request = await build_user_input_request(
        agent_label='Claude Code',
        output='',
        result={
            'events': [{'type': 'result'}],
            'result_event': {
                'type': 'result',
                'deferred_tool_use': {
                    'id': 'tooluse_ask_1',
                    'name': 'AskUserQuestion',
                    'input': {
                        'questions': [
                            {
                                'header': 'Choice',
                                'question': 'Which option?',
                            }
                        ]
                    },
                },
            },
        },
        session_id='session-1',
        turn=1,
    )

    assert request.source == 'hook'
    assert request.session_id == 'session-1'
    assert request.hook is not None
    assert request.hook['name'] == 'AskUserQuestion'
    assert request.hint == 'Claude Code asks for user input.\n1. Choice: Which option?'
    assert request.metadata == {
        'turn': 1,
        'agent': 'Claude Code',
        'latest_screen': None,
        'recent_events': [],
    }


@pytest.mark.asyncio
async def test_session_agent_stops_when_input_provider_returns_empty(monkeypatch):
    calls: list[str] = []
    requests: list[UserInputRequest | str | None] = []
    runtime_kwargs: list[dict] = []

    async def fake_call_runtime_agent(target, /, **kwargs):
        calls.append(target.__name__)
        runtime_kwargs.append(kwargs)
        if target.__name__ == 'run_claude_code_session_turn':
            return {
                'success': True,
                'session_id': 'session-1',
                'output': 'turn output',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 0.1,
                'events': [{'type': 'result', 'session_id': 'session-1'}],
            }
        return {'success': True, 'session_id': 'session-1', 'stderr': '', 'exit_code': 0}

    async def get_user_input(hint: UserInputRequest | str | None) -> str | None:
        requests.append(hint)
        return ''

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code_multiturn(
        ClaudeCodeMultiturnInput(
            user_prompt='first prompt',
            working_dir='/workspace',
            session_id='session-1',
            get_user_input=get_user_input,
        )
    )

    assert output.success is True
    assert output.turns == 1
    assert output.ended_by_user is True
    assert output.output == 'turn output'
    assert len(requests) == 1
    assert isinstance(requests[0], UserInputRequest)
    assert requests[0].hint == 'turn output'
    assert requests[0].session_id == normalize_stream_json_session_id('session-1', namespace='claude-code-session')
    assert calls == ['run_claude_code_session_turn', 'close_claude_code_session']
    assert runtime_kwargs[0]['enable_user_input_hooks'] is False
    assert runtime_kwargs[0]['record_trace_events'] is True


@pytest.mark.asyncio
async def test_session_agent_sends_deferred_tool_answer_as_tool_result(monkeypatch):
    runtime_kwargs: list[dict] = []

    async def fake_call_runtime_agent(target, /, **kwargs):
        runtime_kwargs.append(kwargs)
        if target.__name__ == 'run_claude_code_session_turn' and len(runtime_kwargs) == 1:
            return {
                'success': True,
                'session_id': 'session-1',
                'output': 'need input',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 0.1,
                'events': [{'type': 'result', 'session_id': 'session-1'}],
                'result_event': {
                    'type': 'result',
                    'deferred_tool_use': {
                        'id': 'tooluse_ask_1',
                        'name': 'AskUserQuestion',
                        'input': {'questions': [{'question': 'Which option?'}]},
                    },
                },
            }
        if target.__name__ == 'run_claude_code_session_turn':
            return {
                'success': True,
                'session_id': 'session-1',
                'output': 'done',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 0.1,
                'events': [{'type': 'result', 'session_id': 'session-1'}],
                'result_event': {'type': 'result'},
            }
        return {'success': True, 'session_id': 'session-1', 'stderr': '', 'exit_code': 0}

    async def get_user_input(hint: UserInputRequest | str | None) -> str | None:
        if isinstance(hint, UserInputRequest) and hint.source == 'hook':
            return 'Alpha'
        return None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code_multiturn(
        ClaudeCodeMultiturnInput(
            user_prompt='first prompt',
            working_dir='/workspace',
            session_id='session-1',
            max_turns=2,
            get_user_input=get_user_input,
        )
    )

    assert output.success is True
    assert output.turns == 2
    assert runtime_kwargs[0]['parent_tool_use_id'] is None
    assert runtime_kwargs[1]['parent_tool_use_id'] == 'tooluse_ask_1'


@pytest.mark.asyncio
async def test_session_agent_can_disable_trace_events(monkeypatch):
    runtime_kwargs: list[dict] = []

    async def fake_call_runtime_agent(target, /, **kwargs):
        runtime_kwargs.append(kwargs)
        if target.__name__ == 'run_claude_code_session_turn':
            return {
                'success': True,
                'session_id': 'session-1',
                'output': 'done',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 0.1,
                'events': [{'type': 'result', 'session_id': 'session-1'}],
                'result_event': {'type': 'result'},
            }
        return {'success': True, 'session_id': 'session-1', 'stderr': '', 'exit_code': 0}

    async def get_user_input(_hint: UserInputRequest | str | None) -> str | None:
        return None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)

    await claude_agent.run_claude_code_multiturn(
        ClaudeCodeMultiturnInput(
            user_prompt='first prompt',
            working_dir='/workspace',
            session_id='session-1',
            record_trace_events=False,
            get_user_input=get_user_input,
        )
    )

    assert runtime_kwargs[0]['record_trace_events'] is False
