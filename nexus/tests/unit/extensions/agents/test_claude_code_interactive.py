import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import Any

import pytest
import tomllib

from pydantic import ValidationError

from nexus.core import UserInputRequest, agent_registry
from nexus.extensions.agents.claude_code import agent as claude_agent
from nexus.extensions.agents.claude_code.agent import (
    ClaudeCodeInput,
    ClaudeCodeInteractiveConfigFilePatch,
    ClaudeCodeInteractiveInput,
    ClaudeCodeInteractivePreStartScript,
)
from nexus.extensions.agents.claude_code.interactive import hooks as hook_bridge
from nexus.extensions.agents.claude_code.interactive import runtime as interactive_runtime
from nexus.extensions.agents.claude_code.interactive import session as interactive_session
from nexus.extensions.agents.claude_code.interactive.hooks import InteractionRequest
from nexus.extensions.agents.claude_code.interactive.session import ClaudeInteractiveSession
from nexus.extensions.agents.installed_common import multiturn


def _target_name(target: str | Any) -> str:
    """Extract function name from a string import path or callable."""
    if isinstance(target, str):
        return target.rsplit('::', 1)[-1] if '::' in target else target
    return target.__name__


def test_claude_code_agents_are_registered_independently():
    assert agent_registry.get('claude_code')
    assert agent_registry.get('claude_code_multiturn')
    assert agent_registry.get('claude_code_interactive')


def test_claude_code_manifest_exposes_all_four_agents():
    manifest_path = Path(claude_agent.__file__).with_name('pyproject.toml')
    agents = tomllib.loads(manifest_path.read_text())['tool']['nexus']['agents']

    assert agents == {
        'claude_code': 'nexus.extensions.agents.claude_code.agent',
        'claude_code_native_bridge': 'nexus.extensions.agents.claude_code.agent',
        'claude_code_multiturn': 'nexus.extensions.agents.claude_code.agent',
        'claude_code_interactive': 'nexus.extensions.agents.claude_code.agent',
    }


def test_interactive_input_is_independent_from_print_input():
    assert not issubclass(ClaudeCodeInteractiveInput, ClaudeCodeInput)
    assert ClaudeCodeInteractiveInput.model_fields['version'].default == 'latest'
    assert ClaudeCodeInteractiveInput.model_fields is not ClaudeCodeInput.model_fields


@pytest.mark.asyncio
async def test_default_input_provider_degrades_to_single_turn():
    """Plain callers (e.g. SWE) omit get_user_input; the default ends after turn 1."""
    inp = ClaudeCodeInteractiveInput(user_prompt='hi', working_dir='/workspace')
    assert inp.max_turns == 20
    assert inp.close_on_finish is True
    assert inp.hook_input_timeout is None
    assert inp.hook_wait_timeout == 700.0
    assert inp.hook_command_timeout == 750.0
    assert await inp.get_user_input('anything') is None


def test_interactive_input_validates_hook_timeout_relationships():
    configured = ClaudeCodeInteractiveInput(
        user_prompt='hi',
        working_dir='/workspace',
        hook_input_timeout=680.0,
        hook_wait_timeout=700.0,
        hook_command_timeout=750.0,
    )
    assert configured.hook_input_timeout == 680.0

    with pytest.raises(ValidationError, match='hook_input_timeout cannot be greater'):
        ClaudeCodeInteractiveInput(
            user_prompt='hi',
            working_dir='/workspace',
            hook_input_timeout=701.0,
        )
    with pytest.raises(ValidationError, match='hook_command_timeout must be greater'):
        ClaudeCodeInteractiveInput(
            user_prompt='hi',
            working_dir='/workspace',
            hook_wait_timeout=750.0,
            hook_command_timeout=750.0,
        )


@pytest.mark.asyncio
async def test_interactive_agent_runs_through_multiturn(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_run_installed_multiturn(input_data, /, **kwargs):
        captured['input'] = input_data
        captured['kwargs'] = kwargs
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.25,
            'session_id': 'sid-1',
            'turns': 2,
            'events': [{'type': 'pty_turn', 'turn_id': 1}, {'type': 'pty_turn', 'turn_id': 2}],
            'ended_by_user': True,
            'reached_max_turns': False,
            'status': 'completed',
            'latest_screen': 'done screen',
            'recent_events': [{'seq': 2, 'turn_id': 2, 'type': 'text', 'text': 'done'}],
            'event_cursor': {'after_seq': 1, 'latest_seq': 2, 'count': 1},
        }

    monkeypatch.setattr(claude_agent, 'run_installed_multiturn', fake_run_installed_multiturn)

    output = await claude_agent.run_claude_code_interactive(
        ClaudeCodeInteractiveInput(
            user_prompt='fix the bug',
            working_dir='/workspace',
            version='2.1.0',
            extra_env={'ENABLE_TOOL_SEARCH': 'true'},
            extra_cli_args=['--verbose'],
            config_file_patches=[
                ClaudeCodeInteractiveConfigFilePatch(path='~/.claude/settings.json', content={'env': {'A': 'B'}})
            ],
            pre_start_scripts=[ClaudeCodeInteractivePreStartScript(interpreter='bash', code='true')],
            mcp_servers=[{'name': 'demo', 'command': 'demo-mcp', 'args': []}],
            rows=40,
            cols=120,
            prompt_settle_seconds=2.0,
            log_frames=True,
            snapshot_seconds=30.0,
            hook_wait_timeout=900.0,
            hook_command_timeout=960.0,
        )
    )

    kwargs = captured['kwargs']
    assert kwargs['agent_key'] == 'claude_code_interactive'
    assert kwargs['session_turn_target'].endswith('::run_claude_code_interactive_session_turn')
    assert kwargs['close_session_target'].endswith('::close_claude_code_interactive_session')

    runtime_kwargs = kwargs['runtime_kwargs']
    assert runtime_kwargs['version'] == '2.1.0'
    assert runtime_kwargs['extra_env'] == {'ENABLE_TOOL_SEARCH': 'true'}
    assert runtime_kwargs['extra_cli_args'] == ['--verbose']
    assert runtime_kwargs['config_file_patches'] == [
        {'path': '~/.claude/settings.json', 'content': {'env': {'A': 'B'}}}
    ]
    assert runtime_kwargs['pre_start_scripts'][0]['code'] == 'true'
    assert runtime_kwargs['mcp_servers'][0]['name'] == 'demo'
    assert (runtime_kwargs['rows'], runtime_kwargs['cols']) == (40, 120)
    assert runtime_kwargs['prompt_settle_seconds'] == 2.0
    assert runtime_kwargs['log_frames'] is True
    assert runtime_kwargs['snapshot_seconds'] == 30.0
    assert runtime_kwargs['hook_wait_timeout'] == 900.0
    assert runtime_kwargs['hook_command_timeout'] == 960.0
    assert runtime_kwargs['enable_input_bridge'] is False
    assert kwargs['respond_session_target'].endswith('::respond_and_wait_claude_code_interactive_session')

    # Multi-turn result fields flow through to the agent output.
    assert output.success is True
    assert output.output == 'done'
    assert output.execution_time == 1.25
    assert output.session_id == 'sid-1'
    assert output.turns == 2
    assert output.ended_by_user is True
    assert len(output.events) == 2
    assert output.status == 'completed'


@pytest.mark.asyncio
async def test_interactive_agent_runs_two_turns_on_one_runtime_session(monkeypatch):
    runtime_calls: list[tuple[str, dict[str, Any]]] = []
    input_requests: list[UserInputRequest | str | None] = []

    async def fake_call_runtime_agent(target, /, **kwargs):
        runtime_calls.append((_target_name(target), kwargs))
        if _target_name(target) == 'close_claude_code_interactive_session':
            return {'success': True, 'stderr': ''}

        turn_id = sum(name == 'run_claude_code_interactive_session_turn' for name, _ in runtime_calls)
        return {
            'success': True,
            'output': f'answer-{turn_id}',
            'stderr': '',
            'exit_code': 0,
            'events': [
                {
                    'type': 'pty_turn',
                    'session_id': kwargs['session_id'],
                    'turn_id': turn_id,
                    'claude_pid': 4321,
                    'completed': True,
                }
            ],
        }

    async def get_user_input(request: UserInputRequest | str | None) -> str | None:
        input_requests.append(request)
        return 'second prompt' if len(input_requests) == 1 else None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code_interactive(
        ClaudeCodeInteractiveInput(
            user_prompt='first prompt',
            working_dir='/workspace',
            session_id='interactive-test-session',
            max_turns=3,
            get_user_input=get_user_input,
        )
    )

    turn_calls = [kwargs for name, kwargs in runtime_calls if name == 'run_claude_code_interactive_session_turn']
    close_calls = [kwargs for name, kwargs in runtime_calls if name == 'close_claude_code_interactive_session']

    assert output.success is True
    assert output.turns == 2
    assert output.output == 'answer-1\nanswer-2'
    assert output.ended_by_user is True
    assert output.reached_max_turns is False
    assert [call['prompt'] for call in turn_calls] == ['first prompt', 'second prompt']
    assert len({call['session_id'] for call in turn_calls}) == 1
    assert output.session_id == turn_calls[0]['session_id']
    assert [event['turn_id'] for event in output.events] == [1, 2]
    assert [event['claude_pid'] for event in output.events] == [4321, 4321]
    assert all(call['enable_input_bridge'] is True for call in turn_calls)
    assert len(input_requests) == 2
    assert all(isinstance(request, UserInputRequest) for request in input_requests)
    assert [request.hint for request in input_requests if isinstance(request, UserInputRequest)] == [
        'answer-1',
        'answer-2',
    ]
    assert [request.metadata['turn'] for request in input_requests if isinstance(request, UserInputRequest)] == [1, 2]
    assert len(close_calls) == 1
    assert close_calls[0]['session_id'] == output.session_id


@pytest.mark.asyncio
async def test_hook_input_resumes_same_turn_without_incrementing_turns(monkeypatch):
    runtime_calls: list[tuple[str, dict[str, Any]]] = []
    input_requests: list[UserInputRequest] = []
    provider_timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def capture_wait_for(awaitable, timeout):
        provider_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    async def fake_call_runtime_agent(target, /, **kwargs):
        runtime_calls.append((_target_name(target), kwargs))
        if _target_name(target) == 'run_claude_code_interactive_session_turn':
            return {
                'success': True,
                'status': 'needs_user_input',
                'session_id': kwargs['session_id'],
                'output': '',
                'stderr': '',
                'exit_code': 0,
                'events': [{'type': 'pty_interaction_requested'}],
                'interaction': {
                    'interaction_id': 'tooluse-1',
                    'tool_name': 'AskUserQuestion',
                    'tool_input': {
                        'questions': [
                            {
                                'question': 'Which option?',
                                'options': [{'label': 'Alpha'}, {'label': 'Beta'}],
                            }
                        ]
                    },
                    'turn_id': 1,
                    'timeout_seconds': 650.0,
                },
            }
        if _target_name(target) == 'respond_and_wait_claude_code_interactive_session':
            return {
                'success': True,
                'status': 'completed',
                'session_id': kwargs['session_id'],
                'output': 'implemented Alpha',
                'stderr': '',
                'exit_code': 0,
                'events': [
                    {'type': 'pty_interaction_resolved'},
                    {'type': 'pty_turn', 'turn_id': 1, 'completed': True},
                ],
            }
        return {'success': True, 'session_id': kwargs['session_id'], 'stderr': '', 'exit_code': 0}

    async def get_user_input(request: UserInputRequest | str | None) -> str | None:
        assert isinstance(request, UserInputRequest)
        input_requests.append(request)
        return 'Alpha' if request.source == 'hook' else None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)
    monkeypatch.setattr(multiturn.asyncio, 'wait_for', capture_wait_for)

    output = await claude_agent.run_claude_code_interactive(
        ClaudeCodeInteractiveInput(
            user_prompt='implement the issue',
            working_dir='/workspace',
            get_user_input=get_user_input,
        )
    )

    assert output.success is True
    assert output.turns == 1
    assert output.output == 'implemented Alpha'
    assert [request.source for request in input_requests] == ['hook', 'turn']
    hook_request = input_requests[0]
    assert hook_request.hook == {
        'id': 'tooluse-1',
        'name': 'AskUserQuestion',
        'input': {
            'questions': [
                {
                    'question': 'Which option?',
                    'options': [{'label': 'Alpha'}, {'label': 'Beta'}],
                }
            ]
        },
    }
    assert hook_request.metadata['turn'] == 1
    assert hook_request.metadata['hook_timeout_seconds'] == 650.0
    assert provider_timeouts == [650.0]

    turn_call = next(kwargs for name, kwargs in runtime_calls if name == 'run_claude_code_interactive_session_turn')
    response_call = next(
        kwargs for name, kwargs in runtime_calls if name == 'respond_and_wait_claude_code_interactive_session'
    )
    assert turn_call['enable_input_bridge'] is True
    assert response_call['session_id'] == turn_call['session_id']
    assert response_call['interaction_id'] == 'tooluse-1'
    assert response_call['answer'] == 'Alpha'


def _make_session(tmp_path) -> ClaudeInteractiveSession:
    return ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
    )


def test_pty_screen_snapshots_are_disabled_by_default(tmp_path):
    session = _make_session(tmp_path)

    assert session.snapshot_seconds is None


def test_pty_screen_snapshot_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match='snapshot_seconds must be greater than zero'):
        ClaudeInteractiveSession(
            claude_path='/bin/true',
            cwd=str(tmp_path),
            env={},
            done_file=str(tmp_path / 'done'),
            snapshot_seconds=0,
        )


@pytest.mark.asyncio
async def test_pty_spawn_calls_setsid_before_claiming_controlling_terminal(monkeypatch):
    import pty

    calls: list[str] = []
    captured: dict[str, Any] = {}

    class _Proc:
        pid = 1234

    monkeypatch.setattr(pty, 'openpty', lambda: (100, 101))
    monkeypatch.setattr(interactive_session, '_set_winsize', lambda *_args: None)
    monkeypatch.setattr(interactive_session.fcntl, 'fcntl', lambda *_args: 0)
    monkeypatch.setattr(interactive_session.os, 'setsid', lambda: calls.append('setsid'))
    monkeypatch.setattr(
        interactive_session.fcntl,
        'ioctl',
        lambda _fd, operation, _arg: (
            calls.append('tiocsctty') if operation == interactive_session.termios.TIOCSCTTY else None
        ),
    )
    monkeypatch.setattr(interactive_session.os, 'close', lambda _fd: None)
    monkeypatch.setattr(interactive_session.PtyProcess, '__init__', lambda *_args, **_kwargs: None)

    async def fake_create_subprocess_exec(*_args, **kwargs):
        captured.update(kwargs)
        kwargs['preexec_fn']()
        return _Proc()

    monkeypatch.setattr(interactive_session.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)

    await interactive_session.PtyProcess.spawn(
        argv=['claude'],
        env={},
        rows=50,
        cols=200,
        cwd='/tmp',
        on_data=lambda _data: None,
        on_exit=lambda: None,
    )

    assert calls == ['setsid', 'tiocsctty']
    assert 'start_new_session' not in captured


def test_read_completion_rejects_stale_turn_and_accepts_matching(tmp_path):
    session = _make_session(tmp_path)
    done = Path(session.done_file)

    # A leftover event from turn 1 must not satisfy turn 2.
    done.write_text(json.dumps({'session_id': 'x', 'turn_id': 1, 'last_assistant_message': 'turn1'}))
    assert session._read_completion(2) == (False, None)

    # The matching turn_id is accepted and carries that turn's message.
    done.write_text(json.dumps({'session_id': 'x', 'turn_id': 2, 'last_assistant_message': 'turn2'}))
    assert session._read_completion(2) == (True, 'turn2')

    # And the turn-2 event still does not satisfy an earlier turn.
    assert session._read_completion(1) == (False, None)


def test_read_completion_ignores_empty_or_malformed(tmp_path):
    session = _make_session(tmp_path)
    done = Path(session.done_file)
    done.write_text('')
    assert session._read_completion(1) == (False, None)
    done.write_text('not-json')
    assert session._read_completion(1) == (False, None)


class _FakeInteractionBridge:
    def __init__(self, done_file: Path, *, response_status: str = 'written') -> None:
        self.done_file = done_file
        self.response_status = response_status
        self.pending = InteractionRequest(
            version=1,
            session_id='session-1',
            turn_id=1,
            interaction_id='tooluse-1',
            tool_name='AskUserQuestion',
            tool_input={'questions': [{'question': 'Which option?'}]},
            created_at=1.0,
        )
        self.responses: list[tuple[str, str | None]] = []
        self.closed = False

    async def wait_request(self, *, timeout: float = 0.0):
        del timeout
        request, self.pending = self.pending, None
        return request

    async def respond(self, interaction_id: str, answer: str | None):
        self.responses.append((interaction_id, answer))
        self.done_file.write_text(
            json.dumps({
                'session_id': 'session-1',
                'turn_id': 1,
                'last_assistant_message': 'continued in the same turn',
            }),
            encoding='utf-8',
        )
        return {'status': self.response_status}

    async def close(self):
        self.closed = True


class _FakeLivePty:
    class _Proc:
        pid = 2468

    proc = _Proc()

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)

    def killpg(self) -> None:
        return

    def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_pty_screen_snapshot_logging_is_opt_in(tmp_path, monkeypatch, caplog):
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        snapshot_seconds=1,
    )
    session.pty = _FakeLivePty()
    session._tail.extend(b'current terminal output')
    session._active_turn_id = 1
    session._active_turn_started = 0.0
    session._active_turn_deadline = float('inf')

    completion_states = iter(((False, None), (True, 'done')))
    monkeypatch.setattr(session, '_read_completion', lambda _turn_id: next(completion_states))

    async def no_sleep(_delay: float) -> None:
        return

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.interactive.session.asyncio.sleep',
        no_sleep,
    )

    with caplog.at_level(logging.DEBUG, logger='nexus.extensions.agents.claude_code.interactive.session'):
        result = await session._wait_for_active_turn()

    assert result['completed'] is True
    assert '[pty_session snapshot turn=1' in caplog.text
    assert 'current terminal output' in caplog.text


@pytest.mark.asyncio
async def test_pty_hook_interaction_resumes_without_starting_a_new_turn(tmp_path):
    done_file = tmp_path / 'done'
    bridge = _FakeInteractionBridge(done_file)
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(done_file),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        settle_seconds=0,
        snapshot_seconds=60,
        interaction_bridge=bridge,
    )
    session.pty = _FakeLivePty()
    await session.start()

    first_state = await session.send_turn('initial prompt', timeout=2)
    assert first_state['status'] == 'needs_user_input'
    assert first_state['turn_id'] == 1
    assert first_state['interaction']['interaction_id'] == 'tooluse-1'

    final_state = await session.respond_and_wait('tooluse-1', 'Alpha')
    assert final_state['status'] == 'completed'
    assert final_state['turn_id'] == 1
    assert final_state['final_message'] == 'continued in the same turn'
    assert session._turn == 1
    assert bridge.responses == [('tooluse-1', 'Alpha')]

    await session.close()
    assert bridge.closed is True


@pytest.mark.asyncio
async def test_expired_hook_response_continues_waiting_without_closing_session(tmp_path):
    done_file = tmp_path / 'done'
    bridge = _FakeInteractionBridge(done_file, response_status='expired')
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(done_file),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        settle_seconds=0,
        snapshot_seconds=60,
        interaction_bridge=bridge,
    )
    session.pty = _FakeLivePty()
    await session.start()

    first_state = await session.send_turn('initial prompt', timeout=2)
    final_state = await session.respond_and_wait('tooluse-1', 'late answer')
    already_finished_state = await session.respond_and_wait('tooluse-1', 'even later answer')

    assert first_state['status'] == 'needs_user_input'
    assert final_state['status'] == 'completed'
    assert final_state['interaction_response_status'] == 'expired'
    assert already_finished_state['status'] == 'interaction_expired'
    assert already_finished_state['session_reusable'] is True
    assert session._closed is False
    assert session._turn == 1

    await session.close()


@pytest.mark.asyncio
async def test_real_hook_timeout_falls_back_then_same_turn_accepts_next_hook(monkeypatch, tmp_path):
    """A timed-out real file Hook does not prevent a later Hook in the same turn."""
    interaction_dir = tmp_path / 'interactions'
    done_file = tmp_path / 'done'
    turn_file = tmp_path / 'turn'
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(done_file),
        final_file=str(tmp_path / 'final'),
        turn_file=str(turn_file),
        settle_seconds=0,
        snapshot_seconds=60,
        interaction_bridge=bridge,
    )
    session.pty = _FakeLivePty()
    await session.start()

    monkeypatch.setenv(hook_bridge.INTERACTION_DIR_ENV, str(interaction_dir))
    monkeypatch.setenv(hook_bridge.SESSION_ID_ENV, 'session-timeout-chain')
    monkeypatch.setenv(hook_bridge.TURN_FILE_ENV, str(turn_file))

    hook_inputs = [
        (
            'tooluse-timeout-1',
            {'questions': [{'question': 'First question?'}]},
        ),
        (
            'tooluse-answered-2',
            {'questions': [{'question': 'Second question?'}]},
        ),
    ]
    monkeypatch.setattr(hook_bridge, '_tool_input_from_stdin', lambda: hook_inputs.pop(0))

    hook_outputs: dict[str, dict[str, Any]] = {}
    hook_errors: list[BaseException] = []

    def invoke_hook(name: str, *, wait_timeout: float, finish_turn: bool = False) -> None:
        try:
            hook_outputs[name] = hook_bridge.run_hook('AskUserQuestion', wait_timeout=wait_timeout)
            if finish_turn:
                done_file.write_text(
                    json.dumps({
                        'session_id': 'session-timeout-chain',
                        'turn_id': 1,
                        'last_assistant_message': 'continued after the second hook',
                    }),
                    encoding='utf-8',
                )
        except BaseException as exc:  # pragma: no cover - re-raised below
            hook_errors.append(exc)

    async def wait_thread(thread: threading.Thread, *, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not thread.is_alive(), f'{thread.name} did not finish'
        if hook_errors:
            raise hook_errors[0]

    send_task = asyncio.create_task(session.send_turn('initial prompt', timeout=5))
    deadline = time.monotonic() + 1.0
    while not turn_file.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert turn_file.read_text(encoding='utf-8') == '1'

    first_hook = threading.Thread(
        target=invoke_hook,
        kwargs={'name': 'first', 'wait_timeout': 0.75},
        daemon=True,
        name='first-hook',
    )
    first_hook.start()
    first_state = await asyncio.wait_for(send_task, timeout=3.0)

    assert first_state['status'] == 'needs_user_input'
    assert first_state['turn_id'] == 1
    assert first_state['interaction']['interaction_id'] == 'tooluse-timeout-1'
    assert first_state['interaction']['timeout_seconds'] == 0.75

    await wait_thread(first_hook)
    assert hook_outputs['first']['hookSpecificOutput']['permissionDecision'] == 'deny'

    second_hook = threading.Thread(
        target=invoke_hook,
        kwargs={'name': 'second', 'wait_timeout': 2.0, 'finish_turn': True},
        daemon=True,
        name='second-hook',
    )
    second_hook.start()

    second_state = await asyncio.wait_for(
        session.respond_and_wait('tooluse-timeout-1', 'late answer'),
        timeout=3.0,
    )
    assert second_state['status'] == 'needs_user_input'
    assert second_state['turn_id'] == 1
    assert second_state['interaction']['interaction_id'] == 'tooluse-answered-2'
    assert second_state['interaction_response_status'] == 'expired'

    final_state = await asyncio.wait_for(
        session.respond_and_wait('tooluse-answered-2', 'yes'),
        timeout=3.0,
    )
    await wait_thread(second_hook)

    assert hook_outputs['second']['hookSpecificOutput']['permissionDecision'] == 'allow'
    assert final_state['status'] == 'completed'
    assert final_state['turn_id'] == 1
    assert final_state['final_message'] == 'continued after the second hook'
    assert final_state['interaction_response_status'] == 'written'
    assert session.claude_pid == 2468
    assert session._turn == 1
    assert session._closed is False
    assert list(interaction_dir.iterdir()) == []

    # This sandbox cannot make progress in asyncio.to_thread(), independently of
    # the callback it receives. Run the same bridge cleanup inline so this test
    # still exercises session.close() and verifies that the directory is removed.
    async def run_cleanup_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(hook_bridge.asyncio, 'to_thread', run_cleanup_inline)
    session._exited.set()
    await session.close()
    assert session._closed is True
    assert not interaction_dir.exists()


class _FakePty:
    def __init__(self, completed: bool, pid: int, *, exit_code: int | None = None, reason: str | None = None) -> None:
        self._completed = completed
        self._pid = pid
        self._exit_code = exit_code
        self._reason = reason
        self.turns = 0
        self.closed = False

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        self.turns += 1
        result: dict[str, Any] = {
            'completed': self._completed,
            'turn_id': self.turns,
            'final_message': f'answer-{self.turns}' if self._completed else '',
            'screen_tail': '',
            'elapsed': 0.1,
            'reason': self._reason or ('stop_hook' if self._completed else 'timeout'),
            'claude_pid': self._pid,
        }
        if self._exit_code is not None:
            result['exit_code'] = self._exit_code
        return result

    async def close(self) -> None:
        self.closed = True


class _FakeHookPty(_FakePty):
    def __init__(self, completed: bool, pid: int, expected_answer: str | None = 'approve') -> None:
        super().__init__(completed=completed, pid=pid)
        self.expected_answer = expected_answer

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        del prompt, timeout
        self.turns += 1
        return {
            'status': 'needs_user_input',
            'completed': False,
            'turn_id': self.turns,
            'interaction': {
                'interaction_id': 'tooluse-runtime-1',
                'tool_name': 'ExitPlanMode',
                'tool_input': {'plan': 'implement and test'},
                'turn_id': self.turns,
            },
            'screen_tail': '',
            'elapsed': 0.1,
            'reason': 'needs_user_input',
            'claude_pid': self._pid,
        }

    async def respond_and_wait(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        assert interaction_id == 'tooluse-runtime-1'
        assert answer == self.expected_answer
        return {
            'status': 'completed',
            'completed': True,
            'turn_id': self.turns,
            'final_message': 'implemented',
            'screen_tail': '',
            'elapsed': 0.2,
            'reason': 'stop_hook',
            'claude_pid': self._pid,
        }


class _FakeExpiredResponsePty(_FakeHookPty):
    async def respond_and_wait(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        result = await super().respond_and_wait(interaction_id, answer)
        result['interaction_response_status'] = 'expired'
        return result


class _FakeInactiveResponsePty(_FakePty):
    async def respond_and_wait(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        del interaction_id, answer
        return {
            'status': 'interaction_expired',
            'completed': False,
            'turn_id': self.turns,
            'final_message': '',
            'screen_tail': '',
            'elapsed': 0.1,
            'reason': 'interaction_expired',
            'claude_pid': self._pid,
            'session_reusable': True,
            'interaction_response_status': 'expired',
        }


class _FakeCrashingPty(_FakePty):
    def __init__(self, pid: int) -> None:
        super().__init__(completed=False, pid=pid)
        self._active_turn_id = 1
        self._turn = 1

    @property
    def claude_pid(self) -> int:
        return self._pid

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        del prompt, timeout
        raise RuntimeError('simulated PTY crash')

    def screen_tail(self) -> str:
        return 'crash tail'

    def latest_screen(self) -> str:
        return 'crash screen'


class _FakeProxy:
    def __init__(self) -> None:
        self.stopped = False
        self._tool_events: list[dict[str, Any]] = []
        self._next_event_seq = 1
        self._last_emitted_seq = 0
        self._current_turn_id = 0

    def mark_turn_start(self) -> None:
        self._current_turn_id += 1

    def append_events(self, events: list[dict[str, Any]]) -> None:
        for raw_event in events:
            event = {
                **raw_event,
                'seq': self._next_event_seq,
                'turn_id': self._current_turn_id,
            }
            self._next_event_seq += 1
            self._tool_events.append(event)

    def get_event_delta(self, max_events: int = 1000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        pending = [event for event in self._tool_events if event['seq'] > self._last_emitted_seq]
        latest_seq = pending[-1]['seq'] if pending else self._last_emitted_seq
        dropped_count = max(0, len(pending) - max_events)
        delivered = pending[-max_events:] if dropped_count else pending
        return delivered, {
            'after_seq': self._last_emitted_seq,
            'latest_seq': latest_seq,
            'count': len(delivered),
            'truncated': dropped_count > 0,
            'dropped_count': dropped_count,
        }

    def commit_event_cursor(self, seq: int) -> None:
        self._last_emitted_seq = max(self._last_emitted_seq, seq)
        self._tool_events = [event for event in self._tool_events if event['seq'] > self._last_emitted_seq]

    async def stop(self) -> None:
        self.stopped = True


def _register_fake(monkeypatch, wrapper) -> None:
    async def fake_prepare(session_id, **_kwargs):
        return wrapper

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)


@pytest.mark.asyncio
async def test_session_turn_reuses_same_pty_and_reports_pid(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakePty(completed=True, pid=4321), proxy=_FakeProxy()
    )
    prepared = {'count': 0, 'hook_wait_timeout': None, 'hook_command_timeout': None}

    async def fake_prepare(session_id, **kwargs):
        prepared['count'] += 1
        prepared['hook_wait_timeout'] = kwargs['hook_wait_timeout']
        prepared['hook_command_timeout'] = kwargs['hook_command_timeout']
        return wrapper

    monkeypatch.setattr(interactive_runtime, '_prepare_runtime_session', fake_prepare)

    r1 = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='s2',
        prompt='remember X',
        hook_wait_timeout=620.5,
        hook_command_timeout=690,
    )
    r2 = await interactive_runtime.run_claude_code_interactive_session_turn(session_id='s2', prompt='what is X')

    assert prepared['count'] == 1  # session created once, reused for turn 2
    assert prepared['hook_wait_timeout'] == 620.5
    assert prepared['hook_command_timeout'] == 690
    assert r1['success'] is True
    assert r2['success'] is True
    assert r1['claude_pid'] == r2['claude_pid'] == 4321
    assert r1['events'][0] == {
        'type': 'pty_turn',
        'session_id': 's2',
        'turn_id': 1,
        'claude_pid': 4321,
        'completed': True,
    }
    assert r2['events'][0]['turn_id'] == 2
    assert 's2' in interactive_runtime._PTY_SESSIONS

    await interactive_runtime.close_claude_code_interactive_session(session_id='s2')
    assert 's2' not in interactive_runtime._PTY_SESSIONS


@pytest.mark.asyncio
async def test_runtime_keeps_session_for_hook_input_and_completes_same_turn(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakeHookPty(completed=True, pid=9876),
        proxy=_FakeProxy(),
    )
    _register_fake(monkeypatch, wrapper)

    pending = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='hook-session',
        prompt='implement',
        enable_input_bridge=True,
    )

    assert pending['success'] is True
    assert pending['status'] == 'needs_user_input'
    assert pending['turn_id'] == 1
    assert pending['interaction']['interaction_id'] == 'tooluse-runtime-1'
    assert pending['events'][0]['type'] == 'pty_interaction_requested'
    assert 'hook-session' in interactive_runtime._PTY_SESSIONS

    completed = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='hook-session',
        interaction_id='tooluse-runtime-1',
        answer='approve',
    )

    assert completed['success'] is True
    assert completed['status'] == 'completed'
    assert completed['turn_id'] == 1
    assert completed['output'] == 'implemented'
    assert [event['type'] for event in completed['events']] == ['pty_interaction_resolved', 'pty_turn']
    assert 'hook-session' in interactive_runtime._PTY_SESSIONS

    await interactive_runtime.close_claude_code_interactive_session(session_id='hook-session')


@pytest.mark.asyncio
async def test_runtime_reports_expired_response_without_closing_completed_session(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakeExpiredResponsePty(completed=True, pid=9878),
        proxy=_FakeProxy(),
    )
    _register_fake(monkeypatch, wrapper)

    await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='expired-hook-session',
        prompt='implement',
        enable_input_bridge=True,
    )
    completed = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='expired-hook-session',
        interaction_id='tooluse-runtime-1',
        answer='approve',
    )

    assert completed['success'] is True
    assert completed['status'] == 'completed'
    assert completed['interaction_response_status'] == 'expired'
    assert completed['events'][0]['type'] == 'pty_interaction_resolved'
    assert completed['events'][0]['resolution'] == 'expired'
    assert completed['events'][0]['submitted'] is False
    assert 'expired-hook-session' in interactive_runtime._PTY_SESSIONS
    assert wrapper.closed is False

    await interactive_runtime.close_claude_code_interactive_session(session_id='expired-hook-session')


@pytest.mark.asyncio
async def test_inactive_expired_response_is_nonfatal_to_runtime_session():
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakeInactiveResponsePty(completed=False, pid=9879),
        proxy=_FakeProxy(),
    )
    interactive_runtime._PTY_SESSIONS['inactive-hook-session'] = wrapper

    result = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='inactive-hook-session',
        interaction_id='tooluse-old',
        answer='late answer',
    )

    assert result['success'] is True
    assert result['status'] == 'interaction_expired'
    assert result['session_reusable'] is True
    assert result['interaction_response_status'] == 'expired'
    assert 'inactive-hook-session' in interactive_runtime._PTY_SESSIONS
    assert wrapper.closed is False

    await interactive_runtime.close_claude_code_interactive_session(session_id='inactive-hook-session')


@pytest.mark.asyncio
async def test_runtime_returns_only_events_added_after_previous_hook_boundary(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    fake_proxy = _FakeProxy()
    original_mark = fake_proxy.mark_turn_start

    def mark_then_add():
        original_mark()
        fake_proxy.append_events([{'type': 'tool_use', 'id': 'ask-1', 'name': 'AskUserQuestion', 'input': {}}])

    fake_proxy.mark_turn_start = mark_then_add
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakeHookPty(completed=True, pid=9877),
        proxy=fake_proxy,
    )
    _register_fake(monkeypatch, wrapper)

    pending = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='hook-delta',
        prompt='implement',
        enable_input_bridge=True,
    )
    assert [event['seq'] for event in pending['recent_events']] == [1]
    assert pending['event_cursor']['after_seq'] == 0
    assert pending['event_cursor']['latest_seq'] == 1

    fake_proxy.append_events([
        {
            'type': 'tool_result',
            'tool_use_id': 'ask-1',
            'content': 'approved',
            'is_error': False,
        }
    ])
    completed = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='hook-delta',
        interaction_id='tooluse-runtime-1',
        answer='approve',
    )

    assert [event['seq'] for event in completed['recent_events']] == [2]
    assert completed['recent_events'][0]['type'] == 'tool_result'
    assert completed['event_cursor']['after_seq'] == 1
    assert completed['event_cursor']['latest_seq'] == 2

    await interactive_runtime.close_claude_code_interactive_session(session_id='hook-delta')


@pytest.mark.asyncio
async def test_incomplete_turn_drops_session_from_registry(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakePty(completed=False, pid=777), proxy=_FakeProxy()
    )
    _register_fake(monkeypatch, wrapper)

    result = await interactive_runtime.run_claude_code_interactive_session_turn(session_id='s3', prompt='hi')

    assert result['success'] is False
    assert result['events'][0]['completed'] is False
    # A non-completed turn must not leave a broken PTY behind for the next turn.
    assert 's3' not in interactive_runtime._PTY_SESSIONS
    assert wrapper.closed is True
    assert wrapper.pty_session.closed is True
    assert wrapper.proxy.stopped is True


@pytest.mark.asyncio
async def test_process_exit_code_is_propagated(monkeypatch):
    """When the PTY process exits with a signal, the exit_code is passed through (not -1)."""
    interactive_runtime._PTY_SESSIONS.clear()
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakePty(completed=False, pid=555, exit_code=-15, reason='process_exit'),
        proxy=_FakeProxy(),
    )
    _register_fake(monkeypatch, wrapper)

    result = await interactive_runtime.run_claude_code_interactive_session_turn(session_id='s-sig', prompt='hi')

    assert result['success'] is False
    assert result['exit_code'] == -15
    assert 's-sig' not in interactive_runtime._PTY_SESSIONS


@pytest.mark.asyncio
async def test_turn_crash_returns_and_commits_diagnostic_context(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    fake_proxy = _FakeProxy()
    original_mark = fake_proxy.mark_turn_start

    def mark_then_add():
        original_mark()
        fake_proxy.append_events([{'type': 'text', 'text': 'before crash'}])

    fake_proxy.mark_turn_start = mark_then_add
    wrapper = interactive_runtime._InteractiveRuntimeSession(
        pty_session=_FakeCrashingPty(pid=556),
        proxy=fake_proxy,
    )
    _register_fake(monkeypatch, wrapper)

    result = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='s-crash',
        prompt='hi',
    )

    assert result['success'] is False
    assert result['status'] == 'error'
    assert result['latest_screen'] == 'crash screen'
    assert result['recent_events'][0] == {
        'type': 'text',
        'text': 'before crash',
        'seq': 1,
        'turn_id': 1,
    }
    assert result['event_cursor']['latest_seq'] == 1
    assert fake_proxy._last_emitted_seq == 1
    assert 's-crash' not in interactive_runtime._PTY_SESSIONS


@pytest.mark.asyncio
async def test_close_session_is_idempotent():
    interactive_runtime._PTY_SESSIONS.clear()
    first = await interactive_runtime.close_claude_code_interactive_session(session_id='missing')
    second = await interactive_runtime.close_claude_code_interactive_session(session_id='missing')
    assert first['success'] is True
    assert second['success'] is True


def test_ensure_skip_permissions_adds_flag_once():
    flag = interactive_runtime._SKIP_PERMISSIONS_FLAG
    # Added when absent.
    assert interactive_runtime._ensure_skip_permissions(None) == [flag]
    assert interactive_runtime._ensure_skip_permissions(['--verbose']) == [flag, '--verbose']
    # Not duplicated when the caller already passed it.
    assert interactive_runtime._ensure_skip_permissions([flag]) == [flag]
    assert interactive_runtime._ensure_skip_permissions(['--verbose', flag]) == ['--verbose', flag]


def test_interactive_hooks_are_static_and_file_bridge_capable():
    settings = interactive_runtime._interactive_hook_settings(sys.executable)
    hooks = settings['hooks']
    assert hook_bridge.DEFAULT_HOOK_WAIT_TIMEOUT == 700
    assert interactive_runtime.DEFAULT_HOOK_COMMAND_TIMEOUT == 750
    assert hooks['Stop'][0]['hooks'][0]['command'].startswith(f'{sys.executable} -c ')
    pre_tool_use = {entry['matcher']: entry for entry in hooks['PreToolUse']}

    assert set(pre_tool_use) == {'AskUserQuestion', 'ExitPlanMode'}
    for tool_name, hook in pre_tool_use.items():
        command = hook['hooks'][0]['command']
        assert 'nexus.extensions.agents.claude_code.interactive.hooks' in command
        assert tool_name in command
        assert hook['hooks'][0]['timeout'] == 750


def test_interactive_hook_timeout_configures_command_watchdog():
    settings = interactive_runtime._interactive_hook_settings(
        sys.executable,
        hook_wait_timeout=620.5,
        hook_command_timeout=690,
    )

    for hook in settings['hooks']['PreToolUse']:
        assert hook['hooks'][0]['timeout'] == 690

    with pytest.raises(ValueError, match='positive finite number'):
        interactive_runtime._interactive_hook_settings(sys.executable, hook_wait_timeout=0)
    with pytest.raises(ValueError, match='must be greater than hook_wait_timeout'):
        interactive_runtime._interactive_hook_settings(
            sys.executable,
            hook_wait_timeout=700,
            hook_command_timeout=700,
        )


def test_unattended_interactive_hooks_delegate_ask_user_question_to_claude():
    settings = interactive_runtime._interactive_hook_settings(sys.executable)
    pre_tool_use = {entry['matcher']: entry for entry in settings['hooks']['PreToolUse']}
    command = pre_tool_use['AskUserQuestion']['hooks'][0]['command']
    env = dict(os.environ)
    env.pop('NEXUS_PTY_INTERACTION_DIR', None)
    result = subprocess.run(
        ['/bin/sh', '-c', command],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = json.loads(result.stdout)['hookSpecificOutput']

    assert output['hookEventName'] == 'PreToolUse'
    assert output['permissionDecision'] == 'deny'
    reason = output['permissionDecisionReason']
    assert 'unattended task' in reason
    assert 'Infer the best decision' in reason
    assert 'without asking the user again' in reason
    assert 'first option' not in reason


def test_unattended_interactive_hooks_allow_exit_plan_mode():
    settings = interactive_runtime._interactive_hook_settings(sys.executable)
    pre_tool_use = {entry['matcher']: entry for entry in settings['hooks']['PreToolUse']}
    command = pre_tool_use['ExitPlanMode']['hooks'][0]['command']
    env = dict(os.environ)
    env.pop('NEXUS_PTY_INTERACTION_DIR', None)
    result = subprocess.run(
        ['/bin/sh', '-c', command],
        input=json.dumps({'tool_input': {'plan': 'implement and test'}}),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = json.loads(result.stdout)['hookSpecificOutput']
    assert output['permissionDecision'] == 'allow'
    assert output['updatedInput'] == {'plan': 'implement and test'}


def test_exit_plan_mode_hook_context_uses_latest_plan_file_without_changing_tool_input(monkeypatch, tmp_path):
    home = tmp_path / 'home'
    plan_dir = home / '.claude' / 'plans'
    plan_dir.mkdir(parents=True)
    older = plan_dir / 'older.md'
    latest = plan_dir / 'latest.md'
    older.write_text('old plan', encoding='utf-8')
    latest.write_text('full current plan', encoding='utf-8')
    os.utime(older, ns=(1, 1))
    os.utime(latest, ns=(2, 2))

    interaction_dir = tmp_path / 'interactions'
    captured: list[dict[str, Any]] = []
    original_write = hook_bridge._atomic_write_json

    def capture_write(path, value):
        captured.append(value)
        original_write(path, value)

    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv(hook_bridge.INTERACTION_DIR_ENV, str(interaction_dir))
    monkeypatch.setattr(hook_bridge, '_tool_input_from_stdin', lambda: ('exit-plan-1', {}))
    monkeypatch.setattr(hook_bridge, '_atomic_write_json', capture_write)

    output = hook_bridge.run_hook('ExitPlanMode', wait_timeout=0)

    assert captured[0]['tool_input'] == {
        'plan': 'full current plan',
        'plan_path': str(latest),
    }
    assert output['hookSpecificOutput']['updatedInput'] == {}


def test_exit_plan_mode_hook_context_keeps_plan_from_tool_input(monkeypatch, tmp_path):
    home = tmp_path / 'home'
    plan_dir = home / '.claude' / 'plans'
    plan_dir.mkdir(parents=True)
    (plan_dir / 'latest.md').write_text('file plan', encoding='utf-8')
    monkeypatch.setenv('HOME', str(home))

    assert hook_bridge._context_tool_input('ExitPlanMode', {'plan': 'tool plan'}) == {'plan': 'tool plan'}


def test_tool_reference_is_accepted_by_base_proxy_model():
    """tool_reference in ToolResultContent is accepted by MessagesRequest after model change."""
    from nexus.runtime.server.services.anthropic_proxy.models import MessagesRequest

    body = {
        'model': 'test',
        'max_tokens': 100,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tool-1',
                        'content': [
                            {'type': 'tool_reference', 'tool_name': 'TaskCreate'},
                            {'type': 'text', 'text': 'kept'},
                        ],
                    }
                ],
            }
        ],
    }
    request = MessagesRequest.model_validate(body)
    content = request.messages[0].content
    assert isinstance(content, list)
    tool_result = content[0]
    assert tool_result.type == 'tool_result'
    assert isinstance(tool_result.content, list)
    assert tool_result.content[0].type == 'tool_reference'
    assert tool_result.content[0].tool_name == 'TaskCreate'


# ---------------------------------------------------------------------------
# Stop Hook protection: user patch cannot override Nexus hooks
# ---------------------------------------------------------------------------


def test_user_patch_cannot_override_nexus_stop_hook(tmp_path):
    """config_file_patches targeting hooks.Stop must not suppress completion detection."""
    settings_path = tmp_path / '.claude' / 'settings.json'
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text('{}')
    user_stop_hook = {'hooks': [{'type': 'command', 'command': 'echo user-hook'}]}
    user_pre_tool_hook = {
        'matcher': 'Bash',
        'hooks': [{'type': 'command', 'command': 'echo user-pre-tool-hook'}],
    }

    # Simulate the runtime sequence: user patches applied first, Nexus hooks last.
    interactive_runtime._apply_json_patch(
        str(settings_path),
        {'hooks': {'Stop': [user_stop_hook], 'PreToolUse': [user_pre_tool_hook]}},
    )
    interactive_runtime._append_required_hooks(
        str(settings_path),
        interactive_runtime._interactive_hook_settings(sys.executable),
    )
    # Re-applying required hooks must not duplicate them.
    interactive_runtime._append_required_hooks(
        str(settings_path),
        interactive_runtime._interactive_hook_settings(sys.executable),
    )

    final = json.loads(settings_path.read_text())
    stop_hooks = final['hooks']['Stop']
    assert user_stop_hook in stop_hooks
    # Nexus Stop Hook must be present (contains the STOP_HOOK_CODE python snippet).
    assert any('NEXUS_PTY_DONE_FILE' in h['hooks'][0].get('command', '') for h in stop_hooks)
    assert len(stop_hooks) == 2
    # User PreToolUse hooks and both Nexus bridge hooks must coexist.
    matchers = {entry['matcher'] for entry in final['hooks']['PreToolUse']}
    assert matchers == {'Bash', 'ExitPlanMode', 'AskUserQuestion'}
    assert len(final['hooks']['PreToolUse']) == 3


def test_required_hooks_create_missing_settings_directory(tmp_path):
    settings_path = tmp_path / 'missing-home' / '.claude' / 'settings.json'

    interactive_runtime._append_required_hooks(
        str(settings_path),
        interactive_runtime._interactive_hook_settings(sys.executable),
    )

    final = json.loads(settings_path.read_text())
    assert final['hooks']['Stop']
    assert {entry['matcher'] for entry in final['hooks']['PreToolUse']} == {
        'ExitPlanMode',
        'AskUserQuestion',
    }


# ---------------------------------------------------------------------------
# Multiturn orchestration: provider timeout / exception / consecutive hooks
# ---------------------------------------------------------------------------


class _FakeMultiHookPty(_FakePty):
    """PTY that returns needs_user_input twice before completing."""

    def __init__(self, pid: int) -> None:
        super().__init__(completed=True, pid=pid)
        self._hook_count = 0
        self._hooks = [
            {
                'interaction_id': 'hook-1',
                'tool_name': 'ExitPlanMode',
                'tool_input': {'plan': 'step A'},
            },
            {
                'interaction_id': 'hook-2',
                'tool_name': 'AskUserQuestion',
                'tool_input': {'questions': [{'question': 'Continue?'}]},
            },
        ]
        self.responses: list[tuple[str, str | None]] = []

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict[str, Any]:
        self.turns += 1
        return self._next_hook_or_complete()

    async def respond_and_wait(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        self.responses.append((interaction_id, answer))
        return self._next_hook_or_complete()

    def _next_hook_or_complete(self) -> dict[str, Any]:
        if self._hook_count < len(self._hooks):
            hook = self._hooks[self._hook_count]
            self._hook_count += 1
            return {
                'status': 'needs_user_input',
                'completed': False,
                'turn_id': self.turns,
                'interaction': hook,
                'screen_tail': '',
                'elapsed': 0.1,
                'reason': 'needs_user_input',
                'claude_pid': self._pid,
            }
        return {
            'status': 'completed',
            'completed': True,
            'turn_id': self.turns,
            'final_message': 'all done',
            'screen_tail': '',
            'elapsed': 0.2,
            'reason': 'stop_hook',
            'claude_pid': self._pid,
        }


@pytest.mark.asyncio
async def test_multiturn_consecutive_hook_interactions(monkeypatch):
    """Two consecutive needs_user_input in one turn are both resolved."""
    interactive_runtime._PTY_SESSIONS.clear()
    pty = _FakeMultiHookPty(pid=1111)
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=pty, proxy=_FakeProxy())
    _register_fake(monkeypatch, wrapper)

    answers_given: list[str] = []

    async def provider(request):
        answer = 'approve' if request.hook and request.hook.get('name') == 'ExitPlanMode' else 'yes'
        answers_given.append(answer)
        return answer

    r1 = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='multi-hook', prompt='go', enable_input_bridge=True
    )
    assert r1['status'] == 'needs_user_input'

    r2 = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='multi-hook', interaction_id='hook-1', answer='approve'
    )
    assert r2['status'] == 'needs_user_input'

    r3 = await interactive_runtime.respond_and_wait_claude_code_interactive_session(
        session_id='multi-hook', interaction_id='hook-2', answer='yes'
    )
    assert r3['status'] == 'completed'
    assert r3['output'] == 'all done'
    assert pty.responses == [('hook-1', 'approve'), ('hook-2', 'yes')]

    await interactive_runtime.close_claude_code_interactive_session(session_id='multi-hook')


@pytest.mark.asyncio
async def test_multiturn_provider_timeout_falls_back_to_none(monkeypatch):
    """When get_user_input exceeds hook_input_timeout, answer=None is used."""
    import asyncio as _asyncio

    interactive_runtime._PTY_SESSIONS.clear()
    pty = _FakeHookPty(completed=True, pid=2222, expected_answer=None)
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=pty, proxy=_FakeProxy())
    _register_fake(monkeypatch, wrapper)

    async def slow_provider(_request):
        await _asyncio.sleep(10)
        return 'too late'

    # Patch call_runtime_agent to run the function directly (no real runtime).
    call_log: list[tuple[str, str | None]] = []

    async def mock_call(target, **kwargs):
        if target is interactive_runtime.respond_and_wait_claude_code_interactive_session:
            call_log.append((kwargs['interaction_id'], kwargs['answer']))
        runtime_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                'no_thinking_request_args',
                'no_thinking_request_args_mode',
                'haiku_model',
                'runtime_trace_context',
                'channel_id',
                'persistent_channel',
                'openai_client_name',
                'disable_invoke_wait_timeout',
            )
        }
        return await target(**runtime_kwargs)

    monkeypatch.setattr(multiturn, 'call_runtime_agent', mock_call)

    input_data = ClaudeCodeInteractiveInput(
        user_prompt='do something',
        working_dir='/tmp/test',
        get_user_input=slow_provider,
        hook_input_timeout=0.05,
        session_id='timeout-test',
        max_turns=1,
    )
    result = await multiturn.run_installed_multiturn(
        input_data,
        agent_key='claude_code_interactive',
        agent_label='Test PTY',
        session_namespace='test',
        session_turn_target=interactive_runtime.run_claude_code_interactive_session_turn,
        close_session_target=interactive_runtime.close_claude_code_interactive_session,
        respond_session_target=interactive_runtime.respond_and_wait_claude_code_interactive_session,
        runtime_kwargs={'enable_input_bridge': True},
    )

    assert result['success'] is True
    assert call_log[0] == ('tooluse-runtime-1', None)


@pytest.mark.asyncio
async def test_multiturn_cancels_timed_out_provider_then_answers_next_hook(monkeypatch):
    """One provider timeout falls back without ending the current multi-Hook turn."""
    interactive_runtime._PTY_SESSIONS.clear()
    pty = _FakeMultiHookPty(pid=2223)
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=pty, proxy=_FakeProxy())
    _register_fake(monkeypatch, wrapper)

    provider_calls = 0
    first_provider_cancelled = False

    async def provider(_request):
        nonlocal provider_calls, first_provider_cancelled
        provider_calls += 1
        if provider_calls == 1:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                first_provider_cancelled = True
                raise
        return 'yes'

    async def mock_call(target, **kwargs):
        runtime_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in (
                'no_thinking_request_args',
                'no_thinking_request_args_mode',
                'haiku_model',
                'runtime_trace_context',
                'channel_id',
                'persistent_channel',
                'openai_client_name',
                'disable_invoke_wait_timeout',
            )
        }
        return await target(**runtime_kwargs)

    monkeypatch.setattr(multiturn, 'call_runtime_agent', mock_call)

    result = await multiturn.run_installed_multiturn(
        ClaudeCodeInteractiveInput(
            user_prompt='do something',
            working_dir='/tmp/test',
            get_user_input=provider,
            hook_input_timeout=0.05,
            session_id='timeout-next-hook-test',
            max_turns=1,
        ),
        agent_key='claude_code_interactive',
        agent_label='Test PTY',
        session_namespace='test',
        session_turn_target=interactive_runtime.run_claude_code_interactive_session_turn,
        close_session_target=interactive_runtime.close_claude_code_interactive_session,
        respond_session_target=interactive_runtime.respond_and_wait_claude_code_interactive_session,
        runtime_kwargs={'enable_input_bridge': True},
    )

    assert result['success'] is True
    assert first_provider_cancelled is True
    assert provider_calls == 2
    assert pty.responses == [('hook-1', None), ('hook-2', 'yes')]
    assert pty.turns == 1


@pytest.mark.asyncio
async def test_multiturn_provider_exception_falls_back_to_none(monkeypatch):
    """When get_user_input raises, answer=None is used (no crash)."""
    interactive_runtime._PTY_SESSIONS.clear()
    pty = _FakeHookPty(completed=True, pid=3333, expected_answer=None)
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=pty, proxy=_FakeProxy())
    _register_fake(monkeypatch, wrapper)

    async def bad_provider(_request):
        raise RuntimeError('provider exploded')

    call_log: list[tuple[str, str | None]] = []

    async def mock_call(target, **kwargs):
        if target is interactive_runtime.respond_and_wait_claude_code_interactive_session:
            call_log.append((kwargs['interaction_id'], kwargs['answer']))
        runtime_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                'no_thinking_request_args',
                'no_thinking_request_args_mode',
                'haiku_model',
                'runtime_trace_context',
                'channel_id',
                'persistent_channel',
                'openai_client_name',
                'disable_invoke_wait_timeout',
            )
        }
        return await target(**runtime_kwargs)

    monkeypatch.setattr(multiturn, 'call_runtime_agent', mock_call)

    input_data = ClaudeCodeInteractiveInput(
        user_prompt='do something',
        working_dir='/tmp/test',
        get_user_input=bad_provider,
        hook_input_timeout=1.0,
        session_id='exception-test',
        max_turns=1,
    )
    result = await multiturn.run_installed_multiturn(
        input_data,
        agent_key='claude_code_interactive',
        agent_label='Test PTY',
        session_namespace='test',
        session_turn_target=interactive_runtime.run_claude_code_interactive_session_turn,
        close_session_target=interactive_runtime.close_claude_code_interactive_session,
        respond_session_target=interactive_runtime.respond_and_wait_claude_code_interactive_session,
        runtime_kwargs={'enable_input_bridge': True},
    )

    assert result['success'] is True
    assert call_log[0] == ('tooluse-runtime-1', None)


# ---------------------------------------------------------------------------
# latest_screen — pyte viewport rendering
# ---------------------------------------------------------------------------


def test_latest_screen_handles_cursor_movement_and_overwrite(tmp_path):
    """pyte correctly handles cursor moves and overwrites, unlike screen_tail."""
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        rows=5,
        cols=40,
    )
    # Write "AAAA" on line 1, then move cursor back to col 1 and overwrite with "BBBB".
    session._on_data(b'AAAA\x1b[1;1HBBBB')
    screen = session.latest_screen()
    assert 'BBBB' in screen
    assert 'AAAA' not in screen


def test_latest_screen_handles_clear_screen(tmp_path):
    """After clear-screen, pyte viewport is empty."""
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        rows=5,
        cols=40,
    )
    session._on_data(b'Hello world\r\n')
    session._on_data(b'\x1b[2J\x1b[H')  # Clear screen + cursor home
    screen = session.latest_screen()
    assert 'Hello' not in screen


def test_latest_screen_preserves_multiline_layout(tmp_path):
    """Multi-line TUI output preserves row structure."""
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        rows=10,
        cols=80,
    )
    session._on_data(b'Line 1\r\nLine 2\r\nLine 3\r\n')
    screen = session.latest_screen()
    lines = screen.split('\n')
    assert lines[0] == 'Line 1'
    assert lines[1] == 'Line 2'
    assert lines[2] == 'Line 3'


def test_latest_screen_in_turn_result(tmp_path):
    """_turn_result includes latest_screen field from pyte."""
    session = ClaudeInteractiveSession(
        claude_path='/bin/true',
        cwd=str(tmp_path),
        env={},
        done_file=str(tmp_path / 'done'),
        final_file=str(tmp_path / 'final'),
        turn_file=str(tmp_path / 'turn'),
        rows=5,
        cols=40,
    )
    session._on_data(b'visible content')
    session._active_turn_id = 1
    session._active_turn_started = 0.0
    result = session._turn_result(status='completed', completed=True, reason='stop_hook', final_message='done')
    assert 'latest_screen' in result
    assert 'visible content' in result['latest_screen']


# ---------------------------------------------------------------------------
# recent_events — proxy event tracking and truncation
# ---------------------------------------------------------------------------


def test_proxy_event_truncation_head_tail():
    from nexus.extensions.agents.claude_code.interactive.events import _truncate_str

    # Test truncation with head+tail
    text = 'A' * 2000
    truncated, was_cut = _truncate_str(text, head=700, tail=300)
    assert was_cut is True
    assert len(truncated) < 2000
    assert truncated.startswith('A' * 700)
    assert truncated.endswith('A' * 300)
    assert '[truncated]' in truncated

    # No truncation for short strings
    short = 'hello'
    result_str, was_cut = _truncate_str(short, head=700, tail=300)
    assert result_str == 'hello'
    assert was_cut is False


def test_extract_request_events_from_tool_result():
    from nexus.extensions.agents.claude_code.interactive.events import extract_request_events as _extract_request_events

    body = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tu-1',
                        'content': 'short result',
                        'is_error': False,
                    }
                ],
            }
        ]
    }
    events = _extract_request_events(body)
    assert len(events) == 1
    assert events[0]['type'] == 'tool_result'
    assert events[0]['tool_use_id'] == 'tu-1'
    assert events[0]['content'] == 'short result'
    assert events[0]['is_error'] is False
    assert 'truncated' not in events[0]


def test_extract_request_events_truncates_large_content():
    from nexus.extensions.agents.claude_code.interactive.events import extract_request_events as _extract_request_events

    body = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tu-2',
                        'content': 'X' * 5000,
                        'is_error': False,
                    }
                ],
            }
        ]
    }
    events = _extract_request_events(body)
    assert events[0]['truncated'] is True
    assert len(events[0]['content']) < 5000
    assert '[truncated]' in events[0]['content']


def test_extract_request_events_error_gets_larger_limit():
    from nexus.extensions.agents.claude_code.interactive.events import extract_request_events as _extract_request_events

    body = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'tu-3',
                        'content': 'E' * 1500,
                        'is_error': True,
                    }
                ],
            }
        ]
    }
    events = _extract_request_events(body)
    # 1500 chars < 2000 error limit, so no truncation
    assert 'truncated' not in events[0]
    assert events[0]['content'] == 'E' * 1500


def test_extract_response_events_tool_use_and_text():
    from nexus.extensions.agents.claude_code.interactive.events import (
        extract_response_events as _extract_response_events,
    )

    response = {
        'content': [
            {'type': 'text', 'text': 'I will fix the bug'},
            {'type': 'tool_use', 'id': 'tu-5', 'name': 'Edit', 'input': {'file_path': '/a.py', 'old_string': 'x'}},
            {'type': 'thinking', 'thinking': 'internal reasoning'},
        ]
    }
    events = _extract_response_events(response)
    assert len(events) == 2  # thinking is skipped
    assert events[0] == {'type': 'text', 'text': 'I will fix the bug'}
    assert events[1]['type'] == 'tool_use'
    assert events[1]['name'] == 'Edit'
    assert events[1]['input']['file_path'] == '/a.py'


def test_extract_response_events_truncates_tool_input():
    from nexus.extensions.agents.claude_code.interactive.events import (
        extract_response_events as _extract_response_events,
    )

    long_input = 'Z' * 500
    response = {
        'content': [
            {'type': 'tool_use', 'id': 'tu-6', 'name': 'Write', 'input': {'content': long_input}},
        ]
    }
    events = _extract_response_events(response)
    assert events[0]['truncated'] is True
    assert len(events[0]['input']['content']) < 500
    assert '[truncated]' in events[0]['input']['content']


def test_proxy_event_delta_is_incremental_across_hooks_and_turns():
    from nexus.extensions.agents.claude_code.interactive.events import EventBuffer

    buf = EventBuffer()
    buf.mark_turn_start()
    buf.append([{'type': 'tool_use', 'id': 'new-1', 'name': 'Edit', 'input': {}}])

    first_events, first_cursor = buf.get_delta()
    assert first_events == [
        {
            'seq': 1,
            'turn_id': 1,
            'type': 'tool_use',
            'id': 'new-1',
            'name': 'Edit',
            'input': {},
        }
    ]
    assert first_cursor == {
        'after_seq': 0,
        'latest_seq': 1,
        'count': 1,
        'truncated': False,
        'dropped_count': 0,
    }
    buf.commit(first_cursor['latest_seq'])

    buf.append([{'type': 'tool_use', 'id': 'new-2', 'name': 'Bash', 'input': {}}])
    second_events, second_cursor = buf.get_delta()
    assert [event['id'] for event in second_events] == ['new-2']
    assert second_events[0]['seq'] == 2
    assert second_events[0]['turn_id'] == 1
    assert second_cursor['after_seq'] == 1
    buf.commit(second_cursor['latest_seq'])

    buf.mark_turn_start()
    buf.append([{'type': 'text', 'text': 'turn two'}])
    third_events, third_cursor = buf.get_delta()
    assert third_events[0]['seq'] == 3
    assert third_events[0]['turn_id'] == 2
    buf.commit(third_cursor['latest_seq'])

    empty_events, empty_cursor = buf.get_delta()
    assert empty_events == []
    assert empty_cursor['after_seq'] == empty_cursor['latest_seq'] == 3


def test_proxy_event_delta_has_high_safety_limit():
    from nexus.extensions.agents.claude_code.interactive.events import EventBuffer

    buf = EventBuffer()
    buf.mark_turn_start()
    buf.append([{'type': 'text', 'text': str(index)} for index in range(1200)])

    events, cursor = buf.get_delta()

    assert len(events) == 1000
    assert events[0]['seq'] == 201
    assert events[-1]['seq'] == 1200
    assert cursor == {
        'after_seq': 0,
        'latest_seq': 1200,
        'count': 1000,
        'truncated': True,
        'dropped_count': 200,
    }
    buf.commit(cursor['latest_seq'])
    assert buf.get_delta()[0] == []


# ---------------------------------------------------------------------------
# runtime integration: latest_screen + recent_events in formatted result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_runtime_turn_result_includes_new_fields(monkeypatch):
    interactive_runtime._PTY_SESSIONS.clear()
    fake_pty = _FakePty(completed=True, pid=9999)
    fake_proxy = _FakeProxy()
    # Simulate an event appended during the turn (after mark_turn_start runs).
    original_mark = fake_proxy.mark_turn_start

    def mark_then_add():
        original_mark()
        fake_proxy.append_events([{'type': 'tool_use', 'id': 'tu-1', 'name': 'Read', 'input': {'file_path': '/a.py'}}])

    fake_proxy.mark_turn_start = mark_then_add
    wrapper = interactive_runtime._InteractiveRuntimeSession(pty_session=fake_pty, proxy=fake_proxy)
    _register_fake(monkeypatch, wrapper)

    result = await interactive_runtime.run_claude_code_interactive_session_turn(
        session_id='fields-test', prompt='do something'
    )

    assert result['success'] is True
    assert 'latest_screen' in result
    assert 'recent_events' in result
    assert result['recent_events'] == [
        {
            'type': 'tool_use',
            'id': 'tu-1',
            'name': 'Read',
            'input': {'file_path': '/a.py'},
            'seq': 1,
            'turn_id': 1,
        },
    ]
    assert result['event_cursor'] == {
        'after_seq': 0,
        'latest_seq': 1,
        'count': 1,
        'truncated': False,
        'dropped_count': 0,
    }
    assert fake_proxy._last_emitted_seq == 1
    assert fake_proxy.get_event_delta()[0] == []

    await interactive_runtime.close_claude_code_interactive_session(session_id='fields-test')


# ---------------------------------------------------------------------------
# multiturn propagation: metadata carries latest_screen + recent_events
# ---------------------------------------------------------------------------


async def test_build_user_input_request_propagates_latest_screen_and_recent_events():
    result = {
        'success': True,
        'output': 'done',
        'events': [{'type': 'pty_turn'}],
        'latest_screen': 'Line 1\nLine 2',
        'recent_events': [{'type': 'tool_use', 'name': 'Read'}],
        'event_cursor': {
            'after_seq': 0,
            'latest_seq': 1,
            'count': 1,
            'truncated': False,
            'dropped_count': 0,
        },
    }
    request = await multiturn.build_user_input_request(
        agent_label='Test',
        output='done',
        result=result,
        session_id='s1',
        turn=1,
    )
    assert request.metadata['latest_screen'] == 'Line 1\nLine 2'
    assert request.metadata['recent_events'] == [{'type': 'tool_use', 'name': 'Read'}]
    assert request.metadata['event_cursor']['latest_seq'] == 1


async def test_build_user_input_request_can_log_pty_context(monkeypatch, caplog):
    monkeypatch.setenv('NEXUS_PTY_CONTEXT_DEBUG', '1')
    caplog.set_level(logging.DEBUG, logger=multiturn.__name__)
    result = {
        'latest_screen': 'Line 1\nLine 2',
        'recent_events': [{'type': 'tool_use', 'name': 'Read'}],
        'event_cursor': {
            'after_seq': 0,
            'latest_seq': 1,
            'count': 1,
            'truncated': False,
            'dropped_count': 0,
        },
    }

    await multiturn.build_user_input_request(
        agent_label='Claude Code',
        output='done',
        result=result,
        session_id='session-1',
        turn=1,
    )

    context_record = next(record for record in caplog.records if record.message.startswith('[pty-user-context] '))
    assert 'session_id=session-1' in context_record.message
    assert 'turn=1' in context_record.message
    assert 'latest_screen_chars=13' in context_record.message
    assert 'recent_events=1' in context_record.message


async def test_build_user_input_request_can_dump_exact_pty_context(monkeypatch, tmp_path):
    debug_path = tmp_path / 'pty-context.jsonl'
    monkeypatch.setenv('NEXUS_PTY_CONTEXT_DUMP_PATH', str(debug_path))
    result = {
        'latest_screen': 'Line 1\nLine 2',
        'recent_events': [{'type': 'tool_use', 'name': 'Read'}],
        'event_cursor': {
            'after_seq': 0,
            'latest_seq': 1,
            'count': 1,
            'truncated': False,
            'dropped_count': 0,
        },
    }

    await multiturn.build_user_input_request(
        agent_label='Claude Code',
        output='done',
        result=result,
        session_id='session-1',
        turn=1,
    )

    await multiturn.build_user_input_request(
        agent_label='Claude Code',
        output='done',
        result=result,
        session_id='session-1',
        turn=2,
    )

    lines = debug_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])['turn'] == 1
    payload = json.loads(lines[1])
    assert payload == {
        'session_id': 'session-1',
        'turn': 2,
        'hook': None,
        'latest_screen': 'Line 1\nLine 2',
        'recent_events': [{'type': 'tool_use', 'name': 'Read'}],
        'event_cursor': {
            'after_seq': 0,
            'latest_seq': 1,
            'count': 1,
            'truncated': False,
            'dropped_count': 0,
        },
    }


async def test_build_user_input_request_dumps_full_exit_plan_hook(monkeypatch, tmp_path):
    debug_path = tmp_path / 'pty-context.jsonl'
    monkeypatch.setenv('NEXUS_PTY_CONTEXT_DUMP_PATH', str(debug_path))
    result = {
        'interaction': {
            'interaction_id': 'exit-plan-1',
            'tool_name': 'ExitPlanMode',
            'tool_input': {
                'plan': '1. Inspect\n2. Implement\n3. Test',
                'plan_path': '/root/.claude/plans/current.md',
            },
        },
        'latest_screen': 'Plan ready',
        'recent_events': [{'type': 'tool_use', 'name': 'ExitPlanMode', 'input': {}}],
    }

    request = await multiturn.build_user_input_request(
        agent_label='Claude Code',
        output='',
        result=result,
        session_id='session-1',
        turn=1,
    )

    payload = json.loads(debug_path.read_text().strip().splitlines()[-1])
    assert request.hook == payload['hook']
    assert request.hook['input']['plan'] == '1. Inspect\n2. Implement\n3. Test'
    assert request.hint == 'Claude Code wants to exit plan mode with this plan:\n1. Inspect\n2. Implement\n3. Test'
