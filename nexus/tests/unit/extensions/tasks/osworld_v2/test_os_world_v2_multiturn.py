"""Actual hook file protocol tests without a model endpoint."""

import asyncio
import json
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.claude_code.interactive import hooks
from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeRunRequest
from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import InteractiveAGSAdapter
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['user_limit', 'done', 'cli_limit', 'api_error', 'cancel'])
async def test_real_profile_distinguishes_loop_limit_from_completion(monkeypatch, outcome):
    from nexus.core import ExecutionContext
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeMultiturnInput
    from nexus.extensions.tasks.os_world_v2.adapter import claude_multiturn

    callback = AsyncMock(return_value='continue' if outcome == 'user_limit' else None)
    inp = ClaudeCodeMultiturnInput(
        user_prompt='fixture',
        working_dir='/home/user',
        execution_profile='osworld_v2',
        execution_options={'guide_version': 'v2', 'max_turns': 1},
        max_turns=1,
        get_user_input=callback,
    )
    terminal = {
        'type': 'result',
        'session_id': 'parent',
        'subtype': 'error_max_turns' if outcome == 'cli_limit' else 'success',
        'is_error': outcome in {'cli_limit', 'api_error'},
        'result': 'API Error: upstream failed'
        if outcome == 'api_error'
        else 'TASK_DONE'
        if outcome == 'done'
        else 'Continue work',
    }
    stdout = '\n'.join(
        json.dumps(x)
        for x in [
            {'type': 'system', 'subtype': 'init', 'session_id': 'parent'},
            terminal,
        ]
    )
    result = SimpleNamespace(exit_code=1 if terminal['is_error'] else 0, stdout=stdout, stderr='')
    adapter = SimpleNamespace(
        ensure_claude=AsyncMock(return_value='/bin/claude'),
        run_claude=AsyncMock(
            return_value=result, side_effect=asyncio.CancelledError() if outcome == 'cancel' else None
        ),
        diagnostics={},
        interactions=[],
    )
    monkeypatch.setattr(claude_multiturn, 'InteractiveAGSAdapter', lambda *a, **kw: adapter)
    profile = claude_multiturn.OSWorldV2MultiturnProfile(inp)
    async with ExecutionContext(task_name='os_world_v2', agent_name='claude_code_multiturn') as ctx:
        if outcome == 'cancel':
            with pytest.raises(asyncio.CancelledError):
                await profile.execute(runtime=object(), launch=object(), timeout=10)
            return
        raw = await profile.execute(runtime=object(), launch=object(), timeout=10)
        normalized = ctx.tool_state['osworld_agent_output']
    expected = {
        'user_limit': 'max_iterations',
        'done': 'finished',
        'cli_limit': 'max_iterations',
        'api_error': 'inference_error',
    }[outcome]
    assert normalized.exit_status == expected
    assert normalized.exit_code == (-1 if outcome == 'api_error' else 0)
    assert normalized.process_exit_code == result.exit_code
    assert raw['reached_max_turns'] is (outcome in {'user_limit', 'cli_limit'})
    if outcome in {'user_limit', 'done'}:
        callback.assert_awaited_once()
    else:
        callback.assert_not_awaited()
    # The CLI limit is delivered separately from the outer user-turn budget.
    args = adapter.run_claude.await_args.args[0].extra_cli_args
    assert args[args.index('--max-turns') + 1] == '1'
    adapter.run_claude.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(('native_protocol', 'expected'), [(None, 'osworld-anthropic'), ('anthropic', 'anthropic')])
async def test_v2_document_conversion_follows_client_capability(monkeypatch, native_protocol, expected):
    from nexus.extensions.tasks.os_world_v2.adapter import claude_multiturn
    from nexus.extensions.tasks.os_world_v2.execution.claude_code import OSWorldExecutionOptions

    call = AsyncMock()
    monkeypatch.setattr(claude_multiturn, 'call_native_bridge_runtime_agent', call)
    profile = object.__new__(claude_multiturn.OSWorldV2MultiturnProfile)
    profile.options = OSWorldExecutionOptions()
    await profile.run(SimpleNamespace(native_request_protocol=native_protocol))
    assert call.await_args.kwargs['api_protocol'] == expected


class LocalRuntime:
    from nexus.runtime.providers.ags.handle import AGSInstanceHandle as _Handle
    from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime as _BaseRuntime

    _build_osworld_execute_payload = staticmethod(_BaseRuntime._build_osworld_execute_payload)
    handle = _Handle('fixture', 'fixture', 'example.invalid')
    server_port = 5000

    async def read_only_request(self, method, url, **kwargs):
        from nexus.utils.async_http import HTTPResponse

        command = kwargs['json']['command']
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        text = json.dumps({'returncode': process.returncode, 'output': out.decode(), 'error': err.decode()})
        return HTTPResponse(200, {}, text, text.encode(), url, 'utf-8')

    async def run_command(self, command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        return SimpleNamespace(return_code=process.returncode, stdout=out.decode(), stderr=err.decode())

    async def setup_upload_bytes(self, path, data, **kwargs):
        Path(path).write_bytes(data)
        return SimpleNamespace(status_code=200)


@pytest.mark.asyncio
async def test_real_ask_user_hook_roundtrip(tmp_path):
    adapter = InteractiveAGSAdapter(object.__new__(AGSOSWorldRuntime), AsyncMock(return_value='blue'))
    adapter.runtime = LocalRuntime()
    adapter.interaction_dir = str(tmp_path)
    provider = adapter.get_user_input
    import os

    environment = {**os.environ, hooks.INTERACTION_DIR_ENV: str(tmp_path), hooks.HOOK_WAIT_TIMEOUT_ENV: '10'}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        hooks.__file__,
        'AskUserQuestion',
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    responder = asyncio.create_task(adapter._service_interactions(adapter.runtime))
    try:
        out, err = await asyncio.wait_for(
            process.communicate(
                json.dumps({
                    'tool_use_id': 'test-iid',
                    'tool_input': {'questions': [{'question': 'Which color?'}]},
                }).encode()
            ),
            15,
        )
        assert process.returncode == 0, err.decode()
        reply = json.loads(out)['hookSpecificOutput']
        assert reply['updatedInput']['answers'] == {'Which color?': 'blue'}
        assert provider.await_args.args[0].source == 'hook'
        assert provider.await_args.args[0].hook['tool_name'] == 'AskUserQuestion'
    finally:
        responder.cancel()
        await asyncio.gather(responder, return_exceptions=True)
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_input_hooks_preserve_bash_protections(tmp_path):
    adapter = InteractiveAGSAdapter(object.__new__(AGSOSWorldRuntime), AsyncMock(return_value='ok'))
    adapter.runtime = LocalRuntime()
    adapter.process_guard = True
    request = ClaudeRunRequest(
        'claude', SimpleNamespace(local_base_url='http://127.0.0.1:9281'), 'test', str(tmp_path), 30
    )
    result = await adapter._install_guard(request, str(tmp_path))
    path = result.extra_cli_args[-1]
    settings = json.loads(Path(path).read_text())
    entries = settings['hooks']['PreToolUse']
    assert [entry['matcher'] for entry in entries] == ['Bash', 'AskUserQuestion', 'ExitPlanMode']
    assert len(entries[0]['hooks']) == 2
    assert result.extra_env[hooks.INTERACTION_DIR_ENV] == str(tmp_path / 'interactions')


@pytest.mark.asyncio
async def test_empty_simulator_answer_fails_closed(tmp_path):
    adapter = InteractiveAGSAdapter(object.__new__(AGSOSWorldRuntime), AsyncMock(return_value=None))
    adapter.runtime = LocalRuntime()
    adapter.interaction_dir = str(tmp_path)
    (tmp_path / 'test.request.json').write_text(
        json.dumps({
            'interaction_id': 'test',
            'tool_name': 'ExitPlanMode',
            'tool_input': {},
        })
    )
    with pytest.raises(RuntimeError, match='refusing unattended approval'):
        await adapter._service_interactions(adapter.runtime)
    assert not list(tmp_path.glob('*.response.json'))


@pytest.mark.asyncio
async def test_unattended_hook_denies_without_creating_interaction(tmp_path):
    adapter = InteractiveAGSAdapter(object.__new__(AGSOSWorldRuntime), AsyncMock(), user_simulator=False)
    adapter.runtime = LocalRuntime()
    request = ClaudeRunRequest(
        'claude', SimpleNamespace(local_base_url='http://127.0.0.1:9281'), 'test', str(tmp_path), 30
    )
    await adapter._install_guard(request, str(tmp_path))
    for tool in ('AskUserQuestion', 'ExitPlanMode'):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(tmp_path / 'input-hook.py'),
            tool,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await process.communicate(b'{"tool_input":{"questions":[{"question":"Approve?"}]}}')
        assert process.returncode == 0
        reply = json.loads(out)['hookSpecificOutput']
        assert reply['permissionDecision'] == 'deny'
        assert 'updatedInput' not in reply
    assert not list((tmp_path / 'interactions').iterdir())
    adapter.get_user_input.assert_not_called()


@pytest.mark.asyncio
async def test_unattended_poll_does_not_start_interaction_service(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    adapter = InteractiveAGSAdapter(object.__new__(AGSOSWorldRuntime), AsyncMock(), user_simulator=False)
    adapter._service_interactions = AsyncMock()
    monkeypatch.setattr(OSWorldClaudeNativeBridgeAdapter, '_poll_until_done', AsyncMock(return_value=(0, False, False)))
    paths = SimpleNamespace(exit_code='/tmp/test/exit', pid='/tmp/test/pid')
    assert await adapter._poll_until_done(paths, 1) == (0, False, False)
    adapter._service_interactions.assert_not_called()


def test_v2_parent_drain_uses_root_and_verified_cleanup():
    from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import parent_drain_finished

    events = [
        {'type': 'assistant', 'session_id': 'root', 'message': {'content': [{'type': 'text', 'text': 'TASK_DONE'}]}},
        {'type': 'result', 'session_id': 'child', 'parent_tool_use_id': 'sub', 'is_error': False, 'result': 'child'},
    ]
    diagnostics = {
        'budget_reason': 'parent_terminal_grace',
        'budget_stop_confirmed': True,
        'parent_terminal_session': 'root',
        'parent_terminal_marker': 'TASK_DONE',
    }
    raw = '\n'.join(map(json.dumps, events))
    assert parent_drain_finished(raw, -1, diagnostics, 60)
    assert not parent_drain_finished(raw, 1, diagnostics, 60)
    assert not parent_drain_finished(raw, -1, {**diagnostics, 'budget_stop_confirmed': False}, 60)
    events.append({
        'type': 'assistant',
        'session_id': 'root',
        'message': {'content': [{'type': 'text', 'text': 'more work'}]},
    })
    assert not parent_drain_finished('\n'.join(map(json.dumps, events)), -1, diagnostics, 60)


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_v2_default_guide_only_requests_done(version):
    from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import default_guide

    guide = default_guide(version)
    assert 'TASK_DONE' in guide
    assert 'TASK_INFEASIBLE' not in guide


def test_v2_infeasible_does_not_arm_or_confirm_parent_drain():
    from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import (
        V2ParentDrainState,
        parent_drain_finished,
    )
    from nexus.extensions.tasks.os_world_v2.execution.claude_code import _ParentDrainState

    event = {
        'type': 'assistant',
        'session_id': 'root',
        'message': {'content': [{'type': 'text', 'text': 'TASK_INFEASIBLE'}]},
    }
    state = V2ParentDrainState(1)
    state.feed(event, 0)
    assert state.deadline is None
    assert not state.expired(10)
    diagnostics = {
        'budget_reason': 'parent_terminal_grace',
        'budget_stop_confirmed': True,
        'parent_terminal_session': 'root',
        'parent_terminal_marker': 'TASK_INFEASIBLE',
    }
    assert not parent_drain_finished(json.dumps(event), -1, diagnostics, 1)
    v1 = _ParentDrainState(1)
    v1.feed(event, 0)
    assert v1.expired(10)


@pytest.mark.asyncio
async def test_guest_supervisor_ignores_child_result(tmp_path):
    from nexus.extensions.tasks.os_world_v2.adapter import cc_session

    script = tmp_path / 'fake_cli.py'
    script.write_text(
        'import json,sys\n'
        'init=json.loads(input())\n'
        'print(json.dumps({"type":"control_response","response":{"request_id":init["request_id"],"subtype":"success"}}),flush=True)\n'
        'prompt=json.loads(input())\n'
        'print(json.dumps({"type":"result","session_id":"child","parent_tool_use_id":"sub","is_error":True}),flush=True)\n'
        'print(json.dumps({"type":"result","session_id":prompt["session_id"],"is_error":False,"result":"TASK_DONE"}),flush=True)\n'
    )
    (tmp_path / 'prompt.txt').write_text('test')
    spec = tmp_path / 'spec.json'
    spec.write_text(
        json.dumps({
            'command': [sys.executable, str(script)],
            'prompt_file': str(tmp_path / 'prompt.txt'),
            'session_id': 'root',
            'env': {},
        })
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        cc_session.__file__,
        str(spec),
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await asyncio.wait_for(process.communicate(), 15)
    assert process.returncode == 0
    results = [json.loads(line) for line in out.splitlines() if json.loads(line)['type'] == 'result']
    assert len(results) == 2
    assert results[-1]['session_id'] == 'root'
