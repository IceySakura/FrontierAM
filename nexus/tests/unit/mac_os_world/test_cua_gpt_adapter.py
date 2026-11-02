import base64
import copy
import json
import re
import sys

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.cua.cua_gpt_agent import agent as cua_agent
from nexus.extensions.agents.cua.macosworld_cua import gpt as mac
from nexus.extensions.agents.cua.macosworld_cua.gpt import MacOSWorldGptInput as CuaGptAgentInput


class _FakeResponsesClient:
    def __init__(self, completions: list[Any]) -> None:
        self.completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    async def create_completion(self, messages: list[Any], model: str | None = None, **kwargs: Any) -> Any:
        self.calls.append({'messages': copy.deepcopy(messages), 'model': model, **copy.deepcopy(kwargs)})
        return self.completions.pop(0)


class _FakeRuntime:
    def __init__(self, screenshots: list[str] | None = None) -> None:
        self.screenshots = iter(screenshots or [base64.b64encode(b'screenshot').decode()])
        self.codes: list[str] = []
        self.events: list[Any] = []

    async def gui_screenshot(self) -> dict[str, str]:
        self.events.append('screenshot')
        return {'image': next(self.screenshots)}

    async def gui_pyautogui(self, code: str, timeout: float) -> dict[str, Any]:
        self.events.append(('action', code))
        self.codes.append(code)
        return {'success': True}


def _context(client: _FakeResponsesClient, runtime: _FakeRuntime) -> SimpleNamespace:
    async def get_runtime(name: str) -> _FakeRuntime:
        assert name == 'main'
        return runtime

    def get_openai_client(name: str) -> _FakeResponsesClient:
        assert name == 'main'
        return client

    return SimpleNamespace(get_runtime=get_runtime, get_openai_client=get_openai_client, tool_state={})


def _tool_call(call_id: str, arguments: str, *, name: str = 'computer') -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _computer_call(
    call_id: str,
    actions: list[dict[str, Any]],
    *,
    safety: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    return _tool_call(
        call_id,
        json.dumps({
            'actions': actions,
            'pending_safety_checks': safety or [],
            'raw_item': {'type': 'computer_call', 'call_id': call_id},
        }),
    )


def _completion(
    response_id: str,
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[Any] | None = None,
    status: str | None = 'completed',
    finish_reason: str | None = None,
) -> SimpleNamespace:
    provider_fields: dict[str, Any] = {}
    if status is not None:
        provider_fields['response_status'] = status
        provider_fields['response_state'] = {'status': status}
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason or ('tool_calls' if tool_calls else 'stop'),
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
            )
        ],
        provider_specific_fields=provider_fields,
    )


def _clipboard_text(code: str) -> str:
    match = re.search(r"base64\.b64decode\('([^']+)'\)", code)
    assert match is not None
    return base64.b64decode(match.group(1)).decode('utf-8')


@pytest.mark.parametrize('text', ['你好', 'echo $HOME | cat', 'first line\nsecond line\n'])
def test_shared_text_input_executes_command_v_on_macos(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[tuple[str, ...]] = []
    clipboard: list[str] = []
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(hotkey=lambda *args: keys.append(args)))
    monkeypatch.setitem(sys.modules, 'pyperclip', SimpleNamespace(copy=clipboard.append, paste=lambda: clipboard[-1]))
    monkeypatch.setattr('time.sleep', lambda _: None)
    code = mac._convert_action_to_pyautogui('type', {'text': text}, platform_name='macos')
    assert code is not None
    exec(code, {})
    assert clipboard == [text]
    assert keys == [('command', 'v')]


def test_type_macos_non_ascii_uses_command_v_clipboard_paste() -> None:
    code = mac._convert_action_to_pyautogui('type', {'text': '你好'}, platform_name='macOS')

    assert code is not None
    assert _clipboard_text(code) == '你好'
    assert "keys = ('command', 'v')" in code
    assert "pyautogui.hotkey('ctrl', 'v')" not in code
    assert 'pyautogui.typewrite' not in code


def test_type_multiline_ascii_uses_clipboard_paste_preserving_newline() -> None:
    text = 'first line\nsecond line\n'

    code = mac._convert_action_to_pyautogui('type', {'text': text}, platform_name='ubuntu')

    assert code is not None
    assert _clipboard_text(code) == text
    assert "keys = ('ctrl', 'v')" in code
    assert 'pyautogui.typewrite' not in code
    assert "pyautogui.press('enter')" not in code


def test_type_macos_shell_special_uses_command_v_clipboard_paste() -> None:
    text = 'echo $HOME | cat'

    code = mac._convert_action_to_pyautogui('type', {'text': text}, platform_name='macOS')

    assert code is not None
    assert _clipboard_text(code) == text
    assert "keys = ('command', 'v')" in code
    assert "pyautogui.hotkey('ctrl', 'v')" not in code
    assert 'pyautogui.typewrite' not in code


def test_type_short_ascii_uses_typewrite() -> None:
    code = mac._convert_action_to_pyautogui('type', {'text': 'echo hello'}, platform_name='macOS')

    assert code == "import pyautogui\npyautogui.typewrite('echo hello', interval=0.03)"


@pytest.mark.parametrize(
    ('platform_name', 'button', 'expected_hotkey'),
    [
        ('ubuntu', 'back', "pyautogui.hotkey('alt', 'left')"),
        ('ubuntu', 'forward', "pyautogui.hotkey('alt', 'right')"),
        ('macOS', 'back', "pyautogui.hotkey('command', '[')"),
        ('macOS', 'forward', "pyautogui.hotkey('command', ']')"),
    ],
)
def test_navigation_click_buttons_use_platform_safe_shortcuts(
    platform_name: str,
    button: str,
    expected_hotkey: str,
) -> None:
    args = {'x': 10, 'y': 20, 'button': button}

    cua_agent._validate_computer_action('click', args)
    code = mac._convert_action_to_pyautogui('click', args, platform_name=platform_name)

    assert code == f'pyautogui.moveTo(10, 20)\n{expected_hotkey}'


@pytest.mark.asyncio
@pytest.mark.parametrize('home_dir', [None, '/Users/bkdevops'])
async def test_agent_uses_context_responses_client_and_preserves_macos_behavior(
    monkeypatch: pytest.MonkeyPatch,
    home_dir: str | None,
) -> None:
    initial_b64 = base64.b64encode(b'initial').decode()
    after_actions_b64 = base64.b64encode(b'after-actions').decode()
    actions = [
        {'type': 'type', 'text': 'echo $HOME'},
        {'type': 'keypress', 'keys': ['ENTER']},
    ]
    safety = [{'id': 'safe-1'}]
    client = _FakeResponsesClient([
        _completion(
            'resp-1',
            reasoning='Act on the desktop.',
            tool_calls=[_computer_call('call-1', actions, safety=safety)],
        ),
        _completion('resp-2', content='Done.'),
    ])
    runtime = _FakeRuntime([initial_b64, after_actions_b64])
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish the task',
            model='gpt-cua-test',
            platform_name='macOS',
            home_dir=home_dir,
            max_iterations=2,
            max_output_tokens=321,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    assert len(client.calls) == 2
    first_call = client.calls[0]
    assert first_call['model'] == 'gpt-cua-test'
    assert first_call['tools'] == [{'type': 'computer'}]
    assert first_call['parallel_tool_calls'] is False
    assert first_call['reasoning'] == {
        'context': 'all_turns',
        'effort': 'xhigh',
        'summary': 'concise',
        'mode': 'standard',
    }
    assert first_call['truncation'] == 'auto'
    assert first_call['max_output_tokens'] == 321
    assert 'previous_response_id' not in first_call
    assert f'- The home directory is "{home_dir or "/Users/ec2-user"}".' in first_call['instructions']
    assert '/home/user' not in first_call['instructions']
    assert client.calls[1]['previous_response_id'] == 'resp-1'
    assert client.calls[1]['messages'] == [
        {
            'type': 'computer_call_output',
            'call_id': 'call-1',
            'output': {
                'type': 'computer_screenshot',
                'image_url': f'data:image/png;base64,{after_actions_b64}',
                'detail': 'original',
            },
            'acknowledged_safety_checks': safety,
        }
    ]

    expected_type = mac._convert_action_to_pyautogui(
        'type',
        {'text': 'echo $HOME'},
        platform_name='macOS',
    )
    assert expected_type is not None
    expected_keypress = "pyautogui.hotkey('enter')"
    assert runtime.codes == [expected_type, expected_keypress]
    assert "keys = ('command', 'v')" in runtime.codes[0]
    assert output.exit_status == 'finished'
    assert output.final_answer == 'Done.'
    assert output.main_trajectory[0]['execution_results'] == [
        {'action_type': 'type', 'code': expected_type, 'status': 'success'},
        {'action_type': 'keypress', 'code': expected_keypress, 'status': 'success'},
    ]
    assert output.main_trajectory[0]['action_result'] == 'Success'


@pytest.mark.asyncio
async def test_agent_preserves_multiple_computer_call_order(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_b64 = base64.b64encode(b'initial').decode()
    after_first_b64 = base64.b64encode(b'after-first').decode()
    after_second_b64 = base64.b64encode(b'after-second').decode()
    client = _FakeResponsesClient([
        _completion(
            'resp-1',
            tool_calls=[
                _computer_call('call-1', [{'type': 'click', 'x': 10, 'y': 20}]),
                _computer_call(
                    'call-2',
                    [{'type': 'keypress', 'key': 'ENTER'}],
                    safety=[{'id': 'safe-2'}],
                ),
            ],
        ),
        _completion('resp-2', content='Done.'),
    ])
    runtime = _FakeRuntime([initial_b64, after_first_b64, after_second_b64])
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish',
            max_iterations=2,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    assert output.exit_status == 'finished'
    assert runtime.codes == [
        "pyautogui.moveTo(10, 20)\npyautogui.click(button='left')",
        "pyautogui.hotkey('enter')",
    ]
    assert [item['call_id'] for item in client.calls[1]['messages']] == ['call-1', 'call-2']
    assert client.calls[1]['messages'][1]['acknowledged_safety_checks'] == [{'id': 'safe-2'}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status', 'finish_reason'),
    [
        ('incomplete', 'tool_calls'),
        (None, 'length'),
    ],
)
async def test_incomplete_response_never_executes_or_marks_done(
    status: str | None,
    finish_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeResponsesClient([
        _completion(
            'resp-incomplete',
            content='Partial text.',
            tool_calls=[_computer_call('call-partial', [{'type': 'click', 'x': 1, 'y': 2}])],
            status=status,
            finish_reason=finish_reason,
        )
    ])
    runtime = _FakeRuntime()
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish',
            max_iterations=1,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    assert output.exit_status == 'incomplete'
    assert output.final_answer == 'Model response was incomplete.'
    assert output.main_trajectory[0]['incomplete'] is True
    assert output.main_trajectory[0]['num_actions'] == 0
    assert runtime.codes == []
    assert ctx.tool_state['os_world_last_action'] == {'action_type': 'FAIL'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('call_id', 'name', 'arguments'),
    [
        ('call-bad-json', 'computer', '{"actions":'),
        ('call-not-object', 'computer', '[]'),
        ('call-bad-actions', 'computer', json.dumps({'actions': {}, 'pending_safety_checks': []})),
        ('call-bad-safety', 'computer', json.dumps({'actions': [], 'pending_safety_checks': {}})),
        ('', 'computer', json.dumps({'actions': [{'type': 'click', 'x': 1, 'y': 2}], 'pending_safety_checks': []})),
        (
            'call-wrong-tool',
            'shell',
            json.dumps({'actions': [{'type': 'click', 'x': 1, 'y': 2}], 'pending_safety_checks': []}),
        ),
    ],
)
async def test_invalid_tool_call_fails_safely_without_gui_execution(
    call_id: str,
    name: str,
    arguments: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeResponsesClient([_completion('resp-bad', tool_calls=[_tool_call(call_id, arguments, name=name)])])
    runtime = _FakeRuntime()
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish',
            max_iterations=1,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    safe_call_id = call_id or '<missing>'
    assert output.exit_status == 'tool_error'
    assert output.final_answer == f'Invalid computer tool call: {safe_call_id}'
    assert output.main_trajectory[0]['tool_error'] == output.final_answer
    assert arguments not in output.final_answer
    assert runtime.codes == []
    assert ctx.tool_state['os_world_last_action'] == {'action_type': 'FAIL'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'invalid_action',
    [
        {'type': 'click', 'x': "1); __import__('os').system('id') #", 'y': 20},
        {'type': 'move', 'x': float('inf'), 'y': 20},
        {'type': 'double_click', 'x': True, 'y': 20},
        {'type': 'drag', 'path': [[1, 2], [3, '4']]},
        {'type': 'type', 'text': {'unexpected': 'mapping'}},
        {'type': 'keypress', 'keys': ['CTRL', 7]},
        {'type': 'scroll', 'scroll_y': float('nan')},
        {'type': 'wait', 'ms': True},
    ],
)
async def test_later_invalid_action_makes_entire_turn_atomic(
    invalid_action: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeResponsesClient([
        _completion(
            'resp-invalid',
            tool_calls=[
                _computer_call('call-valid', [{'type': 'click', 'x': 10, 'y': 20}]),
                _computer_call('call-invalid', [invalid_action]),
            ],
        )
    ])
    runtime = _FakeRuntime()
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish',
            max_iterations=1,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    assert output.exit_status == 'tool_error'
    assert output.final_answer == 'Invalid computer tool call: call-invalid'
    assert output.main_trajectory[0]['num_actions'] == 0
    assert output.main_trajectory[0]['actions'] == []
    assert runtime.codes == []
    assert ctx.tool_state['os_world_last_action'] == {'action_type': 'FAIL'}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_mode', ['response', 'exception'])
async def test_first_action_failure_stops_later_actions_and_records_safe_metadata(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_error = 'api_key=do-not-persist'

    class _FailureRuntime(_FakeRuntime):
        async def gui_pyautogui(self, code: str, timeout: float) -> dict[str, Any]:
            self.events.append(('action', code))
            self.codes.append(code)
            if len(self.codes) == 1:
                return {'success': True}
            if failure_mode == 'exception':
                raise RuntimeError(unsafe_error)
            return {'success': False, 'error': unsafe_error, 'stderr': unsafe_error}

    actions = [
        {'type': 'click', 'x': 10, 'y': 20},
        {'type': 'keypress', 'keys': ['CTRL', 'L']},
        {'type': 'type', 'text': 'must not execute'},
    ]
    client = _FakeResponsesClient([_completion('resp-failure', tool_calls=[_computer_call('call-failure', actions)])])
    runtime = _FailureRuntime()
    ctx = _context(client, runtime)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: ctx)

    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='finish',
            max_iterations=1,
            sleep_after_execution=0,
            step_pause=0,
        )
    )

    expected_click = "pyautogui.moveTo(10, 20)\npyautogui.click(button='left')"
    expected_keypress = "pyautogui.hotkey('ctrl', 'l')"
    expected_type = "import pyautogui\npyautogui.typewrite('must not execute', interval=0.03)"
    assert runtime.codes == [expected_click, expected_keypress]
    assert output.exit_status == 'action_error'
    assert output.final_answer == 'Computer action execution failed.'
    assert output.main_trajectory[0]['execution_results'] == [
        {'action_type': 'click', 'code': expected_click, 'status': 'success'},
        {
            'action_type': 'keypress',
            'code': expected_keypress,
            'status': 'failed',
            'error': 'Computer action execution failed.',
            **({'diagnostics': {'exception_type': 'RuntimeError'}} if failure_mode == 'exception' else {}),
        },
        {'action_type': 'type', 'code': expected_type, 'status': 'not_executed'},
    ]
    assert output.main_trajectory[0]['action_result'] == 'Failed'
    assert output.main_trajectory[0]['action_error'] == 'Computer action execution failed.'
    assert unsafe_error not in repr(output.main_trajectory)
    assert ctx.tool_state['os_world_last_action'] == {'action_type': 'FAIL'}


@pytest.mark.asyncio
async def test_command_failure_records_status_without_leaking_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailureRuntime(_FakeRuntime):
        async def run_command(self, **kwargs):
            return {'status': 'TimeLimitExceeded', 'return_code': 124, 'stderr': 'api_key=do-not-persist'}

    client = _FakeResponsesClient([
        _completion('response', tool_calls=[_computer_call('call', [{'type': 'wait'}])]),
    ])
    monkeypatch.setattr(cua_agent, 'require_context', lambda: _context(client, FailureRuntime()))
    output = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='wait',
            max_iterations=1,
            sleep_after_execution=0,
            step_pause=0,
            pyautogui_execution_mode='run_command',
        )
    )
    assert output.exit_status == 'action_error'
    assert output.main_trajectory[0]['execution_results'][0]['diagnostics'] == {
        'command_status': 'TimeLimitExceeded',
        'return_code': 124,
    }
    assert 'do-not-persist' not in repr(output.main_trajectory)


def _with_native_output(completion: Any, call_id: str) -> Any:
    completion.choices[0].message.provider_specific_fields = {
        'responses_output_items': [
            {'type': 'reasoning', 'id': f'rs-{call_id}', 'encrypted_content': f'encrypted-{call_id}'},
            {
                'type': 'computer_call',
                'id': f'cu-{call_id}',
                'call_id': call_id,
                'actions': [{'type': 'wait'}],
                'status': 'completed',
            },
        ],
    }
    return completion


@pytest.mark.asyncio
async def test_screenshot_window_preserves_native_reasoning_and_call_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = [
        _with_native_output(
            _completion(f'resp-{i}', tool_calls=[_computer_call(f'call-{i}', [{'type': 'wait'}])]), f'call-{i}'
        )
        for i in range(5)
    ]
    client = _FakeResponsesClient(completions)
    screenshots = [base64.b64encode(f'image-{i}'.encode()).decode() for i in range(6)]
    runtime = _FakeRuntime(screenshots)
    monkeypatch.setattr(cua_agent, 'require_context', lambda: _context(client, runtime))
    result = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='Task',
            max_iterations=30,
            max_screenshots=5,
            max_recent_screenshots=3,
            prompt_style='macosworld_paper',
            platform_name='macos',
            home_dir='/Users/tester',
            sleep_after_execution=0,
            step_pause=0,
        )
    )
    assert result.exit_status == 'max_iterations'
    assert result.iterations == result.screenshots_sent == len(client.calls) == 5
    assert 'Screenshot budget exhausted' in result.final_answer
    assert len(runtime.codes) == 5
    for i, call in enumerate(client.calls):
        assert 'previous_response_id' not in call
        assert call['include'] == ['reasoning.encrypted_content']
        assert call['store'] is False
        assert '```DONE```' in call['instructions']
        assert '`tester`' in call['instructions']
        assert 'bundle multiple' not in call['instructions']
        images = mac._screenshot_parts(call['messages'])
        assert len(images) == i + 1
        assert sum(x['image_url'] != mac._BLACK_SCREENSHOT for x in images) == min(i + 1, 3)
        reasoning = [x for x in call['messages'] if x.get('type') == 'reasoning']
        assert [x['encrypted_content'] for x in reasoning] == [f'encrypted-call-{j}' for j in range(i)]
        outputs = [x for x in call['messages'] if x.get('type') == 'computer_call_output']
        assert [x['call_id'] for x in outputs] == [f'call-{j}' for j in range(i)]
    # Pruning the fifth request must not mutate earlier requests or original data.
    assert mac._screenshot_parts(client.calls[0]['messages'])[0]['image_url'].endswith(screenshots[0])


@pytest.mark.asyncio
async def test_observation_budget_does_not_send_partial_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeResponsesClient([
        _completion(
            'resp-1',
            tool_calls=[
                _computer_call('call-1', [{'type': 'wait'}]),
                _computer_call('call-2', [{'type': 'wait'}]),
            ],
        )
    ])
    runtime = _FakeRuntime([base64.b64encode(x).decode() for x in (b'a', b'b', b'c')])
    monkeypatch.setattr(cua_agent, 'require_context', lambda: _context(client, runtime))
    result = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='Task',
            max_iterations=30,
            max_screenshots=2,
            sleep_after_execution=0,
            step_pause=0,
        )
    )
    assert len(client.calls) == 1  # Two pending images would exceed the remaining budget of one.
    assert result.screenshots_sent == 1
    assert result.exit_status == 'max_iterations'
    assert len(runtime.codes) == 2


@pytest.mark.parametrize(('marker', 'expected'), [('```FAIL```', 'infeasible'), ('```DONE```', 'finished')])
@pytest.mark.asyncio
async def test_paper_completion_markers(monkeypatch: pytest.MonkeyPatch, marker: str, expected: str) -> None:
    client = _FakeResponsesClient([_completion('resp-1', content=marker)])
    runtime = _FakeRuntime()
    monkeypatch.setattr(cua_agent, 'require_context', lambda: _context(client, runtime))
    result = await mac.run_macosworld_cua_gpt(
        CuaGptAgentInput(
            user_prompt='Task',
            platform_name='macos',
            prompt_style='macosworld_paper',
        )
    )
    assert result.exit_status == expected
    assert runtime.codes == []


def test_windowed_history_rejects_lossy_reasoning() -> None:
    message = SimpleNamespace(
        provider_specific_fields={
            'responses_output_items': [
                {'type': 'reasoning', 'id': 'rs-only-server-id'},
            ]
        }
    )
    with pytest.raises(ValueError, match='encrypted reasoning'):
        mac._native_history_output(message)


@pytest.mark.asyncio
async def test_screenshot_window_rejects_server_side_history(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeResponsesClient([])
    client.stateful_previous_response = True
    runtime = _FakeRuntime()
    monkeypatch.setattr(cua_agent, 'require_context', lambda: _context(client, runtime))
    with pytest.raises(ValueError, match='stateful_previous_response=false'):
        await mac.run_macosworld_cua_gpt(CuaGptAgentInput(user_prompt='Task', max_recent_screenshots=3))
    assert client.calls == []
    assert runtime.codes == []
