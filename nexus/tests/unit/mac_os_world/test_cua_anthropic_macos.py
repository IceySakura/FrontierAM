import base64
import io
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from PIL import Image

from nexus.extensions.agents.cua.cua_anthropic_agent import agent
from nexus.extensions.agents.cua.macosworld_cua import anthropic as mac


@pytest.mark.parametrize('platform', ['macos', 'macOS', 'Darwin', 'OS X'])
def test_macos_prompt_uses_the_task_home_and_gui_instructions(platform):
    prompt = mac.build_system_prompt(platform, '2026-10-21', home_dir='/Users/test-user')
    assert 'macOS' in prompt
    assert '/Users/test-user' in prompt
    assert 'Command+Space' in prompt
    assert 'Ubuntu' not in prompt
    assert 'DISPLAY=' not in prompt
    assert '[DONE]' in prompt


@pytest.mark.parametrize('action', ['key', 'hold_key', 'left_click', 'scroll'])
@pytest.mark.parametrize(
    ('key', 'expected'), [('super_l', 'command'), ('cmd', 'command'), ('option', 'alt'), ('ctrl', 'ctrl')]
)
def test_macos_modifier_keys_work_across_actions_and_batches(action, key, expected):
    call = {'input': {'actions': [{'action': action, 'text': key, 'scroll_direction': 'down', 'scroll_amount': 1}]}}
    code = mac._parse_actions_from_tool_call(call, platform_name='macos')
    events = []
    pyautogui = SimpleNamespace(
        keyDown=lambda k: events.append(('down', k)),
        keyUp=lambda k: events.append(('up', k)),
        click=lambda: None,
        scroll=lambda n: None,
    )
    exec(code, {'pyautogui': pyautogui})
    assert events[0] == ('down', expected)
    if action != 'hold_key':
        assert events[-1] == ('up', expected)


def test_existing_linux_key_mapping_and_prompt_are_preserved():
    code = mac._parse_actions_from_tool_call({'input': {'action': 'key', 'text': 'Super_L+a'}})
    assert "keyDown('win')" in code
    assert 'Ubuntu' in mac.build_system_prompt('ubuntu')
    assert 'Windows' in mac.build_system_prompt('Windows')


def test_paper_model_selects_the_20250124_computer_tool():
    profile = mac.get_claude_runtime_profile('claude-3-7-sonnet-20250219')
    assert profile.computer_use_type == 'computer_20250124'
    assert profile.computer_use_beta_flag == 'computer-use-2025-01-24'
    assert profile.thinking_mode == 'legacy'


def _completion(tool_input=None, *, name='computer'):
    calls = (
        []
        if tool_input is None
        else [SimpleNamespace(id='tool-1', function=SimpleNamespace(name=name, arguments=json.dumps(tool_input)))]
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='[DONE]' if not calls else '', tool_calls=calls, thinking_blocks=[]),
                finish_reason='tool_calls' if calls else 'stop',
            )
        ]
    )


def _context(monkeypatch, *, tool_name='computer', screen_success=True):
    image = io.BytesIO()
    # Retina image size differs from the logical dimensions used for clicking.
    Image.new('RGB', (2048, 1536)).save(image, format='PNG')
    runtime = SimpleNamespace(
        gui_screenshot=AsyncMock(return_value={'success': True, 'image': base64.b64encode(image.getvalue()).decode()}),
        gui_get_screen_size=AsyncMock(return_value={'success': screen_success, 'width': 1024, 'height': 768}),
        run_command=AsyncMock(return_value={'status': 'Finished', 'return_code': 0}),
    )
    client = SimpleNamespace(
        api_style='messages',
        get_effective_request_args=lambda: {'model': 'claude-3-7-sonnet-20250219'},
        create_completion=AsyncMock(
            side_effect=[
                _completion(
                    {
                        'actions': [
                            {'action': 'left_click', 'coordinate': [640, 360]},
                            {'action': 'key', 'text': 'super_l+a'},
                        ]
                    },
                    name=tool_name,
                ),
                _completion(),
            ]
        ),
    )
    ctx = SimpleNamespace(
        get_runtime=AsyncMock(return_value=runtime), get_openai_client=lambda n: client, tool_state={}
    )
    monkeypatch.setattr(agent, 'require_context', lambda: ctx)
    return runtime, client, ctx


def _input():
    return mac.MacOSWorldAnthropicInput(
        user_prompt='Open Dictionary.',
        platform_name='macos',
        home_dir='/Users/test-user',
        transport='client',
        computer_only=True,
        auto_screen_size=True,
        max_iterations=3,
        no_thinking=True,
        sleep_after_execution=0,
        step_pause=0,
        pyautogui_execution_mode='run_command',
    )


@pytest.mark.asyncio
async def test_macos_agent_roundtrip_scales_to_logical_screen_and_uses_shared_client(monkeypatch):
    runtime, client, ctx = _context(monkeypatch)
    result = await mac.run_macosworld_cua_anthropic(_input())
    assert result.exit_status == 'finished'
    assert result.iterations == 2
    assert ctx.tool_state['os_world_last_action']['action_type'] == 'DONE'
    runtime.run_command.assert_awaited_once()
    command = runtime.run_command.await_args.kwargs['command']
    # Decode the executed snippet, rather than relying on the encoder's layout.
    import re

    encoded = re.search(r"code = base64.b64decode\('([^']+)'\)", command).group(1)
    code = base64.b64decode(encoded).decode()
    assert 'pyautogui.click(512, 384)' in code
    assert "pyautogui.keyDown('command')" in code
    request = client.create_completion.await_args_list[0].kwargs
    assert [t['name'] for t in request['tools']] == ['computer']
    assert request['tools'][0]['type'] == 'computer_20250124'
    assert 'macOS' in request['messages'][0]['content']
    assert '/Users/test-user' in request['messages'][0]['content']
    feedback = client.create_completion.await_args_list[1].kwargs['messages']
    assert any(m['role'] == 'tool' and m['tool_call_id'] == 'tool-1' for m in feedback)


@pytest.mark.asyncio
async def test_unknown_screen_size_stops_before_model_or_actions(monkeypatch):
    runtime, client, _ = _context(monkeypatch, screen_success=False)
    with pytest.raises(RuntimeError, match='desktop size'):
        await mac.run_macosworld_cua_anthropic(_input())
    client.create_completion.assert_not_awaited()
    runtime.run_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_computer_only_rejects_unadvertised_shell_call(monkeypatch):
    runtime, _, _ = _context(monkeypatch, tool_name='bash')
    with pytest.raises(RuntimeError, match='bash is disabled'):
        await mac.run_macosworld_cua_anthropic(_input())
    runtime.run_command.assert_not_awaited()


@pytest.mark.parametrize(('action', 'count'), [('double_click', 2), ('triple_click', 3)])
@pytest.mark.parametrize('coordinate', [None, [100, 200]])
def test_macos_multiclick_posts_incrementing_native_click_states(monkeypatch, action, count, coordinate):
    import sys

    events = []
    quartz = SimpleNamespace(
        kCGMouseEventClickState=1,
        kCGHIDEventTap=0,
        CGEventCreateMouseEvent=lambda source, kind, point, button: {'kind': kind, 'point': point, 'button': button},
        CGEventSetIntegerValueField=lambda event, field, value: event.update(count=value),
        CGEventPost=lambda tap, event: events.append(event),
    )
    monkeypatch.setitem(sys.modules, 'Quartz', quartz)
    positions = []
    pg = SimpleNamespace(moveTo=lambda x, y: positions.append((x, y)), position=lambda: (100, 200))
    code = mac._parse_actions_from_tool_call(
        {'input': {'action': action, 'coordinate': coordinate}}, platform_name='macos'
    )
    exec(code, {'pyautogui': pg, 'time': SimpleNamespace(sleep=lambda t: None)})
    assert [e['count'] for e in events] == [i for i in range(1, count + 1) for _ in range(2)]
    assert [e['kind'] for e in events] == [1, 2] * count
    assert positions == [tuple(coordinate) if coordinate else (None, None)]


def test_macos_drag_uses_explicit_left_button():
    drags = []
    code = mac._parse_actions_from_tool_call(
        {'input': {'action': 'left_click_drag', 'coordinate': [300, 400]}}, platform_name='macos'
    )
    exec(code, {'pyautogui': SimpleNamespace(dragTo=lambda *args, **kwargs: drags.append((args, kwargs)))})
    assert drags == [((300, 400), {'duration': 0.5, 'button': 'left'})]


def test_strict_window_includes_initial_image_and_preserves_tool_results():
    def image(i):
        return {'type': 'image', 'source': {'data': str(i)}}

    initial = {'role': 'user', 'content': [image(0), {'type': 'text', 'text': 'task'}]}
    results = [
        {'type': 'tool_result', 'tool_use_id': str(i), 'content': [{'type': 'text', 'text': 'Success'}, image(i)]}
        for i in range(1, 5)
    ]
    messages = [initial, {'role': 'user', 'content': results}]
    mac._trim_image_window(messages, 3)
    assert initial['content'] == [{'type': 'text', 'text': 'task'}]
    assert [r['tool_use_id'] for r in results] == ['1', '2', '3', '4']
    assert len(results[0]['content']) == 1
    assert [r['content'][-1]['source']['data'] for r in results[1:]] == ['2', '3', '4']


@pytest.mark.asyncio
async def test_screenshot_budget_stops_before_executing_an_unobservable_action(monkeypatch):
    runtime, client, _ = _context(monkeypatch)
    action = _completion({'action': 'left_click', 'coordinate': [100, 100]})
    client.create_completion.side_effect = [action, action]
    params = _input().model_copy(update={'max_screenshots': 2})
    result = await mac.run_macosworld_cua_anthropic(params)
    assert result.exit_status == 'max_iterations'
    assert result.screenshots == 2
    assert client.create_completion.await_count == 2
    assert runtime.run_command.await_count == 1


@pytest.mark.asyncio
async def test_action_failure_returns_remote_error_detail():
    runtime = SimpleNamespace(
        run_command=AsyncMock(return_value={'status': 'Finished', 'return_code': 1, 'stderr': 'ValueError: bad button'})
    )
    ok, error = await mac._execute_computer_code(runtime, 'pass', execution_mode='run_command', timeout=30)
    assert not ok
    assert 'ValueError: bad button' in error


@pytest.mark.asyncio
async def test_done_marker_does_not_discard_pending_computer_actions(monkeypatch):
    runtime, client, _ = _context(monkeypatch)
    action = _completion({'action': 'left_click', 'coordinate': [100, 100]})
    action.choices[0].message.content = 'I will click then say [DONE].'
    client.create_completion.side_effect = [action, _completion()]
    result = await mac.run_macosworld_cua_anthropic(_input())
    assert result.iterations == 2
    runtime.run_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_messages_toolset_uses_native_member_names_and_preserves_pairing(monkeypatch):
    runtime, client, _ = _context(monkeypatch)
    client.create_completion.side_effect = [_completion({'coordinate': [640, 360]}, name='left_click'), _completion()]
    params = _input().model_copy(update={'computer_tool_schema': 'computer_toolset_20260801'})
    result = await mac.run_macosworld_cua_anthropic(params)
    assert result.exit_status == 'finished'
    runtime.run_command.assert_awaited_once()
    request = client.create_completion.await_args_list[0].kwargs
    assert request['tools'] == [{'type': 'computer_toolset_20260801'}]
    assert '1280x720' in request['messages'][0]['content']
    history = client.create_completion.await_args_list[1].kwargs['messages']
    assert any(m.get('tool_call_id') == 'tool-1' for m in history)
    assistant = next(m for m in history if m['role'] == 'assistant')
    assert assistant['tool_calls'][0]['function']['name'] == 'left_click'


@pytest.mark.parametrize('tool_type', ['computer_20250124', 'computer_toolset_20260801'])
def test_macosworld_messages_transport_preserves_computer_tool_schema(tool_type):
    from nexus.core import OpenAIConfig
    from nexus.extensions.tasks.mac_os_world.clients import MacOSWorldAnthropicClient

    client = MacOSWorldAnthropicClient(
        OpenAIConfig(
            client_args={'api_key': 'test', 'base_url': 'https://fixture.invalid'},
            request_args={'model': 'claude-test'},
        )
    )
    tools = [{'type': tool_type}]
    payload = client._build_request_payload(
        messages=[{'role': 'user', 'content': 'task'}], request_args={'tools': tools}
    )
    assert payload['tools'] == tools


def test_string_encoded_coordinates_are_decoded_before_scaling():
    code = mac._parse_actions_from_tool_call(
        {
            'input': {
                'action': 'left_click_drag',
                'coordinate': '[36, 7]',
                'start_coordinate': '[12, 3]',
                'duration': '1',
            }
        },
        resize_factor=(2.0, 2.0),
        platform_name='macos',
    )
    events = []
    pg = SimpleNamespace(moveTo=lambda *a, **k: events.append(a), dragTo=lambda *a, **k: events.append((a, k)))
    exec(code, {'pyautogui': pg})
    assert events == [(24, 6), ((72, 14), {'duration': 1.0, 'button': 'left'})]


@pytest.mark.parametrize('coordinate', ['not JSON', '[1, 2, 3]', '["1", "2"]', '[NaN, 2]'])
def test_malformed_coordinates_cannot_become_executable_code(coordinate):
    with pytest.raises(ValueError, match='coordinate|Expecting value'):
        mac._parse_actions_from_tool_call({'input': {'action': 'left_click', 'coordinate': coordinate}})


@pytest.mark.asyncio
async def test_parse_failure_is_reported_even_when_last_tool_succeeds(monkeypatch, tmp_path):
    runtime, client, _ = _context(monkeypatch)
    action = _completion({'action': 'left_click', 'coordinate': '[1, 2, 3]'})
    screenshot = _completion({'action': 'screenshot'}).choices[0].message.tool_calls[0]
    screenshot.id = 'tool-2'
    action.choices[0].message.tool_calls.append(screenshot)
    client.create_completion.side_effect = [action, _completion()]
    params = _input().model_copy(update={'trajectory_dir': str(tmp_path)})
    await mac.run_macosworld_cua_anthropic(params)
    runtime.run_command.assert_awaited_once()
    history = client.create_completion.await_args_list[1].kwargs['messages']
    feedback = next(m for m in history if m.get('tool_call_id') == 'tool-1')
    assert 'Invalid computer action' in str(feedback['content'])
    record = json.loads((tmp_path / 'trajectory.jsonl').read_text().splitlines()[0])
    assert [r['success'] for r in record['action_results']] == [False, True]
    assert record['action_result'] != 'Success'


@pytest.mark.parametrize('action', ['zoom', 'cursor_position'])
def test_unsupported_observation_actions_are_not_reported_as_success(action):
    with pytest.raises(ValueError, match='not supported'):
        mac._parse_actions_from_tool_call({'input': {'action': action}})
