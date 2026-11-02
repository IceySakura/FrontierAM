import base64
import io

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from PIL import Image

from nexus.extensions.agents.cua.macosworld_vlm import agent
from nexus.extensions.agents.cua.macosworld_vlm.upstream import parse_agent_output
from nexus.extensions.tasks.mac_os_world import permissions


def image_bytes():
    stream = io.BytesIO()
    Image.new('RGB', (1920, 1080)).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


def test_original_parser_ignores_code_and_preserves_upstream_whitelist():
    assert parse_agent_output('```\nimport os\nmove_to x=0.5 y=0.25\ntype_text hello world\ndone\n```') == [
        {'action': 'move_to', 'x': 0.5, 'y': 0.25},
        {'action': 'type_text', 'text': 'hello world'},
        {'action': 'done'},
    ]
    # Upstream advertises these but omits them from valid_actions. Do not silently expand it.
    assert parse_agent_output('drag_to 0.1 0.2\ntriple_click\nmouse_down left\nscroll_left 0.1') == []


def test_normalized_coordinates_use_actual_desktop_size():
    assert agent.action_code({'action': 'move_to', 'x': 0.5, 'y': 1}, 1920, 1080) == 'pyautogui.moveTo(960, 1079)'
    with pytest.raises(ValueError, match='Non-finite'):
        agent.action_code({'action': 'move_to', 'x': float('nan'), 'y': 0}, 1920, 1080)


async def test_budget_and_three_images_without_prior_action_messages(monkeypatch):
    shot = image_bytes()
    runtime = SimpleNamespace(
        gui_get_screen_size=AsyncMock(return_value=SimpleNamespace(success=True, width=1920, height=1080)),
        gui_screenshot=AsyncMock(return_value=SimpleNamespace(success=True, image=shot)),
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=0)),
    )
    client = SimpleNamespace(
        create_completion=AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='```\nleft_click\n```'))],
            )
        )
    )
    ctx = SimpleNamespace(get_runtime=AsyncMock(return_value=runtime), get_openai_client=lambda _: client)
    monkeypatch.setattr(agent, 'require_context', lambda: ctx)
    result = await agent.run_macosworld_vlm(
        agent.MacOSWorldVLMInput(
            user_prompt='Task',
            max_screenshots=4,
            max_iterations=30,
            action_interval=0,
        )
    )
    assert result.exit_status == 'max_iterations'
    assert result.iterations == result.screenshots_sent == 4
    assert runtime.run_command.await_count == 4
    assert runtime.gui_screenshot.await_count == 4
    for i, call in enumerate(client.create_completion.call_args_list):
        messages = call.args[0]
        assert [m['role'] for m in messages] == ['system', 'user']
        assert len(messages[1]['content']) == 1 + min(i + 1, 3)
        assert call.kwargs == {'temperature': 1.0, 'top_p': 0.9}


async def test_initialization_clicks_and_verifies_without_agent_actions(monkeypatch):
    monkeypatch.setattr(permissions.asyncio, 'sleep', AsyncMock())
    runtime = SimpleNamespace(gui_screenshot=AsyncMock(return_value=SimpleNamespace(success=True, image='image')))
    probe = AsyncMock(side_effect=[{'matched': 1, 'clicked': 1}, {'matched': 0, 'clicked': 0}])
    monkeypatch.setattr(permissions, '_capture_dialog', probe)
    result = await permissions.prepare_screen_capture_permission(runtime)
    assert result['success']
    assert result['clicked'] == 1
    assert [call.kwargs['click'] for call in probe.call_args_list] == [True, False]
    assert runtime.gui_screenshot.await_count == 2


async def test_unresolved_permission_does_not_reach_agent(monkeypatch):
    monkeypatch.setattr(permissions.asyncio, 'sleep', AsyncMock())
    runtime = SimpleNamespace(gui_screenshot=AsyncMock(return_value=SimpleNamespace(success=True, image='image')))
    monkeypatch.setattr(permissions, '_capture_dialog', AsyncMock(return_value={'matched': 1, 'clicked': 0}))
    with pytest.raises(RuntimeError, match='remains visible'):
        await permissions.prepare_screen_capture_permission(runtime)


async def test_permission_probe_failure_is_not_treated_as_absent():
    runtime = SimpleNamespace(run_command=AsyncMock(return_value=SimpleNamespace(return_code=1, stderr='not trusted')))
    with pytest.raises(RuntimeError, match='not trusted'):
        await permissions._capture_dialog(runtime, click=True)


@pytest.mark.parametrize(
    ('response_text', 'expected_status'), [('done\nleft_click', 'finished'), ('fail', 'infeasible')]
)
async def test_terminal_command_stops_actions_and_reuses_initial_image(monkeypatch, response_text, expected_status):
    shot = image_bytes()
    runtime = SimpleNamespace(
        gui_get_screen_size=AsyncMock(return_value=SimpleNamespace(success=True, width=1920, height=1080)),
        gui_screenshot=AsyncMock(),
        run_command=AsyncMock(),
    )
    client = SimpleNamespace(
        create_completion=AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))],
            )
        )
    )
    monkeypatch.setattr(
        agent,
        'require_context',
        lambda: SimpleNamespace(
            get_runtime=AsyncMock(return_value=runtime),
            get_openai_client=lambda _: client,
        ),
    )
    result = await agent.run_macosworld_vlm(
        agent.MacOSWorldVLMInput(
            user_prompt=[
                {'type': 'text', 'text': 'Task'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + shot}},
            ],
            action_interval=0,
        )
    )
    assert result.exit_status == expected_status
    assert result.iterations == 1
    runtime.gui_screenshot.assert_not_awaited()
    runtime.run_command.assert_not_awaited()


async def test_capture_failure_is_infrastructure_error_not_finished(monkeypatch):
    runtime = SimpleNamespace(
        gui_get_screen_size=AsyncMock(return_value=SimpleNamespace(success=True, width=1920, height=1080)),
        gui_screenshot=AsyncMock(return_value=SimpleNamespace(success=False, image=None, error='HTTP 503')),
    )
    client = SimpleNamespace(create_completion=AsyncMock())
    monkeypatch.setattr(
        agent,
        'require_context',
        lambda: SimpleNamespace(
            get_runtime=AsyncMock(return_value=runtime),
            get_openai_client=lambda _: client,
        ),
    )
    result = await agent.run_macosworld_vlm(agent.MacOSWorldVLMInput(user_prompt='Task'))
    assert result.exit_status == 'runtime_error'
    assert 'HTTP 503' in result.error
    client.create_completion.assert_not_awaited()
