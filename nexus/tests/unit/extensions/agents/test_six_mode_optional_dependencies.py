"""Optional action observation must not be a startup dependency."""

import ast
import base64
import io

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from PIL import Image

import nexus


ROOT = Path(nexus.__file__).resolve().parent


def test_no_action_trace_imports_in_six_mode_entry_modules():
    removed = {
        'ComputerActionObservation',
        'ComputerActionTraceConfig',
        'computer_action_trace_context',
        'computer_action_trace_enabled',
        'get_computer_action_trace_config',
        'ingest_computer_action_event',
        'observe_computer_action',
    }
    for name in (
        'trace/__init__.py',
        'extensions/agents/cua/cua_anthropic_agent/agent.py',
        'extensions/agents/claude_code/native_bridge/agent_runtime.py',
    ):
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != 'nexus.trace.computer_action'
                if node.module == 'nexus.trace':
                    assert not removed & {alias.name for alias in node.names}


@pytest.mark.asyncio
@pytest.mark.parametrize('execution_mode', ['run_command', 'gui_pyautogui'])
async def test_claude_cua_keeps_action_execution_and_screenshots(monkeypatch, execution_mode):
    from nexus.extensions.agents.cua.cua_anthropic_agent import agent

    png = io.BytesIO()
    Image.new('RGB', (4, 4)).save(png, format='PNG')
    events = []

    async def screenshot():
        events.append('screenshot')
        return {'success': True, 'image': base64.b64encode(png.getvalue()).decode()}

    async def execute(*args, **kwargs):
        code = kwargs.get('command') if execution_mode == 'run_command' else args[0]
        assert 'pyautogui' in code
        assert kwargs['timeout'] == 17
        events.append('action')
        return {'status': 'Finished', 'return_code': 0, 'success': True}

    runtime = SimpleNamespace(
        gui_screenshot=AsyncMock(side_effect=screenshot),
        run_command=AsyncMock(side_effect=execute),
        gui_pyautogui=AsyncMock(side_effect=execute),
    )
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason='tool_calls',
                message=SimpleNamespace(
                    content='Click the target.',
                    thinking_blocks=[],
                    tool_calls=[
                        SimpleNamespace(
                            id='click-1',
                            function=SimpleNamespace(
                                name='computer', arguments='{"action":"left_click","coordinate":[1,1]}'
                            ),
                        )
                    ],
                ),
            )
        ]
    )
    client = SimpleNamespace(
        api_style='messages',
        get_effective_request_args=lambda: {'model': 'claude-sonnet-4-5'},
        create_completion=AsyncMock(return_value=completion),
    )
    ctx = SimpleNamespace(
        get_runtime=AsyncMock(return_value=runtime), get_openai_client=lambda _: client, tool_state={}
    )
    monkeypatch.setattr(agent, 'require_context', lambda: ctx)
    result = await agent.run_cua_anthropic_agent(
        agent.CuaAnthropicAgentInput(
            user_prompt='Click the target.',
            transport='client',
            max_iterations=1,
            no_thinking=True,
            pyautogui_execution_mode=execution_mode,
            pyautogui_timeout=17,
            screen_width=4,
            screen_height=4,
            model_display_width=4,
            model_display_height=4,
            sleep_after_execution=0,
            step_pause=0,
        )
    )
    assert result.exit_status == 'max_iterations'
    assert events == ['screenshot', 'action', 'screenshot']
    client.create_completion.assert_awaited_once()
    selected = runtime.run_command if execution_mode == 'run_command' else runtime.gui_pyautogui
    unused = runtime.gui_pyautogui if execution_mode == 'run_command' else runtime.run_command
    selected.assert_awaited_once()
    unused.assert_not_awaited()


def test_native_cc_does_not_relay_optional_action_events():
    tree = ast.parse((ROOT / 'extensions/agents/claude_code/native_bridge/agent_runtime.py').read_text())
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'setup_and_run_native_bridge_claude_code'
    )
    assert 'action_event_file' not in {arg.arg for arg in entry.args.kwonlyargs}
