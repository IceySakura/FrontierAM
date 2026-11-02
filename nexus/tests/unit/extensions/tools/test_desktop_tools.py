from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_desktop_computer_13_tool_calls_runtime(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopComputer13Input, computer_13_tool

    calls: list[tuple[str, dict[str, object]]] = []

    class FakeRuntime:
        async def gui_computer_13(self, action_type, parameters=None):
            calls.append((action_type, parameters or {}))
            return SimpleNamespace(success=True, error=None)

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await computer_13_tool(
        DesktopComputer13Input(action_type='CLICK', parameters={'x': 10, 'y': 20}),
        context=None,
    )

    assert calls == [('CLICK', {'x': 10, 'y': 20})]
    assert response.meta == {'success': True, 'error': None}
    assert response.context_updates == {'desktop_last_tool': 'desktop/computer_13'}


@pytest.mark.asyncio
async def test_desktop_screenshot_tool_marks_observation_context(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import ScreenshotInput, screenshot_tool

    class FakeRuntime:
        async def gui_screenshot(self):
            return SimpleNamespace(success=True, image='abc', format='png', width=1280, height=800, error=None)

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await screenshot_tool(ScreenshotInput(), context=None)

    assert response.meta['success'] is True
    assert response.context_updates == {'desktop_last_tool': 'desktop/screenshot'}


@pytest.mark.asyncio
async def test_desktop_computer_13_tool_accepts_top_level_action_parameters(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopComputer13Input, computer_13_tool

    calls: list[tuple[str, dict[str, object]]] = []

    class FakeRuntime:
        async def gui_computer_13(self, action_type, parameters=None):
            calls.append((action_type, parameters or {}))
            return SimpleNamespace(success=True, done=False, fail=False, error=None)

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await computer_13_tool(
        DesktopComputer13Input.model_validate({'action_type': 'HOTKEY', 'keys': ['CTRL', 'L']}),
        context=None,
    )

    assert calls == [('HOTKEY', {'keys': ['CTRL', 'L']})]
    assert response.meta['success'] is True


@pytest.mark.asyncio
async def test_desktop_pyautogui_tool_returns_stdout(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopPyAutoGUIInput, pyautogui_tool

    class FakeRuntime:
        def __init__(self):
            self.user = None
            self.home_dir = None

        async def gui_pyautogui(self, code, *, timeout=30.0, user=None, home_dir=None):
            self.user = user
            self.home_dir = home_dir
            return SimpleNamespace(success=True, stdout='ok\n', stderr='', error=None)

    runtime = FakeRuntime()

    async def fake_runtime():
        return runtime

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await pyautogui_tool(DesktopPyAutoGUIInput(code='pyautogui.click(1, 2)'), context=None)

    assert response.meta['success'] is True
    assert response.meta['stdout'] == 'ok\n'
    assert response.output == [{'type': 'text', 'text': 'ok\n'}]
    assert runtime.user == 'user'
    assert runtime.home_dir == '/home/user'


@pytest.mark.asyncio
async def test_desktop_pyautogui_tool_uses_tool_configured_user_environment(monkeypatch):
    from nexus.core import ToolContext
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopPyAutoGUIInput, pyautogui_tool

    class FakeRuntime:
        def __init__(self):
            self.code = ''
            self.timeout = None
            self.user = None
            self.home_dir = None

        async def gui_pyautogui(self, code, *, timeout=30.0, user=None, home_dir=None):
            self.code = code
            self.timeout = timeout
            self.user = user
            self.home_dir = home_dir
            return SimpleNamespace(success=True, stdout='ok\n', stderr='', error=None)

    runtime = FakeRuntime()

    async def fake_runtime():
        return runtime

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await pyautogui_tool(
        DesktopPyAutoGUIInput(code='time.sleep(0.1)', timeout=7),
        context=ToolContext(tool_configs={'desktop/pyautogui': {'user': 'user', 'home_dir': '/home/user'}}),
    )

    assert response.meta['success'] is True
    assert runtime.timeout == 7
    assert runtime.code == 'time.sleep(0.1)'
    assert runtime.user == 'user'
    assert runtime.home_dir == '/home/user'


def test_desktop_pyautogui_tool_description_does_not_suggest_position_probe():
    from nexus.extensions.tools.desktop.tools import PYAUTOGUI_CODE_DESCRIPTION

    assert 'pyautogui.position()' not in PYAUTOGUI_CODE_DESCRIPTION


@pytest.mark.asyncio
async def test_desktop_pyautogui_tool_rejects_code_outside_official_action_subset(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopPyAutoGUIInput, pyautogui_tool

    class FakeRuntime:
        code = ''
        called = False

        async def gui_pyautogui(self, code, *, timeout=30.0, user=None, home_dir=None):
            self.called = True
            self.code = code
            return SimpleNamespace(success=True, stdout='passed\n', stderr='', error=None)

    runtime = FakeRuntime()

    async def fake_runtime():
        return runtime

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    code = 'import os\nprint(pyautogui.position())\nopen("/tmp/osworld.txt", "w").write("ok")'
    response = await pyautogui_tool(DesktopPyAutoGUIInput(code=code), context=None)

    assert runtime.called is False
    assert response.meta['success'] is False
    assert 'not allowed' in response.meta['error']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'code',
    [
        "pyautogui.hotkey('ctrl', 'alt', 't')",
        "pyautogui.hotkey('alt', 'f2')",
        "pyautogui.write('http://localhost:3000/events/new')",
    ],
)
async def test_desktop_pyautogui_tool_does_not_apply_cua_gym_policy_blocks(monkeypatch, code):
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopPyAutoGUIInput, pyautogui_tool

    class FakeRuntime:
        called = False
        code = ''

        async def gui_pyautogui(self, code, *, timeout=30.0, user=None, home_dir=None):
            self.called = True
            self.code = code
            return SimpleNamespace(success=True, stdout='passed\n', stderr='', error=None)

    runtime = FakeRuntime()

    async def fake_runtime():
        return runtime

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    ctx = ExecutionContext(task_name='cua_gym', runtime_configs={}, tool_configs={})
    async with ctx:
        response = await pyautogui_tool(DesktopPyAutoGUIInput(code=code), context=None)

    assert runtime.called is True
    assert runtime.code == code
    assert response.meta['success'] is True


@pytest.mark.asyncio
async def test_desktop_launch_app_tool_uses_default_user_session(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import LaunchAppInput, launch_app_tool

    calls: list[dict[str, object]] = []

    class FakeRuntime:
        async def gui_launch_app(self, app, *, user=None, home_dir=None):
            calls.append({'app': app, 'user': user, 'home_dir': home_dir})
            return SimpleNamespace(success=True, error=None)

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await launch_app_tool(LaunchAppInput(app='xterm'), context=None)

    assert response.meta['success'] is True
    assert calls == [{'app': 'xterm', 'user': 'user', 'home_dir': '/home/user'}]


@pytest.mark.asyncio
async def test_desktop_bash_tool_uses_default_user_session(monkeypatch):
    from nexus.extensions.tools.desktop import tools
    from nexus.extensions.tools.desktop.tools import DesktopBashInput, bash_tool

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=180, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='hello\n', stderr='')

    async def fake_runtime():
        return FakeRuntime()

    monkeypatch.setattr(tools, '_runtime', fake_runtime)

    response = await bash_tool(DesktopBashInput(command='printf hello'), context=None)

    assert len(commands) == 1
    assert 'runuser -u user' in commands[0]
    assert 'HOME=/home/user' in commands[0]
    assert 'cd /home/user && printf hello' in commands[0]
    assert response.meta['success'] is True
    assert response.output == [{'type': 'text', 'text': 'hello\n'}]


def test_desktop_manifest_exposes_desktop_tool_set():
    manifest = Path('nexus/extensions/tools/desktop/pyproject.toml').read_text(encoding='utf-8')

    expected_tools = {
        'desktop/start',
        'desktop/close_app',
        'desktop/launch_app',
        'desktop/screenshot',
        'desktop/screen_size',
        'desktop/computer_13',
        'desktop/pyautogui',
        'desktop/bash',
    }

    for tool_name in expected_tools:
        assert f'"{tool_name}" = "nexus.extensions.tools.desktop.tools"' in manifest

    assert '"desktop/click"' not in manifest
    assert '"desktop/type_text"' not in manifest
    assert '"desktop/press_key"' not in manifest


def test_desktop_computer_13_schema_describes_official_osworld_action_space():
    from nexus.core import get_tool_schema
    from nexus.extensions.tools.desktop.tools import computer_13_tool

    schema = get_tool_schema(computer_13_tool)
    description = schema['function']['description']
    parameters = schema['function']['parameters']
    action_description = parameters['properties']['action_type']['description']
    parameter_description = parameters['properties']['parameters']['description']

    for action_type in [
        'MOVE_TO',
        'CLICK',
        'MOUSE_DOWN',
        'MOUSE_UP',
        'RIGHT_CLICK',
        'DOUBLE_CLICK',
        'DRAG_TO',
        'SCROLL',
        'TYPING',
        'PRESS',
        'KEY_DOWN',
        'KEY_UP',
        'HOTKEY',
    ]:
        assert action_type in description
        assert action_type in action_description

    for marker in ['WAIT', 'DONE', 'FAIL']:
        assert marker in description
        assert marker in action_description
        assert marker not in parameters['properties']['action_type']['enum']

    for parameter_hint in [
        'x/y',
        'button',
        'num_clicks',
        'dx/dy',
        'text',
        'key',
        'keys',
    ]:
        assert parameter_hint in parameter_description

    marker_guidance = (
        'For OSWorld-style tasks where desktop/fixed_marker is available, WAIT, DONE, and FAIL are fixed actions '
        'represented by desktop/fixed_marker, not by desktop/computer_13.'
    )
    assert 'official OSWorld computer_13' in description
    assert 'Use exactly one action per call' in description
    assert 'duration' not in description
    assert '- DRAG_TO: {"x": number, "y": number}' in parameter_description
    assert 'duration' not in parameter_description
    assert marker_guidance in description
    assert marker_guidance in action_description
    assert marker_guidance in parameter_description


def test_desktop_pyautogui_schema_describes_official_osworld_code_action_space():
    from nexus.core import get_tool_schema
    from nexus.extensions.tools.desktop.tools import pyautogui_tool

    schema = get_tool_schema(pyautogui_tool)
    description = schema['function']['description']
    parameters = schema['function']['parameters']
    code_description = parameters['properties']['code']['description']
    timeout_description = parameters['properties']['timeout']['description']

    for expected in [
        'official OSWorld pyautogui',
        'grounded to the current observation',
        'Only direct pyautogui.* calls, time.sleep(...), and read-only platform helpers are available',
        'Do not call pyautogui.screenshot()',
        'Do not use pyautogui.locate',
        'Do not use hidden window-inspection helpers',
        'Do not import modules',
        'Do not read or write files',
        'Do not call shell commands, network APIs, or mock-service APIs',
        'Do not type javascript: or data:text/html URLs',
        'Return one line or multiple lines of python code',
        'no variables or function can be shared from history',
        'time.sleep(0.5)',
        'specify coordinates',
    ]:
        assert expected in description
        assert expected in code_description

    assert 'timeout' in timeout_description
    assert 'seconds' in timeout_description


def test_desktop_design_docs_describe_pyautogui_guard():
    from pathlib import Path

    docs = '\n'.join(
        [
            Path('docs/developing/container-gui-runtime-design.md').read_text(encoding='utf-8'),
            Path('docs/developing/cua-gym-task-design.md').read_text(encoding='utf-8'),
        ]
    )

    assert 'pyautogui guard' in docs
    assert 'pyautogui.*' in docs
    assert 'time.sleep' in docs
    assert 'platform' in docs
    assert 'pyautogui.screenshot' in docs
    assert 'pyautogui.locate' in docs
    assert 'window-inspection' in docs
    assert 'data:text/html' in docs
