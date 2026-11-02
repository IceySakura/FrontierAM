"""Unit coverage for optional PyAutoGUI backend selection."""

from __future__ import annotations

import platform
import shlex
import sys

from types import SimpleNamespace

import pytest


def test_gui_module_selects_concrete_backend_classes(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'xdotool')
    assert isinstance(gui._automation_backend(), gui.XDoToolBackend)

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'pyautogui')
    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(FAILSAFE=True))
    assert isinstance(gui._automation_backend(), gui.PyAutoGUIBackend)


def test_windows_gui_always_uses_pyautogui(monkeypatch):
    from nexus.runtime.server import gui

    fake_pyautogui = SimpleNamespace(FAILSAFE=True)
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'xdotool')
    monkeypatch.setitem(sys.modules, 'pyautogui', fake_pyautogui)

    backend = gui._automation_backend()

    assert isinstance(backend, gui.PyAutoGUIBackend)
    assert backend.pyautogui is fake_pyautogui


class FakeImage:
    """Tiny stand-in for a Pillow image returned by PyAutoGUI screenshots."""

    def __init__(self, data: bytes):
        self.data = data

    def save(self, fp, format=None):  # noqa: A002,N803 - mirrors Pillow's public API.
        fp.write(self.data)


def test_auto_backend_prefers_pyautogui_when_importable(monkeypatch):
    from nexus.runtime.server import gui

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakePyAutoGUI:
        FAILSAFE = True

        @staticmethod
        def click(**kwargs):
            calls.append(('click', (), kwargs))

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'auto')
    monkeypatch.setitem(sys.modules, 'pyautogui', FakePyAutoGUI)
    monkeypatch.setattr(gui, '_run_gui_command', lambda *args, **kwargs: calls.append(('xdotool', args, kwargs)))

    gui._click_sync(123, 456)

    assert FakePyAutoGUI.FAILSAFE is False
    assert calls == [('click', (), {'button': 'left', 'clicks': 1, 'x': 123, 'y': 456})]


def test_auto_backend_falls_back_to_xdotool_when_pyautogui_is_unavailable(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls: list[list[str]] = []

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'auto')
    monkeypatch.setitem(sys.modules, 'pyautogui', None)
    monkeypatch.setattr(gui, '_run_gui_command', lambda command, **kwargs: calls.append(command))

    gui._click_sync(123, 456)

    assert calls == [['xdotool', 'mousemove', '--sync', '123', '456', 'click', '1']]


def test_pyautogui_loader_stubs_mouseinfo_when_tkinter_is_missing(monkeypatch):
    from nexus.runtime.server import gui

    fake_pyautogui = SimpleNamespace(FAILSAFE=True)
    imports: list[str] = []

    def fake_import_module(name: str):
        imports.append(name)
        if name == 'pyautogui' and len(imports) == 1:
            raise SystemExit('NOTE: You must install tkinter on Linux to use MouseInfo.')
        return fake_pyautogui

    monkeypatch.delitem(sys.modules, 'mouseinfo', raising=False)
    monkeypatch.setattr(gui.importlib, 'import_module', fake_import_module)

    module = gui._load_pyautogui()

    assert module is fake_pyautogui
    assert fake_pyautogui.FAILSAFE is False
    assert imports == ['pyautogui', 'pyautogui']
    assert hasattr(sys.modules['mouseinfo'], 'MouseInfoWindow')


def test_explicit_xdotool_backend_uses_xdotool_even_when_pyautogui_is_available(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls: list[list[str]] = []

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'xdotool')
    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(click=lambda x, y: None))
    monkeypatch.setattr(gui, '_run_gui_command', lambda command, **kwargs: calls.append(command))

    gui._click_sync(123, 456)

    assert calls == [['xdotool', 'mousemove', '--sync', '123', '456', 'click', '1']]


def test_pyautogui_screenshot_returns_png_bytes_and_size(monkeypatch):
    from nexus.runtime.server import gui

    class FakePyAutoGUI:
        FAILSAFE = True

        @staticmethod
        def screenshot():
            return FakeImage(b'pyautogui-png')

        @staticmethod
        def size():
            return 1440, 900

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'pyautogui')
    monkeypatch.setitem(sys.modules, 'pyautogui', FakePyAutoGUI)

    image, width, height = gui._screenshot_sync()

    assert image == b'pyautogui-png'
    assert (width, height) == (1440, 900)


def test_windows_start_stack_attaches_to_interactive_desktop(monkeypatch):
    from nexus.runtime.server import gui

    backend = SimpleNamespace(screen_size=lambda: (1920, 1080), screenshot=lambda: (b'png', 1920, 1080))
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    monkeypatch.setattr(gui, '_enable_windows_dpi_awareness', lambda: None)
    monkeypatch.setattr(gui, '_automation_backend', lambda: backend)

    result = gui._start_stack_sync()

    assert result['success'] is True
    assert result['display'] == 'windows-interactive-desktop'
    assert result['desktop_session'] == 'windows'
    assert result['vnc_port'] is None
    assert result['novnc_port'] is None


@pytest.mark.parametrize('windows', [False, True])
def test_ascii_typing_uses_pyautogui(monkeypatch, windows):
    from nexus.runtime.server import gui

    calls = []
    fake_pyautogui = SimpleNamespace(write=lambda text, **kwargs: calls.append((text, kwargs)))
    monkeypatch.setattr(gui, 'is_windows', lambda: windows)

    gui.PyAutoGUIBackend(fake_pyautogui).type_text('ASCII input')

    assert calls == [('ASCII input', {'interval': 0.001})]


def test_linux_unicode_typing_uses_xdotool(monkeypatch):
    from nexus.runtime.server import gui

    calls = []
    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    monkeypatch.setattr(gui, 'resolve_first_existing_path', lambda *args, **kwargs: '/nix/runtime/nix-env/bin/xdotool')
    monkeypatch.setattr(gui, '_run_gui_command', lambda command, **kwargs: calls.append(command))

    gui.PyAutoGUIBackend(SimpleNamespace()).type_text('GUI 中文输入 OK')

    assert calls == [
        ['/nix/runtime/nix-env/bin/xdotool', 'type', '--delay', '50', '--clearmodifiers', '--', 'GUI 中文输入 OK']
    ]


def test_linux_unicode_typing_propagates_failure(monkeypatch):
    from nexus.runtime.server import gui

    def fail(*args, **kwargs):
        raise RuntimeError('X11 unavailable')

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    monkeypatch.setattr(gui, '_run_gui_command', fail)

    with pytest.raises(RuntimeError, match='X11 unavailable'):
        gui.PyAutoGUIBackend(SimpleNamespace()).type_text('中文')


def test_windows_unicode_typing_uses_clipboard_and_restores_it(monkeypatch):
    from nexus.runtime.server import gui

    clipboard = {'value': 'original'}
    calls = []
    fake_pyautogui = SimpleNamespace(hotkey=lambda *keys: calls.append(keys))
    fake_pyperclip = SimpleNamespace(
        paste=lambda: clipboard['value'],
        copy=lambda value: clipboard.update(value=value),
    )
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    monkeypatch.setitem(sys.modules, 'pyperclip', fake_pyperclip)

    gui.PyAutoGUIBackend(fake_pyautogui).type_text('你好')

    assert calls == [('ctrl', 'v')]
    assert clipboard['value'] == 'original'


def test_device_info_reports_selected_pyautogui(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setenv(gui.AUTOMATION_BACKEND_ENV, 'pyautogui')
    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(FAILSAFE=True))
    monkeypatch.setattr(gui, '_screen_size_sync', lambda: (1280, 800))
    monkeypatch.setattr(gui, '_active_window_name', lambda: 'Terminal')
    monkeypatch.setattr(gui, '_active_window_pid', lambda: '1234')

    assert gui._selected_automation_name() == 'pyautogui'


def test_run_pyautogui_as_user_executes_with_home_user_environment(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls = []

    def fake_run(command, *, check, capture_output, text, env, timeout):
        calls.append(
            {
                'command': command,
                'check': check,
                'capture_output': capture_output,
                'text': text,
                'env': env,
                'timeout': timeout,
            }
        )
        return SimpleNamespace(returncode=0, stdout='ok\n', stderr='')

    monkeypatch.setattr(gui.subprocess, 'run', fake_run)

    result = gui.PyAutoGUIBackend().run_code("pyautogui.press('enter')", user='user', home_dir='/home/user')

    assert result == {'success': True, 'stdout': 'ok\n', 'stderr': '', 'error': None}
    assert len(calls) == 1
    command = calls[0]['command']
    assert command[:2] == [gui.GUI_BASH, '-lc']
    assert 'runuser -u user' in command[2]
    assert 'useradd --uid 1000 --create-home --shell /bin/bash user' in command[2]
    assert 'HOME=/home/user' in command[2]
    assert 'USER=user' in command[2]
    assert 'LOGNAME=user' in command[2]
    assert 'base64.b64decode' in command[2]
    assert calls[0]['check'] is False


def test_pyautogui_runner_preloads_platform_for_guarded_desktop_code(monkeypatch):
    from nexus.runtime.server import gui

    calls = []

    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(FAILSAFE=True, press=lambda key: calls.append(key)))

    command = gui.PyAutoGUIBackend()._runner_command(
        "if platform.system() == 'Linux':\n    pyautogui.press('enter')\nelse:\n    time.sleep(0.1)"
    )
    parts = shlex.split(command)
    runner = parts[parts.index('-c') + 1]

    exec(runner, {})

    assert calls == (['enter'] if platform.system() == 'Linux' else [])


def test_run_pyautogui_without_user_uses_same_subprocess_runner(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls = []

    def fake_run(command, *, check, capture_output, text, env, timeout):
        calls.append(
            {
                'command': command,
                'check': check,
                'capture_output': capture_output,
                'text': text,
                'env': env,
                'timeout': timeout,
            }
        )
        return SimpleNamespace(returncode=0, stdout='local-ok\n', stderr='')

    monkeypatch.setattr(gui.subprocess, 'run', fake_run)

    result = gui.PyAutoGUIBackend().run_code('time.sleep(0.1)', user='')

    assert result == {'success': True, 'stdout': 'local-ok\n', 'stderr': '', 'error': None}
    assert len(calls) == 1
    command = calls[0]['command']
    assert command[:2] == [gui.GUI_BASH, '-lc']
    assert 'runuser -u' not in command[2]
    assert 'base64.b64decode' in command[2]
    assert calls[0]['check'] is False


def test_run_pyautogui_uses_current_python_on_windows(monkeypatch):
    from nexus.runtime.server import gui

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='windows-ok\n', stderr='')

    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    monkeypatch.setattr(gui.subprocess, 'run', fake_run)

    result = gui.PyAutoGUIBackend().run_code("pyautogui.press('enter')")

    assert result == {'success': True, 'stdout': 'windows-ok\n', 'stderr': '', 'error': None}
    assert calls[0][0][0:2] == [sys.executable, '-c']
    assert 'base64.b64decode' in calls[0][0][2]
    assert calls[0][1]['creationflags'] == gui.subprocess_creation_flags()


def test_start_stack_defaults_to_user_session(monkeypatch):
    from pathlib import Path

    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls = []

    def fake_run(command, *, check, capture_output, text, env, timeout):
        calls.append({'command': command, 'env': env})
        return SimpleNamespace(returncode=0, stdout='started', stderr='')

    monkeypatch.setattr(gui, '_gui_stack_script', lambda: Path('/tmp/nexus-gui-stack.sh'))
    monkeypatch.setattr(gui.subprocess, 'run', fake_run)

    result = gui._start_stack_sync(desktop_session='gnome-session')

    assert result['success'] is True
    assert calls[0]['env']['NEXUS_GUI_SESSION_USER'] == 'user'
    assert calls[0]['env']['HOME'] == '/home/user'


def test_launch_app_defaults_to_user_session(monkeypatch):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    calls = []

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, *, stdout, stderr, stdin, env, preexec_fn):
        calls.append({'command': command, 'env': env})
        return FakeProcess()

    gui._APP_PROCESSES.clear()
    monkeypatch.setattr(gui.subprocess, 'Popen', fake_popen)

    gui._launch_app_sync('xterm')

    assert len(calls) == 1
    assert calls[0]['command'][:2] == [gui.GUI_BASH, '-lc']
    assert 'runuser -u user' in calls[0]['command'][2]
    assert 'useradd --uid 1000 --create-home --shell /bin/bash user' in calls[0]['command'][2]
    assert 'HOME=/home/user' in calls[0]['command'][2]


def test_launch_app_reports_immediate_process_failure(monkeypatch, tmp_path):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)

    class FakeProcess:
        pid = 4321

        def poll(self):
            return 127

    def fake_popen(command, *, stdout, stderr, stdin, env, preexec_fn):
        stdout.write(b'missing-app: command not found\n')
        stdout.flush()
        return FakeProcess()

    gui._APP_PROCESSES.clear()
    monkeypatch.setattr(gui, '_GUI_LAUNCH_LOG_DIR', tmp_path)
    monkeypatch.setattr(gui, '_GUI_LAUNCH_FAILURE_GRACE_SECONDS', 0)
    monkeypatch.setattr(gui.subprocess, 'Popen', fake_popen)

    result = gui._launch_app_sync('missing-app')

    assert result['success'] is False
    assert result['return_code'] == 127
    assert 'missing-app: command not found' in result['error']
    assert gui._APP_PROCESSES == {}


def test_launch_app_allows_immediate_success_with_output(monkeypatch, tmp_path):
    from nexus.runtime.server import gui

    monkeypatch.setattr(gui, 'is_windows', lambda: False)

    class FakeProcess:
        pid = 4321

        def poll(self):
            return 0

    def fake_popen(command, *, stdout, stderr, stdin, env, preexec_fn):
        stdout.write(b'Connecting to org.freedesktop.Tracker3.Miner.Files\n')
        stdout.flush()
        return FakeProcess()

    gui._APP_PROCESSES.clear()
    monkeypatch.setattr(gui, '_GUI_LAUNCH_LOG_DIR', tmp_path)
    monkeypatch.setattr(gui, '_GUI_LAUNCH_FAILURE_GRACE_SECONDS', 0)
    monkeypatch.setattr(gui.subprocess, 'Popen', fake_popen)

    result = gui._launch_app_sync('nautilus --new-window /home/user')

    assert result['success'] is True
    assert result['return_code'] == 0
    assert result['error'] is None
    assert 'Connecting to org.freedesktop.Tracker3.Miner.Files' in result['output']
    assert gui._APP_PROCESSES == {}


def test_windows_launch_app_tracks_process(monkeypatch, tmp_path):
    from nexus.runtime.server import gui

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    gui._APP_PROCESSES.clear()
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    monkeypatch.setattr(gui, '_GUI_LAUNCH_LOG_DIR', tmp_path)
    monkeypatch.setattr(gui, '_GUI_LAUNCH_FAILURE_GRACE_SECONDS', 0)
    monkeypatch.setattr(gui.subprocess, 'Popen', fake_popen)

    result = gui._launch_app_sync('notepad.exe')

    assert result['success'] is True
    assert calls[0][0] == 'notepad.exe'
    assert calls[0][1]['creationflags'] == gui.subprocess_creation_flags()
    assert 'notepad.exe' in gui._APP_PROCESSES
    gui._APP_PROCESSES.clear()
