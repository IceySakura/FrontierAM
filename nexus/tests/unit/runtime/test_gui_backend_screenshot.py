"""Unit coverage for GUI backend screenshot command selection."""


def test_start_stack_runs_gui_stack_script_with_requested_session(monkeypatch, tmp_path):
    """The explicit GUI start path should delegate to the runtime gui_stack script."""
    from nexus.runtime.server import gui

    script = tmp_path / 'gui_stack.sh'
    script.write_text('#!/usr/bin/env bash\n', encoding='utf-8')
    captured: dict[str, object] = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = 'started'
        stderr = ''

    def fake_run(command, *, check, capture_output, text, env, timeout):
        captured['command'] = command
        captured['env'] = env
        captured['timeout'] = timeout
        return FakeCompletedProcess()

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    monkeypatch.setattr(gui, '_gui_stack_script', lambda: script)
    monkeypatch.setattr(gui.subprocess, 'run', fake_run)

    result = gui._start_stack_sync(
        desktop_session='gnome-session',
        window_manager='none',
        screen_width=1440,
        screen_height=900,
        gnome_session_name='ubuntu',
    )

    env = captured['env']
    assert captured['command'] == [gui.GUI_BASH, str(script), 'start']
    assert captured['timeout'] == 120
    assert env['NEXUS_GUI_DESKTOP_SESSION'] == 'gnome-session'
    assert env['NEXUS_GUI_GNOME_SESSION_NAME'] == 'ubuntu'
    assert env['NEXUS_GUI_WINDOW_MANAGER'] == 'none'
    assert env['NEXUS_GUI_SCREEN_WIDTH'] == '1440'
    assert env['NEXUS_GUI_SCREEN_HEIGHT'] == '900'
    assert result['success'] is True
    assert result['desktop_session'] == 'gnome-session'
    assert result['error'] is None


def test_start_stack_reports_script_failure(monkeypatch, tmp_path):
    """Failed GUI startup should return structured error details to the executor."""
    from nexus.runtime.server import gui

    script = tmp_path / 'gui_stack.sh'
    script.write_text('#!/usr/bin/env bash\n', encoding='utf-8')

    class FakeCompletedProcess:
        returncode = 7
        stdout = 'partial'
        stderr = 'boom'

    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    monkeypatch.setattr(gui, '_gui_stack_script', lambda: script)
    monkeypatch.setattr(gui.subprocess, 'run', lambda *args, **kwargs: FakeCompletedProcess())

    result = gui._start_stack_sync()

    assert result['success'] is False
    assert result['stdout'] == 'partial'
    assert result['stderr'] == 'boom'
    assert result['error'] == 'gui stack start failed with return code 7'


def test_screenshot_prefers_gnome_screenshot_when_available(monkeypatch, tmp_path):
    """GNOME Shell sessions should use the compositor-aware screenshot tool when present."""
    from nexus.runtime.server import gui

    screenshot_path = tmp_path / 'screenshot.png'
    calls: list[list[str]] = []

    def fake_run_gui_command(command: list[str], *, check: bool = True, timeout: float = 30.0):
        calls.append(command)
        screenshot_path.write_bytes(b'gnome-png')

    monkeypatch.setattr(gui, '_SCROT_SCREENSHOT_PATH', screenshot_path)
    monkeypatch.setattr(gui.XDoToolBackend, 'screen_size', lambda self: (1280, 800))
    monkeypatch.setattr(gui.shutil, 'which', lambda command: '/usr/bin/gnome-screenshot')
    monkeypatch.setattr(gui, '_run_gui_command', fake_run_gui_command)

    image, width, height = gui.XDoToolBackend().screenshot()

    assert image == b'gnome-png'
    assert (width, height) == (1280, 800)
    assert calls == [['gnome-screenshot', '-f', str(screenshot_path)]]


def test_screenshot_falls_back_to_scrot_when_gnome_screenshot_is_missing(monkeypatch, tmp_path):
    """Base runtime screenshots should keep using scrot when GNOME tooling is absent."""
    from nexus.runtime.server import gui

    screenshot_path = tmp_path / 'screenshot.png'
    calls: list[list[str]] = []

    def fake_run_gui_command(command: list[str], *, check: bool = True, timeout: float = 30.0):
        calls.append(command)
        screenshot_path.write_bytes(b'scrot-png')

    monkeypatch.setattr(gui, '_SCROT_SCREENSHOT_PATH', screenshot_path)
    monkeypatch.setattr(gui.XDoToolBackend, 'screen_size', lambda self: (800, 600))
    monkeypatch.setattr(gui.shutil, 'which', lambda command: None)
    monkeypatch.setattr(gui, '_run_gui_command', fake_run_gui_command)

    image, width, height = gui.XDoToolBackend().screenshot()

    assert image == b'scrot-png'
    assert (width, height) == (800, 600)
    assert calls == [['scrot', '-z', '-q', '100', str(screenshot_path)]]
