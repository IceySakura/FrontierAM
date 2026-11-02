import base64

import pytest


@pytest.mark.asyncio
async def test_gui_start_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiStartRequest, gui_start

    calls: list[dict[str, object]] = []

    async def _fake_start_stack(**kwargs):
        calls.append(kwargs)
        return {
            'success': True,
            'display': ':99',
            'vnc_port': 5900,
            'novnc_port': 6080,
            'desktop_session': 'gnome-session',
            'stdout': 'started',
            'stderr': '',
            'error': None,
        }

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.start_stack', _fake_start_stack)

    response = await gui_start(
        GuiStartRequest(
            desktop_session='gnome-session',
            window_manager='none',
            screen_width=1280,
            screen_height=800,
            session_options={'gnome_session_name': 'ubuntu'},
        )
    )

    assert response.success is True
    assert response.display == ':99'
    assert response.desktop_session == 'gnome-session'
    assert calls == [
        {
            'desktop_session': 'gnome-session',
            'gnome_session_name': 'ubuntu',
            'session_user': 'user',
            'session_home_dir': '/home/user',
            'window_manager': 'none',
            'screen_width': 1280,
            'screen_height': 800,
        }
    ]


@pytest.mark.asyncio
async def test_gui_click_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiClickRequest, gui_click

    calls: list[tuple[int | None, int | None, str, int]] = []

    async def _fake_click(x: int | None, y: int | None, *, button: str, num_clicks: int):
        calls.append((x, y, button, num_clicks))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.click', _fake_click)

    response = await gui_click(GuiClickRequest(x=321, y=654))

    assert response.success is True
    assert calls == [(321, 654, 'left', 1)]


@pytest.mark.asyncio
async def test_gui_double_click_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiDoubleClickRequest, gui_double_click

    calls: list[tuple[int, int]] = []

    async def _fake_double_click(x: int, y: int):
        calls.append((x, y))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.double_click', _fake_double_click)

    response = await gui_double_click(GuiDoubleClickRequest(x=210, y=420))

    assert response.success is True
    assert calls == [(210, 420)]


@pytest.mark.asyncio
async def test_gui_long_click_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiLongClickRequest, gui_long_click

    calls: list[tuple[int, int, int]] = []

    async def _fake_long_click(x: int, y: int, *, duration_ms: int):
        calls.append((x, y, duration_ms))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.long_click', _fake_long_click)

    response = await gui_long_click(GuiLongClickRequest(x=100, y=200, duration=350))

    assert response.success is True
    assert calls == [(100, 200, 350)]


@pytest.mark.asyncio
async def test_gui_drag_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiDragRequest, gui_drag

    calls: list[tuple[int, int, int, int, int]] = []

    async def _fake_drag(x1: int, y1: int, x2: int, y2: int, *, duration_ms: int):
        calls.append((x1, y1, x2, y2, duration_ms))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.drag', _fake_drag)

    response = await gui_drag(GuiDragRequest(x1=10, y1=20, x2=30, y2=40, duration=500))

    assert response.success is True
    assert calls == [(10, 20, 30, 40, 500)]


@pytest.mark.asyncio
async def test_gui_screenshot_executor_returns_base64_png_from_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiScreenshotRequest, gui_screenshot

    async def _fake_screenshot():
        return b'fake-png', 1280, 800

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.screenshot', _fake_screenshot)

    response = await gui_screenshot(GuiScreenshotRequest())

    assert response.success is True
    assert response.width == 1280
    assert response.height == 800
    assert response.format == 'png'
    assert response.image == base64.b64encode(b'fake-png').decode('ascii')


@pytest.mark.asyncio
async def test_gui_screen_size_executor_returns_backend_dimensions(monkeypatch):
    from nexus.runtime.executors.gui import GuiScreenSizeRequest, gui_get_screen_size

    async def _fake_screen_size():
        return 1600, 900

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.screen_size', _fake_screen_size)

    response = await gui_get_screen_size(GuiScreenSizeRequest())

    assert response.success is True
    assert response.width == 1600
    assert response.height == 900


@pytest.mark.asyncio
async def test_gui_launch_and_close_app_use_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import (
        GuiCloseAppRequest,
        GuiLaunchAppRequest,
        gui_close_app,
        gui_launch_app,
    )

    launches: list[tuple[str, dict[str, object]]] = []
    closes: list[str] = []

    async def _fake_launch(command: str, **kwargs):
        launches.append((command, kwargs))
        return {'success': True}

    async def _fake_close(app_name: str):
        closes.append(app_name)
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.launch_app', _fake_launch)
    monkeypatch.setattr('nexus.runtime.executors.gui.gui.close_app', _fake_close)

    launch_response = await gui_launch_app(GuiLaunchAppRequest(app='xterm -geometry 80x24+120+80'))
    close_response = await gui_close_app(GuiCloseAppRequest(app='xterm'))

    assert launch_response.success is True
    assert close_response.success is True
    assert launches == [('xterm -geometry 80x24+120+80', {'user': 'user', 'home_dir': '/home/user'})]
    assert closes == ['xterm']


@pytest.mark.asyncio
async def test_gui_type_executor_uses_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import GuiTypeRequest, gui_type

    typed: list[str] = []

    async def _fake_type_text(text: str):
        typed.append(text)
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.type_text', _fake_type_text)

    response = await gui_type(GuiTypeRequest(text='hello world'))

    assert response.success is True
    assert typed == ['hello world']


@pytest.mark.asyncio
async def test_gui_press_key_normalizes_special_key_names(monkeypatch):
    from nexus.runtime.executors.gui import GuiPressKeyRequest, gui_press_key

    pressed_keys: list[str] = []

    async def _fake_press_key(key: str):
        pressed_keys.append(key)
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.press_key', _fake_press_key)

    response = await gui_press_key(GuiPressKeyRequest(key='Enter'))

    assert response.success is True
    assert pressed_keys == ['Enter']


@pytest.mark.asyncio
async def test_gui_computer_13_mouse_executors_use_local_backend(monkeypatch):
    from nexus.runtime.executors.gui import (
        GuiHotkeyRequest,
        GuiKeyDownRequest,
        GuiKeyUpRequest,
        GuiMouseDownRequest,
        GuiMouseUpRequest,
        GuiMoveToRequest,
        GuiPyAutoGUIRequest,
        GuiRightClickRequest,
        GuiScrollRequest,
        gui_hotkey,
        gui_key_down,
        gui_key_up,
        gui_mouse_down,
        gui_mouse_up,
        gui_move_to,
        gui_pyautogui,
        gui_right_click,
        gui_scroll,
    )

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def _record(name: str, *args: object, **kwargs: object):
        calls.append((name, args, kwargs))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.move_to', lambda *a, **kw: _record('move_to', *a, **kw))
    monkeypatch.setattr(
        'nexus.runtime.executors.gui.gui.mouse_down', lambda *a, **kw: _record('mouse_down', *a, **kw)
    )
    monkeypatch.setattr(
        'nexus.runtime.executors.gui.gui.mouse_up', lambda *a, **kw: _record('mouse_up', *a, **kw)
    )
    monkeypatch.setattr(
        'nexus.runtime.executors.gui.gui.right_click', lambda *a, **kw: _record('right_click', *a, **kw)
    )
    monkeypatch.setattr('nexus.runtime.executors.gui.gui.scroll', lambda *a, **kw: _record('scroll', *a, **kw))
    monkeypatch.setattr(
        'nexus.runtime.executors.gui.gui.key_down', lambda *a, **kw: _record('key_down', *a, **kw)
    )
    monkeypatch.setattr('nexus.runtime.executors.gui.gui.key_up', lambda *a, **kw: _record('key_up', *a, **kw))
    monkeypatch.setattr('nexus.runtime.executors.gui.gui.hotkey', lambda *a, **kw: _record('hotkey', *a, **kw))
    monkeypatch.setattr(
        'nexus.runtime.executors.gui.gui.run_pyautogui',
        lambda *a, **kw: _record('run_pyautogui', *a, **kw),
    )

    assert (await gui_move_to(GuiMoveToRequest(x=11, y=22))).success is True
    assert (await gui_mouse_down(GuiMouseDownRequest(button='right'))).success is True
    assert (await gui_mouse_up(GuiMouseUpRequest(button='right'))).success is True
    assert (await gui_right_click(GuiRightClickRequest(x=33, y=44))).success is True
    assert (await gui_scroll(GuiScrollRequest(dx=5, dy=-9))).success is True
    assert (await gui_key_down(GuiKeyDownRequest(key='Control'))).success is True
    assert (await gui_key_up(GuiKeyUpRequest(key='Control'))).success is True
    assert (await gui_hotkey(GuiHotkeyRequest(keys=['Control', 'S']))).success is True
    assert (
        await gui_pyautogui(GuiPyAutoGUIRequest(code='pyautogui.press("enter")', user='user', home_dir='/home/user'))
    ).success is True

    assert calls == [
        ('move_to', (11, 22), {}),
        ('mouse_down', (), {'button': 'right'}),
        ('mouse_up', (), {'button': 'right'}),
        ('right_click', (33, 44), {}),
        ('scroll', (), {'dx': 5, 'dy': -9}),
        ('key_down', ('Control',), {}),
        ('key_up', ('Control',), {}),
        ('hotkey', (['Control', 'S'],), {}),
        ('run_pyautogui', ('pyautogui.press("enter")',), {'timeout': 30.0, 'user': 'user', 'home_dir': '/home/user'}),
    ]


@pytest.mark.asyncio
async def test_gui_computer_13_dispatches_osworld_action_dicts(monkeypatch):
    from nexus.runtime.executors.gui import GuiComputer13Request, gui_computer_13

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def _record(name: str, *args: object, **kwargs: object):
        calls.append((name, args, kwargs))
        return {'success': True}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.drag_to', lambda *a, **kw: _record('drag_to', *a, **kw))
    monkeypatch.setattr('nexus.runtime.executors.gui.gui.hotkey', lambda *a, **kw: _record('hotkey', *a, **kw))
    monkeypatch.setattr('nexus.runtime.executors.gui.asyncio.sleep', lambda delay: _record('sleep', delay))

    drag_response = await gui_computer_13(
        GuiComputer13Request(action_type='DRAG_TO', parameters={'x': 44, 'y': 55, 'duration': 700})
    )
    hotkey_response = await gui_computer_13(
        GuiComputer13Request(action_type='HOTKEY', parameters={'keys': ['Control', 'S']})
    )
    wait_response = await gui_computer_13(GuiComputer13Request(action_type='WAIT', parameters={'duration': 1.25}))
    done_response = await gui_computer_13(GuiComputer13Request(action_type='DONE'))
    fail_response = await gui_computer_13(GuiComputer13Request(action_type='FAIL'))

    assert drag_response.success is True
    assert hotkey_response.success is True
    assert wait_response.success is True
    assert done_response.done is True
    assert fail_response.fail is True
    assert calls == [
        ('drag_to', (44, 55), {'duration_ms': 700}),
        ('hotkey', (['Control', 'S'],), {}),
        ('sleep', (1.25,), {}),
    ]


@pytest.mark.asyncio
async def test_gui_reset_apps_executor_returns_stopped_packages(monkeypatch):
    from nexus.runtime.executors.gui import GuiResetAppsRequest, gui_reset_apps

    async def _fake_reset_apps():
        return {'success': True, 'stopped_packages': ['xterm', 'xmessage']}

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.reset_apps', _fake_reset_apps)

    response = await gui_reset_apps(GuiResetAppsRequest())

    assert response.success is True
    assert response.stopped_packages == ['xterm', 'xmessage']


@pytest.mark.asyncio
async def test_gui_get_device_info_executor_returns_backend_fields(monkeypatch):
    from nexus.runtime.executors.gui import GuiGetDeviceInfoRequest, gui_get_device_info

    async def _fake_get_device_info():
        return {
            'success': True,
            'device_name': 'nexus-linux-gui',
            'platform_version': 'Linux-test',
            'automation_name': 'xdotool',
            'window_width': 1280,
            'window_height': 800,
            'wm_size': '1280x800',
            'wm_density': '1',
            'current_package': 'Openbox',
            'current_activity': None,
            'error': None,
        }

    monkeypatch.setattr('nexus.runtime.executors.gui.gui.get_device_info', _fake_get_device_info)

    response = await gui_get_device_info(GuiGetDeviceInfoRequest())

    assert response.success is True
    assert response.device_name == 'nexus-linux-gui'
    assert response.platform_version == 'Linux-test'
    assert response.automation_name == 'xdotool'
    assert response.window_width == 1280
    assert response.window_height == 800
    assert response.wm_size == '1280x800'
    assert response.wm_density == '1'
    assert response.current_package == 'Openbox'
    assert response.current_activity is None
