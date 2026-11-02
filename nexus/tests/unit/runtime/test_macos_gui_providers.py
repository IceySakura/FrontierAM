"""Remote desktop action semantics without importing host PyAutoGUI."""

import asyncio

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.smartrun.client import SmartRunRuntime


@pytest.fixture(params=['devcloud', 'smartrun'])
def action_code(request):
    def generate(name, arguments):
        runtime = (
            DevCloudRuntime(base_url='http://test')
            if request.param == 'devcloud'
            else SmartRunRuntime(
                device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
            )
        )
        target = '_run_python_controller_code' if isinstance(runtime, DevCloudRuntime) else '_desktop_python'
        with patch.object(runtime, target, new_callable=AsyncMock) as execute:
            execute.return_value = {'returncode': 0}
            result = asyncio.run(runtime._run_gui_pyautogui(name, arguments))
            assert result['success'], result
            return execute.call_args.args[0], 30.0

    return generate


@pytest.mark.parametrize(
    ('name', 'params', 'method', 'kwargs'),
    [
        (
            'gui_click',
            {'x': 1, 'y': 2, 'button': 'right', 'num_clicks': 1},
            'click',
            {'x': 1, 'y': 2, 'button': 'right', 'clicks': 1},
        ),
        ('gui_right_click', {}, 'rightClick', {'x': None, 'y': None}),
        ('gui_mouse_down', {'button': 'middle'}, 'mouseDown', {'button': 'middle'}),
        ('gui_mouse_up', {'button': 'right'}, 'mouseUp', {'button': 'right'}),
    ],
)
def test_generated_code_preserves_pointer_arguments(action_code, name, params, method, kwargs):
    desktop = MagicMock()
    code, _ = action_code(name, params)
    exec(code, {'pyautogui': desktop})
    getattr(desktop, method).assert_called_once_with(**kwargs)


def test_drag_scroll_and_modifier_semantics(action_code):
    desktop = MagicMock()
    for name, params in [
        ('gui_drag_to', {'x': 4, 'y': 5, 'duration': 700}),
        ('gui_scroll', {'dx': -2, 'dy': 3}),
        ('gui_key_down', {'key': 'CMD'}),
        ('gui_key_up', {'key': 'CMD'}),
        ('gui_hotkey', {'keys': ['cmd', 'shift', 's']}),
    ]:
        code, _ = action_code(name, params)
        exec(code, {'pyautogui': desktop})
    desktop.dragTo.assert_called_once_with(4, 5, duration=0.7, button='left')
    desktop.hscroll.assert_called_once_with(-2)
    desktop.scroll.assert_called_once_with(3)
    desktop.keyDown.assert_called_once_with('command')
    desktop.keyUp.assert_called_once_with('command')
    desktop.hotkey.assert_called_once_with('command', 'shift', 's')


def test_long_click_releases_on_sleep_failure(action_code):
    desktop = MagicMock()
    clock = MagicMock()
    clock.sleep.side_effect = RuntimeError('interrupted')
    code, _ = action_code('gui_long_click', {'x': 1, 'y': 2})
    with pytest.raises(RuntimeError):
        exec(code, {'pyautogui': desktop, 'time': clock})
    desktop.mouseUp.assert_called_once()


@pytest.fixture(params=['devcloud', 'smartrun'])
def runtime(request):
    if request.param == 'devcloud':
        return DevCloudRuntime(base_url='http://test')
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    runtime._connected = True
    return runtime


async def test_public_interfaces_and_restricted_code(runtime):
    target = '_run_python_controller_code' if isinstance(runtime, DevCloudRuntime) else '_desktop_python'
    with patch.object(runtime, target, new_callable=AsyncMock) as execute:
        execute.return_value = {'returncode': 0, 'output': '', 'error': ''}
        assert (await runtime.gui_move_to(1, 2)).success
        assert (await runtime.gui_scroll(dx=2, dy=-3)).success
        execute.assert_awaited()
        execute.reset_mock()
        if isinstance(runtime, DevCloudRuntime):
            with pytest.raises(ValueError, match='pyautogui code is not allowed'):
                await runtime.gui_pyautogui('import os; os.system("id")')
        else:
            assert not (await runtime.gui_pyautogui('import os; os.system("id")')).success
        execute.assert_not_awaited()
        assert (await runtime.gui_pyautogui('pyautogui.moveTo(1, 2)')).success
        assert (await runtime.gui_start()).success
        if isinstance(runtime, DevCloudRuntime):
            with pytest.raises(ValueError, match='Linux desktop configuration'):
                await runtime.gui_start(desktop_session='xfce')
        else:
            assert not (await runtime.gui_start(desktop_session='xfce')).success


async def test_device_info_and_missing_dependencies(runtime):
    target = '_run_python_controller_code' if isinstance(runtime, DevCloudRuntime) else '_desktop_python'
    with patch.object(runtime, target, new_callable=AsyncMock) as execute:
        execute.return_value = {
            'returncode': 0,
            'output': 'NEXUS_GUI_INFO:{"device_name":"mac","window_width":1920}',
            'error': '',
        }
        result = await runtime.gui_get_device_info()
        assert result.success
        assert result.device_name == 'mac'
        assert result.window_width == 1920
        execute.side_effect = RuntimeError('No module named pyautogui')
        with pytest.raises(RuntimeError, match='No module named pyautogui'):
            await runtime.gui_move_to(10, 20)


@pytest.mark.parametrize(
    ('name', 'params'),
    [
        ('gui_double_click', {'x': 10, 'y': 20}),
        ('gui_click', {'num_clicks': 2}),
    ],
)
def test_macos_double_click_sets_system_click_count(action_code, name, params):
    import sys

    quartz, desktop, clock = MagicMock(), MagicMock(), MagicMock()
    desktop.position.return_value = (10, 20)
    code, _ = action_code(name, params)
    with patch.dict(sys.modules, {'Quartz': quartz}):
        exec(code, {'pyautogui': desktop, 'time': clock})
    assert [c.args[2] for c in quartz.CGEventSetIntegerValueField.call_args_list] == [1, 1, 2, 2]
    assert quartz.CGEventPost.call_count == 4


def test_unicode_typing_keeps_non_bmp_characters(action_code):
    import sys

    quartz = MagicMock()
    code, _ = action_code('gui_type', {'text': '中😀'})
    with patch.dict(sys.modules, {'Quartz': quartz}):
        exec(code, {'time': MagicMock()})
    assert [(c.args[1], c.args[2]) for c in quartz.CGEventKeyboardSetUnicodeString.call_args_list] == [
        (1, '中'),
        (1, '中'),
        (2, '😀'),
        (2, '😀'),
    ]


async def test_devcloud_gui_propagates_timeout():
    runtime = DevCloudRuntime(base_url='http://test')
    with patch.object(runtime, '_run_python_controller_code', new_callable=AsyncMock) as execute:
        execute.side_effect = asyncio.TimeoutError('controller timed out')
        with pytest.raises(asyncio.TimeoutError, match='controller timed out'):
            await runtime.gui_move_to(10, 20)


async def test_devcloud_gui_only_applies_explicit_script_timeout():
    runtime = DevCloudRuntime(base_url='http://test', request_timeout=3600)
    with (
        patch.object(runtime, '_run_python_controller_code', new_callable=AsyncMock) as execute,
        patch('nexus.runtime.providers.devcloud.client.asyncio.wait_for', wraps=asyncio.wait_for) as wait,
    ):
        execute.return_value = {'returncode': 0}
        assert (await runtime.gui_move_to(10, 20)).success
        wait.assert_not_called()
        assert runtime._timeout.total == 3600
        assert (await runtime.gui_pyautogui('pyautogui.moveTo(10, 20)', timeout=120)).success
        wait.assert_called_once()
        assert wait.call_args.kwargs['timeout'] == 120


async def test_smartrun_gui_omits_timeout_except_for_explicit_script():
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    runtime._connected = True
    with patch.object(runtime, '_call', new_callable=AsyncMock) as call:
        call.return_value = {'return_code': 0}
        assert (await runtime.gui_move_to(10, 20)).success
        assert 'timeout' not in call.call_args.args[1]
        assert (await runtime.gui_pyautogui('pyautogui.moveTo(10, 20)', timeout=120)).success
        assert call.call_args.args[1]['timeout'] == 120
