import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeRuntime:
    """Records the GUI/executor calls made by the mobile/* tools."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def gui_tap(self, x, y):
        self.calls.append(('gui_tap', x, y))
        return SimpleNamespace(success=True, error=None)

    async def gui_double_tap(self, x, y):
        self.calls.append(('gui_double_tap', x, y))
        return SimpleNamespace(success=True, error=None)

    async def gui_long_press(self, x, y):
        self.calls.append(('gui_long_press', x, y))
        return SimpleNamespace(success=True, error=None)

    async def gui_type(self, text):
        self.calls.append(('gui_type', text))
        return SimpleNamespace(success=True, error=None)

    async def gui_press_key(self, key):
        self.calls.append(('gui_press_key', key))
        return SimpleNamespace(success=True, error=None)

    async def gui_launch_app(self, app):
        self.calls.append(('gui_launch_app', app))
        return SimpleNamespace(success=True, error=None)

    async def call_executor(self, name, arg):
        self.calls.append((name, arg))
        return {'success': True}


def _ctx_with_runtime(screen=(1000, 2000)):
    """ExecutionContext with a fake android runtime and a published screen size."""
    from nexus.core.context import ExecutionContext
    from nexus.extensions.tools.mobile.observation import (
        MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE,
        MOBILE_OBSERVATION_SCREEN_WIDTH_STATE,
    )

    ctx = ExecutionContext(task_name='android_world', runtime_configs={}, tool_configs={})
    ctx._runtimes['android'] = _FakeRuntime()
    ctx.tool_state[MOBILE_OBSERVATION_SCREEN_WIDTH_STATE] = screen[0]
    ctx.tool_state[MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE] = screen[1]
    return ctx


# --- Coordinate scaling -------------------------------------------------------


def test_scale_point_normalized_to_pixels():
    from nexus.extensions.tools.mobile.tools import _scale_point

    ctx = _ctx_with_runtime(screen=(1000, 2000))

    async def run():
        async with ctx:
            return _scale_point([0.5, 0.25])

    import asyncio

    assert asyncio.run(run()) == (500, 500)


def test_scale_point_clamps_to_screen_bounds():
    from nexus.extensions.tools.mobile.tools import _scale_point

    ctx = _ctx_with_runtime(screen=(1080, 2400))

    async def run():
        async with ctx:
            return _scale_point([1.0, 1.0])

    import asyncio

    x, y = asyncio.run(run())
    assert x == 1079
    assert y == 2399


# --- Per-tool dispatch --------------------------------------------------------


@pytest.mark.asyncio
async def test_mobile_click_scales_and_taps():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import ClickInput, mobile_click_tool

    ctx = _ctx_with_runtime(screen=(1000, 2000))
    async with ctx:
        resp = await mobile_click_tool(ClickInput(point=[0.5, 0.5]), context=ToolContext())

    assert resp.meta['success'] is True
    assert ctx._runtimes['android'].calls[-1] == ('gui_tap', 500, 1000)


@pytest.mark.asyncio
async def test_mobile_type_optionally_focuses_then_types():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import TypeInput, mobile_type_tool

    ctx = _ctx_with_runtime(screen=(1000, 2000))
    async with ctx:
        await mobile_type_tool(TypeInput(text='hello', point=[0.1, 0.2]), context=ToolContext())

    calls = ctx._runtimes['android'].calls
    assert calls[0] == ('gui_tap', 100, 400)
    assert calls[1] == ('gui_type', 'hello')


@pytest.mark.asyncio
async def test_mobile_swipe_scales_both_endpoints():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import SwipeInput, mobile_swipe_tool

    ctx = _ctx_with_runtime(screen=(1000, 2000))
    async with ctx:
        # 'from' is aliased; construct via dict to use the alias.
        params = SwipeInput.model_validate({'from': [0.5, 0.8], 'to': [0.5, 0.2]})
        await mobile_swipe_tool(params, context=ToolContext())

    assert ctx._runtimes['android'].calls[-1] == ('aw_swipe', {'direction': '500,1600,500,400'})


@pytest.mark.asyncio
async def test_mobile_long_press_and_double_click():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import (
        DoubleClickInput,
        LongPressInput,
        mobile_double_click_tool,
        mobile_long_press_tool,
    )

    ctx = _ctx_with_runtime(screen=(1000, 2000))
    async with ctx:
        await mobile_long_press_tool(LongPressInput(point=[0.5, 0.5]), context=ToolContext())
        await mobile_double_click_tool(DoubleClickInput(point=[0.5, 0.5]), context=ToolContext())

    calls = ctx._runtimes['android'].calls
    assert ('gui_long_press', 500, 1000) in calls
    assert ('gui_double_tap', 500, 1000) in calls


@pytest.mark.parametrize(
    ('button', 'expected_key'),
    [('back', 'Back'), ('home', 'Home'), ('enter', 'Enter')],
)
@pytest.mark.asyncio
async def test_mobile_button_press_maps_keys(button, expected_key):
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import ButtonPressInput, mobile_button_press_tool

    ctx = _ctx_with_runtime()
    async with ctx:
        await mobile_button_press_tool(ButtonPressInput(button=button), context=ToolContext())

    assert ctx._runtimes['android'].calls[-1] == ('gui_press_key', expected_key)


@pytest.mark.asyncio
async def test_mobile_open_app():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import OpenAppInput, mobile_open_app_tool

    ctx = _ctx_with_runtime()
    async with ctx:
        await mobile_open_app_tool(OpenAppInput(package='Settings'), context=ToolContext())

    assert ctx._runtimes['android'].calls[-1] == ('gui_launch_app', 'Settings')


@pytest.mark.asyncio
async def test_mobile_answer_submits_and_requests_termination():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import (
        MOBILE_ANSWER_STATE,
        MOBILE_LAST_ACTION_STATE,
        AnswerInput,
        mobile_answer_tool,
    )

    ctx = _ctx_with_runtime()
    async with ctx:
        resp = await mobile_answer_tool(AnswerInput(text='42'), context=ToolContext())

    assert resp.meta['termination_requested'] is True
    assert resp.context_updates[MOBILE_LAST_ACTION_STATE] == 'answer'
    assert resp.context_updates[MOBILE_ANSWER_STATE] == '42'
    assert ctx._runtimes['android'].calls[-1] == ('aw_answer', {'answer': '42'})


@pytest.mark.asyncio
async def test_mobile_terminate_requests_termination_with_status():
    from nexus.core import ToolContext
    from nexus.extensions.tools.mobile.tools import MOBILE_LAST_ACTION_STATE, TerminateInput, mobile_terminate_tool

    ctx = _ctx_with_runtime()
    async with ctx:
        resp = await mobile_terminate_tool(TerminateInput(status='success'), context=ToolContext())

    assert resp.meta['termination_requested'] is True
    assert resp.context_updates[MOBILE_LAST_ACTION_STATE] == 'terminate'


@pytest.mark.parametrize(
    ('tool_name', 'input_factory'),
    [
        ('mobile_click_tool', lambda m: m.ClickInput(point=[0.1, 0.1])),
        ('mobile_swipe_tool', lambda m: m.SwipeInput.model_validate({'from': [0.1, 0.1], 'to': [0.2, 0.2]})),
        ('mobile_button_press_tool', lambda m: m.ButtonPressInput(button='home')),
    ],
)
@pytest.mark.asyncio
async def test_non_terminal_tools_do_not_request_termination(tool_name, input_factory):
    import nexus.extensions.tools.mobile.tools as m

    from nexus.core import ToolContext

    ctx = _ctx_with_runtime()
    tool = getattr(m, tool_name)
    async with ctx:
        resp = await tool(input_factory(m), context=ToolContext())

    assert resp.meta['termination_requested'] is False


# --- Input model validation ---------------------------------------------------


def test_click_input_rejects_extra_keys():
    from pydantic import ValidationError

    from nexus.extensions.tools.mobile.tools import ClickInput

    with pytest.raises(ValidationError):
        ClickInput(point=[0.5, 0.5], x=50, y=60)


def test_terminate_input_rejects_bad_status():
    from pydantic import ValidationError

    from nexus.extensions.tools.mobile.tools import TerminateInput

    with pytest.raises(ValidationError):
        TerminateInput(status='done')


# --- Manifest / registration ---------------------------------------------------


def test_mobile_pyproject_registers_all_tools():
    manifest = Path('nexus/extensions/tools/mobile/pyproject.toml').read_text(encoding='utf-8')
    for name in [
        'mobile/click', 'mobile/type', 'mobile/press', 'mobile/wait', 'mobile/answer',
        'mobile/terminate', 'mobile/swipe', 'mobile/long_press', 'mobile/double_click',
        'mobile/button_press', 'mobile/open_app',
    ]:
        assert f'"{name}" = "nexus.extensions.tools.mobile.tools"' in manifest
    assert '"mobile/computer"' not in manifest


def test_android_world_import_registers_all_mobile_tools():
    code = """\
from nexus.core import Registry, task_registry, tool_registry
Registry.import_module('nexus.extensions.tasks.android_world')
task_registry.get_entry('android_world')
for n in ['mobile/click','mobile/type','mobile/press','mobile/wait','mobile/answer','mobile/terminate',
          'mobile/swipe','mobile/long_press','mobile/double_click','mobile/button_press','mobile/open_app']:
    tool_registry.get_entry(n)
print('ok')
"""
    completed = subprocess.run([sys.executable, '-c', code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.strip() == 'ok'
