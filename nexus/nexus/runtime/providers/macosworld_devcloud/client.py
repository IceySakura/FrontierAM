"""DevCloud OSWorld runtime client."""

from __future__ import annotations

import asyncio
import base64
import json
import shlex

from typing import Any

import aiohttp

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.providers.devcloud.client import DevCloudRuntime as BaseDevCloudRuntime
from nexus.runtime.providers.devcloud.client import _normalize_gui_key
from nexus.runtime.providers.macosworld_devcloud.gui import mac_multi_click
from nexus.utils.async_http import http_client


class MacOSWorldDevCloudRuntime(BaseDevCloudRuntime):
    """Explicit macOSWorld adaptation; legacy provider behavior is inherited unchanged."""

    native_bridge_kind = NativeBridgeRuntimeKind.DEVCLOUD_MACOS

    _GUI_SCREENSHOT_RETRY_ATTEMPTS = 5

    _GUI_SCREENSHOT_RETRY_BACKOFF_SECONDS = 1.0

    _GUI_SCREENSHOT_REQUEST_TIMEOUT_SECONDS = 10.0

    _PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

    async def _run_python_controller_code(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        python_cmd = f'import pyautogui; import time; import platform; pyautogui.FAILSAFE = False; {code}'
        if timeout is not None:
            command = ['bash', '-c', f'timeout --kill-after=5s {timeout:g}s python3 -c {shlex.quote(python_cmd)}']
        else:
            command = ['python3', '-c', python_cmd]
        payload = {'command': command, 'shell': False}
        http_timeout = aiohttp.ClientTimeout(total=timeout + 5) if timeout is not None else None
        body, text, status_code = await self._post_json('/execute', payload, timeout=http_timeout)
        if status_code >= 400 or not isinstance(body, dict):
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        return body

    async def _run_command(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, RunCommandResponse

        request = RunCommandRequest.model_validate(arg)
        payload = self._build_osworld_execute_payload(
            command=request.command,
            cwd=request.cwd,
            timeout=request.timeout,
            stdin=request.stdin,
            extra_env=request.extra_env,
        )
        started = asyncio.get_running_loop().time()
        body, text, status_code = await self._execute_command_payload(payload, request.timeout)
        if status_code >= 400:
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        data = body if isinstance(body, dict) else {}
        return_code = data.get('returncode')
        if return_code == 124:
            status = CommandRunStatus.TimeLimitExceeded
        elif return_code == 0:
            status = CommandRunStatus.Finished
        else:
            status = CommandRunStatus.Error
        return RunCommandResponse(
            status=status,
            execution_time=asyncio.get_running_loop().time() - started,
            return_code=return_code,
            stdout=data.get('output'),
            stderr=data.get('error'),
        ).model_dump()

    async def _gui_screenshot(self) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiScreenshotResponse

        image: bytes | None = None
        last_error: str | None = None
        max_attempts = max(1, int(self._GUI_SCREENSHOT_RETRY_ATTEMPTS))
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await http_client.get(
                    f'{self._base_url}/screenshot',
                    timeout=aiohttp.ClientTimeout(
                        total=min(self.request_timeout, self._GUI_SCREENSHOT_REQUEST_TIMEOUT_SECONDS)
                    ),
                    proxy=self.proxy_url,
                )
                image = self._validate_gui_screenshot_response(resp)
                break
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
                if attempt >= max_attempts:
                    return GuiScreenshotResponse(success=False, error=last_error[:500]).model_dump()
                delay = self._gui_screenshot_retry_backoff(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        if image is None:
            return GuiScreenshotResponse(success=False, error=last_error or 'DevCloud /screenshot failed').model_dump()

        screen = await self._gui_get_screen_size()
        return GuiScreenshotResponse(
            success=True,
            image=base64.b64encode(image).decode('utf-8'),
            width=screen.get('width'),
            height=screen.get('height'),
            format='png',
        ).model_dump()

    def _gui_screenshot_retry_backoff(self, attempt: int) -> float:
        return max(0.0, float(self._GUI_SCREENSHOT_RETRY_BACKOFF_SECONDS) * (2 ** (attempt - 1)))

    @classmethod
    def _validate_gui_screenshot_response(cls, resp: Any) -> bytes:
        headers = getattr(resp, 'headers', {}) or {}
        content_type = headers.get('content-type') or headers.get('Content-Type') or ''
        if resp.status_code >= 400:
            raise RuntimeError(f'DevCloud /screenshot failed: HTTP {resp.status_code} {resp.text[:500]}')

        content = getattr(resp, 'content', None)
        if not isinstance(content, (bytes, bytearray)) or not content:
            raise RuntimeError(
                f'DevCloud /screenshot returned no image bytes: HTTP {resp.status_code} content-type={content_type!r}'
            )
        if not content.startswith(cls._PNG_SIGNATURE):
            raise RuntimeError(
                f'DevCloud /screenshot returned invalid PNG bytes: HTTP {resp.status_code} '
                f'content-type={content_type!r} len={len(content)}'
            )
        return bytes(content)

    async def _run_gui_pyautogui(self, executor_name: str, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models import gui
        from nexus.runtime.pyautogui_guard import validate_pyautogui_code

        r: Any
        timeout = None
        if executor_name == 'gui_click':
            r = gui.GuiClickRequest.model_validate(arg)
            code = f'pyautogui.click(x={r.x!r}, y={r.y!r}, button={r.button!r}, clicks={r.num_clicks})'
            if r.num_clicks > 1:
                code = mac_multi_click(r.x, r.y, r.button, r.num_clicks)
        elif executor_name in ('gui_double_click', 'gui_right_click'):
            r = {'gui_double_click': gui.GuiDoubleClickRequest, 'gui_right_click': gui.GuiRightClickRequest}[
                executor_name
            ].model_validate(arg)
            method = 'doubleClick' if executor_name == 'gui_double_click' else 'rightClick'
            code = (
                mac_multi_click(r.x, r.y, 'left', 2)
                if executor_name == 'gui_double_click'
                else f'pyautogui.{method}(x={r.x!r}, y={r.y!r})'
            )
        elif executor_name == 'gui_move_to':
            r = gui.GuiMoveToRequest.model_validate(arg)
            code = f'pyautogui.moveTo({r.x}, {r.y})'
        elif executor_name in ('gui_mouse_down', 'gui_mouse_up'):
            r = {'gui_mouse_down': gui.GuiMouseDownRequest, 'gui_mouse_up': gui.GuiMouseUpRequest}[
                executor_name
            ].model_validate(arg)
            method = 'mouseDown' if executor_name == 'gui_mouse_down' else 'mouseUp'
            code = f'pyautogui.{method}(button={r.button!r})'
        elif executor_name == 'gui_long_click':
            r = gui.GuiLongClickRequest.model_validate(arg)
            code = (
                f'pyautogui.mouseDown(x={r.x}, y={r.y})\ntry:\n'
                f' time.sleep({max(r.duration, 0) / 1000})\nfinally:\n pyautogui.mouseUp()'
            )
        elif executor_name == 'gui_drag':
            r = gui.GuiDragRequest.model_validate(arg)
            code = (
                f'pyautogui.moveTo({r.x1}, {r.y1}); '
                f'pyautogui.dragTo({r.x2}, {r.y2}, duration={max(r.duration, 0) / 1000}, button="left")'
            )
        elif executor_name == 'gui_drag_to':
            r = gui.GuiDragToRequest.model_validate(arg)
            code = f'pyautogui.dragTo({r.x}, {r.y}, duration={max(r.duration, 0) / 1000}, button="left")'
        elif executor_name == 'gui_scroll':
            r = gui.GuiScrollRequest.model_validate(arg)
            code = f'pyautogui.hscroll({r.dx}); pyautogui.scroll({r.dy})'
        elif executor_name == 'gui_type':
            r = gui.GuiTypeRequest.model_validate(arg)
            code = f'pyautogui.write({r.text!r})'
            if not r.text.isascii():
                code = (
                    'import Quartz\n'
                    f'for character in {r.text!r}:\n'
                    ' for pressed in (True, False):\n'
                    '  event = Quartz.CGEventCreateKeyboardEvent(None, 0, pressed)\n'
                    '  Quartz.CGEventKeyboardSetUnicodeString(\n'
                    '   event, len(character.encode("utf-16-le")) // 2, character)\n'
                    '  Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)\n'
                    '  time.sleep(0.01)'
                )
        elif executor_name in ('gui_press_key', 'gui_key_down', 'gui_key_up'):
            r = {
                'gui_press_key': gui.GuiPressKeyRequest,
                'gui_key_down': gui.GuiKeyDownRequest,
                'gui_key_up': gui.GuiKeyUpRequest,
            }[executor_name].model_validate(arg)
            method = {'gui_press_key': 'press', 'gui_key_down': 'keyDown', 'gui_key_up': 'keyUp'}[executor_name]
            key = _normalize_gui_key(r.key)
            code = f'pyautogui.{method}({key!r})'
        elif executor_name == 'gui_hotkey':
            r = gui.GuiHotkeyRequest.model_validate(arg)
            code = f'pyautogui.hotkey(*{[_normalize_gui_key(k) for k in r.keys]!r})'
        elif executor_name == 'gui_pyautogui':
            r = gui.GuiPyAutoGUIRequest.model_validate(arg)
            validate_pyautogui_code(r.code)
            code, timeout = r.code, r.timeout
        elif executor_name == 'gui_get_device_info':
            r = gui.GuiGetDeviceInfoRequest.model_validate(arg)
            code = (
                'import json; w,h=pyautogui.size(); '
                'print("NEXUS_GUI_INFO:"+json.dumps(dict(device_name=platform.node(), '
                'platform_version=platform.platform(), automation_name="pyautogui", '
                'window_width=w, window_height=h, wm_size=str(w)+"x"+str(h))))'
            )
        else:
            r = gui.GuiStartRequest.model_validate(arg)
            if any((r.desktop_session, r.window_manager, r.screen_width, r.screen_height, r.session_options)):
                raise ValueError('macOS uses an existing desktop; Linux desktop configuration is not supported')
            code = 'pyautogui.size()'
        if timeout is None:
            data = await self._run_python_controller_code(code)
        else:
            data = await asyncio.wait_for(self._run_python_controller_code(code, timeout=timeout), timeout=timeout)
        result = {'success': data.get('returncode') == 0, 'error': data.get('error') or None}
        if executor_name == 'gui_pyautogui':
            result.update(stdout=data.get('output', ''), stderr=data.get('error', ''))
        if executor_name == 'gui_get_device_info' and result['success']:
            line = next((x for x in data.get('output', '').splitlines() if x.startswith('NEXUS_GUI_INFO:')), None)
            if line is None:
                raise RuntimeError('Desktop returned no device information')
            result.update(json.loads(line.split(':', 1)[1]))
        return result

    def create_native_bridge_adapter(self, runtime: Any):
        from nexus.runtime.providers.macosworld_devcloud.bridge import MacOSWorldBridgeAdapter

        # Keep task-local wrappers (PATH and screenshot policies) in the bridge path.
        return MacOSWorldBridgeAdapter(runtime)
