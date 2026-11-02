"""SmartRun cloud real-device runtime implementation."""

from __future__ import annotations

import base64
import json
import logging
import shlex

from importlib.resources import files
from typing import TYPE_CHECKING, Any

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.client.base import Runtime
from nexus.utils.async_http import http_client


if TYPE_CHECKING:
    from nexus.runtime.models import InvokeRequest

logger = logging.getLogger(__name__)


class SmartRunRuntime(Runtime):
    """SmartRun-based runtime for iOS/Android/macOS cloud devices.

    Translates nexus GUI executor calls into SmartRun MCP HTTP requests.
    Supports iOS, Android and macOS; platform differences are handled internally.

    Example:
        runtime = SmartRunRuntime(
            device_id='00008020-000E6D863C01002E',
            session_id='session-abc',
            screen_size=(1125, 2436),
            platform='ios',
            api_base_url='https://smartrun.woa.com',
            token='mcp_xxx',
        )
        await runtime.connect()
        result = await runtime.gui_screenshot()
        await runtime.disconnect()
    """

    def __init__(
        self,
        device_id: str,
        session_id: str,
        screen_size: tuple[int, int],
        platform: str,
        *,
        api_base_url: str = 'https://smartrun.woa.com',
        token: str,
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self.session_id = session_id
        self.screen_size = screen_size
        self.platform = platform
        self._api_base_url = api_base_url.rstrip('/')
        self._token = token
        self._connected = False

    @property
    def native_bridge_kind(self) -> NativeBridgeRuntimeKind | None:
        return NativeBridgeRuntimeKind.SMARTRUN_MACOS if self.platform == 'macos' else None

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Verify session is valid and mark connected."""
        data = await self._call('get_session_status', {'session_id': self.session_id, 'json_output': True})
        if not data.get('success'):
            raise RuntimeError(f'SmartRun session {self.session_id} is not valid: {data}')
        self._connected = True
        logger.debug(
            'SmartRunRuntime connected: device=%s session=%s platform=%s screen=%s',
            self.device_id,
            self.session_id,
            self.platform,
            self.screen_size,
        )
        return True

    async def disconnect(self) -> bool:
        """Mark disconnected. Actual release is handled by the provider."""
        self._connected = False
        return True

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request to SmartRun MCP and return the parsed result dict.

        Raises:
            RuntimeError: when SmartRun returns isError=true or HTTP error.
        """
        import json

        url = f'{self._api_base_url}/mcp'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json',
        }
        payload = {
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {'name': tool, 'arguments': arguments},
            'id': 1,
        }

        resp = await http_client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f'SmartRun HTTP error {resp.status_code} for tool={tool}: {resp.text[:512]}')

        # SmartRun returns SSE lines; find the data: line
        if resp.text.lstrip().startswith('{'):
            rpc = json.loads(resp.text)
        else:
            data_line = next((line[5:].strip() for line in resp.text.splitlines() if line.startswith('data:')), '')
            if not data_line:
                raise RuntimeError(f'SmartRun no data line for tool={tool}')
            rpc = json.loads(data_line)
        if 'error' in rpc:
            raise RuntimeError(f'SmartRun {tool} JSON-RPC error: {rpc["error"]}')
        result = rpc.get('result', {})
        content = result.get('content', [{}])
        text = content[0].get('text', '{}') if content else '{}'
        if result.get('isError'):
            # MCP marks nonzero shell exits as tool errors. Preserve their exit
            # code so callers can probe files and handle ordinary command failures.
            if tool == 'shell':
                try:
                    shell_result = json.loads(text)
                except json.JSONDecodeError:
                    shell_result = {}
                if isinstance(shell_result, dict) and type(shell_result.get('return_code')) is int:
                    return shell_result
            raise RuntimeError(f'SmartRun tool={tool} error: {text}')
        return json.loads(text)

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        """Execute a GUI action via SmartRun API."""
        if not self._connected:
            raise RuntimeError('SmartRunRuntime is not connected. Call connect() first.')
        try:
            return await self._dispatch(executor_name, arg)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error('SmartRunRuntime executor %s failed: %s', executor_name, e)
            return {'success': False, 'error': str(e)}

    async def _dispatch(self, executor_name: str, arg: Any) -> dict[str, Any]:
        """Route executor_name to the appropriate SmartRun tool call."""
        base = {'device_id': self.device_id, 'session_id': self.session_id, 'json_output': True}

        if executor_name == 'gui_screenshot':
            result = await self._call('screenshot', base)
            url = result.get('screenshot_url', '')
            if not url:
                return {'success': False, 'error': 'no screenshot_url in response'}
            image_resp = await http_client.get(url)
            image_b64 = base64.b64encode(image_resp.content).decode('utf-8')
            return {
                'success': True,
                'image': image_b64,
                'width': self.screen_size[0],
                'height': self.screen_size[1],
                'format': 'jpg',
            }

        if executor_name == 'gui_get_screen_size':
            return {'success': True, 'width': self.screen_size[0], 'height': self.screen_size[1]}

        if self.platform == 'macos':
            if executor_name in {'gui_start_recording', 'gui_stop_recording'}:
                return await self._gui_recording(executor_name, arg)
            if executor_name in ('gui_press_key', 'gui_hotkey'):
                keys = arg['keys'] if executor_name == 'gui_hotkey' else [arg['key']]
                if not keys or any(not key for key in keys):
                    return {'success': False, 'error': 'empty key combination'}
                if keys[-1] in ('Home', 'Back'):
                    return {'success': False, 'error': 'macOS has no mobile Home/Back action'}
            if executor_name in {
                'gui_move_to',
                'gui_mouse_down',
                'gui_mouse_up',
                'gui_drag_to',
                'gui_scroll',
                'gui_press_key',
                'gui_key_down',
                'gui_key_up',
                'gui_hotkey',
                'gui_pyautogui',
                'gui_get_device_info',
                'gui_start',
            }:
                return await self._run_gui_pyautogui(executor_name, arg)
            if executor_name in ('gui_click', 'gui_double_click', 'gui_right_click') and (
                arg.get('x') is None
                or arg.get('y') is None
                or arg.get('button', 'left') != 'left'
                or arg.get('num_clicks', 1) != 1
            ):
                return await self._run_gui_pyautogui(executor_name, arg)

            if executor_name == 'gui_right_click':
                if arg.get('x') is None or arg.get('y') is None:
                    return {'success': False, 'error': 'SmartRun requires explicit click coordinates'}
                result = await self._call('right_click', {**base, 'x': arg['x'], 'y': arg['y']})
                return {'success': result.get('success', False)}
            if executor_name == 'run_command':
                if arg.get('stdin') is not None:
                    raise RuntimeError('SmartRun shell does not yet support stdin')
                command = arg['command']
                if arg.get('extra_env'):
                    command = shlex.join([
                        'env',
                        *[f'{k}={v}' for k, v in arg['extra_env'].items()],
                        'sh',
                        '-c',
                        command,
                    ])
                if arg.get('cwd'):
                    command = f'cd {shlex.quote(arg["cwd"])} && {command}'
                result = await self._call('shell', {**base, 'command': command, 'timeout': arg['timeout']})
                return {
                    'status': 'Finished' if isinstance(result.get('return_code'), int) else 'Error',
                    'return_code': result.get('return_code'),
                    'stdout': result.get('stdout', ''),
                    'stderr': result.get('stderr', ''),
                }

        if executor_name == 'gui_click':
            if arg.get('x') is None or arg.get('y') is None:
                return {'success': False, 'error': 'SmartRun requires explicit click coordinates'}
            if arg.get('button', 'left') != 'left' or arg.get('num_clicks', 1) != 1:
                return {'success': False, 'error': 'Use gui_right_click or gui_double_click for this action'}
            result = await self._call('tap', {**base, 'x': arg['x'], 'y': arg['y']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_double_click':
            result = await self._call('double_tap', {**base, 'x': arg['x'], 'y': arg['y']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_long_click':
            result = await self._call(
                'long_press',
                {**base, 'x': arg['x'], 'y': arg['y'], 'duration': arg.get('duration', 1000)},
            )
            return {'success': result.get('success', False)}

        if executor_name == 'gui_drag':
            result = await self._call(
                'swipe',
                {
                    **base,
                    'start_x': arg['x1'],
                    'start_y': arg['y1'],
                    'end_x': arg['x2'],
                    'end_y': arg['y2'],
                    'duration': arg.get('duration', 300),
                },
            )
            return {'success': result.get('success', False)}

        if executor_name == 'gui_type':
            result = await self._call('input_text', {**base, 'text': arg['text']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_press_key':
            key = arg.get('key', '')
            if key == 'Home':
                result = await self._call('press_home', base)
            elif key == 'Back':
                result = await self._call('press_back', base)
            else:
                return {'success': False, 'error': f'unsupported key: {key}'}
            return {'success': result.get('success', False)}

        if executor_name == 'gui_launch_app':
            result = await self._call('launch_app', {**base, 'package_name': arg['app']})
            return {'success': result.get('success', False)}

        if executor_name == 'gui_close_app':
            result = await self._call('stop_app', {**base, 'package_name': arg['app']})
            return {'success': result.get('success', False)}

        return {'success': False, 'error': f'unsupported executor: {executor_name}'}

    async def _gui_recording(self, executor_name: str, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.gui import GuiStartRecordingRequest, GuiStopRecordingRequest

        request_type = GuiStartRecordingRequest if executor_name == 'gui_start_recording' else GuiStopRecordingRequest
        request = request_type.model_validate(arg)
        source = files('nexus.runtime.utils').joinpath('_recording_worker.py').read_bytes()
        encoded = base64.b64encode(source).decode('ascii')
        payload = json.dumps({
            'action': 'start' if executor_name == 'gui_start_recording' else 'stop',
            'gui_session': True,
            **request.model_dump(),
        })
        code = (
            f'import base64;source=base64.b64decode({encoded!r}).decode();'
            f'ns={{"__name__":"nexus_recording"}};exec(source,ns);ns["main"]({payload!r},source)'
        )
        result = await self.run_command(command=shlex.join(['python3', '-c', code]))
        for line in (result.stdout or '').splitlines():
            if line.startswith('NEXUS_RECORDING_RESULT:'):
                return request_type.response_type.model_validate(json.loads(line.split(':', 1)[1])).model_dump()
        return request_type.response_type(success=False, error=result.stderr or 'Missing recording result').model_dump()

    async def _run_gui_pyautogui(self, executor_name: str, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models import gui
        from nexus.runtime.pyautogui_guard import validate_pyautogui_code

        r: Any
        timeout = None
        if executor_name == 'gui_click':
            r = gui.GuiClickRequest.model_validate(arg)
            code = f'pyautogui.click(x={r.x!r}, y={r.y!r}, button={r.button!r}, clicks={r.num_clicks})'
            if r.num_clicks > 1:
                code = _mac_multi_click(r.x, r.y, r.button, r.num_clicks)
        elif executor_name in ('gui_double_click', 'gui_right_click'):
            r = {'gui_double_click': gui.GuiDoubleClickRequest, 'gui_right_click': gui.GuiRightClickRequest}[
                executor_name
            ].model_validate(arg)
            method = 'doubleClick' if executor_name == 'gui_double_click' else 'rightClick'
            code = (
                _mac_multi_click(r.x, r.y, 'left', 2)
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
        data = await self._desktop_python(code, timeout)
        result = {'success': data.get('returncode') == 0, 'error': data.get('error') or None}
        if executor_name == 'gui_pyautogui':
            result.update(stdout=data.get('output', ''), stderr=data.get('error', ''))
        if executor_name == 'gui_get_device_info' and result['success']:
            line = next((x for x in data.get('output', '').splitlines() if x.startswith('NEXUS_GUI_INFO:')), None)
            if line is None:
                raise RuntimeError('Desktop returned no device information')
            result.update(json.loads(line.split(':', 1)[1]))
        return result

    async def _desktop_python(self, code: str, timeout: float | None) -> dict[str, Any]:
        script = 'import pyautogui, time, platform; pyautogui.FAILSAFE = False; ' + code
        result = await self._call(
            'shell',
            {
                'device_id': self.device_id,
                'session_id': self.session_id,
                'json_output': True,
                'command': shlex.join(['python3', '-c', script]),
                **({'timeout': timeout} if timeout is not None else {}),
            },
        )
        return {
            'returncode': result.get('return_code'),
            'output': result.get('stdout', ''),
            'error': result.get('stderr', ''),
        }

    async def _invoke(self, request: InvokeRequest) -> Any:
        raise NotImplementedError('SmartRunRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict, context: Any) -> Any:
        raise NotImplementedError('SmartRunRuntime does not support call_tool')


def _mac_multi_click(x: int | None, y: int | None, button: str, clicks: int) -> str:
    # PyAutoGUI's macOS backend emits separate single clicks without click-state counts.
    return (
        'import Quartz\n'
        f'pyautogui.moveTo({x!r}, {y!r})\n'
        'point = tuple(pyautogui.position())\n'
        f'button = {{"left": 0, "right": 1, "middle": 2}}[{button!r}]\n'
        'down, up = {0: (1, 2), 1: (3, 4), 2: (25, 26)}[button]\n'
        f'for count in range(1, {clicks + 1}):\n'
        ' for kind in (down, up):\n'
        '  event = Quartz.CGEventCreateMouseEvent(None, kind, point, button)\n'
        '  Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, count)\n'
        '  Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)\n'
        '  time.sleep(0.05)'
    )


def _normalize_gui_key(key: str) -> str:
    return {'cmd': 'command', 'ctrl': 'ctrl', 'option': 'option', 'return': 'enter', 'esc': 'esc'}.get(
        key.lower(), key.lower()
    )


__all__ = ['SmartRunRuntime']
