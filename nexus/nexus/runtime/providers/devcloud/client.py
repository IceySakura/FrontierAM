"""DevCloud OSWorld runtime client."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import shlex

from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any

import aiohttp

from aiohttp import FormData

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.client.base import Runtime
from nexus.utils.async_http import http_client


class DevCloudRuntime(Runtime):
    """Runtime client for an OSWorld HTTP server running in a DevCloud macOS sandbox."""

    native_bridge_kind = NativeBridgeRuntimeKind.DEVCLOUD_MACOS

    def __init__(
        self,
        *,
        base_url: str,
        request_timeout: float = 1800.0,
        metadata: dict[str, Any] | None = None,
        proxy_url: str | None = None,
        execute_mode: str = 'sync',
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip('/')
        self.request_timeout = float(request_timeout)
        self.metadata = metadata or {}
        self._timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self.proxy_url = proxy_url
        if execute_mode not in {'sync', 'async'}:
            raise ValueError('execute_mode must be sync or async')
        if execute_mode == 'async' and (not math.isfinite(self.request_timeout) or self.request_timeout <= 0):
            raise ValueError('async execute requires a finite, positive request_timeout')
        self.execute_mode = execute_mode

    def _proxy_kwargs(self) -> dict[str, Any]:
        # Omit the option entirely on the original direct path; never alter global clients.
        return {'proxy': self.proxy_url} if self.proxy_url else {}

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def _post_json(
        self, path: str, payload: dict[str, Any], timeout: aiohttp.ClientTimeout | None = None
    ) -> tuple[Any, str, int]:
        if path == '/execute' and self.execute_mode == 'async':
            from nexus.runtime.providers.devcloud.async_execute import execute

            budget = self.request_timeout if timeout is None or timeout.total is None else timeout.total
            return await execute(self._base_url, payload, proxy_url=self.proxy_url, timeout=budget)
        resp = await http_client.post(
            f'{self._base_url}{path}',
            json=payload,
            timeout=self._timeout if timeout is None else timeout,
            **self._proxy_kwargs(),
        )
        try:
            body = resp.json()
        except Exception:
            body = None
        return body, resp.text, resp.status_code

    async def _execute_command_payload(self, payload: dict[str, Any], timeout: float | None) -> tuple[Any, str, int]:
        if self.execute_mode == 'async' and timeout is not None:
            # Bound result collection as well as the command, allowing bootstrap/termination overhead.
            wait_budget = min(self.request_timeout, max(float(timeout), 0.001) + 30.0)
            return await self._post_json('/execute', payload, timeout=aiohttp.ClientTimeout(total=wait_budget))
        return await self._post_json('/execute', payload)

    async def _run_python_controller_code(self, code: str) -> dict[str, Any]:
        payload = {
            'command': [
                'python3',
                '-c',
                f'import pyautogui; import time; import platform; pyautogui.FAILSAFE = False; {code}',
            ],
            'shell': False,
        }
        body, text, status_code = await self._post_json('/execute', payload)
        if status_code >= 400 or not isinstance(body, dict):
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        return body

    @staticmethod
    def _build_osworld_execute_payload(
        *,
        command: str,
        cwd: str | None,
        timeout: float | None,
        stdin: str | None,
        extra_env: dict[str, str] | None,
    ) -> dict[str, Any]:
        script_lines: list[str] = []
        for key, value in (extra_env or {}).items():
            script_lines.append(f'export {key}={shlex.quote(str(value))}')
        if cwd:
            script_lines.append(f'cd {shlex.quote(cwd)}')
        runner = command
        if stdin is not None:
            runner = '\n'.join(["cat <<'__NEXUS_STDIN__' | (", command, ')', stdin, '__NEXUS_STDIN__'])
        script_lines.append(runner)
        wrapped_script = '\n'.join(script_lines)
        shell_command = f'bash -lc {shlex.quote(wrapped_script)}'
        if timeout is not None:
            timeout_value = max(float(timeout), 0.001)
            shell_command = f'timeout --kill-after=5s {timeout_value:g}s {shell_command}'
        return {'command': ['bash', '-lc', shell_command], 'shell': False}

    async def _gui_recording(self, executor_name: str, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.gui import GuiStartRecordingRequest, GuiStopRecordingRequest

        request_type = GuiStartRecordingRequest if executor_name == 'gui_start_recording' else GuiStopRecordingRequest
        request = request_type.model_validate(arg)
        source = files('nexus.runtime.utils').joinpath('_recording_worker.py').read_bytes()
        encoded = base64.b64encode(source).decode('ascii')
        payload = json.dumps({
            'action': 'start' if executor_name == 'gui_start_recording' else 'stop',
            **request.model_dump(),
        })
        code = (
            f'import base64;source=base64.b64decode({encoded!r}).decode();'
            f'ns={{"__name__":"nexus_recording"}};exec(source,ns);ns["main"]({payload!r},source)'
        )
        command = ['bash', '-lc', shlex.join(['python3', '-c', code])]
        try:
            body, text, status = await self._post_json('/execute', {'command': command, 'shell': False})
            if isinstance(body, dict):
                for line in str(body.get('output') or '').splitlines():
                    if line.startswith('NEXUS_RECORDING_RESULT:'):
                        return request_type.response_type.model_validate(json.loads(line.split(':', 1)[1])).model_dump()
            detail = body.get('error') if isinstance(body, dict) else text
            raise RuntimeError(f'Recording command failed (HTTP {status}): {str(detail)[:2000]}')
        except Exception as exc:
            return request_type.response_type(success=False, error=str(exc)).model_dump()

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name in {'gui_start_recording', 'gui_stop_recording'}:
            return await self._gui_recording(executor_name, arg)
        if executor_name == 'run_command':
            return await self._run_command(arg)
        if executor_name in {
            'gui_click',
            'gui_double_click',
            'gui_right_click',
            'gui_long_click',
            'gui_drag',
            'gui_move_to',
            'gui_mouse_down',
            'gui_mouse_up',
            'gui_drag_to',
            'gui_scroll',
            'gui_type',
            'gui_press_key',
            'gui_key_down',
            'gui_key_up',
            'gui_hotkey',
            'gui_pyautogui',
            'gui_get_device_info',
            'gui_start',
        }:
            return await self._run_gui_pyautogui(executor_name, arg)
        if executor_name == 'gui_screenshot':
            return await self._gui_screenshot()
        if executor_name == 'gui_get_screen_size':
            return await self._gui_get_screen_size()
        if executor_name == 'gui_launch_app':
            return await self._setup_route('/setup/launch', {'command': ['open', arg['app']], 'shell': False})
        if executor_name == 'gui_close_app':
            app = PurePosixPath(arg['app']).stem
            return await self._setup_route('/setup/close_window', {'window_name': app})
        if executor_name == 'download_file':
            return await self._download_file(arg)
        if executor_name == 'upload_file':
            return await self._upload_file(arg)
        raise NotImplementedError(f'Unsupported DevCloud executor: {executor_name!r}')

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
        if status_code >= 400 and not isinstance(body, dict):
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

        resp = await http_client.get(f'{self._base_url}/screenshot', timeout=self._timeout, **self._proxy_kwargs())
        if resp.status_code >= 400:
            return GuiScreenshotResponse(success=False, error=resp.text[:500]).model_dump()
        screen = await self._gui_get_screen_size()
        return GuiScreenshotResponse(
            success=True,
            image=base64.b64encode(resp.content).decode('utf-8'),
            width=screen.get('width'),
            height=screen.get('height'),
            format='png',
        ).model_dump()

    async def _gui_get_screen_size(self) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiScreenSizeResponse

        body, text, status_code = await self._post_json('/screen_size', {})
        if status_code >= 400 or not isinstance(body, dict):
            return GuiScreenSizeResponse(success=False, error=text[:500]).model_dump()
        return GuiScreenSizeResponse(success=True, width=body.get('width'), height=body.get('height')).model_dump()

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
        if timeout is None:
            data = await self._run_python_controller_code(code)
        else:
            data = await asyncio.wait_for(self._run_python_controller_code(code), timeout=timeout)
        result = {'success': data.get('returncode') == 0, 'error': data.get('error') or None}
        if executor_name == 'gui_pyautogui':
            result.update(stdout=data.get('output', ''), stderr=data.get('error', ''))
        if executor_name == 'gui_get_device_info' and result['success']:
            line = next((x for x in data.get('output', '').splitlines() if x.startswith('NEXUS_GUI_INFO:')), None)
            if line is None:
                raise RuntimeError('Desktop returned no device information')
            result.update(json.loads(line.split(':', 1)[1]))
        return result

    async def _setup_route(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        from nexus.runtime.executors.gui import GuiCloseAppResponse, GuiLaunchAppResponse

        _, text, status_code = await self._post_json(path, payload)
        response_cls = GuiLaunchAppResponse if path.endswith('/setup/launch') else GuiCloseAppResponse
        return response_cls(success=status_code == 200, error=None if status_code == 200 else text[:500]).model_dump()

    async def _download_file(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.file import (
            DownloadFileRequest,
            DownloadFileResponse,
            FileItem,
            FileOperationResult,
        )

        request = DownloadFileRequest.model_validate(arg)
        files: list[FileItem] = []
        errors: list[FileOperationResult] = []
        for item in request.files:
            try:
                resp = await http_client.post(
                    f'{self._base_url}/file',
                    data={'file_path': item.path},
                    timeout=self._timeout,
                    **self._proxy_kwargs(),
                )
                if resp.status_code >= 400:
                    errors.append(FileOperationResult(path=item.path, success=False, message=resp.text[:500]))
                    continue
                if item.encoding == 'base64':
                    content = base64.b64encode(resp.content).decode('utf-8')
                else:
                    content = resp.content.decode('utf-8')
                files.append(FileItem(path=item.path, content=content, encoding=item.encoding))
            except Exception as exc:
                errors.append(FileOperationResult(path=item.path, success=False, message=f'Error: {exc}'))
        return DownloadFileResponse(files=files, errors=errors if errors else None).model_dump()

    async def _upload_file(self, arg: Any) -> dict[str, Any]:
        from nexus.runtime.executors.models.file import FileOperationResult, UploadFileRequest, UploadFileResponse

        request = UploadFileRequest.model_validate(arg)
        results: list[FileOperationResult] = []
        for item in request.files:
            if item.content is None:
                results.append(FileOperationResult(path=item.path, success=False, message='Content is required'))
                continue
            try:
                if item.encoding == 'base64':
                    file_bytes = base64.b64decode(item.content)
                else:
                    file_bytes = item.content.encode('utf-8')
                form = FormData()
                form.add_field('file_path', item.path)
                form.add_field('file_data', file_bytes, filename=item.path.rsplit('/', 1)[-1] or 'upload.bin')
                resp = await http_client.post(
                    f'{self._base_url}/setup/upload', data=form, timeout=self._timeout, **self._proxy_kwargs()
                )
                if resp.status_code >= 400:
                    results.append(FileOperationResult(path=item.path, success=False, message=resp.text[:500]))
                else:
                    results.append(FileOperationResult(path=item.path, success=True, message=resp.text))
            except (binascii.Error, ValueError) as exc:
                results.append(
                    FileOperationResult(path=item.path, success=False, message=f'Invalid base64 content: {exc}')
                )
            except Exception as exc:
                results.append(FileOperationResult(path=item.path, success=False, message=f'Error: {exc}'))
        return UploadFileResponse(results=results).model_dump()

    async def _invoke(self, request: Any) -> Any:
        raise NotImplementedError('DevCloudRuntime does not support invoke')

    async def call_tool(self, import_path: str, params: dict[str, Any], context: Any) -> Any:
        raise NotImplementedError('DevCloudRuntime does not support call_tool')


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


__all__ = ['DevCloudRuntime']
