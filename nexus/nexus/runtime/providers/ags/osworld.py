"""AGS OSWorld runtime implementations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex

from importlib.resources import files
from typing import Any

import aiohttp

from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind
from nexus.runtime.providers.ags.base import AGSRawRuntime, _BaseAGSRuntime
from nexus.runtime.providers.ags.cdp_relay import CDPRelay
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.utils.async_http import HTTPResponse
from nexus.utils.async_http import http_client_with_trust_env as http_client


logger = logging.getLogger(__name__)


class AGSOSWorldRuntime(_BaseAGSRuntime):
    """Limited runtime backed by an OSWorld-compatible server in AGS."""

    native_bridge_kind = NativeBridgeRuntimeKind.AGS_OSWORLD

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        server_port: int = 5000,
        chromium_port: int = 9222,
        vnc_port: int = 5910,
        vlc_port: int = 8080,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(handle, request_timeout=request_timeout, resource_spec=resource_spec)
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.vnc_port = vnc_port
        self.vlc_port = vlc_port

        self.port_mappings[server_port] = self.handle.url(server_port)
        self.port_mappings[chromium_port] = self.handle.url(chromium_port)
        self.port_mappings[vnc_port] = self.handle.url(vnc_port)
        self.port_mappings[vlc_port] = self.handle.url(vlc_port)

        logger.debug('AGSOSWorldRuntime initialized with vnc_url: %s', self.handle.url(self.vnc_port))

    @property
    def base_url(self) -> str | None:
        return self.handle.url(self.server_port)

    async def _post_osworld_json(
        self, path: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> tuple[Any, str, int]:
        url = self.handle.url(self.server_port, path)
        if timeout is None:
            return await self._http_post_json(url, payload)
        # Each attempt has both a transport timeout and a host wall-clock limit.
        # Retrying /execute is at-least-once: a timed-out command may already
        # have run in the guest. Do not extend this policy to other endpoints.
        attempts = 3 if path == '/execute' else 1
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self._http_request('POST', url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)),
                    timeout=timeout,
                )
                break
            except TimeoutError as exc:
                logger.warning(
                    'AGS OSWorld %s HTTP deadline exceeded: attempt=%d/%d timeout=%gs; '
                    'guest command completion is unknown; %s',
                    path,
                    attempt,
                    attempts,
                    timeout,
                    'retrying may repeat side effects' if attempt < attempts else 'retry budget exhausted',
                )
                if attempt == attempts:
                    raise TimeoutError(
                        f'AGS OSWorld {path} HTTP deadline exceeded after {timeout:g}s per attempt '
                        f'({attempts} attempts); guest command completion is unknown'
                    ) from exc
        try:
            body = response.json()
        except Exception:
            body = None
        return body, response.text, response.status_code

    async def _run_python_controller_code(self, code: str) -> dict[str, Any]:
        payload = {
            'command': [
                'python',
                '-c',
                f'import pyautogui\nimport time\npyautogui.FAILSAFE = False\n{code}',
            ],
            'shell': False,
        }
        body, text, status_code = await self._post_osworld_json('/execute', payload)
        if status_code >= 400:
            raise RuntimeError(f'OSWorld execute failed: HTTP {status_code} {text[:500]}')
        if not isinstance(body, dict):
            raise RuntimeError(f'OSWorld execute returned unexpected payload: {text[:500]}')
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
            runner = '\n'.join([
                f"cat <<'__NEXUS_STDIN__' | bash -lc {shlex.quote(command)}",
                stdin,
                '__NEXUS_STDIN__',
            ])
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
            body, text, status = await self._post_osworld_json('/execute', {'command': command, 'shell': False})
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
            host_timeout = max(float(request.timeout), 0.001) if request.timeout is not None else None
            if self.request_timeout > 0:
                host_timeout = (
                    min(host_timeout, self.request_timeout) if host_timeout is not None else self.request_timeout
                )
            body, text, status_code = await self._post_osworld_json('/execute', payload, timeout=host_timeout)
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
            response = RunCommandResponse(
                status=status,
                execution_time=asyncio.get_running_loop().time() - started,
                return_code=return_code,
                stdout=data.get('output'),
                stderr=data.get('error'),
            )
            return response.model_dump()

        if executor_name == 'gui_click':
            from nexus.runtime.executors.gui import GuiClickRequest, GuiClickResponse

            request = GuiClickRequest.model_validate(arg)
            result = await self._run_python_controller_code(f'pyautogui.click(x={request.x}, y={request.y})')
            return GuiClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_double_click':
            from nexus.runtime.executors.gui import GuiDoubleClickRequest, GuiDoubleClickResponse

            request = GuiDoubleClickRequest.model_validate(arg)
            result = await self._run_python_controller_code(f'pyautogui.doubleClick(x={request.x}, y={request.y})')
            return GuiDoubleClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_long_click':
            from nexus.runtime.executors.gui import GuiLongClickRequest, GuiLongClickResponse

            request = GuiLongClickRequest.model_validate(arg)
            code = (
                f'pyautogui.mouseDown(x={request.x}, y={request.y}); '
                f'time.sleep({request.duration / 1000.0}); '
                'pyautogui.mouseUp()'
            )
            result = await self._run_python_controller_code(code)
            return GuiLongClickResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_drag':
            from nexus.runtime.executors.gui import GuiDragRequest, GuiDragResponse

            request = GuiDragRequest.model_validate(arg)
            duration = max(request.duration / 1000.0, 0.0)
            code = (
                f'pyautogui.moveTo({request.x1}, {request.y1}); '
                f'pyautogui.dragTo({request.x2}, {request.y2}, duration={duration}, button="left", mouseDownUp=True)'
            )
            result = await self._run_python_controller_code(code)
            return GuiDragResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_type':
            from nexus.runtime.executors.gui import GuiTypeRequest, GuiTypeResponse

            request = GuiTypeRequest.model_validate(arg)
            result = await self._run_python_controller_code(f'pyautogui.typewrite({request.text!r})')
            return GuiTypeResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_press_key':
            from nexus.runtime.executors.gui import GuiPressKeyRequest, GuiPressKeyResponse

            request = GuiPressKeyRequest.model_validate(arg)
            result = await self._run_python_controller_code(f'pyautogui.press({request.key.lower()!r})')
            return GuiPressKeyResponse(success=result.get('returncode', 1) == 0).model_dump()

        if executor_name == 'gui_screenshot':
            from nexus.runtime.executors.gui import GuiScreenshotResponse

            response = await http_client.get(
                self.handle.url(self.server_port, '/screenshot'),
                timeout=self._timeout,
                headers=self.handle.headers,
            )
            if response.status_code >= 400:
                return GuiScreenshotResponse(success=False, error=response.text[:500]).model_dump()

            screen_size = await self.call_executor('gui_get_screen_size', {})
            return GuiScreenshotResponse(
                success=True,
                image=base64.b64encode(response.content).decode('utf-8'),
                width=screen_size.get('width'),
                height=screen_size.get('height'),
                format='png',
            ).model_dump()

        if executor_name == 'gui_get_screen_size':
            from nexus.runtime.executors.gui import GuiScreenSizeResponse

            body, text, status_code = await self._post_osworld_json('/screen_size', {})
            if status_code >= 400 or not isinstance(body, dict):
                return GuiScreenSizeResponse(success=False, error=text[:500]).model_dump()
            return GuiScreenSizeResponse(success=True, width=body.get('width'), height=body.get('height')).model_dump()

        if executor_name == 'gui_launch_app':
            from nexus.runtime.executors.gui import GuiLaunchAppRequest, GuiLaunchAppResponse

            request = GuiLaunchAppRequest.model_validate(arg)
            _, text, status_code = await self._post_osworld_json('/setup/open_file', {'path': request.app})
            return GuiLaunchAppResponse(
                success=status_code == 200, error=None if status_code == 200 else text[:500]
            ).model_dump()

        if executor_name == 'gui_close_app':
            from nexus.runtime.executors.gui import GuiCloseAppRequest, GuiCloseAppResponse

            request = GuiCloseAppRequest.model_validate(arg)
            _, text, status_code = await self._post_osworld_json('/setup/close_window', {'window_name': request.app})
            return GuiCloseAppResponse(
                success=status_code == 200, error=None if status_code == 200 else text[:500]
            ).model_dump()

        raise NotImplementedError(f'Unsupported AGS OSWorld executor: {executor_name!r}')


class AGSRawOSWorldRuntime(AGSRawRuntime):
    """Raw OSWorld-compatible runtime that preserves the original HTTP surface."""

    native_bridge_kind = NativeBridgeRuntimeKind.AGS_OSWORLD

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        server_port: int = 5000,
        chromium_port: int = 9222,
        vnc_port: int = 5910,
        vlc_port: int = 8080,
        enable_cdp_relay: bool = False,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            handle,
            kind='osworld',
            default_port=server_port,
            request_timeout=request_timeout,
            resource_spec=resource_spec,
        )
        self.server_port = server_port
        self.chromium_port = chromium_port
        self.vnc_port = vnc_port
        self.vlc_port = vlc_port
        self._cdp_relay = CDPRelay(self.chromium_url, self.handle.headers) if enable_cdp_relay else None

        self.port_mappings[server_port] = self.handle.url(server_port)
        self.port_mappings[chromium_port] = self.handle.url(chromium_port)
        self.port_mappings[vnc_port] = self.handle.url(vnc_port)
        self.port_mappings[vlc_port] = self.handle.url(vlc_port)

        logger.debug('AGSRawOSWorldRuntime initialized with vnc_url: %s', self.vnc_url)

    @property
    def server_url(self) -> str:
        return self.handle.url(self.server_port)

    @property
    def chromium_url(self) -> str:
        return self.handle.url(self.chromium_port)

    async def chromium_cdp_url(self) -> str:
        if self._cdp_relay is None:
            return self.chromium_url
        return await self._cdp_relay.start()

    async def close(self) -> None:
        if self._cdp_relay is not None:
            await self._cdp_relay.close()
        await super().close()

    @property
    def vnc_url(self) -> str:
        return self.handle.url(self.vnc_port)

    @property
    def vlc_url(self) -> str:
        return self.handle.url(self.vlc_port)

    async def screenshot(self) -> bytes:
        return await self.request_bytes('GET', '/screenshot', port=self.server_port)

    async def accessibility_tree(self) -> Any:
        return await self.request_json('GET', '/accessibility', port=self.server_port)

    async def terminal_output(self) -> Any:
        return await self.request_json('GET', '/terminal', port=self.server_port)

    async def get_file(self, file_path: str) -> bytes:
        return await self.request_bytes('POST', '/file', port=self.server_port, data={'file_path': file_path})

    async def setup_upload(self, file_path: str, file_data: bytes, *, filename: str = 'upload.bin') -> str:
        import aiohttp

        form = aiohttp.FormData()
        form.add_field('file_path', file_path)
        form.add_field('file_data', file_data, filename=filename, content_type='application/octet-stream')
        response = await self.request('POST', '/setup/upload', port=self.server_port, data=form)
        if response.status_code >= 400:
            raise RuntimeError(f'OSWorld setup_upload failed: HTTP {response.status_code} {response.text[:500]}')
        return response.text

    async def screen_size(self) -> Any:
        return await self.request_json('POST', '/screen_size', port=self.server_port, json={})

    async def execute_raw(self, payload: dict[str, Any]) -> Any:
        return await self.request_json('POST', '/execute', port=self.server_port, json=payload)

    async def execute(self, command: str | list[str], shell: bool = False) -> Any:
        return await self.execute_raw({'command': command, 'shell': shell})

    async def run_python(self, code: str) -> Any:
        return await self.request_json('POST', '/run_python', port=self.server_port, json={'code': code})

    async def run_bash_script(
        self,
        script: str,
        timeout: int = 30,
        working_dir: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {'script': script, 'timeout': timeout, 'working_dir': working_dir}
        return await self.request_json('POST', '/run_bash_script', port=self.server_port, json=payload)

    async def setup_open_file(self, path: str) -> HTTPResponse:
        # /setup/* 成功时返回空 body，用 request 返回原始响应（避免对空 body 调 .json() 报错）。
        return await self.request('POST', '/setup/open_file', port=self.server_port, json={'path': path})

    async def setup_close_window(self, window_name: str) -> HTTPResponse:
        return await self.request(
            'POST',
            '/setup/close_window',
            port=self.server_port,
            json={'window_name': window_name},
        )

    def to_nexus(self) -> AGSOSWorldRuntime:
        """Convert to AGSOSWorldRuntime (executor-based interface)."""
        return AGSOSWorldRuntime(
            self.handle,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vnc_port=self.vnc_port,
            vlc_port=self.vlc_port,
            request_timeout=self.request_timeout,
            resource_spec=self.resource_spec,
        )


AGSOSWorldExecutorRuntime = AGSOSWorldRuntime | AGSRawOSWorldRuntime


def require_ags_osworld_runtime(runtime: Any) -> AGSOSWorldRuntime:
    """Ensure we have an AGSOSWorldRuntime, converting raw OSWorld runtimes if needed."""
    if isinstance(runtime, AGSOSWorldRuntime):
        return runtime
    if isinstance(runtime, AGSRawOSWorldRuntime):
        return runtime.to_nexus()
    raise TypeError(f'Expected AGSOSWorldRuntime or AGSRawOSWorldRuntime, got {type(runtime).__name__}')
