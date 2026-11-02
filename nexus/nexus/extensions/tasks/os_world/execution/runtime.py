"""Task-owned AGS OSWorld capabilities on the standard provider runtime."""

from __future__ import annotations

import asyncio
import base64
import logging
import posixpath
import shlex

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import aiohttp

from nexus.runtime.providers.ags.base import _BaseAGSRuntime
from nexus.runtime.providers.ags.cdp_relay import CDPRelay
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime as BaseAGSOSWorldRuntime
from nexus.utils.async_http import HTTPResponse
from nexus.utils.async_http import http_client_with_trust_env as http_client
from nexus.utils.host_network import websocket_proxy


logger = logging.getLogger(__name__)

_READ_ONLY_RETRY_ATTEMPTS = 5
_READ_ONLY_RETRY_BASE_DELAY_SECONDS = 0.5
_SCREENSHOT_ATTEMPT_TIMEOUTS = (30.0, 60.0, 120.0)
_READ_ONLY_RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_READ_ONLY_RETRY_EXCEPTIONS = (
    aiohttp.ClientPayloadError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
    aiohttp.ClientConnectionError,
    asyncio.TimeoutError,
)
_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
_JPEG_MAGIC = b'\xff\xd8\xff'
_CDP_PROXY_MODES = {'auto', 'image', 'off'}
_FILE_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120)


class _OSWorldReadOnlyRuntimeMixin:
    handle: AGSInstanceHandle
    server_port: int

    async def _http_request(self, method: str, url: str, **kwargs: Any) -> HTTPResponse:
        headers = self.handle.headers
        extra_headers = kwargs.pop('headers', None)
        if extra_headers:
            headers.update({str(key): str(value) for key, value in extra_headers.items()})
        # Preserve the standard provider HTTP session and authentication.
        return await cast(_BaseAGSRuntime, super())._http_request(method, url, headers=headers, **kwargs)

    async def read_only_request(
        self,
        method: str,
        url: str,
        *,
        retry_label: str,
        response_validator: Callable[[HTTPResponse], bool] | None = None,
        attempt_timeouts: tuple[float, ...] | None = None,
        **kwargs: Any,
    ) -> HTTPResponse:
        """Retry the idempotent OSWorld read endpoints: screenshot, screen size.

        A method rather than a module helper so a runtime with a different transport
        policy supplies it by overriding, the same direction the native bridge already
        asks a runtime for its data proxy. Core therefore names no subclass.
        """

        attempts = len(attempt_timeouts) if attempt_timeouts is not None else _READ_ONLY_RETRY_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                request_kwargs = dict(kwargs)
                headers = {str(key): str(value) for key, value in (request_kwargs.pop('headers', None) or {}).items()}
                headers.setdefault('Connection', 'close')
                if attempt_timeouts is None:
                    response = await self._http_request(method, url, headers=headers, **request_kwargs)
                else:
                    timeout = attempt_timeouts[attempt - 1]
                    request_kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout)
                    response = await asyncio.wait_for(
                        self._http_request(method, url, headers=headers, **request_kwargs),
                        timeout=timeout,
                    )
            except _READ_ONLY_RETRY_EXCEPTIONS as exc:
                if attempt >= attempts:
                    raise
                logger.warning(
                    'AGS OSWorld read-only %s failed with %s; retrying (%d/%d)',
                    retry_label,
                    type(exc).__name__,
                    attempt,
                    attempts,
                )
            else:
                should_retry_status = response.status_code in _READ_ONLY_RETRY_STATUS_CODES
                should_retry_payload = (
                    response.status_code == 200 and response_validator is not None and not response_validator(response)
                )
                if not should_retry_status and not should_retry_payload:
                    return response
                if attempt >= attempts:
                    return response
                if should_retry_status:
                    logger.warning(
                        'AGS OSWorld read-only %s returned HTTP %d; retrying (%d/%d)',
                        retry_label,
                        response.status_code,
                        attempt,
                        attempts,
                    )
                else:
                    logger.warning(
                        'AGS OSWorld read-only %s returned invalid payload; retrying (%d/%d)',
                        retry_label,
                        attempt,
                        attempts,
                    )

            await asyncio.sleep(_READ_ONLY_RETRY_BASE_DELAY_SECONDS * attempt)

        raise RuntimeError(f'AGS OSWorld read-only {retry_label} failed after {attempts} attempts')

    async def read_only_post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        retry_label: str,
    ) -> tuple[Any, str, int]:
        response = await self.read_only_request('POST', url, retry_label=retry_label, json=payload)
        try:
            body = response.json()
        except Exception:
            body = None
        return body, response.text, response.status_code


def _header_value(headers: dict[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return ''


def _screenshot_format(response: HTTPResponse) -> str | None:
    content = response.content
    if not content:
        return None
    if content.startswith(_PNG_MAGIC):
        return 'png'
    if content.startswith(_JPEG_MAGIC):
        return 'jpeg'

    content_type = _header_value(response.headers, 'content-type').lower()
    if 'image/png' in content_type:
        return 'png'
    if 'image/jpeg' in content_type or 'image/jpg' in content_type:
        return 'jpeg'
    return None


def _is_screenshot_response(response: HTTPResponse) -> bool:
    return _screenshot_format(response) is not None


def _normalize_cdp_proxy_mode(value: str | None) -> str:
    mode = (value or 'auto').strip().lower()
    if mode not in _CDP_PROXY_MODES:
        raise ValueError(f'Unsupported AGS OSWorld cdp_proxy_mode: {value!r}')
    return mode


def _build_osworld_execute_payload(
    *,
    command: str,
    cwd: str | None,
    timeout: float,
    stdin: str | None,
    extra_env: dict[str, str] | None,
) -> dict[str, Any]:
    return BaseAGSOSWorldRuntime._build_osworld_execute_payload(
        command=command, cwd=cwd, timeout=timeout, stdin=stdin, extra_env=extra_env
    )


async def _run_osworld_bash_for_cdp_setup(
    runtime: Any,
    script: str,
    *,
    timeout: int = 60,
    stdin: str | None = None,
) -> dict[str, Any]:
    if stdin is None and hasattr(runtime, 'run_bash_script'):
        result = await runtime.run_bash_script(script, timeout=timeout)
        return result if isinstance(result, dict) else {'returncode': 0, 'output': str(result), 'error': ''}

    payload = _build_osworld_execute_payload(
        command=script,
        cwd=None,
        timeout=timeout,
        stdin=stdin,
        extra_env=None,
    )
    if hasattr(runtime, '_post_osworld_json'):
        body, text, status_code = await runtime._post_osworld_json('/execute', payload)
    else:
        response = await runtime.request('POST', '/execute', port=runtime.server_port, json=payload)
        try:
            body = response.json()
        except Exception:
            body = None
        text = response.text
        status_code = response.status_code
    if status_code >= 400:
        return {'returncode': -1, 'output': '', 'error': text[:500]}
    return body if isinstance(body, dict) else {'returncode': -1, 'output': '', 'error': text[:500]}


def _osworld_command_returncode(result: dict[str, Any]) -> int | None:
    value = result.get('returncode', result.get('return_code'))
    return int(value) if value is not None else None


def _osworld_command_text(result: dict[str, Any]) -> str:
    return '\n'.join(
        str(result.get(key) or '') for key in ('output', 'stdout', 'error', 'stderr') if result.get(key) is not None
    )


def _sudo_cdp_handoff_command(script: str) -> str:
    return f"sudo -S -p '' bash -lc {shlex.quote(script)}"


async def _run_cdp_handoff_script(
    runtime: Any,
    script: str,
    *,
    client_password: str,
    marker: str,
) -> None:
    result = await _run_osworld_bash_for_cdp_setup(
        runtime,
        _sudo_cdp_handoff_command(script),
        timeout=60,
        stdin=f'{client_password}\n',
    )
    result_text = _osworld_command_text(result)
    if _osworld_command_returncode(result) != 0 or marker not in result_text:
        raise RuntimeError(f'AGS OSWorld CDP handoff failed before {marker}: {result_text[:500]}')


async def _suspend_image_cdp_proxy(runtime: Any, *, client_password: str) -> None:
    script = r"""
set -eu

contract_mismatch() {
  echo "cdp-handoff-contract-mismatch:$1" >&2
  exit 1
}

port_9222_is_unbound() {
  python3 - <<'PY'
import sys

for path in ('/proc/net/tcp', '/proc/net/tcp6'):
    try:
        lines = open(path, encoding='utf-8').read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[3] == '0A' and fields[1].rsplit(':', 1)[-1].upper() == '2406':
            sys.exit(1)
PY
}

active_state="$(systemctl show cdp_proxy.service --property=ActiveState --value)"
sub_state="$(systemctl show cdp_proxy.service --property=SubState --value)"
fragment_path="$(systemctl show cdp_proxy.service --property=FragmentPath --value)"
exec_start="$(systemctl show cdp_proxy.service --property=ExecStart --value)"
restart_policy="$(systemctl show cdp_proxy.service --property=Restart --value)"

[ "$active_state" = active ] || contract_mismatch "ActiveState=$active_state"
[ "$sub_state" = running ] || contract_mismatch "SubState=$sub_state"
[ "$fragment_path" = /etc/systemd/system/cdp_proxy.service ] || contract_mismatch "FragmentPath=$fragment_path"
case "$exec_start" in
  *"/usr/local/bin/osworld-cdp-proxy"*) ;;
  *) contract_mismatch "ExecStart binary" ;;
esac
printf '%s\n' "$exec_start" | grep -Eq 'listen[^;]*9222([^0-9]|$)' || contract_mismatch "ExecStart listen"
printf '%s\n' "$exec_start" | grep -Eq 'upstream[^;]*1337([^0-9]|$)' || contract_mismatch "ExecStart upstream"
[ "$restart_policy" = always ] || contract_mismatch "Restart=$restart_policy"

service_stopped=0
restore_after_failure() {
  status=$?
  if [ "$service_stopped" = 1 ]; then
    systemctl start cdp_proxy.service >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap restore_after_failure EXIT

systemctl stop cdp_proxy.service
service_stopped=1
for _ in $(seq 1 50); do
  if port_9222_is_unbound; then
    trap - EXIT
    echo cdp-handoff-suspended
    exit 0
  fi
  sleep 0.1
done
echo cdp-handoff-port-9222-still-bound >&2
exit 1
"""
    await _run_cdp_handoff_script(
        runtime,
        script,
        client_password=client_password,
        marker='cdp-handoff-suspended',
    )


async def _restore_image_cdp_proxy(runtime: Any, *, client_password: str) -> None:
    script = r"""
set -eu

port_9222_is_listening() {
  python3 - <<'PY'
import sys

for path in ('/proc/net/tcp', '/proc/net/tcp6'):
    try:
        lines = open(path, encoding='utf-8').read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[3] == '0A' and fields[1].rsplit(':', 1)[-1].upper() == '2406':
            sys.exit(0)
sys.exit(1)
PY
}

command -v fuser >/dev/null 2>&1 || {
  echo cdp-handoff-fuser-missing >&2
  exit 1
}
fuser -k 9222/tcp >/dev/null 2>&1 || true
for _ in $(seq 1 25); do
  if ! port_9222_is_listening; then
    break
  fi
  sleep 0.1
done
if port_9222_is_listening; then
  fuser -k -9 9222/tcp >/dev/null 2>&1 || true
fi
for _ in $(seq 1 25); do
  if ! port_9222_is_listening; then
    break
  fi
  sleep 0.1
done
if port_9222_is_listening; then
  echo cdp-handoff-task-listener-still-running >&2
  exit 1
fi

systemctl start cdp_proxy.service
for _ in $(seq 1 50); do
  active_state="$(systemctl show cdp_proxy.service --property=ActiveState --value)"
  sub_state="$(systemctl show cdp_proxy.service --property=SubState --value)"
  if [ "$active_state" = active ] && [ "$sub_state" = running ] && port_9222_is_listening; then
    echo cdp-handoff-restored
    exit 0
  fi
  sleep 0.1
done
echo cdp-handoff-restore-timeout >&2
systemctl show cdp_proxy.service --property=ActiveState,SubState --no-pager >&2 || true
exit 1
"""
    await _run_cdp_handoff_script(
        runtime,
        script,
        client_password=client_password,
        marker='cdp-handoff-restored',
    )


async def _restore_image_cdp_proxy_cancellation_safe(runtime: Any, *, client_password: str) -> None:
    restore_task = asyncio.create_task(_restore_image_cdp_proxy(runtime, client_password=client_password))
    try:
        await asyncio.shield(restore_task)
    except asyncio.CancelledError:
        try:
            await restore_task
        except Exception:
            logger.exception('AGS OSWorld CDP proxy restoration failed during cancellation')
        raise


@asynccontextmanager
async def _task_owned_chromium_cdp(runtime: Any, *, client_password: str) -> AsyncIterator[None]:
    mode = _normalize_cdp_proxy_mode(getattr(runtime, 'cdp_proxy_mode', 'auto'))
    if mode != 'image':
        raise RuntimeError(f'Task-owned Chromium CDP requires cdp_proxy_mode="image", got {mode!r}')

    suspend_task = asyncio.create_task(_suspend_image_cdp_proxy(runtime, client_password=client_password))
    try:
        await asyncio.shield(suspend_task)
    except asyncio.CancelledError:
        try:
            await suspend_task
        except Exception:
            logger.exception('AGS OSWorld CDP proxy suspension failed during cancellation')
        else:
            await _restore_image_cdp_proxy_cancellation_safe(runtime, client_password=client_password)
        raise

    try:
        yield
    finally:
        await _restore_image_cdp_proxy_cancellation_safe(runtime, client_password=client_password)


async def _get_ags_chromium_cdp_connection(
    runtime: Any,
    *,
    ensure_target: bool = True,
) -> tuple[str, dict[str, str]]:
    # Kept as a call-compatible argument; setup owns Chrome. Connecting must
    # never inject a proxy or create another browser/profile.
    del ensure_target
    return runtime.handle.url(runtime.chromium_port), runtime.handle.headers


class OSWorldTaskRuntime(_OSWorldReadOnlyRuntimeMixin, BaseAGSOSWorldRuntime):
    """Explicit task extension; no guest CDP proxy injection or browser replacement."""

    def __init__(self, handle, *, cdp_proxy_mode='image', tunnel_port=32000, sudo_password='password', **kwargs):
        super().__init__(handle, **kwargs)
        self.cdp_proxy_mode = _normalize_cdp_proxy_mode(cdp_proxy_mode)
        self.tunnel_port = tunnel_port
        self._sudo_password = sudo_password
        self._cdp_relay: CDPRelay | None = None

    @classmethod
    def from_runtime(cls, runtime: BaseAGSOSWorldRuntime, *, cdp_proxy_mode, tunnel_port, sudo_password):
        """Adopt acquired runtime state, without acquiring another sandbox or resetting it."""
        adapter = cls.__new__(cls)
        # Preserve runtime IDs, attached stores, port mappings, resource limits and
        # transport state. Calling the base constructor would reset these fields.
        adapter.__dict__.update(vars(runtime))
        adapter.cdp_proxy_mode = _normalize_cdp_proxy_mode(cdp_proxy_mode)
        adapter.tunnel_port = tunnel_port
        adapter._sudo_password = sudo_password
        adapter._cdp_relay = None
        return adapter

    async def initialize(self) -> None:
        """Prepare shared memory once at acquisition, before any harness/setup action."""
        if getattr(self, '_shm_initialized', False):
            return

        async def capacity() -> int:
            probe = await self.run_command(
                command='/usr/bin/python3 -I -c \'import os; s=os.statvfs("/dev/shm"); print(s.f_frsize*s.f_blocks)\'',
                timeout=10,
            )
            raw = (probe.stdout or '').strip()
            if probe.return_code != 0 or not raw.isdecimal():
                raise RuntimeError('OSWorld AGS setup: cannot determine /dev/shm capacity')
            return int(raw)

        before = await capacity()
        if before < 512 * 1024**2:
            result = await self.run_command(
                command="sudo -S -k -p '' /usr/bin/mount -o remount,size=512M /dev/shm",
                stdin=f'{self._sudo_password}\n',
                timeout=30,
            )
            if result.return_code != 0:
                raise RuntimeError(f'OSWorld AGS setup: /dev/shm remount to 512 MiB failed (rc={result.return_code})')
            after = await capacity()
            if after < 512 * 1024**2:
                raise RuntimeError(f'OSWorld AGS setup: /dev/shm remains below 512 MiB ({after} bytes)')
        else:
            after = before
        self._shm_initialized = True
        logger.info('OSWorld AGS setup: /dev/shm capacity %s -> %s bytes', before, after)

    async def prepare_setup_command(self, command: str | list) -> str | list:
        # The V1 setup installs a per-user .desktop file immediately before this
        # bare command. Update that user's cache, not root-owned system caches.
        if command not in ('update-desktop-database', ['update-desktop-database']):
            return command
        probe = await self.run_command(command='printf "%s" "$HOME"', timeout=10)
        home = (probe.stdout or '').strip()
        if probe.return_code != 0 or not home.startswith('/'):
            raise RuntimeError('Cannot resolve user desktop cache directory')
        return ['update-desktop-database', str(Path(home) / '.local/share/applications')]

    async def accessibility_tree(self) -> Any:
        body, text, status_code = await self._http_get(self.handle.url(self.server_port, '/accessibility'))
        if status_code >= 400:
            raise RuntimeError(f'OSWorld accessibility tree failed: HTTP {status_code} {text[:500]}')
        if isinstance(body, dict) and isinstance(body.get('AT'), str):
            return body['AT']
        return body if body is not None else text

    async def terminal_output(self) -> Any:
        """GET /terminal — the focused gnome-terminal's visible text, or None.

        The endpoint answers ``{'output': <str|None>, 'status': 'success'}`` and
        returns ``output=None`` unless exactly one focused terminal sits in an active frame,
        so callers must treat None as "could not read", not as empty output.
        """
        body, text, status_code = await self._http_get(self.handle.url(self.server_port, '/terminal'))
        if status_code >= 400:
            raise RuntimeError(f'OSWorld terminal output failed: HTTP {status_code} {text[:500]}')
        return body if body is not None else text

    async def get_chromium_cdp_connection(self, *, ensure_target: bool = True) -> tuple[str, dict[str, str]]:
        return await _get_ags_chromium_cdp_connection(self, ensure_target=ensure_target)

    async def chromium_cdp_url(self) -> str:
        chromium_url = self.handle.url(self.chromium_port)
        # Playwright's CDP WebSocket does not honor HTTP(S)_PROXY; route it through the runtime-owned relay.
        if not websocket_proxy(chromium_url):
            return chromium_url
        if self._cdp_relay is None:
            self._cdp_relay = CDPRelay(chromium_url, self.handle.headers)
        return await self._cdp_relay.start()

    async def close(self) -> None:
        if self._cdp_relay is not None:
            await self._cdp_relay.close()
        await super().close()

    @asynccontextmanager
    async def task_owned_chromium_cdp(self, *, client_password: str) -> AsyncIterator[None]:
        async with _task_owned_chromium_cdp(self, client_password=client_password):
            yield

    async def setup_upload_file(self, local_path: str | Path, remote_path: str) -> HTTPResponse:
        path = Path(local_path)
        return await self.setup_upload_bytes(
            remote_path,
            path.read_bytes(),
            filename=posixpath.basename(remote_path) or path.name or 'file',
        )

    async def setup_upload_bytes(
        self,
        remote_path: str,
        content: bytes,
        *,
        filename: str = 'file',
    ) -> HTTPResponse:
        form = aiohttp.FormData()
        form.add_field('file_path', remote_path)
        form.add_field(
            'file_data',
            BytesIO(content),
            filename=filename,
            content_type='application/octet-stream',
        )
        return await self._http_request(
            'POST',
            self.handle.url(self.server_port, '/setup/upload'),
            data=form,
        )

    async def setup_launch(self, command: list[str] | str, *, shell: bool = False) -> HTTPResponse:
        return await self._http_request(
            'POST',
            self.handle.url(self.server_port, '/setup/launch'),
            json={'command': command, 'shell': shell},
        )

    async def setup_activate_window(
        self,
        window_name: str,
        *,
        strict: bool = False,
        by_class: bool = False,
    ) -> HTTPResponse:
        return await self._http_request(
            'POST',
            self.handle.url(self.server_port, '/setup/activate_window'),
            json={'window_name': window_name, 'strict': strict, 'by_class': by_class},
        )

    async def setup_close_window(
        self,
        window_name: str,
        *,
        strict: bool = False,
        by_class: bool = False,
    ) -> HTTPResponse:
        return await self._http_request(
            'POST',
            self.handle.url(self.server_port, '/setup/close_window'),
            json={'window_name': window_name, 'strict': strict, 'by_class': by_class},
        )

    async def get_file(self, file_path: str) -> bytes:
        response = await http_client.post(
            self.handle.url(self.server_port, '/file'),
            data={'file_path': file_path},
            timeout=_FILE_REQUEST_TIMEOUT,
            headers=self.handle.headers,
        )
        response.raise_for_status()
        return response.content

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

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name == 'download_file':
            from nexus.runtime.executors.models.file import (
                DownloadFileRequest,
                DownloadFileResponse,
                FileItem,
                FileOperationResult,
            )

            request = DownloadFileRequest.model_validate(arg)
            files: list[FileItem] = []
            errors: list[FileOperationResult] = []
            for file_item in request.files:
                try:
                    content = await self.get_file(file_item.path)
                    if file_item.encoding == 'base64':
                        encoded = base64.b64encode(content).decode('utf-8')
                    else:
                        encoded = content.decode('utf-8')
                    files.append(FileItem(path=file_item.path, content=encoded, encoding=file_item.encoding))
                except Exception as exc:
                    errors.append(FileOperationResult(path=file_item.path, success=False, message=f'Error: {exc}'))

            return DownloadFileResponse(files=files, errors=errors or None).model_dump()

        if executor_name == 'upload_file':
            from nexus.runtime.executors.models.file import FileOperationResult, UploadFileRequest, UploadFileResponse

            request = UploadFileRequest.model_validate(arg)
            results: list[FileOperationResult] = []
            for file_item in request.files:
                if file_item.content is None:
                    results.append(
                        FileOperationResult(path=file_item.path, success=False, message='Content is required')
                    )
                    continue

                try:
                    if file_item.encoding == 'base64':
                        content = base64.b64decode(file_item.content)
                    else:
                        content = file_item.content.encode('utf-8')

                    form = aiohttp.FormData()
                    form.add_field('file_path', file_item.path)
                    form.add_field(
                        'file_data',
                        content,
                        filename=posixpath.basename(file_item.path) or 'file',
                        content_type='application/octet-stream',
                    )
                    response = await self._http_request(
                        'POST',
                        self.handle.url(self.server_port, '/setup/upload'),
                        data=form,
                    )
                    results.append(
                        FileOperationResult(
                            path=file_item.path,
                            success=response.status_code == 200,
                            message=response.text[:500]
                            if response.status_code != 200
                            else 'File uploaded successfully',
                        )
                    )
                except Exception as exc:
                    results.append(FileOperationResult(path=file_item.path, success=False, message=f'Error: {exc}'))

            return UploadFileResponse(results=results).model_dump()

        if executor_name == 'gui_screenshot':
            from nexus.runtime.executors.gui import GuiScreenshotResponse

            response = await self.read_only_request(
                'GET',
                self.handle.url(self.server_port, '/screenshot'),
                retry_label='OSWorld /screenshot',
                response_validator=_is_screenshot_response,
                attempt_timeouts=_SCREENSHOT_ATTEMPT_TIMEOUTS,
            )
            if response.status_code >= 400:
                return GuiScreenshotResponse(success=False, error=response.text[:500]).model_dump()
            image_format = _screenshot_format(response)
            if image_format is None:
                return GuiScreenshotResponse(success=False, error='Invalid OSWorld screenshot payload').model_dump()

            screen_size = await self.call_executor('gui_get_screen_size', {})
            return GuiScreenshotResponse(
                success=True,
                image=base64.b64encode(response.content).decode('utf-8'),
                width=screen_size.get('width'),
                height=screen_size.get('height'),
                format=image_format,
            ).model_dump()

        if executor_name == 'gui_get_screen_size':
            from nexus.runtime.executors.gui import GuiScreenSizeResponse

            body, text, status_code = await self.read_only_post_json(
                self.handle.url(self.server_port, '/screen_size'),
                {},
                retry_label='OSWorld /screen_size',
            )
            if status_code >= 400 or not isinstance(body, dict):
                return GuiScreenSizeResponse(success=False, error=text[:500]).model_dump()
            return GuiScreenSizeResponse(success=True, width=body.get('width'), height=body.get('height')).model_dump()

        if executor_name == 'gui_pyautogui':
            # Native computer-use agents (Anthropic / OpenAI Responses API)
            # emit batched pyautogui code as their native action space. Reuse
            # _run_python_controller_code, which already imports pyautogui + time
            # with FAILSAFE=False. user/home_dir are Linux runtime knobs; the
            # OSWorld image runs pyautogui as the same shell user the /execute
            # endpoint dispatches to, so we drop them.
            from nexus.runtime.executors.gui import GuiPyAutoGUIRequest, GuiPyAutoGUIResponse

            request = GuiPyAutoGUIRequest.model_validate(arg)
            try:
                result = await self._run_python_controller_code(request.code)
            except Exception as exc:  # noqa: BLE001
                return GuiPyAutoGUIResponse(success=False, error=str(exc)[:500]).model_dump()
            return_code = result.get('returncode')
            return GuiPyAutoGUIResponse(
                success=return_code == 0,
                stdout=result.get('output'),
                stderr=result.get('error'),
                error=None if return_code == 0 else (result.get('error') or f'exit {return_code}'),
            ).model_dump()

        return await super().call_executor(executor_name, arg)


async def get_osworld_runtime(ctx, name: str):
    """Install task capabilities in both owners; AGS retains release ownership."""
    runtime = await ctx.get_runtime(name)
    if isinstance(runtime, OSWorldTaskRuntime) or not isinstance(runtime, BaseAGSOSWorldRuntime):
        return runtime
    if type(runtime) is not BaseAGSOSWorldRuntime:
        raise TypeError('OSWorld task cannot replace an unknown AGS runtime subclass')
    # Concurrent task-side consumers must not initialize/publish two adapters.
    lock = getattr(runtime, '_osworld_adapter_lock', None)
    if lock is None:
        lock = runtime._osworld_adapter_lock = asyncio.Lock()
    async with lock:
        current = ctx._runtimes.get(name)
        if isinstance(current, OSWorldTaskRuntime):
            return current
        provider = ctx.runtime_provider
        if current is not runtime or provider is None or provider._active_runtimes.get(name) is not runtime:
            raise RuntimeError('OSWorld runtime owners changed before task adaptation')
        params = provider.get_params(name)
        params.update(ctx.runtime_configs.get(name, {}))
        try:
            adapter = OSWorldTaskRuntime.from_runtime(
                runtime,
                cdp_proxy_mode=str(
                    params.get('cdp_proxy_mode') or provider.config.get('osworld_cdp_proxy_mode') or 'auto'
                ),
                tunnel_port=int(params.get('tunnel_port') or 32000),
                sudo_password=params.get('sudo_password', 'password'),
            )
            await adapter.initialize()
        except BaseException:
            # Initialization failure still belongs to provisioning. Do not leak
            # a sandbox or replace the original failure with a cleanup exception.
            cleanup = asyncio.create_task(ctx.release_runtime(name, force=True))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                try:
                    await cleanup
                except BaseException:
                    logger.exception('OSWorld runtime initialization cleanup failed')
            except BaseException:
                logger.exception('OSWorld runtime initialization cleanup failed')
            raise
        if ctx._runtimes.get(name) is not runtime or provider._active_runtimes.get(name) is not runtime:
            raise RuntimeError('OSWorld runtime owners changed during task adaptation')
        # No await between publications. Subsequent agent/tool/context/provider
        # lookups see exactly one object, and release closes that same adapter.
        ctx._runtimes[name] = adapter
        provider._active_runtimes[name] = adapter
        # Context callbacks capture the runtime object. Rebind even though the
        # standard AGS event methods are no-ops today.
        for subscriptions in (ctx._runtime_shared_unsubs, ctx._task_files_runtime_unsubs):
            for unsubscribe in subscriptions.pop(id(runtime), []):
                unsubscribe()
        ctx._attach_runtime_shared(adapter)
        ctx._attach_task_files_runtime(adapter)
        return adapter
