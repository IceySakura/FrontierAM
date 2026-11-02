"""SmartRun macOS bridge lifecycle through HTTP port proxies."""

from __future__ import annotations

import shlex

from pathlib import Path
from typing import Any

import aiohttp

from nexus.runtime.bridge.adapters.base import BasePosixBridgeAdapter, BridgeConnectionInfo, BridgeLaunchOptions
from nexus.runtime.providers.smartrun.client import SmartRunRuntime
from nexus.utils.async_http import http_client_with_trust_env as http_client


# Keep bridge Python separate from the interpreter authorized for GUI automation.
_PREPARE_PYTHON = """
set -eu
[ "$(uname -m)" = arm64 ] || { echo 'SmartRun native bridge requires macOS arm64' >&2; exit 1; }
root="$HOME/.local/share/nexus-native-bridge"
if ! "$root/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
    python3 -m venv "$root/bootstrap"
    "$root/bootstrap/bin/pip" install --disable-pip-version-check uv
    UV_PYTHON_INSTALL_DIR="$root/python" "$root/bootstrap/bin/uv" venv --python 3.11 "$root/venv"
fi
"""

# This server only accepts the one artifact path passed on argv. It is reachable
# on loopback and through the short-lived authenticated SmartRun upload proxy.
_UPLOAD_SERVER = """
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import TCPServer
class UploadServer(HTTPServer):
    def server_bind(self):
        # macOS reverse lookup can stall for 35s; this loopback server needs no DNS.
        TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]
class Handler(BaseHTTPRequestHandler):
    def do_PUT(self):
        remaining = int(self.headers['Content-Length'])
        offset = int(self.headers['X-Upload-Offset'])
        path = Path(sys.argv[1])
        if offset and (not path.exists() or path.stat().st_size != offset):
            self.send_error(409)
            return
        with path.open('ab' if offset else 'wb') as out:
            while remaining:
                chunk = self.rfile.read(min(remaining, 1024 * 1024))
                if not chunk:
                    self.send_error(400)
                    return
                out.write(chunk)
                remaining -= len(chunk)
        self.send_response(204)
        self.end_headers()
UploadServer(('127.0.0.1', int(sys.argv[2])), Handler).serve_forever()
"""


def detached_shell_command(command: str, pid_file: str) -> str:
    """Start a shell without macOS nohup's controlling-console requirement."""
    code = (
        'import subprocess; from pathlib import Path; '
        f'p = subprocess.Popen(["bash", "-c", {command!r}], stdin=subprocess.DEVNULL, '
        'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True); '
        f'Path({pid_file!r}).write_text(str(p.pid)); print("LAUNCHED")'
    )
    return shlex.join(['python3', '-c', code])


class SmartRunMacOSBridgeAdapter(BasePosixBridgeAdapter):
    target_label = 'SmartRun macOS'
    _upload_port = 18774

    def __init__(self, runtime: SmartRunRuntime) -> None:
        super().__init__(runtime)
        self.runtime: SmartRunRuntime
        self._tunnel: dict[str, Any] | None = None

    async def _open_tunnel(self, port: int, ttl: int) -> dict[str, Any]:
        result = await self.runtime._call(
            'get_tunnel',
            {
                'device_id': self.runtime.device_id,
                'session_id': self.runtime.session_id,
                'remote_port': port,
                'protocol': 'http',
                'ttl': ttl,
                'json_output': True,
            },
        )
        if not result.get('success'):
            raise RuntimeError('SmartRun could not create the bridge port proxy')
        return result

    async def _close_tunnel(self, tunnel: dict[str, Any]) -> None:
        result = await self.runtime._call(
            'close_tunnel',
            {
                'device_id': self.runtime.device_id,
                'session_id': self.runtime.session_id,
                'token': tunnel['token'],
                'json_output': True,
            },
        )
        if not result.get('success'):
            raise RuntimeError('SmartRun could not close the bridge port proxy')

    async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
        prepared = await self.runtime.run_command(_PREPARE_PYTHON, timeout=600)
        if prepared.return_code != 0:
            raise RuntimeError(f'SmartRun bridge Python setup failed: {prepared.stderr or prepared.stdout}')
        pid_file = f'{self.bridge_home}.upload.pid'
        launcher = (
            'import subprocess; from pathlib import Path; '
            f'log = open({self.bridge_home + ".upload.log"!r}, "w"); '
            f'p = subprocess.Popen({["python3", "-u", "-c", _UPLOAD_SERVER, remote_artifact, str(self._upload_port)]!r}, '
            'stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); '
            f'Path({pid_file!r}).write_text(str(p.pid))\n'
            'import socket, time\n'
            'deadline = time.monotonic() + 30\n'
            'while time.monotonic() < deadline:\n'
            '    if p.poll() is not None:\n'
            '        raise RuntimeError("Upload server exited before becoming ready")\n'
            '    try:\n'
            f'        with socket.create_connection(("127.0.0.1", {self._upload_port}), timeout=0.5):\n'
            '            break\n'
            '    except OSError:\n'
            '        time.sleep(0.05)\n'
            'else:\n'
            '    raise RuntimeError("Upload server did not become ready within 30 seconds")\n'
        )
        tunnel = None
        try:
            started = await self.runtime.run_command(
                shlex.join(['python3', '-c', launcher]),
                timeout=40,
            )
            if started.return_code != 0:
                raise RuntimeError(f'SmartRun bridge upload server failed: {started.stderr}')
            tunnel = await self._open_tunnel(self._upload_port, ttl=600)
            with artifact_path.open('rb') as data:
                offset = 0
                # Keep each proxy request small; large streaming PUTs can stall
                # in the SmartRun HTTP-to-device transport.
                while chunk := data.read(512 * 1024):
                    response = await http_client.put(
                        tunnel['consumer_url'],
                        data=chunk,
                        headers={'X-Upload-Offset': str(offset)},
                        timeout=aiohttp.ClientTimeout(total=60),
                    )
                    if response.status != 204:
                        raise RuntimeError(f'SmartRun bridge artifact upload failed: HTTP {response.status}')
                    offset += len(chunk)
        finally:
            try:
                await self.runtime.run_command(
                    f'if [ -f {shlex.quote(pid_file)} ]; then '
                    f'kill "$(cat {shlex.quote(pid_file)})" 2>/dev/null || true; '
                    f'rm -f {shlex.quote(pid_file)}; fi',
                )
            finally:
                if tunnel is not None:
                    await self._close_tunnel(tunnel)

    async def launch_bridge(self, options: BridgeLaunchOptions | None = None):
        # TTL caps new connections, not established ones. Use the platform maximum
        # so reconnects remain possible for as much of the device lease as allowed.
        self._tunnel = await self._open_tunnel(self.bridge_port, ttl=7200)
        return await super().launch_bridge(options)

    def _connection_info(self) -> BridgeConnectionInfo:
        if self._tunnel is None:
            raise RuntimeError('SmartRun bridge port proxy has not been created')
        return BridgeConnectionInfo(base_url=self._tunnel['consumer_url'].rstrip('/'))

    async def stop_bridge(self) -> None:
        try:
            await super().stop_bridge()
        finally:
            if self._tunnel is not None:
                await self._close_tunnel(self._tunnel)
                self._tunnel = None
