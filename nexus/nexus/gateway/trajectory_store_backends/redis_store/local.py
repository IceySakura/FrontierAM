"""Start a local ``redis-server`` subprocess for tests / benchmarks.

Does not replace production Redis. Requires ``redis-server`` on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path


def _free_port(host: str = '127.0.0.1') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_ready(host: str, port: int, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.05)
    raise TimeoutError(f'redis-server did not become ready on {host}:{port}: {last_err}')


@dataclass
class LocalRedis:
    """Managed local Redis process (ephemeral port, no persistence)."""

    host: str = '127.0.0.1'
    port: int | None = None
    redis_server_bin: str | None = None
    start_timeout_s: float = 10.0

    _proc: subprocess.Popen[bytes] | None = None
    _tmpdir: tempfile.TemporaryDirectory[str] | None = None
    _bound_port: int | None = None

    @property
    def url(self) -> str:
        if self._bound_port is None:
            raise RuntimeError('LocalRedis is not started')
        return f'redis://{self.host}:{self._bound_port}/0'

    def start(self) -> str:
        if self._proc is not None:
            return self.url

        binary = self.redis_server_bin or os.environ.get('NEXUS_REDIS_SERVER') or shutil.which('redis-server')
        if not binary:
            raise FileNotFoundError(
                'redis-server not found on PATH. Install Redis or pass --redis-url to an existing instance.'
            )

        port = self.port if self.port is not None else _free_port(self.host)
        self._tmpdir = tempfile.TemporaryDirectory(prefix='nexus-redis-')
        workdir = Path(self._tmpdir.name)
        cmd = [
            binary,
            '--bind',
            self.host,
            '--port',
            str(port),
            '--dir',
            str(workdir),
            '--dbfilename',
            'dump.rdb',
            '--save',
            '',
            '--appendonly',
            'no',
            '--protected-mode',
            'no',
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=str(workdir),
        )
        try:
            _wait_ready(self.host, port, timeout_s=self.start_timeout_s)
        except Exception:
            self.stop()
            raise
        self._bound_port = port
        return self.url

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self._bound_port = None
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:  # noqa: SIM105 - Preserve the original best-effort process cleanup.
                        proc.kill()
                    except Exception:
                        pass
                    try:  # noqa: SIM105 - Preserve the original best-effort process cleanup.
                        proc.wait(timeout=2)
                    except Exception:
                        pass
            # Drain stderr so we don't leak fds; ignore content unless debugging.
            try:
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass
        if self._tmpdir is not None:
            try:  # noqa: SIM105 - Preserve the original best-effort process cleanup.
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None

    def __enter__(self) -> LocalRedis:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_local_redis(**kwargs: object) -> LocalRedis:
    """Start a local Redis and return the running ``LocalRedis`` handle."""
    server = LocalRedis(**kwargs)  # type: ignore[arg-type]
    server.start()
    return server


def redis_server_available(redis_server_bin: str | None = None) -> bool:
    return bool(redis_server_bin or shutil.which('redis-server') or os.environ.get('NEXUS_REDIS_SERVER'))
