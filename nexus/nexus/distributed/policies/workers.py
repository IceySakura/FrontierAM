"""Supervised reusable processes; policy code never runs on the scheduler event loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

from typing import Any


MAX_MESSAGE_BYTES = 128 * 1024


class PolicyWorkerError(RuntimeError):
    pass


class PolicyWorker:
    def __init__(self, *, startup_seconds: float = 30) -> None:
        self.startup_seconds = startup_seconds
        self.process: asyncio.subprocess.Process | None = None

    async def run(self, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        encoded = json.dumps(request, allow_nan=False).encode() + b'\n'
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise PolicyWorkerError('policy_input_too_large')
        try:
            if self.process is None:
                await asyncio.wait_for(self._start(), timeout=self.startup_seconds)
            return await asyncio.wait_for(self._exchange(encoded), timeout=timeout)
        except asyncio.TimeoutError:
            await self.close()
            raise TimeoutError('policy_timeout') from None
        except BaseException:
            await self.close()
            raise

    async def _start(self) -> None:
        if self.process is None:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                '-m',
                'nexus.distributed.policies.worker',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=MAX_MESSAGE_BYTES + 1,
                # Bound native thread pools before SDK imports to avoid multiplying threads across workers.
                env={
                    **os.environ,
                    'OPENBLAS_NUM_THREADS': '1',
                    'OMP_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1',
                    'BLIS_NUM_THREADS': '1',
                    'VECLIB_MAXIMUM_THREADS': '1',
                },
            )
        assert self.process is not None
        assert self.process.stdout is not None
        ready = await self.process.stdout.readline()
        if ready != b'{"ready":true}\n':
            raise PolicyWorkerError('policy_worker_startup_failed')

    async def _exchange(self, encoded: bytes) -> dict[str, Any]:
        process = self.process
        assert process is not None
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(encoded)
        await process.stdin.drain()
        line = await process.stdout.readline()
        if not line or len(line) > MAX_MESSAGE_BYTES:
            raise PolicyWorkerError('policy_worker_protocol_error')
        response = json.loads(line)
        if not isinstance(response, dict) or not response.get('ok'):
            raise PolicyWorkerError('policy_evaluation_failed')
        return response['result']

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()


class PolicyWorkerPool:
    def __init__(self, *, size: int, startup_seconds: float = 30) -> None:
        self.workers = [PolicyWorker(startup_seconds=startup_seconds) for _ in range(size)]
        self.available: asyncio.Queue[PolicyWorker] = asyncio.Queue()
        for worker in self.workers:
            self.available.put_nowait(worker)

    async def run(self, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        worker = await self.available.get()
        try:
            return await worker.run(request, timeout=timeout)
        finally:
            self.available.put_nowait(worker)

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers))
