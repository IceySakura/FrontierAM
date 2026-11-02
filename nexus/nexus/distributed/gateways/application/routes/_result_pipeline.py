"""Admission for result preparation and responses, owned by the HTTP event loop."""

from __future__ import annotations

import time

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

from aiohttp import web

from nexus.distributed.artifacts.admission import BoundedAdmission, TransferAdmissionBusyError
from nexus.distributed.artifacts.result_stream import iter_result_output
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import TemporaryUnavailableError
from nexus.utils.common import await_complete, run_in_executor_complete, to_thread_complete
from nexus.utils.platform_metrics import measure_operation, record


T = TypeVar('T')


class ResultEncoder:
    """Worker-owned iterator; the request retains ownership even when preparation is cancelled."""

    def __init__(self, path: Path, *, maximum: int, chunk_bytes: int) -> None:
        self.path = path
        self.maximum = maximum
        self.chunk_bytes = chunk_bytes
        self._chunks: Generator[bytes, None, None] | None = None
        self._first: bytes | None = None

    def prepare(self) -> None:
        self._chunks = iter_result_output(self.path, maximum=self.maximum, chunk_bytes=self.chunk_bytes)
        self._first = next(self._chunks, None)

    def next_chunk(self) -> bytes | None:
        assert self._chunks is not None
        if self._first is not None:
            first, self._first = self._first, None
            return first
        # StopIteration cannot cross an asyncio Future boundary.
        return next(self._chunks, None)

    def close(self) -> None:
        self._first = None
        if self._chunks is not None:
            self._chunks.close()
            self._chunks = None


class ResultPipeline:
    def __init__(self, config: DistributedConfig) -> None:
        self._config = config
        self.max_inflight = config.artifact_run_result_max_inflight
        self.max_waiters = config.artifact_run_result_max_waiters
        self.timeout = config.artifact_run_result_admission_timeout
        self._admission = BoundedAdmission(
            total=self.max_inflight,
            lanes={'result': self.max_inflight},
            max_waiters=self.max_waiters,
            timeout=self.timeout,
        )
        self._cpu_concurrency = config.artifact_run_result_decode_concurrency
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    @property
    def active(self) -> int:
        return self._admission.active_count

    @property
    def waiting(self) -> int:
        return self._admission.waiting_count

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        try:
            async with self._admission.reserve('result'):
                yield
        except TransferAdmissionBusyError as exc:
            error = 'result_queue_full' if str(exc) == 'queue_full' else 'result_queue_timeout'
            raise TemporaryUnavailableError(error) from exc

    async def transform(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        stage: str = 'result_transform',
        timings: dict[str, float] | None = None,
    ) -> T:
        if self._closed:
            raise RuntimeError('result pipeline is closed')
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._cpu_concurrency, thread_name_prefix='result-cpu')
        queued_at = time.perf_counter()

        def run() -> T:
            started = time.perf_counter()
            wait = started - queued_at
            record(
                'persistence_duration_distribution', wait, stage='result_cpu_wait', operation=stage, outcome='success'
            )
            try:
                with measure_operation(stage, family='persistence'):
                    return func(*args)
            finally:
                if timings is not None:
                    timings['result_cpu_wait'] = timings.get('result_cpu_wait', 0) + wait
                    timings[stage] = timings.get(stage, 0) + time.perf_counter() - started

        return await run_in_executor_complete(self._executor, run)

    @asynccontextmanager
    async def stream(self, path: Path, *, timings: dict[str, float]) -> AsyncIterator[AsyncIterator[bytes]]:
        config = self._config
        encoder = ResultEncoder(
            path,
            maximum=config.artifact_run_result_max_decoded_size_bytes,
            chunk_bytes=config.artifact_run_result_response_chunk_bytes,
        )

        async def chunks() -> AsyncGenerator[bytes, None]:
            while (
                chunk := await self.transform(encoder.next_chunk, stage='result_encode', timings=timings)
            ) is not None:
                yield chunk

        iterator = chunks()

        async def cleanup() -> None:
            try:
                await iterator.aclose()
            finally:
                await self.transform(encoder.close, stage='result_cleanup', timings=timings)

        try:
            await self.transform(encoder.prepare, stage='result_decode', timings=timings)
            yield iterator
        finally:
            await await_complete(cleanup())

    async def close(self) -> None:
        self._closed = True
        if self._executor is not None:
            await to_thread_complete(self._executor.shutdown, wait=True)
            self._executor = None


RESULT_PIPELINE_KEY = web.AppKey('result_pipeline', ResultPipeline)


async def result_pipeline_context(app: web.Application) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await app[RESULT_PIPELINE_KEY].close()
