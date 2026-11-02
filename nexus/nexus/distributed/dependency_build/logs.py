from __future__ import annotations

import hashlib
import hmac
import logging
import os
import queue
import threading

from contextlib import nullcontext, suppress
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

from aiohttp import web

from nexus.distributed.observability.logging import async_tail_log_file
from nexus.utils.log.formatters import JsonLineFormatter


BUILD_LOG_QUEUE_SIZE = 4096
BUILD_LOG_DRAIN_SECONDS = 2
BUILD_LOG_PORT = 8093
BUILD_LOG_MAX_BYTES = 8 * 1024 * 1024
BUILD_LOG_TAIL_BYTES = 256 * 1024
output_logger = logging.getLogger('nexus.dependency_build.output')
output_logger.propagate = False


def build_log_token(key: str, build_id: str, attempt: int) -> str:
    message = f'nexus:build-log:read:{build_id}:{attempt}'.encode()
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def _write_stdout(text: str) -> None:
    # Avoid holding Python stdout's buffered lock in a possibly stuck daemon thread at exit.
    data = text.encode('utf-8', errors='replace')
    while data:
        data = data[os.write(1, data) :]


class _ListenerOnlyLock(nullcontext):
    # Support both acquire/release and context-manager logging APIs without
    # making interpreter shutdown wait for the single listener's I/O.
    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass


class _BuildLogFile(RotatingFileHandler):
    # Only the listener performs file I/O. Interpreter shutdown must neither wait
    # for its lock nor flush a stream that may be stuck in an OS write.
    _writer_id: int | None = None

    def createLock(self) -> None:  # noqa: N802 - stdlib override
        self.lock = _ListenerOnlyLock()  # ty: ignore[invalid-assignment]

    def _open(self):
        self._writer_id = threading.get_ident()
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()

    def flush(self) -> None:
        if threading.get_ident() == self._writer_id:
            super().flush()

    def close(self) -> None:
        logging.Handler.close(self)

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        self.setLevel(logging.CRITICAL + 1)

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802
        if self.stream is None:
            self.stream = self._open()
        return self.stream.tell() + len(f'{self.format(record)}\n'.encode()) >= self.maxBytes


class _ConsoleStream:
    def write(self, text: str) -> None:
        with suppress(OSError):
            _write_stdout(text)

    def flush(self) -> None:
        pass


class _BuildLogListener(QueueListener):
    _sentinel = None
    dropped = 0

    def dequeue(self, block):
        record = super().dequeue(block)
        if record is self._sentinel:
            if self.dropped:
                self.handle(
                    logging.makeLogRecord({
                        'msg': f'[build] {self.dropped} log records dropped because the log writer could not keep up',
                        'levelno': logging.WARNING,
                        'levelname': 'WARNING',
                        '_build_console': True,
                    })
                )
            logging.FileHandler.close(self.handlers[0])
        return record

    def stop(self) -> None:
        self.enqueue_sentinel()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=BUILD_LOG_DRAIN_SECONDS)


class _BuildLogQueueHandler(QueueHandler):
    listener: _BuildLogListener

    def __init__(self, path: Path, formatter: logging.Formatter) -> None:
        # One reserved slot lets shutdown enqueue its sentinel without dropping data.
        self._queue: queue.Queue[logging.LogRecord | None] = queue.Queue(maxsize=BUILD_LOG_QUEUE_SIZE + 1)
        super().__init__(self._queue)
        self._closing = False
        file = _BuildLogFile(path, maxBytes=BUILD_LOG_MAX_BYTES // 2, backupCount=1, encoding='utf-8', delay=True)
        file.setFormatter(formatter)
        console = logging.StreamHandler(_ConsoleStream())
        console.lock = _ListenerOnlyLock()  # ty: ignore[invalid-assignment]
        console.addFilter(lambda record: getattr(record, '_build_console', False))
        self.listener = _BuildLogListener(self.queue, file, console, respect_handler_level=True)
        self.listener.start()

    @property
    def dropped(self) -> int:
        return self.listener.dropped

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record = super().prepare(record)
        record.msg = record.message = record.getMessage()[:8192]
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        if self._closing:
            return
        if self._queue.qsize() >= BUILD_LOG_QUEUE_SIZE:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.listener.dropped += 1
            except queue.Empty:
                pass
        self._queue.put_nowait(record)

    def close(self) -> None:
        self.acquire()
        try:
            if self._closing:
                return
            self._closing = True
        finally:
            self.release()
        self.listener.stop()
        super().close()


def configure_build_log(path: Path, *, build_id: str, attempt: str) -> _BuildLogQueueHandler:
    handler = _BuildLogQueueHandler(
        path,
        JsonLineFormatter(
            component='dependency_builder',
            identity=build_id,
            context_provider=lambda: {'dependency_build_id': build_id, 'build_attempt': attempt},
            top_level_extra_fields=('dependency_build_id', 'build_attempt'),
        ),
    )
    output_logger.setLevel(logging.INFO)
    output_logger.addHandler(handler)
    return handler


def create_build_log_app(path: Path, *, token: str) -> web.Application:
    if not token:
        raise ValueError('build log read token is required')

    async def read_logs(request: web.Request) -> web.Response:
        supplied = request.headers.get('Authorization', '')
        if not hmac.compare_digest(supplied, f'Bearer {token}'):
            raise web.HTTPUnauthorized(text='build_log_auth_required')
        try:
            lines = min(1000, max(1, int(request.query.get('lines', '200'))))
            text = await async_tail_log_file(path, lines=lines, max_bytes=BUILD_LOG_TAIL_BYTES)
        except ValueError as exc:
            raise web.HTTPBadRequest(text='invalid lines') from exc
        except FileNotFoundError as exc:
            raise web.HTTPNotFound(text='log_not_found') from exc
        return web.Response(text=text, content_type='text/plain', headers={'Cache-Control': 'no-store'})

    app = web.Application()
    app.router.add_get('/v1/internal/logs', read_logs)
    return app
