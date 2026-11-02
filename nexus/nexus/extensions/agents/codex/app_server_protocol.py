"""Minimal line-delimited JSON-RPC transport for ``codex app-server``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .events import CodexEvent, CodexEventStream, app_server_message_to_event


logger = logging.getLogger(__name__)


_STREAM_READER_LIMIT = 10 * 1024 * 1024

AppServerRequestHandler = Callable[[dict[str, Any]], Awaitable[Any]]
NotificationHandler = Callable[[dict[str, Any]], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]


class AppServerRpcError(RuntimeError):
    """Error response returned by app-server."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.code = error.get('code')
        self.data = error.get('data')
        super().__init__(str(error.get('message') or error))


class AppServerProcessError(RuntimeError):
    """The app-server process exited or its protocol stream became unusable."""


@dataclass
class _PendingRequest:
    method: str
    future: asyncio.Future[Any]


@dataclass
class _Notification:
    method: str
    handlers: tuple[NotificationHandler, ...]
    params: dict[str, Any]


@dataclass
class _Failure:
    handlers: tuple[ErrorHandler, ...]
    error: Exception


class _CallbackDispatcher:
    """Own async subscriptions and one ordered consumer for a client's lifetime."""

    def __init__(self, disconnect: Callable[[Exception], None]) -> None:
        self.handlers: dict[str, list[NotificationHandler]] = {}
        self.error_handlers: list[ErrorHandler] = []
        self.error: Exception | None = None
        self.queue: asyncio.Queue[_Notification | _Failure] = asyncio.Queue()
        self._disconnect = disconnect
        self.task = asyncio.create_task(self._run(), name='codex-app-server-callbacks')

    def subscribe(self, method: str, handler: NotificationHandler) -> Callable[[], None]:
        handlers = self.handlers.setdefault(method, [])
        handlers.append(handler)

        def unregister() -> None:
            handlers.remove(handler)
            if not handlers:
                del self.handlers[method]

        return unregister

    def subscribe_error(self, handler: ErrorHandler) -> Callable[[], None]:
        self.error_handlers.append(handler)
        if self.error is not None:
            self.queue.put_nowait(_Failure((handler,), self.error))
        return lambda: self.error_handlers.remove(handler)

    def publish(self, method: str, params: dict[str, Any]) -> None:
        handlers = tuple(self.handlers.get(method, ()))
        if handlers:
            self.queue.put_nowait(_Notification(method, handlers, params))

    def fail(self, error: Exception) -> None:
        self.error = error
        self.queue.put_nowait(_Failure(tuple(self.error_handlers), error))

    async def _deliver_failure(self, failure: _Failure) -> None:
        for handler in failure.handlers:
            if handler not in self.error_handlers:
                continue
            try:
                await handler(failure.error)
            except Exception:
                logger.exception('Codex app-server error handler failed')

    async def _run(self) -> None:
        callbacks_failed = False
        while True:
            event = await self.queue.get()
            try:
                if isinstance(event, _Failure):
                    await self._deliver_failure(event)
                elif not callbacks_failed:
                    for handler in event.handlers:
                        if handler in self.handlers.get(event.method, ()):
                            await handler(event.params)
            except Exception as exc:
                # Do not let a queued completion overwrite a callback failure.
                callbacks_failed = True
                self._disconnect(AppServerProcessError(f'Codex app-server notification handler failed: {exc}'))
            finally:
                self.queue.task_done()

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.queue.empty():
            event = self.queue.get_nowait()
            try:
                if isinstance(event, _Failure):
                    await self._deliver_failure(event)
            finally:
                self.queue.task_done()


class AppServerClient:
    """Bidirectional JSON-RPC client over app-server's stdio JSONL transport."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.stderr_lines: list[str] = []
        self.event_stream = CodexEventStream(source='app_server')
        self._pending: dict[int, _PendingRequest] = {}
        self._request_handlers: dict[str, AppServerRequestHandler] = {}
        self._request_handler_tasks: set[asyncio.Task[None]] = set()
        self._callbacks = _CallbackDispatcher(self._disconnect)
        self._next_request_id = 1
        self._write_lock = asyncio.Lock()
        self._reader_tasks: list[asyncio.Task[None]] = []
        self._closed = False

    @classmethod
    async def start(
        cls,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
    ) -> AppServerClient:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_READER_LIMIT,
        )
        client = cls(process)
        client._reader_tasks = [
            asyncio.create_task(client._read_stdout(), name='codex-app-server-stdout'),
            asyncio.create_task(client._read_stderr(), name='codex-app-server-stderr'),
        ]
        return client

    async def request(self, method: str, params: dict[str, Any], *, timeout: float | None = None) -> Any:
        if self._callbacks.error is not None:
            raise self._callbacks.error
        request_id = self._next_request_id
        self._next_request_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _PendingRequest(method, future)
        try:
            await self._send({'id': request_id, 'method': method, 'params': params})
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({'method': method, 'params': params})

    def register_request_handler(self, method: str, handler: AppServerRequestHandler) -> None:
        """Register or replace the handler for one app-server initiated RPC method."""
        self._request_handlers[method] = handler

    def register_notification_handler(self, method: str, handler: NotificationHandler) -> Callable[[], None]:
        return self._callbacks.subscribe(method, handler)

    def register_error_handler(self, handler: ErrorHandler) -> Callable[[], None]:
        return self._callbacks.subscribe_error(handler)

    async def respond(
        self, request_id: Any, *, method: str, result: Any = None, error: dict[str, Any] | None = None
    ) -> None:
        response = {'id': request_id, **({'error': error} if error is not None else {'result': result})}
        await self._send(response, correlated_method=method)

    async def _send(self, message: dict[str, Any], *, correlated_method: str | None = None) -> None:
        if self._closed or self.process.stdin is None:
            raise AppServerProcessError('Codex app-server stdin is closed')
        payload = (json.dumps(message, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        async with self._write_lock:
            self.process.stdin.write(payload)
            request_id = message.get('id')
            request_methods = (
                {request_id: correlated_method}
                if correlated_method is not None and isinstance(request_id, (int, str))
                else {}
            )
            self.event_stream.events.append(
                app_server_message_to_event(
                    message,
                    seq=len(self.event_stream.events),
                    timestamp=time.time(),
                    direction='outbound',
                    request_methods=request_methods,
                )
            )
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            while line := await self.process.stdout.readline():
                text = line.decode(errors='replace')
                try:
                    message = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.warning('Ignoring invalid Codex app-server JSONL message: %s', e)
                    self._append_event(
                        'protocol.invalid_message',
                        {'raw': text.rstrip(), 'error': str(e)},
                        direction='inbound',
                    )
                    continue
                if not isinstance(message, dict):
                    logger.warning('Ignoring non-object Codex app-server message: %r', message)
                    self._append_event('protocol.invalid_message', {'raw': message}, direction='inbound')
                    continue
                self._dispatch_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._disconnect(AppServerProcessError(f'Failed reading Codex app-server stdout: {e}'))
        finally:
            return_code = await self.process.wait()
            error = AppServerProcessError(f'Codex app-server exited with code {return_code}')
            self._disconnect(error)
            received_at = time.time()
            self._append_event('protocol.process_exited', {'exitCode': return_code}, timestamp=received_at)

    def _dispatch_message(self, message: dict[str, Any]) -> None:
        received_at = time.time()
        self.event_stream.events.append(
            app_server_message_to_event(
                message,
                seq=len(self.event_stream.events),
                timestamp=received_at,
                direction='inbound',
                request_methods={key: pending.method for key, pending in self._pending.items()},
            )
        )
        message['_received_at'] = received_at
        if 'method' in message:
            server_request_id = message.get('id')
            if isinstance(server_request_id, (int, str)) and isinstance(message.get('method'), str):
                self._start_request_handler(message)
            else:
                self._callbacks.publish(message['method'], message.get('params') or {})
            return

        request_id = message.get('id')
        pending = self._pending.get(request_id) if isinstance(request_id, int) else None
        if pending is None or pending.future.done():
            logger.debug('Ignoring response for unknown Codex app-server request id %r', request_id)
            return
        error = message.get('error')
        if isinstance(error, dict):
            pending.future.set_exception(AppServerRpcError(error))
        else:
            pending.future.set_result(message.get('result'))

    def _start_request_handler(self, request: dict[str, Any]) -> None:
        task = asyncio.create_task(self._handle_server_request(request), name='codex-app-server-request')
        self._request_handler_tasks.add(task)
        task.add_done_callback(self._request_handler_done)

    def _request_handler_done(self, task: asyncio.Task[None]) -> None:
        self._request_handler_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error('Codex app-server failed to send a server request response: %s', error)

    async def _handle_server_request(self, request: dict[str, Any]) -> None:
        request_id = request['id']
        method = request['method']
        handler = self._request_handlers.get(method)
        if handler is None:
            logger.warning('Rejecting unsupported Codex app-server request method %r', method)
            await self.respond(
                request_id,
                method=method,
                error={'code': -32601, 'message': f'Method not supported by Nexus: {method}'},
            )
            return

        params = request.get('params')
        if not isinstance(params, dict):
            await self.respond(
                request_id, method=method, error={'code': -32602, 'message': f'Invalid params for method: {method}'}
            )
            return

        try:
            result = await handler(params)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception('Codex app-server request handler failed for method %r', method)
            await self.respond(request_id, method=method, error={'code': -32603, 'message': str(e)})
        else:
            await self.respond(request_id, method=method, result=result)

    def _append_event(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        timestamp: float | None = None,
        direction: Literal['inbound', 'outbound'] | None = None,
    ) -> None:
        self.event_stream.events.append(
            CodexEvent(
                seq=len(self.event_stream.events),
                timestamp=time.time() if timestamp is None else timestamp,
                direction=direction,
                type=event_type,
                data=data,
            )
        )

    async def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while line := await self.process.stderr.readline():
                self.stderr_lines.append(line.decode(errors='replace').rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning('Failed reading Codex app-server stderr: %s', e)

    def _disconnect(self, error: Exception) -> None:
        if self._callbacks.error is not None:
            return
        # RPCs fail immediately; callbacks observe failure after earlier notifications.
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        self._callbacks.fail(error)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        request_handler_tasks = tuple(self._request_handler_tasks)
        for task in request_handler_tasks:
            task.cancel()
        await asyncio.gather(*request_handler_tasks, return_exceptions=True)

        if self.process.stdin is not None:
            self.process.stdin.close()
            with contextlib.suppress(AttributeError, BrokenPipeError, ConnectionResetError):
                await self.process.stdin.wait_closed()

        if self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()

        for task in self._reader_tasks:
            task.cancel()
        await asyncio.gather(*self._reader_tasks, return_exceptions=True)
        self._disconnect(AppServerProcessError('Codex app-server client closed'))
        await self._callbacks.close()


__all__ = ['AppServerClient', 'AppServerProcessError', 'AppServerRequestHandler', 'AppServerRpcError']
