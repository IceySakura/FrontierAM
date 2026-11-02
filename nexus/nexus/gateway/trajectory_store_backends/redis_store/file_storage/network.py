"""JSON control headers plus an opaque bytes frame over ROUTER/DEALER.

Each socket belongs to exactly one I/O thread. Public request() is safe to call
from Redis worker threads; request IDs correlate concurrent replies. This is a
trusted-network protocol, not an authenticated public endpoint.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time

from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import cast, overload
from uuid import uuid4

import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


logger = logging.getLogger(__name__)


class MsgType(str, Enum):
    HANDSHAKE = 'handshake'
    DISCOVER = 'discover'
    HEARTBEAT = 'heartbeat'
    STATUS = 'status'
    PUT = 'put'
    GET = 'get'
    DELETE = 'delete'
    OK = 'ok'
    ERROR = 'error'


@dataclass
class Message:
    msg_type: str
    body: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    sender_id: str = ''
    payload: bytes | bytearray | memoryview = field(default=b'', repr=False)

    @classmethod
    def create(
        cls,
        msg_type: str,
        body: dict | None = None,
        *,
        sender_id: str = '',
        payload: bytes | bytearray | memoryview = b'',
    ):
        return cls(msg_type, body if body is not None else {}, sender_id=sender_id, payload=payload)

    def reply(self, body: dict | None = None, *, sender_id: str = '', payload: bytes | bytearray | memoryview = b''):
        return Message(MsgType.OK, body if body is not None else {}, self.request_id, sender_id, payload)

    def error_reply(self, code: str, message: str, *, sender_id: str = ''):
        return Message(MsgType.ERROR, {'code': code, 'message': message}, self.request_id, sender_id)


class Codec:
    @staticmethod
    def encode(
        message: Message, blob: bytes | bytearray | memoryview | None = None
    ) -> list[bytes | bytearray | memoryview]:
        blob = message.payload if blob is None else blob
        data = {name: getattr(message, name) for name in ('msg_type', 'body', 'request_id', 'sender_id')}
        header = json.dumps(data, separators=(',', ':')).encode()
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TypeError('payload must be bytes-like')
        return [header, blob]

    @staticmethod
    def decode_wire(frames: list[bytes | bytearray | memoryview]) -> tuple[Message, bytes | bytearray | memoryview]:
        if len(frames) != 2:
            raise ValueError('expected a JSON header and one opaque frame')
        data = json.loads(cast(bytes, frames[0]))
        message = Message(**data, payload=frames[1])
        if (
            not isinstance(message.body, dict)
            or not isinstance(message.request_id, str)
            or not message.request_id
            or not isinstance(message.sender_id, str)
            or not isinstance(message.msg_type, str)
        ):
            raise ValueError('invalid message')
        return message, frames[1]

    decode = decode_wire


def _socket(kind: int, context: zmq.Context | None = None):
    sock = (context or zmq.Context.instance()).socket(kind)
    sock.setsockopt(zmq.LINGER, 0)
    # No application size or queue quota; callers own the memory budget.
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    sock.setsockopt(zmq.MAXMSGSIZE, -1)
    return sock


class RouterServer:
    """Socket ownership stays on one thread; deferred handlers release their worker."""

    def __init__(
        self,
        endpoint: Endpoint,
        handler=None,
        *,
        wire_handler=None,
        server_id: str | None = None,
        control_operations: Collection[str] = (),
        request_workers: int = 1,
    ):
        if (handler is None) == (wire_handler is None):
            raise ValueError('provide exactly one handler or wire_handler')
        if type(request_workers) is not int or request_workers < 1:
            raise ValueError('request_workers must be a positive integer')
        self.endpoint = endpoint
        self.handler = handler
        self.wire_handler = wire_handler
        self.server_id = server_id or uuid4().hex
        self._control_operations = frozenset(control_operations)
        self._request_workers = request_workers
        self._stop = threading.Event()
        self._ready: Future = Future()
        self._thread = threading.Thread(target=self._serve, name='hykv-fs-router', daemon=True)

    def _build_reply(self, message, handler_result):
        if self.wire_handler is not None:
            if not isinstance(handler_result, Message) or handler_result.request_id != message.request_id:
                raise ValueError('wire_handler must return a Message with the same request_id')
            return handler_result
        body, output = handler_result
        return message.reply(body, sender_id=self.server_id, payload=output)

    def _execute_request(self, message: Message) -> Message | Future:
        try:
            if self.wire_handler is not None:
                handler_result = self.wire_handler(message)
            else:
                handler_result = cast(Callable, self.handler)(message.msg_type, message.body, message.payload)
            if isinstance(handler_result, Future):
                reply_future: Future = Future()
                request_metadata = replace(message, payload=b'')

                def on_handler_done(completed):
                    try:
                        response = self._build_reply(request_metadata, completed.result())
                    except Exception as exc:
                        response = request_metadata.error_reply(
                            getattr(exc, 'code', 'server_error'), str(exc), sender_id=self.server_id
                        )
                    reply_future.set_result(response)

                handler_result.add_done_callback(on_handler_done)
                return reply_future
            return self._build_reply(message, handler_result)
        except Exception as exc:
            return message.error_reply(getattr(exc, 'code', 'server_error'), str(exc), sender_id=self.server_id)

    @staticmethod
    def _send_reply(sock, identity, reply):
        try:
            sock.send_multipart([identity, *Codec.encode(reply)], flags=zmq.DONTWAIT)
        except zmq.ZMQError:
            # The client may already have timed out or disconnected.
            logger.debug('could not deliver FS reply', exc_info=True)

    def start(self):
        self._thread.start()
        self._ready.result(timeout=10)
        return self

    def _serve(self):
        sock = _socket(zmq.ROUTER)
        sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        executors = []
        completed_replies: queue.Queue[tuple[bytes, Future]] = queue.Queue()
        wake_sockets: list[socket.socket] = []
        wake_pending = threading.Event()
        poller = None
        pending_reply_count = 0

        def queue_completed_reply(identity, future):
            reply_or_future = future.result()
            if isinstance(reply_or_future, Future):
                reply_or_future.add_done_callback(lambda result: queue_completed_reply(identity, result))
                return
            completed_replies.put_nowait((identity, future))
            if not wake_pending.is_set():
                wake_pending.set()
                wake_sockets[1].send(b'\x00')

        try:
            if self.endpoint.port:
                sock.bind(self.endpoint.bind_address)
            else:
                port = sock.bind_to_random_port(f'tcp://{self.endpoint.host}')
                self.endpoint = Endpoint(self.endpoint.host, port)
            if self._control_operations or self._request_workers != 1:
                wake_sockets.extend(socket.socketpair())
                poller = zmq.Poller()
                poller.register(sock, zmq.POLLIN)
                poller.register(wake_sockets[0].fileno(), zmq.POLLIN)
                executors.append(
                    ThreadPoolExecutor(max_workers=self._request_workers, thread_name_prefix='hykv-fs-data')
                )
                executors.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix='hykv-fs-control'))
            self._ready.set_result(None)
            while not self._stop.is_set() or pending_reply_count:
                for _ in range(completed_replies.qsize()):
                    identity, future = completed_replies.get_nowait()
                    pending_reply_count -= 1
                    reply = future.result()
                    self._send_reply(sock, identity, reply)
                    future = reply = None
                if self._stop.is_set():
                    if pending_reply_count:
                        time.sleep(0.01)
                    continue
                if poller is not None:
                    events = dict(poller.poll(10))
                    if wake_sockets[0].fileno() in events:
                        wake_sockets[0].recv(4096)
                        wake_pending.clear()
                    if sock not in events:
                        continue
                elif not sock.poll(10):
                    continue
                try:
                    frames = sock.recv_multipart()
                    identity, wire = frames[0], frames[1:]
                    try:
                        message = Codec.decode(wire)[0]
                    except (ValueError, TypeError, KeyError):
                        logger.warning('discarded malformed FS request')
                        continue
                    if executors:
                        executor = executors[1 if message.msg_type in self._control_operations else 0]
                        reply = executor.submit(self._execute_request, message)
                    else:
                        reply = self._execute_request(message)
                    if isinstance(reply, Future):
                        if poller is None:
                            wake_sockets.extend(socket.socketpair())
                            poller = zmq.Poller()
                            poller.register(sock, zmq.POLLIN)
                            poller.register(wake_sockets[0].fileno(), zmq.POLLIN)
                        pending_reply_count += 1
                        reply.add_done_callback(
                            lambda result, identity=identity: queue_completed_reply(identity, result)
                        )
                    else:
                        self._send_reply(sock, identity, reply)
                finally:
                    # Idle polling must not retain the last PUT or a temporary SSD GET buffer.
                    frames = wire = []
                    message = reply = future = None
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                logger.exception('FS router stopped unexpectedly')
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
            for notification in wake_sockets:
                notification.close()
            sock.close()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise FutureTimeoutError('FS handler did not stop; check Redis socket timeouts')


class DealerClient:
    """Multiplexed RPC without request/byte quotas. timeout=None waits indefinitely."""

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        timeout: float | None = 120,
        context: zmq.Context | None = None,
        zero_copy: bool = False,
    ):
        if timeout is not None and timeout <= 0:
            raise ValueError('timeout must be positive or None')
        self.endpoint = endpoint
        self.timeout = timeout
        self._context = context
        self._zero_copy = zero_copy
        self.client_id = uuid4().hex
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name='hykv-fs-dealer', daemon=True)
        self._thread.start()

    @property
    def closed(self) -> bool:
        return self._stop.is_set() or not self._thread.is_alive()

    @property
    def thread_alive(self) -> bool:
        """True while the I/O thread may still own its socket."""
        return self._thread.is_alive()

    @overload
    def request(
        self, msg_type: str, body: dict | None = None, blob: bytes | bytearray | memoryview = b''
    ) -> tuple[dict, bytes | bytearray | memoryview]: ...

    @overload
    def request(self, msg_type: Message, body: None = None, blob: bytes | bytearray | memoryview = b'') -> Message: ...

    def request(
        self, msg_type: str | Message, body: dict | None = None, blob: bytes | bytearray | memoryview = b''
    ) -> tuple[dict, bytes | bytearray | memoryview] | Message:
        """Message calls return a Message; string calls keep the compact RPC API.

        Each submitted Message must have a fresh request_id, including retries.
        Transport errors raise; a Message-mode remote error is an error_reply.
        """
        if self._stop.is_set():
            raise RedisClientError('FS client is closed', code='closed')
        message_mode = isinstance(msg_type, Message)
        if isinstance(msg_type, Message):
            if body is not None or blob:
                raise ValueError('put body and payload inside the Message')
            message = replace(msg_type, sender_id=msg_type.sender_id or self.client_id)
        else:
            message = Message.create(msg_type, body, sender_id=self.client_id, payload=blob)
        wire = Codec.encode(message)
        future: Future = Future()
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        self._queue.put_nowait((message.request_id, wire, future, deadline))
        try:
            reply = future.result(timeout=self.timeout)
        except FutureTimeoutError as exc:
            raise RedisClientError('FS request timed out; write outcome may be unknown', code='timeout') from exc
        if message_mode:
            return reply
        if reply.msg_type == MsgType.ERROR:
            raise RedisClientError(reply.body.get('message', 'FS error'), code=reply.body.get('code', 'server_error'))
        return reply.body, reply.payload

    @staticmethod
    def _fail(future, error):
        if not future.done():
            future.set_exception(error)

    def _serve(self):
        sock = _socket(zmq.DEALER, self._context)
        pending = {}
        try:
            sock.connect(self.endpoint.tcp_address)
            while not self._stop.is_set():
                try:
                    # Drain the current queue snapshot, then service replies. No request is rejected.
                    for _ in range(self._queue.qsize()):
                        try:
                            rid, wire, future, deadline = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if deadline is not None and time.monotonic() >= deadline:
                            self._fail(future, RedisClientError('FS request expired before send', code='timeout'))
                            continue
                        if rid in pending:
                            self._fail(
                                future, RedisClientError('request_id is already pending', code='duplicate_request')
                            )
                            continue
                        try:
                            sock.send_multipart(wire, flags=zmq.DONTWAIT)
                            pending[rid] = (future, deadline)
                        except zmq.ZMQError as exc:
                            self._fail(
                                future, RedisClientError(f'FS transport send failed: {exc}', code='transport_error')
                            )
                    if sock.poll(5):
                        try:
                            if self._zero_copy:
                                # The payload memoryview owns the ZMQ frame; no Python copy of large blobs.
                                header, payload = sock.recv_multipart(copy=False)
                                reply = Codec.decode([header.bytes, payload.buffer])[0]
                            else:
                                reply = Codec.decode(sock.recv_multipart())[0]
                        except (ValueError, TypeError, KeyError):
                            continue
                        item = pending.pop(reply.request_id, None)
                        if item is not None and not item[0].done():
                            item[0].set_result(reply)
                    now = time.monotonic()
                    for rid, (future, deadline) in list(pending.items()):
                        if future.done() or (deadline is not None and now >= deadline):
                            pending.pop(rid)
                            self._fail(
                                future, RedisClientError('FS request timed out; outcome unknown', code='timeout')
                            )
                finally:
                    # Pending requests and callers own buffers; idle I/O locals must not retain them.
                    wire = []
                    future = item = header = payload = reply = None
        except Exception:
            logger.exception('FS dealer stopped unexpectedly')
        finally:
            self._stop.set()
            sock.close()
            error = RedisClientError('FS connection closed', code='closed')
            for future, _ in pending.values():
                self._fail(future, error)
            while True:
                try:
                    _, _, future, _ = self._queue.get_nowait()
                    self._fail(future, error)
                except queue.Empty:
                    break

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)
