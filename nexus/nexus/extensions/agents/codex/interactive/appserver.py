"""App-server process management plus a relay that taps its traffic.

The Codex app-server's ``unix://`` listener speaks WebSocket (``GET /rpc`` then an
Upgrade), carrying JSON-RPC objects that omit the ``jsonrpc`` field. The relay sits
between the PTY's TUI and that listener:

    TUI ──► relay ──► app-server
              │
              ├─ decodes the server→client direction
              └─ may answer a captured server request on the same connection

Forwarding is byte-exact in both directions. The initial prompt is a native TUI
CLI argument; follow-ups use the PTY. Structured
app-server requests are answered on the TUI's same WebSocket connection using the
captured request id and the method-specific Codex result schema verified across
the supported stable range.

Server→client frames are unmasked per RFC 6455, so decoding needs no key handling;
client→server frames are forwarded untouched. Injected client frames are masked,
as RFC 6455 requires, and share a write lock with forwarded TUI traffic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import struct

from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)

_READ_CHUNK = 65536
_HEADER_TERMINATOR = b'\r\n\r\n'
# Kept only to explain a failure; the rest of the app-server log is discarded.
_STDERR_TAIL_BYTES = 8192


def encode_masked_text_frame(payload: str, *, mask: bytes | None = None) -> bytes:
    """Encode one RFC 6455 client text frame.

    Client frames must be masked. ``mask`` is injectable only so tests can assert
    the exact wire bytes; production always uses fresh entropy.
    """
    data = payload.encode('utf-8')
    mask = os.urandom(4) if mask is None else mask
    if len(mask) != 4:
        raise ValueError('WebSocket mask must be exactly four bytes')
    length = len(data)
    if length < 126:
        header = bytes((0x81, 0x80 | length))
    elif length < (1 << 16):
        header = bytes((0x81, 0x80 | 126)) + struct.pack('>H', length)
    else:
        header = bytes((0x81, 0x80 | 127)) + struct.pack('>Q', length)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
    return header + mask + masked


class ClientStreamFramer:
    """Splits a client→server byte stream on WebSocket frame boundaries.

    The relay forwards TUI bytes verbatim, but it also injects its own response
    frames onto the same connection. A read chunk is not a frame: a prompt larger
    than the read size arrives in several chunks, and releasing the write lock
    between them would let an injected frame land inside a half-written TUI frame,
    corrupting the stream for the app-server.

    So this buffers any trailing partial frame and hands back only whole frames,
    letting the caller hold the write lock across a complete frame run. Payloads
    are never decoded or altered -- only their lengths and the FIN/opcode bits.

    Delivery is cut on complete *messages*, not merely complete frames. A
    fragmented message is a data frame with FIN clear followed by continuation
    frames, and an injected frame placed between them is an illegal sequence for
    the upstream -- it reads the injected payload as continuation bytes of the
    message the TUI is still sending.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._handshake_done = False

    def feed(self, data: bytes) -> bytes:
        """Return the prefix of buffered bytes that forms complete frames."""
        self._buf.extend(data)
        if not self._handshake_done:
            # The client opens with the HTTP upgrade request; forward it as-is.
            end = self._buf.find(_HEADER_TERMINATOR)
            if end < 0:
                return b''
            cut = end + len(_HEADER_TERMINATOR)
            head = bytes(self._buf[:cut])
            del self._buf[:cut]
            self._handshake_done = True
            return head + self._take_whole_frames()
        return self._take_whole_frames()

    def flush(self) -> bytes:
        """Return whatever is still buffered, complete or not.

        Used when the client goes away: holding a partial frame back forever would
        silently truncate the stream instead of letting the upstream see the end.
        """
        rest = bytes(self._buf)
        self._buf.clear()
        return rest

    def _take_whole_frames(self) -> bytes:
        """Return the buffered prefix that ends on a message boundary."""
        scanned = 0
        deliverable = 0
        fragmented = False
        while True:
            frame = self._frame_at(self._buf, scanned)
            if frame is None:
                break
            size, fin, opcode = frame
            scanned += size
            if opcode >= 0x8:
                # Control frames may be interleaved inside a fragmented message.
                # They end nothing, so one is a cut point only outside a fragment.
                if not fragmented:
                    deliverable = scanned
                continue
            fragmented = not fin
            if fin:
                deliverable = scanned
        if deliverable == 0:
            return b''
        out = bytes(self._buf[:deliverable])
        del self._buf[:deliverable]
        return out

    @staticmethod
    def _frame_at(buf: bytearray, start: int) -> tuple[int, bool, int] | None:
        """Return (byte length, FIN, opcode) of the frame at *start*, or None."""
        available = len(buf) - start
        if available < 2:
            return None
        first = buf[start]
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        second = buf[start + 1]
        masked = bool(second & 0x80)
        length = second & 0x7F
        header = 2
        if length == 126:
            if available < header + 2:
                return None
            length = struct.unpack('>H', buf[start + header : start + header + 2])[0]
            header += 2
        elif length == 127:
            if available < header + 8:
                return None
            length = struct.unpack('>Q', buf[start + header : start + header + 8])[0]
            header += 8
        if masked:
            header += 4
        total = header + length
        if available < total:
            return None
        return total, fin, opcode


class WebSocketFrameDecoder:
    """Incremental WebSocket reader for one server→client byte stream.

    Handles the HTTP upgrade preamble, fragmented text frames, and close frames.
    Binary/ping/pong frames are ignored: the app-server only sends text.
    """

    CLOSE_SENTINEL = '__ws_close__'

    def __init__(self) -> None:
        self._buf = bytearray()
        self._fragment = bytearray()
        self._handshake_done = False

    def feed(self, data: bytes) -> list[str]:
        """Return complete text payloads (and close sentinels) decoded from *data*."""
        self._buf.extend(data)
        out: list[str] = []
        if not self._handshake_done:
            end = self._buf.find(_HEADER_TERMINATOR)
            if end < 0:
                return out
            del self._buf[: end + len(_HEADER_TERMINATOR)]
            self._handshake_done = True
        while True:
            frame = self._take_frame()
            if frame is None:
                return out
            opcode, fin, payload = frame
            if opcode == 0x8:
                out.append(self.CLOSE_SENTINEL)
                continue
            if opcode not in (0x0, 0x1):
                continue
            self._fragment.extend(payload)
            if fin:
                out.append(self._fragment.decode('utf-8', 'replace'))
                self._fragment.clear()

    def _take_frame(self) -> tuple[int, bool, bytes] | None:
        buf = self._buf
        if len(buf) < 2:
            return None
        b0, b1 = buf[0], buf[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        offset = 2
        if length == 126:
            if len(buf) < offset + 2:
                return None
            length = struct.unpack('>H', buf[offset : offset + 2])[0]
            offset += 2
        elif length == 127:
            if len(buf) < offset + 8:
                return None
            length = struct.unpack('>Q', buf[offset : offset + 8])[0]
            offset += 8
        mask = None
        if masked:
            if len(buf) < offset + 4:
                return None
            mask = bytes(buf[offset : offset + 4])
            offset += 4
        if len(buf) < offset + length:
            return None
        payload = bytes(buf[offset : offset + length])
        del buf[: offset + length]
        if mask is not None:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return opcode, fin, payload


class CodexAppServer:
    """Owns one session-private app-server process and the relay in front of it.

    Two unix sockets live under the session directory: the upstream socket the
    app-server binds, and the relay socket the TUI is pointed at. Keeping both
    per-session (and never using ``codex app-server daemon``, which is a
    machine-wide shared service) is what isolates concurrent sessions.
    """

    def __init__(
        self,
        *,
        codex_argv: list[str],
        env: dict[str, str],
        upstream_socket: str,
        relay_socket: str,
        on_message: Callable[[dict[str, Any]], None],
        on_event_channel_down: Callable[[], None],
        startup_timeout: float = 60.0,
    ) -> None:
        self._codex_argv = list(codex_argv)
        self._env = dict(env)
        self.upstream_socket = upstream_socket
        self.relay_socket = relay_socket
        self._on_message = on_message
        self._on_event_channel_down = on_event_channel_down
        self._startup_timeout = startup_timeout

        self.proc: asyncio.subprocess.Process | None = None
        self._relay: asyncio.AbstractServer | None = None
        self._pumps: set[asyncio.Task] = set()
        self._links: list[tuple[asyncio.StreamWriter, asyncio.StreamWriter]] = []
        self._upstream_writer: asyncio.StreamWriter | None = None
        self._upstream_write_lock = asyncio.Lock()
        self._exit_watch: asyncio.Task | None = None
        self._stderr_drain: asyncio.Task | None = None
        self._stderr_tail = bytearray()
        self._process_group_id: int | None = None
        self.channel_down = False
        self.channel_error: str | None = None
        self._closed = False
        self._closing = False
        self._close_lock = asyncio.Lock()

    async def start(self) -> None:
        """Launch the app-server, wait for its socket, then open the relay."""
        argv = [*self._codex_argv, 'app-server', '--listen', f'unix://{self.upstream_socket}']
        logger.info('[codex-pty] starting app-server on unix://%s', self.upstream_socket)
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            env=self._env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            # Own process group, so teardown can killpg the whole tree without
            # ever signalling the worker's own group.
            start_new_session=True,
        )
        self._process_group_id = self.proc.pid
        # Drain from the moment the process exists: a server that logs heavily
        # before binding its socket can fill the pipe and block inside startup,
        # which the wait loop below would see only as a missing socket.
        self._stderr_drain = asyncio.create_task(self._drain_stderr())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        while loop.time() < deadline:
            if os.path.exists(self.upstream_socket):
                break
            if self.proc.returncode is not None:
                # Give the drain task a moment to pick up the final writes rather
                # than reading the pipe here -- two readers would split the output.
                await asyncio.sleep(0.1)
                raise RuntimeError(
                    f'Codex app-server exited during startup (rc={self.proc.returncode}): {self.recent_stderr()[:500]}'
                )
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f'Codex app-server did not create its socket within {self._startup_timeout}s')

        self._relay = await asyncio.start_unix_server(self._handle_client, path=self.relay_socket)
        self._exit_watch = asyncio.create_task(self._watch_exit())
        logger.info('[codex-pty] relay ready: %s -> %s', self.relay_socket, self.upstream_socket)

    def recent_stderr(self) -> str:
        """Last bytes the app-server wrote to stderr, decoded."""
        return bytes(self._stderr_tail).decode('utf-8', 'replace')

    async def _drain_stderr(self) -> None:
        """Consume app-server stderr so a full pipe cannot block the process.

        Reads raw bytes rather than lines: ``readline()`` raises once a single log
        line exceeds StreamReader's limit, and the exception would stop the drain
        for good, leaving the pipe to fill and stall the server mid-turn.
        """
        if self.proc is None or self.proc.stderr is None:
            return
        stream = self.proc.stderr
        while True:
            try:
                chunk = await stream.read(_READ_CHUNK)
            except (asyncio.CancelledError, asyncio.IncompleteReadError):
                raise
            except Exception as error:  # noqa: BLE001 - keep draining whatever the stream raises
                logger.debug('[codex-pty] app-server stderr read failed: %s', error)
                await asyncio.sleep(0.05)
                continue
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > _STDERR_TAIL_BYTES:
                del self._stderr_tail[: len(self._stderr_tail) - _STDERR_TAIL_BYTES]

    async def _watch_exit(self) -> None:
        if self.proc is None:
            return
        await self.proc.wait()
        logger.warning('[codex-pty] app-server exited rc=%s', self.proc.returncode)
        self._mark_channel_down()

    def _mark_channel_down(self, error: str | None = None) -> None:
        if error is not None and self.channel_error is None:
            self.channel_error = error
        if self.channel_down or self._closed:
            return
        self.channel_down = True
        with contextlib.suppress(Exception):
            self._on_event_channel_down()

    async def _handle_client(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_unix_connection(self.upstream_socket)
        except OSError as exc:
            logger.warning('[codex-pty] relay could not reach upstream: %s', exc)
            with contextlib.suppress(Exception):
                client_writer.close()
            self._mark_channel_down()
            return
        self._links.append((client_writer, upstream_writer))
        if self._upstream_writer is not None and not self._upstream_writer.is_closing():
            logger.error('[codex-pty] refusing a second simultaneous relay client')
            client_writer.close()
            upstream_writer.close()
            return
        self._upstream_writer = upstream_writer
        decoder = WebSocketFrameDecoder()

        framer = ClientStreamFramer()

        async def to_upstream() -> None:
            with contextlib.suppress(ConnectionError, OSError, asyncio.CancelledError):
                while True:
                    data = await client_reader.read(_READ_CHUNK)
                    if not data:
                        break
                    # Forward only whole frames: an injected response must never
                    # be written into the middle of a partially forwarded frame.
                    forwardable = framer.feed(data)
                    if not forwardable:
                        continue
                    async with self._upstream_write_lock:
                        upstream_writer.write(forwardable)
                        await upstream_writer.drain()
                trailing = framer.flush()
                if trailing:
                    # The client stopped mid-frame; pass the remainder through so
                    # the upstream sees the truncation instead of a silent stall.
                    logger.debug('[codex-pty] flushing %d trailing client bytes', len(trailing))
                    async with self._upstream_write_lock:
                        upstream_writer.write(trailing)
                        await upstream_writer.drain()
            with contextlib.suppress(Exception):
                upstream_writer.close()

        async def to_client() -> None:
            with contextlib.suppress(ConnectionError, OSError, asyncio.CancelledError):
                while True:
                    data = await upstream_reader.read(_READ_CHUNK)
                    if not data:
                        break
                    for payload in decoder.feed(data):
                        self._dispatch(payload)
                    client_writer.write(data)
                    await client_writer.drain()
            with contextlib.suppress(Exception):
                client_writer.close()
            self._mark_channel_down()

        task = asyncio.create_task(self._join(to_upstream(), to_client()))
        self._pumps.add(task)
        task.add_done_callback(self._pumps.discard)
        try:
            await task
        finally:
            if self._upstream_writer is upstream_writer:
                self._upstream_writer = None

    async def send_response(self, request_id: str | int, result: dict[str, Any]) -> None:
        """Send one JSON-RPC result on the TUI's active WebSocket connection.

        Codex omits the ``jsonrpc`` member on the wire. The request id is kept in
        its original JSON type; normalising an integer id to a string would answer
        a different request.
        """
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            raise TypeError('Codex request id must be a string or integer')
        if not isinstance(result, dict):
            raise TypeError('Codex response result must be an object')
        writer = self._upstream_writer
        if self._closed or self._closing or self.channel_down or writer is None or writer.is_closing():
            raise RuntimeError('Codex app-server WebSocket is not available')
        payload = json.dumps({'id': request_id, 'result': result}, ensure_ascii=False, separators=(',', ':'))
        frame = encode_masked_text_frame(payload)
        async with self._upstream_write_lock:
            if writer is not self._upstream_writer or writer.is_closing():
                raise RuntimeError('Codex app-server WebSocket changed before the response was sent')
            writer.write(frame)
            await writer.drain()

    @staticmethod
    async def _join(*coros) -> None:
        await asyncio.gather(*coros, return_exceptions=True)

    def _dispatch(self, payload: str) -> None:
        if payload == WebSocketFrameDecoder.CLOSE_SENTINEL:
            logger.info('[codex-pty] observed WebSocket close from app-server')
            self._mark_channel_down()
            return
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as error:
            logger.error('[codex-pty] app-server sent an invalid JSON-RPC text frame: %s', error)
            self._mark_channel_down(f'app-server sent invalid JSON-RPC: {error}')
            return
        if not isinstance(message, dict):
            self._mark_channel_down('app-server sent a JSON-RPC payload that is not an object')
            return
        try:
            self._on_message(message)
        except Exception as error:
            logger.exception('[codex-pty] app-server message callback failed')
            self._mark_channel_down(f'app-server message callback failed: {type(error).__name__}: {error}')

    async def close(self) -> None:
        """Tear down relay, links, tasks, and the app-server process tree.

        A failed or cancelled cleanup remains retryable; ``_closed`` is set only
        after every owned resource has been handled.
        """
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            errors: list[tuple[str, Exception]] = []
            try:
                if self._relay is not None:
                    self._relay.close()
                    try:
                        await asyncio.shield(self._relay.wait_closed())
                    except Exception as error:
                        errors.append(('relay', error))
                    else:
                        self._relay = None
                for client_writer, upstream_writer in self._links:
                    for writer in (upstream_writer, client_writer):
                        try:
                            writer.close()
                        except Exception as error:
                            errors.append(('websocket link', error))
                self._links.clear()
                self._upstream_writer = None

                tasks = [t for t in (*self._pumps, self._exit_watch, self._stderr_drain) if t is not None]
                for task in tasks:
                    task.cancel()
                if tasks:
                    results = await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
                    for result in results:
                        if isinstance(result, Exception):
                            errors.append(('relay task', result))
                self._pumps.clear()
                self._exit_watch = None
                self._stderr_drain = None

                try:
                    await self._terminate_process_tree()
                except Exception as error:
                    errors.append(('app-server process', error))
                for path in (self.relay_socket, self.upstream_socket):
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        errors.append((path, error))
                if errors:
                    details = '; '.join(f'{stage}: {error}' for stage, error in errors)
                    raise RuntimeError(f'Codex app-server cleanup failed: {details}') from errors[0][1]
                self._closed = True
            finally:
                self._closing = False

    async def _terminate_process_tree(self) -> None:
        """Kill the app-server and anything it spawned.

        Codex runs the model's shell commands from the app-server, not from the
        PTY, so those children are in neither the TUI's process group nor its
        descendant tree. Killing only the parent would strand them.
        """
        if self.proc is None:
            return
        pid = self.proc.pid
        process_group_id = self._process_group_id or pid
        children: list[Any] = []
        if self.proc.returncode is None:
            try:
                import psutil

                children = psutil.Process(pid).children(recursive=True)
            except Exception:  # noqa: BLE001 - psutil missing or process already gone
                children = []

        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGTERM)
        if self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self.proc.wait()), timeout=3)

        # The app-server may have exited before cleanup while a command remains
        # in its process group. Signal the saved group id even when the parent is
        # already gone.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGKILL)
        if self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self.proc.wait()), timeout=3)

        if self.proc.returncode is None:
            raise RuntimeError(f'Codex app-server process {pid} did not exit')

        for child in children:
            with contextlib.suppress(Exception):
                if child.is_running():
                    child.kill()


__all__ = ['CodexAppServer', 'WebSocketFrameDecoder', 'encode_masked_text_frame']
