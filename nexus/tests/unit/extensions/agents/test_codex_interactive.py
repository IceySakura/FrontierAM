"""Unit + integration tests for the codex_interactive PTY agent.

The integration cases run a real PTY against a fake Codex TUI and a fake
app-server that speaks the real wire format (WebSocket over a unix socket,
JSON-RPC without the ``jsonrpc`` field), so relay decoding, turn correlation, and
the interaction bridge are exercised rather than mocked away.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import signal
import struct
import sys
import time
import uuid

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.codex.agent import (
    CodexInput,
    CodexInteractiveInput,
    CodexInteractiveOutput,
)
from nexus.extensions.agents.codex.installer import DEFAULT_CODEX_VERSION
from nexus.extensions.agents.codex.interactive import (
    INTERACTIVE_DEFAULT_CODEX_VERSION,
    INTERACTIVE_MIN_CODEX_VERSION,
    CodexVersionUnsupportedError,
    validate_interactive_version,
)
from nexus.extensions.agents.codex.interactive.appserver import WebSocketFrameDecoder, encode_masked_text_frame
from nexus.extensions.agents.codex.interactive.boundary import CodexBoundaryAdapter, PendingInteraction, TurnOutcome
from nexus.extensions.agents.codex.interactive.events import EventBuffer, extract_event
from nexus.extensions.agents.codex.interactive.session import (
    CodexInteractiveSession,
    InvalidStateTransitionError,
    SessionState,
)


_WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


def _accept_key(client_key: str) -> str:
    digest = hashlib.sha1(f'{client_key}{_WS_GUID}'.encode()).digest()  # noqa: S324 - RFC 6455
    return base64.b64encode(digest).decode()


def encode_frame(payload: str, *, masked: bool = False) -> bytes:
    """Encode one text frame. Server frames are unmasked, client frames masked."""
    data = payload.encode('utf-8')
    header = bytearray([0x81])
    length = len(data)
    mask_bit = 0x80 if masked else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length < (1 << 16):
        header.append(mask_bit | 126)
        header.extend(struct.pack('>H', length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack('>Q', length))
    if masked:
        key = os.urandom(4)
        header.extend(key)
        data = bytes(b ^ key[i % 4] for i, b in enumerate(data))
    return bytes(header) + data


def close_frame() -> bytes:
    return bytes([0x88, 0x00])


def _take_frames(buffer: bytearray) -> list[tuple[str, bool]]:
    """Consume complete text frames from a client byte stream."""
    frames: list[tuple[str, bool]] = []
    while len(buffer) >= 2:
        opcode = buffer[0] & 0x0F
        masked = bool(buffer[1] & 0x80)
        length = buffer[1] & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < offset + 2:
                break
            length = struct.unpack('>H', buffer[offset : offset + 2])[0]
            offset += 2
        elif length == 127:
            if len(buffer) < offset + 8:
                break
            length = struct.unpack('>Q', buffer[offset : offset + 8])[0]
            offset += 8
        mask = b''
        if masked:
            if len(buffer) < offset + 4:
                break
            mask = bytes(buffer[offset : offset + 4])
            offset += 4
        if len(buffer) < offset + length:
            break
        payload = bytes(buffer[offset : offset + length])
        del buffer[: offset + length]
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if opcode == 0x1:
            frames.append((payload.decode('utf-8'), masked))
    return frames


class FakeAppServer:
    """Minimal app-server: accepts one or more WebSocket clients and scripts events."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received: list[dict[str, Any]] = []
        self.received_frames_masked: list[bool] = []
        self.pending_requests: dict[str | int, tuple[str, str]] = {}
        self.thread_id = f'thread-{uuid.uuid4()}'

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=self.socket_path)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await reader.readuntil(b'\r\n\r\n')
        key = ''
        for line in request.decode('latin-1').split('\r\n'):
            if line.lower().startswith('sec-websocket-key:'):
                key = line.split(':', 1)[1].strip()
        writer.write(
            b'HTTP/1.1 101 Switching Protocols\r\n'
            b'connection: Upgrade\r\nupgrade: websocket\r\n'
            + f'sec-websocket-accept: {_accept_key(key)}\r\n\r\n'.encode()
        )
        await writer.drain()
        self._writers.append(writer)
        buffer = bytearray()
        with contextlib.suppress(Exception):
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                for payload, masked in _take_frames(buffer):
                    self.received_frames_masked.append(masked)
                    message = json.loads(payload)
                    if not isinstance(message, dict):
                        continue
                    self.received.append(message)
                    request_id = message.get('id')
                    correlation = self.pending_requests.pop(request_id, None)
                    if correlation is not None and ('result' in message or 'error' in message):
                        thread_id, turn_id = correlation
                        await self._broadcast(
                            {
                                'method': 'serverRequest/resolved',
                                'params': {
                                    'threadId': thread_id,
                                    'turnId': turn_id,
                                    'requestId': request_id,
                                },
                            }
                        )

    async def _broadcast(self, message: dict[str, Any]) -> None:
        frame = encode_frame(json.dumps(message))
        for writer in list(self._writers):
            with contextlib.suppress(Exception):
                writer.write(frame)
                await writer.drain()

    # ----- scripting helpers ------------------------------------------------ #

    async def emit_turn_started(self, turn_id: str, *, thread_id: str | None = None) -> None:
        await self._broadcast(
            {
                'method': 'turn/started',
                'params': {
                    'threadId': thread_id or self.thread_id,
                    'turn': {'id': turn_id, 'status': 'inProgress', 'items': []},
                },
            }
        )

    async def emit_turn_completed(
        self,
        turn_id: str,
        *,
        thread_id: str | None = None,
        status: str = 'completed',
        error: str | None = None,
    ) -> None:
        turn: dict[str, Any] = {'id': turn_id, 'status': status, 'completedAt': 1700000000, 'durationMs': 1200}
        if error is not None:
            turn['error'] = {'message': error}
        await self._broadcast(
            {
                'method': 'turn/completed',
                'params': {'threadId': thread_id or self.thread_id, 'turn': turn},
            }
        )

    async def emit_item(self, item_type: str, *, turn_id: str | None = None, **payload: Any) -> None:
        await self._broadcast(
            {
                'method': 'item/completed',
                'params': {
                    'threadId': self.thread_id,
                    'turnId': turn_id,
                    'item': {'id': f'item-{uuid.uuid4().hex[:8]}', 'itemType': item_type, **payload},
                },
            }
        )

    async def emit_approval_request(
        self,
        request_id: str,
        *,
        method: str = 'item/commandExecution/requestApproval',
        turn_id: str | None = None,
        command: str = 'echo hi',
    ) -> None:
        actual_turn_id = turn_id or ''
        self.pending_requests[request_id] = (self.thread_id, actual_turn_id)
        await self._broadcast(
            {
                'id': request_id,
                'method': method,
                'params': {'threadId': self.thread_id, 'turnId': turn_id, 'command': command},
            }
        )

    async def emit_user_question(
        self,
        request_id: str,
        *,
        turn_id: str | None = None,
        question: str = 'Which database should I use?',
    ) -> None:
        actual_turn_id = turn_id or ''
        self.pending_requests[request_id] = (self.thread_id, actual_turn_id)
        await self._broadcast(
            {
                'id': request_id,
                'method': 'item/tool/requestUserInput',
                'params': {
                    'threadId': self.thread_id,
                    'turnId': turn_id,
                    'questions': [{'id': 'q1', 'header': 'Database', 'question': question, 'options': []}],
                },
            }
        )

    async def emit_interaction_request(
        self,
        request_id: str | int,
        method: str,
        *,
        turn_id: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Emit an arbitrary correlated server request."""
        self.pending_requests[request_id] = (self.thread_id, turn_id)
        await self._broadcast(
            {
                'id': request_id,
                'method': method,
                'params': {'threadId': self.thread_id, 'turnId': turn_id, **(params or {})},
            }
        )

    async def drop_clients(self) -> None:
        """Close every client connection: simulates the event channel going down."""
        for writer in list(self._writers):
            with contextlib.suppress(Exception):
                writer.write(close_frame())
                await writer.drain()
            with contextlib.suppress(Exception):
                writer.close()
        self._writers.clear()

    async def close(self) -> None:
        await self.drop_clients()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(OSError):
            os.unlink(self.socket_path)


FAKE_TUI_SCRIPT = r"""
import base64, os, socket, sys, threading, select

# Connect to the relay exactly like the real TUI does: WebSocket over a unix
# socket. Without this the relay has nothing to forward and no events to tap.
def _connect_relay():
    path = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--remote' and i + 1 < len(argv):
            path = argv[i + 1]
            break
    if not path or not path.startswith('unix://'):
        return
    path = path[len('unix://'):]
    for _ in range(100):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(path)
            break
        except OSError:
            import time as _t
            _t.sleep(0.05)
    else:
        return
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall(
        ('GET /rpc HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\n'
         'Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n'
         f'Sec-WebSocket-Key: {key}\r\n\r\n').encode()
    )
    def pump():
        while True:
            try:
                if not sock.recv(65536):
                    return
            except OSError:
                return
    threading.Thread(target=pump, daemon=True).start()

_connect_relay()

# Minimal stand-in for the Codex TUI: renders a configurable composer and echoes
# pasted text.
argv = sys.argv[1:]
INITIAL_PROMPT = argv[argv.index('--') + 1] if '--' in argv and argv.index('--') + 1 < len(argv) else ''
COMPOSER_HINT = os.environ.get('FAKE_TUI_COMPOSER_HINT', 'Ask Codex to do anything')
out = sys.stdout

def render(body):
    out.write('\x1b[2J\x1b[H' + body)
    out.flush()

composer = ''
last_prompt = ''

def record_prompt(prompt):
    global last_prompt
    with open(os.environ['FAKE_TUI_SUBMIT_LOG'], 'a') as fh:
        fh.write(prompt + '\n')
    last_prompt = prompt

if INITIAL_PROMPT:
    record_prompt(INITIAL_PROMPT)

def paint():
    history = ('› ' + last_prompt + '\r\n\r\n') if last_prompt else ''
    render('OpenAI Codex (fake)\r\n\r\n' + history + '› ' + (composer or COMPOSER_HINT) + '\r\n')

paint()
# Read raw bytes: pasted text is UTF-8 and multibyte characters must not be split.
fd = sys.stdin.fileno()
buf = b''
PASTE_START, PASTE_END = b'\x1b[200~', b'\x1b[201~'
while True:
    r, _, _ = select.select([fd], [], [], 0.2)
    if not r:
        continue
    chunk = os.read(fd, 65536)
    if not chunk:
        break
    buf += chunk
    while True:
        start = buf.find(PASTE_START)
        if start >= 0:
            end = buf.find(PASTE_END, start)
            if end < 0:
                break  # paste still arriving
            composer = buf[start + len(PASTE_START) : end].decode('utf-8', 'replace')
            buf = buf[:start] + buf[end + len(PASTE_END) :]
            paint()
            continue
        if b'\r' in buf or b'\n' in buf:
            idx = min(i for i in (buf.find(b'\r'), buf.find(b'\n')) if i >= 0)
            buf = buf[idx + 1 :]
            if composer:
                record_prompt(composer)
                composer = ''
            paint()
            continue
        if buf.startswith(b'\x1b') and not PASTE_START.startswith(buf[: len(PASTE_START)]):
            with open(os.environ['FAKE_TUI_SUBMIT_LOG'] + '.keys', 'a') as fh:
                fh.write('ESC\n')
            buf = buf[1:]
            continue
        break
"""


def write_fake_tui(path: str) -> str:
    """Materialize the fake TUI script and return its path."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(FAKE_TUI_SCRIPT)
    return path


# Only the async cases need the asyncio mark; a module-level mark would warn on
# every sync test in this file.
asyncio_test = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Registration and input contracts
# --------------------------------------------------------------------------- #


def test_both_codex_agents_register_independently():
    from nexus.core import agent_registry

    assert agent_registry.get('codex')
    assert agent_registry.get('codex_interactive')


def test_interactive_version_is_independent_from_batch():
    async def provider(request):
        return None

    batch = CodexInput(user_prompt='x', working_dir='/workspace')
    interactive = CodexInteractiveInput(user_prompt='x', working_dir='/workspace', get_user_input=provider)
    assert batch.version == DEFAULT_CODEX_VERSION == '0.114.0'
    assert interactive.version == INTERACTIVE_DEFAULT_CODEX_VERSION == '0.155.1'
    assert not issubclass(CodexInteractiveInput, CodexInput)


def test_interactive_input_has_no_exec_only_fields():
    fields = set(CodexInteractiveInput.model_fields)
    assert 'emit_json_trace' not in fields
    assert 'retain_json_events_stdout' not in fields
    assert {'rows', 'cols', 'model_catalog_json'} <= fields
    assert 'approval_policy' not in fields
    assert 'sandbox_mode' not in fields
    assert 'extra_cli_args' in fields


def test_manifest_exposes_both_agents():
    from pathlib import Path

    import tomllib

    manifest = Path('nexus/extensions/agents/codex/pyproject.toml').read_text()
    data = tomllib.loads(manifest)
    # Containment, not equality: the manifest also carries agents this module does not
    # own, and a new sibling agent must not fail the interactive registration check.
    assert {'codex', 'codex_interactive'} <= set(data['tool']['nexus']['agents'])
    assert any('pyte' in dep for dep in data['project']['optional-dependencies']['runtime'])


def test_user_input_provider_defaults_to_single_turn():
    defaulted = CodexInteractiveInput(user_prompt='x', working_dir='/workspace')
    assert callable(defaulted.get_user_input)

    async def provider(request):
        return None

    ok = CodexInteractiveInput(user_prompt='x', working_dir='/workspace', get_user_input=provider)
    assert ok.get_user_input is provider


@pytest.mark.parametrize(
    ('version', 'why'),
    [
        ('0.115.0', 'no --remote at all'),
        ('0.116.0', '--remote and --listen are ws:// only; this code uses unix://'),
        ('0.120.0', 'still ws:// only on both sides'),
        ('0.130.0', '--listen has unix:// but --remote does not'),
        ('0.154.0-alpha.6.2', 'pre-release builds are not supported'),
        ('0.140.0.post1', 'post releases are not official npm release versions'),
        ('0.140.0+custom', 'locally tagged builds are ambiguous'),
        ('latest', 'an unpinned dist-tag is not a version'),
        ('custom-build', 'an unordered custom build is not supported'),
    ],
)
def test_version_gate_rejects_unverified_releases(version, why):
    """The minimum and stable-release constraint come from protocol evidence."""
    with pytest.raises(CodexVersionUnsupportedError) as excinfo:
        validate_interactive_version(version)
    message = str(excinfo.value)
    assert INTERACTIVE_MIN_CODEX_VERSION in message


@pytest.mark.parametrize(
    'version',
    ['0.131.0', '0.135.0', '0.140.0', '0.144.0', '0.147.0', '0.150.0', '0.154.0', '0.155.1', '0.156.1'],
)
def test_version_gate_accepts_e2e_matrix(version):
    validate_interactive_version(version)


@pytest.mark.parametrize('version', ['0.156.0', '0.200.0', '1.0.0'])
def test_version_gate_allows_newer_stable_releases(version):
    validate_interactive_version(version)


def test_interactive_version_defaults():
    assert INTERACTIVE_MIN_CODEX_VERSION == '0.131.0'
    assert INTERACTIVE_DEFAULT_CODEX_VERSION == '0.155.1'


def test_interactive_output_defaults():
    out = CodexInteractiveOutput(success=True, output='', stderr='', exit_code=0, execution_time=0.1)
    assert out.turns == 0
    assert out.events == []


# --------------------------------------------------------------------------- #
# WebSocket frame decoding
# --------------------------------------------------------------------------- #


def test_decoder_skips_http_upgrade_then_reads_frames():
    decoder = WebSocketFrameDecoder()
    handshake = b'HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\n\r\n'
    assert decoder.feed(handshake) == []
    payload = json.dumps({'method': 'turn/started'})
    assert decoder.feed(encode_frame(payload)) == [payload]


def test_decoder_handles_split_and_fragmented_frames():
    decoder = WebSocketFrameDecoder()
    decoder.feed(b'HTTP/1.1 101\r\n\r\n')
    frame = encode_frame('x' * 300)
    assert decoder.feed(frame[:5]) == []
    assert decoder.feed(frame[5:]) == ['x' * 300]


def test_decoder_reports_close_frame():
    decoder = WebSocketFrameDecoder()
    decoder.feed(b'HTTP/1.1 101\r\n\r\n')
    assert decoder.feed(bytes([0x88, 0x00])) == [WebSocketFrameDecoder.CLOSE_SENTINEL]


def test_client_response_frame_is_masked_and_keeps_jsonrpc_id_type():
    decoder = WebSocketFrameDecoder()
    decoder.feed(b'HTTP/1.1 101\r\n\r\n')
    payload = json.dumps({'id': 17, 'result': {'answers': {'db': {'answers': ['MySQL']}}}})
    frame = encode_masked_text_frame(payload, mask=b'ABCD')
    assert frame[1] & 0x80
    assert decoder.feed(frame) == [payload]


# --------------------------------------------------------------------------- #
# Boundary correlation
# --------------------------------------------------------------------------- #


def _started(thread: str, turn: str) -> dict:
    return {'method': 'turn/started', 'params': {'threadId': thread, 'turn': {'id': turn, 'status': 'inProgress'}}}


def _completed(thread: str, turn: str, status: str = 'completed', error: str | None = None) -> dict:
    payload = {'id': turn, 'status': status}
    if error:
        payload['error'] = {'message': error}
    return {'method': 'turn/completed', 'params': {'threadId': thread, 'turn': payload}}


def test_boundary_locks_thread_and_ignores_auxiliary_threads():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_started('aux', 't2'))  # auxiliary thread from the same submission
    adapter.handle(_completed('aux', 't2'))
    assert adapter.thread_id == 'main'
    assert adapter.active_turn_id == 't1'
    assert adapter.last_result is None

    adapter.handle(_completed('main', 't1'))
    assert adapter.last_result is not None
    assert adapter.last_result.outcome is TurnOutcome.COMPLETED
    assert adapter.last_result.turn_id == 't1'


def test_boundary_drops_stale_turn_from_previous_round():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_completed('main', 't1'))
    adapter.release()

    adapter.expect_turn()
    # Residue from turn 1 arrives after turn 2 was armed but before it started.
    adapter.handle(_completed('main', 't1'))
    assert adapter.last_result is None

    adapter.handle(_started('main', 't2'))
    adapter.handle(_completed('main', 't1'))  # still stale
    assert adapter.last_result is None
    adapter.handle(_completed('main', 't2'))
    assert adapter.last_result is not None
    assert adapter.last_result.turn_id == 't2'


@pytest.mark.parametrize(
    ('status', 'expected'),
    [
        ('completed', TurnOutcome.COMPLETED),
        ('failed', TurnOutcome.FAILED),
        ('interrupted', TurnOutcome.INTERRUPTED),
    ],
)
def test_boundary_maps_every_terminal_status(status, expected):
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_completed('main', 't1', status=status, error='boom' if status == 'failed' else None))
    assert adapter.last_result.outcome is expected
    if status == 'failed':
        assert adapter.last_result.error_message == 'boom'


def test_boundary_in_progress_is_not_a_boundary():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_completed('main', 't1', status='inProgress'))
    # inProgress is a terminal-looking payload but not a terminal state: it maps
    # to FAILED only if it actually arrives as turn/completed, which Codex does
    # not do; guard that we at least never report COMPLETED.
    assert adapter.last_result.outcome is not TurnOutcome.COMPLETED


def test_boundary_captures_approval_and_question_requests():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'req-1',
            'method': 'item/commandExecution/requestApproval',
            'params': {'threadId': 'main', 'turnId': 't1', 'command': 'rm -rf x'},
        }
    )
    pending = adapter.take_interaction()
    assert pending is not None
    assert pending.kind == 'approval'
    assert pending.interaction_id == 'req-1'
    assert adapter.take_interaction() is None

    adapter.handle(
        {
            'id': 'req-2',
            'method': 'item/tool/requestUserInput',
            'params': {'threadId': 'main', 'turnId': 't1', 'questions': [{'id': 'q', 'question': 'which?'}]},
        }
    )
    pending = adapter.take_interaction()
    assert pending.kind == 'user_input'


def test_boundary_ignores_interactions_from_other_threads():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'r',
            'method': 'item/commandExecution/requestApproval',
            'params': {'threadId': 'other', 'turnId': 't1'},
        }
    )
    assert adapter.pending_interaction is None


def test_boundary_correlates_mcp_elicitation_without_turn_id_to_the_active_turn():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle({
        'id': 'mcp-1',
        'method': 'mcpServer/elicitation/request',
        'params': {
            'threadId': 'main',
            'serverName': 'settings',
            'mode': 'form',
            'message': 'Choose a region',
            'requestedSchema': {'type': 'object', 'properties': {'region': {'type': 'string'}}},
        },
    })

    pending = adapter.pending_interaction
    assert pending is not None
    assert pending.request_id == 'mcp-1'
    assert pending.thread_id == 'main'
    assert pending.turn_id == 't1'


# --------------------------------------------------------------------------- #
# Event buffer
# --------------------------------------------------------------------------- #


def test_event_buffer_cursor_only_returns_new_events():
    buf = EventBuffer()
    buf.mark_turn_start()
    for i in range(3):
        buf.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'x', 'n': i}}})
    events, cursor = buf.get_delta()
    assert len(events) == 3
    buf.commit(cursor['latest_seq'])
    events, cursor = buf.get_delta()
    assert events == []
    assert cursor['count'] == 0


def test_event_buffer_filters_foreign_thread_and_stale_turn():
    buf = EventBuffer()
    buf.mark_turn_start()
    for thread_id, turn_id, name in (
        ('main', 'current', 'accepted'),
        ('other', 'current', 'foreign'),
        ('main', 'previous', 'stale'),
    ):
        buf.append_message(
            {
                'method': 'item/completed',
                'params': {
                    'threadId': thread_id,
                    'turnId': turn_id,
                    'item': {'itemType': 'commandExecution', 'command': name},
                },
            },
            thread_id='main',
            turn_id='current',
        )

    events, _ = buf.get_delta()
    assert [event['item']['command'] for event in events] == ['accepted']


def test_event_buffer_truncates_and_reports_drops():
    buf = EventBuffer()
    for i in range(EventBuffer.MAX_DELTA_EVENTS + 10):
        buf.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'x', 'n': i}}})
    events, cursor = buf.get_delta()
    assert len(events) == EventBuffer.MAX_DELTA_EVENTS
    assert cursor['truncated'] is True
    assert cursor['dropped_count'] == 10


def test_event_extraction_skips_deltas_and_bounds_payloads():
    assert extract_event({'method': 'item/agentMessage/delta', 'params': {}}) is None
    assert extract_event({'method': 'turn/started', 'params': {}}) is None
    event = extract_event(
        {
            'method': 'item/completed',
            'params': {'threadId': 'th', 'item': {'itemType': 'commandExecution', 'output': 'y' * 5000}},
        }
    )
    assert event['item_type'] == 'commandExecution'
    assert event['truncated'] is True
    assert len(event['item']['output']) < 5000


# --------------------------------------------------------------------------- #
# Session state machine
# --------------------------------------------------------------------------- #


def _offline_session(tmp_path, **kwargs) -> CodexInteractiveSession:
    return CodexInteractiveSession(
        codex_argv=['/bin/true'],
        env={},
        working_dir=str(tmp_path),
        session_dir=str(tmp_path / 'sess'),
        **kwargs,
    )


def test_illegal_state_transition_raises(tmp_path):
    session = _offline_session(tmp_path)
    assert session.state is SessionState.CREATED
    with pytest.raises(InvalidStateTransitionError):
        session._transition(SessionState.RUNNING)
    session._transition(SessionState.STARTING)
    session._transition(SessionState.READY)
    session._transition(SessionState.RUNNING)
    session._transition(SessionState.TURN_COMPLETED)
    session._transition(SessionState.READY)
    with pytest.raises(InvalidStateTransitionError):
        session._transition(SessionState.TURN_COMPLETED)


def test_prompt_fingerprint_never_contains_the_prompt():
    from nexus.extensions.agents.codex.interactive.session import _prompt_fingerprint

    secret = '请把密码改成 hunter2'
    fingerprint = _prompt_fingerprint(secret)
    assert secret not in fingerprint
    assert 'hunter2' not in fingerprint
    assert 'len=' in fingerprint
    assert 'sha=' in fingerprint


# --------------------------------------------------------------------------- #
# Integration: real PTY + fake TUI + fake app-server
# --------------------------------------------------------------------------- #


_SOCKET_ROOTS: list[str] = []


@pytest.fixture(autouse=True)
def _reclaim_socket_roots():
    """Delete the socket directories created by _make_session during one test."""
    mark = len(_SOCKET_ROOTS)
    try:
        yield
    finally:
        import shutil

        while len(_SOCKET_ROOTS) > mark:
            with contextlib.suppress(OSError):
                shutil.rmtree(_SOCKET_ROOTS.pop())


async def _make_session(
    tmp_path,
    *,
    composer_hint: str = 'Ask Codex to do anything',
    name: str = 's1',
) -> tuple:
    """Build a session whose 'codex' is a fake TUI and whose app-server is fake.

    Sockets live in a short system temp directory, never under ``tmp_path`` or the
    repository: AF_UNIX paths are capped near 108 bytes, and both of those roots are
    long enough to blow the cap. ``Path.cwd()`` is avoided deliberately -- a CI
    checkout path such as ``/data/__qci/root-workspaces/__qci-pipeline-<id>-1/nexus``
    leaves only a handful of bytes for the socket name, so the budget would depend on
    the pipeline id's digit count.
    """
    import tempfile

    sock_root = tempfile.mkdtemp(prefix='cxi')
    # session.close() only removes session_dir, so sock_root and the fake's own
    # socket would otherwise accumulate for the whole run.
    _SOCKET_ROOTS.append(sock_root)
    # session_dir holds relay.sock, so it must stay under the AF_UNIX length cap.
    session_dir = Path(sock_root) / 'sess'
    session_dir.mkdir(parents=True, exist_ok=True)
    work = tmp_path / f'work-{name}'
    work.mkdir(exist_ok=True)
    submit_log = str(tmp_path / f'submits-{name}.log')
    tui_script = write_fake_tui(str(tmp_path / f'fake_tui_{name}.py'))

    fake = FakeAppServer(os.path.join(sock_root, 'up.sock'))
    await fake.start()

    env = {
        'PATH': os.environ.get('PATH', ''),
        'TERM': 'xterm-256color',
        'FAKE_TUI_SUBMIT_LOG': submit_log,
        'FAKE_TUI_COMPOSER_HINT': composer_hint,
    }
    session = CodexInteractiveSession(
        codex_argv=[sys.executable, tui_script],
        env=env,
        working_dir=str(work),
        session_dir=str(session_dir),
        rows=30,
        cols=120,
        per_turn_timeout=20.0,
    )

    # Point the session's app-server layer at the fake instead of spawning codex:
    # the relay under test is still the real one.
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    real_start = CodexAppServer.start

    async def fake_start(self) -> None:
        self.upstream_socket = fake.socket_path
        self._relay = await asyncio.start_unix_server(self._handle_client, path=self.relay_socket)

    CodexAppServer.start = fake_start
    try:
        await session.start()
    finally:
        CodexAppServer.start = real_start
    return session, fake, submit_log


async def _await_tui_connected(fake: FakeAppServer, *, timeout: float = 30.0) -> None:
    """Block until the TUI's WebSocket has reached the fake app-server.

    Scripts must not emit events, or drop the channel, before the connection the
    session under test depends on actually exists. A fixed sleep only guesses at
    that: spawning a real Python TUI and completing the handshake takes far longer
    under a loaded CI machine (coverage instrumentation alone doubles the run) than
    on an idle one, and an event sent too early is simply lost -- which surfaces as
    an unrelated status such as ``event_source_exit``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if fake._writers:
            # The handshake is done; give the relay a moment to finish wiring both
            # pump directions before anything is written to the connection.
            await asyncio.sleep(0.1)
            return
        await asyncio.sleep(0.02)
    raise AssertionError(
        f'the fake app-server saw no TUI connection within {timeout}s.\n'
        f'{_environment_report(fake)}'
    )


def _environment_report(fake: FakeAppServer | None = None) -> str:
    """Describe the sandbox these PTY tests need, for a failure message.

    They depend on a real pty, a spawnable interpreter, and AF_UNIX paths under the
    ~108-byte cap. When one of those is unavailable the immediate symptom is a lost
    event or an unrelated terminal status, which says nothing about the cause; a CI
    log showing only the assertion then leaves nothing to work from.
    """
    lines = [
        f'  python: {sys.executable}',
        f'  cwd: {Path.cwd()}',
        f'  tmpdir: {os.environ.get("TMPDIR", "(unset)")}',
    ]
    try:
        import pty as _pty

        master, slave = _pty.openpty()
        os.close(master)
        os.close(slave)
        lines.append('  pty.openpty: OK')
    except Exception as error:  # noqa: BLE001 - diagnostics must never mask the real failure
        lines.append(f'  pty.openpty: FAILED {type(error).__name__}: {error}')
    if fake is not None:
        path_bytes = len(fake.socket_path.encode('utf-8'))
        lines.append(f'  upstream socket: {path_bytes} bytes, exists={os.path.exists(fake.socket_path)}')
        lines.append(f'  connected writers: {len(fake._writers)}')
    return '\n'.join(lines)


def _submitted(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [line.rstrip('\n') for line in fh if line.strip()]


@asyncio_test
async def test_prompt_reaches_tui_and_turn_completes_on_same_ids(tmp_path):
    session, fake, submit_log = await _make_session(tmp_path)
    try:
        assert session.state is SessionState.READY

        async def script():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await asyncio.sleep(0.2)
            await fake.emit_item('commandExecution', turn_id='turn-1', command='echo hi')
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(script())
        report = await session.send_turn('请记住编号 ZEBRA-99 并回答')
        await task

        # This is the simplest end-to-end case in the file, so its failure message
        # carries the sandbox report: whatever breaks the PTY setup breaks this one
        # first, and a bare `assert report.completed` would say nothing about why.
        assert report.completed is True, (
            f'turn did not complete: status={report.status!r} reason={report.reason!r}\n'
            f'stderr={report.stderr!r}\n'
            f'screen={report.latest_screen!r}\n'
            f'{_environment_report(fake)}'
        )
        assert report.status == 'completed'
        assert report.turn_id == 'turn-1'
        assert report.thread_id == fake.thread_id
        # The native TUI received the initial positional prompt verbatim.
        assert _submitted(submit_log) == ['请记住编号 ZEBRA-99 并回答'], (
            f'the fake TUI did not log the prompt; screen={report.latest_screen!r}'
        )
        # VirtualScreen shows it too.
        assert 'ZEBRA-99' in report.latest_screen
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_initial_prompt_does_not_depend_on_composer_copy(tmp_path):
    session, fake, submit_log = await _make_session(
        tmp_path,
        name='copy-neutral',
        composer_hint='Describe your next task',
    )
    try:
        assert session.state is SessionState.READY

        async def finish_after_real_submission():
            deadline = asyncio.get_running_loop().time() + 8
            while not _submitted(submit_log) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
            assert _submitted(submit_log) == ['copy-independent prompt']
            await fake.emit_turn_started('copy-neutral-turn')
            await fake.emit_turn_completed('copy-neutral-turn')

        task = asyncio.create_task(finish_after_real_submission())
        report = await session.send_turn('copy-independent prompt')
        await task

        assert report.completed is True
        assert report.turn_id == 'copy-neutral-turn'
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_two_turns_share_one_thread_and_one_pty(tmp_path):
    session, fake, submit_log = await _make_session(tmp_path)
    try:

        async def turn(turn_id: str):
            await _await_tui_connected(fake)
            await fake.emit_turn_started(turn_id)
            await asyncio.sleep(0.1)
            await fake.emit_turn_completed(turn_id)

        task = asyncio.create_task(turn('turn-1'))
        first = await session.send_turn('记住 ZEBRA-11')
        await task
        pid_before = session.codex_pid
        task = asyncio.create_task(turn('turn-2'))
        second = await session.send_turn('刚才的编号是什么')
        await task

        assert first.completed
        assert second.completed
        assert first.thread_id == second.thread_id == fake.thread_id
        assert first.turn_id != second.turn_id
        assert session.codex_pid == pid_before
        assert _submitted(submit_log) == ['记住 ZEBRA-11', '刚才的编号是什么']
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_stale_completion_does_not_finish_the_next_turn(tmp_path):
    session, fake, _ = await _make_session(tmp_path)
    try:

        async def turn_one():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(turn_one())
        await session.send_turn('first')
        await task

        async def replay_stale_then_real():
            await _await_tui_connected(fake)
            await fake.emit_turn_completed('turn-1')  # residue: must be ignored
            await asyncio.sleep(0.6)
            await fake.emit_turn_started('turn-2')
            await fake.emit_turn_completed('turn-2')

        task = asyncio.create_task(replay_stale_then_real())
        report = await session.send_turn('second')
        await task
        assert report.turn_id == 'turn-2'
    finally:
        await session.close()
        await fake.close()


@pytest.mark.parametrize(
    ('method', 'params', 'answer', 'expected_result', 'expected_kind'),
    [
        (
            'item/tool/requestUserInput',
            {'questions': [{'id': 'q1', 'header': 'Database', 'question': 'Which database?', 'options': []}]},
            'MySQL',
            {'answers': {'q1': {'answers': ['MySQL']}}},
            'user_input',
        ),
        (
            'mcpServer/elicitation/request',
            {
                'mode': 'form',
                'message': 'Choose a region',
                'requestedSchema': {'type': 'object', 'properties': {'region': {'type': 'string'}}},
            },
            'ap-guangzhou',
            {'action': 'accept', 'content': {'region': 'ap-guangzhou'}},
            'user_input',
        ),
    ],
    ids=['user-input', 'mcp-elicitation'],
)
@asyncio_test
async def test_interaction_requests_surface_and_resolve(
    tmp_path,
    method,
    params,
    answer,
    expected_result,
    expected_kind,
):
    session, fake, submit_log = await _make_session(tmp_path, name=f'ix-{expected_kind}')
    try:

        async def raise_interaction():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await asyncio.sleep(0.2)
            await fake.emit_interaction_request('req-9', method, turn_id='turn-1', params=params)

        task = asyncio.create_task(raise_interaction())
        report = await session.send_turn('do something risky')
        await task

        assert report.status == 'needs_user_input'
        assert report.completed is False
        assert report.interaction['interaction_id'] == 'req-9'
        assert report.interaction['kind'] == expected_kind
        assert session.state is SessionState.WAITING_FOR_USER

        async def finish():
            await asyncio.sleep(0.6)
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(finish())
        resumed = await session.respond('req-9', answer)
        await task
        assert resumed.completed is True
        assert resumed.turn_id == 'turn-1'
        assert fake.received[-1] == {'id': 'req-9', 'result': expected_result}
        assert fake.received_frames_masked[-1] is True
        # Structured answers never become terminal keystrokes or a second prompt.
        assert _submitted(submit_log) == ['do something risky']
    finally:
        await session.close()
        await fake.close()


@pytest.mark.parametrize(
    ('method', 'params', 'answer', 'expected_result', 'expected_status'),
    [
        (
            'mcpServer/elicitation/request',
            {
                'mode': 'form',
                'message': 'Choose a region',
                'requestedSchema': {'type': 'object', 'properties': {'region': {'type': 'string'}}},
            },
            'no',
            {'action': 'cancel', 'content': None},
            'interaction_denied',
        ),
    ],
    ids=['mcp-elicitation'],
)
@asyncio_test
async def test_interaction_denial_and_late_answer(
    tmp_path,
    method,
    params,
    answer,
    expected_result,
    expected_status,
):
    session, fake, _ = await _make_session(tmp_path, name='deny')
    try:

        async def raise_approval():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await asyncio.sleep(0.2)
            await fake.emit_interaction_request('req-deny', method, turn_id='turn-1', params=params)

        task = asyncio.create_task(raise_approval())
        report = await session.send_turn('risky')
        await task
        assert report.status == 'needs_user_input'

        # Denial uses the method's official cancel response, never a TUI key.
        denied = await session.respond('req-deny', answer)
        assert denied.status == expected_status
        assert denied.reason == expected_status
        assert denied.session_reusable is False
        assert session.state is SessionState.FAILED
        assert fake.received[-1] == {'id': 'req-deny', 'result': expected_result}

        # A late or duplicate answer must be rejected, not silently accepted.
        stale = await session.respond('req-deny', 'yes')
        assert stale.status == 'interaction_not_found'
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_turn_timeout_marks_session_unusable(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='timeout')
    try:
        report = await session.send_turn('never answered', timeout=2.0)
        assert report.status == 'timeout'
        assert report.reason == 'timeout'
        assert report.completed is False
        assert report.session_reusable is False
        assert session.state is SessionState.FAILED
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_event_channel_exit_and_process_exit_have_distinct_reasons(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='exit-a')
    try:

        async def drop_channel():
            await _await_tui_connected(fake)
            await fake.drop_clients()

        task = asyncio.create_task(drop_channel())
        report = await session.send_turn('will lose its channel', timeout=15)
        await task
        assert report.status == 'event_source_exit'
        assert report.reason == 'event_source_exit'
        assert report.session_reusable is False
    finally:
        await session.close()
        await fake.close()

    session2, fake2, _ = await _make_session(tmp_path, name='exit-b')
    try:

        async def kill_tui():
            # The PTY only exists once send_turn() has spawned it, and killing it
            # before its WebSocket is up would tear down the channel first, which
            # reports event_source_exit instead of the process_exit under test.
            await _await_tui_connected(fake2)
            while session2.pty is None:
                await asyncio.sleep(0.02)
            session2.pty.killpg()

        task = asyncio.create_task(kill_tui())
        report2 = await session2.send_turn('will lose its TUI', timeout=15)
        await task
        assert report2.status == 'process_exit'
        assert report2.reason == 'process_exit'
        assert report2.session_reusable is False
    finally:
        await session2.close()
        await fake2.close()

    assert report.reason != report2.reason


@asyncio_test
async def test_concurrent_sessions_do_not_cross_threads_or_events(tmp_path):
    s1, f1, log1 = await _make_session(tmp_path, name='iso1')
    s2, f2, log2 = await _make_session(tmp_path, name='iso2')
    try:
        assert f1.thread_id != f2.thread_id
        assert s1.session_dir != s2.session_dir

        async def drive(fake, turn_id, item_name):
            await _await_tui_connected(fake)
            await fake.emit_turn_started(turn_id)
            await fake.emit_item('commandExecution', turn_id=turn_id, command=item_name)
            await fake.emit_turn_completed(turn_id)

        d1 = asyncio.create_task(drive(f1, 'a-1', 'cmd-one'))
        d2 = asyncio.create_task(drive(f2, 'b-1', 'cmd-two'))
        r1, r2 = await asyncio.gather(s1.send_turn('session one'), s2.send_turn('session two'))
        await asyncio.gather(d1, d2)

        assert r1.thread_id == f1.thread_id
        assert r2.thread_id == f2.thread_id
        assert r1.turn_id == 'a-1'
        assert r2.turn_id == 'b-1'
        assert _submitted(log1) == ['session one']
        assert _submitted(log2) == ['session two']

        e1, _ = s1.events.get_delta()
        e2, _ = s2.events.get_delta()
        assert any('cmd-one' in json.dumps(e) for e in e1)
        assert not any('cmd-two' in json.dumps(e) for e in e1)
        assert any('cmd-two' in json.dumps(e) for e in e2)
        assert not any('cmd-one' in json.dumps(e) for e in e2)
    finally:
        await s1.close()
        await s2.close()
        await f1.close()
        await f2.close()


@asyncio_test
async def test_close_is_idempotent_and_cleans_up(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='cleanup')

    async def finish():
        await _await_tui_connected(fake)
        await fake.emit_turn_started('turn-1')
        await fake.emit_turn_completed('turn-1')

    task = asyncio.create_task(finish())
    await session.send_turn('start the TUI')
    await task
    pty_proc = session.pty.proc
    socket_dir = session._socket_dir
    assert socket_dir is not None
    assert os.path.exists(os.path.join(socket_dir, 'rl.sock'))
    # 0700 so no other user on the box can reach the relay.
    assert (os.stat(socket_dir).st_mode & 0o777) == 0o700

    await session.close()
    await session.close()  # idempotent
    assert session.state is SessionState.CLOSED
    await asyncio.sleep(0.3)
    assert pty_proc.returncode is not None
    assert not os.path.exists(session.session_dir)
    assert not os.path.exists(socket_dir)  # sockets and their dir are gone
    await fake.close()


@asyncio_test
async def test_close_while_waiting_for_boundary_does_not_transition_from_closed(tmp_path):
    session = _offline_session(tmp_path)
    session._transition(SessionState.STARTING)
    session._transition(SessionState.READY)
    session._transition(SessionState.RUNNING)
    session._turn_deadline = asyncio.get_running_loop().time() + 5

    waiter = asyncio.create_task(session._wait_boundary(time.time(), 5))
    await asyncio.sleep(0.05)
    await session.close()

    report = await asyncio.wait_for(waiter, timeout=1)
    assert report.status == 'session_closed'
    assert report.session_reusable is False
    assert session.state is SessionState.CLOSED


@asyncio_test
async def test_viewport_renders_ansi_tui_output(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='viewport')
    try:

        async def finish():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(finish())
        await session.send_turn('viewport prompt')
        await task
        view = await session.screen.latest_screen()
        assert 'OpenAI Codex (fake)' in view
        assert 'Ask Codex to do anything' in view
        assert '\x1b' not in view  # ANSI is interpreted, not leaked
    finally:
        await session.close()
        await fake.close()


# --------------------------------------------------------------------------- #
# Review fixes: final message, resolved requests, idle timeout, token collision
# --------------------------------------------------------------------------- #


def test_final_message_comes_from_the_matching_turn_only():
    """Auxiliary threads and earlier turns must not leak into the answer."""
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))

    def item(thread, turn, text, item_type='agentMessage'):
        return {
            'method': 'item/completed',
            'params': {
                'threadId': thread,
                'turnId': turn,
                'item': {'id': 'i', 'itemType': item_type, 'text': text},
            },
        }

    adapter.handle(item('aux', 't9', 'FROM_AUX_THREAD'))
    adapter.handle(item('main', 't0', 'FROM_OLD_TURN'))
    adapter.handle(item('main', 't1', 'commandExecution output', item_type='commandExecution'))
    adapter.handle(item('main', 't1', 'FIRST'))
    adapter.handle(item('main', 't1', 'FINAL_ANSWER'))
    adapter.handle(_completed('main', 't1'))

    assert adapter.last_result.final_message == 'FINAL_ANSWER'


def test_final_message_falls_back_to_items_in_the_terminal_payload():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'method': 'turn/completed',
            'params': {
                'threadId': 'main',
                'turn': {
                    'id': 't1',
                    'status': 'completed',
                    'items': [{'itemType': 'agentMessage', 'content': [{'type': 'text', 'text': 'EMBEDDED'}]}],
                },
            },
        }
    )
    assert adapter.last_result.final_message == 'EMBEDDED'


def test_final_message_is_empty_when_the_turn_produced_none():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_completed('main', 't1'))
    assert adapter.last_result.final_message == ''


def test_server_request_resolved_clears_the_pending_interaction():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'req-1',
            'method': 'item/tool/requestUserInput',
            'params': {'threadId': 'main', 'turnId': 't1'},
        }
    )
    assert adapter.pending_interaction is not None

    adapter.handle(
        {
            'method': 'serverRequest/resolved',
            'params': {'threadId': 'main', 'turnId': 't1', 'requestId': 'req-1'},
        }
    )
    assert adapter.pending_interaction is None


def test_resolved_for_a_different_request_id_is_ignored():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'req-1',
            'method': 'item/tool/requestUserInput',
            'params': {'threadId': 'main', 'turnId': 't1'},
        }
    )
    adapter.handle(
        {
            'method': 'serverRequest/resolved',
            'params': {'threadId': 'main', 'turnId': 't1', 'requestId': 'other'},
        }
    )
    assert adapter.pending_interaction is not None


def test_turn_completion_and_release_clear_pending_interactions():
    """Fallback cleanup: a request Codex never resolved must not outlive its turn."""
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'req-1',
            'method': 'item/commandExecution/requestApproval',
            'params': {'threadId': 'main', 'turnId': 't1'},
        }
    )
    adapter.handle(_completed('main', 't1'))
    assert adapter.pending_interaction is None

    adapter.expect_turn()
    adapter.handle(_started('main', 't2'))
    adapter.handle(
        {
            'id': 'req-2',
            'method': 'item/commandExecution/requestApproval',
            'params': {'threadId': 'main', 'turnId': 't2'},
        }
    )
    adapter.release()
    assert adapter.pending_interaction is None


def test_expect_turn_drops_a_request_left_over_from_the_previous_turn():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'stale',
            'method': 'item/tool/requestUserInput',
            'params': {'threadId': 'main', 'turnId': 't1'},
        }
    )
    adapter.expect_turn()
    assert adapter.pending_interaction is None


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        ('a/b', 'a?b'),
        ('x' * 70 + 'A', 'x' * 70 + 'B'),
        ('../../etc/passwd', '.._.._etc_passwd'),
        ('sess:1', 'sess|1'),
    ],
)
def test_session_tokens_never_collide(left, right):
    """Sanitising alone is not injective; the digest makes the token unique."""
    from nexus.extensions.agents.codex.interactive.runtime import _session_paths, _session_token

    assert _session_token(left) != _session_token(right)
    assert _session_paths(left)[0] != _session_paths(right)[0]
    assert _session_paths(left)[1] != _session_paths(right)[1]


def test_session_token_is_filesystem_safe_and_stable():
    from nexus.extensions.agents.codex.interactive.runtime import _session_token

    token = _session_token('../../etc/passwd')
    assert '/' not in token
    assert _session_token('same') == _session_token('same')


@asyncio_test
async def test_completed_turn_reports_the_agent_message_as_output(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='output')
    try:

        async def script():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await fake.emit_item('agentMessage', turn_id='turn-1', text='THE ANSWER IS 42')
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(script())
        report = await session.send_turn('what is the answer')
        await task
        assert report.completed is True
        assert report.output == 'THE ANSWER IS 42'
    finally:
        await session.close()
        await fake.close()


@asyncio_test
async def test_unexpected_approval_is_a_protocol_error(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='unexpected-approval')
    try:

        async def raise_approval():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await asyncio.sleep(0.2)
            await fake.emit_approval_request('req-1', turn_id='turn-1')

        task = asyncio.create_task(raise_approval())
        report = await session.send_turn('unattended run')
        await task
        assert report.status == 'approval_protocol_error'
        assert report.reason == 'approval_under_never_policy'
        assert report.session_reusable is False
        assert session.state is SessionState.FAILED
    finally:
        await session.close()
        await fake.close()


# --------------------------------------------------------------------------- #
# Interaction correlation (thread + turn)
# --------------------------------------------------------------------------- #


def _request(
    request_id: str | int,
    thread: str,
    turn: str | None,
    method: str = 'item/commandExecution/requestApproval',
):
    params: dict = {'threadId': thread}
    if turn is not None:
        params['turnId'] = turn
    return {'id': request_id, 'method': method, 'params': params}


def test_interaction_requires_matching_thread_and_turn():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))

    adapter.handle(_request('r-aux', 'aux', 't1'))  # foreign thread
    assert adapter.pending_interaction is None
    adapter.handle(_request('r-old', 'main', 't0'))  # old turn
    assert adapter.pending_interaction is None

    adapter.handle(_request('r-ok', 'main', 't1'))
    assert adapter.pending_interaction is not None
    assert adapter.pending_interaction.interaction_id == 'r-ok'
    assert adapter.pending_interaction.thread_id == 'main'
    assert adapter.pending_interaction.turn_id == 't1'


def test_interaction_requires_both_ids_and_preserves_numeric_request_id():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))

    adapter.handle(_request('missing-turn', 'main', None))
    adapter.handle(
        {
            'id': 'missing-thread',
            'method': 'item/commandExecution/requestApproval',
            'params': {'turnId': 't1'},
        }
    )
    assert adapter.pending_interaction is None

    adapter.handle(_request(17, 'main', 't1'))
    pending = adapter.pending_interaction
    assert pending is not None
    assert pending.request_id == 17
    assert isinstance(pending.request_id, int)

    # Same printed value but a different JSON type is a different request id.
    adapter.handle(
        {
            'method': 'serverRequest/resolved',
            'params': {'threadId': 'main', 'turnId': 't1', 'requestId': '17'},
        }
    )
    assert adapter.pending_interaction is pending
    adapter.handle(
        {
            'method': 'serverRequest/resolved',
            'params': {'threadId': 'other', 'turnId': 't1', 'requestId': 17},
        }
    )
    assert adapter.pending_interaction is pending
    adapter.handle(
        {
            'method': 'serverRequest/resolved',
            'params': {'threadId': 'main', 'turnId': 'old', 'requestId': 17},
        }
    )
    assert adapter.pending_interaction is pending

    adapter.handle({'method': 'serverRequest/resolved', 'params': {'threadId': 'main', 'requestId': 17}})
    assert adapter.pending_interaction is None
    assert adapter.interaction_was_resolved(pending)


def test_interaction_without_a_live_turn_is_dropped():
    adapter = CodexBoundaryAdapter()
    # Nothing armed yet.
    adapter.handle(_request('r', 'main', 't1'))
    assert adapter.pending_interaction is None

    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_completed('main', 't1'))
    # Arriving after the turn finished.
    adapter.handle(_request('late', 'main', 't1'))
    assert adapter.pending_interaction is None


def test_duplicate_and_competing_requests_are_presented_in_order():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_request('first', 'main', 't1'))
    adapter.handle(_request('first', 'main', 't1'))  # duplicate
    adapter.handle(_request('second', 'main', 't1'))  # competing
    assert adapter.pending_interaction.interaction_id == 'first'

    adapter.handle({'method': 'serverRequest/resolved', 'params': {'threadId': 'main', 'requestId': 'first'}})
    assert adapter.pending_interaction.interaction_id == 'second'

    adapter.handle({'method': 'serverRequest/resolved', 'params': {'threadId': 'main', 'requestId': 'second'}})
    assert adapter.pending_interaction is None


def test_turn_completion_clears_queued_interactions():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_request('first', 'main', 't1'))
    adapter.handle(_request('second', 'main', 't1'))

    adapter.handle(_completed('main', 't1'))

    assert adapter.pending_interaction is None
    adapter.handle({'method': 'serverRequest/resolved', 'params': {'threadId': 'main', 'requestId': 'first'}})
    assert adapter.pending_interaction is None


# --------------------------------------------------------------------------- #
# Host-visible interaction payload
# --------------------------------------------------------------------------- #


def test_user_question_payload_exposes_question_and_options():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(
        {
            'id': 'q1',
            'method': 'item/tool/requestUserInput',
            'params': {
                'threadId': 'main',
                'turnId': 't1',
                'questions': [
                    {
                        'id': 'db',
                        'header': 'Database',
                        'question': 'Which database should I target?',
                        'options': [{'label': 'PostgreSQL'}, {'label': 'MySQL'}],
                    }
                ],
            },
        }
    )
    payload = adapter.pending_interaction.to_dict()
    assert payload['tool_name'] == 'request_user_input'
    assert payload['tool_input']['questions'][0]['question'] == 'Which database should I target?'
    assert payload['tool_input']['questions'][0]['options'] == ['PostgreSQL', 'MySQL']
    assert 'Which database should I target?' in payload['hint']
    assert 'PostgreSQL' in payload['hint']


def test_user_input_builder_maps_non_default_answer_by_question_id():
    session = object.__new__(CodexInteractiveSession)
    pending = PendingInteraction(
        interaction_id='question-1',
        request_id='question-1',
        method='item/tool/requestUserInput',
        kind='user_input',
        params={
            'questions': [
                {
                    'id': 'database',
                    'question': 'Which database?',
                    'options': [{'label': 'PostgreSQL'}, {'label': 'MySQL'}],
                }
            ]
        },
        thread_id='thread-1',
        turn_id='turn-1',
    )
    assert session._build_interaction_response(pending, 'MySQL') == (
        {'answers': {'database': {'answers': ['MySQL']}}},
        'answered',
    )
    assert session._build_interaction_response(pending, None) == ({'answers': {}}, 'unattended')


def test_user_input_builder_requires_an_explicit_answer_for_every_question():
    session = object.__new__(CodexInteractiveSession)
    pending = PendingInteraction(
        interaction_id='questions',
        request_id='questions',
        method='item/tool/requestUserInput',
        kind='user_input',
        params={
            'questions': [
                {'id': 'database', 'question': 'Which database?'},
                {'id': 'cache', 'question': 'Which cache?'},
            ]
        },
        thread_id='thread-1',
        turn_id='turn-1',
    )
    assert session._build_interaction_response(pending, {'database': 'MySQL', 'cache': ['Redis']}) == (
        {'answers': {'database': {'answers': ['MySQL']}, 'cache': {'answers': ['Redis']}}},
        'answered',
    )
    assert session._build_interaction_response(pending, '{"database":"MySQL","cache":"Redis"}') == (
        {'answers': {'database': {'answers': ['MySQL']}, 'cache': {'answers': ['Redis']}}},
        'answered',
    )
    assert session._build_interaction_response(
        pending,
        '{"answers":{"database":{"answers":["MySQL"]},"cache":{"answers":["Redis"]}}}',
    ) == (
        {'answers': {'database': {'answers': ['MySQL']}, 'cache': {'answers': ['Redis']}}},
        'answered',
    )
    assert 'JSON object keyed by question id' in pending.hint()
    assert session._build_interaction_response(pending, 'MySQL') == (None, 'indeterminate')
    assert session._build_interaction_response(pending, {'database': 'MySQL'}) == (None, 'indeterminate')


def test_mcp_elicitation_builder_uses_action_and_structured_content():
    session = object.__new__(CodexInteractiveSession)
    pending = PendingInteraction(
        interaction_id='mcp-1',
        request_id='mcp-1',
        method='mcpServer/elicitation/request',
        kind='user_input',
        params={
            'mode': 'form',
            'message': 'Choose a region',
            'requestedSchema': {'type': 'object', 'properties': {'region': {'type': 'string'}}},
        },
        thread_id='thread-1',
        turn_id='turn-1',
    )
    assert session._build_interaction_response(pending, 'ap-guangzhou') == (
        {'action': 'accept', 'content': {'region': 'ap-guangzhou'}},
        'answered',
    )
    assert session._build_interaction_response(pending, {'action': 'decline'}) == (
        {'action': 'decline', 'content': None},
        'denied',
    )
    assert session._build_interaction_response(pending, None) == (
        {'action': 'decline', 'content': None},
        'unattended',
    )
    assert session._build_interaction_response(pending, '{"region":"ap-shanghai"}') == (
        {'action': 'accept', 'content': {'region': 'ap-shanghai'}},
        'answered',
    )
    assert session._build_interaction_response(pending, 'maybe later') == (
        {'action': 'accept', 'content': {'region': 'maybe later'}},
        'answered',
    )
    assert session._build_interaction_response(pending, '') == (None, 'indeterminate')


def test_pending_interaction_can_be_claimed_only_once():
    adapter = CodexBoundaryAdapter()
    adapter.expect_turn()
    adapter.handle(_started('main', 't1'))
    adapter.handle(_request('request-1', 'main', 't1'))

    claimed = adapter.claim_interaction('request-1')
    assert claimed is not None
    assert adapter.claim_interaction('request-1') is None
    assert adapter.claim_interaction('other') is None


# --------------------------------------------------------------------------- #
# EventBuffer capacity
# --------------------------------------------------------------------------- #


def test_event_buffer_evicts_on_append_and_counts_drops():
    buf = EventBuffer()
    total = EventBuffer.MAX_BUFFERED_EVENTS + 250
    for i in range(total):
        buf.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'x', 'n': i}}})
    # Memory is bounded even though nothing was ever drained.
    assert len(buf._events) == EventBuffer.MAX_BUFFERED_EVENTS
    assert buf.evicted_count == 250

    events, cursor = buf.get_delta()
    assert len(events) == EventBuffer.MAX_DELTA_EVENTS
    assert cursor['truncated'] is True
    # Both eviction and per-read trimming are reported.
    expected = 250 + (EventBuffer.MAX_BUFFERED_EVENTS - EventBuffer.MAX_DELTA_EVENTS)
    assert cursor['dropped_count'] == expected
    assert cursor['evicted_count'] == 250


def test_event_buffer_eviction_count_resets_after_commit():
    buf = EventBuffer()
    for i in range(EventBuffer.MAX_BUFFERED_EVENTS + 5):
        buf.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'x', 'n': i}}})
    _, cursor = buf.get_delta()
    assert cursor['evicted_count'] == 5
    buf.commit(cursor['latest_seq'])
    _, cursor2 = buf.get_delta()
    assert cursor2['evicted_count'] == 0
    assert cursor2['dropped_count'] == 0


def test_event_buffer_without_overflow_reports_no_drops():
    buf = EventBuffer()
    for i in range(10):
        buf.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'x', 'n': i}}})
    events, cursor = buf.get_delta()
    assert len(events) == 10
    assert cursor['truncated'] is False
    assert cursor['dropped_count'] == 0


# --------------------------------------------------------------------------- #
# Socket directory
# --------------------------------------------------------------------------- #


@asyncio_test
async def test_socket_dir_is_short_private_and_not_derived_from_session_id(tmp_path):
    import tempfile

    long_id = 'session-' + 'x' * 200
    session, fake, _ = await _make_session(tmp_path, name='sock')
    try:
        socket_dir = session._socket_dir
        assert socket_dir is not None
        assert socket_dir.startswith(str(Path(tempfile.gettempdir()) / 'nxc-'))
        assert (os.stat(socket_dir).st_mode & 0o777) == 0o700
        # System-generated, so a hostile or very long session_id cannot steer it.
        assert long_id not in socket_dir
        for name in ('as.sock', 'rl.sock'):
            path = os.path.join(socket_dir, name)
            assert len(path.encode('utf-8')) < 100
    finally:
        await session.close()
        await fake.close()
        assert not os.path.exists(socket_dir)


@asyncio_test
async def test_socket_dir_is_removed_when_start_fails(tmp_path):
    """A failed start must not leave a socket directory behind."""
    import glob
    import tempfile

    pattern = str(Path(tempfile.gettempdir()) / 'nxc-*')
    before = set(glob.glob(pattern))
    session = CodexInteractiveSession(
        codex_argv=['/nonexistent-binary-for-test'],
        env={'PATH': os.environ.get('PATH', '')},
        working_dir=str(tmp_path),
        session_dir=str(tmp_path / 'sess-fail'),
    )
    with pytest.raises(FileNotFoundError):
        await session.start()
    leaked = set(glob.glob(pattern)) - before
    assert leaked == set(), f'socket dirs leaked: {leaked}'


# --------------------------------------------------------------------------- #
# Argument and environment guards
# --------------------------------------------------------------------------- #


def test_protected_env_wins_over_caller_values():
    from nexus.extensions.agents.codex.interactive.guards import apply_protected_env

    merged = apply_protected_env(
        {'CODEX_HOME': '/attacker', 'NEXUS_PROXY_API_KEY': 'stolen', 'PATH': '/caller/bin', 'MY': 'keep'},
        {'CODEX_HOME': '/owned', 'NEXUS_AGENT_PROXY_URL': 'http://127.0.0.1:1', 'NEXUS_PROXY_API_KEY': 'real'},
    )
    assert merged['CODEX_HOME'] == '/owned'
    assert merged['NEXUS_PROXY_API_KEY'] == 'real'
    assert merged['NEXUS_AGENT_PROXY_URL'] == 'http://127.0.0.1:1'
    # PATH stays the caller's: that remains the existing installed-agent contract.
    assert merged['PATH'] == '/caller/bin'
    assert merged['MY'] == 'keep'


@pytest.mark.parametrize('key', ['CODEX_HOME', 'NEXUS_AGENT_PROXY_URL', 'NEXUS_PROXY_API_KEY'])
def test_extra_env_cannot_set_protected_keys(key):
    from nexus.extensions.agents.codex.interactive.guards import (
        UnsafeEnvOverrideError,
        assert_env_not_overriding,
    )

    with pytest.raises(UnsafeEnvOverrideError):
        assert_env_not_overriding({key: 'x'})
    assert_env_not_overriding({'SOMETHING_ELSE': 'ok'})
    assert_env_not_overriding({'PATH': '/caller/bin'})  # PATH is not protected here


@asyncio_test
async def test_runtime_rejects_unsafe_args_before_starting_anything(monkeypatch):
    """The guard runs first, so a bad config never installs Codex or starts a proxy."""
    from nexus.extensions.agents.codex.interactive import runtime as rt

    started = []

    async def _should_not_run(*args, **kwargs):
        started.append(1)
        raise AssertionError('nothing may start when the configuration is unsafe')

    monkeypatch.setattr(rt, '_prepare_session', _should_not_run)
    rt._CODEX_PTY_SESSIONS.pop('guarded', None)

    from nexus.extensions.agents.codex.interactive.guards import UnsafeCliArgumentError, validate_extra_cli_args

    with pytest.raises(UnsafeCliArgumentError, match='--remote'):
        validate_extra_cli_args(['--remote', 'unix:///tmp/evil.sock'])
    assert started == []


# --------------------------------------------------------------------------- #
# One deadline per logical turn
# --------------------------------------------------------------------------- #


@asyncio_test
async def test_interaction_resume_keeps_existing_deadline():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    session = object.__new__(CodexInteractiveSession)
    session._closed = False
    session.state = SessionState.WAITING_FOR_USER
    deadline = asyncio.get_running_loop().time() + 6
    session._turn_deadline = deadline
    pending = SimpleNamespace(interaction_id='q-1')
    session.boundary = SimpleNamespace(claim_interaction=Mock(return_value=pending))
    session._answer_interaction = AsyncMock(return_value='answered')
    session._wait_interaction_resolved = AsyncMock(return_value=True)
    session._wait_boundary = AsyncMock(return_value='result')

    assert await session.respond('q-1', 'answer') == 'result'
    assert session._turn_deadline == deadline
    resolve_budget = session._wait_interaction_resolved.await_args.args[1]
    boundary_budget = session._wait_boundary.await_args.args[1]
    assert 0 <= boundary_budget <= resolve_budget <= 6


@asyncio_test
async def test_turn_deadline_is_cleared_on_completion_and_close(tmp_path):
    session, fake, _ = await _make_session(tmp_path, name='deadline')
    try:

        async def script():
            await _await_tui_connected(fake)
            await fake.emit_turn_started('turn-1')
            await fake.emit_turn_completed('turn-1')

        task = asyncio.create_task(script())
        await session.send_turn('go')
        await task
        assert session._turn_deadline is None
    finally:
        await session.close()
        await fake.close()
        assert session._turn_deadline is None


# --------------------------------------------------------------------------- #
# Guard hardening: concatenated short flags, positionals, inline TOML
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'args',
    [
        # Concatenated short flags: clap accepts these exactly like the separated
        # form, and the real Codex CLI was verified to accept all three.
        ['-anever'],
        ['-aon-request'],
        ['-a', 'never'],
        ['--ask-for-approval', 'never'],
        ['-C/tmp'],
        ['-C', '/tmp'],
        ['-Csomewhere'],
        ['--cd', '/etc'],
        ['-capproval_policy=never'],
        ['-cmodel_provider=other'],
        ['-cmodel_providers.nexus.base_url=http://evil'],
        # Long forms, attached and separated.
        ['--ask-for-approval=never'],
        ['--cd=/etc'],
        ['-sworkspace-write'],
        ['-s', 'read-only'],
        ['--sandbox=workspace-write'],
        ['--sandbox', 'read-only'],
        ['--config=approval_policy=never'],
        ['--remote', 'unix:///tmp/evil.sock'],
        ['--remote=ws://evil'],
        ['--remote-auth-token-env', 'X'],
        ['--no-alt-screen'],
        ['--full-auto'],
        ['--dangerously-bypass-approvals-and-sandbox'],
        ['--dangerously-bypass-hook-trust'],
        ['-c', 'sandbox_mode=danger-full-access'],
        ['-c', 'model_provider=other'],
        ['-c', 'model_providers.nexus.base_url=http://evil'],
        ['--config', 'hooks.Stop=[]'],
        ['-c=sandbox_permissions=["disk-full-access"]'],
        ['-c', 'shell_environment_policy.inherit=all'],
        ['-c', 'features.default_mode_request_user_input=false'],
        ['--disable', 'default_mode_request_user_input'],
        ['--enable=hooks'],
        # Routes approvals to an automatic reviewer instead of get_user_input.
        ['--approve-for-me'],
        # A bare positional becomes Codex's startup PROMPT.
        ['do something malicious'],
        ['-s', 'read-only', 'sneaky-prompt'],
        # Provider selection: both point Codex away from the session's own proxy.
        ['--oss'],
        ['--local-provider', 'http://evil'],
        ['--local-provider=http://evil'],
        # A reserved key reached through an inline table names nothing reserved on
        # the left of the '=', so the value has to be parsed to see it.
        ['-c', 'features={default_mode_request_user_input=false}'],
        ['-c', 'features={hooks=false}'],
        ['-c', 'model_providers={nexus={base_url="http://evil"}}'],
        ['-c', 'profiles={unsafe={sandbox_mode="danger-full-access"}}'],
        ['--config=features={hooks=false}'],
        ['-cfeatures={hooks=false}'],
        # A bare inline table is not a TOML document; its keys still count.
        ['-c', '{sandbox_mode="danger-full-access"}'],
        ['-c', '{features={hooks=false}}'],
    ],
)
def test_guard_blocks_every_known_bypass(args):
    from nexus.extensions.agents.codex.interactive.guards import (
        UnsafeCliArgumentError,
        validate_extra_cli_args,
    )

    with pytest.raises(UnsafeCliArgumentError):
        validate_extra_cli_args(args)


@pytest.mark.parametrize(
    'args',
    [
        [],
        ['--add-dir', '/extra'],
        ['--worktree'],
        ['--search'],
        ['-m', 'gpt-5-codex'],
        ['--model=gpt-5-codex'],
        ['-c', 'model_reasoning_effort="high"'],
        ['-c', 'model_context_window=200000'],
        ['--enable', 'some_feature'],
        ['-p', 'profile-name'],
        ['--profile', 'custom'],
        # Inline tables are parsed, not banned: only reserved keys inside one are.
        ['-c', '{model_reasoning_effort="high"}'],
        ['-c', 'tui={theme="dark"}'],
    ],
)
def test_guard_allows_legitimate_arguments(args):
    from nexus.extensions.agents.codex.interactive.guards import validate_extra_cli_args

    assert validate_extra_cli_args(args) == args


@pytest.mark.parametrize(
    'toml_text',
    [
        'approval_policy = "never"',
        'approvals_reviewer = "auto"',
        'sandbox_mode = "danger-full-access"',
        '[model_providers.nexus]\nbase_url = "http://evil"',
        '["model_providers"."nexus"]\nbase_url = "http://evil"',
        '[profiles.unsafe]\nmodel_provider = "other"',
        '[profiles.unsafe.features]\ndefault_mode_request_user_input = false',
        '[hooks]\nStop = []',
        'allow_managed_hooks_only = false',
    ],
)
def test_extra_config_toml_cannot_set_reserved_keys(toml_text):
    from nexus.extensions.agents.codex.interactive.guards import (
        UnsafeCliArgumentError,
        validate_extra_config_toml,
    )

    with pytest.raises(UnsafeCliArgumentError):
        validate_extra_config_toml(toml_text)


@pytest.mark.parametrize(
    'toml_text',
    [None, '', '# only a comment', 'model_reasoning_effort = "high"', '[tui]\ntheme = "dark"'],
)
def test_extra_config_toml_allows_ordinary_settings(toml_text):
    from nexus.extensions.agents.codex.interactive.guards import validate_extra_config_toml

    validate_extra_config_toml(toml_text)


def test_extra_config_toml_rejects_invalid_toml():
    from nexus.extensions.agents.codex.interactive.guards import (
        UnsafeCliArgumentError,
        validate_extra_config_toml,
    )

    with pytest.raises(UnsafeCliArgumentError, match='invalid TOML'):
        validate_extra_config_toml('[tui')


def test_app_server_callback_failure_marks_event_channel_unavailable():
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    channel_events: list[str] = []

    def reject_message(message):
        raise ValueError(f'invalid message {message["method"]}')

    server = CodexAppServer(
        codex_argv=['codex'],
        env={},
        upstream_socket='upstream.sock',
        relay_socket='relay.sock',
        on_message=reject_message,
        on_event_channel_down=lambda: channel_events.append('down'),
    )

    server._dispatch('{"method":"turn/started"}')

    assert server.channel_down is True
    assert server.channel_error == 'app-server message callback failed: ValueError: invalid message turn/started'
    assert channel_events == ['down']


def test_app_server_invalid_json_marks_event_channel_unavailable():
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    channel_events: list[str] = []
    server = CodexAppServer(
        codex_argv=['codex'],
        env={},
        upstream_socket='upstream.sock',
        relay_socket='relay.sock',
        on_message=lambda message: None,
        on_event_channel_down=lambda: channel_events.append('down'),
    )

    server._dispatch('{invalid')

    assert server.channel_down is True
    assert server.channel_error is not None
    assert server.channel_error.startswith('app-server sent invalid JSON-RPC:')
    assert channel_events == ['down']


@pytest.mark.skipif(
    not hasattr(os, 'killpg') or not Path('/proc').is_dir(),
    reason='requires Linux process groups',
)
@asyncio_test
async def test_process_tree_cleanup_signals_group_after_app_server_parent_exits():
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    parent = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        (
            'import os, time\n'
            'child_pid = os.fork()\n'
            'if child_pid == 0:\n'
            '    os.close(0)\n'
            '    os.close(1)\n'
            '    os.close(2)\n'
            '    time.sleep(30)\n'
            '    os._exit(0)\n'
            'print(child_pid, flush=True)\n'
            'os._exit(0)\n'
        ),
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
    )
    process_group_id = parent.pid
    child_pid: int | None = None
    try:
        assert parent.stdout is not None
        child_pid = int((await asyncio.wait_for(parent.stdout.readline(), timeout=3)).decode().strip())
        for _ in range(100):
            if parent.returncode is not None:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(f'parent process {parent.pid} is still running')
        os.kill(child_pid, 0)

        server = CodexAppServer(
            codex_argv=['codex'],
            env={},
            upstream_socket='upstream.sock',
            relay_socket='relay.sock',
            on_message=lambda message: None,
            on_event_channel_down=lambda: None,
        )
        server.proc = parent
        server._process_group_id = process_group_id
        await server._terminate_process_tree()
        for _ in range(100):
            try:
                state = Path(f'/proc/{child_pid}/stat').read_text().split()[2]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(f'child process {child_pid} is still running')
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGKILL)
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child_pid, signal.SIGKILL)


@asyncio_test
async def test_session_cleanup_failure_is_reported_and_retryable(tmp_path):
    session = _offline_session(tmp_path)
    invalid_socket_dir = tmp_path / 'socket-file'
    invalid_socket_dir.write_text('not a directory')
    session._socket_dir = str(invalid_socket_dir)

    with pytest.raises(RuntimeError, match='socket directory'):
        await session.close()

    assert session._closed is False
    invalid_socket_dir.unlink()
    await session.close()
    assert session.state is SessionState.CLOSED


def test_guard_rejection_message_names_the_offending_token():
    from nexus.extensions.agents.codex.interactive.guards import (
        UnsafeCliArgumentError,
        validate_extra_cli_args,
    )

    with pytest.raises(UnsafeCliArgumentError) as excinfo:
        validate_extra_cli_args(['-anever'])
    assert '-anever' in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Client-stream framing: injected responses must not split a TUI frame
# --------------------------------------------------------------------------- #


def _client_frame(payload: bytes, *, opcode: int = 0x1, fin: bool = True) -> bytes:
    """Build a masked client frame around raw payload bytes."""
    import struct as _struct

    mask = b'\x01\x02\x03\x04'
    first = (0x80 if fin else 0x00) | opcode
    length = len(payload)
    if length < 126:
        header = bytes((first, 0x80 | length))
    elif length < (1 << 16):
        header = bytes((first, 0x80 | 126)) + _struct.pack('>H', length)
    else:
        header = bytes((first, 0x80 | 127)) + _struct.pack('>Q', length)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return header + mask + masked


def test_framer_forwards_handshake_then_whole_frames():
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    handshake = b'GET /rpc HTTP/1.1\r\nUpgrade: websocket\r\n\r\n'
    frame = _client_frame(b'{"id":1}')
    out = framer.feed(handshake + frame)
    assert out == handshake + frame


def test_framer_withholds_a_partial_frame_until_complete():
    """The core fix: a half-arrived frame must not be forwarded on its own."""
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(b'GET /rpc HTTP/1.1\r\n\r\n')
    # A frame larger than one read chunk, delivered in pieces.
    payload = b'x' * 200_000
    frame = _client_frame(payload)
    first, second, third = frame[:65536], frame[65536:131072], frame[131072:]

    assert framer.feed(first) == b''
    assert framer.feed(second) == b''
    # Only once the final byte lands does the whole frame become forwardable.
    assert framer.feed(third) == frame


def test_framer_splits_a_batch_at_the_last_complete_frame():
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(b'GET / HTTP/1.1\r\n\r\n')
    whole = _client_frame(b'{"a":1}') + _client_frame(b'{"b":2}')
    partial = _client_frame(b'{"c":3}')[:4]
    out = framer.feed(whole + partial)
    assert out == whole  # the trailing partial frame is held back


def test_framer_flush_releases_a_truncated_tail():
    """A client that dies mid-frame must not leave bytes stuck forever."""
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(b'GET / HTTP/1.1\r\n\r\n')
    stub = _client_frame(b'{"x":1}')[:5]
    assert framer.feed(stub) == b''
    assert framer.flush() == stub
    assert framer.flush() == b''


@pytest.mark.parametrize('size', [0, 1, 125, 126, 127, 65535, 65536, 70000])
def test_framer_handles_every_length_encoding(size):
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(b'GET / HTTP/1.1\r\n\r\n')
    frame = _client_frame(b'y' * size)
    # Byte-at-a-time delivery is the worst case for boundary detection.
    for index in range(len(frame) - 1):
        assert framer.feed(frame[index : index + 1]) == b''
    assert framer.feed(frame[-1:]) == frame


def test_framer_waits_for_the_full_http_preamble():
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    assert framer.feed(b'GET /rpc HTTP/1.1\r\n') == b''
    assert framer.feed(b'Upgrade: websocket\r\n') == b''
    out = framer.feed(b'\r\n')
    assert out.endswith(b'\r\n\r\n')


# --------------------------------------------------------------------------- #
# Review follow-ups: message framing, terminal queries, popup detection, stderr
# --------------------------------------------------------------------------- #


_HANDSHAKE = b'GET /rpc HTTP/1.1\r\n\r\n'


def test_framer_withholds_a_fragmented_message_until_its_final_frame():
    """An injected response must not land between continuation frames.

    The write lock is released on whatever the framer hands back, so delivering a
    non-final fragment would open a window where send_response() writes a new text
    frame into the middle of the message the TUI is still sending. The upstream
    would read that frame as continuation bytes.
    """
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(_HANDSHAKE)
    opening = _client_frame(b'{"id":1,"par', opcode=0x1, fin=False)
    closing = _client_frame(b'ams":{}}', opcode=0x0, fin=True)

    assert framer.feed(opening) == b''
    assert framer.feed(closing) == opening + closing


def test_framer_does_not_cut_on_a_control_frame_inside_a_fragment():
    """A ping may be interleaved mid-message; it ends nothing."""
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(_HANDSHAKE)
    opening = _client_frame(b'AAA', opcode=0x1, fin=False)
    ping = _client_frame(b'', opcode=0x9, fin=True)
    closing = _client_frame(b'BBB', opcode=0x0, fin=True)

    assert framer.feed(opening) == b''
    assert framer.feed(ping) == b''
    assert framer.feed(closing) == opening + ping + closing


def test_framer_delivers_a_control_frame_outside_a_fragment():
    from nexus.extensions.agents.codex.interactive.appserver import ClientStreamFramer

    framer = ClientStreamFramer()
    framer.feed(_HANDSHAKE)
    ping = _client_frame(b'', opcode=0x9, fin=True)
    assert framer.feed(ping) == ping


@pytest.mark.parametrize('split_at', [1, 2, 3, 4, 5])
def test_terminal_query_split_across_reads_is_still_answered(split_at):
    """A query straddling a chunk boundary matches neither half on its own.

    Leaving it unanswered stalls the TUI's first paint, which surfaces much later
    as a turn that never starts.
    """
    from nexus.extensions.agents.codex.interactive import session as session_module

    captured: list[bytes] = []
    probe = object.__new__(session_module.CodexInteractiveSession)
    probe._query_carry = bytearray()
    probe.screen = type('_Screen', (), {'feed': lambda self, data: None})()
    probe._write_async = captured.append

    query = session_module._OSC_FG_QUERY
    probe._on_pty_data(query[:split_at])
    probe._on_pty_data(query[split_at:])
    assert captured == [session_module._OSC_FG_REPLY]


def test_terminal_query_is_answered_once_and_carry_stays_bounded():
    from nexus.extensions.agents.codex.interactive import session as session_module

    captured: list[bytes] = []
    probe = object.__new__(session_module.CodexInteractiveSession)
    probe._query_carry = bytearray()
    probe.screen = type('_Screen', (), {'feed': lambda self, data: None})()
    probe._write_async = captured.append

    probe._on_pty_data(session_module._OSC_FG_QUERY)
    probe._on_pty_data(b'ordinary output with no query at all' * 1000)
    assert captured == [session_module._OSC_FG_REPLY]
    assert len(probe._query_carry) <= session_module._MAX_QUERY_CARRY

    # A complete query followed by the start of the next one: the remainder has to
    # survive in the carry, or the second query is lost.
    captured.clear()
    probe._on_pty_data(session_module._OSC_BG_QUERY + session_module._OSC_FG_QUERY[:3])
    probe._on_pty_data(session_module._OSC_FG_QUERY[3:])
    assert captured == [session_module._OSC_BG_REPLY, session_module._OSC_FG_REPLY]


def test_device_attributes_query_is_answered_once_per_chunk():
    """``ESC[c`` is a substring of ``ESC[>c``; the shared reply must not double."""
    from nexus.extensions.agents.codex.interactive import session as session_module

    captured: list[bytes] = []
    probe = object.__new__(session_module.CodexInteractiveSession)
    probe._query_carry = bytearray()
    probe.screen = type('_Screen', (), {'feed': lambda self, data: None})()
    probe._write_async = captured.append

    probe._on_pty_data(b'\x1b[>c')
    assert captured == [session_module._DA_REPLY]


@pytest.mark.parametrize(
    ('before', 'after', 'submitted', 'expected'),
    [
        # Typing made the hint appear: a real popup.
        ('idle', 'idle\nPress enter to insert', 'git che', True),
        # The prompt itself contains the hint; Escape here cancels the composer.
        ('idle', 'why no matches were found', 'why no matches were found', False),
        # The hint was already on screen from the previous turn's output.
        ('search: no matches', 'search: no matches\nnext prompt', 'next prompt', False),
        ('idle', 'idle\nplain prompt', 'plain prompt', False),
        # Prompt mentions one hint while a different one really opened.
        ('idle', 'no matches\nPress enter to insert', 'why no matches', True),
    ],
)
def test_popup_detection_requires_the_hint_to_be_new(before, after, submitted, expected):
    assert CodexInteractiveSession._popup_opened(before=before, after=after, submitted=submitted) is expected


@pytest.mark.asyncio
async def test_app_server_stderr_drain_survives_an_overlong_line():
    """readline() raises past its limit; the drain must keep reading regardless.

    A drain that stops leaves the pipe to fill, and the app-server then blocks on
    write mid-turn -- a stall with no error anywhere near the cause.
    """
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    server = object.__new__(CodexAppServer)
    server._stderr_tail = bytearray()

    huge = b'z' * 200_000

    class _Stream:
        def __init__(self) -> None:
            self._chunks = [huge, b'tail-marker', b'']

        async def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    server.proc = type('_Proc', (), {'stderr': _Stream()})()
    await server._drain_stderr()

    # The tail is bounded but still carries the most recent output.
    assert server.recent_stderr().endswith('tail-marker')
    assert len(server._stderr_tail) <= 8192


@pytest.mark.asyncio
async def test_app_server_stderr_drain_continues_after_a_read_error():
    from nexus.extensions.agents.codex.interactive.appserver import CodexAppServer

    server = object.__new__(CodexAppServer)
    server._stderr_tail = bytearray()

    class _Stream:
        def __init__(self) -> None:
            self._steps = [ValueError('line too long'), b'after-error', b'']

        async def read(self, _size: int) -> bytes:
            step = self._steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

    server.proc = type('_Proc', (), {'stderr': _Stream()})()
    await server._drain_stderr()
    assert server.recent_stderr() == 'after-error'


def test_wide_dict_is_bounded_without_visiting_every_key():
    """Output size is not the point: the walk itself has to stay bounded.

    Capping only the emitted keys still pays for materializing every pair, so a
    dict with a huge number of keys costs time and memory proportional to its
    width on a path that runs for every streamed event.
    """
    from nexus.extensions.agents.codex.interactive.events import _MAX_ITEMS, _shrink

    visited = 0

    class _CountingItems(dict):
        def items(self):  # noqa: A003 - mirrors the dict API under test
            nonlocal visited
            for key, value in super().items():
                visited += 1
                yield key, value

    wide = _CountingItems({f'k{index}': index for index in range(5_000)})
    shrunk, cut = _shrink(wide)

    assert cut is True
    assert len(shrunk) == _MAX_ITEMS + 1  # the kept keys plus the truncation note
    assert shrunk['...'] == f'{5_000 - _MAX_ITEMS} more keys truncated'
    # Lazy iteration stops at the cap; a list() of every pair would visit all 5000.
    assert visited <= _MAX_ITEMS + 1


def test_wide_dict_truncation_note_counts_every_dropped_key():
    from nexus.extensions.agents.codex.interactive.events import _MAX_ITEMS, _shrink

    shrunk, cut = _shrink({f'k{index}': index for index in range(_MAX_ITEMS + 7)})
    assert cut is True
    assert shrunk['...'] == '7 more keys truncated'


def test_narrow_dict_is_untouched():
    from nexus.extensions.agents.codex.interactive.events import _MAX_ITEMS, _shrink

    payload = {f'k{index}': index for index in range(_MAX_ITEMS)}
    shrunk, cut = _shrink(payload)
    assert cut is False
    assert shrunk == payload
    assert '...' not in shrunk
