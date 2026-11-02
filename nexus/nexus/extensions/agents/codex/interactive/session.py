"""PTY driver and state machine for one interactive Codex session.

The first prompt is passed to the native TUI as its positional CLI argument;
later prompts are typed through the PTY. Nothing calls ``turn/start`` directly,
so the TUI remains the thing actually driving Codex. Turn boundaries come from
the app-server notifications the relay taps out of the TUI's own traffic.

One behaviour here was established by probing the real TUI and is load-bearing:

* Some inputs open a completion popup where Enter means "insert completion", not
  "submit". Submitting therefore checks the viewport and dismisses the popup first;
  a blind Escape is avoided because with nothing to dismiss it can exit the TUI.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from nexus.runtime.executors.terminal.pty_io import PtyProcess

from .appserver import CodexAppServer
from .boundary import CodexBoundaryAdapter, PendingInteraction, TurnOutcome
from .events import EventBuffer
from .screen import CodexScreen


logger = logging.getLogger(__name__)

# Terminal capability queries the TUI issues at startup. Leaving them unanswered
# stalls rendering. CPR (ESC[6n) is deliberately NOT answered: its reply is written
# back into the same PTY and Codex's composer consumes it as literal text.
_OSC_FG_QUERY = b'\x1b]10;?'
_OSC_BG_QUERY = b'\x1b]11;?'
_OSC_FG_REPLY = b'\x1b]10;rgb:e6e6/eded/f3f3\x1b\\'
_OSC_BG_REPLY = b'\x1b]11;rgb:1717/1717/2121\x1b\\'
_DA_REPLY = b'\x1b[?1;2c'
_DSR_REPLY = b'\x1b[0n'

# Query -> reply, scanned against each PTY chunk plus a short carry-over prefix.
_TERMINAL_QUERY_REPLIES: tuple[tuple[bytes, bytes], ...] = (
    (_OSC_FG_QUERY, _OSC_FG_REPLY),
    (_OSC_BG_QUERY, _OSC_BG_REPLY),
    (b'\x1b[c', _DA_REPLY),
    (b'\x1b[>c', _DA_REPLY),
    (b'\x1b[5n', _DSR_REPLY),
)
# One byte short of the longest query: enough to complete a split sequence, and
# never enough to re-match a query that was already answered in full.
_MAX_QUERY_CARRY = max(len(query) for query, _ in _TERMINAL_QUERY_REPLIES) - 1

_POPUP_HINTS = ('Press enter to insert', 'no matches')
_INTERACTION_RESOLVE_TIMEOUT = 5.0
_BETWEEN_TURNS_SETTLE = 1.5

# Linux caps sun_path at 108 bytes including the terminator; stay clear of it.
_MAX_UNIX_SOCKET_BYTES = 100


class SessionState(str, Enum):
    CREATED = 'CREATED'
    STARTING = 'STARTING'
    READY = 'READY'
    RUNNING = 'RUNNING'
    WAITING_FOR_USER = 'WAITING_FOR_USER'
    TURN_COMPLETED = 'TURN_COMPLETED'
    FAILED = 'FAILED'
    EXITED = 'EXITED'
    CLOSED = 'CLOSED'


_ALLOWED: dict[SessionState, set[SessionState]] = {
    SessionState.CREATED: {SessionState.STARTING, SessionState.CLOSED},
    SessionState.STARTING: {SessionState.READY, SessionState.FAILED, SessionState.CLOSED},
    SessionState.READY: {
        SessionState.RUNNING,
        SessionState.EXITED,
        SessionState.FAILED,
        SessionState.CLOSED,
    },
    SessionState.RUNNING: {
        SessionState.WAITING_FOR_USER,
        SessionState.TURN_COMPLETED,
        SessionState.FAILED,
        SessionState.EXITED,
        SessionState.CLOSED,
    },
    SessionState.WAITING_FOR_USER: {
        SessionState.RUNNING,
        SessionState.FAILED,
        SessionState.EXITED,
        SessionState.CLOSED,
    },
    SessionState.TURN_COMPLETED: {SessionState.READY, SessionState.CLOSED, SessionState.FAILED},
    SessionState.FAILED: {SessionState.CLOSED},
    SessionState.EXITED: {SessionState.CLOSED},
    SessionState.CLOSED: set(),
}


class InvalidStateTransitionError(RuntimeError):
    """Raised instead of silently tolerating an impossible transition."""


ELICITATION_ACCEPT_WORDS = frozenset({'yes', 'y', 'approve', 'approved', 'allow', 'ok', 'accept', 'grant', 'true'})
ELICITATION_DENY_WORDS = frozenset({'no', 'n', 'deny', 'denied', 'reject', 'decline', 'quit', 'abort', 'false'})


@dataclass
class TurnReport:
    """Outcome of one ``send_turn`` / ``respond`` call."""

    status: str
    completed: bool
    reason: str
    thread_id: str | None = None
    turn_id: str | None = None
    output: str = ''
    stderr: str = ''
    latest_screen: str = ''
    elapsed: float = 0.0
    exit_code: int | None = None
    interaction: dict[str, Any] | None = None
    session_reusable: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = {
            'status': self.status,
            'completed': self.completed,
            'reason': self.reason,
            'thread_id': self.thread_id,
            'turn_id': self.turn_id,
            'output': self.output,
            'stderr': self.stderr,
            'latest_screen': self.latest_screen,
            'elapsed': self.elapsed,
            'session_reusable': self.session_reusable,
        }
        if self.exit_code is not None:
            data['exit_code'] = self.exit_code
        if self.interaction is not None:
            data['interaction'] = self.interaction
        return data


def _prompt_fingerprint(prompt: str) -> str:
    """Length + hash only: prompts must never be logged verbatim."""
    digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]
    return f'len={len(prompt)} sha={digest}'


class CodexInteractiveSession:
    """One Codex TUI behind a PTY, plus its private app-server and relay."""

    def __init__(
        self,
        *,
        codex_argv: list[str],
        env: dict[str, str],
        working_dir: str,
        session_dir: str,
        rows: int = 50,
        cols: int = 200,
        per_turn_timeout: float = 3600.0,
        extra_cli_args: list[str] | None = None,
        owns_session_dir: bool = True,
    ) -> None:
        self._codex_argv = list(codex_argv)
        self._env = dict(env)
        self.working_dir = working_dir
        self.session_dir = session_dir
        self.rows = rows
        self.cols = cols
        self.per_turn_timeout = per_turn_timeout
        self.extra_cli_args = list(extra_cli_args or [])
        self.owns_session_dir = owns_session_dir

        self.state = SessionState.CREATED
        self.screen = CodexScreen(rows=rows, cols=cols)
        self.boundary = CodexBoundaryAdapter()
        self.events = EventBuffer()
        self.app_server: CodexAppServer | None = None
        self.pty: PtyProcess | None = None

        self._turn_index = 0
        self._tui_exited = asyncio.Event()
        self._channel_down = False
        self._pending_writes: set[asyncio.Task] = set()
        self._closed = False
        self._closing = False
        self._close_lock = asyncio.Lock()
        # One deadline per logical turn, spanning every interaction inside it.
        self._turn_deadline: float | None = None
        self._socket_dir: str | None = None
        # Trailing PTY bytes that may be the start of a split terminal query.
        self._query_carry = bytearray()

    # ----- state machine ---------------------------------------------------- #

    def _transition(self, target: SessionState) -> None:
        if target == self.state:
            return
        if target not in _ALLOWED[self.state]:
            raise InvalidStateTransitionError(f'illegal session transition {self.state.value} -> {target.value}')
        logger.info('[codex-pty] state %s -> %s', self.state.value, target.value)
        self.state = target

    @property
    def codex_pid(self) -> int | None:
        return self.pty.proc.pid if self.pty is not None else None

    # ----- lifecycle -------------------------------------------------------- #

    async def start(self) -> None:
        """Start the app-server and relay; the first turn starts the TUI."""
        self._transition(SessionState.STARTING)
        os.makedirs(self.session_dir, exist_ok=True)
        # Sockets live in a short, mode-0700, system-generated directory. AF_UNIX
        # paths are capped near 108 bytes, and a session_id-derived path both risks
        # exceeding that and makes the name predictable.
        self._socket_dir = tempfile.mkdtemp(prefix='nxc-')
        os.chmod(self._socket_dir, 0o700)
        upstream = os.path.join(self._socket_dir, 'as.sock')
        relay = os.path.join(self._socket_dir, 'rl.sock')
        for path in (upstream, relay):
            encoded = len(path.encode('utf-8'))
            if encoded >= _MAX_UNIX_SOCKET_BYTES:
                raise RuntimeError(
                    f'unix socket path is {encoded} bytes, over the {_MAX_UNIX_SOCKET_BYTES}-byte limit: {path}'
                )
        self.app_server = CodexAppServer(
            codex_argv=self._codex_argv,
            env=self._env,
            upstream_socket=upstream,
            relay_socket=relay,
            on_message=self._on_app_server_message,
            on_event_channel_down=self._on_channel_down,
        )
        try:
            await self.app_server.start()
            self._transition(SessionState.READY)
        except BaseException:
            self.state = SessionState.FAILED
            await self.close()
            raise

    async def _spawn_tui(self, prompt: str) -> None:
        """Start the native TUI with its first prompt as a positional argument."""
        if self.app_server is None:
            raise RuntimeError('Codex app-server is not running')
        argv = [
            *self._codex_argv,
            '--remote',
            f'unix://{self.app_server.relay_socket}',
            '-C',
            self.working_dir,
            '-a',
            'never',
            '--sandbox',
            'workspace-write',
            '--no-alt-screen',
            *self.extra_cli_args,
            '--',
            prompt,
        ]
        self.pty = await PtyProcess.spawn(
            argv=argv,
            env=self._env,
            rows=self.rows,
            cols=self.cols,
            on_data=self._on_pty_data,
            on_exit=self._on_pty_exit,
        )

    async def close(self) -> None:
        """Idempotent teardown of TUI, relay, app-server, PTY, and session dir."""
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            errors: list[tuple[str, Exception]] = []
            try:
                self._turn_deadline = None
                self.boundary.clear_interactions()
                pending = list(self._pending_writes)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                self._pending_writes.clear()
                if self.pty is not None:
                    pty_failed = False
                    try:
                        await self.pty.terminate_descendants(grace_seconds=1.0)
                    except Exception as error:
                        errors.append(('PTY descendants', error))
                        pty_failed = True
                    for stage, operation in (('PTY process group', self.pty.killpg), ('PTY', self.pty.close)):
                        try:
                            operation()
                        except Exception as error:
                            errors.append((stage, error))
                            pty_failed = True
                    if not pty_failed:
                        self.pty = None
                if self.app_server is not None:
                    try:
                        await self.app_server.close()
                    except Exception as error:
                        errors.append(('app-server', error))
                    else:
                        self.app_server = None
                try:
                    self.screen.close()
                except Exception as error:
                    errors.append(('screen', error))
                if self._socket_dir is not None and os.path.exists(self._socket_dir):
                    try:
                        shutil.rmtree(self._socket_dir)
                    except OSError as error:
                        errors.append(('socket directory', error))
                    else:
                        self._socket_dir = None
                if self.owns_session_dir and os.path.exists(self.session_dir):
                    try:
                        shutil.rmtree(self.session_dir)
                    except OSError as error:
                        errors.append(('session directory', error))
                if errors:
                    details = '; '.join(f'{stage}: {error}' for stage, error in errors)
                    raise RuntimeError(f'Codex interactive session cleanup failed: {details}') from errors[0][1]
                self._closed = True
                self.state = SessionState.CLOSED
            finally:
                self._closing = False

    # ----- callbacks -------------------------------------------------------- #

    def _on_app_server_message(self, message: dict[str, Any]) -> None:
        self.boundary.handle(message)
        scope = self.boundary.live_turn_scope()
        if scope is None:
            return
        self.events.append_message(
            message,
            thread_id=scope[0],
            turn_id=scope[1],
        )

    def _on_channel_down(self) -> None:
        self._channel_down = True

    def _on_pty_data(self, data: bytes) -> None:
        self.screen.feed(data)
        # A query can straddle two reads, and a chunk boundary inside one would
        # otherwise match neither half, leaving the TUI waiting on a reply it needs
        # before it paints. Scan a short carry-over prefix together with this chunk.
        scan = bytes(self._query_carry) + data
        reply = b''
        consumed_through = 0
        answered: set[bytes] = set()
        for query, answer in _TERMINAL_QUERY_REPLIES:
            position = scan.rfind(query)
            if position < 0:
                continue
            consumed_through = max(consumed_through, position + len(query))
            # ``ESC[c`` is a substring of ``ESC[>c``; both ask for the device
            # attributes, so the shared reply must be sent once.
            if answer not in answered:
                reply += answer
                answered.add(answer)
        if reply:
            self._write_async(reply)
        # Carry only what follows the last answered query: a chunk can hold both a
        # complete query and the start of the next one, and dropping the remainder
        # would lose the second. Bounded so a stream without queries cannot grow it.
        self._query_carry = bytearray(scan[consumed_through:][-_MAX_QUERY_CARRY:])

    def _on_pty_exit(self) -> None:
        self._tui_exited.set()

    def _write_async(self, data: bytes) -> None:
        if self.pty is None or not data:
            return
        task = asyncio.create_task(self._safe_write(data))
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

    async def _safe_write(self, data: bytes) -> None:
        if self.pty is None:
            return
        with contextlib.suppress(OSError):
            await self.pty.write(data)

    # ----- input ------------------------------------------------------------ #

    async def _type_and_submit(self, text: str) -> None:
        """Paste one follow-up prompt and press Enter once, like Claude Code."""
        if self.pty is None:
            raise RuntimeError('PTY is not running')
        # Snapshot before typing: a hint string that is already on screen -- or that
        # the prompt itself contains -- is not evidence of a popup. Matching it would
        # send an Escape that cancels the composer, or exits the TUI outright.
        before = await self.screen.latest_screen()
        await self.pty.write(b'\x1b[200~' + text.encode('utf-8') + b'\x1b[201~')
        # Give the TUI time to repaint before reading the viewport: some inputs
        # open a completion popup where Enter means "insert completion" rather
        # than "submit", so the popup has to be dismissed first. A blind Escape
        # is avoided -- with nothing to dismiss it can exit the TUI.
        await asyncio.sleep(0.25)
        after = await self.screen.latest_screen()
        if self._popup_opened(before=before, after=after, submitted=text):
            logger.debug('[codex-pty] completion popup open; dismissing before submit')
            await self.pty.write(b'\x1b')
            await asyncio.sleep(0.25)
        await self.pty.write(b'\r')

    @staticmethod
    def _popup_opened(*, before: str, after: str, submitted: str) -> bool:
        """Whether typing opened a completion popup.

        A hint only counts when typing made it appear: present beforehand it belongs
        to the previous turn's output, and present in the prompt it is just echoed
        text in the composer.
        """
        for hint in _POPUP_HINTS:
            if hint not in after:
                continue
            if hint in before or hint in submitted:
                continue
            return True
        return False

    # ----- turns ------------------------------------------------------------ #

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> TurnReport:
        """Start the TUI with the first prompt, or type a follow-up through PTY."""
        if self._closed or self._closing:
            raise RuntimeError('session is closed')
        if self.state not in (SessionState.READY, SessionState.TURN_COMPLETED):
            raise InvalidStateTransitionError(f'cannot submit a prompt from state {self.state.value}')
        return await self._submit_and_wait(prompt, timeout)

    async def _submit_and_wait(self, prompt: str, timeout: float | None) -> TurnReport:
        """Submit one prompt through the native TUI and wait for its boundary."""
        self._turn_index += 1
        started = time.time()
        logger.info('[codex-pty] turn %s submit (%s)', self._turn_index, _prompt_fingerprint(prompt))
        if self.state is SessionState.TURN_COMPLETED:
            self._transition(SessionState.READY)
        budget = timeout if timeout is not None else self.per_turn_timeout
        # One budget spans the complete logical turn and any interaction in it.
        self._turn_deadline = asyncio.get_running_loop().time() + budget
        self.events.mark_turn_start()
        self.boundary.expect_turn()
        self._transition(SessionState.RUNNING)
        if self.pty is None:
            await self._spawn_tui(prompt)
        else:
            await asyncio.sleep(min(_BETWEEN_TURNS_SETTLE, self._remaining_budget()))
            await self._type_and_submit(prompt)
        return await self._wait_boundary(started, self._remaining_budget())

    def _remaining_budget(self, override: float | None = None) -> float:
        """Seconds left on this turn's deadline (never negative)."""
        if override is not None:
            return override
        if self._turn_deadline is None:
            return self.per_turn_timeout
        return max(0.0, self._turn_deadline - asyncio.get_running_loop().time())

    async def respond(self, interaction_id: str, answer: str | None, *, timeout: float | None = None) -> TurnReport:
        """Answer a structured request through JSON-RPC."""
        if self._closed:
            raise RuntimeError('session is closed')
        started = time.time()
        pending = self.boundary.claim_interaction(interaction_id)
        if pending is None:
            return TurnReport(
                status='interaction_not_found',
                completed=False,
                reason='interaction_not_found',
                latest_screen=await self.screen.latest_screen(),
                elapsed=time.time() - started,
            )
        verdict = await self._answer_interaction(pending, answer)
        if verdict == 'indeterminate':
            self._transition(SessionState.FAILED)
            return await self._report(
                status='interaction_unanswered',
                completed=False,
                reason='user_input_unanswered',
                started=started,
                stderr=f'no unambiguous answer was supplied for {pending.method}',
                reusable=False,
            )

        if verdict in {'answered', 'unattended'} and self.state is SessionState.WAITING_FOR_USER:
            self._transition(SessionState.RUNNING)
        resolved = await self._wait_interaction_resolved(pending, self._remaining_budget(timeout))
        if not resolved:
            self._transition(SessionState.FAILED)
            return await self._report(
                status='interaction_response_unconfirmed',
                completed=False,
                reason='interaction_response_unconfirmed',
                started=started,
                stderr=f'Codex did not resolve request {pending.interaction_id} after its JSON-RPC response',
                reusable=False,
            )

        if verdict == 'denied':
            self._transition(SessionState.FAILED)
            return await self._report(
                status='interaction_denied',
                completed=False,
                reason='interaction_denied',
                started=started,
                stderr=f'{pending.method} was cancelled through the Codex app-server protocol',
                reusable=False,
            )
        # Resume on the ORIGINAL turn budget: answering must not hand the turn a
        # fresh full timeout, or a chain of interactions could run unbounded.
        return await self._wait_boundary(started, self._remaining_budget(timeout))

    async def _answer_interaction(self, pending: PendingInteraction, answer: object) -> str:
        """Build the method-specific result and send it over JSON-RPC."""
        if self.app_server is None:
            raise RuntimeError('Codex app-server is not running')
        result, verdict = self._build_interaction_response(pending, answer)
        if result is None:
            return verdict
        await self.app_server.send_response(pending.request_id, result)
        return verdict

    def _build_interaction_response(
        self,
        pending: PendingInteraction,
        answer: object,
    ) -> tuple[dict[str, Any] | None, str]:
        """Return a Codex response result and its local verdict."""
        method = pending.method
        if answer is None:
            if method == 'item/tool/requestUserInput':
                return {'answers': {}}, 'unattended'
            if method == 'mcpServer/elicitation/request':
                return {'action': 'decline', 'content': None}, 'unattended'
        if method == 'item/tool/requestUserInput':
            result = self._build_user_input_response(pending, answer)
            return (result, 'answered') if result is not None else (None, 'indeterminate')
        if method == 'mcpServer/elicitation/request':
            return self._build_mcp_elicitation_response(pending, answer)
        return None, 'indeterminate'

    @staticmethod
    def _build_user_input_response(pending: PendingInteraction, answer: object) -> dict[str, Any] | None:
        questions = pending.params.get('questions')
        if not isinstance(questions, list) or not questions:
            return None
        question_ids: list[str] = []
        for question in questions:
            if not isinstance(question, dict):
                return None
            question_data: dict[str, Any] = dict(question)
            question_id = question_data.get('id')
            if not isinstance(question_id, str):
                return None
            question_ids.append(question_id)

        supplied: object = answer
        if isinstance(answer, dict):
            answer_data = cast(dict[str, Any], answer)
            if isinstance(answer_data.get('answers'), dict):
                supplied = answer_data['answers']
        if isinstance(supplied, str):
            text = supplied.strip()
            if not text:
                return None
            if len(question_ids) == 1:
                values: dict[str, object] = {question_ids[0]: text}
            else:
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    return None
                if not isinstance(decoded, dict):
                    return None
                nested_answers = decoded.get('answers')
                values = dict(nested_answers) if isinstance(nested_answers, dict) else decoded
        elif isinstance(supplied, dict):
            values = dict(supplied)
        else:
            return None

        encoded: dict[str, dict[str, list[str]]] = {}
        for question_id in question_ids:
            value = values.get(question_id)
            if isinstance(value, dict):
                nested = cast(dict[str, Any], value)
                value = nested.get('answers')
            if isinstance(value, str):
                answers = [value.strip()] if value.strip() else []
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                string_values = [item for item in value if isinstance(item, str)]
                answers = [item.strip() for item in string_values if item.strip()]
            else:
                answers = []
            if not answers:
                return None
            encoded[question_id] = {'answers': answers}
        return {'answers': encoded}

    @staticmethod
    def _build_mcp_elicitation_response(
        pending: PendingInteraction,
        answer: object,
    ) -> tuple[dict[str, Any] | None, str]:
        if isinstance(answer, dict):
            answer_data = cast(dict[str, Any], answer)
            if answer_data.get('action') in {'accept', 'decline', 'cancel'}:
                action = str(answer_data['action'])
                content = answer_data.get('content') if action == 'accept' else None
                return {'action': action, 'content': content}, 'answered' if action == 'accept' else 'denied'
        if answer is None:
            return None, 'indeterminate'
        if isinstance(answer, str):
            text = answer.strip()
            if not text:
                return None, 'indeterminate'
            if text.casefold() in ELICITATION_DENY_WORDS:
                return {'action': 'cancel', 'content': None}, 'denied'
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    return {'action': 'accept', 'content': decoded}, 'answered'
            schema = pending.params.get('requestedSchema')
            properties = schema.get('properties') if isinstance(schema, dict) else None
            if isinstance(properties, dict) and len(properties) == 1:
                field = next(iter(properties))
                return {'action': 'accept', 'content': {field: text}}, 'answered'
            if pending.params.get('mode') == 'url' and text.casefold() in ELICITATION_ACCEPT_WORDS:
                return {'action': 'accept', 'content': None}, 'answered'
            return None, 'indeterminate'
        if isinstance(answer, dict):
            return {'action': 'accept', 'content': dict(answer)}, 'answered'
        return None, 'indeterminate'

    async def _wait_interaction_resolved(self, pending: PendingInteraction, remaining: float) -> bool:
        """Wait briefly for Codex's exact ``serverRequest/resolved`` acknowledgement."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(_INTERACTION_RESOLVE_TIMEOUT, max(0.0, remaining))
        while loop.time() < deadline:
            if self.boundary.interaction_was_resolved(pending):
                return True
            if self._channel_down or self._tui_exited.is_set():
                return False
            await asyncio.sleep(0.02)
        return self.boundary.interaction_was_resolved(pending)

    def _closed_turn_report(self, started: float) -> TurnReport:
        """Return the terminal result when teardown wins a race with a waiter."""
        self._turn_deadline = None
        return TurnReport(
            status='session_closed',
            completed=False,
            reason='session_closed',
            thread_id=self.boundary.thread_id,
            turn_id=self.boundary.active_turn_id,
            stderr='the interactive Codex session was closed while a turn was in flight',
            elapsed=time.time() - started,
            session_reusable=False,
        )

    async def _wait_boundary(self, started: float, timeout: float) -> TurnReport:
        """Poll for the terminal condition of the in-flight turn.

        Bounded by the turn's own deadline, so an interaction resume inherits the
        remaining budget rather than restarting the clock.
        """
        loop = asyncio.get_running_loop()
        deadline = min(loop.time() + timeout, self._turn_deadline or (loop.time() + timeout))
        while loop.time() < deadline:
            if self._closed or self._closing:
                return self._closed_turn_report(started)
            if self._tui_exited.is_set():
                self._transition(SessionState.EXITED)
                self._turn_deadline = None
                return await self._report(
                    status='process_exit',
                    completed=False,
                    reason='process_exit',
                    started=started,
                    exit_code=self.pty.proc.returncode if self.pty is not None else None,
                    reusable=False,
                )
            if self._channel_down:
                self._transition(SessionState.FAILED)
                self._turn_deadline = None
                return await self._report(
                    status='event_source_exit',
                    completed=False,
                    reason='event_source_exit',
                    started=started,
                    stderr=(
                        self.app_server.channel_error
                        if self.app_server is not None and self.app_server.channel_error
                        else 'the app-server event channel closed while a turn was in flight'
                    ),
                    reusable=False,
                )
            interaction = self.boundary.pending_interaction
            if interaction is not None:
                if interaction.kind == 'approval':
                    self.boundary.take_interaction()
                    self._transition(SessionState.FAILED)
                    self._turn_deadline = None
                    return await self._report(
                        status='approval_protocol_error',
                        completed=False,
                        reason='approval_under_never_policy',
                        started=started,
                        stderr=(
                            f'Codex raised {interaction.method} while approval_policy=never. '
                            'Refusing to auto-approve; the session is terminated.'
                        ),
                        reusable=False,
                    )
                self._transition(SessionState.WAITING_FOR_USER)
                return await self._report(
                    status='needs_user_input',
                    completed=False,
                    reason=interaction.kind,
                    started=started,
                    interaction=interaction.to_dict(),
                )
            result = self.boundary.last_result
            if result is not None:
                self.boundary.release()
                self._turn_deadline = None  # the logical turn is over
                if result.outcome is TurnOutcome.COMPLETED:
                    self._transition(SessionState.TURN_COMPLETED)
                    return await self._report(
                        status='completed',
                        completed=True,
                        reason='turn_completed',
                        started=started,
                        thread_id=result.thread_id,
                        turn_id=result.turn_id,
                        output=result.final_message,
                    )
                self._transition(SessionState.FAILED)
                reason = 'turn_interrupted' if result.outcome is TurnOutcome.INTERRUPTED else 'turn_failed'
                return await self._report(
                    status=result.outcome.value,
                    completed=False,
                    reason=reason,
                    started=started,
                    thread_id=result.thread_id,
                    turn_id=result.turn_id,
                    stderr=result.error_message or '',
                    reusable=False,
                )
            await asyncio.sleep(0.2)

        if self._closed:
            return self._closed_turn_report(started)
        self._transition(SessionState.FAILED)
        self._turn_deadline = None
        return await self._report(
            status='timeout',
            completed=False,
            reason='timeout',
            started=started,
            stderr=f'no turn boundary within {timeout}s',
            reusable=False,
        )

    async def _report(
        self,
        *,
        status: str,
        completed: bool,
        reason: str,
        started: float,
        thread_id: str | None = None,
        turn_id: str | None = None,
        stderr: str = '',
        exit_code: int | None = None,
        interaction: dict[str, Any] | None = None,
        reusable: bool = True,
        output: str = '',
    ) -> TurnReport:
        return TurnReport(
            status=status,
            completed=completed,
            reason=reason,
            thread_id=thread_id or self.boundary.thread_id,
            turn_id=turn_id or self.boundary.active_turn_id,
            output=output,
            stderr=stderr,
            latest_screen=await self.screen.latest_screen(),
            elapsed=time.time() - started,
            exit_code=exit_code,
            interaction=interaction,
            session_reusable=reusable,
        )


__all__ = [
    'CodexInteractiveSession',
    'InvalidStateTransitionError',
    'SessionState',
    'TurnReport',
]
