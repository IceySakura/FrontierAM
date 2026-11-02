"""Keep a runtime-local CodeBuddy ACP process alive across prompt turns.

Sessions preserve CLI context until explicitly closed; dead or quarantined sessions
are never silently replaced. Extension factories are trusted startup configuration,
not model-generated tool arguments. A factory must be an async context manager and
may be entered and exited by different runtime invoke tasks. It must not carry an
AnyIO cancel scope or task group across its yield: task-affine resources need their
own owner task, with startup/shutdown coordinated by the extension. A failing
``__aenter__`` must roll back its own partially acquired resources.

``acp_cli='workbuddy'`` runs ``wb --acp`` (ACP version 2) behind the same session lifecycle. Its
``session/prompt`` returns once accepted, so a turn waits for the matching ``idle`` state update;
a running turn can take a steer, and ``fork_source_session_id`` branches a running session with
``session/fork`` on the source's process, with its own extension and Host channels. A btw question
is ``_codebuddy.ai/sideQuestion`` on the session itself: the peer reads that session's context,
uses no tools, and returns one answer without opening a turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast, get_args

from nexus.extensions.agents.codebuddy.installer import DEFAULT_CODEBUDDY_VERSION, install_codebuddy_via_npm
from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_nexus_runtime_binary,
)
from nexus.extensions.agents.installed_common.acp import (
    ACP_ERROR_META,
    ACP_ERROR_STOP_REASON,
    ACP_PROMPT_META,
    ACP_PROTOCOL_VERSION,
    ACP_PROTOCOL_VERSION_V2,
    ACP_REQUEST_ID_META,
    AcpConnection,
    AcpProtocolError,
    AcpRemoteError,
    DeferredPermissionHandler,
    ElicitationHandler,
    PermissionPolicy,
    accepted_request_id,
    assistant_text_from_updates,
    side_question_turn_limit,
    supports_prompt_steer,
    supports_session_fork,
    supports_side_question,
)

from .runtime_setup import (
    DEFAULT_INSTALL_DIR,
    DEFAULT_MODEL_ID,
    _apply_json_patch,
    _run_pre_start_script,
    _write_models_json,
    build_codebuddy_permission_cli_args,
)


logger = logging.getLogger(__name__)

_MAX_STDERR_LINES = 200
_INITIALIZE_TIMEOUT = 60.0
_STARTUP_TIMEOUT = 300.0
_SIDE_QUESTION_TIMEOUT_S = 330.0
"""Client wait for one btw answer; the peer's own budget is five minutes and then it returns -32603."""
_SIDE_QUESTION_LAND_TIMEOUT_S = 120.0
"""How long a btw question waits for the session's first prompt to be accepted."""
_SIDE_QUESTION_LAND_POLL_S = 0.05
# The CLI takes about 10s to confirm an ACP cancel; a shorter grace quarantines every cancelled session,
# and closing a session with a live turn waits for that same confirmation.
_CANCEL_GRACE_TIMEOUT = 30.0
_RESOURCE_CLOSE_TIMEOUT = 20.0
_DEFERRED_PERMISSION_TIMEOUT = 300.0
_SESSION_STATE_ROOT = Path('/tmp/nexus-codebuddy-sessions')
# NOTE(wallyshi): wb exits with code 0 on stdin EOF once it has reaped the agent processes it started;
# a signal sent first can leave those behind, so the signal only follows this wait.
_WORKBUDDY_EXIT_TIMEOUT = 10.0
_FORK_CLOSE_TIMEOUT = 10.0
_FORK_SOURCE_POLL_S = 0.5
"""How often a fork rechecks the registry of the source session whose start it is waiting out."""
_WORKBUDDY_PERMISSION_CONFIG = 'permission_mode'
_WORKBUDDY_MODE_CONFIG = 'mode'

AcpCli = Literal['codebuddy', 'workbuddy']
_ACP_CLIS: tuple[str, ...] = get_args(AcpCli)

_CleanupResult = TypeVar('_CleanupResult')
_ProcessStopper = Callable[[asyncio.subprocess.Process], Awaitable[None]]


class TurnUpdateRelay(Protocol):
    """Receive prompt boundaries so pushed updates can name the turn they belong to.

    A version 2 session also passes its ACP ``session_id``: a forked session shares the source's
    connection, so a relay keeps only the updates of the session whose turn it is stamping.
    """

    def begin_turn(self, turn_id: str, *, session_id: str | None = None) -> None: ...

    def end_turn(self) -> int: ...

    async def flush(self) -> None: ...


@dataclass(frozen=True)
class CodeBuddySessionExtension:
    """Provide project-independent resources owned by one session extension.

    Args:
        permission_handler: Deferred approval callback, required for the defer policy.
        mcp_servers: Additional ACP MCP server specifications for session/new.
        update_relay: Told when each prompt that carries a ``relay_turn_id`` starts and ends.
        client_capabilities: ACP ``clientCapabilities`` declared at initialize; None declares none.
        elicitation_handler: Answers ACP ``elicitation/create``; without one they are answered ``cancel``.
        on_turn_end: Called after every prompt ends, once its deferred permissions have been withdrawn.
    """

    permission_handler: DeferredPermissionHandler | None = None
    mcp_servers: tuple[dict[str, Any], ...] = ()
    update_relay: TurnUpdateRelay | None = None
    client_capabilities: dict[str, Any] | None = None
    elicitation_handler: ElicitationHandler | None = None
    on_turn_end: Callable[[], None] | None = None


@dataclass
class _CodeBuddySessionStart:
    """Track a soft interrupt before the first prompt has been created."""

    cancelled: bool = False


class _CodeBuddySessionStartCancelledError(Exception):
    """Signal controlled cancellation before the first prompt starts."""


@dataclass
class _CodeBuddyTurn:
    """Keep cancellation scoped to the exact prompt that was interrupted."""

    prompt_task: asyncio.Task[dict[str, Any]]
    done: asyncio.Future[dict[str, Any]]
    cancel_task: asyncio.Task[dict[str, Any] | None] | None = None


_SESSIONS: dict[str, _CodeBuddyAcpSession] = {}
_SESSION_STARTS: dict[str, _CodeBuddySessionStart] = {}
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_SESSION_CLEANUPS: dict[str, asyncio.Task[str]] = {}
_INSTALL_LOCK = asyncio.Lock()


@dataclass
class _CodeBuddyAcpSession:
    """Own one long-lived CLI, its ACP transport, extension, and model proxy.

    A forked session (``owns_process`` False) shares the source's process, connection, and proxy and
    owns only its remote ACP session and its extension; ``fork_source`` names the source, whose
    ``forks`` lists it back.
    """

    session_id: str
    remote_session_id: str
    proc: asyncio.subprocess.Process
    proxy: Any
    connection: AcpConnection
    cwd: str
    started_at: float
    stderr_lines: list[str]
    stderr_task: asyncio.Task[None]
    model_id: str
    home: str = ''
    extension_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    update_relay: TurnUpdateRelay | None = None
    on_turn_end: Callable[[], None] | None = None
    start_config: dict[str, Any] = field(default_factory=dict)
    acp_cli: AcpCli = 'codebuddy'
    protocol_version: int = ACP_PROTOCOL_VERSION
    initialize_result: dict[str, Any] = field(default_factory=dict)
    owns_process: bool = True
    fork_source: str | None = None
    forks: set[str] = field(default_factory=set)
    unbind_handlers: Callable[[], None] | None = None
    interrupted: bool = False
    reusable: bool = True
    closing: bool = False
    active_turn: _CodeBuddyTurn | None = field(default=None, init=False, repr=False)
    active_request_id: str | None = field(default=None, init=False, repr=False)
    """Request id of the version 2 prompt that is running now; a steer must name it."""
    prompt_landed: bool = False
    """True once a prompt has been accepted; a btw question before that is refused as a draft session."""
    side_turns: list[dict[str, str]] = field(default_factory=list)
    """btw answers this client keeps and sends back; the peer does not store them."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cleanup_task: asyncio.Task[str] | None = field(default=None, init=False, repr=False)

    @property
    def is_running(self) -> bool:
        return self.proc.returncode is None

    async def send_user_message(
        self, prompt: str, *, timeout: float, relay_turn_id: str | None = None
    ) -> dict[str, Any]:
        """Run one serialized turn, budgeting prompt time only after taking its lock.

        With ``relay_turn_id`` and an update relay, updates pushed during this prompt carry that
        id and the result reports how many were pushed as ``relay_frames``. A version 2 turn lasts
        until its ``idle`` state update; one that ended with ``_codebuddy.ai/error`` is ``failed``
        and leaves the session reusable.
        """
        async with self.lock:
            if self.closing:
                raise RuntimeError(f'CodeBuddy session {self.session_id} is closing or closed; recreate it explicitly')
            if not self.is_running:
                raise RuntimeError(f'CodeBuddy session {self.session_id} is not running; close it before recreating it')
            if not self.reusable:
                raise RuntimeError(f'CodeBuddy session {self.session_id} is quarantined; close it before recreating it')
            self.connection.drain_updates(session_id=self.remote_session_id)
            relay = self.update_relay if relay_turn_id else None
            if relay is not None and relay_turn_id:
                if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
                    relay.begin_turn(relay_turn_id, session_id=self.remote_session_id)
                else:
                    relay.begin_turn(relay_turn_id)
            start_time = time.monotonic()
            # NOTE(wallyshi): The outer business deadline leaves time for the original ACP terminal to arrive.
            prompt_timeout = timeout + _CANCEL_GRACE_TIMEOUT
            if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
                prompt_call = self._prompt_until_idle(prompt, timeout=prompt_timeout)
            else:
                prompt_call = self.connection.prompt(
                    session_id=self.remote_session_id, text=prompt, timeout=prompt_timeout
                )
            turn = _CodeBuddyTurn(
                prompt_task=asyncio.create_task(prompt_call),
                done=asyncio.get_running_loop().create_future(),
            )
            self.active_turn = turn
            status = 'ok'
            response: dict[str, Any] = {}
            stderr = ''
            raised_error: BaseException | None = None
            result: dict[str, Any] = {}
            try:
                try:
                    completed, _ = await asyncio.wait((turn.prompt_task,), timeout=timeout)
                    if not completed:
                        raise asyncio.TimeoutError
                    if turn.prompt_task.cancelled():
                        raise RuntimeError(
                            'CodeBuddy ACP prompt ended without a terminal; close and recreate the session'
                        )
                    response = turn.prompt_task.result()
                    stop_reason = response.get('stopReason')
                    if stop_reason == 'cancelled':
                        status = 'cancelled'
                    elif self.protocol_version >= ACP_PROTOCOL_VERSION_V2 and stop_reason == ACP_ERROR_STOP_REASON:
                        status = 'failed'
                        stderr = _turn_error_text(response)
                    else:
                        status = 'ok'
                except asyncio.TimeoutError:
                    status = 'timeout'
                    stderr = f'CodeBuddy session {self.session_id} timed out after {timeout:g}s'
                    self._ensure_turn_cancel(turn, grace_timeout=_CANCEL_GRACE_TIMEOUT)
                except asyncio.CancelledError as error:
                    status = 'cancelled'
                    raised_error = error
                    self._ensure_turn_cancel(turn, grace_timeout=_CANCEL_GRACE_TIMEOUT)
                except Exception as error:
                    self.reusable = False
                    status = 'failed'
                    stderr = str(error) or type(error).__name__
                    raised_error = error
                finally:
                    # NOTE(wallyshi): Keep the turn lock until cancel write/drain finishes even if the prompt finished first.
                    if turn.cancel_task is not None:
                        cancel_response, cancellation_error = await _wait_for_cleanup(turn.cancel_task)
                        if cancel_response is not None:
                            response = cancel_response
                        elif status == 'ok':
                            status = 'failed'
                            stderr = 'ACP cancellation did not finish safely; close and recreate the session'
                        if cancellation_error is not None:
                            raised_error = cancellation_error
                            status = 'cancelled'
                    if self.on_turn_end is not None:
                        self.on_turn_end()
                    updates = self.connection.drain_updates(session_id=self.remote_session_id)
                    relay_frames = None
                    if relay is not None:
                        relay_frames = relay.end_turn()
                        # NOTE(wallyshi): Frames must leave before the result that counts them, or the Host
                        # sees fewer than reported and replays a turn it already has.
                        _, flush_cancellation = await _wait_for_cleanup(asyncio.ensure_future(relay.flush()))
                        if flush_cancellation is not None and raised_error is None:
                            raised_error = flush_cancellation
                    result = {
                        'success': status in ('ok', 'cancelled') and self.reusable,
                        'status': status,
                        'session_id': self.session_id,
                        'output': assistant_text_from_updates(updates),
                        'stderr': stderr,
                        'exit_code': 0,
                        'execution_time': time.monotonic() - start_time,
                        'events': updates,
                        'acp_stop_reason': response.get('stopReason'),
                        'partial': status != 'ok',
                        'relay_frames': relay_frames,
                    }
            except BaseException:
                self.reusable = False
                raise
            finally:
                # NOTE(wallyshi): Interrupt waiters must settle even if update conversion or cancellation cleanup raises.
                if not turn.done.done():
                    turn.done.set_result(result)
                if self.active_turn is turn:
                    self.active_turn = None
            if raised_error is not None:
                raise raised_error
            return result

    async def _prompt_until_idle(self, prompt: str, *, timeout: float) -> dict[str, Any]:
        """Send a version 2 prompt and return its terminal once the matching ``idle`` update arrives.

        ``timeout`` bounds only the acceptance; the turn itself is bounded by the caller. The idle
        update is matched on its request id, and one that arrives before the prompt result is kept
        until that id is known. The turn's permissions and questions are withdrawn only after it
        has ended, never at acceptance.
        """
        idles: dict[str, dict[str, Any]] = {}
        arrived = asyncio.Event()

        def record_idle(params: dict[str, Any]) -> None:
            update = params.get('update')
            if params.get('sessionId') != self.remote_session_id or not isinstance(update, dict):
                return
            if update.get('sessionUpdate') != 'state_update' or update.get('state') != 'idle':
                return
            meta = update.get('_meta')
            request_id = meta.get(ACP_REQUEST_ID_META) if isinstance(meta, dict) else None
            if not isinstance(request_id, str) or not request_id:
                logger.warning(
                    'ACP session %s went idle without a request id; ending the running turn', self.session_id
                )
                request_id = ''
            idles[request_id] = update
            arrived.set()

        remove_listener = self.connection.add_update_listener(record_idle)
        accepted = False
        try:
            response = await self.connection.prompt(session_id=self.remote_session_id, text=prompt, timeout=timeout)
            request_id = accepted_request_id(response)
            if request_id is None:
                raise AcpProtocolError(f'ACP peer accepted session/prompt without a request id: {response!r}')
            accepted = True
            self.prompt_landed = True
            self.active_request_id = request_id
            while (idle := idles.get(request_id, idles.get(''))) is None:
                arrived.clear()
                await arrived.wait()
            return _idle_terminal(response, idle)
        finally:
            self.active_request_id = None
            remove_listener()
            if accepted:
                self.connection.finish_prompt(self.remote_session_id)

    async def steer(self, prompt: str, *, timeout: float) -> dict[str, Any]:
        """Insert ``prompt`` into the version 2 turn that is running now, without taking the turn lock.

        Nothing is inserted when no turn runs or the peer does not advertise steering. A refusal by
        the peer reports its JSON-RPC code as ``error_code``: -32600 means the turn cannot take the
        input right now (starting or ending), so the caller may retry once or send a new turn.
        """

        def refused(reason: str, error_code: int | None = None) -> dict[str, Any]:
            return {
                'success': False,
                'steered': False,
                'session_id': self.session_id,
                'stderr': reason,
                'error_code': error_code,
            }

        if self.closing or not self.is_running:
            return refused(f'CodeBuddy session {self.session_id} is not running or is closing')
        if self.protocol_version < ACP_PROTOCOL_VERSION_V2 or not supports_prompt_steer(self.initialize_result):
            return refused(f'The ACP peer of session {self.session_id} does not advertise steering a running prompt')
        request_id = self.active_request_id
        if not request_id:
            return refused(f'Session {self.session_id} has no running prompt that can take steer input')
        try:
            response = await self.connection.prompt(
                session_id=self.remote_session_id,
                text=prompt,
                timeout=timeout,
                meta={ACP_PROMPT_META: {'mode': 'steer', 'expectedRequestId': request_id}},
            )
        except AcpRemoteError as error:
            detail = cast(dict[str, Any], error.error) if isinstance(error.error, dict) else {}
            code = detail.get('code')
            return refused(str(error), code if isinstance(code, int) else None)
        except (AcpProtocolError, asyncio.TimeoutError) as error:
            return refused(str(error) or type(error).__name__)
        accepted_turn = accepted_request_id(response)
        if accepted_turn != request_id:
            logger.warning(
                'ACP steer of session %s was accepted for turn %s, expected %s',
                self.session_id,
                accepted_turn,
                request_id,
            )
            return refused(f'Steer was accepted for turn {accepted_turn!r}, not the running turn {request_id!r}')
        return {'success': True, 'steered': True, 'session_id': self.session_id, 'stderr': '', 'error_code': None}

    async def ask_side_question(self, question: str, *, timeout: float = _SIDE_QUESTION_TIMEOUT_S) -> dict[str, Any]:
        """Ask ``question`` beside this session and remember the answer for the next btw question.

        Does not take the turn lock: the peer answers from a snapshot while the turn keeps running,
        and writes nothing into the session. A question sent before the first prompt is accepted waits
        until it is, then fails if that wait runs out.
        """

        def failed(reason: str, error_code: int | None = None) -> dict[str, Any]:
            return {
                'success': False,
                'answer': '',
                'session_id': self.session_id,
                'stderr': reason,
                'error_code': error_code,
            }

        if self.closing or not self.is_running:
            return failed(f'CodeBuddy session {self.session_id} is not running or is closing', -32602)
        if self.protocol_version < ACP_PROTOCOL_VERSION_V2 or not supports_side_question(self.initialize_result):
            return failed(f'The ACP peer of session {self.session_id} does not advertise side questions', -32601)
        if not question:
            return failed('A side question must be a non-empty string', -32602)
        landed = await self._wait_until_prompt_landed()
        if not landed:
            return failed(f'Session {self.session_id} has not accepted a prompt yet', -32600)
        limit = side_question_turn_limit(self.initialize_result)
        turns = list(self.side_turns[-limit:])
        try:
            answer = await self.connection.side_question(
                session_id=self.remote_session_id,
                question=question,
                turns=turns,
                timeout=timeout,
            )
        except AcpRemoteError as error:
            detail = cast(dict[str, Any], error.error) if isinstance(error.error, dict) else {}
            code = detail.get('code')
            return failed(str(error), code if isinstance(code, int) else None)
        except (AcpProtocolError, asyncio.TimeoutError) as error:
            return failed(str(error) or type(error).__name__)
        self.side_turns.append({'question': question, 'answer': answer})
        del self.side_turns[:-limit]
        return {
            'success': True,
            'answer': answer,
            'session_id': self.session_id,
            'stderr': '',
            'error_code': None,
        }

    async def _wait_until_prompt_landed(self) -> bool:
        """Wait until this session has accepted a prompt; False when the wait runs out or it is closing."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _SIDE_QUESTION_LAND_TIMEOUT_S
        while not self.prompt_landed:
            if self.closing or not self.is_running or loop.time() >= deadline:
                return False
            await asyncio.sleep(_SIDE_QUESTION_LAND_POLL_S)
        return True

    def _ensure_turn_cancel(self, turn: _CodeBuddyTurn, *, grace_timeout: float) -> asyncio.Task[dict[str, Any] | None]:
        if turn.cancel_task is None:
            turn.cancel_task = asyncio.create_task(self._cancel_and_drain_prompt(turn, grace_timeout=grace_timeout))
        return turn.cancel_task

    async def _cancel_and_drain_prompt(self, turn: _CodeBuddyTurn, *, grace_timeout: float) -> dict[str, Any] | None:
        deadline = asyncio.get_running_loop().time() + grace_timeout
        try:
            if not turn.prompt_task.done():
                await asyncio.wait_for(
                    self.connection.cancel(session_id=self.remote_session_id),
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
            return await asyncio.wait_for(
                asyncio.shield(turn.prompt_task),
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
        except (asyncio.CancelledError, Exception):
            self.reusable = False
            logger.warning(
                'CodeBuddy session %s cancellation did not confirm the original prompt terminal', self.session_id
            )
            turn.prompt_task.cancel()
            await asyncio.gather(turn.prompt_task, return_exceptions=True)
            return None

    async def interrupt(self, *, grace_timeout: float = _CANCEL_GRACE_TIMEOUT) -> dict[str, Any]:
        """Cancel a snapshot of the active turn, never a prompt arriving afterward."""
        self.interrupted = True
        if not self.is_running or self.closing:
            return {
                'success': False,
                'session_id': self.session_id,
                'stopped': False,
                'stderr': f'CodeBuddy session {self.session_id} is not running or is closing',
            }
        turn = self.active_turn
        if turn is None:
            return {'success': True, 'session_id': self.session_id, 'stopped': False, 'alive': True, 'stderr': ''}
        # NOTE(wallyshi): Publish the shared cancellation before yielding, so the target retains its turn lock.
        cancel_task = self._ensure_turn_cancel(turn, grace_timeout=grace_timeout)
        response, cancellation_error = await _wait_for_cleanup(cancel_task)
        turn_result, done_cancellation = await _wait_for_cleanup(turn.done)
        if cancellation_error is not None:
            raise cancellation_error
        if done_cancellation is not None:
            raise done_cancellation
        if response is None:
            return {
                'success': False,
                'session_id': self.session_id,
                'stopped': False,
                'alive': self.is_running,
                'stderr': 'ACP cancel did not confirm the original prompt terminal; close and recreate the session',
                'quarantined': True,
                'output': turn_result.get('output', ''),
                'events': list(turn_result.get('events') or []),
                'partial': True,
            }
        return {
            'success': True,
            'session_id': self.session_id,
            'stopped': True,
            'alive': self.is_running,
            'output': turn_result.get('output', ''),
            'events': list(turn_result.get('events') or []),
            'partial': True,
        }

    async def close(self) -> dict[str, Any]:
        """Close resources exactly once, despite repeated caller cancellation.

        A session that owns its process closes its forks first. A fork closes only its remote
        session and its extension; the source's process keeps running.
        """
        self.closing = True
        if self._cleanup_task is None:
            cleanup = self._close_process() if self.owns_process else self._close_fork()
            self._cleanup_task = asyncio.create_task(cleanup)
            _SESSION_CLEANUPS[self.session_id] = self._cleanup_task
        stderr = await _wait_for_resource_cleanup(self.session_id, self._cleanup_task)
        if stderr:
            self.reusable = False
        return {'success': not stderr, 'session_id': self.session_id, 'stderr': stderr}

    async def _close_process(self) -> str:
        for fork_id in sorted(self.forks):
            fork = _SESSIONS.get(fork_id)
            if fork is None:
                continue
            result = await fork.close()
            if result['success'] and _SESSIONS.get(fork_id) is fork:
                _SESSIONS.pop(fork_id, None)
            elif not result['success']:
                logger.warning(
                    'Fork %s of session %s did not close cleanly: %s', fork_id, self.session_id, result['stderr']
                )
        self.forks.clear()
        return await _close_codebuddy_resources(
            proc=self.proc,
            proxy=self.proxy,
            connection=self.connection,
            stderr_task=self.stderr_task,
            extension_stack=self.extension_stack,
            stop_process=_process_stopper(self.acp_cli),
        )

    async def _close_fork(self) -> str:
        errors: list[str] = []
        try:
            if not self.is_running:
                logger.info('Fork %s outlived its source process; there is no remote session to close', self.session_id)
            else:
                await self.connection.close_session(session_id=self.remote_session_id, timeout=_FORK_CLOSE_TIMEOUT)
        except Exception as error:  # noqa: BLE001
            # NOTE(wallyshi): A fork dies with its source's process, so a transport that closed meanwhile is not a leak.
            if self.is_running:
                errors.append(f'Failed to close forked ACP session: {str(error) or type(error).__name__}')
                logger.exception('Failed to close forked ACP session %s', self.session_id)
            else:
                logger.info('Source process of fork %s exited while closing it: %s', self.session_id, error)
        finally:
            if self.unbind_handlers is not None:
                self.unbind_handlers()
        try:
            await self.extension_stack.aclose()
        except BaseException as error:
            errors.append(f'Failed to close CodeBuddy session extension: {error}')
            logger.exception('Failed to close the extension of forked session %s', self.session_id)
        source = _SESSIONS.get(self.fork_source) if self.fork_source else None
        if source is not None:
            source.forks.discard(self.session_id)
        return '\n'.join(errors)


async def _wait_for_cleanup(
    cleanup_task: asyncio.Future[_CleanupResult],
) -> tuple[_CleanupResult, asyncio.CancelledError | None]:
    cancellation_error: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(cleanup_task), cancellation_error
        except asyncio.CancelledError as error:
            if cleanup_task.cancelled():
                raise
            if cancellation_error is None:
                cancellation_error = error


async def _wait_for_resource_cleanup(session_id: str, cleanup_task: asyncio.Task[str]) -> str:
    waiter = asyncio.create_task(asyncio.wait((cleanup_task,), timeout=_RESOURCE_CLOSE_TIMEOUT))
    (completed, _), cancellation_error = await _wait_for_cleanup(waiter)
    if completed:
        stderr = cleanup_task.result()
        if not stderr and _SESSION_CLEANUPS.get(session_id) is cleanup_task:
            _SESSION_CLEANUPS.pop(session_id, None)
    else:
        stderr = f'CodeBuddy session {session_id} resource cleanup is still pending; retry close after it finishes'
        logger.warning('%s', stderr)
    if cancellation_error is not None:
        raise cancellation_error
    return stderr


async def _close_codebuddy_resources(
    *,
    proc: asyncio.subprocess.Process | None,
    proxy: Any | None,
    connection: AcpConnection | None,
    stderr_task: asyncio.Task[None] | None,
    extension_stack: AsyncExitStack,
    stop_process: _ProcessStopper | None = None,
) -> str:
    """Release one session's resources; ``stop_process`` defaults to terminating the CLI."""
    errors: list[str] = []
    connection_close_task: asyncio.Task[None] | None = None
    if connection is not None:
        connection_close_task = asyncio.create_task(connection.close())
        # NOTE(wallyshi): Do not cancel ACP.close: its shielded callback join must finish before extension teardown.
        await asyncio.wait((connection_close_task,), timeout=_RESOURCE_CLOSE_TIMEOUT)
    if stderr_task is not None:
        stderr_task.cancel()
        try:
            await stderr_task
        except BaseException as error:
            if not stderr_task.cancelled():
                errors.append(f'Failed to stop CodeBuddy stderr reader: {error}')
                logger.exception('Failed to stop CodeBuddy stderr reader')
    if proc is not None:
        try:
            await (stop_process or _terminate_process)(proc)
        except BaseException as error:
            errors.append(f'Failed to stop CodeBuddy process: {error}')
            logger.exception('Failed to stop CodeBuddy process')
    if connection_close_task is not None:
        try:
            await connection_close_task
        except BaseException as error:
            errors.append(f'Failed to close CodeBuddy ACP connection: {error}')
            logger.exception('Failed to close CodeBuddy ACP connection')
    try:
        await extension_stack.aclose()
    except BaseException as error:
        errors.append(f'Failed to close CodeBuddy session extension: {error}')
        logger.exception('Failed to close CodeBuddy session extension')
    if proxy is not None:
        try:
            await proxy.stop()
        except BaseException as error:
            errors.append(f'Failed to stop CodeBuddy LLM proxy: {error}')
            logger.exception('Failed to stop CodeBuddy LLM proxy')
    return '\n'.join(errors)


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)


async def _stop_workbuddy_process(proc: asyncio.subprocess.Process) -> None:
    """Close stdin so wb exits on its own, then stop whatever is left of its process group."""
    from .workbuddy_acp import stop_workbuddy_process_group

    if proc.returncode is None and proc.stdin is not None:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_WORKBUDDY_EXIT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning('WorkBuddy did not exit %gs after stdin closed; signalling it', _WORKBUDDY_EXIT_TIMEOUT)
    # NOTE(wallyshi): wb runs in its own process group, so agent processes it left behind are stopped too.
    await stop_workbuddy_process_group(proc.pid)
    if proc.returncode is None:
        await asyncio.wait_for(proc.wait(), timeout=5)


def _process_stopper(acp_cli: AcpCli) -> _ProcessStopper:
    return _stop_workbuddy_process if acp_cli == 'workbuddy' else _terminate_process


async def _read_stderr(stream: asyncio.StreamReader, *, buffer: list[str]) -> None:
    while True:
        line = await stream.readline()
        if not line:
            return
        buffer.append(line.decode('utf-8', errors='replace').rstrip())
        del buffer[:-_MAX_STDERR_LINES]


def _idle_terminal(response: dict[str, Any], idle: dict[str, Any]) -> dict[str, Any]:
    """Merge a version 2 prompt result with the ``idle`` update that ended its turn."""
    stop_reason = idle.get('stopReason')
    terminal = {**response, 'stopReason': stop_reason if isinstance(stop_reason, str) and stop_reason else 'end_turn'}
    meta = idle.get('_meta')
    error = meta.get(ACP_ERROR_META) if isinstance(meta, dict) else None
    if isinstance(error, dict):
        terminal['error'] = {'code': error.get('code'), 'message': error.get('message')}
    return terminal


def _turn_error_text(response: dict[str, Any]) -> str:
    error = response.get('error')
    if not isinstance(error, dict):
        return 'The ACP turn ended with _codebuddy.ai/error and no error details'
    message = str(error.get('message') or 'The ACP turn failed without a message')
    return f'{message} (code {error["code"]})' if error.get('code') else message


def _session_home(session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:24]
    return _SESSION_STATE_ROOT / digest / 'home'


def _supports_http_mcp(initialize_result: dict[str, Any]) -> bool:
    capabilities = initialize_result.get('agentCapabilities')
    if not isinstance(capabilities, dict):
        return False
    mcp_capabilities = capabilities.get('mcpCapabilities')
    if isinstance(mcp_capabilities, dict) and mcp_capabilities.get('http') is True:
        return True
    session_capabilities = capabilities.get('session')
    if isinstance(session_capabilities, dict):
        mcp = session_capabilities.get('mcp')
        if isinstance(mcp, dict) and mcp.get('http') is True:
            return True
    return False


def _validate_acp_cli(
    acp_cli: str, workbuddy_options: dict[str, Any] | None, fork_source_session_id: str | None
) -> None:
    if acp_cli not in _ACP_CLIS:
        raise ValueError(f'Unsupported acp_cli {acp_cli!r}; choose one of {list(_ACP_CLIS)}')
    if acp_cli != 'workbuddy' and (workbuddy_options is not None or fork_source_session_id is not None):
        raise ValueError('workbuddy_options and fork_source_session_id need acp_cli="workbuddy"')


def _resolve_extension_factory(extension_factory: str | None) -> Callable[..., Any] | None:
    """Resolve the trusted ``module::factory`` path of a session extension."""
    if extension_factory is None:
        return None
    # NOTE(wallyshi): Runtime transport imports stay lazy so host-side code can import this invocation target.
    from nexus.runtime.models import resolve_func

    parts = extension_factory.split('::')
    if len(parts) != 2 or not all(part and all(name.isidentifier() for name in part.split('.')) for part in parts):
        raise ValueError('extension_factory must use the trusted module::factory import path, not a registry name')
    factory = resolve_func(extension_factory)
    if not callable(factory):
        raise TypeError(f'extension_factory {extension_factory!r} must resolve to a callable async context manager')
    return factory


async def _enter_extension(
    extension_stack: AsyncExitStack,
    factory: Callable[..., Any],
    *,
    session_id: str,
    sio: Any,
    connection: AcpConnection,
    config: dict[str, Any] | None,
) -> CodeBuddySessionExtension:
    extension = await extension_stack.enter_async_context(
        factory(session_id=session_id, sio=sio, connection=connection, config=config or {})
    )
    if not isinstance(extension, CodeBuddySessionExtension):
        raise TypeError('extension_factory must yield CodeBuddySessionExtension')
    if extension.permission_handler is not None and not callable(extension.permission_handler):
        raise TypeError('CodeBuddySessionExtension.permission_handler must be callable or None')
    if extension.elicitation_handler is not None and not callable(extension.elicitation_handler):
        raise TypeError('CodeBuddySessionExtension.elicitation_handler must be callable or None')
    if extension.on_turn_end is not None and not callable(extension.on_turn_end):
        raise TypeError('CodeBuddySessionExtension.on_turn_end must be callable or None')
    return extension


def _with_workbuddy_defaults(workbuddy_options: dict[str, Any] | None) -> dict[str, Any]:
    # NOTE(wallyshi): Only wb sessions need the WorkBuddy package helpers, so CodeBuddy never imports them.
    from .workbuddy_acp import DEFAULT_WORKBUDDY_ACP_VERSION

    options = dict(workbuddy_options or {})
    options['version'] = options.get('version') or DEFAULT_WORKBUDDY_ACP_VERSION
    return options


async def _install_workbuddy_cli(workbuddy_options: dict[str, Any]) -> tuple[Path, str]:
    """Return ``wb.js`` and the Node binary that runs it, unpacking the uploaded package once."""
    from .workbuddy_acp import ensure_workbuddy_cli

    node_path = find_nexus_runtime_binary('node')
    if not node_path:
        raise RuntimeError('Node.js is required but not found in runtime')
    return await ensure_workbuddy_cli(workbuddy_options, node_path=node_path), node_path


def _workbuddy_launch(
    *,
    node_path: str,
    wb_js: Path,
    session_home: Path,
    proxy_port: int,
    model_id: str,
    managed_env: dict[str, str],
    workbuddy_options: dict[str, Any],
    extra_cli_args: list[str] | None,
) -> tuple[list[str], dict[str, str]]:
    """Write the session's WorkBuddy data directory and return the ``wb --acp`` command and environment."""
    from .workbuddy_acp import build_workbuddy_acp_argv, build_workbuddy_acp_env, prepare_workbuddy_config_dir

    config_dir = session_home / 'workbuddy'
    proxy_base_url = f'http://127.0.0.1:{proxy_port}/v1'
    # NOTE(wallyshi): wb reads models.json once at startup, so it must name this session's proxy before spawning.
    prepare_workbuddy_config_dir(
        config_dir, proxy_url=f'{proxy_base_url}/chat/completions', model_id=model_id, api_key=DEFAULT_PROXY_API_KEY
    )
    env = build_workbuddy_acp_env(
        base_env=managed_env,
        home=str(session_home),
        config_dir=str(config_dir),
        proxy_base_url=proxy_base_url,
        api_key=managed_env['CODEBUDDY_API_KEY'],
    )
    cmd = build_workbuddy_acp_argv(
        node_path,
        str(wb_js),
        model_id=model_id,
        version=str(workbuddy_options['version']),
        preset=workbuddy_options.get('preset'),
        mode=workbuddy_options.get('mode'),
        effort=workbuddy_options.get('effort'),
        features=workbuddy_options.get('features'),
        extra_cli_args=tuple(extra_cli_args or ()),
    )
    return cmd, env


def _config_option_values(option: dict[str, Any]) -> set[Any]:
    """Selectable values of one ``configOptions`` entry, looking inside option groups."""
    values: set[Any] = set()
    pending = list(option.get('options') or [])
    while pending:
        item = pending.pop()
        if not isinstance(item, dict):
            continue
        if isinstance(item.get('options'), list):
            pending.extend(item['options'])
        elif 'value' in item:
            values.add(item['value'])
    return values


async def _apply_workbuddy_permission_mode(
    connection: AcpConnection,
    remote_session_id: str,
    *,
    workbuddy_options: dict[str, Any],
    skip_permissions: bool,
) -> None:
    """Set the session's ``permission_mode`` to the explicit option, else derive it from ``skip_permissions``.

    The Host option defaults to ``bypassPermissions``, which still asks before dangerous commands.
    Without one, ``skip_permissions`` maps to ``fullAccess`` like CodeBuddy's ``--dangerously-skip-permissions``,
    otherwise ``default`` asks for each tool; wb's own default is ``bypassPermissions`` in craft and quick.
    Plan mode offers no permission mode, so nothing is set.
    """
    target = workbuddy_options.get('permission_mode') or ('fullAccess' if skip_permissions else 'default')
    options = connection.config_options.get(remote_session_id) or []
    option = next(
        (item for item in options if isinstance(item, dict) and item.get('configId') == _WORKBUDDY_PERMISSION_CONFIG),
        None,
    )
    if option is None:
        logger.info(
            'WorkBuddy session %s offers no permission_mode (plan mode); keeping its default', remote_session_id
        )
        return
    values = _config_option_values(option)
    if target not in values:
        raise ValueError(
            f'WorkBuddy permission_mode {target!r} is not offered by this session; '
            f'choose one of {sorted(map(str, values))} in workbuddy.permission_mode'
        )
    if option.get('currentValue') == target:
        return
    await connection.set_config_option(
        session_id=remote_session_id,
        config_id=_WORKBUDDY_PERMISSION_CONFIG,
        value=target,
        timeout=_INITIALIZE_TIMEOUT,
    )


async def _start_session(
    *,
    session_id: str,
    working_dir: str,
    version: str,
    npm_registry: str | None,
    model_id: str,
    extra_env: dict[str, str] | None,
    extra_cli_args: list[str] | None,
    config_file_patches: list[dict[str, Any]] | None,
    pre_start_scripts: list[dict[str, Any]] | None,
    permission_policy: PermissionPolicy,
    mcp_servers: list[dict[str, Any]] | None,
    extension_factory: str | None = None,
    extension_config: dict[str, Any] | None = None,
    skip_permissions: bool = True,
    deferred_permission_timeout: float | None = _DEFERRED_PERMISSION_TIMEOUT,
    acp_cli: AcpCli = 'codebuddy',
    workbuddy_options: dict[str, Any] | None = None,
    fork_source_session_id: str | None = None,
) -> _CodeBuddyAcpSession:
    """Acquire startup resources, bind an optional extension, and negotiate ACP.

    ``acp_cli='workbuddy'`` launches ``wb --acp`` and negotiates ACP version 2. With
    ``fork_source_session_id`` no process is started: the running source session is branched
    with ``session/fork`` instead (see :func:`_attach_fork`).
    """
    # NOTE(wallyshi): Runtime transport imports stay lazy so host-side code can import this invocation target.
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy
    from nexus.utils.shell import get_os_env_without_ld_library_path

    _validate_acp_cli(acp_cli, workbuddy_options, fork_source_session_id)
    if permission_policy not in ('allow', 'reject', 'defer'):
        raise ValueError(f'Unsupported permission_policy {permission_policy!r}; choose allow, reject, or defer')
    if permission_policy == 'defer' and extension_factory is None:
        raise ValueError('permission_policy=defer requires extension_factory=module::factory with a permission handler')
    factory = _resolve_extension_factory(extension_factory)
    workbuddy = acp_cli == 'workbuddy'
    wb_options = _with_workbuddy_defaults(workbuddy_options) if workbuddy else {}
    if fork_source_session_id is not None:
        return await _attach_fork(
            session_id=session_id,
            fork_source_session_id=fork_source_session_id,
            working_dir=working_dir,
            permission_policy=permission_policy,
            factory=factory,
            extension_config=extension_config,
            workbuddy_options=wb_options,
            skip_permissions=skip_permissions,
        )
    runtime_control = require_runtime_control()
    codebuddy_path: Path | None = None
    wb_js: Path | None = None
    wb_node_path = ''
    if workbuddy:
        wb_js, wb_node_path = await _install_workbuddy_cli(wb_options)
    else:
        # NOTE(wallyshi): Concurrent sessions share npm state, but proxy and process startup remain independent.
        async with _INSTALL_LOCK:
            codebuddy_path = await install_codebuddy_via_npm(
                install_dir=DEFAULT_INSTALL_DIR, version=version, npm_registry=npm_registry
            )
    proxy: Any | None = None
    proc: asyncio.subprocess.Process | None = None
    connection: AcpConnection | None = None
    stderr_task: asyncio.Task[None] | None = None
    extension_stack = AsyncExitStack()
    try:
        proxy_port = find_free_port()
        proxy = OpenAIProxy(
            sio=runtime_control.sio,
            listen_port=proxy_port,
            listen_host='127.0.0.1',
            request_timeout=3600,
            models=[model_id],
            channel_id=session_id,
        )
        await proxy.start()
        effective_mcp_servers = list(mcp_servers or [])
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        session_home = _session_home(session_id)
        session_home.mkdir(parents=True, exist_ok=True)
        home = str(session_home)
        if not workbuddy:
            codebuddy_home = session_home / '.codebuddy'
            codebuddy_home.mkdir(parents=True, exist_ok=True)
            # NOTE(wallyshi): models.json is read once by the CLI and must point at this session's proxy before spawning.
            _write_models_json(
                codebuddy_home, f'http://127.0.0.1:{proxy_port}/v1/chat/completions', model_id, DEFAULT_PROXY_API_KEY
            )
        managed_env = {
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
            'NO_COLOR': '1',
            'CI': '1',
            # NOTE(wallyshi): An isolated HOME has no CLI login; the local proxy uses this non-secret startup credential.
            'CODEBUDDY_API_KEY': DEFAULT_PROXY_API_KEY,
        }
        managed_env.update(extra_env or {})
        # NOTE(wallyshi): Pin HOME after overrides so callers cannot collapse per-session settings isolation.
        managed_env['HOME'] = home
        if wb_js is not None:
            cmd, managed_env = _workbuddy_launch(
                node_path=wb_node_path,
                wb_js=wb_js,
                session_home=session_home,
                proxy_port=proxy_port,
                model_id=model_id,
                managed_env=managed_env,
                workbuddy_options=wb_options,
                extra_cli_args=extra_cli_args,
            )
        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {}, home=home)
        for script in pre_start_scripts or []:
            await _run_pre_start_script(script, working_dir, managed_env)
        if wb_js is None:
            node_path = find_nexus_runtime_binary('node')
            if not node_path:
                raise RuntimeError('Node.js is required but not found in runtime')
            # NOTE(wallyshi): ACP session/new owns identity; passing --session-id would compete with its handshake.
            cmd = [
                node_path,
                str(codebuddy_path),
                '--acp',
                '--acp-transport',
                'stdio',
                '--model',
                model_id,
                *build_codebuddy_permission_cli_args(skip_permissions=skip_permissions, extra_cli_args=extra_cli_args),
            ]
        logger.debug('Starting %s ACP session %s', 'WorkBuddy' if workbuddy else 'CodeBuddy', session_id)
        launch_env = get_os_env_without_ld_library_path()
        launch_env['BASH_ENV'] = '~/.bashrc'
        launch_env.update(managed_env)
        # NOTE(wallyshi): wb gets its own process group so the agent processes it starts can be stopped with it.
        spawn_options: dict[str, Any] = {'start_new_session': True} if workbuddy else {}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_path),
            env=launch_env,
            limit=DEFAULT_CLI_STREAM_READER_LIMIT,
            **spawn_options,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError('CodeBuddy process was created without stdio pipes')
        stderr_lines: list[str] = []
        stderr_task = asyncio.create_task(_read_stderr(proc.stderr, buffer=stderr_lines))
        connection = AcpConnection(
            proc.stdout,
            proc.stdin,
            permission_policy=permission_policy,
            tolerate_noise=True,
            deferred_permission_timeout=deferred_permission_timeout,
        )
        update_relay = None
        on_turn_end = None
        client_capabilities = None
        if factory is not None:
            extension = await _enter_extension(
                extension_stack,
                factory,
                session_id=session_id,
                sio=runtime_control.sio,
                connection=connection,
                config=extension_config,
            )
            connection.deferred_permission_handler = extension.permission_handler
            connection.elicitation_handler = extension.elicitation_handler
            effective_mcp_servers.extend(extension.mcp_servers)
            update_relay = extension.update_relay
            on_turn_end = extension.on_turn_end
            client_capabilities = extension.client_capabilities
        if permission_policy == 'defer' and connection.deferred_permission_handler is None:
            raise ValueError('permission_policy=defer requires extension_factory to yield a permission_handler')
        if workbuddy and effective_mcp_servers:
            raise ValueError(
                'WorkBuddy ACP does not accept mcpServers; drop host_tools and mcp_servers for acp_cli="workbuddy"'
            )
        connection.start()
        if workbuddy:
            initialize_result = await connection.initialize(
                timeout=_INITIALIZE_TIMEOUT,
                client_capabilities=client_capabilities,
                protocol_version=ACP_PROTOCOL_VERSION_V2,
            )
        else:
            initialize_result = await connection.initialize(
                timeout=_INITIALIZE_TIMEOUT, client_capabilities=client_capabilities
            )
        if any(server.get('type') == 'http' for server in effective_mcp_servers) and not _supports_http_mcp(
            initialize_result
        ):
            raise RuntimeError('CodeBuddy ACP peer does not advertise HTTP MCP support')
        remote_session_id = await connection.new_session(
            cwd=str(work_path), timeout=_INITIALIZE_TIMEOUT, mcp_servers=effective_mcp_servers
        )
        if workbuddy:
            await _apply_workbuddy_permission_mode(
                connection, remote_session_id, workbuddy_options=wb_options, skip_permissions=skip_permissions
            )
        return _CodeBuddyAcpSession(
            session_id=session_id,
            remote_session_id=remote_session_id,
            proc=proc,
            proxy=proxy,
            connection=connection,
            cwd=str(work_path),
            started_at=time.time(),
            stderr_lines=stderr_lines,
            stderr_task=stderr_task,
            model_id=model_id,
            home=home,
            extension_stack=extension_stack,
            update_relay=update_relay,
            on_turn_end=on_turn_end,
            acp_cli=acp_cli,
            protocol_version=ACP_PROTOCOL_VERSION_V2 if workbuddy else ACP_PROTOCOL_VERSION,
            initialize_result=initialize_result if isinstance(initialize_result, dict) else {},
        )
    except BaseException:
        cleanup_task = asyncio.create_task(
            _close_codebuddy_resources(
                proc=proc,
                proxy=proxy,
                connection=connection,
                stderr_task=stderr_task,
                extension_stack=extension_stack,
                stop_process=_process_stopper(acp_cli),
            )
        )
        _SESSION_CLEANUPS[session_id] = cleanup_task
        await _wait_for_resource_cleanup(session_id, cleanup_task)
        raise


async def _attach_fork(
    *,
    session_id: str,
    fork_source_session_id: str,
    working_dir: str,
    permission_policy: PermissionPolicy,
    factory: Callable[..., Any] | None,
    extension_config: dict[str, Any] | None,
    workbuddy_options: dict[str, Any],
    skip_permissions: bool,
) -> _CodeBuddyAcpSession:
    """Branch the running ``fork_source_session_id`` with ``session/fork`` on its own process.

    The fork enters its own extension under its own session id, so its approvals, questions, and
    pushed updates reach the Host on the fork's channels; its handlers are bound to the fork's ACP
    session. ``workbuddy_options['fork_mode']`` is applied while the fork is still idle, then the
    permission mode is set again because a mode change resets it. Model requests still use the
    source's proxy and LLM channel: wb reads ``models.json`` once per process.
    """
    source = await _await_fork_source(fork_source_session_id)
    if source is None or source.closing or not source.is_running or not source.reusable:
        raise RuntimeError(
            f'Cannot fork CodeBuddy session {fork_source_session_id}: it is not running; start it before forking'
        )
    if not source.owns_process:
        raise ValueError(f'Session {fork_source_session_id} is itself a fork; fork its source instead')
    if source.protocol_version < ACP_PROTOCOL_VERSION_V2 or not supports_session_fork(source.initialize_result):
        raise RuntimeError(f'The ACP peer of session {fork_source_session_id} does not advertise session/fork')
    if Path(working_dir).resolve() != Path(source.cwd).resolve():
        raise ValueError(f'A fork must keep its source cwd {source.cwd}; got working_dir={working_dir}')
    connection = source.connection
    if permission_policy != connection.permission_policy:
        raise ValueError(
            f'A fork shares its source connection, whose permission_policy is {connection.permission_policy!r}; '
            f'got {permission_policy!r}'
        )
    # NOTE(wallyshi): Runtime transport imports stay lazy so host-side code can import this invocation target.
    from nexus.runtime.context import require_runtime_control

    extension_stack = AsyncExitStack()
    remote_session_id: str | None = None
    unbind: Callable[[], None] | None = None
    extension: CodeBuddySessionExtension | None = None
    try:
        if factory is not None:
            extension = await _enter_extension(
                extension_stack,
                factory,
                session_id=session_id,
                sio=require_runtime_control().sio,
                connection=connection,
                config=extension_config,
            )
            if extension.mcp_servers:
                raise ValueError('WorkBuddy ACP does not accept mcpServers; drop host_tools for a fork')
            if permission_policy == 'defer' and extension.permission_handler is None:
                raise ValueError('permission_policy=defer requires extension_factory to yield a permission_handler')
        remote_session_id = await connection.fork_session(
            session_id=source.remote_session_id, cwd=source.cwd, timeout=_INITIALIZE_TIMEOUT
        )
        if extension is not None:
            unbind = connection.bind_session_handlers(
                remote_session_id,
                permission_handler=extension.permission_handler,
                elicitation_handler=extension.elicitation_handler,
            )
        fork_mode = workbuddy_options.get('fork_mode')
        if fork_mode:
            await connection.set_config_option(
                session_id=remote_session_id,
                config_id=_WORKBUDDY_MODE_CONFIG,
                value=fork_mode,
                timeout=_INITIALIZE_TIMEOUT,
            )
        await _apply_workbuddy_permission_mode(
            connection, remote_session_id, workbuddy_options=workbuddy_options, skip_permissions=skip_permissions
        )
    except BaseException:
        if unbind is not None:
            unbind()
        if remote_session_id is not None:
            try:
                await connection.close_session(session_id=remote_session_id, timeout=_FORK_CLOSE_TIMEOUT)
            except Exception:  # noqa: BLE001
                logger.exception('Failed to close fork %s after its setup failed', session_id)
        await extension_stack.aclose()
        raise
    fork = _CodeBuddyAcpSession(
        session_id=session_id,
        remote_session_id=remote_session_id,
        proc=source.proc,
        proxy=source.proxy,
        connection=connection,
        cwd=source.cwd,
        started_at=time.time(),
        stderr_lines=source.stderr_lines,
        stderr_task=source.stderr_task,
        model_id=source.model_id,
        home=source.home,
        extension_stack=extension_stack,
        update_relay=extension.update_relay if extension is not None else None,
        on_turn_end=extension.on_turn_end if extension is not None else None,
        acp_cli='workbuddy',
        protocol_version=source.protocol_version,
        initialize_result=source.initialize_result,
        owns_process=False,
        fork_source=source.session_id,
        unbind_handlers=unbind,
    )
    source.forks.add(session_id)
    logger.info('Forked ACP session %s from %s', session_id, fork_source_session_id)
    return fork


async def _cancel_session_start(start_task: asyncio.Task[_CodeBuddyAcpSession]) -> None:
    start_task.cancel()
    result = await asyncio.gather(start_task, return_exceptions=True)
    if isinstance(result[0], _CodeBuddyAcpSession):
        await result[0].close()


async def _await_fork_source(session_id: str) -> _CodeBuddyAcpSession | None:
    """Return the session ``session_id``, waiting out a start that is still in flight, if any.

    A follow-up question is dispatched the moment its source turn is handed to the backend, which is
    before that session's startup registered it. Polling the start registry, not the creation lock,
    keeps this safe on every Python the runtime ships: a cancelled lock acquire can leave the lock
    held on 3.10 and 3.11.
    """
    source = _SESSIONS.get(session_id)
    if source is not None or session_id not in _SESSION_STARTS:
        return source
    loop = asyncio.get_running_loop()
    # NOTE(wallyshi): Startup itself is capped at _STARTUP_TIMEOUT and its rollback waits are bounded,
    # so the registry entry always leaves; the margin covers a cleanup that outlives the startup cap.
    deadline = loop.time() + _STARTUP_TIMEOUT + _RESOURCE_CLOSE_TIMEOUT
    while session_id in _SESSION_STARTS and loop.time() < deadline:
        await asyncio.sleep(_FORK_SOURCE_POLL_S)
    return _SESSIONS.get(session_id)


async def _get_or_create_session(*, session_id: str, **start_kwargs: Any) -> _CodeBuddyAcpSession:
    lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = _SESSIONS.get(session_id)
        if session is None and session_id in _SESSION_CLEANUPS:
            raise RuntimeError(f'CodeBuddy session {session_id} still owns resource cleanup; explicitly close it first')
        if session is not None:
            if session.closing:
                raise RuntimeError(f'CodeBuddy session {session_id} is closing; close it before recreating it')
            if not session.is_running:
                raise RuntimeError(f'CodeBuddy session {session_id} is not running; close it before recreating it')
            if not session.reusable:
                raise RuntimeError(f'CodeBuddy session {session_id} is quarantined; close it before recreating it')
            if session.start_config != start_kwargs:
                raise ValueError(
                    f'CodeBuddy session {session_id} startup configuration changed; close it before recreating it'
                )
            return session
        session_start = _CodeBuddySessionStart()
        _SESSION_STARTS[session_id] = session_start
        start_config = deepcopy(start_kwargs)
        start_task = asyncio.create_task(_start_session(session_id=session_id, **deepcopy(start_kwargs)))
        try:
            try:
                # NOTE(wallyshi): Shield startup so repeated invoke cancellation cannot detach its resource rollback.
                session = await asyncio.wait_for(asyncio.shield(start_task), timeout=_STARTUP_TIMEOUT)
            except BaseException as error:
                cleanup_task = asyncio.create_task(_cancel_session_start(start_task))
                _, cancellation_error = await _wait_for_cleanup(cleanup_task)
                if cancellation_error is not None:
                    raise cancellation_error from None
                if session_start.cancelled and isinstance(error, Exception):
                    raise _CodeBuddySessionStartCancelledError from None
                raise
            if session_start.cancelled:
                await session.close()
                raise _CodeBuddySessionStartCancelledError
            session.start_config = start_config
            _SESSIONS[session_id] = session
            return session
        finally:
            if _SESSION_STARTS.get(session_id) is session_start:
                _SESSION_STARTS.pop(session_id, None)


async def run_codebuddy_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600.0,
    version: str = DEFAULT_CODEBUDDY_VERSION,
    npm_registry: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    permission_policy: PermissionPolicy = 'allow',
    mcp_servers: list[dict[str, Any]] | None = None,
    extension_factory: str | None = None,
    extension_config: dict[str, Any] | None = None,
    skip_permissions: bool = True,
    deferred_permission_timeout: float | None = _DEFERRED_PERMISSION_TIMEOUT,
    relay_turn_id: str | None = None,
    acp_cli: AcpCli = 'codebuddy',
    workbuddy_options: dict[str, Any] | None = None,
    fork_source_session_id: str | None = None,
) -> dict[str, Any]:
    """Drive a turn of an existing or new runtime-local CodeBuddy session.

    Startup configuration must stay identical across turns. ``timeout`` budgets the
    prompt after acquiring its turn lock; startup has a separate bounded deadline.
    ``deferred_permission_timeout`` bounds each deferred approval, including its
    handler, and each extension-handled ``elicitation/create``; ``None`` waits until
    the Host decides or the prompt ends.
    ``parent_tool_use_id`` is accepted for the shared host loop but has no ACP field.
    ``extension_factory`` is a trusted ``module::factory`` import path, called with
    keyword-only session_id, sio, connection, and config. It must return an async
    context manager yielding CodeBuddySessionExtension; see the module task-affinity
    contract. ``extension_config`` is passed to that factory, not to the CLI.
    ``relay_turn_id`` tags the updates the extension's relay pushes during this prompt.
    ``acp_cli='workbuddy'`` runs ``wb --acp`` configured by ``workbuddy_options`` (version,
    package_parts, package_sha256, preset, mode, effort, features, fork_mode, permission_mode);
    ``fork_source_session_id`` creates this session as a ``session/fork`` of that running session.
    """
    del parent_tool_use_id
    start_time = time.monotonic()
    try:
        session = await _get_or_create_session(
            session_id=session_id,
            working_dir=working_dir,
            version=version,
            npm_registry=npm_registry,
            model_id=model_id,
            extra_env=extra_env,
            extra_cli_args=extra_cli_args,
            config_file_patches=config_file_patches,
            pre_start_scripts=pre_start_scripts,
            permission_policy=permission_policy,
            mcp_servers=mcp_servers,
            extension_factory=extension_factory,
            extension_config=extension_config,
            skip_permissions=skip_permissions,
            deferred_permission_timeout=deferred_permission_timeout,
            acp_cli=acp_cli,
            workbuddy_options=workbuddy_options,
            fork_source_session_id=fork_source_session_id,
        )
    except _CodeBuddySessionStartCancelledError:
        return {
            'success': True,
            'status': 'cancelled',
            'session_id': session_id,
            'output': '',
            'stderr': '',
            'exit_code': 0,
            'execution_time': time.monotonic() - start_time,
            'events': [],
            'acp_stop_reason': 'cancelled',
            'partial': True,
            'relay_frames': None,
        }
    return await session.send_user_message(prompt, timeout=timeout, relay_turn_id=relay_turn_id)


async def side_question_codebuddy_session(
    *, session_id: str, question: str, timeout: float = _SIDE_QUESTION_TIMEOUT_S
) -> dict[str, Any]:
    """Ask one btw question of the running ``session_id`` and return ``{success, answer, error_code}``.

    The session must already be open on this runtime. Earlier btw answers stay on the session and go
    out with the next question. ``error_code`` is the peer's JSON-RPC code when it refused.
    """
    session = await _await_fork_source(session_id)
    if session is None:
        return {
            'success': False,
            'answer': '',
            'session_id': session_id,
            'stderr': f'Unknown CodeBuddy session {session_id}',
            'error_code': -32602,
        }
    return await session.ask_side_question(question, timeout=timeout)


async def steer_codebuddy_session(*, session_id: str, prompt: str, timeout: float = 30.0) -> dict[str, Any]:
    """Insert ``prompt`` into the turn ``session_id`` is running now, on a peer that supports steering.

    Returns:
        ``{success, steered, session_id, stderr, error_code}``; ``steered`` is False when nothing was
        inserted, and ``error_code`` carries the peer's JSON-RPC code when it refused the steer.
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        return {
            'success': False,
            'steered': False,
            'session_id': session_id,
            'stderr': f'Unknown CodeBuddy session {session_id}',
            'error_code': None,
        }
    return await session.steer(prompt, timeout=timeout)


async def interrupt_codebuddy_session(
    *, session_id: str, grace_timeout: float = _CANCEL_GRACE_TIMEOUT
) -> dict[str, Any]:
    """Soft-interrupt through a direct invoke, without requiring an LLM bridge."""
    session = _SESSIONS.get(session_id)
    if session is not None:
        return await session.interrupt(grace_timeout=grace_timeout)
    session_start = _SESSION_STARTS.get(session_id)
    if session_start is not None:
        session_start.cancelled = True
        return {'success': True, 'session_id': session_id, 'stopped': False, 'pending': True, 'stderr': ''}
    return {
        'success': False,
        'session_id': session_id,
        'stopped': False,
        'stderr': f'Unknown CodeBuddy session {session_id}',
    }


async def close_codebuddy_session(*, session_id: str) -> dict[str, Any]:
    """Close an idle session; reject busy startup or active turns without disrupting them."""
    # NOTE(wallyshi): Keep creation locks registered because deleting one can strand pre-existing waiters.
    creation_lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
    if creation_lock.locked():
        session = _SESSIONS.get(session_id)
        if session is not None and session.closing:
            return await session.close()
        return {'success': False, 'session_id': session_id, 'stderr': 'CodeBuddy session is busy in another operation'}
    async with creation_lock:
        session = _SESSIONS.get(session_id)
        if session is None:
            cleanup_task = _SESSION_CLEANUPS.get(session_id)
            stderr = await _wait_for_resource_cleanup(session_id, cleanup_task) if cleanup_task is not None else ''
            return {'success': not stderr, 'session_id': session_id, 'stderr': stderr}
        if session.lock.locked():
            return {
                'success': False,
                'session_id': session_id,
                'stderr': 'CodeBuddy session has an active or draining turn',
            }
        # NOTE(wallyshi): Mark closing before yielding; a caller holding a stale lookup must not acquire a new turn.
        session.closing = True
        try:
            return await session.close()
        finally:
            cleanup_task = session._cleanup_task
            if (
                _SESSIONS.get(session_id) is session
                and cleanup_task is not None
                and cleanup_task.done()
                and not cleanup_task.cancelled()
                and cleanup_task.exception() is None
                and not cleanup_task.result()
            ):
                _SESSIONS.pop(session_id, None)


__all__ = [
    'AcpCli',
    'CodeBuddySessionExtension',
    'close_codebuddy_session',
    'interrupt_codebuddy_session',
    'run_codebuddy_session_turn',
    'side_question_codebuddy_session',
    'steer_codebuddy_session',
]
