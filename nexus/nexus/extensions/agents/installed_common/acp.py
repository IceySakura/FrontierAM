"""Shared ACP JSON-RPC client for installed agents.

CodeBuddy speaks protocol version 1: ``session/prompt`` returns when the turn stops, and the
response withdraws whatever permission or question that turn still had open. WorkBuddy
``wb --acp`` speaks version 2: ``session/prompt`` returns as soon as the prompt is accepted,
the turn ends later with an ``idle`` ``state_update``, and the caller then calls
:meth:`AcpConnection.finish_prompt`. A steer is another ``session/prompt`` carrying
``_meta[ACP_PROMPT_META]``; a branch is ``session/fork``; deliverables arrive as the
``_codebuddy.ai/command`` notification, re-emitted here as a session update of that kind.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast


logger = logging.getLogger(__name__)

ACP_PROTOCOL_VERSION = 1
ACP_PROTOCOL_VERSION_V2 = 2
"""WorkBuddy ``wb --acp`` speaks only this version; ``session/prompt`` returns once the prompt is accepted."""
ACP_PROMPT_META = 'codebuddy.ai/prompt'
"""``_meta`` key of a ``session/prompt`` that steers the running turn: ``{mode: 'steer', expectedRequestId}``."""
ACP_REQUEST_ID_META = 'codebuddy.ai/requestId'
"""``_meta`` key naming the turn on a prompt result and on ``running`` / ``idle`` state updates."""
ACP_ERROR_META = 'codebuddy.ai/error'
"""``_meta`` key of an ``idle`` state update carrying ``{code, message}`` of a failed turn."""
ACP_ERROR_STOP_REASON = '_codebuddy.ai/error'
"""``stopReason`` of an ``idle`` state update that ended a failed turn."""
ACP_COMMAND_METHOD = '_codebuddy.ai/command'
"""WorkBuddy notification naming deliverables; ``AcpConnection`` re-emits it as a session update of this kind."""
ACP_SIDE_QUESTION_METHOD = '_codebuddy.ai/sideQuestion'
"""WorkBuddy btw question: one tool-less answer beside the session, returned whole, not as a turn."""
ACP_SIDE_QUESTION_META = 'codebuddy.ai/sideQuestion'
"""``capabilities.session._meta`` key advertising btw; the value names its limits."""
_SIDE_QUESTION_TURN_LIMIT = 20
"""How many earlier btw answers the client sends back when the peer does not name a limit."""
ACP_COMMAND_ACTIONS = frozenset({'presentFiles', 'openResultView', 'openBrowser'})
ACP_CANCEL_REQUEST_METHOD = '$/cancel_request'
"""Notification by which the peer withdraws one of its own requests, such as a permission or an elicitation."""
PermissionPolicy = Literal['allow', 'reject', 'defer']
DeferredPermissionHandler = Callable[[dict[str, Any], int], Awaitable[dict[str, Any] | None]]
ElicitationHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
"""Answers one ``elicitation/create`` params dict with an ``{action, content?}`` result."""
UpdateListener = Callable[[dict[str, Any]], None]
"""Receives each ``session/update`` params dict synchronously, before it is queued."""
ELICITATION_CREATE_METHOD = 'elicitation/create'
_PERMISSION_KINDS = {'allow_once', 'allow_always', 'reject_once', 'reject_always'}
_ELICITATION_ACTIONS = {'accept', 'decline', 'cancel'}
_CANCELLED_ELICITATION: dict[str, Any] = {'action': 'cancel'}
_MAX_NOISE_LINES = 100
_MAX_NOISE_LINE_LENGTH = 2048
_MAX_CACHED_TOOL_CALLS = 512
_MAX_CACHED_TEXT_CHARS = 2000
_CACHED_TOOL_CALL_KEYS = ('title', 'kind', 'rawInput', 'locations')
_TOOL_NAME_META = 'codebuddy.ai/toolName'


def accepted_request_id(result: object) -> str | None:
    """Return the turn id of a version 2 ``session/prompt`` result, or None when it names none.

    The ``_meta`` request id names the turn. ``messageId`` names the user message, which equals the
    turn id for a regular prompt but not for a steer, so it is only the fallback.
    """
    if not isinstance(result, dict):
        return None
    meta = cast(dict[str, Any], result).get('_meta')
    if isinstance(meta, dict):
        request_id = cast(dict[str, Any], meta).get(ACP_REQUEST_ID_META)
        if isinstance(request_id, str) and request_id:
            return request_id
    message_id = cast(dict[str, Any], result).get('messageId')
    return message_id if isinstance(message_id, str) and message_id else None


def supports_session_fork(initialize_result: object) -> bool:
    """Tell whether an ``initialize`` result advertises ``session/fork``."""
    session = _initialize_session_capabilities(initialize_result)
    return isinstance(session.get('fork'), dict)


def supports_side_question(initialize_result: object) -> bool:
    """Tell whether an ``initialize`` result advertises ``_codebuddy.ai/sideQuestion``."""
    return _side_question_capability(initialize_result) is not None


def side_question_turn_limit(initialize_result: object) -> int:
    """How many earlier btw answers to send back; the peer's ``maxTurns``, or 20."""
    capability = _side_question_capability(initialize_result)
    limit = capability.get('maxTurns') if capability is not None else None
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return limit
    return _SIDE_QUESTION_TURN_LIMIT


def _side_question_capability(initialize_result: object) -> dict[str, Any] | None:
    session = _initialize_session_capabilities(initialize_result)
    meta = session.get('_meta')
    if not isinstance(meta, dict):
        return None
    capability = meta.get(ACP_SIDE_QUESTION_META)
    return capability if isinstance(capability, dict) else None


def supports_prompt_steer(initialize_result: object) -> bool:
    """Tell whether an ``initialize`` result advertises steering a running prompt."""
    prompt = _initialize_session_capabilities(initialize_result).get('prompt')
    meta = prompt.get('_meta') if isinstance(prompt, dict) else None
    steer = meta.get(ACP_PROMPT_META) if isinstance(meta, dict) else None
    modes = steer.get('modes') if isinstance(steer, dict) else None
    return isinstance(modes, list) and 'steer' in modes


def _initialize_session_capabilities(initialize_result: object) -> dict[str, Any]:
    if not isinstance(initialize_result, dict):
        return {}
    capabilities = cast(dict[str, Any], initialize_result).get('capabilities')
    session = capabilities.get('session') if isinstance(capabilities, dict) else None
    return cast(dict[str, Any], session) if isinstance(session, dict) else {}


class AcpProtocolError(RuntimeError):
    """The ACP stream ended or contained an invalid frame."""


class AcpRemoteError(AcpProtocolError):
    """The ACP server returned a JSON-RPC error response."""

    def __init__(self, error: object):
        self.error = error
        super().__init__(f'ACP request failed: {error!r}')


@dataclass
class _DeferredPermission:
    session_id: str
    options: list[dict[str, Any]]
    future: asyncio.Future[dict[str, Any]]
    deadline: float | None


@dataclass
class _PendingElicitation:
    session_id: str | None
    callback: asyncio.Future[dict[str, Any]]


@dataclass(frozen=True)
class _SessionHandlers:
    """Handlers that answer the reverse requests of one remote session instead of the connection's."""

    permission_handler: DeferredPermissionHandler | None
    elicitation_handler: ElicitationHandler | None


class AcpConnection:
    """Correlate ACP requests, updates, permission and elicitation callbacks over NDJSON."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: Any,
        *,
        permission_policy: PermissionPolicy = 'allow',
        tolerate_noise: bool = False,
        deferred_permission_handler: DeferredPermissionHandler | None = None,
        deferred_permission_timeout: float | None = 300.0,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> None:
        if permission_policy not in ('allow', 'reject', 'defer'):
            raise ValueError(f'Invalid permission_policy {permission_policy!r}; choose allow, reject, or defer')
        if deferred_permission_timeout is not None and (
            not math.isfinite(deferred_permission_timeout) or deferred_permission_timeout <= 0
        ):
            raise ValueError('deferred_permission_timeout must be positive and finite, or explicitly None')
        self.reader = reader
        self.writer = writer
        self.permission_policy = permission_policy
        self.tolerate_noise = tolerate_noise
        self.deferred_permission_handler = deferred_permission_handler
        self.deferred_permission_timeout = deferred_permission_timeout
        self.elicitation_handler = elicitation_handler
        """Answers ``elicitation/create``; without one every elicitation is answered ``cancel``.
        Each answer is bounded by ``deferred_permission_timeout`` like a deferred permission;
        None waits until the handler answers or the prompt ends or is cancelled."""
        self.noise: list[str] = []
        self.updates: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._update_listeners: list[UpdateListener] = []
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._prompt_sessions: dict[int, str] = {}
        self._deferred_permissions: dict[int, _DeferredPermission] = {}
        self._elicitations: dict[Any, _PendingElicitation] = {}
        self._permission_tasks: set[asyncio.Future[Any]] = set()
        self._cancelled_permission_tasks: set[asyncio.Future[Any]] = set()
        self._cancelled_sessions: set[str] = set()
        self.protocol_version = ACP_PROTOCOL_VERSION
        """Version negotiated by :meth:`initialize`; version 2 prompts return once they are accepted."""
        self.config_options: dict[str, list[dict[str, Any]]] = {}
        """Latest version 2 ``configOptions`` per remote session, replaced whole on every change."""
        self._open_turns: set[str] = set()
        self._session_handlers: dict[str, _SessionHandlers] = {}
        self._tool_calls: dict[str, dict[str, dict[str, Any]]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._close_waiter: asyncio.Future[Any] | None = None
        self._closed = False

    def add_update_listener(self, listener: UpdateListener) -> Callable[[], None]:
        """Observe every ``session/update`` as it arrives, without consuming it from ``updates``.

        Listeners run inline on the read loop in registration order, so they must not block;
        an exception is logged and never reaches the queue or the other listeners.

        Returns:
            A callable that removes the listener; calling it again is a no-op.
        """
        self._update_listeners.append(listener)

        def remove() -> None:
            if listener in self._update_listeners:
                self._update_listeners.remove(listener)

        return remove

    def bind_session_handlers(
        self,
        remote_session_id: str,
        *,
        permission_handler: DeferredPermissionHandler | None = None,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> Callable[[], None]:
        """Answer the permissions and elicitations of one remote session with its own handlers.

        Several sessions can share one process, such as a ``session/fork`` and its source. A bound
        session uses only these handlers: a None handler answers like a connection without one
        (``cancelled`` / ``cancel``), never the connection-level handler of another session.
        Sessions that are not bound keep the connection-level handlers.

        Returns:
            A callable that removes the binding; calling it again is a no-op.

        Raises:
            ValueError: ``remote_session_id`` is empty or already bound.
        """
        if not remote_session_id:
            raise ValueError('remote_session_id must not be empty')
        if remote_session_id in self._session_handlers:
            raise ValueError(f'ACP session {remote_session_id!r} already has bound handlers; unbind them first')
        handlers = _SessionHandlers(permission_handler=permission_handler, elicitation_handler=elicitation_handler)
        self._session_handlers[remote_session_id] = handlers

        def unbind() -> None:
            if self._session_handlers.get(remote_session_id) is handlers:
                del self._session_handlers[remote_session_id]

        return unbind

    def start(self) -> None:
        if self._closed:
            raise AcpProtocolError('ACP connection is closed')
        if self.permission_policy == 'defer' and self.deferred_permission_handler is None:
            raise ValueError('permission_policy="defer" requires a deferred_permission_handler before start')
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(self._read_loop())

    async def initialize(
        self,
        *,
        timeout: float = 60.0,
        client_capabilities: dict[str, Any] | None = None,
        protocol_version: int = ACP_PROTOCOL_VERSION,
    ) -> dict[str, Any]:
        """Negotiate ACP, declaring ``client_capabilities`` (none by default).

        Version 1 names the client ``clientInfo`` / ``clientCapabilities``; version 2, the only one
        ``wb --acp`` speaks, names them ``info`` / ``capabilities``. The peer must answer with the
        same version; the result is returned unchanged.

        Raises:
            ValueError: ``protocol_version`` is neither 1 nor 2.
            AcpProtocolError: The peer negotiated another version.
        """
        if protocol_version not in (ACP_PROTOCOL_VERSION, ACP_PROTOCOL_VERSION_V2):
            raise ValueError(
                f'Unsupported ACP protocol_version {protocol_version!r}; '
                f'choose {ACP_PROTOCOL_VERSION} or {ACP_PROTOCOL_VERSION_V2}'
            )
        capabilities = deepcopy(client_capabilities or {})
        if protocol_version == ACP_PROTOCOL_VERSION_V2:
            params: dict[str, Any] = {
                'protocolVersion': protocol_version,
                'info': {'name': 'nexus', 'version': '1'},
                'capabilities': capabilities,
            }
        else:
            params = {
                'protocolVersion': protocol_version,
                'clientCapabilities': capabilities,
                'clientInfo': {'name': 'nexus', 'version': '1'},
            }
        result = await self.request('initialize', params, timeout=timeout)
        if not isinstance(result, dict) or result.get('protocolVersion') != protocol_version:
            raise AcpProtocolError(f'ACP peer negotiated an unexpected protocol version: {result!r}')
        self.protocol_version = protocol_version
        return result

    async def new_session(
        self, *, cwd: str, timeout: float = 60.0, mcp_servers: list[dict[str, Any]] | None = None
    ) -> str:
        """Open a session in ``cwd`` and return its remote id.

        A version 2 request names ``mcpServers`` only when some are given and never
        ``additionalDirectories``: ``wb --acp`` answers either key with -32602. The CodeBuddy session
        runtime refuses MCP servers for WorkBuddy before it gets here.
        """
        if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
            params: dict[str, Any] = {'cwd': cwd}
            if mcp_servers:
                params['mcpServers'] = list(mcp_servers)
        else:
            params = {'cwd': cwd, 'mcpServers': list(mcp_servers or []), 'additionalDirectories': []}
        result = await self.request('session/new', params, timeout=timeout)
        session_id = result.get('sessionId') if isinstance(result, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise AcpProtocolError(f'ACP peer returned an invalid session/new response: {result!r}')
        self._record_config_options(session_id, result)
        return session_id

    async def fork_session(self, *, session_id: str, cwd: str, timeout: float = 60.0) -> str:
        """Branch ``session_id`` into a new idle session and return the new remote id.

        The branch copies the source up to its last finished turn, even while the source is running.
        WorkBuddy requires ``cwd`` to be the source's own directory.

        Raises:
            AcpProtocolError: The peer is not version 2 or returned no session id.
        """
        self._require_protocol_v2('session/fork')
        result = await self.request('session/fork', {'sessionId': session_id, 'cwd': cwd}, timeout=timeout)
        forked = result.get('sessionId') if isinstance(result, dict) else None
        if not isinstance(forked, str) or not forked:
            raise AcpProtocolError(f'ACP peer returned an invalid session/fork response: {result!r}')
        self._record_config_options(forked, result)
        return forked

    async def close_session(self, *, session_id: str, timeout: float = 60.0) -> None:
        """Close one remote session; its running turn, permissions and questions end with it.

        The local state of the session is forgotten even when the request fails, since the peer
        treats a closed or unknown session the same way.

        Raises:
            AcpProtocolError: The peer is not version 2, the request failed, or the transport closed.
        """
        self._require_protocol_v2('session/close')
        try:
            await self.request('session/close', {'sessionId': session_id}, timeout=timeout)
        finally:
            self._open_turns.discard(session_id)
            self._tool_calls.pop(session_id, None)
            self.config_options.pop(session_id, None)
            self._cancel_session_permissions(session_id)

    async def set_config_option(
        self,
        *,
        session_id: str,
        config_id: str,
        value: str | bool,
        value_type: str = 'id',
        timeout: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Set one ``configOptions`` entry, such as ``permission_mode`` or ``mode``, and return them all.

        The value must be one the session's latest ``configOptions`` lists; the peer answers others
        with -32602 and a ``mode`` change during a turn with -32600.

        Raises:
            AcpProtocolError: The peer is not version 2 or returned no ``configOptions``.
        """
        self._require_protocol_v2('session/set_config_option')
        result = await self.request(
            'session/set_config_option',
            {'sessionId': session_id, 'configId': config_id, 'type': value_type, 'value': value},
            timeout=timeout,
        )
        if not isinstance(result, dict) or not isinstance(result.get('configOptions'), list):
            raise AcpProtocolError(f'ACP peer returned an invalid session/set_config_option response: {result!r}')
        self._record_config_options(session_id, result)
        return self.config_options[session_id]

    async def prompt(
        self,
        *,
        session_id: str,
        text: str,
        timeout: float,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one prompt; what the result means depends on the negotiated version.

        Version 1 returns ``stopReason`` when the turn ends and withdraws the turn's permissions.
        Version 2 returns the accepted request id (see :func:`accepted_request_id`); the turn goes
        on until an ``idle`` update, after which the caller calls :meth:`finish_prompt`. ``meta``
        is the version 2 ``_meta``, such as a steer; a steer joins the running turn, so it neither
        reopens a cancelled session nor opens a turn.

        Raises:
            ValueError: ``meta`` is given to a version 1 peer.
            AcpProtocolError: The result has no ``stopReason`` (version 1) or no request id (version 2).
        """
        if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
            return await self._accept_prompt(session_id=session_id, text=text, timeout=timeout, meta=meta)
        if meta is not None:
            raise ValueError('ACP version 1 session/prompt carries no _meta; steering needs a version 2 peer')
        self._cancelled_sessions.discard(session_id)
        try:
            result = await self.request(
                'session/prompt',
                {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': text}]},
                timeout=timeout,
            )
            if not isinstance(result, dict) or not isinstance(result.get('stopReason'), str):
                raise AcpProtocolError(f'ACP peer returned an invalid session/prompt response: {result!r}')
            return result
        finally:
            self._cancel_session_permissions(session_id)

    async def _accept_prompt(
        self, *, session_id: str, text: str, timeout: float, meta: dict[str, Any] | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': text}]}
        was_cancelled = session_id in self._cancelled_sessions
        if meta is None:
            # NOTE(wallyshi): Reopen before sending; the turn's first permission can be read in the same
            # buffer as the accept result, before this coroutine resumes.
            self._cancelled_sessions.discard(session_id)
        else:
            params['_meta'] = deepcopy(meta)
        try:
            result = await self.request('session/prompt', params, timeout=timeout)
            if accepted_request_id(result) is None:
                raise AcpProtocolError(f'ACP peer accepted session/prompt without a request id: {result!r}')
        except BaseException:
            # NOTE(wallyshi): A refused prompt (such as -32600 while a turn runs) must not reopen a session
            # whose turn is being cancelled.
            if was_cancelled and meta is None:
                self._cancelled_sessions.add(session_id)
            raise
        if meta is None:
            self._open_turns.add(session_id)
        return cast(dict[str, Any], result)

    def finish_prompt(self, session_id: str) -> None:
        """End a version 2 turn locally once its ``idle`` update arrived: withdraw what it left open.

        Permissions and questions of the session still waiting are answered ``cancelled`` /
        ``cancel``, later ones are refused until the next regular prompt, and the cached tool calls
        of the turn are dropped.
        """
        self._open_turns.discard(session_id)
        self._tool_calls.pop(session_id, None)
        self._cancel_session_permissions(session_id)

    async def cancel(self, *, session_id: str) -> None:
        self._cancel_session_permissions(session_id)
        await self.notify('session/cancel', {'sessionId': session_id})

    async def extension_request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
        """Send one ACP extension request; ACP reserves method names starting with ``_`` for them."""
        if not method.startswith('_') or len(method) < 2:
            raise ValueError(f'ACP extension methods must start with "_", got {method!r}')
        return await self.request(method, params, timeout=timeout)

    async def side_question(
        self,
        *,
        session_id: str,
        question: str,
        turns: list[dict[str, str]] | None = None,
        timeout: float,
    ) -> str:
        """Ask one tool-less question beside ``session_id`` and return the whole answer.

        The peer reads the session's own context. ``turns`` are earlier btw answers this client kept;
        the peer does not store them. Cancelling this call sends ``$/cancel_request`` for it alone.

        Raises:
            AcpProtocolError: The peer is not version 2, the request failed, or the answer is not text.
            ValueError: ``question`` is empty.
        """
        self._require_protocol_v2(ACP_SIDE_QUESTION_METHOD)
        if not question:
            raise ValueError('A side question must be a non-empty string')
        params: dict[str, Any] = {'sessionId': session_id, 'question': question}
        if turns:
            params['turns'] = [{'question': turn['question'], 'answer': turn['answer']} for turn in turns]
        result = await self.request(ACP_SIDE_QUESTION_METHOD, params, timeout=timeout, cancel_on_cancel=True)
        answer = result.get('answer') if isinstance(result, dict) else None
        if not isinstance(answer, str):
            raise AcpProtocolError(f'ACP peer returned an invalid side question response: {result!r}')
        return answer

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float,
        cancel_on_cancel: bool = False,
    ) -> Any:
        if self._closed:
            raise AcpProtocolError('ACP connection is closed')
        self.start()
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        # NOTE(wallyshi): Only a version 1 prompt result ends the turn. A version 2 result just means the
        # prompt was accepted, so tracking it would withdraw every later permission and question of the turn.
        if (
            method == 'session/prompt'
            and self.protocol_version < ACP_PROTOCOL_VERSION_V2
            and isinstance(params.get('sessionId'), str)
        ):
            self._prompt_sessions[request_id] = params['sessionId']
        try:
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.CancelledError:
            if cancel_on_cancel:
                await self._cancel_peer_request(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            self._prompt_sessions.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # NOTE(wallyshi): A write failure may bypass the await of an already-failed response future.
                future.exception()

    async def _cancel_peer_request(self, request_id: int) -> None:
        """Tell the peer to drop request ``request_id``; the notification outlives this call being cancelled."""
        notify = asyncio.create_task(
            self.notify(ACP_CANCEL_REQUEST_METHOD, {'requestId': request_id}),
            name=f'acp-cancel-request:{request_id}',
        )
        try:
            await asyncio.shield(notify)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning('Failed to cancel ACP request %s', request_id, exc_info=True)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if self._closed:
            return
        await self._write({'jsonrpc': '2.0', 'method': method, 'params': params})

    def drain_updates(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        while True:
            try:
                update = self.updates.get_nowait()
            except asyncio.QueueEmpty:
                break
            if session_id is None or update.get('sessionId') == session_id:
                updates.append(update)
            else:
                retained.append(update)
        for update in retained:
            self.updates.put_nowait(update)
        return updates

    async def close(self) -> None:
        self._shutdown(AcpProtocolError('ACP connection closed'))
        # NOTE(wallyshi): A callback can close its own connection; it must not join the task waiting for it.
        if asyncio.current_task() in self._permission_tasks:
            return
        if self._close_waiter is not None:
            await asyncio.shield(self._close_waiter)

    def _shutdown(self, error: AcpProtocolError) -> None:
        if self._closed:
            return
        self._closed = True
        self._fail_pending(error)
        self._pending.clear()
        self._prompt_sessions.clear()
        self._open_turns.clear()
        self._tool_calls.clear()
        for permission in self._deferred_permissions.values():
            permission.future.cancel()
        self._deferred_permissions.clear()
        self._elicitations.clear()
        tasks = set(self._permission_tasks)
        if self._reader_task is not None:
            tasks.add(self._reader_task)
        current_task = asyncio.current_task()
        for task in tasks:
            if task is not current_task and task not in self._cancelled_permission_tasks:
                if task in self._permission_tasks:
                    self._cancelled_permission_tasks.add(task)
                task.cancel()
        if tasks:
            self._close_waiter = asyncio.gather(*tasks, return_exceptions=True)

    async def _write(self, frame: dict[str, Any]) -> None:
        data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        async with self._write_lock:
            if self._closed:
                raise AcpProtocolError('ACP connection is closed')
            try:
                self.writer.write(data)
                await self.writer.drain()
            except Exception as exc:
                error = AcpProtocolError(f'Failed to write ACP frame: {exc}')
                self._shutdown(error)
                raise error from exc

    async def _read_loop(self) -> None:
        error: AcpProtocolError | None = None
        try:
            while True:
                line = await self.reader.readline()
                if not line:
                    error = AcpProtocolError('ACP peer stdout closed')
                    break
                try:
                    frame = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if self.tolerate_noise:
                        self._record_noise(line)
                        continue
                    error = AcpProtocolError(f'ACP peer emitted invalid JSON: {exc}')
                    break
                if not isinstance(frame, dict) or frame.get('jsonrpc') != '2.0':
                    if self.tolerate_noise:
                        self._record_noise(line)
                        continue
                    error = AcpProtocolError(f'ACP peer emitted an invalid frame: {frame!r}')
                    break
                await self._dispatch(frame)
        except asyncio.CancelledError:
            error = AcpProtocolError('ACP reader was cancelled')
            raise
        except Exception as exc:
            error = AcpProtocolError(f'ACP reader failed: {exc}')
            logger.exception('ACP reader failed')
        finally:
            if error is not None:
                self._shutdown(error)

    def _record_noise(self, line: bytes) -> None:
        logger.warning('Ignoring non-ACP stdout from peer: %r', line[:200])
        self.noise.append(line[:_MAX_NOISE_LINE_LENGTH].decode('utf-8', errors='replace').rstrip())
        del self.noise[:-_MAX_NOISE_LINES]

    def _require_protocol_v2(self, method: str) -> None:
        if self.protocol_version < ACP_PROTOCOL_VERSION_V2:
            raise AcpProtocolError(
                f'{method} needs an ACP version {ACP_PROTOCOL_VERSION_V2} peer, but version '
                f'{self.protocol_version} was negotiated; initialize with protocol_version={ACP_PROTOCOL_VERSION_V2}'
            )

    def _record_config_options(self, session_id: str, result: object) -> None:
        options = cast(dict[str, Any], result).get('configOptions') if isinstance(result, dict) else None
        if isinstance(options, list):
            self.config_options[session_id] = [dict(option) for option in options if isinstance(option, dict)]

    def _notify_update_listeners(self, params: dict[str, Any]) -> None:
        for listener in list(self._update_listeners):
            try:
                listener(params)
            except Exception:
                logger.exception('ACP update listener failed; the update is still queued')

    def _observe_update(self, params: dict[str, Any]) -> None:
        """Keep what a version 2 permission request leaves out: the tool call it is about, and config changes."""
        session_id = params.get('sessionId')
        update = params.get('update')
        if not isinstance(session_id, str) or not isinstance(update, dict):
            return
        kind = update.get('sessionUpdate')
        if kind == 'config_option_update':
            self._record_config_options(session_id, update)
            return
        if kind not in ('tool_call', 'tool_call_update'):
            return
        tool_call_id = update.get('toolCallId')
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return
        calls = self._tool_calls.setdefault(session_id, {})
        entry = calls.get(tool_call_id)
        if entry is None:
            if len(calls) >= _MAX_CACHED_TOOL_CALLS:
                calls.pop(next(iter(calls)))
            entry = calls[tool_call_id] = {}
        for key in _CACHED_TOOL_CALL_KEYS:
            if update.get(key) is not None:
                entry[key] = _cut_cached(update[key])
        meta = update.get('_meta')
        tool_name = meta.get(_TOOL_NAME_META) if isinstance(meta, dict) else None
        if isinstance(tool_name, str) and tool_name:
            entry['name'] = tool_name

    def _with_cached_tool_call(self, session_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Give a version 2 permission the ``toolCall`` a version 1 one carries, from the cached tool updates.

        Version 2 names only ``subject.toolCall`` ``{toolCallId, name}``, so without this the Host cannot
        show what the tool is about to do. The copy carries no ``_meta``: a tool name there would make the
        Host treat the request as CodeBuddy's ``AskUserQuestion``.
        """
        if 'toolCall' in params:
            return params
        subject = params.get('subject')
        tool = subject.get('toolCall') if isinstance(subject, dict) else None
        if not isinstance(tool, dict):
            return params
        tool_call_id = tool.get('toolCallId')
        cached = self._tool_calls.get(session_id, {}).get(tool_call_id) if isinstance(tool_call_id, str) else None
        tool_call = deepcopy(cached) if cached is not None else {}
        tool_call.update({key: tool[key] for key in ('toolCallId', 'name') if key in tool})
        return {**params, 'toolCall': tool_call}

    @staticmethod
    def _command_update(raw: object) -> dict[str, Any] | None:
        """Re-shape a ``_codebuddy.ai/command`` notification as a session update; None for unknown actions."""
        if not isinstance(raw, dict):
            return None
        params = cast(dict[str, Any], raw)
        action = params.get('action')
        if action not in ACP_COMMAND_ACTIONS:
            logger.info('Ignoring ACP command with unknown action %r', action)
            return None
        command_params = params.get('params')
        return {
            'sessionId': params.get('sessionId'),
            'update': {
                'sessionUpdate': ACP_COMMAND_METHOD,
                'action': action,
                'params': deepcopy(command_params) if isinstance(command_params, dict) else {},
            },
        }

    def _withdraw_peer_request(self, request_id: object) -> None:
        """Stop waiting on a permission or elicitation the peer took back; it is still answered once."""
        if type(request_id) is int and self.cancel_deferred_permission(request_id):
            logger.info('ACP peer withdrew permission request %s', request_id)
            return
        elicitation = self._elicitations.get(request_id) if isinstance(request_id, (int, str)) else None
        if elicitation is not None and not elicitation.callback.done():
            logger.info('ACP peer withdrew elicitation %s', request_id)
            elicitation.callback.cancel()
            return
        logger.debug('ACP peer withdrew request %r, which is not waiting here', request_id)

    async def _dispatch(self, frame: dict[str, Any]) -> None:
        method = frame.get('method')
        request_id = frame.get('id')
        if method == 'session/update' and request_id is None:
            params = frame.get('params')
            if isinstance(params, dict):
                if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
                    self._observe_update(params)
                self._notify_update_listeners(params)
                await self.updates.put(params)
            return
        if method == ACP_COMMAND_METHOD and request_id is None:
            command = self._command_update(frame.get('params'))
            if command is not None:
                self._notify_update_listeners(command)
                await self.updates.put(command)
            return
        if method == ACP_CANCEL_REQUEST_METHOD and request_id is None:
            params = frame.get('params')
            self._withdraw_peer_request(params.get('requestId') if isinstance(params, dict) else None)
            return

        if isinstance(method, str) and request_id is not None:
            if method == 'session/request_permission':
                params = frame.get('params')
                session_id = params.get('sessionId') if isinstance(params, dict) else None
                bound = self._session_handlers.get(session_id) if isinstance(session_id, str) else None
                handler = bound.permission_handler if bound is not None else self.deferred_permission_handler
                if (
                    self.permission_policy == 'defer'
                    and isinstance(params, dict)
                    and isinstance(session_id, str)
                    and session_id not in self._cancelled_sessions
                    and type(request_id) is int
                    and (bound is None or handler is not None)
                ):
                    if request_id in self._deferred_permissions:
                        raise AcpProtocolError(f'ACP peer reused pending permission request ID {request_id}')
                    if self.protocol_version >= ACP_PROTOCOL_VERSION_V2:
                        params = self._with_cached_tool_call(session_id, params)
                    loop = asyncio.get_running_loop()
                    timeout = self.deferred_permission_timeout
                    permission = _DeferredPermission(
                        session_id=session_id,
                        options=self._permission_options(params),
                        future=loop.create_future(),
                        deadline=None if timeout is None else loop.time() + timeout,
                    )
                    self._deferred_permissions[request_id] = permission
                    task = asyncio.create_task(self._run_deferred_permission(params, request_id, permission, handler))
                    self._permission_tasks.add(task)
                    task.add_done_callback(self._permission_task_done)
                    return
                result = (
                    {'outcome': {'outcome': 'cancelled'}}
                    if (
                        self.permission_policy == 'defer'
                        or not isinstance(session_id, str)
                        or session_id in self._cancelled_sessions
                    )
                    else self._permission_response(params)
                )
                await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})
            elif method == ELICITATION_CREATE_METHOD:
                await self._start_elicitation(frame.get('params'), request_id)
            else:
                await self._write({
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'error': {'code': -32601, 'message': f'Method not found: {method}'},
                })
            return

        if method is None and isinstance(request_id, int):
            future = self._pending.get(request_id)
            if future is None or future.done():
                return
            session_id = self._prompt_sessions.get(request_id)
            if session_id is not None:
                self._cancel_session_permissions(session_id)
            if 'error' in frame:
                future.set_exception(AcpRemoteError(frame['error']))
            elif 'result' in frame:
                future.set_result(frame['result'])
            else:
                future.set_exception(AcpProtocolError(f'ACP response has neither result nor error: {frame!r}'))

    async def _run_deferred_permission(
        self,
        params: dict[str, Any],
        request_id: int,
        permission: _DeferredPermission,
        handler: DeferredPermissionHandler | None,
    ) -> None:
        callback: asyncio.Future[dict[str, Any] | None] | None = None
        result: dict[str, Any] = {'outcome': {'outcome': 'cancelled'}}
        try:
            if not permission.future.done():
                if handler is None:
                    raise ValueError('Deferred permission handler was removed after start')
                callback = asyncio.ensure_future(handler(params, request_id))
                self._permission_tasks.add(callback)
                callback.add_done_callback(self._permission_task_done)
                remaining = (
                    None
                    if permission.deadline is None
                    else max(0.0, permission.deadline - asyncio.get_running_loop().time())
                )
                done, _ = await asyncio.wait(
                    {callback, permission.future}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    raise asyncio.TimeoutError
                if not permission.future.done():
                    if callback.cancelled():
                        logger.warning('ACP deferred permission callback %s was cancelled', request_id)
                        permission.future.set_result(result)
                    else:
                        decision = callback.result()
                        if decision is not None:
                            permission.future.set_result(self._validate_permission_result(decision, permission.options))
            remaining = (
                None
                if permission.deadline is None
                else max(0.0, permission.deadline - asyncio.get_running_loop().time())
            )
            result = await asyncio.wait_for(permission.future, timeout=remaining)
        except asyncio.TimeoutError:
            logger.warning('ACP deferred permission %s timed out', request_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('ACP deferred permission %s failed; cancelling permission', request_id)
        finally:
            if callback is not None and not callback.done() and not self._closed:
                # NOTE(wallyshi): Python 3.10 lacks Task.cancelling; do not interrupt async cleanup with a second cancel.
                self._cancelled_permission_tasks.add(callback)
                callback.cancel()
            if not permission.future.done():
                permission.future.cancel()
            self._deferred_permissions.pop(request_id, None)
        if not self._closed:
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

    async def _start_elicitation(self, params: object, request_id: Any) -> None:
        """Hand one ``elicitation/create`` to the handler, or answer ``cancel`` when nobody can ask."""
        parameters = cast(dict[str, Any], params) if isinstance(params, dict) else None
        raw_session_id = parameters.get('sessionId') if parameters is not None else None
        session_id = raw_session_id if isinstance(raw_session_id, str) else None
        bound = self._session_handlers.get(session_id) if session_id is not None else None
        handler = bound.elicitation_handler if bound is not None else self.elicitation_handler
        if (
            handler is None
            or parameters is None
            or (session_id is not None and session_id in self._cancelled_sessions)
            or (session_id is None and not self._prompt_sessions and not self._open_turns)
            or not isinstance(request_id, (int, str))
            or request_id in self._elicitations
        ):
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': dict(_CANCELLED_ELICITATION)})
            return
        callback = asyncio.ensure_future(handler(parameters))
        self._permission_tasks.add(callback)
        callback.add_done_callback(self._permission_task_done)
        self._elicitations[request_id] = _PendingElicitation(session_id=session_id, callback=callback)
        task = asyncio.create_task(self._run_elicitation(request_id, callback))
        self._permission_tasks.add(task)
        task.add_done_callback(self._permission_task_done)

    async def _run_elicitation(self, request_id: Any, callback: asyncio.Future[dict[str, Any]]) -> None:
        result: dict[str, Any] = dict(_CANCELLED_ELICITATION)
        try:
            done, _ = await asyncio.wait({callback}, timeout=self.deferred_permission_timeout)
            if not done:
                raise asyncio.TimeoutError
            if callback.cancelled():
                logger.info('ACP elicitation %s was withdrawn', request_id)
            else:
                result = self._validate_elicitation_result(callback.result())
        except asyncio.TimeoutError:
            logger.warning('ACP elicitation %s timed out', request_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('ACP elicitation %s failed; cancelling it', request_id)
        finally:
            if not callback.done() and not self._closed:
                self._cancelled_permission_tasks.add(callback)
                callback.cancel()
            self._elicitations.pop(request_id, None)
        if not self._closed:
            await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

    @staticmethod
    def _validate_elicitation_result(result: object) -> dict[str, Any]:
        response = cast(dict[str, Any], result) if isinstance(result, dict) else {}
        action = response.get('action')
        if action not in _ELICITATION_ACTIONS:
            # The result may carry the user's answer, so only its shape is logged.
            logger.warning(
                'Invalid ACP elicitation result; cancelling it: %s',
                f'keys={sorted(str(key) for key in response)}' if isinstance(result, dict) else type(result).__name__,
            )
            return dict(_CANCELLED_ELICITATION)
        if action == 'accept' and isinstance(response.get('content'), dict):
            return {'action': 'accept', 'content': response['content']}
        return {'action': action}

    def _permission_task_done(self, task: asyncio.Future[Any]) -> None:
        self._permission_tasks.discard(task)
        self._cancelled_permission_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error('ACP permission task failed', exc_info=task.exception())

    def resolve_deferred_permission(self, request_id: int, *, option_id: str) -> bool:
        """Resolve one pending permission using an exact option ID supplied by the peer."""
        permission = self._deferred_permissions.get(request_id)
        if permission is None or permission.future.done():
            return False
        result = self._validate_permission_result(
            {'outcome': {'outcome': 'selected', 'optionId': option_id}}, permission.options
        )
        permission.future.set_result(result)
        return result['outcome']['outcome'] == 'selected'

    def cancel_deferred_permission(self, request_id: int) -> bool:
        permission = self._deferred_permissions.get(request_id)
        if permission is None or permission.future.done():
            return False
        permission.future.set_result({'outcome': {'outcome': 'cancelled'}})
        return True

    def has_deferred_permission(self, request_id: int) -> bool:
        """Tell whether this deferred permission still waits for an answer."""
        permission = self._deferred_permissions.get(request_id)
        return permission is not None and not permission.future.done()

    def _cancel_session_permissions(self, session_id: str) -> None:
        """Answer every deferred permission and elicitation of this session ``cancelled``.

        An elicitation without a ``sessionId`` is cancelled with any session's prompt, so a
        ``None`` timeout cannot leave it waiting after the prompt that asked it is over.
        """
        self._cancelled_sessions.add(session_id)
        for request_id, permission in self._deferred_permissions.items():
            if permission.session_id == session_id:
                self.cancel_deferred_permission(request_id)
        for elicitation in self._elicitations.values():
            if elicitation.session_id in (session_id, None) and not elicitation.callback.done():
                elicitation.callback.cancel()

    @staticmethod
    def _permission_options(params: object) -> list[dict[str, Any]]:
        parameters = cast(dict[str, Any], params) if isinstance(params, dict) else {}
        options = parameters.get('options')
        if not isinstance(options, list):
            return []
        return [
            {'optionId': option['optionId'], 'kind': option['kind']}
            for option in options
            if isinstance(option, dict)
            and isinstance(option.get('optionId'), str)
            and option['optionId']
            and isinstance(option.get('kind'), str)
            and option['kind'] in _PERMISSION_KINDS
        ]

    @staticmethod
    def _validate_permission_result(result: object, options: list[dict[str, Any]]) -> dict[str, Any]:
        response = cast(dict[str, Any], result) if isinstance(result, dict) else {}
        outcome = response.get('outcome')
        if isinstance(outcome, dict):
            if outcome.get('outcome') == 'cancelled':
                return {'outcome': {'outcome': 'cancelled'}}
            option_id = outcome.get('optionId')
            if outcome.get('outcome') == 'selected' and any(option['optionId'] == option_id for option in options):
                return {'outcome': {'outcome': 'selected', 'optionId': option_id}}
        logger.warning('Invalid ACP permission outcome; cancelling permission: %r', result)
        return {'outcome': {'outcome': 'cancelled'}}

    def _permission_response(self, params: object) -> dict[str, Any]:
        choices = self._permission_options(params)
        kind = 'allow_once' if self.permission_policy == 'allow' else 'reject_once'
        selected = next((option for option in choices if option['kind'] == kind), None)
        if selected is not None:
            return {'outcome': {'outcome': 'selected', 'optionId': selected['optionId']}}
        return {'outcome': {'outcome': 'cancelled'}}

    def _fail_pending(self, error: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)


def _cut_cached(value: Any) -> Any:
    """Copy a cached tool field with long strings cut, so a written file's content is not kept whole."""
    if isinstance(value, str):
        return value[:_MAX_CACHED_TEXT_CHARS]
    if isinstance(value, dict):
        return {key: _cut_cached(item) for key, item in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_cut_cached(item) for item in cast(list[Any], value)]
    return value


def assistant_text_from_updates(updates: list[dict[str, Any]]) -> str:
    """Concatenate committed assistant text carried by ACP updates."""
    chunks: list[str] = []
    for notification in updates:
        update = notification.get('update')
        if not isinstance(update, dict) or update.get('sessionUpdate') != 'agent_message_chunk':
            continue
        content = update.get('content')
        if isinstance(content, dict) and content.get('type') == 'text' and isinstance(content.get('text'), str):
            chunks.append(content['text'])
    return ''.join(chunks)


__all__ = [
    'ACP_CANCEL_REQUEST_METHOD',
    'ACP_COMMAND_ACTIONS',
    'ACP_COMMAND_METHOD',
    'ACP_ERROR_META',
    'ACP_ERROR_STOP_REASON',
    'ACP_PROMPT_META',
    'ACP_PROTOCOL_VERSION',
    'ACP_PROTOCOL_VERSION_V2',
    'ACP_REQUEST_ID_META',
    'ACP_SIDE_QUESTION_META',
    'ACP_SIDE_QUESTION_METHOD',
    'ELICITATION_CREATE_METHOD',
    'AcpConnection',
    'AcpProtocolError',
    'AcpRemoteError',
    'DeferredPermissionHandler',
    'ElicitationHandler',
    'PermissionPolicy',
    'UpdateListener',
    'accepted_request_id',
    'assistant_text_from_updates',
    'side_question_turn_limit',
    'supports_prompt_steer',
    'supports_session_fork',
    'supports_side_question',
]
