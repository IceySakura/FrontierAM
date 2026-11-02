"""Runtime-side channel RPC from sandboxed agent sessions to Host-owned services.

A runtime process shares one Socket.IO link across sessions, and python-socketio keeps
only the last handler per event and namespace. Each response event therefore owns one
shared listener per underlying link and routes frames by ``channel_id``, the session's
Host routing key. Host-to-runtime events keep the ``task_files:session_response:``
prefix, which the runtime Base already relays to worker venvs.

Wire contract:

- ``acp_permission_request`` → Host: ``{channel_id, acp_request_id, permission_token, params}``.
- Host → ``task_files:session_response:acp_permission``: ``{channel_id, permission_token,
  option_id, ext_requests?}``; the Host echoes the token unchanged. Only a token issued by the
  live session and not yet used is honored, and only while its ACP request still waits. A
  missing or empty ``option_id`` cancels the ACP request. When the relay is built with
  ``client_questions=True``, ``ext_requests`` (``[{method, params}]``, only
  ``_codebuddy.ai/resolveInterruption``) are sent to the agent in order before the permission
  is answered; a malformed list or any other method cancels the permission.
- ``acp_elicitation_request`` → Host: ``{channel_id, elicitation_token, params}``.
- Host → ``task_files:session_response:acp_elicitation``: ``{channel_id, elicitation_token,
  result}`` where ``result`` is ``{action: accept|decline|cancel, content?}``.
- ``host_tool_request`` → Host: ``{request_id, channel_id, tool_name, arguments}``.
- Host → ``task_files:session_response:host_tool``: ``{request_id, channel_id, success,
  result}`` or ``{..., success: false, error: {code, message}}``.
- ``acp_session_update`` → Host: ``{channel_id, turn_id, seq, update}``, one per
  ``trim_acp_update``-trimmed ``session/update`` received during a prompt. ``seq`` starts at 1
  per turn; the turn result reports the last one as ``relay_frames``. A turn that names its ACP
  session (a WorkBuddy session, which may share its process with forks) relays only that
  session's updates, including the ``_codebuddy.ai/command`` updates ``AcpConnection`` re-emits.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import re
import uuid

from collections.abc import Callable
from typing import Any, Protocol, cast
from weakref import WeakKeyDictionary


logger = logging.getLogger(__name__)

ACP_PERMISSION_REQUEST_EVENT = 'acp_permission_request'
ACP_PERMISSION_RESOLVE_EVENT = 'task_files:session_response:acp_permission'
HOST_TOOL_REQUEST_EVENT = 'host_tool_request'
HOST_TOOL_RESPONSE_EVENT = 'task_files:session_response:host_tool'
ACP_SESSION_UPDATE_EVENT = 'acp_session_update'
ACP_ELICITATION_REQUEST_EVENT = 'acp_elicitation_request'
ACP_ELICITATION_RESOLVE_EVENT = 'task_files:session_response:acp_elicitation'

RESOLVE_INTERRUPTION_METHOD = '_codebuddy.ai/resolveInterruption'

_UPDATE_CLOSE_TIMEOUT = 2.0
_EXT_REQUEST_TIMEOUT = 40.0
# NOTE(wallyshi): The Host never reads these; ACP version 2 sends usage and config changes many times a turn.
_DROPPED_UPDATES = frozenset({'available_commands_update', 'usage_update', 'config_option_update'})
# NOTE(wallyshi): The Host reads at most 400 length units of a pushed tool string, a unit being a CJK
# character or a whitespace-separated word of at most 32 characters, so at most 12800 non-whitespace
# characters even after JSON encoding. A cut string keeps twice that many non-whitespace characters and at
# most twice as many characters overall, which bounds whitespace-only text. Assistant text is never cut:
# the Host keeps a turn's closing narration whole.
_PUSHED_VISIBLE_CHARS = 25600
# NOTE(wallyshi): The Host names a file by the last segment of these, which a cut would change, and
# registers or opens the file or page a WorkBuddy command names.
_WHOLE_TEXT_KEYS = frozenset({'path', 'file_path', 'filePath', 'focusFile', 'targetFile', 'url'})
_NON_WHITESPACE = re.compile(r'\S')

_FrameHandler = Callable[[dict[str, Any]], None]
_ROUTES: WeakKeyDictionary[Any, dict[tuple[str | None, str], dict[str, _FrameHandler]]] = WeakKeyDictionary()


class DeferredPermissionTarget(Protocol):
    """Resolve or cancel one deferred ACP permission request."""

    def resolve_deferred_permission(self, request_id: int, *, option_id: str) -> bool: ...

    def cancel_deferred_permission(self, request_id: int) -> bool: ...

    def has_deferred_permission(self, request_id: int) -> bool: ...

    async def extension_request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any: ...


class HostToolError(RuntimeError):
    """Report a failed Host tool call with a machine-readable code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _bind_channel(sio: Any, event: str, channel_id: str, handler: _FrameHandler) -> None:
    if not channel_id:
        raise ValueError('channel_id must not be empty')
    link = getattr(sio, '_sio', sio)
    by_event = _ROUTES.setdefault(link, {})
    key = (getattr(sio, '_ns', None), event)
    routes = by_event.get(key)
    if routes is None:
        routes = {}
        by_event[key] = routes

        async def dispatch(data: Any) -> None:
            if not isinstance(data, dict):
                return
            route = routes.get(data.get('channel_id'))
            if route is not None:
                route(data)

        sio.on(event)(dispatch)
    if channel_id in routes:
        raise ValueError(f'Channel {channel_id!r} already owns a {event!r} route on this runtime link')
    routes[channel_id] = handler


def _unbind_channel(sio: Any, event: str, channel_id: str, handler: _FrameHandler) -> None:
    routes = _ROUTES.get(getattr(sio, '_sio', sio), {}).get((getattr(sio, '_ns', None), event))
    if routes is not None and routes.get(channel_id) == handler:
        routes.pop(channel_id, None)


class AcpPermissionRelay:
    """Forward one session's deferred ACP permissions to the Host and apply its decisions.

    Args:
        sio: Runtime Socket.IO transport.
        channel_id: Host routing key of the owning session.
        connection: ACP connection that owns the deferred requests.
        client_questions: Honour the Host's ``ext_requests``; off ignores them.
    """

    def __init__(
        self, *, sio: Any, channel_id: str, connection: DeferredPermissionTarget, client_questions: bool = False
    ) -> None:
        self.sio = sio
        self.channel_id = channel_id
        self.connection = connection
        self.client_questions = client_questions
        self._closed = False
        # NOTE(wallyshi): ACP request ids restart per CLI process and option ids are fixed, so a session recreated
        # under the same channel would accept a stale approval unless each request carries its own token.
        self._pending_tokens: dict[str, int] = {}
        self._ext_tasks: set[asyncio.Task[None]] = set()
        _bind_channel(sio, ACP_PERMISSION_RESOLVE_EVENT, channel_id, self._apply_resolution)

    async def request(self, params: dict[str, Any], request_id: int) -> None:
        """Publish a permission request; the Host answers asynchronously through the resolve event."""
        if self._closed:
            raise RuntimeError(f'ACP permission relay for channel {self.channel_id!r} is closed')
        token = uuid.uuid4().hex
        self._pending_tokens[token] = request_id
        logger.info('ACP permission request channel=%s request=%s', self.channel_id, request_id)
        try:
            await self.sio.emit(
                ACP_PERMISSION_REQUEST_EVENT,
                {
                    'channel_id': self.channel_id,
                    'acp_request_id': request_id,
                    'permission_token': token,
                    'params': params,
                },
            )
        except BaseException:
            self._pending_tokens.pop(token, None)
            raise

    def _apply_resolution(self, data: dict[str, Any]) -> None:
        token = data.get('permission_token')
        request_id = self._pending_tokens.pop(token, None) if isinstance(token, str) else None
        if request_id is None:
            logger.warning(
                'Ignoring ACP permission resolution channel=%s: token was not issued by this session or was used',
                self.channel_id,
            )
            return
        if not self._is_waiting(request_id):
            return
        option_id = data.get('option_id')
        option_id = option_id if isinstance(option_id, str) else ''
        raw_ext_requests = data.get('ext_requests')
        if raw_ext_requests is None or not self.client_questions:
            self._answer(request_id, option_id)
            return
        ext_requests = _parse_ext_requests(raw_ext_requests)
        if ext_requests is None:
            logger.warning(
                'Cancelling ACP permission channel=%s request=%s: malformed or disallowed ext_requests',
                self.channel_id,
                request_id,
            )
            self.connection.cancel_deferred_permission(request_id)
            return
        if not ext_requests:
            self._answer(request_id, option_id)
            return
        task = asyncio.get_running_loop().create_task(self._answer_after(request_id, option_id, ext_requests))
        self._ext_tasks.add(task)
        task.add_done_callback(self._ext_tasks.discard)

    async def _answer_after(
        self, request_id: int, option_id: str, ext_requests: list[tuple[str, dict[str, Any]]]
    ) -> None:
        # Params, results and peer error messages carry the user's answer, so none of them is logged.
        for method, params in ext_requests:
            if not self._is_waiting(request_id):
                return
            try:
                result = await self.connection.extension_request(method, params, timeout=_EXT_REQUEST_TIMEOUT)
            except Exception as error:  # noqa: BLE001
                failure = type(error).__name__
            else:
                if isinstance(result, dict) and cast(dict[str, Any], result).get('resolved'):
                    continue
                failure = 'unresolved'
            # CodeBuddy returns {} on a resolveInterruption it accepted, so a failure must not override the answer.
            logger.warning(
                'ACP extension request failed (%s) channel=%s request=%s method=%s; answering the permission as chosen',
                failure,
                self.channel_id,
                request_id,
                method,
            )
        if not self._closed:
            self._answer(request_id, option_id)

    def _is_waiting(self, request_id: int) -> bool:
        if self.connection.has_deferred_permission(request_id):
            return True
        logger.info(
            'Dropping ACP permission resolution channel=%s request=%s: the request was already answered or withdrawn',
            self.channel_id,
            request_id,
        )
        return False

    def _answer(self, request_id: int, option_id: str) -> None:
        if option_id:
            self.connection.resolve_deferred_permission(request_id, option_id=option_id)
        else:
            self.connection.cancel_deferred_permission(request_id)

    def end_turn(self) -> None:
        """Forget the tokens of requests the agent no longer waits on, once a prompt has ended."""
        # NOTE(wallyshi): JSON-RPC keeps ids unique only among requests in flight, so a token outliving its
        # request could later match a new request that reuses the id.
        self._pending_tokens = {
            token: request_id
            for token, request_id in self._pending_tokens.items()
            if self.connection.has_deferred_permission(request_id)
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_tokens.clear()
        for task in self._ext_tasks:
            task.cancel()
        self._ext_tasks.clear()
        _unbind_channel(self.sio, ACP_PERMISSION_RESOLVE_EVENT, self.channel_id, self._apply_resolution)


def _parse_ext_requests(value: object) -> list[tuple[str, dict[str, Any]]] | None:
    """Validate a Host ``ext_requests`` list; None when any entry is malformed or not allowed."""
    if not isinstance(value, list):
        return None
    requests: list[tuple[str, dict[str, Any]]] = []
    for item in cast(list[Any], value):
        if not isinstance(item, dict):
            return None
        entry = cast(dict[str, Any], item)
        method = entry.get('method')
        params = entry.get('params')
        if method != RESOLVE_INTERRUPTION_METHOD or not isinstance(params, dict):
            return None
        requests.append((method, cast(dict[str, Any], params)))
    return requests


class AcpElicitationRelay:
    """Forward one session's ACP ``elicitation/create`` requests to the Host and await its answers.

    ``request`` is an ``AcpConnection`` elicitation handler; the connection bounds how long it
    waits with ``deferred_permission_timeout`` and answers ``cancel`` when it gives up or the
    prompt ends. The Host is not told when that happens.

    Args:
        sio: Runtime Socket.IO transport.
        channel_id: Host routing key of the owning session.
    """

    def __init__(self, *, sio: Any, channel_id: str) -> None:
        self.sio = sio
        self.channel_id = channel_id
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._closed = False
        _bind_channel(sio, ACP_ELICITATION_RESOLVE_EVENT, channel_id, self._apply_resolution)

    async def request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Publish one elicitation and wait for the Host's ``{action, content?}`` result."""
        if self._closed:
            raise RuntimeError(f'ACP elicitation relay for channel {self.channel_id!r} is closed')
        token = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[token] = future
        logger.info('ACP elicitation request channel=%s mode=%s', self.channel_id, params.get('mode'))
        try:
            await self.sio.emit(
                ACP_ELICITATION_REQUEST_EVENT,
                {'channel_id': self.channel_id, 'elicitation_token': token, 'params': params},
            )
            return await future
        finally:
            self._pending.pop(token, None)

    def _apply_resolution(self, data: dict[str, Any]) -> None:
        token = data.get('elicitation_token')
        future = self._pending.get(token) if isinstance(token, str) else None
        if future is None or future.done():
            logger.warning(
                'Ignoring ACP elicitation resolution channel=%s: token was not issued by this session or was used',
                self.channel_id,
            )
            return
        result = data.get('result')
        future.set_result(result if isinstance(result, dict) else {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _unbind_channel(self.sio, ACP_ELICITATION_RESOLVE_EVENT, self.channel_id, self._apply_resolution)
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()


class HostToolClient:
    """Call Host-owned tools on behalf of one session.

    Args:
        sio: Runtime Socket.IO transport.
        channel_id: Host routing key of the owning session.
    """

    def __init__(self, *, sio: Any, channel_id: str) -> None:
        self.sio = sio
        self.channel_id = channel_id
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._closed = False
        _bind_channel(sio, HOST_TOOL_RESPONSE_EVENT, channel_id, self._apply_response)

    async def call(self, *, tool_name: str, arguments: dict[str, Any], timeout: float) -> Any:
        """Send one tool call to the Host and wait for its result."""
        if self._closed:
            raise HostToolError('Host tool client is closed', code='HOST_TOOL_CLIENT_CLOSED')
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self.sio.emit(
                HOST_TOOL_REQUEST_EVENT,
                {
                    'request_id': request_id,
                    'channel_id': self.channel_id,
                    'tool_name': tool_name,
                    'arguments': arguments,
                },
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as error:
            logger.warning(
                'Host tool timed out channel=%s request=%s tool=%s timeout=%s',
                self.channel_id,
                request_id,
                tool_name,
                timeout,
            )
            raise HostToolError(
                f'Host tool {tool_name!r} timed out after {timeout:g}s', code='HOST_TOOL_TIMEOUT'
            ) from error
        finally:
            self._pending.pop(request_id, None)

    def _apply_response(self, data: dict[str, Any]) -> None:
        request_id = data.get('request_id')
        future = self._pending.get(request_id) if isinstance(request_id, str) else None
        if future is None or future.done():
            return
        if data.get('success') is True:
            future.set_result(data.get('result'))
            return
        error = data.get('error')
        error = error if isinstance(error, dict) else {}
        future.set_exception(
            HostToolError(
                str(error.get('message') or 'Host tool call failed'), code=str(error.get('code') or 'HOST_TOOL_ERROR')
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _unbind_channel(self.sio, HOST_TOOL_RESPONSE_EVENT, self.channel_id, self._apply_response)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(HostToolError('Host tool client closed', code='HOST_TOOL_CLIENT_CLOSED'))
        self._pending.clear()


def trim_acp_update(update: dict[str, Any]) -> dict[str, Any] | None:
    """Drop the parts of one ACP update the Host never reads.

    Tool calls keep every field: the Host records their output and the ``_meta`` tool name, so a
    pushed tool call must carry what the same update in the invoke result's events carries. Tool
    output and file contents can still make one frame large enough to break the link, so every
    long string is cut to the prefix the Host can read; keys, lists and file paths stay whole.
    Assistant text is pushed whole, since the Host hands a turn's closing narration on uncut.

    Args:
        update (dict[str, Any]): The ``update`` field of a ``session/update`` notification.

    Returns:
        A trimmed copy, or None when the whole update is irrelevant to the Host.
    """
    kind = update.get('sessionUpdate')
    if kind in _DROPPED_UPDATES:
        return None
    if kind in ('tool_call', 'tool_call_update'):
        return _cut_strings(update)
    if kind == 'agent_thought_chunk':
        # Only the fact that the model reasoned is projected, never what it reasoned.
        return {'sessionUpdate': kind}
    stripped = {key: value for key, value in update.items() if key != '_meta'}
    return stripped if kind == 'agent_message_chunk' else _cut_strings(stripped)


def _cut_strings(value: Any) -> Any:
    if isinstance(value, str):
        return _cut_text(value)
    if isinstance(value, dict):
        return {
            key: item if key in _WHOLE_TEXT_KEYS and isinstance(item, str) else _cut_strings(item)
            for key, item in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [_cut_strings(item) for item in cast(list[Any], value)]
    return value


def _cut_text(text: str) -> str:
    if len(text) <= _PUSHED_VISIBLE_CHARS:
        return text
    head = text[: 2 * _PUSHED_VISIBLE_CHARS]
    last_visible = next(itertools.islice(_NON_WHITESPACE.finditer(head), _PUSHED_VISIBLE_CHARS - 1, None), None)
    return head if last_visible is None else text[: last_visible.end()]


class AcpUpdateRelay:
    """Push one session's ACP ``session/update`` notifications to the Host as they arrive.

    ``publish`` is an ``AcpConnection`` update listener: it runs inline on the ACP read loop,
    so it only enqueues, and a single sender task emits in order. Only updates that arrive
    between ``begin_turn`` and ``end_turn`` are sent, stamped with that turn's id; a turn begun
    with a ``session_id`` sends only that ACP session's updates.

    Args:
        sio: Runtime Socket.IO transport.
        channel_id: Host routing key of the owning session.
        close_timeout: How long ``flush`` and ``aclose`` wait for queued updates to be sent.
    """

    def __init__(self, *, sio: Any, channel_id: str, close_timeout: float = _UPDATE_CLOSE_TIMEOUT) -> None:
        if not channel_id:
            raise ValueError('channel_id must not be empty')
        self.sio = sio
        self.channel_id = channel_id
        self.close_timeout = close_timeout
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None
        self._turn_id: str | None = None
        self._turn_session_id: str | None = None
        self._turn_frames = 0
        self._closed = False

    def begin_turn(self, turn_id: str, *, session_id: str | None = None) -> None:
        """Start stamping updates with ``turn_id``, numbering them from 1.

        Args:
            turn_id (str): Id stamped on every update of this turn.
            session_id (str | None): ACP session whose updates belong to the turn; None keeps all,
                as for a CodeBuddy process that runs one session.
        """
        if not turn_id:
            raise ValueError('turn_id must not be empty')
        self._turn_id = turn_id
        self._turn_session_id = session_id
        self._turn_frames = 0

    def end_turn(self) -> int:
        """Stop sending updates and return how many this turn published, lost emits included."""
        frames, self._turn_id, self._turn_session_id, self._turn_frames = self._turn_frames, None, None, 0
        return frames

    async def flush(self) -> None:
        """Wait up to ``close_timeout`` for queued frames to be emitted."""
        if self._sender is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self.close_timeout)
        except asyncio.TimeoutError:
            logger.warning('Gave up on %s queued ACP session updates channel=%s', self._queue.qsize(), self.channel_id)

    def publish(self, params: dict[str, Any]) -> None:
        """Queue one ``session/update`` params dict for the Host; ignored outside a turn or once closed."""
        if self._closed or self._turn_id is None:
            return
        if self._turn_session_id is not None and params.get('sessionId') != self._turn_session_id:
            return
        update = params.get('update')
        trimmed = trim_acp_update(update) if isinstance(update, dict) else None
        if trimmed is None:
            return
        self._turn_frames += 1
        self._queue.put_nowait({
            'channel_id': self.channel_id,
            'turn_id': self._turn_id,
            'seq': self._turn_frames,
            'update': trimmed,
        })
        if self._sender is None:
            self._sender = asyncio.get_running_loop().create_task(self._send_loop())

    async def _send_loop(self) -> None:
        while True:
            payload = await self._queue.get()
            try:
                await self.sio.emit(ACP_SESSION_UPDATE_EVENT, payload)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    'Dropped ACP session update channel=%s seq=%s: %s', self.channel_id, payload['seq'], error
                )
            finally:
                self._queue.task_done()

    async def aclose(self) -> None:
        """Stop accepting updates, send what is queued within ``close_timeout``, then stop."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.flush()
        finally:
            sender, self._sender = self._sender, None
            if sender is not None:
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender


__all__ = [
    'ACP_ELICITATION_REQUEST_EVENT',
    'ACP_ELICITATION_RESOLVE_EVENT',
    'ACP_PERMISSION_REQUEST_EVENT',
    'ACP_PERMISSION_RESOLVE_EVENT',
    'ACP_SESSION_UPDATE_EVENT',
    'HOST_TOOL_REQUEST_EVENT',
    'HOST_TOOL_RESPONSE_EVENT',
    'RESOLVE_INTERRUPTION_METHOD',
    'AcpElicitationRelay',
    'AcpPermissionRelay',
    'AcpUpdateRelay',
    'DeferredPermissionTarget',
    'HostToolClient',
    'HostToolError',
    'trim_acp_update',
]
