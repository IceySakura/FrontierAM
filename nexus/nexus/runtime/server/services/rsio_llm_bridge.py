"""Shared reliable-SIO (RSIO) LLM bridge for runtime-side proxies.

Rides entirely on the existing ``llm_request`` / ``llm_response`` Socket.IO
events. Protocol metadata lives in the ``rsio`` sub-field of the payload;
business fields are untouched.

Two modes, chosen per bridge instance:

- **RSIO mode** (default): send ``rsio`` metadata, resend on ack/report
  timeout, treat ``running`` reports as keep-alives.
- **Legacy mode** — chosen when either
    - the host sets the ``NEXUS_DISABLE_LLM_RSIO`` env var on the runtime
      (escape hatch for an *old host* that can't digest ``rsio`` fields), or
    - ``RSIOLLMConfig.enable`` is False.
  Emit the request once and wait for the final response, exactly like the
  pre-RSIO bridge. No ``rsio`` field is sent (so it can't leak into an old
  host's LLM params) and there is no resend (so an old host without a dedup
  guard is never double-charged).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary, WeakSet


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge


# Runtime-side escape hatch for old-host compatibility. Set on the runtime
# provider's environment (e.g. via docker ``environment`` / gongfeng env_vars)
# to force the RSIO bridge back to legacy emit+wait behaviour — used when the
# host is an old version that can't handle rsio fields on the wire.
_DISABLE_ENV_VAR = 'NEXUS_DISABLE_LLM_RSIO'

# Runtime-to-host metadata. Runtime-side proxies attach these fields so the host can wrap the
# real summary inference in a live compaction span. They must never reach an upstream provider.
NEXUS_AGENT_COMPACTION_FIELD = '_nexus_agent_compaction'
NEXUS_COMPACTION_DETECTOR_FIELD = '_nexus_compaction_detector'


def _env_disables_rsio() -> bool:
    return os.getenv(_DISABLE_ENV_VAR, '').strip().lower() in {'1', 'true', 'yes', 'on'}


logger = logging.getLogger(__name__)


class LLMRequestError(Exception):
    """A host-side model call that came back as a failed ``llm_response``.

    Carries the host's machine-readable ``error_code`` (absent on an old host) so a proxy can
    name the cause to its in-sandbox agent instead of collapsing every failure into an opaque
    HTTP 500.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


def bridge_error_payload(exc: BaseException) -> tuple[int, dict[str, Any]] | None:
    """Render a coded bridge failure as an OpenAI-shaped client error; None when uncoded.

    The status code carries this, not the body. Measured against codex 0.136 and 0.142
    (``codex exec --json``) with a proxy returning each shape in turn:

    * 500, any body -- codex retries up to 30 times and reports a canned "high demand" message,
      so the upstream text never reaches its transcript and one refusal costs 30 host calls.
    * 400 -- returned verbatim in ``turn.failed.error.message`` after a single call.

    400 is also the faithful status: a prompt over the budget is a client error, and retrying it
    unchanged cannot succeed. The code is repeated as a message prefix because the agent records
    the body as opaque text, so only what is inside the message survives to be read back.
    """
    error_code = getattr(exc, 'error_code', None)
    if not error_code:
        return None
    return 400, {
        'error': {
            'message': f'[{error_code}] {exc}',
            'type': 'invalid_request_error',
            'code': error_code,
        }
    }


@dataclass(frozen=True)
class RSIOLLMConfig:
    enable: bool = True  # False → bridge falls back to legacy emit+wait (no rsio, no resend)
    ack_timeout: float = 5.0  # 首次探活：发出 request 后多久无回包就重发
    report_interval: float = 15.0  # Host running 心跳间隔（供 handler 使用）
    # 收过 running 后多久无新回包就重发。也是断连/丢包后的最坏恢复延迟上限。
    # 取 3×report_interval：容忍正常长调用期间偶发丢 1~2 条心跳而不误重发。
    report_timeout: float = 45.0


@dataclass
class _PendingCall:
    request_id: str
    params: dict[str, Any]
    future: asyncio.Future
    deadline_at: float | None
    attempt: int = 0
    last_report_at: float = 0.0
    saw_running: bool = False


# Per-sio-client shared dispatch state. Multiple bridge instances in the same
# worker process (one per concurrent agent's AnthropicProxy) ride the SAME
# underlying Socket.IO client. python-socketio keeps only the LAST handler
# registered for a given event/namespace, so if every bridge did
# ``@self.sio.on('llm_response')`` in its own __init__, only the last bridge's
# handler would survive — the other bridges' request_ids would arrive, hit a
# handler whose _pending doesn't know them, and be dropped, causing an endless
# RESEND loop. Fix: register ONE handler per underlying client and share a
# single request_id → _PendingCall table across all bridges on that client.
#
# Weak keys keep distinct clients (distinct sids) isolated without retaining a
# stopped client forever or confusing a later client that reuses its Python id.
_SHARED_PENDING: WeakKeyDictionary[Any, dict[str, _PendingCall]] = WeakKeyDictionary()
_SHARED_HANDLER_REGISTERED: WeakSet[Any] = WeakSet()


def _underlying_client(sio: SIOBridge) -> Any:
    """Return the real Socket.IO client behind a bridge wrapper.

    Falls back to the bridge itself if the private attribute is unavailable.
    """
    return getattr(sio, '_sio', sio)


class RSIOLLMBridge:
    """Initiate reliable LLM calls with optional outbound channel routing.

    Args:
        sio: Runtime Socket.IO transport shared with other bridges.
        config: Reliability settings; defaults to RSIO mode.
        channel_id: Host routing metadata, supported only in RSIO mode. Responses
            still match by request ID so hosts need not echo the channel.
    """

    def __init__(self, sio: SIOBridge, config: RSIOLLMConfig | None = None, *, channel_id: str | None = None) -> None:
        self.sio = sio
        self.cfg = config or RSIOLLMConfig()
        self.channel_id = channel_id
        self._closed = False
        self._owned_requests: dict[str, asyncio.Task[Any]] = {}
        # Legacy fallback: host set NEXUS_DISABLE_LLM_RSIO on the runtime, or
        # this instance was constructed with enable=False. Either way, degrade
        # to pre-RSIO emit+wait behaviour so an old host isn't confused by
        # rsio fields on the wire or double-charged by resends.
        self._legacy = not self.cfg.enable or _env_disables_rsio()
        if channel_id is not None and self._legacy:
            raise ValueError('channel_id requires RSIO mode; enable RSIO and unset NEXUS_DISABLE_LLM_RSIO')
        # All bridges sharing one underlying sio client share one pending table,
        # so responses for any request_id reach the bridge that owns it.
        self._client = _underlying_client(sio)
        self._pending: dict[str, _PendingCall] = _SHARED_PENDING.setdefault(self._client, {})
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        # Register the shared dispatcher exactly once per underlying client.
        # Later bridges reuse the same _pending table (see __init__) so their
        # request_ids are dispatched by the single existing handler.
        if self._client in _SHARED_HANDLER_REGISTERED:
            return
        _SHARED_HANDLER_REGISTERED.add(self._client)
        pending_table = self._pending

        @self.sio.on('llm_response')
        async def on_llm_response(data: dict[str, Any]) -> None:
            request_id = data.get('request_id')
            if not request_id:
                logger.warning('rsio.worker: llm_response without request_id, ignoring')
                return
            pending = pending_table.get(request_id)
            if pending is None:
                # Unknown / already-completed request. Common and harmless: a
                # duplicate final for a call we already resolved and popped.
                logger.debug('rsio.worker: llm_response for unknown request_id=%s, ignoring', request_id)
                return
            if pending.future.done():
                logger.debug('rsio.worker: llm_response for already-resolved request_id=%s, ignoring', request_id)
                return
            rsio = data.get('rsio') or {}
            kind = rsio.get('kind', 'final')  # 未知/缺省视为 final（兼容老 Host）
            if kind == 'running':
                pending.last_report_at = asyncio.get_running_loop().time()
                pending.saw_running = True
                logger.debug('rsio.worker: running heartbeat request_id=%s (attempt=%s)', request_id, pending.attempt)
                return
            if data.get('success'):
                logger.debug('rsio.worker: final OK request_id=%s attempts=%s', request_id, pending.attempt)
                pending.future.set_result(data.get('response', {}))
            else:
                error = data.get('error', 'Unknown error')
                error_code = data.get('error_code') or None
                logger.warning(
                    'rsio.worker: final ERROR request_id=%s attempts=%s code=%s error=%s',
                    request_id,
                    pending.attempt,
                    error_code,
                    error,
                )
                pending.future.set_exception(LLMRequestError(error, error_code=error_code))

    def close(self) -> None:
        """Cancel this bridge's calls without disrupting sibling bridges on the same socket."""
        if self._closed:
            return
        self._closed = True
        for request_id, task in self._owned_requests.items():
            pending = self._pending.pop(request_id, None)
            if pending is not None:
                pending.future.cancel()
            task.cancel()
        self._owned_requests.clear()

    async def request(self, openai_params: dict[str, Any], timeout: float | None = 300) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError('LLM bridge is closed; create a new bridge before making requests')
        loop = asyncio.get_running_loop()
        request_id = str(uuid.uuid4())
        pending = _PendingCall(
            request_id=request_id,
            params=openai_params,
            future=loop.create_future(),
            deadline_at=loop.time() + timeout if timeout is not None else None,
        )
        self._pending[request_id] = pending
        mode = 'legacy' if self._legacy else 'rsio'
        timeout_description = f'{timeout:.0f}s' if timeout is not None else 'disabled'
        logger.debug('rsio.worker: call start request_id=%s timeout=%s mode=%s', request_id, timeout_description, mode)
        try:
            # NOTE(wallyshi): Track the caller to cancel blocked emits without adding a task hop to legacy shutdown.
            task = asyncio.current_task()
            assert task is not None, 'LLM requests must run inside an asyncio task'
            self._owned_requests[request_id] = task
            if self._legacy:
                return await self._drive_legacy(pending, timeout)
            return await self._drive(pending)
        finally:
            self._pending.pop(request_id, None)
            self._owned_requests.pop(request_id, None)
            pending.future.cancel()

    async def _drive_legacy(self, pending: _PendingCall, timeout: float | None) -> dict[str, Any]:
        """Old-host path: emit once (no rsio), wait for the final. No resend.

        An old host does not strip an ``rsio`` field (it would leak into the LLM
        params) and has no dedup guard (a resend would double-charge), so we
        must behave exactly like the pre-RSIO bridge.
        """
        envelope = {**pending.params, 'request_id': pending.request_id}
        logger.debug('rsio.worker: send llm_request request_id=%s (legacy)', pending.request_id)
        await self.sio.emit('llm_request', envelope)
        try:
            if timeout is None:
                return await pending.future
            return await asyncio.wait_for(pending.future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error('rsio.worker: legacy request timed out request_id=%s', pending.request_id)
            raise TimeoutError(f'LLM request timed out (request_id={pending.request_id})') from None

    async def _drive(self, pending: _PendingCall) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        await self._send(pending)
        while not pending.future.done():
            now = loop.time()
            if pending.deadline_at is not None and now >= pending.deadline_at:
                logger.error(
                    'rsio.worker: deadline exceeded request_id=%s attempts=%s saw_running=%s',
                    pending.request_id,
                    pending.attempt,
                    pending.saw_running,
                )
                raise TimeoutError(f'LLM request timed out (request_id={pending.request_id})')
            gap = self.cfg.report_timeout if pending.saw_running else self.cfg.ack_timeout
            wait = gap - (now - pending.last_report_at)
            if pending.deadline_at is not None:
                wait = min(wait, pending.deadline_at - now)
            if wait <= 0:
                # Silent past `gap` — resend the same request_id (host dedups).
                logger.warning(
                    'rsio.worker: no report for %.1fs (gap=%.0fs) request_id=%s, resending',
                    now - pending.last_report_at,
                    gap,
                    pending.request_id,
                )
                await self._send(pending)
                continue
            # Wait up to `wait` for the final response. If it times out, fall
            # back to the loop top, which recomputes `wait`; if the peer is
            # truly silent past `gap` the `wait <= 0` branch resends. A running
            # heartbeat that arrived mid-wait pushed last_report_at forward, so
            # wait stays > 0 and we keep waiting instead of resending.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(pending.future), timeout=wait)
        return pending.future.result()

    async def _send(self, pending: _PendingCall) -> None:
        loop = asyncio.get_running_loop()
        pending.attempt += 1
        pending.last_report_at = loop.time()
        envelope: dict[str, Any] = {
            **pending.params,
            'request_id': pending.request_id,
            'rsio': {
                'attempt': pending.attempt,
                'deadline_at': pending.deadline_at,
            },
        }
        if self.channel_id is not None:
            envelope['rsio']['channel_id'] = self.channel_id
        log = logger.debug if pending.attempt == 1 else logger.warning
        log(
            'rsio.worker: %s llm_request request_id=%s attempt=%s',
            'send' if pending.attempt == 1 else 'RESEND',
            pending.request_id,
            pending.attempt,
        )
        try:
            await self.sio.emit('llm_request', envelope)
        except Exception as exc:
            # Disconnected / transient transport failure. Do NOT fail the call;
            # the drive loop resends on the next probe tick (lazy reconnect).
            logger.warning(
                'rsio.worker: llm_request emit failed request_id=%s attempt=%s (%s); will retry',
                pending.request_id,
                pending.attempt,
                exc,
            )
