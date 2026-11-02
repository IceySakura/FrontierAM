"""ToolProxyDispatcher — sits between the LLM loop and the tool_proxy HTTP RPC.

Responsibilities:

* Look each tool_call up in :mod:`yb_tool_registry`, validate params via
  the tool's Pydantic schema, and construct a JSON ``/batch_tool_call`` body on
  the ``context_in`` path.
* Route through :meth:`ToolProxyClient.batch_tool_call_with_retries`
  (rate-limit + retries + total deadline).
* Parse JSON responses into OpenAI-format tool messages and accumulate
  ``context_out.session`` fields in :class:`_SessionState`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nexus.extensions.projects.yb_search.tool_proxy_client import (
    ToolProxyClient,
    ToolProxyConfig,
)


if TYPE_CHECKING:
    from nexus.extensions.projects.yb_search.tools.context import (
        ToolProxyContextIn,
    )


logger = logging.getLogger(__name__)
wire_logger = logging.getLogger('nexus.yb.tool_proxy.wire')


# Wire-level dumping is opt-in: either set NEXUS_YB_TOOL_WIRE_LOG=1
# (read in ToolProxyDispatcher.__init__) or call set_wire_log_enabled(True)
# directly from tests. The flag is decoupled from wire_logger.level so
# pytest's caplog.at_level() can't spuriously activate dumping.
_wire_log_enabled: bool = False


def set_wire_log_enabled(enabled: bool) -> None:
    """Toggle wire-level payload dumping globally."""
    global _wire_log_enabled
    _wire_log_enabled = bool(enabled)


def is_wire_log_enabled() -> bool:
    return _wire_log_enabled


@dataclass
class ToolCallStatus:
    """Structured status for a single tool call."""

    call_id: str
    name: str = ''
    status_code: int | None = None
    status_msg: str = ''
    category: str = 'success'
    content_empty: bool = False


@dataclass
class DispatchStats:
    """Per-batch stats returned alongside tool messages."""

    elapsed_ms: float = 0.0
    rate_limit_wait_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    batch_status_code: int = 0
    call_statuses: list[ToolCallStatus] = field(default_factory=list)


@dataclass
class _SessionState:
    """Mutable session fields round-tripped via ``context_out.session``."""

    cursors: dict[str, int] = field(default_factory=dict)
    visited_urls: list[str] = field(default_factory=list)
    sandbox_state: Any | None = None
    image_refs: dict[str, Any] = field(default_factory=dict)
    biz_state: dict[str, Any] = field(default_factory=dict)
    extra_out_history: list[Any] = field(default_factory=list)


def _dump_wire(label: str, message: Any, traj_id: str) -> None:
    """Dump a JSON request/response body to the wire logger."""
    if not _wire_log_enabled:
        return
    try:
        payload = json.dumps(message, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover - defensive
        wire_logger.warning(
            '%s serialize failed (%s: %s); falling back to str',
            label,
            type(exc).__name__,
            exc,
        )
        payload = f'{type(message).__name__}: {str(message)[:1000]}'

    wire_logger.debug(
        '===== %s begin traj=%s =====\n%s\n===== %s end traj=%s =====',
        label,
        traj_id,
        payload,
        label,
        traj_id,
    )


class ToolProxyDispatcher:
    """High-level dispatcher above the HTTP JSON transport."""

    def __init__(
        self,
        config: ToolProxyConfig,
        context_in: ToolProxyContextIn,
    ) -> None:
        self._transport = ToolProxyClient(config)
        self._context_in_cfg = context_in
        self._session_state = _SessionState()

        if os.environ.get('NEXUS_YB_TOOL_WIRE_LOG', '') in ('1', 'true', 'True'):
            set_wire_log_enabled(True)
            wire_logger.setLevel(logging.DEBUG)
            if not wire_logger.handlers:
                _handler = logging.StreamHandler()
                _handler.setLevel(logging.DEBUG)
                _handler.setFormatter(logging.Formatter('%(asctime)s | WIRE | %(name)s | %(message)s'))
                wire_logger.addHandler(_handler)
                wire_logger.propagate = False

    @property
    def is_configured(self) -> bool:
        return self._transport.is_configured

    @property
    def session_state(self) -> _SessionState:
        return self._session_state

    # ---------- request construction ----------

    @staticmethod
    def _local_tool_error(
        *,
        tc: dict[str, Any],
        idx: int,
        name: str,
        message: str,
        category: str = 'format_error',
    ) -> tuple[dict[str, Any], ToolCallStatus]:
        call_id = tc.get('id') or f'call_{idx}'
        display_category = category.replace('_', ' ')
        return (
            {
                'role': 'tool',
                'tool_call_id': call_id,
                'content': f'Tool call {display_category}: {message}',
            },
            ToolCallStatus(
                call_id=call_id,
                name=name,
                status_msg=message,
                category=category,
                content_empty=False,
            ),
        )

    @staticmethod
    def _history_to_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Flatten OpenAI messages (incl. content-parts list) to JSON messages."""
        out: list[dict[str, str]] = []
        for msg in history:
            role = str(msg.get('role', ''))
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = [
                    part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text'
                ]
                content = '\n'.join(text_parts)
            elif not isinstance(content, str):
                content = str(content) if content else ''
            out.append({'role': role, 'content': content})
        return out

    def _build_req(
        self,
        *,
        raw_tool_calls: list[dict[str, Any]],
        history: list[dict[str, Any]],
        raw_query: str,
        round_idx: int,
        enable_multimodal: bool,
        tool_versions: dict[str, str] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[ToolCallStatus]]:
        """Build a JSON ``/batch_tool_call`` body on the ``context_in`` path."""
        from nexus.extensions.agents.base.utils import parse_func_arguments  # noqa: PLC0415
        from nexus.extensions.projects.yb_search.tools.registry import (  # noqa: PLC0415
            yb_tool_registry,
        )

        built_calls: list[dict[str, Any]] = []
        dispatchable_raw_tool_calls: list[dict[str, Any]] = []
        local_tool_messages: list[dict[str, Any]] = []
        local_call_statuses: list[ToolCallStatus] = []
        extras: list[dict[str, Any]] = []
        for idx, tc in enumerate(raw_tool_calls):
            func = tc.get('function', {}) or {}
            name = str(func.get('name') or '')
            version = str(tc.get('version') or func.get('version') or (tool_versions or {}).get(name) or '')
            tool = yb_tool_registry.get(name, version) if name and version else None
            if tool is None:
                message, status = self._local_tool_error(
                    tc=tc,
                    idx=idx,
                    name=name,
                    message=f'tool {name!r}@{version!r} is not registered',
                )
                local_tool_messages.append(message)
                local_call_statuses.append(status)
                continue

            try:
                raw_args = parse_func_arguments(func.get('arguments', '{}'))
                params = tool.Params(**raw_args)
            except Exception as exc:
                message, status = self._local_tool_error(
                    tc=tc,
                    idx=idx,
                    name=name,
                    message=f'param validation failed for {name}@{version}: {exc}',
                )
                local_tool_messages.append(message)
                local_call_statuses.append(status)
                continue

            call_id = tc.get('id') or f'call_{uuid.uuid4().hex[:12]}'
            built_calls.append({
                'call_id': call_id,
                'name': name,
                'version': version,
                # 用 exclude_unset 保留 JSON "字段存在性" 语义：
                # task_schedule/update_task 会判断用户是否显式传入字段。
                'arguments': params.model_dump(exclude_unset=True),
            })
            parse_tc = dict(tc)
            parse_tc['id'] = call_id
            dispatchable_raw_tool_calls.append(parse_tc)

            extra = tool.build_extra(params, self._context_in_cfg, self._session_state)
            if extra is not None:
                if not isinstance(extra, dict):
                    raise TypeError(f'{name}@{version} build_extra must return dict | None')
                extras.append(extra)

        req: dict[str, Any] = {
            'calls': built_calls,
            'options': {'is_vlm': enable_multimodal},
        }

        context_in: dict[str, Any] = {}
        user_cfg = self._context_in_cfg.user
        user = {
            'user_id': user_cfg.user_id,
            'client_version': user_cfg.client_version,
            'wx_openid': user_cfg.wx_openid,
        }
        if user_cfg.location:
            user['location'] = dict(user_cfg.location)
        context_in['user'] = user

        caller_cfg = self._context_in_cfg.caller
        context_in['caller'] = {
            'call_source': caller_cfg.call_source,
            'model_name': caller_cfg.model_name,
            'exp': caller_cfg.exp,
            'is_illustrated': caller_cfg.is_illustrated,
        }

        session_cfg = self._context_in_cfg.session
        session: dict[str, Any] = {
            'round': round_idx + 1,
            'raw_query': raw_query,
            'conversation_id': session_cfg.conversation_id,
            'task_id': session_cfg.task_id,
            'history': self._history_to_messages(history),
        }
        if self._session_state.cursors:
            session['cursors'] = dict(self._session_state.cursors)
        if self._session_state.visited_urls:
            session['visited_urls'] = list(self._session_state.visited_urls)
        if self._session_state.sandbox_state is not None:
            session['sandbox_state'] = self._session_state.sandbox_state
        context_in['session'] = session

        if extras:
            context_in['extra'] = extras
        req['context_in'] = context_in
        return req, dispatchable_raw_tool_calls, local_tool_messages, local_call_statuses

    # ---------- response parsing ----------

    @staticmethod
    def _classify_result_status(status_code: int | None, status_msg: str, *, content_empty: bool) -> str:
        """Classify tool_proxy per-call status from ``result.status``."""
        msg = status_msg.lower()
        if status_code is None or status_code == 0:
            return 'empty_result' if content_empty else 'success'
        if status_code == 20601:
            return 'partial_success'
        if (
            'timeout' in msg
            or 'deadline' in msg
            or 'timed out' in msg
            or status_code in {408, 504}
            or 40800 <= status_code < 40900
            or 50400 <= status_code < 50500
        ):
            return 'timeout'
        if 'rate limit' in msg or 'rate_limited' in msg or status_code == 429 or 42900 <= status_code < 43000:
            return 'rate_limited'
        if status_code == 40401 or 40400 <= status_code < 40500 or '工具不存在' in msg or 'not found' in msg:
            return 'tool_not_found'
        if status_code == 40001 or 400 <= status_code < 500 or 40000 <= status_code < 50000:
            return 'format_error'
        if status_code >= 500 or status_code >= 50000:
            return 'error'
        return 'error'

    def _parse_rsp(
        self,
        rsp: dict[str, Any],
        raw_tool_calls: list[dict[str, Any]],
        *,
        enable_multimodal: bool,
    ) -> tuple[list[dict[str, Any]], list[ToolCallStatus]]:
        """Convert a JSON BatchToolCall response to OpenAI-format tool messages."""
        from nexus.extensions.projects.yb_search.tool_proxy_client import (  # noqa: PLC0415
            _build_content_from_for_model,
        )

        results = rsp.get('results') or []
        result_map: dict[str, dict[str, Any]] = {
            str(result.get('call_id') or ''): result for result in results if isinstance(result, dict)
        }
        batch_status = rsp.get('status') or {}
        batch_status_code = batch_status.get('code')
        batch_status_msg = str(batch_status.get('msg') or '')

        messages: list[dict[str, Any]] = []
        statuses: list[ToolCallStatus] = []
        for idx, tc in enumerate(raw_tool_calls):
            func = tc.get('function', {}) or {}
            name = str(func.get('name') or '')
            call_id = tc.get('id', '') or f'call_{idx}'
            result = result_map.get(call_id)
            if result is None and idx < len(results) and isinstance(results[idx], dict):
                result = results[idx]

            status_code: int | None = None
            status_msg = ''
            if result is None:
                status_code = batch_status_code if batch_status_code not in (None, 0, 20601) else None
                status_msg = batch_status_msg
                content_parts = [
                    {
                        'type': 'text',
                        'text': f'Error: no result returned for tool call {call_id}',
                    }
                ]
                category = self._classify_result_status(status_code, status_msg, content_empty=True)
                if category == 'empty_result':
                    category = 'error'
                content_empty = True
            else:
                result_status = result.get('status') or {}
                if result_status:
                    status_code = result_status.get('code')
                    status_msg = str(result_status.get('msg') or '')
                content_parts = _build_content_from_for_model(
                    list(result.get('for_model') or []),
                    enable_multimodal=enable_multimodal,
                )
                content_empty = not content_parts
                category = self._classify_result_status(status_code, status_msg, content_empty=content_empty)
                if not content_parts:
                    if status_code not in (None, 0):
                        content_parts = [
                            {
                                'type': 'text',
                                'text': f'Tool error (code={status_code}): {status_msg}',
                            }
                        ]
                    elif status_msg:
                        content_parts = [{'type': 'text', 'text': status_msg}]
                    else:
                        content_parts = [{'type': 'text', 'text': 'Tool returned no content.'}]

            text_parts = [
                part.get('text', '') for part in content_parts if isinstance(part, dict) and part.get('type') == 'text'
            ]
            messages.append({
                'role': 'tool',
                'tool_call_id': call_id,
                'content': '\n'.join(text_parts),
            })
            statuses.append(
                ToolCallStatus(
                    call_id=call_id,
                    name=name,
                    status_code=status_code,
                    status_msg=status_msg,
                    category=category,
                    content_empty=content_empty,
                )
            )
        return messages, statuses

    def _apply_context_out(self, rsp: dict[str, Any]) -> None:
        """Merge ``context_out.session`` into ``_session_state`` (overwrite, not union)."""
        context_out = rsp.get('context_out') or {}
        session = context_out.get('session') or {}
        if session.get('cursors'):
            self._session_state.cursors = {str(k): int(v) for k, v in session['cursors'].items()}
        if session.get('visited_urls'):
            self._session_state.visited_urls = [str(url) for url in session['visited_urls']]
        if 'sandbox_state' in session:
            self._session_state.sandbox_state = session['sandbox_state']
        if session.get('image_refs'):
            self._session_state.image_refs = dict(session['image_refs'])
        if session.get('biz_state'):
            self._session_state.biz_state = dict(session['biz_state'])
        for extra_out in context_out.get('extra') or []:
            self._session_state.extra_out_history.append(extra_out)

    async def execute_tool_calls(
        self,
        raw_tool_calls: list[dict[str, Any]],
        *,
        history: list[dict[str, Any]],
        raw_query: str,
        round_idx: int,
        enable_multimodal: bool,
        agent_id: str,
        tool_versions: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], DispatchStats]:
        """Dispatch a batch of OpenAI-format tool_calls via the context_in path."""
        call_descriptors = [
            {
                'name': (tc.get('function', {}) or {}).get('name', '?'),
                'version': (tc.get('function', {}) or {}).get('version', '')
                or (tool_versions or {}).get((tc.get('function', {}) or {}).get('name', ''), ''),
                'call_id': tc.get('id', ''),
            }
            for tc in raw_tool_calls
        ]
        logger.debug(
            'TOOL_DISPATCH_START traj=%s batch_size=%d calls=%s',
            agent_id,
            len(raw_tool_calls),
            call_descriptors,
        )

        stats = DispatchStats()
        start = time.monotonic()
        req, dispatchable_raw_tool_calls, local_tool_messages, local_call_statuses = self._build_req(
            raw_tool_calls=raw_tool_calls,
            history=history,
            raw_query=raw_query,
            round_idx=round_idx,
            enable_multimodal=enable_multimodal,
            tool_versions=tool_versions,
        )

        tool_messages = list(local_tool_messages)
        stats.call_statuses = list(local_call_statuses)
        if req.get('calls'):
            _dump_wire('TOOL_DISPATCH_REQ', req, agent_id)
            try:
                rsp = await self._transport.batch_tool_call_with_retries(req, traj_id=agent_id)
                stats.rate_limit_wait_ms = self._transport.last_rate_limit_wait_s * 1000
            except Exception as exc:
                stats.rate_limit_wait_ms = self._transport.last_rate_limit_wait_s * 1000
                logger.exception('dispatcher transport failed: %s', exc)
                is_timeout = 'timeout' in str(exc).lower() or 'timeout' in type(exc).__name__.lower()
                category = 'timeout' if is_timeout else 'transport_error'
                tool_messages.extend(
                    {
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'content': f'Tool proxy call failed: {type(exc).__name__}: {exc}',
                    }
                    for tc in dispatchable_raw_tool_calls
                )
                stats.call_statuses.extend(
                    ToolCallStatus(
                        call_id=tc.get('id', ''),
                        name=(tc.get('function', {}) or {}).get('name', ''),
                        status_msg=f'{type(exc).__name__}: {exc}',
                        category=category,
                        content_empty=False,
                    )
                    for tc in dispatchable_raw_tool_calls
                )
            else:
                _dump_wire('TOOL_DISPATCH_RSP', rsp, agent_id)
                status = rsp.get('status') or {}
                if status.get('code', 0) != 0:
                    stats.batch_status_code = int(status.get('code') or 0)
                    logger.warning(
                        'dispatcher batch status non-zero: code=%s msg=%s',
                        status.get('code'),
                        status.get('msg', ''),
                    )
                parsed_tool_messages, parsed_call_statuses = self._parse_rsp(
                    rsp,
                    dispatchable_raw_tool_calls,
                    enable_multimodal=enable_multimodal,
                )
                tool_messages.extend(parsed_tool_messages)
                stats.call_statuses.extend(parsed_call_statuses)
                self._apply_context_out(rsp)

        stats.elapsed_ms = (time.monotonic() - start) * 1000
        stats.success_count = sum(1 for status in stats.call_statuses if status.category == 'success')
        stats.error_count = len(stats.call_statuses) - stats.success_count

        logger.info(
            'TOOL_DISPATCH_DONE traj=%s elapsed_ms=%.1f rate_limit_wait_ms=%.1f '
            'success=%d error=%d cursors=%s visited=%d',
            agent_id,
            stats.elapsed_ms,
            stats.rate_limit_wait_ms,
            stats.success_count,
            stats.error_count,
            dict(self._session_state.cursors),
            len(self._session_state.visited_urls),
        )
        return tool_messages, stats
