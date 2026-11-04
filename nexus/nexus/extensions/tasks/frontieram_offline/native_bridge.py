"""Protocol v5 transport: Codex Responses to an upstream Responses endpoint, without protocol conversion.

Protocol v4 converted every Codex Responses request into Chat Completions. Models whose native interface is Responses
then went through the gateway's reverse conversion, which corrupted some tool calls (for example the tool name
``mcp__workspace<parameter=command``). This bridge forwards the Codex request itself and keeps the v4 boundary:

* exactly one candidate-facing tool, the offline workspace ``exec``; every other declared tool is removed, and the
  conversation history may contain no other tool call;
* one immutable development deadline and a fixed HTTP budget shared by retries;
* the credential stays in this host-side process, never in Codex or its tools;
* provider-default reasoning and a fixed per-response output cap, as in v4.

v5 changes only what happens to an unusable response. A response truncated at the output cap, or one calling a tool
that does not exist, is not executed; it is requested again with the identical payload (at most three retries per
request, counted against the HTTP budget), so one runaway sample no longer ends the whole attempt.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import math
import random
import socket
import time

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from nexus.extensions.openai_clients.cache_task_id import normalize_cache_task_id

from .client import BoundaryError


WORKSPACE_NAMESPACE = 'mcp__workspace'
WORKSPACE_TOOL = 'exec'
FLAT_NAME = f'{WORKSPACE_NAMESPACE}__{WORKSPACE_TOOL}'
MAX_ARGUMENT_BYTES = 256 * 1024
RETRIES_PER_REQUEST = 3
UNUSABLE = ('incomplete_response', 'rejected_tool', 'failed_response')


def _sse(kind: str, sequence: int, **fields: Any) -> bytes:
    payload = {'type': kind, 'sequence_number': sequence, **fields}
    return f'event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'.encode()


def responses_events(payload: dict[str, Any]) -> bytes:
    """Deliver a completed non-streaming Responses object through the event stream Codex consumes."""
    sequence = 0
    chunks: list[bytes] = []

    def emit(kind: str, **fields: Any) -> None:
        nonlocal sequence
        chunks.append(_sse(kind, sequence, **fields))
        sequence += 1

    initial = copy.deepcopy(payload)
    initial.update(status='in_progress', output=[], usage=None)
    emit('response.created', response=initial)
    emit('response.in_progress', response=initial)
    for index, item in enumerate(payload.get('output') or []):
        fields = {'output_index': index, 'item_id': item.get('id', '')}
        added = copy.deepcopy(item)
        if 'status' in added:
            added['status'] = 'in_progress'
        kind = item.get('type')
        if kind == 'message':
            added['content'] = []
        elif kind == 'function_call':
            added['arguments'] = ''
        elif kind == 'reasoning':
            added['summary'] = []
            added.pop('encrypted_content', None)
            if 'content' in added:
                added['content'] = []
        emit('response.output_item.added', output_index=index, item=added)
        if kind == 'message':
            for content_index, part in enumerate(item.get('content') or []):
                part_fields = {**fields, 'content_index': content_index}
                start = copy.deepcopy(part)
                if part.get('type') == 'output_text':
                    start.update(text='', annotations=[])
                emit('response.content_part.added', **part_fields, part=start)
                if part.get('type') == 'output_text':
                    emit('response.output_text.delta', **part_fields, delta=part.get('text', ''))
                    emit('response.output_text.done', **part_fields, text=part.get('text', ''))
                emit('response.content_part.done', **part_fields, part=part)
        elif kind == 'function_call':
            emit('response.function_call_arguments.delta', **fields, delta=item.get('arguments', ''))
            emit('response.function_call_arguments.done', **fields, arguments=item.get('arguments', ''))
        elif kind == 'reasoning':
            for summary_index, part in enumerate(item.get('summary') or []):
                part_fields = {**fields, 'summary_index': summary_index}
                emit('response.reasoning_summary_part.added', **part_fields, part={**part, 'text': ''})
                emit('response.reasoning_summary_text.delta', **part_fields, delta=part.get('text', ''))
                emit('response.reasoning_summary_text.done', **part_fields, text=part.get('text', ''))
                emit('response.reasoning_summary_part.done', **part_fields, part=part)
        emit('response.output_item.done', output_index=index, item=item)
    emit('response.completed', response=payload)
    return b''.join(chunks)


def _is_workspace_function(tool: dict[str, Any], namespace: str | None) -> bool:
    if tool.get('type') != 'function':
        return False
    if namespace == WORKSPACE_NAMESPACE:
        return tool.get('name') == WORKSPACE_TOOL
    return namespace is None and tool.get('name') == FLAT_NAME


def outbound_request(body: dict[str, Any], *, max_output_tokens: int, record) -> dict[str, Any]:
    """The Codex request with only the workspace tool, flattened, and the fixed v4 inference settings."""
    if not isinstance(body, dict) or not isinstance(body.get('input', []), list):
        raise BoundaryError('request_structure')
    body = copy.deepcopy(body)
    kept = []
    for tool in body.get('tools') or []:
        if not isinstance(tool, dict):
            raise BoundaryError('request_structure')
        nested = tool.get('tools') if tool.get('type') == 'namespace' else [tool]
        namespace = tool.get('name') if tool.get('type') == 'namespace' else None
        for inner in nested if isinstance(nested, list) else []:
            if isinstance(inner, dict) and _is_workspace_function(inner, namespace):
                kept.append({**{k: v for k, v in inner.items() if k != 'name'}, 'name': FLAT_NAME})
            else:
                record('tool_schema_filtered', namespace=namespace, name=str((inner or {}).get('name'))[:64])
    if len(kept) > 1:
        raise BoundaryError('workspace_tool_count')
    if body.get('tools') and not kept:
        raise BoundaryError('workspace_tool_count')
    if kept:
        body['tools'] = kept
    else:
        body.pop('tools', None)
    choice = body.get('tool_choice')
    if isinstance(choice, dict) and choice.get('name') not in (FLAT_NAME, WORKSPACE_TOOL):
        raise BoundaryError('selected_tool_identity')
    history = []
    for item in body['input']:
        if not isinstance(item, dict):
            history.append(item)
            continue
        if item.get('type') in ('custom_tool_call', 'tool_search_call'):
            raise BoundaryError('history_tool_identity')
        if item.get('type') == 'reasoning':
            # Stateless requests (store=false) cannot reference earlier reasoning items; v4 carried none either.
            continue
        if item.get('type') == 'function_call':
            namespace = item.pop('namespace', None)
            if not _is_workspace_function({'type': 'function', 'name': item.get('name')}, namespace):
                raise BoundaryError('history_tool_identity')
            item['name'] = FLAT_NAME
        history.append(item)
    body['input'] = history
    # Standard Responses fields only, provider-default reasoning and one output cap for every model, as in v4.
    allowed = ('model', 'instructions', 'input', 'tools', 'tool_choice', 'parallel_tool_calls', 'prompt_cache_key')
    body = {key: body[key] for key in allowed if key in body}
    body.update(stream=False, max_output_tokens=max_output_tokens, store=False)
    return body


def usable_response(payload: dict[str, Any], record) -> str | None:
    """None when Codex may execute the response; otherwise the reason it must be sampled again."""
    status = payload.get('status')
    if status == 'incomplete':
        record('incomplete_response', reason=str((payload.get('incomplete_details') or {}).get('reason'))[:64],
               usage=payload.get('usage'))
        return 'incomplete_response'
    if status != 'completed':
        record('failed_response', status=str(status)[:32], error=str(payload.get('error'))[:200])
        return 'failed_response'
    for item in payload.get('output') or []:
        if not isinstance(item, dict):
            return 'failed_response'
        if item.get('type') in ('custom_tool_call', 'tool_search_call', 'mcp_call', 'web_search_call'):
            record('rejected_tool', name=str(item.get('type'))[:64])
            return 'rejected_tool'
        if item.get('type') != 'function_call':
            continue
        if item.get('name') != FLAT_NAME or item.get('namespace') not in (None, ''):
            record('rejected_tool', name=str(item.get('name'))[:64])
            return 'rejected_tool'
        arguments = item.get('arguments')
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else None
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict) or len(arguments.encode()) > MAX_ARGUMENT_BYTES:
            record('rejected_tool', name=FLAT_NAME, reason='arguments')
            return 'rejected_tool'
    return None


def inbound_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore the namespace Codex dispatches MCP calls by."""
    payload = copy.deepcopy(payload)
    for item in payload.get('output') or []:
        if item.get('type') == 'function_call':
            item['name'] = WORKSPACE_TOOL
            item['namespace'] = WORKSPACE_NAMESPACE
    return payload


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class NativeResponsesBridge:
    """Both the loopback server Codex talks to and the bounded upstream client (agent.py uses it for both roles)."""

    def __init__(
        self,
        *,
        upstream_url: str,
        api_key: str,
        token: str,
        deadline: float,
        retry_policy: dict[str, Any],
        max_http_attempts: int = 800,
        max_output_tokens: int = 64000,
        cache_task_id: str | None = None,
        provider_timeout_max_seconds: int | None = None,
        upstream_headers: dict[str, str] | None = None,
        audit_path: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not token or len(token) < 16:
            raise ValueError('A per-run loopback token of at least 16 characters is required')
        if not math.isfinite(deadline) or not 1 <= max_http_attempts <= 800 or not 1 <= max_output_tokens <= 128000:
            raise ValueError('Invalid immutable inference budget')
        policy = retry_policy.get('data', {})
        if retry_policy.get('code') != 0 or not isinstance(policy.get('dispositions'), list):
            raise ValueError('A successful frozen retry policy is required')
        self.url = upstream_url.rstrip('/') + '/responses'
        self._key = normalize_cache_task_id(api_key, cache_task_id) if cache_task_id else api_key
        if any(k.lower() in ('authorization', 'content-type', 'host') for k in upstream_headers or {}):
            raise ValueError('Fixed upstream headers cannot replace transport or credential headers')
        self._headers = dict(upstream_headers or {})
        self._token = token
        self.deadline = deadline
        self.max_http_attempts = max_http_attempts
        self.max_output_tokens = max_output_tokens
        self.provider_timeout_max_seconds = provider_timeout_max_seconds
        self.http_attempts = self.retry_count = self.logical_requests = 0
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.audit_path = audit_path
        if audit_path is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.touch(exist_ok=False)
        self._clock, self._sleep, self._jitter = clock, sleep, jitter
        self._retry_codes = {
            item['code'] for item in policy['dispositions']
            if isinstance(item, dict) and item.get('is_retryable') is True and isinstance(item.get('code'), int)
        }
        self._retry_statuses = {429, 500, 502, 503, 504} & set(policy.get('backoff', {}).get('retryable_http_status', []))
        self._http = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False)
        self._stopped = False
        self._lock = asyncio.Lock()
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self.port = 0

    # Shared client/proxy interface ---------------------------------------------------------------------------------

    def record(self, kind: str, **fields: Any) -> None:
        event = {'kind': kind, 'at_monotonic': self._clock(), **fields}
        self.events.append(event)
        if kind == 'error':
            self.errors.append(event)
        if self.audit_path is not None:
            with self.audit_path.open('a') as output:
                output.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')

    def check_deadline(self) -> float:
        remaining = self.deadline - self._clock()
        if self._stopped or remaining <= 0:
            raise BoundaryError('deadline' if remaining <= 0 else 'stopped')
        return remaining

    def stop(self) -> None:
        self._stopped = True

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}/v1'

    # Upstream --------------------------------------------------------------------------------------------------------

    async def _send(self, body: bytes) -> httpx.Response:
        if self.http_attempts >= self.max_http_attempts:
            raise BoundaryError('http_budget')
        remaining = self.check_deadline()
        self.http_attempts += 1
        self.record('http_attempt', attempt=self.http_attempts, payload_sha256=hashlib.sha256(body).hexdigest(),
                    remaining_seconds=remaining)
        url = self.url
        try:
            async with asyncio.timeout(remaining):
                response = await self._http.post(
                    url,
                    content=body,
                    headers={**self._headers, 'Authorization': f'Bearer {self._key}', 'Content-Type': 'application/json'},
                    timeout=httpx.Timeout(remaining + 30.0, connect=30.0),
                )
        except TimeoutError:
            raise BoundaryError('deadline') from None
        except httpx.HTTPError as error:
            self.record('transport_error', exception_type=type(error).__name__)
            raise BoundaryError('upstream_transport') from None
        self.record('http_response', attempt=self.http_attempts, status=response.status_code)
        return response

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """One logical request: identical-payload retries for retryable HTTP errors and unusable responses."""
        async with self._lock:
            self.logical_requests += 1
            if self.provider_timeout_max_seconds is not None:
                # The gateway's per-request provider timeout, bounded by the remaining development time (as in v4).
                request = {**request, 'timeout': max(1, min(int(self.check_deadline()), self.provider_timeout_max_seconds))}
            body = json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode()
            for retry in range(RETRIES_PER_REQUEST + 1):
                response = await self._send(body)
                reason, delay = None, None
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {'status': 'failed', 'error': 'non-JSON body'}
                    reason = usable_response(payload, self.record)
                    if reason is None:
                        self.check_deadline()
                        return payload
                    delay = 0.0
                else:
                    try:
                        code = (response.json().get('error') or {}).get('code')
                        code = int(code) if code is not None else None
                    except (ValueError, TypeError, AttributeError):
                        code = None
                    if response.status_code not in self._retry_statuses or code not in self._retry_codes:
                        self.record('error', category='upstream_http', status=response.status_code, gateway_code=code)
                        raise BoundaryError('upstream_http')
                    reason = f'http_{response.status_code}'
                    delay = _retry_after(response.headers.get('retry-after'))
                    if delay is None:
                        delay = self._jitter(0.0, float(2**retry))
                if retry == RETRIES_PER_REQUEST:
                    category = reason if reason in UNUSABLE else 'upstream_http'
                    self.record('error', category=category, retries=retry)
                    raise BoundaryError(category)
                if delay >= self.check_deadline() or self.http_attempts >= self.max_http_attempts:
                    self.record('error', category='retry_budget')
                    raise BoundaryError('retry_budget')
                self.retry_count += 1
                self.record('retry', retry=self.retry_count, reason=reason, delay=delay)
                if delay:
                    await self._sleep(delay)
        raise AssertionError('unreachable')

    # Loopback server -------------------------------------------------------------------------------------------------

    def _app(self) -> FastAPI:
        app = FastAPI()

        @app.post('/v1/responses')
        async def responses(request: Request):
            if not hmac.compare_digest(request.headers.get('authorization', ''), f'Bearer {self._token}'):
                return JSONResponse(status_code=401, content={'error': {'code': 'local_authentication_failed'}})
            try:
                self.check_deadline()
                outbound = outbound_request(
                    json.loads(await request.body()), max_output_tokens=self.max_output_tokens, record=self.record
                )
                payload = await self.complete(outbound)
            except BoundaryError as error:
                category = error.error_code.removeprefix('frontieram_')
                if not any(e.get('category') == category for e in self.errors[-1:]):
                    self.record('error', category=category)
                status = 400 if category in ('deadline', 'stopped', 'http_budget') else 502
                return JSONResponse(status_code=status, content={'error': {'code': error.error_code}})
            except ValueError:
                self.record('error', category='request_structure')
                return JSONResponse(status_code=400, content={'error': {'code': 'frontieram_request_structure'}})
            return Response(responses_events(inbound_response(payload)), media_type='text/event-stream')

        @app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
        async def other(path: str):
            # Includes /v1/responses/compact: remote compaction stays disabled; Codex compacts locally.
            return JSONResponse(status_code=404, content={'error': {'code': 'frontieram_route_disabled'}})

        return app

    async def start(self) -> None:
        self.check_deadline()
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('127.0.0.1', 0))
        self._socket.listen(128)
        self._socket.setblocking(False)
        self.port = self._socket.getsockname()[1]
        config = uvicorn.Config(self._app(), log_level='error', access_log=False, timeout_graceful_shutdown=0)
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        for _ in range(200):
            if self._server.started:
                return
            if self._server_task.done():
                await self._server_task
                raise BoundaryError('proxy_startup')
            await asyncio.sleep(0.01)
        await self.stop_server()
        raise BoundaryError('proxy_startup')

    async def stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=5)
            except (asyncio.TimeoutError, Exception):
                self._server_task.cancel()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
