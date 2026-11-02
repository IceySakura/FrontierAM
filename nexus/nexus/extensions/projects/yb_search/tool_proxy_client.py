"""HTTP JSON transport for the tool-proxy service.

The production agent calls ``POST /batch_tool_call`` with a JSON body that
matches the tool_proxy protojson field names (snake_case), but no longer
depends on the tRPC runtime or generated Python stubs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request

from dataclasses import dataclass, field
from functools import cache
from typing import Any

from nexus.utils.rate_limiter import get_rate_limiter


logger = logging.getLogger(__name__)

DEFAULT_TOOL_PROXY_TIMEOUT = int(os.environ.get('NEXUS_SEARCH_TIMEOUT', '100'))
MAX_TOOL_PROXY_RETRIES = int(os.environ.get('NEXUS_SEARCH_MAX_RETRIES', '1'))
TOOL_PROXY_RETRY_SLEEP = int(os.environ.get('NEXUS_SEARCH_RETRY_SLEEP', '10'))
# Total timeout for the entire retry loop (incl. rate-limit wait);
# aligned with NEXUS_PROMPT_SEARCH_TOTAL_TIMEOUT in agts.
TOOL_PROXY_TOTAL_TIMEOUT = int(os.environ.get('NEXUS_SEARCH_TOTAL_TIMEOUT', '1000'))
# Per-attempt timeout derived from total / retries (used as HTTP call timeout).
PER_ATTEMPT_TIMEOUT = TOOL_PROXY_TOTAL_TIMEOUT // max(MAX_TOOL_PROXY_RETRIES, 1)
# Per-worker concurrency cap (aligned with NEXUS_SEARCH_RATE_LIMIT in agts).
TOOL_PROXY_RATE_LIMIT = int(os.environ.get('NEXUS_SEARCH_RATE_LIMIT', '1'))

DEFAULT_TOOL_PROXY_HTTP_SERVICE = 'trpc.yb_agent.tool_proxy.http'
LEGACY_TOOL_PROXY_TRPC_SERVICE = 'trpc.yb_agent.tool_proxy.trpc'
DEFAULT_TOOL_PROXY_HTTP_PATH = '/batch_tool_call'
AUTH_HEADER = 'wemeet-interface-auth-token'
TRPC_TRANS_INFO_HEADER = 'Trpc-Trans-Info'


# ====== Configuration ======


@dataclass
class ToolProxyConfig:
    """Configuration for the tool proxy HTTP client.

    Endpoint resolution priority:
      1. ``TOOL_PROXY_HTTP_URL`` environment variable.
      2. ``TOOL_PROXY_HTTP_HOST`` + ``TOOL_PROXY_HTTP_PORT`` environment variables.
      3. ``service`` + ``namespace`` via Polaris service discovery.
    """

    service: str | None = None
    namespace: str = 'Production'
    metadata: dict[str, str] = field(default_factory=dict)
    env: str | None = None
    src_service: str | None = None
    timeout: int = DEFAULT_TOOL_PROXY_TIMEOUT
    caller: str = ''
    auth: dict[str, Any] = field(default_factory=dict)


# ====== Polaris Discovery ======


@cache
def _get_polaris_consumer_api():
    """Get or create a Polaris Consumer API instance (cached)."""
    from polaris.api.consumer import create_consumer_by_config  # noqa: PLC0415

    return create_consumer_by_config('')


def _discover_service(
    namespace: str,
    service: str,
    metadata: dict[str, str] | None = None,
    src_service: str | None = None,
) -> tuple[str, int]:
    """Discover host:port via Polaris service discovery."""
    from polaris.pkg.model.service import GetOneInstanceRequest  # noqa: PLC0415

    consumer_api = _get_polaris_consumer_api()
    request = GetOneInstanceRequest(namespace=namespace, service=service)

    if src_service:
        parts = src_service.split(':', 1)
        if len(parts) == 2:
            request.set_src_service_key(parts[0], parts[1])
        else:
            request.set_src_service_key(namespace, src_service)

    if metadata:
        for key, value in metadata.items():
            request.add_service_metadata(key, value)

    response = consumer_api.get_one_instance_resp(request)
    instance = response[0]
    return instance.get_host(), instance.get_port()


def _resolve_http_url(transport: dict[str, Any], *, dry: bool = False) -> str:
    """Resolve the ``POST /batch_tool_call`` URL."""
    direct = os.getenv('TOOL_PROXY_HTTP_URL', '').strip()
    if direct:
        return direct.rstrip('/') + DEFAULT_TOOL_PROXY_HTTP_PATH

    host = os.getenv('TOOL_PROXY_HTTP_HOST', '').strip()
    port = os.getenv('TOOL_PROXY_HTTP_PORT', '').strip()
    if host and port:
        return f'http://{host}:{port}{DEFAULT_TOOL_PROXY_HTTP_PATH}'

    service = _normalize_http_service(str(transport.get('service') or DEFAULT_TOOL_PROXY_HTTP_SERVICE))
    namespace = str(transport.get('namespace') or os.getenv('TOOL_PROXY_NAMESPACE', 'Production'))
    metadata = dict(transport.get('metadata') or {})
    env = transport.get('env') or os.getenv('env')  # noqa: SIM112
    if env:
        metadata.setdefault('env', str(env))

    if dry:
        env_suffix = f'?env={metadata["env"]}' if metadata.get('env') else ''
        return f'http://<polaris:{namespace}/{service}{env_suffix}>{DEFAULT_TOOL_PROXY_HTTP_PATH}'

    host, port = _discover_service(
        namespace=namespace,
        service=service,
        metadata=metadata or None,
        src_service=transport.get('src_service'),
    )
    logger.debug('Discovered tool proxy HTTP endpoint via Polaris at %s:%s', host, port)
    return f'http://{host}:{port}{DEFAULT_TOOL_PROXY_HTTP_PATH}'


def _normalize_http_service(service: str) -> str:
    """Map legacy tRPC service names embedded in old data to the HTTP service."""
    if service == LEGACY_TOOL_PROXY_TRPC_SERVICE:
        return DEFAULT_TOOL_PROXY_HTTP_SERVICE
    return service


# ====== Conversion Helpers ======


def _build_content_from_for_model(
    for_model_list: list[Any],
    *,
    enable_multimodal: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible ``content`` from Tool Proxy ``for_model`` chunks."""
    content_parts: list[dict[str, Any]] = []

    for chunk in for_model_list:
        if not isinstance(chunk, dict):
            continue
        chunk_type = chunk.get('type') or 'text'
        text = chunk.get('content') or ''

        if chunk_type == 'image' and enable_multimodal:
            image_url = chunk.get('url') or ''
            if not image_url:
                continue
            content_parts.append({
                'type': 'image_url',
                'image_url': {'url': image_url},
            })
        elif chunk_type == 'text':
            content_parts.append({'type': 'text', 'text': text})

    return content_parts


# ====== Auth / HTTP helpers ======


def _auth_token(auth: dict[str, Any] | None = None, *, now: int | None = None) -> str | None:
    """Build the interface_auth token for trpc-go HTTP metadata."""
    auth = dict(auth or {})
    ak = str(auth.get('ak') or os.getenv('TOOL_PROXY_AK') or 'kit').strip()
    sk = str(auth.get('sk') or os.getenv('TOOL_PROXY_SK') or '').strip()
    version = str(auth.get('version') or '20231031').strip()
    if not sk:
        return None

    timestamp = int(time.time()) if now is None else int(now)
    plain = f'&time={timestamp}&method={DEFAULT_TOOL_PROXY_HTTP_PATH}'.encode()
    if version == '20240521':
        sign = hmac.new(sk.encode(), plain, hashlib.sha256).digest()
    else:
        sign = plain + hmac.new(sk.encode(), b'', hashlib.sha256).digest()

    return json.dumps(
        {
            'a': ak,
            't': timestamp,
            's': base64.b64encode(sign).decode(),
            'v': version,
        },
        separators=(',', ':'),
    )


def _attach_auth_headers(headers: dict[str, str], token: str) -> None:
    """Attach interface_auth token through trpc-go HTTP transport metadata."""
    trans_info = {
        AUTH_HEADER: base64.b64encode(token.encode()).decode(),
    }
    headers[TRPC_TRANS_INFO_HEADER] = json.dumps(trans_info, separators=(',', ':'))


def _http_post(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode()
        except Exception:
            raw = ''
        raise RuntimeError(f'HTTP {exc.code} from tool_proxy: {raw[:500]}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'tool_proxy HTTP request failed: {exc}') from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'tool_proxy response is not JSON: {raw[:500]}') from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f'tool_proxy response must be a JSON object, got {type(parsed).__name__}')
    return parsed


# ====== Logging helpers ======


def _format_json_for_log(message: Any) -> str:
    """Best-effort JSON formatter for logs."""
    try:
        return json.dumps(message, ensure_ascii=False, indent=2)
    except Exception:
        try:
            return str(message)
        except Exception:
            return '<unprintable JSON>'


def _summarize_req_for_log(req: dict[str, Any]) -> str:
    """Generate a compact summary of a request body for INFO logging."""
    try:
        parts: list[str] = []
        context_in = req.get('context_in') or {}
        session = context_in.get('session') or {}
        if session:
            parts.append(f'round={session.get("round", "")}')
            parts.append(f'history={len(session.get("history") or [])} msgs')
            if session.get('cursors'):
                parts.append(f'cursors={session["cursors"]}')
            if session.get('visited_urls'):
                parts.append(f'visited_urls={len(session["visited_urls"])}')

        call_summaries: list[str] = []
        for call in req.get('calls') or []:
            arguments = call.get('arguments') or {}
            call_summaries.append(
                f'{call.get("name", "")}(call_id={call.get("call_id", "")}, '
                f'v={call.get("version", "")}, args={list(arguments.keys())})'
            )
        parts.append(f'tools=[{", ".join(call_summaries)}]')
        return ', '.join(parts)
    except Exception:
        return '<failed to summarize request>'


def _summarize_rsp_for_log(rsp: dict[str, Any]) -> str:
    """Generate a compact summary of a response body for INFO logging."""
    try:
        parts: list[str] = []
        status = rsp.get('status') or {}
        if status:
            parts.append(f'status(code={status.get("code", 0)}, msg={status.get("msg", "")})')
        else:
            parts.append('status=none')

        result_summaries: list[str] = []
        for result in rsp.get('results') or []:
            for_model = result.get('for_model') or []
            for_model_chars = sum(len(chunk.get('content') or '') for chunk in for_model if isinstance(chunk, dict))
            result_status = result.get('status') or {}
            status_desc = f'code={result_status.get("code", 0)}' if result_status else 'no-status'
            result_summaries.append(
                f'{result.get("name", "")}(call_id={result.get("call_id", "")}, {status_desc}, '
                f'for_model={len(for_model)} chunks/{for_model_chars} chars, '
                f'contents={len(result.get("contents") or [])})'
            )
        parts.append(f'results=[{", ".join(result_summaries)}]')
        return ', '.join(parts)
    except Exception:
        return '<failed to summarize response>'


# ====== Client Proxy ======


class ToolProxyClient:
    """HTTP JSON transport for the tool-proxy service."""

    def __init__(self, config: ToolProxyConfig) -> None:
        self._config = config
        self.last_rate_limit_wait_s = 0.0

    @property
    def is_configured(self) -> bool:
        """Whether the client has enough config to make calls."""
        return bool(
            self._config.service
            or os.getenv('TOOL_PROXY_HTTP_URL')
            or (os.getenv('TOOL_PROXY_HTTP_HOST') and os.getenv('TOOL_PROXY_HTTP_PORT'))
        )

    def _transport_dict(self) -> dict[str, Any]:
        service = (
            os.getenv('TOOL_PROXY_HTTP_SERVICE', '').strip()
            or os.getenv('TOOL_PROXY_SERVICE', '').strip()
            or self._config.service
            or DEFAULT_TOOL_PROXY_HTTP_SERVICE
        )
        namespace = os.getenv('TOOL_PROXY_NAMESPACE', '').strip() or self._config.namespace
        return {
            'service': _normalize_http_service(service),
            'namespace': namespace,
            'metadata': self._config.metadata,
            'env': self._config.env,
            'src_service': self._config.src_service,
            'auth': self._config.auth,
        }

    async def batch_tool_call(self, req: dict[str, Any]) -> dict[str, Any]:
        """Single BatchToolCall request over HTTP JSON (no retry / rate-limit)."""
        transport = self._transport_dict()
        url = _resolve_http_url(transport)
        headers = {'Content-Type': 'application/json'}
        token = _auth_token(transport.get('auth'))
        if token:
            _attach_auth_headers(headers, token)
        timeout = int(self._config.timeout or DEFAULT_TOOL_PROXY_TIMEOUT)
        return await asyncio.to_thread(
            _http_post,
            url,
            req,
            headers=headers,
            timeout=timeout,
        )

    async def batch_tool_call_with_retries(
        self,
        req: dict[str, Any],
        *,
        traj_id: str = '',
    ) -> dict[str, Any]:
        """Batch call with rate-limit, per-attempt retry and a total deadline."""
        log_prefix = f'[ToolProxy][{traj_id}]' if traj_id else '[ToolProxy]'

        if not self.is_configured:
            raise ValueError(
                f'{log_prefix} HTTP transport requires tool_proxy_service or '
                'TOOL_PROXY_HTTP_URL / TOOL_PROXY_HTTP_HOST+PORT to be set'
            )

        logger.debug('%s request: %s', log_prefix, _summarize_req_for_log(req))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s request (full): %s', log_prefix, _format_json_for_log(req))

        rate_limiter = get_rate_limiter('tool_proxy', TOOL_PROXY_RATE_LIMIT)
        tool_names = [call.get('name', '') for call in req.get('calls') or []]
        rate_limit_wait_s = 0.0
        self.last_rate_limit_wait_s = 0.0

        async def _do_request() -> dict[str, Any]:
            nonlocal rate_limit_wait_s
            last_error: Exception | None = None
            for attempt in range(MAX_TOOL_PROXY_RETRIES):
                try:
                    wait_start = time.monotonic()
                    acquired_rate_limit = False
                    try:
                        async with rate_limiter:
                            acquired_rate_limit = True
                            wait_s = time.monotonic() - wait_start
                            rate_limit_wait_s += wait_s
                            if wait_s > 0.1:
                                logger.debug(
                                    '%s rate-limit wait %.3fs before attempt %d/%d, tools=%s',
                                    log_prefix,
                                    wait_s,
                                    attempt + 1,
                                    MAX_TOOL_PROXY_RETRIES,
                                    tool_names,
                                )
                            return await asyncio.wait_for(
                                self.batch_tool_call(req),
                                timeout=PER_ATTEMPT_TIMEOUT,
                            )
                    finally:
                        if not acquired_rate_limit:
                            rate_limit_wait_s += time.monotonic() - wait_start
                except asyncio.TimeoutError:
                    last_error = TimeoutError(
                        f'{log_prefix} attempt {attempt + 1}/{MAX_TOOL_PROXY_RETRIES} '
                        f'timed out after {PER_ATTEMPT_TIMEOUT}s, tools={tool_names}'
                    )
                    logger.warning(str(last_error))
                    if attempt < MAX_TOOL_PROXY_RETRIES - 1:
                        await asyncio.sleep(TOOL_PROXY_RETRY_SLEEP)
                except Exception as exc:
                    last_error = exc
                    is_timeout = 'timeout' in str(exc).lower() or 'timeout' in type(exc).__name__.lower()
                    tag = 'timeout' if is_timeout else type(exc).__name__
                    logger.warning(
                        '%s attempt %d/%d failed (%s), tools=%s: %s',
                        log_prefix,
                        attempt + 1,
                        MAX_TOOL_PROXY_RETRIES,
                        tag,
                        tool_names,
                        exc,
                    )
                    if attempt < MAX_TOOL_PROXY_RETRIES - 1:
                        await asyncio.sleep(TOOL_PROXY_RETRY_SLEEP)
            logger.error(
                '%s all %d retries exhausted, tools=%s, last_error=%s',
                log_prefix,
                MAX_TOOL_PROXY_RETRIES,
                tool_names,
                last_error,
            )
            raise last_error  # type: ignore[misc]

        try:
            rsp = await _do_request()
        finally:
            self.last_rate_limit_wait_s = rate_limit_wait_s

        logger.debug(
            '%s response: %s, rate_limit_wait=%.3fs',
            log_prefix,
            _summarize_rsp_for_log(rsp),
            self.last_rate_limit_wait_s,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s response (full): %s', log_prefix, _format_json_for_log(rsp))

        status = rsp.get('status') or {}
        if status.get('code', 0) != 0:
            logger.warning(
                '%s status code=%s, msg=%s',
                log_prefix,
                status.get('code'),
                status.get('msg', ''),
            )

        return rsp
