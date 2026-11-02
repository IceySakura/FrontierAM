"""Search engine backend for the white-box CC search agent.

A small, self-contained backend that gives the agent two atomic
capabilities:

- :meth:`CCSearchBackend.search` \u2014 Google search via Serper, with
  Hugging Face / domain filtering and Claude Code style rendering.
- :meth:`CCSearchBackend.fetch` \u2014 fetch a single URL via the Jina
  reader and return its **raw page text** (no LLM extraction).

Inspired by :mod:`nexus.extensions.tools.search_stateful.backend`
(``GoogleApiBackend``): one class, two ``async`` methods, credentials
injected at construction time, no orchestration beyond network +
caching. The Claude Code WebSearch / WebFetch tools wrap the two
methods in :mod:`web_search` and :mod:`web_fetch`.

External dependencies kept to the bare minimum:

- :mod:`nexus.extensions.agents.cc_search_agent_whitebox.tools.http_cache` \u2014 generic
  file-cache + single-flight + filelock; reused as cross-cutting infra.
- :mod:`nexus.utils.async_http` \u2014 async HTTP client.

Public surface (kept stable so :mod:`web_search` / :mod:`web_fetch`
need no churn):

- :class:`CCSearchBackend` with :meth:`from_context` /
  :meth:`from_tool_configs` factories.
- :class:`SearchHit` / :class:`SearchResult` / :class:`FetchResult`.
- :class:`BackendError` / :class:`MissingCredentialsError`.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import hmac as hmac_mod
import json
import logging
import os
import random
import re
import traceback
import uuid

from dataclasses import dataclass, field
from typing import Any, NamedTuple
from urllib.parse import urlparse

from nexus.core import ToolContext
from nexus.extensions.agents.base.errors import (
    NonRetryableProviderError,
    is_non_retryable_provider_error,
)
from nexus.extensions.agents.cc_search_agent_whitebox.tools.http_cache import http_cache
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)


# ===========================================================================
# Result dataclasses + errors + small constants
# ===========================================================================


@dataclass
class SearchHit:
    """Single Serper organic result, post domain-filtering."""

    title: str
    url: str
    snippet: str
    date: str | None = None
    source: str | None = None


@dataclass
class SearchResult:
    """Outcome of :meth:`CCSearchBackend.search`."""

    query: str
    hits: list[SearchHit]
    rendered: str
    """Pre-formatted Claude Code style WebSearch summary."""
    success: bool = True


@dataclass
class FetchResult:
    """Outcome of :meth:`CCSearchBackend.fetch`."""

    url: str
    content: str
    """Raw page text returned by the Jina reader."""
    success: bool
    title: str | None = None


class BackendError(Exception):
    """Backend operation error (network / credential / parsing)."""


class MissingCredentialsError(BackendError):
    """Raised when a required credential is missing for the requested op."""


class HttpStatusError(BackendError):
    """Marker for 4xx-style upstream status codes that should not be retried.

    The status code is preserved on the exception so the caller can decide
    whether to surface a tailored message, but no extra body parsing is
    performed here.
    """

    def __init__(self, label: str, status_code: int, body: str) -> None:
        super().__init__(f'{label} returned HTTP {status_code}')
        self.status_code = status_code
        self.body = body


# ===========================================================================
# Summary model (HMAC-based LLM extraction, same as visit.py)
# ===========================================================================


class SummaryModelConfig(NamedTuple):
    """Credentials for one summary model (mirrors visit.py)."""

    model: str
    app_id: str
    app_key: str


SUMMARY_LLM_BASE_URL = 'http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval'
SUMMARY_API_VERSION = 'v2.03'
SUMMARY_SOURCE = 'nexus_tools_search_visit'

# Sentinel strings (same as visit.py) used for flow control.
SUMMARY_LLM_ERROR = '[visit llm error]'
SUMMARY_CONTEXT_LENGTH_ERROR = '[visit context length exceeded]'
SUMMARY_LLM_UNKNOWN_FORMAT = '[visit unknown response format]'

CONTEXT_LENGTH_ERROR_PATTERNS = (
    "requested token count exceeds the model's maximum context length",
    "longer than the model's context length",
    'maximum input length',
    'maximum context length',
    'exceeds the maximum length',
    'context_length_exceeded',
    'input_tokens',
)


class SummaryContextLengthError(BackendError):
    """The summary model returned a context-length error (retryable with truncation)."""


def _get_simple_auth(source: str, secret_id: str, secret_key: str) -> tuple[str, str]:
    """Generate HMAC auth headers (same as visit.py get_simple_auth)."""
    date_time = datetime.datetime.now(datetime.timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = f'hmac id="{secret_id}", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = f'date: {date_time}\nsource: {source}'
    sign = hmac_mod.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


async def _call_llm_server(
    messages: list[dict],
    app_id: str,
    app_key: str,
    model: str,
    max_tries: int = 3,
) -> str:
    """Call trpc-gpt-eval LLM server with HMAC auth (mirrors visit.py call_llm_server)."""
    sign, date_time = _get_simple_auth(SUMMARY_SOURCE, app_id, app_key)
    headers = {
        'Apiversion': SUMMARY_API_VERSION,
        'Authorization': sign,
        'Date': date_time,
        'Source': SUMMARY_SOURCE,
    }
    params: dict[str, Any] = {'model': model, 'thinking': {'type': 'disabled'}, 'enable_thinking': False}
    if 'gpt' in model.lower():
        params = {'model': model, 'max_completion_tokens': 8192}
        if 'gpt-5' in model.lower() or 'gpt5' in model.lower():
            params['service_tier'] = 'flex'
            params['reasoning_effort'] = 'minimal'
    data = {
        'request_id': str(uuid.uuid4()),
        'model_marker': model,
        'messages': messages,
        'params': params,
        'timeout': 1200,
    }

    response = None
    for attempt in range(max_tries):
        try:
            response = await http_client.post(SUMMARY_LLM_BASE_URL, headers=headers, json=data, timeout=1800)
            if response.status_code != 200:
                logger.warning(
                    '[summary] LLM server returned HTTP %d for model=%s: %.500s',
                    response.status_code,
                    model,
                    response.text,
                )
                if is_non_retryable_provider_error(response.text):
                    raise NonRetryableProviderError('Provider returned a non-retryable error.')
                raise ValueError(f'LLM server returned HTTP {response.status_code}')
            result = response.json()

            if not isinstance(result, dict):
                raise TypeError(f'LLM response is not a dict: {type(result).__name__}')

            request_detail = result.get('request_detail') or {}
            if not isinstance(request_detail, dict):
                request_detail = {}
            resp = request_detail.get('response') or {}
            if not isinstance(resp, dict):
                raise TypeError(f'LLM response payload is not a dict: {type(resp).__name__}')

            if 'output' in resp:
                return resp['output']['choices'][0]['message']['content']
            if 'choices' in resp:
                return resp['choices'][0]['message']['content']
            if 'content' in resp:
                return resp['content'][0]['text']

            logger.error('[summary] Unknown response format from model=%s: %r', model, result)
            if attempt == max_tries - 1:
                return SUMMARY_LLM_UNKNOWN_FORMAT
        except NonRetryableProviderError:
            raise
        except Exception:
            error_msg = traceback.format_exc()
            if response is not None:
                error_msg += f', response.text={response.text}'
            if any(pattern in error_msg.lower() for pattern in CONTEXT_LENGTH_ERROR_PATTERNS):
                raise SummaryContextLengthError('context length exceeded') from None
            logger.error(
                '[summary] attempt %d/%d failed for model=%s: %s',
                attempt + 1,
                max_tries,
                model,
                error_msg[:500],
            )
            if attempt == max_tries - 1:
                return SUMMARY_LLM_ERROR
    return SUMMARY_LLM_ERROR


# Hostnames excluded by default from Serper organic results.
EXCLUDED_SEARCH_SITES: tuple[str, ...] = ('huggingface.co', 'hf.co')

DEFAULT_JINA_BASE_URL = os.environ.get(
    'DEFAULT_JINA_URL',
    'http://trpc-gpt-eval.production.polaris:8080/api/jina_ai/search',
)

# 4xx status codes don't get better on retry — bail out fast and let the
# caller surface the response so the LLM can adjust strategy.
NON_RETRYABLE_HTTP_STATUSES: tuple[int, ...] = (400, 401, 403, 404, 422)

# Actionable hints surfaced to the LLM when the Jina reader returns a 4xx.
# Mirrors :mod:`nexus.extensions.tools.web_search.visit`'s ``status_hints``
# so the agent can adjust its strategy (try alternative source, fix typo, …)
# instead of getting a bare HTTP code.
_JINA_STATUS_HINTS: dict[int, str] = {
    400: ('The URL may be malformed or contain invalid characters. Try verifying and correcting the URL format.'),
    403: (
        'Access to this page is forbidden (likely behind a paywall, login wall, or anti-bot protection). '
        'Try finding the same information from an alternative source.'
    ),
    404: (
        'This page does not exist or has been removed. The URL may be outdated or contain a typo. '
        'Try searching for updated links.'
    ),
    422: (
        'The reader service could not process this page. Possible causes include unsupported content '
        'type, timeout, SSL error, or anti-bot protection. Try finding an HTML version of this content '
        'or an alternative source.'
    ),
}

# Hugging Face leaderboards/datasets/spaces leak benchmark answers. The
# whitebox agent is benchmark-evaluated, so ``WebFetch`` blocks them by
# default (matching :mod:`nexus.extensions.tools.web_search.visit`).
_HUGGINGFACE_BLOCKED_PATH_PATTERNS: tuple[str, ...] = ('huggingface.co/datasets', 'huggingface.co/spaces')

# Bearer-token masker for log/error lines. Keeps the first/last 4 chars
# so logs stay useful for correlating which credential pair was in use.
_BEARER_PATTERN = re.compile(
    r'(Bearer\s+)(?P<app_id>[^:\s]+):(?P<app_key>[^?\s]+)(\?provider=(?P<provider>[^\s,)]+))?',
    re.IGNORECASE,
)


# ===========================================================================
# Cached HTTP primitives
# ===========================================================================
#
# Two ``http_cache``-wrapped helpers, one per atomic capability. Both
# return ``str`` (so the cache layer can persist them) and expose
# ``use_cache`` / ``cache_dir`` as required by the decorator. ``cache_keys``
# is the ALLOW LIST: only listed argument names participate in the cache
# key, so credentials never get hashed and key rotation does not bust
# the cache.


def _check_response_status(label: str, response: Any) -> None:
    """Raise the appropriate backend exception when a 4xx/5xx slips through.

    Provider-level "no_budget_error" / "预算不足" markers and classic 4xx
    codes won't get better on retry. Anything else falls through to a
    generic ``BackendError`` (which the caller still retries).
    """
    status_code = getattr(response, 'status_code', 200)
    if status_code == 200:
        return
    body = getattr(response, 'text', '') or ''
    if is_non_retryable_provider_error(body):
        raise NonRetryableProviderError('Provider returned a non-retryable error.')
    if status_code in NON_RETRYABLE_HTTP_STATUSES:
        raise HttpStatusError(label, status_code, body)
    raise BackendError(f'{label} returned HTTP {status_code}: {body[:500]}')


def _require_non_empty_body(label: str, response: Any) -> str:
    """Return ``response.text`` after rejecting empty bodies.

    Some upstream proxies occasionally return ``200`` with an empty body
    (connection drop, edge-side timeout, …). The downstream caller would
    then call ``json.loads('')`` and raise a useless decode error that
    ``http_cache`` would happily persist. Treat empty bodies as transient
    backend failures so :meth:`_with_retry` can retry and ``http_cache``
    skips the write (``BackendError`` propagates up the leader path).
    """
    body = getattr(response, 'text', '') or ''
    if not body.strip():
        raise BackendError(f'{label} returned HTTP 200 with empty body')
    return body


# ---------------------------------------------------------------------------
# Cache namespace choice (Serper)
# ---------------------------------------------------------------------------
# Use ``cc_serper`` so the on-disk cache files do NOT collide with the
# legacy ``google_serper_search`` tool, which writes pre-rendered Markdown
# under ``serper/``. The two formats are NOT interchangeable: this backend
# stores the raw Serper JSON body and parses it via :func:`_filter_hits`,
# while ``google_serper_search`` stores the already-rendered string. Sharing
# the same namespace previously caused "serper returned non-JSON body:
# Expecting value: line 1 column 1 (char 0)" failures when the whitebox
# agent read a file written by the legacy tool.
#
# Jina is intentionally left on the shared ``jina`` namespace: the cached
# value there is the raw response body, byte-compatible with
# :func:`nexus.extensions.tools.web_search.visit.jina_readpage`.
SERPER_CACHE_NAMESPACE = 'cc_serper'


@http_cache(namespace=SERPER_CACHE_NAMESPACE, cache_keys=['query', 'num'])
async def _cached_serper_post(
    url: str,
    query: str,
    num: int,
    *,
    app_id: str,
    app_key: str,
    timeout: float,
    use_cache: bool = False,
    cache_dir: str | None = None,
    cache_read_only: bool = False,
) -> str:
    """POST to Serper and return the raw JSON body as a string.

    The cache value is the **unfiltered** Serper response — domain filters
    (``block_huggingface_search`` / ``allowed_domains`` / ``blocked_domains``)
    run downstream in :func:`_filter_hits`, so they intentionally do NOT
    appear in ``cache_keys``: only inputs that actually change what Serper
    returns belong here. ``url`` is the API endpoint and is treated as a
    deployment constant.

    Stored under the dedicated :data:`SERPER_CACHE_NAMESPACE` namespace so
    files never collide with the legacy ``google_serper_search`` cache,
    which writes pre-rendered Markdown under ``serper/`` and would
    otherwise poison this reader (see module docstring for context).
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {app_id}:{app_key}?provider=serper',
    }
    response = await http_client.post(
        url,
        headers=headers,
        json={'q': query, 'num': num},
        timeout=timeout,
    )
    _check_response_status('serper', response)
    return _require_non_empty_body('serper', response)


@http_cache(namespace='jina', cache_keys=['url'])
async def _cached_jina_fetch(
    url: str,
    *,
    app_id: str,
    app_key: str,
    provider: str,
    base_url: str,
    timeout: float,
    use_cache: bool = False,
    cache_dir: str | None = None,
    cache_read_only: bool = False,
) -> str:
    """POST to Jina and return the raw JSON body as a string.

    Headers and namespace are byte-for-byte aligned with
    :func:`nexus.extensions.tools.web_search.visit.jina_readpage` so that the
    on-disk ``{cache_dir}/jina/<md5(url)>.json`` cache files are interchangeable
    between this agent and the legacy ``visit`` tool — same URL hits the same
    cache regardless of which path warmed it.
    """
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {app_id}:{app_key}?provider={provider}',
    }
    response = await http_client.post(base_url, headers=headers, json={'url': url}, timeout=timeout)
    _check_response_status('jina', response)
    return _require_non_empty_body('jina', response)


# ===========================================================================
# Pure helpers (search-side filtering + rendering)
# ===========================================================================


def _split_url(url: str) -> tuple[str, str]:
    """Return (lowercase host, path) for an http(s) URL. Empty host on
    parse failure or non-http(s) scheme (``javascript:``, ``data:``, …).

    Note: ``mailto:`` / ``javascript:`` / ``data:`` are opaque schemes with no
    ``//``, so we cannot use ``'://' in url`` to detect "scheme present".
    Parse first, then fall back to ``https://`` only when no scheme was found.
    """
    if not isinstance(url, str):
        return '', ''
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ('http', 'https'):
        return '', ''
    if not parsed.scheme:
        parsed = urlparse(f'https://{url}')
    return (parsed.hostname or '').lower(), parsed.path or ''


def _split_pattern(pattern: str) -> tuple[str, str] | None:
    """Parse an Anthropic-style domain pattern into (host, path-with-optional-`*`).

    Returns ``None`` for malformed patterns (scheme present, host wildcard,
    multiple ``*``, empty host, …) so the caller treats the entry as a no-op
    instead of silently matching everything or nothing.

    Reference: https://docs.claude.com/en/docs/agents-and-tools/tool-use/server-tools#domain-filtering
    """
    pattern = (pattern or '').strip().lower()
    if not pattern or '://' in pattern:
        return None
    host, slash, path = pattern.partition('/')
    if not host or '*' in host:  # `*.example.com` and `ex*.com` are invalid.
        return None
    path = (slash + path) if slash else ''
    if path.count('*') > 1:  # Anthropic permits at most one wildcard, in the path.
        return None
    return host, path


def _path_matches(url_path: str, pattern_path: str) -> bool:
    """Anthropic path matching: empty pattern matches anything; otherwise
    prefix match with at most one ``*`` standing in for arbitrary text."""
    if not pattern_path:
        return True
    if '*' not in pattern_path:
        return url_path.startswith(pattern_path)
    head, _, tail = pattern_path.partition('*')
    return url_path.startswith(head) and tail in url_path[len(head) :]


def _domain_matches(url: str, pattern: str) -> bool:
    """Anthropic-style host+path match. Strict subdomain semantics:
    ``docs.example.com`` matches ``docs.example.com`` and its sub-subdomains
    (``api.docs.example.com``) but NOT ``example.com`` or ``api.example.com``.
    """
    parsed = _split_pattern(pattern)
    if parsed is None:
        return False
    pat_host, pat_path = parsed
    url_host, url_path = _split_url(url)
    if not url_host:
        return False
    if url_host != pat_host and not url_host.endswith(f'.{pat_host}'):
        return False
    return _path_matches(url_path, pat_path)


def _is_excluded_search_result(url: str, *, block_huggingface: bool) -> bool:
    if not block_huggingface or not url:
        return False
    host, _ = _split_url(url)
    return any(host == site or host.endswith(f'.{site}') for site in EXCLUDED_SEARCH_SITES)


def is_huggingface_dataset_or_space_url(url: str) -> bool:
    """Return True for ``huggingface.co/datasets/...`` or ``/spaces/...`` URLs.

    Used by ``WebFetch`` to refuse benchmark-leaking URLs by default. We
    treat the substring match conservatively (no scheme/host parsing) so a
    rewritten URL like ``//huggingface.co/datasets/foo`` is still caught.
    """
    return bool(url) and any(marker in url for marker in _HUGGINGFACE_BLOCKED_PATH_PATTERNS)


def _mask_auth_bearer(content: str) -> str:
    """Mask ``app_id`` / ``app_key`` inside a ``Bearer ...`` header for logs.

    Operates on the whole string (not just the leading match), because
    bearer tokens can appear anywhere — e.g. inside a stringified
    ``response`` object that we drop into a log line.
    """

    def _mask_token(token: str) -> str:
        if len(token) <= 8:
            return '****'
        return f'{token[:4]}****{token[-4:]}'

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        provider = match.group('provider') or 'unknown'
        return f'{prefix}{_mask_token(match.group("app_id"))}:{_mask_token(match.group("app_key"))}?provider={provider}'

    return _BEARER_PATTERN.sub(_replace, content)


def _filter_hits(
    raw: dict,
    *,
    block_huggingface: bool,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> list[SearchHit]:
    organic = raw.get('organic') if isinstance(raw, dict) else None
    if not isinstance(organic, list):
        return []
    # An explicit allow-list is a stronger user intent than the default
    # huggingface block, so don't second-guess the caller.
    apply_hf_block = block_huggingface and not allowed_domains
    hits: list[SearchHit] = []
    for page in organic:
        if not isinstance(page, dict):
            continue
        link = page.get('link', '')
        # Drop entries we cannot map to a real http(s) host: a non-string
        # `link` (Serper has historically returned dict/null in odd cases) or
        # a non-http scheme (`javascript:`, `data:`) would otherwise bypass
        # both the huggingface block and the allow/block filters.
        if not _split_url(link)[0]:
            continue
        if _is_excluded_search_result(link, block_huggingface=apply_hf_block):
            continue
        if allowed_domains and not any(_domain_matches(link, d) for d in allowed_domains):
            continue
        if blocked_domains and any(_domain_matches(link, d) for d in blocked_domains):
            continue
        hits.append(
            SearchHit(
                title=page.get('title', ''),
                url=link,
                snippet=page.get('snippet', ''),
                date=page.get('date'),
                source=page.get('source'),
            )
        )
    return hits


def _render_hits(query: str, hits: list[SearchHit]) -> str:
    """Render hits as a Claude Code style WebSearch summary string."""
    if not hits:
        return f"No results found for '{query}'. Try with a more general query."
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        date_published = f'\nDate published: {hit.date}' if hit.date else ''
        source = f'\nSource: {hit.source}' if hit.source else ''
        snippet_block = f'\n{hit.snippet}' if hit.snippet else ''
        blocks.append(f'{idx}. [{hit.title}]({hit.url}){date_published}{source}{snippet_block}')
    return f'Web search results for query: "{query}"\n\nFound {len(blocks)} results:\n\n## Web Results\n' + '\n\n'.join(
        blocks
    )


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, '')
    return raw.lower() in ('1', 'true', 'yes', 'on') if raw else default


# ===========================================================================
# CCSearchBackend
# ===========================================================================


@dataclass
class CCSearchBackend:
    """Search engine backend for the white-box CC search agent.

    Each instance is a stateless wrapper around a fixed set of credentials;
    construction is cheap. Use :meth:`from_context` (preferred) or
    :meth:`from_tool_configs` to build one from a ``ToolContext``.
    """

    # ---- Serper (search) ----
    # Defaults tuned against ``google_serper_search``: 3 attempts is enough
    # for transient 5xx / connection resets; 5s spacing leaves room for the
    # upstream pool to recover (1s was effectively "retry immediately").
    serper_url: str = ''
    serper_app_id: str = ''
    serper_app_key: str = ''
    serper_num_results: int = 10
    serper_timeout: float = 60.0
    serper_max_retries: int = 3
    serper_retry_delay: float = 5.0
    # Drops huggingface.co / hf.co (incl. subdomains) from Serper organic
    # results. Env override: ``DEFAULT_SERPER_BLOCK_HUGGINGFACE``.
    block_huggingface_search: bool = True

    # ---- Jina (fetch) ----
    # Jina reader fronts third-party scrapers and is noticeably flakier
    # than Serper, so we follow ``visit.py`` and allocate more headroom:
    # 5 attempts with exponential backoff (10s, 20s, 40s, 80s) tolerates
    # short-lived upstream brownouts without burning the whole agent loop
    # on a single bad URL.
    jina_base_url: str = DEFAULT_JINA_BASE_URL
    jina_app_id: str = ''
    jina_app_key: str = ''
    jina_provider: str = 'aws_jina'
    jina_timeout: float = 60.0
    jina_max_retries: int = 5
    jina_retry_delay: float = 10.0
    # Refuses fetching ``huggingface.co/datasets/*`` and ``/spaces/*`` paths.
    # Env override: ``DEFAULT_JINA_BLOCK_HUGGINGFACE``.
    block_huggingface_fetch: bool = True

    # ---- Summary (LLM extraction) ----
    # Defaults mirror :mod:`nexus.extensions.tools.web_search.visit`:
    # cap raw page text at 200k chars before calling the summary model;
    # if the model raises a context-length error, retry with the page
    # progressively shrunk by ``context_truncate_chars_per_attempt`` per
    # attempt (50k, 100k, ... chars dropped from the tail).
    webcontent_maxlength: int = 200_000
    context_truncate_chars_per_attempt: int = 50_000
    summary_max_attempts: int = 3

    # ---- Cache (on by default; caller opts out via tool_configs.use_cache=false) ----
    use_cache: bool = False
    cache_dir: str | None = None
    # Read-only mode: reads from cache on hit, but never writes on miss.
    # Useful when CephFS MDS write latency is too high but existing cache is valuable.
    cache_read_only_search: bool = False
    cache_read_only_fetch: bool = False

    # ---- Summary model (HMAC-based, from env vars like visit.py) ----
    summary_models: list[SummaryModelConfig] = field(default_factory=list)

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_tool_configs(
        cls,
        tool_configs: dict[str, dict[str, Any]],
        *,
        search_registry_name: str | None = None,
        fetch_registry_name: str | None = None,
    ) -> CCSearchBackend:
        """Build a backend from a ``ToolContext.tool_configs`` mapping.

        Each lookup falls back to the legacy upstream key
        (``google_serper_search`` / ``visit``) and finally to environment
        variables, so existing search-agent YAML configs keep working.
        """
        search_cfg = (tool_configs.get(search_registry_name, {}) if search_registry_name else {}) or tool_configs.get(
            'google_serper_search', {}
        )
        fetch_cfg = (tool_configs.get(fetch_registry_name, {}) if fetch_registry_name else {}) or tool_configs.get(
            'visit', {}
        )

        # ---- Serper ----
        serper_url = search_cfg.get('serper_url', os.environ.get('SEARCH_SERPER_URL', ''))
        serper_app_id = search_cfg.get('serper_app_id', os.environ.get('SEARCH_SERPER_APP_ID', ''))
        serper_app_key = search_cfg.get('serper_app_key', os.environ.get('SEARCH_SERPER_APP_KEY', ''))
        block_hf_search = search_cfg.get(
            'block_huggingface', _bool_env('DEFAULT_SERPER_BLOCK_HUGGINGFACE', default=True)
        )
        num_results = int(search_cfg.get('num_results', 10))
        serper_max_retries = int(search_cfg.get('max_retries', cls.serper_max_retries))
        serper_retry_delay = float(search_cfg.get('retry_delay', cls.serper_retry_delay))

        # ---- Jina ----
        jina_app_id = fetch_cfg.get('jina_app_id', os.environ.get('DEFAULT_JINA_APP_ID', ''))
        jina_app_key = fetch_cfg.get('jina_app_key', os.environ.get('DEFAULT_JINA_APP_KEY', ''))
        jina_provider = fetch_cfg.get('jina_provider', os.environ.get('DEFAULT_JINA_PROVIDER', 'aws_jina'))
        jina_base_url = fetch_cfg.get('jina_url', os.environ.get('DEFAULT_JINA_URL', DEFAULT_JINA_BASE_URL))
        jina_max_retries = int(fetch_cfg.get('max_retries', cls.jina_max_retries))
        jina_retry_delay = float(fetch_cfg.get('retry_delay', cls.jina_retry_delay))
        block_hf_fetch = fetch_cfg.get('block_huggingface', _bool_env('DEFAULT_JINA_BLOCK_HUGGINGFACE', default=True))

        # ---- Summary (LLM extraction) ----
        webcontent_maxlength = int(fetch_cfg.get('webcontent_maxlength', cls.webcontent_maxlength))
        context_truncate_chars_per_attempt = int(
            fetch_cfg.get('context_truncate_chars_per_attempt', cls.context_truncate_chars_per_attempt)
        )
        summary_max_attempts = int(fetch_cfg.get('summary_max_attempts', cls.summary_max_attempts))

        # ---- Cache ----
        use_cache_search = search_cfg.get('use_cache', _bool_env('SEARCH_SERPER_USE_CACHE', default=True))
        use_cache_fetch = fetch_cfg.get('use_cache', _bool_env('JINA_USE_CACHE', default=True))
        # Single backend-wide flag; both atoms must opt out before we disable.
        use_cache = bool(use_cache_search and use_cache_fetch)
        cache_dir = search_cfg.get('cache_dir') or fetch_cfg.get('cache_dir') or os.environ.get('SEARCH_CACHE_DIR')
        # Read-only mode: reads from cache but never writes (avoids CephFS MDS write storms).
        cache_read_only_search = search_cfg.get(
            'cache_read_only', _bool_env('SEARCH_SERPER_CACHE_READ_ONLY', default=False)
        )
        cache_read_only_fetch = fetch_cfg.get('cache_read_only', _bool_env('JINA_CACHE_READ_ONLY', default=False))

        # ---- Summary models (from tool_configs / env vars, same as visit.py) ----
        summary_models: list[SummaryModelConfig] = []
        model = fetch_cfg.get('jina_model', os.environ.get('DEFAULT_JINA_MODEL', ''))
        model_app_id = fetch_cfg.get('jina_model_app_id', os.environ.get('DEFAULT_JINA_MODEL_APP_ID', ''))
        model_app_key = fetch_cfg.get('jina_model_app_key', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY', ''))
        if model and model_app_id and model_app_key:
            summary_models.append(SummaryModelConfig(model, model_app_id, model_app_key))
            # Backup model (optional, _2 suffix)
            model_2 = fetch_cfg.get('jina_model_2', os.environ.get('DEFAULT_JINA_MODEL_2', ''))
            if model_2:
                model_app_id_2 = (
                    fetch_cfg.get('jina_model_app_id_2', os.environ.get('DEFAULT_JINA_MODEL_APP_ID_2', ''))
                    or model_app_id
                )
                model_app_key_2 = (
                    fetch_cfg.get('jina_model_app_key_2', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY_2', ''))
                    or model_app_key
                )
                summary_models.append(SummaryModelConfig(model_2, model_app_id_2, model_app_key_2))

        return cls(
            serper_url=serper_url,
            serper_app_id=serper_app_id,
            serper_app_key=serper_app_key,
            serper_num_results=num_results,
            serper_max_retries=serper_max_retries,
            serper_retry_delay=serper_retry_delay,
            block_huggingface_search=bool(block_hf_search),
            jina_base_url=jina_base_url,
            jina_app_id=jina_app_id,
            jina_app_key=jina_app_key,
            jina_provider=jina_provider,
            jina_max_retries=jina_max_retries,
            jina_retry_delay=jina_retry_delay,
            block_huggingface_fetch=bool(block_hf_fetch),
            webcontent_maxlength=webcontent_maxlength,
            context_truncate_chars_per_attempt=context_truncate_chars_per_attempt,
            summary_max_attempts=summary_max_attempts,
            use_cache=use_cache,
            cache_dir=cache_dir,
            cache_read_only_search=bool(cache_read_only_search),
            cache_read_only_fetch=bool(cache_read_only_fetch),
            summary_models=summary_models,
        )

    @classmethod
    def from_context(cls, context: ToolContext) -> CCSearchBackend:
        """Build a backend using the package's registered tool keys.

        Hardcoded registry keys avoid importing the sibling tool modules
        (which would re-import this module and create a cycle). Keep these
        in sync with ``WEB_SEARCH_TOOL_REGISTRY_NAME`` in ``web_search.py``,
        ``WEB_FETCH_TOOL_REGISTRY_NAME`` in ``web_fetch.py``, and
        ``[tool.nexus.tools]`` in ``pyproject.toml``.
        """
        return cls.from_tool_configs(
            context.tool_configs,
            search_registry_name='cc_search_agent_whitebox/web_search',
            fetch_registry_name='cc_search_agent_whitebox/web_fetch',
        )

    # ---------------------------------------------------------------- checks
    def _require_search_creds(self) -> None:
        if not (self.serper_url and self.serper_app_id and self.serper_app_key):
            raise MissingCredentialsError(
                'CCSearchBackend.search requires serper_url + serper_app_id + serper_app_key. '
                'Configure tool_configs (or SEARCH_SERPER_URL / SEARCH_SERPER_APP_ID / SEARCH_SERPER_APP_KEY env vars).'
            )

    def _require_fetch_creds(self) -> None:
        if not (self.jina_app_id and self.jina_app_key):
            raise MissingCredentialsError(
                'CCSearchBackend.fetch requires jina_app_id + jina_app_key. '
                'Configure tool_configs (or DEFAULT_JINA_APP_ID / DEFAULT_JINA_APP_KEY env vars).'
            )

    def _require_summary_creds(self) -> None:
        if not self.summary_models:
            raise MissingCredentialsError(
                'CCSearchBackend.call_summary_model requires summary model credentials. '
                'Configure tool_configs (jina_model / jina_model_app_id / jina_model_app_key) '
                'or env vars (DEFAULT_JINA_MODEL / DEFAULT_JINA_MODEL_APP_ID / DEFAULT_JINA_MODEL_APP_KEY).'
            )

    @property
    def has_summary_models(self) -> bool:
        """True when at least one HMAC summary model credential is configured."""
        return bool(self.summary_models)

    # --------------------------------------------------------------- summary
    async def call_summary_model(self, messages: list[dict], *, attempt: int = 0) -> str:
        """Call the summary LLM via HMAC auth (same pattern as visit.py).

        Selects a model from ``summary_models`` based on the attempt index
        (rotates through available models). Raises :class:`SummaryContextLengthError`
        if the model reports context-length exceeded (caller should truncate and retry).
        Returns the extracted text content on success.
        """
        self._require_summary_creds()
        candidates = random.sample(self.summary_models, k=len(self.summary_models))
        selected = candidates[min(attempt, len(candidates) - 1)]
        result = await _call_llm_server(messages, selected.app_id, selected.app_key, selected.model, max_tries=3)
        if result == SUMMARY_LLM_ERROR:
            raise BackendError(f'Summary model {selected.model} failed after retries')
        if result == SUMMARY_LLM_UNKNOWN_FORMAT:
            raise BackendError(f'Summary model {selected.model} returned unknown response format')
        return result

    # ---------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        *,
        num_results: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> SearchResult:
        """Run a Google search via Serper and return filtered, rendered hits."""
        self._require_search_creds()
        query = (query or '').strip()
        if not query:
            logger.warning('[CCSearchBackend.search] empty query, returning no results')
            return SearchResult(query=query, hits=[], rendered="No results found for ''.", success=False)

        topn = int(num_results or self.serper_num_results)
        raw_text = await self._with_retry(
            label='serper',
            attempts=self.serper_max_retries,
            delay=self.serper_retry_delay,
            call=lambda: _cached_serper_post(
                self.serper_url,
                query,
                topn,
                app_id=self.serper_app_id,
                app_key=self.serper_app_key,
                timeout=self.serper_timeout,
                use_cache=self.use_cache,
                cache_dir=self.cache_dir,
                cache_read_only=self.cache_read_only_search,
            ),
        )

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error('[CCSearchBackend.search] non-JSON response: %.200s', raw_text)
            raise BackendError(f'serper returned non-JSON body: {exc}') from exc

        hits = _filter_hits(
            raw,
            block_huggingface=self.block_huggingface_search,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
        organic_count = len(raw.get('organic', [])) if isinstance(raw, dict) else 0
        if organic_count != len(hits):
            logger.debug('[CCSearchBackend.search] query=%s organic=%d filtered=%d', query, organic_count, len(hits))
        return SearchResult(
            query=query,
            hits=hits,
            rendered=_render_hits(query, hits),
            success=bool(hits),
        )

    # ----------------------------------------------------------------- fetch
    async def fetch(self, url: str) -> FetchResult:
        """Fetch ``url`` via Jina and return its raw page text.

        Refuses Hugging Face dataset/space pages when
        ``block_huggingface_fetch`` is on. Mirrors the search-side filter
        in :func:`_filter_hits` (gated by ``block_huggingface_search``):
        benchmark-evaluated agents have leaked answers from leaderboard
        pages historically, so both default to deny.
        """
        self._require_fetch_creds()
        url = (url or '').strip()
        if not url:
            logger.warning('[CCSearchBackend.fetch] empty URL provided')
            return FetchResult(url=url, content='Error: Missing URL.', success=False)
        if self.block_huggingface_fetch and is_huggingface_dataset_or_space_url(url):
            logger.warning('[CCSearchBackend.fetch] blocked HuggingFace dataset/space URL: %s', url)
            return FetchResult(
                url=url,
                content=(
                    f'Error: Refusing to fetch "{url}" — Hugging Face dataset/space pages are blocked '
                    'to avoid benchmark answer leakage. Try the project homepage or paper instead.'
                ),
                success=False,
            )

        try:
            raw_text = await self._with_retry(
                label='jina',
                attempts=self.jina_max_retries,
                delay=self.jina_retry_delay,
                call=lambda: _cached_jina_fetch(
                    url,
                    app_id=self.jina_app_id,
                    app_key=self.jina_app_key,
                    provider=self.jina_provider,
                    base_url=self.jina_base_url,
                    timeout=self.jina_timeout,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    cache_read_only=self.cache_read_only_fetch,
                ),
            )
        except HttpStatusError as exc:
            hint = _JINA_STATUS_HINTS.get(exc.status_code, 'An unexpected client error occurred.')
            logger.warning('[CCSearchBackend.fetch] HTTP %d for url=%s: %s', exc.status_code, url, hint)
            return FetchResult(
                url=url,
                content=f'Error fetching {url}: HTTP {exc.status_code}. {hint}',
                success=False,
            )
        except BackendError as exc:
            logger.warning('[CCSearchBackend.fetch] failed url=%s: %s', url, exc)
            return FetchResult(url=url, content=f'Error fetching {url}: {exc}', success=False)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return FetchResult(url=url, content=raw_text, success=bool(raw_text))

        data = (payload.get('data') or {}) if isinstance(payload, dict) else {}
        content = (data.get('content') or data.get('text')) if isinstance(data, dict) else None
        title = data.get('title') if isinstance(data, dict) else None

        if not content:
            logger.warning('[CCSearchBackend.fetch] empty content for url=%s', url)
            return FetchResult(
                url=url, content=f'Error fetching {url}: empty page content.', success=False, title=title
            )
        return FetchResult(url=url, content=content, success=True, title=title)

    # ------------------------------------------------------------- internals
    async def _with_retry(self, *, label: str, attempts: int, delay: float, call):
        """Run ``call()`` with bounded retry + exponential backoff.

        :class:`NonRetryableProviderError` (budget exhausted, …) and
        :class:`HttpStatusError` (4xx) are re-raised immediately because
        retrying would just trip the same upstream gate. Everything else
        (timeouts, 5xx, connection resets) waits ``delay * 2**attempt``
        before the next attempt — same shape as ``visit.py``'s LLM
        extraction loop, capped by ``attempts``.
        """
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                result = await call()
                if attempt > 0:
                    logger.info('[CCSearchBackend] %s: succeeded on attempt %d/%d', label, attempt + 1, attempts)
                return result
            except (NonRetryableProviderError, HttpStatusError):
                raise
            except Exception as exc:
                last_exc = exc
                next_delay = delay * (2**attempt)
                logger.warning(
                    '[CCSearchBackend] %s attempt %d/%d failed: %s. Retry in %.1fs.',
                    label,
                    attempt + 1,
                    attempts,
                    _mask_auth_bearer(repr(exc)),
                    next_delay,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(next_delay)
        reason = type(last_exc).__name__ if last_exc else 'unknown error'
        raise BackendError(f'{label} failed after {attempts} retries: {reason}')


__all__ = [
    'BackendError',
    'CCSearchBackend',
    'FetchResult',
    'HttpStatusError',
    'MissingCredentialsError',
    'SearchHit',
    'SearchResult',
    'SummaryContextLengthError',
    'SummaryModelConfig',
    'is_huggingface_dataset_or_space_url',
]
