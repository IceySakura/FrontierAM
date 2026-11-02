"""Search tool for the S2T (speech-to-text) Search Agent — yuanbao-reward glue.

This is the SAME PromptSearch backend used by the original ``search_s2t_rl``
project (``SearchClient`` + ``extract_web_docs`` from the vendored
``search_api_client.py``). It is re-registered here under the
``search_s2t_yb_glue/search`` canonical key and exposed to the model as
``web_search`` (sp.md contract: a single ``query`` array of atomic sub-queries).
No other backends are provided — the yuanbao ``picture_search`` /
``video_search`` / ``finance_search`` tools are intentionally NOT used.

Resolution order for the client implementation:

1. ``SEARCH_S2T_CLIENT_PATH`` environment variable (explicit override).
2. The vendored sibling ``search_api_client.py`` next to this module.
3. A repo-level ``agentic_rl/search_api_client.py`` discovered by walking up
   the directory tree (legacy standalone-nexus layout).

Configuration (all optional) is read from ``tool_configs['search_s2t_yb_glue']``
(falling back to ``tool_configs['search_s2t']``) in the run config, then to
environment variables, then to the defaults baked into ``search_api_client.py``::

    tool_configs:
      search_s2t_yb_glue:
        search_url: "https://.../PromptSearch"   # or SEARCH_URL
        search_source: "agentic_search"           # or SEARCH_SOURCE
        top_k: 10                                  # or SEARCH_S2T_TOPK (render caps at MAX_DOCS_FOR_RENDER)
        timeout: 30                                # or SEARCH_TIMEOUT
        connect_timeout: 5                         # or SEARCH_CONNECT_TIMEOUT
        max_total_chars: 10000                     # or SEARCH_S2T_MAX_TOTAL_CHARS (global content budget)
        max_attempts: 3                            # or SEARCH_S2T_MAX_ATTEMPTS (API-error retries)
        max_connections: 256                       # or SEARCH_S2T_MAX_CONNECTIONS (worker-level pool cap)
        max_keepalive_connections: 256             # or SEARCH_S2T_MAX_KEEPALIVE_CONNECTIONS
        keepalive_expiry: 60                       # or SEARCH_S2T_KEEPALIVE_EXPIRY (seconds)

The rendered ``role=tool`` payload mirrors ``agent_loop.py``'s
``render_web_search_result``: each kept doc is emitted as a
``[web_doc N begin] … [web_doc N end]`` block carrying search query / title /
site / authority level / publish time / content, under a global content-char
budget with a first-doc-overflow special case; an empty / failed search feeds
the fixed Chinese text ``未查询到对应搜索结果`` (the upstream error is never
surfaced to the model).
"""

import asyncio
import importlib.util
import logging
import os
import weakref

from functools import lru_cache
from pathlib import Path
from types import ModuleType

import httpx

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


logger = logging.getLogger(__name__)


# Rendering / retry budgets — aligned with agent_loop.py.
MAX_DOCS_FOR_RENDER = 10
MAX_TOTAL_CHARS = 10000
TRUNCATE_TAIL = '......'
MAX_SEARCH_ATTEMPTS = 3
MAX_SUBQUERIES_PER_CALL = 3

# Worker-level connection pool defaults. The cap should stay at or above the
# rollout concurrency of a single worker (``gen_batch_size * rollout.n /
# agent.num_workers``, plus validation tasks that overlap with it): once the pool
# is saturated, requests queue for a free connection and eventually fail with
# PoolTimeout, which costs the trajectory its search result.
#
# Keepalive is pinned to the cap rather than kept smaller: httpcore prunes idle
# connections while the *total* connection count exceeds
# max_keepalive_connections (AsyncConnectionPool._assign_requests_to_connections
# measures the whole pool, not just its idle members), so any keepalive below the
# peak concurrency closes idle connections on every pool pass and drops reuse to
# zero for the duration of the burst.
DEFAULT_MAX_CONNECTIONS = 256
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 256
DEFAULT_KEEPALIVE_EXPIRY = 60.0
# Fixed text fed to the model when a search yields no usable docs OR fails after
# all retries. Mirrors agent_loop.py: the raw upstream error is never surfaced.
EMPTY_SEARCH_TEXT = '未查询到对应搜索结果'


def _resolve_client_path() -> Path:
    """Locate ``search_api_client.py``.

    Resolution order:
    1. ``SEARCH_S2T_CLIENT_PATH`` environment variable (explicit override).
    2. The vendored sibling ``search_api_client.py`` next to this module — the
       in-package copy that ships with the Nexus extension. This is the path
       used during HyTuner RL training because the project directory is packaged
       and shipped to Ray workers as a single unit.
    3. Walk up the directory tree from this file looking for
       ``agentic_rl/search_api_client.py`` (the legacy standalone-nexus layout).
    """
    override = os.getenv('SEARCH_S2T_CLIENT_PATH')
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()

    sibling = here.parent / 'search_api_client.py'
    if sibling.exists():
        return sibling

    for parent in here.parents:
        candidate = parent / 'agentic_rl' / 'search_api_client.py'
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        'Could not locate search_api_client.py. Expected a vendored copy next to '
        f'{here.parent}/search_api_client.py, a repo-level agentic_rl/search_api_client.py, '
        'or the SEARCH_S2T_CLIENT_PATH environment variable.'
    )


@lru_cache(maxsize=1)
def _load_search_module() -> ModuleType:
    """Dynamically import ``search_api_client.py`` as a module (cached)."""
    client_path = _resolve_client_path()
    spec = importlib.util.spec_from_file_location('search_s2t_yb_glue_search_api_client', client_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to build import spec for {client_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One httpx client per event loop per transport config, reused by every
# agent-loop coroutine on that worker. Keyed weakly so a finished loop's client
# becomes collectable; a client can never be shared across loops or processes.
_POOLED_HTTP_CLIENTS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[float, float, int, int, float], httpx.AsyncClient]
] = weakref.WeakKeyDictionary()


def _get_pooled_http_client(
    *,
    timeout: float,
    connect_timeout: float,
    max_connections: int,
    max_keepalive_connections: int,
    keepalive_expiry: float,
) -> httpx.AsyncClient:
    """Return the worker-level client for this event loop and transport config.

    Creating the client is synchronous, so no lock is needed: the event loop
    cannot interleave another coroutine between the lookup and the insert.
    """
    loop = asyncio.get_running_loop()
    clients = _POOLED_HTTP_CLIENTS.get(loop)
    if clients is None:
        clients = {}
        _POOLED_HTTP_CLIENTS[loop] = clients

    key = (timeout, connect_timeout, max_connections, max_keepalive_connections, keepalive_expiry)
    client = clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            # ``pool`` bounds how long a request waits for a free connection.
            # Reuse the request timeout so a saturated pool degrades like a slow
            # backend instead of failing fast.
            timeout=httpx.Timeout(timeout, connect=connect_timeout, pool=timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
            ),
        )
        clients[key] = client
        logger.debug(
            'search_s2t_yb_glue: created pooled http client (timeout=%s connect=%s max_connections=%s)',
            timeout,
            connect_timeout,
            max_connections,
        )
    return client


async def aclose_pooled_http_clients() -> None:
    """Close every pooled client owned by the current event loop.

    Call this on worker shutdown. Reusing a keep-alive connection can still fail
    when the peer closed it in the meantime, which is why the search retry loop
    stays in place regardless of pooling.
    """
    loop = asyncio.get_running_loop()
    clients = _POOLED_HTTP_CLIENTS.pop(loop, None) or {}
    for client in clients.values():
        if not client.is_closed:
            await client.aclose()


class SearchS2TInput(BaseModel):
    query: list[str] = Field(
        min_length=1,
        max_length=MAX_SUBQUERIES_PER_CALL,
        description='用于联网检索的子任务查询列表，每个元素为一个独立、原子化的搜索 query。'
        '拆解原则：按问题结构拆分、避免冗余、优先核心高频词；时间/地点显式化，专有名词保持完整；语种与用户保持一致。',
    )


def _render_doc_block(doc: dict, idx: int, content: str) -> str:
    """Render a single web doc as a ``[web_doc N begin] … [web_doc N end]`` block.

    Field layout mirrors agent_loop.py::render_web_search_result exactly so the
    policy sees the same web-doc schema at RL rollout as it does when served.
    """
    authority_text = doc.get('authority_level_text') or '权威性未知'
    publish_time = doc.get('publish_time') or '发布时间未知'
    return (
        f'[web_doc {idx} begin]\n'
        f'[search query] {doc.get("search_query", "")}\n'
        f'[web_doc_title] {doc.get("title", "")}\n'
        f'[web_doc_site] {doc.get("site_name", "")}\n'
        f'[web_doc_authority_level] {authority_text}\n'
        f'[web_doc_publish_time] {publish_time}\n'
        f'[web_doc_content begin]\n'
        f'{content}\n'
        f'[web_doc_content end]\n'
        f'[web_doc {idx} end]'
    )


def _render_web_search_result(docs: list[dict], *, max_total_chars: int, max_docs: int) -> str:
    """Render extracted web docs into the model-facing tool payload.

    Port of agent_loop.py::render_web_search_result:
      * keep at most ``max_docs`` docs;
      * cumulative content length across kept docs must not exceed
        ``max_total_chars`` — once adding a doc would overflow, stop appending;
      * special case: if the *first doc with content* alone exceeds
        ``max_total_chars``, truncate it to ``max_total_chars`` + ``......`` and
        make it the sole rendered doc;
      * docs whose content (``content`` → ``abstract`` fallback) is empty are
        skipped; if nothing survives, return ``未查询到对应搜索结果``.
    """
    if not docs:
        return EMPTY_SEARCH_TEXT

    blocks: list[str] = []
    kept_idx = 0
    total_chars = 0

    for doc in docs:
        # content → abstract fallback, matching agent_loop.py exactly (no strip:
        # whitespace-only content is kept, and the budget counts raw len()).
        content = doc.get('content') or doc.get('abstract') or ''
        if not content:
            continue

        # Special case: first doc with content already blows the whole budget.
        if kept_idx == 0 and len(content) > max_total_chars:
            content = content[:max_total_chars] + TRUNCATE_TAIL
            kept_idx += 1
            blocks.append(_render_doc_block(doc, kept_idx, content))
            break

        # Normal case: skip (and stop) once this doc would overflow the budget.
        if total_chars + len(content) > max_total_chars:
            break

        kept_idx += 1
        total_chars += len(content)
        blocks.append(_render_doc_block(doc, kept_idx, content))

        if kept_idx >= max_docs:
            break

    if not blocks:
        return EMPTY_SEARCH_TEXT

    return '\n'.join(blocks)


@tool_registry.register(
    'search_s2t_yb_glue/search',
    meta={
        'name': 'web_search',
        'description': '联网搜索工具，基于query执行网络信息检索，返回相关检索结果。',
        'tool_type': ToolType.LOCAL,
    },
)
async def search_s2t_yb_glue_tool(params: SearchS2TInput, context: ToolContext) -> ToolResponse:
    """Retrieve relevant web documents for the given queries via PromptSearch.

    Args:
        params: SearchS2TInput containing the list of sub-task search queries.
        context: ToolContext for the current tool invocation.

    Returns:
        ToolResponse with formatted search results (title / source / time / content).
    """
    module = _load_search_module()
    search_client_cls = module.SearchClient
    extract_web_docs = module.extract_web_docs

    cfg = context.tool_configs.get('search_s2t_yb_glue') or context.tool_configs.get('search_s2t', {})

    search_url = cfg.get('search_url') or os.getenv('SEARCH_URL') or module.DEFAULT_SEARCH_URL
    search_source = cfg.get('search_source') or os.getenv('SEARCH_SOURCE') or module.DEFAULT_SEARCH_SOURCE
    ext_params = cfg.get('ext_params') or None
    # Per-row cutoff date: supplied via task_config (parquet `time_range` column)
    # -> agent -> ExecutionContext.tool_configs. Falls back to the SEARCH_TIME_RANGE
    # env var (global cutoff) when not provided per row. Empty string = no filter.
    time_range = cfg.get('time_range') or os.getenv('SEARCH_TIME_RANGE') or ''
    top_k = int(cfg.get('top_k', os.getenv('SEARCH_S2T_TOPK', str(MAX_DOCS_FOR_RENDER))))
    timeout = float(cfg.get('timeout', os.getenv('SEARCH_TIMEOUT', '30')))
    connect_timeout = float(cfg.get('connect_timeout', os.getenv('SEARCH_CONNECT_TIMEOUT', '5')))
    max_total_chars = int(cfg.get('max_total_chars', os.getenv('SEARCH_S2T_MAX_TOTAL_CHARS', str(MAX_TOTAL_CHARS))))
    max_attempts = int(cfg.get('max_attempts', os.getenv('SEARCH_S2T_MAX_ATTEMPTS', str(MAX_SEARCH_ATTEMPTS))))
    max_connections = int(
        cfg.get('max_connections', os.getenv('SEARCH_S2T_MAX_CONNECTIONS', str(DEFAULT_MAX_CONNECTIONS)))
    )
    max_keepalive_connections = int(
        cfg.get(
            'max_keepalive_connections',
            os.getenv('SEARCH_S2T_MAX_KEEPALIVE_CONNECTIONS', str(DEFAULT_MAX_KEEPALIVE_CONNECTIONS)),
        )
    )
    keepalive_expiry = float(
        cfg.get('keepalive_expiry', os.getenv('SEARCH_S2T_KEEPALIVE_EXPIRY', str(DEFAULT_KEEPALIVE_EXPIRY)))
    )

    # ``query`` is a list of atomic sub-queries (sp.md web_search contract). Drop
    # empties, keep order; PromptSearch takes a prompt + a sub_query list, so we
    # use the first sub-query as the prompt and forward the full list as queries.
    # Pydantic exposes maxItems to the policy; the slice is an executor-side
    # defense in depth in case this function is called outside ToolExecutor.
    queries = [q.strip() for q in params.query if isinstance(q, str) and q.strip()][:MAX_SUBQUERIES_PER_CALL]
    if not queries:
        # Malformed call: no usable query. Feed the fixed empty text so the model
        # can still produce a final answer (mirrors agent_loop.py's empty branch).
        return ToolResponse(
            output=[{'type': 'text', 'text': EMPTY_SEARCH_TEXT}],
            meta={
                'success': False,
                'result_count': 0,
                'empty': True,
                'error': 'empty query list',
                'subquery_count': 0,
            },
        )
    prompt = queries[0]

    client = search_client_cls(url=search_url, source=search_source, ext_params=ext_params, time_range=time_range)
    if time_range:
        logger.debug('search_s2t_yb_glue_tool: applying time_range filter=%r', time_range)

    http_client = _get_pooled_http_client(
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
    )

    # API-error retry (agent_loop.py parity): retry only on transport/HTTP errors
    # (result.ok is False). A successful call that returns zero docs is NOT an API
    # error and is not retried — it falls through to the empty-render branch.
    result = None
    for attempt in range(1, max_attempts + 1):
        result = await client.call(http_client, prompt=prompt, queries=queries)
        if result.ok:
            break
        if attempt < max_attempts:
            await asyncio.sleep(0.3 * attempt)

    # All attempts failed: never surface the raw upstream error to the model —
    # feed the fixed empty text so it can answer from prior context.
    if result is None or not result.ok:
        http_status = result.http_status if result is not None else 0
        request_uuid = result.request_uuid if result is not None else ''
        error = (result.error if result is not None else '') or f'HTTP {http_status}'
        return ToolResponse(
            output=[{'type': 'text', 'text': EMPTY_SEARCH_TEXT}],
            meta={
                'success': False,
                'give_up': True,
                'empty': True,
                'result_count': 0,
                'http_status': http_status,
                'request_uuid': request_uuid,
                'error': error,
                'subquery_count': len(queries),
            },
        )

    docs = extract_web_docs(result.raw_text, max_docs=top_k)
    formatted = _render_web_search_result(docs, max_total_chars=max_total_chars, max_docs=MAX_DOCS_FOR_RENDER)

    return ToolResponse(
        output=[{'type': 'text', 'text': formatted}],
        meta={
            'success': True,
            'result_count': len(docs),
            'empty': len(docs) == 0,
            'latency_sec': result.latency_sec,
            'request_uuid': result.request_uuid,
            'subquery_count': len(queries),
        },
    )
