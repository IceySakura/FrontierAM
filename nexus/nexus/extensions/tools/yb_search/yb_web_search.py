"""Internal Yuanbao text-search adapter."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.utils.rate_limiter import get_qps_limiter, get_rate_limiter

from ._config_utils import as_bool, as_float, as_int
from ._yb_utils import get_current_time_str, get_service_discovery, get_shared_httpx_client, make_request_timeout


logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = 'Production'
_DEFAULT_SERVICE = 'trpc.tsbs.vr_mcp_server.SearchHttp'
_DEFAULT_PATH = '/trpc.tsbs.vr_mcp_server.VrMcpServerHttp/PromptSearch'

_AUTHORITY_LABELS = {
    -1: '权威性中等',
    0: '权威性极低',
    1: '权威性较低',
    2: '权威性中等',
    3: '权威性较高',
    4: '权威性极高',
    5: '权威性很高',
}


def _value(config: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    value = config.get(key)
    if value is not None and value != '':
        return value
    value = os.environ.get(env_name)
    return value if value not in (None, '') else default


def _env_value(primary: str, legacy: str, default: Any) -> Any:
    value = os.environ.get(primary)
    return value if value not in (None, '') else os.environ.get(legacy, default)


def _positive(name: str, value: int | float) -> int | float:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f'{name} must be > 0, got {value}')
    return value


def _non_negative(name: str, value: int | float) -> int | float:
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}')
    return value


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    aliases = {
        'service_namespace': 'namespace',
        'service_name': 'service',
        'path': 'service_path',
        'max_concurrent': 'concurrency',
        'timeout': 'request_timeout',
        'retries': 'retry_time',
        'max_queries': 'max_query_num',
    }
    for canonical, alias in aliases.items():
        if canonical not in config and alias in config:
            config[canonical] = config[alias]
    ca_bundle = _value(config, 'ca_bundle', 'YB_WEB_SEARCH_CA_BUNDLE', '')
    insecure = as_bool(_value(config, 'insecure', 'YB_WEB_SEARCH_INSECURE', False))
    max_docs_value = config.get('max_docs', config.get('max_doc_num'))
    if max_docs_value is None:
        max_docs_value = os.environ.get('YB_WEB_SEARCH_MAX_DOCS', 5)
    max_queries_value = _value(config, 'max_queries', 'YB_WEB_SEARCH_MAX_QUERIES', 0)
    max_concurrent = as_int(
        _value(
            config,
            'max_concurrent',
            'YB_WEB_SEARCH_MAX_CONCURRENT',
            _env_value('YB_WEB_SEARCH_MAX_CONCURRENT', 'YB_WEB_SEARCH_RATE_LIMIT', 2),
        ),
        2,
    )
    timeout = as_float(_value(config, 'timeout', 'YB_WEB_SEARCH_TIMEOUT', 60.0), 60.0)
    hard_timeout = as_float(_value(config, 'hard_timeout', 'YB_WEB_SEARCH_HTTP_HARD_TIMEOUT', 65.0), 65.0)
    connect_timeout = as_float(_value(config, 'connect_timeout', 'YB_WEB_SEARCH_CONNECT_TIMEOUT', 10.0), 10.0)
    retries = as_int(_value(config, 'retries', 'YB_WEB_SEARCH_RETRIES', 3), 3)
    max_docs = as_int(max_docs_value, 5)
    max_queries = as_int(max_queries_value, 0)
    discovery_ttl = as_float(_value(config, 'discovery_ttl', 'YB_WEB_SEARCH_DISCOVERY_TTL', 60.0), 60.0)
    return {
        'namespace': str(_value(config, 'service_namespace', 'YB_WEB_SEARCH_NAMESPACE', _DEFAULT_NAMESPACE)),
        'service': str(_value(config, 'service_name', 'YB_WEB_SEARCH_SERVICE', _DEFAULT_SERVICE)),
        'path': str(_value(config, 'path', 'YB_WEB_SEARCH_PATH', _DEFAULT_PATH)),
        'discovery_ttl': _non_negative('discovery_ttl', discovery_ttl),
        'qps': _non_negative(
            'qps',
            as_float(_value(config, 'qps', 'YB_WEB_SEARCH_QPS', 1.0), 1.0),
        ),
        'max_concurrent': _positive('max_concurrent', max_concurrent),
        'timeout': _positive('timeout', timeout),
        'hard_timeout': _positive('hard_timeout', hard_timeout),
        'connect_timeout': _positive('connect_timeout', connect_timeout),
        'retries': _positive('retries', retries),
        'max_docs': _positive('max_docs', max_docs),
        'max_queries': _non_negative('max_queries', max_queries),
        'search_type': as_int(_value(config, 'search_type', 'YB_WEB_SEARCH_TYPE', 0), 0),
        'proxy': _value(config, 'proxy', 'YB_WEB_SEARCH_HTTP_PROXY', '') or None,
        'verify': False if insecure else (str(ca_bundle) if ca_bundle else True),
    }


def _format_doc(doc: dict[str, Any], index: int) -> str:
    content = doc.get('content')
    content = str(content).strip('"') if content else str(doc.get('abstract') or '').strip('"')[:10000]
    authority = _AUTHORITY_LABELS.get(doc.get('doc_authority_level', 2), '权威性中等')
    if doc.get('invisible') == 1:
        location = '[web_doc_displayable]False'
    else:
        location = f'[web_doc_site]{doc.get("site_name", "")}\n[web_doc_url]{doc.get("url", "")}'
    return (
        f'[web_doc {index} begin]\n'
        f'[search query]{doc.get("recall_query", "")}\n'
        f'[web_doc_title]{doc.get("title", "")}\n'
        f'{location}\n'
        f'[web_doc_authority_level]{authority}\n'
        f'[web_doc_publish_time]{doc.get("publish_time", "")}\n'
        f'[web_doc_content begin]{content}\n'
        '[web_doc_content end]\n'
        f'[web_doc {index} end]'
    )


async def _prompt_search(prompt: str, query_list: list[str], config: dict[str, Any]) -> dict[str, Any] | None:
    discovery = get_service_discovery(config['namespace'], config['service'], config['discovery_ttl'])
    limiter_name = f'yb_web_search:{config["namespace"]}:{config["service"]}:{config["qps"]}:{config["max_concurrent"]}'
    concurrency_limiter = get_rate_limiter(limiter_name, config['max_concurrent'])
    qps_limiter = get_qps_limiter(limiter_name, config['qps']) if config['qps'] > 0 else nullcontext()
    client = get_shared_httpx_client(
        limiter_name,
        timeout=make_request_timeout(config['timeout'], config['connect_timeout']),
        max_connections=config['max_concurrent'] + 1,
        max_keepalive_connections=config['max_concurrent'] + 1,
        proxy=config['proxy'],
        verify=config['verify'],
    )
    request_id = str(uuid.uuid4())
    payload = {
        'header': {
            'session_id': f'sid-{request_id}',
            'uuid': request_id,
            'trace_id': f'tid-{request_id}',
            'yb_traffic_type': 1,
        },
        'source': 'vlm-agent-rl',
        'prompt': prompt,
        'sub_query_info': [{'subq': query, 'search_type': config['search_type']} for query in query_list],
        'params': {
            'ext_params': {
                'open_google_api': '0',
                'open_wx_video': '0',
                'filter_level': '0',
                'enable_full_content': '1',
            }
        },
    }

    retries = max(config['retries'], 1)
    for attempt in range(retries):
        try:
            url = discovery.get_url(config['path'])
            async with concurrency_limiter, qps_limiter:
                response = await asyncio.wait_for(
                    client.post(url, headers={'Content-Type': 'application/json'}, json=payload),
                    timeout=config['hard_timeout'],
                )
            response.raise_for_status()
            result = response.json()
            return result if isinstance(result, dict) else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            discovery.invalidate()
            logger.warning(
                'YB web search request failed (queries=%d, attempt=%d/%d, error=%s)',
                len(query_list),
                attempt + 1,
                retries,
                type(exc).__name__,
            )
            if attempt + 1 < retries:
                await asyncio.sleep(min(2**attempt, 4))
    return None


_WEB_SEARCH_DESCRIPTION = (
    '文本搜索工具，基于子任务查询query列表，在数据库中对每个query基于语义执行全文检索，'
    '每个query可以返回多条相关的文本结果，为大模型提供参考信息，适用于信息检索、内容丰富等场景。'
)
_WEB_SEARCH_QUERY_DESCRIPTION = (
    '用于文本检索的子任务查询query列表，每个query应明确表达单一意图。\n'
    '- 查询query应使用通用、搜索引擎友好且易于检索的关键词，以提高搜索效率和结果相关性。'
    '避免直接使用用户问题中难以有效检索的短语。'
    '例如：避免使用相对的术语，如"今天"或"明年"——应改用精确的日期，如"2025年9月3日"\n'
    '- 文本搜索工具只承接简单的基于相关性匹配的文档返回，不应包含任何需要进一步思考、筛选需求，'
    '例如：查询query中避免使用"第三个作者"这样的term，因为"第三个作者"一般不会直接出现在'
    '检索库的文档中，应改用"作者列表"这样更通用的表述\n'
    '- 查询query应保持简洁，去除不必要的辅助词、连词和疑问词。主要使用关键词组合（用空格分隔）。'
    '理想情况下不应超过15个字。避免使用特殊标点符号'
)


class YbWebSearchInput(BaseModel):
    query: list[str] | str = Field(..., description=_WEB_SEARCH_QUERY_DESCRIPTION)


@tool_registry.register(
    'yb_web_search',
    meta={'name': 'web_search', 'description': _WEB_SEARCH_DESCRIPTION, 'tool_type': ToolType.LOCAL},
)
async def yb_web_search_tool(params: YbWebSearchInput, context: ToolContext) -> ToolResponse:
    now = get_current_time_str()
    raw_queries = params.query if isinstance(params.query, list) else [params.query]
    query_list = [str(query) for query in raw_queries if query is not None and str(query) != '']
    if not query_list:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'当前时间：{now}\nError: Empty query list.'}],
            meta={'success': False, 'error': 'empty_query', 'doc_segments': []},
        )

    config = _runtime_config(dict(context.tool_configs.get('yb_web_search', {}) or {}))
    if config['max_queries'] > 0:
        query_list = query_list[: config['max_queries']]
    user_question = context.tool_state.get('yb_search.user_question', '')
    prompt = user_question if isinstance(user_question, str) else ''

    try:
        result = await _prompt_search(prompt, query_list, config)
    except Exception as exc:
        logger.error('[yb_web_search] prompt_search raised: %s', exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'当前时间：{now}\nWeb search failed: {exc}'}],
            meta={'success': False, 'error': str(exc), 'doc_segments': []},
        )
    if result is None:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': (f'当前时间：{now}\nWeb search failed: backend returned None (network or service error).'),
                }
            ],
            meta={'success': False, 'error': 'backend_none', 'doc_segments': []},
        )

    segments: list[str] = []
    seen: set[tuple[str, str]] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for doc in result.get('docs', []) or []:
        if not isinstance(doc, dict):
            continue
        recall_query = str(doc.get('recall_query', ''))
        signature = (str(doc.get('url', '')), str(doc.get('title', '')))
        if counts[recall_query] >= config['max_docs'] or signature in seen:
            continue
        counts[recall_query] += 1
        seen.add(signature)
        segments.append(_format_doc(doc, len(segments) + 1))

    no_resp_queries = [query for query in query_list if counts.get(query, 0) == 0]
    header = [f'当前时间：{now}']
    if no_resp_queries:
        joined = '、'.join(f'"{query}"' for query in no_resp_queries)
        header.append(f'使用web_search检索工具， {joined}未能检索到相关信息；')
        if segments:
            header.append('通过其它search query获取的检索信息如下：')
    rendered = '\n'.join(header)
    if segments:
        rendered = f'{rendered}\n{chr(10).join(segments)}'
    return ToolResponse(
        output=[{'type': 'text', 'text': rendered}],
        meta={
            'success': True,
            'count': len(segments),
            'doc_segments': segments,
            'no_resp_queries': no_resp_queries,
        },
    )
