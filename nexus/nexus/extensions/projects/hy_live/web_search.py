"""Yuanbao PromptSearch behind the cerebellum's ``web_search`` tool.

The request payload is this module's; the doc extraction, block rendering, limits, empty text and failure
text that fix the tool result the cerebellum reads are hylive's shared cerebellum rules. The endpoint needs
no credentials.
"""

from __future__ import annotations

import logging
import uuid

from typing import Any, Literal, Protocol, runtime_checkable

import httpx

from hylive.cerebellum_rules import (
    EMPTY_TEXT,
    FAILURE_PREFIX,
    MAX_DOCS,
    MAX_TOTAL_CHARS,
    NOT_CONNECTED_TEXT,
    TRUNCATE_TAIL,
    extract_docs,
    latest_user_text,
    render_docs,
)


logger = logging.getLogger(__name__)

SEARCH_URL = 'https://vr-mcp-server.woa.com/trpc.tsbs.vr_mcp_server.VrMcpServerHttp/PromptSearch'
SEARCH_SOURCE = 'agentic_search'
EXT_PARAMS = {
    'open_google_api': '0',
    'open_wx_video': '0',
    'filter_level': '0',
    'enable_full_content': '1',
}
SEARCH_ATTEMPTS = 3

WebSearchProvider = Literal['none', 'yuanbao']
"""Serializable choice of search backend; ``none`` leaves the tool answering ``NOT_CONNECTED_TEXT``."""


@runtime_checkable
class WebSearchBackend(Protocol):
    async def search(self, raw_query: Any, *, prompt: str = '') -> str:
        """Return the text the cerebellum reads for one ``web_search`` call.

        Args:
            raw_query (Any): The tool's ``query`` argument as the model sent it, normally a list of strings.
            prompt (str): The user utterance that led to the search; empty means the first query.

        Returns:
            The rendered doc blocks, or ``EMPTY_TEXT`` when nothing usable came back.

        Raises:
            Exception: The search failed; the caller reports it as ``搜索失败：{error}``.
        """
        ...


def queries_of(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def search_payload(prompt: str, queries: list[str]) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    return {
        'header': {
            'session_id': request_id,
            'uuid': request_id,
            'trace_id': request_id,
            'yb_traffic_type': 0,
        },
        'prompt': prompt,
        'source': SEARCH_SOURCE,
        'sub_query_info': [{'subq': query, 'search_type': 0} for query in queries],
        'messages': [{'role': 'user', 'content': prompt}],
        'params': {'ext_params': EXT_PARAMS},
    }


class YuanbaoWebSearch:
    """Query Yuanbao PromptSearch: three attempts, no wait between them."""

    def __init__(
        self,
        *,
        url: str = SEARCH_URL,
        timeout: httpx.Timeout | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout or httpx.Timeout(60.0, connect=5.0)
        self.transport = transport

    async def search(self, raw_query: Any, *, prompt: str = '') -> str:
        queries = queries_of(raw_query)
        if not queries:
            raise ValueError('web_search: query 不能为空')
        prompt = prompt.strip() or queries[0]
        last_error = ''
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(1, SEARCH_ATTEMPTS + 1):
                try:
                    response = await client.post(self.url, json=search_payload(prompt, queries))
                    response.raise_for_status()
                except Exception as exc:
                    last_error = str(exc).strip() or exc.__class__.__name__
                    logger.warning('Yuanbao search attempt %s/%s failed: %s', attempt, SEARCH_ATTEMPTS, last_error)
                    continue
                text = render_docs(extract_docs(response.text))
                logger.info('Yuanbao search queries=%s returned %d chars', queries, len(text))
                return text
        # NOTE(wallyshi): the caller adds the prefix again, so the model reads "搜索失败：搜索失败：...";
        # the doubled prefix is part of the tool result format.
        raise RuntimeError(f'{FAILURE_PREFIX}{last_error}')


def build_web_search_backend(provider: WebSearchProvider) -> WebSearchBackend | None:
    if provider == 'yuanbao':
        return YuanbaoWebSearch()
    return None


__all__ = [
    'EMPTY_TEXT',
    'FAILURE_PREFIX',
    'MAX_DOCS',
    'MAX_TOTAL_CHARS',
    'NOT_CONNECTED_TEXT',
    'SEARCH_URL',
    'TRUNCATE_TAIL',
    'WebSearchBackend',
    'WebSearchProvider',
    'YuanbaoWebSearch',
    'build_web_search_backend',
    'extract_docs',
    'latest_user_text',
    'queries_of',
    'render_docs',
    'search_payload',
]
