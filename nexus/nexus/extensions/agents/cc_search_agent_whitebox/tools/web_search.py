"""Claude Code style ``WebSearch`` tool.

Thin wrapper around :meth:`CCSearchBackend.search` that handles input
validation, mutually-exclusive ``allowed_domains`` / ``blocked_domains``
checks and translates backend errors into ``ToolResponse`` messages.
"""

import json
import logging
import time

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .backend import BackendError, CCSearchBackend, MissingCredentialsError, SearchHit


logger = logging.getLogger(__name__)


# Registry key (matches ``[tool.nexus.tools]`` in pyproject.toml). Used by the
# agent to request this tool by id, by ``tool_configs`` lookups, and by
# ``CCSearchBackend.from_context``.
WEB_SEARCH_TOOL_REGISTRY_NAME = 'cc_search_agent_whitebox/web_search'
# Public ``meta['name']`` exposed to the LLM. Kept as the Claude Code name
# (no slashes) so prompts and trajectories stay aligned with the black-box
# CC search agent.
WEB_SEARCH_TOOL_NAME = 'WebSearch'


def _current_month_year() -> str:
    """Return the local month/year phrase used in the Claude WebSearch prompt."""
    return datetime.now().strftime('%B %Y')


WEB_SEARCH_DESCRIPTION = f"""
- Allows the agent to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks
- Use this tool for accessing information beyond the model knowledge cutoff

Usage notes:
  - Domain filtering is supported to include or block specific websites

IMPORTANT - Use the correct year in search queries:
  - The current month is {_current_month_year()}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation" with the current year, NOT last year
""".strip()


class WebSearchInput(BaseModel):
    model_config = {'extra': 'forbid'}

    query: str = Field(description='The search query to use.')
    allowed_domains: list[str] | None = Field(
        default=None,
        description='Only include search results from these domains',
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description='Never include search results from these domains',
    )

    @model_validator(mode='before')
    @classmethod
    def _normalize_missing_required_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if not isinstance(normalized.get('query'), str):
                normalized['query'] = ''
            return normalized
        return data


def _hit_to_link(hit: SearchHit) -> dict[str, Any]:
    """Flatten a :class:`SearchHit` into the JSON-serialisable link dict.

    Includes ``snippet`` / ``date`` / ``source`` whenever Serper returned them
    so the LLM has enough context to answer without an immediate WebFetch
    follow-up. Empty / ``None`` fields are dropped to keep the payload
    compact and to avoid noisy ``"date": null`` entries in the JSON.
    """
    link: dict[str, Any] = {'title': hit.title, 'url': hit.url}
    if hit.snippet:
        link['snippet'] = hit.snippet
    if hit.date:
        link['date'] = hit.date
    if hit.source:
        link['source'] = hit.source
    return link


def _format_search_output(query: str, result_rendered: str, links: list[dict[str, Any]]) -> str:
    """Map backend search hits to Claude WebSearch tool_result text."""
    formatted = f'Web search results for query: "{query}"\n\n'
    if links:
        formatted += f'Links: {json.dumps(links, ensure_ascii=False)}'
    elif result_rendered:
        formatted += result_rendered
    else:
        formatted += 'No links found.'
    return formatted.strip()


@tool_registry.register(
    WEB_SEARCH_TOOL_REGISTRY_NAME,
    meta={
        'name': WEB_SEARCH_TOOL_NAME,
        'description': WEB_SEARCH_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def web_search_tool(params: WebSearchInput, context: ToolContext) -> ToolResponse:
    """Claude Code style WebSearch — thin wrapper around ``CCSearchBackend.search``."""
    query = params.query.strip()
    logger.debug('[WebSearch] called: query=%s domains=%s', query, params.allowed_domains or params.blocked_domains)

    if not query:
        logger.warning('[WebSearch] rejected: missing query')
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Missing required parameter "query"'}],
            meta={'success': False},
        )

    if params.allowed_domains and params.blocked_domains:
        logger.warning('[WebSearch] rejected: both allowed_domains and blocked_domains specified')
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': 'Error: Cannot specify both allowed_domains and blocked_domains in the same request',
                }
            ],
            meta={'success': False},
        )

    start = time.monotonic()
    try:
        backend = CCSearchBackend.from_context(context)
        result = await backend.search(
            query,
            allowed_domains=params.allowed_domains,
            blocked_domains=params.blocked_domains,
        )
    except MissingCredentialsError as exc:
        logger.error('[WebSearch] missing credentials: %s', exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: {exc}'}],
            meta={'success': False},
        )
    except BackendError as exc:
        duration_seconds = time.monotonic() - start
        logger.error('[WebSearch] backend error after %.2fs: query=%s, error=%s', duration_seconds, query, exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error during web search: {exc}'}],
            meta={'success': False},
        )

    duration_seconds = time.monotonic() - start
    links = [_hit_to_link(hit) for hit in result.hits if hit.title and hit.url]
    text = _format_search_output(query, result.rendered, links)

    logger.debug('[WebSearch] ok | query=%s results=%d duration=%.2fs', query, len(links), duration_seconds)

    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={
            'success': result.success,
            'query': query,
            'duration_seconds': duration_seconds,
            'result_count': len(links),
        },
    )


__all__ = [
    'WEB_SEARCH_DESCRIPTION',
    'WEB_SEARCH_TOOL_NAME',
    'WEB_SEARCH_TOOL_REGISTRY_NAME',
    'WebSearchInput',
    '_format_search_output',
    '_hit_to_link',
    'web_search_tool',
]
