"""Bridge tools for client-side Tool Search.

When tool search is enabled, "deferred" tools are hidden from the model's main
tool schema. The model discovers and invokes them through three bridge tools:

* ``tool_search``   — fuzzy-search deferred tools by name/description.
* ``tool_describe`` — fetch the full input schema for one deferred tool.
* ``tool_call``     — invoke one deferred tool by its schema-visible name.

Each bridge tool reaches the deferred-tool registry through
``ToolContext.executor`` (a back-reference to the owning :class:`ToolExecutor`),
which the executor injects only for these LOCAL bridge tools.
"""

from __future__ import annotations

import json
import re

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import (
    ToolArgumentValidationError,
    ToolContext,
    ToolNameResolutionError,
    ToolResponse,
    ToolType,
    tool_registry,
)


class ToolSearchInput(BaseModel):
    query: str = Field(
        default='',
        description=(
            'Search query matched against tool names and descriptions. '
            'Space-separated words are ranked individually (e.g. "read file"). '
            'Prefix a word with "+" to require it (e.g. "+browser click" only '
            'returns tools matching "browser"). '
            'Use "select:<tool_name>" to fetch the full input schema for one '
            'tool by its exact name.'
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description='Maximum number of matches to return (1-10).',
    )


class ToolDescribeInput(BaseModel):
    tool_name: str = Field(description="A tool's exact name as returned in the 'name' field of a tool_search match.")


class ToolCallInput(BaseModel):
    tool_name: str = Field(
        description="A tool's exact name as returned in the 'name' field of a tool_search or tool_describe result."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description='Arguments object for the target tool, matching its input_schema (from tool_search/tool_describe).',
    )


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    """Wrap a JSON-serializable payload as a ToolResponse.

    The same payload is emitted both as deterministic text (sorted keys, so
    cached/replayed runs are stable) for the model and as structured ``meta``
    for programmatic consumers.
    """
    return ToolResponse(
        output=[{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
        meta=payload,
    )


def _error_response(message: str) -> ToolResponse:
    """Return a model-readable error so the loop can self-correct rather than abort."""
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Tool error: {message}'}],
        meta={'error': message},
    )


def _tokenize_name(name: str) -> list[str]:
    """Split a tool name into lowercase search tokens.

    Handles camelCase, acronym runs, and the ``/ _ -`` separators used in tool
    names so each word is independently searchable:

    * ``readFile``    -> ``['read', 'file']``    (camelCase boundary)
    * ``HTTPServer``  -> ``['http', 'server']``  (acronym boundary)
    * ``pkg/my_tool`` -> ``['pkg', 'my', 'tool']`` (separators)
    """
    # 1) lower/digit followed by an upper starts a new word: "readFile" -> "read File".
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # 2) an acronym run before an upper+lower starts a new word: "HTTPServer" -> "HTTP Server".
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return [part.lower() for part in re.split(r'[/_\-\s]+', spaced) if part]


def _query_terms(query: str) -> tuple[list[str], list[str]]:
    """Split a search query into (required, optional) terms.

    A term prefixed with ``+`` (e.g. ``+read``) is *required*: a tool must match
    it to be returned at all. All other whitespace-separated terms are
    *optional* and only contribute to ranking. A lone ``+`` is treated as a
    normal optional term.
    """
    required: list[str] = []
    optional: list[str] = []
    for raw in query.split():
        term = raw.strip().lower()
        if not term:
            continue
        if term.startswith('+') and len(term) > 1:
            required.append(term[1:])
        else:
            optional.append(term)
    return required, optional


def _score_tool(query: str, info: dict[str, Any]) -> int:
    """Rank a single tool against a query; higher is more relevant, 0 = excluded.

    Required (``+``-prefixed) terms act as a hard filter — any miss returns 0.
    Remaining terms accumulate a score weighted by where they match, strongest
    first: exact name (100) > name token (60) > name substring (30) >
    description word (10). An empty term list (e.g. a punctuation-only query)
    scores 1 so the tool is kept but unranked.
    """
    name = str(info['name']).lower()
    description = str(info.get('description') or '').lower()
    name_tokens = _tokenize_name(str(info['name']))
    token_text = ' '.join(name_tokens)
    required, optional = _query_terms(query)
    terms = required + optional

    def _matches_required(term: str) -> bool:
        # Match on word boundaries (whole name, a name token, or a description
        # word) rather than raw substring, so "+read" does not match an
        # unrelated name like "spreadsheet_export".
        if term == name:
            return True
        if re.search(rf'\b{re.escape(term)}\b', token_text):
            return True
        return bool(re.search(rf'\b{re.escape(term)}\b', description))

    if required and not all(_matches_required(term) for term in required):
        return 0

    if not terms:
        return 1

    score = 0
    for term in terms:
        if term == name:
            score += 100
        if term in name_tokens:
            score += 60
        if term in name and term != name:
            score += 30
        if re.search(rf'\b{re.escape(term)}\b', description):
            score += 10
    return score


def _get_executor(context: ToolContext) -> Any:
    """Return the executor back-reference, or raise if tool search isn't wired up.

    ``ToolContext.executor`` is populated only for bridge tools running under an
    executor with tool search enabled; its absence is a configuration error.
    """
    executor = context.executor
    if executor is None:
        raise RuntimeError('Tool Search bridge tools require ToolContext.executor')
    return executor


@tool_registry.register(
    'tool_search',
    meta={
        'name': 'tool_search',
        'description': (
            'Discover additional tools that are not listed directly. Search by keyword '
            'to find relevant tools, then invoke one with tool_call. '
            'Query tips: space-separated words are ranked (e.g. "read file"); prefix a '
            'word with "+" to require it. '
            'Returns {"matches": [{"name", "description", "source", ...}], "total"}; '
            'when exactly one tool matches, its full "input_schema" is included so you '
            'can call it right away. Otherwise use tool_describe to get a schema first.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def tool_search_tool(params: ToolSearchInput, context: ToolContext) -> ToolResponse:
    """Search deferred tools and return ranked matches.

    Query forms:

    * ``"select:<name>"`` — shortcut to describe one tool (full schema), as if
      the model had picked it from a prior search.
    * any other text      — fuzzy match via :func:`_score_tool`, best first.

    When the search narrows to exactly one tool, its full input schema is
    included so the model can call it without a separate ``tool_describe`` step.
    """
    try:
        executor = _get_executor(context)
    except RuntimeError as exc:
        return _error_response(str(exc))

    query = params.query.strip()
    include_schema = False

    # "select:<name>" — describe a specific tool directly (full schema).
    if query.startswith('select:'):
        selected_name = query.removeprefix('select:').strip()
        try:
            info = executor.describe_deferred_tool(selected_name)
        except ToolNameResolutionError as exc:
            return _error_response(exc.message_for_model)
        payload = {'matches': [info], 'total': 1}
        return _json_response(payload)

    infos = executor.get_deferred_tool_infos(include_schema=False)
    # Always score-and-rank: there is no "list everything" mode. The hidden-tool
    # catalog is surfaced to the model via the prompt reminder instead.
    scored = [(info, _score_tool(query, info)) for info in infos]
    scored = [(info, score) for info, score in scored if score > 0]
    scored.sort(key=lambda item: (-item[1], item[0]['name']))
    infos = [info for info, _ in scored]

    total = len(infos)
    matches = infos[: params.max_results]
    # Sole match: include its full schema so the model can call it immediately.
    if total == 1:
        matches = [executor.describe_deferred_tool(matches[0]['name'])]
        include_schema = True

    payload = {'matches': matches, 'total': total}
    if include_schema:
        payload['schema_included'] = True
    return _json_response(payload)


@tool_registry.register(
    'tool_describe',
    meta={
        'name': 'tool_describe',
        'description': (
            'Get the full input schema for one tool found via tool_search, before calling it. '
            'Returns {"name", "description", "input_schema", ...}. Pass the exact "name" from '
            'a tool_search match.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def tool_describe_tool(params: ToolDescribeInput, context: ToolContext) -> ToolResponse:
    """Return the full input schema for one deferred tool by its visible name."""
    try:
        executor = _get_executor(context)
    except RuntimeError as exc:
        return _error_response(str(exc))

    try:
        payload = executor.describe_deferred_tool(params.tool_name)
    except ToolNameResolutionError as exc:
        return _error_response(exc.message_for_model)
    return _json_response(payload)


@tool_registry.register(
    'tool_call',
    meta={
        'name': 'tool_call',
        'description': (
            'Invoke a tool discovered via tool_search/tool_describe. Pass the exact tool '
            'name and a params object matching its input_schema. Use this instead of calling '
            'the discovered tool directly, since it is not listed in the main tool set.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def tool_call_tool(params: ToolCallInput, context: ToolContext) -> ToolResponse:
    """Invoke one discovered deferred tool by its schema-visible name.

    Name-resolution and argument-validation errors are returned to the model as
    correctable text; any other error propagates to the caller unchanged.
    """
    executor = _get_executor(context)
    try:
        return await executor.call_discovered_tool(params.tool_name, params.params)
    except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
        return _error_response(exc.message_for_model)
