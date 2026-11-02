"""LeanSearch MCP server backed by the internal GPU search service.

This standalone server exposes one read-only ``leansearch`` tool over MCP's
stdio transport.  It follows the request and response format used by
``lean-lsp-mcp`` and uses the official Python MCP SDK for protocol handling.

LeanSearch is described in "A Semantic Search Engine for Mathlib4":
https://arxiv.org/abs/2403.13310

Environment variables:
  LEANSEARCH_URL      Search endpoint. Defaults to the Polaris GPU service.
  LEANSEARCH_TIMEOUT  Per-request timeout in seconds. Defaults to 30.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field


DEFAULT_LEANSEARCH_URL = 'http://leansearch-gpu.production.polaris:8080/search'
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_REQUEST_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0

logger = logging.getLogger(__name__)


class LeanSearchResult(BaseModel):
    """One ranked Mathlib declaration returned by LeanSearch."""

    name: str = Field(description='Fully qualified Lean declaration name.')
    module_name: str = Field(description='Mathlib module in which the declaration is defined.')
    kind: str | None = Field(default=None, description='Declaration kind, such as theorem, lemma, or definition.')
    type: str | None = Field(default=None, description='Lean type signature / formal theorem statement.')


class LeanSearchResults(BaseModel):
    """Structured LeanSearch response."""

    items: list[LeanSearchResult] = Field(description='Ranked Mathlib declarations, most relevant first.')


TOOL_DESCRIPTION = """Semantically search Mathlib4 declarations with LeanSearch.

Use this read-only retrieval tool when you know the mathematical meaning of a
needed theorem or definition but not its exact Lean name, namespace, module, or
type signature. Unlike literal name/substring search, LeanSearch embeds the
query and Mathlib declarations and ranks them by semantic similarity, so the
query may be:

- an informal mathematical description, such as "sum of two even numbers is
  even" or "Cauchy-Schwarz inequality";
- a Lean proposition/type or goal fragment, such as
  "{f : A → B} (hf : Function.Injective f) : ∃ g, Function.LeftInverse g f";
- a short mixed Lean/natural-language phrase when some relevant identifiers are
  already known.

Results are ordered by estimated relevance and include the fully qualified
declaration name, defining module, declaration kind, and Lean type signature.
Use the returned declarations as candidate premises or imports, especially
after an "unknown identifier/constant" error. Search for reusable lemmas rather
than asking for a complete proof; if results are weak, reformulate the query or
split a complex goal into its key mathematical facts. Semantic relevance does
not guarantee that a result applies in the current context or Mathlib version,
and this tool does not elaborate or verify Lean code, so confirm candidates
with the local environment or a Lean proof checker before relying on them."""


mcp = FastMCP(
    name='leansearch',
    instructions=(
        'Use the leansearch tool to retrieve semantically relevant Mathlib4 declarations. '
        'Verify returned candidates with Lean before using them in a proof.'
    ),
)


def _timeout_seconds() -> float:
    raw = os.environ.get('LEANSEARCH_TIMEOUT')
    if raw is None or not raw.strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            'Invalid LEANSEARCH_TIMEOUT=%r; using %.1f seconds',
            raw,
            DEFAULT_TIMEOUT_SECONDS,
        )
        return DEFAULT_TIMEOUT_SECONDS
    if timeout <= 0:
        logger.warning(
            'LEANSEARCH_TIMEOUT must be positive; using %.1f seconds',
            DEFAULT_TIMEOUT_SECONDS,
        )
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """Send one blocking JSON request; called in a worker thread."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'nexus-leansearch-mcp/0.1',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode('utf-8', errors='replace').strip()
        suffix = f': {detail}' if detail else ''
        raise RuntimeError(f'LeanSearch returned HTTP {exc.code}{suffix}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Could not reach LeanSearch at {url}: {exc.reason}') from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError('LeanSearch returned invalid JSON') from exc


async def _post_json_with_retries(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """Call LeanSearch up to three times, counting the initial attempt."""
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(_post_json, url, payload, timeout)
        except Exception as exc:
            if attempt == MAX_REQUEST_ATTEMPTS:
                logger.error(
                    'LeanSearch request failed after %d attempts: %s',
                    MAX_REQUEST_ATTEMPTS,
                    exc,
                )
                raise

            delay = RETRY_DELAY_SECONDS * attempt
            logger.warning(
                'LeanSearch request attempt %d/%d failed: %s; retrying in %.1fs',
                attempt,
                MAX_REQUEST_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError('LeanSearch retry loop exited unexpectedly')


def _dotted_name(value: Any) -> str:
    """Convert the backend's list-of-name-parts representation to Lean syntax."""
    if isinstance(value, list):
        return '.'.join(str(part) for part in value)
    if isinstance(value, str):
        return value
    return ''


def _parse_results(raw: Any, num_results: int) -> LeanSearchResults:
    """Validate and normalize LeanSearch's batched response."""
    if not isinstance(raw, list):
        raise RuntimeError(f'Unexpected LeanSearch response type: {type(raw).__name__}')
    if not raw or raw[0] is None:
        return LeanSearchResults(items=[])
    if not isinstance(raw[0], list):
        raise RuntimeError('Unexpected LeanSearch response: first query result is not a list')

    items: list[LeanSearchResult] = []
    for entry in raw[0][:num_results]:
        if not isinstance(entry, dict):
            continue
        result = entry.get('result', entry)
        if not isinstance(result, dict):
            continue

        name = _dotted_name(result.get('name'))
        module_name = _dotted_name(result.get('module_name'))
        if not name:
            logger.warning('Skipping LeanSearch result without a declaration name: %r', result)
            continue
        items.append(
            LeanSearchResult(
                name=name,
                module_name=module_name,
                kind=result.get('kind'),
                type=result.get('type'),
            )
        )
    return LeanSearchResults(items=items)


@mcp.tool(
    name='leansearch',
    title='LeanSearch for Mathlib4',
    description=TOOL_DESCRIPTION,
    annotations=ToolAnnotations(
        title='LeanSearch for Mathlib4',
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
    structured_output=True,
)
async def leansearch(
    query: Annotated[
        str,
        Field(
            min_length=1,
            description='Natural-language, Lean, or mixed query describing the declaration to find.',
        ),
    ],
    num_results: Annotated[
        int,
        Field(ge=1, description='Maximum number of ranked declarations to return.'),
    ] = 5,
) -> LeanSearchResults:
    """Run one semantic Mathlib search against the configured GPU backend."""
    query = query.strip()
    if not query:
        raise ValueError('query must contain non-whitespace characters')

    url = os.environ.get('LEANSEARCH_URL', DEFAULT_LEANSEARCH_URL).strip()
    if not url:
        raise ValueError('LEANSEARCH_URL must not be empty')

    # This matches leansearch.net's batched API used by lean-lsp-mcp.
    payload = {'num_results': str(num_results), 'query': [query]}
    raw = await _post_json_with_retries(url, payload, _timeout_seconds())
    return _parse_results(raw, num_results)


if __name__ == '__main__':
    mcp.run(transport='stdio')
