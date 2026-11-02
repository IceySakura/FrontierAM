"""Claude Code style ``WebFetch`` tool.

The Claude Code WebFetch tool definition is the contract presented to
the LLM. The schema keeps ``url`` and ``prompt`` required while the tool
layer handles validation and summary-model extraction.

Internally, the tool delegates the network call to
:meth:`CCSearchBackend.fetch` (which only knows about ``url`` \u2014 it is
the atomic fetch primitive) and handles the ``prompt`` argument inside
the tool layer.

Summary-model extraction prefers the backend's HMAC summary model
(configured via ``jina_model`` / ``jina_model_app_id`` / ``jina_model_app_key``
in ``tool_configs`` or ``DEFAULT_JINA_MODEL*`` env vars), falling back to the
``summary`` OpenAI client when no HMAC credentials are configured.
"""

import asyncio
import logging

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry

from .backend import BackendError, CCSearchBackend, MissingCredentialsError, SummaryContextLengthError


logger = logging.getLogger(__name__)


# Registry key (matches ``[tool.nexus.tools]`` in pyproject.toml). Used by the
# agent to request this tool by id, by ``tool_configs`` lookups, and by
# ``CCSearchBackend.from_context``.
WEB_FETCH_TOOL_REGISTRY_NAME = 'cc_search_agent_whitebox/web_fetch'
# Public ``meta['name']`` exposed to the LLM. Kept as the Claude Code name
# (no slashes) so prompts and trajectories stay aligned with the black-box
# CC search agent.
WEB_FETCH_TOOL_NAME = 'WebFetch'
MAX_URL_LENGTH = 2000
_TRUNCATE_SUFFIX = '\n\n[Content truncated due to length...]'

WEB_FETCH_DESCRIPTION = (
    'IMPORTANT: WebFetch WILL FAIL for authenticated or private URLs (e.g. Google Docs, Confluence, Jira, GitHub '
    'private repos). Skip this tool for those.\n'
    '\n'
    '- Fetches content from a specified URL and processes it using an AI model\n'
    '- Takes a URL and a prompt as input\n'
    '- Fetches the URL content, converts HTML to markdown\n'
    '- Processes the content with the prompt using a small, fast model\n'
    "- Returns the model's response about the content\n"
    '- Use this tool when you need to retrieve and analyze web content\n'
    '\n'
    'Usage notes:\n'
    '  - The URL must be a fully-formed valid URL\n'
    '  - The prompt should describe what information you want to extract from the page\n'
    '  - Results may be summarized if the content is very large\n'
    '  - For GitHub URLs, prefer using the gh CLI via Bash instead (e.g., gh pr view, gh issue view, gh api).'
)


class WebFetchInput(BaseModel):
    model_config = {'extra': 'forbid'}

    url: str = Field(description='The URL to fetch content from')
    prompt: str = Field(description='The prompt to run on the fetched content')

    @model_validator(mode='before')
    @classmethod
    def _normalize_missing_required_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if not isinstance(normalized.get('url'), str):
                normalized['url'] = ''
            if not isinstance(normalized.get('prompt'), str):
                normalized['prompt'] = ''
            return normalized
        return data


def _validate_url(url: str) -> str | None:
    """Validate and normalize URL. Returns an error message, or None when valid."""
    if len(url) > MAX_URL_LENGTH:
        return f'URL too long ({len(url)} > {MAX_URL_LENGTH} chars).'
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.hostname:
            return f'Invalid URL "{url}".'
        if parsed.username or parsed.password:
            return f'Invalid URL "{url}". URLs with username/password are not allowed.'
        hostname = parsed.hostname
        if hostname is None or len(hostname.split('.')) < 2:
            return f'Invalid hostname in URL "{url}".'
    except Exception:
        return f'Invalid URL "{url}". The URL provided could not be parsed.'
    return None


def _truncate_content(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + _TRUNCATE_SUFFIX


def _make_secondary_model_prompt(markdown_content: str, prompt: str) -> str:
    """Build the prompt used by the summary model, matching Claude WebFetch."""
    guidelines = (
        'Provide a concise response based only on the content above. In your response:\n'
        ' - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is '
        'ok as long as we respect the license.\n'
        ' - Use quotation marks for exact language from articles; any language outside of the quotation should '
        'never be word-for-word the same.\n'
        ' - You are not a lawyer and never comment on the legality of your own prompts and responses.\n'
        ' - Never produce or reproduce exact song lyrics.'
    )
    return f'\nWeb page content:\n---\n{markdown_content}\n---\n\n{prompt}\n\n{guidelines}\n'


def _truncate_for_attempt(content: str, attempt: int, chars_per_attempt: int) -> str:
    """Drop ``attempt * chars_per_attempt`` chars from the tail of ``content``.

    Attempt 0 keeps the page intact; each subsequent attempt shrinks it so a
    repeated context-length error eventually fits. Mirrors ``visit.py``'s
    ``context_truncate_chars_per_attempt`` knob.
    """
    drop = attempt * chars_per_attempt
    if drop <= 0 or len(content) <= drop:
        return content
    return content[:-drop] + _TRUNCATE_SUFFIX


async def _apply_prompt_to_markdown(
    prompt: str,
    markdown_content: str,
    *,
    backend: CCSearchBackend,
    max_attempts: int = 1,
    chars_per_attempt: int = 0,
) -> str:
    """Run the summary model with progressive truncation + exponential backoff.

    Prefers the backend's HMAC summary model (configured via ``jina_model`` /
    ``jina_model_app_id`` / ``jina_model_app_key``); falls back to the ``summary``
    OpenAI client when no HMAC credentials are configured. Each retry shrinks the
    page via :func:`_truncate_for_attempt` and waits ``2 ** (attempt + 1)`` seconds
    before the next call. ``max_attempts=1`` reproduces a single-shot call
    (no retry, no truncation).
    """
    use_backend = backend.has_summary_models
    client = None if use_backend else require_context().get_openai_client('summary')
    for attempt in range(max_attempts):
        page = _truncate_for_attempt(markdown_content, attempt, chars_per_attempt)
        try:
            if client is None:
                messages = [
                    {'role': 'user', 'content': [{'value': _make_secondary_model_prompt(page, prompt), 'type': 'text'}]}
                ]
                content = await backend.call_summary_model(messages, attempt=attempt)
            else:
                response = await client.create_completion(
                    messages=[{'role': 'user', 'content': _make_secondary_model_prompt(page, prompt)}]
                )
                content = response.choices[0].message.content if response.choices else None
        except SummaryContextLengthError:
            if attempt == max_attempts - 1:
                raise
            logger.warning(
                '[WebFetch] summary context length exceeded on attempt %d/%d, truncating', attempt + 1, max_attempts
            )
            await asyncio.sleep(2 ** (attempt + 1))
            continue
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            logger.warning('[WebFetch] summary attempt %d/%d failed: %s', attempt + 1, max_attempts, exc)
            await asyncio.sleep(2 ** (attempt + 1))
            continue
        if content:
            if attempt > 0:
                logger.debug('[WebFetch] summary succeeded after %d retries', attempt)
            return content
        logger.warning('[WebFetch] summary model returned empty on attempt %d/%d', attempt + 1, max_attempts)
        return 'No response from model'
    raise RuntimeError('_apply_prompt_to_markdown requires max_attempts >= 1')


@tool_registry.register(
    WEB_FETCH_TOOL_REGISTRY_NAME,
    meta={
        'name': WEB_FETCH_TOOL_NAME,
        'description': WEB_FETCH_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def web_fetch_tool(params: WebFetchInput, context: ToolContext) -> ToolResponse:
    """Claude Code style WebFetch - fetch via backend, then summarize with prompt."""
    url = params.url.strip()
    prompt = params.prompt.strip()
    logger.debug('[WebFetch] called: url=%s prompt=%.100s', url, prompt)

    missing_fields = []
    if not url:
        missing_fields.append('url')
    if not prompt:
        missing_fields.append('prompt')
    if missing_fields:
        fields = ', '.join(f'"{field}"' for field in missing_fields)
        logger.warning('[WebFetch] rejected: missing required parameter(s): %s', fields)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: Missing required parameter(s): {fields}'}],
            meta={'success': False},
        )

    if url_error := _validate_url(url):
        logger.warning('[WebFetch] invalid URL: %s', url_error)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: {url_error}'}],
            meta={'success': False},
        )

    try:
        backend = CCSearchBackend.from_context(context)
        result = await backend.fetch(url)
    except MissingCredentialsError as exc:
        logger.error('[WebFetch] missing credentials: %s', exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: {exc}'}],
            meta={'success': False},
        )
    except BackendError as exc:
        logger.error('[WebFetch] backend error for url=%s: %s', url, exc)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error fetching {url}: {exc}'}],
            meta={'success': False},
        )

    content = _truncate_content(result.content, backend.webcontent_maxlength)
    if len(content) < len(result.content):
        logger.debug('[WebFetch] content truncated: %d -> %d chars', len(result.content), len(content))

    if result.success:
        try:
            text = await _apply_prompt_to_markdown(
                prompt,
                content,
                backend=backend,
                max_attempts=backend.summary_max_attempts,
                chars_per_attempt=backend.context_truncate_chars_per_attempt,
            )
        except Exception as exc:
            logger.error('[WebFetch] summary model failed for url=%s: %s', url, exc)
            return ToolResponse(
                output=[{'type': 'text', 'text': f'Error processing {url} with summary model: {exc}'}],
                meta={'success': False, 'url': url, 'title': result.title, 'prompt': prompt},
            )
        logger.debug('[WebFetch] ok | url=%s title=%s response_len=%d', url, result.title, len(text))
    else:
        text = content

    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={'success': result.success, 'url': url, 'title': result.title, 'prompt': prompt},
    )


__all__ = [
    'MAX_URL_LENGTH',
    'WEB_FETCH_DESCRIPTION',
    'WEB_FETCH_TOOL_NAME',
    'WEB_FETCH_TOOL_REGISTRY_NAME',
    'WebFetchInput',
    '_apply_prompt_to_markdown',
    '_make_secondary_model_prompt',
    '_truncate_for_attempt',
    'web_fetch_tool',
]
