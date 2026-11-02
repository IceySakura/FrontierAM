"""Goal-oriented page visiting for search_stateful.

An alternative to the ``open``/``find`` paging loop. Those two return a ~1024-token slice of
the page and expect the model to scroll or search its way through a long document, which
costs several turns per page. ``visit`` instead fetches the page once and has a summariser
model pull out what matters for a stated goal, answering in a single turn.

Both styles share this module's fetch path, cache and leak filtering, and both register their
pages in the same browser state, so an agent may visit a page for the summary and then still
``find`` inside it. Pick a style per task by listing either ``search_stateful/visit`` or
``search_stateful/open`` + ``search_stateful/find`` in the agent's tool list.

By default, the summariser call reproduces the retired ``web_search_cache/visit_cache`` tool:
same gpt-eval endpoint, HMAC auth, prompt and ``jina_model*`` config keys. Setting
``tool_configs.search_stateful.summary_client`` to ``openai`` (or exporting
``SEARCH_STATEFUL_SUMMARY_CLIENT=openai``) switches only the summariser to AIHub 2.0 through
Nexus's existing OpenAI client.
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import logging
import os
import uuid

from pydantic import BaseModel, Field

from nexus.core import (
    OpenAICompatibleClient,
    OpenAIConfig,
    ToolContext,
    ToolResponse,
    ToolType,
    require_context,
    tool_registry,
)
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.utils.async_http import http_client

from .backend import BackendError, GoogleApiBackend
from .config import get_default_config
from .page_contents import PageContents
from .state import SIMPLE_BROWSER_STATE_LIST


logger = logging.getLogger(__name__)

API_VERSION = 'v2.03'
SUMMARY_URL = 'http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval'
SUMMARY_SOURCE = 'nexus_tools_search_visit'
SUMMARY_MAX_ATTEMPTS = 3
SUMMARY_TIMEOUT_SECONDS = 1200
SUMMARY_FAILURE_TEXT = '[visit] Failed to parse using llm.'
DEFAULT_SUMMARY_MODEL = 'api_doubao_DeepSeek-V3.1-250821'
DEFAULT_OPENAI_SUMMARY_MODEL = 'deepseek-v4-flash'
DEFAULT_OPENAI_SUMMARY_BASE_URL = 'http://api.aihub.woa.com/standard/v1'
WEBCONTENT_MAXLENGTH = 100000
BLOCKED_PAGE_MESSAGE = 'This page has been blocked!'

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format:
EVIDENCE:
<evidence>

SUMMARY:
<summary>
"""


class VisitInput(BaseModel):
    """Visit Tool Input Parameters"""

    url: str | list[str] = Field(
        description='The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.'
    )
    goal: str = Field(description='The goal of the visit for webpage(s).')


class SummaryConfig(BaseModel):
    """Credentials and limits for the summariser call."""

    model: str
    app_id: str
    app_key: str
    webcontent_maxlength: int = WEBCONTENT_MAXLENGTH


def _get_simple_auth(source: str, secret_id: str, secret_key: str) -> tuple[str, str]:
    """Build the HMAC-SHA1 auth header the gpt-eval endpoint expects."""
    date_time = datetime.datetime.now(datetime.timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = f'hmac id="{secret_id}", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = f'date: {date_time}\nsource: {source}'
    sign = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


def _extract_summary_content(result: dict) -> str:
    """Pull the message text out of a gpt-eval response.

    Providers behind this endpoint disagree on shape: deepseek nests under ``output``, doubao
    returns ``choices`` directly, and claude returns a ``content`` array.
    """
    response = result.get('request_detail', {}).get('response', {})

    if 'output' in response:
        return response['output']['choices'][0]['message']['content']
    if 'choices' in response:
        return response['choices'][0]['message']['content']
    if 'content' in response:
        return response['content'][0]['text']

    logger.warning('[visit] Unknown response format: %s', result)
    raise KeyError('Unknown response format')


async def _call_summary_model(messages: list, config: SummaryConfig) -> str:
    """Call the gpt-eval summariser, returning SUMMARY_FAILURE_TEXT when it cannot be reached."""
    sign, date_time = _get_simple_auth(SUMMARY_SOURCE, config.app_id, config.app_key)
    headers = {
        'Apiversion': API_VERSION,
        'Authorization': sign,
        'Date': date_time,
        'Source': SUMMARY_SOURCE,
    }
    payload = {
        'request_id': str(uuid.uuid4()),
        'model_marker': config.model,
        'messages': messages,
        'params': {'model': config.model, 'thinking': {'type': 'disabled'}, 'enable_thinking': False},
        'timeout': SUMMARY_TIMEOUT_SECONDS,
    }

    try:
        response = await http_client.post(SUMMARY_URL, headers=headers, json=payload, timeout=SUMMARY_TIMEOUT_SECONDS)
        return _extract_summary_content(response.json())
    except Exception as e:
        logger.warning('[visit] Summary call failed for model=%s: %s', config.model, e)
        return SUMMARY_FAILURE_TEXT


def _get_openai_summary_client(tool_config: dict) -> OpenAICompatibleClient | None:
    """Return or lazily create the optional OpenAI summary client.

    A fully configured ``openai_clients.summary`` still takes precedence. HyTuner
    datasets can instead opt in with ``tool_configs.search_stateful.summary_client:
    openai`` or ``SEARCH_STATEFUL_SUMMARY_CLIENT=openai``; this keeps all
    AIHub-specific defaults inside search_stateful and avoids changes to either
    runner. The per-dataset setting takes precedence over the environment.
    """
    try:
        execution_context = require_context()
        return execution_context.get_openai_client('summary')
    except KeyError:
        pass
    except RuntimeError:
        execution_context = None

    client_name = tool_config.get('summary_client')
    if client_name is None:
        client_name = os.environ.get('SEARCH_STATEFUL_SUMMARY_CLIENT')
    if not client_name:
        return None
    if client_name != 'openai':
        raise ValueError(f'Unsupported search_stateful summary_client: {client_name!r}')

    api_key = tool_config.get('jina_model_app_key') or os.environ.get('DEFAULT_JINA_MODEL_APP_KEY', '')
    if not api_key:
        raise ValueError(
            'Missing AIHub 2.0 summary API key. Set jina_model_app_key under '
            'tool_configs.search_stateful or DEFAULT_JINA_MODEL_APP_KEY.'
        )

    model = tool_config.get('jina_model') or os.environ.get('DEFAULT_JINA_MODEL') or DEFAULT_OPENAI_SUMMARY_MODEL
    base_url = (
        tool_config.get('summary_base_url')
        or os.environ.get('SEARCH_STATEFUL_SUMMARY_BASE_URL')
        or DEFAULT_OPENAI_SUMMARY_BASE_URL
    )
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': api_key,
                'base_url': base_url,
                'max_retries': SUMMARY_MAX_ATTEMPTS,
                'timeout': SUMMARY_TIMEOUT_SECONDS,
            },
            request_args={
                'model': model,
                'extra_body': {'thinking': {'type': 'disabled'}},
            },
        )
    )
    client._client_key = 'summary'
    if execution_context is not None:
        # Reuse one SDK client for every visit call in this trajectory.
        client = execution_context.openai_clients.setdefault('summary', client)
    return client


async def _call_openai_summary_model(messages: list[dict], client: OpenAICompatibleClient) -> str:
    """Call the named OpenAI-compatible summary client without legacy fallback."""
    try:
        response = await client.create_completion(messages=messages)
        if not response.choices:
            raise ValueError('Summary model returned no choices')
        content = response.choices[0].message.content
        if not content:
            raise ValueError('Summary model returned empty content')
        return content
    except Exception as e:
        logger.warning('[visit] OpenAI summary call failed: %s', e)
        return SUMMARY_FAILURE_TEXT


async def _summarize(
    page: PageContents,
    goal: str,
    config: SummaryConfig,
    openai_client: OpenAICompatibleClient | None = None,
) -> str:
    """Summarise a page, retrying the model a few times before giving up."""
    prompt = EXTRACTOR_PROMPT.format(webpage_content=page.text[: config.webcontent_maxlength], goal=goal)
    if openai_client is not None:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        # Keep the legacy wire format byte-compatible with gpt-eval data_eval.
        messages = [{'role': 'user', 'content': [{'value': prompt, 'type': 'text'}]}]

    for attempt in range(SUMMARY_MAX_ATTEMPTS):
        if openai_client is not None:
            raw = await _call_openai_summary_model(messages, openai_client)
        else:
            raw = await _call_summary_model(messages, config)
        if raw and raw != SUMMARY_FAILURE_TEXT:
            return f'The useful information in {page.url} for user goal {goal} as follows: \n\n{raw}'
        logger.info('[visit] Summary attempt %d/%d failed for %s', attempt + 1, SUMMARY_MAX_ATTEMPTS, page.url)

    return f'{SUMMARY_FAILURE_TEXT} URL: {page.url}'


def _build_backend(tool_config: dict) -> GoogleApiBackend:
    """Construct a backend for fetching pages, reading config the way open and search do."""
    default_config = get_default_config()
    return GoogleApiBackend(
        source='google api',
        url_block_list_file=tool_config.get('url_block_list_file', default_config['url_block_list_file']),
        user=tool_config.get('user', default_config.get('user', '')),
        apikey=tool_config.get('apikey', default_config.get('apikey', '')),
        jina_provider=tool_config.get('jina_provider', default_config.get('jina_provider', 'aws_jina')),
        use_cache=tool_config.get('use_cache', default_config.get('use_cache', True)),
        search_cache_path=tool_config.get('search_cache_path', default_config.get('search_cache_path')),
        visit_cache_path=tool_config.get('visit_cache_path', default_config.get('visit_cache_path')),
    )


def _build_summary_config(tool_config: dict) -> SummaryConfig:
    """Read summariser settings.

    The summariser is a service of its own, distinct from search and page fetching, so it may
    run on a separate app. Key names and env vars follow web_search's ``jina_model*`` -- a
    misnomer inherited from the tool this one replaces, where fetching and summarising shared a
    module. Unset, they fall back to the pair the rest of search_stateful uses.
    """
    default_config = get_default_config()
    return SummaryConfig(
        model=(tool_config.get('jina_model') or os.environ.get('DEFAULT_JINA_MODEL') or DEFAULT_SUMMARY_MODEL),
        app_id=(
            tool_config.get('jina_model_app_id')
            or os.environ.get('DEFAULT_JINA_MODEL_APP_ID')
            or tool_config.get('user')
            or default_config.get('user', '')
        ),
        app_key=(
            tool_config.get('jina_model_app_key')
            or os.environ.get('DEFAULT_JINA_MODEL_APP_KEY')
            or tool_config.get('apikey')
            or default_config.get('apikey', '')
        ),
        webcontent_maxlength=int(tool_config.get('webcontent_maxlength', WEBCONTENT_MAXLENGTH)),
    )


def _load_url_block_list(file_path: str) -> list[str]:
    """Read the URL block list, tolerating a missing or unreadable file."""
    if not file_path:
        return []
    try:
        with open(file_path, encoding='utf-8') as file:
            return [line.strip() for line in file.read().split('\n') if line.strip()]
    except Exception as e:
        logger.warning('Failed to read URL block list: %s', e)
        return []


def _is_url_blocked(url: str, url_block_list: list[str]) -> bool:
    return any(blocked in url for blocked in url_block_list)


async def _fetch_and_filter(
    url: str, backend: GoogleApiBackend, browser_state, url_block_list: list[str]
) -> PageContents:
    """Fetch a page, refusing blocked URLs and blanking pages that leak the question."""
    # NOTE(albertyi): the backend only applies the block list to search hits, never to fetch,
    # so a directly supplied URL would otherwise bypass it. visit_cache checked here too.
    if _is_url_blocked(url, url_block_list):
        logger.warning('URL blocked by block-list: %s', url)
        return PageContents(url=url, text=BLOCKED_PAGE_MESSAGE, title='Blocked', urls={})

    page = await backend.fetch(url)

    # NOTE(albertyi): blanking the text before summarising keeps a leaking page away from the
    # summariser, which would otherwise launder the answer into the summary.
    if browser_state.check_fetch_page(page):
        logger.warning('Page blocked due to question leak detection: %s', url)
        page.text = BLOCKED_PAGE_MESSAGE

    return page


def resolve_summariser(tool_config: dict) -> tuple[SummaryConfig, OpenAICompatibleClient | None]:
    """Return the summariser settings, raising ValueError when no usable credentials are set."""
    summary_config = _build_summary_config(tool_config)
    openai_summary_client = _get_openai_summary_client(tool_config)
    if openai_summary_client is None and (not summary_config.app_id or not summary_config.app_key):
        raise ValueError(
            'Missing summariser credentials. Set jina_model_app_id and jina_model_app_key '
            'under tool_configs.search_stateful (or DEFAULT_JINA_MODEL_APP_ID / '
            'DEFAULT_JINA_MODEL_APP_KEY), or set the user / apikey pair they fall back to.'
        )
    return summary_config, openai_summary_client


async def visit_pages(
    urls: list[str],
    goal: str,
    backend: GoogleApiBackend,
    browser_state,
    url_block_list: list[str],
    summary_config: SummaryConfig,
    openai_summary_client: OpenAICompatibleClient | None = None,
) -> tuple[str, bool]:
    """Fetch and summarise pages, returning the combined text and whether any page succeeded.

    Shared by the native ``visit`` tool and the stdio MCP server so both render identical output.
    """
    fetched = await asyncio.gather(
        *(_fetch_and_filter(url, backend, browser_state, url_block_list) for url in urls), return_exceptions=True
    )

    # Register pages in request order so the cursor ids an agent sees are deterministic.
    pages: list[PageContents | BaseException] = []
    for url, result in zip(urls, fetched, strict=True):
        if isinstance(result, BaseException):
            logger.error('Fetch failed for %s: %s', url, result)
            pages.append(result)
            continue
        browser_state.add_page(result)
        pages.append(result)

    async def _render(url: str, page: PageContents | BaseException) -> str:
        if isinstance(page, BackendError):
            return f'Failed to fetch URL {url}: {page}'
        if isinstance(page, BaseException):
            return f'Error fetching {url}: {page!r}'
        if page.text == BLOCKED_PAGE_MESSAGE:
            return f'{BLOCKED_PAGE_MESSAGE} URL: {url}'
        return await _summarize(page, goal, summary_config, openai_summary_client)

    texts = await asyncio.gather(*(_render(url, page) for url, page in zip(urls, pages, strict=True)))
    combined = '\n=======\n'.join(texts)
    success = any(
        not text.startswith(('Failed to fetch URL', 'Error fetching', SUMMARY_FAILURE_TEXT)) for text in texts
    )
    return combined, success


@tool_registry.register(
    'search_stateful/visit',
    meta={
        'name': 'visit',
        'description': (
            'Visit webpage(s) and return the summary of the content. '
            'Use this tool to extract relevant information from web pages based on a specific goal.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def visit_tool(params: VisitInput, context: ToolContext) -> ToolResponse:
    """Visit webpage(s) and return goal-oriented summaries."""
    goal = params.goal
    if not goal or not goal.strip():
        return ToolResponse(output=[{'type': 'text', 'text': 'Error: Empty goal provided.'}], meta={'success': False})

    request_id = context.run_id
    if not request_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No run_id available in context. Cannot maintain state.'}],
            meta={'success': False},
        )

    tool_config = context.tool_configs.get('search_stateful', {})
    try:
        summary_config, openai_summary_client = resolve_summariser(tool_config)
    except ValueError as e:
        return ToolResponse(output=[{'type': 'text', 'text': f'Error: {e}'}], meta={'success': False})

    browser_state = await SIMPLE_BROWSER_STATE_LIST.get_browser_state(request_id)
    if not browser_state:
        default_config = get_default_config()
        origin_question = context.tool_state.get('origin_question', '')
        if not origin_question:
            logger.warning(
                "tool_state['origin_question'] is unset; question-leak detection is inactive. "
                'The task should seed it before running the agent.'
            )
        state_config = {
            'testset_name_to_filter_file': tool_config.get(
                'testset_name_to_filter_file', default_config['testset_name_to_filter_file']
            ),
            'url_block_list_file': tool_config.get('url_block_list_file', default_config['url_block_list_file']),
        }
        browser_state = await SIMPLE_BROWSER_STATE_LIST.add_browser_state(request_id, origin_question, state_config)

    urls = [params.url] if isinstance(params.url, str) else list(params.url)
    backend = _build_backend(tool_config)
    url_block_list = _load_url_block_list(
        tool_config.get('url_block_list_file', get_default_config()['url_block_list_file'])
    )

    combined, success = await visit_pages(
        urls, goal, backend, browser_state, url_block_list, summary_config, openai_summary_client
    )
    return ToolResponse(output=[{'type': 'text', 'text': combined}], meta={'success': success})
