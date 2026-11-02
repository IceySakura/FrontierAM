"""
开发信息: DrakosXiao(2025.12)
工具名称: 网页访问工具
工具类型: LOCAL
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import random
import re
import traceback
import uuid

from typing import NamedTuple

from pydantic import BaseModel, Field

from nexus.core import (
    ContextLengthExceededError,
    OpenAICompatibleClient,
    OpenAIConfig,
    ToolContext,
    ToolResponse,
    ToolType,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.base.errors import (
    NonRetryableProviderError,
    raise_for_non_retryable_provider_error,
)
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.extensions.tools.web_search._config_utils import as_float as _as_float
from nexus.extensions.tools.web_search._config_utils import as_int as _as_int
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

# ============== 常量定义 ==============
API_VERSION = 'v2.03'
BASE_URL = os.environ.get('DEFAULT_JINA_URL', 'http://trpc-gpt-eval.production.polaris:8080/api/jina_ai/search')
PROVIDER = os.environ.get('DEFAULT_JINA_PROVIDER', 'aws_jina')
VISIT_ERROR_TAG = '[visit error]'
VISIT_LLM_ERROR_TAG = '[visit llm error]'
VISIT_LLM_UNKNOWN_FORMAT_TAG = '[visit unknown response format]'
VISIT_CONTEXT_LENGTH_ERROR_TAG = '[visit context length exceeded]'
VISIT_FAILURE_MARKERS = (VISIT_ERROR_TAG, VISIT_LLM_ERROR_TAG, VISIT_LLM_UNKNOWN_FORMAT_TAG, '[visit] Failed')
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
JINA_FALLBACK_HEADER_TEMPLATE = """{llm_error_tag} LLM extraction failed after {max_attempts} attempts. \
Falling back to raw webpage content ({returned_length}/{original_length} characters shown).

## User Goal
{goal}

## Raw Webpage Content from {url}
"""
WEBCONTENT_MAXLENGTH = 200000
CONTEXT_TRUNCATE_CHARS_PER_ATTEMPT = 50000
WEBCONTENT_LENGTH_SETTINGS = {
    'webcontent_maxlength': WEBCONTENT_MAXLENGTH,
    'context_truncate_chars_per_attempt': CONTEXT_TRUNCATE_CHARS_PER_ATTEMPT,
}
JINA_FALLBACK_CONTENT_LENGTH = 5000
CONTEXT_LENGTH_ERROR_PATTERNS = (
    "requested token count exceeds the model's maximum context length",
    "longer than the model's context length",
    'maximum input length',
    'maximum context length',
    'exceeds the maximum length',
    'context_length_exceeded',
    'input_tokens',
)

SUMMARY_MAX_ATTEMPTS = 3

# AIHub 2.0 summariser. Opting in routes only the content-extraction model off the
# data_eval HMAC gateway, whose per-model platform limit returns HTTP 429, onto AIHub's
# OpenAI-compatible chat endpoint. Page fetching still goes through jina_readpage.
SUMMARY_CLIENT_KEY = 'summary'
SUPPORTED_SUMMARY_CLIENTS = ('openai',)
# NOTE(jaykaizhang): AIHub 2.0 exposes two endpoints whose model catalogues differ. This
# default matches search_stateful, but deepseek-v4.1-flash is served only by
# compatible-mode/v1, which answers standard/v1 requests with UserModelNotFound(2004) and
# additionally requires ?provider= on the bearer key. Datasets using those models must
# point VISIT_SUMMARY_BASE_URL at http://api.aihub.woa.com/compatible-mode/v1.
DEFAULT_SUMMARY_BASE_URL = 'http://api.aihub.woa.com/standard/v1'
DEFAULT_SUMMARY_TIMEOUT_SECONDS = 300.0
# Output cap for the extraction call. EXTRACTOR_PROMPT asks for the full original context as
# evidence, so a long page legitimately needs a long completion; too low a cap truncates the
# evidence block mid-sentence instead of failing loudly.
AIHUB_V2_MAX_OUTPUT_TOKENS = 40960
# 'none' is the OpenAI-standard "no reasoning" value and is valid on deepseek-v4.1-flash,
# which otherwise defaults to 'high'. Models speaking a different dialect (hy3 wants
# 'no_think') override it through VISIT_SUMMARY_REASONING_EFFORT.
DEFAULT_SUMMARY_REASONING_EFFORT = 'none'


class SummaryModelConfig(NamedTuple):
    model: str
    app_id: str
    app_key: str


# ============== 辅助函数 ==============
def get_simple_auth(source: str, secret_id: str, secret_key: str) -> tuple[str, str]:
    """生成 HMAC 认证头"""
    date_time = datetime.datetime.now(datetime.timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = f'hmac id="{secret_id}", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = f'date: {date_time}\nsource: {source}'
    sign = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


def get_openai_summary_client(tool_config: dict) -> OpenAICompatibleClient | None:
    """Return the AIHub 2.0 summary client, or None when the data_eval path stays in use.

    A runner-provided ``openai_clients.summary`` always wins. Otherwise a dataset opts in
    with ``tool_configs.visit.summary_client: openai`` or ``VISIT_SUMMARY_CLIENT=openai``,
    which keeps the AIHub defaults here instead of spreading them across runner configs.
    The per-dataset setting takes precedence over the environment.

    Args:
        tool_config (dict): The ``visit`` entry of ``context.tool_configs``.

    Returns:
        A summary client when the OpenAI summariser is enabled, otherwise None.

    Raises:
        ValueError: If an unsupported client name or no bearer key is configured.
    """
    # NOTE(jaykaizhang): bind before the try so every failure path leaves it defined. Only
    # get_openai_client is expected to raise KeyError, after the assignment has run, but
    # relying on that makes the later `if execution_context is not None` one refactor away
    # from a NameError.
    execution_context = None
    try:
        execution_context = require_context()
        return execution_context.get_openai_client(SUMMARY_CLIENT_KEY)
    except KeyError:
        # A context exists but carries no 'summary' client, so build and cache one below.
        pass
    except RuntimeError:
        # No active context, or a RUNTIME tool: build a client that nothing can cache.
        execution_context = None

    client_name = tool_config.get('summary_client') or os.environ.get('VISIT_SUMMARY_CLIENT', '')
    if not client_name:
        return None
    if client_name not in SUPPORTED_SUMMARY_CLIENTS:
        raise ValueError(
            f'Unsupported visit summary_client: {client_name!r}. Choose one of {list(SUPPORTED_SUMMARY_CLIENTS)}'
        )

    api_key = tool_config.get('jina_model_app_key') or os.environ.get('DEFAULT_JINA_MODEL_APP_KEY', '')
    if not api_key:
        raise ValueError(
            'Missing AIHub 2.0 summary API key. Set jina_model_app_key under tool_configs.visit '
            'or DEFAULT_JINA_MODEL_APP_KEY.'
        )

    provider = tool_config.get('summary_provider') or os.environ.get('VISIT_SUMMARY_PROVIDER', '')
    if provider and '?provider=' not in api_key:
        # NOTE(jaykaizhang): AIHub 2.0 selects the upstream vendor from the credential itself,
        # not from a query param on the request, so the suffix belongs on the bearer key.
        api_key = f'{api_key}?provider={provider}'

    base_url = (
        tool_config.get('summary_base_url') or os.environ.get('VISIT_SUMMARY_BASE_URL') or DEFAULT_SUMMARY_BASE_URL
    )
    timeout = _as_float(
        tool_config.get('summary_timeout') or os.environ.get('VISIT_SUMMARY_TIMEOUT'),
        DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    )
    max_output_tokens = _as_int(
        tool_config.get('summary_max_output_tokens') or os.environ.get('AIHUB_V2_MAX_OUTPUT_TOKENS'),
        AIHUB_V2_MAX_OUTPUT_TOKENS,
    )
    # Reasoning is pure waste for extraction, but the two vendors spell "off" differently:
    # deepseek-v4.1-flash takes reasoning_effort='none' (and defaults to 'high'), while hy3
    # only accepts 'no_think' and rejects the standard values. Both speak the standard
    # reasoning_effort field, so stay on it rather than sending a vendor extra_body, and let
    # a dataset override the value its model documents. Set the knob empty to send nothing.
    reasoning_effort = tool_config.get('summary_reasoning_effort')
    if reasoning_effort is None:
        reasoning_effort = os.environ.get('VISIT_SUMMARY_REASONING_EFFORT', DEFAULT_SUMMARY_REASONING_EFFORT)
    request_args: dict = {'max_tokens': max_output_tokens}
    if reasoning_effort:
        request_args['reasoning_effort'] = reasoning_effort
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': api_key,
                'base_url': base_url,
                'max_retries': SUMMARY_MAX_ATTEMPTS,
                'timeout': timeout,
            },
            # NOTE(jaykaizhang): model is deliberately absent. OpenAIClient merges request_args
            # over the per-call model argument, which would pin every attempt to one model and
            # defeat the model rotation readpage performs across retries.
            request_args=request_args,
        )
    )
    client._client_key = SUMMARY_CLIENT_KEY
    if execution_context is not None:
        # Reuse one SDK client, and its connection pool, for every visit in this trajectory.
        client = execution_context.openai_clients.setdefault(SUMMARY_CLIENT_KEY, client)
    return client


def _flatten_summary_messages(messages: list) -> list[dict]:
    """Convert data_eval message content into the plain strings the chat wire expects."""
    flattened = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            text = '\n'.join(str(part.get('value', part.get('text', ''))) for part in content if isinstance(part, dict))
        else:
            text = str(content or '')
        flattened.append({'role': message.get('role', 'user'), 'content': text})
    return flattened


async def _call_openai_summary_model(messages: list, model: str, client: OpenAICompatibleClient) -> str:
    """Summarise through the AIHub 2.0 client, mapping failures onto the visit error tags.

    Makes a single call: the client retries transient errors itself, and readpage owns the
    outer loop that rotates models and truncates the page.
    """
    try:
        response = await client.create_completion(messages=_flatten_summary_messages(messages), model=model)
    except NonRetryableProviderError:
        logger.error(
            f'{VISIT_LLM_ERROR_TAG} summary(aihub) stopped on terminal platform error for model={model}',
            exc_info=True,
        )
        raise
    except ContextLengthExceededError:
        logger.warning(f'{VISIT_CONTEXT_LENGTH_ERROR_TAG} summary(aihub) context too long for model={model}')
        return VISIT_CONTEXT_LENGTH_ERROR_TAG
    except Exception:
        error_msg = traceback.format_exc()
        if any(pattern in error_msg.lower() for pattern in CONTEXT_LENGTH_ERROR_PATTERNS):
            return VISIT_CONTEXT_LENGTH_ERROR_TAG
        raise_for_non_retryable_provider_error(error_msg)
        logger.error(f'{VISIT_LLM_ERROR_TAG} summary(aihub) failed for model={model}: {error_msg[:800]}')
        return VISIT_LLM_ERROR_TAG

    if not response.choices:
        logger.error(f'{VISIT_LLM_UNKNOWN_FORMAT_TAG} summary(aihub) returned no choices for model={model}')
        return VISIT_LLM_UNKNOWN_FORMAT_TAG
    content = response.choices[0].message.content
    if not content:
        logger.error(f'{VISIT_LLM_UNKNOWN_FORMAT_TAG} summary(aihub) returned empty content for model={model}')
        return VISIT_LLM_UNKNOWN_FORMAT_TAG
    return content


async def call_llm_server(
    messages: list,
    app_id: str,
    app_key: str,
    model: str,
    max_tries: int = 3,
    summary_client: OpenAICompatibleClient | None = None,
) -> str | dict:
    """调用 LLM 服务进行内容解析"""
    # AIHub 2.0 summariser when the caller resolved one, otherwise the data_eval HMAC
    # gateway. app_id and app_key are unused on the AIHub path, which authenticates with
    # the bearer key already held by the client.
    if summary_client is not None:
        return await _call_openai_summary_model(messages, model, summary_client)

    source = 'nexus_tools_search_visit'
    sign, date_time = get_simple_auth(source, app_id, app_key)
    headers = {
        'Apiversion': API_VERSION,
        'Authorization': sign,
        'Date': date_time,
        'Source': source,
    }
    base_url = 'http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval'
    params = {'model': model, 'thinking': {'type': 'disabled'}, 'enable_thinking': False}
    if 'gpt' in model.lower():
        params = {
            'model': model,
            'max_completion_tokens': 8192,
        }
        # Add cost-saving parameters for GPT-5 models
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
            response = await http_client.post(base_url, headers=headers, json=data, timeout=1800)
            if response.status_code != 200:
                logger.warning(
                    f'[visit] LLM server returned HTTP {response.status_code} for model={model}: {response.text[:500]}'
                )
                raise_for_non_retryable_provider_error(response.text)
                raise ValueError(f'LLM server returned HTTP {response.status_code}')
            result = response.json()

            if not isinstance(result, dict):
                logger.warning(f'[visit] Unexpected LLM response type: {type(result).__name__}, result={result!r}')
                raise TypeError('LLM response is not a dict')

            request_detail = result.get('request_detail') or {}
            if not isinstance(request_detail, dict):
                logger.warning(f'[visit] Unexpected request_detail type: {type(request_detail).__name__}')
                request_detail = {}

            resp = request_detail.get('response') or {}
            if not isinstance(resp, dict):
                logger.warning(f'[visit] Unexpected response payload type: {type(resp).__name__}, resp={resp!r}')
                raise TypeError('LLM response payload is not a dict')

            if 'output' in resp:  # 格式1: deepseek-v3.2 (带output层级)
                return resp['output']['choices'][0]['message']['content']

            if 'choices' in resp:  # 格式2: doubao-seed-1.6 (直接choices)
                return resp['choices'][0]['message']['content']

            if 'content' in resp:  # 格式3: claude-sonnet-4.5 (content数组)
                return resp['content'][0]['text']

            logger.error(f'{VISIT_LLM_UNKNOWN_FORMAT_TAG} model={model}: Unknown response format: {result}')
            if attempt == max_tries - 1:
                return VISIT_LLM_UNKNOWN_FORMAT_TAG
        except NonRetryableProviderError:
            logger.error(
                f'{VISIT_LLM_ERROR_TAG} LLM parsing stopped on terminal platform error for model={model}', exc_info=True
            )
            raise
        except Exception:
            error_msg = traceback.format_exc()
            if response is not None:
                error_msg += f', {response.text=}'
            if any(pattern in error_msg.lower() for pattern in CONTEXT_LENGTH_ERROR_PATTERNS):
                return VISIT_CONTEXT_LENGTH_ERROR_TAG
            error_details = {'error': error_msg, 'response': response}
            logger.error(
                f'{VISIT_LLM_ERROR_TAG} LLM parsing attempt {attempt + 1}/{max_tries} failed for model={model}: {error_details}'
            )
            if attempt == max_tries - 1:
                return VISIT_LLM_ERROR_TAG
    return VISIT_LLM_ERROR_TAG


def _mask_auth_bearer(content: str) -> str:
    """Mask sensitive information in bearer tokens for logging"""

    def _mask_token(token: str) -> str:
        if len(token) <= 8:
            return '****'
        return token[:4] + '****' + token[-4:]

    pattern = re.compile(
        r'(Bearer\s+)(?P<app_id>[^:]+):(?P<app_key>[^?]+)(\?provider=(?P<provider>.+))?', re.IGNORECASE
    )
    m = pattern.match(content)
    if m:
        app_id = m.group('app_id')
        app_key = m.group('app_key')
        provider = m.group('provider') or 'unknown'
        masked_app_id = _mask_token(app_id)
        masked_app_key = _mask_token(app_key)
        return f'Bearer {masked_app_id}:{masked_app_key}?provider={provider}'
    return content


def _build_jina_fallback_content(url: str, goal: str, content: str, max_attempts: int) -> str:
    """Build a truncated raw-content fallback when LLM extraction keeps failing"""
    original_length = len(content)
    excerpt = content[:JINA_FALLBACK_CONTENT_LENGTH]
    returned_length = len(excerpt)
    remaining = original_length - returned_length
    truncated_notice = f'\n\n[Content truncated. {remaining} more characters not shown.]' if remaining > 0 else ''
    header = JINA_FALLBACK_HEADER_TEMPLATE.format(
        llm_error_tag=VISIT_LLM_ERROR_TAG,
        url=url,
        goal=goal,
        max_attempts=max_attempts,
        returned_length=returned_length,
        original_length=original_length,
    )
    return f'{header}{excerpt}{truncated_notice}\n'


def _extract_reader_failure_reason(response_text: str) -> str:
    """Extract a short readable reason from the reader service response."""
    reason = ''
    try:
        err = json.loads(response_text)
        if isinstance(err, dict):
            reason = err.get('readableMessage') or err.get('message') or ''
    except Exception:
        reason = response_text or ''
    return reason[:200].strip()


def _is_huggingface_dataset_or_space_url(url: str) -> bool:
    """Check whether the URL points to a Hugging Face dataset or space page."""
    if not url:
        return False
    return 'huggingface.co/datasets' in url or 'huggingface.co/spaces' in url


async def jina_readpage(
    url: str,
    app_id: str,
    app_key: str,
    provider: str,
    base_url: str,
    use_cache: bool = False,
    cache_dir: str | None = None,
) -> str:
    """使用 Jina 服务读取网页内容"""
    jina_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {app_id}:{app_key}?provider={provider}',
    }
    max_retries = 5
    retry_delay = 10
    timeout = 1800
    logger.debug(
        f'[visit] Calling Jina readpage with url={url}, provider={provider}, auth={_mask_auth_bearer(jina_headers["Authorization"])}, base_url={base_url}'
    )

    response = None
    for attempt in range(max_retries):
        try:
            response = await http_client.post(base_url, json={'url': url}, headers=jina_headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            raise_for_non_retryable_provider_error(response.text)
            # 400/403/404 多为 URL 或访问权限问题，不用重试
            # 422 常见于 unexpected content type application/xml，也通常与网址格式有关
            if response.status_code in [400, 403, 404, 422]:
                logger.error(
                    f'[visit] jina_readpage got status {response.status_code} for {url}: {response.text[:1000]}'
                )
                # 返回一句话，让大模型可以判断这个访问失败的原因，改变访问策略
                status_hints = {
                    400: 'The URL may be malformed or contain invalid characters. Try verifying and correcting the URL format.',
                    403: 'Access to this page is forbidden (likely behind a paywall, login wall, or anti-bot protection). Try finding the same information from an alternative source.',
                    404: 'This page does not exist or has been removed. The URL may be outdated or contain a typo. Try searching for updated links.',
                    422: (
                        'The reader service could not process this page. '
                        f'Reason: {_extract_reader_failure_reason(response.text) or "unavailable"}. '
                        'Possible causes include unsupported content type, timeout, '
                        'SSL error, or anti-bot protection. Try finding an HTML version of this content or an alternative source.'
                    ),
                }
                hint = status_hints.get(response.status_code, 'An unexpected client error occurred.')
                return f'{VISIT_ERROR_TAG} Failed to read page (HTTP {response.status_code}). {hint}'
            raise ValueError(response.text)
        except NonRetryableProviderError:
            logger.error(f'{VISIT_ERROR_TAG} Jina readpage stopped on terminal platform error for {url}', exc_info=True)
            raise
        except Exception:
            error_msg = traceback.format_exc()
            if response is not None:
                error_msg += f', {response.text=}'
            error_details = {'url': url, 'error_msg': error_msg, 'response': response}
            logger.warning(
                f'{VISIT_ERROR_TAG} Jina readpage attempt {attempt + 1}/{max_retries} failed, will retry in {retry_delay} seconds: {error_details}'
            )
            if attempt == max_retries - 1:
                return f'{VISIT_ERROR_TAG} An unexpected client error occurred. response={response.text if response else None}'
            await asyncio.sleep(retry_delay)

    return f'{VISIT_ERROR_TAG} Failed to read page.'


async def readpage(
    url: str,
    goal: str,
    app_id: str,
    app_key: str,
    provider: str,
    summary_models: list[SummaryModelConfig],
    base_url: str,
    webcontent_maxlength: int = WEBCONTENT_MAXLENGTH,
    context_truncate_chars_per_attempt: int = CONTEXT_TRUNCATE_CHARS_PER_ATTEMPT,
    use_cache: bool = False,
    cache_dir: str | None = None,
    summary_client: OpenAICompatibleClient | None = None,
) -> str:
    """读取并解析单个网页"""
    max_attempts = SUMMARY_MAX_ATTEMPTS

    content = await jina_readpage(url, app_id, app_key, provider, base_url, use_cache=use_cache, cache_dir=cache_dir)

    if content and not content.startswith(VISIT_ERROR_TAG):
        # 检查是否获取到有效内容
        content = content[:webcontent_maxlength]
        raw = ''
        model_candidates = random.sample(summary_models, k=len(summary_models))
        logger.debug(
            f'[visit] Selected summary model order for {url}: {[candidate.model for candidate in model_candidates]}'
        )
        for attempt in range(max_attempts):
            selected = model_candidates[min(attempt, len(model_candidates) - 1)]
            truncate_chars = context_truncate_chars_per_attempt * attempt
            webpage_content = content
            if truncate_chars > 0 and len(webpage_content) > truncate_chars:
                webpage_content = webpage_content[:-truncate_chars] + '\n[...truncated]'
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'value': EXTRACTOR_PROMPT.format(webpage_content=webpage_content, goal=goal),
                            'type': 'text',
                        }
                    ],
                }
            ]
            raw = await call_llm_server(
                messages,
                selected.app_id,
                selected.app_key,
                selected.model,
                max_tries=3,
                summary_client=summary_client,
            )

            if raw and raw != VISIT_LLM_ERROR_TAG:
                if raw == VISIT_CONTEXT_LENGTH_ERROR_TAG:
                    logger.warning(
                        f'{VISIT_LLM_ERROR_TAG} readpage context too long for {url}, '
                        f'model={selected.model}, truncated_chars={truncate_chars}, '
                        f'attempt={attempt + 1}/{max_attempts}'
                    )
                    continue
                useful_information = f'The useful information in {url} for user goal {goal} as follows: \n\n'
                raw_text = raw if isinstance(raw, str) else str(raw)
                useful_information += raw_text
                return useful_information
            retry_delay = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            logger.warning(
                f'{VISIT_LLM_ERROR_TAG} readpage LLM attempt {attempt + 1}/{max_attempts} failed for {url}, '
                f'model={selected.model}, will retry in {retry_delay}s, messages={messages!r}, raw={raw!r}'
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
        # 读取成功，但LLM解析失败，报错
        logger.error(
            f'{VISIT_LLM_ERROR_TAG} call_llm_server failed for {url} after {max_attempts} attempts, last_response={raw!r}'
        )
        return _build_jina_fallback_content(url, goal, content, max_attempts)
    # 读取失败，返回原始错误信息，纠正访问逻辑
    logger.warning(f'jina_readpage failed for {url}, but return hint text, response={content!r}')
    return f'Read page for {url}. {content}'


async def visit_pages(
    urls: list[str],
    goal: str,
    app_id: str,
    app_key: str,
    provider: str,
    summary_models: list[SummaryModelConfig],
    base_url: str,
    webcontent_maxlength: int = WEBCONTENT_MAXLENGTH,
    context_truncate_chars_per_attempt: int = CONTEXT_TRUNCATE_CHARS_PER_ATTEMPT,
    use_cache: bool = False,
    cache_dir: str | None = None,
    summary_client: OpenAICompatibleClient | None = None,
) -> str:
    """并发访问多个网页"""
    tasks = [
        readpage(
            url,
            goal,
            app_id,
            app_key,
            provider,
            summary_models,
            base_url,
            webcontent_maxlength=webcontent_maxlength,
            context_truncate_chars_per_attempt=context_truncate_chars_per_attempt,
            use_cache=use_cache,
            cache_dir=cache_dir,
            summary_client=summary_client,
        )
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, NonRetryableProviderError):
            raise result
    results = [
        result if isinstance(result, str) else f'Error fetch content from {url}, {result=}'
        for url, result in zip(urls, results, strict=False)
    ]
    return '\n=======\n'.join(results)


# ============== Nexus 工具适配层 ==============
class VisitInput(BaseModel):
    url: str | list[str] = Field(
        description='The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.'
    )
    goal: str = Field(description='The goal of the visit for webpage(s).')


@tool_registry.register(
    'visit',
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
    """访问网页并提取相关信息"""
    url = params.url
    goal = params.goal

    if not goal or not goal.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty goal provided.'}],
            meta={'success': False},
        )

    # 从 task_config 读取配置，如果没有则使用环境变量
    tool_config = context.tool_configs.get('visit', {})
    webcontent_maxlength = tool_config.get('webcontent_maxlength', WEBCONTENT_LENGTH_SETTINGS['webcontent_maxlength'])
    context_truncate_chars_per_attempt = tool_config.get(
        'context_truncate_chars_per_attempt', WEBCONTENT_LENGTH_SETTINGS['context_truncate_chars_per_attempt']
    )
    app_id = tool_config.get('jina_app_id', os.environ.get('DEFAULT_JINA_APP_ID', ''))
    app_key = tool_config.get('jina_app_key', os.environ.get('DEFAULT_JINA_APP_KEY', ''))
    provider = tool_config.get('jina_provider', os.environ.get('DEFAULT_JINA_PROVIDER', 'aws_jina'))
    base_url = tool_config.get('jina_url', os.environ.get('DEFAULT_JINA_URL', BASE_URL))
    model = tool_config.get('jina_model', os.environ.get('DEFAULT_JINA_MODEL', 'api_doubao_DeepSeek-V3.1-250821'))
    model_app_id = tool_config.get('jina_model_app_id', os.environ.get('DEFAULT_JINA_MODEL_APP_ID', ''))
    model_app_key = tool_config.get('jina_model_app_key', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY', ''))

    try:
        summary_client = get_openai_summary_client(tool_config)
    except ValueError as e:
        logger.error(f'[visit] Summary client misconfigured: {e}')
        return ToolResponse(output=[{'type': 'text', 'text': f'Error: {e}'}], meta={'success': False})

    # The AIHub 2.0 client carries its own bearer credential, so the data_eval model app
    # id/key pair is only required while the summariser still goes through data_eval.
    summary_credentials_missing = summary_client is None and (not model_app_id or not model_app_key)
    if not app_id or not app_key or summary_credentials_missing:
        logger.error(
            f'[visit] Missing credentials: app_id={bool(app_id)}, app_key={bool(app_key)}, '
            f'model_app_id={bool(model_app_id)}, model_app_key={bool(model_app_key)}, '
            f'summary_client={type(summary_client).__name__ if summary_client else None}'
        )
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': 'Error: Missing app_id or app_key configuration. Check tool_configs["visit"] or env vars.',
                }
            ],
            meta={'success': False},
        )

    summary_models = [SummaryModelConfig(model, model_app_id, model_app_key)]
    model_2 = tool_config.get('jina_model_2', os.environ.get('DEFAULT_JINA_MODEL_2', ''))
    if model_2:
        model_app_id_2 = (
            tool_config.get('jina_model_app_id_2', os.environ.get('DEFAULT_JINA_MODEL_APP_ID_2', '')) or model_app_id
        )
        model_app_key_2 = (
            tool_config.get('jina_model_app_key_2', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY_2', '')) or model_app_key
        )
        summary_models.append(SummaryModelConfig(model_2, model_app_id_2, model_app_key_2))
        logger.debug(f'[visit] Dual summary models enabled: [{model}, {model_2}]')

    use_cache = tool_config.get('use_cache', os.environ.get('JINA_USE_CACHE', '').lower() in ('1', 'true', 'yes', 'on'))
    cache_dir = tool_config.get('cache_dir', os.environ.get('SEARCH_CACHE_DIR', None))

    # 默认情况下，禁止访问 Hugging Face 数据集和空间页面，避免作弊导致的分数污染
    default_block_huggingface = os.getenv('DEFAULT_JINA_BLOCK_HUGGINGFACE', 'true').lower() in (
        '1',
        'true',
        'yes',
        'on',
    )
    block_huggingface = tool_config.get('block_huggingface', default_block_huggingface)
    if block_huggingface and (
        _is_huggingface_dataset_or_space_url(url)
        if isinstance(url, str)
        else any(_is_huggingface_dataset_or_space_url(item) for item in url)
    ):
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'{VISIT_ERROR_TAG} You are trying to visit a Hugging Face dataset or space URL. Please do not use the visit tool for this purpose.',
                }
            ],
            meta={'success': False},
        )

    if isinstance(url, str):
        result = await readpage(
            url,
            goal,
            app_id,
            app_key,
            provider,
            summary_models,
            base_url,
            webcontent_maxlength=webcontent_maxlength,
            context_truncate_chars_per_attempt=context_truncate_chars_per_attempt,
            use_cache=use_cache,
            cache_dir=cache_dir,
            summary_client=summary_client,
        )
    else:
        result = await visit_pages(
            url,
            goal,
            app_id,
            app_key,
            provider,
            summary_models,
            base_url,
            webcontent_maxlength=webcontent_maxlength,
            context_truncate_chars_per_attempt=context_truncate_chars_per_attempt,
            use_cache=use_cache,
            cache_dir=cache_dir,
            summary_client=summary_client,
        )

    if not result or any(marker in result for marker in VISIT_FAILURE_MARKERS):
        return ToolResponse(
            output=[{'type': 'text', 'text': result or 'Visit failed.'}],
            meta={'success': False},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': result.strip()}],
        meta={'success': True},
    )


if __name__ == '__main__':
    import asyncio

    tool_config = {}
    app_id = tool_config.get('jina_app_id', os.environ.get('DEFAULT_JINA_APP_ID', ''))
    app_key = tool_config.get('jina_app_key', os.environ.get('DEFAULT_JINA_APP_KEY', ''))
    base_url = tool_config.get('jina_url', os.environ.get('DEFAULT_JINA_URL', BASE_URL))
    model = tool_config.get('jina_model', os.environ.get('DEFAULT_JINA_MODEL', 'api_doubao_doubao-seed-1-6-251015'))
    model_app_id = tool_config.get('jina_model_app_id', os.environ.get('DEFAULT_JINA_MODEL_APP_ID', ''))
    model_app_key = tool_config.get('jina_model_app_key', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY', ''))
    summary_models = [SummaryModelConfig(model, model_app_id, model_app_key)]
    model_2 = tool_config.get('jina_model_2', os.environ.get('DEFAULT_JINA_MODEL_2', ''))
    if model_2:
        model_app_id_2 = (
            tool_config.get('jina_model_app_id_2', os.environ.get('DEFAULT_JINA_MODEL_APP_ID_2', '')) or model_app_id
        )
        model_app_key_2 = (
            tool_config.get('jina_model_app_key_2', os.environ.get('DEFAULT_JINA_MODEL_APP_KEY_2', '')) or model_app_key
        )
        summary_models.append(SummaryModelConfig(model_2, model_app_id_2, model_app_key_2))
    data = {
        'name': 'visit',
        'arguments': {
            'goal': 'Find information about where clown anemonefish was found as a nonnative species in the United States before 2020, including specific locations and zip codes',
            'url': 'https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243',
        },
    }
    # data = {
    #     'name': 'visit',
    #     'arguments': {
    #         'url': 'https://finance.sina.cn/2025-01-01/detail-inecmhca9877278.d.html?cre=picnextw&mod=picg&loc=2&r=0&rfunc=63&tj=cxvertical_wap_picne%E2%94%82%E2%94%82xtw_picg&cref=cj&from=nextalbum',
    #         'goal': '找到2024年上半年最后一个交易日上证主板市值前十的股票详细数据',
    #     },
    # }
    url = data['arguments']['url']  # type: ignore[index]
    goal = data['arguments']['goal']  # type: ignore[index]
    provider = tool_config.get('jina_provider', os.environ.get('DEFAULT_JINA_PROVIDER', 'aws_jina'))
    summary_client = get_openai_summary_client(tool_config)
    print(f'{url=}, {goal=}, summary_client={type(summary_client).__name__ if summary_client else None}')
    print(
        asyncio.run(
            visit_pages([url], goal, app_id, app_key, provider, summary_models, base_url, summary_client=summary_client)
        )
    )
