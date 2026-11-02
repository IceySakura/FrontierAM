"""Internal Yuanbao reverse-image-search adapter."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import uuid

from contextlib import nullcontext
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.utils.rate_limiter import get_qps_limiter, get_rate_limiter

from ._config_utils import as_bool, as_float, as_int
from ._yb_utils import (
    compress_image,
    download_image,
    download_image_as_data_uri,
    get_current_time_str,
    get_service_discovery,
    get_shared_httpx_client,
    make_request_timeout,
)


logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = 'Production'
_DEFAULT_SERVICE = 'trpc.search_image.image_agent.SearchHttp'
_DEFAULT_PATH = '/trpc.search_image.image_agent.Searcher/ImageSearchImage'
_USER_IMAGE_RE = re.compile(r'^user_image\s+(\d+)$')


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
    }
    for canonical, alias in aliases.items():
        if canonical not in config and alias in config:
            config[canonical] = config[alias]
    rpc_ca = _value(config, 'ca_bundle', 'YB_VISION_SEARCH_CA_BUNDLE', '')
    rpc_insecure = as_bool(_value(config, 'insecure', 'YB_VISION_SEARCH_INSECURE', False))
    image_ca = _value(config, 'image_download_ca_bundle', 'YB_IMAGE_DOWNLOAD_CA_BUNDLE', '')
    image_insecure = as_bool(_value(config, 'image_download_insecure', 'YB_IMAGE_DOWNLOAD_INSECURE', False))
    max_docs_value = config.get('max_docs', config.get('max_doc_num'))
    if max_docs_value is None:
        max_docs_value = os.environ.get('YB_VISION_SEARCH_MAX_DOCS', 2)
    max_concurrent = as_int(
        _value(
            config,
            'max_concurrent',
            'YB_VISION_SEARCH_MAX_CONCURRENT',
            _env_value('YB_VISION_SEARCH_MAX_CONCURRENT', 'YB_VISION_SEARCH_RATE_LIMIT', 1),
        ),
        1,
    )
    timeout = as_float(_value(config, 'timeout', 'YB_VISION_SEARCH_TIMEOUT', 60.0), 60.0)
    hard_timeout = as_float(_value(config, 'hard_timeout', 'YB_VISION_SEARCH_HTTP_HARD_TIMEOUT', 65.0), 65.0)
    connect_timeout = as_float(_value(config, 'connect_timeout', 'YB_VISION_SEARCH_CONNECT_TIMEOUT', 10.0), 10.0)
    retries = as_int(_value(config, 'retries', 'YB_VISION_SEARCH_RETRIES', 3), 3)
    max_docs = as_int(max_docs_value, 2)
    start_idx = as_int(_value(config, 'start_idx', 'YB_VISION_SEARCH_START_INDEX', 1), 1)
    max_image_bytes = as_int(
        _value(config, 'max_image_bytes', 'YB_VISION_SEARCH_MAX_IMAGE_BYTES', 1024 * 1024),
        1024 * 1024,
    )
    image_download_timeout = as_float(
        _value(config, 'image_download_timeout', 'YB_IMAGE_DOWNLOAD_TIMEOUT', 30.0),
        30.0,
    )
    discovery_ttl = as_float(_value(config, 'discovery_ttl', 'YB_VISION_SEARCH_DISCOVERY_TTL', 60.0), 60.0)
    result_max_bytes = as_int(
        _value(config, 'result_image_max_bytes', 'YB_RESULT_IMAGE_MAX_BYTES', 20 * 1024 * 1024),
        20 * 1024 * 1024,
    )
    tool_max_pixels = as_int(
        _value(config, 'tool_max_pixels', 'YB_VISION_SEARCH_TOOL_MAX_PIXELS', 1024 * 1024),
        1024 * 1024,
    )
    tool_min_pixels = as_int(
        _value(config, 'tool_min_pixels', 'YB_VISION_SEARCH_TOOL_MIN_PIXELS', 256 * 256),
        256 * 256,
    )
    resize_factor = as_int(_value(config, 'resize_factor', 'YB_VISION_SEARCH_RESIZE_FACTOR', 32), 32)
    max_ratio = as_int(_value(config, 'max_ratio', 'YB_VISION_SEARCH_MAX_RATIO', 200), 200)
    return {
        'namespace': str(_value(config, 'service_namespace', 'YB_VISION_SEARCH_NAMESPACE', _DEFAULT_NAMESPACE)),
        'service': str(_value(config, 'service_name', 'YB_VISION_SEARCH_SERVICE', _DEFAULT_SERVICE)),
        'path': str(_value(config, 'path', 'YB_VISION_SEARCH_PATH', _DEFAULT_PATH)),
        'discovery_ttl': _non_negative('discovery_ttl', discovery_ttl),
        'qps': _non_negative(
            'qps',
            as_float(_value(config, 'qps', 'YB_VISION_SEARCH_QPS', 0.75), 0.75),
        ),
        'max_concurrent': _positive('max_concurrent', max_concurrent),
        'timeout': _positive('timeout', timeout),
        'hard_timeout': _positive('hard_timeout', hard_timeout),
        'connect_timeout': _positive('connect_timeout', connect_timeout),
        'retries': _positive('retries', retries),
        'max_docs': _positive('max_docs', max_docs),
        'start_idx': _positive('start_idx', start_idx),
        'max_image_bytes': _positive('max_image_bytes', max_image_bytes),
        'server': str(_value(config, 'server', 'YB_VISION_SEARCH_SERVER', 'default')),
        'request_source': str(_value(config, 'request_source', 'YB_VISION_SEARCH_REQUEST_SOURCE', 'yuanbao')),
        'proxy': _value(config, 'proxy', 'YB_VISION_SEARCH_HTTP_PROXY', '') or None,
        'verify': False if rpc_insecure else (str(rpc_ca) if rpc_ca else True),
        'image_download_timeout': _positive('image_download_timeout', image_download_timeout),
        'image_download_proxy': config.get('image_download_proxy')
        or _env_value('YB_IMAGE_DOWNLOAD_HTTP_PROXY', 'YB_IMG_DOWNLOAD_HTTP_PROXY', '')
        or None,
        'image_download_verify': False if image_insecure else (str(image_ca) if image_ca else True),
        'result_image_max_bytes': _positive('result_image_max_bytes', result_max_bytes),
        'tool_max_pixels': _positive('tool_max_pixels', tool_max_pixels),
        'tool_min_pixels': _positive('tool_min_pixels', tool_min_pixels),
        'resize_factor': _positive('resize_factor', resize_factor),
        'max_ratio': _positive('max_ratio', max_ratio),
    }


def _resolve_user_image(source: str, images: list[str]) -> str | None:
    if not isinstance(source, str):
        return None
    match = _USER_IMAGE_RE.fullmatch(source.strip())
    if match is None:
        return None
    index = int(match.group(1)) - 1
    if index < 0 or index >= len(images):
        return None
    image = images[index]
    return image if isinstance(image, str) and image else None


def _encode_backend_image(image: Any) -> bytes:
    """Preserve the legacy ImageSearchImage JPEG encoding contract."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


async def _prepare_image(image_source: str, config: dict[str, Any]) -> tuple[Any, int, bytes] | None:
    source: str | bytes = image_source
    if image_source.startswith(('http://', 'https://')):
        try:
            source = await download_image(
                image_source,
                timeout=config['image_download_timeout'],
                proxy=config['image_download_proxy'],
                verify=config['image_download_verify'],
            )
        except Exception as exc:
            logger.warning('Failed to download a YB vision-search input image: %s', type(exc).__name__)
            return None
    _, quality, image, _ = await asyncio.to_thread(
        compress_image,
        source,
        max_size=config['max_image_bytes'],
        quality=100,
    )
    if image is None or quality < 0:
        return None
    # The source implementation used the quality-controlled encode only for
    # the size check, then encoded the PIL image again with Pillow defaults
    # for the backend request. Keep that wire contract because JPEG quality
    # can affect reverse-image-search recall.
    jpeg_bytes = await asyncio.to_thread(_encode_backend_image, image)
    return image, quality, jpeg_bytes


async def _image_search(image_bytes: bytes, prompt: str, config: dict[str, Any]) -> list[dict[str, Any]] | None:
    discovery = get_service_discovery(config['namespace'], config['service'], config['discovery_ttl'])
    limiter_name = (
        f'yb_vision_search:{config["namespace"]}:{config["service"]}:{config["qps"]}:{config["max_concurrent"]}'
    )
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
    payload = {
        'prompt': prompt,
        'server': config['server'],
        'request_id': str(uuid.uuid4()),
        'request_source': config['request_source'],
        'img_data': base64.b64encode(image_bytes).decode('ascii'),
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
            if not isinstance(result, dict):
                return None
            if result.get('code', 0) != 0:
                logger.warning('YB vision search backend returned code=%s', result.get('code'))
                return []
            image_data = result.get('imgs') or []
            if not isinstance(image_data, list):
                return None
            return [item for item in image_data[: config['max_docs']] if isinstance(item, dict)]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            discovery.invalidate()
            logger.warning(
                'YB vision search request failed (attempt=%d/%d, error=%s)',
                attempt + 1,
                retries,
                type(exc).__name__,
            )
            if attempt + 1 < retries:
                await asyncio.sleep(min(2**attempt, 4))
    return None


async def _render_hits(
    hits: list[dict[str, Any]],
    source: str,
    text: str,
    start_index: int,
    config: dict[str, Any],
) -> tuple[list[Any], int, int, list[tuple[int, int]]]:
    output: list[Any] = []
    index = start_index
    failures = 0
    image_sizes: list[tuple[int, int]] = []
    query = json.dumps({'source': source, 'text': text}, ensure_ascii=False)
    hit_urls = [(hit, str(hit.get('image_url') or hit.get('content_url') or '')) for hit in hits]
    downloadable = [(idx, hit, url) for idx, (hit, url) in enumerate(hit_urls) if url]
    downloads = await asyncio.gather(
        *(
            download_image_as_data_uri(
                url,
                timeout=config['image_download_timeout'],
                proxy=config['image_download_proxy'],
                verify=config['image_download_verify'],
                max_bytes=config['result_image_max_bytes'],
                max_pixels=config['tool_max_pixels'],
                min_pixels=config['tool_min_pixels'],
                factor=config['resize_factor'],
                max_ratio=config['max_ratio'],
            )
            for _, _, url in downloadable
        ),
        return_exceptions=True,
    )
    download_by_index = {idx: downloaded for (idx, _, _), downloaded in zip(downloadable, downloads, strict=True)}
    for idx, (hit, image_url) in enumerate(hit_urls):
        downloaded = download_by_index.get(idx)
        output.append({
            'type': 'text',
            'text': (
                f'[web_image {index} begin]\n'
                f'[search query]{query}\n'
                '[web_image_title]\n'
                f'[web_image_url]{image_url}\n'
                '[web_image_authority_level]\n'
                f'[web_image_publish_time]{hit.get("date_published", "")}\n'
                f'[web_image_description]{hit.get("caption", "")}\n'
                '[web_image_recommend_text]\n'
                '[web_image_binary]'
            ),
        })
        if isinstance(downloaded, tuple):
            data_uri, width, height = downloaded
            # ``source_url`` is not part of OpenAI's image content-part schema and ToolResponse's
            # Pydantic validation drops it. The original URL is already preserved in the adjacent
            # ``[web_image_url]`` text field, so omit the non-standard field explicitly.
            output.append({'type': 'image_url', 'image_url': {'url': data_uri}})
            image_sizes.append((width, height))
        else:
            failures += 1
            output.append({'type': 'text', 'text': '[image load failed]'})
        output.append({'type': 'text', 'text': f'\n[web_image {index} end]'})
        index += 1
    return output, index, failures, image_sizes


_VISION_SEARCH_DESCRIPTION = (
    '根据用户图片的视觉特征与prompt的语义需求，在千亿图库中搜索相似图片，'
    '并返回相似图片的文本描述信息，以此补充用户图片的实体信息。'
    '适用于需要明确用户图片实体信息的场景，如品牌IP介绍、景观介绍、名人认知、'
    '生物识别、实体消费品介绍、艺术介绍、影视认知、动漫游戏认知、网络梗图认知等'
)
_VISION_SEARCH_QUERY_DESCRIPTION = '待搜索的图片id及需求，见内层字段的描述'


class VisionSearchQueryItem(BaseModel):
    source: str = Field(
        ...,
        description=(
            '一个待搜索的图片id。以图片id代指用户图片，如user_image 1代表用户输入第一张图，'
            '具体id形式和图片的对应关系以实际内容为准。每个query仅能指定 1 个图片id；'
            '若需搜索多个图片，需使用多个query。'
        ),
    )
    text: str = Field(
        ...,
        description=(
            '对用户输入文字的转述、改写，或是对图片中需着重关注部分的写实简述。'
            '这里的语言使用应该与用户输入文字的主体语言一致，无明确指引时倾向于使用中文表述。'
            '无论是否确认实体信息，都一定避免强行在此强行猜测。'
        ),
    )


class YbVisionSearchInput(BaseModel):
    query: list[VisionSearchQueryItem] = Field(..., description=_VISION_SEARCH_QUERY_DESCRIPTION)


@tool_registry.register(
    'yb_vision_search',
    meta={'name': 'vision_search', 'description': _VISION_SEARCH_DESCRIPTION, 'tool_type': ToolType.LOCAL},
)
async def yb_vision_search_tool(params: YbVisionSearchInput, context: ToolContext) -> ToolResponse:
    now = get_current_time_str()
    if not params.query:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'当前时间：{now}\nError: empty query list.'}],
            meta={'success': False, 'error': 'empty_query'},
        )
    raw_images = context.tool_state.get('yb_search.images', []) or []
    images = list(raw_images) if isinstance(raw_images, (list, tuple)) else []
    if not images:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': (
                        f'当前时间：{now}\n'
                        'Error: no user images attached to this task; '
                        'vision_search requires images uploaded by the user.'
                    ),
                }
            ],
            meta={'success': False, 'error': 'no_images'},
        )

    config = _runtime_config(dict(context.tool_configs.get('yb_vision_search', {}) or {}))
    state = context.tool_state
    if state.get('yb_search.tool_image_max_pixels') is not None:
        config['tool_max_pixels'] = int(state['yb_search.tool_image_max_pixels'])
    if state.get('yb_search.tool_image_min_pixels') is not None:
        config['tool_min_pixels'] = int(state['yb_search.tool_image_min_pixels'])
    _positive('tool_max_pixels', config['tool_max_pixels'])
    _positive('tool_min_pixels', config['tool_min_pixels'])
    if config['tool_min_pixels'] > config['tool_max_pixels']:
        raise ValueError('tool_min_pixels must be <= tool_max_pixels')

    begin_index = int(state.get('yb_search.next_image_idx', config['start_idx']))
    rendered_parts: list[Any] = []
    no_hit_items: list[dict[str, str]] = []
    backend_errors: list[str] = []
    index = begin_index
    image_download_failures = 0
    image_sizes: list[tuple[int, int]] = []
    for item in params.query:
        item_dict = {'source': item.source, 'text': item.text or ''}
        image_source = _resolve_user_image(item.source, images)
        if image_source is None:
            no_hit_items.append(item_dict)
            continue
        prepared = await _prepare_image(image_source, config)
        if prepared is None:
            no_hit_items.append(item_dict)
            continue
        _, _, jpeg_bytes = prepared
        hits = await _image_search(jpeg_bytes, item.text, config)
        if hits is None:
            backend_errors.append(item.source)
            no_hit_items.append(item_dict)
            continue
        parts, index, failures, sizes = await _render_hits(hits, item.source, item.text, index, config)
        if not parts:
            no_hit_items.append(item_dict)
            continue
        rendered_parts.extend(parts)
        image_download_failures += failures
        image_sizes.extend(sizes)

    header = [f'当前时间：{now}']
    if no_hit_items:
        joined = '、'.join(f'"{json.dumps(item, ensure_ascii=False)}"' for item in no_hit_items)
        header.append(f'使用vision_search检索工具，{joined}未能检索到相关信息；')
        if rendered_parts:
            header.append('通过其它search query获取的检索信息如下：')
    header_text = '\n'.join(header)
    if rendered_parts:
        header_text += '\n'
    output: list[Any] = [{'type': 'text', 'text': header_text}, *rendered_parts]

    image_tokens = 0
    token_counter = state.get('token_counter')
    if token_counter is not None:
        for width, height in image_sizes:
            image_tokens += int(token_counter.count_image_tokens(width, height))
    context_updates: dict[str, Any] = {}
    rendered_count = max(index - begin_index, 0)
    if rendered_count:
        context_updates['yb_search.next_image_idx'] = index
    if image_tokens:
        previous = int(state.get('yb_search.round_image_tokens', 0) or 0)
        context_updates['yb_search.round_image_tokens'] = previous + image_tokens

    meta: dict[str, Any] = {
        'success': rendered_count > 0 or not backend_errors,
        'rendered_count': rendered_count,
        'no_hit_sources': [item['source'] for item in no_hit_items],
        'image_download_failures': image_download_failures,
        'image_tokens': image_tokens,
        'image_sizes': image_sizes,
    }
    if backend_errors:
        meta['backend_errors'] = backend_errors
    return ToolResponse(output=output, meta=meta, context_updates=context_updates)
