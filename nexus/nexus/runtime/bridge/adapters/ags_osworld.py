"""AGS OSWorld bridge assembly: deployment and explicitly selected protocol policies.

Shared executors retain their defaults; importing this module in the guest does
not import host clients or task registration. Host handler loading is lazy.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import sys
import time
import uuid

from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from weakref import WeakKeyDictionary, WeakSet

import aiohttp
import socketio
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.bridge.app import _ServerSio
from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard as _BaseRPCReplayGuard
from nexus.runtime.server.rpc_replay_guard import _ReplayEntry as _Base_ReplayEntry
from nexus.runtime.server.services.anthropic_proxy import converter as _default_converter
from nexus.runtime.server.services.anthropic_proxy.converter import (
    _initial_conversion_log_level,
    _system_summary,
    _tools_summary,
    convert_document_to_openai,
    convert_stop_reason,
    convert_system_content_to_openai,
    convert_system_to_openai,
    convert_tool_choice_to_openai,
    convert_tools_to_openai,
)
from nexus.runtime.server.services.anthropic_proxy.converter import client_request_meta as client_request_meta
from nexus.runtime.server.services.anthropic_proxy.models import (
    ContentBlock,
    DocumentContent,
    ErrorDetail,
    ErrorResponse,
    ImageContent,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingContent,
    TextContent,
    ThinkingConfig,
    ThinkingContent,
    ToolReferenceContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
    get_thinking_config,
)
from nexus.runtime.server.services.anthropic_proxy.native_response import extract_native_anthropic_response
from nexus.runtime.server.services.anthropic_proxy.streaming import (
    _sse_event,
    simulate_native_streaming,
    simulate_streaming,
)
from nexus.runtime.server.services.rsio_llm_bridge import LLMRequestError as LLMRequestError
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge as _BaseRSIOLLMBridge
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMConfig as _BaseRSIOLLMConfig
from nexus.runtime.server.services.rsio_llm_bridge import _PendingCall as _BasePendingCall
from nexus.runtime.server.services.rsio_llm_bridge import _env_disables_rsio, _underlying_client
from nexus.runtime.server.services.rsio_llm_bridge import bridge_error_payload as bridge_error_payload


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge
    from nexus.runtime.providers.ags.osworld import AGSOSWorldExecutorRuntime, AGSOSWorldRuntime


OSWORLD_BRIDGE_UPLOAD_TIMEOUT = 300.0
OSWORLD_BRIDGE_UPLOAD_ATTEMPTS = 3
OSWORLD_BRIDGE_UPLOAD_RETRY_DELAY = 2.0


# Deployment: host-only runtime access is deferred to construction.


logger = logging.getLogger(__name__)

# Conversion policies: shared converters with task-specific callbacks.

conversion_logger = logging.getLogger('nexus.osworld.anthropic_proxy.converter')

_CLIENT_META_HEADERS = {
    'x-claude-code-agent-id': 'agent_id',
    'x-claude-code-session-id': 'session_id',
}

_PDF_MAX_PAGES_ENV = 'NEXUS_ANTHROPIC_PDF_MAX_PAGES'

_PDF_MAX_SIDE_ENV = 'NEXUS_ANTHROPIC_PDF_MAX_SIDE'

_DEFAULT_PDF_MAX_PAGES = 20

_DEFAULT_PDF_MAX_SIDE = 1920

_PDF_MAX_BYTES = 32 * 1024 * 1024

_PDF_MAX_PAGE_PIXELS = 4_000_000

_PDF_MAX_TOTAL_PIXELS = 40_000_000

_conversion_log_level: int = _initial_conversion_log_level()

conversion_logger.setLevel(_conversion_log_level)


def set_conversion_log_level(level: str | int) -> None:
    """Set the level used for anthropic<->openai conversion dumps."""
    global _conversion_log_level
    if isinstance(level, str):
        name = level.upper()
        # ``getLevelNamesMapping`` is 3.11+; ``_nameToLevel`` is stable across
        # the supported range and avoids the deprecated ``getLevelName`` path
        # that Pylance complains about.
        mapping = getattr(logging, '_nameToLevel', None)
        if not isinstance(mapping, dict) or name not in mapping:
            raise ValueError(f'Invalid log level: {level!r}')
        _conversion_log_level = mapping[name]
    else:
        _conversion_log_level = int(level)
    conversion_logger.setLevel(_conversion_log_level)


def _dump(direction: str, before: Any, after: Any) -> None:
    """Emit a multi-line before/after log entry for one conversion step."""
    from nexus.utils.common import pformat

    if not conversion_logger.isEnabledFor(_conversion_log_level):
        return
    message = (
        f'\n=== conversion: {direction} ===\n'
        f'before:\n{pformat(_redact_data_urls(before))}\n'
        f'after:\n{pformat(_redact_data_urls(after))}'
    )
    conversion_logger.log(_conversion_log_level, '%s', message)


def _redact_data_urls(value: Any) -> Any:
    """Replace binary payloads with size summaries before formatting conversion logs."""

    if isinstance(value, str) and value.startswith('data:'):
        separator = value.find(',')
        if separator >= 0 and ';base64' in value[:separator]:
            return f'<{value[:separator]}; {len(value) - separator - 1} base64 chars>'
    if isinstance(value, dict):
        media_type = value.get('media_type')
        encoded = value.get('data')
        if isinstance(media_type, str) and isinstance(encoded, str):
            return {
                key: (f'<{len(encoded)} base64 chars>' if key == 'data' else _redact_data_urls(item))
                for key, item in value.items()
            }
        return {key: _redact_data_urls(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_data_urls(item) for item in value)
    return value


_DOCUMENT_MEDIA_EXTENSIONS: dict[str, str] = {
    'application/pdf': 'pdf',
    'text/plain': 'txt',
    'text/markdown': 'md',
    'text/csv': 'csv',
    'application/json': 'json',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning('Invalid %s=%r; using %s', name, raw, default)
        return default
    if value < 1:
        logger.warning('%s must be positive, got %s; using %s', name, value, default)
        return default
    return value


def _pdf_fallback_text(reason: str) -> list[dict[str, Any]]:
    return [
        {
            'type': 'text',
            'text': (
                f'[PDF attachment could not be rendered by the model proxy: {reason}. '
                'The original PDF remains available in the sandbox; use pdftoppm to render it to PNG files, '
                'then read those images.]'
            ),
        }
    ]


def _convert_document_to_openai_parts(document: DocumentContent | dict[str, Any]) -> list[dict[str, Any]]:
    """Render an Anthropic base64 PDF document into OpenAI image parts."""
    if isinstance(document, dict):
        source = document.get('source') or {}
        media_type = source.get('media_type')
        encoded = source.get('data')
    else:
        source = document.source
        media_type = source.media_type
        encoded = source.data

    if media_type != 'application/pdf':
        return _pdf_fallback_text(f'unsupported document media type {media_type!r}')
    if not isinstance(encoded, str) or not encoded:
        return _pdf_fallback_text('missing base64 data')
    if len(encoded) > ((_PDF_MAX_BYTES + 2) // 3) * 4:
        return _pdf_fallback_text(f'the encoded PDF exceeds the {_PDF_MAX_BYTES // (1024 * 1024)} MiB limit')

    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        logger.warning('Could not decode Anthropic PDF document', exc_info=True)
        return _pdf_fallback_text(f'invalid base64 data ({type(exc).__name__})')
    if len(pdf_bytes) > _PDF_MAX_BYTES:
        return _pdf_fallback_text(f'the PDF exceeds the {_PDF_MAX_BYTES // (1024 * 1024)} MiB limit')

    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            logger.warning('PyMuPDF is unavailable; cannot render Anthropic PDF document')
            return _pdf_fallback_text('PyMuPDF is unavailable')

    max_pages = _positive_int_env(_PDF_MAX_PAGES_ENV, _DEFAULT_PDF_MAX_PAGES)
    max_side = _positive_int_env(_PDF_MAX_SIDE_ENV, _DEFAULT_PDF_MAX_SIDE)
    parts: list[dict[str, Any]] = []
    rendered_pixels = 0
    rendered_count = 0
    try:
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as pdf:
            page_count = len(pdf)
            if page_count < 1:
                return _pdf_fallback_text('the PDF contains no pages')
            if getattr(pdf, 'needs_pass', False):
                return _pdf_fallback_text('the PDF is password-protected')
            render_count = min(page_count, max_pages)
            for page_index in range(render_count):
                page = pdf[page_index]
                width = max(float(page.rect.width), 1.0)
                height = max(float(page.rect.height), 1.0)
                longest_side = max(width, height)
                scale = min(max_side / longest_side, (_PDF_MAX_PAGE_PIXELS / (width * height)) ** 0.5)
                page_pixels = math.ceil(width * scale) * math.ceil(height * scale)
                if rendered_pixels + page_pixels > _PDF_MAX_TOTAL_PIXELS:
                    break
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                png = pixmap.tobytes('png')
                parts.extend([
                    {'type': 'text', 'text': f'[PDF page {page_index + 1} of {page_count}]'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')},
                    },
                ])
                rendered_pixels += page_pixels
                rendered_count += 1
            if rendered_count < page_count:
                parts.append({
                    'type': 'text',
                    'text': (
                        f'[PDF has {page_count} pages; rendered the first {rendered_count}. '
                        'Use pdftoppm in the sandbox to inspect the remaining pages.]'
                    ),
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning('Could not render Anthropic PDF document', exc_info=True)
        return _pdf_fallback_text(f'{type(exc).__name__}: {exc}')

    logger.info('Rendered Anthropic PDF document into %d image page(s)', rendered_count)
    return parts


def _document_parts(document: DocumentContent | dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == 'images':
        return _convert_document_to_openai_parts(document)
    if mode != 'file':
        raise ValueError('document_mode must be file or images')
    if isinstance(document, dict):
        return [convert_document_to_openai(document.get('source'), document.get('title'))]
    return [convert_document_to_openai(document.source, document.title)]


def convert_content_to_openai(
    content: str | list[ContentBlock], *, document_mode: str = 'file'
) -> str | list[dict[str, Any]]:
    """Convert Anthropic content to OpenAI format.

    Args:
        content: Anthropic content (string or list of content blocks)

    Returns:
        OpenAI format content (string or list of content parts)
    """
    if document_mode == 'file':
        return _default_converter.convert_content_to_openai(content)
    if isinstance(content, str):
        return content

    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get('type')
            if block_type == 'text':
                result.append({'type': 'text', 'text': block.get('text', '')})
            elif block_type == 'image':
                source = block.get('source', {})
                media_type = source.get('media_type', 'image/png')
                data = source.get('data', '')
                result.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{media_type};base64,{data}'},
                })
            elif block_type == 'document':
                result.extend(_document_parts(block, document_mode))
            elif block_type == 'tool_result':
                # Tool results need special handling - convert text and images
                tool_content = block.get('content', '')
                if isinstance(tool_content, list):
                    for item in tool_content:
                        if item.get('type') == 'text':
                            result.append({'type': 'text', 'text': item.get('text', '')})
                        elif item.get('type') == 'image':
                            source = item.get('source', {})
                            media_type = source.get('media_type', 'image/png')
                            data = source.get('data', '')
                            result.append({
                                'type': 'image_url',
                                'image_url': {'url': f'data:{media_type};base64,{data}'},
                            })
                        elif item.get('type') == 'document':
                            result.extend(_document_parts(item, document_mode))
                        elif item.get('type') == 'tool_reference':
                            tool_name = item.get('tool_name', '')
                            result.append({'type': 'text', 'text': f'[ToolSearch] Tool now available: {tool_name}'})
                else:
                    result.append({'type': 'text', 'text': tool_content})
            elif block_type == 'tool_use':
                # Tool use in content - shouldn't happen in user messages
                result.append({
                    'type': 'text',
                    'text': f'[Tool call: {block.get("name")}({json.dumps(block.get("input", {}))})]',
                })
        elif isinstance(block, TextContent):
            result.append({'type': 'text', 'text': block.text})
        elif isinstance(block, ImageContent):
            result.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{block.source.media_type};base64,{block.source.data}'},
            })
        elif isinstance(block, DocumentContent):
            result.extend(_document_parts(block, document_mode))
        elif isinstance(block, ToolResultContent):
            tool_content = block.content
            if isinstance(tool_content, list):
                for item in tool_content:
                    if isinstance(item, TextContent):
                        result.append({'type': 'text', 'text': item.text})
                    elif isinstance(item, ImageContent):
                        result.append({
                            'type': 'image_url',
                            'image_url': {'url': f'data:{item.source.media_type};base64,{item.source.data}'},
                        })
                    elif isinstance(item, DocumentContent):
                        result.extend(_document_parts(item, document_mode))
                    elif isinstance(item, ToolReferenceContent):
                        result.append({'type': 'text', 'text': f'[ToolSearch] Tool now available: {item.tool_name}'})
            else:
                result.append({'type': 'text', 'text': tool_content})
        elif isinstance(block, ToolUseContent):
            result.append({
                'type': 'text',
                'text': f'[Tool call: {block.name}({json.dumps(block.input)})]',
            })

    # Simplify if only one text block
    if len(result) == 1 and result[0].get('type') == 'text':
        return result[0]['text']

    return result


def _reorder_tool_messages_by_call_order(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align tool results with the preceding assistant tool-call order.

    Anthropic clients may emit results in tool-completion order, while some
    OpenAI-compatible backends expect the following ``role=tool`` messages to
    use the same order as the assistant's ``tool_calls`` array. Match by ID,
    reorder only the contiguous result batch for that assistant turn, and keep
    unknown result IDs in their original positions. Missing, duplicate, and
    unmatched results are preserved; this helper never invents or drops data.
    """
    reordered: list[dict[str, Any]] = []
    pending_order: dict[str, int] | None = None
    pending_tools: list[dict[str, Any]] = []

    def flush_pending_tools() -> None:
        order = pending_order
        if order and len(pending_tools) > 1:
            matched_positions: list[int] = []
            matched_messages: list[tuple[int, dict[str, Any]]] = []
            for original_position, tool_message in enumerate(pending_tools):
                tool_call_id = tool_message.get('tool_call_id')
                if isinstance(tool_call_id, str) and tool_call_id in order:
                    matched_positions.append(original_position)
                    matched_messages.append((original_position, tool_message))

            matched_messages.sort(
                key=lambda item: (
                    order[item[1]['tool_call_id']],
                    item[0],
                )
            )
            for destination, (_, tool_message) in zip(matched_positions, matched_messages, strict=True):
                pending_tools[destination] = tool_message

        reordered.extend(pending_tools)
        pending_tools.clear()

    for message in messages:
        if message.get('role') == 'tool' and pending_order is not None:
            pending_tools.append(message)
            continue

        flush_pending_tools()
        reordered.append(message)
        pending_order = None

        if message.get('role') == 'assistant':
            tool_calls = message.get('tool_calls')
            if isinstance(tool_calls, list):
                call_order: dict[str, int] = {}
                for position, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get('id')
                    if isinstance(tool_call_id, str) and tool_call_id:
                        call_order.setdefault(tool_call_id, position)
                if call_order:
                    pending_order = call_order

    flush_pending_tools()
    return reordered


def convert_messages_to_openai(
    messages: list[Message],
    thinking_config: ThinkingConfig | None = None,
    *,
    document_mode: str = 'file',
) -> list[dict[str, Any]]:
    """Convert Anthropic messages to OpenAI format.

    Handles the special case where tool_result blocks need to be converted
    to separate tool messages in OpenAI format.

    Args:
        messages: List of Anthropic messages
        thinking_config: How to encode thinking content in OpenAI format

    Returns:
        List of OpenAI format messages
    """
    thinking_config = thinking_config or get_thinking_config()
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
        else:
            role = msg.role
            content = msg.content

        # Check for tool results that need special handling
        if role == 'user' and isinstance(content, list):
            tool_results = []
            other_content = []

            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'tool_result':
                        tool_results.append(block)
                    else:
                        other_content.append(block)
                elif isinstance(block, ToolResultContent):
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.tool_use_id,
                        'content': block.content,
                        'is_error': block.is_error,
                    })
                else:
                    other_content.append(block)

            # Add tool result messages first
            for tool_result in tool_results:
                tool_content = tool_result.get('content', '')
                tool_parts: list[dict[str, Any]] = []

                if isinstance(tool_content, list):
                    for item in tool_content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                tool_parts.append({'type': 'text', 'text': item.get('text', '')})
                            elif item.get('type') == 'image':
                                source = item.get('source', {})
                                media_type = source.get('media_type', 'image/png')
                                data = source.get('data', '')
                                tool_parts.append({
                                    'type': 'image_url',
                                    'image_url': {'url': f'data:{media_type};base64,{data}'},
                                })
                            elif item.get('type') == 'document':
                                tool_parts.extend(_document_parts(item, document_mode))
                            elif item.get('type') == 'tool_reference':
                                tool_name = item.get('tool_name', '')
                                tool_parts.append({
                                    'type': 'text',
                                    'text': f'[ToolSearch] Tool now available: {tool_name}',
                                })
                        elif isinstance(item, TextContent):
                            tool_parts.append({'type': 'text', 'text': item.text})
                        elif isinstance(item, ImageContent):
                            tool_parts.append({
                                'type': 'image_url',
                                'image_url': {'url': f'data:{item.source.media_type};base64,{item.source.data}'},
                            })
                        elif isinstance(item, DocumentContent):
                            tool_parts.extend(_document_parts(item, document_mode))
                        elif isinstance(item, ToolReferenceContent):
                            tool_parts.append({
                                'type': 'text',
                                'text': f'[ToolSearch] Tool now available: {item.tool_name}',
                            })
                else:
                    tool_parts.append({'type': 'text', 'text': tool_content})

                result.append({
                    'role': 'tool',
                    'tool_call_id': tool_result.get('tool_use_id', ''),
                    'content': tool_parts,
                })

            # Add remaining content as user message if any
            if other_content:
                result.append({
                    'role': 'user',
                    'content': convert_content_to_openai(other_content, document_mode=document_mode),
                })
        elif role == 'assistant' and isinstance(content, list):
            text_parts: list[str] = []
            thinking_blocks: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'tool_use':
                        tool_calls.append({
                            'id': block.get('id', str(uuid.uuid4())),
                            'type': 'function',
                            'function': {
                                'name': block.get('name', ''),
                                'arguments': json.dumps(block.get('input', {})),
                            },
                        })
                    elif block_type == 'thinking' or block_type == 'redacted_thinking':
                        thinking_blocks.append(block)
                    elif block_type == 'text':
                        text_parts.append(block.get('text', ''))
                elif isinstance(block, ToolUseContent):
                    tool_calls.append({
                        'id': block.id,
                        'type': 'function',
                        'function': {
                            'name': block.name,
                            'arguments': json.dumps(block.input),
                        },
                    })
                elif isinstance(block, ThinkingContent):
                    thinking_blocks.append({
                        'type': 'thinking',
                        'thinking': block.thinking,
                        'signature': block.signature,
                    })
                elif isinstance(block, RedactedThinkingContent):
                    thinking_blocks.append({'type': 'redacted_thinking', 'data': block.data})
                elif isinstance(block, TextContent):
                    text_parts.append(block.text)

            joined_text = '\n'.join(text_parts)
            # Upstream Anthropic-format converters (litellm's anthropic_messages_pt, used by
            # Bedrock invoke / gcp_anthropic / anthropic routes) rewrite blank content into a
            # visible "[System: Empty message content sanitised to satisfy protocol]" text block.
            # null is the OpenAI-native way to say "tool calls only, no text".
            text_content: str | None = None if tool_calls and not joined_text.strip() else joined_text

            msg_dict: dict[str, Any] = {
                'role': 'assistant',
                'content': text_content,
            }

            if thinking_blocks:
                if thinking_config.format == 'blocks':
                    msg_dict[thinking_config.request_field] = thinking_blocks
                else:
                    reasoning_parts = [b.get('thinking', '') for b in thinking_blocks if b.get('type') == 'thinking']
                    if reasoning_parts:
                        msg_dict[thinking_config.request_field] = '\n'.join(reasoning_parts)

            if tool_calls:
                msg_dict['tool_calls'] = tool_calls

            result.append(msg_dict)
        else:
            result.append({
                'role': role,
                'content': convert_content_to_openai(content, document_mode=document_mode),
            })

    return _reorder_tool_messages_by_call_order(result)


def _dump_recent_message_conversion(
    request: MessagesRequest,
    converted_messages: list[dict[str, Any]],
) -> None:
    """Log the tail of the conversation and its openai translation.

    The Anthropic Messages API is stateless — every request carries the full
    history, so we cannot reliably compute "what's new since last time" from
    inside the proxy (clients may restart sessions or truncate history).
    Dumping the last two messages covers the usual pattern ``(assistant,
    user/tool_result)`` appended per turn, while still giving enough context
    to spot a conversion bug. Tools and system prompt are summarised rather
    than dumped in full to keep the log short.
    """
    if not conversion_logger.isEnabledFor(_conversion_log_level):
        return
    if not request.messages:
        return
    tail = request.messages[-2:]
    before = {
        'messages_tail': [m.model_dump(exclude_none=True) if hasattr(m, 'model_dump') else m for m in tail],
        'total_messages': len(request.messages),
        'tools': _tools_summary(request),
        'system': _system_summary(request),
    }
    after = {'messages_tail': converted_messages[-4:]}
    _dump(direction='anthropic->openai (last <=2 msgs)', before=before, after=after)


def anthropic_to_openai(
    request: MessagesRequest,
    thinking_config: ThinkingConfig | None = None,
    *,
    document_mode: str = 'file',
) -> dict[str, Any]:
    """Convert Anthropic MessagesRequest to OpenAI Chat Completion format.

    Args:
        request: Anthropic MessagesRequest
        thinking_config: How to encode thinking content in OpenAI format

    Returns:
        OpenAI Chat Completion request parameters
    """
    thinking_config = thinking_config or get_thinking_config()
    messages = convert_messages_to_openai(
        request.messages, thinking_config=thinking_config, document_mode=document_mode
    )
    _dump_recent_message_conversion(request, messages)

    # Add system message if present
    system = convert_system_content_to_openai(request.system)
    if system:
        messages.insert(0, {'role': 'system', 'content': system})

    result: dict[str, Any] = {
        'model': request.model,
        'messages': messages,
    }

    if request.max_tokens is not None:
        result['max_tokens'] = request.max_tokens

    result['_client_has_thinking'] = request.thinking is not None
    result['_client_requests_haiku'] = 'haiku' in request.model.lower()

    if request.tools:
        tools_data = [t.model_dump() if hasattr(t, 'model_dump') else t for t in request.tools]
        for tool in tools_data:
            if str(tool.get('type', '')).startswith('computer_'):
                raise ValueError(f'Native tool {tool["type"]} requires the native Messages client')
        result['tools'] = convert_tools_to_openai(tools_data)

    if request.tool_choice:
        choice_data = (
            request.tool_choice.model_dump() if hasattr(request.tool_choice, 'model_dump') else request.tool_choice
        )
        result['tool_choice'] = convert_tool_choice_to_openai(choice_data)

    if request.temperature is not None:
        result['temperature'] = request.temperature

    if request.top_p is not None:
        result['top_p'] = request.top_p

    if request.stop_sequences:
        result['stop'] = request.stop_sequences

    return result


def _parse_tool_arguments(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f'OpenAI tool arguments must be a JSON object or string, got {type(raw).__name__}')
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('OpenAI tool arguments contain invalid JSON') from exc
    if not isinstance(parsed, dict):
        raise ValueError(f'OpenAI tool arguments must decode to a JSON object, got {type(parsed).__name__}')
    return parsed


def convert_openai_content_to_anthropic(
    message: dict[str, Any],
    thinking_config: ThinkingConfig | None = None,
) -> list[TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent]:
    """Validate arguments locally, then use the shared content/thinking converter."""
    normalized = dict(message)
    if message.get('tool_calls'):
        normalized['tool_calls'] = [
            {
                **call,
                'function': {
                    **call.get('function', {}),
                    'arguments': json.dumps(_parse_tool_arguments(call.get('function', {}).get('arguments', '{}'))),
                },
            }
            for call in message['tool_calls']
        ]
    return _default_converter.convert_openai_content_to_anthropic(normalized, thinking_config=thinking_config)


def openai_to_anthropic(
    response: dict[str, Any] | Any,
    model: str,
    thinking_config: ThinkingConfig | None = None,
) -> MessagesResponse:
    """Convert OpenAI Chat Completion response to Anthropic MessagesResponse.

    Args:
        response: OpenAI Chat Completion response (dict or Pydantic model)
        model: Original model name from the request
        thinking_config: How to read thinking content from OpenAI format

    Returns:
        Anthropic MessagesResponse
    """
    thinking_config = thinking_config or get_thinking_config()

    # Handle both dict and object responses - convert Pydantic models to dict
    model_dump_fn = getattr(response, 'model_dump', None)
    response_dict: dict[str, Any] = (
        model_dump_fn() if model_dump_fn is not None and callable(model_dump_fn) else response
    )
    # Anthropic clients use this id to delimit assistant turns, while an OpenAI
    # backend may reuse one response id for the whole Gateway session.
    message_id = f'msg_{uuid.uuid4().hex[:24]}'

    choices = cast(list[dict[str, Any]], response_dict.get('choices', []))
    if not choices:
        return MessagesResponse(
            id=message_id,
            model=model,
            content=[TextContent(type='text', text='')],
            stop_reason='end_turn',
            stop_sequence=None,
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    # Some gateways split one assistant turn across several choices: choices[0] carries
    # only the preamble text with finish_reason='stop', while a later choice carries the
    # actual tool_calls (observed on gpt-6-astra via the aihub azure Responses passthrough).
    # Taking choices[0] blindly drops those tool calls, so the agent sees a plain sentence,
    # treats the turn as finished and never runs a single tool. Anthropic represents one
    # assistant turn as text + tool_use blocks together, so merge the tool-calling choice
    # into the leading one instead of discarding either half.
    choice = choices[0]
    message = dict(choice.get('message') or {})
    if not message.get('tool_calls'):
        for extra in choices[1:]:
            extra_message = extra.get('message') or {}
            extra_tool_calls = extra_message.get('tool_calls')
            if not extra_tool_calls:
                continue
            message['tool_calls'] = extra_tool_calls
            if not message.get('content'):
                message['content'] = extra_message.get('content')
            choice = {**extra, 'message': message}
            break

    content = convert_openai_content_to_anthropic(message, thinking_config=thinking_config)

    # Ensure at least one content block
    if not content:
        content = [TextContent(type='text', text='')]

    # Handle usage - use 'or {}' to handle both missing key and None value
    usage_data = response_dict.get('usage') or {}

    prompt_tokens = usage_data.get('prompt_tokens') or 0
    completion_tokens = usage_data.get('completion_tokens') or 0
    raw_finish_reason = choice.get('finish_reason')
    finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None

    # ExtendedUsage carries Anthropic cache metrics at the top level. Keep the
    # nested prompt-details fallback for ordinary OpenAI-compatible responses.
    cache_creation = int(usage_data.get('cache_creation_input_tokens') or 0)
    cache_read = int(usage_data.get('cache_read_input_tokens') or 0)
    details = usage_data.get('prompt_tokens_details') or {}
    if not cache_creation:
        cache_creation = int(details.get('cache_creation_tokens') or 0)
    if not cache_read:
        cache_read = int(details.get('cached_tokens') or 0)

    anthropic_response = MessagesResponse(
        id=message_id,
        model=model,
        content=content,
        stop_reason=convert_stop_reason(finish_reason),
        stop_sequence=None,
        usage=Usage(
            input_tokens=max(prompt_tokens - cache_read - cache_creation, 0),
            output_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )
    # Only dump the new assistant message (and its content blocks), not the
    # surrounding usage/id metadata or the entire raw OpenAI response.
    _dump(
        direction='openai->anthropic (new message)',
        before={'message': message, 'finish_reason': finish_reason},
        after={'content': [c.model_dump(exclude_none=True) for c in content]},
    )
    return anthropic_response


__all__ = [
    'anthropic_to_openai',
    'convert_content_to_openai',
    'convert_document_to_openai',
    'convert_messages_to_openai',
    'convert_openai_content_to_anthropic',
    'convert_stop_reason',
    'convert_system_content_to_openai',
    'convert_system_to_openai',
    'convert_tool_choice_to_openai',
    'convert_tools_to_openai',
    'openai_to_anthropic',
    'set_conversion_log_level',
]

ResultT = TypeVar('ResultT')


# Replay: cancellation tombstones and bounded retained state.


class RPCRequestCancelledError(RuntimeError):
    """Cached terminal result for an explicitly abandoned RPC request."""


@dataclass(frozen=True)
class _ReplayOutcome:
    value: object | None = None
    exception: BaseException | None = None


@dataclass
class _OSWorldReplayEntry(_Base_ReplayEntry):
    created_at: float
    future: asyncio.Future[_ReplayOutcome]
    completed_at: float | None = None
    completed_size_bytes: int = 0


_MAX_REPLAY_EXCEPTION_TEXT_CHARS = 4096

_MAX_REPLAY_EXCEPTION_NOTES = 4

_MAX_REPLAY_EXCEPTION_NOTE_CHARS = 1024


def _bounded_exception_text(exc: BaseException) -> str:
    try:
        text = str(exc)
    except Exception:
        text = '<unprintable exception>'
    if len(text) > _MAX_REPLAY_EXCEPTION_TEXT_CHARS:
        return text[:_MAX_REPLAY_EXCEPTION_TEXT_CHARS] + '…'
    return text


def _is_small_exception_arg(arg: object) -> bool:
    if arg is None or isinstance(arg, (bool, float)):
        return True
    if isinstance(arg, int):
        return arg.bit_length() <= 256
    return isinstance(arg, str) and len(arg) <= _MAX_REPLAY_EXCEPTION_TEXT_CHARS


def _lightweight_exception_clone(exc: BaseException) -> BaseException:
    """Create a traceback-free replay exception without retaining request state.

    Built-in and conventional exceptions reconstruct from ``args`` and preserve
    their exact type/message. Exceptions with custom keyword-only constructors
    fall back to a small ``RuntimeError`` rather than keeping the original
    exception's traceback, context, response, or request graph alive.

    ``RPCRequestCancelledError`` is the replay protocol's own safe tombstone.
    Preserve that exact type so late frames cannot mistake an explicit cancel
    for an ordinary provider failure and restart the abandoned request.
    """
    safe_args = all(_is_small_exception_arg(arg) for arg in exc.args)
    fallback = f'{type(exc).__name__[:256]}: {_bounded_exception_text(exc)}'
    if type(exc) is RPCRequestCancelledError:
        clone: BaseException = RPCRequestCancelledError(_bounded_exception_text(exc))
    elif type(exc).__module__ == 'builtins' and safe_args:
        try:
            clone = type(exc)(*exc.args)
        except Exception:
            clone = RuntimeError(fallback)
    else:
        clone = RuntimeError(fallback)
    clone.__traceback__ = None
    clone.__cause__ = None
    clone.__context__ = None
    notes = getattr(exc, '__notes__', None)
    if notes:
        cast(Any, clone).__notes__ = [
            str(note)[:_MAX_REPLAY_EXCEPTION_NOTE_CHARS] for note in notes[:_MAX_REPLAY_EXCEPTION_NOTES]
        ]
    return clone


def _estimate_retained_size(value: Any, seen: set[int] | None = None) -> int:
    """Best-effort retained heap size for ordinary RPC results.

    RPC results are predominantly JSON-like dict/list trees. Walk those exact
    containers without invoking user-defined iteration or serialization hooks;
    unfamiliar objects are charged their shallow size. Shared children are
    counted once.
    """
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    size = sys.getsizeof(value, 0)
    if isinstance(value, dict):
        size += sum(
            _estimate_retained_size(key, seen) + _estimate_retained_size(item, seen) for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_estimate_retained_size(item, seen) for item in value)
    elif isinstance(value, BaseException):
        size += _estimate_retained_size(value.args, seen)
    return size


class RPCReplayGuard(_BaseRPCReplayGuard):
    async def _discard_entry(self, cache_key: str, future: asyncio.Future[Any]) -> None:
        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None and entry.future is future:
                self._remove_locked(cache_key)

    async def _discard_after(self, cache_key: str, future: asyncio.Future[Any], delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None and entry.future is future:
                self._remove_locked(cache_key)

    async def discard(self, scope: str, request_id: str | None, delay: float = 0.0) -> bool:
        """Discard a completed replay result now or after a grace period.

        Delayed deletion is tied to the current replay future so it can never
        remove a newer entry that later reuses the same request id. Returns whether
        a completed entry was deleted or scheduled for deletion.
        """
        if not request_id:
            return False

        cache_key = f'{scope}:{request_id}'
        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None or not entry.future.done():
                return False

            if delay <= 0:
                self._remove_locked(cache_key)
            else:
                task = asyncio.create_task(self._discard_after(cache_key, entry.future, delay))
                # The event loop keeps only weak references to tasks. Retain this fire-and-forget
                # task until completion, then remove it from the set to avoid accumulating tasks.
                self._discard_tasks.add(task)
                task.add_done_callback(self._discard_tasks.discard)
            return True

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_entries: int = 2048,
        max_completed_bytes: int | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError('max_entries must be positive')
        if max_completed_bytes is not None and max_completed_bytes <= 0:
            raise ValueError('max_completed_bytes must be positive or None')
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._max_completed_bytes = max_completed_bytes
        self._completed_bytes = 0
        self._entries: dict[str, _OSWorldReplayEntry] = {}
        self._discard_tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    async def run(self, scope: str, request_id: str | None, handler) -> ResultT:
        """Run *handler* once and replay the cached result for duplicates."""
        if not request_id:
            return await handler()

        cache_key = f'{scope}:{request_id}'
        owner = False
        future: asyncio.Future[_ReplayOutcome]

        async with self._lock:
            # Protect the requested key from capacity eviction during lookup so
            # a just-disconnected caller can always consume a still-live cached
            # final. TTL expiry is still applied to protected keys.
            self._prune_locked(protected_key=cache_key)
            entry = self._entries.get(cache_key)
            if entry is None:
                future = cast(asyncio.Future[_ReplayOutcome], asyncio.get_running_loop().create_future())
                self._entries[cache_key] = _OSWorldReplayEntry(created_at=time.monotonic(), future=future)
                owner = True
            else:
                future = entry.future

        if not owner:
            status = 'completed' if future.done() else 'in-flight'
            logger.warning('%s: duplicate request_id=%s detected, replaying %s result', scope, request_id, status)
            outcome = await asyncio.shield(future)
            if outcome.exception is not None:
                # Raise a fresh clone on every replay. The template retained by
                # the Future is never raised and therefore remains traceback-free.
                raise _lightweight_exception_clone(outcome.exception)
            return cast(ResultT, outcome.value)

        try:
            result = await handler()
        except asyncio.CancelledError:
            # Normal task/shutdown cancellation is retryable and discards the
            # entry. Explicit protocol cancellation calls ``cancel()`` first,
            # leaving a completed exception tombstone that late frames replay
            # instead of resurrecting the side effect.
            if future.done():
                raise
            future.cancel()
            await self._discard_entry(cache_key, cast(asyncio.Future[object], future))
            raise
        except Exception as exc:
            replay_exception = _lightweight_exception_clone(exc)
            if not future.done():
                future.set_result(_ReplayOutcome(exception=replay_exception))
                await self._mark_completed(cache_key, future, replay_exception)
            raise

        if not future.done():
            future.set_result(_ReplayOutcome(value=result))
            await self._mark_completed(cache_key, future, result)
            return result

        # A concurrent explicit cancellation may have installed a terminal
        # tombstone while a cancellation-resistant handler was unwinding.
        # Replaying it here prevents the owner from emitting a late success.
        outcome = future.result()
        if outcome.exception is not None:
            raise _lightweight_exception_clone(outcome.exception)
        return cast(ResultT, outcome.value)

    async def cancel(self, scope: str, request_id: str, *, reason: str = 'request cancelled') -> bool:
        """Install a terminal cancellation tombstone for ``scope/request_id``.

        Returns True when an in-flight entry existed. A tombstone is also
        created for an unknown id so a cancel racing ahead of a late/replayed
        request cannot start the handler. This method deliberately does not
        cancel an asyncio task; the owner (which knows that task) must do so and
        await it before acknowledging cancellation to the caller.
        """
        cache_key = f'{scope}:{request_id}'
        async with self._lock:
            self._prune_locked(protected_key=cache_key)
            entry = self._entries.get(cache_key)
            was_inflight = entry is not None and not entry.future.done()
            if entry is None:
                future = cast(asyncio.Future[_ReplayOutcome], asyncio.get_running_loop().create_future())
                entry = _OSWorldReplayEntry(created_at=time.monotonic(), future=future)
                self._entries[cache_key] = entry
            if not entry.future.done():
                tombstone = RPCRequestCancelledError(reason)
                entry.future.set_result(_ReplayOutcome(exception=tombstone))
                entry.completed_at = time.monotonic()
                entry.completed_size_bytes = (
                    _estimate_retained_size(tombstone) if self._max_completed_bytes is not None else 0
                )
                self._completed_bytes += entry.completed_size_bytes
                self._prune_locked(protected_key=cache_key)
            return was_inflight

    async def _mark_completed(
        self,
        cache_key: str,
        future: asyncio.Future[_ReplayOutcome],
        result: object,
    ) -> None:
        """Record completion time/size and enforce completed-cache limits."""
        async with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None or entry.future is not future or entry.completed_at is not None:
                return
            entry.completed_at = time.monotonic()
            entry.completed_size_bytes = _estimate_retained_size(result) if self._max_completed_bytes is not None else 0
            self._completed_bytes += entry.completed_size_bytes
            self._prune_locked(protected_key=cache_key)

    def _remove_locked(self, cache_key: str) -> None:
        entry = self._entries.pop(cache_key, None)
        if entry is not None:
            self._completed_bytes = max(0, self._completed_bytes - entry.completed_size_bytes)

    def _over_capacity_locked(self) -> bool:
        return len(self._entries) > self._max_entries or (
            self._max_completed_bytes is not None and self._completed_bytes > self._max_completed_bytes
        )

    def _prune_locked(self, *, protected_key: str | None = None) -> None:
        now = time.monotonic()
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.completed_at is not None and now - entry.completed_at > self._ttl_seconds
        ]
        for key in expired_keys:
            self._remove_locked(key)

        while self._over_capacity_locked():
            candidates = [
                (entry.completed_at, key)
                for key, entry in self._entries.items()
                if key != protected_key and entry.completed_at is not None
            ]
            if not candidates:
                # In-flight entries (and the just-completed protected final) are
                # never evicted. Capacity may exceed the target transiently until
                # another request arrives, at which point this entry is eligible.
                break
            _, oldest_key = min(candidates)
            oldest_entry = self._entries[oldest_key]
            logger.info(
                'RPC replay cache capacity reached; evicting completed key=%s '
                '(entry_bytes=%d completed_bytes=%d max_completed_bytes=%s entries=%d max_entries=%d)',
                oldest_key,
                oldest_entry.completed_size_bytes,
                self._completed_bytes,
                self._max_completed_bytes,
                len(self._entries),
                self._max_entries,
            )
            self._remove_locked(oldest_key)


_DISABLE_ENV_VAR = 'NEXUS_DISABLE_LLM_RSIO'

NEXUS_AGENT_COMPACTION_FIELD = '_nexus_agent_compaction'

NEXUS_COMPACTION_DETECTOR_FIELD = '_nexus_compaction_detector'


# RSIO: shared dispatcher/driver with explicit cancellation.


def terminal_stream_failure_event(exc: BaseException) -> dict[str, Any] | None:
    """Render a coded bridge failure as a ``response.failed`` SSE event; None when uncoded.

    Used once SSE headers are out, where ``bridge_error_payload``'s 400 is no longer
    available. Closing the stream instead would look like a dropped connection and be
    reconnected, so the refusal has to arrive as a terminal Responses event.
    """
    error_code = getattr(exc, 'error_code', None)
    if not error_code:
        return None
    return {
        'type': 'response.failed',
        'response': {
            'status': 'failed',
            'error': {
                'message': f'[{error_code}] {exc}',
                'type': 'invalid_request_error',
                'code': error_code,
            },
        },
    }


@dataclass(frozen=True)
class RSIOLLMConfig(_BaseRSIOLLMConfig):
    enable: bool = True  # False selects one-shot emit+wait without RSIO or resend.
    ack_timeout: float = 5.0  # 首次探活：发出 request 后多久无回包就重发
    report_interval: float = 15.0  # Host running 心跳间隔（供 handler 使用）
    report_timeout: float = 45.0
    cancel_timeout: float = 5.0
    emit_timeout: float = 5.0


@dataclass
class _PendingCall(_BasePendingCall):
    cancel_ack: asyncio.Future = field(kw_only=True)
    final_received: bool = False


_SHARED_PENDING: WeakKeyDictionary[Any, dict[str, _PendingCall]] = WeakKeyDictionary()

_SHARED_HANDLER_REGISTERED: WeakSet[Any] = WeakSet()


class RSIOLLMBridge(_BaseRSIOLLMBridge):
    def __init__(self, sio: SIOBridge, config: RSIOLLMConfig | None = None, *, channel_id: str | None = None) -> None:
        self.sio = sio
        self.cfg = config or RSIOLLMConfig()
        self.channel_id = channel_id
        self._closed = False
        self._owned_requests: dict[str, asyncio.Task[Any]] = {}
        # Disabled RSIO uses one-shot emit+wait, without an envelope or resends.
        self._legacy = not self.cfg.enable or _env_disables_rsio()
        if channel_id is not None and self._legacy:
            raise ValueError('channel_id requires RSIO mode; enable RSIO and unset NEXUS_DISABLE_LLM_RSIO')
        # All bridges sharing one underlying sio client share one pending table,
        # so responses for any request_id reach the bridge that owns it.
        self._client = _underlying_client(sio)
        self._pending: dict[str, _PendingCall] = _SHARED_PENDING.setdefault(self._client, {})
        self._setup_handlers()

    cfg: RSIOLLMConfig
    _pending: dict[str, Any]
    _config_type = RSIOLLMConfig
    _pending_tables = _SHARED_PENDING
    _registered_clients = _SHARED_HANDLER_REGISTERED

    @staticmethod
    def _on_final(pending):
        pending.final_received = True

    def _setup_handlers(self) -> None:
        # Register the shared dispatcher exactly once per underlying client.
        # Later bridges reuse the same _pending table (see __init__) so their
        # request_ids are dispatched by the single existing handler.
        if self._client in _SHARED_HANDLER_REGISTERED:
            return
        _SHARED_HANDLER_REGISTERED.add(self._client)
        pending_table = self._pending

        @self.sio.on('llm_response')
        async def on_llm_response(data: dict[str, Any]) -> None:
            request_id = data.get('request_id')
            if not request_id:
                logger.warning('rsio.worker: llm_response without request_id, ignoring')
                return
            pending = pending_table.get(request_id)
            if pending is None:
                # Unknown / already-completed request. Common and harmless: a
                # duplicate final for a call we already resolved and popped.
                logger.debug('rsio.worker: llm_response for unknown request_id=%s, ignoring', request_id)
                return
            if pending.future.done():
                logger.debug('rsio.worker: llm_response for already-resolved request_id=%s, ignoring', request_id)
                return
            rsio = data.get('rsio') or {}
            kind = rsio.get('kind', 'final')  # 未知/缺省视为 final（兼容老 Host）
            if kind == 'running':
                pending.last_report_at = asyncio.get_running_loop().time()
                pending.saw_running = True
                logger.debug('rsio.worker: running heartbeat request_id=%s (attempt=%s)', request_id, pending.attempt)
                return
            pending.final_received = True
            if data.get('success'):
                logger.debug('rsio.worker: final OK request_id=%s attempts=%s', request_id, pending.attempt)
                pending.future.set_result(data.get('response', {}))
            else:
                error = data.get('error', 'Unknown error')
                error_code = data.get('error_code') or None
                logger.warning(
                    'rsio.worker: final ERROR request_id=%s attempts=%s code=%s error=%s',
                    request_id,
                    pending.attempt,
                    error_code,
                    error,
                )
                pending.future.set_exception(LLMRequestError(error, error_code=error_code))

        @self.sio.on('llm_cancelled')
        async def on_llm_cancelled(data: dict[str, Any]) -> None:
            request_id = data.get('request_id')
            if not request_id:
                logger.warning('rsio.worker: llm_cancelled without request_id, ignoring')
                return
            pending = self._pending.get(request_id)
            if pending is None or pending.cancel_ack.done():
                logger.debug('rsio.worker: late/duplicate llm_cancelled request_id=%s, ignoring', request_id)
                return
            pending.cancel_ack.set_result(bool(data.get('cancelled')))

    async def request(self, openai_params: dict[str, Any], timeout: float | None = 300) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError('LLM bridge is closed; create a new bridge before making requests')
        loop = asyncio.get_running_loop()
        request_id = str(uuid.uuid4())
        pending = _PendingCall(
            request_id=request_id,
            params=openai_params,
            future=loop.create_future(),
            cancel_ack=loop.create_future(),
            deadline_at=loop.time() + timeout if timeout is not None else None,
        )
        self._pending[request_id] = pending
        mode = 'legacy' if self._legacy else 'rsio'
        timeout_description = f'{timeout:.0f}s' if timeout is not None else 'disabled'
        logger.debug('rsio.worker: call start request_id=%s timeout=%s mode=%s', request_id, timeout_description, mode)
        try:
            # Track the caller so shutdown can cancel a blocked emit directly.
            task = asyncio.current_task()
            assert task is not None, 'LLM requests must run inside an asyncio task'
            self._owned_requests[request_id] = task
            if self._legacy:
                return await asyncio.wait_for(self._drive_legacy(pending, timeout), timeout=timeout)
            return await self._drive(pending)
        except (asyncio.CancelledError, Exception) as exc:
            # Drop the pending entry BEFORE awaiting the cancellation ack.
            # _pending is the ledger of what *this side* is still waiting for,
            # and once we are unwinding we are not waiting any more -- the ack
            # round-trip (up to cancel_timeout, 5s) only tells the host to stop.
            # Keeping the entry across that await made a caller that observes
            # the bridge right after shutdown still see the abandoned request.
            self._pending.pop(request_id, None)
            # A host final (including a provider error) means the host call is
            # already over.  Otherwise this is a local timeout/disconnect/
            # cancellation: explicitly tear down the host-side provider task
            # before allowing the caller to retry the logical request.
            if not self._legacy and not pending.final_received:
                # On local cancellation the caller has gone away; fire-and-forget
                # so shutdown is not held up by the ack round-trip.
                await self._cancel_remote(
                    pending,
                    wait_for_ack=not isinstance(exc, asyncio.CancelledError),
                )
            raise
        finally:
            self._pending.pop(request_id, None)
            self._owned_requests.pop(request_id, None)
            pending.future.cancel()

    async def _cancel_remote(self, pending: _PendingCall, *, wait_for_ack: bool = True) -> None:
        logger.info('rsio.worker: cancelling host request request_id=%s', pending.request_id)
        deadline = asyncio.get_running_loop().time() + self.cfg.cancel_timeout
        try:
            await asyncio.wait_for(
                self.sio.emit(
                    'llm_cancel',
                    {'request_id': pending.request_id, 'rsio': {'attempt': pending.attempt}},
                ),
                timeout=self.cfg.cancel_timeout,
            )
        except Exception as exc:
            logger.warning(
                'rsio.worker: llm_cancel emit failed request_id=%s (%s)',
                pending.request_id,
                exc,
            )
            return

        if not wait_for_ack:
            # The caller is already gone (its task was cancelled), so nobody is
            # left to act on the ack. Telling the host to stop is the part that
            # matters; awaiting up to cancel_timeout here only keeps the ASGI
            # task alive past shutdown.
            return

        try:
            cancelled = await asyncio.wait_for(
                asyncio.shield(pending.cancel_ack),
                timeout=max(0, deadline - asyncio.get_running_loop().time()),
            )
            logger.info(
                'rsio.worker: host cancellation acknowledged request_id=%s active=%s',
                pending.request_id,
                cancelled,
            )
        except asyncio.TimeoutError:
            logger.warning(
                'rsio.worker: host cancellation ack timed out request_id=%s after %.1fs',
                pending.request_id,
                self.cfg.cancel_timeout,
            )

    async def _send(self, pending: _BasePendingCall) -> None:
        loop = asyncio.get_running_loop()
        pending.attempt += 1
        pending.last_report_at = loop.time()
        envelope: dict[str, Any] = {
            **pending.params,
            'request_id': pending.request_id,
            'rsio': {
                'attempt': pending.attempt,
                'deadline_at': pending.deadline_at,
            },
        }
        if self.channel_id is not None:
            envelope['rsio']['channel_id'] = self.channel_id
        log = logger.debug if pending.attempt == 1 else logger.warning
        log(
            'rsio.worker: %s llm_request request_id=%s attempt=%s',
            'send' if pending.attempt == 1 else 'RESEND',
            pending.request_id,
            pending.attempt,
        )
        # `deadline_at` is None when the caller disabled the timeout (codex sets
        # request_timeout=None and manages its own), so the emit budget is just
        # emit_timeout -- the same None guard the drive loop uses above.
        emit_budget = self.cfg.emit_timeout
        if pending.deadline_at is not None:
            emit_budget = min(emit_budget, max(0, pending.deadline_at - loop.time()))
        try:
            await asyncio.wait_for(
                self.sio.emit('llm_request', envelope),
                timeout=emit_budget,
            )
        except Exception as exc:
            # Disconnected / transient transport failure. Do NOT fail the call;
            # the drive loop resends on the next probe tick (lazy reconnect).
            logger.warning(
                'rsio.worker: llm_request emit failed request_id=%s attempt=%s (%s); will retry',
                pending.request_id,
                pending.attempt,
                exc,
            )


_STREAM_KEEPALIVE_ENABLED_ENV = 'NEXUS_LLM_STREAM_KEEPALIVE'

_STREAM_KEEPALIVE_INTERVAL_ENV = 'NEXUS_LLM_STREAM_KEEPALIVE_INTERVAL'


# HTTP protocol adapters: shared Messages routes and native Responses.


def _context_limit_message(exc: Exception) -> str | None:
    """Restore the Anthropic too-long signal after the bridge serializes errors.

    CC uses both the error type and this wording for reactive compaction. Do not
    turn a deterministic context rejection into a retryable 500/api_error.
    """
    message = str(exc)
    lowered = message.lower()
    signals = (
        'context_length_exceeded',
        'context length exceeded',
        'context window exceeded',
        'prompt is too long',
        'prompt is longer than the model',
        "exceeds model's maximum context length",
        'maximum context length is',
    )
    if any(signal in lowered for signal in signals):
        return f'prompt is too long: {message}'
    return None


class LLMBridge(RSIOLLMBridge):
    """Bridge for LLM requests via Socket.IO.

    Thin alias over the shared :class:`RSIOLLMBridge`; kept as a named
    subclass so existing imports (including responses_proxy) stay valid.
    """


class LLMRequestBridgeProtocol(Protocol):
    """Minimal interface required by the Anthropic app factory."""

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]: ...


def create_anthropic_app(
    *,
    bridge: LLMRequestBridgeProtocol,
    request_timeout: float = 300,
    conversion_log_level: str | int = 'INFO',
    thinking_config: str | dict[str, Any] | ThinkingConfig | None = None,
    document_mode: str = 'file',
    title: str = 'Anthropic API Proxy',
    health_payload: Callable[[], dict[str, Any]] | None = None,
    include_health_route: bool = True,
    on_request_body: Callable[[dict[str, Any]], None] | None = None,
    on_response: Callable[[Any], None] | None = None,
) -> FastAPI:
    """Create an Anthropic-compatible FastAPI app backed by an LLM request bridge.

    ``on_request_body``/``on_response`` are optional observers: the first sees the
    raw request body before validation, the second every final response (native
    dict or ``MessagesResponse``). ``PtyAnthropicProxy`` uses them to build a
    structured tool-call event log without duplicating any route.

    ``include_health_route=False`` serves health on ``'/'`` only. Callers that
    mount their own richer ``'/health'`` afterwards (the sandbox-local bridge
    publishes ``bridge``/``launch_id`` there, which the host's ``wait_ready``
    polls for) need this: FastAPI matches the first registered route, so a
    default ``'/health'`` here would shadow theirs.
    """
    resolved_thinking_config = get_thinking_config(thinking_config)
    set_conversion_log_level(conversion_log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info('%s starting', title)
        yield
        logger.info('%s shutting down', title)

    app = FastAPI(
        title=title,
        description='Proxy Anthropic Messages API to OpenAI-compatible backends',
        lifespan=lifespan,
    )

    def _health_payload() -> dict[str, Any]:
        if health_payload is not None:
            return health_payload()
        return {
            'status': 'ok',
            'bridge_ready': True,
        }

    @app.get('/')
    async def root_health():
        """Root health check endpoint."""
        return _health_payload()

    if include_health_route:

        @app.get('/health')
        async def health():
            """Health check endpoint."""
            return _health_payload()

    @app.post('/v1/messages')
    async def messages(raw_request: Request):
        """Handle Anthropic Messages API requests."""
        body = await raw_request.json()
        if on_request_body is not None:
            on_request_body(body)

        try:
            request = MessagesRequest.model_validate(body)
        except Exception as e:
            logger.error('Request validation failed: %s', e)
            raise HTTPException(status_code=422, detail=str(e)) from e

        try:
            # Convert Anthropic request to OpenAI format (always non-streaming to backend).
            # The converter is a synchronous, CPU-bound function; large payloads (e.g. a
            # multi-thousand-character thinking block) can hold the event loop for hundreds
            # of milliseconds. Offload to a worker thread so the proxy loop stays responsive.
            openai_params = await asyncio.to_thread(
                anthropic_to_openai, request, thinking_config=resolved_thinking_config, document_mode=document_mode
            )
            request_meta = client_request_meta(raw_request.headers)
            if request_meta:
                openai_params['request_meta'] = request_meta
            logger.debug(f'Converted request to OpenAI format: {openai_params.get("model")}')

            if request.stream:
                # Reverse-streaming with keep-alive. The backend (host LLM) is
                # non-streaming, so we can only emit real SSE events once the full
                # response is back. But we open the SSE response immediately and emit
                # `: keepalive` comments while waiting, so the client's underlying HTTP
                # stack (Node/undici defaults headersTimeout/bodyTimeout to 300s) never
                # sees a long silence and drops the socket. Comments are ignored by the
                # Anthropic SDK decoder (lines starting with ':'), so they don't pollute
                # the business event stream.
                bridge_task_timeout = request_timeout
                model = request.model
                current_thinking_config = resolved_thinking_config

                async def stream_with_keepalive() -> AsyncGenerator[str, None]:
                    bridge_task = asyncio.create_task(bridge.request(openai_params, timeout=bridge_task_timeout))
                    keepalive_interval = _stream_keepalive_interval()
                    try:
                        if keepalive_interval > 0:
                            # First byte immediately so response headers are flushed and the
                            # client's headersTimeout window is satisfied right away.
                            yield ': keepalive\n\n'
                            while not bridge_task.done():
                                done, _ = await asyncio.wait({bridge_task}, timeout=keepalive_interval)
                                if not done:
                                    # Host still working — refresh the body idle timer.
                                    yield ': keepalive\n\n'
                        else:
                            logger.info('Anthropic streaming keepalive disabled by environment')
                            if not bridge_task.done():
                                await bridge_task

                        try:
                            openai_response = bridge_task.result()
                            native_response = extract_native_anthropic_response(openai_response)
                            if native_response is not None:
                                if on_response is not None:
                                    on_response(native_response)
                                stream = simulate_native_streaming(native_response)
                            else:
                                # Same CPU-bound concern as the request converter above.
                                anthropic_response = await asyncio.to_thread(
                                    openai_to_anthropic,
                                    openai_response,
                                    model,
                                    thinking_config=current_thinking_config,
                                )
                                if on_response is not None:
                                    on_response(anthropic_response)
                                stream = simulate_streaming(anthropic_response)
                            async for chunk in stream:
                                yield chunk
                        except Exception as exc:
                            # Headers (200) already sent — cannot raise HTTP 500 now;
                            # signal bridge, native-envelope, validation, and encoding failures
                            # through the established Anthropic-style SSE error event.
                            logger.error(f'Error processing streaming request: {exc}')
                            context_message = _context_limit_message(exc)
                            coded = None if context_message else bridge_error_payload(exc)
                            if coded is not None:
                                # Coded bridge failure (e.g. prompt over budget): keep the
                                # upstream type/message so the agent transcript records it
                                # instead of a generic api_error.
                                _, content = coded
                                error = content['error']
                                yield _sse_event(
                                    'error',
                                    {
                                        'type': 'error',
                                        'error': {'type': error['type'], 'message': error['message']},
                                    },
                                )
                                return
                            yield _sse_event(
                                'error',
                                {
                                    'type': 'error',
                                    'error': {
                                        'type': 'invalid_request_error' if context_message else 'api_error',
                                        'message': context_message or str(exc),
                                    },
                                },
                            )
                            return
                    except asyncio.CancelledError:
                        # Client disconnected — the finally below cancels the in-flight bridge
                        # request so LLMBridge.request()'s finally clears pending_requests.
                        raise
                    finally:
                        if not bridge_task.done():
                            bridge_task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await bridge_task

                return StreamingResponse(
                    stream_with_keepalive(),
                    media_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                    },
                )

            # Non-streaming: block until the host returns the full response, then return it
            # directly. Errors here still surface as HTTP 400/500 (headers not yet sent).
            openai_response = await bridge.request(openai_params, timeout=request_timeout)
            native_response = extract_native_anthropic_response(openai_response)
            if native_response is not None:
                if on_response is not None:
                    on_response(native_response)
                return native_response
            anthropic_response = await asyncio.to_thread(
                openai_to_anthropic,
                openai_response,
                request.model,
                thinking_config=resolved_thinking_config,
            )
            if on_response is not None:
                on_response(anthropic_response)
            return JSONResponse(content=anthropic_response.model_dump(mode='json', exclude_none=True))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error processing request: {e}')
            context_message = _context_limit_message(e)
            if context_message:
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(
                        error=ErrorDetail(type='invalid_request_error', message=context_message)
                    ).model_dump(),
                )
            coded = bridge_error_payload(e)
            if coded is not None:
                # 400 (not 500) for coded bridge failures: codex retries 500s up to 30
                # times and swallows the message; see bridge_error_payload's docstring.
                status_code, content = coded
                return JSONResponse(status_code=status_code, content=content)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post('/v1/messages/count_tokens')
    async def count_tokens(request: Request):
        """Token counting endpoint (stub implementation)."""
        body = await request.json()
        # Rough estimate (~4 chars per token); actual counting would need a tokenizer.
        messages = body.get('messages', [])
        total_chars = sum(len(str(m.get('content', ''))) for m in messages)
        return {'input_tokens': total_chars // 4}

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f'Unhandled exception: {exc}')
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    type='internal_error',
                    message=str(exc),
                )
            ).model_dump(),
        )

    return app


def flatten_calls(response: dict, body: dict) -> dict:
    """Only map namespaces declared in this request, as Codex expects."""
    allowed = {
        (tool['name'], child['name'])
        for tool in body.get('tools', [])
        if tool.get('type') == 'namespace'
        for child in tool.get('tools', [])
    }
    output = []
    for item in response.get('output', []):
        item = dict(item)
        pair = (item.get('namespace'), item.get('name'))
        if item.get('type') in {'function_call', 'custom_tool_call'} and pair in allowed:
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
            item['name'] = pair[0] + '__' + pair[1]
            item.pop('namespace', None)
        output.append(item)
    return {**response, 'output': output}


KEEPALIVE_SECONDS = 15.0


def _terminal_error(exc: Exception) -> LLMRequestError | None:
    # Only known terminal codes cross this boundary, never arbitrary exception
    # text (which may contain credentials or image payloads).
    messages = {
        'content_policy_violation': 'Upstream rejected the request due to content policy.',
        'context_length_exceeded': 'The request exceeds the model context window.',
    }
    code = getattr(exc, 'error_code', None)
    if code in messages:
        return LLMRequestError(messages[code], error_code=code)
    return None


def response_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    status = response.get('status')
    final_type = {'failed': 'response.failed', 'incomplete': 'response.incomplete'}.get(status, 'response.completed')
    return [
        {'type': 'response.created', 'response': {'id': response.get('id', ''), 'object': 'response'}},
        *[
            {'type': 'response.output_item.done', 'output_index': index, 'item': item}
            for index, item in enumerate(response.get('output') or [])
        ],
        {'type': final_type, 'response': response},
    ]


def _sse(event: dict[str, Any]) -> str:
    return f'event: {event["type"]}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n'


def attach_responses_endpoint(app: FastAPI, bridge: Any, *, timeout: float) -> None:
    # Scoped to this endpoint/bridge, bounded, no prompts or exception text kept.
    # Codex 0.153.4 reconnects even after a coded response.failed. Its next exact
    # replay must receive HTTP 400, not repeat a known rejected upstream request.
    refused: OrderedDict[str, LLMRequestError] = OrderedDict()

    def fingerprint(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def rejected(error):
        rendered = bridge_error_payload(error)
        assert rendered is not None
        status, payload = rendered
        return JSONResponse(status_code=status, content=payload)

    @app.post('/v1/responses')
    async def responses(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={'error': {'message': 'Expected a JSON object'}})
        if refused:
            key = await asyncio.to_thread(fingerprint, body)
            if key in refused:
                return rejected(refused[key])
        params = {'_osworld_responses_body': body}
        if not body.get('stream', False):
            try:
                return JSONResponse(flatten_calls(await bridge.request(params, timeout=timeout), body))
            except Exception as exc:
                if terminal := _terminal_error(exc):
                    return rejected(terminal)
                # Upstream bodies can contain credentials or very large image payloads.
                return JSONResponse(status_code=502, content={'error': {'message': 'Responses bridge failed'}})

        pending = asyncio.create_task(bridge.request(params, timeout=timeout))
        try:
            # A quick terminal rejection can still use its faithful HTTP status
            # before SSE headers commit. Long calls keep their decoded heartbeat.
            await asyncio.wait({pending}, timeout=KEEPALIVE_SECONDS)
            if pending.done() and not pending.cancelled():
                exc = pending.exception()
                if isinstance(exc, Exception) and (terminal := _terminal_error(exc)):
                    return rejected(terminal)
        except BaseException:
            pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
            raise

        async def stream():
            # Codex's idle timer waits for decoded SSE events, not raw bytes.
            # Comments keep HTTP alive but cannot keep its event reader alive.
            waiting = {
                'type': 'response.in_progress',
                'response': {'id': f'resp_wait_{uuid.uuid4().hex}', 'object': 'response', 'status': 'in_progress'},
            }
            try:
                yield _sse(waiting)
                while not pending.done():
                    done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_SECONDS)
                    if not done:
                        yield _sse(waiting)
                for event in response_events(flatten_calls(pending.result(), body)):
                    yield _sse(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if terminal := _terminal_error(exc):
                    key = await asyncio.to_thread(fingerprint, body)
                    refused[key] = terminal
                    if len(refused) > 16:
                        refused.popitem(last=False)
                    event = terminal_stream_failure_event(terminal)
                    assert event is not None
                    yield _sse(event)
                    return
                yield _sse({
                    'type': 'error',
                    'error': {
                        'type': 'api_error',
                        'message': 'Responses bridge failed; see host diagnostics',
                    },
                })
            finally:
                if not pending.done():
                    pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending

        return StreamingResponse(
            stream(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )


# Host-only Infer Proxy policy, selected by the OSWorld handler, not the registry.


def _infer_proxy_retry_wait():
    from tenacity import wait_exponential_jitter

    return wait_exponential_jitter(max=10 * 60, jitter=30)


def adapt_infer_proxy_client(client):
    """Create a handler-local view without changing the configured client or SDK."""
    from nexus.extensions.openai_clients.infer_proxy import InferProxyClient

    # Subclasses may implement their own wire/decoding policy. Only the standard
    # registered InferProxyClient is adapted; bridge_litellm and other clients pass through.
    if type(client) is not InferProxyClient:
        return client
    cls = globals().get('_InferProxyAdapter')
    if cls is None:
        cls = _load_infer_proxy_adapter_class()
        globals()['_InferProxyAdapter'] = cls
    return cls(client)


def _load_infer_proxy_adapter_class():
    import copy
    import re

    from dataclasses import replace

    import openai

    from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

    from nexus.core import ContextLengthExceededError
    from nexus.core.types import ExtendedChatCompletion
    from nexus.extensions.openai_clients.infer_proxy import InferProxyClient
    from nexus.utils.common import before_retry_sleep

    context_limit_re = re.compile(r'maximum context length is\s+(\d+)\s+tokens', re.IGNORECASE)
    input_tokens_re = re.compile(r'request has\s+(\d+)\s+input tokens', re.IGNORECASE)

    def _is_retryable_request_error(exc: BaseException) -> bool:
        """Retry only failures that can plausibly recover without changing the request."""

        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, TimeoutError, ConnectionError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code == 429 or 500 <= exc.status_code < 600
        return False

    def _split_embedded_reasoning(content):
        """Split Infer Proxy's ``reasoning</think>visible`` content convention."""

        if not isinstance(content, str) or '</think>' not in content:
            return None, content
        # Split at the first delimiter. Some malformed long responses repeat the
        # delimiter; rsplit would leak an earlier delimiter back into reasoning.
        reasoning, visible = content.split('</think>', 1)
        reasoning = reasoning.strip()
        if reasoning.startswith('<think>'):
            reasoning = reasoning[len('<think>') :].lstrip()
        return reasoning or None, visible.lstrip()

    def _adaptive_context_max_tokens(exc: BaseException, current_max_tokens: object, *, margin: int) -> int | None:
        """Derive a smaller output budget from an Infer Proxy context rejection."""

        if not isinstance(current_max_tokens, int) or isinstance(current_max_tokens, bool) or current_max_tokens < 1:
            return None
        detail = str(exc)
        context_match = context_limit_re.search(detail)
        input_match = input_tokens_re.search(detail)
        if context_match is None or input_match is None:
            return None
        available = int(context_match.group(1)) - int(input_match.group(1)) - margin
        if available < 1 or available >= current_max_tokens:
            return None
        return available

    def _validate_function_tools(tools: object, tool_choice: object = None) -> None:
        """Reject provider-hosted or malformed tools before entering HTTP retries.

        Infer Proxy implements OpenAI function tools. Anthropic server tools such as
        ``web_search_*`` must be executed by a host-side adapter and must never be
        forwarded as though they contained an OpenAI ``function`` object.
        """

        if tools is None:
            if isinstance(tool_choice, dict):
                raise ValueError('Infer Proxy function tool_choice requires a tools list')
            return
        if not isinstance(tools, list):
            raise TypeError('Infer Proxy tools must be a list')

        names: set[str] = set()
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise TypeError(f'Infer Proxy tools[{index}] must be an object')
            tool_obj = cast(dict[str, Any], tool)
            tool_type = tool_obj.get('type')
            if tool_type != 'function':
                raise ValueError(
                    f'Infer Proxy only supports OpenAI function tools; tools[{index}].type={tool_type!r} '
                    'must be handled by a host-side provider-tool adapter'
                )
            function = tool_obj.get('function')
            if not isinstance(function, dict):
                raise ValueError(f'Infer Proxy tools[{index}].function must be an object')
            name = function.get('name')
            if not isinstance(name, str) or not name:
                raise ValueError(f'Infer Proxy tools[{index}].function.name must be a non-empty string')
            parameters = function.get('parameters')
            if parameters is not None and not isinstance(parameters, dict):
                raise ValueError(f'Infer Proxy tools[{index}].function.parameters must be an object')
            names.add(name)

        if isinstance(tool_choice, dict):
            tool_choice_obj = cast(dict[str, Any], tool_choice)
            function_choice = tool_choice_obj.get('function')
            if tool_choice_obj.get('type') != 'function' or not isinstance(function_choice, dict):
                raise ValueError('Infer Proxy object tool_choice must select an OpenAI function')
            choice_name = cast(dict[str, Any], function_choice).get('name')
            if not isinstance(choice_name, str) or not choice_name:
                raise ValueError('Infer Proxy tool_choice.function.name must be a non-empty string')
            if choice_name not in names:
                raise ValueError(f'Infer Proxy tool_choice selects unknown function {choice_name!r}')

    class InferProxyAdapter(InferProxyClient):
        def __init__(self, source):
            # Preserve tracing tags and shared helpers, but isolate request config
            # and retry options. SDK copies share its HTTP transport, owned by source.
            self.__dict__.update(vars(source))
            self.config = replace(
                source.config,
                client_args=dict(source.config.client_args),
                request_args=copy.deepcopy(source.config.request_args),
                enforced_request_args=copy.deepcopy(source.config.enforced_request_args),
            )
            self.client = source.client.with_options(max_retries=0)
            self.inject_request_id = True
            self.split_think_content = True
            self.normalize_tool_finish_reason = True
            self.adaptive_context_max_tokens = True
            self.context_token_margin = 4096

        async def _create_completion(self, messages, model=None, **kwargs):
            # Historical config calls this ``max_retries``, but the existing retry
            # helper interpreted it as the total number of attempts. Preserve that
            # contract while sharing the budget with an adaptive parameter retry.
            max_attempts = max(1, int(self.config.client_args.get('max_retries', 15)))

            # Merge extra_body from request_args and runtime kwargs. Runtime values
            # remain authoritative, including an explicitly supplied request_id.
            extra_body = {
                **self.config.request_args.get('extra_body', {}),
                **kwargs.pop('extra_body', {}),
            }
            if self.inject_request_id:
                # Infer Proxy may serve an unrelated cached answer when request_id is
                # absent. Generate one ID per logical _create_completion call; retry
                # attempts of that same call deliberately reuse it.
                extra_body.setdefault('request_id', f'nexus_{uuid.uuid4().hex}')

            # Build final request parameters
            request_params = {
                'model': model,
                **{k: v for k, v in self.config.request_args.items() if k != 'extra_body'},
                **kwargs,
                'messages': messages,
            }
            if extra_body:
                request_params['extra_body'] = extra_body
            request_params = self._apply_enforced_request_args(request_params)

            _validate_function_tools(request_params.get('tools'), request_params.get('tool_choice'))

            attempts_used = 0

            async def send(params: dict[str, Any], *, attempts: int):
                async def create():
                    nonlocal attempts_used
                    attempts_used += 1
                    return await self.client.chat.completions.create(**params)

                return await AsyncRetrying(
                    retry=retry_if_exception(_is_retryable_request_error),
                    stop=stop_after_attempt(attempts),
                    wait=_infer_proxy_retry_wait(),
                    before_sleep=before_retry_sleep,
                    reraise=True,
                )(create)

            try:
                try:
                    response = await send(request_params, attempts=max_attempts)
                except openai.BadRequestError as first_error:
                    effective = self._resolve_effective_request_args(request_params)
                    token_param = next(
                        (name for name in ('max_tokens', 'max_completion_tokens') if name in effective),
                        None,
                    )
                    adjusted_max_tokens = (
                        _adaptive_context_max_tokens(
                            first_error,
                            effective.get(token_param) if token_param else None,
                            margin=self.context_token_margin,
                        )
                        if self.adaptive_context_max_tokens
                        else None
                    )
                    remaining_attempts = max_attempts - attempts_used
                    pinned = self.config.enforced_request_args
                    pinned_body = pinned.get('extra_body') or {}
                    if (
                        token_param is None
                        or adjusted_max_tokens is None
                        or remaining_attempts < 1
                        or token_param in pinned
                        or token_param in pinned_body
                    ):
                        raise
                    adjusted_request_params = dict(request_params)
                    adjusted_request_params[token_param] = adjusted_max_tokens
                    adjusted_extra_body = dict(adjusted_request_params.get('extra_body') or {})
                    # SDK extra_body wins on the wire. Remove a stale output limit
                    # rather than sending the rejected value again.
                    adjusted_extra_body.pop(token_param, None)
                    if (
                        self.inject_request_id or 'request_id' in adjusted_extra_body
                    ) and 'request_id' not in pinned_body:
                        adjusted_extra_body['request_id'] = f'nexus_{uuid.uuid4().hex}'
                    if adjusted_extra_body or 'extra_body' in adjusted_request_params:
                        adjusted_request_params['extra_body'] = adjusted_extra_body
                    logger.warning(
                        'Infer Proxy reduced %s from %s to %s after a context-window rejection',
                        token_param,
                        effective[token_param],
                        adjusted_max_tokens,
                    )
                    response = await send(adjusted_request_params, attempts=remaining_attempts)
                result = ExtendedChatCompletion.from_openai(response)
                # Enrich with reasoning_content (not covered by standard from_openai)
                for orig, ext in zip(response.choices, result.choices, strict=False):
                    explicit_reasoning = getattr(orig.message, 'reasoning_content', None)
                    if (
                        self.split_think_content
                        and isinstance(getattr(ext.message, 'content', None), str)
                        and '</think>' in ext.message.content
                    ):
                        embedded_reasoning, visible_content = _split_embedded_reasoning(
                            getattr(ext.message, 'content', None)
                        )
                        # Keep private chain-of-thought out of the visible Anthropic
                        # text block. An explicit provider field remains authoritative.
                        ext.message.content = visible_content or None
                        if explicit_reasoning is None:
                            explicit_reasoning = embedded_reasoning
                    ext.message.reasoning_content = explicit_reasoning
                    # This endpoint can return valid named tool calls with
                    # finish_reason="stop". Normalize locally so AnthropicProxy maps
                    # the turn to stop_reason="tool_use" and Claude Code executes it.
                    if self.normalize_tool_finish_reason and ext.message.tool_calls and ext.finish_reason == 'stop':
                        ext.finish_reason = 'tool_calls'
                return result
            except openai.APITimeoutError as e:
                raise TimeoutError(f'Infer Proxy request timed out after {self.timeout}s: {e}') from e
            except openai.BadRequestError as e:
                error_message = str(e)
                if (
                    'context_length_exceeded' in error_message.lower()
                    or 'maximum context length' in error_message.lower()
                ):
                    raise ContextLengthExceededError(
                        f'Infer Proxy context length exceeded: {error_message}', original_error=e
                    ) from e
                raise

    return InferProxyAdapter


# Host handler: lazy loading keeps the guest free of host dependencies.


def _terminal_error_code(exc: BaseException) -> str | None:
    # Host-only dependencies: the sandbox imports the adapter without core clients.
    from nexus.core.openai_client import (
        CONTENT_POLICY_VIOLATION_CODE,
        CONTEXT_LENGTH_EXCEEDED_CODE,
        ContentPolicyViolationError,
        ContextLengthExceededError,
    )

    if isinstance(exc, ContextLengthExceededError):
        return CONTEXT_LENGTH_EXCEEDED_CODE
    if isinstance(exc, ContentPolicyViolationError):
        return CONTENT_POLICY_VIOLATION_CODE
    return None


def span_context(*args, **kwargs):
    # Tracing is host-only; importing the guest assembly must not load nexus.core.
    from nexus.trace import span_context as trace_span_context

    return trace_span_context(*args, **kwargs)


def _load_handler_class():
    import asyncio
    import contextlib
    import copy
    import json
    import os

    from typing import Any, cast

    from nexus.runtime.server.services.llm_handler import LLMRequestHandler as _BaseLLMRequestHandler
    from nexus.runtime.server.services.llm_handler import sanitize_openai_params

    final_replay_grace_seconds = float(os.environ.get('NEXUS_FINAL_REPLAY_GRACE_SECONDS') or 90.0)

    class OSWorldLLMRequestHandler(_BaseLLMRequestHandler):
        _guard: RPCReplayGuard

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.openai_client = adapt_infer_proxy_client(self.openai_client)
            self._unacknowledged_reasoning_frontiers = {}
            self._reasoning_frontier_limit = 32
            self._active_request_tasks = {}
            self._guard = RPCReplayGuard(ttl_seconds=300.0, max_entries=512, max_completed_bytes=64 * 1024 * 1024)
            self._emit_timeout = RSIOLLMConfig().emit_timeout
            self._delivery_timeout = RSIOLLMConfig().report_timeout
            self._delivery_retry_interval = 1.0
            self._final_delivery_tasks = {}

        def _inject_thinking_blocks(self, messages: list[dict[str, Any]]) -> int:
            """Restore thinking state (blocks and/or message-level signatures).

            Returns the number of assistant messages that received an injection.
            Matching is intentionally conservative: tool-call turns require both
            tool_call_id and exact plaintext content to match. Plain assistant turns
            without tool calls can match by unique exact plaintext content. Missing
            matches are skipped rather than falling back to turn position, because a
            wrong signed thinking block is worse than none.
            """
            if not self._thinking_cache:
                return 0
            injected = 0
            used = [False] * len(self._thinking_cache)
            for msg in messages:
                if msg.get('role') != 'assistant':
                    continue
                if self._has_replayable_thinking(msg):
                    continue

                entry = self._find_cached_thinking_entry(msg=msg, used=used)
                if entry is None:
                    continue

                restored = False
                if entry.get('blocks'):
                    # deepcopy, not [dict(b) ...]: blocks carry nested payloads (e.g. per-block
                    # metadata) that a shallow copy would leave aliased to the cache, letting a
                    # downstream mutation of the replayed message corrupt later replays.
                    msg['thinking_blocks'] = copy.deepcopy(entry['blocks'])
                    restored = True
                if entry.get('provider_specific_fields') and not msg.get('provider_specific_fields'):
                    msg['provider_specific_fields'] = copy.deepcopy(entry['provider_specific_fields'])
                    restored = True
                if restored:
                    injected += 1
            return injected

        def _register_reasoning_frontier(self, frontier_id: str | None) -> str | None:
            """Protect a completed response until its turn appears in an input history."""
            if not frontier_id:
                return None
            self._unacknowledged_reasoning_frontiers.pop(frontier_id, None)
            self._unacknowledged_reasoning_frontiers[frontier_id] = None
            while len(self._unacknowledged_reasoning_frontiers) > self._reasoning_frontier_limit:
                expired_id = next(iter(self._unacknowledged_reasoning_frontiers))
                self._unacknowledged_reasoning_frontiers.pop(expired_id, None)
                # This is a hard bound for responses that never return through a
                # scaffold. Acknowledged historical entries have already left this
                # registry and remain available until ordinary compaction pruning.
                self._thinking_cache = [
                    entry for entry in self._thinking_cache if entry.get('_frontier_id') != expired_id
                ]
                self._reasoning_items_cache = [
                    entry for entry in self._reasoning_items_cache if entry.get('_frontier_id') != expired_id
                ]
            return frontier_id

        def _cache_thinking_from_response(
            self,
            response_data: dict[str, Any],
            *,
            frontier_id: str | None = None,
        ) -> tuple[int, str | None]:
            """Cache signed blocks and message-level signatures with a bounded frontier."""
            choices = response_data.get('choices') or []
            if not choices:
                return 0, None
            msg = choices[0].get('message') or {}
            blocks_raw = [b for b in msg.get('thinking_blocks') or [] if isinstance(b, dict)]
            # Anthropic-style providers sign per block: keep any signed block, plus
            # redacted_thinking blocks (signed implicitly by their opaque ``data``), and drop
            # unsigned ones — replaying an unsigned block is rejected by the provider.
            blocks = [
                copy.deepcopy(block)
                for block in blocks_raw
                if block.get('signature') or (block.get('type') == 'redacted_thinking' and block.get('data'))
            ]

            signatures = self._message_thought_signatures(msg)
            provider_fields = {'thought_signatures': signatures} if signatures else None
            # Gemini signs at the message level and leaves every block unsigned, so the
            # per-block filter above drops all of them. Keep the unsigned blocks in that case:
            # they only make sense replayed together with the message-level signatures.
            if provider_fields and not blocks:
                blocks = [copy.deepcopy(block) for block in blocks_raw]

            if not blocks and not provider_fields:
                return 0, None

            tc_id: str | None = None
            for tc in msg.get('tool_calls') or []:
                if (tc or {}).get('id'):
                    tc_id = str(tc['id'])
                    break
            protected_frontier_id = self._register_reasoning_frontier(frontier_id)
            entry = {
                'tool_call_id': tc_id,
                'content': self._plaintext_content(msg),
                'blocks': blocks,
                'provider_specific_fields': provider_fields,
            }
            if protected_frontier_id is not None:
                entry['_frontier_id'] = protected_frontier_id
            self._thinking_cache.append(entry)
            return len(blocks) or len(signatures), tc_id

        @staticmethod
        def _normalized_tool_calls_key(tool_calls: list[dict[str, Any]]) -> tuple[tuple[Any, Any, Any], ...]:
            """Hashable form of :meth:`_tool_calls_for_match` output."""
            return tuple((call.get('id'), call.get('name'), call.get('arguments')) for call in tool_calls)

        def _prune_unreachable_reasoning_state(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
            """Drop opaque state for assistant turns removed by context compaction.

            A short unrelated request (for example a summarizer/helper call) must not
            erase the main conversation's signed state. Therefore each cache is
            pruned only when the incoming history contains an exact anchor for that
            cache. Once anchored, older entries that cannot match any assistant
            message in the current history are no longer reachable by its injector
            and are safe to release.

            Returns ``(thinking_removed, reasoning_removed)``.
            """
            if not self._thinking_cache and not self._reasoning_items_cache:
                return 0, 0

            assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get('role') == 'assistant']
            if not assistant_messages:
                return 0, 0

            # Build replay-surface indexes once. A pairwise cache×history scan would
            # become unnecessarily expensive near a 300-turn context.
            thinking_surfaces: set[tuple[str | None, str | None]] = set()
            reasoning_content_surfaces: set[str] = set()
            reasoning_tool_surfaces: dict[
                tuple[tuple[str, ...], tuple[tuple[Any, Any, Any], ...]],
                set[str | None],
            ] = {}
            for msg in assistant_messages:
                content = self._plaintext_content(msg)
                tool_call_id = self._first_tool_call_id(msg)
                if tool_call_id or content:
                    thinking_surfaces.add((tool_call_id, content))
                if content:
                    reasoning_content_surfaces.add(content)

                tool_call_ids = self._tool_call_ids(msg)
                if tool_call_ids:
                    key = (
                        tuple(tool_call_ids),
                        self._normalized_tool_calls_key(self._tool_calls_for_match(msg)),
                    )
                    reasoning_tool_surfaces.setdefault(key, set()).add(content)

            thinking_keep = [
                bool(entry.get('blocks')) and (entry.get('tool_call_id'), entry.get('content')) in thinking_surfaces
                for entry in self._thinking_cache
            ]
            reasoning_keep: list[bool] = []
            for entry in self._reasoning_items_cache:
                if not entry.get('items'):
                    reasoning_keep.append(False)
                    continue
                entry_tool_call_ids = entry.get('tool_call_ids') or []
                if not entry_tool_call_ids:
                    reasoning_keep.append(
                        bool(entry.get('content')) and entry.get('content') in reasoning_content_surfaces
                    )
                    continue
                key = (
                    tuple(entry_tool_call_ids),
                    self._normalized_tool_calls_key(entry.get('tool_calls') or []),
                )
                message_contents = reasoning_tool_surfaces.get(key)
                entry_content = entry.get('content')
                reasoning_keep.append(
                    message_contents is not None
                    and (
                        not entry_content
                        or any(
                            not message_content or message_content == entry_content
                            for message_content in message_contents
                        )
                    )
                )

            old_thinking_count = len(self._thinking_cache)
            old_reasoning_count = len(self._reasoning_items_cache)
            protected_frontiers = set(self._unacknowledged_reasoning_frontiers)
            matched_frontiers = {
                entry.get('_frontier_id')
                for entries, keep_flags in (
                    (self._thinking_cache, thinking_keep),
                    (self._reasoning_items_cache, reasoning_keep),
                )
                for entry, keep in zip(entries, keep_flags, strict=True)
                if keep and entry.get('_frontier_id')
            }

            # Arm the two caches independently: a Responses-reasoning anchor says
            # nothing about whether an Anthropic thinking entry belongs to this
            # request, and vice versa. Every unacknowledged response frontier remains
            # protected, so overlapping calls cannot delete one another's signatures.
            if any(thinking_keep):
                self._thinking_cache = [
                    entry
                    for entry, keep in zip(self._thinking_cache, thinking_keep, strict=True)
                    if keep or entry.get('_frontier_id') in protected_frontiers
                ]
            if any(reasoning_keep):
                self._reasoning_items_cache = [
                    entry
                    for entry, keep in zip(self._reasoning_items_cache, reasoning_keep, strict=True)
                    if keep or entry.get('_frontier_id') in protected_frontiers
                ]
            for frontier_id in matched_frontiers:
                self._unacknowledged_reasoning_frontiers.pop(frontier_id, None)
            return (
                old_thinking_count - len(self._thinking_cache),
                old_reasoning_count - len(self._reasoning_items_cache),
            )

        def _cache_reasoning_items_from_response(
            self,
            response_data: dict[str, Any],
            *,
            frontier_id: str | None = None,
        ) -> tuple[int, str | None]:
            """Cache this assistant turn's Responses reasoning_items for replay."""
            total_items = 0
            first_key: str | None = None

            protected_frontier_id: str | None = None
            for choice in response_data.get('choices') or []:
                msg = (choice or {}).get('message') or {}
                raw_items = msg.get('reasoning_items') or []
                if not isinstance(raw_items, list):
                    continue

                items = [
                    normalized
                    for item in raw_items
                    if isinstance(item, dict)
                    for normalized in [self._normalize_reasoning_item(item)]
                    if normalized is not None
                ]
                if not items:
                    continue

                tool_call_ids = self._tool_call_ids(msg)
                tool_calls = self._tool_calls_for_match(msg)
                content = self._plaintext_content(msg)
                if (
                    os.environ.get('NEXUS_LLM_DEDUP_RS_ID', '').strip().lower() in {'1', 'true', 'yes', 'on'}
                    and len(items) > 1
                ):
                    _seen_ids: set[str] = set()
                    _deduped: list[dict[str, Any]] = []
                    for _it in items:
                        _rid = _it.get('id')
                        if _rid and _rid in _seen_ids:
                            logger.warning(
                                'rs_dedup: dropping duplicate rs_id=%s (%d bytes encrypted_content)',
                                _rid,
                                len(_it.get('encrypted_content') or ''),
                            )
                            continue
                        if _rid:
                            _seen_ids.add(_rid)
                        _deduped.append(_it)
                    items = _deduped
                if protected_frontier_id is None:
                    protected_frontier_id = self._register_reasoning_frontier(frontier_id)
                entry = {
                    'tool_call_ids': tool_call_ids,
                    'tool_calls': tool_calls,
                    'content': content,
                    'items': items,
                    'output_items': copy.deepcopy(
                        (msg.get('provider_specific_fields') or {}).get('responses_output_items')
                    ),
                }
                if protected_frontier_id is not None:
                    entry['_frontier_id'] = protected_frontier_id
                self._reasoning_items_cache.append(entry)
                total_items += len(items)
                if first_key is None:
                    first_key = ','.join(tool_call_ids) if tool_call_ids else '<content>'

            return total_items, first_key

        async def _produce_response(
            self, request_id: str, data: dict[str, Any], *, rsio_mode: bool = True
        ) -> dict[str, Any]:
            """Run the host-side LLM call and return the response dump.

            ``data`` has already been copied and had ``rsio`` / ``request_id``
            stripped by :meth:`handle_llm_request`. Raises on failure. The periodic
            ``running`` heartbeat is emitted only when the request selects RSIO.
            """
            responses_body = data.pop('_osworld_responses_body', None)
            if responses_body is not None:
                create_responses = getattr(self.openai_client, 'create_responses', None)
                if not callable(create_responses):
                    raise ValueError('This run has no Responses passthrough client configured')
                heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
                try:
                    return await create_responses(responses_body)
                finally:
                    if heartbeat is not None:
                        heartbeat.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat

            operation = data.pop('_nexus_operation', None)
            if operation == 'responses_compact':
                request_body = data.pop('request_body', None)
                compact = getattr(self.openai_client, 'compact_response', None)
                supports_native = bool(getattr(self.openai_client, 'supports_native_responses_compaction', False))
                if not supports_native or not callable(compact):
                    raise RuntimeError(
                        'configured OpenAI client does not support provider-native Responses compaction; '
                        'HyTuner training must use local summary compaction'
                    )
                if not isinstance(request_body, dict) or data:
                    raise ValueError('invalid Responses compaction bridge payload')
                heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
                try:
                    with span_context(
                        'agent.compaction',
                        attributes={
                            'type': 'agent_compaction',
                            'compacted': False,
                            'source': 'provider_native',
                            'trainable': False,
                        },
                    ) as compaction_span:
                        response = await compact(request_body)
                        output = response.get('output') if isinstance(response, dict) else None
                        compaction_items = (
                            [item for item in output if isinstance(item, dict) and item.get('type') == 'compaction']
                            if isinstance(output, list)
                            else []
                        )
                        if (
                            not isinstance(output, list)
                            or len(compaction_items) != 1
                            or not isinstance(compaction_items[0].get('id'), str)
                            or not isinstance(compaction_items[0].get('encrypted_content'), str)
                        ):
                            raise ValueError(
                                'native Responses compaction must return exactly one compaction item among '
                                'the output items, with string id and encrypted_content'
                            )
                        compaction_span.update(attributes={'compacted': True})
                        return response
                finally:
                    if heartbeat is not None:
                        heartbeat.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat
            if operation is not None:
                raise ValueError(f'unsupported Nexus LLM operation: {operation}')

            # Proxies own compaction classification. Consume their transport-only metadata before
            # sanitization and provider dispatch; message text must not override that classification.
            is_agent_compaction = data.pop(NEXUS_AGENT_COMPACTION_FIELD, False) is True
            runtime_detector = data.pop(NEXUS_COMPACTION_DETECTOR_FIELD, None)
            compaction_detector = runtime_detector if isinstance(runtime_detector, str) else 'runtime_marker'
            client_supports_request_meta = getattr(self.openai_client, 'supports_request_meta', False)
            if is_agent_compaction and client_supports_request_meta:
                request_meta = data.get('request_meta')
                if request_meta is None:
                    request_meta = {}
                elif not isinstance(request_meta, dict):
                    raise TypeError('Compaction request_meta must be a dictionary')
                data['request_meta'] = {
                    **request_meta,
                    'invocation_kind': 'summary_inference',
                }

            requested_model = data.get('model')
            host_request_args = self.openai_client.config.request_args
            host_model = host_request_args.get('model')
            if host_model:
                data.pop('model', None)
            # Preserve the historical handler contract: request_args.reasoning_effort
            # is host-pinned, while max_tokens remains a default unless the client
            # explicitly declares it under enforced_request_args.
            if host_request_args.get('reasoning_effort') is not None:
                data.pop('reasoning_effort', None)

            try:
                client_has_thinking = data.pop('_client_has_thinking', None)
                client_requests_haiku = data.pop('_client_requests_haiku', None)

                if client_has_thinking is False and self.no_thinking_request_args is not None:
                    no_think_args = copy.deepcopy(self.no_thinking_request_args)
                    if self.no_thinking_request_args_mode == 'inherit':
                        no_think_overrides = no_think_args
                        no_think_args = copy.deepcopy(host_request_args)
                        no_think_args.update(no_think_overrides)
                        for field in ('thinking', 'reasoning_effort', 'output_config'):
                            if field not in no_think_overrides:
                                no_think_args.pop(field, None)
                        extra_body = no_think_args.get('extra_body')
                        override_extra_body = no_think_overrides.get('extra_body')
                        if isinstance(extra_body, dict):
                            extra_body = dict(extra_body)
                            for field in ('thinking', 'output_config'):
                                if not isinstance(override_extra_body, dict) or field not in override_extra_body:
                                    extra_body.pop(field, None)
                            if extra_body:
                                no_think_args['extra_body'] = extra_body
                            else:
                                no_think_args.pop('extra_body', None)
                    if client_requests_haiku and self.haiku_model:
                        no_think_args.pop('model', None)
                    data['_request_args_override'] = no_think_args

                if client_requests_haiku and self.haiku_model:
                    data['model'] = self.haiku_model

                if 'request_meta' in data and not client_supports_request_meta:
                    data.pop('request_meta', None)

                final_model = data.get('model') or host_model or requested_model
                thinking_str = 'no_thinking' if client_has_thinking is False else 'thinking'
                haiku_str = ' [haiku]' if client_requests_haiku else ''
                logger.info(
                    f'Handling LLM request: {request_id}, '
                    f'requested_model={requested_model}, final_model={final_model}, '
                    f'{thinking_str}{haiku_str}'
                )

                request_data = sanitize_openai_params(data)

                if self.preserve_thinking and isinstance(request_data.get('messages'), list):
                    thinking_pruned, reasoning_pruned = self._prune_unreachable_reasoning_state(
                        request_data['messages']
                    )
                    if thinking_pruned or reasoning_pruned:
                        logger.info(
                            'preserve_reasoning: released compacted state (thinking_entries=%d reasoning_entries=%d) [%s]',
                            thinking_pruned,
                            reasoning_pruned,
                            request_id,
                        )
                    injected = self._inject_thinking_blocks(request_data['messages'])
                    if injected:
                        logger.info(
                            'preserve_thinking: injected thinking_blocks into %d assistant message(s) [%s]',
                            injected,
                            request_id,
                        )
                    reasoning_injected = self._inject_reasoning_items(request_data['messages'])
                    if reasoning_injected:
                        logger.info(
                            'preserve_reasoning: injected reasoning_items into %d assistant message(s) [%s]',
                            reasoning_injected,
                            request_id,
                        )
                    tampered = self._maybe_tamper_for_test(request_data['messages'])
                    if tampered is not None:
                        logger.warning(
                            'preserve_thinking: TAMPERED signature on assistant turn #%d for diagnostics [%s]',
                            tampered,
                            request_id,
                        )

                if isinstance(request_data.get('messages'), list):
                    unsigned_dropped = self._drop_unreplayable_thinking_blocks(request_data['messages'])
                    if unsigned_dropped:
                        logger.debug(
                            'dropped %d unsigned thinking block(s) before replay [%s]',
                            unsigned_dropped,
                            request_id,
                        )

                heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
                _dump_dir = os.getenv('NEXUS_DUMP_LLM_REQ_DIR')
                if _dump_dir:
                    try:
                        os.makedirs(_dump_dir, exist_ok=True)
                        _dump_path = os.path.join(_dump_dir, f'req_{request_id}.json')
                        with open(_dump_path, 'w') as _fh:
                            json.dump(request_data, _fh, default=str)
                        logger.debug(
                            '[DUMP] wrote request payload -> %s (bytes=%d)', _dump_path, os.path.getsize(_dump_path)
                        )
                    except Exception as _e:
                        logger.debug('[DUMP] failed to dump request payload: %s', _e)
                try:
                    logger.debug(
                        '[DUMP] calling create_completion request_id=%s model=%s stream=%s tools=%d msgs=%d',
                        request_id,
                        request_data.get('model'),
                        request_data.get('stream'),
                        len(request_data.get('tools') or []),
                        len(request_data.get('messages') or []),
                    )

                    async def _complete() -> Any:
                        # The hosted-tool handler, when configured, serves the request itself
                        # (server-side tools); None means it declined and the ordinary
                        # completion path applies.
                        return await self.openai_client.create_completion(**request_data)

                    if is_agent_compaction:
                        with span_context(
                            'agent.compaction',
                            attributes={
                                'type': 'agent_compaction',
                                'compacted': False,
                                'source': 'runtime_marker',
                                'detector': compaction_detector,
                            },
                        ) as compaction_span:
                            response = await _complete()
                            compaction_span.update(attributes={'compacted': True})
                    else:
                        response = await _complete()
                    logger.debug('[DUMP] create_completion returned request_id=%s', request_id)
                finally:
                    if heartbeat is not None:
                        heartbeat.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat

                response_dump = cast(Any, response).model_dump() if hasattr(response, 'model_dump') else response

                if self.preserve_thinking and isinstance(response_dump, dict):
                    frontier_id = request_id if request_id != '<no-request-id>' else None
                    cached_n, cached_key = self._cache_thinking_from_response(
                        response_dump,
                        frontier_id=frontier_id,
                    )
                    if cached_n:
                        logger.info(
                            'preserve_thinking: cached %d thinking block(s) (key=%s) [%s]',
                            cached_n,
                            cached_key or '<content>',
                            request_id,
                        )
                    reasoning_cached_n, reasoning_cached_key = self._cache_reasoning_items_from_response(
                        response_dump,
                        frontier_id=frontier_id,
                    )
                    if reasoning_cached_n:
                        logger.info(
                            'preserve_reasoning: cached %d reasoning item(s) (key=%s) [%s]',
                            reasoning_cached_n,
                            reasoning_cached_key or '<unknown>',
                            request_id,
                        )

                # Notify host-side tool_call observer (used by kilo to detect 'suggest').
                if self.on_tool_call is not None and isinstance(response_dump, dict):
                    for choice in response_dump.get('choices') or []:
                        message = (choice or {}).get('message') or {}
                        for tc in message.get('tool_calls') or []:
                            fn = (tc or {}).get('function') or {}
                            fn_name = fn.get('name') or ''
                            if fn_name:
                                try:
                                    self.on_tool_call(fn_name)
                                except Exception:
                                    logger.exception('on_tool_call callback raised; ignoring')

                if self.openai_client.config.client_args.get('log_llm_usage'):
                    rd = response_dump
                    usage = (rd.get('usage') or {}) if isinstance(rd, dict) else {}
                    cache_read = usage.get('cache_read_input_tokens', 0) or 0
                    cache_creation = usage.get('cache_creation_input_tokens', 0) or 0
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0) or 0
                    total_cached = cache_read + cache_creation
                    hit_rate = (cache_read / total_cached * 100) if total_cached > 0 else 0
                    prompt_details = usage.get('prompt_tokens_details') or {}
                    input_details = usage.get('input_tokens_details') or {}
                    cached_tokens = 0
                    if isinstance(prompt_details, dict):
                        cached_tokens = (
                            prompt_details.get('gpu_cached_tokens', 0) or prompt_details.get('cached_tokens', 0) or 0
                        )
                    if not cached_tokens and isinstance(input_details, dict):
                        cached_tokens = (
                            input_details.get('gpu_cached_tokens', 0) or input_details.get('cached_tokens', 0) or 0
                        )
                    cached_input_rate = (
                        (cached_tokens / input_tokens * 100) if input_tokens > 0 and cached_tokens > 0 else 0
                    )
                    logger.info(
                        'LLM usage [%s]: input=%d output=%d cache_read=%d cache_creation=%d hit_rate=%.1f%% cached_tokens=%d cached_input_rate=%.1f%%',
                        request_id,
                        input_tokens,
                        output_tokens,
                        cache_read,
                        cache_creation,
                        hit_rate,
                        cached_tokens,
                        cached_input_rate,
                    )

                return response_dump

            except Exception as e:
                logger.error(f'LLM request {request_id} failed: {e}')
                raise

        async def _emit_running(self, request_id: str, *, kind: str = 'heartbeat') -> bool:
            """Emit one ``running`` report for *request_id*.

            ``kind`` is only for the local log line (``ack`` vs ``heartbeat``); the
            wire payload is identical. Both submit acknowledgements and periodic
            heartbeats stay at DEBUG to avoid flooding normal logs. Only ever
            called in RSIO mode; one-shot clients expect a final response with
            a ``success`` field, not a ``running`` report.

            Delivery is best-effort: if the socket is currently disconnected the
            report is dropped silently. The worker's rsio driver will resend the
            original request when it stops hearing heartbeats, and the replay
            guard will short-circuit the duplicate to the cached final anyway.
            """
            logger.debug('rsio.host: running %s request_id=%s', kind, request_id)
            try:
                await asyncio.wait_for(
                    self.runtime.emit('llm_response', {'request_id': request_id, 'rsio': {'kind': 'running'}}),
                    timeout=self._emit_timeout,
                )
            except Exception as exc:
                logger.warning('rsio.host: running %s emit dropped request_id=%s (%s)', kind, request_id, exc)
                return False
            return True

        async def _emit_final(
            self,
            request_id: str,
            *,
            success: bool,
            response: dict[str, Any] | None = None,
            error: str | None = None,
            error_code: str | None = None,
            rsio_mode: bool = True,
        ) -> bool:
            """Emit the terminal ``llm_response`` for *request_id*.

            Retry delivery of the already-computed payload within a bounded window.
            No provider invocation happens here. Silent packet loss still uses the
            worker's same-id replay; cancellation and teardown stop delivery retries.
            True means Socket.IO accepted the emit, not a worker ACK.
            """
            payload: dict[str, Any] = {'request_id': request_id, 'success': success}
            if rsio_mode:
                # Envelope fields are emitted only for an RSIO request.
                payload['rsio'] = {'kind': 'final'}
            if success:
                payload['response'] = response
            else:
                payload['error'] = error or 'Unknown error'
                if error_code:
                    # Machine-readable companion to the free-text ``error``. The worker turns it back
                    # into a typed failure so the in-sandbox proxy can name the cause to its agent,
                    # which otherwise only ever sees an opaque HTTP 500. Old workers ignore it.
                    payload['error_code'] = error_code
            owner = asyncio.current_task()
            if owner is not None:
                self._final_delivery_tasks.setdefault(request_id, set()).add(owner)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._delivery_timeout
            attempt = 0
            try:
                while loop.time() < deadline:
                    attempt += 1
                    try:
                        await asyncio.wait_for(
                            self.runtime.emit('llm_response', payload),
                            timeout=min(self._emit_timeout, max(0, deadline - loop.time())),
                        )
                        logger.debug(
                            'rsio.host: final sent request_id=%s success=%s delivery_attempt=%s (not worker ACK)',
                            request_id,
                            success,
                            attempt,
                        )
                        return True
                    except Exception as exc:
                        logger.warning(
                            'rsio.host: final delivery failed request_id=%s attempt=%s (%s); retrying same response',
                            request_id,
                            attempt,
                            exc,
                        )
                    # RemoteRuntime's explicit reconnect is serialized and restores
                    # subscriptions if its Socket.IO client has to be replaced.
                    reconnect = getattr(self.runtime, 'ws_connect', None)
                    if getattr(self.runtime, 'ws_connected', None) is False and callable(reconnect):
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(
                                reconnect(),
                                timeout=min(self._emit_timeout, max(0, deadline - loop.time())),
                            )
                    await asyncio.sleep(min(self._delivery_retry_interval, max(0, deadline - loop.time())))
                logger.warning(
                    'rsio.host: delivery window expired request_id=%s; cached same-id replay remains', request_id
                )
                return False
            finally:
                tasks = self._final_delivery_tasks.get(request_id)
                if tasks is not None and owner is not None:
                    tasks.discard(owner)
                    if not tasks:
                        self._final_delivery_tasks.pop(request_id, None)

        async def _handle_llm_request(self, data: dict[str, Any]) -> None:
            """Handle an LLM request from the runtime side.

            Dedups overlapping retries by ``request_id`` (same host process), emits
            ``running`` heartbeats while the model call runs, and always attempts a
            terminal ``llm_response``. A completed replay result is retained until
            that final emit completes without raising.

            Requests with an ``rsio`` envelope receive running heartbeats and
            framed final responses. Requests without it receive only a plain final.
            """
            data = dict(data)
            rsio = data.pop('rsio', None)  # protocol envelope — never forward to the provider
            rsio_mode = rsio is not None  # No envelope selects one-shot responses.
            request_id = data.pop('request_id', None) or '<no-request-id>'
            attempt = (rsio or {}).get('attempt')
            logger.debug(
                'rsio.host: recv llm_request request_id=%s attempt=%s mode=%s',
                request_id,
                attempt,
                'rsio' if rsio_mode else 'legacy',
            )

            ran = False

            async def once() -> dict[str, Any]:
                nonlocal ran
                ran = True
                owner_task = asyncio.current_task()
                if owner_task is None:  # pragma: no cover - asyncio always has one here
                    raise RuntimeError('LLM request handler has no owning asyncio task')
                self._active_request_tasks[request_id] = owner_task
                try:
                    # Ack with a `running` report before starting the model call, and
                    # only on the owning frame (guard runs `once` once per request_id).
                    # This tells the caller "received, working on it" so its probe window
                    # switches from ack_timeout to the longer report_timeout — a normal
                    # multi-second LLM call is no longer mistaken for a lost request and
                    # needlessly resent. In-flight/cached resends don't re-ack: the owner
                    # already did, and a resend gets the final directly anyway.
                    # Heartbeats require an RSIO request envelope.
                    if rsio_mode:
                        await self._emit_running(request_id, kind='ack')
                    return await self._produce_response(request_id, data, rsio_mode=rsio_mode)
                finally:
                    if self._active_request_tasks.get(request_id) is owner_task:
                        self._active_request_tasks.pop(request_id, None)

            try:
                result = await self._guard.run('llm_request', request_id, once)
            except Exception as e:
                error_code = _terminal_error_code(e)
                self._record_outcome(request_id, error_code)
                logger.warning('rsio.host: request_id=%s failed (code=%s): %s', request_id, error_code, e)
                delivered = await self._emit_final(
                    request_id,
                    success=False,
                    error=str(e),
                    error_code=error_code,
                    rsio_mode=rsio_mode,
                )
                if delivered:
                    await self._guard.discard('llm_request', request_id, delay=final_replay_grace_seconds)
                return
            self._record_outcome(request_id, None)
            if not ran:
                # Guard replayed an in-flight/cached result: this is a resend of a
                # request already being handled or already completed. The model call
                # was NOT re-executed — we just re-emit the final (a prior final may
                # have been the frame that got lost and triggered this resend).
                logger.info('rsio.host: dedup replay request_id=%s attempt=%s (no model re-call)', request_id, attempt)
            # Re-emit final on every attempt (cache hit or fresh run): a prior final
            # may have been the message that got lost and triggered the resend.
            delivered = await self._emit_final(request_id, success=True, response=result, rsio_mode=rsio_mode)
            if delivered:
                await self._guard.discard('llm_request', request_id, delay=final_replay_grace_seconds)

        async def handle_llm_cancel(self, data: dict[str, Any]) -> None:
            """Cancel and await the provider task for an abandoned RSIO request.

            The acknowledgement is emitted only after the owner task has finished
            unwinding.  Therefore a runtime caller that waits for ``llm_cancelled``
            can safely retry without overlapping the old LiteLLM/Tenacity chain.
            Unknown/already-finished request ids are acknowledged as inactive.
            """
            request_id = str((data or {}).get('request_id') or '')
            if not request_id:
                logger.warning('rsio.host: llm_cancel without request_id, ignoring')
                return

            task = self._active_request_tasks.get(request_id)
            cancelled = task is not None and not task.done()
            # Install the terminal tombstone before cancelling the owner.  The
            # guard distinguishes this explicit protocol cancellation from normal
            # process/shutdown cancellation, and late same-id frames can therefore
            # never resurrect the provider call.
            await self._guard.cancel(
                'llm_request',
                request_id,
                reason=f'LLM request cancelled by runtime (request_id={request_id})',
            )
            deliveries = list(self._final_delivery_tasks.get(request_id, ()))
            for delivery in deliveries:
                if delivery is not asyncio.current_task():
                    delivery.cancel()
            for delivery in deliveries:
                if delivery is not asyncio.current_task():
                    with contextlib.suppress(asyncio.CancelledError):
                        await delivery
            if cancelled and task is not None:
                logger.info('rsio.host: cancelling active request_id=%s', request_id)
                task.cancel()
                if task is not asyncio.current_task():
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            else:
                logger.info('rsio.host: cancel request_id=%s already inactive', request_id)

            try:
                await asyncio.wait_for(
                    self.runtime.emit('llm_cancelled', {'request_id': request_id, 'cancelled': cancelled}),
                    timeout=self._emit_timeout,
                )
            except Exception as exc:
                logger.warning('rsio.host: llm_cancelled emit dropped request_id=%s (%s)', request_id, exc)

        def register(self):
            if self._handler_registered:
                return
            self.runtime.on('llm_cancel', cast(Any, self.handle_llm_cancel))
            super().register()

        def unregister(self):
            for tasks in self._final_delivery_tasks.values():
                for task in tasks:
                    task.cancel()
            if self._handler_registered:
                self.runtime.off('llm_cancel', cast(Any, self.handle_llm_cancel))
            super().unregister()

        async def aclose(self) -> None:
            """Opt-in terminal teardown; ordinary unregister/replay behavior is unchanged.

            Cancellation tombstones prevent late replay frames from restarting a
            model call after this task-scoped handler has closed.
            """
            self.unregister()
            current = asyncio.current_task()
            tasks = {task for task in self._active_request_tasks.values() if task is not current}
            tasks.update(task for group in self._final_delivery_tasks.values() for task in group if task is not current)
            for request_id in tuple(self._active_request_tasks):
                await self._guard.cancel('llm_request', request_id, reason='Native bridge closed')
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    return OSWorldLLMRequestHandler


def __getattr__(name):
    # The guest bundle has no core clients/tracing. Import host code only on demand.
    if name == 'AGSOSWorldBridgeAdapter':
        result = globals().get(name) or _load_deployment_class()
        globals()[name] = result
        return result
    if name == 'BridgeHostClient':
        result = globals().get(name) or _load_host_client_class()
        globals()[name] = result
        return result
    if name == 'LLMRequestHandler':
        handler = globals().get(name) or _load_handler_class()
        globals()[name] = handler
        return handler
    if name == 'sanitize_openai_params':
        from nexus.runtime.server.services.llm_handler import sanitize_openai_params

        return sanitize_openai_params
    raise AttributeError(name)


# Assembly: explicit extension and execution-profile selection.

BUNDLE_REQUIREMENTS = ('PyMuPDF==1.24.14', 'psutil==7.2.1')


def create_handler(**kwargs):
    return __getattr__('LLMRequestHandler')(**kwargs)


def claude_bridge_options(options, client):
    """Model/client policy for the shared Claude Code execution profile."""
    thinking = options.thinking_preset
    if thinking == 'auto':
        thinking = 'bedrock-claude' if getattr(client, 'native_request_protocol', None) == 'anthropic' else 'hunyuan'
    return {
        'timeout': options.timeout,
        'preserve_thinking': True,
        'api_protocol': 'anthropic',
        'thinking_config': thinking or 'hunyuan',
    }


def codex_bridge_options(options, client):
    """Native Responses is capability-selected; other clients retain Chat conversion."""
    native = callable(getattr(client, 'create_responses', None))
    return {
        'timeout': options.timeout,
        'preserve_thinking': False,
        'api_protocol': 'osworld-responses' if native else 'responses-chat',
        'bridge_request_timeout': 300.0,
    }


def create_app(*, sio, api_protocol, request_timeout, conversion_log_level, thinking_config, health_payload):
    from nexus.runtime.server.services.responses_proxy.converter import responses_to_openai
    from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy

    bridge = LLMBridge(sio)
    if api_protocol in {'anthropic', 'osworld-anthropic', 'osworld-responses'}:
        app = create_anthropic_app(
            bridge=bridge,
            request_timeout=request_timeout,
            conversion_log_level=conversion_log_level,
            thinking_config=thinking_config,
            document_mode='images' if api_protocol.startswith('osworld-') else 'file',
            title='Nexus LLM Bridge',
            health_payload=health_payload,
        )
        if api_protocol == 'osworld-responses':
            attach_responses_endpoint(app, bridge, timeout=86400)
    elif api_protocol in {'responses', 'responses-chat'}:

        def convert(*args, **kwargs):
            params = responses_to_openai(*args, **kwargs)
            if params.get('reasoning_effort') == 'none':
                params.pop('reasoning_effort')
            if api_protocol == 'responses-chat':
                for message in params.get('messages', []):
                    message.pop('reasoning_items', None)
                    for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                        image = part.get('image_url')
                        if isinstance(image, dict) and image.get('detail') == 'original':
                            image.pop('detail')
            return params

        proxy = ResponsesProxy(sio=sio, request_timeout=request_timeout, responses_to_openai_converter=convert)
        proxy.bridge = cast(Any, bridge)
        app = proxy.create_app()

        @app.get('/health')
        async def health():
            return health_payload()

    else:
        raise ValueError(f'Unsupported OSWorld bridge API protocol: {api_protocol}')
    return app, bridge


def _stream_keepalive_interval() -> float:
    enabled = os.getenv(_STREAM_KEEPALIVE_ENABLED_ENV)
    if enabled is not None and enabled.strip().lower() in {'0', 'false', 'no', 'off', 'disabled'}:
        return 0.0

    raw = os.getenv(_STREAM_KEEPALIVE_INTERVAL_ENV)
    if not raw:
        return 15.0
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            'Invalid %s=%r; using default stream keepalive interval',
            _STREAM_KEEPALIVE_INTERVAL_ENV,
            raw,
        )
        return 15.0


def _load_deployment_class():
    from nexus.runtime.bridge.adapters.base import BasePosixBridgeAdapter, BridgeConnectionInfo

    class AGSOSWorldBridgeAdapter(BasePosixBridgeAdapter):
        target_label = 'OSWorld'
        # AGS OSWorld's gateway only routes specific ports; 9281 is the one exposed for the bridge.
        bridge_port = 9281

        def __init__(self, runtime: AGSOSWorldExecutorRuntime) -> None:
            from nexus.runtime.providers.ags.osworld import require_ags_osworld_runtime

            super().__init__(require_ags_osworld_runtime(runtime))
            self.runtime: AGSOSWorldRuntime

        def _connection_info(self) -> BridgeConnectionInfo:
            return BridgeConnectionInfo(
                base_url=self.runtime.handle.url(self.bridge_port),
                headers=self.runtime.handle.headers,
            )

        async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
            last_error: Exception | None = None
            for attempt in range(1, OSWORLD_BRIDGE_UPLOAD_ATTEMPTS + 1):
                try:
                    form = aiohttp.FormData()
                    form.add_field('file_path', remote_artifact)
                    with artifact_path.open('rb') as file_data:
                        form.add_field(
                            'file_data',
                            file_data,
                            filename=artifact_path.name,
                            content_type='application/octet-stream',
                        )
                        response = await self.runtime._http_request(
                            'POST',
                            self.runtime.handle.url(self.runtime.server_port, '/setup/upload'),
                            data=form,
                            timeout=aiohttp.ClientTimeout(total=OSWORLD_BRIDGE_UPLOAD_TIMEOUT),
                        )
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f'OSWorld bridge artifact upload failed: HTTP {response.status_code} {response.text[:500]}'
                        )
                    return
                except (aiohttp.ClientError, TimeoutError) as exc:
                    last_error = exc
                    if attempt >= OSWORLD_BRIDGE_UPLOAD_ATTEMPTS:
                        raise
                    await asyncio.sleep(OSWORLD_BRIDGE_UPLOAD_RETRY_DELAY)
            if last_error is not None:
                raise last_error

    return AGSOSWorldBridgeAdapter


def _load_host_client_class():
    import contextlib

    import aiohttp as _aiohttp

    from socketio.exceptions import BadNamespaceError

    from nexus.runtime.bridge.host_client import BridgeHostClient as DefaultHostClient
    from nexus.utils.socketio_proxy import create_socketio_client

    ws_max_msg_size = int(os.getenv('NEXUS_BRIDGE_MAX_MSG_SIZE_MB', '200')) * 1024 * 1024

    def _build_http_session() -> _aiohttp.ClientSession:
        """Build a SOCKS-aware Socket.IO HTTP session.

        NEXUS_BRIDGE_SOCKS_PROXY selects an explicit aiohttp_socks connector.
        Without it, trust_env enables environment HTTP/HTTPS proxies. HTTP
        reachability does not imply WebSocket upgrades are accepted by that proxy.
        SOCKS selection disables ambient HTTP proxy handling to avoid conflicts.
        """
        socks_url = os.getenv('NEXUS_BRIDGE_SOCKS_PROXY', '').strip()
        if not socks_url:
            return _aiohttp.ClientSession(trust_env=True)
        try:
            from aiohttp_socks import ProxyConnector
        except ImportError:
            logger.warning(
                'NEXUS_BRIDGE_SOCKS_PROXY ignored: aiohttp_socks is not installed; falling back to trust_env',
            )
            return _aiohttp.ClientSession(trust_env=True)
        logger.info('LLM bridge Socket.IO via configured SOCKS proxy')
        # trust_env stays off: the ambient http_proxy points at the HTTP listener, and letting it
        # win here would put us back on the transport that rejects the upgrade.
        return _aiohttp.ClientSession(connector=ProxyConnector.from_url(socks_url))

    class OSWorldHostClient(DefaultHostClient):
        def __init__(
            self,
            *,
            base_url: str,
            headers: dict[str, str] | None = None,
            request_timeout: float = 60.0,
            reconnection: bool = True,
            keepalive_interval: float | None = 25.0,
            initial_connect_attempts: int | None = None,
            trust_env: bool = True,
            proxy: str | None = None,
        ) -> None:
            self._initial_connect_attempts = (
                int(os.getenv('NEXUS_BRIDGE_WS_CONNECT_ATTEMPTS', '5'))
                if initial_connect_attempts is None
                else initial_connect_attempts
            )
            if self._initial_connect_attempts < 1:
                raise ValueError('initial_connect_attempts must be positive')
            self._trust_env = trust_env
            self._explicit_proxy = proxy
            self.base_url = base_url.rstrip('/')
            self.headers = headers or {}
            self.request_timeout = request_timeout
            self.keepalive_interval = keepalive_interval
            self._ws_connected = False
            self._connect_lock = asyncio.Lock()
            self._event_listeners: dict[str, list[Callable[..., Awaitable[None]]]] = {}
            self._keepalive_task: asyncio.Task[None] | None = None
            self._reconnection = reconnection
            self._reconnect_task: asyncio.Task[None] | None = None
            self._closed = False
            self._connect_attempted = False
            self._reconnect_interval = 1.0
            self._reconnect_timeout = 5.0
            self._sio = self._create_client()

        def _create_client(self) -> socketio.AsyncClient:
            sio = create_socketio_client(
                trust_env=self._trust_env,
                http_session_factory=(
                    (lambda: _aiohttp.ClientSession(proxy=self._explicit_proxy, trust_env=False))
                    if self._explicit_proxy
                    else _build_http_session
                    if os.getenv('NEXUS_BRIDGE_SOCKS_PROXY', '').strip()
                    else None
                ),
                # One owner must serialize recovery. Socket.IO's own reconnect task
                # otherwise races explicit final-response recovery on a stale client.
                reconnection=False,
                request_timeout=self.request_timeout,
                logger=False,
                engineio_logger=False,
                # aiohttp's websocket max_msg_size defaults to 4 MiB; LLM messages carrying
                # base64 screenshots exceed it (~4.7 MB), and the client aborts the connection
                # ("Message size ... exceeds limit 4194304"), dropping the in-flight request.
                websocket_extra_options={'max_msg_size': ws_max_msg_size},
            )

            @sio.event
            async def connect() -> None:
                if self._sio is not sio or self._closed:
                    return
                self._ws_connected = True
                self._start_keepalive()
                logger.info('LLM bridge host connected: %s', self.base_url)

            @sio.event
            async def disconnect(*_args: Any) -> None:
                if self._sio is not sio:
                    return
                self._ws_connected = False
                await self._stop_keepalive()
                logger.info('LLM bridge host disconnected: %s', self.base_url)
                self._start_reconnect()

            for event in self._event_listeners:
                sio.on(event, self._make_event_dispatcher(event))
            return sio

        def _start_reconnect(self) -> None:
            if self._closed or not self._reconnection:
                return
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

        async def _reconnect_loop(self) -> None:
            attempt = 0
            while not self._closed and not self.ws_connected:
                attempt += 1
                await asyncio.sleep(min(self._reconnect_interval * attempt, 5.0))
                if self._closed:
                    return
                try:
                    await asyncio.wait_for(self.ws_connect(), timeout=self._reconnect_timeout)
                except Exception as exc:
                    # Handshake exceptions can contain headers / credential-bearing
                    # URLs. Keep only the type, not their raw text.
                    logger.warning('LLM bridge reconnect failed: attempt=%d error=%s', attempt, type(exc).__name__)

        async def _close_client(self, sio: socketio.AsyncClient) -> None:
            try:
                await asyncio.wait_for(sio.shutdown(), timeout=self._reconnect_timeout)
                eio = getattr(sio, 'eio', None)
                if eio is not None and eio.state != 'disconnected':
                    await asyncio.wait_for(eio.disconnect(abort=True), timeout=self._reconnect_timeout)
            except Exception as exc:
                logger.warning('LLM bridge client cleanup failed: %s', type(exc).__name__)
            finally:
                # A cancelled/failed handshake can leave an Engine.IO-owned session
                # open even though there is no Socket.IO namespace to disconnect.
                http = getattr(getattr(sio, 'eio', None), 'http', None)
                if http is not None and not http.closed and not sio.eio.external_http:
                    await http.close()

        async def _keepalive_loop(self) -> None:
            try:
                while self._ws_connected and self._sio.connected:
                    await asyncio.sleep(self.keepalive_interval or 0)
                    if not self._ws_connected or not self._sio.connected:
                        break
                    await self._sio.emit('bridge_keepalive', {'time': time.time()})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug('LLM bridge keepalive stopped after emit failure', exc_info=True)
                self._ws_connected = False
                self._start_reconnect()

        async def ws_connect(self) -> None:
            # Final delivery, background recovery and duplicate requests share this
            # lock. Replace stale Engine.IO state but retain handler/replay ownership.
            async with self._connect_lock:
                if self.ws_connected:
                    return
                self._closed = False
                # Only the initial handshake has the team's bounded retry budget.
                # Later recovery is already bounded by the caller/background loop.
                attempts = 1 if self._connect_attempted else self._initial_connect_attempts
                for attempt in range(attempts):
                    if self._connect_attempted:
                        await self._stop_keepalive()
                        old = self._sio
                        self._sio = self._create_client()
                        await self._close_client(old)
                    self._connect_attempted = True
                    self._ws_connected = False
                    try:
                        await self._sio.connect(self.base_url, headers=self.headers or None)
                    except Exception as exc:
                        if attempt + 1 == attempts:
                            raise
                        logger.warning(
                            'LLM bridge handshake retry: attempt=%d/%d error=%s',
                            attempt + 1,
                            attempts,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(min(2**attempt, 8))
                    else:
                        self._ws_connected = True
                        self._start_keepalive()
                        return

        async def ws_disconnect(self) -> None:
            self._closed = True
            task, self._reconnect_task = self._reconnect_task, None
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            async with self._connect_lock:
                self._ws_connected = False
                await self._stop_keepalive()
                await self._close_client(self._sio)

        @property
        def ws_connected(self) -> bool:
            return self._ws_connected and self._sio.connected

        async def emit(self, event: str, *args: Any, ack_callback: Callable[..., None] | None = None) -> None:
            if not self.ws_connected:
                raise RuntimeError('Socket.IO is not connected. Call ws_connect() first.')
            data = args[0] if len(args) == 1 else args if args else None
            try:
                if ack_callback is not None:
                    await self._sio.emit(event, data, callback=ack_callback)
                else:
                    await self._sio.emit(event, data)
            except (BadNamespaceError, ConnectionError):
                self._ws_connected = False
                self._start_reconnect()
                raise

    return OSWorldHostClient


def create_bridge_fastapi_app(
    *,
    api_protocol: str = 'anthropic',
    request_timeout: float = 300,
    conversion_log_level: str = 'INFO',
    thinking_config: str | dict[str, Any] | None = 'hunyuan',
    launch_id: str = '',
) -> tuple[FastAPI, LLMBridge]:
    sio = socketio.AsyncServer(
        async_mode='asgi',
        cors_allowed_origins='*',
        logger=False,
        engineio_logger=False,
        ping_interval=15.0,
        ping_timeout=120.0,
        # Large LLM responses flow host -> bridge; match the client's 200 MiB frame limit
        # so a big completion isn't dropped at the default 1 MiB and killed as a transport error.
        max_http_buffer_size=200 * 1024 * 1024,
    )
    connected_hosts: set[str] = set()

    @sio.on('connect')
    async def on_connect(sid: str, environ: dict | None = None, auth: dict | None = None) -> None:
        connected_hosts.add(str(sid))
        logger.info('LLM bridge host connected: sid=%s hosts=%d', sid, len(connected_hosts))

    @sio.on('disconnect')
    async def on_disconnect(sid: str) -> None:
        connected_hosts.discard(str(sid))
        logger.info('LLM bridge host disconnected: sid=%s hosts=%d', sid, len(connected_hosts))

    # _ServerSio is a dependency-free SIOBridge stand-in; the proxies only use on()/emit().
    bridge_sio = cast('SIOBridge', _ServerSio(sio))
    bridge = None

    def health_payload():
        return {
            'status': 'ok',
            'bridge': 'nexus-llm-bridge',
            'launch_id': launch_id,
            'connected_hosts': len(connected_hosts),
            'pending_requests': len(getattr(bridge, '_pending', {})),
        }

    app, bridge = create_app(
        sio=bridge_sio,
        api_protocol=api_protocol,
        request_timeout=request_timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_config,
        health_payload=health_payload,
    )
    app.state.sio = sio
    app.state.bridge = bridge
    return app, bridge


def create_bridge_asgi_app(
    *,
    api_protocol: str = 'anthropic',
    request_timeout: float = 300,
    conversion_log_level: str = 'INFO',
    thinking_config: str | dict[str, Any] | None = 'hunyuan',
    launch_id: str = '',
) -> socketio.ASGIApp:
    fastapi_app, _bridge = create_bridge_fastapi_app(
        api_protocol=api_protocol,
        request_timeout=request_timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_config,
        launch_id=launch_id,
    )
    return socketio.ASGIApp(fastapi_app.state.sio, other_asgi_app=fastapi_app)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Nexus LLM Bridge')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--api-protocol', default='anthropic')
    parser.add_argument('--request-timeout', type=float, default=300.0)
    parser.add_argument('--conversion-log-level', default='INFO')
    parser.add_argument('--thinking-config', default='hunyuan')
    parser.add_argument('--launch-id', default='')
    parser.add_argument('--log-level', default='info')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    app = create_bridge_asgi_app(
        api_protocol=args.api_protocol,
        request_timeout=args.request_timeout,
        conversion_log_level=args.conversion_log_level,
        thinking_config=args.thinking_config,
        launch_id=args.launch_id,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # Allow large websocket frames (LLM responses); uvicorn defaults to 16 MiB.
        ws_max_size=200 * 1024 * 1024,
    )


async def call_native_bridge_runtime_agent(
    runtime_agent: Callable[..., Awaitable[Any]],
    /,
    *,
    timeout: float = 3600.0,
    no_thinking_request_args: dict[str, Any] | None = None,
    haiku_model: str | None = None,
    preserve_thinking: bool = True,
    conversion_log_level: str = 'INFO',
    thinking_config: str = 'hunyuan',
    api_protocol: str = 'anthropic',
    bridge_request_timeout: float | None = None,
    pass_bridge_client: bool = False,
    bridge_transport_timeout: float | None = None,
    bridge_proxy: str | None = None,
    stop_bridge_after_run: bool = False,
    close_llm_handler: bool = False,
    **kwargs: Any,
) -> Any:
    """Run a host-orchestrated installed agent through a sandbox-local bridge.

    This mirrors :func:`call_runtime_agent` at the installed-agent API boundary,
    but it intentionally does not call ``runtime.invoke``.  The supplied
    ``runtime_agent`` callable runs in the host process and drives the native sandbox
    through the provider runtime plus bridge adapter.
    """
    from nexus.core.context import require_context
    from nexus.extensions.agents.installed_common.native_bridge import _await_cleanup, _build_adapter
    from nexus.runtime.bridge.adapters.base import BridgeLaunchOptions
    from nexus.utils.host_network import host_trust_env

    ctx = require_context()
    runtime = await ctx.get_runtime()

    adapter = _build_adapter(runtime)
    native_transport = api_protocol.startswith('osworld-') or (
        api_protocol == 'responses-chat' and isinstance(adapter, __getattr__('AGSOSWorldBridgeAdapter'))
    )
    artifact_path = await build_bridge_artifact_cached(runtime)
    launch = None
    bridge_may_be_running = False
    bridge_client: Any = None
    handler: Any = None
    try:
        await adapter.upload_artifact(artifact_path)
        bridge_may_be_running = True
        launch = await adapter.launch_bridge(
            BridgeLaunchOptions(
                conversion_log_level=conversion_log_level,
                thinking_config=thinking_config,
                api_protocol=api_protocol,
                request_timeout=timeout if bridge_request_timeout is None else bridge_request_timeout,
            )
        )
        await adapter.wait_ready(launch, timeout=90.0 if native_transport else 300.0)

        client_options: dict[str, Any] = {}
        if bridge_transport_timeout is not None:
            client_options['request_timeout'] = bridge_transport_timeout
        if bridge_proxy is not None:
            client_options['proxy'] = bridge_proxy
        bridge_client = __getattr__('BridgeHostClient')(
            base_url=launch.base_url,
            headers=launch.headers,
            initial_connect_attempts=1 if native_transport else None,
            trust_env=host_trust_env() if native_transport else True,
            **client_options,
        )
        handler = await connect_bridge_llm_handler(
            cast(Any, bridge_client),
            ctx.get_openai_client(),
            no_thinking_request_args=no_thinking_request_args,
            haiku_model=haiku_model,
            preserve_thinking=preserve_thinking,
        )
        if pass_bridge_client:
            kwargs['bridge_client'] = bridge_client
        result = await runtime_agent(
            runtime=runtime,
            launch=launch,
            timeout=timeout,
            **kwargs,
        )
        if isinstance(result, dict):
            metadata_result = cast(dict[str, Any], result)
            metadata_result['llm_request_outcomes'] = handler.request_outcomes
            metadata_result['last_llm_error_code'] = handler.last_request_error_code
            metadata_result['last_llm_finish_reason'] = handler.last_request_finish_reason
            metadata_result['last_llm_unparsed_tool_call'] = handler.last_request_left_tool_call_unparsed
        return result
    finally:
        cancelled: asyncio.CancelledError | None = None
        if handler is not None:
            if close_llm_handler:
                try:
                    await _await_cleanup(handler.aclose(), label='LLM handler')
                except asyncio.CancelledError as exc:
                    cancelled = exc
            else:
                handler.unregister()
        if bridge_client is not None:
            try:
                # _await_cleanup logs the failure warning (label) and bounds the disconnect with
                # BRIDGE_CLEANUP_TIMEOUT, so a wedged socket cannot stall teardown.
                await _await_cleanup(bridge_client.ws_disconnect(), label='Socket.IO disconnect')
            except asyncio.CancelledError as exc:
                cancelled = exc
        # Deployment adapter owns retention; explicit caller override may force cleanup.
        if bridge_may_be_running and (
            stop_bridge_after_run or not isinstance(adapter, __getattr__('AGSOSWorldBridgeAdapter'))
        ):
            try:
                await _await_cleanup(adapter.stop_bridge(), label='stop')
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled


async def connect_bridge_llm_handler(
    bridge_client,
    openai_client,
    *,
    no_thinking_request_args=None,
    haiku_model=None,
    preserve_thinking=True,
    handler_factory=None,
):
    await bridge_client.ws_connect()
    handler = (handler_factory or create_handler)(
        runtime=bridge_client,
        openai_client=openai_client,
        no_thinking_request_args=no_thinking_request_args,
        haiku_model=haiku_model,
        preserve_thinking=preserve_thinking,
    )
    handler.register()
    return handler


def _bridge_content_hash(target, default_port):
    from nexus.runtime.bridge import artifacts

    digest = hashlib.sha256(artifacts._bridge_content_hash(target, default_port).encode())
    digest.update(b'ags-osworld-adapter-v1')
    digest.update(repr(BUNDLE_REQUIREMENTS).encode())
    for rel in ('nexus/runtime/server/rpc_replay_guard.py', 'nexus/utils/shell.py', 'nexus/utils/async_http.py'):
        digest.update((artifacts._repo_root() / rel).read_bytes())
    return digest.hexdigest()[:16]


def build_app_bundle(*, output, target, default_port, include_vendor=True):
    """Extend the standard archive without changing global builder functions."""
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import zipfile

    from nexus.runtime.bridge import artifacts

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        original = tmp / 'base.tar.gz'
        artifacts.build_app_bundle(
            output=original, target=target, default_port=default_port, include_vendor=include_vendor
        )
        with tarfile.open(original) as archive:
            # This archive was just produced locally by the shared builder.
            if hasattr(tarfile, 'data_filter'):
                archive.extractall(tmp, filter='data')
            else:
                archive.extractall(tmp)
        bundle = tmp / 'nexus-llm-bridge'
        # Shared source copier already includes this adapter; only add the missing base.
        for rel in ('nexus/runtime/server/rpc_replay_guard.py', 'nexus/utils/shell.py', 'nexus/utils/async_http.py'):
            shutil.copy2(artifacts._repo_root() / rel, bundle / 'src' / rel)
        manifest = bundle / 'manifest.json'
        metadata = json.loads(manifest.read_text())
        metadata['entrypoint'] = 'python -m nexus.runtime.bridge.adapters.ags_osworld'
        manifest.write_text(json.dumps(metadata, indent=2))
        for filename in ('launch_bridge.sh', 'launch_bridge.ps1'):
            launcher = bundle / 'bin' / filename
            text = launcher.read_text()
            text = text.replace('nexus.runtime.bridge', 'nexus.runtime.bridge.adapters.ags_osworld')
            if filename.endswith('.sh'):
                text = text.replace('nohup ', 'nohup setsid ', 1)
            launcher.write_text(text)
        if include_vendor:
            spec = artifacts._BUNDLE_TARGET_SPECS[artifacts._coerce_bundle_target(target)]
            wheels = tmp / 'extra-wheels'
            wheels.mkdir()
            command = [
                sys.executable,
                '-m',
                'pip',
                'download',
                '--only-binary=:all:',
                '--dest',
                str(wheels),
            ]
            if spec.platform_tag:
                command += ['--platform', spec.platform_tag, '--python-version', spec.python_tag.replace('py', '')]
            command += list(BUNDLE_REQUIREMENTS)
            subprocess.run(command, check=True, capture_output=True, text=True)
            for wheel in wheels.glob('*.whl'):
                with zipfile.ZipFile(wheel) as archive:
                    archive.extractall(bundle / 'vendor')
        with tarfile.open(output, 'w:gz') as archive:
            archive.add(bundle, arcname='nexus-llm-bridge')
    return output


def _build_cached_bundle(target, default_port):
    import tempfile

    import filelock

    from nexus.runtime.bridge import artifacts

    key = f'{target.value}-osworld-{_bridge_content_hash(target, default_port)}'
    artifacts.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(artifacts.CACHE_DIR / f'{key}.lock'):
        dest = artifacts.CACHE_DIR / key / artifacts._ARTIFACT_NAMES[target]
        if not dest.is_file():
            with tempfile.TemporaryDirectory(dir=artifacts.CACHE_DIR) as tmp:
                built = Path(tmp) / dest.name
                build_app_bundle(output=built, target=target, default_port=default_port)
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Publish atomically; an interrupted copy must not poison the cache.
                built.replace(dest)
        return dest


_bundle_builds: dict[tuple[Any, int, str], asyncio.Task[Path]] = {}


async def build_bridge_artifact_cached(runtime, *, default_port=8765):
    from nexus.runtime.bridge.runtime_kind import native_bridge_bundle_target_for_runtime

    target = native_bridge_bundle_target_for_runtime(runtime)
    digest = await asyncio.to_thread(_bridge_content_hash, target, default_port)
    key = (asyncio.get_running_loop(), default_port, f'{target.value}-{digest}')
    task = _bundle_builds.get(key)
    if task is None:
        task = asyncio.create_task(asyncio.to_thread(_build_cached_bundle, target, default_port))
        _bundle_builds[key] = task

        def completed(done):
            _bundle_builds.pop(key, None)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
    # Cancellation of one case must not cancel the build shared by other cases.
    return await asyncio.shield(task)


if __name__ == '__main__':
    main()
