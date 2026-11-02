# /// script
# dependencies = []
#
# [project]
# name = "vertex"
# authors = [
#     { name = "kechizhang" },
# ]
#
# [tool.nexus]
# openai_clients = { vertex = "nexus.extensions.openai_clients.vertex" }
# ///

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
import uuid

from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import httpx
import openai

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from nexus.core import (
    ContextLengthExceededError,
    OpenAICompatibleClient,
    OpenAIConfig,
    apply_enforced_request_args,
    openai_client_registry,
    resolve_effective_request_args,
)
from nexus.core.types import (
    ExtendedChatCompletion,
    ExtendedChoice,
    ExtendedCompletionTokensDetails,
    ExtendedMessage,
    ExtendedPromptTokensDetails,
    ExtendedUsage,
    RedactedThinkingBlock,
    ThinkingBlock,
)
from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing
from nexus.utils.common import serialize_object


logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = 'vertex-2023-10-16'
_DEFAULT_MAX_TOKENS = 64000
_OPUS_4_8_MAX_TOKENS = 128000


class VertexAnthropicHTTPError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f'Vertex Anthropic request failed ({status}): {message}')
        self.status = status
        self.message = message
        self.body = message


class VertexRequestError(RuntimeError):
    """Safe public error raised after a Vertex provider request fails."""

    def __init__(self, protocol: str, status: int | None = None):
        protocol_name = 'Anthropic' if protocol == 'anthropic' else 'OpenAI'
        message = f'Vertex {protocol_name} request failed'
        if status is not None:
            message = f'{message} (status {status})'
        super().__init__(message)
        self.protocol = protocol
        self.status = status


class _VertexAnthropicResponseError(RuntimeError):
    def __init__(self, status: int, description: str, body: str):
        super().__init__(f'Vertex Anthropic {description}')
        self.status = status
        self.body = body


class _VertexOpenAIStreamError(RuntimeError):
    def __init__(self, message: str, chunks: list[dict[str, Any]]):
        super().__init__(message)
        self.body = {'stream_chunks': list(chunks)}


class _VertexAnthropicStreamError(RuntimeError):
    def __init__(self, raw_sse_events: list[str], partial_response: dict[str, Any]):
        super().__init__('Vertex Anthropic stream read failed')
        self.body = {
            'raw_sse_events': list(raw_sse_events),
            'partial_response': partial_response,
        }


_STREAM_HISTORY_TAIL = 20
_RETRYABLE_STATUS_CODES = {408, 409, 429}
_ANTHROPIC_NATIVE_TOOL_TYPES = {'bash_20250124', 'computer_20251124'}


def _append_endpoint_path(endpoint: Any, path: str) -> str:
    value = str(endpoint or '')
    try:
        parsed = urlsplit(value)
    except ValueError:
        return f'{value.rstrip("/")}/{path.lstrip("/")}'
    combined_path = f'{parsed.path.rstrip("/")}/{path.lstrip("/")}'
    return urlunsplit((parsed.scheme, parsed.netloc, combined_path, parsed.query, parsed.fragment))


def _openai_transport_from_endpoint(endpoint: Any) -> tuple[str, str]:
    value = str(endpoint or '')
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError('vertex openai client_args.endpoint must be a full Chat Completions URL') from exc

    if (
        parsed.scheme.lower() not in {'http', 'https'}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith('/chat/completions')
    ):
        raise ValueError(
            'vertex openai client_args.endpoint must be an HTTP(S) URL ending exactly in /chat/completions '
            'without query parameters or a fragment'
        )

    base_path = parsed.path[: -len('chat/completions')]
    base_url = urlunsplit((parsed.scheme, parsed.netloc, base_path, '', ''))
    exact_endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
    return base_url, exact_endpoint


def _provider_value(value: Any) -> Any:
    """Return a complete JSON-serializable provider value without redaction."""

    if isinstance(value, Mapping):
        return {str(key): _provider_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_provider_value(item) for item in value]
    return serialize_object(value)


def _contains_context_length_marker(value: Any) -> bool:
    if isinstance(value, str):
        normalized = re.sub(r'[\s-]+', '_', value.lower())
        return any(
            marker in normalized
            for marker in (
                'context_length_exceeded',
                'context_window_exceeded',
                'maximum_context_length',
                'max_context_length',
                'input_too_long',
            )
        )
    if isinstance(value, Mapping):
        return any(
            _contains_context_length_marker(key) or _contains_context_length_marker(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_context_length_marker(item) for item in value)
    return False


def _status_is_retryable(status: int | None) -> bool:
    return status in _RETRYABLE_STATUS_CODES or (status is not None and status >= 500)


def _is_context_length_error(error: BaseException) -> bool:
    if _error_status(error) == 413:
        return True
    if _contains_context_length_marker(str(error)):
        return True
    for attribute in ('code', 'type', 'body'):
        if _contains_context_length_marker(getattr(error, attribute, None)):
            return True
    cause = error.__cause__
    return cause is not None and cause is not error and _is_context_length_error(cause)


def _is_anthropic_retryable(error: BaseException) -> bool:
    if isinstance(error, _VertexAnthropicStreamError):
        return True
    if isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    if isinstance(error, VertexAnthropicHTTPError) and _status_is_retryable(error.status):
        return True
    cause = error.__cause__
    return cause is not None and cause is not error and _is_anthropic_retryable(cause)


def _error_status(error: BaseException) -> int | None:
    status = getattr(error, 'status', None)
    if isinstance(status, int):
        return status
    status_code = getattr(error, 'status_code', None)
    return status_code if isinstance(status_code, int) else None


def _error_detail(error: BaseException, limit: int = 2000) -> str:
    """Return the provider error body for logging, truncated to keep logs readable.

    ``VertexRequestError`` deliberately drops the body before it reaches the caller, so the upstream response is only observable if it is logged here.
    """
    detail = getattr(error, 'body', None)
    if detail is None and isinstance(error, VertexAnthropicHTTPError):
        detail = error.message
    if detail is None:
        detail = str(error) or type(error).__name__
    text = detail if isinstance(detail, str) else repr(detail)
    text = ' '.join(text.split())
    return text if len(text) <= limit else f'{text[:limit]}... (truncated, {len(text)} chars)'


def _log_attempt_failure(
    protocol: str,
    attempt: int,
    max_retries: int,
    error: BaseException,
    *,
    will_retry: bool,
) -> None:
    logger.log(
        logging.INFO if will_retry else logging.WARNING,
        'vertex %s attempt %d/%d failed: %s status=%s will_retry=%s detail=%s',
        protocol,
        attempt,
        max_retries,
        type(error).__name__,
        _error_status(error),
        will_retry,
        _error_detail(error),
    )


def _attempt_error(attempt: int, error: BaseException) -> dict[str, Any]:
    status = _error_status(error)
    message = str(error) or type(error).__name__
    record: dict[str, Any] = {
        'attempt': attempt,
        'status': 'error',
        'error_type': type(error).__name__,
        'message': message,
    }
    if status is not None:
        record['status_code'] = status
    return record


def _set_final_error(provider_call: dict[str, Any], error: BaseException) -> None:
    status = _error_status(error)
    message = str(error) or type(error).__name__
    provider_call['error'] = {
        'type': type(error).__name__,
        'message': message,
    }
    if status is not None:
        provider_call['error']['status_code'] = status


def _attach_provider_call(
    completion: ExtendedChatCompletion,
    provider_call: dict[str, Any],
    record_provider_call: Callable[[dict[str, Any]], None],
) -> ExtendedChatCompletion:
    completion.provider_specific_fields = {
        **(completion.provider_specific_fields or {}),
        'provider_call': provider_call,
    }
    record_provider_call(provider_call)
    return completion


provider_value = _provider_value
provider_attempt_error = _attempt_error
set_provider_call_error = _set_final_error
attach_provider_call = _attach_provider_call
provider_error_status = _error_status
is_provider_context_length_error = _is_context_length_error


def _tag_message_cache_control(entry: dict[str, Any]) -> bool:
    # For tool_result messages, Vertex requires cache_control at message level, not content block level
    if entry.get('role') == 'tool':
        entry['cache_control'] = {'type': 'ephemeral'}
        return True

    content = entry.get('content')
    if content is None:
        return False

    if isinstance(content, str) and content.strip():
        entry['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        return True

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if isinstance(text, str) and text.strip():
                    block['cache_control'] = {'type': 'ephemeral'}
                    return True

    return False


def apply_cache_control_claude_code(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_messages = copy.deepcopy(messages)
    for entry in reversed(new_messages):
        if entry.get('role') in {'user', 'tool'} and _tag_message_cache_control(entry):
            break
    return new_messages


def apply_tools_cache_control(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tools:
        return tools
    new_tools = copy.deepcopy(tools)
    new_tools[-1]['cache_control'] = {'type': 'ephemeral'}
    return new_tools


def _content_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get('text') or block.get('input_text')
                if isinstance(text, str):
                    parts.append(text)
        return ''.join(parts)
    return str(content)


def _convert_text_part(block: dict[str, Any]) -> dict[str, Any]:
    result = {'type': 'text', 'text': block.get('text') or block.get('input_text') or ''}
    if block.get('cache_control'):
        result['cache_control'] = block['cache_control']
    return result


def _convert_image_url(block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = block.get('image_url') or {}
    url = image_url.get('url') if isinstance(image_url, dict) else None
    if not isinstance(url, str) or not url.startswith('data:') or ';base64,' not in url:
        return None

    header, data = url.split(';base64,', 1)
    media_type = header.removeprefix('data:') or 'image/png'
    return {
        'type': 'image',
        'source': {
            'type': 'base64',
            'media_type': media_type,
            'data': data,
        },
    }


def _convert_user_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _content_text(content)

    converted: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            converted.append({'type': 'text', 'text': block})
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get('type')
        if block_type in {'text', 'input_text'}:
            converted.append(_convert_text_part(block))
        elif block_type == 'image_url':
            image = _convert_image_url(block)
            if image is not None:
                converted.append(image)

    if len(converted) == 1 and converted[0].get('type') == 'text' and not converted[0].get('cache_control'):
        return converted[0].get('text', '')
    return converted


def _convert_tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    converted = _convert_user_content(content)
    if isinstance(converted, list):
        return converted or ''
    return converted


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    model_dump = getattr(tool_call, 'model_dump', None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return {}


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ''):
        return {}
    if not isinstance(value, str):
        raise ValueError(f'Tool arguments must be a JSON object, got {type(value).__name__}')
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError('Tool arguments contain invalid JSON') from exc
    if not isinstance(loaded, dict):
        raise ValueError(f'Tool arguments must decode to a JSON object, got {type(loaded).__name__}')
    return loaded


_safe_json_loads = _parse_tool_arguments


def _convert_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    # Fast path: when the message was produced from a native Anthropic response we captured the original
    # content blocks in `provider_specific_fields['anthropic_content']`. Replaying them verbatim preserves the
    # exact block order (e.g. interleaved [thinking, text, thinking, text]) and each thinking block's signature.
    # Re-assembling from the individual fields below forces a fixed thinking -> text -> tool_use order, which
    # corrupts such interleaved sequences and makes Anthropic reject the follow-up request.
    provider_fields = message.get('provider_specific_fields')
    if isinstance(provider_fields, dict):
        anthropic_content = provider_fields.get('anthropic_content')
        if isinstance(anthropic_content, list) and anthropic_content:
            return {'role': 'assistant', 'content': copy.deepcopy(anthropic_content)}

    # Fallback path: general-purpose reconstruction for messages without the original Anthropic content
    # (e.g. other providers, hand-crafted history, or legacy records). This must be preserved.
    content_blocks: list[dict[str, Any]] = []

    for block in message.get('thinking_blocks') or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'thinking':
            converted = {
                'type': 'thinking',
                'thinking': block.get('thinking', ''),
            }
            if block.get('signature'):
                converted['signature'] = block['signature']
            content_blocks.append(converted)
        elif block_type == 'redacted_thinking':
            converted = {'type': 'redacted_thinking', 'data': block.get('data', '')}
            content_blocks.append(converted)

    content = message.get('content')
    if isinstance(content, str) and content:
        content_blocks.append({'type': 'text', 'text': content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                content_blocks.append({'type': 'text', 'text': block})
            elif isinstance(block, dict) and block.get('type') in {'text', 'input_text'}:
                text_block = _convert_text_part(block)
                if text_block.get('text'):
                    content_blocks.append(text_block)

    for tool_call in message.get('tool_calls') or []:
        call = _tool_call_to_dict(tool_call)
        function = call.get('function') or {}
        content_blocks.append({
            'type': 'tool_use',
            'id': call.get('id') or f'toolu_{uuid.uuid4().hex[:24]}',
            'name': function.get('name') or '',
            'input': _parse_tool_arguments(function.get('arguments')),
        })

    return {'role': 'assistant', 'content': content_blocks or ''}


_MID_CONVERSATION_SYSTEM_MODEL_MARKERS = (
    'claude-fable-5',
    'claude-mythos-5',
    'claude-opus-4-8',
    'claude-opus-5',
    'gemini-4',
)


def _supports_mid_conversation_system(model: str | None) -> bool:
    normalized_model = (model or '').lower().replace('.', '-').replace('_', '-')
    return any(marker in normalized_model for marker in _MID_CONVERSATION_SYSTEM_MODEL_MARKERS)


def _anthropic_message_role(message: dict[str, Any]) -> str:
    role = message.get('role')
    if role in {'system', 'developer'}:
        return 'system'
    if role == 'assistant':
        return 'assistant'
    return 'user'


def _has_valid_mid_conversation_system_placement(messages: list[dict[str, Any]]) -> bool:
    roles = [_anthropic_message_role(message) for message in messages]
    index = 0
    while index < len(roles) and roles[index] == 'system':
        index += 1

    while index < len(roles):
        if roles[index] != 'system':
            index += 1
            continue

        group_end = index + 1
        while group_end < len(roles) and roles[group_end] == 'system':
            group_end += 1
        if roles[index - 1] != 'user':
            return False
        if group_end < len(roles) and roles[group_end] != 'assistant':
            return False
        index = group_end

    return True


def convert_messages_to_anthropic(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []
    # Legacy hoisting remains the compatibility fallback for models or message layouts that reject this feature.
    preserve_mid_system = _supports_mid_conversation_system(model) and _has_valid_mid_conversation_system_placement(
        messages
    )
    conversation_started = False

    for message in messages:
        role = message.get('role')
        if role in {'system', 'developer'}:
            content = message.get('content')
            if conversation_started and preserve_mid_system:
                converted_content = _convert_user_content(content)
                if converted_content:
                    anthropic_messages.append({'role': 'system', 'content': converted_content})
            else:
                text = _content_text(content)
                if text:
                    system_parts.append(text)
            continue

        conversation_started = True
        if role == 'tool':
            tool_result = {
                'type': 'tool_result',
                'tool_use_id': message.get('tool_call_id') or '',
                'content': _convert_tool_result_content(message.get('content')),
            }
            if message.get('cache_control'):
                tool_result['cache_control'] = message['cache_control']
            anthropic_messages.append({
                'role': 'user',
                'content': [tool_result],
            })
            continue

        if role == 'assistant':
            anthropic_messages.append(_convert_assistant_message(message))
            continue

        anthropic_messages.append({
            'role': 'user',
            'content': _convert_user_content(message.get('content', '')),
        })

    system = '\n'.join(system_parts) if system_parts else None
    return system, anthropic_messages


def convert_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    converted_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get('type', '')
        if tool_type and tool_type != 'function':
            if tool_type not in _ANTHROPIC_NATIVE_TOOL_TYPES:
                raise ValueError(f'Unsupported Anthropic native tool type: {tool_type}')

            # Approved native Anthropic types flatten function.name/parameters to the top level.
            if 'function' in tool:
                func = tool['function']
                if not isinstance(func, dict):
                    raise ValueError(f'Anthropic native tool {tool_type} requires a function object')
                flattened = {k: v for k, v in tool.items() if k != 'function'}
                if 'name' not in flattened and func.get('name'):
                    flattened['name'] = func['name']
                for k, v in (func.get('parameters') or {}).items():
                    if k not in flattened:
                        flattened[k] = v
                converted_tools.append(flattened)
            else:
                converted_tools.append(tool)
            continue

        function = tool.get('function') or {}
        converted = {
            'name': function.get('name') or tool.get('name') or '',
            'description': function.get('description') or tool.get('description') or '',
            'input_schema': function.get('parameters') or tool.get('input_schema') or {'type': 'object'},
        }
        if tool.get('cache_control'):
            converted['cache_control'] = tool['cache_control']
        converted_tools.append(converted)

    return converted_tools or None


def _thinking_enabled(thinking: dict[str, Any] | None) -> bool:
    return bool(thinking and thinking.get('type') != 'disabled')


def _normalize_thinking(thinking: Any) -> dict[str, Any] | None:
    if not thinking:
        return None
    if thinking is True:
        return {'type': 'adaptive'}
    if isinstance(thinking, dict):
        normalized = {key: value for key, value in thinking.items() if value is not None}
        normalized['type'] = normalized.get('type') or 'adaptive'
        return normalized
    return None


def _normalize_output_config(request_args: dict[str, Any]) -> dict[str, Any] | None:
    output_config = request_args.get('output_config')
    if isinstance(output_config, dict):
        return output_config

    reasoning_effort = request_args.get('reasoning_effort')
    if isinstance(reasoning_effort, str) and reasoning_effort:
        return {'effort': reasoning_effort}
    if isinstance(reasoning_effort, dict) and reasoning_effort.get('effort'):
        return {'effort': reasoning_effort['effort']}
    return None


def _convert_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == 'none':
        return None
    if tool_choice == 'auto':
        return {'type': 'auto'}
    if tool_choice == 'required':
        return {'type': 'any'}
    if isinstance(tool_choice, dict):
        if tool_choice.get('type') == 'function':
            function = tool_choice.get('function') or {}
            return {'type': 'tool', 'name': function.get('name') or ''}
        if tool_choice.get('type') in {'auto', 'any', 'tool'}:
            return dict(tool_choice)
    return None


def _default_max_tokens_for_model(model: Any) -> int:
    if isinstance(model, str) and 'claude-opus-4-8' in model:
        return _OPUS_4_8_MAX_TOKENS
    return _DEFAULT_MAX_TOKENS


def _build_payload(
    *,
    messages: list[dict[str, Any]],
    request_args: dict[str, Any],
) -> dict[str, Any]:
    system, anthropic_messages = convert_messages_to_anthropic(messages, model=request_args.get('model'))
    max_tokens = request_args.get('max_tokens')
    if max_tokens is None:
        max_tokens = _default_max_tokens_for_model(request_args.get('model'))

    payload: dict[str, Any] = {
        'anthropic_version': request_args.get('anthropic_version') or _ANTHROPIC_VERSION,
        'messages': anthropic_messages,
        'max_tokens': max_tokens,
        'stream': bool(request_args.get('stream', False)),
    }
    if system:
        payload['system'] = system

    thinking = _normalize_thinking(request_args.get('thinking'))
    if thinking is not None:
        payload['thinking'] = thinking

    output_config = _normalize_output_config(request_args)
    if output_config is not None:
        payload['output_config'] = output_config

    tools = convert_tools_to_anthropic(request_args.get('tools'))
    if tools is not None:
        payload['tools'] = tools

    tool_choice = _convert_tool_choice(request_args.get('tool_choice'))
    if tool_choice is not None:
        forced_choice = tool_choice.get('type') in {'any', 'tool'}
        drop_for_thinking = request_args.get('drop_forced_tool_choice_with_thinking', True)
        if _thinking_enabled(thinking) and forced_choice and drop_for_thinking:
            logger.warning('Dropping forced tool_choice because Vertex Anthropic rejects it when thinking is enabled')
        else:
            payload['tool_choice'] = tool_choice

    if request_args.get('temperature') is not None:
        payload['temperature'] = request_args['temperature']
    if request_args.get('top_p') is not None:
        payload['top_p'] = request_args['top_p']
    if request_args.get('top_k') is not None:
        payload['top_k'] = request_args['top_k']
    if request_args.get('metadata') is not None:
        payload['metadata'] = request_args['metadata']
    if request_args.get('stop') is not None:
        stop = request_args['stop']
        payload['stop_sequences'] = stop if isinstance(stop, list) else [stop]

    return payload


def _stop_reason_to_finish_reason(stop_reason: str | None) -> str:
    mapping = {
        'end_turn': 'stop',
        'max_tokens': 'length',
        'refusal': 'stop',
        'stop_sequence': 'stop',
        'tool_use': 'tool_calls',
    }
    return mapping.get(stop_reason or '', 'stop')


def _refusal_text(response: Mapping[str, Any]) -> str | None:
    stop_details = response.get('stop_details')
    is_refusal = response.get('stop_reason') == 'refusal'
    if isinstance(stop_details, Mapping):
        is_refusal = is_refusal or stop_details.get('type') == 'refusal'
    if not is_refusal:
        return None

    if isinstance(stop_details, Mapping):
        explanation = stop_details.get('explanation')
        if isinstance(explanation, str) and explanation.strip():
            return explanation
    return 'The model refused to fulfill the request.'


def _convert_response(response: dict[str, Any], model: str) -> ExtendedChatCompletion:
    if not isinstance(response.get('content'), list):
        raise ValueError('Vertex Anthropic returned an invalid Messages response: content must be an array')

    native_response = copy.deepcopy(response)
    native_response.pop('raw_sse_events', None)

    text_parts: list[str] = []
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []

    for block in response.get('content') or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'text':
            text_parts.append(block.get('text', ''))
        elif block_type == 'thinking':
            thinking_blocks.append(
                ThinkingBlock(
                    type='thinking',
                    thinking=block.get('thinking', ''),
                    signature=block.get('signature') or None,
                )
            )
        elif block_type == 'redacted_thinking':
            thinking_blocks.append(RedactedThinkingBlock(type='redacted_thinking', data=block.get('data', '')))
        elif block_type == 'tool_use':
            arguments = _parse_tool_arguments(block.get('input'))
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=block.get('id') or f'toolu_{uuid.uuid4().hex[:24]}',
                    type='function',
                    function=Function(
                        name=block.get('name') or '',
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    usage_data = response.get('usage') or {}
    input_tokens = int(usage_data.get('input_tokens') or 0)
    output_tokens = int(usage_data.get('output_tokens') or 0)
    cache_creation_tokens = int(usage_data.get('cache_creation_input_tokens') or 0)
    cache_read_tokens = int(usage_data.get('cache_read_input_tokens') or 0)
    prompt_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
    thinking_tokens = (usage_data.get('output_tokens_details') or {}).get('thinking_tokens')

    usage = ExtendedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=ExtendedPromptTokensDetails(
            cached_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ),
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
        completion_tokens_details=ExtendedCompletionTokensDetails(reasoning_tokens=thinking_tokens)
        if thinking_tokens is not None
        else None,
    )

    message = ExtendedMessage(
        role='assistant',
        content=''.join(text_parts) if text_parts else None,
        refusal=_refusal_text(response),
        tool_calls=tool_calls or None,
        thinking_blocks=thinking_blocks or None,
        provider_specific_fields={'anthropic_content': response.get('content') or []},
    )

    choice = ExtendedChoice(
        finish_reason=_stop_reason_to_finish_reason(response.get('stop_reason')),
        index=0,
        message=message,
        provider_specific_fields={
            'stop_reason': response.get('stop_reason'),
            'stop_sequence': response.get('stop_sequence'),
            'stop_details': response.get('stop_details'),
        },
    )

    return ExtendedChatCompletion(
        id=response.get('id') or f'chatcmpl-{uuid.uuid4().hex}',
        choices=[choice],
        created=int(response.get('created') or time.time()),
        model=response.get('model') or model,
        usage=usage,
        provider_specific_fields={
            'native_response': {
                'protocol': 'anthropic',
                'body': native_response,
            }
        },
    )


class _StreamAccumulator:
    def __init__(self):
        self.message: dict[str, Any] = {'content': []}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.tool_json_parts: dict[int, list[str]] = {}
        self.message_stopped = False

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get('type')
        if event_type == 'error':
            error = event.get('error') or {}
            raise RuntimeError(f'Vertex Anthropic stream error: {json.dumps(error, ensure_ascii=False)}')

        if event_type == 'message_start':
            message = event.get('message') or {}
            if isinstance(message, dict):
                self.message.update(message)
            return

        if event_type == 'content_block_start':
            index = int(event.get('index') or 0)
            block = copy.deepcopy(event.get('content_block') or {})
            block_type = block.get('type')
            if block_type == 'text':
                block.setdefault('text', '')
            elif block_type == 'thinking':
                block.setdefault('thinking', '')
            elif block_type == 'tool_use':
                block.setdefault('input', {})
                self.tool_json_parts[index] = []
            self.blocks[index] = block
            return

        if event_type == 'content_block_delta':
            index = int(event.get('index') or 0)
            block = self.blocks.setdefault(index, {})
            delta = event.get('delta') or {}
            if not isinstance(delta, dict):
                return

            delta_type = delta.get('type')
            if delta_type == 'text_delta' or 'text' in delta:
                block['type'] = 'text'
                block['text'] = block.get('text', '') + (delta.get('text') or '')
            elif delta_type == 'thinking_delta' or 'thinking' in delta:
                block['type'] = 'thinking'
                block['thinking'] = block.get('thinking', '') + (delta.get('thinking') or '')
            elif delta_type == 'signature_delta' or 'signature_delta' in delta or 'signature' in delta:
                block['signature'] = block.get('signature', '') + (
                    delta.get('signature') or delta.get('signature_delta') or ''
                )
            elif delta_type == 'input_json_delta' or 'partial_json' in delta:
                block['type'] = 'tool_use'
                self.tool_json_parts.setdefault(index, []).append(delta.get('partial_json') or '')
            return

        if event_type == 'content_block_stop':
            index = int(event.get('index') or 0)
            block = self.blocks.get(index)
            if block and block.get('type') == 'tool_use':
                raw_input = ''.join(self.tool_json_parts.get(index, []))
                if raw_input:
                    try:
                        parsed = json.loads(raw_input)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Vertex Anthropic tool block {index} contains invalid JSON') from exc
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f'Vertex Anthropic tool block {index} must decode to a JSON object, '
                            f'got {type(parsed).__name__}'
                        )
                    block['input'] = parsed
                else:
                    try:
                        block['input'] = _parse_tool_arguments(block.get('input'))
                    except ValueError as exc:
                        raise ValueError(f'Vertex Anthropic tool block {index} has invalid input: {exc}') from exc
            return

        if event_type == 'message_delta':
            delta = event.get('delta') or {}
            if isinstance(delta, dict):
                if 'stop_reason' in delta:
                    self.message['stop_reason'] = delta.get('stop_reason')
                if 'stop_sequence' in delta:
                    self.message['stop_sequence'] = delta.get('stop_sequence')
                if 'stop_details' in delta:
                    self.message['stop_details'] = delta.get('stop_details')
            usage = event.get('usage') or (delta.get('usage') if isinstance(delta, dict) else None)
            if isinstance(usage, dict):
                self.message.setdefault('usage', {}).update(usage)
            return

        if event_type == 'message_stop':
            self.message_stopped = True

    def build(self, model: str) -> dict[str, Any]:
        content = [self.blocks[index] for index in sorted(self.blocks)]
        self.message['content'] = content
        self.message.setdefault('model', model)
        return self.message


def _parse_sse_event(raw_event: str) -> dict[str, Any] | None:
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith('data:'):
            data_lines.append(line[len('data:') :].strip())
    if not data_lines:
        return None
    raw_data = '\n'.join(data_lines)
    try:
        event = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Vertex Anthropic returned invalid SSE data: {raw_data[:500]}') from exc
    if not isinstance(event, dict):
        raise RuntimeError(f'Vertex Anthropic returned unexpected SSE data type: {type(event).__name__}')
    return event


def _append_stream_history(history: list[Any], item: Any, capture_raw_io: bool) -> None:
    history.append(item)
    if not capture_raw_io and len(history) > _STREAM_HISTORY_TAIL:
        del history[:-_STREAM_HISTORY_TAIL]


async def _read_stream_response(
    response: aiohttp.ClientResponse,
    model: str,
    *,
    capture_raw_io: bool = True,
) -> dict[str, Any]:
    accumulator = _StreamAccumulator()
    event_lines: list[str] = []
    raw_sse_events: list[str] = []

    def consume_event(lines: list[str]) -> None:
        raw_event = '\n'.join(lines)
        _append_stream_history(raw_sse_events, raw_event, capture_raw_io)
        event = _parse_sse_event(raw_event)
        if event is not None:
            accumulator.handle(event)

    try:
        async for raw_line in response.content:
            line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
            if line == '':
                if event_lines:
                    pending_event = event_lines
                    event_lines = []
                    consume_event(pending_event)
                continue
            event_lines.append(line)

        if event_lines:
            pending_event = event_lines
            event_lines = []
            consume_event(pending_event)
    except asyncio.CancelledError as error:
        if event_lines:
            _append_stream_history(raw_sse_events, '\n'.join(event_lines), capture_raw_io)
        vars(error)['body'] = {
            'raw_sse_events': list(raw_sse_events),
            'partial_response': accumulator.build(model),
        }
        raise
    except Exception as error:
        if event_lines:
            _append_stream_history(raw_sse_events, '\n'.join(event_lines), capture_raw_io)
        raise _VertexAnthropicStreamError(raw_sse_events, accumulator.build(model)) from error

    if not accumulator.message_stopped:
        raise _VertexAnthropicStreamError(raw_sse_events, accumulator.build(model))

    result = accumulator.build(model)
    if capture_raw_io:
        result['raw_sse_events'] = raw_sse_events
    return result


def _openai_model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, 'model_dump', None)
    if not callable(model_dump):
        raise TypeError(f'Vertex OpenAI stream returned unsupported chunk type: {type(value).__name__}')
    dumped = model_dump(mode='json', exclude_none=True)
    if not isinstance(dumped, dict):
        raise TypeError(f'Vertex OpenAI stream returned unsupported chunk data: {type(dumped).__name__}')
    return dumped


async def _aggregate_openai_stream(
    stream: Any,
    model: str,
    *,
    capture_raw_io: bool = True,
) -> tuple[ChatCompletion, list[dict[str, Any]]]:
    response: dict[str, Any] = {
        'id': '',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
    }
    choices: dict[int, dict[str, Any]] = {}
    raw_chunks: list[dict[str, Any]] = []

    stream_iterator = stream.__aiter__()
    while True:
        try:
            raw_chunk = await anext(stream_iterator)
        except StopAsyncIteration:
            break
        except asyncio.CancelledError as error:
            vars(error)['body'] = {'stream_chunks': raw_chunks}
            raise
        except Exception as error:
            raise _VertexOpenAIStreamError('Vertex OpenAI stream read failed', raw_chunks) from error
        chunk = _openai_model_dump(raw_chunk)
        _append_stream_history(raw_chunks, chunk, capture_raw_io)
        for field in ('id', 'created', 'model', 'system_fingerprint', 'service_tier'):
            if chunk.get(field) is not None:
                response[field] = chunk[field]
        if chunk.get('usage') is not None:
            response['usage'] = chunk['usage']
        provider_specific_fields = chunk.get('provider_specific_fields')
        if isinstance(provider_specific_fields, Mapping):
            response.setdefault('provider_specific_fields', {}).update(provider_specific_fields)

        for raw_choice in chunk.get('choices') or []:
            if not isinstance(raw_choice, dict):
                continue
            index = int(raw_choice.get('index') or 0)
            choice = choices.setdefault(
                index,
                {
                    'index': index,
                    'finish_reason': None,
                    'message': {
                        'role': 'assistant',
                        '_content_parts': [],
                        '_refusal_parts': [],
                        '_reasoning_parts': [],
                        '_reasoning_items': [],
                        '_provider_specific_fields': {},
                        '_tool_calls': {},
                        '_function_call': {},
                    },
                    '_provider_specific_fields': {},
                    '_logprobs': {'content': [], 'refusal': []},
                },
            )
            if raw_choice.get('finish_reason') is not None:
                choice['finish_reason'] = raw_choice['finish_reason']
            provider_specific_fields = raw_choice.get('provider_specific_fields')
            if isinstance(provider_specific_fields, Mapping):
                choice['_provider_specific_fields'].update(provider_specific_fields)
            raw_logprobs = raw_choice.get('logprobs')
            if isinstance(raw_logprobs, Mapping):
                for field in ('content', 'refusal'):
                    if isinstance(raw_logprobs.get(field), list):
                        choice['_logprobs'][field].extend(raw_logprobs[field])

            delta = raw_choice.get('delta') or {}
            if not isinstance(delta, dict):
                continue
            message = choice['message']
            if delta.get('role'):
                message['role'] = delta['role']
            if delta.get('content') is not None:
                message['_content_parts'].append(str(delta['content']))
            if delta.get('refusal') is not None:
                message['_refusal_parts'].append(str(delta['refusal']))
            if delta.get('reasoning_content') is not None:
                message['_reasoning_parts'].append(str(delta['reasoning_content']))
            reasoning_items = delta.get('reasoning_items')
            if isinstance(reasoning_items, list):
                message['_reasoning_items'].extend(reasoning_items)
            provider_specific_fields = delta.get('provider_specific_fields')
            if isinstance(provider_specific_fields, Mapping):
                message['_provider_specific_fields'].update(provider_specific_fields)

            function_call = delta.get('function_call')
            if isinstance(function_call, dict):
                if function_call.get('name'):
                    message['_function_call']['name'] = function_call['name']
                if function_call.get('arguments') is not None:
                    message['_function_call']['arguments'] = message['_function_call'].get('arguments', '') + str(
                        function_call['arguments']
                    )

            for raw_tool_call in delta.get('tool_calls') or []:
                if not isinstance(raw_tool_call, dict):
                    continue
                tool_index = int(raw_tool_call.get('index') or 0)
                tool_call = message['_tool_calls'].setdefault(
                    tool_index,
                    {
                        'id': '',
                        'type': 'function',
                        'function': {'name': '', 'arguments': ''},
                    },
                )
                if raw_tool_call.get('id'):
                    tool_call['id'] = raw_tool_call['id']
                if raw_tool_call.get('type'):
                    tool_call['type'] = raw_tool_call['type']
                raw_function = raw_tool_call.get('function') or {}
                if isinstance(raw_function, dict):
                    if raw_function.get('name'):
                        tool_call['function']['name'] += str(raw_function['name'])
                    if raw_function.get('arguments') is not None:
                        tool_call['function']['arguments'] += str(raw_function['arguments'])

    if not choices:
        raise _VertexOpenAIStreamError('Vertex OpenAI stream completed without any choices', raw_chunks)

    final_choices: list[dict[str, Any]] = []
    for index in sorted(choices):
        choice = choices[index]
        message = choice['message']
        content_parts = message.pop('_content_parts')
        refusal_parts = message.pop('_refusal_parts')
        reasoning_parts = message.pop('_reasoning_parts')
        reasoning_items = message.pop('_reasoning_items')
        provider_specific_fields = message.pop('_provider_specific_fields')
        tool_calls = message.pop('_tool_calls')
        function_call = message.pop('_function_call')
        message['content'] = ''.join(content_parts) if content_parts else None
        if refusal_parts:
            message['refusal'] = ''.join(refusal_parts)
        if tool_calls:
            message['tool_calls'] = [tool_calls[tool_index] for tool_index in sorted(tool_calls)]
        if function_call:
            message['function_call'] = function_call
        if reasoning_parts:
            message['reasoning_content'] = ''.join(reasoning_parts)
        if reasoning_items:
            message['reasoning_items'] = reasoning_items
        if provider_specific_fields:
            message['provider_specific_fields'] = provider_specific_fields
        if choice['finish_reason'] is None:
            raise _VertexOpenAIStreamError(
                f'Vertex OpenAI stream ended before choice {index} reported finish_reason',
                raw_chunks,
            )
        logprobs = choice.pop('_logprobs')
        provider_specific_fields = choice.pop('_provider_specific_fields')
        if provider_specific_fields:
            choice['provider_specific_fields'] = provider_specific_fields
        if logprobs['content'] or logprobs['refusal']:
            choice['logprobs'] = {field: values if values else None for field, values in logprobs.items()}
        final_choices.append(choice)

    response['id'] = response['id'] or f'chatcmpl-{uuid.uuid4().hex}'
    response['choices'] = final_choices
    return ChatCompletion.model_validate(response), raw_chunks


class _VertexAdapter:
    protocol: str

    def __init__(
        self,
        config: OpenAIConfig,
        record_provider_call: Callable[[dict[str, Any]], None],
    ) -> None:
        self.config = config
        self.capture_raw_io = bool(config.client_args.get('capture_raw_io', False))
        self.max_retries = max(1, int(config.client_args.get('max_retries', 3)))
        raw_api_key = str(config.client_args.get('api_key') or '')
        gateway_routing = config.client_args.get('gateway_routing', {})
        self.api_key = merge_gateway_routing(raw_api_key, gateway_routing) if gateway_routing else raw_api_key
        self._record_provider_call = record_provider_call

    def _new_provider_call(
        self,
        model: str = '',
        endpoint: str = '',
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_call: dict[str, Any] = {
            'provider': 'vertex',
            'protocol': self.protocol,
            'model': _provider_value(model),
            'endpoint': str(endpoint),
            'attempts': [],
        }
        if self.capture_raw_io and payload is not None:
            provider_call['request'] = _provider_value(payload)
        return provider_call

    def _set_provider_call_request(
        self,
        provider_call: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        provider_call['_provider_sensitive_source'] = _provider_value(payload)
        if self.capture_raw_io:
            provider_call['request'] = _provider_value(payload)

    def _record_failure(self, provider_call: dict[str, Any], error: BaseException) -> None:
        _set_final_error(provider_call, error)
        self._record_provider_call(provider_call)

    def _capture_error_response(
        self,
        provider_call: dict[str, Any],
        error: BaseException,
    ) -> dict[str, Any] | None:
        if not self.capture_raw_io:
            return None
        body = getattr(error, 'body', None)
        if body is None and isinstance(error, VertexAnthropicHTTPError):
            body = error.message
        if body is None:
            return None
        response: dict[str, Any] = {'body': _provider_value(body)}
        status = _error_status(error)
        if status is not None:
            response['status_code'] = status
        provider_call['response'] = response
        return response


class VertexAnthropicAdapter(_VertexAdapter):
    """Adapter for Vertex-hosted Anthropic Messages ``rawPredict`` endpoints."""

    protocol = 'anthropic'

    def __init__(
        self,
        config: OpenAIConfig,
        record_provider_call: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(config, record_provider_call)
        self.base_url = str(config.client_args.get('base_url') or '').rstrip('/')
        self.endpoint = config.client_args.get('endpoint')
        self.timeout = float(config.client_args.get('timeout', 900))

    def _resolve_endpoint(self, model: str, *, stream: bool) -> str:
        if self.endpoint:
            return str(self.endpoint)
        if not self.base_url:
            raise ValueError('vertex anthropic client_args.base_url is required')
        method = 'streamRawPredict' if stream else 'rawPredict'
        return f'{self.base_url}/v1/publishers/anthropic/models/{model}:{method}'

    async def _post_once(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'anthropic-beta': 'computer-use-2025-11-24',
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status < 400 and payload.get('stream'):
                    return await _read_stream_response(
                        response,
                        str(payload.get('model') or ''),
                        capture_raw_io=self.capture_raw_io,
                    )
                text = await response.text()
                if response.status >= 400:
                    raise VertexAnthropicHTTPError(response.status, text)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise _VertexAnthropicResponseError(response.status, 'returned a non-JSON response', text) from exc
                if not isinstance(data, dict):
                    raise _VertexAnthropicResponseError(
                        response.status,
                        f'returned unexpected response type {type(data).__name__}',
                        text,
                    )
                return data

    async def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        provider_call: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._post_once(endpoint, payload)
            except asyncio.CancelledError as error:
                attempt_record = _attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                raise
            except Exception as error:
                attempt_record = _attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)

                will_retry = _is_anthropic_retryable(error) and attempt < self.max_retries
                _log_attempt_failure(self.protocol, attempt, self.max_retries, error, will_retry=will_retry)
                if not will_retry:
                    raise
                await asyncio.sleep(min(2 ** (attempt - 1), 30))
                continue

            provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
            return response

        raise RuntimeError('Vertex Anthropic retry loop exhausted without a response')

    async def create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        provider_call = self._new_provider_call(endpoint=str(self.endpoint or ''))
        try:
            request_args_override = kwargs.pop('_request_args_override', None)
            request_args = {
                **(request_args_override if request_args_override is not None else self.config.request_args),
                **kwargs,
            }
            configured_model = request_args.pop('model', None)

            normalized_messages = [dict(message) for message in messages]
            if self.config.client_args.get('cache_control_enabled'):
                strategy = self.config.client_args.get('cache_control_strategy', 'default')
                if strategy != 'claude_code':
                    logger.warning('vertex anthropic only supports claude_code cache_control_strategy; using it')
                normalized_messages = apply_cache_control_claude_code(normalized_messages)
                if isinstance(request_args.get('tools'), list):
                    request_args['tools'] = apply_tools_cache_control(request_args['tools'])

            effective_request = {
                **request_args,
                'model': model or configured_model,
                'messages': normalized_messages,
            }
            if self.config.enforced_request_args:
                effective_request = resolve_effective_request_args(
                    effective_request,
                    self.config.enforced_request_args,
                )
            final_model = str(effective_request.pop('model', '') or '')
            normalized_messages = effective_request.pop('messages')
            provider_call['model'] = final_model
            if not final_model:
                raise ValueError('vertex anthropic requires a model in request_args or create_completion(model=...)')
            payload = _build_payload(
                messages=normalized_messages,
                request_args={**effective_request, 'model': final_model},
            )
            self._set_provider_call_request(provider_call, payload)
            endpoint = self._resolve_endpoint(final_model, stream=payload.get('stream') is True)
            provider_call['endpoint'] = endpoint
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            raise VertexRequestError(self.protocol, _error_status(error)) from error

        try:
            response = await self._post_json(endpoint, payload, provider_call)
            if self.capture_raw_io:
                provider_call['response'] = _provider_value(response)
            completion = _convert_response(response, final_model)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            if _is_context_length_error(error):
                raise ContextLengthExceededError('Vertex Anthropic context length exceeded') from error
            raise VertexRequestError(self.protocol, _error_status(error)) from error
        return _attach_provider_call(completion, provider_call, self._record_provider_call)


class VertexOpenAIAdapter(_VertexAdapter):
    """Adapter for Vertex endpoints exposing OpenAI Chat Completions."""

    protocol = 'openai'

    def __init__(
        self,
        config: OpenAIConfig,
        record_provider_call: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(config, record_provider_call)
        self._configured_endpoint = config.client_args.get('endpoint')
        self._client_args = {
            key: value
            for key, value in config.client_args.items()
            if key
            not in {
                'protocol',
                'capture_raw_io',
                'endpoint',
                'gateway_routing',
                'max_retries',
                'cache_control_enabled',
                'cache_control_strategy',
            }
        }
        if self.api_key:
            self._client_args['api_key'] = self.api_key
        self._client_args['max_retries'] = 0
        self.client: Any = None
        self.endpoint = str(self._configured_endpoint or '')

    def _resolve_transport(self) -> tuple[Any, str]:
        if self._configured_endpoint:
            return _openai_transport_from_endpoint(self._configured_endpoint)

        base_url = self._client_args.get('base_url')
        if not base_url:
            raise ValueError('vertex openai requires client_args.base_url or client_args.endpoint')
        return base_url, _append_endpoint_path(base_url, 'chat/completions')

    def _get_client(self, base_url: Any) -> Any:
        if self.client is None:
            client_args = {**self._client_args, 'base_url': base_url}
            self.client = openai.AsyncOpenAI(**client_args)
        return self.client

    @staticmethod
    def _is_context_length_error(error: BaseException) -> bool:
        return _is_context_length_error(error)

    @staticmethod
    def _is_retryable(error: BaseException) -> bool:
        if isinstance(
            error,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                httpx.TransportError,
                _VertexOpenAIStreamError,
            ),
        ):
            return True
        if isinstance(error, openai.APIStatusError) and _status_is_retryable(_error_status(error)):
            return True
        cause = error.__cause__
        return cause is not None and cause is not error and VertexOpenAIAdapter._is_retryable(cause)

    async def create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        provider_call = self._new_provider_call(endpoint=self.endpoint)
        try:
            request_args_override = kwargs.pop('_request_args_override', None)
            request_args = {
                **(request_args_override if request_args_override is not None else self.config.request_args),
                **kwargs,
            }
            configured_model = request_args.pop('model', None)
            base_url, endpoint = self._resolve_transport()
            self.endpoint = endpoint
            provider_call['endpoint'] = endpoint
            payload = apply_enforced_request_args(
                {
                    'model': model or configured_model,
                    **request_args,
                    'messages': messages,
                },
                self.config.enforced_request_args,
            )
            effective_request = (
                resolve_effective_request_args(payload, None) if self.config.enforced_request_args else payload
            )
            final_model = str(effective_request.get('model') or '')
            provider_call['model'] = final_model
            if not final_model:
                raise ValueError('vertex openai requires a model in request_args or create_completion(model=...)')
            self._set_provider_call_request(provider_call, payload)
            provider_call['endpoint'] = endpoint
            client = self._get_client(base_url)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            raise VertexRequestError(self.protocol, _error_status(error)) from error

        response: ChatCompletion | None = None
        stream_chunks: list[dict[str, Any]] | None = None
        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    raw_response = await client.chat.completions.create(**payload)
                    if effective_request.get('stream') is True:
                        response, stream_chunks = await _aggregate_openai_stream(
                            raw_response,
                            final_model,
                            capture_raw_io=self.capture_raw_io,
                        )
                    else:
                        response = raw_response
                except asyncio.CancelledError as error:
                    attempt_record = _attempt_error(attempt, error)
                    captured_response = self._capture_error_response(provider_call, error)
                    if captured_response is not None:
                        attempt_record['response'] = captured_response
                    provider_call['attempts'].append(attempt_record)
                    raise
                except Exception as error:
                    attempt_record = _attempt_error(attempt, error)
                    captured_response = self._capture_error_response(provider_call, error)
                    if captured_response is not None:
                        attempt_record['response'] = captured_response
                    provider_call['attempts'].append(attempt_record)
                    will_retry = self._is_retryable(error) and attempt < self.max_retries
                    _log_attempt_failure(self.protocol, attempt, self.max_retries, error, will_retry=will_retry)
                    if not will_retry:
                        raise
                    await asyncio.sleep(min(2 ** (attempt - 1), 30))
                    continue

                provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
                break

            if response is None:
                raise RuntimeError('Vertex OpenAI retry loop exhausted without a response')
            if self.capture_raw_io:
                response_data = response.model_dump(mode='json')
                if stream_chunks is not None:
                    response_data['stream_chunks'] = stream_chunks
                provider_call['response'] = _provider_value(response_data)
            completion = ExtendedChatCompletion.from_openai(response)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            if self._is_context_length_error(error):
                raise ContextLengthExceededError('Vertex OpenAI context length exceeded') from error
            raise VertexRequestError(self.protocol, _error_status(error)) from error
        return _attach_provider_call(completion, provider_call, self._record_provider_call)


@openai_client_registry.register('vertex')
class VertexClient(OpenAICompatibleClient):
    """Unified Vertex client with explicit Anthropic or OpenAI protocol routing."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        protocol = config.client_args.get('protocol')
        if protocol not in {'anthropic', 'openai'}:
            raise ValueError("vertex client_args.protocol must be 'anthropic' or 'openai'")
        self.protocol = protocol
        self.capture_raw_io = bool(config.client_args.get('capture_raw_io', False))
        self._adapter = (
            VertexAnthropicAdapter(config, self._record_provider_call)
            if protocol == 'anthropic'
            else VertexOpenAIAdapter(config, self._record_provider_call)
        )

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        return await self._adapter.create_completion(messages, model=model, **kwargs)


__all__ = [
    'VertexAnthropicAdapter',
    'VertexAnthropicHTTPError',
    'VertexClient',
    'VertexOpenAIAdapter',
    'VertexRequestError',
    'attach_provider_call',
    'apply_cache_control_claude_code',
    'apply_tools_cache_control',
    'convert_messages_to_anthropic',
    'convert_tools_to_anthropic',
    'is_provider_context_length_error',
    'provider_attempt_error',
    'provider_error_status',
    'provider_value',
    'set_provider_call_error',
    '_StreamAccumulator',
    '_build_payload',
    '_convert_response',
    '_parse_tool_arguments',
    '_safe_json_loads',
]
