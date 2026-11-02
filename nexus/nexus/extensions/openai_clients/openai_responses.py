# /// script
# dependencies = []
#
# [project]
# name = "openai_responses"
#
# [tool.nexus]
# openai_clients = { openai_responses = "nexus.extensions.openai_clients.openai_responses" }
# ///

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
import openai

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.openai_client import ContentPolicyViolationError
from nexus.core.types import (
    ExtendedChatCompletion,
    ExtendedChoice,
    ExtendedCompletionTokensDetails,
    ExtendedMessage,
    ExtendedPromptTokensDetails,
    ExtendedUsage,
)
from nexus.extensions.openai_clients.cache_task_id import normalize_cache_task_id
from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing
from nexus.extensions.openai_clients.responses_trace import trace_response
from nexus.utils.common import _resolve_retry_total_timeout, with_call_deadline


logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 409, 429}
_RETRYABLE_RESPONSE_ERROR_CODES = {
    'rate_limit_exceeded',
    'server_error',
    'vector_store_timeout',
}
_NEXUS_CLIENT_ARGS = {
    'cache_control_enabled',
    'cache_control_strategy',
    'cache_task_id',
    'prompt_cache_task_id',
    'capture_raw_io',
    'encrypted_content_archive_dir',
    'endpoint',
    'ensure_v1_path',
    'gateway_routing',
    'max_retries',
    'min_reasoning_effort',
    'protocol',
    'stateful_previous_response',
    'require_encrypted_reasoning',
}
#: Reasoning effort from weakest to strongest, for the ``min_reasoning_effort`` floor.
#: Values outside this tuple are left alone rather than guessed at.
_REASONING_EFFORT_ORDER = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
_SUPPORTED_FINAL_STATUSES = {'completed', 'incomplete'}
_RESPONSES_CALL_TYPES = {'computer_call', 'function_call'}
_RESPONSES_INPUT_ITEM_TYPES = {
    'agent_message',
    'compaction',
    'computer_call',
    'computer_call_output',
    'function_call',
    'function_call_output',
    'item_reference',
    'message',
    'reasoning',
}
_RESPONSES_OUTPUT_ITEM_TYPES = {
    'computer_call',
    'function_call',
    'message',
    'output_text',
    'reasoning',
}
_DESKTOP_SCREENSHOT_TEXT = 'Current desktop screenshot.'
_IMAGE_HISTORY_OMISSION_TEXT = '[previous image omitted from inference history]'


class OpenAIResponsesRequestError(RuntimeError):
    """Safe public error raised after a Responses request fails."""

    def __init__(self, status: int | None = None):
        message = 'OpenAI Responses request failed'
        if status is not None:
            message = f'{message} (status {status})'
        super().__init__(message)
        self.status = status


class EncryptedContentArchiveError(RuntimeError):
    """Raised after a Responses result cannot be archived for later recovery."""


class _OpenAIResponsesStateError(RuntimeError):
    def __init__(self, status: str | None, response: dict[str, Any]):
        super().__init__('OpenAI Responses returned a non-final or failed response')
        self.response_status = status
        self.body = response


def _json_value(value: Any) -> Any:
    """Serialize dicts and SDK/Pydantic objects without depending on generated classes."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        try:
            return _json_value(model_dump(mode='json', exclude_unset=True, warnings=False))
        except TypeError:
            try:
                return _json_value(model_dump(mode='json', exclude_unset=True))
            except TypeError:
                try:
                    return _json_value(model_dump(mode='json'))
                except TypeError:
                    return _json_value(model_dump())

    if hasattr(value, '__dict__'):
        return {str(key): _json_value(item) for key, item in vars(value).items() if not str(key).startswith('_')}
    return str(value)


def _response_header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    get = getattr(headers, 'get', None)
    value = get(name) if callable(get) else None
    if value not in (None, ''):
        return str(value)
    if isinstance(headers, Mapping):
        lowered_name = name.lower()
        for key, item in headers.items():
            if str(key).lower() == lowered_name and item not in (None, ''):
                return str(item)
    return None


def _encrypted_content_archive(
    response_data: Mapping[str, Any],
    *,
    headers: Any,
    requested_model: str | None,
    endpoint: str,
    request_sequence: int,
    gateway_cache_task_id: str | None,
) -> dict[str, Any]:
    output = response_data.get('output')
    output_items = output if isinstance(output, list) else []
    output_order: list[dict[str, Any]] = []
    reasoning_blocks: list[dict[str, Any]] = []
    for output_index, raw_item in enumerate(output_items):
        if not isinstance(raw_item, Mapping):
            output_order.append({'output_index': output_index, 'type': None, 'id': None})
            continue
        item_type = raw_item.get('type')
        item_id = raw_item.get('id')
        output_order.append({'output_index': output_index, 'type': item_type, 'id': item_id})
        encrypted_content = raw_item.get('encrypted_content')
        if item_type != 'reasoning' or encrypted_content in (None, ''):
            continue
        reasoning_blocks.append({
            'index': len(reasoning_blocks),
            'output_index': output_index,
            'reasoning_id': item_id,
            'status': raw_item.get('status'),
            'encrypted_content': encrypted_content,
        })

    response_id = response_data.get('id')
    model = response_data.get('model') or requested_model
    region = _response_header(headers, 'x-ms-region')
    missing_fields = []
    if reasoning_blocks and not region:
        missing_fields.append('x-ms-region')
    return {
        'schema_version': 'nexus-openai-responses-encrypted-content-v1',
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'request_sequence': request_sequence,
        'provider': 'openai',
        'protocol': 'responses',
        'endpoint': endpoint,
        'gateway_cache_task_id': gateway_cache_task_id,
        'response_id': response_id,
        'model': model,
        'region': region,
        'output_order': output_order,
        'reasoning_blocks': reasoning_blocks,
        'encrypted_block_count': len(reasoning_blocks),
        'archive_complete': not missing_fields,
        'missing_fields': missing_fields,
    }


def _write_encrypted_content_archive(archive_dir: Path, archive: Mapping[str, Any]) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(archive_dir, 0o700)
    response_id = str(archive.get('response_id') or 'unknown-response')
    safe_response_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', response_id).strip('._') or 'unknown-response'
    request_sequence = int(archive.get('request_sequence') or 0)
    final_path = archive_dir / f'{request_sequence:06d}-{safe_response_id}.enc.json'
    temp_path = archive_dir / f'.{final_path.name}.{uuid4().hex}.tmp'
    payload = json.dumps(_json_value(archive), ensure_ascii=False, indent=2).encode('utf-8') + b'\n'
    try:
        descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, final_path)
        os.chmod(final_path, 0o600)
    finally:
        temp_path.unlink(missing_ok=True)
    return final_path


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _tool_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return json.dumps(_json_value(content), ensure_ascii=False, separators=(',', ':'))


def _function_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(_json_value(arguments or {}), ensure_ascii=False, separators=(',', ':'))


def _responses_content(
    content: Any,
    *,
    omit_image_url: str | None = None,
    text_type: str = 'input_text',
) -> Any:
    """Convert Chat Completions content parts into Responses input content.

    Assistant replay uses ``text_type='output_text'``: Polarist/OpenAI reject
    ``input_text`` on ``role=assistant``.
    """

    if not isinstance(content, list):
        return content

    converted: list[Any] = []
    for original_part in content:
        part = _json_value(original_part)
        if not isinstance(part, Mapping):
            converted.append(part)
            continue
        part_type = part.get('type')
        if part_type == 'text':
            converted.append({'type': text_type, 'text': str(part.get('text') or '')})
            continue
        if part_type == 'image_url':
            image = part.get('image_url')
            if isinstance(image, Mapping):
                image_url = image.get('url')
                detail = image.get('detail')
            else:
                image_url = image
                detail = None
            if not isinstance(image_url, str) or not image_url:
                raise ValueError('Responses image_url content requires a non-empty URL')
            if omit_image_url is not None and image_url == omit_image_url:
                continue
            converted_image: dict[str, Any] = {'type': 'input_image', 'image_url': image_url}
            if detail is not None:
                converted_image['detail'] = detail
            converted.append(converted_image)
            continue
        converted.append(dict(part))
    return converted


def _function_tool_output(content: Any) -> Any:
    if not isinstance(content, list):
        return _tool_output(content)
    converted = _responses_content(content)
    if all(
        isinstance(part, Mapping) and part.get('type') in {'input_text', 'input_image', 'input_file'}
        for part in converted
    ):
        return converted
    return _tool_output(content)


def _responses_tools(tools: Any) -> Any:
    """Convert function schemas and allow only the native Computer Use tool.

    AIHub/Azure ``/v1/responses`` treats a missing ``strict`` as ``strict: true`` and then expands ``required`` to every property (forcing models to emit ``""`` for optional fields). OpenAI's documented default is non-strict, so when ``strict`` is absent we set ``false``. Explicit ``true`` / ``false`` are preserved.
    """

    if isinstance(tools, (str, bytes, Mapping)) or not isinstance(tools, Iterable):
        raise ValueError('OpenAI Responses tools must be an iterable of objects')

    converted: list[Any] = []
    for original_tool in tools:
        tool = _json_value(original_tool)
        if not isinstance(tool, Mapping):
            raise ValueError('OpenAI Responses tools must be objects')

        tool_type = tool.get('type')
        if tool_type == 'computer':
            converted.append(original_tool)
            continue
        if tool_type != 'function':
            raise ValueError(f'Unsupported OpenAI Responses native tool type {tool_type!r}')

        function = tool.get('function')
        if not isinstance(function, Mapping):
            if not isinstance(tool.get('name'), str) or not tool['name']:
                raise ValueError('OpenAI Responses function tools require a non-empty name')
            converted.append(_with_default_function_strict(tool))
            continue

        flattened = {key: value for key, value in tool.items() if key != 'function'}
        for key in ('name', 'description', 'parameters', 'strict'):
            if key not in flattened and key in function:
                flattened[key] = function[key]
        converted.append(_with_default_function_strict(flattened))

    return converted


def _with_default_function_strict(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Default missing function-tool ``strict`` to false for Responses upstreams."""
    normalized = dict(tool)
    if not isinstance(normalized.get('strict'), bool):
        normalized['strict'] = False
    return normalized


def _raise_reasoning_effort_to_floor(request: dict[str, Any], floor: str | None) -> str | None:
    """Lift ``reasoning.effort`` to the weakest value this wire actually accepts.

    Scaffolds ask for a no-thinking side request (a title, a summary) by sending
    ``reasoning.effort='none'``. Whether that is legal is a per-model fact, not a
    protocol one: measured 2026-10-16 on AIHub, gpt-6-sol / gpt-5.6-sol /
    deepseek-v4.1-flash accept ``none``, while gpt-6.1-sol and gpt-6-astra answer
    400 ``unsupported_value`` and want ``low`` (``minimal`` is rejected too). Left
    alone, every such side request is a wasted round trip -- 303 of 7142 calls
    (4.2%) in one WorkBuddy x gpt-6.1-sol run.

    The llm profile declares ``client_args.min_reasoning_effort`` for a wire with a
    floor; wires that accept ``none`` declare nothing and are untouched. Applied
    after the enforced-args policy on purpose: an illegal value stays illegal no
    matter which layer chose it.

    Returns the value replaced, so the caller can record that it happened.
    """
    if floor is None:
        return None
    if not isinstance(floor, str) or floor not in _REASONING_EFFORT_ORDER:
        raise ValueError(f'min_reasoning_effort must be one of {_REASONING_EFFORT_ORDER}, got {floor!r}')
    reasoning = request.get('reasoning')
    if not isinstance(reasoning, Mapping):
        return None
    effort = reasoning.get('effort')
    # An unknown spelling is the caller's business: raising it would be a guess.
    if not isinstance(effort, str) or effort not in _REASONING_EFFORT_ORDER:
        return None
    if _REASONING_EFFORT_ORDER.index(effort) >= _REASONING_EFFORT_ORDER.index(floor):
        return None
    request['reasoning'] = {**reasoning, 'effort': floor}
    return effort


_NESTED_CONTROL_KEYS = ('text', 'reasoning')


def _reapply_enforced_nested_controls(
    request: dict[str, Any],
    enforced_request_args: Mapping[str, Any] | None,
) -> None:
    """Restore enforced sub-keys of ``text`` / ``reasoning`` after normalization.

    The enforced policy is applied before normalization so that values injected by
    it are translated too. That ordering lets the translation write back into the
    same composite keys the policy owns: a request-level ``response_format`` or
    ``reasoning_effort`` becomes ``text.format`` / ``reasoning.effort`` and
    replaces whatever the policy put there. Re-applying per sub-key keeps the
    policy authoritative without discarding sibling keys the translation produced
    (an enforced ``text.format`` must not erase a translated ``text.verbosity``).
    """
    if not enforced_request_args:
        return
    for key in _NESTED_CONTROL_KEYS:
        enforced_value = enforced_request_args.get(key)
        if not isinstance(enforced_value, Mapping):
            continue
        current = request.get(key)
        if not isinstance(current, Mapping):
            # Normalization dropped or never created it; the policy value already
            # stands from the pre-normalization merge.
            continue
        merged = dict(current)
        merged.update(enforced_value)
        if merged != current:
            request[key] = merged


def _normalize_chat_request_controls(request: dict[str, Any]) -> None:
    """Translate Chat Completions controls to their Responses equivalents in place."""

    max_tokens = request.pop('max_tokens', None)
    if max_tokens is not None and request.get('max_output_tokens') is None:
        request['max_output_tokens'] = max_tokens

    # Hermes (and other chat clients targeting GPT-5+) send max_completion_tokens;
    # Responses uses max_output_tokens. Always pop so it never reaches the SDK.
    max_completion_tokens = request.pop('max_completion_tokens', None)
    if max_completion_tokens is not None and request.get('max_output_tokens') is None:
        request['max_output_tokens'] = max_completion_tokens

    reasoning_effort = request.pop('reasoning_effort', None)
    if reasoning_effort is not None:
        reasoning = request.get('reasoning')
        if reasoning is None:
            normalized_reasoning: dict[str, Any] = {}
        elif isinstance(reasoning, Mapping):
            normalized_reasoning = dict(reasoning)
        else:
            raise TypeError('openai_responses reasoning must be a mapping or null')

        normalized_reasoning['effort'] = reasoning_effort
        request['reasoning'] = normalized_reasoning

    # Reasoning models reject non-default sampling (Polarist 400:
    # "Unsupported parameter: 'temperature' is not supported with this model.").
    if request.get('reasoning') is not None:
        for sampling in ('temperature', 'top_p', 'top_k'):
            request.pop(sampling, None)

    response_format = request.pop('response_format', None)
    verbosity = request.pop('verbosity', None)
    if response_format is not None or verbosity is not None:
        text = request.get('text')
        if text is None:
            normalized_text: dict[str, Any] = {}
        elif isinstance(text, Mapping):
            normalized_text = dict(text)
        else:
            raise TypeError('openai_responses text must be a mapping or null')

        if response_format is not None:
            serialized_format = _json_value(response_format)
            if not isinstance(serialized_format, Mapping):
                raise TypeError('openai_responses response_format must be a mapping or null')

            format_type = serialized_format.get('type')
            if format_type == 'json_schema':
                json_schema = serialized_format.get('json_schema')
                if not isinstance(json_schema, Mapping):
                    raise TypeError('openai_responses json_schema response_format requires a mapping')
                normalized_text['format'] = {'type': 'json_schema', **dict(json_schema)}
            elif format_type in {'json_object', 'text'}:
                normalized_text['format'] = {'type': format_type}
            else:
                raise ValueError(f'Unsupported openai_responses response_format type {format_type!r}')

        if verbosity is not None:
            normalized_text['verbosity'] = verbosity
        request['text'] = normalized_text


def _validate_responses_item_type(
    item: Mapping[str, Any],
    supported_types: set[str],
    *,
    location: str,
) -> str:
    item_type = item.get('type')
    if not isinstance(item_type, str):
        raise ValueError(f'OpenAI Responses {location} item requires a supported type')
    if item_type not in supported_types:
        raise ValueError(f'Unsupported OpenAI Responses {location} item type {item_type!r}')
    return item_type


def _register_response_call(call_items: dict[str, dict[str, Any]], item: Any) -> None:
    data = _json_value(item)
    if not isinstance(data, Mapping) or data.get('type') not in _RESPONSES_CALL_TYPES:
        return
    call_id = data.get('call_id')
    if isinstance(call_id, str) and call_id:
        call_items[call_id] = dict(data)


def _assistant_response_items(item: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    fields = item.get('provider_specific_fields')
    if not isinstance(fields, Mapping):
        return None
    raw_items = fields.get('responses_output_items')
    if not isinstance(raw_items, list):
        return None
    serialized = _json_value(raw_items)
    if not isinstance(serialized, list) or not all(isinstance(raw, Mapping) for raw in serialized):
        raise ValueError('responses_output_items must be a list of Responses item objects')
    return [dict(raw) for raw in serialized]


def _content_image_url(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    for original_part in content:
        part = _json_value(original_part)
        if not isinstance(part, Mapping):
            continue
        if part.get('type') == 'input_image':
            image_url = part.get('image_url')
            if isinstance(image_url, str) and image_url:
                return image_url
        if part.get('type') != 'image_url':
            continue
        image = part.get('image_url')
        image_url = image.get('url') if isinstance(image, Mapping) else image
        if isinstance(image_url, str) and image_url:
            return image_url
    return None


def _content_has_omitted_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, Mapping)
        and part.get('type') in {'text', 'input_text'}
        and part.get('text') == _IMAGE_HISTORY_OMISSION_TEXT
        for part in content
    )


def _without_omitted_image_parts(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [
        part
        for part in content
        if not (
            isinstance(part, Mapping)
            and part.get('type') in {'text', 'input_text'}
            and part.get('text') == _IMAGE_HISTORY_OMISSION_TEXT
        )
    ]


def _following_user_content(messages: list[Any], tool_index: int) -> tuple[int, Any] | None:
    for candidate_index in range(tool_index + 1, len(messages)):
        candidate = _json_value(messages[candidate_index])
        if not isinstance(candidate, Mapping):
            return None
        role = candidate.get('role')
        if role == 'tool':
            continue
        if role != 'user':
            return None
        return candidate_index, candidate.get('content')
    return None


def _collect_response_calls(messages: list[Any]) -> dict[str, dict[str, Any]]:
    response_calls: dict[str, dict[str, Any]] = {}
    for original in messages:
        item = _json_value(original)
        if not isinstance(item, Mapping):
            continue
        if 'type' in item:
            _register_response_call(response_calls, item)
            continue
        if item.get('role') != 'assistant':
            continue
        for raw_response_item in _assistant_response_items(item) or []:
            _register_response_call(response_calls, raw_response_item)
    return response_calls


def _omitted_computer_history(
    messages: list[Any],
    response_calls: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], set[int]]:
    omitted_call_ids: set[str] = set()
    omitted_message_indices: set[int] = set()
    for message_index, original in enumerate(messages):
        item = _json_value(original)
        if not isinstance(item, Mapping) or item.get('role') != 'tool':
            continue
        call_id = str(item.get('tool_call_id') or item.get('call_id') or '')
        source_call = response_calls.get(call_id)
        if source_call is None or source_call.get('type') != 'computer_call':
            continue
        if _content_image_url(item.get('content')) is not None:
            continue
        following = _following_user_content(messages, message_index)
        if following is None:
            continue
        screenshot_index, screenshot_content = following
        if _content_image_url(screenshot_content) is None and _content_has_omitted_image(screenshot_content):
            omitted_call_ids.add(call_id)
            omitted_message_indices.add(screenshot_index)
    return omitted_call_ids, omitted_message_indices


def _native_tool_output(
    source_call: Mapping[str, Any] | None,
    call_id: str,
    content: Any,
    *,
    screenshot_url: str | None = None,
) -> dict[str, Any]:
    call_type = source_call.get('type') if source_call is not None else 'function_call'
    if call_type == 'function_call':
        return {'type': 'function_call_output', 'call_id': call_id, 'output': _function_tool_output(content)}
    if call_type == 'computer_call':
        if screenshot_url is None:
            raise ValueError(f'Responses computer_call {call_id!r} requires a screenshot result')
        output: dict[str, Any] = {
            'type': 'computer_call_output',
            'call_id': call_id,
            'output': {'type': 'computer_screenshot', 'image_url': screenshot_url},
        }
        safety_checks = source_call.get('pending_safety_checks') if source_call is not None else None
        if isinstance(safety_checks, list) and safety_checks:
            output['acknowledged_safety_checks'] = _json_value(safety_checks)
        return output
    raise ValueError(f'Unsupported Responses tool call type {call_type!r}')


def _align_tool_outputs(items: list[Any], call_items: list[Any] | None = None) -> list[Any]:
    """Keep contiguous result batches in call order for backends that render them positionally."""

    # Bind each batch to the latest preceding calls; call_id may be reused across turns.
    call_order: dict[Any, int] = {}
    position = 0

    def remember(item: Any) -> None:
        nonlocal position
        if _get(item, 'type') in _RESPONSES_CALL_TYPES and _get(item, 'call_id'):
            call_order[_get(item, 'call_id')] = position
        position += 1

    for item in call_items or []:
        remember(item)
    aligned: list[Any] = []
    for is_output, group in groupby(
        items, key=lambda item: _get(item, 'type') in {'function_call_output', 'computer_call_output'}
    ):
        batch = list(group)
        if is_output:
            batch.sort(key=lambda item: call_order.get(_get(item, 'call_id'), position))
        else:
            for item in batch:
                remember(item)
        aligned.extend(batch)
    return items if all(a is b for a, b in zip(items, aligned, strict=True)) else aligned


def _tail_input_since_last_assistant(messages: list[Any]) -> list[Any]:
    """Convert messages added since the last assistant response.

    The parent response ID represents earlier history. Use the full input for
    the first request or when no new messages follow the last assistant response.
    """
    last_assistant = -1
    for index, message in enumerate(messages):
        role = message.get('role') if isinstance(message, Mapping) else getattr(message, 'role', None)
        if getattr(role, 'value', role) == 'assistant':
            last_assistant = index
    if last_assistant < 0:
        return _responses_input(messages)
    tail = messages[last_assistant + 1 :]
    # Use the full input when the new-message slice is empty.
    return _responses_input(tail) if tail else _responses_input(messages)


def _responses_input(messages: list[Any]) -> list[Any]:
    """Convert Chat Completions history and align native tool results to their calls."""

    converted: list[Any] = []
    changed = False
    indexed_response_calls = _collect_response_calls(messages)
    omitted_call_ids, omitted_message_indices = _omitted_computer_history(messages, indexed_response_calls)
    response_calls: dict[str, dict[str, Any]] = {}
    consumed_screenshots: dict[int, str] = {}
    for message_index, original in enumerate(messages):
        item = _json_value(original)
        if isinstance(item, Mapping) and 'type' in item:
            _validate_responses_item_type(item, _RESPONSES_INPUT_ITEM_TYPES, location='input')
        if not isinstance(item, Mapping) or 'type' in item:
            if (
                isinstance(item, Mapping)
                and item.get('type') == 'computer_call'
                and item.get('call_id') in omitted_call_ids
            ):
                changed = True
                continue
            converted.append(original)
            _register_response_call(response_calls, item)
            continue

        role = item.get('role')
        if role == 'tool':
            changed = True
            call_id = item.get('tool_call_id') or item.get('call_id') or ''
            if str(call_id) in omitted_call_ids:
                continue
            source_call = response_calls.get(str(call_id))
            screenshot_url = _content_image_url(item.get('content'))
            if source_call is not None and source_call.get('type') == 'computer_call' and screenshot_url is None:
                following = _following_user_content(messages, message_index)
                if following is not None:
                    screenshot_index, screenshot_content = following
                    screenshot_url = _content_image_url(screenshot_content)
                    if screenshot_url is not None:
                        if screenshot_index in consumed_screenshots:
                            raise ValueError('One desktop screenshot cannot satisfy multiple Responses computer calls')
                        consumed_screenshots[screenshot_index] = screenshot_url
            converted.append(
                _native_tool_output(
                    source_call,
                    str(call_id),
                    item.get('content'),
                    screenshot_url=screenshot_url,
                )
            )
            continue

        if role not in {'developer', 'system', 'user', 'assistant'}:
            converted.append(original)
            continue

        changed = True
        if role == 'assistant':
            raw_response_items = _assistant_response_items(item)
            if raw_response_items is not None:
                for raw_response_item in raw_response_items:
                    _validate_responses_item_type(
                        raw_response_item,
                        _RESPONSES_INPUT_ITEM_TYPES,
                        location='input',
                    )
                    if (
                        raw_response_item.get('type') == 'computer_call'
                        and raw_response_item.get('call_id') in omitted_call_ids
                    ):
                        continue
                    converted.append(raw_response_item)
                    _register_response_call(response_calls, raw_response_item)
                continue

        content = item.get('content')
        if content is not None:
            omit_image_url = consumed_screenshots.get(message_index)
            if message_index in omitted_message_indices:
                content = _without_omitted_image_parts(content)
            responses_content = _responses_content(
                content,
                omit_image_url=omit_image_url,
                text_type='output_text' if role == 'assistant' else 'input_text',
            )
            if not (
                role == 'user'
                and (omit_image_url is not None or message_index in omitted_message_indices)
                and responses_content == [{'type': 'input_text', 'text': _DESKTOP_SCREENSHOT_TEXT}]
            ):
                converted.append({'role': role, 'content': responses_content})

        if role != 'assistant':
            continue
        for reasoning_item in item.get('reasoning_items') or []:
            serialized_reasoning = _json_value(reasoning_item)
            if isinstance(serialized_reasoning, Mapping):
                converted.append(dict(serialized_reasoning))
        for raw_tool_call in item.get('tool_calls') or []:
            tool_call = _json_value(raw_tool_call)
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get('function')
            if not isinstance(function, Mapping):
                continue
            function_call = {
                'type': 'function_call',
                'call_id': tool_call.get('id') or '',
                'name': function.get('name') or '',
                'arguments': _function_arguments(function.get('arguments')),
            }
            converted.append(function_call)
            _register_response_call(response_calls, function_call)

    return _align_tool_outputs(converted if changed else messages)


def _append_endpoint_path(base_url: Any, path: str) -> str:
    value = str(base_url or '')
    try:
        parsed = urlsplit(value)
    except ValueError:
        return f'{value.rstrip("/")}/{path.lstrip("/")}'
    combined_path = f'{parsed.path.rstrip("/")}/{path.lstrip("/")}'
    return urlunsplit((parsed.scheme, parsed.netloc, combined_path, parsed.query, parsed.fragment))


def _ensure_v1_base_url(base_url: Any) -> str:
    value = str(base_url or '')
    try:
        parsed = urlsplit(value)
    except ValueError:
        path = value.rstrip('/')
        return path if path.endswith('/v1') else f'{path}/v1'
    if parsed.path.rstrip('/').endswith('/v1'):
        return value
    return _append_endpoint_path(value, 'v1')


def _responses_transport(client_args: Mapping[str, Any]) -> tuple[Any, str]:
    """Return the SDK base URL and the exact Responses endpoint it will address."""

    endpoint = client_args.get('endpoint')
    if endpoint is None:
        base_url = client_args.get('base_url') or 'https://api.openai.com/v1'
        if client_args.get('ensure_v1_path'):
            base_url = _ensure_v1_base_url(base_url)
        return base_url, _append_endpoint_path(base_url, 'responses')

    endpoint_value = str(endpoint)
    try:
        parsed = urlsplit(endpoint_value)
        hostname = parsed.hostname
        _ = parsed.port
    except (TypeError, ValueError) as error:
        raise ValueError('openai_responses endpoint must be a valid HTTP(S) /responses URL') from error
    if (
        parsed.scheme.lower() not in {'http', 'https'}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith('/responses')
        or any(character.isspace() for character in endpoint_value)
    ):
        raise ValueError('openai_responses endpoint must be a safe HTTP(S) URL ending in /responses')

    endpoint_path = parsed.path
    base_path = endpoint_path[: -len('responses')]
    normalized_endpoint = urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, '', ''))
    sdk_base_url = urlunsplit((parsed.scheme, parsed.netloc, base_path, '', ''))
    return sdk_base_url, normalized_endpoint


def _error_status(error: BaseException) -> int | None:
    for attribute in ('status', 'status_code'):
        status = getattr(error, attribute, None)
        if isinstance(status, int):
            return status
    response = getattr(error, 'response', None)
    status = getattr(response, 'status_code', None)
    return status if isinstance(status, int) else None


def _status_is_retryable(status: int | None) -> bool:
    return status in _RETRYABLE_STATUS_CODES or (status is not None and status >= 500)


def _response_error_code(response: Mapping[str, Any]) -> str | None:
    response_error = response.get('error')
    if not isinstance(response_error, Mapping):
        return None
    code = response_error.get('code')
    return code if isinstance(code, str) else None


def _is_retryable(error: BaseException) -> bool:
    if (
        isinstance(error, _OpenAIResponsesStateError)
        and error.response_status == 'failed'
        and _response_error_code(error.body) in _RETRYABLE_RESPONSE_ERROR_CODES
    ):
        return True
    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            httpx.TransportError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
        ),
    ):
        return True
    if _status_is_retryable(_error_status(error)):
        return True
    cause = error.__cause__
    return cause is not None and cause is not error and _is_retryable(cause)


def _input_fingerprints(items: list[Any]) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(json.dumps(_json_value(item), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for item in items
    )


def _require_self_contained_items(items: list[Any]) -> None:
    for raw_item in items:
        item = _json_value(raw_item)
        if not isinstance(item, Mapping) or item.get('type') == 'item_reference':
            raise ValueError('Encrypted history cannot contain bare server-side item references')
        if item.get('type') in {'reasoning', 'compaction'}:
            encrypted = item.get('encrypted_content')
            if not isinstance(encrypted, str) or not encrypted:
                raise ValueError('Native reasoning is missing its encrypted_content; refusing lossy continuation')


def _require_complete_encrypted_output(response: Mapping[str, Any]) -> None:
    output = response.get('output') or []
    _require_self_contained_items(output)
    has_state = any(item.get('type') in {'reasoning', 'compaction'} for item in output)
    usage = response.get('usage') or {}
    reasoning_tokens = (usage.get('output_tokens_details') or {}).get('reasoning_tokens')
    if not has_state and reasoning_tokens != 0:
        raise ValueError('Response does not account for its native reasoning state')


def _is_content_policy_error(error: BaseException) -> bool:
    body = getattr(error, 'body', None)
    # SDK status errors may unwrap the top-level `error` object.
    if isinstance(body, Mapping) and (
        body.get('code') == 'content_policy_violation' or _response_error_code(body) == 'content_policy_violation'
    ):
        return True
    cause = error.__cause__
    return cause is not None and cause is not error and _is_content_policy_error(cause)


def _contains_context_length_marker(value: Any) -> bool:
    value = _json_value(value)
    if isinstance(value, str):
        normalized = re.sub(r'[\s-]+', '_', value.lower())
        return any(
            marker in normalized
            for marker in (
                'context_length_exceeded',
                'context_window_exceeded',
                'input_too_long',
                'max_context_length',
                'maximum_context_length',
            )
        )
    if isinstance(value, Mapping):
        return any(
            _contains_context_length_marker(key) or _contains_context_length_marker(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_context_length_marker(item) for item in value)
    return False


def _is_context_length_error(error: BaseException) -> bool:
    if _error_status(error) == 413 or _contains_context_length_marker(str(error)):
        return True
    for attribute in ('body', 'code', 'type'):
        if _contains_context_length_marker(getattr(error, attribute, None)):
            return True
    cause = error.__cause__
    return cause is not None and cause is not error and _is_context_length_error(cause)


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
    provider_call['error'] = {'type': type(error).__name__, 'message': message}
    if status is not None:
        provider_call['error']['status_code'] = status


# Request-echo modes for Responses ``reasoning.summary``. Never treat these as
# visible thinking text — they are config enums, not model summaries.
_REASONING_SUMMARY_MODES = frozenset({'auto', 'concise', 'detailed', 'none'})


def _summary_text(item: Mapping[str, Any]) -> list[str]:
    for field in ('summary', 'content'):
        texts: list[str] = []
        value = item.get(field)
        if isinstance(value, str) and value:
            if value.strip().lower() in _REASONING_SUMMARY_MODES:
                continue
            texts.append(value)
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, Mapping):
                    text = part.get('text')
                    if isinstance(text, str) and text:
                        texts.append(text)
        if texts:
            return texts
    return []


def _computer_actions(item: Any) -> list[Any]:
    actions = _get(item, 'actions')
    if actions is not None:
        if isinstance(actions, (list, tuple)):
            return [_json_value(action) for action in actions]
        return [_json_value(actions)]
    action = _get(item, 'action')
    return [] if action is None else [_json_value(action)]


def _usage_from_response(value: Any) -> ExtendedUsage | None:
    usage = _json_value(value)
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
        return None

    prompt_tokens = int(usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0)
    completion_tokens = int(usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0)
    total_tokens = int(usage.get('total_tokens', prompt_tokens + completion_tokens) or 0)

    input_details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details')
    cached_tokens = 0
    cache_write_tokens = 0
    prompt_details = None
    if isinstance(input_details, Mapping):
        cached_tokens = int(input_details.get('cached_tokens', 0) or 0)
        cache_write_tokens = int(
            input_details.get('cache_write_tokens', input_details.get('cache_creation_tokens', 0)) or 0
        )
        prompt_details = ExtendedPromptTokensDetails(
            audio_tokens=input_details.get('audio_tokens'),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_write_tokens,
        )

    output_details = usage.get('output_tokens_details') or usage.get('completion_tokens_details')
    completion_details = None
    if isinstance(output_details, Mapping):
        completion_details = ExtendedCompletionTokensDetails(
            accepted_prediction_tokens=output_details.get('accepted_prediction_tokens'),
            audio_tokens=output_details.get('audio_tokens'),
            reasoning_tokens=output_details.get('reasoning_tokens'),
            rejected_prediction_tokens=output_details.get('rejected_prediction_tokens'),
        )

    return ExtendedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        cache_creation_input_tokens=cache_write_tokens,
        cache_read_input_tokens=cached_tokens,
    )


def _incomplete_finish_reason(response: Mapping[str, Any]) -> str:
    incomplete_details = response.get('incomplete_details')
    if not isinstance(incomplete_details, Mapping):
        return 'incomplete'
    reason = incomplete_details.get('reason')
    if not isinstance(reason, str) or not reason:
        return 'incomplete'
    if reason == 'max_output_tokens':
        return 'length'
    return reason


def _serialize_response(response: Any) -> dict[str, Any]:
    response_data = _json_value(response)
    if not isinstance(response_data, Mapping):
        raise TypeError('OpenAI Responses returned an invalid response object')
    return dict(response_data)


def _convert_response_data(
    response_data: dict[str, Any],
    recorded_response: dict[str, Any],
    requested_model: str | None,
) -> ExtendedChatCompletion:
    """Convert a serialized Responses payload into a completion."""

    status_value = response_data.get('status')
    status = str(status_value) if status_value is not None else None
    if response_data.get('error') is not None or status not in _SUPPORTED_FINAL_STATUSES:
        raise _OpenAIResponsesStateError(status, response_data)
    if not response_data.get('id'):
        raise ValueError('OpenAI Responses returned a response without an id')

    output = response_data.get('output') or []
    if not isinstance(output, list):
        raise TypeError('OpenAI Responses output must be a list')
    recorded_output = recorded_response.get('output')
    if not isinstance(recorded_output, list):
        recorded_output = []

    content_parts: list[str] = []
    refusals: list[str] = []
    reasoning_summaries: list[str] = []
    reasoning_items: list[dict[str, Any]] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []

    for raw_item in output:
        if not isinstance(raw_item, Mapping):
            raise ValueError('OpenAI Responses output item requires a supported type')
        item = dict(raw_item)
        item_type = _validate_responses_item_type(item, _RESPONSES_OUTPUT_ITEM_TYPES, location='output')

        if item_type == 'message':
            message_content = item.get('content') or []
            if isinstance(message_content, list):
                for part in message_content:
                    if not isinstance(part, Mapping):
                        continue
                    part_data = part
                    if part_data.get('type') == 'output_text':
                        text = part_data.get('text')
                        if isinstance(text, str):
                            content_parts.append(text)
                    elif part_data.get('type') == 'refusal':
                        refusal = part_data.get('refusal', part_data.get('text'))
                        if isinstance(refusal, str):
                            refusals.append(refusal)
            continue

        if item_type == 'output_text':
            text = item.get('text')
            if isinstance(text, str):
                content_parts.append(text)
            continue

        if item_type == 'reasoning':
            reasoning_items.append(item)
            reasoning_summaries.extend(_summary_text(item))
            continue

        if item_type == 'function_call':
            arguments = item.get('arguments', '')
            if not isinstance(arguments, str):
                arguments = json.dumps(_json_value(arguments), ensure_ascii=False, separators=(',', ':'))
            if not item.get('call_id'):
                if status == 'incomplete':
                    continue
                raise ValueError('OpenAI Responses function_call is missing call_id')
            call_id = str(item['call_id'])
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=call_id,
                    type='function',
                    function=Function(
                        name=str(item.get('name') or ''),
                        arguments=arguments,
                    ),
                )
            )
            continue

        if item_type == 'computer_call':
            actions = _computer_actions(item)
            safety_checks = item.get('pending_safety_checks') or []
            arguments = {
                'actions': actions,
                'pending_safety_checks': safety_checks,
                'raw_item': item,
            }
            if not item.get('call_id'):
                if status == 'incomplete':
                    continue
                raise ValueError('OpenAI Responses computer_call is missing call_id')
            call_id = str(item['call_id'])
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=call_id,
                    type='function',
                    function=Function(
                        name='computer',
                        arguments=json.dumps(arguments, ensure_ascii=False, separators=(',', ':')),
                    ),
                )
            )

    # Top-level ``reasoning`` may be the request echo ({effort, summary: auto|...}).
    # Only harvest list-shaped summary/content fields as visible thinking text.
    top_level_reasoning = response_data.get('reasoning')
    if isinstance(top_level_reasoning, Mapping):
        summary = top_level_reasoning.get('summary')
        content = top_level_reasoning.get('content')
        if isinstance(summary, list) or isinstance(content, list):
            reasoning_summaries.extend(_summary_text(top_level_reasoning))

    finish_reason = (
        _incomplete_finish_reason(response_data) if status == 'incomplete' else 'tool_calls' if tool_calls else 'stop'
    )
    message = ExtendedMessage(
        role='assistant',
        # Empty string (not None): chat_tracker omits null fields on dump; SFT schema
        # expects assistant ``content`` to always be present on tool-only turns.
        content='\n'.join(content_parts) if content_parts else '',
        refusal='\n'.join(refusals) if refusals else None,
        tool_calls=tool_calls or None,
        reasoning_content='\n'.join(reasoning_summaries) if reasoning_summaries else None,
        reasoning_items=reasoning_items or None,
        provider_specific_fields={'responses_output_items': [dict(item) for item in output]},
    )
    response_id = str(response_data['id'])
    response_model_value = response_data.get('model') or requested_model
    response_model = str(response_model_value) if response_model_value is not None else None
    created = int(response_data.get('created_at') or 0)
    provider_fields = {
        'provider': 'openai',
        'protocol': 'responses',
        'response_id': recorded_response.get('id'),
        'response_model': recorded_response.get('model'),
        'response_status': status,
        'response_output': recorded_output,
        'response_error': recorded_response.get('error'),
        'response_incomplete_details': recorded_response.get('incomplete_details'),
        'response_state': {key: value for key, value in recorded_response.items() if key != 'output'},
    }
    return ExtendedChatCompletion(
        id=response_id,
        choices=[
            ExtendedChoice(
                index=0,
                finish_reason=finish_reason,
                message=message,
            )
        ],
        created=created,
        model=response_model,
        usage=_usage_from_response(response_data.get('usage')),
        provider_specific_fields=provider_fields,
    )


def convert_response(
    response: Any,
    requested_model: str | None = None,
) -> ExtendedChatCompletion:
    """Convert a non-streaming Responses object into Nexus' OpenAI-shaped completion."""

    response_data = _serialize_response(response)
    return _convert_response_data(response_data, response_data, requested_model)


@openai_client_registry.register('openai_responses')
class OpenAIResponsesClient(OpenAICompatibleClient):
    """Responses transport with both native and OpenAI-shaped Nexus interfaces."""

    supports_native_responses_compaction = True
    supports_agent_message = True

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.capture_raw_io = bool(config.client_args.get('capture_raw_io', False))
        archive_dir = config.client_args.get('encrypted_content_archive_dir')
        self.encrypted_content_archive_dir = Path(str(archive_dir)) if archive_dir not in (None, '') else None
        self._archive_request_sequence = 0
        self.max_retries = max(1, int(config.client_args.get('max_retries', 5)))
        self._regions: dict[str, dict[str, str]] = {}
        transport_base_url, self.endpoint = _responses_transport(config.client_args)

        # Continuation mode is explicit per model. Stateful Responses use the
        # server's parent ID. Stateless reasoning uses complete native output
        # with include=['reasoning.encrypted_content']; availability and account
        # affinity must be validated on the actual provider. Each run owns its
        # client, so neither parent IDs nor history fingerprints cross tasks.
        self.stateful_previous_response = bool(config.client_args.get('stateful_previous_response', False))
        self._last_response_id: str | None = None
        self._last_request_input_hash: str | None = None
        self._last_request_parent_id: str | None = None
        self.require_encrypted_reasoning = bool(config.client_args.get('require_encrypted_reasoning', False))
        if self.require_encrypted_reasoning and self.stateful_previous_response:
            raise ValueError('require_encrypted_reasoning uses full history, not stateful_previous_response')
        self._last_encrypted_input: tuple[str, ...] | None = None
        self._next_encrypted_prefix: tuple[str, ...] | None = None

        client_args = {key: value for key, value in config.client_args.items() if key not in _NEXUS_CLIENT_ARGS}
        gateway_routing = config.client_args.get('gateway_routing')
        if gateway_routing:
            client_args['api_key'] = merge_gateway_routing(client_args.get('api_key'), gateway_routing)
        if config.client_args.get('endpoint') is not None or (
            config.client_args.get('ensure_v1_path') and 'base_url' in client_args
        ):
            client_args['base_url'] = transport_base_url

        self.gateway_cache_task_id: str | None = None
        cache_task_id = config.client_args.get('cache_task_id')
        if cache_task_id not in (None, ''):
            api_key = client_args.get('api_key')
            if not isinstance(api_key, str) or not api_key:
                raise ValueError('openai_responses cache_task_id requires a non-empty api_key')
            cache_hash = hashlib.md5(str(cache_task_id).encode()).hexdigest()
            self.gateway_cache_task_id = cache_hash
            client_args['api_key'] = normalize_cache_task_id(api_key, cache_hash)

        client_args['max_retries'] = 0
        self.client = openai.AsyncOpenAI(**client_args)

    def _validate_encrypted_request(self, request: Mapping[str, Any], messages: list[Any]) -> None:
        if request.get('previous_response_id') is not None or request.get('conversation') is not None:
            raise ValueError('Encrypted full-history requests cannot also reference server-side conversation state')
        include = request.get('include')
        if not isinstance(include, list) or 'reasoning.encrypted_content' not in include:
            raise ValueError('Encrypted reasoning requires include=[reasoning.encrypted_content]')
        items = request['input']
        _require_self_contained_items(items)
        for message in messages:
            item = _json_value(message)
            if (
                isinstance(item, Mapping)
                and item.get('role') == 'assistant'
                and 'type' not in item
                and _assistant_response_items(item) is None
            ):
                raise ValueError('Assistant history is missing the original native Responses output')
        fingerprints = _input_fingerprints(items)
        if (
            self._next_encrypted_prefix is not None
            and fingerprints != self._last_encrypted_input
            and fingerprints[: len(self._next_encrypted_prefix)] != self._next_encrypted_prefix
        ):
            raise ValueError('Encrypted history was changed or pruned; refusing lossy continuation')

    def _new_provider_call(self) -> dict[str, Any]:
        return {
            'provider': 'openai',
            'protocol': 'responses',
            'model': '',
            'endpoint': self.endpoint,
            'attempts': [],
        }

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
        if body is None:
            return None
        response: dict[str, Any] = {'body': _json_value(body)}
        status = _error_status(error)
        if status is not None:
            response['status_code'] = status
        provider_call['response'] = response
        return response

    def _prepare_native_request(self, request_body: Mapping[str, Any]) -> dict[str, Any]:
        defaults = dict(_json_value(self.config.request_args))
        _normalize_chat_request_controls(defaults)
        request = {**defaults, **dict(_json_value(request_body))}
        if isinstance(defaults.get('reasoning'), Mapping) and isinstance(request.get('reasoning'), Mapping):
            request['reasoning'] = {**defaults['reasoning'], **request['reasoning']}
        # The CLI's catalog identity is not necessarily the actual routed model.
        if defaults.get('model'):
            request['model'] = defaults['model']
            if isinstance(request.get('extra_body'), Mapping):
                request['extra_body'] = {key: value for key, value in request['extra_body'].items() if key != 'model'}
        request = self._resolve_effective_request_args(request)
        if not request.get('model'):
            raise ValueError('openai_responses requires a model')
        if request.get('background') is True:
            raise ValueError('openai_responses does not support background=true')
        # The bridge emits SSE and keepalives; the host always requests a complete object.
        request['stream'] = False
        cache_id = self.config.client_args.get('prompt_cache_task_id')
        if cache_id:
            request.setdefault('prompt_cache_key', hashlib.sha256(str(cache_id).encode()).hexdigest())
        return request

    async def create_responses(self, request_body: Mapping[str, Any]) -> dict[str, Any]:
        """Forward native items/tools without the Chat conversion whitelist.

        Use the same SDK transport, credentials, policy and retry loop as
        create_completion. Unknown provider fields stay JSON, not SDK kwargs.
        """
        provider_call = self._new_provider_call()
        try:
            request = self._prepare_native_request(request_body)
            provider_call['model'] = _json_value(request['model'])
            provider_call['_provider_sensitive_source'] = _json_value(request)
            if self.capture_raw_io:
                provider_call['request'] = _json_value(request)
        except Exception as error:
            self._record_failure(provider_call, error)
            raise OpenAIResponsesRequestError(_error_status(error)) from error
        return await self._execute_response(request, provider_call, native=True)

    async def compact_response(self, request_body: Mapping[str, Any]) -> dict[str, Any]:
        """Forward one native Responses compaction request to the provider."""
        request = dict(_json_value(request_body))
        configured_model = self.config.request_args.get('model')
        enforced_model = self.config.enforced_request_args.get('model')
        if enforced_model not in (None, ''):
            request['model'] = enforced_model
        elif configured_model not in (None, ''):
            request['model'] = configured_model

        final_model = request.get('model')
        provider_call = self._new_provider_call()
        provider_call['model'] = _json_value(final_model or '')
        provider_call['endpoint'] = f'{self.endpoint}/compact'
        provider_call['_provider_sensitive_source'] = _json_value(request)
        if self.capture_raw_io:
            provider_call['request'] = _json_value(request)
        if not final_model:
            error = ValueError('openai_responses compact requires a model')
            self._record_failure(provider_call, error)
            raise OpenAIResponsesRequestError() from error

        # Enforce separate limits for each request and the accumulated retry time.
        _retry_deadline = _resolve_retry_total_timeout()
        _t0 = time.monotonic()
        _create_compact = with_call_deadline()(self.client.responses.compact)
        for attempt in range(1, self.max_retries + 1):
            if _retry_deadline > 0 and time.monotonic() - _t0 >= _retry_deadline:
                error = TimeoutError(
                    f'openai_responses: retry total budget {_retry_deadline}s exhausted after {attempt - 1} attempt(s)'
                )
                self._record_failure(provider_call, error)
                raise OpenAIResponsesRequestError() from error
            try:
                response = await _create_compact(**request)
                response_data = _serialize_response(response)
                if self.capture_raw_io:
                    provider_call['response'] = response_data
            except asyncio.CancelledError as error:
                provider_call['attempts'].append(_attempt_error(attempt, error))
                self._record_failure(provider_call, error)
                raise
            except Exception as error:
                attempt_record = _attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                if _is_retryable(error) and attempt < self.max_retries:
                    try:
                        await asyncio.sleep(min(2 ** (attempt - 1), 30))
                    except asyncio.CancelledError as cancellation:
                        self._record_failure(provider_call, cancellation)
                        raise
                    continue
                self._record_failure(provider_call, error)
                raise OpenAIResponsesRequestError(_error_status(error)) from error

            provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
            self._record_provider_call(provider_call)
            return response_data

        raise RuntimeError('OpenAI Responses compact retry loop exhausted without a response')

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        provider_call = self._new_provider_call()
        stateful_input_hash: str | None = None
        try:
            request_args_override = kwargs.pop('_request_args_override', None)
            request = {
                **(request_args_override if request_args_override is not None else self.config.request_args),
                **kwargs,
            }
            configured_model = request.pop('model', None)
            request['model'] = model if model is not None else configured_model
            request['input'] = _responses_input(messages)
            # Send only new input when the parent response ID already represents earlier history.
            if self.stateful_previous_response:
                stateful_input_hash = hashlib.sha256(
                    json.dumps(request['input'], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
                ).hexdigest()
                parent_id = self._last_response_id
                if stateful_input_hash == self._last_request_input_hash:
                    # The caller rejected the last response and is retrying the
                    # same history. Branch from its original parent; otherwise
                    # old tool outputs would be attached to new tool calls.
                    parent_id = self._last_request_parent_id
                if parent_id:
                    request['previous_response_id'] = parent_id
                    # The delta omits call declarations; retain their order from the full input.
                    request['input'] = _align_tool_outputs(_tail_input_since_last_assistant(messages), request['input'])

            request = self._apply_enforced_request_args(request)
            _normalize_chat_request_controls(request)
            _reapply_enforced_nested_controls(request, self.config.enforced_request_args)
            lifted_effort = _raise_reasoning_effort_to_floor(
                request, self.config.client_args.get('min_reasoning_effort')
            )
            if lifted_effort is not None:
                provider_call['reasoning_effort_floor'] = _json_value(
                    f'{lifted_effort}->{request["reasoning"]["effort"]}'
                )
            if request.get('tools') is None:
                request.pop('tools', None)
            else:
                request['tools'] = _responses_tools(request['tools'])
            effective_request = (
                self._resolve_effective_request_args(request)
                if self.config.enforced_request_args or self.require_encrypted_reasoning
                else request
            )
            if self.require_encrypted_reasoning:
                self._validate_encrypted_request(effective_request, messages)
            final_model = effective_request.get('model')
            provider_call['model'] = _json_value(final_model or '')
            if not final_model:
                raise ValueError('openai_responses requires a model in request_args or create_completion(model=...)')
            if effective_request.get('stream') is True:
                raise ValueError('openai_responses does not support stream=true')
            if effective_request.get('background') is True:
                raise ValueError('openai_responses does not support background=true')
            provider_call['endpoint'] = self.endpoint
            provider_call['_provider_sensitive_source'] = _json_value(request)
            if self.capture_raw_io:
                provider_call['request'] = _json_value(request)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            raise OpenAIResponsesRequestError(_error_status(error)) from error

        completion = await self._execute_response(request, provider_call)
        # Advance continuation state only after a successful response. Retries of
        # a rejected response must keep the same parent and full input history.
        if self.require_encrypted_reasoning:
            self._last_encrypted_input = _input_fingerprints(effective_request['input'])
            self._next_encrypted_prefix = self._last_encrypted_input + _input_fingerprints(
                completion.provider_specific_fields['response_output']
            )
        if self.stateful_previous_response:
            self._last_response_id = completion.id
            self._last_request_input_hash = stateful_input_hash
            self._last_request_parent_id = effective_request.get('previous_response_id')
        return completion

    async def _execute_response(
        self,
        request: dict[str, Any],
        provider_call: dict[str, Any],
        *,
        native: bool = False,
    ) -> Any:
        retry_budget = _resolve_retry_total_timeout()
        started_at = time.monotonic()

        @with_call_deadline()
        async def call_provider() -> tuple[Any, dict[str, Any], Any]:
            if native:
                # Keep the native transport's unknown fields and region headers.
                response = await self.client.post('/responses', cast_to=httpx.Response, body=request)
                return response, _serialize_response(response.json()), response.headers
            headers = None
            if self.encrypted_content_archive_dir is None:
                response = await self.client.responses.create(**request)
            else:
                raw_response = await self.client.responses.with_raw_response.create(**request)
                response = raw_response.parse()
                if inspect.isawaitable(response):
                    response = await response
                headers = getattr(getattr(raw_response, 'http_response', None), 'headers', None)
            return response, _serialize_response(response), headers

        for attempt in range(1, self.max_retries + 1):
            if retry_budget > 0 and time.monotonic() - started_at >= retry_budget:
                error = TimeoutError(
                    f'openai_responses: retry total budget {retry_budget}s exhausted after {attempt - 1} attempt(s)'
                )
                self._record_failure(provider_call, error)
                raise OpenAIResponsesRequestError() from error
            response_data: dict[str, Any] | None = None
            try:
                archive_path: Path | None = None
                archive: dict[str, Any] | None = None
                headers = None
                archive_request_sequence = self._archive_request_sequence
                if self.encrypted_content_archive_dir is not None:
                    self._archive_request_sequence += 1
                    archive_request_sequence = self._archive_request_sequence
                response, response_data, headers = await call_provider()
                if self.encrypted_content_archive_dir is not None:
                    archive = _encrypted_content_archive(
                        response_data,
                        headers=headers,
                        requested_model=str(provider_call['model']),
                        endpoint=self.endpoint,
                        request_sequence=archive_request_sequence,
                        gateway_cache_task_id=self.gateway_cache_task_id,
                    )
                    archive_path = _write_encrypted_content_archive(self.encrypted_content_archive_dir, archive)
                    if not archive['archive_complete']:
                        missing = ', '.join(archive['missing_fields'])
                        raise EncryptedContentArchiveError(
                            f'encrypted content was preserved at {archive_path}, but recovery metadata is missing: {missing}'
                        )
                if self.capture_raw_io:
                    provider_call['response'] = response_data
                if native:
                    status = response_data.get('status')
                    if response_data.get('error') is not None or status not in _SUPPORTED_FINAL_STATUSES:
                        raise _OpenAIResponsesStateError(status, response_data)
                    if (
                        response_data.get('object') != 'response'
                        or not response_data.get('id')
                        or not isinstance(response_data.get('output'), list)
                    ):
                        raise ValueError('OpenAI Responses returned an invalid response object')
                    result = response_data
                else:
                    result = _convert_response_data(response_data, response_data, str(provider_call['model']))
                    if self.require_encrypted_reasoning:
                        _require_complete_encrypted_output(response_data)
                    if archive_path is not None and archive is not None:
                        result.provider_specific_fields = {
                            **(result.provider_specific_fields or {}),
                            'encrypted_content_archive': {
                                'path': str(archive_path),
                                'region': archive['region'],
                                'encrypted_block_count': archive['encrypted_block_count'],
                                'archive_complete': archive['archive_complete'],
                            },
                        }
            except asyncio.CancelledError as error:
                attempt_record = _attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                self._record_failure(provider_call, error)
                raise
            except EncryptedContentArchiveError as error:
                provider_call['attempts'].append(_attempt_error(attempt, error))
                self._record_failure(provider_call, error)
                raise
            except Exception as error:
                attempt_record = _attempt_error(attempt, error)
                if response_data is not None and self.capture_raw_io:
                    attempt_record['response'] = response_data
                else:
                    captured_response = self._capture_error_response(provider_call, error)
                    if captured_response is not None:
                        attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                native_terminal = native and (_is_content_policy_error(error) or _is_context_length_error(error))
                if not native_terminal and _is_retryable(error) and attempt < self.max_retries:
                    try:
                        await asyncio.sleep(min(2 ** (attempt - 1), 30))
                    except asyncio.CancelledError as cancellation:
                        self._record_failure(provider_call, cancellation)
                        raise
                    continue
                self._record_failure(provider_call, error)
                if native and _is_content_policy_error(error):
                    raise ContentPolicyViolationError('OpenAI Responses content policy violation') from error
                if _is_context_length_error(error):
                    raise ContextLengthExceededError('OpenAI Responses context length exceeded') from error
                raise OpenAIResponsesRequestError(_error_status(error)) from error

            provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
            if not native:
                completion = cast(ExtendedChatCompletion, result)
                completion.provider_specific_fields = {
                    **(completion.provider_specific_fields or {}),
                    'provider_call': provider_call,
                }
            self._record_provider_call(provider_call)
            if native:
                self._trace_native_response(request, cast(dict, result), response.headers, provider_call)
            return result

        raise RuntimeError('OpenAI Responses retry loop exhausted without a response')

    def _trace_native_response(
        self, request: dict, response: dict, headers: httpx.Headers, provider_call: dict
    ) -> None:
        try:
            items = request.get('input', [])
            reachable = (
                {item.get('encrypted_content') for item in items if isinstance(item, dict)}
                if isinstance(items, list)
                else set()
            )
            regions = {key: value for key, value in self._regions.items() if key in reachable}
            region = {
                key: headers[header]
                for key, header in (('region', 'x-ms-region'), ('account_id', 'x-account-id'))
                if header in headers
            }
            for item in response['output']:
                if (
                    isinstance(item, dict)
                    and item.get('type') == 'reasoning'
                    and item.get('encrypted_content')
                    and region
                ):
                    regions[item['encrypted_content']] = region
            # Concurrent calls trace against their own snapshot; metadata is best-effort.
            self._regions = regions
            trace_response(request, response, regions, provider_call)
        except Exception:
            # New item kinds or trace processors must not invalidate a successful call.
            logger.warning('Native Responses trace projection failed (non-fatal)')


__all__ = [
    'EncryptedContentArchiveError',
    'OpenAIResponsesClient',
    'OpenAIResponsesRequestError',
    'convert_response',
]
