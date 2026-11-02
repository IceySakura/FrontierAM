"""Convert between Codex Responses API payloads and OpenAI chat completions."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import uuid

from dataclasses import dataclass
from typing import Any, Literal


logger = logging.getLogger(__name__)

_CHAT_TOOL_NAME_MAX_LENGTH = 64
_INVALID_CHAT_TOOL_NAME_CHARACTER = re.compile(r'[^a-zA-Z0-9_-]')


_EXEC_COMMAND_NAME = 'exec_command'
_EXEC_COMMAND_CHUNK_PREFIX = 'Chunk ID: '
_EXEC_COMMAND_WALL_TIME_PREFIX = 'Wall time: '


def _normalize_exec_command_output_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) < 2:
        return text
    if not lines[0].startswith(_EXEC_COMMAND_CHUNK_PREFIX) or not lines[1].startswith(_EXEC_COMMAND_WALL_TIME_PREFIX):
        return text

    def line_ending(line: str) -> str:
        if line.endswith('\r\n'):
            return '\r\n'
        if line.endswith('\n'):
            return '\n'
        return ''

    lines[0] = f'{_EXEC_COMMAND_CHUNK_PREFIX}deterministic{line_ending(lines[0])}'
    lines[1] = f'{_EXEC_COMMAND_WALL_TIME_PREFIX}deterministic seconds{line_ending(lines[1])}'
    return ''.join(lines)


def _normalize_exec_command_output_value(output: Any) -> Any:
    if isinstance(output, str):
        return _normalize_exec_command_output_text(output)
    if isinstance(output, list):
        normalized_parts = []
        for part in output:
            if isinstance(part, dict) and isinstance(part.get('text'), str):
                normalized_parts.append({**part, 'text': _normalize_exec_command_output_text(part['text'])})
            else:
                normalized_parts.append(part)
        return normalized_parts
    return output


def normalize_exec_command_output_metadata(request_body: dict[str, Any]) -> dict[str, Any]:
    """Normalize non-semantic Codex exec headers without mutating the wire request."""
    normalized = copy.deepcopy(request_body)
    raw_input = normalized.get('input')
    if not isinstance(raw_input, list):
        return normalized

    tool_names_by_call_id: dict[str, str] = {}
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type in {'function_call', 'custom_tool_call'}:
            call_id = item.get('call_id')
            name = item.get('name')
            if isinstance(call_id, str) and isinstance(name, str):
                tool_names_by_call_id[call_id] = name
            continue
        if item_type not in {'function_call_output', 'custom_tool_call_output'}:
            continue
        if tool_names_by_call_id.get(item.get('call_id')) != _EXEC_COMMAND_NAME:
            continue
        item['output'] = _normalize_exec_command_output_value(item.get('output'))
    return normalized


@dataclass(frozen=True)
class ToolIdentity:
    """Original Responses identity for one function exposed through Chat Completions."""

    tool_type: str
    name: str
    namespace: str | None = None
    execution: str | None = None

    def to_metadata(self) -> dict[str, str | None]:
        """Return a transport-safe identity snapshot for Gateway policy inspection."""
        return {
            'tool_type': self.tool_type,
            'name': self.name,
            'namespace': self.namespace,
            'execution': self.execution,
        }


class ResponsesProxyConversionError(ValueError):
    """Structured invalid-request error raised while converting Responses payloads."""

    def __init__(self, message: str, *, code: str, status_code: int = 422):
        super().__init__(f'[{code}] {message}')
        self.code = code
        self.status_code = status_code

    def response_payload(self) -> dict[str, Any]:
        return {
            'error': {
                'message': str(self),
                'type': 'invalid_request_error',
                'code': self.code,
            }
        }


def responses_to_openai(
    request_body: dict[str, Any],
    *,
    collect_custom_tools: set[str] | None = None,
    collect_tool_name_mappings: dict[str, ToolIdentity] | None = None,
    supports_tool_references: bool = False,
    tool_search_output_format: Literal['tool_reference', 'tool_search_output'] = 'tool_reference',
) -> dict[str, Any]:
    """Convert a Responses API request into OpenAI chat completion parameters.

    ``collect_custom_tools``: optional output set. When provided, the names of tools whose
    Responses type is ``custom`` (e.g. Codex's freeform ``apply_patch``) are added to it, so
    the caller can pass the same set to ``openai_to_responses_events`` and have the matching
    function_call restored as a ``custom_tool_call`` on the return trip. Defaults to None →
    behavior identical to before (no collection).

    ``supports_tool_references`` enables structured search results and deferred tool schemas.
    Defaults to False, using text search results and ordinary Chat tool schemas. Enable only
    when the endpoint implements function.defer_loading and the selected search result format.
    ``tool_search_output_format`` selects references for HY4 text templates or embedded Chat
    schemas for HY4 serving/VLM templates. The latter requires supports_tool_references=True.
    """
    if tool_search_output_format not in ('tool_reference', 'tool_search_output'):
        raise ResponsesProxyConversionError(
            f'unsupported tool_search_output_format={tool_search_output_format!r}; '
            'choose tool_reference or tool_search_output',
            code='tool_search_output_format_invalid',
        )
    if tool_search_output_format == 'tool_search_output' and not supports_tool_references:
        raise ResponsesProxyConversionError(
            'tool_search_output format requires supports_tool_references=True to keep schemas deferred',
            code='tool_search_output_requires_deferred_loading',
        )
    tool_name_mappings = collect_tool_name_mappings if collect_tool_name_mappings is not None else {}
    tools = _convert_tools(
        _request_tools(request_body),
        collect_custom_tools=collect_custom_tools,
        collect_tool_name_mappings=tool_name_mappings,
    )
    if not supports_tool_references:
        for tool in tools or []:
            tool['function'].pop('defer_loading', None)
    openai_params: dict[str, Any] = {
        'messages': _build_messages(
            request_body,
            tool_name_mappings=tool_name_mappings,
            supports_tool_references=supports_tool_references,
            tool_search_output_format=tool_search_output_format,
        ),
    }

    if request_body.get('model') is not None:
        openai_params['model'] = request_body['model']

    _convert_request_controls(request_body, openai_params)

    # Tool controls are invalid and meaningless when every input tool is absent or filtered out.
    if tools:
        openai_params['tools'] = tools
        if request_body.get('tool_choice') is not None:
            openai_params['tool_choice'] = _convert_tool_choice(
                request_body['tool_choice'],
                tool_name_mappings=tool_name_mappings,
            )
        if request_body.get('parallel_tool_calls') is not None:
            openai_params['parallel_tool_calls'] = request_body['parallel_tool_calls']
    elif request_body.get('tool_choice') not in (None, 'none') or request_body.get('parallel_tool_calls') is not None:
        logger.warning('responses_proxy: dropping tool controls because no supported tools remain after conversion')

    return openai_params


def _convert_request_controls(request_body: dict[str, Any], openai_params: dict[str, Any]) -> None:
    """Map Responses generation controls that have direct Chat Completions equivalents."""
    reasoning = request_body.get('reasoning')
    if isinstance(reasoning, dict) and isinstance(reasoning.get('effort'), str):
        openai_params['reasoning_effort'] = reasoning['effort']

    text = request_body.get('text')
    if not isinstance(text, dict):
        return
    if isinstance(text.get('verbosity'), str):
        openai_params['verbosity'] = text['verbosity']

    text_format = text.get('format')
    if not isinstance(text_format, dict):
        return
    if text_format.get('type') != 'json_schema':
        logger.warning('responses_proxy: dropping unsupported text format type=%s', text_format.get('type'))
        return

    schema = text_format.get('schema')
    json_schema: dict[str, Any] = {
        'name': text_format.get('name') or 'codex_output_schema',
        'schema': {} if schema is None else schema,
    }
    if isinstance(text_format.get('strict'), bool):
        json_schema['strict'] = text_format['strict']
    openai_params['response_format'] = {
        'type': 'json_schema',
        'json_schema': json_schema,
    }


def _convert_tool_choice(value: Any, *, tool_name_mappings: dict[str, ToolIdentity]) -> Any:
    """Translate Responses tool-choice objects into Chat Completions shape."""
    if not isinstance(value, dict):
        return value
    choice_type = value.get('type')
    logger.debug('responses_proxy: normalizing tool_choice type=%r fields=%s', choice_type, sorted(value))
    if choice_type == 'auto':
        return 'auto'
    if choice_type in {'none', 'required'}:
        return choice_type
    if choice_type in {'function', 'custom'} and isinstance(value.get('name'), str):
        namespace = value.get('namespace')
        selected_name = next(
            (
                chat_name
                for chat_name, identity in tool_name_mappings.items()
                if identity.tool_type == choice_type
                and identity.name == value['name']
                and identity.namespace == namespace
            ),
            None,
        )
        if selected_name is None:
            raise ResponsesProxyConversionError(
                f'tool_choice selects an unavailable {choice_type} tool {value["name"]!r}',
                code='tool_choice_tool_not_found',
            )
        return {
            'type': 'function',
            'function': {'name': selected_name},
        }
    return value


def openai_to_responses_events(
    openai_response: dict[str, Any],
    *,
    custom_tool_names: set[str] | None = None,
    tool_name_mappings: dict[str, ToolIdentity] | None = None,
    wrap_plaintext_reasoning_content: bool = True,
) -> list[dict[str, Any]]:
    """Convert an OpenAI chat completion response into Responses API SSE events.

    ``custom_tool_names``: names of tools that were originally Responses ``custom`` tools (e.g.
    Codex freeform ``apply_patch``), collected by ``responses_to_openai``. A tool_call whose
    name is in this set is emitted as a ``custom_tool_call`` (the form Codex >=0.136 requires;
    its apply_patch handler only accepts ToolPayload::Custom) instead of a ``function_call``.
    Defaults to None → every tool_call becomes a function_call, identical to before.

    ``wrap_plaintext_reasoning_content``: origin/main default True. True keeps main's wrap of
    ``reasoning_content`` and mints missing reasoning-item ids. False follows the GPT/Codex
    path: do not wrap plaintext, do not mint ``rs_*``, and drop encrypted blobs that have no id.
    Native ``reasoning_items`` with a provider id are always forwarded.
    """
    response_id = openai_response.get('id') or f'resp_{uuid.uuid4().hex}'
    events: list[dict[str, Any]] = [
        {
            'type': 'response.created',
            'response': {
                'id': response_id,
            },
        }
    ]

    choices = openai_response.get('choices') or []
    for choice in choices:
        message = choice.get('message') or {}

        reasoning_items = message.get('reasoning_items') or []
        emitted_reasoning = False
        if isinstance(reasoning_items, list):
            for reasoning_item in reasoning_items:
                item = _normalize_reasoning_item(
                    reasoning_item,
                    wrap_plaintext_reasoning_content=wrap_plaintext_reasoning_content,
                )
                if item is None:
                    continue
                events.append({
                    'type': 'response.output_item.done',
                    'item': item,
                })
                emitted_reasoning = True

        reasoning_content = message.get('reasoning_content')
        if wrap_plaintext_reasoning_content and reasoning_content and not emitted_reasoning:
            events.append({
                'type': 'response.output_item.done',
                'item': _reasoning_item_from_text(str(reasoning_content)),
            })

        text = _assistant_content_to_text(message.get('content'))
        if text:
            events.append({
                'type': 'response.output_item.done',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'id': f'msg_{uuid.uuid4().hex}',
                    'content': [
                        {
                            'type': 'output_text',
                            'text': text,
                        }
                    ],
                },
            })

        tool_calls = message.get('tool_calls') or []
        for tool_call in tool_calls:
            function = tool_call.get('function') or {}
            chat_name = function.get('name', '')
            tool_identity = (tool_name_mappings or {}).get(chat_name)
            name = tool_identity.name if tool_identity is not None else chat_name
            call_id = tool_call.get('id') or f'call_{uuid.uuid4().hex}'
            if tool_identity is not None:
                is_custom_tool = tool_identity.tool_type == 'custom'
            else:
                is_custom_tool = bool(custom_tool_names and chat_name in custom_tool_names)
            if tool_identity is not None and tool_identity.tool_type == 'tool_search':
                raw_arguments = function.get('arguments')
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except (TypeError, ValueError):
                        arguments = raw_arguments
                else:
                    arguments = raw_arguments if raw_arguments is not None else {}
                events.append({
                    'type': 'response.output_item.done',
                    'item': {
                        'type': 'tool_search_call',
                        'call_id': call_id,
                        'execution': tool_identity.execution or 'client',
                        'arguments': arguments,
                    },
                })
            elif is_custom_tool:
                # Original tool was a Responses custom tool (e.g. apply_patch). Restore it as a
                # custom_tool_call; Codex >=0.136 routes only custom_tool_call to the apply_patch
                # handler (function_call would raise "apply_patch handler received unsupported
                # payload"). _extract_custom_tool_input recovers the raw input from the function
                # arguments we downgraded on the request side.
                item = {
                    'type': 'custom_tool_call',
                    'call_id': call_id,
                    'name': name,
                    'input': _extract_custom_tool_input(function.get('arguments')),
                }
                if tool_identity is not None and tool_identity.namespace is not None:
                    item['namespace'] = tool_identity.namespace
                events.append({
                    'type': 'response.output_item.done',
                    'item': item,
                })
            else:
                if tool_identity is None and custom_tool_names and len(custom_tool_names) == 1 and len(tool_calls) == 1:
                    # The request advertised exactly one custom tool but this call's name did not
                    # match (host may have namespaced/renamed it). Emitting function_call here
                    # silently reintroduces the 0.136 "unsupported payload" crash, so warn.
                    only_custom = next(iter(custom_tool_names))
                    logger.warning(
                        'responses_proxy: tool_call name %r did not match the sole custom tool %r; '
                        'emitting function_call may break Codex >=0.136 apply_patch',
                        chat_name,
                        only_custom,
                    )
                item = {
                    'type': 'function_call',
                    'call_id': call_id,
                    'name': name,
                    'arguments': _ensure_json_string(function.get('arguments')),
                }
                if tool_identity is not None and tool_identity.namespace is not None:
                    item['namespace'] = tool_identity.namespace
                events.append({
                    'type': 'response.output_item.done',
                    'item': item,
                })

    usage = _convert_usage(openai_response.get('usage'))
    events.append({
        'type': 'response.completed',
        'response': {
            'id': response_id,
            'usage': usage,
        },
    })

    return events


def _build_messages(
    request_body: dict[str, Any],
    *,
    tool_name_mappings: dict[str, ToolIdentity] | None = None,
    supports_tool_references: bool = False,
    tool_search_output_format: Literal['tool_reference', 'tool_search_output'] = 'tool_reference',
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    instructions = request_body.get('instructions')
    if isinstance(instructions, str) and instructions.strip():
        messages.append({'role': 'system', 'content': instructions})

    raw_input = request_body.get('input')
    if isinstance(raw_input, str):
        messages.append({'role': 'user', 'content': raw_input})
        return messages

    if not isinstance(raw_input, list):
        return messages

    pending_assistant_message: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] = []
    pending_reasoning_items: list[dict[str, Any]] = []
    outstanding_tool_call_ids: set[str] = set()
    deferred_interstitial_messages: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant_message, pending_tool_calls, pending_reasoning_items
        if pending_assistant_message is not None:
            message = dict(pending_assistant_message)
            if pending_reasoning_items:
                message['reasoning_items'] = pending_reasoning_items
                _attach_reasoning_content(message, pending_reasoning_items)
            if pending_tool_calls:
                message['tool_calls'] = pending_tool_calls
                outstanding_tool_call_ids.update(tool_call['id'] for tool_call in pending_tool_calls)
            messages.append(message)
            pending_assistant_message = None
            pending_tool_calls = []
            pending_reasoning_items = []
            return
        if pending_tool_calls or pending_reasoning_items:
            message = {
                'role': 'assistant',
                'content': '',
            }
            if pending_reasoning_items:
                message['reasoning_items'] = pending_reasoning_items
                _attach_reasoning_content(message, pending_reasoning_items)
            if pending_tool_calls:
                message['tool_calls'] = pending_tool_calls
                outstanding_tool_call_ids.update(tool_call['id'] for tool_call in pending_tool_calls)
            messages.append(message)
            pending_tool_calls = []
            pending_reasoning_items = []

    def append_tool_output(message: dict[str, Any]) -> None:
        messages.append(message)
        outstanding_tool_call_ids.discard(message['tool_call_id'])
        if not outstanding_tool_call_ids:
            messages.extend(deferred_interstitial_messages)
            deferred_interstitial_messages.clear()

    for item in raw_input:
        if not isinstance(item, dict):
            continue

        item_type = item.get('type')
        if item_type == 'compaction':
            # Codex remote compaction replaces most of the conversation with this opaque
            # Responses item. It must survive the chat-shaped runtime bridge unchanged so the
            # host-side Responses client can replay it to the provider on the next request.
            flush_pending_assistant()
            messages.append(dict(item))
            continue
        if item_type == 'compaction_trigger':
            raise ResponsesProxyConversionError(
                'Codex remote compaction v2 is unsupported; configure Codex to use local compaction.',
                code='remote_compaction_v2_unsupported',
            )
        if item_type == 'reasoning':
            reasoning_item = _normalize_reasoning_item(item)
            if reasoning_item is not None:
                # Reasoning after a message/tool call starts a new assistant group.
                # Flush before collecting it so it cannot attach to the previous output.
                if pending_assistant_message is not None or pending_tool_calls:
                    flush_pending_assistant()
                pending_reasoning_items.append(reasoning_item)
            continue

        if item_type in {'function_call', 'custom_tool_call', 'tool_search_call'}:
            if item_type == 'tool_search_call':
                _validate_tool_search_item(item)
            pending_tool_calls.append(_convert_tool_call_item(item, tool_name_mappings=tool_name_mappings))
            continue

        if item_type == 'agent_message':
            # Multi-Agent V2 encrypts inter-agent payloads in this native item. The provider must
            # receive the original structure so it can decrypt the payload instead of seeing ciphertext.
            flush_pending_assistant()
            messages.append(dict(item))
            continue

        if item_type == 'message':
            message = _convert_message_item(item)
            if message is not None:
                if message.get('role') == 'assistant':
                    if pending_assistant_message is not None or pending_tool_calls:
                        flush_pending_assistant()
                    pending_assistant_message = message
                else:
                    flush_pending_assistant()
                    if outstanding_tool_call_ids:
                        deferred_interstitial_messages.append(message)
                    else:
                        messages.append(message)
        elif item_type in {'function_call_output', 'custom_tool_call_output'}:
            flush_pending_assistant()
            append_tool_output(_convert_tool_output_item(item))
        elif item_type == 'additional_tools':
            # Responses Lite carries the current tool declarations as an input item. They are
            # converted separately by _request_tools and are not conversation content.
            continue
        elif item_type == 'tool_search_output':
            _validate_tool_search_item(item, require_tools=True)
            flush_pending_assistant()
            append_tool_output(
                _convert_tool_search_output_item(
                    item,
                    tool_name_mappings=tool_name_mappings,
                    supports_tool_references=supports_tool_references,
                    tool_search_output_format=tool_search_output_format,
                )
            )
        elif item_type is not None:
            logger.warning('responses_proxy: dropping unknown input item type=%s', item_type)

    flush_pending_assistant()
    messages.extend(deferred_interstitial_messages)
    return messages


def _convert_message_item(item: dict[str, Any]) -> dict[str, Any] | None:
    role = item.get('role', 'user')
    if role == 'developer':
        role = 'system'

    content = _convert_content(item.get('content'))
    if content is None:
        return None

    return {
        'role': role,
        'content': content,
    }


def _convert_content(content: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ''

    has_images = any(isinstance(part, dict) and part.get('type') == 'input_image' for part in content)
    if has_images:
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get('type')
            if part_type in {'input_text', 'output_text', 'text'}:
                parts.append({'type': 'text', 'text': part.get('text', '')})
            elif part_type == 'input_image':
                url = part.get('image_url')
                if not isinstance(url, str) or not url:
                    # An empty url reaches the backend as a fetch failure that kills the whole
                    # request, so keep the part visible as text instead of sending a broken image.
                    logger.warning('responses_proxy: input_image part carries no image_url')
                    parts.append({'type': 'text', 'text': json.dumps(part, ensure_ascii=False)})
                    continue
                image_url = {'url': url}
                if part.get('detail') is not None:
                    image_url['detail'] = part['detail']
                parts.append({'type': 'image_url', 'image_url': image_url})
            else:
                logger.warning('responses_proxy: dropping unsupported message content type=%s', part_type)
        return parts

    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get('type') in {'input_text', 'output_text', 'text'}:
            text_parts.append(part.get('text', ''))
        else:
            logger.warning('responses_proxy: dropping unsupported message content type=%s', part.get('type'))

    return '\n\n'.join(text_parts)


def _convert_tool_call_item(
    item: dict[str, Any],
    *,
    tool_name_mappings: dict[str, ToolIdentity] | None = None,
) -> dict[str, Any]:
    item_type = item.get('type')
    if item_type == 'custom_tool_call':
        arguments = json.dumps({'input': item.get('input', '')}, ensure_ascii=False)
    else:
        arguments = _ensure_json_string(item.get('arguments'))
    default_name = 'tool_search' if item_type == 'tool_search_call' else item.get('name', '')
    namespace = item.get('namespace')
    if isinstance(namespace, str) and namespace:
        # Codex sends local compaction requests without tools, so history calls cannot be
        # resolved through this request's mappings. Keep the name the model saw in earlier turns.
        default_name = _bounded_chat_tool_name(_normalized_chat_tool_name(namespace, default_name), digest=None)
    chat_name = _chat_tool_name_for_item(item, tool_name_mappings) or default_name

    return {
        'id': item.get('call_id') or f'call_{uuid.uuid4().hex}',
        'type': 'function',
        'function': {
            'name': chat_name,
            'arguments': arguments,
        },
    }


def _convert_tool_output_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'role': 'tool',
        'tool_call_id': item.get('call_id', ''),
        'content': _convert_tool_output(item.get('output')),
    }


def _convert_tool_output(output: Any) -> Any:
    """Preserve Responses content arrays so images remain multimodal downstream."""
    if isinstance(output, list):
        converted = _convert_content(output)
        if isinstance(converted, list):
            return converted
    return _tool_output_to_text(output)


def _convert_tool_search_output_item(
    item: dict[str, Any],
    *,
    tool_name_mappings: dict[str, ToolIdentity] | None = None,
    supports_tool_references: bool = False,
    tool_search_output_format: Literal['tool_reference', 'tool_search_output'] = 'tool_reference',
) -> dict[str, Any]:
    # NOTE(luxyhliu): Text templates resolve registry references; serving/VLM templates require embedded schemas.
    # NOTE(luxyhliu): Both render definitions in tool results while defer_loading keeps them out of the system prompt.
    tools = _convert_tools(item['tools'], collect_tool_name_mappings=tool_name_mappings)
    if not supports_tool_references:
        for tool in tools or []:
            tool['function'].pop('defer_loading', None)
    if supports_tool_references and tool_search_output_format == 'tool_search_output':
        content = [{'type': 'tool_search_output', 'tools': tools or []}]
    elif tools and supports_tool_references:
        content = [{'type': 'tool_reference', 'name': tool['function']['name']} for tool in tools]
    else:
        content = json.dumps(
            {
                'status': item.get('status'),
                'execution': item['execution'],
                'tools': tools or [],
            },
            ensure_ascii=False,
        )
    return {
        'role': 'tool',
        'tool_call_id': item['call_id'],
        'content': content,
    }


def _tool_output_to_text(output: Any) -> str:
    if output is None:
        return ''
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text_parts = []
        for part in output:
            if isinstance(part, dict):
                if 'text' in part:
                    text_parts.append(str(part.get('text', '')))
                else:
                    text_parts.append(json.dumps(part, ensure_ascii=False))
            else:
                text_parts.append(str(part))
        return '\n'.join(text_parts)
    return json.dumps(output, ensure_ascii=False)


def _request_tools(request_body: dict[str, Any]) -> list[Any]:
    """Collect declared, Lite, and client-loaded Responses tools in wire order."""
    tools: list[Any] = []
    top_level_tools = request_body.get('tools')
    if isinstance(top_level_tools, list):
        tools.extend(top_level_tools)

    raw_input = request_body.get('input')
    if not isinstance(raw_input, list):
        return tools
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type == 'additional_tools':
            additional_tools = item.get('tools')
            if not isinstance(additional_tools, list):
                logger.warning('responses_proxy: additional_tools item has non-list tools payload')
                continue
            tools.extend(additional_tools)
        elif item_type == 'tool_search_output':
            _validate_tool_search_item(item, require_tools=True)
            tools.extend(item['tools'])
    return tools


def _validate_tool_search_execution(execution: Any) -> None:
    if execution != 'client':
        raise ResponsesProxyConversionError(
            f'only client-executed tool_search is supported, got execution={execution!r}',
            code='tool_search_execution_unsupported',
        )


def _validate_tool_search_item(item: dict[str, Any], *, require_tools: bool = False) -> str:
    _validate_tool_search_execution(item.get('execution'))
    call_id = item.get('call_id')
    if not isinstance(call_id, str) or not call_id:
        raise ResponsesProxyConversionError(
            'client tool_search call/output requires a non-empty call_id',
            code='tool_search_call_id_missing',
        )
    if require_tools and not isinstance(item.get('tools'), list):
        raise ResponsesProxyConversionError(
            'client tool_search_output requires tools to be a list',
            code='tool_search_tools_invalid',
        )
    return call_id


def _convert_tools(
    tools: Any,
    *,
    collect_custom_tools: set[str] | None = None,
    collect_tool_name_mappings: dict[str, ToolIdentity] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None

    tool_name_mappings = collect_tool_name_mappings if collect_tool_name_mappings is not None else {}
    converted_tools: list[dict[str, Any]] = []
    converted_tool_names: set[str] = set()

    def append_converted_tool(converted: dict[str, Any] | None) -> None:
        if converted is None:
            return
        chat_name = converted['function']['name']
        if chat_name in converted_tool_names:
            logger.debug('responses_proxy: coalescing duplicate tool declaration name=%s', chat_name)
            return
        converted_tool_names.add(chat_name)
        converted_tools.append(converted)

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get('type')
        if tool_type in {'function', 'custom'}:
            converted = _convert_callable_tool(
                tool,
                namespace=None,
                collect_custom_tools=collect_custom_tools,
                collect_tool_name_mappings=tool_name_mappings,
            )
            append_converted_tool(converted)
        elif tool_type == 'namespace':
            namespace = tool.get('name')
            nested_tools = tool.get('tools')
            if not isinstance(namespace, str) or not namespace:
                logger.warning('responses_proxy: dropping namespace tool without a name')
                continue
            if not isinstance(nested_tools, list):
                logger.warning('responses_proxy: dropping namespace=%s with non-list tools payload', namespace)
                continue
            for nested_tool in nested_tools:
                if not isinstance(nested_tool, dict):
                    continue
                nested_type = nested_tool.get('type')
                if nested_type not in {'function', 'custom'}:
                    logger.warning(
                        'responses_proxy: dropping unsupported nested tool type=%s in namespace=%s',
                        nested_type,
                        namespace,
                    )
                    continue
                converted = _convert_callable_tool(
                    nested_tool,
                    namespace=namespace,
                    collect_custom_tools=collect_custom_tools,
                    collect_tool_name_mappings=tool_name_mappings,
                )
                append_converted_tool(converted)
        elif tool_type == 'tool_search':
            execution = tool.get('execution')
            _validate_tool_search_execution(execution)
            identity = ToolIdentity(
                tool_type='tool_search',
                name='tool_search',
                execution='client',
            )
            chat_name = _allocate_chat_tool_name(identity, tool_name_mappings)
            append_converted_tool({
                'type': 'function',
                'function': {
                    'name': chat_name,
                    'description': tool.get('description', ''),
                    'parameters': tool.get('parameters')
                    or {
                        'type': 'object',
                        'properties': {},
                        'additionalProperties': False,
                    },
                },
            })
        else:
            logger.warning('responses_proxy: dropping unknown tool type=%s', tool_type)

    return converted_tools or None


def _convert_callable_tool(
    tool: dict[str, Any],
    *,
    namespace: str | None,
    collect_custom_tools: set[str] | None,
    collect_tool_name_mappings: dict[str, ToolIdentity],
) -> dict[str, Any] | None:
    tool_type = str(tool.get('type') or '')
    name = tool.get('name')
    if not isinstance(name, str) or not name:
        logger.warning('responses_proxy: dropping %s tool without a name', tool_type or 'unknown')
        return None

    identity = ToolIdentity(tool_type=tool_type, name=name, namespace=namespace)
    chat_name = _allocate_chat_tool_name(identity, collect_tool_name_mappings)
    if tool_type == 'custom':
        # Chat Completions has no freeform/custom tool. Downgrade it to one string argument and
        # retain its identity so the response can be restored as custom_tool_call.
        if collect_custom_tools is not None:
            collect_custom_tools.add(name)
        function = {
            'name': chat_name,
            'description': tool.get('description', ''),
            'parameters': {
                'type': 'object',
                'properties': {
                    'input': {
                        'type': 'string',
                        'description': 'Raw tool input.',
                    }
                },
                'required': ['input'],
                'additionalProperties': False,
            },
        }
    else:
        function = {
            'name': chat_name,
            'description': tool.get('description', ''),
            'parameters': tool.get('parameters')
            or {
                'type': 'object',
                'properties': {},
                'additionalProperties': False,
            },
        }
        if isinstance(tool.get('strict'), bool):
            function['strict'] = tool['strict']
    if isinstance(tool.get('defer_loading'), bool):
        function['defer_loading'] = tool['defer_loading']
    return {'type': 'function', 'function': function}


def _allocate_chat_tool_name(
    identity: ToolIdentity,
    tool_name_mappings: dict[str, ToolIdentity],
) -> str:
    normalized_name = _normalized_chat_tool_name(identity.namespace, identity.name)
    identity_key = f'{identity.tool_type}\0{identity.namespace or ""}\0{identity.name}'
    digest = hashlib.sha256(identity_key.encode()).hexdigest()[:8]
    candidate = _bounded_chat_tool_name(normalized_name, digest=None)
    existing_identity = tool_name_mappings.get(candidate)
    if existing_identity is not None and existing_identity != identity:
        candidate = _bounded_chat_tool_name(normalized_name, digest=digest)
    tool_name_mappings[candidate] = identity
    return candidate


def _normalized_chat_tool_name(namespace: str | None, name: str) -> str:
    raw_name = name if namespace is None else f'{namespace}__{name}'
    return _INVALID_CHAT_TOOL_NAME_CHARACTER.sub('_', raw_name).strip('_') or 'tool'


def _bounded_chat_tool_name(name: str, *, digest: str | None) -> str:
    suffix = f'__{digest}' if digest is not None else ''
    prefix_length = _CHAT_TOOL_NAME_MAX_LENGTH - len(suffix)
    return f'{name[:prefix_length]}{suffix}'


def _chat_tool_name_for_item(
    item: dict[str, Any],
    tool_name_mappings: dict[str, ToolIdentity] | None,
) -> str | None:
    if not tool_name_mappings:
        return None
    item_type = item.get('type')
    if item_type == 'custom_tool_call':
        expected_tool_type = 'custom'
    elif item_type == 'tool_search_call':
        expected_tool_type = 'tool_search'
    else:
        expected_tool_type = 'function'
    name = 'tool_search' if item_type == 'tool_search_call' else item.get('name')
    namespace = item.get('namespace')
    for chat_name, identity in tool_name_mappings.items():
        if identity.tool_type == expected_tool_type and identity.name == name and identity.namespace == namespace:
            return chat_name
    return None


def _summary_from_text(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    return [{'type': 'summary_text', 'text': text}]


def _attach_reasoning_content(message: dict[str, Any], reasoning_items: list[dict[str, Any]]) -> None:
    """Mirror replayed reasoning_items into the assistant message's reasoning_content.

    HY-style chat models continue their chain-of-thought across turns only when the prior turn's
    reasoning is replayed in the ``reasoning_content`` field (a plain string) alongside ``content``;
    they never read the Responses ``reasoning_items`` structure. We keep ``reasoning_items`` intact
    (GPT-5.5 / Responses models continue via its encrypted_content + item ids) and ADD a flattened
    ``reasoning_content`` next to it -- each item's ``summary[].text`` (falling back to
    ``content[].text``). litellm's Responses path simply drops this extra field, so GPT-5.5 is
    unaffected (verified on the wire: reasoning_items count and encrypted_content are unchanged),
    while chat models pick it up. Only set when there is textual reasoning and the message does not
    already carry reasoning_content (opaque encrypted_content-only items yield nothing)."""
    if message.get('reasoning_content'):
        return
    parts: list[str] = []
    for item in reasoning_items:
        if not isinstance(item, dict):
            continue
        chunks: list[str] = []
        for key in ('summary', 'content'):
            section = item.get(key)
            if not isinstance(section, list):
                continue
            for entry in section:
                if isinstance(entry, dict):
                    text = entry.get('text')
                    if isinstance(text, str) and text:
                        chunks.append(text)
            if chunks:
                # Prefer summary; only fall back to content when summary had no text.
                break
        if chunks:
            parts.append('\n'.join(chunks))
    if parts:
        message['reasoning_content'] = '\n'.join(parts)


def _reasoning_item_from_text(text: str) -> dict[str, Any]:
    return {
        'type': 'reasoning',
        'id': f'rs_{uuid.uuid4().hex}',
        'summary': _summary_from_text(text),
    }


def _normalize_reasoning_item(
    item: Any,
    *,
    wrap_plaintext_reasoning_content: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    encrypted_content = item.get('encrypted_content')
    item_id = item.get('id')
    if wrap_plaintext_reasoning_content:
        normalized: dict[str, Any] = {
            'type': 'reasoning',
            'id': item_id or f'rs_{uuid.uuid4().hex}',
        }
    else:
        # GPT/Codex: never invent an rs_* that Codex would replay to a stateless upstream.
        if encrypted_content and not item_id:
            logger.warning('responses_proxy: skipping encrypted reasoning item without id')
            return None
        normalized = {'type': 'reasoning'}
        if item_id:
            normalized['id'] = str(item_id)

    summary = item.get('summary')
    if isinstance(summary, list):
        normalized['summary'] = summary
    elif isinstance(summary, str):
        normalized['summary'] = _summary_from_text(summary)
    else:
        normalized['summary'] = []

    content = item.get('content')
    if isinstance(content, list):
        normalized['content'] = content

    if encrypted_content:
        normalized['encrypted_content'] = encrypted_content

    status = item.get('status')
    if status:
        normalized['status'] = status

    return normalized


def _assistant_content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') in {'text', 'output_text'}:
                    text_parts.append(part.get('text', ''))
                elif 'text' in part:
                    text_parts.append(str(part.get('text', '')))
        return '\n'.join(text_parts)
    return str(content)


def _convert_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
    else:
        prompt_tokens = int(usage.get('prompt_tokens') or 0)
        completion_tokens = int(usage.get('completion_tokens') or 0)
        total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))

    return {
        'input_tokens': prompt_tokens,
        'input_tokens_details': None,
        'output_tokens': completion_tokens,
        'output_tokens_details': None,
        'total_tokens': total_tokens,
    }


def _ensure_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _extract_custom_tool_input(arguments: Any) -> str:
    """Recover the raw input string for a restored custom_tool_call.

    On the request side a custom tool is downgraded to a function with a single ``input`` string
    param (see _convert_tools). The model usually replies with arguments ``{"input": "..."}``, but
    it may not follow that shape, so handle each case without ever turning a non-patch value into
    a string that codex would try to parse as a patch.
    """
    if arguments is None:
        return ''  # do NOT json.dumps(None) -> "null"; codex would treat it as a patch
    if not isinstance(arguments, str):
        if isinstance(arguments, dict) and isinstance(arguments.get('input'), str):
            return arguments['input']
        return json.dumps(arguments, ensure_ascii=False)
    # arguments is a str: either {"input": "..."} JSON, or a bare patch body.
    try:
        obj = json.loads(arguments)
    except (ValueError, TypeError):
        return arguments  # bare patch (not JSON) -> pass through verbatim
    if isinstance(obj, dict):
        if isinstance(obj.get('input'), str):
            return obj['input']  # normal: {"input": "*** Begin Patch..."}
        # dict but no "input" key (model used a different schema, e.g. {"patch": ...}).
        # Do not feed the whole JSON as a patch; prefer the sole string value, else warn.
        str_vals = [v for v in obj.values() if isinstance(v, str)]
        if len(str_vals) == 1:
            logger.warning('responses_proxy: custom tool arguments missing "input", using sole string field')
            return str_vals[0]
        logger.warning('responses_proxy: custom tool arguments has no usable input field: %s', list(obj))
        return arguments  # last resort (may still fail downstream, but logged)
    return arguments  # JSON but not a dict (rare) -> verbatim


__all__ = [
    'ResponsesProxyConversionError',
    'ToolIdentity',
    'openai_to_responses_events',
    'responses_to_openai',
]
