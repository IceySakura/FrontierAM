"""OpenCode-compatible Responses conversion used only by EnterpriseBench."""

from __future__ import annotations

import copy
import json
import logging
import uuid

from typing import Any


logger = logging.getLogger(__name__)


def responses_to_openai(
    request_body: dict[str, Any],
    *,
    collect_custom_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Convert supported inline Judge requests; reject lossy fallbacks.

    collect_custom_tools is accepted for the shared proxy interface only.
    This Judge accepts function tools, never downgraded native/custom tools.
    """
    supported = {
        'model',
        'input',
        'instructions',
        'tools',
        'tool_choice',
        'parallel_tool_calls',
        'reasoning',
        'include',
        'max_output_tokens',
        'temperature',
        'top_p',
        'text',
        'truncation',
        'store',
        'metadata',
        'service_tier',
        'prompt_cache_key',
        'prompt_cache_retention',
        'safety_identifier',
        'user',
        'max_tool_calls',
        'stream',
        'background',
        'previous_response_id',
        'conversation',
    }
    unknown = set(request_body) - supported
    if unknown:
        raise ValueError(f'Unsupported Judge request fields: {sorted(unknown)}')
    for key in ('previous_response_id', 'conversation'):
        if request_body.get(key) is not None:
            raise ValueError(f'Judge requires complete inline history; {key} is unsupported')
    if request_body.get('background'):
        raise ValueError('Judge background requests are unsupported')
    if request_body.get('stream') is False:
        raise ValueError('OpenCode Judge proxy requires streaming responses')
    if request_body.get('truncation') not in (None, 'disabled'):
        raise ValueError('Judge automatic truncation is unsupported')
    openai_params: dict[str, Any] = {'messages': _build_messages(request_body)}
    local = {'input', 'instructions', 'tools', 'stream', 'background', 'previous_response_id', 'conversation'}
    for key, value in request_body.items():
        if key not in local and value is not None:
            openai_params[key] = copy.deepcopy(value)
    # The public host handler strips these fields for generic clients. Carry them
    # only to the task-owned Responses wrapper, which applies YAML precedence.
    controls = {key: openai_params.pop(key) for key in ('store', 'metadata', 'service_tier') if key in openai_params}
    if controls:
        openai_params['_judge_response_controls'] = controls
    if request_body.get('tools') is not None:
        openai_params['tools'] = _convert_tools(request_body['tools'], collect_custom_tools=collect_custom_tools)
    return openai_params


def openai_to_responses_events(
    openai_response: dict[str, Any],
    *,
    custom_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert an OpenAI chat completion response into Responses API SSE events.

    ``custom_tool_names``: names of tools that were originally Responses ``custom`` tools (e.g.
    Codex freeform ``apply_patch``), collected by ``responses_to_openai``. A tool_call whose
    name is in this set is emitted as a ``custom_tool_call`` (the form Codex >=0.136 requires;
    its apply_patch handler only accepts ToolPayload::Custom) instead of a ``function_call``.
    Defaults to None → every tool_call becomes a function_call, identical to before.
    """
    response_id = openai_response.get('id') or f'resp_{uuid.uuid4().hex}'
    response_model = openai_response.get('model')
    created_at = int(openai_response.get('created') or 0)
    output_items: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        {
            'type': 'response.created',
            'response': {
                'id': response_id,
                'object': 'response',
                'created_at': created_at,
                'model': response_model,
                'output': [],
                'status': 'in_progress',
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
                item = _normalize_reasoning_item(reasoning_item)
                if item is None:
                    continue
                # Some Responses providers (notably DSV4 Flash) return the
                # complete visible reasoning under ``content[].reasoning_text``
                # while leaving ``summary`` empty. OpenCode renders and records
                # only the summary text in its JSONL reasoning event, so mirror
                # the visible content there for trajectory persistence. Keep
                # the original content/encrypted fields intact for replay.
                if not item.get('summary'):
                    reasoning_text = _reasoning_content_text(item)
                    if reasoning_text:
                        item['summary'] = _summary_from_text(reasoning_text)
                item.setdefault('status', 'completed')
                output_index = len(output_items)
                events.append({
                    'type': 'response.output_item.added',
                    'output_index': output_index,
                    'item': {**item, 'summary': [], 'status': 'in_progress'},
                })
                events.extend(_reasoning_summary_events(item, output_index=output_index))
                events.append({
                    'type': 'response.output_item.done',
                    'output_index': output_index,
                    'item': item,
                })
                output_items.append(item)
                emitted_reasoning = True

        reasoning_content = message.get('reasoning_content')
        if reasoning_content and not emitted_reasoning:
            item = _reasoning_item_from_text(str(reasoning_content))
            item['status'] = 'completed'
            output_index = len(output_items)
            events.extend([
                {
                    'type': 'response.output_item.added',
                    'output_index': output_index,
                    'item': {**item, 'status': 'in_progress'},
                },
                {
                    'type': 'response.output_item.done',
                    'output_index': output_index,
                    'item': item,
                },
            ])
            output_items.append(item)

        text = _assistant_content_to_text(message.get('content'))
        if text:
            output_index = len(output_items)
            message_id = f'msg_{uuid.uuid4().hex}'
            content_part = {'type': 'output_text', 'text': text, 'annotations': []}
            item = {
                'type': 'message',
                'role': 'assistant',
                'id': message_id,
                'status': 'completed',
                'content': [content_part],
            }
            events.extend([
                {
                    'type': 'response.output_item.added',
                    'output_index': output_index,
                    'item': {
                        'type': 'message',
                        'role': 'assistant',
                        'id': message_id,
                        'status': 'in_progress',
                        'content': [],
                    },
                },
                {
                    'type': 'response.content_part.added',
                    'item_id': message_id,
                    'output_index': output_index,
                    'content_index': 0,
                    'part': {'type': 'output_text', 'text': '', 'annotations': []},
                },
                {
                    'type': 'response.output_text.delta',
                    'item_id': message_id,
                    'output_index': output_index,
                    'content_index': 0,
                    'delta': text,
                    'logprobs': [],
                },
                {
                    'type': 'response.output_text.done',
                    'item_id': message_id,
                    'output_index': output_index,
                    'content_index': 0,
                    'text': text,
                    'logprobs': [],
                },
                {
                    'type': 'response.content_part.done',
                    'item_id': message_id,
                    'output_index': output_index,
                    'content_index': 0,
                    'part': content_part,
                },
                {
                    'type': 'response.output_item.done',
                    'output_index': output_index,
                    'item': item,
                },
            ])
            output_items.append(item)

        tool_calls = message.get('tool_calls') or []
        for tool_call in tool_calls:
            function = tool_call.get('function') or {}
            name = function.get('name', '')
            call_id = tool_call.get('id') or f'call_{uuid.uuid4().hex}'
            output_index = len(output_items)
            if custom_tool_names and name in custom_tool_names:
                # Original tool was a Responses custom tool (e.g. apply_patch). Restore it as a
                # custom_tool_call; Codex >=0.136 routes only custom_tool_call to the apply_patch
                # handler (function_call would raise "apply_patch handler received unsupported
                # payload"). _extract_custom_tool_input recovers the raw input from the function
                # arguments we downgraded on the request side.
                item = {
                    'type': 'custom_tool_call',
                    'id': f'ctc_{uuid.uuid4().hex}',
                    'call_id': call_id,
                    'name': name,
                    'input': _extract_custom_tool_input(function.get('arguments')),
                    'status': 'completed',
                }
                events.append({
                    'type': 'response.output_item.done',
                    'output_index': output_index,
                    'item': item,
                })
            else:
                if custom_tool_names and len(custom_tool_names) == 1 and len(tool_calls) == 1:
                    # The request advertised exactly one custom tool but this call's name did not
                    # match (host may have namespaced/renamed it). Emitting function_call here
                    # silently reintroduces the 0.136 "unsupported payload" crash, so warn.
                    only_custom = next(iter(custom_tool_names))
                    logger.warning(
                        'responses_proxy: tool_call name %r did not match the sole custom tool %r; '
                        'emitting function_call may break Codex >=0.136 apply_patch',
                        name,
                        only_custom,
                    )
                arguments = _ensure_json_string(function.get('arguments'))
                item_id = f'fc_{uuid.uuid4().hex}'
                item = {
                    'type': 'function_call',
                    'id': item_id,
                    'call_id': call_id,
                    'name': name,
                    'arguments': arguments,
                    'status': 'completed',
                }
                events.extend([
                    {
                        'type': 'response.output_item.added',
                        'output_index': output_index,
                        'item': {**item, 'arguments': '', 'status': 'in_progress'},
                    },
                    {
                        'type': 'response.function_call_arguments.delta',
                        'item_id': item_id,
                        'output_index': output_index,
                        'delta': arguments,
                    },
                    {
                        'type': 'response.function_call_arguments.done',
                        'item_id': item_id,
                        'output_index': output_index,
                        'arguments': arguments,
                    },
                    {
                        'type': 'response.output_item.done',
                        'output_index': output_index,
                        'item': item,
                    },
                ])
            output_items.append(item)

    usage = _convert_usage(openai_response.get('usage'))
    events.append({
        'type': 'response.completed',
        'response': {
            'id': response_id,
            'object': 'response',
            'created_at': created_at,
            'model': response_model,
            'output': output_items,
            'status': 'completed',
            'usage': usage,
        },
    })

    return events


def _build_messages(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    instructions = request_body.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError('Judge instructions must be a string')
    if isinstance(instructions, str) and instructions.strip():
        messages.append({'role': 'system', 'content': instructions})

    raw_input = request_body.get('input')
    if isinstance(raw_input, str):
        messages.append({'role': 'user', 'content': raw_input})
        return messages

    if not isinstance(raw_input, list):
        raise ValueError('Judge input must be a string or a list of inline items')

    pending_native_items: list[dict[str, Any]] = []
    pending_assistant_message: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] = []
    pending_reasoning_items: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant_message, pending_tool_calls, pending_reasoning_items, pending_native_items
        if pending_assistant_message is not None:
            message = dict(pending_assistant_message)
            if pending_reasoning_items:
                message['reasoning_items'] = pending_reasoning_items
                _attach_reasoning_content(message, pending_reasoning_items)
            if pending_tool_calls:
                message['tool_calls'] = pending_tool_calls
            message['provider_specific_fields'] = {'responses_output_items': pending_native_items}
            pending_native_items = []
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
            message['provider_specific_fields'] = {'responses_output_items': pending_native_items}
            pending_native_items = []
            messages.append(message)
            pending_tool_calls = []
            pending_reasoning_items = []

    for item in raw_input:
        if not isinstance(item, dict):
            raise ValueError('Judge history items must be objects')

        item_type = item.get('type')
        if item_type == 'reasoning':
            # A new reasoning item starts a new turn after standalone assistant
            # text (notably a compaction summary). Do not merge that summary
            # into the next tool-call turn, which the replay proxy replaces.
            if pending_assistant_message is not None or pending_tool_calls:
                flush_pending_assistant()
            reasoning_item = _normalize_reasoning_item(item)
            if reasoning_item is not None:
                pending_reasoning_items.append(reasoning_item)
                pending_native_items.append(copy.deepcopy(item))
            continue

        if item_type == 'function_call':
            if not item.get('call_id') or not item.get('name'):
                raise ValueError('Judge function calls require call_id and name')
            pending_tool_calls.append(_convert_tool_call_item(item))
            pending_native_items.append(copy.deepcopy(item))
            continue

        if item_type == 'message' or (
            item_type is None and item.get('role') in {'developer', 'system', 'user', 'assistant'}
        ):
            # The Responses API accepts both typed message items and the
            # role/content shorthand emitted by @ai-sdk/openai (OpenCode).
            message = _convert_message_item(item)
            if message is not None:
                if message.get('role') == 'assistant':
                    if pending_assistant_message is not None or pending_tool_calls:
                        flush_pending_assistant()
                    pending_assistant_message = message
                    native_message = copy.deepcopy(item)
                    native_message.setdefault('type', 'message')
                    pending_native_items.append(native_message)
                else:
                    flush_pending_assistant()
                    messages.append(message)
        elif item_type == 'function_call_output':
            if not item.get('call_id'):
                raise ValueError('Judge tool results require call_id')
            flush_pending_assistant()
            messages.append(_convert_tool_output_item(item))
        else:
            raise ValueError(f'Unsupported Judge input item type: {item_type!r}')

    flush_pending_assistant()
    return messages


def _convert_message_item(item: dict[str, Any]) -> dict[str, Any] | None:
    role = item.get('role', 'user')
    if role not in {'developer', 'system', 'user', 'assistant'}:
        raise ValueError(f'Unsupported Judge message role: {role!r}')

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
        raise ValueError('Judge message content must be text or content blocks')

    # Preserve text block boundaries as well as media. Assistant-native items
    # retain their complete original content separately for exact replay.
    return _multimodal_content(content)


def _convert_tool_call_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get('type')
    if item_type == 'custom_tool_call':
        arguments = json.dumps({'input': item.get('input', '')}, ensure_ascii=False)
    else:
        arguments = _ensure_json_string(item.get('arguments'))

    return {
        'id': item.get('call_id') or f'call_{uuid.uuid4().hex}',
        'type': 'function',
        'function': {
            'name': item.get('name', ''),
            'arguments': arguments,
        },
    }


def _multimodal_content(parts: list[Any]) -> list[dict[str, Any]]:
    """Preserve media bytes and metadata across the Runtime/Host boundary."""
    converted = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError('Judge content blocks must be objects')
        if part.get('type') in {'input_text', 'output_text', 'text'}:
            converted.append({'type': 'text', 'text': str(part.get('text', ''))})
        elif part.get('type') in {'input_image', 'image_url', 'input_file'}:
            converted.append(copy.deepcopy(part))
        else:
            # Unknown blocks must not silently turn binary payloads into text.
            raise ValueError(f'Unsupported multimodal content block: {part.get("type")!r}')
    return converted


def _convert_tool_output_item(item: dict[str, Any]) -> dict[str, Any]:
    output = item.get('output')
    if isinstance(output, list):
        blocks = _multimodal_content(output)
        # Preserve the existing string representation for ordinary text tools.
        # Media stays structured; unsupported blocks have already been rejected.
        has_media = any(part['type'] != 'text' for part in blocks)
        content = blocks if has_media else '\n'.join(part['text'] for part in blocks)
    else:
        content = _tool_output_to_text(output)
    return {
        'role': 'tool',
        'tool_call_id': item.get('call_id', ''),
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


def _convert_tools(
    tools: Any,
    *,
    collect_custom_tools: set[str] | None = None,
) -> list[dict[str, Any]] | None:
    del collect_custom_tools
    if not isinstance(tools, list):
        raise ValueError('Judge tools must be a list')
    converted_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            raise ValueError('Judge supports function tools only; native/custom tools are unsupported')
        unknown = set(tool) - {'type', 'name', 'description', 'parameters', 'strict'}
        if unknown:
            raise ValueError(f'Unsupported Judge function tool fields: {sorted(unknown)}')
        if not tool.get('name'):
            raise ValueError('Judge function tools require a name')
        converted_tools.append({
            'type': 'function',
            'function': copy.deepcopy({key: value for key, value in tool.items() if key != 'type'}),
        })
    return converted_tools


def _summary_from_text(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    return [{'type': 'summary_text', 'text': text}]


def _reasoning_summary_events(item: dict[str, Any], *, output_index: int) -> list[dict[str, Any]]:
    """Emit the streaming summary events OpenCode uses for visible reasoning."""
    item_id = str(item.get('id') or '')
    summary = item.get('summary')
    if not item_id or not isinstance(summary, list):
        return []
    events: list[dict[str, Any]] = []
    for summary_index, raw_part in enumerate(summary):
        if not isinstance(raw_part, dict):
            continue
        text = raw_part.get('text')
        if not isinstance(text, str) or not text:
            continue
        part = {'type': 'summary_text', 'text': text}
        events.extend([
            {
                'type': 'response.reasoning_summary_part.added',
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'part': {'type': 'summary_text', 'text': ''},
            },
            {
                'type': 'response.reasoning_summary_text.delta',
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'delta': text,
            },
            {
                'type': 'response.reasoning_summary_text.done',
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'text': text,
            },
            {
                'type': 'response.reasoning_summary_part.done',
                'item_id': item_id,
                'output_index': output_index,
                'summary_index': summary_index,
                'part': part,
            },
        ])
    return events


def _reasoning_content_text(item: dict[str, Any]) -> str:
    content = item.get('content')
    if not isinstance(content, list):
        return ''
    parts = [
        str(part.get('text'))
        for part in content
        if isinstance(part, dict) and part.get('type') == 'reasoning_text' and part.get('text')
    ]
    return '\n'.join(parts)


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


def _normalize_reasoning_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    normalized: dict[str, Any] = {
        'type': 'reasoning',
        'id': item.get('id') or f'rs_{uuid.uuid4().hex}',
    }

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

    encrypted_content = item.get('encrypted_content')
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
    'openai_to_responses_events',
    'responses_to_openai',
]
