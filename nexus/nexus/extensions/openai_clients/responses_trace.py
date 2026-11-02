"""Trace-only projections of native Responses; never use them as model input."""

from __future__ import annotations

import json

from typing import Any

from nexus.trace.tracer import start_span


def _content(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    parts = []
    for part in value:
        if not isinstance(part, dict):
            parts.append({'type': 'text', 'text': str(part)})
        elif part.get('type') == 'input_image':
            parts.append({
                'type': 'image_url',
                'image_url': {'url': part.get('image_url', ''), 'detail': part.get('detail', 'auto')},
            })
        elif part.get('type') in {'input_text', 'output_text', 'text'}:
            parts.append({'type': 'text', 'text': part.get('text', '')})
        else:
            parts.append(dict(part))
    if parts and all(p.get('type') == 'text' for p in parts):
        return '\n'.join(p['text'] for p in parts)
    return parts


def messages_for_trace(body: dict[str, Any], regions: dict[str, dict[str, str]] | None = None) -> list[dict]:
    """Project without dropping custom calls or detaching tool images from IDs."""
    messages: list[dict] = []
    if body.get('instructions'):
        messages.append({'role': 'system', 'content': body['instructions']})
    items = body.get('input', [])
    if isinstance(items, str):
        return [*messages, {'role': 'user', 'content': items}]
    for item in items:
        kind = item.get('type', 'message')
        if kind in {'function_call_output', 'custom_tool_call_output'}:
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': _content(item['output'])})
        elif kind == 'message':
            role = item.get('role', 'user')
            content = _content(item.get('content', ''))
            if role == 'assistant' and messages and messages[-1]['role'] == 'assistant':
                messages[-1]['content'] = content
            else:
                messages.append({'role': 'system' if role == 'developer' else role, 'content': content})
        elif kind in {'reasoning', 'function_call', 'custom_tool_call'}:
            if not messages or messages[-1]['role'] != 'assistant':
                messages.append({'role': 'assistant', 'content': ''})
            if kind == 'reasoning':
                reasoning = dict(item)
                reasoning.update((regions or {}).get(str(item.get('encrypted_content', '')), {}))
                messages[-1].setdefault('reasoning_items', []).append(reasoning)
            else:
                args = (
                    json.dumps({'input': item.get('input', '')}, ensure_ascii=False)
                    if kind == 'custom_tool_call'
                    else item.get('arguments', '')
                )
                messages[-1].setdefault('tool_calls', []).append({
                    'id': item['call_id'],
                    'type': 'function',
                    'function': {'name': item['name'], 'arguments': args},
                })
        else:
            messages.append({'role': 'assistant', 'content': '', 'responses_items': [dict(item)]})
    return messages


def trace_response(body: dict, raw: dict, regions: dict, provider_call: dict) -> None:
    projected = messages_for_trace({'input': raw['output']}, regions)
    message = (
        projected[0]
        if len(projected) == 1
        else {
            'role': 'assistant',
            'content': '',
            'responses_items': raw['output'],
        }
    )
    usage = raw.get('usage') or {}
    completion = {
        'id': raw.get('id'),
        'object': 'chat.completion',
        'model': body['model'],
        'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
        'usage': {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'prompt_tokens_details': usage.get('input_tokens_details') or {},
            'completion_tokens_details': usage.get('output_tokens_details') or {},
        },
    }
    span = start_span(
        'openai_completion',
        {
            'inputs': {'messages': messages_for_trace(body, regions), 'kwargs': {'tools': body.get('tools', [])}},
            'provider_call': provider_call,
        },
    )
    span.update(attributes={'outputs': completion})
    span.finish()
