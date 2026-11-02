"""Render buffered native Responses output without relabelling incomplete responses."""

import copy

from typing import Any


def response_events(completion: dict[str, Any], **_kwargs: Any) -> list[dict[str, Any]]:
    """Emit buffered native-item SSE, not a reconstruction from Chat projections."""
    from nexus.extensions.tasks.context_agent.enterprisebench_native.openai_converter import _reasoning_summary_events

    fields = completion.get('provider_specific_fields') or {}
    state = fields.get('response_state')
    output = fields.get('response_output')
    if not isinstance(state, dict) or not isinstance(output, list):
        raise ValueError('Host Judge requires native Responses output/state; refusing Chat fallback')
    response = copy.deepcopy(state)
    response['output'] = copy.deepcopy(output)
    status = response.get('status')
    if status not in {'completed', 'incomplete', 'failed'}:
        raise ValueError(f'Unsupported Responses terminal state: {status!r}')
    initial = {**response, 'output': [], 'status': 'in_progress', 'usage': None, 'error': None}
    events = [{'type': 'response.created', 'response': initial}]
    for index, original in enumerate(output):
        item = copy.deepcopy(original)
        kind = item.get('type')
        added = copy.deepcopy(item)
        if kind == 'function_call':
            added['arguments'] = ''
        elif kind == 'message':
            added['content'] = []
        elif kind == 'reasoning':
            added['summary'] = []
        added['status'] = 'in_progress'
        events.append({'type': 'response.output_item.added', 'output_index': index, 'item': added})
        common = {'item_id': item.get('id'), 'output_index': index}
        if kind == 'function_call':
            arguments = item.get('arguments', '')
            events.extend([
                {'type': 'response.function_call_arguments.delta', **common, 'delta': arguments},
                {'type': 'response.function_call_arguments.done', **common, 'arguments': arguments},
            ])
        elif kind == 'reasoning':
            events.extend(_reasoning_summary_events(item, output_index=index))
        elif kind == 'message':
            for ci, part in enumerate(item.get('content') or []):
                pc = {**common, 'content_index': ci}
                part_type = part.get('type')
                field = 'text' if part_type == 'output_text' else 'refusal' if part_type == 'refusal' else None
                if field is None:
                    raise ValueError(f'Unsupported Responses message content: {part_type!r}')
                events.append({'type': 'response.content_part.added', **pc, 'part': {**part, field: ''}})
                event_kind = 'output_text' if field == 'text' else 'refusal'
                events.extend([
                    {'type': f'response.{event_kind}.delta', **pc, 'delta': part.get(field, '')},
                    {'type': f'response.{event_kind}.done', **pc, field: part.get(field, '')},
                    {'type': 'response.content_part.done', **pc, 'part': part},
                ])
        else:
            raise ValueError(f'Unsupported Responses output item: {kind!r}')
        events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    events.append({'type': f'response.{status}', 'response': response})
    for seq, event in enumerate(events):
        event['sequence_number'] = seq
    return events
