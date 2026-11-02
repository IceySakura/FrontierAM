"""Runtime-side Responses proxy used by the EnterpriseBench OpenCode judge."""

from __future__ import annotations

import copy
import json
import logging
import uuid

from typing import Any


logger = logging.getLogger(__name__)

_PROXIES: dict[str, Any] = {}


def opencode_responses_to_openai(
    request_body: dict[str, Any],
    *,
    collect_custom_tools: set[str] | None = None,
    collect_tool_name_mappings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an OpenCode Responses request and retain Responses controls."""
    from nexus.extensions.tasks.context_agent.enterprisebench_native.openai_converter import responses_to_openai

    # Enterprise accepts flat function tools and preserves their names verbatim.
    # The shared proxy passes this collector even when no renaming is needed.
    del collect_tool_name_mappings
    params = responses_to_openai(request_body, collect_custom_tools=collect_custom_tools)
    # OpenCode may issue several file reads in one turn. Replaying a Responses
    # turn containing multiple parallel function calls through the internal
    # gateway is rejected on the following request, so keep the Judge agentic
    # while making its tool loop sequential and deterministic.
    params['parallel_tool_calls'] = False
    return params


class _OpenCodeResponsesReplay:
    """Keep provider-native Responses history while OpenCode executes tools."""

    def __init__(self) -> None:
        self._output_turns_by_call_id: dict[str, dict[str, Any]] = {}
        self._next_turn_index = 0

    def request_to_openai(
        self,
        request_body: dict[str, Any],
        *,
        collect_custom_tools: set[str] | None = None,
        collect_tool_name_mappings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = opencode_responses_to_openai(
            request_body,
            collect_custom_tools=collect_custom_tools,
            collect_tool_name_mappings=collect_tool_name_mappings,
        )
        injected = self._inject_output_turns(params['messages'])
        if injected:
            logger.info('EnterpriseBench Judge replayed %d exact Responses output turn(s)', injected)
        return params

    def response_to_events(
        self,
        openai_response: dict[str, Any],
        *,
        custom_tool_names: set[str] | None = None,
        tool_name_mappings: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from nexus.extensions.tasks.context_agent.enterprisebench_native.responses_events import response_events

        # Native output carries the original tool names; it must not be reconstructed.
        del tool_name_mappings
        # Do not retain partial tool calls as executable history.
        status = (openai_response.get('provider_specific_fields') or {}).get('response_status')
        cached = self._cache_output_turns(openai_response) if status == 'completed' else 0
        if cached:
            logger.info('EnterpriseBench Judge cached %d exact Responses output turn(s)', cached)
        return response_events(openai_response, custom_tool_names=custom_tool_names)

    def _cache_output_turns(self, response: dict[str, Any]) -> int:
        cached = 0
        for choice in response.get('choices') or []:
            message = (choice or {}).get('message') or {}
            provider_fields = message.get('provider_specific_fields')
            raw_output = provider_fields.get('responses_output_items') if isinstance(provider_fields, dict) else None
            if not isinstance(raw_output, list):
                continue
            output_items = [copy.deepcopy(item) for item in raw_output if isinstance(item, dict)]
            call_ids = [
                str(item['call_id'])
                for item in output_items
                if item.get('type') in {'function_call', 'computer_call', 'custom_tool_call'} and item.get('call_id')
            ]
            if not call_ids:
                continue
            if len(set(call_ids)) != len(call_ids):
                raise ValueError('Provider response contains duplicate tool call IDs')
            existing_turns = [self._output_turns_by_call_id.get(call_id) for call_id in call_ids]
            if any(turn is not None for turn in existing_turns):
                existing = next(turn for turn in existing_turns if turn is not None)
                if (
                    any(turn is not existing for turn in existing_turns)
                    or existing['call_ids'] != call_ids
                    or existing['output_items'] != output_items
                ):
                    raise ValueError('Conflicting Responses output for an already cached tool call')
                # Identical delivery is idempotent: preserve its position in history.
                continue
            turn = {
                'index': self._next_turn_index,
                'call_ids': call_ids,
                'output_items': output_items,
            }
            self._next_turn_index += 1
            for call_id in call_ids:
                self._output_turns_by_call_id[call_id] = turn
            cached += 1
        return cached

    def _inject_output_turns(self, messages: list[dict[str, Any]]) -> int:
        pending: list[str] = []
        completed: set[str] = set()
        for message in messages:
            calls = message.get('tool_calls') or []
            if calls:
                if pending:
                    raise ValueError('Judge history contains calls before prior tool results')
                pending = [str(call['id']) for call in calls]
                if len(set(pending)) != len(pending) or completed.intersection(pending):
                    raise ValueError('Judge history contains duplicate tool call IDs')
            elif message.get('role') == 'tool':
                call_id = str(message.get('tool_call_id') or '')
                if pending and call_id != pending.pop(0):
                    raise ValueError('Judge tool results are not in call order')
                if call_id in completed:
                    raise ValueError('Judge history contains duplicate tool results')
                completed.add(call_id)
            elif pending:
                raise ValueError('Judge cannot replay messages interleaved between calls and results')
        if pending:
            raise ValueError('Judge history contains unfinished tool calls')
        # Keep standalone assistant/reasoning items in their original order.
        # Only tool-linked assistant projections are replaced by provider output.
        messages[:] = [message for message in messages if not message.get('tool_calls')]
        tool_messages = [message for message in messages if message.get('role') == 'tool']
        if not tool_messages:
            return 0

        observed_by_turn: dict[int, list[str]] = {}
        turns_by_index: dict[int, dict[str, Any]] = {}
        turn_order: list[int] = []
        for message in tool_messages:
            call_id = str(message.get('tool_call_id') or message.get('call_id') or '')
            turn = self._output_turns_by_call_id.get(call_id)
            if turn is None:
                raise ValueError(f'Responses tool result {call_id!r} has no cached provider output turn')
            turn_index = int(turn['index'])
            if turn_index not in observed_by_turn:
                observed_by_turn[turn_index] = []
                turns_by_index[turn_index] = turn
                turn_order.append(turn_index)
            observed_by_turn[turn_index].append(call_id)

        if turn_order != sorted(turn_order):
            raise ValueError('Responses tool result turns are not in provider response order')
        for turn_index, observed_call_ids in observed_by_turn.items():
            expected_call_ids = turns_by_index[turn_index]['call_ids']
            if observed_call_ids != expected_call_ids:
                raise ValueError(
                    f'Responses tool result order {observed_call_ids!r} does not match '
                    f'provider tool call order {expected_call_ids!r}'
                )

        # A single provider response may be split at reasoning/text boundaries
        # by the input converter. Remove only fragments owned by turns that will
        # actually be replayed, not unrelated standalone assistant messages.
        item_owners: dict[str, tuple[int, dict[str, Any]]] = {}
        for turn_index, turn in turns_by_index.items():
            for item in turn['output_items']:
                item_id = item.get('id')
                if not item_id:
                    continue
                if item_id in item_owners:
                    raise ValueError('Cached provider turns contain duplicate output item IDs')
                item_owners[item_id] = (turn_index, item)
        filtered: list[dict[str, Any]] = []
        fragment_turn: int | None = None
        consumed_turns: set[int] = set()
        for message in messages:
            raw_items = (message.get('provider_specific_fields') or {}).get('responses_output_items')
            owners = []
            if message.get('role') == 'assistant' and isinstance(raw_items, list):
                for item in raw_items:
                    owner = item_owners.get(item.get('id'))
                    if owner is not None:
                        if item != owner[1]:
                            raise ValueError('Standalone history fragment conflicts with cached provider output')
                        owners.append(owner[0])
            if owners and isinstance(raw_items, list):
                if len(owners) != len(raw_items) or len(set(owners)) != 1:
                    raise ValueError('Ambiguous ownership of standalone Judge history fragment')
                owner_index = owners[0]
                if owner_index in consumed_turns or fragment_turn not in (None, owner_index):
                    raise ValueError('Judge history fragments are not in provider response order')
                fragment_turn = owner_index
                continue
            if message.get('role') == 'tool':
                turn_index = self._output_turns_by_call_id[str(message['tool_call_id'])]['index']
                if fragment_turn not in (None, turn_index):
                    raise ValueError('Judge history fragment does not precede its own tool result')
                fragment_turn = None
                consumed_turns.add(turn_index)
            elif fragment_turn is not None:
                raise ValueError('Judge cannot move messages across a cached response fragment')
            filtered.append(message)
        if fragment_turn is not None:
            raise ValueError('Judge history fragment has no following tool result')
        messages[:] = filtered

        rebuilt: list[dict[str, Any]] = []
        emitted_turns: set[int] = set()
        for message in messages:
            if message.get('role') == 'tool':
                call_id = str(message.get('tool_call_id') or message.get('call_id') or '')
                turn = self._output_turns_by_call_id[call_id]
                turn_index = int(turn['index'])
                if turn_index not in emitted_turns:
                    rebuilt.append(_responses_output_items_to_chat_message(turn['output_items']))
                    emitted_turns.add(turn_index)
            rebuilt.append(message)
        messages[:] = rebuilt
        return len(emitted_turns)


def _responses_output_items_to_chat_message(output_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror native Responses output into a trace-friendly Chat message.

    ``provider_specific_fields.responses_output_items`` remains the authoritative,
    lossless replay representation. The ordinary Chat-compatible fields are only
    a projection of the same data, allowing processors such as Langfuse to display
    assistant text, reasoning metadata, and tool calls without understanding the
    private provider field. ``OpenAIResponsesClient`` prefers the provider field
    when rebuilding ``input``, so these mirrored fields are not sent twice.
    """
    content_parts: list[str] = []
    reasoning_items: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in output_items:
        item_type = item.get('type')
        if item_type == 'reasoning':
            reasoning_items.append(copy.deepcopy(item))
            for key in ('summary', 'content'):
                section = item.get(key)
                if not isinstance(section, list):
                    continue
                section_parts = [
                    str(part['text'])
                    for part in section
                    if isinstance(part, dict) and isinstance(part.get('text'), str) and part['text']
                ]
                if section_parts:
                    reasoning_parts.extend(section_parts)
                    break
            continue

        if item_type == 'message':
            for part in item.get('content') or []:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                    content_parts.append(part['text'])
                elif part.get('type') == 'refusal':
                    refusal = part.get('refusal', part.get('text'))
                    if isinstance(refusal, str):
                        content_parts.append(refusal)
            continue

        if item_type == 'output_text' and isinstance(item.get('text'), str):
            content_parts.append(item['text'])
            continue

        call_id = item.get('call_id')
        if item_type == 'function_call' and call_id:
            arguments = item.get('arguments', '')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))
            tool_calls.append({
                'id': str(call_id),
                'type': 'function',
                'function': {
                    'name': str(item.get('name') or ''),
                    'arguments': arguments,
                },
            })
            continue

        if item_type == 'custom_tool_call' and call_id:
            tool_calls.append({
                'id': str(call_id),
                'type': 'function',
                'function': {
                    'name': str(item.get('name') or ''),
                    'arguments': json.dumps({'input': item.get('input', '')}, ensure_ascii=False),
                },
            })
            continue

        if item_type == 'computer_call' and call_id:
            tool_calls.append({
                'id': str(call_id),
                'type': 'function',
                'function': {
                    'name': 'computer',
                    'arguments': json.dumps(
                        {
                            'actions': item.get('actions') or ([item['action']] if item.get('action') else []),
                            'pending_safety_checks': item.get('pending_safety_checks') or [],
                            'raw_item': item,
                        },
                        ensure_ascii=False,
                        separators=(',', ':'),
                    ),
                },
            })

    message: dict[str, Any] = {
        'role': 'assistant',
        'content': '\n'.join(content_parts) if content_parts else None,
        'provider_specific_fields': {
            'responses_output_items': copy.deepcopy(output_items),
        },
    }
    if reasoning_items:
        message['reasoning_items'] = reasoning_items
    if reasoning_parts:
        message['reasoning_content'] = '\n'.join(reasoning_parts)
    if tool_calls:
        message['tool_calls'] = tool_calls
    return message


async def start_responses_proxy(*, request_timeout: float = 7200.0) -> dict[str, str]:
    """Start a Runtime-local Responses proxy and return its connection details."""
    from nexus.extensions.agents.installed_common import find_free_port
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.responses_proxy import ResponsesProxy

    runtime_control = require_runtime_control()
    proxy_id = uuid.uuid4().hex
    port = find_free_port()
    replay = _OpenCodeResponsesReplay()
    proxy = ResponsesProxy(
        sio=runtime_control.sio,
        listen_host='127.0.0.1',
        listen_port=port,
        request_timeout=request_timeout,
        responses_to_openai_converter=replay.request_to_openai,
        openai_to_responses_events_converter=replay.response_to_events,
    )
    await proxy.start()
    _PROXIES[proxy_id] = proxy
    return {
        'proxy_id': proxy_id,
        'base_url': f'http://127.0.0.1:{port}/v1',
    }


async def stop_responses_proxy(proxy_id: str) -> bool:
    """Stop a previously started Runtime-local Responses proxy."""
    proxy = _PROXIES.pop(proxy_id, None)
    if proxy is None:
        return False
    await proxy.stop()
    return True


__all__ = ['opencode_responses_to_openai', 'start_responses_proxy', 'stop_responses_proxy']
