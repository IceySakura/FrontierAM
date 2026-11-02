"""Export cerebellum and midbrain message logs in the hy-live trajectory format.

One record per model: ``{"tools": [...], "messages": [...]}``. Each message keeps only ``role``,
``content`` (never None), ``at`` (Beijing time with microseconds), parsed ``tool_calls``, and for
tool results ``tool_call_id`` plus ``name``. Private keys such as ``_at``/``_injected`` and
provider fields such as ``reasoning_content`` are dropped.
"""

from __future__ import annotations

import contextlib
import json

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any


BEIJING = timezone(timedelta(hours=8))


def stamp(at: Any) -> str:
    """Format an epoch timestamp as a Beijing ISO time with microseconds, or '' when unset."""
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not at:
        return ''
    return datetime.fromtimestamp(at, BEIJING).isoformat(timespec='microseconds')


def normalize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Render model messages in the exported trajectory format."""
    names: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        item: dict[str, Any] = {'role': message.get('role'), 'content': '' if content is None else content}
        if isinstance(message.get('_at'), (int, float)) and not isinstance(message.get('_at'), bool):
            item['at'] = stamp(message['_at'])
        calls = [call for call in message.get('tool_calls') or [] if isinstance(call, dict)]
        if calls:
            rendered = []
            for call in calls:
                function = call.get('function') or {}
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        arguments = json.loads(arguments)
                name = str(function.get('name') or '')
                call_id = str(call.get('id') or '')
                rendered.append({
                    'id': call_id,
                    'type': 'function',
                    'function': {'name': name, 'arguments': {} if arguments is None else arguments},
                })
                if call_id and name:
                    names[call_id] = name
            item['tool_calls'] = rendered
        if message.get('role') == 'tool':
            call_id = str(message.get('tool_call_id') or '')
            item['tool_call_id'] = call_id
            item['name'] = str(message.get('name') or names.get(call_id, ''))
        normalized.append(item)
    return normalized


def trajectory_record(tools: Iterable[dict[str, Any]] | None, messages: Iterable[Any] | None) -> dict[str, Any]:
    """One strict-format sample: the tools offered to that model, then its full message list."""
    return {'tools': list(tools or []), 'messages': normalize_messages(messages or [])}


def _layer_entries(layer: str, messages: Iterable[Any] | None) -> list[tuple[float, str, dict[str, Any], bool]]:
    """Each message as exported, with when it was added and whether the system wrote it."""
    raw = [item for item in messages or [] if isinstance(item, dict)]
    entries: list[tuple[float, str, dict[str, Any], bool]] = []
    for source, item in zip(raw, normalize_messages(raw), strict=True):
        at = source.get('_at')
        moment = float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) else 0.0
        entries.append((moment, layer, item, bool(source.get('_injected'))))
    return entries


def dialogue_record(
    cerebellum_tools: Iterable[dict[str, Any]] | None,
    cerebellum_messages: Iterable[Any] | None,
    midbrain_tools: Iterable[dict[str, Any]] | None,
    midbrain_messages: Iterable[Any] | None,
) -> dict[str, Any]:
    """Both models in one timeline, as written to ``dialogue.jsonl``.

    Each real user line opens a turn; what both models did until the next one follows it in the
    order it happened. Injected cerebellum messages never open a turn.
    """
    cerebellum = _layer_entries('cerebellum', cerebellum_messages)
    midbrain = _layer_entries('midbrain', midbrain_messages)
    merged: list[tuple[float, str, dict[str, Any], bool]] = []
    left = right = 0
    while left < len(cerebellum) or right < len(midbrain):
        take_left = right >= len(midbrain) or (left < len(cerebellum) and cerebellum[left][0] <= midbrain[right][0])
        if take_left:
            merged.append(cerebellum[left])
            left += 1
        else:
            merged.append(midbrain[right])
            right += 1
    system: dict[str, str] = {}
    turns: list[dict[str, Any]] = []
    for at, layer, item, injected in merged:
        if item['role'] == 'system':
            system.setdefault(layer, item['content'])
            continue
        content = item['content']
        if layer == 'cerebellum' and item['role'] == 'user' and isinstance(content, str) and not injected:
            turns.append({'turn': len(turns) + 1, 'at': stamp(at), 'user': content, 'messages': []})
            continue
        if not turns:
            turns.append({'turn': 0, 'at': stamp(at), 'user': '', 'messages': []})
        turns[-1]['messages'].append({'layer': layer, **item})
    return {
        'tools': {'cerebellum': list(cerebellum_tools or []), 'midbrain': list(midbrain_tools or [])},
        'system': {layer: system[layer] for layer in ('cerebellum', 'midbrain') if layer in system},
        'turns': turns,
    }


__all__ = ['dialogue_record', 'trajectory_record']
