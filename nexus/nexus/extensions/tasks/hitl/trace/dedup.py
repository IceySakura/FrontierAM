"""Export-time deduplication rules for chat-tracker trajectories.

Two rules, applied in order: drop a trajectory whose normalized message sequence
is a strict prefix of a longer one, then drop one whose ``tool_call_id`` set is a
strict subset of a longer one's. Both compare within a single branch.

Not applied during a run -- see ``docs/shared/05-chat-tracker-dedup.md``. These are
for post-processing, which reads an exported jsonl and can be re-run.
"""

import re

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast


_BILLING_HEADER_RE = re.compile(r'^x-anthropic-billing-header:.*?\n', re.MULTILINE)


def deduplicate_tracker_result(result: dict[str, object]) -> dict[str, object]:
    """Apply both rules to a tracker result, leaving the input untouched."""
    copied = deepcopy(result)
    trajectories = copied.get('trajectories')
    if not isinstance(trajectories, list):
        return copied

    copied['trajectories'] = _deduplicate_trajectories(cast(list[object], trajectories))
    return copied


def _deduplicate_trajectories(trajectories: list[object]) -> list[object]:
    """Remove partial/retried trajectory snapshots.

    This mirrors the old HITL export logic:
    1. remove trajectories whose normalized message sequence is a strict prefix
       of a longer one;
    2. remove trajectories whose tool_call_id set is a strict subset of a
       longer trajectory's set.

    Both steps are conservative and export-only. The underlying chat tree is
    preserved for debugging.
    """

    if len(trajectories) <= 1:
        return trajectories

    step1 = _deduplicate_by_message_prefix(trajectories)
    if len(step1) <= 1:
        return step1
    return _deduplicate_by_tool_call_ids(step1)


def _deduplicate_by_message_prefix(trajectories: list[object]) -> list[object]:
    # Longest first: a shorter trajectory is only recognised as a prefix once a longer
    # one is in ``kept``. The length comes from ``item[1]`` -- given the whole
    # ``(index, trajectory)`` pair, _trajectory_messages returns [] for every entry,
    # which sorts them all equal and leaves the original order, shortest first.
    indexed = sorted(
        enumerate(trajectories),
        key=lambda item: -len(_trajectory_messages(item[1])),
    )
    kept: list[tuple[int, list[str]]] = []

    for original_index, trajectory in indexed:
        fingerprint = [_normalize_message_for_dedup(message) for message in _trajectory_messages(trajectory)]
        is_prefix = False
        for _, longer_fingerprint in kept:
            if len(longer_fingerprint) <= len(fingerprint):
                continue
            if longer_fingerprint[: len(fingerprint)] == fingerprint:
                is_prefix = True
                break
        if not is_prefix:
            kept.append((original_index, fingerprint))

    return [trajectories[index] for index in sorted(index for index, _ in kept)]


def _deduplicate_by_tool_call_ids(trajectories: list[object]) -> list[object]:
    tool_id_sets = [_extract_tool_call_ids(_trajectory_messages(trajectory)) for trajectory in trajectories]
    ordered_indices = sorted(range(len(trajectories)), key=lambda index: -len(tool_id_sets[index]))
    kept: list[int] = []

    for index in ordered_indices:
        tool_ids = tool_id_sets[index]
        if not tool_ids:
            kept.append(index)
            continue
        is_subset = False
        for kept_index in kept:
            if len(tool_id_sets[kept_index]) > len(tool_ids) and tool_ids < tool_id_sets[kept_index]:
                is_subset = True
                break
        if not is_subset:
            kept.append(index)

    return [trajectories[index] for index in sorted(kept)]


def _trajectory_messages(trajectory: object) -> list[dict[str, object]]:
    if isinstance(trajectory, list):
        return [cast(dict[str, object], m) for m in trajectory if isinstance(m, dict)]
    if isinstance(trajectory, dict):
        messages = cast(Mapping[str, Any], trajectory).get('messages')
        if isinstance(messages, list):
            return [cast(dict[str, object], m) for m in messages if isinstance(m, dict)]
    return []


def _normalize_message_for_dedup(message: dict[str, object]) -> str:
    role = str(message.get('role') or '')
    content = _normalize_content_for_dedup(message.get('content'))
    tool_signature = _tool_calls_signature(message.get('tool_calls'))
    return f'{role}:{content}:{tool_signature}'


def _normalize_content_for_dedup(content: object) -> str:
    if isinstance(content, str):
        return _BILLING_HEADER_RE.sub('', content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                block = cast(Mapping[str, Any], item)
                text = block.get('text') or block.get('content') or block.get('thinking')
                if isinstance(text, str):
                    parts.append(_BILLING_HEADER_RE.sub('', text))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return '|'.join(parts)
    return '' if content is None else str(content)


def _tool_calls_signature(tool_calls: object) -> str:
    if not isinstance(tool_calls, list):
        return ''
    signatures: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = cast(Mapping[str, Any], tool_call).get('function')
        if not isinstance(function, dict):
            continue
        name = str(function.get('name') or '')
        arguments = function.get('arguments') or ''
        if isinstance(arguments, str):
            arguments = _normalize_json_string(arguments)
        signatures.append(f'{name}:{arguments}')
    return '|'.join(signatures)


def _normalize_json_string(value: str) -> str:
    import json

    try:
        return json.dumps(json.loads(value), sort_keys=True, separators=(',', ':'))
    except (json.JSONDecodeError, TypeError):
        return value


def _extract_tool_call_ids(messages: list[dict[str, object]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        tool_call_id = message.get('tool_call_id')
        if isinstance(tool_call_id, str) and tool_call_id:
            ids.add(tool_call_id)
        tool_calls = message.get('tool_calls')
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = cast(Mapping[str, Any], tool_call).get('id')
            if isinstance(tool_id, str) and tool_id:
                ids.add(tool_id)
    return ids


__all__ = [
    'deduplicate_tracker_result',
]
