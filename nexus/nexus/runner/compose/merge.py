"""Run part merge rules."""

from __future__ import annotations

import copy

from typing import Any


LIST_OPS = {'append', 'append_unique', 'replace'}


def deep_merge(
    base: dict[str, Any], overlay: dict[str, Any], merge_ops: dict[str, str] | None = None
) -> dict[str, Any]:
    """Merge ``overlay`` into ``base`` using Nexus run part rules."""

    result = copy.deepcopy(base)
    _merge_into(result, copy.deepcopy(overlay), merge_ops or {}, path='')
    return result


def set_path(base: dict[str, Any], path: str, value: Any, *, list_op: str | None = None) -> dict[str, Any]:
    """Return a copy of ``base`` with a dotted path assigned."""

    if list_op is not None and list_op not in LIST_OPS:
        raise ValueError(f'Unsupported list merge op: {list_op}')
    overlay: dict[str, Any] = {}
    current = overlay
    parts = path.split('.')
    if any(not part for part in parts):
        raise ValueError(f'Invalid dotted path: {path}')
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = [value] if list_op in {'append', 'append_unique'} else value
    merge_ops = {path: list_op} if list_op else {}
    return deep_merge(base, overlay, merge_ops)


def _merge_into(target: dict[str, Any], overlay: dict[str, Any], merge_ops: dict[str, str], *, path: str) -> None:
    for key, value in overlay.items():
        child_path = f'{path}.{key}' if path else key
        op = merge_ops.get(child_path)
        if op is not None and op not in LIST_OPS:
            raise ValueError(f'Unsupported merge op for {child_path}: {op}')

        if value is None:
            target.pop(key, None)
            continue

        if op in {'append', 'append_unique'}:
            target[key] = _merge_list(target.get(key), value, op)
            continue
        if op == 'replace':
            target[key] = value
            continue

        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_into(existing, value, merge_ops, path=child_path)
        else:
            target[key] = value


def _merge_list(existing: Any, incoming: Any, op: str) -> list[Any]:
    current = list(existing) if isinstance(existing, list) else []
    values = incoming if isinstance(incoming, list) else [incoming]
    if op == 'append':
        current.extend(values)
        return current

    for item in values:
        if item not in current:
            current.append(item)
    return current
