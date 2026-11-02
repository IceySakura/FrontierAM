"""Resolve raw run parts to partial RunInput bodies."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import yaml

from nexus.core.registry import EntryNotFoundError
from nexus.dataset.item import resolve_task_files_input
from nexus.runner.compose.merge import set_path
from nexus.runner.compose.model import RawRunPart, RunPart, UseSpec
from nexus.runner.compose.parser import parse_set_expr, parse_use_spec
from nexus.runner.compose.registry import run_part_registry


_RUN_PART_FILE_KEYS = {'body', 'merge', 'source'}


def resolve_non_data_part(raw: RawRunPart) -> RunPart:
    """Resolve a raw non-data run part."""

    if raw.kind == 'use':
        spec = raw.value if isinstance(raw.value, UseSpec) else parse_use_spec(raw.value)
        if spec.name == 'data':
            raise ValueError('data: run parts are handled by the compose compiler')
        return _resolve_use(spec.name, spec.params)

    if raw.kind == 'input':
        if not isinstance(raw.value, dict):
            raise TypeError(f'{raw.source} input must be a mapping')
        return RunPart(body=raw.value, merge=raw.merge, source=raw.source)

    if raw.kind in {'set', 'set_json'}:
        path, value, op = parse_set_expr(raw.value, json_value=raw.kind == 'set_json')
        return RunPart(body=set_path({}, path, value, list_op=op), merge={path: op} if op else {}, source=raw.source)

    raise ValueError(f'Unsupported raw part kind for resolver: {raw.kind}')


def _resolve_use(name: str, params: dict[str, Any]) -> RunPart:
    path = Path(name).expanduser()
    if path.exists() and path.is_file():
        return _load_run_part_file(path, params=params)
    try:
        provider = run_part_registry.get(name)
    except EntryNotFoundError as exc:
        raise KeyError(f'Unknown run part: {name}') from exc
    part = provider(dict(params))
    if not isinstance(part, RunPart):
        raise TypeError(f'Run part {name!r} returned {type(part).__name__}, expected RunPart')
    return part


def _load_run_part_file(path: Path, *, params: dict[str, Any]) -> RunPart:
    if params:
        raise ValueError(f'Run part file {path} does not accept parameters: {sorted(params)}')
    with open(path, encoding='utf-8') as f:
        data = json.load(f) if path.suffix == '.json' else yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f'Run part file must contain a mapping: {path}')

    if _looks_like_run_part_file(data):
        return _parse_run_part_file(data, path=path)

    return _parse_run_input_file(data, path=path)


def _looks_like_run_part_file(data: dict[str, Any]) -> bool:
    return any(key in data for key in _RUN_PART_FILE_KEYS)


def _parse_run_part_file(data: dict[str, Any], *, path: Path) -> RunPart:
    unexpected = sorted(set(data) - _RUN_PART_FILE_KEYS)
    if unexpected:
        raise ValueError(f'Run part file {path} has unsupported top-level keys: {unexpected}')
    body = data.get('body')
    merge = data.get('merge', {})
    source = data.get('source', str(path))
    if not isinstance(body, dict):
        raise TypeError(f'Run part body must be a mapping: {path}')
    if not isinstance(merge, dict):
        raise TypeError(f'Run part merge must be a mapping: {path}')
    if not isinstance(source, str):
        raise TypeError(f'Run part source must be a string: {path}')
    if 'task_files' in body:
        body['task_files'] = resolve_task_files_input(body['task_files'], base_dir=path.parent)
    return RunPart(body=body, merge={str(k): str(v) for k, v in merge.items()}, source=source)


def _parse_run_input_file(data: dict[str, Any], *, path: Path) -> RunPart:
    from nexus.runner.single import RunInput

    if 'task_files' in data:
        data['task_files'] = resolve_task_files_input(data['task_files'], base_dir=path.parent)
    try:
        RunInput.model_validate(data)
    except Exception as exc:
        raise ValueError(f'Run part file {path} must contain either a RunPart mapping or a RunInput mapping') from exc
    return RunPart(body=data, source=str(path))
