"""Ordered CLI run part parser."""

from __future__ import annotations

import json
import shlex

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from nexus.dataset.item import resolve_task_files_input
from nexus.dataset.layout import is_local_dataset_path
from nexus.runner.compose.model import RawPartKind, RawRunPart, RunComposition, UseSpec


_VALUE_OPTIONS: dict[str, RawPartKind] = {
    '--data': 'data',
    '--use': 'use',
    '--set': 'set',
    '--set-json': 'set_json',
}


def parse_ordered_parts(argv: list[str]) -> list[RawRunPart]:
    """Extract ordered compose options from raw CLI argv."""

    parts: list[RawRunPart] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        matched = False
        for option, kind in _VALUE_OPTIONS.items():
            if token == option:
                if index + 1 >= len(argv):
                    raise ValueError(f'{option} requires a value')
                value = argv[index + 1]
                parts.append(RawRunPart(kind=kind, value=value, source=f'{option} {value}'))
                index += 2
                matched = True
                break
            if token.startswith(f'{option}='):
                value = token[len(option) + 1 :]
                if not value:
                    raise ValueError(f'{option} requires a value')
                parts.append(RawRunPart(kind=kind, value=value, source=f'{option}={value}'))
                index += 1
                matched = True
                break
        if not matched:
            index += 1
    return parts


def load_run_file(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML or JSON run file without resolving it."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f'Config file not found: {source_path}')
    with open(source_path, encoding='utf-8') as f:
        data = json.load(f) if source_path.suffix == '.json' else yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f'Config file must contain a mapping, got: {type(data).__name__}')
    return data, source_path


def parse_run_composition(data: Mapping[str, Any], *, source_path: str | Path) -> RunComposition:
    """Parse the ordered ``compose`` form of a run file."""

    source = Path(source_path).expanduser().resolve()
    unexpected = sorted(set(data) - {'compose'})
    if unexpected:
        raise ValueError(f'Compose file {source} has unsupported top-level keys: {unexpected}')
    raw_parts = data.get('compose')
    if not isinstance(raw_parts, list) or not raw_parts:
        raise TypeError(f'Compose file {source} requires a non-empty compose list')

    parts: list[RawRunPart] = []
    item_id: str | None = None
    split: str | None = None
    limit: int | None = None
    seen_data = False
    for index, raw_part in enumerate(raw_parts):
        part_source = f'{source}:compose[{index}]'
        if not isinstance(raw_part, Mapping):
            raise TypeError(f'{part_source} must be a mapping')
        operations = [key for key in ('data', 'use', 'input') if key in raw_part]
        if len(operations) != 1:
            raise ValueError(f'{part_source} must contain exactly one of data, use, or input')
        operation = operations[0]
        allowed = {operation, 'merge'} if operation == 'input' else {operation}
        extra = sorted(set(raw_part) - allowed)
        if extra:
            raise ValueError(f'{part_source} has unsupported keys: {extra}')

        if operation == 'data':
            if seen_data:
                raise ValueError(f'{part_source}: only one data part is supported')
            seen_data = True
            uri, part_split, part_item_id, part_limit = _parse_data_part(raw_part['data'], source=part_source)
            item_id, split, limit = part_item_id, part_split, part_limit
            uri = _resolve_data_uri(uri, base_dir=source.parent)
            parts.append(RawRunPart(kind='data', value=uri, source=part_source))
            continue

        if operation == 'use':
            use = _parse_use_part(raw_part['use'], source=part_source)
            use = UseSpec(name=_resolve_use_name(use.name, base_dir=source.parent), params=use.params)
            parts.append(RawRunPart(kind='use', value=use, source=part_source))
            continue

        body = raw_part['input']
        if not isinstance(body, Mapping):
            raise TypeError(f'{part_source}.input must be a mapping')
        normalized_body = dict(body)
        if 'task_files' in normalized_body:
            normalized_body['task_files'] = resolve_task_files_input(
                normalized_body['task_files'], base_dir=source.parent
            )
        merge = raw_part.get('merge', {})
        if not isinstance(merge, Mapping):
            raise TypeError(f'{part_source}.merge must be a mapping')
        parts.append(
            RawRunPart(
                kind='input',
                value=normalized_body,
                source=part_source,
                merge={str(key): str(value) for key, value in merge.items()},
            )
        )

    return RunComposition(parts=parts, item_id=item_id, split=split, limit=limit, source=str(source))


def parse_use_spec(value: str) -> UseSpec:
    """Parse ``name:param=value`` use syntax."""

    if value.startswith('data:'):
        return UseSpec(name='data', params={'uri': value[len('data:') :]})

    name, sep, raw_params = value.partition(':')
    if not name:
        raise ValueError(f'Invalid run part: {value}')
    if not sep:
        return UseSpec(name=name, params={})
    params = _parse_params(raw_params)
    return UseSpec(name=name, params=params)


def _parse_data_part(value: Any, *, source: str) -> tuple[str, str | None, str | None, int | None]:
    if isinstance(value, str):
        return value, None, None, None
    if not isinstance(value, Mapping):
        raise TypeError(f'{source}.data must be a dataset URI/path string or mapping')
    unexpected = sorted(set(value) - {'uri', 'split', 'id', 'limit'})
    if unexpected:
        raise ValueError(f'{source}.data has unsupported keys: {unexpected}')
    uri = value.get('uri')
    if not isinstance(uri, str) or not uri:
        raise ValueError(f'{source}.data.uri must be a non-empty string')
    split = value.get('split')
    item_id = value.get('id')
    limit = value.get('limit')
    if split is not None and not isinstance(split, str):
        raise TypeError(f'{source}.data.split must be a string')
    if item_id is not None and not isinstance(item_id, str):
        raise TypeError(f'{source}.data.id must be a string')
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
        raise TypeError(f'{source}.data.limit must be a non-negative integer')
    return uri, split, item_id, limit


def _parse_use_part(value: Any, *, source: str) -> UseSpec:
    if isinstance(value, str):
        return parse_use_spec(value)
    if not isinstance(value, Mapping):
        raise TypeError(f'{source}.use must be a run part string or mapping')
    unexpected = sorted(set(value) - {'name', 'params'})
    if unexpected:
        raise ValueError(f'{source}.use has unsupported keys: {unexpected}')
    name = value.get('name')
    params = value.get('params', {})
    if not isinstance(name, str) or not name:
        raise ValueError(f'{source}.use.name must be a non-empty string')
    if not isinstance(params, Mapping):
        raise TypeError(f'{source}.use.params must be a mapping')
    return UseSpec(name=name, params=dict(params))


def _resolve_data_uri(uri: str, *, base_dir: Path) -> str:
    path = Path(uri).expanduser()
    candidate = path if path.is_absolute() else base_dir / path
    if is_local_dataset_path(candidate) or path.is_absolute() or uri.startswith(('./', '../', '~')):
        return str(candidate.resolve())
    return uri


def _resolve_use_name(name: str, *, base_dir: Path) -> str:
    path = Path(name).expanduser()
    candidate = path if path.is_absolute() else base_dir / path
    return str(candidate.resolve()) if candidate.is_file() else name


def parse_set_expr(value: str, *, json_value: bool = False) -> tuple[str, Any, str | None]:
    """Parse ``path=value`` or ``path+=value`` into path/value/list-op."""

    if '+=' in value:
        path, raw = value.split('+=', 1)
        op = 'append'
    elif '=' in value:
        path, raw = value.split('=', 1)
        op = None
    else:
        raise ValueError(f'Expected KEY=VALUE for --set, got: {value}')

    path = path.strip()
    if not path:
        raise ValueError(f'Set path must not be empty: {value}')

    parsed_value = json.loads(raw) if json_value else _parse_scalar(raw)
    return path, parsed_value, op


def _parse_params(value: str) -> dict[str, Any]:
    if not value:
        return {}
    params: dict[str, Any] = {}
    for token in shlex.split(value.replace(',', ' ')):
        if '=' not in token:
            raise ValueError(f'Expected key=value parameter in run part: {token}')
        key, raw = token.split('=', 1)
        if not key:
            raise ValueError(f'Run part parameter key must not be empty: {value}')
        params[key] = _parse_scalar(raw)
    return params


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        return parsed
    return parsed
