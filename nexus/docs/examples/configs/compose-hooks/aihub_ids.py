"""Compose hook: check or assign per-item IDs in an API key's query and record them in ``metadata.__extra__``.

``key`` is the config path of the API key (default ``openai_clients.main.config.client_args.api_key``; ``+`` joins
several). A path ending in ``<field>__env``, or a ``<field>`` absent next to a ``<field>__env``, means that variable.
Variables are read from the submission's ``--host-env`` first, then from the environment running the hook.
Every name in ``fields`` is a parameter of the key's query (``credential?provider=...&acc_global_cache_id=...``).

- ``mode=check`` (default) records the values and warns about missing or batch-duplicated ones; it never changes the
  config or fails the batch.
- ``mode=unique`` gives every item its own UUID for each field. A literal key has any existing value replaced. A
  variable is shared by the whole batch, so it must not already set the field: the key is written back as
  ``<field>: ${oc.env:NAME}&field=<id>`` (``?`` when the value has no query), which the run resolves, and the
  credential stays out of the task list.

    --hook docs/examples/configs/compose-hooks/aihub_ids.py:fields=acc_global_cache_id,mode=unique
"""

from __future__ import annotations

import logging
import os
import re
import uuid

from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, quote, unquote_plus

from nexus.runner.compose.compose_hooks import compose_hook_host_env


if TYPE_CHECKING:
    from nexus.runner.batch.batch_config import BatchItem


logger = logging.getLogger('nexus.compose_hooks.aihub_ids')

METADATA_KEY = '__extra__'
DEFAULT_KEY = 'openai_clients.main.config.client_args.api_key'

_PARAMS = {'key', 'fields', 'mode', 'seed', 'source'}
_FIELD_NAME = re.compile(r'^[A-Za-z0-9_.-]+$')
_KEY_PATH = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[^.]+)+$')
_ENV_SUFFIX = '__env'
_OC_ENV = re.compile(r'\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}')
_ENV_REFERENCE = re.compile(r'\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}(?:[?&](.*))?', re.DOTALL)
_UUID_NAMESPACE = uuid.UUID('6f1c1c55-9a0e-4b51-8d36-0d7f0b6e3a21')


def _names(value: Any, *, param: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f'aihub_ids {param} must be "+"-separated names')
    names = tuple(dict.fromkeys(name.strip() for name in value.split('+')))
    if '' in names:
        raise ValueError(f'aihub_ids {param} contains an empty name')
    return names


def _env_value(name: str) -> str | None:
    return compose_hook_host_env().get(name) or os.environ.get(name)


def _with_param(api_key: str, name: str, value: str) -> str:
    credential, _, query = api_key.partition('?')
    parts = [part for part in query.split('&') if part and unquote_plus(part.partition('=')[0]) != name]
    parts.append(f'{name}={quote(value, safe="")}')
    return f'{credential}?{"&".join(parts)}'


class _Key:
    """The API key at ``path``: a literal ``<field>``, or the variable named by the ``<field>__env`` next to it.

    A literal ``${oc.env:NAME}`` followed by ``&...`` or ``?...`` (what an earlier hook writes back) is the variable
    with per-item parameters appended, so its query continues the variable's own query.
    """

    def __init__(self, config: Any, path: str) -> None:
        first, *middle, field = path.split('.')
        parent: Any = getattr(config, first, None)
        for part in middle:
            parent = parent.get(part) if isinstance(parent, dict) else None
        if not isinstance(parent, dict):
            raise ValueError(f'{path} is not configured')
        self._parent = parent
        self._field = field.removesuffix(_ENV_SUFFIX)
        literal = parent.get(self._field)
        env_field = parent.get(self._field + _ENV_SUFFIX)
        self.env_name: str | None = None
        self._appended = ''
        if not field.endswith(_ENV_SUFFIX) and isinstance(literal, str) and literal:
            reference = _ENV_REFERENCE.fullmatch(literal)
            if reference and _env_value(reference[1]):
                self.env_name, self._appended = reference[1], reference[2] or ''
            else:
                self.value = literal
        elif isinstance(env_field, str) and env_field.strip():
            self.env_name = env_field.strip()
        else:
            raise ValueError(f'{path} is not configured')
        if self.env_name is None:
            query = self.value.partition('?')[2]
            self._shared: dict[str, str] = {}
        else:
            value = _env_value(self.env_name)
            if not value:
                raise ValueError(
                    f'{path} reads {self.env_name}, which is set neither in --host-env nor in this environment'
                )
            self._separator = '&' if '?' in value else '?'
            self._shared = dict(parse_qsl(value.partition('?')[2], keep_blank_values=True))
            query = '&'.join(part for part in (value.partition('?')[2], self._appended) if part)
        # ${oc.env:NAME} in the query resolves where the run does; written values keep the reference.
        query = _OC_ENV.sub(lambda match: _env_value(match[1]) or match[0], query)
        self.query = dict(parse_qsl(query, keep_blank_values=True))

    def set(self, field: str, value: str) -> None:
        if self.env_name is None:
            self.value = _with_param(self.value, field, value)
            self._parent[self._field] = self.value
        else:
            if field in self._shared:
                raise ValueError(
                    f'{self.env_name} already sets {field} for the whole batch; remove it from the variable to '
                    'assign one per item'
                )
            self._appended = _with_param(f'?{self._appended}', field, value).partition('?')[2]
            self._parent[self._field] = f'${{oc.env:{self.env_name}}}{self._separator}{self._appended}'
            self._parent.pop(self._field + _ENV_SUFFIX, None)
        self.query[field] = value


def create_compose_hook(params: dict[str, Any]):
    unknown = sorted(set(params) - _PARAMS)
    if unknown:
        raise ValueError(f'aihub_ids does not accept parameters: {unknown}')
    keys = _names(params.get('key', DEFAULT_KEY), param='key')
    invalid_keys = [key for key in keys if not _KEY_PATH.fullmatch(key)]
    if invalid_keys:
        raise ValueError(f'aihub_ids key must be config paths, got {invalid_keys}')
    fields = _names(params.get('fields', 'cache_task_id'), param='fields')
    invalid_fields = [field for field in fields if not _FIELD_NAME.fullmatch(field)]
    if invalid_fields:
        raise ValueError(f'aihub_ids fields must be api_key query parameter names, got {invalid_fields}')
    mode = params.get('mode', 'check')
    if mode not in {'check', 'unique'}:
        raise ValueError('aihub_ids mode must be check or unique')
    source = params.get('source')
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError('aihub_ids source must be a metadata field name')
    seed = '' if params.get('seed') is None else str(params['seed'])

    class AihubIds:
        def __init__(self) -> None:
            self.position = 0
            self.owners: dict[tuple[str, str], tuple[int, str | None]] = {}
            self.items_with_issues = 0

        def process(self, item: BatchItem) -> BatchItem:
            index = self.position
            self.position += 1
            item = item.model_copy(deep=True)
            run_id = item.run_id or item.config.run_id
            issues: list[str] = []

            def issue(message: str) -> None:
                if mode == 'unique':
                    raise ValueError(message)
                issues.append(message)

            recorded: dict[str, dict[str, str]] = {field: {} for field in fields}
            for path in keys:
                try:
                    key = _Key(item.config, path)
                except ValueError as exc:
                    issue(str(exc))
                    continue
                for field in fields:
                    if mode == 'unique':
                        seed_key = item.metadata.get(source) if source else run_id
                        if seed_key in (None, ''):
                            raise ValueError(
                                f'generating {field} requires {"metadata." + source if source else "a run_id"}'
                            )
                        key.set(field, str(uuid.uuid5(_UUID_NAMESPACE, '\0'.join((field, seed, str(seed_key))))))
                    value = key.query.get(field)
                    if not value:
                        issue(f'{path} has no {field}')
                    elif '${' in value:
                        issue(f'{path} {field} is resolved only at run time')
                    else:
                        recorded[field][path] = value

            extra = item.metadata.setdefault(METADATA_KEY, {})
            if not isinstance(extra, dict):
                raise ValueError(f'metadata.{METADATA_KEY} must be an object')
            for field, by_key in recorded.items():
                if not by_key:
                    extra.pop(field, None)
                    continue
                extra[field] = next(iter(by_key.values())) if len(by_key) == 1 else by_key
                for value in set(by_key.values()):
                    owner_index, owner_run_id = self.owners.setdefault((field, value), (index, run_id))
                    if owner_index != index:
                        issue(f'{field} is also used by item #{owner_index} (run_id={owner_run_id!r})')
            if not extra:
                item.metadata.pop(METADATA_KEY)
            if issues:
                self.items_with_issues += 1
                logger.warning('aihub_ids check: item #%s (run_id=%r): %s', index, run_id, '; '.join(issues))
            return item

        def finish(self) -> None:
            if self.items_with_issues:
                logger.warning(
                    'aihub_ids check: %s of %s items have missing or duplicated IDs',
                    self.items_with_issues,
                    self.position,
                )

    return AihubIds()


__all__ = ['DEFAULT_KEY', 'METADATA_KEY', 'create_compose_hook']
