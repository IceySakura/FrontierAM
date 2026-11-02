from __future__ import annotations

import json
import os

from typing import Any


_DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_QUEUE_ITEMS = 10_000
_DEFAULT_MAX_QUEUE_BYTES = 1536 * 1024 * 1024
_MEMORY_BYTES_PER_WRITER = 4 * 1024**3
_WRITER_SHARDS_ENV = 'NEXUS_TRAJ_TDSQL_WRITER_SHARDS'


def _positive_int_setting(value: int | None, *, name: str, env_name: str, default: int) -> int:
    if value is None:
        raw = os.getenv(env_name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f'{env_name} must be a positive integer') from None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def resolve_max_queue_items(value: int | None = None) -> int:
    return _positive_int_setting(
        value, name='max_queue_items', env_name='NEXUS_TRAJ_TDSQL_MAX_QUEUE_ITEMS', default=_DEFAULT_MAX_QUEUE_ITEMS
    )


def resolve_max_queue_bytes(value: int | None = None) -> int:
    return _positive_int_setting(
        value, name='max_queue_bytes', env_name='NEXUS_TRAJ_TDSQL_MAX_QUEUE_BYTES', default=_DEFAULT_MAX_QUEUE_BYTES
    )


def _manager_memory_bytes() -> int:
    try:
        spec = json.loads(os.getenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', '{}'))
    except (ValueError, TypeError):
        return 0
    resources = spec.get('resources') if isinstance(spec, dict) else None
    if not isinstance(resources, dict):
        return 0
    total = resources.get('memory_bytes')
    count = resources.get('manager_count')
    if type(total) is not int or type(count) is not int or total <= 0 or count <= 0 or total % count:
        return 0
    return total // count


def resolve_writer_shards(value: int | None = None) -> int:
    if value is not None or _WRITER_SHARDS_ENV in os.environ:
        return _positive_int_setting(value, name='writer_shards', env_name=_WRITER_SHARDS_ENV, default=1)
    memory_bytes = _manager_memory_bytes()
    return max(1, min(memory_bytes // _MEMORY_BYTES_PER_WRITER, 4))


def _connect_timeout() -> int:
    raw = os.environ.get('NEXUS_TRAJ_DB_CONNECT_TIMEOUT')
    if raw is None:
        return _DEFAULT_CONNECT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_CONNECT_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_CONNECT_TIMEOUT_SECONDS


def build_db_config(
    *,
    host: str,
    user: str,
    db: str,
    password: str = '',
    port: int = 3306,
) -> dict[str, Any]:
    for field_name, value in (('host', host), ('user', user), ('db', db)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'TDSQL {field_name} must be a non-empty string')
    if not isinstance(password, str):
        raise ValueError('TDSQL password must be a string')
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError('TDSQL port must be between 1 and 65535')
    return {
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'db': db,
        'charset': 'utf8mb4',
        'autocommit': False,
        'connect_timeout': _connect_timeout(),
    }


__all__ = [
    'build_db_config',
    'resolve_max_queue_bytes',
    'resolve_max_queue_items',
    'resolve_writer_shards',
]
