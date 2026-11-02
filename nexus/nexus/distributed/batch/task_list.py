from __future__ import annotations

import hashlib
import json
import os
import re

from typing import Literal, TypeGuard


MAX_BATCH_ITEM_ID_LENGTH = 64
MAX_BATCH_METADATA_BYTES = 64 * 1024
MAX_BATCH_ITEM_METADATA_BYTES = 256 * 1024
DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES = 128 * 1024
BATCH_ITEM_METADATA_BYTES_ENV = 'NEXUS_SUBMIT_MAX_BATCH_ITEM_METADATA_BYTES'
TaskListVersion = Literal[1, 2, 3]
_BATCH_ITEM_ID_PATTERN = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}', flags=re.ASCII)
_INVALID_BATCH_ITEM_ID_RUN = re.compile(r'[^a-z0-9._-]+', flags=re.ASCII)
_NORMALIZED_ID_DIGEST_LENGTH = 12


def is_valid_batch_item_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _BATCH_ITEM_ID_PATTERN.fullmatch(value) is not None


def normalize_batch_item_id(value: str) -> str:
    if is_valid_batch_item_id(value):
        return value

    digest = hashlib.sha256(value.encode()).hexdigest()[:_NORMALIZED_ID_DIGEST_LENGTH]
    readable = _INVALID_BATCH_ITEM_ID_RUN.sub('-', value.lower()).strip('._-') or 'item'
    max_readable_length = MAX_BATCH_ITEM_ID_LENGTH - len(digest) - 1
    readable = readable[:max_readable_length].rstrip('._-') or 'item'
    return f'{readable}-{digest}'


def get_local_max_batch_item_metadata_bytes() -> int:
    raw = os.getenv(BATCH_ITEM_METADATA_BYTES_ENV, '').strip()
    if not raw:
        return MAX_BATCH_ITEM_METADATA_BYTES
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(f'{BATCH_ITEM_METADATA_BYTES_ENV} must be a positive integer') from exc
    if limit < 1:
        raise ValueError(f'{BATCH_ITEM_METADATA_BYTES_ENV} must be a positive integer')
    return limit


def canonical_batch_item_metadata(value: object, *, max_bytes: int | None = None) -> bytes:
    """Encode metadata; server callers pass their configured limit explicitly."""
    limit = get_local_max_batch_item_metadata_bytes() if max_bytes is None else max_bytes
    return _canonical_metadata(value, label='batch item metadata', max_bytes=limit)


def canonical_batch_metadata(value: object) -> bytes:
    return _canonical_metadata(value, label='batch metadata', max_bytes=MAX_BATCH_METADATA_BYTES)


def _canonical_metadata(value: object, *, label: str, max_bytes: int) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be JSON serializable') from exc
    if len(encoded) > max_bytes:
        raise ValueError(f'{label} exceeds size limit')
    return encoded


__all__ = [
    'BATCH_ITEM_METADATA_BYTES_ENV',
    'DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES',
    'MAX_BATCH_ITEM_ID_LENGTH',
    'MAX_BATCH_ITEM_METADATA_BYTES',
    'MAX_BATCH_METADATA_BYTES',
    'TaskListVersion',
    'canonical_batch_item_metadata',
    'canonical_batch_metadata',
    'get_local_max_batch_item_metadata_bytes',
    'is_valid_batch_item_id',
    'normalize_batch_item_id',
]
