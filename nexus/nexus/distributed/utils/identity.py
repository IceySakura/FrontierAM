from __future__ import annotations

import hashlib
import json
import re
import uuid

from typing import Any

from nexus.distributed.errors import DomainValidationError


ENVIRONMENT_ID_MAX_LENGTH = 64
POOL_ID_MAX_LENGTH = 64
PRINCIPAL_ID_MAX_LENGTH = 64
SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH = 128
IDENTITY_ISSUER_MAX_LENGTH = 64
IDENTITY_TENANT_MAX_LENGTH = 128
IDENTITY_SUBJECT_MAX_LENGTH = 256
IDENTITY_DISPLAY_NAME_MAX_LENGTH = 256
RUN_TRACE_BACKEND_NAME_MAX_LENGTH = 64

_SUBMISSION_IDEMPOTENCY_KEY_PATTERN = re.compile(r'[A-Za-z0-9._:-]+')
_POOL_ID_PATTERN = re.compile(r'[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?')


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex}'


def stable_shard(value: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError('shard_count must be positive')
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % shard_count


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validate_submission_idempotency_key(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH
        or _SUBMISSION_IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None
    ):
        raise DomainValidationError('invalid_submission_idempotency_key')
    return value


def validate_environment_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > ENVIRONMENT_ID_MAX_LENGTH:
        raise DomainValidationError('invalid_environment_id')
    return value


def validate_pool_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > POOL_ID_MAX_LENGTH
        or _POOL_ID_PATTERN.fullmatch(value) is None
    ):
        raise DomainValidationError('invalid_pool_id')
    return value


def validate_run_trace_backend_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > RUN_TRACE_BACKEND_NAME_MAX_LENGTH
        or not all(character.isalnum() or character in {'-', '_'} for character in value)
    ):
        raise ValueError(f'invalid run trace backend name: {value!r}')
    return value
