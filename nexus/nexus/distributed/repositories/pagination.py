from __future__ import annotations

import base64
import binascii
import json

from collections.abc import Mapping

from nexus.distributed.errors import DomainValidationError


def encode_cursor(payload: Mapping[str, str]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=')


def decode_cursor(value: str, *, fields: frozenset[str]) -> dict[str, str]:
    if not value or len(value) > 4096:
        raise DomainValidationError('invalid_cursor')
    try:
        padding = '=' * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b'-_', validate=True)
        payload = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DomainValidationError('invalid_cursor') from exc
    if not isinstance(payload, dict) or set(payload) != fields:
        raise DomainValidationError('invalid_cursor')
    if any(not isinstance(item, str) or not item for item in payload.values()):
        raise DomainValidationError('invalid_cursor')
    return payload


__all__ = ['decode_cursor', 'encode_cursor']
