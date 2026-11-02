"""Purpose-separated HMAC cursors and stable non-authorizing stream identities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time

from collections.abc import Callable, Mapping
from typing import Any

from nexus.distributed.trace.models import AuthorizedTraceTarget, TraceReadError, json_bytes


_ENCODED = re.compile(r'^[A-Za-z0-9_-]+$')


def _pack(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def _unpack(value: str) -> bytes:
    if not _ENCODED.fullmatch(value):
        raise ValueError('invalid encoding')
    data = base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
    if _pack(data) != value:
        raise ValueError('noncanonical encoding')
    return data


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('duplicate cursor field')
    return result


def _load(value: bytes) -> dict[str, Any]:
    data = json.loads(value, object_pairs_hook=_object)
    if not isinstance(data, dict):
        raise ValueError('invalid cursor object')
    return data


def target_scope(target: AuthorizedTraceTarget) -> dict[str, str]:
    return {
        'principal': target.principal_id,
        'owner': target.owner_principal_id,
        'environment': target.environment_id,
        'execution': target.execution_id,
    }


class TraceCursorCodec:
    def __init__(
        self,
        keys: Mapping[str, str],
        *,
        current_key_id: str,
        ttl_seconds: int = 86400,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if current_key_id not in keys or any(len(value.encode()) < 32 for value in keys.values()):
            raise TraceReadError('trace_source_misconfigured', 503)
        if any(not _ENCODED.fullmatch(key) or len(key) > 64 for key in keys):
            raise TraceReadError('trace_source_misconfigured', 503)
        self.keys = {key: value.encode() for key, value in keys.items()}
        self.current_key_id = current_key_id
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def issue(self, scope: Mapping[str, Any], *, purpose: str, position: str) -> str:
        now = int(self.clock())
        payload = {**scope, 'v': 1, 'purpose': purpose, 'position': position, 'iat': now, 'exp': now + self.ttl_seconds}
        body = f'tc1.{self.current_key_id}.{_pack(json_bytes(payload))}'
        signature = _pack(hmac.digest(self.keys[self.current_key_id], body.encode(), hashlib.sha256))
        token = f'{body}.{signature}'
        if len(token) > 4096:
            raise TraceReadError('invalid_trace_cursor', 400)
        return token

    def decode(self, token: str, *, purpose: str, target: AuthorizedTraceTarget) -> dict[str, Any]:
        try:
            if not isinstance(token, str) or not 1 <= len(token) <= 4096:
                raise ValueError('invalid length')
            version, kid, encoded, signature = token.split('.')
            if version != 'tc1' or kid not in self.keys:
                raise ValueError('unknown key or version')
            expected = hmac.digest(self.keys[kid], f'{version}.{kid}.{encoded}'.encode(), hashlib.sha256)
            if not hmac.compare_digest(expected, _unpack(signature)):
                raise ValueError('invalid signature')
            data = _load(_unpack(encoded))
            fields = {
                'v',
                'purpose',
                'position',
                'iat',
                'exp',
                'backend',
                'source_id',
                'source_revision',
                *target_scope(target),
            }
            if purpose == 'events':
                fields.add('span')
            # Only Boundless datasets carry a partition key, so it stays optional here and
            # cursors minted before the column existed keep decoding.
            optional = {'parent_created_at'} & set(data)
            if (
                set(data) - optional != fields
                or (type(data['v']) is not int or data['v'] != 1)
                or data['purpose'] != purpose
            ):
                raise ValueError('invalid purpose or schema')
            if any(data[key] != value for key, value in target_scope(target).items()):
                raise ValueError('invalid scope')
            for key in ('iat', 'exp'):
                if type(data[key]) is not int:
                    raise ValueError('invalid time')
            now = int(self.clock())
            if data['iat'] > now + 60 or data['exp'] <= data['iat']:
                raise ValueError('invalid time range')
            if data['exp'] <= now:
                raise TraceReadError('trace_cursor_expired', 410)
            string_keys = (
                {'position', 'backend', 'source_id', 'source_revision'}
                | ({'span'} if purpose == 'events' else set())
                | optional
            )
            if any(not isinstance(data[key], str) or len(data[key]) > 128 for key in string_keys):
                raise ValueError('invalid identity')
            if purpose == 'events' and (not data['position'].isascii() or not data['position'].isdigit()):
                raise ValueError('invalid event position')
            return data
        except TraceReadError:
            raise
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            raise TraceReadError('invalid_trace_cursor', 400) from None


def stream_id(*, backend: str, source_id: str, source_revision: str, run_id: str, run_span_id: str) -> str:
    return 'ts1.' + _pack(
        json_bytes({
            'v': 1,
            'backend': backend,
            'source_id': source_id,
            'source_revision': source_revision,
            'run_id': run_id,
            'span': run_span_id,
        })
    )


def decode_stream_id(value: str, *, execution_id: str) -> dict[str, Any]:
    try:
        if len(value) > 1024 or not value.startswith('ts1.'):
            raise ValueError('invalid stream')
        data = _load(_unpack(value[4:]))
        if set(data) != {'v', 'backend', 'source_id', 'source_revision', 'run_id', 'span'} or (
            type(data['v']) is not int or data['v'] != 1
        ):
            raise ValueError('invalid stream schema')
        if data['run_id'] != execution_id:
            raise ValueError('stream execution mismatch')
        if any(not isinstance(v, str) or not 1 <= len(v) <= 128 for k, v in data.items() if k != 'v'):
            raise ValueError('invalid stream identity')
        return data
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise TraceReadError('invalid_trace_cursor', 400) from None
