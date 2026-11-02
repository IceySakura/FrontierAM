from __future__ import annotations

import base64
import binascii
import re


_NEXUS_SPAN_ID = re.compile(r'[A-Za-z0-9_-]{11}')
_OTEL_SPAN_ID = re.compile(r'[0-9A-Fa-f]{16}')
_OTEL_TRACE_ID = re.compile(r'[0-9A-Fa-f]{32}')


def nexus_span_id_to_otel_hex(nexus_span_id: str) -> str:
    if _NEXUS_SPAN_ID.fullmatch(nexus_span_id) is None:
        raise ValueError('nexus span_id must be canonical 11-char base64url encoding 8 bytes')
    padded = nexus_span_id + '=' * (-len(nexus_span_id) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b'-_', validate=True)
    except (binascii.Error, ValueError):
        raise ValueError('nexus span_id must decode to 8 bytes') from None
    if len(raw) != 8:
        raise ValueError(f'nexus span_id must decode to 8 bytes, got {len(raw)}')
    if base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii') != nexus_span_id:
        raise ValueError('nexus span_id must be canonical 11-char base64url encoding 8 bytes')
    return raw.hex()


def otel_hex_to_nexus_span_id(otel_span_id: str) -> str:
    if _OTEL_SPAN_ID.fullmatch(otel_span_id) is None:
        raise ValueError(f'otel span_id must be 16 hex chars, got {len(otel_span_id)}')
    raw = bytes.fromhex(otel_span_id)
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def otel_hex_to_nexus_trace_id(otel_trace_id: str) -> str:
    if _OTEL_TRACE_ID.fullmatch(otel_trace_id) is None:
        raise ValueError(f'otel trace_id must be 32 hex chars, got {len(otel_trace_id)}')
    raw = bytes.fromhex(otel_trace_id)
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')
