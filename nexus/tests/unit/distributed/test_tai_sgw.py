from __future__ import annotations

import hashlib

from datetime import datetime, timedelta, timezone

import pytest

from multidict import CIMultiDict

from nexus.distributed.access import taihu
from nexus.distributed.access.taihu import verify_sgw_backend_signature


_APPLICATION_TOKEN = 'a' * 32
_NOW = datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)


def _headers(
    *,
    app_id: str = 'mcp-l2m93f',
    timestamp: datetime = _NOW,
    sequence: str = 'request-sequence-1',
) -> dict[str, str]:
    raw_timestamp = str(int(timestamp.timestamp()))
    signature = (
        hashlib.sha256(f'{raw_timestamp}{_APPLICATION_TOKEN}{sequence}{raw_timestamp}'.encode()).hexdigest().upper()
    )
    return {
        'x-rio-paasid': app_id,
        'timestamp': raw_timestamp,
        'x-rio-seq': sequence,
        'signature': signature,
    }


def test_verify_sgw_backend_signature_returns_caller_app_id() -> None:
    assert (
        verify_sgw_backend_signature(
            _headers(app_id='MCP-L2M93F'),
            application_token=_APPLICATION_TOKEN,
            now=_NOW,
        )
        == 'MCP-L2M93F'
    )


def test_sgw_auth_header_counts_reports_names_without_values() -> None:
    headers = CIMultiDict(
        {
            'x-rio-paasid': 'mcp-l2m93f',
            'x-rio-timestamp': 'caller-timestamp',
            'x-rio-nonce': 'caller-nonce',
            'x-rio-signature': 'caller-signature',
            'x-rio-seq': 'backend-sequence-1',
        }
    )
    headers.add('X-Rio-Seq', 'backend-sequence-2')

    assert taihu.sgw_auth_header_counts(headers) == {
        'signature': 0,
        'timestamp': 0,
        'x-rio-nonce': 1,
        'x-rio-paasid': 1,
        'x-rio-seq': 2,
        'x-rio-signature': 1,
        'x-rio-timestamp': 1,
    }


def test_trusted_sgw_app_id_requires_one_non_empty_header() -> None:
    assert taihu.trusted_sgw_app_id({'x-rio-paasid': ' mcp-l2m93f '}) == 'mcp-l2m93f'

    duplicate = CIMultiDict([('x-rio-paasid', 'first'), ('X-Rio-Paasid', 'second')])
    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        taihu.trusted_sgw_app_id(duplicate)
    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        taihu.trusted_sgw_app_id({'x-rio-paasid': ' '})


def test_has_sgw_backend_signature_headers_ignores_forwarded_sequence() -> None:
    assert taihu.has_sgw_backend_signature_headers({'timestamp': '1'}) is True
    assert taihu.has_sgw_backend_signature_headers({'signature': 'A' * 64}) is True
    assert taihu.has_sgw_backend_signature_headers({'x-rio-seq': 'forwarded-sequence'}) is False
    assert taihu.has_sgw_backend_signature_headers({'x-rio-paasid': 'mcp-l2m93f'}) is False


@pytest.mark.parametrize('offset_seconds', [-181, 181])
def test_verify_sgw_backend_signature_rejects_stale_timestamp(offset_seconds: int) -> None:
    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        verify_sgw_backend_signature(
            _headers(timestamp=_NOW + timedelta(seconds=offset_seconds)),
            application_token=_APPLICATION_TOKEN,
            now=_NOW,
        )


@pytest.mark.parametrize('missing_header', ['x-rio-paasid', 'timestamp', 'x-rio-seq', 'signature'])
def test_verify_sgw_backend_signature_requires_all_headers(missing_header: str) -> None:
    headers = _headers()
    headers.pop(missing_header)

    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        verify_sgw_backend_signature(headers, application_token=_APPLICATION_TOKEN, now=_NOW)


def test_verify_sgw_backend_signature_rejects_invalid_signature() -> None:
    headers = _headers()
    headers['signature'] = '0' * 64

    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        verify_sgw_backend_signature(headers, application_token=_APPLICATION_TOKEN, now=_NOW)


def test_verify_sgw_backend_signature_rejects_lowercase_signature() -> None:
    headers = _headers()
    headers['signature'] = headers['signature'].lower()

    with pytest.raises(PermissionError, match='invalid_tai_app_identity'):
        verify_sgw_backend_signature(headers, application_token=_APPLICATION_TOKEN, now=_NOW)


def test_verify_sgw_backend_signature_requires_timezone_aware_now() -> None:
    with pytest.raises(ValueError, match='now must be timezone-aware'):
        verify_sgw_backend_signature(
            _headers(),
            application_token=_APPLICATION_TOKEN,
            now=_NOW.replace(tzinfo=None),
        )
