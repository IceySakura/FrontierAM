from __future__ import annotations

import base64
import json
import os

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from nexus.distributed.access import taihu


_APPLICATION_TOKEN = 'a' * 32


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def _identity_jwe(
    *,
    staff_id: int | None = 123456,
    login_name: str = 'zhangsan',
    expiration: datetime | None = None,
    protected_header: dict[str, Any] | None = None,
    application_token: str = _APPLICATION_TOKEN,
) -> str:
    header = protected_header or {'alg': 'dir', 'enc': 'A256GCM'}
    protected = _base64url(json.dumps(header, separators=(',', ':')).encode())
    claims = {
        'LoginName': login_name,
        'Expiration': (expiration or datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }
    if staff_id is not None:
        claims['StaffId'] = staff_id
    payload = json.dumps(claims, separators=(',', ':')).encode()
    iv = os.urandom(12)
    encrypted = AESGCM(application_token.encode()).encrypt(iv, payload, protected.encode())
    return '.'.join(
        (
            protected,
            '',
            _base64url(iv),
            _base64url(encrypted[:-16]),
            _base64url(encrypted[-16:]),
        )
    )


def _tamper(value: str) -> str:
    return value[:-1] + ('A' if value[-1] != 'A' else 'B')


def test_decrypt_tai_identity_uses_staff_id_as_stable_subject() -> None:
    identity = taihu.decrypt_tai_identity(
        _identity_jwe(staff_id=123456, login_name='zhangsan'),
        application_token=_APPLICATION_TOKEN,
    )

    assert identity.issuer == 'tai'
    assert identity.tenant == 'default'
    assert identity.subject == '123456'
    assert identity.display_name == 'zhangsan'
    assert identity.principal_id == taihu.principal_id('tai', 'default', '123456')


@pytest.mark.parametrize(
    'protected_header',
    [
        {'alg': 'A256KW', 'enc': 'A256GCM'},
        {'alg': 'dir', 'enc': 'A128GCM'},
    ],
)
def test_decrypt_tai_identity_rejects_unapproved_jwe_algorithms(protected_header: dict[str, str]) -> None:
    with pytest.raises(PermissionError, match='invalid_tai_identity_algorithm'):
        taihu.decrypt_tai_identity(
            _identity_jwe(protected_header=protected_header),
            application_token=_APPLICATION_TOKEN,
        )


def test_decrypt_tai_identity_rejects_expired_ticket() -> None:
    expired = datetime.now(timezone.utc) - timedelta(minutes=4)

    with pytest.raises(PermissionError, match='expired_tai_identity'):
        taihu.decrypt_tai_identity(
            _identity_jwe(expiration=expired),
            application_token=_APPLICATION_TOKEN,
        )


def test_decrypt_tai_identity_rejects_invalid_application_token_length() -> None:
    with pytest.raises(PermissionError, match='invalid_tai_application_token'):
        taihu.decrypt_tai_identity(
            _identity_jwe(),
            application_token='too-short',
        )


@pytest.mark.parametrize(
    'raw_identity',
    [
        '@@@..AAAA.AAAA.AAAA',
        _tamper(_identity_jwe()),
    ],
)
def test_decrypt_tai_identity_rejects_malformed_or_tampered_jwe(raw_identity: str) -> None:
    with pytest.raises(PermissionError, match='invalid_tai_identity'):
        taihu.decrypt_tai_identity(raw_identity, application_token=_APPLICATION_TOKEN)


def test_decrypt_tai_identity_requires_staff_id() -> None:
    with pytest.raises(PermissionError, match='invalid_tai_identity_staff_id'):
        taihu.decrypt_tai_identity(
            _identity_jwe(staff_id=None),
            application_token=_APPLICATION_TOKEN,
        )
