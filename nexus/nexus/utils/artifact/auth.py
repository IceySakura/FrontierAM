"""Authentication helpers for the artifact service.

The artifact service is fronted by the Taihu (太湖) unified-auth gateway. A
caller sends a personal access token as ``Authorization: Bearer <PAT>`` and the
gateway injects the authenticated employee identity before forwarding the
request.

A local or development backend without the gateway can instead accept the
plaintext ``X-Dev-Employee`` header when ``taihu.allow_dev_identity_header`` is
enabled. This development-only mechanism must never be enabled in production.
"""

from __future__ import annotations

import os

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr

from nexus.utils.paths import ARTIFACT_AUTH_KEY_FILE


# Taihu (太湖) unified-auth. A personal access token minted at
# ``https://tai.it.woa.com/user/pat`` is sent as ``Authorization: Bearer <PAT>``
# to the Taihu MCP gateway, which validates it and injects ``x-tai-identity``.
ENV_TAIHU_PAT = 'ARTIFACT_TAIHU_PAT'
# Optional override for the on-disk PAT location (default: ~/.nexus/auth/taihu).
ENV_TAIHU_PAT_FILE = 'ARTIFACT_TAIHU_PAT_FILE'
# Dev-only plaintext identity header, honoured by the backend only when
# ``taihu.allow_dev_identity_header`` is enabled (never in production).
DEV_IDENTITY_HEADER = 'X-Dev-Employee'
ENV_DEV_EMPLOYEE = 'ARTIFACT_DEV_EMPLOYEE'
# Verifiable service credential for the passwordless ``/system`` prefix. The
# distributed store reaches the artifact service through Taihu's no-auth
# ``/system`` routes (no employee identity header); to prove it is the system
# caller it presents the shared secret in this header, which the backend
# constant-time compares. The mere absence of an identity header is not a
# credential on its own.
SYSTEM_SECRET_HEADER = 'X-System-Secret'
ARTIFACT_SHARED_VALUE_NAME = 'artifact_credentials'


class ArtifactCredentials(BaseModel):
    """Artifact credentials kept in memory and redacted from representations."""

    model_config = ConfigDict(frozen=True)

    taihu_pat: SecretStr | None = None

    def to_shared_payload(self) -> dict[str, str]:
        if self.taihu_pat is None:
            return {}
        return {'taihu_pat': self.taihu_pat.get_secret_value()}

    @classmethod
    def from_shared_payload(cls, payload: Mapping[str, Any]) -> ArtifactCredentials:
        taihu_pat = payload.get('taihu_pat')
        if taihu_pat is None:
            return cls()
        if not isinstance(taihu_pat, str):
            raise ValueError('Artifact credentials taihu_pat must be a string')
        normalized = normalize_pat(taihu_pat)
        return cls(taihu_pat=SecretStr(normalized) if normalized else None)


def normalize_pat(pat: str) -> str:
    """Return the bare Taihu PAT, stripping a leading ``Bearer `` scheme (case-insensitive)."""

    pat = pat.strip()
    scheme, _, value = pat.partition(' ')
    if scheme.lower() == 'bearer' and value:
        return value.strip()
    return pat


def taihu_authorization_header(pat: str) -> str:
    """Build one normalized ``Authorization: Bearer <PAT>`` value."""

    return f'Bearer {normalize_pat(pat)}'


def taihu_pat_file(key_file: str | Path | None = None) -> Path:
    """Resolve the on-disk PAT location: explicit > ``$ARTIFACT_TAIHU_PAT_FILE`` > default."""

    path_value = key_file if key_file is not None else os.getenv(ENV_TAIHU_PAT_FILE)
    return Path(path_value).expanduser() if path_value else ARTIFACT_AUTH_KEY_FILE


def load_taihu_pat(
    pat: str | None = None,
    *,
    key_file: str | Path | None = None,
) -> str | None:
    """Resolve a Taihu PAT from an explicit value, the environment, or local file.

    Resolution order (first non-empty wins):

    1. the explicit ``pat`` argument;
    2. ``$ARTIFACT_TAIHU_PAT``;
    3. the on-disk key file (``$ARTIFACT_TAIHU_PAT_FILE`` or
       ``~/.nexus/auth/taihu``), so users can ``nexus artifact login``
       once instead of exporting the PAT for every command.

    Returns the normalized PAT, or ``None`` when no source provides one.
    """

    for candidate in (pat, os.getenv(ENV_TAIHU_PAT)):
        if candidate and candidate.strip():
            return normalize_pat(candidate)

    path = taihu_pat_file(key_file)
    try:
        file_pat = path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f'Failed to read Taihu PAT file {path}: {exc}') from exc
    return normalize_pat(file_pat) if file_pat else None


def load_artifact_credentials(
    pat: str | None = None,
    *,
    key_file: str | Path | None = None,
) -> ArtifactCredentials:
    """Load Host-side Artifact credentials for sharing with Runtime processes."""

    resolved_pat = load_taihu_pat(pat, key_file=key_file)
    return ArtifactCredentials(taihu_pat=SecretStr(resolved_pat) if resolved_pat is not None else None)


def save_taihu_pat(pat: str, *, key_file: str | Path | None = None) -> Path:
    """Persist a Taihu PAT to the local key file with owner-only permissions.

    The PAT is stored as normalized plaintext (no ``Bearer`` prefix) in
    ``~/.nexus/auth/taihu`` (or ``$ARTIFACT_TAIHU_PAT_FILE``). Returns the
    path written. Raises ``ValueError`` for an empty PAT.
    """

    normalized = normalize_pat(pat)
    if not normalized:
        raise ValueError('Taihu PAT must not be empty')
    path = taihu_pat_file(key_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create/truncate with 0600 so the token is never group/world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, normalized.encode('utf-8'))
    finally:
        os.close(fd)
    # Tighten permissions even if the file already existed with looser bits.
    os.chmod(path, 0o600)
    return path


def clear_taihu_pat(*, key_file: str | Path | None = None) -> bool:
    """Delete the local PAT file. Returns ``True`` if a file was removed."""

    path = taihu_pat_file(key_file)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
