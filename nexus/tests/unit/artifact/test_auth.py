"""Unit tests for :mod:`nexus.utils.artifact.auth`."""

import stat

import pytest

from nexus.utils.artifact import auth as auth_module
from nexus.utils.artifact.auth import (
    ARTIFACT_SHARED_VALUE_NAME,
    ENV_TAIHU_PAT,
    ENV_TAIHU_PAT_FILE,
    ArtifactCredentials,
    clear_taihu_pat,
    load_artifact_credentials,
    load_taihu_pat,
    normalize_pat,
    save_taihu_pat,
    taihu_authorization_header,
    taihu_pat_file,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_TAIHU_PAT, raising=False)
    monkeypatch.delenv(ENV_TAIHU_PAT_FILE, raising=False)


def test_artifact_credentials_round_trip_normalizes_and_redacts_pat():
    credentials = ArtifactCredentials.from_shared_payload({'taihu_pat': 'Bearer tai_pat_shared'})

    assert ARTIFACT_SHARED_VALUE_NAME == 'artifact_credentials'
    assert credentials.to_shared_payload() == {'taihu_pat': 'tai_pat_shared'}
    assert 'tai_pat_shared' not in repr(credentials)


def test_artifact_package_exports_runtime_shared_credentials():
    from nexus.utils import artifact

    assert {'ARTIFACT_SHARED_VALUE_NAME', 'ArtifactCredentials'} <= set(artifact.__all__)


def test_load_artifact_credentials_supports_missing_pat(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', tmp_path / 'no-such-key')

    credentials = load_artifact_credentials()

    assert credentials.to_shared_payload() == {}


def test_normalize_pat_strips_bearer_scheme():
    assert normalize_pat('Bearer tai_pat_xyz') == 'tai_pat_xyz'
    assert normalize_pat('bearer tai_pat_xyz') == 'tai_pat_xyz'
    assert normalize_pat('  tai_pat_raw  ') == 'tai_pat_raw'


def test_taihu_authorization_header_adds_single_bearer():
    assert taihu_authorization_header('tai_pat_xyz') == 'Bearer tai_pat_xyz'
    assert taihu_authorization_header('Bearer tai_pat_xyz') == 'Bearer tai_pat_xyz'


def test_taihu_pat_file_prefers_explicit_then_env(monkeypatch, tmp_path):
    explicit = tmp_path / 'explicit-key'
    assert taihu_pat_file(explicit) == explicit

    env_path = tmp_path / 'env-key'
    monkeypatch.setenv(ENV_TAIHU_PAT_FILE, str(env_path))
    assert taihu_pat_file() == env_path


def test_load_taihu_pat_resolution_order(monkeypatch, tmp_path):
    key_file = tmp_path / 'key'
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)

    # Nothing configured -> None.
    assert load_taihu_pat() is None

    # File is the lowest-priority source and is normalized on read.
    key_file.write_text('Bearer tai_pat_file', encoding='utf-8')
    assert load_taihu_pat() == 'tai_pat_file'

    # Environment beats the file.
    monkeypatch.setenv(ENV_TAIHU_PAT, 'tai_pat_env')
    assert load_taihu_pat() == 'tai_pat_env'

    # Explicit argument beats everything.
    assert load_taihu_pat('Bearer tai_pat_explicit') == 'tai_pat_explicit'


def test_save_taihu_pat_writes_normalized_with_owner_only_permissions(tmp_path):
    key_file = tmp_path / 'nested' / 'key'
    path = save_taihu_pat('Bearer tai_pat_saved', key_file=key_file)

    assert path == key_file
    assert key_file.read_text(encoding='utf-8') == 'tai_pat_saved'
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_save_then_load_round_trip(tmp_path):
    key_file = tmp_path / 'key'
    save_taihu_pat('tai_pat_roundtrip', key_file=key_file)
    assert load_taihu_pat(key_file=key_file) == 'tai_pat_roundtrip'


def test_save_taihu_pat_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match='must not be empty'):
        save_taihu_pat('   ', key_file=tmp_path / 'key')


def test_save_taihu_pat_overwrites_and_tightens_permissions(tmp_path):
    key_file = tmp_path / 'key'
    key_file.write_text('old', encoding='utf-8')
    key_file.chmod(0o644)

    save_taihu_pat('tai_pat_new', key_file=key_file)

    assert key_file.read_text(encoding='utf-8') == 'tai_pat_new'
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_clear_taihu_pat(tmp_path):
    key_file = tmp_path / 'key'
    assert clear_taihu_pat(key_file=key_file) is False

    save_taihu_pat('tai_pat_x', key_file=key_file)
    assert clear_taihu_pat(key_file=key_file) is True
    assert not key_file.exists()
