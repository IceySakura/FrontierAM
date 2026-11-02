import base64
import json

from pathlib import Path

import pytest

from nexus.utils.container_registry_auth import (
    RegistryConfigurationError,
    load_registry_credentials,
    registry_from_shared_value_name,
    registry_shared_value_name,
)


def test_registry_credentials_load_from_docker_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('REGISTRY_USERNAME', raising=False)
    monkeypatch.delenv('REGISTRY_PASSWORD', raising=False)
    config = tmp_path / 'config.json'
    auth = base64.b64encode(b'test-user:test-password').decode()
    config.write_text(json.dumps({'auths': {'mirrors.tencent.com': {'auth': auth}}}))

    credentials = load_registry_credentials(docker_config=config)

    assert credentials.to_shared_payload() == {
        'registry': 'mirrors.tencent.com',
        'username': 'test-user',
        'password': 'test-password',
    }


def test_registry_credentials_prefer_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('REGISTRY_USERNAME', 'environment-user')
    monkeypatch.setenv('REGISTRY_PASSWORD', 'environment-password')

    credentials = load_registry_credentials(docker_config=tmp_path / 'missing.json')

    assert credentials.to_shared_payload()['username'] == 'environment-user'


def test_registry_credentials_reject_partial_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('REGISTRY_USERNAME', 'only-user')
    monkeypatch.delenv('REGISTRY_PASSWORD', raising=False)

    with pytest.raises(RegistryConfigurationError, match='must be set together'):
        load_registry_credentials()


def test_registry_shared_value_name_round_trip() -> None:
    name = registry_shared_value_name('mirrors.tencent.com')

    assert registry_from_shared_value_name(name) == 'mirrors.tencent.com'
    assert registry_from_shared_value_name('other') is None
