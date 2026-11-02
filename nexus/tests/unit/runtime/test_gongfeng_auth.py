from pathlib import Path

import pytest

from nexus.runtime.providers import gongfeng as gongfeng_module
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


def _provider(token: str | None = None) -> GongfengRuntimeProvider:
    return GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token=token,
        print_port_mappings=False,
    )


@pytest.fixture
def gongfeng_key_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    key_file = tmp_path / 'gongfeng' / 'key'
    monkeypatch.setattr(gongfeng_module, 'GONGFENG_AUTH_KEY_FILE', key_file)
    monkeypatch.delenv('GONGFENG_TOKEN', raising=False)
    return key_file


def test_gongfeng_token_prefers_explicit_value(monkeypatch: pytest.MonkeyPatch, gongfeng_key_file: Path) -> None:
    gongfeng_key_file.parent.mkdir()
    gongfeng_key_file.write_text('file-token\n')
    monkeypatch.setenv('GONGFENG_TOKEN', 'environment-token')

    assert _provider('explicit-token').token == 'explicit-token'


def test_gongfeng_token_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
    gongfeng_key_file: Path,
) -> None:
    gongfeng_key_file.parent.mkdir()
    gongfeng_key_file.write_text('file-token\n')
    monkeypatch.setenv('GONGFENG_TOKEN', ' environment-token ')

    assert _provider().token == 'environment-token'


def test_gongfeng_token_falls_back_to_default_file(gongfeng_key_file: Path) -> None:
    gongfeng_key_file.parent.mkdir()
    gongfeng_key_file.write_text(' file-token\n')

    assert _provider().token == 'file-token'


def test_gongfeng_token_reports_all_supported_sources(gongfeng_key_file: Path) -> None:
    with pytest.raises(ValueError, match=r'GONGFENG_TOKEN.*gongfeng/key'):
        _provider()
