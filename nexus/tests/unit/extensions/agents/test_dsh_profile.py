from __future__ import annotations

import json

from pathlib import Path

import pytest
import yaml

from nexus.extensions.agents.dsh.profile import prepare_dsh_profile, validate_extra_env


def test_prepare_dsh_profile_builds_acp_only_composition(tmp_path: Path) -> None:
    runtime_installation = tmp_path / 'runtime'
    (runtime_installation / 'node_modules').mkdir(parents=True)
    profile = prepare_dsh_profile(
        install_dir=str(tmp_path),
        runtime_installation=runtime_installation,
        session_id='session-1',
        acp_version='0.1.0-rc.6',
        proxy_base_url='http://127.0.0.1:1234/v1/',
        context_window=100_000,
        max_output_tokens=8_000,
    )

    package = json.loads((profile.profile_dir / 'package.json').read_text())
    patch = yaml.safe_load((profile.profile_dir / 'cordis.patch.yml').read_text())
    by_id = {row['id']: row for row in patch if 'id' in row}

    assert package['dsh']['profile']['bundles'] == ['@deepseek-ai/dsh-base']
    assert package['dependencies']['@deepseek-ai/dsh-acp'] == '0.1.0-rc.6'
    assert (profile.profile_dir / 'node_modules').resolve() == (runtime_installation / 'node_modules').resolve()
    assert by_id['llm-pi-ai']['config']['providers']['nexus']['baseURL'] == 'http://127.0.0.1:1234/v1'
    assert by_id['llm-pi-ai']['config']['providers']['nexus']['models'][0]['contextWindow'] == 100_000
    assert by_id['session-persistence-jsonl']['config'] == {
        'root': str(profile.session_root),
        'compression': 'none',
        'packChunks': False,
    }
    assert by_id['llm-deepseek']['disabled'] is True
    assert by_id['plan-mode']['disabled'] is True
    assert by_id['tool-str-replace-editor']['disabled'] is True
    assert by_id['tool-subagent']['disabled'] is True
    assert patch[-1]['insert'][0]['name'] == '@deepseek-ai/dsh-acp'


@pytest.mark.parametrize('key', ['DSH_HOME', 'DSH_PERMISSION_MODE', 'NEXUS_PROXY_API_KEY', 'PATH'])
def test_validate_extra_env_rejects_managed_variables(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        validate_extra_env({key: 'override'})


def test_validate_extra_env_allows_unrelated_variables() -> None:
    validate_extra_env({'MY_TOOL_TOKEN': 'secret'})


def test_prepare_dsh_profile_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Invalid DSH session id'):
        prepare_dsh_profile(
            install_dir=str(tmp_path),
            runtime_installation=tmp_path,
            session_id='../escape',
            acp_version='0.1.0-rc.6',
            proxy_base_url='http://127.0.0.1:1234/v1',
            context_window=100_000,
            max_output_tokens=8_000,
        )
