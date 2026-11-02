import json

from pathlib import Path

import pytest

from nexus.extensions.tasks.swe import generator
from nexus.runner.single import RunInput


def _write_instance(path: Path) -> None:
    path.write_text(
        json.dumps({'instance_id': 'fixture__repo-1', 'dataset_type': 'swe-bench', 'repo': 'fixture/repo'}) + '\n',
        encoding='utf-8',
    )


def test_generate_interactive_claude_code_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / 'instances.jsonl'
    _write_instance(dataset_path)

    monkeypatch.delenv('SWE_DATASET', raising=False)
    monkeypatch.delenv('SWE_PTY_TOOL_SEARCH', raising=False)
    monkeypatch.setenv('SWE_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SWE_AGENT_NAME', 'claude_code_interactive')
    monkeypatch.setenv('SWE_WORKER_VENV', '1')
    monkeypatch.setenv('CLAUDE_CODE_VERSION', '2.1.153')
    monkeypatch.setenv('OPENAI_MODEL', 'anthropic.claude-opus-4-8')
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-model-secret')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://fixture.invalid/v1')
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'fixture-sandbox-secret')

    item = next(generator.generate_configs())
    config = item.config.model_dump()
    agent_config = config['agent_config']

    assert item.run_id == 'fixture__repo-1'
    assert config['agent_name'] == 'claude_code_interactive'
    assert 'nexus.extensions.agents.claude_code' in config['imports']
    assert config['runtime_configs'] == {
        'main': {'worker_venv': True},
        'eval': {'worker_venv': True},
    }
    assert agent_config['version'] == '2.1.153'
    assert 'interaction_mode' not in agent_config
    assert agent_config['extra_env']['ANTHROPIC_MODEL'] == 'anthropic.claude-opus-4-8'
    assert agent_config['extra_env']['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] == '1'
    assert 'ENABLE_TOOL_SEARCH' not in agent_config['extra_env']
    assert config['runtime_provider_config']['token'] == '${oc.env:GONGFENG_RUNTIME_TOKEN,""}'
    assert config['openai_clients']['main']['config']['client_args'] == {
        'api_key': '${oc.env:OPENAI_API_KEY,""}',
        'base_url': '${oc.env:OPENAI_BASE_URL,""}',
    }
    assert config['openai_clients']['main']['config']['request_args']['model'] == '${oc.env:OPENAI_MODEL,""}'
    persisted_config = json.dumps(config)
    assert 'fixture-model-secret' not in persisted_config
    assert 'https://fixture.invalid/v1' not in persisted_config
    assert 'fixture-sandbox-secret' not in persisted_config

    resolved = RunInput.from_dict(config).model_dump()
    assert resolved['runtime_provider_config']['token'] == 'fixture-sandbox-secret'
    assert resolved['openai_clients']['main']['config']['client_args'] == {
        'api_key': 'fixture-model-secret',
        'base_url': 'https://fixture.invalid/v1',
    }

    metadata = generator.get_batch_metadata()
    assert metadata['agent'] == 'claude_code_interactive'
    assert metadata['claude_code_version'] == '2.1.153'
    assert metadata['model'] == 'anthropic.claude-opus-4-8'
    assert metadata['tool_search'] is False
    assert 'interaction_mode' not in metadata


def test_interactive_claude_code_tool_search_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SWE_PTY_TOOL_SEARCH', '1')
    monkeypatch.setenv('SWE_PTY_TOOL_SEARCH_MODE', 'auto:5')
    config = generator._build_agent_config('claude_code_interactive')

    assert config['extra_env']['ENABLE_TOOL_SEARCH'] == 'auto:5'
    assert 'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS' not in config['extra_env']
    assert 'interaction_mode' not in config
