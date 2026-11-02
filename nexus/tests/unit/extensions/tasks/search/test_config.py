"""Tests for nexus.extensions.tasks.search.config — SearchConfig = RunInput + benchmarks."""

from __future__ import annotations

import textwrap

from pathlib import Path

import pytest

from nexus.extensions.tasks.search.config import (
    SearchConfig,
    load_search_config,
    load_search_config_if_exists,
)


# ============================================================================
# Fixtures
# ============================================================================


def _write_yaml(tmp_path: Path, content: str, name: str = 'config.yaml') -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding='utf-8')
    return p


# ============================================================================
# Schema — SearchConfig is a RunInput subclass with an extra `benchmarks` field.
# ============================================================================


class TestSchema:
    def test_is_runinput_subclass(self) -> None:
        from nexus.runner.single import RunInput

        assert issubclass(SearchConfig, RunInput)

    def test_empty_yaml_defaults(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, '{task_name: search}')
        cfg = load_search_config(p)
        assert cfg.task_name == 'search'
        assert cfg.benchmarks == {}
        assert cfg.agent_config == {}
        assert cfg.task_config == {}
        assert cfg.openai_clients == {}
        assert cfg.tool_configs == {}

    def test_accepts_all_runinput_fields(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path,
            """\
            imports: [nexus.extensions]
            task_name: search
            agent_name: search_agent
            agent_config:
              max_iterations: 15
              toolset_version: v2
              system_prompt: "You are helpful. {finish_instruction}"
            task_config:
              eval_mode: llm_judge
              enable_summary: false
            tool_configs:
              google_serper_search:
                serper_url: http://example.com
            openai_clients:
              main:
                name: litellm
                config:
                  client_args: {base_url: http://api.example.com, api_key: xyz}
                  request_args: {model: gpt-4, temperature: 0.7}
            runner:
              timeout: 3600
            benchmarks:
              seal_0: /data/seal.parquet
              gaia: /data/gaia.parquet
            """,
        )
        cfg = load_search_config(p)

        assert cfg.task_name == 'search'
        assert cfg.agent_name == 'search_agent'
        assert cfg.agent_config['max_iterations'] == 15
        assert cfg.agent_config['toolset_version'] == 'v2'
        assert '{finish_instruction}' in cfg.agent_config['system_prompt']
        assert cfg.task_config['eval_mode'] == 'llm_judge'
        assert cfg.tool_configs['google_serper_search']['serper_url'] == 'http://example.com'
        assert cfg.openai_clients['main']['name'] == 'litellm'
        assert cfg.runner.timeout == 3600
        assert cfg.benchmarks == {'seal_0': '/data/seal.parquet', 'gaia': '/data/gaia.parquet'}


# ============================================================================
# OmegaConf env-var interpolation (inherited from RunInput.from_file).
# ============================================================================


class TestEnvInterpolation:
    def test_resolves_oc_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('MY_SERPER_KEY', 'secret-123')
        p = _write_yaml(
            tmp_path,
            """\
            task_name: search
            tool_configs:
              google_serper_search:
                serper_app_key: ${oc.env:MY_SERPER_KEY}
                serper_url: ${oc.env:MISSING_URL,http://default.example.com}
            """,
        )
        cfg = load_search_config(p)
        assert cfg.tool_configs['google_serper_search']['serper_app_key'] == 'secret-123'
        assert cfg.tool_configs['google_serper_search']['serper_url'] == 'http://default.example.com'

    def test_resolves_in_benchmarks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('MY_DATA_DIR', '/my/data')
        p = _write_yaml(
            tmp_path,
            """\
            task_name: search
            benchmarks:
              seal: ${oc.env:MY_DATA_DIR}/seal.parquet
              gaia: ${oc.env:OTHER_DIR,/fallback}/gaia.parquet
            """,
        )
        cfg = load_search_config(p)
        assert cfg.benchmarks == {'seal': '/my/data/seal.parquet', 'gaia': '/fallback/gaia.parquet'}

    def test_system_prompt_placeholders_preserved(self, tmp_path: Path) -> None:
        """`{today_date}` and `{finish_instruction}` are free-text fields — OmegaConf
        must not mangle them (they're resolved later in the generator per-row)."""
        p = _write_yaml(
            tmp_path,
            """\
            task_name: search
            agent_config:
              system_prompt: |
                Date: {today_date}. {finish_instruction}
            """,
        )
        cfg = load_search_config(p)
        assert '{today_date}' in cfg.agent_config['system_prompt']
        assert '{finish_instruction}' in cfg.agent_config['system_prompt']


# ============================================================================
# Error cases
# ============================================================================


class TestLoadErrors:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match='not found'):
            load_search_config(tmp_path / 'nonexistent.yaml')

    def test_non_mapping_root(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, '"just a string"')
        with pytest.raises(TypeError, match='mapping'):
            load_search_config(p)

    def test_invalid_benchmarks_type(self, tmp_path: Path) -> None:
        # benchmarks must be dict[str, str] — a list or scalar fails pydantic validation.
        p = _write_yaml(tmp_path, 'task_name: search\nbenchmarks: [a, b]\n')
        with pytest.raises(Exception):  # pydantic ValidationError
            load_search_config(p)


# ============================================================================
# load_search_config_if_exists
# ============================================================================


class TestLoadIfExists:
    def test_returns_none_when_no_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SEARCH_CONFIG_PATH', raising=False)
        assert load_search_config_if_exists() is None

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert load_search_config_if_exists(tmp_path / 'nope.yaml') is None

    def test_loads_from_explicit_path(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, 'task_name: search\nagent_config: {max_iterations: 3}\n')
        cfg = load_search_config_if_exists(p)
        assert cfg is not None
        assert cfg.agent_config == {'max_iterations': 3}

    def test_loads_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p = _write_yaml(tmp_path, 'task_name: search\nagent_config: {max_iterations: 7}\n')
        monkeypatch.setenv('SEARCH_CONFIG_PATH', str(p))
        cfg = load_search_config_if_exists()
        assert cfg is not None
        assert cfg.agent_config == {'max_iterations': 7}

    def test_explicit_path_overrides_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p_env = _write_yaml(tmp_path, 'task_name: search\nagent_config: {max_iterations: 1}\n', name='env.yaml')
        p_arg = _write_yaml(tmp_path, 'task_name: search\nagent_config: {max_iterations: 99}\n', name='arg.yaml')
        monkeypatch.setenv('SEARCH_CONFIG_PATH', str(p_env))
        cfg = load_search_config_if_exists(p_arg)
        assert cfg is not None
        assert cfg.agent_config == {'max_iterations': 99}


# ============================================================================
# Example YAML file — sanity check that the shipped template still parses.
# ============================================================================


class TestExampleFile:
    def test_example_file_is_parseable(self) -> None:
        project_root = Path(__file__).resolve().parent
        for _ in range(5):
            project_root = project_root.parent
        example_path = project_root / 'nexus' / 'extensions' / 'tasks' / 'search' / 'search_config.yaml.example'
        if not example_path.is_file():
            pytest.skip(f'Example file not found at {example_path}')

        cfg = load_search_config(example_path)
        assert cfg.task_name == 'search'
        assert cfg.benchmarks
        assert cfg.openai_clients
        assert cfg.agent_config


# ============================================================================
# Custom tools — SearchAgentInput supports custom_tools directly; no separate registry.
# ============================================================================


class TestCustomToolsIntegration:
    def test_custom_tools_via_agent_config(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path,
            """\
            task_name: search
            agent_config:
              custom_tools:
                - google_serper_search
                - base/finish
                - visit
            """,
        )
        cfg = load_search_config(p)
        assert cfg.agent_config['custom_tools'] == ['google_serper_search', 'base/finish', 'visit']

    def test_resolve_tools_by_names_strips_finish(self) -> None:
        from nexus.extensions.agents.search_agent.agent import _resolve_tools_by_names

        tools = _resolve_tools_by_names(['google_serper_search', 'base/finish', 'visit'], include_finish=False)
        assert len(tools) == 2

    def test_resolve_tools_by_names_raises_on_missing(self) -> None:
        from nexus.extensions.agents.search_agent.agent import _resolve_tools_by_names

        with pytest.raises(ValueError, match='not found in registry'):
            _resolve_tools_by_names(['completely_nonexistent_tool_xyz'])
