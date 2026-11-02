"""Hyness Harness profiles use the shared three-axis loader and generators."""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

import pytest

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from nexus.extensions.agents.hyness import HynessInput
from nexus.extensions.configs import generator_swe_template, validate
from nexus.extensions.configs.load_config import SHARED_DIR, load_config


PROFILES = {
    'hyness_opencc': 'opencc',
    'hyness_mini_swe_agent': 'mini-swe-agent',
    'hyness_swe_agent': 'swe-agent',
    'hyness_workbuddy': 'workbuddy',
}

APP_PROFILES = {
    'hyness_mini_swe_agent': 'mini-swe-agent',
    'hyness_swe_agent': 'swe-agent',
    'hyness_workbuddy': 'workbuddy',
}


@pytest.fixture
def select(monkeypatch: pytest.MonkeyPatch):
    def configure(profile: str, llm: str = 'hy3.openai', **overrides: str) -> None:
        for key in list(os.environ):
            if key.startswith(('HYNESS_', 'AGENT_', 'LANGFUSE_')) or key in {
                'RUNTIME_CONFIG',
                'LLM_CONFIG',
                'SWE_LIMIT',
                'SWE_SKIP',
                'RUN_ORACLE',
                'INCLUDE_INSTANCES',
                'EXCLUDE_INSTANCES',
            }:
                monkeypatch.delenv(key, raising=False)
        for key, value in {
            'AGENT_CONFIG': profile,
            'LLM_CONFIG': llm,
            'HR_API_KEY': 'fixture-key',
            'GONGFENG_RUNTIME_TOKEN': 'fixture-token',
            **overrides,
        }.items():
            monkeypatch.setenv(key, value)

    return configure


def resolved() -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.create(load_config()), resolve=True)
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def agent_input(config: dict[str, Any]) -> HynessInput:
    return HynessInput.model_validate({'user_prompt': 'fixture task', **config['agent_config']})


@pytest.mark.parametrize(('profile', 'app'), APP_PROFILES.items())
def test_app_profiles_pin_published_packages_and_native_schema(select, profile, app):
    select(profile)
    selected = agent_input(resolved())
    app_revision = '12' if app == 'workbuddy' else '7'
    engine_revision = '8' if app == 'workbuddy' else '7'
    assert selected.packages == [
        f'@tencent/hyness@0.0.1-scaffold-apps.20261021.{engine_revision}',
        f'@tencent/hyness-{app}@0.1.0-scaffold-apps.20261021.{app_revision}',
    ]
    assert selected.app_config == f'@tencent/hyness-{app}/apps/{app}/app.yaml'
    assert selected.harness_id == app
    assert selected.preserve_tool_schema
    assert selected.session_persistence is (app == 'workbuddy')
    assert 'ANTHROPIC_CONTEXT_WINDOW' not in (selected.extra_env or {})
    assert not any(key.startswith('HYNESS_WORKBUDDY_') for key in selected.extra_env or {})
    assert validate.check_file(SHARED_DIR / 'agents' / f'{profile}.yaml', []) == []


@pytest.mark.parametrize('profile', APP_PROFILES)
def test_app_profiles_allow_explicit_candidate_package_overrides(select, profile):
    packages = ['/opt/verified-app-packages/engine.tgz', '/opt/verified-app-packages/app.tgz']
    select(profile, HYNESS_PACKAGES=json.dumps(packages))
    assert agent_input(resolved()).packages == packages


@pytest.mark.parametrize('profile', PROFILES)
def test_hyness_profile_passes_shared_schema(profile: str) -> None:
    skips: list[str] = []
    assert validate.check_file(SHARED_DIR / 'agents' / f'{profile}.yaml', skips) == []
    assert skips == []


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize(
    ('llm', 'thinking'),
    [('hy3.openai', 'hunyuan'), ('claude-opus-48.openai', 'bedrock-claude'), ('gpt-6-sol.openai_responses', 'hunyuan')],
)
def test_shared_loader_combines_harness_and_model_without_unread_fields(select, profile, llm, thinking):
    select(profile, llm)
    config = resolved()
    selected = agent_input(config)
    assert config['agent_name'] == 'hyness'
    assert selected.harness_id == PROFILES[profile]
    assert selected.thinking_preset == thinking
    assert 'llm' not in config
    assert 'agent_config_by_llm' not in config
    assert len(config['imports']) == len(set(config['imports']))
    if profile == 'hyness_opencc':
        assert len(selected.packages) == 2
        assert selected.app_config_overlay == '@tencent/hyness-opencc/apps/opencc/overlays/training.yaml'
    elif profile in APP_PROFILES:
        assert len(selected.packages) == 2
        assert selected.app_config == f'@tencent/hyness-{APP_PROFILES[profile]}/apps/{APP_PROFILES[profile]}/app.yaml'
        assert selected.app_config_overlay is None
    if llm == 'hy3.openai':
        assert selected.no_thinking_request_args is not None
        assert selected.extra_env is not None
        assert selected.no_thinking_request_args['model'] == 'hy3'
        assert selected.extra_env['ANTHROPIC_MODEL'] == 'claude-opus-4-8[1m]'
    elif llm == 'claude-opus-48.openai':
        assert selected.no_thinking_request_args is not None
        assert selected.no_thinking_request_args['model'] == 'claude-opus-4-8'
    else:
        assert selected.no_thinking_request_args is None


def test_explicit_null_survives_shared_merge_and_disables_opencc_overlay(select):
    # YAML null means deletion in run-part merges; decode it after merging instead.
    select('hyness_opencc', HYNESS_APP_CONFIG_OVERLAY='null', HYNESS_HARNESS_ID='null')
    config = resolved()
    assert config['agent_config']['app_config_overlay'] is None
    selected = agent_input(config)
    assert selected.app_config_overlay is None
    assert selected.harness_id is None


def test_custom_package_app_and_policy_use_existing_env_overrides(select):
    packages = ['@tencent/hyness@1.2.3', '@example/custom-app@2.0.0']
    select(
        'hyness',
        HYNESS_PACKAGES=json.dumps(packages),
        HYNESS_APP_CONFIG='@example/custom-app/apps/custom/app.yaml',
        HYNESS_APP_CONFIG_OVERLAY='/opt/policies/rollout.yaml',
        HYNESS_HARNESS_ID='custom',
        AGENT_TIMEOUT='700',
        AGENT_MAX_TURNS='20',
    )
    selected = agent_input(resolved())
    assert selected.packages == packages
    assert selected.app_config == '@example/custom-app/apps/custom/app.yaml'
    assert selected.app_config_overlay == '/opt/policies/rollout.yaml'
    assert selected.harness_id == 'custom'
    assert selected.timeout == 700
    assert selected.max_turns == 20


def test_opencc_profile_retains_full_swe_deny_boundary(select):
    select('hyness_opencc')
    selected = agent_input(resolved())
    arguments = selected.extra_cli_args
    assert arguments[arguments.index('--disallowedTools') + 1].split(',') == [
        'WebSearch',
        'WebFetch',
        'NotebookEdit',
        'AskUserQuestion',
        'EnterPlanMode',
        'ExitPlanMode',
    ]


def test_workbuddy_profile_prepares_bundled_native_tools_without_asset_env(select):
    select('hyness_workbuddy')
    selected = agent_input(resolved())
    assert selected.extra_env is not None
    assert not any(key.startswith('HYNESS_WORKBUDDY_') for key in selected.extra_env)
    assert selected.pre_start_scripts is not None
    assert len(selected.pre_start_scripts) == 1
    assert 'workbuddy-app-plugin/prepare.mjs' in selected.pre_start_scripts[0].code
    assert '--build-native' in selected.pre_start_scripts[0].code
    assert selected.session_persistence is True


@pytest.mark.parametrize('profile', ['hyness_opencc', 'hyness_mini_swe_agent', 'hyness_swe_agent'])
def test_other_profiles_keep_session_persistence_disabled(select, profile):
    select(profile)
    assert agent_input(resolved()).session_persistence is False


@pytest.mark.parametrize('profile', PROFILES)
def test_existing_swe_generator_emits_hyness_harness_and_preserves_case(select, monkeypatch, tmp_path: Path, profile):
    case = {'instance_id': 'fixture__one', 'problem_statement': 'Fix the fixture.', 'repo': 'fixture/repo'}
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(json.dumps(case) + '\n')
    select(profile, RUNTIME_CONFIG='swe_runtime', DATASET_PATH=str(dataset), SWE_LIMIT='1')
    (item,) = generator_swe_template.generate_configs()
    assert item.run_id == 'fixture__one'
    assert item.config.task_config['instance'] == case
    assert item.config.agent_name == 'hyness'
    selected = agent_input({'agent_config': item.config.agent_config})
    assert selected.harness_id == PROFILES[profile]
    assert item.config.openai_clients['main']['config']['request_args']['model'] == 'hy3'
    assert generator_swe_template.get_total_count() == 1


@pytest.mark.parametrize('profile', ['mini_swe_agent', 'swe_agent', 'workbuddy'])
def test_native_scaffold_names_keep_their_original_agents(select, profile):
    select(profile)
    assert resolved()['agent_name'] == profile


@pytest.mark.parametrize('profile', ['hyness_mini_swe_agent_app', 'hyness_swe_agent_app'])
def test_retired_app_profile_names_fail_before_generating_runs(select, profile):
    select(profile)
    with pytest.raises(FileNotFoundError, match=profile):
        resolved()


def test_generic_hyness_profile_passes_shared_schema():
    skips: list[str] = []
    assert validate.check_file(SHARED_DIR / 'agents/hyness.yaml', skips) == []
    assert skips == []


@pytest.mark.parametrize('missing', ['HYNESS_PACKAGES', 'HYNESS_APP_CONFIG'])
def test_generic_profile_requires_explicit_package_and_app(select, monkeypatch, missing):
    select('hyness', HYNESS_PACKAGES='["@tencent/hyness@1.0.0"]', HYNESS_APP_CONFIG='@example/app/app.yaml')
    monkeypatch.delenv(missing)
    with pytest.raises(InterpolationResolutionError, match=missing):
        resolved()


@pytest.mark.parametrize('app', ['@example/custom-app/app.yaml', '@tencent/hyness-opencc/apps/opencc/app.yaml'])
def test_generic_profile_leaves_harness_and_policy_to_selected_app(select, app):
    select('hyness', HYNESS_PACKAGES='["@tencent/hyness@1.0.0","@example/custom-app@2.0.0"]', HYNESS_APP_CONFIG=app)
    config = resolved()
    selected = agent_input(config)
    assert selected.app_config == app
    assert selected.app_config_overlay is None
    assert selected.harness_id is None
    assert '--disallowedTools' not in selected.extra_cli_args
    assert not any(k in (selected.extra_env or {}) for k in ['DISABLE_COMPACT', 'DISABLE_AUTO_COMPACT'])


def test_generic_profile_selects_unregistered_harness_without_python_changes(select):
    select(
        'hyness',
        HYNESS_PACKAGES='["@tencent/hyness@1.0.0","@example/custom-app@2.0.0"]',
        HYNESS_APP_CONFIG='@example/custom-app/app.yaml',
        HYNESS_HARNESS_ID='new-research-agent',
        HYNESS_APP_CONFIG_OVERLAY='/opt/policies/research.yaml',
    )
    selected = agent_input(resolved())
    assert selected.packages == ['@tencent/hyness@1.0.0', '@example/custom-app@2.0.0']
    assert selected.harness_id == 'new-research-agent'
    assert selected.app_config_overlay == '/opt/policies/research.yaml'


def test_generic_profile_does_not_retain_opencc_policy_when_switching_apps(select):
    values = {'HYNESS_PACKAGES': '["@tencent/hyness@1.0.0","@example/custom-app@2.0.0"]'}
    select('hyness_opencc', **values)
    assert agent_input(resolved()).app_config_overlay is not None
    select('hyness', HYNESS_APP_CONFIG='@example/custom-app/app.yaml', **values)
    custom = agent_input(resolved())
    assert custom.app_config_overlay is None
    assert custom.harness_id is None


def test_generic_hyness_works_with_existing_swe_generator(select, tmp_path):
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(json.dumps({'instance_id': 'custom__one'}) + '\n')
    select(
        'hyness',
        RUNTIME_CONFIG='swe_runtime',
        DATASET_PATH=str(dataset),
        HYNESS_PACKAGES='["@tencent/hyness@1.0.0","@example/custom-app@2.0.0"]',
        HYNESS_APP_CONFIG='@example/custom-app/apps/custom/app.yaml',
        HYNESS_HARNESS_ID='custom',
    )
    (item,) = generator_swe_template.generate_configs()
    selected = agent_input({'agent_config': item.config.agent_config})
    assert item.run_id == 'custom__one'
    assert selected.harness_id == 'custom'
    assert selected.app_config_overlay is None
