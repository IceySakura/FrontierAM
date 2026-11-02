"""V2 uses the same non-cgroup CC profile as V1."""

import pytest

from pydantic import ValidationError

from nexus.extensions.tasks.os_world_v2.execution.claude_code import OSWorldExecutionOptions
from nexus.extensions.tasks.os_world_v2.generators.policy import agent_config, selection


@pytest.mark.parametrize('family', ['claude', 'hunyuan'])
def test_v2_cc_retains_grep_and_explicit_process_guard(monkeypatch, family):
    monkeypatch.setenv('OSWORLD_CC_PROCESS_GUARD', 'true')
    name, config = agent_config(selection(model_family=family, harness='cc'))
    assert name == 'claude_code_multiturn'
    assert config['execution_profile'] == 'osworld_v2'
    options = OSWorldExecutionOptions(**config['execution_options'])
    assert options.system_grep is True
    assert options.process_guard is True
    assert (
        not {
            'resource_isolation',
            'cc_memory_max_mib',
            'cc_memory_reserve_mib',
            'cc_tasks_max',
        }
        & options.model_dump().keys()
    )


def test_v2_cc_extra_env_json_extends_default_env(monkeypatch):
    monkeypatch.setenv('CC_EXTRA_ENV_JSON', '{"API_TIMEOUT_MS": 600000, "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "0"}')
    _, config = agent_config(selection(model_family='hunyuan', harness='cc'))
    assert config['execution_options']['extra_env'] == {
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '0',
        'API_TIMEOUT_MS': '600000',
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW': '200000',
        'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '80',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '16384',
    }


def test_v2_claude_does_not_inherit_hunyuan_context_limits(monkeypatch):
    monkeypatch.delenv('CC_EXTRA_ENV_JSON', raising=False)
    _, config = agent_config(selection(model_family='claude', harness='cc'))
    assert 'CLAUDE_CODE_AUTO_COMPACT_WINDOW' not in config['execution_options']['extra_env']


def test_v2_hunyuan_compact_limits_can_be_overridden(monkeypatch):
    monkeypatch.setenv('CC_AUTO_COMPACT_WINDOW', '180000')
    monkeypatch.setenv('CC_AUTOCOMPACT_PCT_OVERRIDE', '75')
    monkeypatch.setenv('CC_MAX_OUTPUT_TOKENS', '8192')
    monkeypatch.setenv('CC_EXTRA_ENV_JSON', '{"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE":"70"}')
    _, config = agent_config(selection(model_family='hunyuan', harness='cc'))
    env = config['execution_options']['extra_env']
    assert env['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '180000'
    assert env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] == '70'
    assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '8192'


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('CLAUDE_CODE_AUTO_COMPACT_WINDOW', '0'),
        ('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '-1'),
        ('CLAUDE_AUTOCOMPACT_PCT_OVERRIDE', '0'),
        ('CLAUDE_AUTOCOMPACT_PCT_OVERRIDE', '101'),
    ],
)
def test_v2_hunyuan_rejects_invalid_final_compact_env(monkeypatch, key, value):
    import json

    monkeypatch.setenv('CC_EXTRA_ENV_JSON', json.dumps({key: value}))
    with pytest.raises(ValueError, match=key):
        agent_config(selection(model_family='hunyuan', harness='cc'))


@pytest.mark.parametrize(
    'field',
    [
        'resource_isolation',
        'cc_memory_max_mib',
        'cc_memory_reserve_mib',
        'cc_tasks_max',
    ],
)
def test_retired_cgroup_options_are_not_silently_accepted(field):
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        OSWorldExecutionOptions(**{field: 1})
