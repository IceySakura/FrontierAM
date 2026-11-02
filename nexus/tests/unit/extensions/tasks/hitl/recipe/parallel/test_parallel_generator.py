"""What the launch script's environment turns into.

A variable the generator does not read costs a whole sandbox run to notice, so the
wiring is checked here: every key the script sets reaches the config it is meant to,
and a script written for the other recipe is refused rather than silently defaulted.
"""

import json

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.recipe.parallel.agent import AGENT_NAME, ParallelAgentInput
from nexus.extensions.tasks.hitl.recipe.parallel.generator import ENVIRONMENT, generate_configs
from nexus.extensions.tasks.hitl.recipe.parallel.task import TASK_NAME, ParallelTaskInput


SCRIPT = [
    {'parallel': [{'to': 'alpha', 'message': 'write ALPHA'}, {'to': 'beta', 'message': 'write BETA'}]},
    {'parallel': [{'to': 'alpha', 'message': 'what did you write?'}]},
]
CHECKS = [
    {'name': 'alpha kept to itself', 'label': 'worker/alpha', 'holds': 'ALPHA', 'lacks': 'BETA'},
    {'name': 'they ran together', 'overlap': 2},
]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """The smallest environment that generates, as a launch script would set it."""
    script = tmp_path / 'script.json'
    script.write_text(json.dumps(SCRIPT))
    checks = tmp_path / 'checks.json'
    checks.write_text(json.dumps(CHECKS))

    values = {
        'HITL_PARALLEL_SCRIPT': str(script),
        'HITL_PARALLEL_CHECKS': str(checks),
        'HITL_PARALLEL_IMAGE': 'example/image:latest',
        'OPENAI_API_KEY': 'k',
        'OPENAI_BASE_URL': 'http://gateway/v1',
        'OPENAI_MODEL': 'openai/some-model',
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    for key in ENVIRONMENT:
        if key not in values:
            monkeypatch.delenv(key, raising=False)
    return values


def only(env: dict[str, str]) -> Any:
    items = list(generate_configs())
    assert len(items) == 1
    return items[0]


def test_the_default_run_opens_two_sessions(env: dict) -> None:
    item = only(env)

    assert item.config.agent_config['sessions'] == ['alpha', 'beta']


def test_the_script_reaches_the_agent(env: dict) -> None:
    item = only(env)

    addressed = [[(send['to'], send['message']) for send in step['parallel']] for step in item.config.agent_config['script']]
    assert addressed == [[(send['to'], send['message']) for send in step['parallel']] for step in SCRIPT]


def test_a_follow_up_in_the_script_reaches_the_agent(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """It is what makes a session answerable the moment it finishes."""
    monkeypatch.setenv(
        'HITL_PARALLEL_SCRIPT',
        json.dumps(
            [{'parallel': [{'to': 'alpha', 'message': 'go', 'follow_up': ['and then this']}, {'to': 'beta', 'message': 'go'}]}]
        ),
    )

    sends = only(env).config.agent_config['script'][0]['parallel']

    assert sends[0]['follow_up'] == ['and then this']
    assert sends[1]['follow_up'] == []


def test_the_checks_reach_the_task(env: dict) -> None:
    item = only(env)

    assert [check['name'] for check in item.config.task_config['checks']] == [check['name'] for check in CHECKS]


def test_the_named_task_and_agent_are_the_ones_registered(env: dict) -> None:
    item = only(env)

    assert item.config.task_name == TASK_NAME
    assert item.config.agent_name == AGENT_NAME


def test_the_tracker_is_configured_to_carry_labels(env: dict) -> None:
    """In ``global_first`` every trajectory exports as a bare list with nowhere to put
    a label, so every session would export unattributed and nothing would say so."""
    item = only(env)

    assert item.config.processors['chat'].args['tools_mode'] == 'branch_last'


def test_the_usage_processor_is_configured(env: dict) -> None:
    """The run asserts one token label per session; without this there are none."""
    item = only(env)

    assert item.config.processors['usage'].name == 'hitl_usage'


def test_the_run_ceiling_reaches_a_field_the_agent_declares(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner builds a task input from the fields the task declares and drops the
    rest, so a ceiling named in ``task_config`` would be silently ignored."""
    monkeypatch.setenv('TIMEOUT_MULTIPLIER', '5')

    item = only(env)

    assert item.config.agent_config['timeout'] == 300.0
    assert 'agent_timeout' not in item.config.task_config


def test_the_sandbox_builds_the_worker_venv(env: dict) -> None:
    """The worker manager invokes this package's code in the sandbox."""
    item = only(env)

    assert item.config.runtime_configs['main']['worker_venv'] is True


def test_every_agent_key_is_one_the_agent_declares(env: dict) -> None:
    item = only(env)

    assert set(item.config.agent_config) <= set(ParallelAgentInput.model_fields)


def test_every_task_key_is_one_the_task_declares(env: dict) -> None:
    item = only(env)

    assert set(item.config.task_config) <= set(ParallelTaskInput.model_fields)


def test_the_sessions_can_be_named_by_the_script(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_SESSIONS', 'one,two,three')
    monkeypatch.setenv(
        'HITL_PARALLEL_SCRIPT',
        json.dumps([{'parallel': [{'to': name, 'message': 'go'} for name in ('one', 'two', 'three')]}]),
    )

    assert only(env).config.agent_config['sessions'] == ['one', 'two', 'three']


def test_one_session_is_a_valid_run(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """How the wiring is checked before spending a sandbox on two."""
    monkeypatch.setenv('HITL_PARALLEL_SESSIONS', 'alpha')
    monkeypatch.setenv('HITL_PARALLEL_SCRIPT', json.dumps([{'parallel': [{'to': 'alpha', 'message': 'go'}]}]))

    assert only(env).config.agent_config['sessions'] == ['alpha']


def test_repeating_yields_one_item_per_run(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_REPEAT', '3')

    items = list(generate_configs())

    assert len(items) == 3
    assert len({item.run_id for item in items}) == 3


def test_each_session_gets_its_own_directory_by_default(env: dict) -> None:
    item = only(env)

    assert item.config.agent_config['shared_workspace'] is False


def test_sharing_a_directory_is_available(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_SHARED_WORKSPACE', '1')

    assert only(env).config.agent_config['shared_workspace'] is True


# -- what the generator refuses ---------------------------------------------


def test_a_script_addressing_an_unopened_session_is_refused_before_the_sandbox(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('HITL_PARALLEL_SCRIPT', json.dumps([{'parallel': [{'to': 'gamma', 'message': 'go'}]}]))

    with pytest.raises(ValueError, match='gamma'):
        list(generate_configs())


def test_a_variable_from_the_other_recipe_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """It would otherwise take this generator's default and run something else."""
    monkeypatch.setenv('HITL_SMOKE_SCRIPT', '/tmp/other.json')

    with pytest.raises(ValueError, match='HITL_SMOKE_SCRIPT'):
        list(generate_configs())


def test_no_script_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HITL_PARALLEL_SCRIPT')

    with pytest.raises(ValueError, match='HITL_PARALLEL_SCRIPT required'):
        list(generate_configs())


def test_no_checks_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HITL_PARALLEL_CHECKS')

    with pytest.raises(ValueError, match='HITL_PARALLEL_CHECKS required'):
        list(generate_configs())


def test_an_empty_check_list_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_CHECKS', '[]')

    with pytest.raises(ValueError, match='reports nothing'):
        list(generate_configs())


def test_a_check_naming_no_kind_is_refused_here(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rather than at the end of a run that already spent a sandbox."""
    monkeypatch.setenv('HITL_PARALLEL_CHECKS', json.dumps([{'name': 'empty'}]))

    with pytest.raises(ValueError, match='sets none of'):
        list(generate_configs())


def test_no_image_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HITL_PARALLEL_IMAGE')

    with pytest.raises(ValueError, match='HITL_PARALLEL_IMAGE required'):
        list(generate_configs())


def test_a_file_that_is_not_there_says_so(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_SCRIPT', '/no/such/script.json')

    with pytest.raises(ValueError, match='neither a JSON list nor a file'):
        list(generate_configs())


def test_a_repeated_session_name_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HITL_PARALLEL_SESSIONS', 'alpha,alpha')

    with pytest.raises(ValueError, match='repeated'):
        list(generate_configs())
