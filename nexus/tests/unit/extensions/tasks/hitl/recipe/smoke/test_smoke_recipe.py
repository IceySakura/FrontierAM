"""The smoke recipe: a run states what to ask for and what to look for.

Its purpose is to check a mechanism reached the worker, so the risk it carries is a
check that cannot fail -- a tool name nothing produces, a command nothing runs -- which
would report a passing run for a broken mechanism. What is pinned here is that a
malformed check is refused before a conversation is paid for, and that the three kinds
read the sources they claim to.

No run happens here. What a real run adds is whether the worker used the tool at all.
"""

import json

from pathlib import Path
from typing import Any

import pytest

from nexus.core import task_registry
from nexus.extensions.tasks.hitl.recipe.smoke import agent as smoke_agent
from nexus.extensions.tasks.hitl.recipe.smoke import generator as generator_mod
from nexus.extensions.tasks.hitl.recipe.smoke.task import (
    SmokeCheck,
    SmokeTaskInput,
    SmokeTaskOutput,
    run_smoke_task,
)
from nexus.extensions.tasks.hitl.user_agent import mcp_config


REQUIRED_ENV = {
    'HITL_SMOKE_MESSAGE': 'say hello',
    'HITL_SMOKE_IMAGE': 'some/image:1',
    'HITL_WORKER_API_KEY': 'k',
    'HITL_WORKER_BASE_URL': 'http://x/v1',
    'HITL_WORKER_MODEL': 'm',
    'HITL_HUMAN_API_KEY': 'k',
    'HITL_HUMAN_BASE_URL': 'http://x/v1',
    'HITL_HUMAN_MODEL': 'm',
}

# Left unset by the fixture, so a test that does not name one gets the default rather
# than whatever the shell that ran pytest had.
RECIPE_ENV = (
    'HITL_SMOKE_CHECKS',
    'HITL_SMOKE_MODE',
    'HITL_SMOKE_SCRIPT',
    'HITL_SMOKE_INJECTIONS',
    'HITL_SMOKE_REPEAT',
    'HITL_MAX_TURNS',
    'HITL_WORKER_MCP_SEARCH',
    'HITL_USER_MCP_SERVERS',
    'HITL_WORKER_MCP_SERVERS',
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """The variables a smoke run needs, with the recipe's own left unset."""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for key in (*generator_mod.FOREIGN_ENV, *RECIPE_ENV):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def one_check(**fields: Any) -> str:
    return json.dumps([{'name': 'it worked', **fields}])


# -- the task is reachable by the name the generator uses ------------------


def test_the_task_is_registered_under_the_name_the_generator_names() -> None:
    assert task_registry.get(generator_mod.TASK_NAME) is not None


# -- a check has to be able to fail ---------------------------------------


def test_a_check_setting_none_of_the_three_fields_is_refused() -> None:
    """It would report neither pass nor fail, so it would have to default to one."""
    with pytest.raises(ValueError, match='none of tool, says or command'):
        SmokeCheck(name='empty').kind()


@pytest.mark.parametrize(
    ('fields', 'expected'),
    [
        ({'tool': 'mcp__search'}, 'tool'),
        ({'says': 'PROBE'}, 'says'),
        ({'command': 'test -f /tmp/x'}, 'command'),
    ],
)
def test_each_kind_is_read_off_the_field_that_is_set(fields: dict, expected: str) -> None:
    assert SmokeCheck(name='c', **fields).kind() == expected


def test_a_run_with_no_checks_is_refused_before_the_conversation() -> None:
    """Otherwise it pays for a worker and reports nothing about it."""
    import asyncio

    async def never_called(**_kwargs: Any) -> Any:
        raise AssertionError('the conversation must not start')

    with pytest.raises(ValueError, match='at least one check'):
        asyncio.run(run_smoke_task(SmokeTaskInput(message='x', checks=[], run_agent=never_called)))


# -- the declaration a launch script writes -------------------------------


def test_checks_are_read_from_a_json_list(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='mcp__search'))

    checks = generator_mod._load_checks()

    assert [check.name for check in checks] == ['it worked']
    assert checks[0].tool == 'mcp__search'


def test_checks_are_read_from_a_file_a_script_points_at(env, tmp_path: Path) -> None:
    """A multi-check declaration in a shell string is where quoting goes wrong."""
    path = tmp_path / 'checks.json'
    path.write_text(one_check(command='test -f /tmp/x'), encoding='utf-8')
    env.setenv('HITL_SMOKE_CHECKS', str(path))

    assert generator_mod._load_checks()[0].command == 'test -f /tmp/x'


def test_a_path_that_names_no_file_is_refused(env, tmp_path: Path) -> None:
    env.setenv('HITL_SMOKE_CHECKS', str(tmp_path / 'typo.json'))

    with pytest.raises(ValueError, match='neither a JSON list nor a file'):
        generator_mod._load_checks()


def test_a_malformed_check_is_refused_by_the_generator(env) -> None:
    """Where it costs nothing. Inside the sandbox it costs the conversation."""
    env.setenv('HITL_SMOKE_CHECKS', one_check())

    with pytest.raises(ValueError, match='none of tool, says or command'):
        generator_mod._load_checks()


def test_an_empty_list_is_refused(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', '[]')

    with pytest.raises(ValueError, match='non-empty list'):
        generator_mod._load_checks()


def test_the_same_check_name_twice_is_refused(env) -> None:
    """The report is read by name, so two of a name make one of them unreadable."""
    env.setenv('HITL_SMOKE_CHECKS', json.dumps([{'name': 'a', 'tool': 'x'}, {'name': 'a', 'tool': 'y'}]))

    with pytest.raises(ValueError, match='more than once'):
        generator_mod._load_checks()


# -- what the generator hands the runner ----------------------------------


def test_a_run_declares_the_task_the_agent_and_the_checks(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='mcp__search'))

    items = list(generator_mod.generate_configs())

    assert len(items) == 1
    config = items[0].config
    assert config.task_name == generator_mod.TASK_NAME
    assert config.agent_name == generator_mod.AGENT_NAME
    assert config.task_config['checks'] == [SmokeCheck(name='it worked', tool='mcp__search').model_dump()]
    assert config.task_config['message'] == 'say hello'


def test_the_agent_is_given_a_working_directory(env) -> None:
    """The smoke task has no dataset to read one from, and the agent requires it."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_config['working_dir'] == generator_mod.DEFAULT_WORKING_DIR


def test_the_chat_tracker_is_configured_in_the_mode_a_label_survives() -> None:
    """``tools_used`` is read off the exported trajectories, which ``global_first``
    exports as bare lists with nowhere to carry a label."""
    processors = generator_mod._processors('default', 'smoke')

    assert processors['chat']['name'] == 'chat_tracker'
    assert processors['chat']['args']['tools_mode'] == 'branch_last'
    assert processors['usage']['name'] == 'hitl_usage'


def test_the_task_module_is_imported_so_the_runner_can_resolve_it(env) -> None:
    """A task named in the config but never imported is not in the registry when the
    worker process looks for it."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    config = next(iter(generator_mod.generate_configs())).config

    assert generator_mod.TASK_IMPORT in config.imports
    assert generator_mod.AGENT_IMPORT in config.imports


def test_repeating_a_configuration_gives_each_run_its_own_id(env) -> None:
    """How a flaky mechanism is told from a broken one."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_SMOKE_REPEAT', '3')

    items = list(generator_mod.generate_configs())

    assert [item.run_id for item in items] == ['smoke__try1', 'smoke__try2', 'smoke__try3']


def test_a_single_run_keeps_the_plain_name(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    assert next(iter(generator_mod.generate_configs())).run_id == 'smoke'


def test_a_variable_from_another_recipe_is_refused(env) -> None:
    """A script for this run is most likely derived from another by editing the
    generator path, which leaves these set."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_MAX_AGENT_ITERATIONS', '40')

    with pytest.raises(ValueError, match='HITL_MAX_AGENT_ITERATIONS'):
        list(generator_mod.generate_configs())


def test_the_worker_mcp_server_reaches_the_worker_config(env) -> None:
    """What a run testing MCP depends on: the entry has to be in the config the
    manager is constructed with, or the sandbox never registers it."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='mcp__search'))
    env.setenv('HITL_WORKER_MCP_SEARCH', '1')
    env.setenv('SEARCH_API_KEY', 'sk-test')

    config = next(iter(generator_mod.generate_configs())).config
    worker_config = config.agent_config['worker_config']

    assert [entry['name'] for entry in worker_config['mcp_servers']] == ['search']
    assert worker_config['extra_env']['SEARCH_API_KEY'] == 'sk-test'


def test_the_user_mcp_server_reaches_the_agent_not_the_worker_config(env, tmp_path: Path) -> None:
    """The two sides are separate mechanisms, and this generator has to carry both.

    Measured as a silent gap: the generator read only the worker's, so a run
    declaring a user server got one that never loaded, and the check looking for
    its tool failed as though the mechanism were broken.
    """
    declared = tmp_path / 'user_mcp.json'
    declared.write_text(
        json.dumps([{'name': 'docs', 'command': '/usr/bin/python3', 'args': ['-m', 'docs']}]), encoding='utf-8'
    )
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='docs_'))
    env.setenv(mcp_config.USER_ENV, str(declared))

    config = next(iter(generator_mod.generate_configs())).config

    assert [entry['name'] for entry in config.agent_config['mcp_servers']] == ['docs']
    assert 'client' in config.agent_config['mcp_servers'][0], 'the user side takes the nested shape'
    assert 'mcp_servers' not in config.agent_config['worker_config'], 'the worker takes a different shape'


def test_the_two_sides_are_declared_by_different_variables(env) -> None:
    """One variable for both would send each side a shape the other cannot read."""
    assert mcp_config.USER_ENV != mcp_config.WORKER_ENV
    assert 'HITL_USER_MCP_SERVERS' in generator_mod.ENVIRONMENT
    assert 'HITL_WORKER_MCP_SERVERS' in generator_mod.ENVIRONMENT


# -- the mode a run is driven in -------------------------------------------


def test_a_run_is_live_unless_it_says_otherwise(env) -> None:
    """The mode that tests what a prompt causes, which is the common ask."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_config['mode'] == 'live'
    assert 'script' not in config.agent_config


def test_a_replay_carries_its_script_to_the_agent(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='mcp__search'))
    env.setenv('HITL_SMOKE_MODE', 'replay')
    env.setenv('HITL_SMOKE_SCRIPT', json.dumps([{'tool': 'send_message', 'args': {'message': 'search it'}}]))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_config['mode'] == 'replay'
    assert config.agent_config['script'] == [{'tool': 'send_message', 'args': {'message': 'search it'}}]


def test_a_replay_with_no_script_is_refused_before_the_sandbox(env) -> None:
    """It would start a container and do nothing in it."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_SMOKE_MODE', 'replay')
    env.delenv('HITL_SMOKE_SCRIPT', raising=False)

    with pytest.raises(ValueError, match='HITL_SMOKE_SCRIPT required'):
        list(generator_mod.generate_configs())


def test_a_mode_nothing_implements_is_refused(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_SMOKE_MODE', 'simulate')

    with pytest.raises(ValueError, match='simulate'):
        list(generator_mod.generate_configs())


def test_injections_reach_a_live_run(env) -> None:
    env.setenv('HITL_SMOKE_CHECKS', one_check(tool='mcp__search'))
    env.setenv('HITL_SMOKE_INJECTIONS', json.dumps([{'turn': 2, 'text': 'search for it now'}]))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_config['injections'] == [{'turn': 2, 'text': 'search for it now'}]


def test_an_injection_past_the_cap_is_refused_by_the_generator(env) -> None:
    """Where it costs nothing. In the sandbox it costs the conversation, and the check
    then fails for the wrong reason."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_MAX_TURNS', '3')
    env.setenv('HITL_SMOKE_INJECTIONS', json.dumps([{'turn': 9, 'text': 'x'}]))

    with pytest.raises(ValueError, match='past the cap'):
        list(generator_mod.generate_configs())


def test_a_declaration_can_be_a_file_or_a_list(env, tmp_path: Path) -> None:
    """A launch script writes whichever is less painful; the quoting is where a
    multi-entry list inline goes wrong."""
    path = tmp_path / 'script.json'
    path.write_text(json.dumps([{'tool': 'accept'}]), encoding='utf-8')
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))
    env.setenv('HITL_SMOKE_MODE', 'replay')
    env.setenv('HITL_SMOKE_SCRIPT', str(path))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_config['script'] == [{'tool': 'accept', 'args': {}}]


def test_the_mode_is_recorded_in_the_metadata(env) -> None:
    """Two modes' numbers are not comparable, so a reader has to be able to group."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    assert next(iter(generator_mod.generate_configs())).metadata['mode'] == 'live'


def test_the_agent_named_is_the_one_that_takes_a_mode(env) -> None:
    """The generic agent drops a field it does not declare, so a mode sent to it would
    be silently ignored and every run would be live."""
    env.setenv('HITL_SMOKE_CHECKS', one_check(says='hi'))

    config = next(iter(generator_mod.generate_configs())).config

    assert config.agent_name == smoke_agent.AGENT_NAME
    assert 'mode' in smoke_agent.SmokeAgentInput.model_fields


# -- the output shape a reader depends on ---------------------------------


def test_the_score_is_the_share_of_checks_that_passed() -> None:
    output = SmokeTaskOutput(
        score=0.5,
        all_passed=False,
        results=[],
    )

    assert output.score == 0.5
    assert not output.all_passed
