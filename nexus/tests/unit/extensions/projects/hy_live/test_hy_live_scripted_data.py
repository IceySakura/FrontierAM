from __future__ import annotations

import json

import pytest

from nexus.extensions.projects.hy_live.example.prepare_scripted_user_data import (
    FIRST_USER_TURN,
    SECOND_USER_TURN,
    build_scripted_user_row,
)
from nexus.extensions.projects.hy_live.task import LiveAgentTaskInput
from nexus.runner.single import _build_input


async def _unused_run_agent(**kwargs):
    raise AssertionError(f'unexpected run_agent call: {kwargs}')


def test_scripted_a_then_b_row_matches_interactive_task_contract() -> None:
    row = build_scripted_user_row()
    nexus_config = json.loads(row['nexus'])
    task_config = nexus_config['task_config']

    task_input = _build_input(
        LiveAgentTaskInput,
        {
            **task_config,
            'user_prompt': row['prompt'][0]['content'],
            'run_agent': _unused_run_agent,
        },
    )

    assert row['agent_name'] == 'session_agent'
    assert nexus_config['task_name'] == 'hy-live'
    assert nexus_config['agent_name'] == 'hy-live'
    assert task_input.prompt == FIRST_USER_TURN
    assert task_input.user_simulator is not None
    assert task_input.user_simulator.type == 'scripted'
    assert task_input.user_simulator.turns == [FIRST_USER_TURN, SECOND_USER_TURN]
    assert task_input.backend_adapter == 'workbuddy'
    assert task_input.work_agent_configs == {'workbuddy': {'working_dir': '/workspace', 'llm_client_name': 'external'}}
    assert 'your_api_key' not in row['nexus']
    assert json.loads(build_scripted_user_row(agent_loop='nexus_agent')['nexus']) == nexus_config
    assert build_scripted_user_row(agent_loop='nexus_agent')['agent_name'] == 'nexus_agent'


@pytest.mark.parametrize('backend', ['codebuddy', 'workbuddy'])
def test_scripted_row_configures_only_the_selected_backend(backend: str) -> None:
    task_config = json.loads(build_scripted_user_row(backend=backend)['nexus'])['task_config']
    task_input = _build_input(LiveAgentTaskInput, {**task_config, 'run_agent': _unused_run_agent})

    assert task_input.backend_adapter == backend
    assert set(task_input.work_agent_configs) == {backend}
    config = task_input.work_agent_configs[backend]
    assert config['llm_client_name'] == 'external'
    assert config.get('version') == ('2.97.2' if backend == 'codebuddy' else None)
