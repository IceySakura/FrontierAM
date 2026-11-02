from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomllib

from nexus.core import agent_registry
from nexus.extensions.agents.dsh.agent import DshInput, _no_more_user_input, run_dsh


def test_dsh_agent_and_manifest_expose_one_acp_surface() -> None:
    assert agent_registry.get('dsh')
    manifest = tomllib.loads((Path(__file__).parents[4] / 'nexus/extensions/agents/dsh/pyproject.toml').read_text())

    assert manifest['tool']['nexus']['agents'] == {'dsh': 'nexus.extensions.agents.dsh.agent'}
    assert manifest['project']['authors'] == [{'name': 'tarschen'}]
    assert 'no_thinking_request_args' not in DshInput.model_fields
    assert 'haiku_model' not in DshInput.model_fields


@pytest.mark.asyncio
async def test_dsh_default_input_provider_ends_after_one_turn() -> None:
    input_data = DshInput(user_prompt='hello', working_dir='/workspace')

    assert input_data.get_user_input is _no_more_user_input
    assert await input_data.get_user_input(None) is None


@pytest.mark.asyncio
async def test_run_dsh_uses_acp_multiturn_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_installed_multiturn(input_data: DshInput, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'session_id': 'session-1',
            'turns': 1,
            'events': [],
            'ended_by_user': True,
            'reached_max_turns': False,
            'status': 'completed',
        }

    monkeypatch.setattr('nexus.extensions.agents.dsh.agent.run_installed_multiturn', fake_run_installed_multiturn)
    output = await run_dsh(DshInput(user_prompt='hello', working_dir='/workspace', permission_policy='reject'))

    assert output.output == 'done'
    assert output.turns == 1
    assert captured['agent_key'] == 'dsh'
    assert captured['runtime_kwargs']['permission_policy'] == 'reject'
    assert captured['runtime_kwargs']['version'] == '0.1.0-rc.6'
    assert captured['runtime_kwargs']['acp_version'] == '0.1.0-rc.6'
    assert captured['runtime_kwargs']['prebuilt_base_url'].endswith('/nexus-dsh/v1')
