from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from nexus.core import AgentRunOutput, ExecutionContext
from nexus.core.tool import MCPServerSpec
from nexus.extensions.harbor_trial.agent import NexusAgent
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding
from nexus.runtime.context import get_execution_user


class FakeAgentOutput(AgentRunOutput):
    final_answer: str


def _environment():
    return SimpleNamespace(
        task_env_config=SimpleNamespace(workdir='/workspace'),
        default_user='agent',
        exec=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_nexus_agent_normalizes_harbor_streamable_http_mcp(tmp_path) -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgentOutput(final_answer='done')

    run_agent.nexus_meta = {
        'name': 'claude_code',
        'type': 'agent',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    binding = HarborTrialBinding(
        execution_context=ExecutionContext(task_config={}, tool_state={}),
        run_agent=run_agent,
    )
    async with harbor_trial_binding(binding):
        agent = NexusAgent(
            logs_dir=tmp_path / 'logs',
            mcp_servers=[
                MCPServerConfig(
                    name='mcp-server',
                    transport='streamable-http',
                    url='http://mcp-server:8000/mcp',
                )
            ],
        )
        await agent.run('use the tool', _environment(), AgentContext())

    [server] = calls[0]['mcp_servers']
    spec = MCPServerSpec.model_validate(server)
    assert spec.name == 'mcp-server'
    assert spec.client.transport.value == 'streamable_http'
    assert spec.client.url == 'http://mcp-server:8000/mcp'
    assert get_execution_user() is None


@pytest.mark.asyncio
async def test_nexus_agent_passes_harbor_skills_to_claude_code(tmp_path) -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgentOutput(final_answer='done')

    run_agent.nexus_meta = {
        'name': 'claude_code',
        'type': 'agent',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    binding = HarborTrialBinding(
        execution_context=ExecutionContext(task_config={}, tool_state={}),
        run_agent=run_agent,
    )
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs', skills_dir='/skills')
        await agent.run('use a skill', _environment(), AgentContext())

    assert calls[0]['skills_dir'] == '/skills'


@pytest.mark.asyncio
async def test_nexus_agent_rejects_harbor_skills_for_unsupported_agent(tmp_path) -> None:
    run_agent = AsyncMock()
    run_agent.nexus_meta = {
        'name': 'codex',
        'type': 'agent',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    binding = HarborTrialBinding(
        execution_context=ExecutionContext(task_config={}, tool_state={}),
        run_agent=run_agent,
    )
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs', skills_dir='/skills')
        with pytest.raises(ValueError, match="Agent 'codex' does not support Harbor skills_dir"):
            await agent.run('use a skill', _environment(), AgentContext())

    run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_nexus_agent_rejects_http_mcp_for_codex(tmp_path) -> None:
    run_agent = AsyncMock()
    run_agent.nexus_meta = {
        'name': 'codex',
        'type': 'agent',
        'meta': {'supports_execution_target': True, 'supports_execution_user': True},
    }
    binding = HarborTrialBinding(
        execution_context=ExecutionContext(task_config={}, tool_state={}),
        run_agent=run_agent,
    )
    async with harbor_trial_binding(binding):
        agent = NexusAgent(
            logs_dir=tmp_path / 'logs',
            mcp_servers=[
                MCPServerConfig(
                    name='mcp-server',
                    transport='streamable-http',
                    url='http://mcp-server:8000/mcp',
                )
            ],
        )
        with pytest.raises(ValueError, match='Codex supports only stdio MCP servers'):
            await agent.run('use the tool', _environment(), AgentContext())

    run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_nexus_agent_rejects_partial_user_support(tmp_path) -> None:
    run_agent = AsyncMock()
    run_agent.nexus_meta = {'name': 'partial', 'meta': {'supports_execution_target': True}}
    binding = HarborTrialBinding(
        execution_context=ExecutionContext(task_config={}, tool_state={}),
        run_agent=run_agent,
    )
    async with harbor_trial_binding(binding):
        with pytest.raises(ValueError, match='complete execution-user'):
            await NexusAgent(logs_dir=tmp_path / 'logs').run('task', _environment(), AgentContext())
    run_agent.assert_not_awaited()
