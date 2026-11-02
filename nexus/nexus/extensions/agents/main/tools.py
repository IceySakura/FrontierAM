"""Expose bounded first-level subagent control as local main-agent tools."""

from __future__ import annotations

import json

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.main.multi_agent import MULTI_AGENT_CONTROLLER_STATE_KEY, SubagentController


class SpawnAgentInput(BaseModel):
    prompt: str = Field(min_length=1, description='A self-contained task for the subagent.')
    name: str | None = Field(default=None, description='Optional label for correlating this subagent result.')


class WaitAgentInput(BaseModel):
    agent_ids: list[str] = Field(min_length=1, description='Subagent IDs returned by spawn_agent.')
    return_when: Literal['all', 'any'] = Field(default='all', description='Return after all or any target finishes.')
    timeout_seconds: float | None = Field(default=None, gt=0, description='Optional wait timeout that does not cancel.')


def _controller(context: ToolContext) -> SubagentController:
    controller = context.tool_state.get(MULTI_AGENT_CONTROLLER_STATE_KEY)
    if not isinstance(controller, SubagentController):
        raise RuntimeError('spawn_agent/wait_agent require an active main-agent multi_agent configuration')
    return controller


def _response(payload: dict[str, object]) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}], meta=payload)


@tool_registry.register(
    'multi_agent/spawn_agent',
    meta={'name': 'spawn_agent', 'description': 'Start one subagent in the background.', 'tool_type': ToolType.LOCAL},
)
async def spawn_agent_tool(params: SpawnAgentInput, context: ToolContext) -> ToolResponse:
    """Start one subagent and return its handle without waiting."""
    return _response((await _controller(context).spawn(params.prompt, name=params.name)).model_dump(mode='json'))


@tool_registry.register(
    'multi_agent/wait_agent',
    meta={'name': 'wait_agent', 'description': 'Wait for one or more spawned subagents.', 'tool_type': ToolType.LOCAL},
)
async def wait_agent_tool(params: WaitAgentInput, context: ToolContext) -> ToolResponse:
    """Return terminal or pending results for requested children."""
    result = await _controller(context).wait(
        params.agent_ids, return_when=params.return_when, timeout_seconds=params.timeout_seconds
    )
    return _response(result.model_dump(mode='json'))


__all__ = [
    'SpawnAgentInput',
    'WaitAgentInput',
    'spawn_agent_tool',
    'wait_agent_tool',
]
