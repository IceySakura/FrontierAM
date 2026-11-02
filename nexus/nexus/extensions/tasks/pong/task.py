"""Task that gives main_agent an isolated local Atari Pong MCP server."""

import json
import sys

from typing import Any

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry


PONG_MCP_SERVER_NAME = 'pong'
PONG_MCP_MODULE = 'nexus.extensions.tasks.pong.mcp_server'
PONG_TOOLS = ['pong_reset', 'pong_step']

DEFAULT_SYSTEM_PROMPT = """You are controlling Atari Pong through stateful tools.
Use the returned contact sheet to track the ball and right-side paddle over time.
Tool calls change the environment, so call exactly one tool at a time and never replay a step speculatively."""


class PongTaskInput(TaskRunInput, WithAgentRunner):
    seed: int = 42
    max_decisions: int = Field(default=1000, gt=0)
    initial_noop_steps: int = Field(default=3, ge=0)
    frame_width: int = Field(default=480, gt=0)
    frame_height: int = Field(default=630, gt=0)
    mcp_timeout_s: float = Field(default=60.0, gt=0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str | None = None


class PongTaskOutput(TaskRunOutput, WithScore):
    episode_return: float
    decisions: int
    done: bool
    done_reason: str | None = None
    actions: list[int] = Field(default_factory=list)
    final_state: dict[str, Any] | None = None
    agent_output: Any | None = None


def build_pong_mcp_server_spec(task_input: PongTaskInput) -> dict[str, Any]:
    """Build a host-local stdio server spec for this task execution."""
    return {
        'name': PONG_MCP_SERVER_NAME,
        'client': {
            'transport': 'stdio',
            'command': sys.executable,
            'args': ['-m', PONG_MCP_MODULE],
            'env': {
                'SDL_VIDEODRIVER': 'dummy',
                'SDL_AUDIODRIVER': 'dummy',
                'NEXUS_PONG_FRAME_WIDTH': str(task_input.frame_width),
                'NEXUS_PONG_FRAME_HEIGHT': str(task_input.frame_height),
                'NEXUS_PONG_INITIAL_NOOP_STEPS': str(task_input.initial_noop_steps),
                'NEXUS_PONG_MAX_DECISIONS': str(task_input.max_decisions),
            },
            'timeout_s': task_input.mcp_timeout_s,
        },
    }


def build_pong_user_prompt(task_input: PongTaskInput) -> str:
    if task_input.user_prompt:
        return task_input.user_prompt
    return f"""Play one episode of Atari Pong.

1. Call `pong_reset` exactly once with seed {task_input.seed}.
2. Inspect the JSON state and the 2x2 frame sheet returned by every tool call.
3. Repeatedly call `pong_step` with one action from 0 through 5. Call only one step at a time because the game is stateful.
4. Continue until `done` is true. The server also stops after {task_input.max_decisions} decisions.
5. Then report the episode return, decision count, and why the episode ended.

The contact sheet is chronological: top-left is oldest and bottom-right is newest."""


def _pong_call_records(tool_state: dict[str, Any]) -> list[dict[str, Any]]:
    records = tool_state.get('__mcp_call_records__', [])
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, dict)
        and record.get('server') == PONG_MCP_SERVER_NAME
        and record.get('tool_name') in {'reset', 'step'}
    ]


def _state_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    content = record.get('response_content', [])
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict) or part.get('type') != 'text':
            continue
        text = part.get('text')
        if not isinstance(text, str):
            continue
        try:
            state = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(state, dict) and {'episode_return', 'decision', 'done'} <= state.keys():
            return state
    return None


def summarize_pong_run(tool_state: dict[str, Any]) -> tuple[dict[str, Any] | None, list[int]]:
    final_state: dict[str, Any] | None = None
    actions: list[int] = []
    for record in _pong_call_records(tool_state):
        if record.get('tool_name') == 'step':
            arguments = record.get('arguments')
            action = arguments.get('action') if isinstance(arguments, dict) else None
            if isinstance(action, int):
                actions.append(action)
        state = _state_from_record(record)
        if state is not None:
            final_state = state
    return final_state, actions


@task_registry.register('pong')
async def run_pong_task(task_input: PongTaskInput) -> PongTaskOutput:
    """Run main_agent against one task-owned, host-local Pong MCP process."""
    ctx = require_context()
    ctx.tool_state['__mcp_call_records__'] = []
    server_spec = build_pong_mcp_server_spec(task_input)

    agent_output: Any | None = None
    try:
        await ctx.load_mcp_tools([server_spec], clear_existing=True)
        agent_output = await task_input.run_agent(
            system_prompt=task_input.system_prompt,
            user_prompt=build_pong_user_prompt(task_input),
            tools=PONG_TOOLS,
            parallel_tool_calls=False,
            multimodal_tool_output=True,
        )
    finally:
        await ctx.close_mcp_tools()

    final_state, actions = summarize_pong_run(ctx.tool_state)
    episode_return = float(final_state.get('episode_return', 0.0)) if final_state else 0.0
    decisions = int(final_state.get('decision', len(actions))) if final_state else len(actions)
    done = bool(final_state.get('done', False)) if final_state else False
    done_reason = str(final_state['done_reason']) if final_state and final_state.get('done_reason') else None

    return PongTaskOutput(
        score=episode_return,
        episode_return=episode_return,
        decisions=decisions,
        done=done,
        done_reason=done_reason,
        actions=actions,
        final_state=final_state,
        agent_output=agent_output,
    )
