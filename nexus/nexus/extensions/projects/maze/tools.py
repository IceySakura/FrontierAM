from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .treasuremaze.env import TreasureMazeEnv, make_env


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool input – the LLM only needs to supply an action string
# ---------------------------------------------------------------------------


class MazeActionInput(BaseModel):
    action: str = Field(
        default='look',
        description='Action to execute in the maze, such as "go north", "take key1", "look", "inventory".',
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_env_and_replay(state: dict) -> tuple[TreasureMazeEnv, str, bool, dict]:
    """Reconstruct the maze environment from *config* and replay *action_history*.

    Because the environment object itself is not serializable we store only the
    deterministic config and the ordered list of actions taken so far.  Each
    tool call rebuilds the env, replays the history, and then applies the new
    action.  With a fixed ``seed`` the replay is fully deterministic.
    """
    config = state.get('config', {})
    action_history: list[str] = state.get('action_history', [])

    env = make_env(
        size=config.get('size', 4),
        num_keys=config.get('num_keys', 3),
        ordered_keys=config.get('ordered_keys', True),
        seed=config.get('seed'),
        min_distractors_per_room=config.get('min_distractors_per_room', 2),
        max_steps=config.get('max_steps'),
    )

    observation, info = env.reset()
    done = env.done
    for action in action_history:
        observation, step_reward, done, info = env.step(action)
        if done:
            break

    return env, observation, done, info


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@tool_registry.register(
    'maze/act',
    meta={
        'name': 'maze_act',
        'description': 'Execute one action in the TreasureMaze environment and return the latest observation.',
        'tool_type': ToolType.LOCAL,
    },
)
async def maze_act_tool(params: MazeActionInput, context: ToolContext) -> ToolResponse:
    """Execute one action in the maze.

    The maze configuration (size, num_keys, seed, …) is expected to be
    pre-initialised in ``tool_state['maze_state']['config']`` by the task
    layer before the first tool call.  If no prior state exists at all the
    tool falls back to a small default maze so that it never crashes.
    """
    previous_state = context.tool_state.get('maze_state')

    if isinstance(previous_state, dict) and 'config' in previous_state:
        # Normal path – state was either pre-initialised by the task or
        # carried forward from a previous tool call.
        state: dict = {
            'config': dict(previous_state['config']),
            'action_history': list(previous_state.get('action_history', [])),
            'episode_reward': float(previous_state.get('episode_reward', 0.0)),
            'last_step_reward': float(previous_state.get('last_step_reward', 0.0)),
        }
    else:
        # Fallback – no state at all.  Use a tiny default config so the tool
        # does not error out; callers should pre-initialise via the task.
        logger.warning(
            'maze_act_tool: no maze_state found in tool_state; '
            'using default config.  Did the task layer forget to initialise it?'
        )
        state = {
            'config': {
                'size': 4,
                'num_keys': 3,
                'ordered_keys': True,
                'seed': 0,
                'min_distractors_per_room': 2,
                'max_steps': 80,
            },
            'action_history': [],
            'episode_reward': 0.0,
            'last_step_reward': 0.0,
        }

    # Rebuild the env from config and replay all previous actions
    env, replay_observation, done, replay_info = _build_env_and_replay(state)

    if done:
        # Episode already finished – return the last observation without
        # appending another action.
        observation = replay_observation
        info = replay_info
        valid_actions = env.valid_actions()
        action_applied = None
        step_reward = 0.0
    else:
        action_to_run = params.action.strip() or 'look'
        observation, step_reward, done, info = env.step(action_to_run)
        valid_actions = env.valid_actions()
        action_applied = action_to_run
        action_history = state['action_history']
        assert isinstance(action_history, list)
        action_history.append(action_to_run)

    prev_reward = state.get('episode_reward', 0.0)
    assert isinstance(prev_reward, (int, float))
    state['episode_reward'] = float(prev_reward) + step_reward
    state['last_step_reward'] = step_reward
    state['last_observation'] = observation
    state['done'] = done
    state['last_info'] = info

    meta = {
        'done': done,
        'success': bool(info.get('success', False)),
        'steps': info.get('steps', 0),
        'room_id': info.get('room_id'),
        'keys_remaining': info.get('keys_remaining', []),
        'next_required_key': info.get('next_required_key'),
        'valid_actions': valid_actions,
        'action_applied': action_applied,
        'step_reward': step_reward,
        'episode_reward': state['episode_reward'],
    }

    return ToolResponse(
        output=[{'type': 'text', 'text': observation}],
        meta=meta,
        context_updates={'maze_state': state},
    )
