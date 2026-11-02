import logging

from typing import Any

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    WithUserPrompt,
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default maze environment parameters – used when the corresponding key is
# absent from ``task_config``.
# ---------------------------------------------------------------------------

_DEFAULT_MAZE_CONFIG: dict[str, Any] = {
    'size': 4,
    'num_keys': 3,
    'ordered_keys': True,
    'seed': 0,
    'min_distractors_per_room': 2,
    'max_steps': 80,
}

# The keys in task_config that belong to the maze environment (not the task itself).
_MAZE_CONFIG_KEYS = frozenset(_DEFAULT_MAZE_CONFIG.keys())


class MazeTaskInput(TaskRunInput, WithAgentRunner, WithUserPrompt):
    expected_success: bool = True
    max_iterations: int = 20
    system_prompt: str | None = None
    # Maze environment parameters – passed through to tool_state
    size: int = _DEFAULT_MAZE_CONFIG['size']
    num_keys: int = _DEFAULT_MAZE_CONFIG['num_keys']
    ordered_keys: bool = _DEFAULT_MAZE_CONFIG['ordered_keys']
    seed: int | None = _DEFAULT_MAZE_CONFIG['seed']
    min_distractors_per_room: int = _DEFAULT_MAZE_CONFIG['min_distractors_per_room']
    max_steps: int | None = _DEFAULT_MAZE_CONFIG['max_steps']


class MazeTaskOutput(TaskRunOutput, WithScore):
    success: bool | None = None
    done: bool | None = None
    steps: int | None = None
    action_count: int = 0
    final_response: str | None = None


def _extract_last_assistant_content(trajectory: list[dict[str, Any]]) -> str | None:
    return next(
        (
            msg.get('content')
            for msg in reversed(trajectory)
            if msg.get('role') == 'assistant' and isinstance(msg.get('content'), str) and msg.get('content')
        ),
        None,
    )


@task_registry.register('maze_task')
async def run_maze_task(task_input: MazeTaskInput) -> MazeTaskOutput:
    logger.debug(f'Running maze task with prompt: {task_input.user_prompt}')

    # ------------------------------------------------------------------
    # Pre-initialise tool_state with the maze environment configuration
    # so that the first maze_act tool call can build the environment
    # without needing config parameters from the LLM.
    # ------------------------------------------------------------------
    ctx = require_context()
    maze_config: dict[str, Any] = {
        'size': task_input.size,
        'num_keys': task_input.num_keys,
        'ordered_keys': task_input.ordered_keys,
        'seed': task_input.seed,
        'min_distractors_per_room': task_input.min_distractors_per_room,
        'max_steps': task_input.max_steps,
    }
    ctx.tool_state['maze_state'] = {
        'config': maze_config,
        'action_history': [],
        'episode_reward': 0.0,
        'last_step_reward': 0.0,
    }
    logger.debug(f'Initialised maze_state with config: {maze_config}')

    # ------------------------------------------------------------------
    # Run the agent loop
    # ------------------------------------------------------------------
    agent_output = await task_input.run_agent(
        user_prompt=task_input.user_prompt,
        max_iterations=task_input.max_iterations,
        system_prompt=task_input.system_prompt,
    )

    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError('Maze task requires agent output to be a subclass of WithMainTrajectory')

    # ------------------------------------------------------------------
    # Extract final maze state and compute score
    # ------------------------------------------------------------------
    maze_state_raw = ctx.tool_state.get('maze_state')
    maze_state = maze_state_raw if isinstance(maze_state_raw, dict) else {}

    last_info_raw = maze_state.get('last_info')
    last_info = last_info_raw if isinstance(last_info_raw, dict) else {}

    success_raw = last_info.get('success')
    success = bool(success_raw) if success_raw is not None else None

    episode_reward_raw = maze_state.get('episode_reward')
    episode_reward = float(episode_reward_raw) if isinstance(episode_reward_raw, (int, float)) else None

    done_raw = maze_state.get('done')
    done = bool(done_raw) if done_raw is not None else None

    steps_raw = last_info.get('steps')
    steps = int(steps_raw) if isinstance(steps_raw, int) else None

    action_history = maze_state.get('action_history')
    action_count = len(action_history) if isinstance(action_history, list) else 0

    if success is None:
        score = 0.0
    else:
        if episode_reward is None:
            score = 1.0 if success == task_input.expected_success else 0.0
        else:
            score = 1.0 if success == task_input.expected_success else episode_reward

    final_response = _extract_last_assistant_content(agent_output.main_trajectory)

    logger.info(
        'Maze task finished. '
        f'success={success}, expected_success={task_input.expected_success}, score={score}, '
        f'done={done}, steps={steps}, action_count={action_count}'
    )

    return MazeTaskOutput(
        score=score,
        success=success,
        done=done,
        steps=steps,
        action_count=action_count,
        final_response=final_response,
    )
