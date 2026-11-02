"""Tau2 benchmark task — runs a customer-service simulation and evaluates the result.

The task delegates execution to the tau2 agent, then uses tau2's evaluator
to compute a reward score (0.0–1.0).
"""

import logging

from typing import Any

from nexus.core import (
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    task_registry,
)


logger = logging.getLogger(__name__)


class Tau2TaskInput(TaskRunInput, WithAgentRunner):
    """Input configuration for a single tau2 benchmark task.

    Fields ``task_data`` and ``db_path`` are set by the generator for each task.
    """

    solo_mode: bool = False
    max_steps: int = 200
    max_errors: int = 10
    seed: int | None = None
    evaluation_type: str = 'all'  # all | env | action | communicate
    add_thinking_tool: bool = False
    task_data: dict[str, Any] | None = None  # tau2 Task dict, provided by generator
    db_path: str | None = None  # per-task DB path (e.g. for RL data)


class Tau2TaskOutput(TaskRunOutput, WithScore):
    """Output of a tau2 benchmark task, including score and diagnostics."""

    reward_info: dict[str, Any] | None = None  # detailed reward breakdown
    termination_reason: str | None = None
    trajectory: list[dict[str, Any]] = []  # full message history
    error: str | None = None


@task_registry.register('tau2')
async def run_tau2_task(task_input: Tau2TaskInput) -> Tau2TaskOutput:
    """Run a tau2-bench simulation and evaluate the result.

    1. Delegates to the agent to run the multi-turn simulation.
    2. Evaluates the simulation using tau2's evaluator.
    3. Returns the evaluation reward as the task score.
    """
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    task_data = task_input.task_data or {}
    task_id = task_data.get('id', 'unknown')

    logger.debug(f'Running tau2 task: task_id={task_id}, solo_mode={task_input.solo_mode}')

    # 1. Call the agent to run the simulation (agent handles execution only)
    agent_output: AgentRunOutput = await task_input.run_agent(
        solo_mode=task_input.solo_mode,
        max_steps=task_input.max_steps,
        max_errors=task_input.max_errors,
        seed=task_input.seed,
        add_thinking_tool=task_input.add_thinking_tool,
        task_data=task_data,
        db_path=task_input.db_path,
    )

    # Extract raw results from agent output
    simulation = getattr(agent_output, 'simulation', None)
    tau2_task = getattr(agent_output, 'tau2_task', None)
    domain = getattr(agent_output, 'domain', '')
    solo_mode = getattr(agent_output, 'solo_mode', task_input.solo_mode)
    termination_reason = getattr(agent_output, 'termination_reason', None)
    trajectory = getattr(agent_output, 'trajectory', [])
    error = getattr(agent_output, 'error', None)

    # 2. Evaluate the simulation (task layer responsibility)
    score = 0.0
    reward_info = None

    if simulation is not None and tau2_task is not None and error is None:
        try:
            eval_type = EvaluationType(task_input.evaluation_type)

            # Build environment_constructor with per-task DB
            if task_input.db_path is None:
                raise ValueError(
                    'db_path is required for evaluation. Ensure generator provides db_path for every task.'
                )

            from nexus.extensions.tasks.tau2.agent import make_env_constructor

            env_constructor = make_env_constructor(domain, task_input.db_path)

            reward_result = evaluate_simulation(
                domain=domain,
                task=tau2_task,
                simulation=simulation,
                evaluation_type=eval_type,
                solo_mode=solo_mode,
                environment_constructor=env_constructor,  # ty: ignore[unknown-argument]
            )
            score = reward_result.reward
            reward_info = reward_result.model_dump()
        except Exception as e:
            logger.error(f'Tau2 evaluation error: {type(e).__name__}: {e}')
            error = str(e)

    logger.info(f'Tau2 task completed: score={score}, termination={termination_reason}')

    return Tau2TaskOutput(
        score=score,
        reward_info=reward_info,
        termination_reason=termination_reason,
        trajectory=trajectory,
        error=error,
    )
