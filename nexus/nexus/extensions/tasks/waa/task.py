"""WAA (Windows Agent Arena) evaluation task.

This task orchestrates the full WAA evaluation pipeline:
1. Environment setup (pre-config actions)
2. Agent execution (navi_agent loop)
3. Task evaluation (getters + metrics → score)
"""

import logging

from typing import Protocol, runtime_checkable

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.waa.evaluator import evaluate_task
from nexus.extensions.tasks.waa.setup import close_all_and_cleanup, setup_environment
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime


logger = logging.getLogger(__name__)


# ============== Agent Output Protocol ==============


@runtime_checkable
class WithExitStatus(Protocol):
    """Protocol for agent outputs that report exit status."""

    @property
    def exit_status(self) -> str: ...


@runtime_checkable
class WithFinalAnswerProto(Protocol):
    """Protocol for agent outputs that report a final answer."""

    @property
    def final_answer(self) -> str | None: ...


def _extract_agent_conclusion(agent_output: object) -> str | None:
    """Extract the agent's final conclusion from its output.

    Returns:
        'DONE' if agent concluded task is complete,
        'FAIL' if agent concluded task is infeasible,
        None if agent did not reach a voluntary conclusion.
    """
    if isinstance(agent_output, WithFinalAnswerProto):
        answer = agent_output.final_answer
        if answer in ('DONE', 'FAIL'):
            return answer
    return None


class WAATaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single WAA benchmark task.

    Attributes:
        task_config: The full task configuration JSON (includes id, instruction,
            config, evaluator, etc.)
        max_steps: Maximum agent steps (passed to agent via run_agent).
        sleep_after_execution: Delay between agent steps.
        skip_setup: If True, skip environment setup (useful for debugging).
        skip_cleanup: If True, skip pre-task cleanup (closing all apps).
    """

    task_config: dict
    skip_setup: bool = False
    skip_cleanup: bool = False


class WAATaskOutput(TaskRunOutput, WithScore):
    """Output of a WAA benchmark task evaluation.

    Only contains evaluation-specific results. Agent execution details
    (terminated, termination_reason, actions) are captured separately
    in RunOutput.agent_output by the runner.

    Attributes:
        score: Evaluation score (0.0 or 1.0 for most tasks).
        task_id: Unique task identifier from the config.
    """

    task_id: str = ''


@task_registry.register('waa')
async def waa_task(task_input: WAATaskInput) -> WAATaskOutput:
    """Execute a single WAA benchmark task.

    Pipeline:
    1. Cleanup previous task state (close all apps, clear files)
    2. Setup environment per task config (launch apps, create files, etc.)
    3. Run the agent with the task instruction
    4. Evaluate the result using the task's evaluator config
    5. Return score and metadata
    """
    ctx = require_context()
    runtime = require_raw_waa_runtime(await ctx.get_runtime())

    task_config = task_input.task_config
    task_id = task_config.get('id', 'unknown')
    instruction = task_config.get('instruction', '')

    logger.debug(f'WAA task started: task_id={task_id}')
    logger.debug(f'Instruction: {instruction}')

    # 1. Pre-task cleanup
    if not task_input.skip_cleanup:
        logger.debug('Cleaning up previous task state...')
        await close_all_and_cleanup(runtime)

    # 2. Environment setup
    config_actions = task_config.get('config', [])
    if config_actions and not task_input.skip_setup:
        logger.debug(f'Running {len(config_actions)} setup actions...')
        try:
            await setup_environment(runtime, config_actions)
        except Exception as e:
            logger.exception('Setup failed for task %s', task_id)
            raise RuntimeError(f'WAA setup failed for task {task_id}') from e

    # 3. Run agent
    logger.debug('Starting agent execution...')
    agent_output = await task_input.run_agent(
        user_prompt=instruction,
        task_id=task_id,
    )

    # 4. Extract agent's final conclusion for evaluator
    conclusion = _extract_agent_conclusion(agent_output)
    logger.info(f'Agent finished: conclusion={conclusion}')

    # 5. Evaluate
    # WAA always runs evaluation regardless of how the agent exited.
    # conclusion='DONE'/'FAIL' is used by evaluator to handle infeasible tasks
    # and feasible-but-agent-said-FAIL cases. When conclusion is None (e.g. max_iterations),
    # we still run evaluate — the agent may have completed the task without explicitly saying DONE.
    evaluator_config = task_config.get('evaluator', {})
    if not evaluator_config:
        logger.warning(f'No evaluator config for task {task_id}, returning score=0')
        score = 0.0
    else:
        logger.debug('Running evaluation...')
        score = await evaluate_task(runtime, evaluator_config, conclusion or 'DONE')

    logger.info(f'WAA task completed: task_id={task_id}, score={score}')

    return WAATaskOutput(score=score, task_id=task_id)
