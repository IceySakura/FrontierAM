import logging
import time

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    task_registry,
)
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.tasks.math.evaluator import MathEvaluator
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


class MathTaskInput(TaskRunInput, WithAgentRunner):
    """Math task input."""

    question: str
    answer: float
    run_oracle: bool = False


class MathTaskOutput(TaskRunOutput, WithScore):
    """Math task output."""

    prediction: str
    answer: float


@task_registry.register('math')
async def math_task(task_input: MathTaskInput) -> MathTaskOutput:
    """
    Execute math task where agent must solve the problem.
    """
    # Execute
    if task_input.run_oracle:
        prediction = str(task_input.answer)
        logger.debug('Oracle mode: bypassed agent')
    else:
        agent_output = await task_input.run_agent(user_prompt=task_input.question)

        if hasattr(agent_output, 'final_answer'):
            prediction = agent_output.final_answer
        elif isinstance(agent_output, WithMainTrajectory):
            prediction = agent_output.main_trajectory[-1]['content']
        else:
            raise RuntimeError(f'Invalid agent output: {type(agent_output)}')

    prediction_val = ''
    if isinstance(prediction, str):
        prediction_val = prediction.strip().lower()
    elif isinstance(prediction, dict) and 'content' in prediction:
        prediction_val = str(prediction['content']).strip().lower()  # type: ignore
    else:
        prediction_val = str(prediction).strip().lower()

    # Evaluate
    start_time = time.time()
    score, _, is_valid_reward = await evaluate_output(
        prediction=prediction_val,
        reference=task_input.answer,
        question=task_input.question,
        evaluator=MathEvaluator(),
    )
    reward_time = time.time() - start_time
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.reward_time': reward_time})
    if not is_valid_reward and span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})
    logger.info(
        f'Question: {task_input.question} | Prediction: {prediction_val} | Reference: {task_input.answer} | Score: {score} | is_valid_reward: {is_valid_reward}'
    )

    return MathTaskOutput(
        score=score,
        prediction=prediction_val,
        answer=task_input.answer,
    )
