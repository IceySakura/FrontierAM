import logging
import os

from typing import Any

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.agents.base.final_answer import generate_final_answer
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.tasks.hy3_search.evaluator import EvalMode, SearchEvaluator
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


def _get_default_eval_mode() -> EvalMode:
    """Get default eval mode from environment variable SEARCH_EVAL_MODE."""
    env_mode = os.environ.get('SEARCH_EVAL_MODE', 'exact_match')
    try:
        return EvalMode(env_mode)
    except ValueError:
        logger.warning(f"Invalid SEARCH_EVAL_MODE '{env_mode}', falling back to exact_match")
        return EvalMode.EXACT_MATCH


class Hy3SearchTaskInput(TaskRunInput, WithAgentRunner):
    """HY3 Search task input.

    Evaluation modes:
    - EXACT_MATCH: Simple string comparison (default)
    - LLM_JUDGE: Use LLM for semantic equivalence (requires judge client config)
    - REWARD_SERVICE: Use reward service API for evaluation (cost-effective)
    - WIDESEARCH: Structured table evaluation using the full WideSearch pipeline
    """

    question: str
    answer: str
    instance_id: str | None = None
    eval_mode: EvalMode = Field(default_factory=_get_default_eval_mode)
    judge_use_tool_calling: bool = False  # Whether the judge should use tool calling (if LLM_JUDGE mode)
    instruction_template: str | None = None
    run_oracle: bool = False
    # LLM judge configuration
    judge_prompt_template: str | None = None
    judge_system_prompt: str | None = None
    # Training mode configuration
    training_length_exceeded: bool = bool(os.environ.get('TRAINING_LENGTH_EXCEEDED', False))
    enable_summary: bool = False  # Whether to summarize on MAX_ITERATIONS
    # WideSearch evaluation configuration (optional; auto-inferred when absent)
    evaluation: dict | None = None
    # 历史对话记录，非必填
    history: list[dict[str, str]] = []


class Hy3SearchTaskOutput(TaskRunOutput, WithScore):
    """Output containing both result and ground truth for immediate debuggability."""

    prediction: str
    answer: str
    trajectory: list[dict[str, Any]] | None = None
    evaluation_details: dict[str, Any] | None = None


@task_registry.register('hy3_search')
async def hy3_search_task(task_input: Hy3SearchTaskInput) -> Hy3SearchTaskOutput:
    """
    Executes a hy3_search task and evaluates the prediction against the answer.
    """
    question = task_input.question
    instance_id = task_input.instance_id
    logger.debug(
        f'Hy3Search task started: eval_mode={task_input.eval_mode!r}, instance_id={instance_id}, question={question}'
    )
    logger.debug(f'eval_model: {task_input.eval_mode}')
    # Standardize ground truth once
    target_answer = task_input.answer.strip().lower()

    trajectory = None

    # 1. Execution
    if task_input.run_oracle:
        logger.debug(f'Oracle Mode: Bypassing agent for "{question[:30]}..."')
        prediction_val = target_answer
    else:
        template = task_input.instruction_template
        if template is None:
            logger.warning('instruction_template is None, falling back to question as prompt')
        prompt = template.replace('{question}', question) if template else question
        agent_output = await task_input.run_agent(user_prompt=prompt, history=task_input.history, tools=['web_search'])

        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError(f'Agent output invalid type: {type(agent_output)}')

        trajectory = agent_output.main_trajectory

        # Handle training length exceeded case
        exit_status = getattr(agent_output, 'exit_status', None)
        termination_reason = None
        if exit_status is not None:
            raw_termination_reason = (
                exit_status.value if isinstance(exit_status, TerminationReason) else str(exit_status)
            )
            termination_reason = MainExitStatus.termination_reason_for(raw_termination_reason)
            if termination_reason == 'UNKNOWN':
                termination_reason = raw_termination_reason
        if task_input.training_length_exceeded and termination_reason in {
            TerminationReason.TRUNCATION.value,
            TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,
        }:
            assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
            logger.info(f'[TerminationReason: {termination_reason}] turn: {assistant_turns}, question: {question}')
            final_answer = getattr(agent_output, 'final_answer', None) or ''
            return Hy3SearchTaskOutput(
                score=0.0,
                prediction=final_answer,
                answer=target_answer,
                trajectory=trajectory,
            )

        should_summary = task_input.enable_summary and exit_status == TerminationReason.MAX_ITERATIONS
        if (
            task_input.eval_mode in (EvalMode.LLM_JUDGE, EvalMode.REWARD_SERVICE, EvalMode.WIDESEARCH)
            and not should_summary
        ):
            last_msg = agent_output.main_trajectory[-1]
            raw_content = last_msg.get('content') or last_msg.get('reasoning_content')
            prediction_val = extract_text_from_content(raw_content)
        elif task_input.eval_mode == EvalMode.EXACT_MATCH or should_summary:
            raw_pred = await generate_final_answer(
                trajectory=agent_output.main_trajectory,
                return_type='str',
            )
            prediction_val = str(raw_pred).strip().lower()
        else:
            raise RuntimeError(f'Unsupported eval_mode: {task_input.eval_mode}')

    # 2. Evaluation — always evaluate via evaluate_output for consistent behavior.
    # In oracle mode, downgrade non-wide evaluators to exact match to avoid unnecessary remote calls.
    effective_eval_mode = task_input.eval_mode
    if task_input.run_oracle and task_input.eval_mode != EvalMode.WIDESEARCH:
        effective_eval_mode = EvalMode.EXACT_MATCH

    evaluator = SearchEvaluator(
        effective_eval_mode,
        judge_use_tool_calling=task_input.judge_use_tool_calling,
        judge_prompt_template=task_input.judge_prompt_template,
        judge_system_prompt=task_input.judge_system_prompt,
        widesearch_evaluation=task_input.evaluation,
        instance_id=instance_id or 'unknown',
    )
    score, details_obj, is_valid_reward = await evaluate_output(
        prediction=prediction_val,
        reference=target_answer,
        question=question,
        evaluator=evaluator,
        need_details=True,
    )
    evaluation_details = details_obj.to_dict() if details_obj else None

    logger.info(
        f'InstanceID: {instance_id or "N/A"} | Score: {score} | is_valid_reward: {is_valid_reward} | '
        f'Question: {question} | Prediction: {prediction_val} | Reference: {target_answer} | '
        f'Evaluation Details: {evaluation_details}'
    )
    span_id = get_current_span_id()
    if span_id and not is_valid_reward:
        update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})

    return Hy3SearchTaskOutput(
        score=score,
        prediction=prediction_val,
        answer=target_answer,
        trajectory=trajectory,
        evaluation_details=evaluation_details,
    )
