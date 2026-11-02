import logging
import time

from enum import Enum
from typing import Any, Literal

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.hycode.agent import HyCodeOutput, extract_hycode_final_answer
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.tasks.search_stateful.evaluator import (
    JUDGE_INFRA_ERROR_PREFIX,
    GeneralEvaluator,
    HLEEvaluator,
    RewardSystemEvaluator,
    SearchStatefulEvaluator,
)
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)

# A rollout that ran out of room, or out of time, never reached the answer it was working
# towards, so ``prediction`` is whichever fragment the model had emitted when the limit hit
# — a half-written sentence, a truncated tool call, or bare reasoning. Judging that
# fragment spends a reward call on a score that describes nothing about the policy, and for
# the length cutoffs it still lands in the GRPO group statistics its siblings are centred
# on. Short-circuit to 0.0 instead; whether the partial sequence trains or is masked stays
# HyTuner's decision, taken from the termination reason.
_CUTOFF_TERMINATION_REASONS = frozenset({
    TerminationReason.TRUNCATION.value,  # prompt outgrew the model window mid-run
    TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,  # one sampling call hit max_tokens
    TerminationReason.TASK_TIMEOUT.value,  # a sampling call or an observation ran out of time
})


def _cutoff_termination_reason(agent_output: Any) -> str | None:
    """Return the canonical termination reason when the rollout was cut short, else None.

    Agents built on ``main_agent`` report the lowercase ``MainExitStatus`` value
    ('output_length_exceeded'); others report the uppercase ``TerminationReason``
    ('TRUNCATION'). Normalize both before matching, and keep an unrecognized value as-is
    so a future exit status cannot silently match a cutoff.

    Enum members are unwrapped by value: ``str()`` on a ``str``-mixin Enum yields
    'MainExitStatus.OUTPUT_LENGTH_EXCEEDED', which matches nothing.
    """
    exit_status = getattr(agent_output, 'exit_status', None)
    if exit_status is None:
        return None
    raw_reason = exit_status.value if isinstance(exit_status, Enum) else str(exit_status)
    reason = MainExitStatus.termination_reason_for(raw_reason)
    if reason == 'UNKNOWN':
        reason = raw_reason
    return reason if reason in _CUTOFF_TERMINATION_REASONS else None


_TRUNCATION_TERMINATION_REASONS = frozenset({'TRUNCATION', 'OUTPUT_LENGTH_EXCEEDED'})


def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get('text') or part.get('content')
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        return '\n'.join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_trajectory_prediction(main_trajectory: list[dict]) -> str:
    for message in reversed(main_trajectory):
        if message.get('role') != 'assistant':
            continue
        content = _content_to_text(message.get('content'))
        if content:
            return content
    logger.warning('No assistant content found in main trajectory; using empty prediction')
    return ''


def _extract_prediction(agent_output: object) -> str:
    """Extract final text from both chat agents and installed CLI agents."""
    success = getattr(agent_output, 'success', True)
    if success is False:
        exit_code = getattr(agent_output, 'exit_code', None)
        stderr = (getattr(agent_output, 'stderr', '') or '').strip()
        if len(stderr) > 1000:
            stderr = stderr[:1000] + '...'
        detail = f'exit_code={exit_code}'
        if stderr:
            detail = f'{detail}, stderr={stderr}'
        raise RuntimeError(f'Agent execution failed: {detail}')
    if hasattr(agent_output, 'final_answer'):
        final_answer = _content_to_text(agent_output.final_answer)
        if final_answer:
            return final_answer
    if isinstance(agent_output, WithMainTrajectory):
        return _extract_trajectory_prediction(agent_output.main_trajectory)

    if isinstance(agent_output, HyCodeOutput):
        return extract_hycode_final_answer(agent_output.output)

    if hasattr(agent_output, 'output'):
        output = _content_to_text(agent_output.output)
        if output:
            return output
        logger.warning('Agent output is empty for %s; using empty prediction', type(agent_output).__name__)
        return ''
    raise RuntimeError(f'Cannot extract prediction from agent output: {type(agent_output)}')


def _termination_reason(agent_output: object) -> str | None:
    reason = getattr(agent_output, 'termination_reason', None)
    if reason is None:
        reason = getattr(agent_output, 'exit_status', None)
    if reason is None:
        return None
    value = getattr(reason, 'value', reason)
    return str(value)


def _default_truncate_score(extra: dict[str, Any] | None) -> float:
    value = (extra or {}).get('default_truncate_score', 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('reward_system_extra.default_truncate_score must be numeric')
    return float(value)


class SearchStatefulTaskInput(TaskRunInput, WithAgentRunner):
    """Math task input."""

    question: str
    answer: str
    run_oracle: bool = False
    evaluator_type: Literal['hle', 'general', 'none', 'olympiad', 'research', 'oss', 'reward_system'] = 'hle'
    evaluator_none_score: float = 0.0

    # reward_system evaluator config
    reward_system_user_token: str | None = None
    reward_system_exp_id: str | None = None
    reward_system_router_id: str | None = None
    reward_system_extra: dict[str, Any] | None = None
    reward_system_public_test: dict[str, Any] | None = None
    reward_system_ci_test: bool = False


class SearchStatefulTaskOutput(TaskRunOutput, WithScore):
    """Math task output."""

    prediction: str
    answer: str
    judge_details: dict | None = None  # Judge details (prompt, response, parsed_result)


@task_registry.register('search_stateful')
async def search_stateful_task(task_input: SearchStatefulTaskInput) -> SearchStatefulTaskOutput:
    """
    Execute search stateful task where agent must solve the problem.
    """
    # Get context and initialize runtime
    ctx = require_context()
    runtime = await ctx.get_runtime('main')
    logger.debug(f'Runtime initialized: {runtime}')
    runtime_id = ctx.get_runtime_id('main')
    if runtime_id:
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})
    if task_input.evaluator_type == 'reward_system':
        logger.debug(
            '[search_stateful_task] reward_system config: router_id=%s, exp_id=%s, ci_test=%s, token=%s, '
            'extra_count=%d, public_test_count=%d',
            task_input.reward_system_router_id,
            task_input.reward_system_exp_id or '<empty>',
            task_input.reward_system_ci_test,
            task_input.reward_system_user_token,
            len(task_input.reward_system_extra or {}),
            len(task_input.reward_system_public_test or {}),
        )

    # Execute
    cutoff_reason: str | None = None
    termination_reason = None
    skip_reward_for_truncation = False
    if task_input.run_oracle:
        prediction = str(task_input.answer)
        logger.debug('Oracle mode: bypassed agent')
    else:
        agent_output = await task_input.run_agent(user_prompt=task_input.question)
        cutoff_reason = _cutoff_termination_reason(agent_output)
        termination_reason = _termination_reason(agent_output)
        if cutoff_reason is None and termination_reason in _CUTOFF_TERMINATION_REASONS:
            cutoff_reason = termination_reason
        skip_reward_for_truncation = (
            task_input.evaluator_type == 'reward_system' and termination_reason in _TRUNCATION_TERMINATION_REASONS
        )
        if cutoff_reason is not None:
            # Preserve partial output for dumps, but do not judge it as a completed answer.
            if hasattr(agent_output, 'output'):
                prediction = _content_to_text(agent_output.output)
            elif hasattr(agent_output, 'final_answer'):
                prediction = agent_output.final_answer
            elif isinstance(agent_output, WithMainTrajectory):
                prediction = _extract_trajectory_prediction(agent_output.main_trajectory)
            else:
                prediction = ''
        else:
            prediction = _extract_prediction(agent_output)

    prediction_val = ''
    if isinstance(prediction, str):
        prediction_val = prediction.strip()
    elif isinstance(prediction, dict) and 'content' in prediction:
        prediction_val = str(prediction['content']).strip()  # type: ignore
    elif prediction is not None:
        # ``final_answer`` is declared ``str | None``, and it stays None whenever the turn
        # carried no content — a cutoff inside the reasoning block, for one. Stringifying
        # that used to hand the judge the literal word 'None' as the model's answer.
        prediction_val = str(prediction).strip()

    # Evaluate
    is_valid_reward = True
    if skip_reward_for_truncation:
        score = _default_truncate_score(task_input.reward_system_extra)
        judge_details = {
            'skipped': 'agent_truncation',
            'termination_reason': termination_reason,
            'score': score,
        }
        span_id = get_current_span_id()
        if span_id:
            update_span(
                span_id,
                attributes={
                    'hytuner.termination_reason': termination_reason,
                    'hytuner.reward_time': 0.0,
                },
            )
    elif task_input.evaluator_type == 'none':
        score = task_input.evaluator_none_score
        judge_details = None
        logger.info('Skipping evaluator, using default score')
    elif cutoff_reason is not None:
        score = 0.0
        judge_details = None
        # main_agent already publishes this on its own span, but an agent that only
        # returns ``exit_status`` would otherwise reach HyTuner as ENV_DONE.
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': cutoff_reason})
        logger.warning(
            '[%s] rollout stopped before producing an answer; skipping evaluation. question=%s | partial prediction=%s',
            cutoff_reason,
            task_input.question,
            prediction_val,
        )
    else:
        if task_input.evaluator_type == 'reward_system':
            if not task_input.reward_system_router_id:
                raise ValueError('reward_system_router_id is required for reward_system evaluator')
            evaluator = RewardSystemEvaluator(
                user_token=task_input.reward_system_user_token,
                exp_id=task_input.reward_system_exp_id or '',
                router_id=task_input.reward_system_router_id,
                extra=task_input.reward_system_extra,
                public_test=task_input.reward_system_public_test,
                ci_test=task_input.reward_system_ci_test,
            )
            logger.debug('Using RewardSystemEvaluator for evaluation: router_id=%s', task_input.reward_system_router_id)
        elif task_input.evaluator_type == 'general':
            evaluator = GeneralEvaluator()
            logger.debug('Using GeneralEvaluator for evaluation')
        elif task_input.evaluator_type == 'hle':
            evaluator = HLEEvaluator()
            logger.debug('Using HLEEvaluator for evaluation')
        else:
            evaluator = SearchStatefulEvaluator(evaluator_type=task_input.evaluator_type)
            logger.debug(f'Using SearchStatefulEvaluator for evaluation: {task_input.evaluator_type}')
        start_time = time.time()
        score, _, is_valid_reward = await evaluate_output(
            prediction=prediction_val,
            reference=task_input.answer,
            question=task_input.question,
            evaluator=evaluator,
        )
        judge_details = evaluator.get_last_judge_result()
        reward_time = time.time() - start_time
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.reward_time': reward_time})
        if not is_valid_reward and span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})

        # Judge 基础设施故障（如 LLM judge 请求失败、5xx、网络错误、鉴权失败等）
        # 需要向上抛异常，避免被静默地记为 0 分而拉低整体指标.
        # 内容审核策略过滤（content_filtered:）属于合法的 0 分，正常返回.
        if judge_details is not None:
            judge_error = judge_details.get('error') or ''
            if isinstance(judge_error, str) and judge_error.startswith(JUDGE_INFRA_ERROR_PREFIX):
                if span_id:
                    update_span(span_id, attributes={'hytuner.termination_reason': 'JUDGE_INFRA_ERROR'})
                raise RuntimeError(f'Judge infrastructure failure, aborting task: {judge_error}')

    logger.info(
        f'Question: {task_input.question} | Prediction: {prediction_val} | Reference: {task_input.answer} | '
        f'Score: {score} | is_valid_reward: {is_valid_reward}'
    )

    return SearchStatefulTaskOutput(
        score=score,
        prediction=prediction_val,
        answer=task_input.answer,
        judge_details=judge_details,
    )
