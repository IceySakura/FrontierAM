import logging
import os
import time

from typing import Any

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.agents.base.final_answer import generate_final_answer
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.tasks.search import search_stateful_shims
from nexus.extensions.tasks.search.evaluator import EvalMode, SearchEvaluator
from nexus.trace import get_current_span_id, update_span
from nexus.trace.phases import phase


logger = logging.getLogger(__name__)

WIDESEARCH_EVALUATION_REQUIRED_KEYS = frozenset({'required', 'unique_columns', 'eval_pipeline'})
JUDGE_EVAL_MODES = frozenset({
    EvalMode.LLM_JUDGE,
    EvalMode.DEEPSEARCH_QA,
    EvalMode.DRACO_RUBRIC_PER_CRITERION,
    EvalMode.DRACO_RUBRIC_ONESHOT,
    EvalMode.REWARD_SERVICE,
    EvalMode.WIDESEARCH,
})
JUDGE_STATUS_SUCCESS = 'success'
JUDGE_STATUS_FAILED = 'failed'


def _get_default_eval_mode() -> EvalMode:
    """Get default eval mode from environment variable SEARCH_EVAL_MODE."""
    env_mode = os.environ.get('SEARCH_EVAL_MODE', 'llm_judge')
    try:
        return EvalMode(env_mode)
    except ValueError:
        logger.warning(f"Invalid SEARCH_EVAL_MODE '{env_mode}', falling back to exact_match")
        return EvalMode.EXACT_MATCH


class SearchTaskInput(TaskRunInput, WithAgentRunner):
    """Search task input.

    Evaluation modes:
    - LLM_JUDGE: Use LLM for semantic equivalence (default, recommended)
    - EXACT_MATCH: Simple string comparison
    - REWARD_SERVICE: Use reward service API for evaluation (cost-effective)
    - WIDESEARCH: Structured table evaluation (auto-detected from dataset evaluation config)
    - INFERENCE_ONLY: Run the agent and produce a prediction, but skip all judging/scoring
    """

    question: str
    answer: str
    instance_id: str | None = None
    eval_mode: EvalMode = Field(default_factory=_get_default_eval_mode)
    judge_use_tool_calling: bool = False  # Whether the judge should use tool calling (if LLM_JUDGE mode)
    instruction_template: str | None = None
    run_oracle: bool = False
    soft_eval: bool = True
    soft_score: float = 1.0
    # Single-turn (last-turn-only) mode: when ``prefix_messages`` is set, the task
    # replays this full conversation history (system + user + assistant/tool turns)
    # as the agent prompt and the model only regenerates the final turn. Used with
    # ``agent_name='single_turn'``. ``tools`` carries the tool schemas so the chat
    # template still renders the historical tool calls / interleaved thinking
    # exactly as in the original multi-turn rollout.
    prefix_messages: list[dict] | None = None
    tools: list[dict] | None = None
    # LLM judge configuration
    judge_prompt_template: str | None = None
    judge_system_prompt: str | None = None
    # Training mode configuration
    training_length_exceeded: bool = bool(os.environ.get('TRAINING_LENGTH_EXCEEDED', False))
    enable_summary: bool = False  # Whether to summarize on MAX_ITERATIONS
    # WideSearch evaluation configuration (optional; auto-inferred when absent)
    evaluation: dict | None = None


class SearchTaskOutput(TaskRunOutput, WithScore):
    """Output containing both result and ground truth for immediate debuggability."""

    prediction: str
    answer: str
    trajectory: list[dict[str, Any]] | None = None
    evaluation_details: dict[str, Any] | None = None
    judge_status: str | None = None
    # Extra fields passed through to the runner / HyTuner (e.g. HTTP cache hit/miss
    # stats from the cc_search_agent_whitebox web tools). Owned here rather than in
    # the generic runner so the runner stays agent-agnostic.
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _resolve_eval_mode(task_input: SearchTaskInput) -> EvalMode:
    """Only force WideSearch samples to use ``wide_search`` eval mode."""
    evaluation = task_input.evaluation
    if isinstance(evaluation, dict) and WIDESEARCH_EVALUATION_REQUIRED_KEYS.issubset(evaluation):
        return EvalMode.WIDESEARCH
    return task_input.eval_mode


def _resolve_judge_status(eval_mode: EvalMode, evaluation_details: dict[str, Any] | None) -> str | None:
    """Return the instance-level judge status for modes that depend on an external judge."""
    if eval_mode not in JUDGE_EVAL_MODES:
        return None
    if evaluation_details and evaluation_details.get('status') == JUDGE_STATUS_FAILED:
        return JUDGE_STATUS_FAILED
    return JUDGE_STATUS_SUCCESS


def extract_agent_result(agent_output: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Normalize agent outputs into a trajectory and optional final answer."""
    if isinstance(agent_output, WithMainTrajectory):
        return agent_output.main_trajectory, getattr(agent_output, 'final_answer', None)

    raw_output = getattr(agent_output, 'output', None)
    if not isinstance(raw_output, str):
        raise RuntimeError(f'Agent output invalid type: {type(agent_output)}')

    from nexus.extensions.tasks.search.utils import parse_claude_stream_output

    return parse_claude_stream_output(raw_output)


@task_registry.register('search')
async def search_task(task_input: SearchTaskInput) -> SearchTaskOutput:
    """
    Executes a search task and evaluates the prediction against the answer.
    """
    # Reset per-trajectory HTTP cache stats before any tool runs. Imported locally
    # (not at module top) so the generic runner never has to depend on this agent's
    # tools; the counters are read back via ``get_cache_stats`` before returning.
    from nexus.extensions.agents.cc_search_agent_whitebox.tools.http_cache import get_cache_stats, reset_cache_stats

    reset_cache_stats()

    question = task_input.question
    instance_id = task_input.instance_id
    resolved_eval_mode = _resolve_eval_mode(task_input)
    logger.debug(
        'Search task started: requested_eval_mode=%r, resolved_eval_mode=%r, instance_id=%s, question=%s',
        task_input.eval_mode,
        resolved_eval_mode,
        instance_id,
        question,
    )
    # Standardize ground truth once
    target_answer = task_input.answer.strip().lower()

    trajectory = None
    evaluation_details = None
    judge_status = None
    right_score = 1.0

    # 1. Execution
    if task_input.run_oracle:
        logger.debug(f'Oracle Mode: Bypassing agent for "{question[:30]}..."')
        final_answer = target_answer
    else:
        try:
            ctx = require_context()
        except RuntimeError:
            ctx = None
        if ctx is not None:
            search_stateful_shims.prepare_tool_configs(ctx.tool_configs)
        template = task_input.instruction_template
        if template is None:
            logger.warning('instruction_template is None, falling back to question as prompt')
        prompt = template.replace('{question}', question) if template else question
        phase('task.agent')
        if task_input.prefix_messages:
            # Single-turn last-turn-only mode: replay the full history and let the
            # agent (single_turn) regenerate only the final turn. Pass tools so the
            # chat template renders historical tool calls / interleaved thinking
            # identically to the original rollout.
            agent_output = await task_input.run_agent(prompt=task_input.prefix_messages, tools=task_input.tools)
        else:
            agent_output = await task_input.run_agent(user_prompt=prompt, prompt=[{'role': 'user', 'content': prompt}])
        trajectory, parsed_final_answer = extract_agent_result(agent_output)

        # Handle training length exceeded case
        exit_status = getattr(agent_output, 'exit_status', None)
        termination_reason = None

        # Propagate termination_reason to HyTuner via span
        if exit_status is not None:
            # cc_search_agent_whitebox delegates to main_agent and returns the raw lowercase
            # MainExitStatus value (e.g. 'finished' / 'context_length_exceeded'); other agents
            # return the uppercase base TerminationReason value (e.g. 'ENV_DONE'). Normalize the
            # lowercase MainExitStatus form to the canonical uppercase hytuner termination_reason
            # so downstream stats / maskout see a single tag. Values that are already uppercase
            # are not MainExitStatus members, so termination_reason_for returns 'UNKNOWN' — keep
            # the original value in that case.
            raw_termination_reason = (
                exit_status.value if isinstance(exit_status, TerminationReason) else str(exit_status)
            )
            termination_reason = MainExitStatus.termination_reason_for(raw_termination_reason)
            if termination_reason == 'UNKNOWN':
                termination_reason = raw_termination_reason
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'hytuner.termination_reason': termination_reason})

        if task_input.training_length_exceeded and termination_reason in {
            TerminationReason.TRUNCATION.value,
            TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,
        }:
            assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
            logger.info(f'[TerminationReason: {termination_reason}] turn: {assistant_turns}, question: {question}')
            final_answer = getattr(agent_output, 'final_answer', None) or parsed_final_answer or ''
            # close 'task.agent' phase span before early return
            phase()
            return SearchTaskOutput(
                score=0.0,
                prediction=final_answer,
                answer=target_answer,
                trajectory=trajectory,
                extra_fields={'cache_stats': get_cache_stats() or {}},
            )

        should_summary = task_input.enable_summary and exit_status == TerminationReason.MAX_ITERATIONS
        # DeepSearch QA 评估始终通过 generate_final_answer 产出预测（强制走 summary 路径），
        # 不依赖 enable_summary 开关与 MAX_ITERATIONS 退出状态。
        if resolved_eval_mode == EvalMode.DEEPSEARCH_QA:
            should_summary = True
        if (
            resolved_eval_mode
            in (
                EvalMode.LLM_JUDGE,
                EvalMode.DEEPSEARCH_QA,
                EvalMode.REWARD_SERVICE,
                EvalMode.WIDESEARCH,
                EvalMode.DRACO_RUBRIC_PER_CRITERION,
                EvalMode.DRACO_RUBRIC_ONESHOT,
                EvalMode.INFERENCE_ONLY,
            )
            and not should_summary
        ):
            final_answer = getattr(agent_output, 'final_answer', None) or parsed_final_answer
            if final_answer is None and task_input.soft_eval:
                last_msg = trajectory[-1]
                raw_content = last_msg.get('content') or last_msg.get('reasoning_content')
                final_answer = extract_text_from_content(raw_content)
                right_score = task_input.soft_score
        elif resolved_eval_mode == EvalMode.EXACT_MATCH or should_summary:
            raw_pred = await generate_final_answer(
                trajectory=trajectory,
                return_type='str',
            )
            final_answer = str(raw_pred).strip().lower()
        else:
            raise RuntimeError(f'Unsupported eval_mode: {resolved_eval_mode}')

    # 2. Evaluation
    # inference_only 模式：只产出预测，不做任何 judge/打分，直接返回。
    if resolved_eval_mode == EvalMode.INFERENCE_ONLY:
        phase()
        logger.info(
            'InstanceID: %s | inference_only mode: skipping evaluation | Prediction: %s',
            instance_id or 'N/A',
            final_answer,
        )
        return SearchTaskOutput(
            score=1.0,
            prediction=final_answer or '',
            answer=target_answer,
            trajectory=trajectory,
            extra_fields={'cache_stats': get_cache_stats() or {}},
        )

    effective_eval_mode = EvalMode.EXACT_MATCH if task_input.run_oracle else resolved_eval_mode
    if final_answer is None or final_answer == '':
        score = 0.0
        if effective_eval_mode in JUDGE_EVAL_MODES:
            evaluation_details = {
                'status': JUDGE_STATUS_FAILED,
                'msg': 'Prediction is empty; judge evaluation skipped.',
            }
            judge_status = JUDGE_STATUS_FAILED
        # 对于需要 details 的评估模式，即使推理失败也返回零值 details，
        # 确保 avg_evaluation_details 的分母与 accuracy 的分母一致。
        if effective_eval_mode == EvalMode.DEEPSEARCH_QA:
            from nexus.extensions.tasks.search.deepsearch_qa_evaluation.evaluation import DeepsearchQaDetails

            details_obj = DeepsearchQaDetails(
                msg='Empty prediction, skipping evaluation',
                precision=0.0,
                recall=0.0,
                f1=0.0,
            )
            evaluation_details = details_obj.to_dict()
            judge_status = JUDGE_STATUS_FAILED
        elif effective_eval_mode in (EvalMode.DRACO_RUBRIC_PER_CRITERION, EvalMode.DRACO_RUBRIC_ONESHOT):
            from nexus.extensions.tasks.search.draco_rubric_evaluation.evaluation import DracoRubricDetails

            details_obj = DracoRubricDetails(
                msg='Empty prediction, skipping evaluation',
                normalized_score=0.0,
                raw_score=0.0,
            )
            evaluation_details = details_obj.to_dict()
            judge_status = JUDGE_STATUS_FAILED
    else:
        with phase('task.scoring'):
            start_time = time.time()
            evaluator = SearchEvaluator(
                effective_eval_mode,
                judge_use_tool_calling=task_input.judge_use_tool_calling,
                judge_prompt_template=task_input.judge_prompt_template,
                judge_system_prompt=task_input.judge_system_prompt,
                widesearch_evaluation=task_input.evaluation,
                instance_id=instance_id or 'unknown',
            )
            score, details_obj, is_valid_reward = await evaluate_output(
                prediction=final_answer,
                reference=target_answer,
                question=question,
                evaluator=evaluator,
                need_details=True,
            )
            reward_time = time.time() - start_time
        # 如果是软评测，且评测结果为1.0，则使用软评测分数
        if score == 1.0:
            score = right_score
        evaluation_details = details_obj.to_dict() if details_obj else None
        judge_status = _resolve_judge_status(effective_eval_mode, evaluation_details)

        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.reward_time': reward_time})
        if not is_valid_reward and span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})

        logger.warning(
            f'InstanceID: {instance_id or "N/A"} | Score: {score} | is_valid_reward: {is_valid_reward} | '
            f'Question: {question} | Prediction: {final_answer} | Reference: {target_answer} | '
            f'Evaluation Details: {evaluation_details} | Judge Status: {judge_status}'
        )
    # close remaining phase spans
    phase()
    return SearchTaskOutput(
        score=score,
        prediction=final_answer or '',
        answer=target_answer,
        trajectory=trajectory,
        evaluation_details=evaluation_details,
        judge_status=judge_status,
        extra_fields={'cache_stats': get_cache_stats() or {}},
    )
