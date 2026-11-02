"""``yb_vlm_oneagent`` task — entry point for oneagent rollouts.

Composition (no new agent — this is purely a task that drives the
existing ``main_agent`` runner):

* Protocol-level input shaping, context-budget defaults and ``ctx.tool_state``
  seeding live in ``protocol.py``.
* ``run_agent`` is invoked with the package's post-iteration hook and the
  pinned oneagent tool set.
* Reward / abnormal-exit handling is delegated to the same code path
  that ``mm_agent_task`` uses; we just import-and-reuse instead of
  forking the implementation.
"""

# NOTE: Do NOT add ``from __future__ import annotations`` to this module.
# ``nexus.core.task_registry`` validates the registered task at import
# time via ``inspect.signature(...)``; under PEP 563 deferred evaluation
# the parameter / return annotations would be plain strings (e.g.
# ``"SearchTaskInput"``) instead of class objects, and the validator's
# ``inspect.isclass(...) and issubclass(..., TaskRunInput)`` check would
# fail with ``RegistryError: Processing failed for 'yb_vlm_oneagent'
# in registry 'task': Task input parameter must be a subclass of
# TaskRunInput, got SearchTaskInput``. Keep annotations eagerly evaluated.

import logging
import os
import time

from typing import Any, Literal, cast

from nexus.core import (
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.projects.yb_vlm_oneagent.context_manage import TokenCounter
from nexus.extensions.projects.yb_vlm_oneagent.protocol import (
    HOOK_PATH,
    PINNED_TOOLS,
    build_initial_payload,
    env_bool,
    env_int,
    extract_agent_config_from_run_agent,
    seed_tool_state,
    validate_pinned_tools,
)
from nexus.extensions.tasks.mm_agent.reward_system import vl_reward_system
from nexus.extensions.tasks.mm_agent.task import (
    _ALWAYS_MASK_STATUSES,
    _ALWAYS_ZERO_STATUSES,
    _UNJUDGED_SCORE,
    SearchTaskOutput,
    _emit_task_span_attrs,
    extract_final_answer,
)

# Symbols we deliberately reuse (per the project's "no duplication"
# directive in the task handover doc — every helper listed here lives
# verbatim in mm_agent.task.py).
from nexus.extensions.tasks.mm_agent.task import (  # noqa: F401
    SearchTaskInput as MMSearchTaskInput,
)


logger = logging.getLogger(__name__)

_WEB_SEARCH_TOOL = 'web_search'
_VISION_SEARCH_TOOL = 'vision_search'
_IN_PROCESS_REWARD_MODE = 'in_process'
_DEFERRED_REWARD_MODE = 'deferred'
_DEFERRED_REWARD_FUNCS = frozenset({'yb_rubric'})


class YbVlmOneAgentTaskOutput(SearchTaskOutput):
    """Yb oneagent output carrying HyTuner-only extra fields."""

    extra_fields: dict[str, Any]


class YbVlmOneAgentTaskInput(MMSearchTaskInput):
    """Yb oneagent input carrying task-local prompt fields."""

    location_info: str | None = None
    # True when this rollout is part of a validation batch (as opposed to
    # a training rollout).
    validate: bool = False
    # Reward execution mode:
    # - in_process: score inside nexus via vl_reward_system (default)
    # - deferred: hand reward ownership to hytuner reward-loop / reward platform
    reward_mode: Literal['in_process', 'deferred'] = _IN_PROCESS_REWARD_MODE


def _extract_assistant_tool_names(trajectory: list[dict[str, Any]]) -> list[str]:
    tool_names: list[str] = []
    for msg in trajectory:
        if msg.get('role') != 'assistant':
            continue
        raw_tool_calls = msg.get('tool_calls')
        if not isinstance(raw_tool_calls, list):
            continue
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            raw_function = tool_call.get('function')
            if not isinstance(raw_function, dict):
                continue
            name = raw_function.get('name')
            if isinstance(name, str) and name:
                tool_names.append(name)
    return tool_names


def _count_tool_call_rounds(trajectory: list[dict[str, Any]]) -> int:
    rounds = 0
    for msg in trajectory:
        if msg.get('role') != 'assistant':
            continue
        raw_tool_calls = msg.get('tool_calls')
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            rounds += 1
    return rounds


def _build_rollout_extra_fields(
    *,
    trajectory: list[dict[str, Any]],
    prediction: str,
    token_counter: TokenCounter,
) -> dict[str, float]:
    tool_names = _extract_assistant_tool_names(trajectory)
    return {
        'tool_call_counts': float(len(tool_names)),
        'tool_call_rounds': float(_count_tool_call_rounds(trajectory)),
        'web_search_call_count': float(tool_names.count(_WEB_SEARCH_TOOL)),
        'vision_search_call_count': float(tool_names.count(_VISION_SEARCH_TOOL)),
        'web_search_call_rate': float(_WEB_SEARCH_TOOL in tool_names),
        'vision_search_call_rate': float(_VISION_SEARCH_TOOL in tool_names),
        'final_answer_length': float(token_counter.count_text(prediction)),
    }


def _should_defer_reward(task_input: YbVlmOneAgentTaskInput) -> bool:
    if task_input.reward_mode != _DEFERRED_REWARD_MODE:
        return False
    if task_input.reward_func not in _DEFERRED_REWARD_FUNCS:
        raise ValueError(
            f'reward_mode=deferred is not supported for reward_func={task_input.reward_func!r}; supported: '
            f'{sorted(_DEFERRED_REWARD_FUNCS)}'
        )
    return True


def _build_hytuner_reward_fields(
    *,
    question: str,
    prediction: str,
    target_answer: str,
    exit_status: str | None,
    reward_mode: str,
    reward_func: str | None,
) -> dict[str, Any]:
    return {
        'reward_decided_by_hytuner': True,
        'reward_mode': reward_mode,
        'reward_func': reward_func,
        'reward_exit_status': exit_status or MainExitStatus.FINISHED.value,
        'reward_system_question': question,
        'reward_system_response': prediction,
        'reward_system_target_answer': target_answer,
    }


@task_registry.register('yb_vlm_oneagent')
async def yb_vlm_oneagent_task(task_input: YbVlmOneAgentTaskInput) -> SearchTaskOutput:
    """Drive a single oneagent rollout via the shared ``main_agent`` runner.

    Behaviour parity with ``mm_agent_task``:

    * Same abnormal-exit handling (``_ALWAYS_MASK_STATUSES`` /
      ``_ALWAYS_ZERO_STATUSES``), ``REWARD_ERROR`` rewrite, span
      emission, and final ``SearchTaskOutput`` shape.
    * Same reward routing through ``vl_reward_system`` selected by
      ``task_input.reward_func``.
    * Same completion-bonus / image-search-bonus env knobs (re-used as
      ``MM_AGENT_COMPLETION_BONUS_SCORE`` / ``MM_AGENT_IMAGE_SEARCH_BONUS_SCORE``
      to keep deployment configs cross-compatible).

    Behaviour deviations specific to oneagent:

    * The system prompt, history and current user message are built from the
      mm_agent-style ``system_prompt`` / ``history`` / ``question`` / ``images``
      fields, then passed through the oneagent context manager inside this task.
    * Tools are pinned to the oneagent set (``yb_web_search`` /
      ``yb_vision_search``); the schema-facing names that the model
      sees stay ``web_search`` / ``vision_search`` (set on the tool
      registrations themselves).
    * ``post_iteration_hook = budget_and_finish_hook`` and
      ``tool_output_truncation = None`` so per-round budgeting is
      handled by the hook, not by the engine.
    * Force-answer is *opt-in* per task instance (same flag as
      mm_agent's force-finish), but the hook injects ``<answer>`` —
      not a user prompt — to align byte-for-byte with AngelAgent's
      ``search_agent_runner.py``.
    """
    question = task_input.question
    instance_id = task_input.instance_id
    logger.debug(
        f'YB oneagent task started: reward_func={task_input.reward_func!r}, '
        f'instance_id={instance_id}, question={question}'
    )
    target_answer = task_input.answer.strip()
    _defer_reward = _should_defer_reward(task_input)

    # ----- Seed ctx.tool_state for tools + hook -----
    ctx = require_context()

    # 用 rollout client 的 tokenizer 构造 TokenCounter，经 tool_state 显式注入给
    # 两个消费方：yb_vision_search（算图片 token）与 budget_and_finish_hook
    # （apply_tool_round_budget）。client 必带 .tokenizer，缺失即配置错误，直接崩溃。
    openai_client = cast(Any, ctx.openai_client)
    token_counter = TokenCounter(openai_client.tokenizer)
    payload = build_initial_payload(task_input, token_counter)

    max_iterations = env_int('MAX_ITERATIONS', 10)
    enable_force_finish = env_bool('YB_VLM_ENABLE_FORCE_FINISH', task_input.enable_force_finish)
    seed_tool_state(
        ctx,
        question=question,
        payload=payload,
        token_counter=token_counter,
        enable_force_finish=enable_force_finish,
    )

    if not payload.system_prompt:
        logger.warning(
            'YB oneagent task: empty system_prompt — task_config.system_prompt may be '
            'missing the __YB_VLM_SYSTEM_PROMPT__ marker'
        )

    # Tool-set guard: if the run_input declares ``agent_config.tools``,
    # it MUST match the two pinned tools (or be omitted entirely). We
    # still pass ``tools=[...]`` to ``run_agent`` below for defence in
    # depth — runner-side merge order already lets the explicit kwarg
    # win — but a mismatched yaml is almost always a dataset/config
    # bug worth surfacing immediately.
    agent_config = extract_agent_config_from_run_agent(task_input.run_agent)
    validate_pinned_tools(agent_config)

    agent_output = await task_input.run_agent(
        user_prompt=payload.user_prompt,
        system_prompt=payload.system_prompt,
        # Pass the dataset-provided conversation history so the agent
        # sees prior turns and training prompt_ids stay consistent
        # with the validation inputs (which use raw_prompt directly).
        history=payload.history,
        post_iteration_hook=HOOK_PATH,
        # The engine must NEVER truncate tool messages for oneagent —
        # all per-round budgeting is doc-granular and lives in the
        # hook. ``None`` disables ``maybe_truncate`` on tool output.
        tool_output_truncation=None,
        # Pin the oneagent tool set; parser, retries and image-token guard
        # are left at the agent's defaults so behaviour matches the mm_agent
        # baseline. Iteration count is a training runtime knob.
        tools=list(PINNED_TOOLS),
        max_iterations=max_iterations,
    )

    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError(f'Agent output invalid type: {type(agent_output)}')

    trajectory = agent_output.main_trajectory

    # -------------------------------------------------------------------
    # Abnormal-exit handling — same logic, status sets and span emission
    # as mm_agent_task. The two implementations stay byte-aligned
    # because we import the constants directly from mm_agent.task.
    # -------------------------------------------------------------------
    exit_status = getattr(agent_output, 'exit_status', None)
    exit_status_value = getattr(exit_status, 'value', exit_status or '')

    if exit_status_value in _ALWAYS_MASK_STATUSES:
        assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
        logger.info(
            f'[ExitStatus: {exit_status_value}] Masked (score={_UNJUDGED_SCORE}), '
            f'turn={assistant_turns}, question={question}'
        )
        final_answer = getattr(agent_output, 'final_answer', None) or ''
        rollout_extra_fields = _build_rollout_extra_fields(
            trajectory=trajectory,
            prediction=final_answer,
            token_counter=token_counter,
        )
        if _defer_reward:
            rollout_extra_fields = {
                **rollout_extra_fields,
                **_build_hytuner_reward_fields(
                    question=question,
                    prediction=final_answer,
                    target_answer=target_answer,
                    exit_status=exit_status_value,
                    reward_mode=task_input.reward_mode,
                    reward_func=task_input.reward_func,
                ),
            }
        _emit_task_span_attrs(exit_status=exit_status_value)
        return YbVlmOneAgentTaskOutput(
            score=0.0 if _defer_reward else _UNJUDGED_SCORE,
            prediction=final_answer,
            answer=target_answer,
            trajectory=trajectory,
            evaluation_details={
                'reward_mode': task_input.reward_mode,
                'scoring_mode': 'deferred_to_hytuner' if _defer_reward else 'nexus_exit_short_circuit',
                'exit_status': exit_status_value,
            },
            extra_fields={
                **rollout_extra_fields,
            },
            exit_status=exit_status_value,
            instance_id=task_input.instance_id,
            reward_func=task_input.reward_func,
            question=question,
            system_prompt_template='yb_vlm_oneagent',
        )

    if exit_status_value in _ALWAYS_ZERO_STATUSES:
        assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
        logger.info(
            f'[ExitStatus: {exit_status_value}] Short-circuited (score=0.0), '
            f'turn={assistant_turns}, question={question}'
        )
        final_answer = getattr(agent_output, 'final_answer', None) or ''
        rollout_extra_fields = _build_rollout_extra_fields(
            trajectory=trajectory,
            prediction=final_answer,
            token_counter=token_counter,
        )
        if _defer_reward:
            rollout_extra_fields = {
                **rollout_extra_fields,
                **_build_hytuner_reward_fields(
                    question=question,
                    prediction=final_answer,
                    target_answer=target_answer,
                    exit_status=exit_status_value,
                    reward_mode=task_input.reward_mode,
                    reward_func=task_input.reward_func,
                ),
            }
        _emit_task_span_attrs(exit_status=exit_status_value)
        return YbVlmOneAgentTaskOutput(
            score=0.0,
            prediction=final_answer,
            answer=target_answer,
            trajectory=trajectory,
            evaluation_details={
                'reward_mode': task_input.reward_mode,
                'scoring_mode': 'deferred_to_hytuner' if _defer_reward else 'nexus_exit_short_circuit',
                'exit_status': exit_status_value,
            },
            extra_fields={
                **rollout_extra_fields,
            },
            exit_status=exit_status_value,
            instance_id=task_input.instance_id,
            reward_func=task_input.reward_func,
            question=question,
            system_prompt_template='yb_vlm_oneagent',
        )

    # ----- Extract prediction from the last assistant turn -----
    last_msg = agent_output.main_trajectory[-1]
    raw_content = last_msg.get('content') or last_msg.get('reasoning_content')
    prediction_val = extract_text_from_content(raw_content) if raw_content is not None else ''
    prediction_val = extract_final_answer(prediction_val)
    logger.debug(f'[YB oneagent] Extracted prediction: {prediction_val}')

    rollout_extra_fields = _build_rollout_extra_fields(
        trajectory=trajectory,
        prediction=prediction_val,
        token_counter=token_counter,
    )

    if _defer_reward:
        exit_status_value = exit_status_value or MainExitStatus.FINISHED.value
        logger.debug(
            f'[DeferredReward] reward_func={task_input.reward_func!r}, reward_mode={task_input.reward_mode!r}; '
            f'delegating reward ownership to hytuner. instance_id={instance_id}'
        )
        _emit_task_span_attrs(exit_status=exit_status_value)
        return YbVlmOneAgentTaskOutput(
            score=0.0,
            prediction=prediction_val,
            answer=target_answer,
            trajectory=trajectory,
            evaluation_details={
                'reward_func': task_input.reward_func,
                'reward_mode': task_input.reward_mode,
                'scoring_mode': 'deferred_to_hytuner',
            },
            extra_fields={
                **rollout_extra_fields,
                **_build_hytuner_reward_fields(
                    question=question,
                    prediction=prediction_val,
                    target_answer=target_answer,
                    exit_status=exit_status_value,
                    reward_mode=task_input.reward_mode,
                    reward_func=task_input.reward_func,
                ),
            },
            exit_status=exit_status_value,
            instance_id=task_input.instance_id,
            reward_func=task_input.reward_func,
            question=question,
            system_prompt_template='yb_vlm_oneagent',
        )

    # ----- Reward -----
    reward_start = time.monotonic()
    score, evaluation_details = await vl_reward_system(
        task_input.reward_func,
        question=question,
        response=prediction_val,
        target_answer=target_answer,
        trajectory=agent_output.main_trajectory,
        history=task_input.history,
        # Forwarded to per-reward routines (e.g. yb_rubric) so validation
        # rollouts can opt into a larger rubric-judge round count via
        # YB_RUBRIC_N_ROUNDS_VAL. Injected upstream by nexus_agent_loop.
        validate=task_input.validate,
    )
    reward_time = time.monotonic() - reward_start

    logger.info(
        f'**vl_reward_system** InstanceID: {instance_id or "N/A"} | Score: {score} | '
        f'Question: {question} | Prediction: {prediction_val} | Reference: {target_answer} | '
        f'Evaluation Details: {evaluation_details}'
    )

    if score is None or float(score) <= -99.0:
        logger.warning(
            f'[RewardError] vl_reward_system returned invalid score={score} '
            f'(<= -99 or None); marking exit_status=REWARD_ERROR. '
            f'instance_id={instance_id}, reward_func={task_input.reward_func!r}'
        )
        score = _UNJUDGED_SCORE
        exit_status_value = 'REWARD_ERROR'
    else:
        exit_status_value = exit_status_value or MainExitStatus.FINISHED.value
        # ``MM_AGENT_COMPLETION_BONUS_SCORE`` / ``MM_AGENT_IMAGE_SEARCH_BONUS_SCORE``
        # remain valid here so the same deployment knob covers both
        # mm_agent and yb_vlm_oneagent rollouts. We deliberately
        # do NOT mint new env vars for the same semantics.
        _bonus_env = os.environ.get('MM_AGENT_COMPLETION_BONUS_SCORE')
        if _bonus_env is not None and _bonus_env.strip() != '':
            try:
                _bonus_val = float(_bonus_env)
            except ValueError:
                logger.warning(f'[CompletionBonus] Ignoring non-numeric MM_AGENT_COMPLETION_BONUS_SCORE={_bonus_env!r}')
                _bonus_val = None
            if (
                _bonus_val is not None
                and exit_status_value == MainExitStatus.FINISHED.value
                and float(score) < _bonus_val
            ):
                logger.info(
                    f'[CompletionBonus] Raising score {score} -> {_bonus_val} '
                    f'(FINISHED but judged wrong). instance_id={instance_id}, '
                    f'reward_func={task_input.reward_func!r}'
                )
                score = _bonus_val

    _emit_task_span_attrs(exit_status=exit_status_value, reward_time=reward_time)

    return YbVlmOneAgentTaskOutput(
        score=score,
        prediction=prediction_val,
        answer=target_answer,
        trajectory=trajectory,
        evaluation_details=evaluation_details,
        extra_fields={
            **rollout_extra_fields,
        },
        exit_status=exit_status_value,
        instance_id=task_input.instance_id,
        reward_func=task_input.reward_func,
        question=question,
        system_prompt_template='yb_vlm_oneagent',
    )
