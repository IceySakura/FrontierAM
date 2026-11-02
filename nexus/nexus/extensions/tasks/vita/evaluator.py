"""Async trajectory evaluator for VitaBench using sliding-window rubric-based evaluation."""

import copy
import json
import logging

from typing import Any

from nexus.extensions.tasks.vita.data_model.message import (
    Message,
    SystemMessage,
    UserMessage,
)
from nexus.extensions.tasks.vita.data_model.simulation import NLRubricCheck, RewardInfo, TerminationReason
from nexus.extensions.tasks.vita.data_model.tasks import EvaluationCriteria, RewardType, Task
from nexus.extensions.tasks.vita.llm_utils import generate
from nexus.extensions.tasks.vita.prompts import get_prompts
from nexus.extensions.tasks.vita.vita_utils import evaluator_extracter, get_weekday


logger = logging.getLogger(__name__)


def _initialize_rubric_states(evaluation_criteria: EvaluationCriteria) -> dict:
    """Initialize rubric states — all start as False (not met)."""
    rubric_states = {}
    rubric_idx = 0
    seen_rubrics: set[str] = set()

    if evaluation_criteria.expected_states:
        for expected_state in evaluation_criteria.expected_states:
            if hasattr(expected_state, 'state_rubrics') and expected_state.state_rubrics:
                for rubric in expected_state.state_rubrics:
                    if rubric in seen_rubrics:
                        continue
                    seen_rubrics.add(rubric)
                    rubric_states[f'rubric_{rubric_idx}'] = {
                        'rubric': rubric,
                        'justification': 'Not evaluated yet',
                        'meetExpectation': False,
                    }
                    rubric_idx += 1

    if evaluation_criteria.overall_rubrics:
        for rubric in evaluation_criteria.overall_rubrics:
            if rubric in seen_rubrics:
                continue
            seen_rubrics.add(rubric)
            rubric_states[f'rubric_{rubric_idx}'] = {
                'rubric': rubric,
                'justification': 'Not evaluated yet',
                'meetExpectation': False,
            }
            rubric_idx += 1

    return rubric_states


def _create_sliding_windows(messages: list[Message], window_size: int, overlap: int = 2) -> list[list[Message]]:
    """Create sliding windows from the message list with overlap."""
    if len(messages) <= window_size:
        return [messages]

    windows = []
    step = window_size - overlap
    i = 0
    while i < len(messages):
        window = messages[i : i + window_size]
        if window:
            windows.append(window)
        if i + window_size >= len(messages):
            break
        i += step
    return windows


def _format_window_content(window: list[Message], window_start_idx: int = 0) -> str:
    """Format window messages into a readable string with global message indices."""
    lines = []
    for i, message in enumerate(window):
        role = getattr(message, 'role', 'unknown')
        content = getattr(message, 'content', '') or ''
        full_content = content

        if role == 'assistant' and hasattr(message, 'tool_calls') and message.tool_calls:
            tool_strs = []
            for tc in message.tool_calls or []:  # type: ignore[union-attr]
                name = getattr(tc, 'name', 'unknown_tool') if hasattr(tc, 'name') else tc.get('name', 'unknown_tool')
                args = getattr(tc, 'arguments', {}) if hasattr(tc, 'arguments') else tc.get('arguments', {})
                args_str = ', '.join(f'{k}={v!r}' for k, v in args.items()) if isinstance(args, dict) else str(args)
                tool_strs.append(f'{name}({args_str})')
            if tool_strs:
                tc_text = '.'.join(tool_strs)
                full_content = f'{full_content} {tc_text}' if full_content else tc_text

        if full_content:
            global_idx = window_start_idx + i + 1
            lines.append(f'[{global_idx}] {role}: {full_content}')

    return '\n'.join(lines)


def _format_rubrics(current_states: dict) -> str:
    """Format current rubric states for LLM input."""
    rubrics_list = [
        {
            'rubric_idx': key,
            'rubric': state['rubric'],
            'justification': state['justification'],
            'meetExpectation': state['meetExpectation'],
        }
        for key, state in current_states.items()
    ]
    return json.dumps(rubrics_list, ensure_ascii=False, indent=2)


def _convert_states_to_checks(final_states: dict) -> list[NLRubricCheck]:
    """Convert final rubric states to NLRubricCheck objects."""
    return [
        NLRubricCheck(
            nl_rubric=state['rubric'],
            met=state['meetExpectation'],
            justification=state['justification'],
        )
        for state in final_states.values()
    ]


async def _evaluate_window(
    openai_client: Any,
    env_info: dict,
    task: Task,
    window: list[Message],
    current_states: dict,
    window_idx: int,
    total_windows: int,
    window_start_idx: int,
    language: str | None = None,
) -> tuple[dict, dict]:
    """Evaluate a single window and update rubric states."""
    window_content = _format_window_content(window, window_start_idx)
    current_rubrics_str = _format_rubrics(current_states)

    prompts = get_prompts(language)
    system_prompt = prompts.sliding_window_eval_template.format(  # type: ignore[attr-defined]
        env_info=env_info,
        user_instruction=task.instructions,
        window_idx=window_idx,
        total_windows=total_windows,
    )

    user_prompt = (
        f'# Input\n<window_content>\n{window_content}\n</window_content>\n\n'
        f'<current_rubrics>\n{current_rubrics_str}\n</current_rubrics>'
    )

    messages = [
        SystemMessage(role='system', content=system_prompt),
        UserMessage(role='user', content=user_prompt),
    ]

    assistant_message = await generate(
        openai_client=openai_client,
        messages=messages,
    )

    window_eval_info = {
        'window_idx': window_idx,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'assistant_message_content': assistant_message.content,
        'assistant_message_usage': assistant_message.usage,
    }

    updated_states = copy.deepcopy(current_states)
    result_data = evaluator_extracter(assistant_message.content or '')

    if result_data:
        for result in result_data:
            rubric_idx = result.get('rubric_idx')
            if rubric_idx and rubric_idx in updated_states:
                updated_states[rubric_idx]['justification'] = result.get('justification', 'No justification provided')
                updated_states[rubric_idx]['meetExpectation'] = result.get(
                    'meetExpectation', updated_states[rubric_idx]['meetExpectation']
                )
    else:
        logger.warning('Failed to parse LLM response for window %d, keeping current states', window_idx)

    return updated_states, window_eval_info


async def evaluate_trajectory(
    openai_client: Any,
    simulation_messages: list[Message],
    task: Task,
    termination_reason: str | None = None,
    evaluation_type: str = 'trajectory',
    language: str | None = None,
) -> RewardInfo:
    """Evaluate a VitaBench simulation trajectory.

    Returns RewardInfo with reward (0.0 or 1.0) and rubric-level details.
    """
    bad_terminations = {
        TerminationReason.TOO_MANY_ERRORS.value,
        TerminationReason.MAX_STEPS.value,
        TerminationReason.INVALID_AGENT_MESSAGE.value,
    }
    if termination_reason in bad_terminations:
        return RewardInfo(
            reward=0.0,
            nl_rubrics=None,
            reward_breakdown=None,
            info={'note': f'Simulation terminated prematurely: {termination_reason}'},
            window_evaluations=None,
        )

    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            nl_rubrics=None,
            reward_breakdown=None,
            info={'note': 'No evaluation criteria'},
            window_evaluations=None,
        )

    evaluation_criteria = task.evaluation_criteria
    if not evaluation_criteria.expected_states and not evaluation_criteria.overall_rubrics:
        return RewardInfo(
            reward=1.0,
            nl_rubrics=[],
            info={'note': 'No rubric to evaluate'},
            reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            window_evaluations=None,
        )

    env_info: dict[str, Any] = {'system_time': '', 'database': []}
    if hasattr(task, 'environment') and task.environment:
        time_str = task.environment.get('time', '')
        if time_str:
            weekday = get_weekday(str(time_str), language)
            env_info['system_time'] = f'{time_str} {weekday or ""}'

    if evaluation_type == 'trajectory':
        return await _evaluate_sliding_window(
            openai_client,
            env_info,
            task,
            simulation_messages,
            evaluation_criteria,
            language,
        )

    return await _evaluate_sliding_window(
        openai_client,
        env_info,
        task,
        simulation_messages,
        evaluation_criteria,
        language,
    )


async def _evaluate_sliding_window(
    openai_client: Any,
    env_info: dict,
    task: Task,
    full_trajectory: list[Message],
    evaluation_criteria: EvaluationCriteria,
    language: str | None = None,
    window_size: int = 10,
    overlap: int = 2,
) -> RewardInfo:
    """Sliding-window rubric-based evaluation."""
    current_rubric_states = _initialize_rubric_states(evaluation_criteria)
    windows = _create_sliding_windows(full_trajectory, window_size, overlap)
    step = window_size - overlap
    window_evaluations = []

    for i, window in enumerate(windows):
        logger.debug('Processing window %d/%d with %d messages', i + 1, len(windows), len(window))
        window_start_idx = i * step
        current_rubric_states, window_eval_info = await _evaluate_window(
            openai_client,
            env_info,
            task,
            window,
            current_rubric_states,
            i + 1,
            len(windows),
            window_start_idx,
            language,
        )
        window_evaluations.append(window_eval_info)

    final_checks = _convert_states_to_checks(current_rubric_states)
    all_met = all(r.met for r in final_checks) and len(final_checks) > 0
    rubric_score = sum(1.0 if r.met else 0.0 for r in final_checks) / len(final_checks) if final_checks else 0.0
    reward = 1.0 if all_met else 0.0

    return RewardInfo(
        reward=reward,
        nl_rubrics=final_checks,
        reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
        info={
            'evaluation_method': 'sliding_window',
            'num_windows': len(windows),
            'window_size': window_size,
        },
        window_evaluations=window_evaluations,
    )
