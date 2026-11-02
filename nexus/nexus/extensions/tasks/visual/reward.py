"""Optional visual-task scoring through the shared reward-system client."""

import asyncio
import logging
import math
import os
import re
import time

from typing import TYPE_CHECKING, Any

from aiohttp import ClientError

from nexus.core import AgentRunOutput
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.agents.visual_agent.resources.images import (
    _part_image_ref,
    _with_image_ref,
    image_data_uri,
    validate_reward_media,
)
from nexus.extensions.compaction.runtime_state import is_compaction_runtime_state_message
from nexus.trace import get_current_span_id, update_span
from nexus.utils.reward_score import ReasoningReward
from nexus.utils.reward_system import RewardSystemClient, resolve_strip_multimodal_config


if TYPE_CHECKING:
    from nexus.extensions.tasks.visual.task import VisualTaskInput


logger = logging.getLogger(__name__)

# Infrastructure failures are unjudged; model errors and interrupted trajectories receive zero.
_MASKED_EXITS = frozenset({
    MainExitStatus.TOOL_EXECUTION_ERROR.value,
    MainExitStatus.INFERENCE_ERROR.value,
    MainExitStatus.INFERENCE_TIMEOUT.value,
    MainExitStatus.OBSERVATION_TIMEOUT.value,
    MainExitStatus.RUNTIME_ERROR.value,
})
_ZERO_EXITS = frozenset({
    MainExitStatus.TOOL_ERROR.value,
    MainExitStatus.MAX_ITERATIONS.value,
    MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value,
    MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value,
})


def create_reward_client() -> RewardSystemClient:
    timeout = float(os.environ.get('REWARD_PLATFORM_TIMEOUT', '180'))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('REWARD_PLATFORM_TIMEOUT must be a finite positive number of seconds')
    return RewardSystemClient(timeout=timeout, max_retries=1, retry_delay=0)


def _prediction(output: AgentRunOutput) -> str:
    value = getattr(output, 'final_answer', None)
    if value is None:
        trajectory = getattr(output, 'main_trajectory', None) or []
        if trajectory and trajectory[-1].get('role') == 'assistant' and not trajectory[-1].get('tool_calls'):
            value = trajectory[-1].get('content') or trajectory[-1].get('reasoning_content')
        else:
            value = getattr(output, 'output', None)
    return extract_text_from_content(value or '').rsplit(os.environ.get('THINK_END_TOKEN', '</think>'), 1)[-1].strip()


def _image_url(source: str) -> str:
    if source.startswith(('http://', 'https://')):
        return source
    return image_data_uri(source)


async def _reward_content(content: Any, strip_multimodal: bool) -> Any:
    if not isinstance(content, list):
        return content
    if strip_multimodal:
        return extract_text_from_content(content)
    validate_reward_media(content)
    parts = list(content)
    for index, part in enumerate(parts):
        if source := _part_image_ref(part):
            parts[index] = _with_image_ref(part, await asyncio.to_thread(_image_url, source))
    return parts


_TRAJECTORY_KEYS = ('role', 'content', 'tool_calls', 'tool_call_id')


def _validate_tool_pairs(messages: list[dict[str, Any]]) -> None:
    pending: set[str] = set()
    for message in messages:
        if message.get('role') == 'tool':
            call_id = message.get('tool_call_id')
            if call_id not in pending:
                raise ValueError('Reward trajectory contains an unmatched tool response')
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError('Reward trajectory contains tool calls without responses')
            if message.get('role') == 'assistant':
                calls = message.get('tool_calls') or []
                for call in calls:
                    call_id = call.get('id')
                    if not isinstance(call_id, str) or not call_id or call_id in pending:
                        raise ValueError('Reward trajectory contains invalid or duplicate tool call IDs')
                    pending.add(call_id)
    if pending:
        raise ValueError('Reward trajectory contains tool calls without responses')


def _without_thinking(content: Any) -> Any:
    marker = os.environ.get('THINK_END_TOKEN', '</think>')
    if isinstance(content, str):
        return content.rsplit(marker, 1)[-1]
    if isinstance(content, list):
        parts = []
        found_end = False
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                text = part['text']
                if found_end:
                    text = ''
                elif marker in text:
                    text = text.rsplit(marker, 1)[-1]
                    found_end = True
                part = {**part, 'text': text}
            parts.append(part)
        return list(reversed(parts))
    return content


async def _trajectory_messages(
    output: AgentRunOutput, max_tool_images: int, strip_multimodal: bool
) -> tuple[list[dict[str, Any]], int]:
    """Use the agent's explicit boundary; compacted inference context is not a full scoring trajectory."""
    if getattr(output, 'trajectory_compacted', False):
        raise ValueError('Reward trajectory scoring does not support compacted trajectories')
    trajectory = getattr(output, 'main_trajectory', None)
    start = getattr(output, 'generated_message_start', None)
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError('reward_system_include_trajectory requires a non-empty main_trajectory')
    if type(start) is not int or not 1 <= start < len(trajectory):
        raise ValueError('Reward trajectory requires a valid generated_message_start from the agent')
    if trajectory[start - 1].get('role') != 'user':
        raise ValueError('Reward trajectory boundary must follow the user prompt')
    generated = [message for message in trajectory[start:] if not is_compaction_runtime_state_message(message)]
    if any(message.get('_is_compacted_summary') for message in trajectory):
        raise ValueError('Reward trajectory scoring does not support compacted trajectories')
    if not generated or any(message.get('role') not in {'assistant', 'tool', 'user'} for message in generated):
        raise ValueError('Reward trajectory contains no generated messages or an unsupported role')
    if getattr(output, 'exit_status', None) == MainExitStatus.REFUSED.value and generated[-1]['role'] == 'assistant':
        # The agent preserves provider tool calls on refusal, but never executes them.
        generated[-1] = {key: value for key, value in generated[-1].items() if key != 'tool_calls'}
    _validate_tool_pairs(generated)
    if generated and generated[-1]['role'] == 'assistant' and not generated[-1].get('tool_calls'):
        generated.pop()  # replaced by the cleaned final answer
    slots = [
        (i, j)
        for i, message in enumerate(generated)
        if not strip_multimodal and message['role'] == 'tool' and isinstance(message.get('content'), list)
        for j, part in enumerate(message['content'])
        if _part_image_ref(part)
    ]
    kept = set(slots[-max_tool_images:]) if max_tool_images else set()
    messages = []
    for i, source in enumerate(generated):
        message = {key: source[key] for key in _TRAJECTORY_KEYS if key in source}
        content = message.get('content')
        if content is None:
            content = ''
        if message['role'] == 'assistant':
            content = _without_thinking(content)
        if not strip_multimodal and message['role'] == 'tool' and isinstance(content, list):
            content = [
                part
                if not _part_image_ref(part) or (i, j) in kept
                else {'type': 'text', 'text': '[earlier tool image omitted]'}
                for j, part in enumerate(content)
            ]
        message['content'] = await _reward_content(content, strip_multimodal)
        messages.append(message)
    return messages, len(kept) if not strip_multimodal else 0


async def score_visual_response(
    task_input: 'VisualTaskInput', output: AgentRunOutput, client: RewardSystemClient
) -> tuple[float, dict[str, Any]]:
    """Judge the original question and final answer, optionally including the uncompressed agent trajectory."""
    router_id = task_input.reward_system_router_id
    if router_id is None:
        raise ValueError('reward_system_router_id is required for scoring')
    details: dict[str, Any] = {'router_id': router_id}
    exit_status = getattr(output, 'exit_status', None)
    if exit_status in _MASKED_EXITS or getattr(output, 'failure_kind', None) == 'infrastructure':
        return float(ReasoningReward.UNJUDGED.value), {**details, 'skipped': exit_status or 'infrastructure'}
    if exit_status in _ZERO_EXITS:
        return 0.0, {**details, 'skipped': exit_status}
    if getattr(output, 'success', True) is False:
        return float(ReasoningReward.TASK_FAILED.value), {**details, 'skipped': 'task_failed'}

    strip_all, strip_routers = resolve_strip_multimodal_config(task_input.reward_system_strip_multimodal_content)
    strip_multimodal = strip_all or router_id in strip_routers
    messages = [
        {'role': 'system', 'content': ''},
        *(dict(message) for message in task_input.history or []),
        {'role': 'user', 'content': task_input.user_prompt},
    ]
    for message in messages:
        message['content'] = await _reward_content(message.get('content'), strip_multimodal)
    if task_input.reward_system_include_trajectory:
        trajectory_messages, tool_images_sent = await _trajectory_messages(
            output, task_input.reward_system_max_tool_images, strip_multimodal
        )
        messages.extend(trajectory_messages)
        details.update(trajectory_messages_sent=len(trajectory_messages), tool_images_sent=tool_images_sent)
    messages.append({'role': 'assistant', 'content': _prediction(output)})
    start = time.monotonic()
    try:
        score = await client.get_score(
            messages=messages,
            ref_answer=task_input.answer.strip(),
            router_id=router_id,
            extra_param=task_input.reward_system_extra,
            unittests=task_input.public_test,
            **({'log_summary': True} if task_input.reward_system_include_trajectory else {}),
        )
    except (ClientError, OSError, TimeoutError) as exc:
        message = re.sub(r"https?://[^\s'\"]+|data:[^\s'\"]+", '<media reference>', str(exc))[:300]
        details['error'] = f'{type(exc).__name__}: {message}'
        logger.warning('Visual reward scoring failed for router %s: %s', router_id, details['error'])
        score = float(ReasoningReward.UNJUDGED.value)
    if score is None or not math.isfinite(score) or score <= ReasoningReward.THRESHOLD.value:
        details.setdefault('error', f'Invalid reward score: {score}')
        score = float(ReasoningReward.UNJUDGED.value)
    attributes: dict[str, Any] = {'hytuner.reward_time': time.monotonic() - start}
    if score <= ReasoningReward.THRESHOLD.value:
        attributes['hytuner.termination_reason'] = 'REWARD_ERROR'
    if span_id := get_current_span_id():
        update_span(span_id, attributes=attributes)
    return score, details
