"""Common score dispatch for reward-system-backed and verifier tasks."""

from __future__ import annotations

import asyncio

from typing import Any

from nexus.extensions.tasks.reward_system.gsm8k import compute_score as compute_gsm8k_score
from nexus.extensions.tasks.reward_system.math import compute_score as compute_math_score
from nexus.gateway.media import REMOTE_CONSUMER_SOURCES, MediaResolver
from nexus.utils.reward_system import RewardSystemClient


async def compute_score(
    *,
    scorer: str,
    response: str,
    reference_answer: Any,
    prompt: list[dict[str, Any]] | None = None,
    question: str | None = None,
    router_id: str | None = None,
    extra: dict[str, Any] | None = None,
    public_test: dict[str, Any] | list[Any] | str | None = None,
    strip_multimodal: bool = False,
    max_retries: int = 1,
    retry_delay: float = 5.0,
    timeout: float = 1800.0,
) -> float | None:
    """Compute one score through the selected backend.

    The task and all verifier implementations share this interface. Verifier
    functions receive only the model response and row reference; the gateway
    remains the owner of inference and trajectory metrics.
    """
    if scorer == 'gsm8k':
        options = extra or {}
        return await asyncio.to_thread(
            compute_gsm8k_score,
            response,
            str(reference_answer),
            method=options.get('method', 'strict'),
            format_score=float(options.get('format_score', 0.0)),
            score=float(options.get('score', 1.0)),
        )
    if scorer == 'math':
        return await compute_math_score(response, reference_answer, question=question)
    if scorer != 'reward_system':
        raise ValueError(f'Unknown score backend: {scorer!r}')
    if prompt is None:
        raise ValueError('Reward System scoring requires the original prompt messages')
    if not router_id:
        raise ValueError('Reward System scoring requires reward.router_id')
    messages = _build_reward_messages(prompt, response, strip_multimodal=strip_multimodal)
    # NOTE(petersheng): the Reward System cannot read the Nexus host's filesystem, so local image
    # paths are inlined as data URLs; stripped images were already dropped and are not read.
    messages = await MediaResolver().resolve(messages, REMOTE_CONSUMER_SOURCES)
    client = RewardSystemClient(max_retries=max_retries, retry_delay=retry_delay, timeout=timeout)
    return await client.get_score(
        messages=messages,
        ref_answer=str(reference_answer),
        router_id=router_id,
        extra_param=extra or {},
        unittests=public_test,
    )


def _build_reward_messages(
    prompt: list[dict[str, Any]], response: str, *, strip_multimodal: bool
) -> list[dict[str, Any]]:
    """Build exactly the message shape used by the old hytuner reward loop."""
    import copy

    if not isinstance(response, str):
        raise ValueError(f'Answer must be string, got {type(response)}')
    messages = copy.deepcopy(prompt)
    if not messages:
        raise ValueError('Reward System scoring requires a nonempty prompt')
    if messages[-1].get('role') == 'assistant':
        raise ValueError('Last message must not be assistant role')
    if messages[0].get('role') != 'system':
        messages.insert(0, {'role': 'system', 'content': ''})
    if strip_multimodal:
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                message['content'] = '\n'.join(
                    part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text'
                )
    messages.append({'role': 'assistant', 'content': response})
    return messages


__all__ = ['compute_score']
