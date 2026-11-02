"""VL Reward System — dispatches to specific reward functions based on reward_func.

This module provides the `vl_reward_system` function that routes evaluation
to different scoring strategies (e.g., consist_judge, exact_match) based on
the `reward_func` parameter specified in the data.
"""

import logging
import os

from typing import Any

from nexus.extensions.tasks.mm_agent.reward_system.consist_judge import consist_judge
from nexus.extensions.tasks.mm_agent.reward_system.reward_platform_client import call_reward_platform_consist_judge
from nexus.extensions.tasks.mm_agent.reward_system.spatial import (
    bbox_reward,
    point_reward,
    seg_reward,
)


logger = logging.getLogger(__name__)


async def _exact_match_reward(
    question: str,
    response: str,
    target_answer: str,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Simple exact match reward function.

    Returns 1.0 if the response matches the target answer (case-insensitive,
    stripped), 0.0 otherwise.
    """
    pred = response.strip().lower()
    ref = target_answer.strip().lower()
    is_match = pred == ref
    score = 1.0 if is_match else 0.0
    return score, {
        'reward_func': 'exact_match',
        'prediction': pred,
        'reference': ref,
        'is_match': is_match,
    }


async def _consist_judge_reward(
    question: str,
    response: str,
    target_answer: str,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Consistency judge reward function.

    By default this uses the reward platform implementation. Set
    ``USE_REWARD_PLATFORM_FOR_CONSIST_JUDGE=0`` to fall back to the legacy
    in-process Taiji judge (text-only: it never sees the images).
    """
    use_platform = os.environ.get('USE_REWARD_PLATFORM_FOR_CONSIST_JUDGE', '1') != '0'
    if use_platform:
        score, details = await call_reward_platform_consist_judge(
            question=question,
            response=response,
            target_answer=target_answer,
            instance_id=kwargs.get('instance_id'),
            router_id=kwargs.get('router_id'),
            images=kwargs.get('images'),
            extra=kwargs.get('reward_extra'),
        )
    else:
        score, details = await consist_judge(
            question=question,
            response=response,
            target_answer=target_answer,
        )
    details['reward_func'] = 'consist_judge'
    return score, details


# Registry of available reward functions
REWARD_FUNC_REGISTRY: dict[str, Any] = {
    'exact_match': _exact_match_reward,
    'consist_judge': _consist_judge_reward,
    'point': point_reward,
    'bbox': bbox_reward,
    'seg': seg_reward,
}

# Reward functions that actually route to the reward platform. Anything else
# scores locally and therefore ignores ``reward_system_router_id``.
_PLATFORM_REWARD_FUNCS = frozenset({'consist_judge'})


async def vl_reward_system(
    reward_func: str | None,
    question: str,
    response: str,
    target_answer: str,
    trajectory: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[float, dict]:
    """Main entry point for the VL reward system.

    Routes to the appropriate reward function based on `reward_func`.

    Args:
        reward_func: Name of the reward function to use. If None, defaults to 'consist_judge'.
        question: The original question.
        response: The model's prediction/response.
        target_answer: The ground truth/reference answer.
        trajectory: The agent's trajectory (optional, for future use).
        **kwargs: Additional keyword arguments passed to the reward function.
            ``router_id`` / ``images`` / ``reward_extra`` are only consumed by
            the reward-platform path.

    Returns:
        A tuple of (score, details).
        score: Float reward score (typically 0.0 or 1.0).
        details: Dictionary with evaluation details.
    """
    if reward_func is None:
        reward_func = 'consist_judge'
        logger.debug(f'reward_func is None, defaulting to: {reward_func}')

    func = REWARD_FUNC_REGISTRY.get(reward_func)
    if func is None:
        available = list(REWARD_FUNC_REGISTRY.keys())
        raise ValueError(f'Unknown reward_func: {reward_func!r}. Available: {available}')

    # A router_id on a locally-scored reward_func is a data misconfiguration:
    # the platform is never called, so the configured router silently does
    # nothing. Surface it instead of dropping it into **kwargs.
    if kwargs.get('router_id') and reward_func not in _PLATFORM_REWARD_FUNCS:
        logger.warning(
            'reward_system_router_id=%r is set but reward_func=%r scores locally and never calls the '
            'reward platform; the router_id is ignored. Platform-backed reward_funcs: %s',
            kwargs['router_id'],
            reward_func,
            sorted(_PLATFORM_REWARD_FUNCS),
        )

    logger.debug(f'VL reward system using reward_func={reward_func!r}')
    score, details = await func(
        question=question,
        response=response,
        target_answer=target_answer,
        trajectory=trajectory,
        **kwargs,
    )
    return score, details
