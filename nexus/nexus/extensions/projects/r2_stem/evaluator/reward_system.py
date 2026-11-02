"""Reward-system evaluator adapter for mini SWE agent tasks."""

from typing import Any

from nexus.utils.reward_system import RewardSystemClient


async def rmsys_evaluate(
    *,
    model_answer: str,
    reference: Any,
    question: str,
    router_id: str,
    user_token: str | None = None,
    exp_id: str = '',
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Evaluate a text answer through the remote reward system."""
    client = RewardSystemClient(user_token=user_token, exp_id=exp_id)
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': model_answer},
    ]
    score = await client.get_score(
        messages=messages,
        ref_answer=reference,
        router_id=router_id,
        extra_param=extra_info,
    )
    return float(score)


__all__ = ['rmsys_evaluate']
