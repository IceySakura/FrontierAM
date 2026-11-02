"""``yb_rubric`` reward function — the user-facing async wrapper.

This is the entry point that ``vl_reward_system`` invokes when
``task_input.reward_func == 'yb_rubric'``. It adapts topic-specific
rubric scorers to the ``async (question, response, target_answer, trajectory)
-> (score, dict)`` contract used by the reward registry.

The wrapper expects ``target_answer`` to carry a rubric envelope JSON string:

``{"rubric_topic": "shiwu|general|medical", "rubrics": [...]}``

``general`` data is dispatched to the Shiwu topic scorer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from collections.abc import Awaitable, Callable
from typing import Any, cast

from .medical_topic import medical_rubric_score
from .personal_management_topic import personal_management_rubric_score
from .shiwu_topic import shiwu_rubric_score
from .trans_topic import trans_rubric_score
from .writing_topic import writing_rubric_score


logger = logging.getLogger(__name__)


# Sentinel returned to ``vl_reward_system`` when the rubric judge fails.
# ``mm_agent_task`` / ``yb_vlm_oneagent_task`` short-circuit on
# ``score <= -99.0`` and mark the sample REWARD_ERROR, so any value
# below that threshold works; -100 matches what ``consist_judge`` uses.
_REWARD_ERROR_SENTINEL: float = -100.0


_SCORER_BY_TOPIC: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    'shiwu': shiwu_rubric_score,
    'general': shiwu_rubric_score,
    'medical': medical_rubric_score,
    'writing': writing_rubric_score,
    'personal_management': personal_management_rubric_score,
    'translate': trans_rubric_score,
}


def _resolve_float(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None or raw == '':
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


async def _call_scorer(
    scorer: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    obj: dict[str, Any],
) -> dict[str, Any]:
    return await scorer(obj)


def _parse_rubric_payload(target_answer: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Parse the required rubric envelope from ``target_answer``.

    Expected format:
        {
            "rubric_topic": "shiwu" | "general" | "medical",
            "rubrics": [...]
        }
    """
    try:
        payload = json.loads(target_answer)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'target_answer must be a rubric envelope JSON string: {exc}') from exc

    if not isinstance(payload, dict):
        raise ValueError('target_answer must be a rubric envelope object')
    payload = cast(dict[str, Any], payload)

    raw_topic = payload.get('rubric_topic') or payload.get('topic') or payload.get('data_type')
    if not isinstance(raw_topic, str) or not raw_topic.strip():
        raise ValueError('rubric envelope missing non-empty rubric_topic')

    topic = raw_topic.strip().lower()
    if topic not in _SCORER_BY_TOPIC:
        raise ValueError(f'unsupported rubric_topic={raw_topic!r}')

    rubrics = payload.get('rubrics')
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError('rubric envelope missing non-empty rubrics list')
    if not all(isinstance(item, dict) for item in rubrics):
        raise ValueError('rubrics must be a list of objects')

    return topic, rubrics, payload


async def yb_rubric_reward(
    question: str,
    response: str,
    target_answer: str,
    trajectory: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[float, dict[str, Any]]:
    """Score ``response`` against a rubric checklist via an LLM judge.

    Args:
        question: The user's question (used in the judge prompt and
            also stored in ``obj['prompt']`` / ``obj['question']``).
        response: The model's final answer text.
        target_answer: Rubric envelope JSON string. It must contain
            ``rubric_topic`` and ``rubrics``. ``general`` routes to the
            ``shiwu`` topic.
        trajectory: Optional current rollout trajectory. It is kept for
            reward-registry compatibility but is not forwarded as rubric
            history.
        history: Optional dataset-provided conversation history;
            forwarded to the judge so the rubric grader can see the
            multi-turn context from prior turns.

    Returns:
        ``(score, details)``. ``score`` is in ``[0.0, 1.0]`` on success
        or ``-100`` on judge failure. ``details`` always has
        ``reward_func='yb_rubric'`` and includes the per-round scores
        plus the last valid scored checklist when available.
    """
    is_validate = bool(kwargs.get('validate', False))

    total_timeout = _resolve_float('YB_RUBRIC_TOTAL_TIMEOUT', 180.0)
    if total_timeout <= 0:
        total_timeout = 180.0

    try:
        rubric_topic, rubrics, rubric_payload = _parse_rubric_payload(target_answer)
    except Exception as exc:
        logger.warning(f'[yb_rubric][PAYLOAD_ERROR] {exc!r}; returning REWARD_ERROR sentinel.')
        return _REWARD_ERROR_SENTINEL, {
            'reward_func': 'yb_rubric',
            'rubric_topic': None,
            'scoring_mode': 'yb_rubric_payload_error',
            'round_scores': None,
            'scored_checklist': None,
            'critic_resp': f'Invalid rubric payload: {exc!r}',
            'question': question,
            'model_response': response,
            'error': 'rubric_payload_error',
        }

    scorer = _SCORER_BY_TOPIC[rubric_topic]

    obj: dict[str, Any] = {
        'question': question,
        'prompt': question,
        'response': response,
        'ref_answer': target_answer,
        'rubrics': rubrics,
        'rubric_topic': rubric_topic,
        'rubric_payload': rubric_payload,
        'history': history or [],
        'validate': is_validate,
    }

    try:
        result = await asyncio.wait_for(
            _call_scorer(scorer, obj),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f'[yb_rubric][RM_TIMEOUT] total_timeout={total_timeout}s; returning REWARD_ERROR sentinel.')
        return _REWARD_ERROR_SENTINEL, {
            'reward_func': 'yb_rubric',
            'rubric_topic': rubric_topic,
            'scoring_mode': 'yb_rubric_timeout',
            'round_scores': None,
            'scored_checklist': None,
            'critic_resp': f'RM timed out after {total_timeout}s',
            'question': question,
            'model_response': response,
            'error': 'rubric_judge_timeout',
        }
    except Exception as exc:
        logger.warning(f'[yb_rubric][RM_ERROR] {exc!r}; returning REWARD_ERROR sentinel.')
        return _REWARD_ERROR_SENTINEL, {
            'reward_func': 'yb_rubric',
            'rubric_topic': rubric_topic,
            'scoring_mode': 'yb_rubric_error',
            'round_scores': None,
            'scored_checklist': None,
            'critic_resp': f'RM failed: {exc!r}',
            'question': question,
            'model_response': response,
            'error': 'rubric_judge_error',
        }

    raw_reward = result.get('reward', -1.0)
    try:
        reward = float(raw_reward)
    except (TypeError, ValueError):
        reward = -1.0

    scoring_mode = result.get('scoring_mode', '')

    details: dict[str, Any] = {
        'reward_func': 'yb_rubric',
        'rubric_topic': rubric_topic,
        'scoring_mode': scoring_mode,
        'round_scores': result.get('round_scores'),
        'scored_checklist': result.get('scored_checklist'),
        'critic_resp': result.get('critic_resp'),
        'question': question,
        'model_response': response,
    }

    # Judge / parse failure → bubble up REWARD_ERROR sentinel.
    # Note: ``yb_rubric_empty_response`` is treated as a legitimate 0.0
    # (the model returned nothing — that's a real wrong answer, not a
    # judge failure) and falls through to the success path below.
    if scoring_mode == 'yb_rubric_failed' or reward < 0.0:
        logger.warning(
            f'[yb_rubric] judge failed: scoring_mode={scoring_mode!r}, '
            f'raw_reward={raw_reward!r}; returning REWARD_ERROR sentinel.'
        )
        details['error'] = f'rubric_judge_failed: mode={scoring_mode}'
        return _REWARD_ERROR_SENTINEL, details

    # Clamp to [0, 1] defensively — get_overall_score should already be
    # in range but rubric prompts evolve over time.
    reward = max(0.0, min(1.0, reward))
    details['score'] = reward
    return reward, details


__all__ = ['yb_rubric_reward']
