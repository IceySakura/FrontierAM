"""Client for the external reward-system judge (RewardService).

Ported (nearly verbatim) from
``ContextAgent/contextagent/eval/rm_system_client.py`` so the formal nexus
context agent scores answers with the SAME RM system as the source repo,
replacing the previous Hunyuan LLM-as-judge (``【评分】`` checklist) scorer.

The reward service is a drop-in judge: given the task prompt + a candidate
answer + the reference answer + a router id, it returns a ``final_score``.

Endpoint / token are shared constants (overridable via env so nothing is baked
in for a single user):
    RM_REWARD_URL, NEXUS_REWARD_SYSTEM_TOKEN, NEXUS_REWARD_SYSTEM_EXP_ID, RM_MODEL_TYPE
    RM_TOKEN and RM_EXP_ID remain compatibility aliases.

Request shape (mirrors the original ``build_req_data``):
    {request_id, exp_id, source:"ppo", model_type, messages, class1, class2,
     ref_answer, router_version, public_test:[], extra}
where ``class1-class2-router_version = reward_system_router_id.split("-")``.
The task's parsing settings are preserved so routers such as ``T1-Omni-0`` can
split the complete thinking/answer response before applying the rubric judge.

Response: score at ``data.reward.final_score`` (``-200`` == error sentinel).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid

from contextlib import suppress
from typing import Any

import aiohttp

from .score import ScoreResult


logger = logging.getLogger(__name__)

# Shared service constants (env-overridable; no per-user hardcoding).
REWARD_URL = os.environ.get(
    'RM_REWARD_URL',
    'http://api-reward-platform-api-reward-platform-354ba565.turbotke.production.polaris:10001/reward/RewardService',
)
REWARD_TOKEN = (
    os.environ.get('NEXUS_REWARD_SYSTEM_TOKEN')
    or os.environ.get('nexus_reward_system_token')  # noqa: SIM112 - legacy Nexus credential name
    or os.environ.get('RM_TOKEN', '')
)
REWARD_HEADERS = {'Content-Type': 'application/json', 'token': REWARD_TOKEN}
DEFAULT_EXP_ID = (
    os.environ.get('NEXUS_REWARD_SYSTEM_EXP_ID')
    or os.environ.get('nexus_reward_system_exp_id')  # noqa: SIM112 - legacy Nexus experiment name
    or os.environ.get('RM_EXP_ID')
    or 'ctxagent_rm_eval'
)
DEFAULT_MODEL_TYPE = os.environ.get('RM_MODEL_TYPE', 'hy3_a20b_offical')

# Score the complete assistant content as the answer without requiring think tags.
WHOLE_ANSWER_PATTERN = r'^()([\s\S]*)$'

# ``-200`` is the reward service's error sentinel for a failed grade.
ERROR_SCORE = -200

# Lazily resolved Ray GlobalRateLimiter handle shared with HyTuner reward_loop.
_RM_RATE_LIMITER = None


def _rm_max_concurrent() -> int:
    raw = os.environ.get('CTXAGENT_RM_MAX_CONCURRENT', '').strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError(f'CTXAGENT_RM_MAX_CONCURRENT must be a non-negative integer, got {raw!r}') from exc


def _get_shared_rm_rate_limiter():
    """Reuse HyTuner's named GlobalRateLimiter actor when concurrency is capped.

    Returns None only when limiting is disabled. Prefer ``ray.get_actor`` so
    we join the reward_loop limiter without resetting it on every agent score
    call. A configured limiter is part of the training contract, so missing
    HyTuner/Ray integration fails explicitly instead of silently sending an
    uncapped request burst.
    """
    global _RM_RATE_LIMITER
    max_concurrent = _rm_max_concurrent()
    if max_concurrent <= 0:
        return None
    if _RM_RATE_LIMITER is not None:
        return _RM_RATE_LIMITER
    import ray

    from hytuner.reward_loop.global_rate_limiter import (
        GLOBAL_RATE_LIMITER_ACTOR_NAME,
        create_global_rate_limiter,
    )

    try:
        _RM_RATE_LIMITER = ray.get_actor(GLOBAL_RATE_LIMITER_ACTOR_NAME)
    except ValueError:
        # Reward loop may not have created the actor yet (or is disabled).
        _RM_RATE_LIMITER = create_global_rate_limiter(max_concurrent)
        logger.debug(
            'created shared RM GlobalRateLimiter max_concurrent=%d',
            max_concurrent,
        )
    return _RM_RATE_LIMITER


async def _with_rm_rate_limit(coro):
    limiter = _get_shared_rm_rate_limiter()
    if limiter is None:
        return await coro
    from hytuner.reward_loop.global_rate_limiter import rate_limited_call

    return await rate_limited_call({'_default': limiter}, coro)


def _split_router_id(router_id: str | None) -> tuple[str, str, str]:
    """Return ``(class1, class2, router_version)`` from a router id string."""
    if not router_id:
        return '', '', ''
    parts = str(router_id).split('-')
    if len(parts) != 3:
        logger.warning('reward_system_router_id %r does not split into 3 parts', router_id)
        return '', '', ''
    return parts[0], parts[1], parts[2]


def build_reward_request(
    prompt: list[dict],
    answer: str,
    *,
    router_id: str | None,
    ref_answer: str = '',
    extra: dict | None = None,
    exp_id: str = DEFAULT_EXP_ID,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict | None:
    """Build the RewardService request body for ``answer`` on a task.

    Mirrors ``build_req_data``: appends the complete model response as the
    assistant turn, splits the router id, and forwards ``ref_answer`` + ``extra``.
    Explicit parsing settings from the dataset take precedence. Returns ``None``
    when there is no prompt or a valid router id.
    """
    if not isinstance(prompt, list) or not prompt:
        logger.warning('task has no prompt list; cannot build reward request')
        return None
    class1, class2, router_version = _split_router_id(router_id)
    if not class1:
        return None

    extra_out = dict(extra or {})
    extra_out.setdefault('answer_pattern', WHOLE_ANSWER_PATTERN)

    messages = list(prompt) + [{'role': 'assistant', 'content': answer}]
    return {
        'request_id': uuid.uuid4().hex,
        'exp_id': exp_id,
        'source': 'ppo',
        'model_type': model_type,
        'messages': messages,
        'class1': class1,
        'class2': class2,
        'ref_answer': ref_answer,
        'router_version': router_version,
        'public_test': [],
        'extra': extra_out,
    }


def extract_final_score(response: dict | None) -> float | None:
    """Pull ``data.reward.final_score`` out of a RewardService response."""
    if response is None:
        return None
    try:
        return response['data']['reward']['final_score']
    except (KeyError, TypeError):
        logger.warning('could not extract final_score from response: %.300s', str(response))
        return None


def _extract_error_msg(response: dict | None) -> str:
    if response is None:
        return 'no response'
    try:
        return response['data']['reward'].get('error_msg') or ''
    except (KeyError, TypeError):
        return ''


def _extract_model_answer_error(value: Any) -> str:
    """Find an inference error hidden inside a nominally successful RM response."""
    if isinstance(value, dict):
        model_answer = value.get('model_answer')
        if isinstance(model_answer, dict) and model_answer.get('error'):
            error = model_answer['error']
            if isinstance(error, dict):
                return str(error.get('message') or error.get('code') or error)
            return str(error)
        for nested in value.values():
            message = _extract_model_answer_error(nested)
            if message:
                return message
    elif isinstance(value, list):
        for nested in value:
            message = _extract_model_answer_error(nested)
            if message:
                return message
    return ''


def score_to_binary(final_score: float | None, threshold: float = 1.0) -> int:
    """Map a (possibly non-binary) final_score to 0/1 for pass@k machinery.

    ``final_score >= threshold`` -> 1, else 0. The error sentinel ``-200`` and
    ``None`` both map to 0. The raw float is kept separately by callers.
    """
    if final_score is None:
        return 0
    try:
        v = float(final_score)
    except (TypeError, ValueError):
        return 0
    if v == ERROR_SCORE:
        return 0
    return 1 if v >= threshold else 0


async def score_answer(
    prompt: list[dict],
    answer: str,
    *,
    router_id: str | None,
    ref_answer: str = '',
    extra: dict | None = None,
    exp_id: str = DEFAULT_EXP_ID,
    model_type: str = DEFAULT_MODEL_TYPE,
    max_retries: int = 2,
    subservice_retry_delay: float = 30.0,
    session: aiohttp.ClientSession | None = None,
) -> ScoreResult:
    """Score one ``answer`` for a task via the reward service.

    Returns ``{final_score, final_score_ori, binary, error_msg, ok, raw, ...}``.
    Retries on 5xx / network errors and on the service's ``-200`` subservice
    failure sentinel. Never raises. Infrastructure-side failures carry
    ``reward_error=True`` so callers can mask the sample instead of training
    the model against a synthetic zero. Opens its own aiohttp session when one
    is not supplied.
    """
    if not answer or not str(answer).strip():
        return {
            'final_score': None,
            'final_score_ori': None,
            'binary': 0,
            'error_msg': 'empty answer',
            'ok': False,
            'raw': None,
            'reward_error': False,
            'request_ids': [],
        }
    if not REWARD_TOKEN:
        return {
            'final_score': None,
            'final_score_ori': None,
            'binary': 0,
            'error_msg': 'NEXUS_REWARD_SYSTEM_TOKEN is not configured (legacy alias: RM_TOKEN)',
            'ok': False,
            'raw': None,
            'reward_error': True,
            'request_ids': [],
        }

    req = build_reward_request(
        prompt,
        answer,
        router_id=router_id,
        ref_answer=ref_answer,
        extra=extra,
        exp_id=exp_id,
        model_type=model_type,
    )
    if req is None:
        return {
            'final_score': None,
            'final_score_ori': None,
            'binary': 0,
            'error_msg': 'no prompt / bad router_id',
            'ok': False,
            'raw': None,
            'reward_error': True,
            'request_ids': [],
        }

    owns_session = session is None
    client = aiohttp.ClientSession() if session is None else session

    async def _score_with_retries() -> ScoreResult:
        last_error = None
        request_ids: list[str] = []
        for attempt in range(max_retries + 1):
            if attempt:
                req['request_id'] = uuid.uuid4().hex
            request_id = str(req['request_id'])
            request_ids.append(request_id)
            try:
                async with client.post(REWARD_URL, headers=REWARD_HEADERS, json=req, ssl=False) as resp:
                    raw = await resp.json(content_type=None)
                    if resp.status >= 500:
                        raise RuntimeError(f'HTTP {resp.status}: {raw}')
                    fs = extract_final_score(raw)
                    fs_ori = None
                    with suppress(KeyError, TypeError):
                        fs_ori = raw['data']['reward'].get('final_score_ori')
                    model_answer_error = _extract_model_answer_error(raw)
                    err = model_answer_error or (_extract_error_msg(raw) if (fs is None or fs == ERROR_SCORE) else '')
                    if fs == ERROR_SCORE:
                        logger.warning(
                            'reward subservice failed: request_id=%s attempt=%d/%d error=%s',
                            request_id,
                            attempt + 1,
                            max_retries + 1,
                            err or 'unknown',
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(subservice_retry_delay)
                            continue
                    if model_answer_error:
                        logger.warning(
                            'reward judge inference failed: request_id=%s error=%s',
                            request_id,
                            model_answer_error,
                        )
                    reward_error = fs is None or fs == ERROR_SCORE or bool(model_answer_error)
                    return {
                        'final_score': fs,
                        'final_score_ori': fs_ori,
                        'binary': score_to_binary(fs),
                        'error_msg': err,
                        'ok': not reward_error,
                        'raw': raw,
                        'reward_error': reward_error,
                        'request_id': request_id,
                        'request_ids': request_ids,
                    }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    'reward request failed: request_id=%s attempt=%d/%d error=%s',
                    request_id,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(min(10.0, 0.2 * (2**attempt)) + random.random() * 0.2)
                    continue
        logger.debug('reward request failed after retries: %s', last_error)
        return {
            'final_score': None,
            'final_score_ori': None,
            'binary': 0,
            'error_msg': f'request failed: {last_error}',
            'ok': False,
            'raw': None,
            'reward_error': True,
            'request_id': request_ids[-1],
            'request_ids': request_ids,
        }

    try:
        # Hold one shared concurrency slot for the whole retry budget so agent
        # and baseline RM traffic share CTXAGENT_RM_MAX_CONCURRENT.
        return await _with_rm_rate_limit(_score_with_retries())
    finally:
        if owns_session:
            await client.close()
