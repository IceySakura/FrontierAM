"""Reward-platform HTTP client for mm_agent consist_judge scoring."""

from __future__ import annotations

import logging
import os
import re
import uuid

from collections.abc import Sequence
from typing import Any

import aiohttp

from nexus.utils.reward_system import build_user_content


logger = logging.getLogger(__name__)

DEFAULT_REWARD_PLATFORM_URL = (
    'http://api-reward-platform-api-reward-platform-354ba565.turbotke.production.polaris:10001/reward/RewardService'
)
FAILED_SCORE = -100.0
# Used only when the sample carries no ``reward_system_router_id`` at all.
DEFAULT_ROUTER = ('MultimodalReward', 'ToolUseModelScoring', '0')


def parse_reward_platform_router_id(router_id: str | None) -> tuple[str, str, str]:
    """Parse ``class1-class2-router_version``.

    An absent ``router_id`` falls back to ``DEFAULT_ROUTER``. A *malformed*
    one raises instead of silently scoring against the default router, which
    would otherwise report a plausible-looking score from the wrong judge.
    """
    if not router_id:
        return DEFAULT_ROUTER

    parts = router_id.rsplit('-', 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f'reward_system_router_id={router_id!r} is not "<class1>-<class2>-<router_version>". '
            'Fix the data instead of relying on a default router.'
        )

    class1, class2, router_version = parts
    return class1, class2, router_version


def _build_request(
    *,
    question: str,
    response: str,
    target_answer: str,
    request_id: str,
    router_id: str | None = None,
    images: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    class1, class2, router_version = parse_reward_platform_router_id(router_id)
    return {
        'request_id': request_id,
        'exp_id': os.environ.get('REWARD_PLATFORM_EXP_ID', 'mm_agent_consist_judge'),
        'source': 'dpo',
        'model_type': os.environ.get('REWARD_PLATFORM_MODEL_TYPE', 'hy_turbo_v2.07'),
        'messages': [
            {'role': 'system', 'content': ''},
            # Images first, then the question — same order the policy saw them,
            # so a multimodal router judges against the picture rather than the
            # question text alone. Text-only routers still get a bare string.
            {'role': 'user', 'content': build_user_content(question, images)},
            {'role': 'assistant', 'content': response},
        ],
        'ref_answer': target_answer,
        'class1': class1,
        'class2': class2,
        'router_version': router_version,
        # Sub-service knobs (e.g. scoring_mode / checklist_mode for
        # MultimodalRewardGeneralV2) travel here.
        'extra': extra or {},
        'public_test': [],
    }


def build_expected_sub_service_classes(class1: str, class2: str) -> tuple[str, ...]:
    """Spell out the sub-service class names a ``class1-class2`` route may answer with.

    The platform names each sub-service after its *implementation*, not after
    the route: ``MMOCR-OCRModelScoring-0`` is answered by
    ``MultimodalRewardOCRModelScoring``, so neither ``class2`` alone nor
    ``class1 + class2`` matches. List every observed spelling, most specific
    first.

    Every candidate here is derived from ``class2``, so a match always comes
    from the judge the caller asked for. Notably absent is the legacy
    ``MultimodalRewardToolUseModelScoring``: accepting it for an arbitrary
    route would score the sample with a different judge under the requested
    router's name — the exact failure ``extract_sub_service_data`` refuses
    to allow via ``sub_responses[0]``. It is reachable only through
    ``DEFAULT_ROUTER``, i.e. when the sample carries no ``router_id`` and
    ``class2`` is genuinely ``ToolUseModelScoring``.
    """
    candidates = [f'{class1}{class2}', f'MultimodalReward{class2}']
    moved = _version_before_kind(class2)
    if moved is not None:
        candidates += [f'{class1}{moved}', f'MultimodalReward{moved}']
    candidates.append(class2)
    return tuple(dict.fromkeys(candidates))


def _version_before_kind(class2: str) -> str | None:
    """Rewrite ``PPTJudgeScoringV2`` as ``PPTJudgeV2Scoring``, the platform's order.

    A route spells its version last while the class that answers it spells the
    version before the kind, so neither ``class1 + class2`` nor the ``class2``
    suffix matches a versioned judge. Derived from ``class2`` alone, so a match
    still comes from the judge the caller asked for.
    """
    match = re.fullmatch(r'(?P<stem>.+?)(?P<kind>Scoring|Server|Judge)(?P<version>V\d+)', class2)
    if match is None:
        return None
    return f'{match["stem"]}{match["version"]}{match["kind"]}'


def extract_sub_service_data(
    resp_json: dict[str, Any],
    expected_classes: tuple[str, ...] = (),
    fallback_class_suffix: str | None = None,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Extract the selected reward sub-service data from a RewardService response.

    Args:
        resp_json: Decoded RewardService response.
        expected_classes: Accepted ``response_datas[].class`` values, tried in order.
        fallback_class_suffix: When no exact name matches, accept the single
            sub-service whose class ends with this suffix (the requested
            ``class2``). Ambiguous suffix matches are rejected.
    """
    if not isinstance(resp_json, dict):
        return None, f'malformed response (not a dict): {type(resp_json).__name__}', None

    status = resp_json.get('status')
    if status != 0:
        return None, f'platform status={status} msg={resp_json.get("msg")!r}', None

    data = resp_json.get('data') or {}

    sub_responses = data.get('response_datas') or []
    if not sub_responses and isinstance(data.get('reward'), dict):
        reward_data = data['reward']
        if reward_data.get('error_code', 0) != 0:
            return (
                None,
                f'reward error_code={reward_data.get("error_code")} msg={reward_data.get("error_msg")!r}',
                'reward',
            )
        return reward_data, '', 'reward'
    if not sub_responses:
        return None, 'response_datas is empty', None

    target = None
    for cls in expected_classes:
        for sub_response in sub_responses:
            if sub_response.get('class') == cls:
                target = sub_response
                break
        if target is not None:
            break

    if target is None and fallback_class_suffix:
        # The platform occasionally prefixes a sub-service with a vendor name
        # we do not know up front. A *unique* suffix match still identifies
        # the requested judge unambiguously.
        suffix_matches = [
            sub_response
            for sub_response in sub_responses
            if str(sub_response.get('class') or '').endswith(fallback_class_suffix)
        ]
        if len(suffix_matches) == 1:
            target = suffix_matches[0]

    if target is None:
        # Never fall back to sub_responses[0]: that silently returns another
        # sub-service's score under the requested router's name.
        available = [sub_response.get('class') for sub_response in sub_responses]
        return None, f'no sub_service matching {list(expected_classes)}; available={available}', None

    sub_class = target.get('class')
    if target.get('error_code') != 0:
        return (
            None,
            f'sub_service {sub_class!r} error_code={target.get("error_code")} msg={target.get("error_msg")!r}',
            sub_class,
        )

    return target.get('data') or {}, '', sub_class


def get_reward_platform_token() -> str:
    return (
        os.environ.get('REWARD_PLATFORM_TOKEN', '').strip()
        or os.environ.get('REWARD_SERVICE_TOKEN', '').strip()
        or os.environ.get('NEXUS_REWARD_SYSTEM_TOKEN', '').strip()
        or os.environ.get('nexus_reward_system_token', '').strip()  # noqa: SIM112
    )


def get_reward_platform_url() -> str:
    """Reward-platform endpoint, from the environment or the default."""
    return (
        os.environ.get('REWARD_PLATFORM_URL', '').strip()
        or os.environ.get('REWARD_SERVICE_URL', '').strip()
        or DEFAULT_REWARD_PLATFORM_URL
    ).strip()


def _get_reward_platform_timeout() -> int:
    try:
        return int(os.environ.get('REWARD_PLATFORM_TIMEOUT', '180'))
    except (TypeError, ValueError):
        return 180


async def post_to_reward_platform(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """POST the payload once.

    The reward platform maintains its own retry/failover, so retrying here
    would only multiply the load on it.
    """
    timeout = aiohttp.ClientTimeout(total=_get_reward_platform_timeout())
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f'reward platform http {resp.status}: {(await resp.text())[:300]}')
            return await resp.json()


async def call_reward_platform_consist_judge(
    *,
    question: str,
    response: str,
    target_answer: str,
    instance_id: str | None = None,
    router_id: str | None = None,
    images: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score a response through the reward platform's consist_judge service."""
    url = get_reward_platform_url()
    token = get_reward_platform_token()
    if not token:
        msg = 'REWARD_PLATFORM_TOKEN, REWARD_SERVICE_TOKEN, or NEXUS_REWARD_SYSTEM_TOKEN is required.'
        logger.error(msg)
        return FAILED_SCORE, _error_details(msg, question, target_answer, response, router_id, images)

    request_id = f'{instance_id or "consist_judge"}_{uuid.uuid4().hex}'
    payload = _build_request(
        question=question,
        response=response,
        target_answer=target_answer,
        request_id=request_id,
        router_id=router_id,
        images=images,
        extra=extra,
    )
    headers = {'Content-Type': 'application/json', 'token': token}

    # Debug-level: on the success path every field here also reaches
    # ``details`` and is logged once by the mm_agent task, so this line would
    # only duplicate it per sample. It earns its keep when the process dies
    # mid-request — then it is the only record that the call went out at all,
    # and which router it went to.
    logger.debug(
        '[reward_platform] request_id=%s, router_id=%s, image_count=%d, extra_keys=%s, question_len=%d, response_len=%d',
        request_id,
        router_id,
        len(images or []),
        sorted((extra or {}).keys()),
        len(question),
        len(response),
    )

    try:
        resp_json = await post_to_reward_platform(url, payload, headers)
    except Exception as exc:
        msg = f'reward platform request failed: {exc!r}'
        logger.error('[reward_platform] request_id=%s failed: %s', request_id, msg)
        return FAILED_SCORE, _error_details(
            msg, question, target_answer, response, router_id, images, platform_request_id=request_id
        )

    class1, class2, _ = parse_reward_platform_router_id(router_id)
    expected_classes = build_expected_sub_service_classes(class1, class2)
    sub_data, err, sub_class = extract_sub_service_data(
        resp_json,
        expected_classes,
        fallback_class_suffix=class2,
    )
    if sub_data is None:
        logger.error('reward platform sub-service extraction failed: %s', err)
        return FAILED_SCORE, _error_details(
            err, question, target_answer, response, router_id, images, platform_request_id=request_id
        )

    reward_score = sub_data.get('reward_score', sub_data.get('final_score'))
    try:
        score = float(reward_score)
    except (TypeError, ValueError):
        msg = f'reward_score is not a float: {reward_score!r}'
        logger.error(msg)
        return FAILED_SCORE, _error_details(
            msg, question, target_answer, response, router_id, images, platform_request_id=request_id
        )

    res_json = sub_data.get('res_json') or {}
    details: dict[str, Any] = {
        'ref_summary': res_json.get('ref_summary', ''),
        'model_summary': res_json.get('model_summary', ''),
        'analysis': res_json.get('analysis', ''),
        'is_consistent': res_json.get('is_consistent'),
        'raw_response': res_json.get('raw_response', sub_data.get('res', '')),
        'parse_method': res_json.get('parse_method', 'reward_platform_v1'),
        'parse_attempts': 1,
        'question': question,
        'target_answer': target_answer,
        'model_response': response,
        'platform_request_id': request_id,
        'router_id_used': router_id,
        'sub_service_class': sub_class,
        'image_count': len(images or []),
        'extra_keys': sorted((extra or {}).keys()),
    }
    return score, details


def _error_details(
    error: str,
    question: str,
    target_answer: str,
    response: str,
    router_id: str | None,
    images: Sequence[str] | None = None,
    *,
    platform_request_id: str = '',
) -> dict[str, Any]:
    return {
        'error': error,
        'question': question,
        'target_answer': target_answer,
        'model_response': response,
        'platform_request_id': platform_request_id,
        'router_id_used': router_id,
        'image_count': len(images or []),
    }
