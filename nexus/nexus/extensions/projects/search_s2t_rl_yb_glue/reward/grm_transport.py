"""GRM (AudioGRMDS) raw-text transport, ported from hytuner — independent of hytuner.

Ported from ``hytuner/utils/reward_score/audio_grm_reward.py`` (``is_grm_router_id``,
``_extract_choice_text``, ``request_grm_raw_text``) on the
``integration/speech-agenticrl-v5-hy-main`` hytuner branch, so ``gate_judge.py`` /
``llm_judge.py`` no longer need to import the training repo at runtime.

Two things are intentionally reused from :mod:`nexus.utils.reward_system` instead of
being duplicated here:

- ``RewardSystemClient.BASE_URL`` / ``REWARD_ENDPOINT``: identical Polaris service name
  hytuner's own ``RewardSystemClient`` points at (verified byte-for-byte equal).
- ``_get_ip_port_random``: the Polaris service-discovery helper, which now caches its
  consumer (see the comment there) — the original hytuner ``get_consumer_api()`` cached
  it globally too; only the nexus port had dropped that caching, not the resolution
  logic (still ``get_all_instances`` + random pick, not a single-instance lookup).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from typing import Any

import aiohttp

from nexus.utils.reward_system import RewardSystemClient, _get_ip_port_random


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1800
DEFAULT_MAX_RETRIES = 3

# hytuner's RewardSystemClient.RETRY_DELAY; kept as a local constant rather than
# reaching into a class attribute nexus's RewardSystemClient does not expose.
_RETRY_DELAY = 5


def is_grm_router_id(router_id: str | None) -> bool:
    """Return whether a reward_system router id should use the AudioGRMDS path."""
    if not router_id:
        return False
    parts = str(router_id).split('-', 2)
    return len(parts) == 3 and parts[0].lower() in {'audiogrmds', 'audiogrm'}


def _extract_choice_text(reward_data: dict[str, Any]) -> str:
    """Pull the raw judge completion out of a reward_system ``reward`` block.

    The AudioGRMDS/GRM routers return the model's raw text at
    ``data.reward.choices[0].message.content``. Handles the three observed choice
    shapes: a bare string, an OpenAI-style ``{"message": {"content": ...}}``, or
    ``{"text": ...}``.
    """
    choices = reward_data.get('choices')
    if not choices:
        return ''
    first = choices[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        message = first.get('message')
        if isinstance(message, dict) and isinstance(message.get('content'), str):
            return message['content']
        if isinstance(first.get('text'), str):
            return first['text']
    return ''


async def _request_rm_system_async(url: str, data: str, headers: dict, timeout: int) -> dict:
    """Resolve a Polaris instance and POST the reward_system request.

    Mirrors ``hytuner.utils.reward_score.reward_system.request_rm_system_async``: the
    input ``url`` is only used to derive the Polaris service name; the actual request
    always lands on ``{resolved_host}:10001/reward/RewardService``.
    """
    ip_port = _get_ip_port_random(url)
    if ip_port is None:
        raise RuntimeError(f'Polaris failed to resolve host for {url}')
    resolved_url = f'http://{ip_port}{RewardSystemClient.REWARD_ENDPOINT}'

    async with aiohttp.ClientSession() as session:
        async with session.post(
            resolved_url, data=data, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            try:
                parsed = await response.json(content_type=None)
            except Exception as exc:
                text = await response.text()
                raise RuntimeError(
                    f'Failed to parse JSON (status={response.status}, body[:200]={text[:200]!r}): {exc}'
                ) from exc
            if parsed is None:
                text = await response.text()
                raise RuntimeError(f'Empty JSON body (status={response.status}, body[:200]={text[:200]!r})')
            return parsed


async def request_grm_raw_text(
    messages: list[dict[str, Any]],
    *,
    router_id: str,
    user_token: str,
    exp_id: str,
    override: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str, str]:
    """Send a client-assembled judge prompt through reward_system and return raw text.

    This is the transport used when the *caller* owns the prompt and parses the raw
    model completion itself (a project-specific LLM judge). It reuses the Polaris
    discovery + HTTP primitive above and only bypasses the score-extraction path that a
    prompt-in / free-text-out GRM router does not need.

    Args:
        messages: Full chat messages to send verbatim (the judge prompt is the last
            user turn).
        router_id: ``class1-class2-router_version`` (exactly 3 hyphen parts).
        user_token / exp_id: reward_system credentials.
        override: Decoding params + ``response_format`` (json_schema) placed under
            request ``extra.override`` to constrain the judge's output structure.
        timeout / max_retries: per-attempt timeout and attempt count.

    Returns:
        ``(raw_model_text, request_id)`` on success.

    Raises:
        ValueError: on a missing/placeholder token.
        RuntimeError: if all ``max_retries`` attempts fail (never returns a sentinel).
    """
    if not user_token or user_token in ('', 'xxx'):
        raise ValueError(f'Invalid reward system user token: {user_token!r}')

    class1, class2, router_version = router_id.split('-')
    base_endpoint = f'{RewardSystemClient.BASE_URL}{RewardSystemClient.REWARD_ENDPOINT}'
    headers = {'token': user_token, 'Content-Type': 'application/json'}
    request_id = str(uuid.uuid4()).replace('-', '_')
    data = {
        'exp_id': exp_id or 'grpo',
        'request_id': request_id,
        'messages': messages,
        'class1': class1,
        'class2': class2,
        'router_version': router_version,
        'extra': {'override': override},
        'public_test': [],
    }
    payload = json.dumps(data)

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            result = await _request_rm_system_async(base_endpoint, payload, headers, timeout)
            # reward_system may return "data"/"reward" present-but-null on a server-side
            # error; `.get(key, {})` only substitutes the default when the key is MISSING,
            # so an explicit null would slip through and crash the next `.get`. Coerce both
            # levels to {} so a malformed response degrades to the empty-text retry path.
            reward_data = (result.get('data') or {}).get('reward') or {}
            error_code = reward_data.get('error_code', 0)
            if error_code != 0:
                logger.warning(
                    f'[request_grm_raw_text] router_id={router_id} error_code={error_code}, '
                    f'retry {attempt + 1}/{max_retries}, request_id={request_id}'
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
            text = _extract_choice_text(reward_data)
            if not text:
                logger.warning(
                    f'[request_grm_raw_text] router_id={router_id} empty judge text, '
                    f'retry {attempt + 1}/{max_retries}, request_id={request_id}'
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
            return text, request_id
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning(
                f'[request_grm_raw_text] router_id={router_id} transport error: {exc}, '
                f'spend={time.time() - start_time:.2f}s, retry {attempt + 1}/{max_retries}, request_id={request_id}'
            )
            await asyncio.sleep(_RETRY_DELAY)
            continue

    raise RuntimeError(
        f'reward_system GRM request failed after {max_retries} attempts, router_id={router_id}, request_id={request_id}'
    )
