"""Async reward system client for Nexus.

Ported from hytuner/utils/reward_score/reward_system.py to be used independently
within Nexus tasks/evaluators without requiring hytuner dependencies.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid

from collections.abc import Sequence
from urllib.parse import urlparse

import aiohttp

from nexus.utils.multimodal import place_images_in_question
from nexus.utils.reward_score import ReasoningReward


logger = logging.getLogger(__name__)

# Polaris consumer creation talks to the local agent over RPC and is expensive enough
# that creating one per request causes real trouble under concurrency. The original
# hytuner implementation (hytuner/utils/reward_score/reward_system.py::get_consumer_api)
# caches it in a module-level global; this port previously called
# create_consumer_by_default_config_file() on every single call, which is a regression.
# Restore the cache here (with a lock, since callers run from multiple asyncio tasks /
# threads).
_consumer_lock = threading.Lock()
_consumer_api = None


def _get_consumer_api():
    """Return a process-wide cached Polaris consumer, creating it at most once."""
    global _consumer_api
    if _consumer_api is None:
        with _consumer_lock:
            if _consumer_api is None:
                from polaris.api.consumer import create_consumer_by_default_config_file

                _consumer_api = create_consumer_by_default_config_file()
    return _consumer_api


def _get_ip_port_random(url: str) -> str | None:
    """Resolve service address via Polaris service discovery."""
    from polaris.api.consumer import GetInstancesRequest

    namespace = 'Production'
    service = urlparse(url.split('.')[0]).netloc
    consumer = _get_consumer_api()
    instances = consumer.get_all_instances(GetInstancesRequest(namespace=namespace, service=service))
    hosts = [inst.get_host() for inst in instances]
    if not hosts:
        logger.error('Polaris returned no hosts for service: %s', service)
        return None
    return f'{random.choice(hosts)}:10001'


def resolve_strip_multimodal_config(
    strip_multimodal_content: bool | list[str] | None,
) -> tuple[bool, set[str]]:
    """Resolve ``strip_multimodal_content`` into ``(global_flag, router_ids)``.

    Mirrors ``hytuner.reward_loop.reward_loop_manager_extensions`` so a router
    configured as text-only on the hytuner side behaves the same when nexus owns
    the scoring call:

    - ``bool``: strip for every router, or for none.
    - ``list[str]``: strip only for the listed router_ids.
    - anything else: strip for none.
    """
    if isinstance(strip_multimodal_content, bool):
        return strip_multimodal_content, set()

    if isinstance(strip_multimodal_content, list):
        if all(isinstance(router_id, str) for router_id in strip_multimodal_content):
            return False, set(strip_multimodal_content)
        logger.warning(
            'strip_multimodal_content has invalid element types, expected list[str]; '
            'treating as empty (no router will be stripped).'
        )
        return False, set()

    if strip_multimodal_content is not None:
        logger.warning(
            'Unexpected type for strip_multimodal_content: %s; treating as empty (no router will be stripped).',
            type(strip_multimodal_content).__name__,
        )
    return False, set()


def build_user_content(question: str, image_urls: Sequence[str] | None) -> str | list[dict]:
    """Build the user message content for one scored row.

    Layout is delegated to ``place_images_in_question``, the same call the actor
    prompt goes through, so the judge reads the question the policy answered
    rather than a second arrangement of the same pieces.

    Returns the bare question string when there is no image, keeping text-only
    routers on the exact payload shape they saw before multimodal support.
    """
    if not image_urls:
        return question

    image_parts: list[dict] = [
        {'type': 'image_url', 'image_url': {'url': _resolved_image_url(url)}} for url in image_urls
    ]
    return place_images_in_question(question, image_parts)


def _resolved_image_url(image_url: str) -> str:
    """Validate that an image is fetchable by the remote reward service."""
    url = image_url.strip() if isinstance(image_url, str) else ''
    if url.startswith(('data:image/', 'http://', 'https://')):
        return url
    # A local path would silently score as a broken image on the reward side,
    # so callers must resolve it to a data URI or an HTTP URL before scoring.
    raise ValueError(f'Reward system image must be a data URI or an HTTP(S) URL, got: {image_url!r:.80}')


# Base64 image payloads run to megabytes; logging them verbatim buries the request log.
_LOGGED_IMAGE_URL_LIMIT = 64
_LOGGED_MESSAGES_LIMIT = 4096


def redact_messages_for_log(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with long media payloads replaced by a size marker."""
    redacted = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            redacted.append(message)
            continue
        redacted.append({**message, 'content': [_redact_content_item(item) for item in content]})
    return redacted


def _redact_content_item(item):
    if not isinstance(item, dict) or item.get('type') not in {'image', 'image_url', 'file', 'video', 'video_url'}:
        return item

    def redact(value):
        if isinstance(value, str) and len(value) > _LOGGED_IMAGE_URL_LIMIT:
            return f'{value[:_LOGGED_IMAGE_URL_LIMIT]}...<{len(value)} chars>'
        if isinstance(value, dict):
            return {key: redact(val) if key in {'url', 'file_data'} else val for key, val in value.items()}
        return value

    return {
        key: redact(value) if key in {'image', 'image_url', 'file', 'video', 'video_url'} else value
        for key, value in item.items()
    }


def _convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists."""
    try:
        import numpy as np
    except ImportError:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    return obj


class RewardSystemClient:
    """Async client for the reward system API.

    Usage::

        client = RewardSystemClient(user_token='xxx', exp_id='my_exp')
        score = await client.get_score(
            messages=[
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': 'Solve x^2 = 4'},
                {'role': 'assistant', 'content': 'x = 2'},
            ],
            ref_answer='x = ±2',
            router_id='math-algebra-v1',
        )
    """

    BASE_URL = 'http://api-reward-platform-api-reward-platform-354ba565.turbotke.production.polaris:10001'
    REWARD_ENDPOINT = '/reward/RewardService'

    def __init__(
        self,
        user_token: str | None = None,
        exp_id: str = '',
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 1800,
    ):
        resolved_user_token = (
            user_token or os.getenv('NEXUS_REWARD_SYSTEM_TOKEN') or os.getenv('nexus_reward_system_token')  # noqa: SIM112
        )
        if not resolved_user_token:
            raise ValueError('user_token is required')
        resolved_exp_id = (
            exp_id or os.getenv('NEXUS_REWARD_SYSTEM_EXP_ID') or os.getenv('nexus_reward_system_exp_id')  # noqa: SIM112
        )
        self.user_token = resolved_user_token
        self.exp_id = resolved_exp_id or 'nexus'
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.headers = {'token': self.user_token, 'Content-Type': 'application/json'}

    def _build_request_data(
        self,
        messages: list[dict],
        ref_answer: str,
        router_id: str,
        extra_param: dict | None = None,
        unittests: dict | list | str | None = None,
    ) -> dict:
        class1, class2, router_version = router_id.split('-')
        extra_param = extra_param or {}
        return {
            'exp_id': self.exp_id,
            'request_id': str(uuid.uuid4()).replace('-', '_'),
            'messages': copy.deepcopy(messages),
            'ref_answer': ref_answer,
            'class1': class1,
            'class2': class2,
            'router_version': router_version,
            'length_penalty_beta': extra_param.get('length_penalty_beta', 0.0),
            'extra': _convert_numpy_to_list(extra_param),
            'public_test': unittests or {},
        }

    async def _request(self, data: dict) -> dict:
        """Send a single request to the reward system endpoint."""
        ip_port = _get_ip_port_random(self.BASE_URL)
        if not ip_port:
            raise RuntimeError('Failed to resolve reward system service address')
        url = f'http://{ip_port}{self.REWARD_ENDPOINT}'
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=json.dumps(data),
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                body_text = await resp.text()
                if resp.status != 200:
                    body_preview = body_text[:500]
                    raise RuntimeError(f'Reward API HTTP {resp.status}, body={body_preview!r}')

                try:
                    return json.loads(body_text)
                except json.JSONDecodeError as exc:
                    content_type = resp.headers.get('Content-Type', '')
                    body_preview = body_text[:500]
                    raise RuntimeError(
                        f'Reward API returned non-JSON response (content-type={content_type!r}), body={body_preview!r}'
                    ) from exc

    async def get_score(
        self,
        messages: list[dict],
        ref_answer: str,
        router_id: str,
        extra_param: dict | None = None,
        unittests: dict | list | str | None = None,
        ci_test: bool = False,
        *,
        log_summary: bool = False,
    ) -> float:
        """Get reward score for a conversation (messages end with assistant reply).

        Returns:
            float score (typically 0.0~1.0, or negative on failure).
        """
        if ci_test:
            return self._ci_test_score(messages, ref_answer, router_id)

        data = self._build_request_data(messages, ref_answer, router_id, extra_param, unittests)
        if log_summary:
            logger.debug(
                '[reward_system] request_id=%s, router_id=%s, message_count=%s',
                data.get('request_id'),
                router_id,
                len(data['messages']),
            )
            if logger.isEnabledFor(logging.DEBUG):
                preview = repr(redact_messages_for_log(data['messages']))
                if len(preview) > _LOGGED_MESSAGES_LIMIT:
                    preview = f'{preview[:_LOGGED_MESSAGES_LIMIT]}...<{len(preview)} chars>'
                logger.debug('[reward_system] request_id=%s, messages=%s', data.get('request_id'), preview)
        else:
            logger.debug(
                '[reward_system] request_id=%s, router_id=%s, messages=%s',
                data.get('request_id'),
                router_id,
                redact_messages_for_log(data.get('messages', [])),
            )

        had_response_failure = False
        had_exception = False

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            request_id = data['request_id']
            try:
                result = await self._request(data)
                reward_data = result.get('data', {}).get('reward', {})
                error_code = reward_data.get('error_code', 1)
                final_score = reward_data.get('final_score', ReasoningReward.FAILED.value)

                if error_code != 0:
                    had_response_failure = True
                    logger.warning(
                        '[reward_system] request_id=%s failed (attempt %d/%d), error_code=%s',
                        request_id,
                        attempt,
                        self.max_retries,
                        error_code,
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue

                if not isinstance(final_score, (float, int)):
                    had_response_failure = True
                    logger.warning('[reward_system] request_id=%s, invalid final_score type', request_id)
                    await asyncio.sleep(self.retry_delay)
                    continue

                reward = max(final_score, 0.0)

                logger.info(
                    '[reward_system] router_id=%s, reward=%.4f, time=%.2fs, request_id=%s',
                    router_id,
                    reward,
                    time.time() - t0,
                    request_id,
                )
                return reward

            except Exception as e:
                had_exception = True
                logger.error(
                    '[reward_system] exception: %s, router_id=%s, time=%.2fs, request_id=%s',
                    e,
                    router_id,
                    time.time() - t0,
                    request_id,
                )
                await asyncio.sleep(self.retry_delay)

        if had_response_failure:
            logger.error(
                '[reward_system] all %d retries failed with invalid reward response for router_id=%s; '
                'return RS_FAILED(%s)',
                self.max_retries,
                router_id,
                ReasoningReward.RS_FAILED.value,
            )
            return float(ReasoningReward.RS_FAILED.value)

        if had_exception:
            logger.error(
                '[reward_system] all %d retries failed with request exceptions for router_id=%s; return FAILED(%s)',
                self.max_retries,
                router_id,
                ReasoningReward.FAILED.value,
            )
            return float(ReasoningReward.FAILED.value)

        logger.error(
            '[reward_system] all %d retries failed for router_id=%s with unknown reason; return FAILED(%s)',
            self.max_retries,
            router_id,
            ReasoningReward.FAILED.value,
        )
        return float(ReasoningReward.FAILED.value)

    def _ci_test_score(self, messages: list[dict], ref_answer: str, router_id: str) -> float:
        """Deterministic score for CI testing via MD5 hash."""
        class1, class2, router_version = router_id.split('-')
        hash_input = str(messages) + ref_answer + class1 + class2 + router_version
        return abs(int(hashlib.md5(hash_input.encode()).hexdigest(), 16)) % 5
