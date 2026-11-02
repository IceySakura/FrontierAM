"""Direct transport for the department GPT-OSS judge service."""

from __future__ import annotations

import asyncio
import json
import random
import uuid

from collections.abc import Callable
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

import aiohttp


DEFAULT_GPTOSS_MODEL = 'hy-t2t-hyeval-critique-gpt-oss-120b'
DEFAULT_CONCURRENCY = 8

T = TypeVar('T')


class GptOssJudgeError(RuntimeError):
    """Raised after a GPT-OSS judge request exhausts its allowed attempts."""


class GptOssHttpError(RuntimeError):
    """HTTP failure from the direct GPT-OSS endpoint."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        super().__init__(f'GPT-OSS HTTP {status}: {body[:1000]}')

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 425, 429} or self.status >= 500


_semaphores: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[int, asyncio.Semaphore],
] = WeakKeyDictionary()


def _semaphore(concurrency: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    by_limit = _semaphores.setdefault(loop, {})
    limit = max(1, concurrency)
    if limit not in by_limit:
        by_limit[limit] = asyncio.Semaphore(limit)
    return by_limit[limit]


def _headers(token: str, wsid: str) -> dict[str, str]:
    return {
        'Content-Type': 'application/json',
        'Authorization': token if token.startswith('Bearer ') else f'Bearer {token}',
        'Wsid': str(wsid),
    }


def build_request(
    *,
    model: str,
    query_prefix: str,
    system_prompt: str,
    user_content: str,
) -> dict[str, Any]:
    """Build the HY-specific, non-streaming critique-model request."""
    return {
        'model': model,
        'query_id': query_prefix + uuid.uuid4().hex,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ],
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': 0,
        'repetition_penalty': 1.0,
        'max_input_seq_len': 131_072,
        'output_seq_len': 8_192,
        'stream': False,
        'openai_infer': True,
    }


async def post_gptoss(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST one request and validate the GPT-OSS response envelope."""
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout, connect=min(60, timeout)),
        connector=connector,
    ) as session:
        async with session.post(url, json=payload, ssl=False) as response:
            text = await response.text()
            if response.status != 200:
                raise GptOssHttpError(response.status, text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'GPT-OSS non-JSON response: {text[:500]}') from exc
    if not isinstance(value, dict):
        raise RuntimeError('GPT-OSS response must be a JSON object')
    return value


async def judge_json(
    *,
    url: str,
    token: str,
    wsid: str,
    model: str,
    timeout: float,
    max_retries: int,
    concurrency: int,
    query_prefix: str,
    system_prompt: str,
    user_content: str,
    validator: Callable[[dict[str, Any]], T],
) -> tuple[T, dict[str, Any]]:
    """Call GPT-OSS until transport, JSON, and the result schema validate."""
    if not url or not token or not wsid or not model:
        raise ValueError('GPT-OSS url, token, wsid, and model are required')

    headers = _headers(token, wsid)
    attempts = max(1, max_retries)
    attempted = 0
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempted += 1
        payload = build_request(
            model=model,
            query_prefix=query_prefix,
            system_prompt=system_prompt,
            user_content=user_content,
        )
        try:
            async with _semaphore(concurrency):
                response = await post_gptoss(
                    url,
                    headers=headers,
                    payload=payload,
                    timeout=timeout,
                )
            choices = response.get('choices') or []
            if not choices:
                raise ValueError('GPT-OSS response has no choices')
            content = ((choices[0] or {}).get('message') or {}).get('content') or ''
            decoder = json.JSONDecoder()
            raw = None
            for index, char in enumerate(content.strip()):
                if char != '{':
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content.strip()[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    raw = candidate
                    break
            if raw is None:
                raise ValueError(f'no JSON object in GPT-OSS response: {content[:300]!r}')
            return validator(raw), raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if isinstance(exc, GptOssHttpError) and not exc.retryable:
                break
            if attempt + 1 < attempts:
                await asyncio.sleep(min(2 ** (attempt + 1), 60) + random.random() * 0.5)

    raise GptOssJudgeError(
        f'GPT-OSS judge failed after {attempted} attempt(s): {type(last_error).__name__}: {last_error}'
    ) from last_error


__all__ = [
    'DEFAULT_GPTOSS_MODEL',
    'GptOssHttpError',
    'GptOssJudgeError',
    'build_request',
    'judge_json',
    'post_gptoss',
]
