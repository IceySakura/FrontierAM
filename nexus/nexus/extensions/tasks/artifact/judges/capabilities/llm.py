"""Multimodal LLM calls, client-role resolution, and structured retry failures."""

from __future__ import annotations

import logging

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput


logger = logging.getLogger(__name__)


async def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    label: str = 'call',
) -> Any:
    """Retry an async callable with exponential-jitter backoff.

    max_attempts counts total attempts; 1 disables outer retries. Client transport
    retries are independent, and this helper does not limit total call duration.
    """
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1.0, max=8.0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            try:
                return await fn()
            except Exception as e:
                logger.warning(
                    'retry_with_backoff[%s]: attempt %d/%d failed: %s',
                    label,
                    attempt.retry_state.attempt_number,
                    max_attempts,
                    e,
                )
                raise
    raise RuntimeError(f'retry_with_backoff[{label}]: unreachable')  # pragma: no cover


async def vlm_score(
    client: OpenAICompatibleClient,
    screenshots_b64: list[str],
    prompt: str,
    *,
    max_attempts: int = 1,
) -> str:
    """Send screenshots and a prompt, returning raw response text.

    max_attempts controls outer retries in addition to client transport retries.
    """
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    for shot in screenshots_b64:
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{shot}', 'detail': 'high'},
        })

    async def _call() -> str:
        response = await client.create_completion(
            messages=[{'role': 'user', 'content': content}],
        )
        return response.choices[0].message.content or ''

    if max_attempts <= 1:
        return await _call()
    return await retry_with_backoff(_call, max_attempts=max_attempts, label='vlm_score')


async def multimodal_completion(
    client: OpenAICompatibleClient,
    content: list[dict[str, Any]],
    *,
    max_attempts: int = 1,
) -> str:
    """Send an already-labelled multimodal content list to an OpenAI-compatible client."""

    async def _call() -> str:
        response = await client.create_completion(messages=[{'role': 'user', 'content': content}])
        return response.choices[0].message.content or ''

    if max_attempts <= 1:
        return await _call()
    return await retry_with_backoff(_call, max_attempts=max_attempts, label='multimodal_completion')


async def call_judge_llm(
    inp: JudgeInput,
    *,
    role: str = 'vlm',
    prompt: str,
    images: list[str] | None = None,
    max_attempts: int = 1,
    request_args: dict[str, Any] | None = None,
    label: str = 'call_judge_llm',
) -> str:
    """Resolve the requested role and return text from a multimodal completion.

    Forward request_args to the client; raise no_client or vlm_failed on failure.
    """
    # Keep capability imports lazy.
    from nexus.extensions.tasks.artifact.judges.base import JudgeExecutionError
    from nexus.extensions.tasks.artifact.judges.registry import resolve_client_role

    client = resolve_client_role(inp.openai_clients, role)
    if client is None:
        raise JudgeExecutionError(
            'no_client',
            f'{label}: no client for role {role!r}; available = {sorted(inp.openai_clients)}',
        )

    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    for shot in images or []:
        if not shot:
            continue
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{shot}', 'detail': 'high'},
        })

    async def _call() -> str:
        response = await client.create_completion(
            messages=[{'role': 'user', 'content': content}],
            **(request_args or {}),
        )
        text = response.choices[0].message.content or ''
        # Reject empty responses inside the retry boundary.
        if not text.strip():
            raise ValueError('empty response from LLM')
        return text

    try:
        if max_attempts <= 1:
            return await _call()
        return await retry_with_backoff(_call, max_attempts=max_attempts, label=label)
    except JudgeExecutionError:
        raise
    except Exception as exc:
        raise JudgeExecutionError(
            'vlm_failed',
            f'{label}: {type(exc).__name__}: {exc}',
        ) from exc


__all__ = ['call_judge_llm', 'multimodal_completion', 'retry_with_backoff', 'vlm_score']
