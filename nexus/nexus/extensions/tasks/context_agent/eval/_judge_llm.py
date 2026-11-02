"""The one Nexus-specific piece of this package: how a judge reaches a model.

The evaluators were written against ``contextagent.llm.LLMClient`` and only ever
use ``await llm.chat(messages, **kwargs) -> str | dict``. Nexus has no such
class, so :class:`JudgeLLM` adapts an ``OpenAICompatibleClient`` to that one
method and this module owns endpoint resolution. Nothing else in ``eval/``
needed changing.

**The judge is never the policy.** Under HyTuner RL ``ctx.openai_client``
(``main``) is the model being trained; grading a rollout with it would score
the policy against itself and feed the result back into its own advantage. So
``main`` is not a fallback here at any point: a routed judge comes from a client
the config named ``judge``, or from an endpoint named in the environment.

Endpoint resolution, in order:

1. ``CONTEXTAGENT_GENERAL_JUDGE_BASE_URL`` / ``_API_KEY``
2. ``TAIJI_BASE_URL`` / ``TAIJI_API_KEY`` — what this chain's own config uses,
   so the common case needs no new variables
3. the model-eval gateway default

Clients are cached per (model, base_url) so a 500-row dataset shares one client,
and therefore one concurrency semaphore, instead of opening one per task.
"""

from __future__ import annotations

import asyncio
import logging
import os

from typing import Any

from nexus.core.openai_client import OpenAIConfig, openai_client_registry


logger = logging.getLogger(__name__)

DEFAULT_JUDGE_BASE_URL = 'http://llm-api.model-eval.woa.com/v1'
DEFAULT_JUDGE_MODEL = 'api_ali_qwen3-235b-a22b'
DEFAULT_CONCURRENCY = 8

# ``AsyncOpenAI`` refuses to construct without a key, but plenty of internal
# OpenAI-compatible gateways take none (this repo ships a ``noauth_openai``
# client for exactly that). Sending a placeholder makes a real auth problem
# surface as an HTTP error from the call, instead of as a constructor error on
# a benchmark that may never ask the judge anything.
_PLACEHOLDER_API_KEY = 'EMPTY'

_cache: dict[tuple[str, str], JudgeLLM] = {}


def _env(name: str) -> str:
    return (os.environ.get(name) or '').strip()


def judge_endpoint() -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the judge endpoint."""
    base_url = _env('CONTEXTAGENT_GENERAL_JUDGE_BASE_URL') or _env('TAIJI_BASE_URL') or DEFAULT_JUDGE_BASE_URL
    api_key = _env('CONTEXTAGENT_GENERAL_JUDGE_API_KEY') or _env('TAIJI_API_KEY')
    return base_url, api_key


class JudgeLLM:
    """Adapts a Nexus OpenAI-compatible client to the eval package's ``.chat()``.

    ``concurrency`` is enforced here rather than left to the client, because the
    evaluators fan out per task (AALCR walks every official answer in sequence,
    but a batch run grades many rows at once) and a pinned judge endpoint is a
    shared resource.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        request_args: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._request_args = dict(request_args or {})
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    async def chat(self, messages: list[dict] | str, **kwargs: Any) -> str:
        """Send one request and return the assistant text.

        Returns ``str``, which the evaluators accept alongside the
        ``{"response": ...}`` shape the harness client could also produce.
        """
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        request: dict[str, Any] = {**self._request_args, **kwargs}
        if self._model:
            request['model'] = self._model

        async with self._semaphore:
            response = await self._client.create_completion(messages=messages, **request)

        if not response.choices:
            return ''
        return response.choices[0].message.content or ''

    async def create_completion(
        self,
        messages: list[dict] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the raw Nexus completion for agentic judges.

        Text evaluators use :meth:`chat`. A deliverable-aware judge needs the
        assistant tool calls and provider-specific history fields as well, so
        it consumes the unflattened completion.
        """
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        request: dict[str, Any] = {**self._request_args, **kwargs}
        if self._model:
            request['model'] = self._model
        async with self._semaphore:
            return await self._client.create_completion(messages=messages, tools=tools, **request)


def build_judge_llm(model: str, *, concurrency: int = DEFAULT_CONCURRENCY) -> JudgeLLM:
    """Build (or reuse) a judge that talks to *model* on the judge endpoint."""
    base_url, api_key = judge_endpoint()
    cached = _cache.get((model, base_url))
    if cached is not None:
        return cached

    client_cls = openai_client_registry.get('openai')
    client = client_cls(
        OpenAIConfig(
            client_args={
                'api_key': api_key or _PLACEHOLDER_API_KEY,
                'base_url': base_url,
                'timeout': 1800,
                'max_retries': 5,
            },
            request_args={'model': model},
        )
    )
    judge = JudgeLLM(client, model=model, concurrency=concurrency)
    _cache[(model, base_url)] = judge
    logger.debug('context_agent_general judge: model=%s base_url=%s', model, base_url)
    return judge


class LazyJudgeLLM:
    """Defers judge construction until something actually asks it a question.

    The five rule-based benchmarks (``GraphWalks``, ``LongBenchV2``,
    ``MRCR_*``, ``oolong``) take an ``llm`` argument and never use it. Building
    a client for them would open a connection pool for nothing and — worse —
    turn a missing judge credential into a failure on a benchmark that needs
    no judge at all.
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._judge: JudgeLLM | None = None

    def resolve(self) -> JudgeLLM:
        """Build (once) and return the underlying judge."""
        if self._judge is None:
            self._judge = self._factory()
        return self._judge

    @property
    def resolved(self) -> bool:
        return self._judge is not None

    async def chat(self, messages: list[dict] | str, **kwargs: Any) -> str:
        return await self.resolve().chat(messages, **kwargs)

    async def create_completion(
        self,
        messages: list[dict] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.resolve().create_completion(messages, tools=tools, **kwargs)


def resolve_judge_llm(ctx: Any = None) -> LazyJudgeLLM:
    """Return the judge for the routed (non-pinned) evaluators.

    Prefers an ``openai_clients.judge`` entry from the run config, so an
    operator can point the judge at a different provider entirely. Falls back to
    the environment. ``main`` is deliberately never used — see the module
    docstring. Construction is deferred; see :class:`LazyJudgeLLM`.
    """

    def factory() -> JudgeLLM:
        if ctx is not None:
            try:
                client = ctx.get_openai_client('judge')
            except Exception:
                client = None
            if client is not None:
                return JudgeLLM(
                    client,
                    model=_env('CONTEXTAGENT_GENERAL_JUDGE_MODEL') or None,
                    concurrency=DEFAULT_CONCURRENCY,
                )
        return build_judge_llm(_env('CONTEXTAGENT_GENERAL_JUDGE_MODEL') or DEFAULT_JUDGE_MODEL)

    return LazyJudgeLLM(factory)
