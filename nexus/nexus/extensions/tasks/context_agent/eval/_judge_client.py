"""Compatibility helpers for evaluators that require a separately resolved Judge.

Current Context Agent evaluators use the Judge supplied by the task, including
AALCR and LongBenchPro. The helpers remain available for external evaluators
that intentionally need a fixed model.

It is built through :mod:`._judge_llm` rather than a private ``AsyncOpenAI``,
which is what the upstream evaluators did. That buys the retry ladder, the
concurrency cap, the hard timeout and endpoint resolution for free, and means
these modules do not each carry their own copy of "how do I reach a model".
(In the harness this went through ``contextagent.llm.LLMClient``; ``_judge_llm``
is the Nexus stand-in and the only adapted file in this package.)

Override the pinned model with ``CTXAGENT_JUDGE_MODEL_<NAME>`` when you have to
— a gateway that has retired it, say — but a run that does so is no longer
comparable with a published figure, so it is deliberately not a CLI flag.
"""

from __future__ import annotations

import logging
import os

from typing import Any


logger = logging.getLogger(__name__)

_clients: dict[str, Any] = {}


def pinned_client(name: str, default_model: str, concurrency: int = 8):
    """Return a cached client for the judge *name* pins.

    Cached per resolved model name so a 500-row dataset shares one client (and
    therefore one concurrency semaphore) instead of opening one per task.
    """
    model = os.environ.get(f'CTXAGENT_JUDGE_MODEL_{name.upper()}', '').strip() or default_model
    client = _clients.get(model)
    if client is None:
        from ._judge_llm import build_judge_llm

        client = build_judge_llm(model, concurrency=concurrency)
        _clients[model] = client
        if model != default_model:
            logger.warning(
                '%s judge model overridden to %r (published numbers used %r)',
                name,
                model,
                default_model,
            )
        else:
            logger.debug('%s judge: %s', name, model)
    return client


async def ask(client, prompt: str, **kwargs: Any) -> str:
    """Send one single-turn prompt and return the text.

    ``LLMClient.chat`` returns ``str`` normally and
    ``{"reasoning_content", "response"}`` from a model that emits a chain of
    thought, so the text has to be unwrapped rather than assumed.
    """
    raw = await client.chat([{'role': 'user', 'content': prompt}], **kwargs)
    if isinstance(raw, dict):
        return raw.get('response') or raw.get('content') or ''
    return raw or ''
