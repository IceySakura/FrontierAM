"""Host-side helpers for the SWE-Cosmos bugmaker pipeline task.

Kept separate from ``task.py`` so the reusable building blocks (e.g. the per-stage LLM router) are isolated from the task's orchestration flow.
"""

from __future__ import annotations

import logging
import os

from typing import Any

from nexus.core.openai_client import OpenAICompatibleClient
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


logger = logging.getLogger(__name__)


def resolve_question_mark_path(path: str) -> str | None:
    """Expand a single ``/?/`` wildcard in *path* to a real file path.

    ``parent/?/suffix`` looks for the first ``parent/<subdir>/suffix`` that is an
    existing file (``subdir`` is any immediate child of ``parent``). Used when the
    concrete task-split directory is unknown at template time (e.g.
    ``.../splits-6000/?/{repo_name}/....jsonl``).

    Returns:
      - *path* unchanged when it does not contain exactly one ``/?/`` segment;
      - the resolved file path when a match is found;
      - ``None`` when ``/?/`` is present but no matching file exists (or ``parent`` is missing).
    """
    if not path:
        return path
    parts = path.split('/?/')
    if len(parts) != 2:
        return path
    parent_dir, suffix = parts
    if not os.path.isdir(parent_dir):
        return None
    for subdir in os.listdir(parent_dir):
        candidate = f'{parent_dir}/{subdir}/{suffix}'
        if os.path.isfile(candidate):
            return candidate
    return None


# ``BUGMAKER_CLIENT`` / ``ISSUEGEN_CLIENT`` are the *stage* keys under which the host registers the two OpenAI clients (in ``ctx.openai_clients``); they only identify which client carries each stage's model / endpoint / key.
#
# The *routing name* the sandbox actually tags its requests with is ``<bare model id>.<stage>``, NOT these stage keys alone. Each stage sends ``openai/<routing-name>`` (litellm strips the ``openai/`` prefix), so the host routes the bare ``<routing-name>`` back to the matching client (falling back to the first client for an unmapped name). Carrying the real model id keeps it visible in the sandbox's logs and in the ``model`` field of the bugmaker / issuegen metadata, which an opaque stage name could not.
BUGMAKER_CLIENT = 'bugmaker'
ISSUEGEN_CLIENT = 'issuegen'


def model_routing_name(model: str | None, stage: str) -> str:
    """Routing name for a model string: its bare id (the last ``/``-segment).

    e.g. ``bedrock/anthropic.claude-opus-4-8`` -> ``anthropic.claude-opus-4-8``.
    """
    model_name = (model or '').split('/')[-1]
    return f'{model_name}.{stage}' if model_name else ''


_PROFILE_OWNED_REQUEST_FIELDS = frozenset({'thinking', 'output_config', 'extra_headers'})


class RoutingLLMRequestHandler:
    """Route each ``llm_request`` to the host OpenAI client whose name matches the request's model.

    All of a sandbox's LLM calls share one Socket.IO ``llm_request`` stream, so the only per-request discriminator is the ``model`` field. Each pipeline stage sends ``openai/<client-name>`` (litellm strips the prefix), so we forward ``<client-name>`` to the like-named :class:`LLMRequestHandler` — each with its own client and thinking-block cache. An unmapped name falls back to ``default_name`` when given, otherwise to the first registered client.
    """

    def __init__(
        self,
        runtime: Any,
        openai_clients: dict[str, OpenAICompatibleClient],
        default_name: str | None = None,
    ) -> None:
        if not openai_clients:
            raise ValueError('RoutingLLMRequestHandler requires at least one OpenAI client.')
        self._runtime = runtime
        self._by_name: dict[str, LLMRequestHandler] = {
            name: LLMRequestHandler(runtime=runtime, openai_client=client) for name, client in openai_clients.items()
        }
        first = sorted(self._by_name)[0]
        if default_name is None:
            default_name = first
        elif default_name not in self._by_name:
            logger.warning("default OpenAI client '%s' not found; using '%s'", default_name, first)
            default_name = first
        self._default: LLMRequestHandler = self._by_name[default_name]
        self._registered = False

    async def _on_request(self, data: dict[str, Any]) -> None:
        model = data.get('model')
        handler = self._by_name.get(model, self._default) if isinstance(model, str) else self._default
        await handler.handle_llm_request({k: v for k, v in data.items() if k not in _PROFILE_OWNED_REQUEST_FIELDS})

    def register(self) -> None:
        if self._registered:
            return
        self._runtime.on('llm_request', self._on_request)
        self._registered = True
        logger.debug('Routing LLM handler registered (clients: %s)', sorted(self._by_name))

    def unregister(self) -> None:
        if not self._registered:
            return
        self._runtime.off('llm_request', self._on_request)
        self._registered = False
