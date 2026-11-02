"""Give each hy-live model conversation its own gateway ``cache_task_id``.

The gateway routes requests that share a ``cache_task_id`` to the same node, which is what lets a
conversation's growing prefix hit the prefix cache. Each conversation (cerebellum, midbrain, every
project Session, user simulator) therefore gets a fresh id of its own for the episode.
"""

from __future__ import annotations

import logging
import uuid

from dataclasses import replace
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from nexus.core import OpenAICompatibleClient


logger = logging.getLogger(__name__)

ClientT = TypeVar('ClientT', bound=OpenAICompatibleClient)

_CACHE_TASK_ID_CLIENTS = frozenset({
    'nexus.extensions.openai_clients.openai.OpenAIClient',
    'nexus.extensions.openai_clients.azure_openai.AzureOpenAIClient',
    'nexus.extensions.openai_clients.infer_proxy.InferProxyClient',
    'nexus.extensions.openai_clients.litellm.LiteLLMClient',
    'nexus.extensions.openai_clients.openai_responses.OpenAIResponsesClient',
})
"""Clients that read ``client_args.cache_task_id``; any other client may reject the unknown argument."""


class CacheTaskIdConfig(BaseModel):
    """Which hy-live conversations send their own ``cache_task_id``."""

    model_config = ConfigDict(extra='forbid')

    cerebellum: bool = False
    """Off by default: under HyTuner the cerebellum client is the trainable policy, which has no gateway."""
    midbrain: bool = True
    backend: bool = True
    """One id per project Session; a question fork shares its source Session's."""
    user_simulator: bool = True


def supports_cache_task_id(client: OpenAICompatibleClient) -> bool:
    cls = type(client)
    return f'{cls.__module__}.{cls.__qualname__}' in _CACHE_TASK_ID_CLIENTS


def with_cache_task_id(client: ClientT, cache_task_id: str) -> ClientT:
    """Return a client of the same kind and config as ``client`` whose requests carry ``cache_task_id``.

    The copy keeps ``_client_key``, so trace processors attribute its calls to the same client.
    """
    config = replace(client.config, client_args={**client.config.client_args, 'cache_task_id': cache_task_id})
    scoped = type(client)(config)
    client_key = getattr(client, '_client_key', None)
    if client_key is not None:
        scoped._client_key = client_key
    return scoped


def conversation_client(client: ClientT, *, conversation: str, enabled: bool, required: bool = False) -> ClientT:
    """Return ``client``, or a copy with a fresh ``cache_task_id`` when ``enabled``.

    A client that cannot carry the id is used unchanged with a warning, unless ``required``.

    Raises:
        ValueError: ``required`` and the client cannot carry a ``cache_task_id``.
    """
    if not enabled:
        return client
    if not supports_cache_task_id(client):
        message = f'{type(client).__name__} cannot send a cache_task_id for the hy-live {conversation}'
        if required:
            raise ValueError(f'{message}; turn off cache_task_id.{conversation}')
        logger.warning('%s; its requests go out without one', message)
        return client
    return with_cache_task_id(client, uuid.uuid4().hex)


__all__ = ['CacheTaskIdConfig', 'conversation_client', 'supports_cache_task_id', 'with_cache_task_id']
