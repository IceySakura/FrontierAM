"""Responses client for AIHub's multi-resource Azure pool.

AIHub load-balances gpt-5.6-sol across Azure resources. Replaying a Responses
item id created by another resource fails with UpstreamParamError 3002
("created under a different Azure OpenAI resource"). Drop the ids and keep
``encrypted_content`` so cross-turn thinking still round-trips.
"""

from __future__ import annotations

from typing import Any

from nexus.core import openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient


_STATEFUL_ITEM_TYPES = frozenset({'computer_call', 'function_call', 'message', 'reasoning'})


def strip_responses_item_ids(value: Any) -> Any:
    """Return a copy of a chat payload with provider-bound Responses ids removed."""
    if isinstance(value, dict):
        stripped = {key: strip_responses_item_ids(item) for key, item in value.items()}
        if stripped.get('type') in _STATEFUL_ITEM_TYPES:
            stripped.pop('id', None)
        return stripped
    if isinstance(value, list):
        return [strip_responses_item_ids(item) for item in value]
    return value


@openai_client_registry.register('aihub_responses')
class AihubResponsesClient(OpenAIResponsesClient):
    """OpenAI Responses client that does not replay Azure-resource-bound item ids."""

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        return await super()._create_completion(strip_responses_item_ids(messages), model, **kwargs)


__all__ = ['AihubResponsesClient', 'strip_responses_item_ids']
