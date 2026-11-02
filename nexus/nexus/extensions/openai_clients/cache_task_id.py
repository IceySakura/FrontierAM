"""Gateway cache task configuration and initialization tracing."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, unquote_plus

from nexus.trace import span_context
from nexus.utils.cache_task_id import get_cache_task_id, get_input_cache_task_id


def normalize_cache_task_id(api_key: Any, cache_task_id: str) -> str:
    """Replace only the cache-task query parameter, preserving all other parameters."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError('cache_task_id requires a non-empty api_key')
    credential, _, query = api_key.partition('?')
    if not credential.strip():
        raise ValueError('cache_task_id requires a non-empty api_key credential')
    parts = query.split('&') if query else []
    parts = [part for part in parts if unquote_plus(part.partition('=')[0]) != 'cache_task_id']
    parts.append(f'cache_task_id={quote(cache_task_id, safe="")}')
    return f'{credential}?{"&".join(parts)}'


def prepare_cache_task_id(client_args: dict[str, Any]) -> dict[str, Any]:
    """Move the Nexus-only option into the API-key query sent to the gateway."""
    client_args = dict(client_args)
    cache_task_id = client_args.pop('cache_task_id', None)
    if cache_task_id not in (None, ''):
        if not isinstance(cache_task_id, str):
            raise TypeError('cache_task_id must be a string')
        client_args['api_key'] = normalize_cache_task_id(client_args.get('api_key'), cache_task_id)
    return client_args


def observe_client_init(client: Any) -> str | None:
    """Record the API-key cache ID once at initialization, including missing IDs."""
    cache_task_id = get_cache_task_id(client)
    with span_context(
        'openai_client_init',
        attributes={
            'input_cache_task_id': get_input_cache_task_id(client),
            'cache_task_id': cache_task_id,
        },
    ):
        pass
    return cache_task_id
