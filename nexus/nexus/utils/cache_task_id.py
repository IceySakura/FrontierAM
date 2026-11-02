"""Read the effective client cache ID."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl


def get_input_cache_task_id(client: Any) -> str | None:
    """Read the selected input before provider-specific transformations."""
    args = getattr(getattr(client, 'config', None), 'client_args', {})
    explicit = args.get('cache_task_id')
    if explicit not in (None, ''):
        return str(explicit)
    routing = args.get('gateway_routing')
    if isinstance(routing, Mapping) and routing.get('cache_task_id') not in (None, ''):
        return str(routing['cache_task_id'])
    api_key = args.get('api_key')
    if not isinstance(api_key, str):
        return None
    return dict(parse_qsl(api_key.partition('?')[2])).get('cache_task_id')


def get_cache_task_id(client: Any) -> str | None:
    # A request may explicitly remove the configured ID, so None must not fall back.
    if hasattr(client, '_request_cache_task_id'):
        return client._request_cache_task_id
    config_args = getattr(getattr(client, 'config', None), 'client_args', {})
    api_keys = (
        getattr(getattr(client, 'client', None), 'api_key', None),
        getattr(getattr(client, '_adapter', None), 'api_key', None),
        getattr(client, 'client_args', {}).get('api_key'),
        config_args.get('api_key'),
    )
    api_key = next((key for key in api_keys if isinstance(key, str)), '')
    return dict(parse_qsl(api_key.partition('?')[2])).get('cache_task_id')
