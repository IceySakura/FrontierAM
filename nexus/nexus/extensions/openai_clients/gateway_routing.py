from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode


_ROUTING_KEYS = {'provider', 'model', 'api', 'timeout', 'cache_task_id'}


def merge_gateway_routing(api_key: Any, routing: Any) -> str:
    """Merge Nexus gateway routing parameters into an API-key query string.

    Gateway credentials use ``credential?key=value`` rather than a URL. Existing
    query parameters (notably ``cache_task_id``) are retained, while explicit
    routing values replace any existing value with the same key.
    """

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError('gateway routing requires a non-empty api_key')
    if not isinstance(routing, Mapping):
        raise TypeError('gateway_routing must be a mapping')
    if not routing:
        return api_key

    if any(not isinstance(key, str) for key in routing):
        raise TypeError('gateway_routing parameter names must be strings')
    unknown = set(routing) - _ROUTING_KEYS
    if unknown:
        raise ValueError(f'Unsupported gateway_routing parameters: {sorted(unknown)}')
    for key, value in routing.items():
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(f'gateway_routing.{key} must be a string or integer')

    credential, separator, raw_query = api_key.partition('?')
    if not credential.strip():
        raise ValueError('gateway routing requires a non-empty api_key credential')

    merged: dict[str, str] = {}
    if separator:
        for key, value in parse_qsl(raw_query, keep_blank_values=True):
            merged[key] = value
    for key, value in routing.items():
        merged[str(key)] = str(value)

    encoded = urlencode(merged)
    return f'{credential}?{encoded}' if encoded else credential


__all__ = ['merge_gateway_routing']
