"""The one ``cache_task_id`` for a trajectory.

SWE and frontend generators both call ``cache_task_overlay``. ``load_config`` does not, so a
topic that never calls it leaves every wire unchanged.

Return the ``openai_clients.main.config`` overlay and the effective id for ``metadata``.
An absent overlay leaves the input unchanged; it may already carry a caller-supplied id.

Every gateway gets an id unless its host is listed in ``GATEWAYS_WITHOUT_CACHE_TASK_ID``.
``openai_responses`` and ``litellm`` receive ``client_args.cache_task_id`` and fold it into the
key themselves: the runner only ``setdefault``s, so this explicit value wins over
``cache_task_id_from_run_id``. Writing the id into the query instead would be replaced by that
derived value, and ``metadata`` would name an id the request never carried. Other clients do not
take the argument, so the id is appended to the api_key query here.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, quote


#: Clients that accept ``cache_task_id`` as a client argument and fold it into the key themselves.
CLIENTS_TAKING_CACHE_ARG = frozenset({'openai_responses', 'litellm'})

#: Gateways that must NOT have the parameter appended. Everything else gets one: the id is a
#: hard requirement downstream, so the default is "carry it" and each omission is named here.
#: Taiji and hunyuanapi-rd take a bare key; rewriting it has produced gateway rejections before.
GATEWAYS_WITHOUT_CACHE_TASK_ID = (
    'api.taiji.woa.com',
    'infer-api.taiji.woa.com',
    'hunyuanapi-rd.woa.com',
)


def _gateway(client_args: dict[str, Any]) -> str:
    """Where this wire points: ``endpoint`` when it has one, else ``base_url``."""
    return str(client_args.get('endpoint') or client_args.get('base_url') or '')


def cache_task_overlay(
    client_name: str,
    client_args: dict[str, Any],
    api_key: str,
    cache_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the client overlay and the id the gateway will receive, preserving caller choices."""
    if not api_key:
        return None, None
    query_ids = [
        value for name, value in parse_qsl(api_key.partition('?')[2], keep_blank_values=True) if name == 'cache_task_id'
    ]
    # Match the clients' query reader: the last non-empty value wins.
    query_id = next((value for value in reversed(query_ids) if value), None)
    if 'cache_task_id' in client_args:
        supplied_id = client_args['cache_task_id']
        if client_name in CLIENTS_TAKING_CACHE_ARG and supplied_id not in (None, ''):
            if not isinstance(supplied_id, str):
                raise TypeError('cache_task_id must be a string')
            return None, supplied_id
        return None, query_id
    if query_ids:
        # Pin a query id as an argument too, so the runner's setdefault cannot replace it.
        overlay = (
            {'client_args': {'cache_task_id': query_id or ''}} if client_name in CLIENTS_TAKING_CACHE_ARG else None
        )
        return overlay, query_id
    gateway = _gateway(client_args)
    if any(host in gateway for host in GATEWAYS_WITHOUT_CACHE_TASK_ID):
        return None, None
    if client_name in CLIENTS_TAKING_CACHE_ARG:
        return {'client_args': {'cache_task_id': cache_id}}, cache_id
    # Percent-encode the value and tolerate a key that already ends in a separator. Today every
    # id is md5 hex so the encoding is a no-op, but the id is an argument: a future caller passing
    # something with '&' or '=' would otherwise forge extra query parameters onto the key.
    stem = api_key.rstrip('?&')
    separator = '&' if '?' in stem else '?'
    return {'client_args': {'api_key': f'{stem}{separator}cache_task_id={quote(cache_id, safe="")}'}}, cache_id
