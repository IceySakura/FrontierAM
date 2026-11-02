"""Server-Sent Events encoding helpers for live event projections."""

from __future__ import annotations

import json

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from nexus.runner.trace import RunTraceStreamEvent


def prefix_runtime_url(value: str | None, base_path: str) -> str | None:
    if not value or not base_path or not value.startswith('/'):
        return value
    if value == base_path or value.startswith(f'{base_path}/'):
        return value
    return f'{base_path}{value}'


def prefix_runtime_urls(data: Any, base_path: str) -> Any:
    """Prefix run-scoped runtime URLs in API/SSE payloads with the live server base path."""
    if not base_path or not isinstance(data, dict):
        return data
    runtimes = data.get('runtimes')
    if not isinstance(runtimes, dict):
        return data
    for runtime_data in runtimes.values():
        if not isinstance(runtime_data, dict):
            continue
        for key in ('proxy_url', 'debug_gui_url'):
            runtime_data[key] = prefix_runtime_url(runtime_data.get(key), base_path)
    return data


def sse_bytes(event: RunTraceStreamEvent, *, base_path: str = '') -> bytes:
    data = {
        'id': event.id,
        'type': event.type,
        'time': datetime.fromtimestamp(event.time_unix_nano / 1e9, tz=timezone.utc).isoformat(),
        'data': deepcopy(event.data),
    }
    prefix_runtime_urls(data['data'], base_path)
    return f'id: {event.id}\nevent: {event.type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'.encode()
