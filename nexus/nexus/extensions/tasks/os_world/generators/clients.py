"""Local InferProxy configuration shared by Hunyuan CC and Codex.

Taiji supplies bridge_litellm independently; neither client chooses a harness.
"""

from __future__ import annotations

import os

from typing import Any

from nexus.extensions.tasks.os_world.data.task_source import OSWorldBatchTask


INFER_PROXY_STRATEGY_TYPE = 'ConsistentHash'


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, '').strip() or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer, got {raw!r}') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer, got {raw!r}')
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, '').strip() or str(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive number, got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive number, got {raw!r}')
    return value


def _normalized_api_base() -> str:
    """Return the OpenAI SDK base URL, accepting a full endpoint for convenience."""

    value = os.getenv('HUNYUAN_BASE_URL', '').strip().rstrip('/')
    suffix = '/chat/completions'
    if value.endswith(suffix):
        value = value[: -len(suffix)].rstrip('/')
    if not value:
        raise ValueError('HUNYUAN_BASE_URL must not be empty')
    return value


def _serving_profile() -> str:
    value = os.getenv('HUNYUAN_SERVING_PROFILE', 'claude_passthrough').strip().lower()
    if value not in {'claude_passthrough', 'native_hunyuan'}:
        raise ValueError(f'HUNYUAN_SERVING_PROFILE must be claude_passthrough or native_hunyuan, got {value!r}')
    return value


def _hunyuan_model() -> str:
    model = os.getenv('HUNYUAN_MODEL', '').strip()
    if not model:
        raise ValueError('HUNYUAN_MODEL must not be empty')
    return model


def _hunyuan_request_args() -> dict[str, Any]:
    model = _hunyuan_model()

    reasoning_effort = os.getenv('HUNYUAN_REASONING_EFFORT', 'high').strip()
    if not reasoning_effort:
        raise ValueError('HUNYUAN_REASONING_EFFORT must not be empty')

    request_args: dict[str, Any] = {
        'model': model,
        'reasoning_effort': reasoning_effort,
        'stream': False,
    }
    if _serving_profile() == 'native_hunyuan':
        request_args.update({
            'temperature': float(os.getenv('HUNYUAN_TEMPERATURE', '0.0')),
            'top_p': float(os.getenv('HUNYUAN_TOP_P', '0.95')),
            'max_tokens': _positive_int_env('HUNYUAN_MAX_TOKENS', 1024),
        })
    return request_args


def _hunyuan_openai_clients(task: OSWorldBatchTask) -> dict[str, Any]:
    # AsyncOpenAI requires a non-empty credential even though this internal
    # endpoint does not authenticate requests. The endpoint accepts this
    # non-secret placeholder; a real key can still be supplied explicitly.
    api_key = '${oc.env:HUNYUAN_API_KEY,unused}'
    request_args = _hunyuan_request_args()
    request_args['extra_headers'] = {
        'StrategyType': INFER_PROXY_STRATEGY_TYPE,
        'SessionID': '${oc.env:HUNYUAN_SESSION_ID,osworld-' + task.task_id + '}',
    }
    enforced = {'extra_headers': request_args['extra_headers']}
    if _serving_profile() == 'native_hunyuan':
        enforced.update({key: request_args[key] for key in ('temperature', 'top_p', 'max_tokens')})

    return {
        'main': {
            'name': 'infer_proxy',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': _normalized_api_base(),
                    'timeout': _positive_float_env('HUNYUAN_REQUEST_TIMEOUT', 600.0),
                    'max_retries': _positive_int_env('HUNYUAN_MAX_REQUEST_RETRIES', 5),
                },
                'request_args': request_args,
                'enforced_request_args': enforced,
            },
        }
    }
