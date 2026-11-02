"""Shared configuration helpers for GUI agent integration flows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MODEL_SERVICE_API_BASE_URL_ENV = 'MODEL_SERVICE_API_BASE_URL'
DEFAULT_MODEL = 'gpt-5.5'
DEFAULT_MODEL_SERVICE_PROVIDER = 'azure_openai'
DEFAULT_MODEL_SERVICE_TIMEOUT = '60'
# Integration flows consume a prebuilt image, not the removed image builder.
DEFAULT_BASE_IMAGE = 'nexus-ubuntu-session:latest'
DEFAULT_RUNTIME_NAME = 'main'
DEFAULT_RUNTIME_EXTRA_ARGS = [
    '--tmpfs',
    '/run',
    '--tmpfs',
    '/run/lock',
    '-e',
    'GTK_A11Y=none',
    '-e',
    'NEXUS_GUI_REQUIRE_SYSTEM_DBUS=1',
]
DEFAULT_SYSTEM_PROMPT = (
    'You are a precise assistant. Follow the requested desktop GUI steps exactly and stop when the task is done.'
)


@dataclass(frozen=True)
class OpenAISettings:
    """Resolved OpenAI client settings for local integration runs."""

    api_key: str
    base_url: str
    model: str


def env_flag(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name, '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, '').strip()
    if not raw:
        return default
    return int(raw)


def resolve_openai_settings(env: Mapping[str, str]) -> OpenAISettings:
    """Resolve OpenAI settings from environment-like input."""
    model = env.get('NEXUS_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL
    api_key = env.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        app_id = env.get('MODEL_SERVICE_API_APP_ID', '').strip()
        app_key = env.get('MODEL_SERVICE_API_APP_KEY', '').strip()
        if not app_id or not app_key:
            raise RuntimeError('OPENAI_API_KEY or MODEL_SERVICE_API_APP_ID/MODEL_SERVICE_API_APP_KEY is required')
        provider = env.get('MODEL_SERVICE_API_PROVIDER', DEFAULT_MODEL_SERVICE_PROVIDER).strip()
        timeout = env.get('MODEL_SERVICE_API_TIMEOUT', DEFAULT_MODEL_SERVICE_TIMEOUT).strip()
        api_key = f'{app_id}:{app_key}?provider={provider}&model={model}&timeout={timeout}'

    base_url = env.get('OPENAI_BASE_URL', '').strip() or env.get(MODEL_SERVICE_API_BASE_URL_ENV, '').strip()
    if not base_url:
        raise RuntimeError(f'OPENAI_BASE_URL or {MODEL_SERVICE_API_BASE_URL_ENV} is required')
    return OpenAISettings(api_key=api_key, base_url=base_url, model=model)


def build_identity_gui_run_config(
    *,
    api_key: str,
    user_prompt: str,
    tools: Sequence[str],
    base_url: str,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 30,
) -> dict[str, Any]:
    """Build the common `RunInput` payload for identity-task GUI demos."""
    return {
        'imports': ['nexus.extensions'],
        'task_name': 'identity',
        'task_config': {},
        'runtime_provider_name': 'docker',
        'runtime_provider_config': {
            'runtime_image': 'current',
            'force_rebuild': False,
            'print_port_mappings': True,
        },
        'runtime_configs': {
            DEFAULT_RUNTIME_NAME: {
                'image': DEFAULT_BASE_IMAGE,
                'exposed_ports': [5900, 6080],
                'extra_args': DEFAULT_RUNTIME_EXTRA_ARGS,
            }
        },
        'agent_name': 'main_agent',
        'agent_config': {
            'user_prompt': user_prompt,
            'system_prompt': DEFAULT_SYSTEM_PROMPT,
            'max_iterations': max_iterations,
            'tools': list(tools),
            'parallel_tool_calls': False,
        },
        'openai_clients': {
            'main': {
                'name': 'openai',
                'config': {
                    'client_args': {
                        'api_key': api_key,
                        'base_url': base_url,
                    },
                    'request_args': {
                        'model': model,
                    },
                },
            }
        },
        'processors': {
            'chat': {'name': 'chat_tracker'},
            'agent_metrics': {'name': 'main_agent_metrics'},
        },
    }


__all__ = [
    'DEFAULT_BASE_IMAGE',
    'DEFAULT_MODEL',
    'MODEL_SERVICE_API_BASE_URL_ENV',
    'DEFAULT_RUNTIME_EXTRA_ARGS',
    'DEFAULT_RUNTIME_NAME',
    'DEFAULT_SYSTEM_PROMPT',
    'OpenAISettings',
    'build_identity_gui_run_config',
    'env_flag',
    'env_int',
    'resolve_openai_settings',
]
