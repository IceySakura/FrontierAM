"""Batch generator for quick HyCode smoke tests.

Usage:
  export OPENAI_API_KEY=...
  export OPENAI_BASE_URL=http://trpc-gpt-eval.production.polaris:8080
  export OPENAI_MODEL=bedrock/anthropic.claude-opus-4-7
  export OPENAI_CLIENT_NAME=litellm
  export LANGFUSE_SECRET_KEY=...
  export LANGFUSE_PUBLIC_KEY=...
  export LANGFUSE_BASE_URL=http://21.91.27.43:3000

  HYCODE_RUNTIME_PROVIDER=gongfeng \
  HYCODE_PROMPT='Create a small counter app in index.html.' \
  uv run nexus run-batch \
    -g ./nexus/extensions/agents/hycode/generator.py \
    -d ./outputs/hycode_smoke.db \
    --max-concurrent 1
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.agents.hycode.installer import DEFAULT_HYCODE_VERSION
from nexus.runner.single import RunInput


DEFAULT_LANGFUSE_BASE_URL = 'http://21.91.27.43:3000'
DEFAULT_NPM_REGISTRY = 'https://mirrors.tencent.com/npm/'
DEFAULT_MODEL = 'bedrock/anthropic.claude-opus-4-7'
DEFAULT_BASE_URL = 'http://trpc-gpt-eval.production.polaris:8080'
DEFAULT_PROMPT = (
    'Create a compact polished counter app in /testbed/index.html. '
    'Include increment, decrement, reset, and a small activity log. '
    'Keep it self-contained with HTML, CSS, and JavaScript.'
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_json(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if not value:
        return default
    return json.loads(value)


def _required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ' or '.join(names)
    raise ValueError(f'Missing required environment variable: {joined}')


def _openai_clients() -> dict[str, dict[str, Any]]:
    request_args: dict[str, Any] = {
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'max_tokens': _env_int('OPENAI_MAX_TOKENS', 16000) or 16000,
    }
    request_args.update(_env_json('OPENAI_REQUEST_ARGS_JSON', {}))

    return {
        'main': {
            'name': os.getenv('OPENAI_CLIENT_NAME', 'litellm'),
            'config': {
                'client_args': {
                    'api_key': _required_env('OPENAI_API_KEY'),
                    'base_url': os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL),
                    'max_retries': _env_int('OPENAI_MAX_RETRIES', 5) or 5,
                    'cache_control_enabled': _env_bool('OPENAI_CACHE_CONTROL_ENABLED', True),
                    'cache_control_strategy': os.getenv('OPENAI_CACHE_CONTROL_STRATEGY', 'claude_code'),
                    'log_llm_usage': _env_bool('OPENAI_LOG_LLM_USAGE', True),
                },
                'request_args': request_args,
            },
        },
    }


def _runtime_configs() -> dict[str, dict[str, Any]]:
    default = {
        'main': {
            'image': os.getenv('HYCODE_RUNTIME_IMAGE', 'node:22-bookworm'),
            'worker_venv': {'modules': []},
        },
    }
    return _env_json('HYCODE_RUNTIME_CONFIGS_JSON', default)


def _gongfeng_runtime_provider_config() -> dict[str, Any]:
    token = _required_env('GONGFENG_RUNTIME_TOKEN', 'SANDBOX_TOKEN', 'HYCODE_RUNTIME_TOKEN')
    config: dict[str, Any] = {
        'api_base_url': os.getenv('GONGFENG_API_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
        'token': token,
        'runtime_image': os.getenv(
            'GONGFENG_RUNTIME_IMAGE',
            f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{os.getenv("GONGFENG_RUNTIME_IMAGE_TAG", "feat-cc-support-proxy-convert")}',
        ),
        'request_timeout': _env_int('GONGFENG_REQUEST_TIMEOUT', 7200) or 7200,
        'runtime_provider': os.getenv('GONGFENG_RUNTIME_PROVIDER', 'gongfeng'),
        'resource_spec': {
            'cpu': _env_int('GONGFENG_CPU', 4) or 4,
            'memory': _env_int('GONGFENG_MEMORY', 16) or 16,
        },
    }
    # Which sandbox quota bucket to bill is a deployment choice, not a code default.
    purpose = os.getenv('GONGFENG_PURPOSE')
    if purpose:
        config['purpose'] = purpose
    return config


def _runtime_provider_config() -> dict[str, Any]:
    if os.getenv('HYCODE_RUNTIME_PROVIDER_CONFIG_JSON'):
        return _env_json('HYCODE_RUNTIME_PROVIDER_CONFIG_JSON', {})
    if os.getenv('HYCODE_RUNTIME_PROVIDER', 'gongfeng') == 'gongfeng':
        return _gongfeng_runtime_provider_config()
    return {}


def _extra_cli_args() -> list[str]:
    args = [
        '--print',
        '--input-format',
        'text',
        '--output-format',
        'stream-json',
        '--verbose',
        '--no-session-persistence',
    ]
    if not _env_bool('HYCODE_ALLOW_NATIVE_WEB', False):
        args.extend(['--disallowedTools', 'WebSearch', 'WebFetch'])
    extra = _env_json('HYCODE_EXTRA_CLI_ARGS_JSON', [])
    if extra:
        args.extend(extra)
    return args


def _pre_start_scripts() -> list[dict[str, Any]]:
    return _env_json('HYCODE_PRE_START_SCRIPTS_JSON', [])


def _agent_config() -> dict[str, Any]:
    working_dir = os.getenv('HYCODE_WORKING_DIR', '/testbed')
    extra_env = {
        'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
    }
    extra_env.update(_env_json('HYCODE_EXTRA_ENV_JSON', {}))
    return {
        'working_dir': working_dir,
        'timeout': _env_int('HYCODE_AGENT_TIMEOUT', 3600) or 3600,
        'version': os.getenv('HYCODE_PACKAGE_VERSION', DEFAULT_HYCODE_VERSION),
        'npm_registry': os.getenv('HYCODE_NPM_REGISTRY', DEFAULT_NPM_REGISTRY),
        'user_prompt': os.getenv('HYCODE_PROMPT', DEFAULT_PROMPT),
        'extra_env': extra_env,
        'extra_cli_args': _extra_cli_args(),
        'pre_start_scripts': _pre_start_scripts(),
        'conversion_log_level': os.getenv('HYCODE_CONVERSION_LOG_LEVEL', 'INFO'),
        'thinking_preset': os.getenv('HYCODE_THINKING_PRESET', 'hunyuan'),
    }


def _processors(run_id: str) -> dict[str, Any]:
    processors: dict[str, Any] = {
        'chat': {
            'name': 'chat_tracker',
            'args': {'tools_mode': os.getenv('HYCODE_TOOLS_MODE', 'branch_last')},
        },
    }
    if _env_bool('HYCODE_DISABLE_LANGFUSE', False):
        return processors

    model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
    experiment = os.getenv('HYCODE_EXPERIMENT', 'hycode_smoke')
    processors['langfuse'] = {
        'name': 'langfuse_tracker',
        'args': {
            'public_key': _required_env('LANGFUSE_PUBLIC_KEY'),
            'secret_key': _required_env('LANGFUSE_SECRET_KEY'),
            'host': os.getenv('LANGFUSE_BASE_URL', DEFAULT_LANGFUSE_BASE_URL),
            'session_id': run_id,
            'trace_name': run_id,
            'tags': [
                'hycode',
                'smoke',
                experiment,
                model,
            ],
            'trace_metadata': {
                'agent_name': 'hycode',
                'experiment': experiment,
                'model': model,
                'package_version': os.getenv('HYCODE_PACKAGE_VERSION', DEFAULT_HYCODE_VERSION),
                'prompt': os.getenv('HYCODE_PROMPT', DEFAULT_PROMPT),
            },
        },
    }
    return processors


def _build_config(repeat_index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    experiment = os.getenv('HYCODE_EXPERIMENT', 'hycode_smoke')
    run_id_prefix = os.getenv('HYCODE_RUN_ID_PREFIX', experiment)
    run_id = f'{run_id_prefix}__hycode__r{repeat_index}'
    config = {
        'imports': [
            'nexus.extensions.tasks.identity',
            'nexus.extensions.agents.hycode',
        ],
        'task_name': 'identity',
        'run_id': run_id,
        'runtime_provider_name': os.getenv('HYCODE_RUNTIME_PROVIDER', 'gongfeng'),
        'runtime_provider_config': _runtime_provider_config(),
        'runtime_configs': _runtime_configs(),
        'agent_name': 'hycode',
        'agent_config': _agent_config(),
        'openai_clients': _openai_clients(),
        'processors': _processors(run_id),
    }
    metadata = {
        'agent_name': 'hycode',
        'repeat_index': repeat_index,
        'experiment': experiment,
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
    }
    return run_id, config, metadata


def _selected_items() -> list[int]:
    repeat = _env_int('HYCODE_REPEAT', 1) or 1
    items = list(range(repeat))
    limit = _env_int('HYCODE_LIMIT')
    if limit is not None:
        items = items[:limit]
    return items


def generate_configs() -> Iterator[BatchItem]:
    for repeat_index in _selected_items():
        run_id, config, metadata = _build_config(repeat_index)
        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=run_id,
            metadata=metadata,
        )


def get_total_count() -> int:
    return len(_selected_items())


def get_batch_metadata() -> dict[str, Any]:
    return {
        'agent_name': 'hycode',
        'experiment': os.getenv('HYCODE_EXPERIMENT', 'hycode_smoke'),
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'runtime_provider': os.getenv('HYCODE_RUNTIME_PROVIDER', 'gongfeng'),
        'langfuse_enabled': not _env_bool('HYCODE_DISABLE_LANGFUSE', False),
        'langfuse_host': os.getenv('LANGFUSE_BASE_URL', DEFAULT_LANGFUSE_BASE_URL),
        'package_version': os.getenv('HYCODE_PACKAGE_VERSION', DEFAULT_HYCODE_VERSION),
        'total_count': get_total_count(),
    }


if __name__ == '__main__':
    print(json.dumps(get_batch_metadata(), ensure_ascii=False, indent=2))
