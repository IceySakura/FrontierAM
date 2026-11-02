#!/usr/bin/env python3
"""WebArena batch generator for `nexus run-batch`.

This generator reads WebArena task configurations from test.raw.json and yields `BatchItem`s,
one per task, suitable for running the `webarena` task at scale.

Usage:
  uv run nexus run-batch -g ./workspace/webarena_generator.py -d ./outputs/webarena_batch.db --max-concurrent 5

Environment variables:
  - WEBARENA_DATASET_PATH: path to test.raw.json (default: ./workspace/webarena/test.raw.json)
  - WEBARENA_TASK_IDS: comma-separated list of specific task_ids to run (default: all)
                       Example: "0,1,2,3"
  - WEBARENA_SKIP: number of tasks to skip before starting (default: 0)
  - WEBARENA_LIMIT: max tasks to yield (default: no limit)
  - WEBARENA_AGENT_NAME: agent name (default: browser_agent)
  - WEBARENA_RUNTIME_BASE_URL: runtime base URL (default: http://localhost:32000)
  - WEBARENA_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: docker)
  - WEBARENA_RUNTIME_TAG: runtime tag to use (default: latest)
  - WEBARENA_RUNTIME_TIMEOUT: runtime timeout in seconds (default: 600)
  - WEBARENA_WITH_NA_HINT: "1"/"true" to include N/A hint (default: false)
  - WEBARENA_WITH_HOMEPAGE_HINT: "1"/"true" to include homepage hint (default: false)
  - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: used to configure OpenAI client

Note: URL placeholders (__GITLAB__, __SHOPPING__, etc.) in test.raw.json are preserved and
substituted at task runtime based on the WebArenaTaskInput.webarena_urls configuration.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


DEFAULT_WEBARENA_DATASET_PATH = './workspace/webarena/test_sample_150.json'


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_list(name: str) -> set[int] | None:
    """Parse comma-separated list of task IDs from env var, return as set for O(1) lookup."""
    v = os.getenv(name)
    if not v:
        return None
    items = []
    for item in v.split(','):
        item = item.strip()
        if item:
            try:
                items.append(int(item))
            except ValueError:
                continue
    return set(items) if items else None


def _load_tasks(dataset_path: str) -> list[dict[str, Any]]:
    """Load all task configurations from test.raw.json without URL substitution.

    URL placeholders (__GITLAB__, __SHOPPING__, etc.) are preserved as-is and will be
    substituted at task runtime.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    main_api_key = os.getenv(
        'OPENAI_API_KEY',
        'app-id:app-key?provider=xxx&model=xxx',
    )
    main_base_url = os.getenv('OPENAI_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080')
    main_model = os.getenv('OPENAI_MODEL', 'xxx')

    judge_api_key = os.getenv('JUDGE_API_KEY', 'app-id:app-key?provider=xxx&model=xxx')
    judge_base_url = os.getenv('JUDGE_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080/openai/v1')
    judge_model = os.getenv('JUDGE_MODEL', 'xxx')
    judge_config = {
        'client_args': {
            'api_key': judge_api_key,
            'base_url': judge_base_url,
        },
        'request_args': {
            'model': judge_model,
        },
    }
    return {
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': main_api_key,
                    'base_url': main_base_url,
                    'timeout': 1800,
                },
                'request_args': {
                    'model': main_model,  # litellm 格式: gemini/model-name
                    'max_tokens': 32768,
                    'extra_body': {
                        'generationConfig': {
                            'thinkingConfig': {'includeThoughts': True, 'thinkingLevel': 'high'},
                            # 'temperature': 1.0
                        }
                    },
                },
            },
        },
        'judge': {
            'name': 'openai',  # Use openai client for OpenAI API
            'config': judge_config,
        },
    }


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = os.getenv('WEBARENA_DATASET_PATH', DEFAULT_WEBARENA_DATASET_PATH)
    skip = _env_int('WEBARENA_SKIP') or 0
    limit = _env_int('WEBARENA_LIMIT')
    task_ids_filter = _env_list('WEBARENA_TASK_IDS')

    agent_name = os.getenv('WEBARENA_AGENT_NAME', 'browser_gym_agent')
    runtime_provider_name = os.getenv('WEBARENA_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('WEBARENA_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    runtime_tag = os.getenv('WEBARENA_RUNTIME_TAG', 'latest')
    runtime_timeout = _env_int('WEBARENA_RUNTIME_TIMEOUT') or 1200
    runtime_cpu = _env_int('WEBARENA_RUNTIME_CPU') or 8
    runtime_memory = _env_int('WEBARENA_RUNTIME_MEMORY') or 16

    with_na_hint = _env_bool('WEBARENA_WITH_NA_HINT', default=False)
    with_homepage_hint = _env_bool('WEBARENA_WITH_HOMEPAGE_HINT', default=False)

    openai_clients = _build_openai_clients()

    # Load all tasks
    all_tasks = _load_tasks(dataset_path)

    skipped = 0
    produced = 0

    for task_config in all_tasks:
        task_id = task_config.get('task_id')
        if task_id is None:
            continue

        # Filter by specific task_ids if provided
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        # Skip tasks until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.webarena',
                'nexus.extensions.agents.browser_gym',
            ],
            'task_name': 'webarena',
            'task_config': {
                'task_id': task_id,
                'task_config': task_config,
                'runtime_name': 'main',
                'timeout': runtime_timeout,
                'with_na_hint': with_na_hint,
                'with_homepage_hint': with_homepage_hint,
            },
            'agent_name': agent_name,
            'agent_config': {
                'max_iterations': int(os.getenv('WEBARENA_MAX_ITERATIONS', '30')),
                'max_context_tokens': _env_int('WEBARENA_MAX_CONTEXT_TOKENS') or 180 * 1024,  # 默认 180K tokens
                'max_actions_history': _env_int('WEBARENA_MAX_ACTIONS_HISTORY') or 30,  # 保留的历史动作数
                'min_keep_iterations': _env_int('WEBARENA_MIN_KEEP_ITERATIONS') or 3,  # 裁剪时最少保留的轮次
                'dom_compression_threshold': _env_int('WEBARENA_DOM_COMPRESSION_THRESHOLD')
                or 15000,  # DOM 压缩触发阈值
                'enable_dom_compression': os.getenv('WEBARENA_ENABLE_DOM_COMPRESSION', 'true').lower() == 'true',
                'enable_delta_observation': os.getenv('WEBARENA_ENABLE_DELTA_OBSERVATION', 'true').lower() == 'true',
            },
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'api_base_url': runtime_base_url,
                'token': '6d12fc43adaa15b0f377d83fceeda61d',
                'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}',
                'request_timeout': runtime_timeout,
                'resource_spec': {
                    'cpu': runtime_cpu,
                    'memory': runtime_memory,
                },
            },
            'runtime_configs': {},
            'openai_clients': openai_clients,
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=f'webarena_{task_id}',
            metadata={
                'task_id': task_id,
                'sites': task_config.get('sites', []),
                'intent_template_id': task_config.get('intent_template_id'),
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    dataset_path = os.getenv('WEBARENA_DATASET_PATH', DEFAULT_WEBARENA_DATASET_PATH)
    skip = _env_int('WEBARENA_SKIP') or 0
    limit = _env_int('WEBARENA_LIMIT')
    task_ids_filter = _env_list('WEBARENA_TASK_IDS')

    all_tasks = _load_tasks(dataset_path)

    n = 0
    skipped = 0
    for task_config in all_tasks:
        task_id = task_config.get('task_id')

        # Filter by specific task_ids if provided
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        # Skip tasks until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        n += 1
        if limit is not None and n >= limit:
            return limit

    return n


def get_batch_metadata() -> dict[str, Any]:
    dataset_path = os.getenv('WEBARENA_DATASET_PATH', DEFAULT_WEBARENA_DATASET_PATH)
    task_ids_filter = _env_list('WEBARENA_TASK_IDS')
    return {
        'task_name': 'webarena',
        'dataset_path': dataset_path,
        'task_ids_filter': list(task_ids_filter) if task_ids_filter else None,
    }


if __name__ == '__main__':
    """When run as a script, output the first task config as YAML."""
    import sys

    try:
        import yaml
    except ImportError:
        print('Error: PyYAML is required. Install with: pip install pyyaml', file=sys.stderr)
        sys.exit(1)

    # Generate first task
    for batch_item in generate_configs():
        # Convert to dict for YAML output
        config_dict = batch_item.config.model_dump(mode='python', exclude_none=True)

        # Output to stdout
        print('# WebArena Task Configuration (First Task)')
        print('# Generated from:', os.getenv('WEBARENA_DATASET_PATH', DEFAULT_WEBARENA_DATASET_PATH))
        print()
        yaml.dump(config_dict, sys.stdout, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Only output first task
        break
