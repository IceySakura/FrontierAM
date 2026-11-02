#!/usr/bin/env python3
"""Custom Web batch generator for `nexus run-batch`.

This generator reads custom web task configurations from delivery_tasks.jsonl and yields `BatchItem`s,
one per task, suitable for running the `custom_web` task at scale.

Usage:
  uv run nexus run-batch -g ./nexus/extensions/tasks/custom_web/generator.py -d ./outputs/custom_web_batch.db --max-concurrent 5

Environment variables:
  - CUSTOM_WEB_DATASET_PATH: path to delivery_tasks.jsonl (default: ./workspace/delivery_tasks.jsonl)
  - CUSTOM_WEB_TASK_IDS: comma-separated list of specific task IDs to run (default: all)
                         Example: "001,002,019"
  - CUSTOM_WEB_SKIP: number of tasks to skip before starting (default: 0)
  - CUSTOM_WEB_LIMIT: max tasks to yield (default: no limit)
  - CUSTOM_WEB_TASK_NAME: task name for image selection (default: delivery)
                          Available: delivery, stock
  - CUSTOM_WEB_AGENT_NAME: agent name (default: browser_gym_agent)
  - CUSTOM_WEB_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: docker)
  - CUSTOM_WEB_RUNTIME_BASE_URL: runtime base URL for gongfeng (default: http://hyrl-sandbox.prod.woa.com:8052)
  - CUSTOM_WEB_RUNTIME_TAG: runtime tag to use (default: latest)
  - CUSTOM_WEB_TIMEOUT: task timeout in seconds (default: 600)
  - CUSTOM_WEB_MAX_ITERATIONS: max agent iterations (default: 30)
  - CUSTOM_WEB_WITH_NA_HINT: "1"/"true" to include N/A hint (default: false)
  - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: used to configure main OpenAI client
  - JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL: used to configure judge OpenAI client (for retrieval verification)
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DATASET_PATH = './workspace/customweb/delivery_tasks.jsonl'

# Task name -> Docker image mapping
# Add more mappings as needed for different web applications
TASK_IMAGE_MAPPING: dict[str, str] = {
    'delivery': 'mirrors.tencent.com/hunyuan_yanguan/webmock.delivery',
    'stock': 'mirrors.tencent.com/hunyuan_yanguan/webmock.stock',
    'shopping': 'mirrors.tencent.com/hunyuan_yanguan/webmock.shopping',
    'ticket': 'mirrors.tencent.com/hunyuan_yanguan/webmock.ticket',
    'emails': 'mirrors.tencent.com/hunyuan_yanguan/webmock.emails',
    'movie': 'mirrors.tencent.com/hunyuan_yanguan/webmock.movie',
    'blog': 'mirrors.tencent.com/hunyuan_yanguan/webmock.blog',
}

# Default image when task name is not in mapping
DEFAULT_IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/webmock.delivery'

# Default service port for Next.js apps
DEFAULT_SERVICE_PORT = 3000


# ============================================================================
# Helper Functions
# ============================================================================


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str) -> int | None:
    """Parse integer from environment variable."""
    v = os.getenv(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_list(name: str) -> set[str] | None:
    """Parse comma-separated list of task IDs from env var, return as set for O(1) lookup."""
    v = os.getenv(name)
    if not v:
        return None
    items = []
    for item in v.split(','):
        item = item.strip()
        if item:
            items.append(item)
    return set(items) if items else None


def _load_tasks(dataset_path: str) -> list[dict[str, Any]]:
    """Load all task configurations from JSONL file.

    Each line in the JSONL file should be a valid JSON object with task data.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    tasks = []
    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON at line {line_num}: {e}') from e

    return tasks


def _get_image_for_task(task_name: str, tag: str = 'latest') -> str:
    """Get Docker image for a given task name with tag."""
    base_image = TASK_IMAGE_MAPPING.get(task_name, DEFAULT_IMAGE)
    return f'{base_image}:{tag}'


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client configurations for main and judge models."""
    # Main model configuration
    # main_api_key = os.getenv('OPENAI_API_KEY','sk-RM3HkiqY0B7oarI02cPvA19YAvpDiBmLeMl4w10N84gdpHCU',)
    # main_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.hunyuan.cloud.tencent.com/v1/')
    # main_model = os.getenv('OPENAI_MODEL', 'hunyuan-2.0-instruct-20251111')
    main_api_key = os.getenv(
        'OPENAI_API_KEY',
        '7auGXNATFSKl7dF',
    )
    main_base_url = os.getenv('OPENAI_BASE_URL', 'http://stream-server-online-openapi.turbotke.production.polaris:8080')
    main_model = os.getenv('OPENAI_MODEL', 'hy2-instruct-251111-browser-1')

    # Judge model configuration (fallback to main if not specified)
    judge_api_key = os.getenv('JUDGE_API_KEY', '$APP_ID:$APP_KEY?provider=azure_openai&model=gpt-4o')
    judge_base_url = os.getenv('JUDGE_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080/openai/v1')
    judge_model = os.getenv('JUDGE_MODEL', 'gpt-4o')

    return {
        'main': {
            'name': 'stream_server',
            'config': {
                'client_args': {'api_key': main_api_key, 'base_url': main_base_url, 'wsid': '10103', 'max_retries': 6},
                'request_args': {
                    'model': main_model,
                },
            },
        },
        'judge': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': judge_api_key,
                    'base_url': judge_base_url,
                },
                'request_args': {
                    'model': judge_model,
                },
            },
        },
    }


def _build_service_config(task_name: str = 'delivery', tag: str = 'latest') -> dict[str, Any]:
    """Build service configuration for custom web task."""
    image = _get_image_for_task(task_name, tag)

    return {
        'image': image,
        'port': DEFAULT_SERVICE_PORT,
        'app_dir': '/app',
        'startup_command': f'PORT={DEFAULT_SERVICE_PORT} node server.js',
        'startup_wait_seconds': 10,
        'health_check_path': '/',
        'skip_health_check': True,
    }


# ============================================================================
# Generator Functions
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem configurations for custom web tasks.

    Yields:
        BatchItem for each task in the dataset.
    """
    dataset_path = os.getenv('CUSTOM_WEB_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('CUSTOM_WEB_SKIP') or 0
    limit = _env_int('CUSTOM_WEB_LIMIT')
    task_ids_filter = _env_list('CUSTOM_WEB_TASK_IDS')

    task_name = os.getenv('CUSTOM_WEB_TASK_NAME', 'delivery')
    agent_name = os.getenv('CUSTOM_WEB_AGENT_NAME', 'browser_gym_agent')
    runtime_provider_name = os.getenv('CUSTOM_WEB_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('CUSTOM_WEB_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    runtime_tag = os.getenv('CUSTOM_WEB_RUNTIME_TAG', 'latest')
    timeout = _env_int('CUSTOM_WEB_TIMEOUT') or 600
    max_iterations = _env_int('CUSTOM_WEB_MAX_ITERATIONS') or 30

    with_na_hint = _env_bool('CUSTOM_WEB_WITH_NA_HINT', default=False)

    openai_clients = _build_openai_clients()
    service_config = _build_service_config(task_name=task_name, tag=runtime_tag)

    # Load all tasks
    all_tasks = _load_tasks(dataset_path)

    skipped = 0
    produced = 0

    for task_data in all_tasks:
        task_id = task_data.get('ID')
        if task_id is None:
            continue

        # Filter by specific task_ids if provided
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        # Skip tasks until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        # Build runtime_provider_config based on provider type
        if runtime_provider_name == 'docker':
            # Docker provider config (matches single.yaml)
            runtime_provider_config: dict[str, Any] = {
                'image': service_config['image'],
                'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}',
                'runtime_port': 8000,
                'exposed_ports': [DEFAULT_SERVICE_PORT],
            }
            runtime_configs: dict[str, Any] = {
                'main': {
                    'extra_args': ['--user', 'root'],
                },
            }
        else:
            # Gongfeng provider config
            runtime_provider_config = {
                'api_base_url': runtime_base_url,
                'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}',
                'request_timeout': timeout,
                'resource_spec': {
                    'cpu': 2,
                    'memory': 4,
                },
            }
            runtime_configs = {}

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.custom_web',
                'nexus.extensions.agents.browser_gym',
            ],
            'task_name': 'custom_web',
            'task_config': {
                'task_data': task_data,
                'service_config': service_config,
                'start_url': '__SERVICE__/',
                'runtime_name': 'main',
                'timeout': timeout,
                'with_na_hint': with_na_hint,
                'judge_client_name': 'judge',
            },
            'agent_name': agent_name,
            'agent_config': {
                'max_iterations': max_iterations,
            },
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': runtime_provider_config,
            'runtime_configs': runtime_configs,
            'openai_clients': openai_clients,
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=f'custom_web_{task_id}',
            metadata={
                'task_id': task_id,
                'task': task_data.get('task', ''),
                'intentions': task_data.get('intentions', []),
                'classes': task_data.get('classes', []),
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    """Get total count of tasks to be processed.

    Returns:
        Number of tasks that will be processed based on filters.
    """
    dataset_path = os.getenv('CUSTOM_WEB_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('CUSTOM_WEB_SKIP') or 0
    limit = _env_int('CUSTOM_WEB_LIMIT')
    task_ids_filter = _env_list('CUSTOM_WEB_TASK_IDS')

    all_tasks = _load_tasks(dataset_path)

    n = 0
    skipped = 0
    for task_data in all_tasks:
        task_id = task_data.get('ID')
        if task_id is None:
            continue

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
    """Get batch-level metadata.

    Returns:
        Dictionary containing batch metadata.
    """
    dataset_path = os.getenv('CUSTOM_WEB_DATASET_PATH', DEFAULT_DATASET_PATH)
    task_ids_filter = _env_list('CUSTOM_WEB_TASK_IDS')
    task_name = os.getenv('CUSTOM_WEB_TASK_NAME', 'delivery')
    runtime_tag = os.getenv('CUSTOM_WEB_RUNTIME_TAG', 'latest')

    return {
        'task_name': 'custom_web',
        'dataset_path': dataset_path,
        'task_ids_filter': list(task_ids_filter) if task_ids_filter else None,
        'image': _get_image_for_task(task_name, runtime_tag),
    }


# ============================================================================
# CLI Entry Point
# ============================================================================


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
        print('# Custom Web Task Configuration (First Task)')
        print('# Generated from:', os.getenv('CUSTOM_WEB_DATASET_PATH', DEFAULT_DATASET_PATH))
        print()
        yaml.dump(config_dict, sys.stdout, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Only output first task
        break
