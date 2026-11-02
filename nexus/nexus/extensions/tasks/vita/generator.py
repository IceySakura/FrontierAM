"""VitaBench batch generator for `nexus run-batch`.

Reads VitaBench task JSON files and yields BatchItems for running vita_task at scale.

Usage:
  nexus run-batch -g nexus/extensions/tasks/vita/generator.py -d ./outputs/vita_batch.db

Environment variables:
  - VITA_DOMAIN: domain name (default: ota). Supports: ota, delivery, instore,
                 or comma-separated for cross-domain (e.g., "delivery,instore,ota")
  - VITA_LANGUAGE: chinese or english (default: chinese)
  - VITA_DATA_DIR: path to vitabench data directory (default: auto-detected)
  - VITA_TASK_IDS: comma-separated list of specific task IDs to run (default: all)
  - VITA_SKIP: number of tasks to skip (default: 0)
  - VITA_LIMIT: max tasks to yield (default: no limit)
  - VITA_MAX_STEPS: max orchestration steps (default: 300)
  - VITA_ENABLE_THINK: enable thinking mode for agent (default: false)
  - VITA_SEED: random seed (default: 300)
  - OPENAI_CLIENT_NAME / OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: agent LLM config
  - VITA_USER_API_KEY / VITA_USER_BASE_URL / VITA_USER_MODEL: user simulator LLM config (optional)
  - VITA_EVAL_API_KEY / VITA_EVAL_BASE_URL / VITA_EVAL_MODEL: evaluator LLM config (optional)
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


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


def _env_list(name: str) -> set[str] | None:
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _get_data_dir() -> Path:
    env_path = os.getenv('VITA_DATA_DIR')
    if env_path:
        return Path(env_path)

    from nexus.extensions.tasks.vita.vita_utils.utils import _find_data_dir

    return _find_data_dir()


def _get_task_file_path(domain: str, language: str) -> Path:
    data_dir = _get_data_dir()
    task_dir = domain if domain != 'cross_domain' else 'cross_domain'
    filename = 'tasks_en.json' if language == 'english' else 'tasks.json'
    return data_dir / 'vita' / 'domains' / task_dir / filename


def _load_tasks(domain: str, language: str) -> list[dict]:
    """Load tasks from VitaBench JSON file."""
    if ',' in domain:
        task_path = _get_task_file_path('cross_domain', language)
    else:
        task_path = _get_task_file_path(domain, language)

    if not task_path.exists():
        raise FileNotFoundError(f'VitaBench task file not found: {task_path}')

    with task_path.open('r', encoding='utf-8') as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError(f'Expected a JSON array in {task_path}')

    return tasks


def _build_openai_client(
    name_env: str = 'OPENAI_CLIENT_NAME',
    key_env: str = 'OPENAI_API_KEY',
    url_env: str = 'OPENAI_BASE_URL',
    model_env: str = 'OPENAI_MODEL',
    defaults: dict[str, str] | None = None,
) -> dict[str, Any]:
    defaults = defaults or {}
    return {
        'name': os.getenv(name_env, defaults.get('name', 'openai')),
        'config': {
            'client_args': {
                'api_key': os.getenv(key_env, defaults.get('api_key', 'sk-placeholder')),
                'base_url': os.getenv(url_env, defaults.get('base_url', 'https://api.openai.com/v1')),
            },
            'request_args': {
                'model': os.getenv(model_env, defaults.get('model', 'gpt-4.1')),
            },
        },
    }


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    clients: dict[str, dict[str, Any]] = {
        'main': _build_openai_client(),
    }

    if os.getenv('VITA_USER_API_KEY') or os.getenv('VITA_USER_BASE_URL'):
        clients['user'] = _build_openai_client(
            key_env='VITA_USER_API_KEY',
            url_env='VITA_USER_BASE_URL',
            model_env='VITA_USER_MODEL',
            defaults={'api_key': os.getenv('OPENAI_API_KEY', ''), 'base_url': os.getenv('OPENAI_BASE_URL', '')},
        )

    if os.getenv('VITA_EVAL_API_KEY') or os.getenv('VITA_EVAL_BASE_URL'):
        clients['evaluator'] = _build_openai_client(
            key_env='VITA_EVAL_API_KEY',
            url_env='VITA_EVAL_BASE_URL',
            model_env='VITA_EVAL_MODEL',
            defaults={'api_key': os.getenv('OPENAI_API_KEY', ''), 'base_url': os.getenv('OPENAI_BASE_URL', '')},
        )

    return clients


def generate_configs() -> Iterator[BatchItem]:
    domain = os.getenv('VITA_DOMAIN', 'ota')
    language = os.getenv('VITA_LANGUAGE', 'chinese')
    skip = _env_int('VITA_SKIP') or 0
    limit = _env_int('VITA_LIMIT')
    task_ids_filter = _env_list('VITA_TASK_IDS')
    max_steps = _env_int('VITA_MAX_STEPS') or 300
    enable_think = _env_bool('VITA_ENABLE_THINK', default=False)
    seed = _env_int('VITA_SEED') or 300

    tasks = _load_tasks(domain, language)
    openai_clients = _build_openai_clients()

    skipped = 0
    produced = 0

    for task_data in tasks:
        task_id = task_data.get('id', '')
        if not task_id:
            continue

        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue

        if skipped < skip:
            skipped += 1
            continue

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.vita',
            ],
            'task_name': 'vita_task',
            'task_config': {
                'task_data': task_data,
                'language': language,
                'evaluation_type': 'trajectory',
            },
            'agent_name': 'vita_agent',
            'agent_config': {
                'max_steps': max_steps,
                'enable_think': enable_think,
                'seed': seed,
            },
            'openai_clients': openai_clients,
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=task_id,
            metadata={
                'task_id': task_id,
                'domain': domain,
                'language': language,
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    domain = os.getenv('VITA_DOMAIN', 'ota')
    language = os.getenv('VITA_LANGUAGE', 'chinese')
    skip = _env_int('VITA_SKIP') or 0
    limit = _env_int('VITA_LIMIT')
    task_ids_filter = _env_list('VITA_TASK_IDS')

    tasks = _load_tasks(domain, language)
    n = 0
    skipped = 0

    for task_data in tasks:
        task_id = task_data.get('id', '')
        if not task_id:
            continue
        if task_ids_filter is not None and task_id not in task_ids_filter:
            continue
        if skipped < skip:
            skipped += 1
            continue
        n += 1
        if limit is not None and n >= limit:
            return limit
    return n


def get_batch_metadata() -> dict[str, Any]:
    domain = os.getenv('VITA_DOMAIN', 'ota')
    language = os.getenv('VITA_LANGUAGE', 'chinese')
    task_ids_filter = _env_list('VITA_TASK_IDS')
    data_dir = _get_data_dir()

    return {
        'task_name': 'vita_task',
        'domain': domain,
        'language': language,
        'data_dir': str(data_dir),
        'task_ids_filter': list(task_ids_filter) if task_ids_filter else None,
    }
