#!/usr/bin/env python3
"""WAA (Windows Agent Arena) batch generator for `nexus run-batch`.

Reads task definitions from WAA's test_all.json and task config JSONs,
yields BatchItems for the WAA evaluation pipeline.

Usage:
  nexus run-batch -g nexus/extensions/tasks/waa/generator.py -d ./outputs/waa.db --max-concurrent 1

Environment variables:
  Task filtering:
  - WAA_TASK_DIR: path to evaluation_examples_windows/ directory (required)
  - WAA_DOMAIN: domain filter, comma-separated or "all" (default: all)
  - WAA_TASK_IDS: comma-separated list of specific task IDs to run (default: all)
  - WAA_SKIP: number of tasks to skip (default: 0)
  - WAA_LIMIT: max tasks to yield (default: no limit)
  - WAA_DIFF_LVL: difficulty level, "normal" or "hard" (default: hard)

  Agent config:
  - WAA_MAX_STEPS: max agent steps per task (default: 15)
  - WAA_SLEEP_AFTER_EXECUTION: seconds to wait between actions (default: 0.5)
  - WAA_INITIAL_OBSERVATION_DELAY: seconds to wait before the first observation,
    to let setup-launched apps finish rendering (default: 2.0)
  - WAA_TOOL_CALL_RETRIES: retries when LLM fails to produce tool call (default: 2)
  - WAA_TRACE_DIR: directory for saving per-step observability data (optional)

  Runtime (AGS):
  - WAA_SECRET_ID: AGS secret ID (required)
  - WAA_SECRET_KEY: AGS secret key (required)
  - WAA_TOOL_ID: AGS sandbox tool ID (required)
  - WAA_ENABLE_CDP_RELAY: enable the runtime CDP proxy relay (default: false)
  - WAA_REGION: AGS region (default: ap-chongqing)
  - WAA_INSTANCE_TIMEOUT: AGS sandbox lifetime, e.g. "1800s" or "30m"
    (optional; AGS defaults to 300s, which is too short for full WAA tasks)

  LLM:
  - OPENAI_API_KEY: LLM API key (required)
  - OPENAI_BASE_URL: LLM base URL (required)
  - OPENAI_MODEL: model name (required)
  - OPENAI_CLIENT_NAME: client name (required)
  - OPENAI_TEMPERATURE: sampling temperature (optional)
  - OPENAI_TOP_P: nucleus sampling top_p (optional)
  - OPENAI_MAX_TOKENS: max output tokens (optional)
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


def _env(name: str, default: str = '') -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_set(name: str) -> set[str] | None:
    """Parse comma-separated list from env var."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _load_test_all(task_dir: str) -> dict[str, list[str]]:
    """Load test_all.json — maps domain → list of task IDs."""
    path = Path(task_dir) / 'test_all.json'
    if not path.exists():
        raise FileNotFoundError(f'test_all.json not found at {path}')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _load_task_config(task_dir: str, domain: str, task_id: str, diff_lvl: str) -> dict[str, Any] | None:
    """Load a single task config JSON."""
    if diff_lvl == 'hard':
        config_path = Path(task_dir) / 'examples_noctxt' / domain / f'{task_id}.json'
    else:
        config_path = Path(task_dir) / 'examples' / domain / f'{task_id}.json'

    if not config_path.exists():
        return None
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    api_key = _env('OPENAI_API_KEY')
    base_url = _env('OPENAI_BASE_URL')
    model = _env('OPENAI_MODEL')
    client_name = _env('OPENAI_CLIENT_NAME')

    if not api_key or not base_url or not model or not client_name:
        return {}

    request_args: dict[str, Any] = {'model': model}

    temperature = os.getenv('OPENAI_TEMPERATURE')
    if temperature:
        request_args['temperature'] = float(temperature)
    top_p = os.getenv('OPENAI_TOP_P')
    if top_p:
        request_args['top_p'] = float(top_p)
    max_tokens = os.getenv('OPENAI_MAX_TOKENS')
    if max_tokens:
        request_args['max_tokens'] = int(max_tokens)

    return {
        'main': {
            'name': client_name,
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': request_args,
            },
        }
    }


def _build_runtime_provider() -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build runtime provider config.

    Returns:
        (provider_name, provider_config, runtime_configs)
    """
    secret_id = _env('WAA_SECRET_ID')
    secret_key = _env('WAA_SECRET_KEY')
    tool_id = _env('WAA_TOOL_ID')
    region = _env('WAA_REGION', 'ap-chongqing')
    instance_timeout = _env('WAA_INSTANCE_TIMEOUT')

    if not secret_id or not secret_key or not tool_id:
        raise ValueError('WAA_SECRET_ID, WAA_SECRET_KEY, and WAA_TOOL_ID are required')

    provider_config: dict[str, Any] = {
        'secret_id': secret_id,
        'secret_key': secret_key,
        'region': region,
        'interface_mode': 'raw',
        'request_timeout': 180.0,
        'instance_ready_timeout': 300.0,
    }

    runtime_configs: dict[str, Any] = {
        'main': {
            'tool_id': tool_id,
            'runtime_kind': 'waa',
            'launch_mode': 'native',
            'interface_mode': 'raw',
            'provision_mode': 'control_plane',
            'enable_cdp_relay': _env('WAA_ENABLE_CDP_RELAY', 'false').lower() in {'1', 'true', 'yes'},
        }
    }
    if instance_timeout:
        runtime_configs['main']['timeout'] = instance_timeout

    return 'ags', provider_config, runtime_configs


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItems for WAA tasks."""
    task_dir = _env('WAA_TASK_DIR', '')
    domain_filter = _env('WAA_DOMAIN', 'all')
    task_ids_filter = _env_set('WAA_TASK_IDS')
    skip = _env_int('WAA_SKIP', 0) or 0
    limit = _env_int('WAA_LIMIT')
    max_steps = _env_int('WAA_MAX_STEPS', 15) or 15
    sleep_after_execution = _env_float('WAA_SLEEP_AFTER_EXECUTION', 0.5)
    # The first observation happens immediately after task setup and before the
    # agent loop's sleep_after_execution delay. Win11 File Explorer's WinUI/XAML
    # toolbar can expose UIA controls before DWM finishes repainting pixels,
    # producing gray top-chrome placeholders in step_0 screenshots.
    initial_observation_delay = _env_float('WAA_INITIAL_OBSERVATION_DELAY', 2.0)
    tool_call_retries = _env_int('WAA_TOOL_CALL_RETRIES', 2) or 2
    trace_dir = _env('WAA_TRACE_DIR')
    diff_lvl = _env('WAA_DIFF_LVL', 'hard')

    # Build configs
    provider_name, provider_config, runtime_configs = _build_runtime_provider()
    openai_clients = _build_openai_clients()

    # Load task list
    test_all = _load_test_all(task_dir)

    # Filter domains
    if domain_filter and domain_filter != 'all':
        domains = [d.strip() for d in domain_filter.split(',')]
        test_all = {d: test_all[d] for d in domains if d in test_all}

    skipped = 0
    produced = 0

    for domain in test_all:
        for task_id in test_all[domain]:
            # Filter by specific task IDs
            if task_ids_filter is not None and task_id not in task_ids_filter:
                continue

            # Skip
            if skipped < skip:
                skipped += 1
                continue

            # Load task config
            task_config = _load_task_config(task_dir, domain, task_id, diff_lvl)
            if task_config is None:
                continue

            config: dict[str, Any] = {
                'imports': [
                    'nexus.runtime.providers.ags',
                    'nexus.extensions.tasks.waa',
                    'nexus.extensions.agents.navi',
                ],
                'task_name': 'waa',
                'task_config': {
                    'task_config': task_config,
                },
                'runtime_provider_name': provider_name,
                'runtime_provider_config': provider_config,
                'runtime_configs': runtime_configs,
                'agent_name': 'navi_agent',
                'agent_config': {
                    'max_iterations': max_steps,
                    'sleep_after_execution': sleep_after_execution,
                    'initial_observation_delay': initial_observation_delay,
                    'tool_call_retries': tool_call_retries,
                    **({'trace_dir': trace_dir} if trace_dir else {}),
                },
                'openai_clients': openai_clients,
            }

            yield BatchItem(
                config=RunInput.model_validate(config),
                run_id=task_id,
                metadata={
                    'task_id': task_id,
                    'domain': domain,
                    'instruction': task_config.get('instruction', ''),
                },
            )

            produced += 1
            if limit is not None and produced >= limit:
                return


def get_total_count() -> int:
    """Return total number of tasks matching the current filter."""
    task_dir = _env('WAA_TASK_DIR', '')
    domain_filter = _env('WAA_DOMAIN', 'all')
    task_ids_filter = _env_set('WAA_TASK_IDS')
    skip = _env_int('WAA_SKIP', 0) or 0
    limit = _env_int('WAA_LIMIT')
    diff_lvl = _env('WAA_DIFF_LVL', 'hard')

    test_all = _load_test_all(task_dir)

    if domain_filter and domain_filter != 'all':
        domains = [d.strip() for d in domain_filter.split(',')]
        test_all = {d: test_all[d] for d in domains if d in test_all}

    n = 0
    skipped = 0
    for domain in test_all:
        for task_id in test_all[domain]:
            if task_ids_filter is not None and task_id not in task_ids_filter:
                continue
            if skipped < skip:
                skipped += 1
                continue
            config_path = (
                Path(task_dir) / ('examples_noctxt' if diff_lvl == 'hard' else 'examples') / domain / f'{task_id}.json'
            )
            if not config_path.exists():
                continue
            n += 1
            if limit is not None and n >= limit:
                return limit
    return n


def get_batch_metadata() -> dict[str, Any]:
    """Return metadata about this batch run."""
    return {
        'task_name': 'waa',
        'domain': _env('WAA_DOMAIN', 'all'),
        'diff_lvl': _env('WAA_DIFF_LVL', 'hard'),
        'tool_id': _env('WAA_TOOL_ID'),
        'region': _env('WAA_REGION', 'ap-chongqing'),
        'max_steps': _env_int('WAA_MAX_STEPS', 15),
    }
