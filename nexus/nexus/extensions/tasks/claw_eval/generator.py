#!/usr/bin/env python3
"""Claw-Eval batch generator for `nexus run-batch`.

Generates batch configs for all claw-eval tasks, supporting automatic task
discovery, flexible LLM configuration via environment variables, and parallel
execution with per-worker port offsets.

Usage:
  nexus run-batch -g nexus/extensions/tasks/claw_eval/generator.py \\
                  -d ./outputs/claw_eval_batch.db --max-concurrent 4

Environment variables:
  - CLAW_EVAL_ROOT: Root path of claw-eval project (required)
  - CLAW_EVAL_TASKS_DIR: Override tasks directory (default: {CLAW_EVAL_ROOT}/tasks)
  - CLAW_EVAL_TASK_IDS: Comma-separated task IDs to run (default: all)
  - CLAW_EVAL_SKIP: Number of tasks to skip (default: 0)
  - CLAW_EVAL_LIMIT: Maximum number of tasks to run (default: no limit)
  - CLAW_EVAL_CATEGORIES: Comma-separated categories to include (default: all)
  - CLAW_EVAL_DIFFICULTIES: Comma-separated difficulties to include (default: all)
  - CLAW_EVAL_AGENT: Agent name to use (default: main)
  - CLAW_EVAL_MODEL: Main LLM model (required)
  - CLAW_EVAL_API_KEY: LLM API key (required)
  - CLAW_EVAL_BASE_URL: LLM API base URL (required)
  - CLAW_EVAL_JUDGE_MODEL: Model for LLM judge (defaults to CLAW_EVAL_MODEL)
  - CLAW_EVAL_JUDGE_API_KEY: API key for judge model (defaults to CLAW_EVAL_API_KEY)
  - CLAW_EVAL_JUDGE_BASE_URL: API base URL for judge model (defaults to CLAW_EVAL_BASE_URL)
  - CLAW_EVAL_TIMEOUT: Override timeout in seconds (default: from task.yaml)
  - CLAW_EVAL_TIMEOUT_MULTIPLIER: Multiply task timeout by this factor (default: 1.0)
                                   Applied after CLAW_EVAL_TIMEOUT override.
  - CLAW_EVAL_REASONING_EFFORT: Reasoning effort level (default: empty, i.e. not set)
  - CLAW_EVAL_INPUT_MODALITIES: Comma-separated list of model input modalities
                                (default: text). Set to "text,image" for vision models.
  - CLAW_EVAL_PORT_BASE_OFFSET: Base port offset (default: 0)
  - CLAW_EVAL_PORT_STEP: Port offset step per worker (default: 100)
"""

from __future__ import annotations

import logging
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nexus import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput, RunnerConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Environment Configuration Helpers
# ============================================================================


def _env(name: str, default: str = '') -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_path(name: str, default: Path | None = None) -> Path | None:
    v = os.getenv(name)
    if not v:
        return default
    return Path(v)


# ============================================================================
# Task Discovery
# ============================================================================


def _discover_tasks(tasks_dir: Path) -> list[Path]:
    """Discover all task directories containing task.yaml."""
    task_dirs = []
    for d in sorted(tasks_dir.iterdir()):
        if d.is_dir() and (d / 'task.yaml').exists():
            task_dirs.append(d)
    logger.debug(f'Discovered {len(task_dirs)} task directories in {tasks_dir}')
    return task_dirs


def _load_task_meta(task_dir: Path) -> dict[str, Any]:
    """Load task metadata from task.yaml."""
    with open(task_dir / 'task.yaml') as f:
        return yaml.safe_load(f)


def _filter_tasks(
    task_dirs: list[Path],
    task_ids: list[str] | None = None,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> list[Path]:
    """Filter task directories by ID, category, and difficulty."""
    filtered = []

    for task_dir in task_dirs:
        try:
            meta = _load_task_meta(task_dir)
        except Exception as e:
            logger.warning(f'Failed to load {task_dir}/task.yaml: {e}')
            continue

        tid = meta.get('task_id', task_dir.name)

        # Filter by task ID
        if task_ids and tid not in task_ids and task_dir.name not in task_ids:
            continue

        # Filter by category
        if categories and meta.get('category', '') not in categories:
            continue

        # Filter by difficulty
        if difficulties and meta.get('difficulty', '') not in difficulties:
            continue

        filtered.append(task_dir)

    return filtered


def _resolve_task_dirs() -> list[Path]:
    """Resolve task directories based on environment configuration."""
    claw_eval_root = _env_path('CLAW_EVAL_ROOT')
    if claw_eval_root is None:
        raise ValueError(
            'CLAW_EVAL_ROOT environment variable is required. Set it to the root path of your claw-eval project.'
        )

    tasks_dir = _env_path('CLAW_EVAL_TASKS_DIR', claw_eval_root / 'tasks')
    if tasks_dir is None or not tasks_dir.exists():
        raise ValueError(f'Tasks directory not found: {tasks_dir}')

    # Discover all tasks
    task_dirs = _discover_tasks(tasks_dir)

    # Apply filters
    task_dirs = _filter_tasks(
        task_dirs,
        task_ids=_env_list('CLAW_EVAL_TASK_IDS'),
        categories=_env_list('CLAW_EVAL_CATEGORIES'),
        difficulties=_env_list('CLAW_EVAL_DIFFICULTIES'),
    )

    # Apply skip/limit
    skip = _env_int('CLAW_EVAL_SKIP', 0) or 0
    limit = _env_int('CLAW_EVAL_LIMIT')

    task_dirs = task_dirs[skip:]
    if limit is not None:
        task_dirs = task_dirs[:limit]

    logger.debug(f'Resolved {len(task_dirs)} tasks to run (skip={skip}, limit={limit})')
    return task_dirs


# ============================================================================
# Configuration Builders
# ============================================================================


def _build_extra_body() -> dict[str, Any]:
    """Build extra_body for request_args based on environment variables.

    Returns an empty dict if CLAW_EVAL_REASONING_EFFORT is not set,
    otherwise returns {'extra_body': {'chat_template_kwargs': {'reasoning_effort': ...}}}.
    """
    reasoning_effort = _env('CLAW_EVAL_REASONING_EFFORT')
    if not reasoning_effort:
        return {}
    return {
        'extra_body': {
            'chat_template_kwargs': {
                'reasoning_effort': reasoning_effort,
            },
        },
    }


def _build_openai_clients(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build OpenAI client configurations."""
    api_key = _env('CLAW_EVAL_API_KEY')
    base_url = _env('CLAW_EVAL_BASE_URL')
    model = _env('CLAW_EVAL_MODEL')
    if not api_key or not base_url or not model:
        raise ValueError(
            'CLAW_EVAL_API_KEY, CLAW_EVAL_BASE_URL, and CLAW_EVAL_MODEL environment variables are required'
        )

    # Judge model configuration - can be customized independently
    judge_model = _env('CLAW_EVAL_JUDGE_MODEL', model)
    judge_api_key = _env('CLAW_EVAL_JUDGE_API_KEY', api_key)
    judge_base_url = _env('CLAW_EVAL_JUDGE_BASE_URL', base_url)

    # Determine client name based on environment variable, default to 'openai'
    client_name = _env('CLAW_EVAL_CLIENT_NAME', 'openai')

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': client_name,
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': {
                    'model': model,
                    'max_tokens': 32768,
                    **(_build_extra_body()),
                },
            },
        },
    }

    # Always add a judge client.  The judge is used not only for
    # communication scoring (which requires judge_rubric) but also for
    # LLM-based classification extraction in graders like T01zh_email_triage.
    # If CLAW_EVAL_JUDGE_MODEL is not set, it defaults to the main model.
    clients['judge'] = {
        'name': 'openai',
        'config': {
            'client_args': {
                'api_key': judge_api_key,
                'base_url': judge_base_url,
            },
            'request_args': {
                'model': judge_model,
                'max_tokens': 8192,
            },
        },
    }

    return clients


def _compute_port_offset(worker_index: int) -> int:
    """Compute port offset for a specific worker.

    Args:
        worker_index: Zero-based worker index

    Returns:
        Port offset value
    """
    base = _env_int('CLAW_EVAL_PORT_BASE_OFFSET', 0) or 0
    step = _env_int('CLAW_EVAL_PORT_STEP', 100) or 100
    return base + worker_index * step


def _build_run_input(
    task_dir: Path,
    claw_eval_root: Path,
    worker_index: int = 0,
) -> tuple[RunInput, dict[str, Any]]:
    """Build RunInput for a single task.

    Args:
        task_dir: Path to task directory
        claw_eval_root: Root of claw-eval project
        worker_index: Worker index for port offset

    Returns:
        Tuple of (RunInput, metadata dict)
    """
    meta = _load_task_meta(task_dir)
    task_id = meta.get('task_id', task_dir.name)

    # Environment settings
    env = meta.get('environment', {})
    timeout = _env_int('CLAW_EVAL_TIMEOUT') or env.get('timeout_seconds', 300)
    timeout_multiplier = float(os.environ.get('CLAW_EVAL_TIMEOUT_MULTIPLIER', '1.0'))
    if timeout_multiplier != 1.0:
        timeout = int(timeout * timeout_multiplier)
    max_turns = env.get('max_turns', 20)

    # Port offset
    port_offset = _compute_port_offset(worker_index)

    # Task config
    task_config = {
        'task_dir': str(task_dir.resolve()),
        'claw_eval_root': str(claw_eval_root.resolve()),
        'port_offset': port_offset,
        'timeout': timeout,
        'max_turns': max_turns,
    }

    # OpenAI clients
    openai_clients = _build_openai_clients(meta)

    # Agent configuration
    agent_name = _env('CLAW_EVAL_AGENT', 'main_agent')
    model = _env('CLAW_EVAL_MODEL')
    if not model:
        raise ValueError('CLAW_EVAL_MODEL environment variable is required')
    agent_config: dict[str, Any] = {
        'timeout': timeout,
    }

    # All tasks need a Docker runtime provider (the main_agent registers a
    # bash tool of type RUNTIME, which requires a configured provider).
    # Sandbox tasks additionally configure a dedicated "sandbox" runtime.
    has_sandbox = bool(meta.get('sandbox_files'))
    docker_image = _env('CLAW_EVAL_DOCKER_IMAGE')
    if not docker_image:
        raise ValueError('CLAW_EVAL_DOCKER_IMAGE environment variable is required')
    runtime_provider_name = _env('CLAW_EVAL_RUNTIME_PROVIDER', 'docker')
    runtime_provider_config: dict[str, Any] = {
        'image': docker_image,
        'runtime_image': 'current',
        'request_timeout': timeout + 120,
    }
    runtime_configs: dict[str, dict[str, Any]] = {}

    if has_sandbox:
        sandbox_image = _env('CLAW_EVAL_SANDBOX_IMAGE', 'claw-eval-agent:latest')
        # Configure the "sandbox" runtime that task.py will acquire
        runtime_configs['sandbox'] = {
            'image': sandbox_image,
        }

    # Build RunInput — claw-eval does NOT use nexus's openclaw agent,
    # but needs any compatible agent that supports MCP tools
    config = RunInput(
        imports=['nexus.extensions'],
        task_name='claw_eval',
        task_config=task_config,
        agent_name=agent_name,
        agent_config=agent_config,
        openai_clients=openai_clients,
        runtime_provider_name=runtime_provider_name,
        runtime_provider_config=runtime_provider_config,
        runtime_configs=runtime_configs,
        processors={
            'chat': ProcessorConfig(name='chat_tracker'),
        },
        runner=RunnerConfig(timeout=timeout + 60),  # Add buffer for setup/teardown
    )

    # Metadata
    metadata = {
        'task_id': task_id,
        'task_name': meta.get('task_name', 'Unknown'),
        'category': meta.get('category', 'unknown'),
        'difficulty': meta.get('difficulty', 'unknown'),
        'timeout_seconds': timeout,
        'max_turns': max_turns,
        'num_services': len(meta.get('services', [])),
        'num_tools': len(meta.get('tools', [])),
        'has_sandbox': bool(meta.get('sandbox_files')),
        'model': model,
    }

    return config, metadata


# ============================================================================
# Main Generator Functions (required by nexus run-batch)
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for claw-eval evaluation.

    Yields:
        BatchItem: Configured batch items for each task
    """
    claw_eval_root = _env_path('CLAW_EVAL_ROOT')
    if claw_eval_root is None:
        raise ValueError('CLAW_EVAL_ROOT environment variable is required')

    task_dirs = _resolve_task_dirs()

    logger.debug(f'Generating configs for {len(task_dirs)} claw-eval tasks')

    for i, task_dir in enumerate(task_dirs):
        try:
            config, metadata = _build_run_input(task_dir, claw_eval_root, worker_index=i)
            task_id = metadata['task_id']

            yield BatchItem(
                config=config,
                run_id=task_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f'Failed to generate config for {task_dir}: {e}')
            raise

    logger.debug(f'Generated {len(task_dirs)} batch items for claw-eval')


def get_total_count() -> int:
    """Return total number of tasks that will be generated."""
    try:
        return len(_resolve_task_dirs())
    except Exception as e:
        logger.error(f'Failed to count tasks: {e}')
        return 0


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata."""
    claw_eval_root = _env_path('CLAW_EVAL_ROOT')
    tasks_dir = _env_path('CLAW_EVAL_TASKS_DIR')
    if tasks_dir is None and claw_eval_root is not None:
        tasks_dir = claw_eval_root / 'tasks'

    return {
        'task_type': 'claw_eval',
        'claw_eval_root': str(claw_eval_root) if claw_eval_root else None,
        'tasks_dir': str(tasks_dir) if tasks_dir else None,
        'model': _env('CLAW_EVAL_MODEL'),
        'input_modalities': _env('CLAW_EVAL_INPUT_MODALITIES', 'text'),
        'judge_model': _env('CLAW_EVAL_JUDGE_MODEL', _env('CLAW_EVAL_MODEL')),
        'agent': _env('CLAW_EVAL_AGENT', 'main'),
        'skip': _env_int('CLAW_EVAL_SKIP', 0) or 0,
        'limit': _env_int('CLAW_EVAL_LIMIT'),
        'task_ids': _env_list('CLAW_EVAL_TASK_IDS'),
        'categories': _env_list('CLAW_EVAL_CATEGORIES'),
        'difficulties': _env_list('CLAW_EVAL_DIFFICULTIES'),
    }
