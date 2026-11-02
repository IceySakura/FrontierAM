#!/usr/bin/env python3
"""PinchBench batch generator for `nexus run-batch`.

Generates batch configs for all PinchBench tasks, supporting automatic task
discovery, flexible LLM configuration via environment variables, and parallel
execution.

Usage:
  nexus run-batch -g datasets/batch_run/pinchbench_generator.py -d ./outputs/pinchbench_batch.db --max-concurrent 4

Environment variables:
  - PINCHBENCH_TASKS_DIR: Path to tasks directory (default: datasets/pinchbench/tasks)
  - PINCHBENCH_ASSETS_DIR: Path to assets directory (default: datasets/pinchbench/assets)
  - PINCHBENCH_TASK_IDS: Comma-separated task IDs to run (default: all tasks)
  - PINCHBENCH_SKIP: Number of tasks to skip (default: 0)
  - PINCHBENCH_LIMIT: Maximum number of tasks to run (default: no limit)
  - PINCHBENCH_AUTOMATED_ONLY: "1"/"true" to only run automated grading tasks (default: false)
  - PINCHBENCH_IMAGE: Docker image to use (required)
  - PINCHBENCH_FORCE_REBUILD: "true" to force rebuild the image (default: false)
  - OPENCLAW_VERSION: OpenClaw agent version (default: 2026.3.13)
  - OPENCLAW_MODEL: Main agent model (required)
  - OPENCLAW_LITELLM_API_KEY: API key for LLM (required)
  - OPENCLAW_LITELLM_BASE_URL: API base URL (required)
  - OPENCLAW_NPM_REGISTRY: NPM registry URL (optional)
  - OPENCLAW_CONTEXT_WINDOW: Context window size for the model (default: 200000)
  - OPENCLAW_MAX_TOKENS: Maximum output tokens for the model (default: 65536)
  - JUDGE_MODEL: Model for LLM judge tasks (defaults to OPENCLAW_MODEL)
  - JUDGE_API_KEY: API key for judge model (defaults to OPENCLAW_LITELLM_API_KEY)
  - JUDGE_BASE_URL: API base URL for judge model (defaults to OPENCLAW_LITELLM_BASE_URL)
  - PINCHBENCH_TIMEOUT: Timeout per task in seconds (default: from task frontmatter or 300)
  - PINCHBENCH_TIMEOUT_MULTIPLIER: Multiplier for timeout (default: 1.0)
  - PINCHBENCH_REASONING_EFFORT: Reasoning effort for the model (e.g. low/medium/high/none, default: not set)
"""

from __future__ import annotations

import logging
import os
import re

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nexus import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput


logger = logging.getLogger(__name__)


# ============================================================================
# Path Constants
# ============================================================================

# Auto-detect paths relative to this file's location
# This file is at: nexus/extensions/tasks/pinchbench/generator.py
GENERATOR_DIR = Path(__file__).parent  # nexus/extensions/tasks/pinchbench
BATCH_RUN_DIR = GENERATOR_DIR
DATASETS_DIR = BATCH_RUN_DIR.parents[3]  # project root
DEFAULT_TASKS_DIR = DATASETS_DIR / 'datasets' / 'pinchbench' / 'tasks'
DEFAULT_ASSETS_DIR = DATASETS_DIR / 'datasets' / 'pinchbench' / 'assets'


# ============================================================================
# Environment Configuration Helpers
# ============================================================================


def _env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Get integer from environment variable with fallback."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    """Parse comma-separated list from environment variable."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_path(name: str, default: Path) -> Path:
    """Get path from environment variable with fallback."""
    v = os.getenv(name)
    if not v:
        return default
    return Path(v)


# ============================================================================
# Task Discovery and Parsing
# ============================================================================


def _load_task_meta(task_file: Path) -> dict:
    """Load task metadata from YAML frontmatter.

    Args:
        task_file: Path to the task markdown file

    Returns:
        Dictionary containing the parsed frontmatter

    Raises:
        ValueError: If no YAML frontmatter is found
    """
    content = task_file.read_text(encoding='utf-8')
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        raise ValueError(f'No YAML frontmatter in {task_file}')
    return yaml.safe_load(match.group(1))


def _needs_assets(meta: dict) -> bool:
    """Check if task needs assets from assets directory.

    Args:
        meta: Task metadata dictionary

    Returns:
        True if workspace_files contains any item with 'source' key
    """
    return any('source' in f for f in (meta.get('workspace_files') or []))


def _needs_judge(meta: dict) -> bool:
    """Check if task needs LLM judge client.

    Args:
        meta: Task metadata dictionary

    Returns:
        True if grading_type is 'llm_judge' or 'hybrid'
    """
    return meta.get('grading_type') in ('llm_judge', 'hybrid')


def _discover_all_tasks(tasks_dir: Path) -> list[Path]:
    """Discover all task files in the tasks directory.

    Args:
        tasks_dir: Path to the tasks directory

    Returns:
        Sorted list of task file paths
    """
    task_files = sorted(tasks_dir.glob('task_*.md'))
    logger.debug(f'Discovered {len(task_files)} task files in {tasks_dir}')
    return task_files


def _filter_tasks_by_ids(task_files: list[Path], task_ids: list[str]) -> list[Path]:
    """Filter task files by specified task IDs.

    Args:
        task_files: List of all task file paths
        task_ids: List of task IDs to filter by (can be stem or frontmatter id)

    Returns:
        Filtered list of task file paths

    Raises:
        ValueError: If any specified task ID is not found
    """
    # Build mapping of ID -> Path
    task_map = {}
    for f in task_files:
        # Map by file stem
        task_map[f.stem] = f
        # Also map by frontmatter ID if available
        try:
            meta = _load_task_meta(f)
            fid = meta.get('id')
            if fid:
                task_map[fid] = f
        except Exception as e:
            logger.debug('Failed to load metadata for task %s: %s', f.stem, e)
            pass

    # Filter by requested IDs
    filtered = []
    for tid in task_ids:
        if tid not in task_map:
            raise ValueError(f"Task ID '{tid}' not found in tasks directory")
        if task_map[tid] not in filtered:
            filtered.append(task_map[tid])

    return filtered


def _filter_automated_only(task_files: list[Path]) -> list[Path]:
    """Filter to only keep automated grading tasks.

    Args:
        task_files: List of task file paths

    Returns:
        Filtered list containing only automated grading tasks
    """
    filtered = []
    for f in task_files:
        try:
            meta = _load_task_meta(f)
            if meta.get('grading_type') == 'automated':
                filtered.append(f)
        except Exception as e:
            logger.warning(f'Failed to load metadata for {f}: {e}')
    logger.debug(f'Filtered to {len(filtered)} automated tasks')
    return filtered


def _resolve_task_files() -> list[Path]:
    """Resolve task files based on environment configuration.

    Returns:
        List of task file paths to process

    Raises:
        ValueError: If tasks directory doesn't exist or requested task not found
    """
    tasks_dir = _env_path('PINCHBENCH_TASKS_DIR', DEFAULT_TASKS_DIR)

    if not tasks_dir.exists():
        raise ValueError(f'Tasks directory not found: {tasks_dir}')

    # Discover all tasks
    task_files = _discover_all_tasks(tasks_dir)

    # Filter by explicit task IDs if provided
    explicit_ids = _env_list('PINCHBENCH_TASK_IDS')
    if explicit_ids:
        logger.debug(f'Using explicit task IDs: {explicit_ids}')
        task_files = _filter_tasks_by_ids(task_files, explicit_ids)

    # Filter to automated only if requested
    if _env_bool('PINCHBENCH_AUTOMATED_ONLY'):
        task_files = _filter_automated_only(task_files)

    # Apply skip/limit
    skip = _env_int('PINCHBENCH_SKIP', 0) or 0
    limit = _env_int('PINCHBENCH_LIMIT')

    task_files = task_files[skip:]
    if limit is not None:
        task_files = task_files[:limit]

    logger.debug(f'Resolved {len(task_files)} tasks to run (skip={skip}, limit={limit})')
    return task_files


# ============================================================================
# Configuration Builders
# ============================================================================


def _build_openai_clients(meta: dict) -> dict[str, dict[str, Any]]:
    """Build OpenAI client configurations from environment variables.

    Args:
        meta: Task metadata dictionary (used to determine if judge is needed)

    Returns:
        Dict with 'main' and optionally 'judge' client configurations
    """
    api_key = os.getenv('OPENCLAW_LITELLM_API_KEY', '')
    base_url = os.getenv('OPENCLAW_LITELLM_BASE_URL', '')
    model = os.getenv('OPENCLAW_MODEL', '')
    if not api_key or not base_url or not model:
        raise ValueError(
            'OPENCLAW_LITELLM_API_KEY, OPENCLAW_LITELLM_BASE_URL, and OPENCLAW_MODEL environment variables are required'
        )

    # Judge model configuration - can be customized independently
    judge_model = os.getenv('JUDGE_MODEL', model)
    judge_api_key = os.getenv('JUDGE_API_KEY', api_key)
    judge_base_url = os.getenv('JUDGE_BASE_URL', base_url)

    # Build main client request_args
    main_request_args: dict[str, Any] = {
        'model': model,
    }

    # Inject reasoning_effort if specified
    reasoning_effort = os.getenv('PINCHBENCH_REASONING_EFFORT', '')
    if reasoning_effort and reasoning_effort.lower() not in ('', 'none'):
        main_request_args['extra_body'] = {'reasoning_effort': reasoning_effort}

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': main_request_args,
            },
        },
    }

    # Add judge client if needed
    if _needs_judge(meta):
        clients['judge'] = {
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
        }

    return clients


def _build_task_config(task_file: Path, assets_dir: Path) -> dict[str, Any]:
    """Build task configuration for a specific task file.

    Args:
        task_file: Path to the task markdown file
        assets_dir: Path to the assets directory

    Returns:
        Task configuration dictionary
    """
    meta = _load_task_meta(task_file)

    # Use absolute path to avoid relative path resolution issues
    # The PinchBench task will use Path.cwd() to resolve relative paths,
    # which may not match our expected base directory
    task_config: dict[str, Any] = {
        'task_file': str(task_file.resolve()),
        'workspace_path': '/workspace',
    }

    # Add assets path if needed
    if _needs_assets(meta):
        task_config['assets_path'] = str(assets_dir.resolve())

    return task_config


def _build_run_input(task_file: Path, assets_dir: Path) -> tuple[RunInput, dict[str, Any]]:
    """Build RunInput configuration for a task.

    Args:
        task_file: Path to the task markdown file
        assets_dir: Path to the assets directory

    Returns:
        Tuple of (RunInput instance, metadata dict)
    """
    meta = _load_task_meta(task_file)
    task_id = meta.get('id', task_file.stem)
    timeout = _env_int('PINCHBENCH_TIMEOUT') or meta.get('timeout_seconds', 300)
    timeout = timeout * (_env_int('PINCHBENCH_TIMEOUT_MULTIPLIER', 1) or 1)

    # Build configurations
    task_config = _build_task_config(task_file, assets_dir)
    openai_clients = _build_openai_clients(meta)

    # Docker image configuration
    docker_image = os.getenv('PINCHBENCH_IMAGE', '')
    if not docker_image:
        raise ValueError('PINCHBENCH_IMAGE environment variable is required')

    # Agent configuration
    model = os.getenv('OPENCLAW_MODEL', '')
    if not model:
        raise ValueError('OPENCLAW_MODEL environment variable is required')
    agent_config = {
        'working_dir': '/workspace',
        'timeout': timeout,
        'version': os.getenv('OPENCLAW_VERSION', '2026.3.13'),
        'model': model,
        'npm_registry': os.getenv('OPENCLAW_NPM_REGISTRY', ''),
        'context_window': _env_int('OPENCLAW_CONTEXT_WINDOW', 200_000) or 200_000,
        'max_tokens': _env_int('OPENCLAW_MAX_TOKENS', 65_536) or 65_536,
    }

    # Build RunInput
    config = RunInput(
        imports=['nexus.extensions'],
        task_name='pinchbench',
        task_config=task_config,
        runtime_provider_name='docker',
        runtime_provider_config={
            'image': docker_image,
            'runtime_image': 'current',
            'force_rebuild': os.getenv('PINCHBENCH_FORCE_REBUILD', 'false').lower() in ['true', '1', 'yes'],
        },
        runtime_configs={},
        agent_name='openclaw',
        agent_config=agent_config,
        openai_clients=openai_clients,
        processors={
            'chat': ProcessorConfig(name='chat_tracker'),
            # Note: trace processor will be auto-added by batch runner
        },
    )

    # Build metadata
    metadata = {
        'task_id': task_id,
        'task_name': meta.get('name', 'Unknown'),
        'category': meta.get('category', 'unknown'),
        'grading_type': meta.get('grading_type', 'unknown'),
        'timeout_seconds': timeout,
        'needs_assets': _needs_assets(meta),
        'needs_judge': _needs_judge(meta),
        'model': model,
    }

    return config, metadata


# ============================================================================
# Main Generator Functions
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for PinchBench evaluation.

    Reads configuration from environment variables, discovers task files,
    and yields BatchItem instances ready for execution.

    Yields:
        BatchItem: Configured batch items for each task

    Raises:
        ValueError: If task discovery or configuration fails
    """
    # Resolve task files and assets directory
    task_files = _resolve_task_files()
    assets_dir = _env_path('PINCHBENCH_ASSETS_DIR', DEFAULT_ASSETS_DIR)

    logger.debug(f'Generating configs for {len(task_files)} PinchBench tasks')
    logger.debug(f'Assets directory: {assets_dir}')

    for task_file in task_files:
        try:
            config, metadata = _build_run_input(task_file, assets_dir)
            task_id = metadata['task_id']

            yield BatchItem(
                config=config,
                run_id=task_id,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f'Failed to generate config for {task_file}: {e}')
            raise

    logger.debug(f'Generated {len(task_files)} batch items for PinchBench')


def get_total_count() -> int:
    """Return total number of tasks that will be generated.

    Takes into account environment configuration (skip, limit, filters).

    Returns:
        Total count of tasks to be generated
    """
    try:
        task_files = _resolve_task_files()
        return len(task_files)
    except Exception as e:
        logger.error(f'Failed to count tasks: {e}')
        return 0


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata for tracking and debugging.

    Returns:
        Dictionary containing batch configuration and environment info
    """
    tasks_dir = _env_path('PINCHBENCH_TASKS_DIR', DEFAULT_TASKS_DIR)
    assets_dir = _env_path('PINCHBENCH_ASSETS_DIR', DEFAULT_ASSETS_DIR)

    return {
        'task_type': 'pinchbench',
        'tasks_dir': str(tasks_dir),
        'assets_dir': str(assets_dir),
        'model': os.getenv('OPENCLAW_MODEL', ''),
        'judge_model': os.getenv('JUDGE_MODEL', os.getenv('OPENCLAW_MODEL', '')),
        'agent_version': os.getenv('OPENCLAW_VERSION', '2026.3.13'),
        'docker_image': os.getenv('PINCHBENCH_IMAGE', ''),
        'automated_only': _env_bool('PINCHBENCH_AUTOMATED_ONLY'),
        'skip': _env_int('PINCHBENCH_SKIP', 0) or 0,
        'limit': _env_int('PINCHBENCH_LIMIT'),
        'explicit_task_ids': _env_list('PINCHBENCH_TASK_IDS'),
    }
