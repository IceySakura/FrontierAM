#!/usr/bin/env python3
"""Tau2-bench batch generator for ``nexus run-batch``.

Reads configuration from environment variables, loads tasks from a .jsonl file,
and yields ``BatchItem`` instances ready for parallel execution.

Usage example::

    uv run nexus run-batch -g nexus/extensions/tasks/tau2/generator.py \
        -d ./outputs/tau2_batch.db --max-concurrent 2

Environment variables
---------------------
General:
    TAU2_TASK_IDS           Comma-separated task IDs to run (default: all)
    TAU2_SKIP               Number of tasks to skip (default: 0)
    TAU2_LIMIT              Maximum number of tasks to yield (default: no limit)
    TAU2_SEED               Random seed for reproducibility
    TAU2_MAX_STEPS          Maximum simulation steps (default: 200)
    TAU2_MAX_ERRORS         Maximum tool errors before termination (default: 10)
    TAU2_SOLO_MODE          "1"/"true" to run without user simulator
    TAU2_EVALUATION_TYPE    all | env | action | communicate (default: all)
    TAU2_ADD_THINKING_TOOL  "1"/"true" to add a thinking tool
    TAU2_AGENT_NAME         Agent name (default: tau2_agent)

Task data:
    TAU2_TASKS_FILE         Path to .jsonl task file (required).
                            One JSON object per line, optionally with 'db_path'.

Agent (main) LLM:
    TAU2_AGENT_API_KEY, TAU2_AGENT_BASE_URL, TAU2_AGENT_MODEL  (all required)
    TAU2_AGENT_MAX_TOKENS, TAU2_AGENT_EXTRA_BODY (JSON), TAU2_AGENT_EXTRA_HEADERS (JSON)
    TAU2_AGENT_REASONING_EFFORT, TAU2_AGENT_TEMPERATURE, TAU2_AGENT_TOP_P
    TAU2_AGENT_REQUEST_ARGS  Extra request args as JSON; overridden by explicit vars above.

User simulator LLM (defaults to agent config if not set):
    TAU2_USER_API_KEY, TAU2_USER_BASE_URL, TAU2_USER_MODEL
    TAU2_USER_MAX_TOKENS, TAU2_USER_EXTRA_BODY (JSON), TAU2_USER_EXTRA_HEADERS (JSON)
    TAU2_USER_REASONING_EFFORT, TAU2_USER_TEMPERATURE, TAU2_USER_TOP_P
    TAU2_USER_REQUEST_ARGS  Extra request args as JSON.
"""

from __future__ import annotations

import json
import logging
import os

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


# --------------- Environment helpers ---------------


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var (accepts 1/true/yes/y/on, case-insensitive)."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Parse an integer env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    """Parse a float env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    """Parse a comma-separated list env var (e.g. ``"0,1,5"``)."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_json(name: str, default: dict | None = None) -> dict | None:
    """Parse a JSON-object env var; returns *default* on missing or invalid value."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        parsed = json.loads(v)
        if isinstance(parsed, dict):
            return parsed
        return default
    except (json.JSONDecodeError, TypeError):
        logger.warning(f'Failed to parse JSON from {name}={v!r}')
        return default


# --------------- Task loading ---------------


def _extract_domain(task_data: dict[str, Any]) -> str:
    """Extract the domain from task_data['user_scenario']['instructions']['domain'].

    Raises ``ValueError`` if the domain field is missing.
    """
    try:
        return task_data['user_scenario']['instructions']['domain']
    except (KeyError, TypeError):
        task_id = task_data.get('id', 'unknown')
        raise ValueError(
            f"Task {task_id} is missing 'user_scenario.instructions.domain'. "
            'Each task must specify its domain in the task data.'
        ) from None


def _load_tasks_from_file(file_path: str) -> list[dict[str, Any]]:
    """Load tasks from a .jsonl file (one JSON object per line).

    Each record may contain an optional ``db_path`` field. Relative paths
    are resolved against the parent directory of *file_path*.
    The ``domain`` is extracted from each task's ``user_scenario.instructions.domain``.

    Returns:
        List of ``{'task_data': dict, 'db_path': str | None, 'domain': str}``.
    """
    file_path = os.path.abspath(file_path)

    # --- Parse records from .jsonl ---
    records: list[dict[str, Any]] = []
    with open(file_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f'Skipping invalid JSON at line {line_num} in {file_path}: {e}')

    # --- Normalize each record into {task_data, db_path, domain} ---
    tasks: list[dict[str, Any]] = []
    for record in records:
        db_path = record.pop('db_path', None)
        domain = _extract_domain(record)
        tasks.append({'task_data': record, 'db_path': db_path, 'domain': domain})

    logger.debug(f'Loaded {len(tasks)} tasks from {file_path}')
    return tasks


def _get_tasks_file() -> str:
    """Return the path from ``TAU2_TASKS_FILE`` env var, or raise if unset."""
    path = os.getenv('TAU2_TASKS_FILE')
    if not path:
        raise RuntimeError(
            'TAU2_TASKS_FILE environment variable is required. '
            'Set it to a .jsonl file path (one task JSON object per line).'
        )
    return path


def _resolve_tasks(tasks_file: str) -> list[dict[str, Any]]:
    """Load tasks from *tasks_file* and optionally filter by ``TAU2_TASK_IDS``.

    Returns:
        Filtered list of ``{'task_data': dict, 'db_path': str | None}``.
    """
    tasks = _load_tasks_from_file(tasks_file)

    # Filter by explicit task IDs if specified
    explicit_ids = _env_list('TAU2_TASK_IDS')
    if explicit_ids:
        id_set = set(explicit_ids)
        tasks = [t for t in tasks if str(t['task_data'].get('id', '')) in id_set]
        logger.debug(f'Filtered to {len(tasks)} tasks by TAU2_TASK_IDS: {explicit_ids}')

    return tasks


# --------------- Configuration builders ---------------


def _build_request_args(
    model: str,
    max_tokens: int | None,
    extra_body: dict | None,
    extra_headers: dict | None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    passthrough_args: dict | None = None,
) -> dict[str, Any]:
    """Build ``request_args`` dict for an OpenAI client.

    *passthrough_args* (from ``TAU2_*_REQUEST_ARGS``) serves as the base layer;
    explicit keyword params always override keys in passthrough.
    """
    # Start with passthrough args as the base layer
    args: dict[str, Any] = dict(passthrough_args) if passthrough_args else {}

    # Explicit parameters override passthrough (higher priority)
    args['model'] = model
    if max_tokens is not None:
        args['max_tokens'] = max_tokens
    if extra_body is not None:
        args['extra_body'] = extra_body
    if extra_headers is not None:
        args['extra_headers'] = extra_headers
    if reasoning_effort is not None:
        args['reasoning_effort'] = reasoning_effort
    if temperature is not None:
        args['temperature'] = temperature
    if top_p is not None:
        args['top_p'] = top_p
    return args


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build ``main`` (agent) and ``user`` (simulator) OpenAI client configs.

    User simulator defaults to agent config when its env vars are not set.
    The ``user`` client is omitted in solo mode.
    """
    # Agent (main) client
    api_key = os.getenv('TAU2_AGENT_API_KEY', '')
    base_url = os.getenv('TAU2_AGENT_BASE_URL', '')
    model = os.getenv('TAU2_AGENT_MODEL', '')
    max_tokens = _env_int('TAU2_AGENT_MAX_TOKENS')
    extra_body = _env_json('TAU2_AGENT_EXTRA_BODY')
    extra_headers = _env_json('TAU2_AGENT_EXTRA_HEADERS')
    reasoning_effort = os.getenv('TAU2_AGENT_REASONING_EFFORT') or None
    temperature = _env_float('TAU2_AGENT_TEMPERATURE')
    top_p = _env_float('TAU2_AGENT_TOP_P')
    passthrough_args = _env_json('TAU2_AGENT_REQUEST_ARGS')

    # User simulator client (defaults to agent config)
    user_api_key = os.getenv('TAU2_USER_API_KEY', api_key)
    user_base_url = os.getenv('TAU2_USER_BASE_URL', base_url)
    user_model = os.getenv('TAU2_USER_MODEL', model)
    user_max_tokens = _env_int('TAU2_USER_MAX_TOKENS') or max_tokens
    user_extra_body = _env_json('TAU2_USER_EXTRA_BODY') or extra_body
    user_extra_headers = _env_json('TAU2_USER_EXTRA_HEADERS') or extra_headers
    user_reasoning_effort = os.getenv('TAU2_USER_REASONING_EFFORT') or reasoning_effort
    user_temperature = _env_float('TAU2_USER_TEMPERATURE')
    user_top_p = _env_float('TAU2_USER_TOP_P')
    user_passthrough_args = _env_json('TAU2_USER_REQUEST_ARGS')

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': _build_request_args(
                    model, max_tokens, extra_body, extra_headers, reasoning_effort, temperature, top_p, passthrough_args
                ),
            },
        },
    }

    # Only add user client if not in solo mode
    if not _env_bool('TAU2_SOLO_MODE'):
        clients['user'] = {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': user_api_key,
                    'base_url': user_base_url,
                },
                'request_args': _build_request_args(
                    user_model,
                    user_max_tokens,
                    user_extra_body,
                    user_extra_headers,
                    user_reasoning_effort,
                    user_temperature,
                    user_top_p,
                    user_passthrough_args,
                ),
            },
        }

    return clients


def _build_task_config(
    task_data: dict | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Build a ``Tau2TaskInput`` config dict from env vars and per-task data."""
    seed = _env_int('TAU2_SEED')
    config: dict[str, Any] = {
        'solo_mode': _env_bool('TAU2_SOLO_MODE'),
        'max_steps': _env_int('TAU2_MAX_STEPS', 200) or 200,
        'max_errors': _env_int('TAU2_MAX_ERRORS', 10) or 10,
        'evaluation_type': os.getenv('TAU2_EVALUATION_TYPE', 'all'),
        'add_thinking_tool': _env_bool('TAU2_ADD_THINKING_TOOL'),
    }

    if seed is not None:
        config['seed'] = seed
    if db_path is not None:
        config['db_path'] = db_path
    if task_data is not None:
        config['task_data'] = task_data

    return config


# --------------- Main generator functions ---------------


def generate_configs() -> Iterator[BatchItem]:
    """Yield ``BatchItem`` instances for each tau2 task.

    Loads tasks from the .jsonl file, applies skip/limit, and produces
    one ``BatchItem`` per task with all LLM and task configuration.
    """
    skip = _env_int('TAU2_SKIP', 0) or 0
    limit = _env_int('TAU2_LIMIT')
    agent_name = os.getenv('TAU2_AGENT_NAME', 'tau2_agent')
    tasks_file = _get_tasks_file()

    tasks = _resolve_tasks(tasks_file)

    # Apply skip/limit
    tasks = tasks[skip:]
    if limit is not None:
        tasks = tasks[:limit]

    # Build shared configurations
    openai_clients = _build_openai_clients()

    logger.debug(f'Generating configs: source={tasks_file}, tasks={len(tasks)}, skip={skip}, limit={limit}')

    for entry in tasks:
        task_data = entry['task_data']
        task_id = str(task_data.get('id', 'unknown'))
        db_path = entry['db_path']
        domain = entry['domain']

        task_config = _build_task_config(task_data=task_data, db_path=db_path)

        config = RunInput(
            imports=[
                'nexus.extensions.tasks.tau2',
            ],
            task_name='tau2',
            task_config=task_config,
            agent_name=agent_name,
            agent_config={},
            openai_clients=openai_clients,
        )

        yield BatchItem(
            config=config,
            run_id=f'{domain}_task{task_id}',
            metadata={
                'domain': domain,
                'task_id': task_id,
                'db_path': db_path,
                'data_source': tasks_file,
                'agent_model': os.getenv('TAU2_AGENT_MODEL', ''),
                'user_model': os.getenv('TAU2_USER_MODEL', os.getenv('TAU2_AGENT_MODEL', '')),
                'agent_name': agent_name,
            },
        )

    logger.debug(f'Generated {len(tasks)} batch items for tau2')


def get_total_count() -> int:
    """Return the total number of tasks that will be generated (after skip/limit)."""
    skip = _env_int('TAU2_SKIP', 0) or 0
    limit = _env_int('TAU2_LIMIT')

    try:
        tasks = _resolve_tasks(_get_tasks_file())
        total = max(0, len(tasks) - skip)
    except Exception:
        return 0

    return min(total, limit) if limit else total


def get_batch_metadata() -> dict[str, Any]:
    """Return batch-level metadata for tracking and debugging."""
    agent_model = os.getenv('TAU2_AGENT_MODEL', '')
    user_model = os.getenv('TAU2_USER_MODEL', agent_model)

    return {
        'task_type': 'tau2',
        'tasks_file': os.getenv('TAU2_TASKS_FILE', ''),
        'agent_model': agent_model,
        'user_model': user_model,
        'agent_name': os.getenv('TAU2_AGENT_NAME', 'tau2_agent'),
        'solo_mode': _env_bool('TAU2_SOLO_MODE'),
        'evaluation_type': os.getenv('TAU2_EVALUATION_TYPE', 'all'),
        'max_steps': _env_int('TAU2_MAX_STEPS', 200) or 200,
        'max_errors': _env_int('TAU2_MAX_ERRORS', 10) or 10,
        'seed': _env_int('TAU2_SEED'),
        'skip': _env_int('TAU2_SKIP', 0) or 0,
        'limit': _env_int('TAU2_LIMIT'),
    }
