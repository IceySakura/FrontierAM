#!/usr/bin/env python3
"""Tau3-bench batch generator for ``nexus run-batch``.

Reads configuration from environment variables, loads tasks from a .jsonl file,
and yields ``BatchItem`` instances ready for parallel execution.

Usage example::

    TAU3_DOMAIN=airline TAU3_TASKS_FILE=tasks.jsonl \
    TAU3_AGENT_API_KEY=... TAU3_AGENT_BASE_URL=... TAU3_AGENT_MODEL=... \
    uv run nexus run-batch -g nexus/extensions/tasks/tau3/generator.py \
        -d ./outputs/tau3_batch.db --max-concurrent 2

Environment variables
---------------------
General:
    TAU3_DOMAIN             Domain: airline | retail | telecom | banking_knowledge | all
                            Comma-separated for multiple domains, or "all" for all 4 (required)
    TAU3_TASK_IDS           Comma-separated task IDs to run (default: all)
    TAU3_TASK_SPLIT         Task split: train | test | base (default: all tasks)
    TAU3_NUM_TASKS          Maximum number of tasks to load from registry (default: all)
    TAU3_SKIP               Number of tasks to skip (default: 0)
    TAU3_LIMIT              Maximum number of tasks to yield (default: no limit)
    TAU3_SEED               Random seed for reproducibility
    TAU3_MAX_STEPS          Maximum simulation steps (default: 200)
    TAU3_MAX_ERRORS         Maximum tool errors before termination (default: 10)
    TAU3_SOLO_MODE          "1"/"true" to run without user simulator
    TAU3_EVALUATION_TYPE    all | env | action | communicate (default: all)
    TAU3_ADD_THINKING_TOOL  "1"/"true" to add a thinking tool
    TAU3_AGENT_NAME         Agent name (default: tau3_agent)

Banking knowledge:
    TAU3_RETRIEVAL_CONFIG        Retrieval strategy (bm25, embedding, llm_rerank, agentic, ...)
    TAU3_RETRIEVAL_CONFIG_KWARGS Extra kwargs as JSON for retrieval strategy

Task data (one of the following):
    TAU3_TASKS_FILE         Path to .jsonl task file (one JSON object per line).
                            If not set, tasks are loaded from tau3's built-in registry
                            using TAU3_DOMAIN (+ optional TAU3_TASK_SPLIT, TAU3_TASK_IDS,
                            TAU3_NUM_TASKS).

Agent (main) LLM:
    TAU3_AGENT_CLIENT_NAME  Client type: openai (default) or taiji
    TAU3_AGENT_API_KEY, TAU3_AGENT_BASE_URL, TAU3_AGENT_MODEL  (required for openai)
    TAU3_AGENT_MAX_TOKENS, TAU3_AGENT_EXTRA_BODY (JSON), TAU3_AGENT_EXTRA_HEADERS (JSON)
    TAU3_AGENT_REASONING_EFFORT, TAU3_AGENT_TEMPERATURE, TAU3_AGENT_TOP_P
    TAU3_AGENT_REQUEST_ARGS  Extra request args as JSON; overridden by explicit vars above.

User simulator LLM (defaults to agent config if not set):
    TAU3_USER_CLIENT_NAME   Client type: openai (default, falls back to agent) or taiji
    TAU3_USER_API_KEY, TAU3_USER_BASE_URL, TAU3_USER_MODEL
    TAU3_USER_MAX_TOKENS, TAU3_USER_EXTRA_BODY (JSON), TAU3_USER_EXTRA_HEADERS (JSON)
    TAU3_USER_REASONING_EFFORT, TAU3_USER_TEMPERATURE, TAU3_USER_TOP_P
    TAU3_USER_REQUEST_ARGS  Extra request args as JSON.

Eval LLM (defaults to OpenAI-compatible, falls back to user/main only if needed):
    TAU3_EVAL_CLIENT_NAME   Client type: openai (default) or taiji
    TAU3_EVAL_API_KEY, TAU3_EVAL_BASE_URL, TAU3_EVAL_MODEL
    TAU3_EVAL_MAX_TOKENS, TAU3_EVAL_EXTRA_BODY (JSON), TAU3_EVAL_EXTRA_HEADERS (JSON)
    TAU3_EVAL_REASONING_EFFORT, TAU3_EVAL_TEMPERATURE, TAU3_EVAL_TOP_P
    TAU3_EVAL_REQUEST_ARGS  Extra request args as JSON.

Taiji-specific (when CLIENT_NAME=taiji, all have defaults):
    TAU3_TAIJI_AUTH             Authorization header (default: Bearer 7auGXNATFSKl7dF)
    TAU3_TAIJI_WSID             Workspace ID (default: 11644)
    TAU3_TAIJI_NS               Polaris namespace (default: Production)
    TAU3_TAIJI_SERVICE          Polaris service name (default: stream-server-online-sbs-11644)
    TAU3_TAIJI_URL_PATTERN      URL pattern: openapi or v1 (default: openapi)
    TAU3_TAIJI_TIMEOUT          Request timeout in seconds (default: 120)
    TAU3_AGENT_ENABLE_THINKING  Enable thinking mode for agent (default: false)
    TAU3_AGENT_TAIJI_SERVICE    Override TAU3_TAIJI_SERVICE for agent only
    TAU3_USER_ENABLE_THINKING   Enable thinking mode for user (default: false)
    TAU3_USER_TAIJI_SERVICE     Override TAU3_TAIJI_SERVICE for user only
    TAU3_EVAL_ENABLE_THINKING   Enable thinking mode for eval (default: false)
    TAU3_EVAL_TAIJI_SERVICE     Override TAU3_TAIJI_SERVICE for eval only
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


def _env_float(name: str, default: float | None = None) -> float | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_json(name: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
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


ALL_DOMAINS = ['airline', 'retail', 'telecom', 'banking_knowledge']


def _ensure_tau2_data_dir() -> None:
    """Fail early with a clear message when tau2's benchmark data is unavailable."""
    data_dir = os.getenv('TAU2_DATA_DIR')
    if not data_dir:
        raise RuntimeError(
            'TAU2_DATA_DIR is required when loading TAU3 tasks from the tau2 registry. '
            'The tau2 package installed in .venv contains Python code but not benchmark data. '
            'Set TAU2_DATA_DIR to the tau2-bench data directory that contains tau2/domains/, '
            'or set TAU3_TASKS_FILE to load tasks from a JSONL file.'
        )

    domains_dir = os.path.join(data_dir, 'tau2', 'domains')
    if os.path.isdir(domains_dir):
        return

    nested_domains_dir = os.path.join(data_dir, 'data', 'tau2', 'domains')
    nested_hint = (
        f' It looks like you passed a tau2-bench repository root; use TAU2_DATA_DIR={os.path.join(data_dir, "data")}.'
        if os.path.isdir(nested_domains_dir)
        else ''
    )
    raise RuntimeError(
        f'TAU2_DATA_DIR does not contain tau2 benchmark data: {data_dir!r}. '
        f'Expected directory: {domains_dir}.{nested_hint}'
    )


def _get_domains() -> list[str]:
    raw = os.getenv('TAU3_DOMAIN', '')
    if not raw:
        raise RuntimeError(
            'TAU3_DOMAIN environment variable is required. '
            'Set to: airline, retail, telecom, banking_knowledge, all, or comma-separated.'
        )
    if raw.strip().lower() == 'all':
        return list(ALL_DOMAINS)
    domains = [d.strip() for d in raw.split(',') if d.strip()]
    if not domains:
        raise RuntimeError(f'TAU3_DOMAIN is empty after parsing: {raw!r}')
    return domains


def _load_tasks_from_file(file_path: str) -> list[dict[str, Any]]:
    file_path = os.path.abspath(file_path)
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

    logger.debug(f'Loaded {len(records)} tasks from {file_path}')
    return records


def _resolve_tasks(domain: str) -> list[dict[str, Any]]:
    """Load tasks from .jsonl file or directly from tau3 registry.

    If ``TAU3_TASKS_FILE`` is set, loads from that file.
    Otherwise, uses tau3's built-in ``get_tasks()`` with the domain name,
    optional ``TAU3_TASK_SPLIT``, ``TAU3_TASK_IDS``, and ``TAU3_NUM_TASKS``.
    """
    tasks_file = os.getenv('TAU3_TASKS_FILE')

    if tasks_file:
        # Load from .jsonl file
        records = _load_tasks_from_file(tasks_file)
        explicit_ids = _env_list('TAU3_TASK_IDS')
        if explicit_ids:
            id_set = set(explicit_ids)
            records = [t for t in records if str(t.get('id', '')) in id_set]
            logger.debug(f'Filtered to {len(records)} tasks by TAU3_TASK_IDS: {explicit_ids}')
        return records

    # Load from tau3 registry
    _ensure_tau2_data_dir()

    from tau2.runner.helpers import get_tasks  # ty: ignore[unresolved-import]

    task_split = os.getenv('TAU3_TASK_SPLIT', 'base')  # default to base split
    task_ids = _env_list('TAU3_TASK_IDS')
    num_tasks = _env_int('TAU3_NUM_TASKS')

    tasks = get_tasks(
        domain,
        task_split_name=task_split,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
    records = [t.model_dump() for t in tasks]
    logger.debug(
        f'Loaded {len(records)} tasks from registry: domain={domain}, '
        f'split={task_split}, ids={task_ids}, num_tasks={num_tasks}'
    )
    return records


# --------------- Configuration builders ---------------


def _build_request_args(
    model: str,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
    extra_headers: dict[str, Any] | None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    passthrough_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = dict(passthrough_args) if passthrough_args else {}
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


def _build_taiji_client_args(
    model: str,
    *,
    service_override_env: str = '',
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Build taiji client_args from TAU3_TAIJI_* env vars."""
    service = (os.getenv(service_override_env) if service_override_env else None) or os.getenv(
        'TAU3_TAIJI_SERVICE', 'stream-server-online-sbs-11644'
    )
    return {
        'model': model,
        'namespace': os.getenv('TAU3_TAIJI_NS', 'Production'),
        'service': service,
        'authorization': os.getenv('TAU3_TAIJI_AUTH', 'Bearer 7auGXNATFSKl7dF'),
        'wsid': os.getenv('TAU3_TAIJI_WSID', '11644'),
        'url_pattern': os.getenv('TAU3_TAIJI_URL_PATTERN', 'openapi'),
        'timeout': float(os.getenv('TAU3_TAIJI_TIMEOUT', '120')),
        'enable_thinking': enable_thinking,
    }


def _clone_client_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': spec['name'],
        'config': {
            'client_args': dict(spec.get('config', {}).get('client_args', {})),
            'request_args': dict(spec.get('config', {}).get('request_args', {})),
        },
    }


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    agent_client_name = os.getenv('TAU3_AGENT_CLIENT_NAME', 'openai')
    model = os.getenv('TAU3_AGENT_MODEL', '')
    max_tokens = _env_int('TAU3_AGENT_MAX_TOKENS')
    extra_body = _env_json('TAU3_AGENT_EXTRA_BODY')
    extra_headers = _env_json('TAU3_AGENT_EXTRA_HEADERS')
    reasoning_effort = os.getenv('TAU3_AGENT_REASONING_EFFORT') or None
    temperature = _env_float('TAU3_AGENT_TEMPERATURE')
    top_p = _env_float('TAU3_AGENT_TOP_P')
    passthrough_args = _env_json('TAU3_AGENT_REQUEST_ARGS')

    if agent_client_name == 'taiji':
        agent_client_args = _build_taiji_client_args(
            model,
            service_override_env='TAU3_AGENT_TAIJI_SERVICE',
            enable_thinking=_env_bool('TAU3_AGENT_ENABLE_THINKING'),
        )
    else:
        api_key = os.getenv('TAU3_AGENT_API_KEY', '')
        base_url = os.getenv('TAU3_AGENT_BASE_URL', '')
        agent_client_args = {'api_key': api_key, 'base_url': base_url}

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': agent_client_name,
            'config': {
                'client_args': agent_client_args,
                'request_args': _build_request_args(
                    model, max_tokens, extra_body, extra_headers, reasoning_effort, temperature, top_p, passthrough_args
                ),
            },
        },
    }

    if not _env_bool('TAU3_SOLO_MODE'):
        user_client_name = os.getenv('TAU3_USER_CLIENT_NAME')
        if not user_client_name:
            if os.getenv('TAU3_USER_API_KEY') or os.getenv('TAU3_USER_BASE_URL'):
                user_client_name = 'openai'
            else:
                user_client_name = agent_client_name
        user_model = os.getenv('TAU3_USER_MODEL', model)
        user_max_tokens = _env_int('TAU3_USER_MAX_TOKENS') or max_tokens
        user_extra_body = _env_json('TAU3_USER_EXTRA_BODY') or extra_body
        user_extra_headers = _env_json('TAU3_USER_EXTRA_HEADERS') or extra_headers
        user_reasoning_effort = os.getenv('TAU3_USER_REASONING_EFFORT') or reasoning_effort
        user_temperature = _env_float('TAU3_USER_TEMPERATURE')
        user_top_p = _env_float('TAU3_USER_TOP_P')
        user_passthrough_args = _env_json('TAU3_USER_REQUEST_ARGS')

        if user_client_name == 'taiji':
            user_client_args = _build_taiji_client_args(
                user_model,
                service_override_env='TAU3_USER_TAIJI_SERVICE',
                enable_thinking=_env_bool('TAU3_USER_ENABLE_THINKING'),
            )
        else:
            user_api_key = os.getenv('TAU3_USER_API_KEY', os.getenv('TAU3_AGENT_API_KEY', ''))
            user_base_url = os.getenv('TAU3_USER_BASE_URL', os.getenv('TAU3_AGENT_BASE_URL', ''))
            user_client_args = {'api_key': user_api_key, 'base_url': user_base_url}

        clients['user'] = {
            'name': user_client_name,
            'config': {
                'client_args': user_client_args,
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

    eval_client_name = os.getenv('TAU3_EVAL_CLIENT_NAME')
    if not eval_client_name:
        if (
            os.getenv('TAU3_EVAL_API_KEY')
            or os.getenv('TAU3_EVAL_BASE_URL')
            or os.getenv('TAU3_USER_API_KEY')
            or os.getenv('TAU3_USER_BASE_URL')
            or os.getenv('TAU3_AGENT_API_KEY')
            or os.getenv('TAU3_AGENT_BASE_URL')
        ):
            eval_client_name = 'openai'
        else:
            eval_client_name = ''

    eval_model = os.getenv('TAU3_EVAL_MODEL') or os.getenv('TAU3_USER_MODEL') or os.getenv('TAU3_AGENT_MODEL', '')
    eval_max_tokens = _env_int('TAU3_EVAL_MAX_TOKENS') or _env_int('TAU3_USER_MAX_TOKENS') or max_tokens
    eval_extra_body = _env_json('TAU3_EVAL_EXTRA_BODY') or _env_json('TAU3_USER_EXTRA_BODY') or extra_body
    eval_extra_headers = _env_json('TAU3_EVAL_EXTRA_HEADERS') or _env_json('TAU3_USER_EXTRA_HEADERS') or extra_headers
    eval_reasoning_effort = (
        os.getenv('TAU3_EVAL_REASONING_EFFORT') or os.getenv('TAU3_USER_REASONING_EFFORT') or reasoning_effort
    )
    eval_temperature = _env_float('TAU3_EVAL_TEMPERATURE')
    if eval_temperature is None:
        eval_temperature = _env_float('TAU3_USER_TEMPERATURE')
    eval_top_p = _env_float('TAU3_EVAL_TOP_P')
    if eval_top_p is None:
        eval_top_p = _env_float('TAU3_USER_TOP_P')
    eval_passthrough_args = _env_json('TAU3_EVAL_REQUEST_ARGS') or _env_json('TAU3_USER_REQUEST_ARGS')

    if eval_client_name == 'taiji':
        clients['eval'] = {
            'name': 'taiji',
            'config': {
                'client_args': _build_taiji_client_args(
                    eval_model,
                    service_override_env='TAU3_EVAL_TAIJI_SERVICE',
                    enable_thinking=_env_bool('TAU3_EVAL_ENABLE_THINKING'),
                ),
                'request_args': _build_request_args(
                    eval_model,
                    eval_max_tokens,
                    eval_extra_body,
                    eval_extra_headers,
                    eval_reasoning_effort,
                    eval_temperature,
                    eval_top_p,
                    eval_passthrough_args,
                ),
            },
        }
    elif eval_client_name:
        eval_api_key = (
            os.getenv('TAU3_EVAL_API_KEY') or os.getenv('TAU3_USER_API_KEY') or os.getenv('TAU3_AGENT_API_KEY', '')
        )
        eval_base_url = (
            os.getenv('TAU3_EVAL_BASE_URL') or os.getenv('TAU3_USER_BASE_URL') or os.getenv('TAU3_AGENT_BASE_URL', '')
        )
        if eval_api_key or eval_base_url or eval_model:
            clients['eval'] = {
                'name': eval_client_name,
                'config': {
                    'client_args': {'api_key': eval_api_key, 'base_url': eval_base_url},
                    'request_args': _build_request_args(
                        eval_model,
                        eval_max_tokens,
                        eval_extra_body,
                        eval_extra_headers,
                        eval_reasoning_effort,
                        eval_temperature,
                        eval_top_p,
                        eval_passthrough_args,
                    ),
                },
            }
        elif 'user' in clients:
            clients['eval'] = _clone_client_spec(clients['user'])
        else:
            clients['eval'] = _clone_client_spec(clients['main'])

    return clients


def _build_task_config(
    domain: str,
    task_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = _env_int('TAU3_SEED')
    config: dict[str, Any] = {
        'domain': domain,
        'solo_mode': _env_bool('TAU3_SOLO_MODE'),
        'max_steps': _env_int('TAU3_MAX_STEPS', 200) or 200,
        'max_errors': _env_int('TAU3_MAX_ERRORS', 10) or 10,
        'evaluation_type': os.getenv('TAU3_EVALUATION_TYPE', 'all'),
        'add_thinking_tool': _env_bool('TAU3_ADD_THINKING_TOOL'),
    }

    if seed is not None:
        config['seed'] = seed
    if task_data is not None:
        config['task_data'] = task_data

    retrieval_config = os.getenv('TAU3_RETRIEVAL_CONFIG', 'golden_retrieval')
    config['retrieval_config'] = retrieval_config
    retrieval_config_kwargs = _env_json('TAU3_RETRIEVAL_CONFIG_KWARGS')
    if retrieval_config_kwargs is not None:
        config['retrieval_config_kwargs'] = retrieval_config_kwargs

    config['eval_client_name'] = 'eval'

    return config


# --------------- Main generator functions ---------------


def generate_configs() -> Iterator[BatchItem]:
    skip = _env_int('TAU3_SKIP', 0) or 0
    limit = _env_int('TAU3_LIMIT')
    agent_name = os.getenv('TAU3_AGENT_NAME', 'tau3_agent')
    domains = _get_domains()
    openai_clients = _build_openai_clients()

    total_generated = 0
    for domain in domains:
        tasks = _resolve_tasks(domain)
        tasks = tasks[skip:]
        if limit is not None:
            tasks = tasks[:limit]

        logger.debug(f'Generating configs: domain={domain}, tasks={len(tasks)}, skip={skip}, limit={limit}')

        for task_data in tasks:
            task_id = str(task_data.get('id', 'unknown'))
            task_config = _build_task_config(domain=domain, task_data=task_data)

            config = RunInput(
                imports=[
                    'nexus.extensions.tasks.tau3',
                ],
                task_name='tau3',
                task_config=task_config,
                agent_name=agent_name,
                agent_config={},
                openai_clients=openai_clients,
                runner={'timeout': 1800},  # ty: ignore[invalid-argument-type]  # 30 min per task
            )

            yield BatchItem(
                config=config,
                run_id=f'{domain}_task{task_id}',
                metadata={
                    'domain': domain,
                    'task_id': task_id,
                    'data_source': os.getenv('TAU3_TASKS_FILE', f'registry:{domain}'),
                    'agent_model': os.getenv('TAU3_AGENT_MODEL', ''),
                    'user_model': os.getenv('TAU3_USER_MODEL', os.getenv('TAU3_AGENT_MODEL', '')),
                    'eval_model': os.getenv(
                        'TAU3_EVAL_MODEL', os.getenv('TAU3_USER_MODEL', os.getenv('TAU3_AGENT_MODEL', ''))
                    ),
                    'agent_name': agent_name,
                },
            )
            total_generated += 1

    logger.debug(f'Generated {total_generated} batch items for tau3 across {len(domains)} domain(s)')


def get_total_count() -> int:
    skip = _env_int('TAU3_SKIP', 0) or 0
    limit = _env_int('TAU3_LIMIT')

    try:
        domains = _get_domains()
        total = 0
        for domain in domains:
            tasks = _resolve_tasks(domain)
            total += max(0, len(tasks) - skip)
            if limit:
                total = min(total, limit * len(domains))
    except Exception:
        return 0

    return total


def get_batch_metadata() -> dict[str, Any]:
    agent_client_name = os.getenv('TAU3_AGENT_CLIENT_NAME', 'openai')
    user_client_name = os.getenv('TAU3_USER_CLIENT_NAME')
    if not user_client_name:
        if os.getenv('TAU3_USER_API_KEY') or os.getenv('TAU3_USER_BASE_URL'):
            user_client_name = 'openai'
        else:
            user_client_name = agent_client_name
    agent_model = os.getenv('TAU3_AGENT_MODEL', '')
    user_model = os.getenv('TAU3_USER_MODEL', agent_model)

    return {
        'task_type': 'tau3',
        'domain': os.getenv('TAU3_DOMAIN', 'all'),
        'tasks_file': os.getenv('TAU3_TASKS_FILE', ''),
        'task_split': os.getenv('TAU3_TASK_SPLIT', ''),
        'agent_client_name': agent_client_name,
        'agent_model': agent_model,
        'user_client_name': user_client_name,
        'user_model': user_model,
        'agent_name': os.getenv('TAU3_AGENT_NAME', 'tau3_agent'),
        'solo_mode': _env_bool('TAU3_SOLO_MODE'),
        'evaluation_type': os.getenv('TAU3_EVALUATION_TYPE', 'all'),
        'max_steps': _env_int('TAU3_MAX_STEPS', 200) or 200,
        'max_errors': _env_int('TAU3_MAX_ERRORS', 10) or 10,
        'seed': _env_int('TAU3_SEED'),
        'skip': _env_int('TAU3_SKIP', 0) or 0,
        'limit': _env_int('TAU3_LIMIT'),
        'retrieval_config': os.getenv('TAU3_RETRIEVAL_CONFIG'),
    }
