#!/usr/bin/env python3
"""SWE-bench batch generator for `nexus run-batch`.

This generator reads a local SWE dataset in JSONL format or fetches instance list from
a remote dataset server, and yields `BatchItem`s, one per instance, suitable for running
the `swe_task` at scale.

Usage:
  uv run nexus run-batch -g ./workspace/swe_generator.py -d ./outputs/swe_batch.db --max-concurrent 5

Environment variables:
  - SWE_DATASET: remote dataset path (e.g., "swe-public/SWE-bench_Verified")
                 When set, fetches instance list from dataset server and uses remote loading.
                 Takes precedence over SWE_DATASET_PATH.
  - SWE_DATASET_SPLIT: split name for remote dataset (default: empty, uses default split)
  - SWE_DATASET_PATH: path to JSONL dataset (default: ./workspace/swe-bench/swe-bench.jsonl)
                      Only used when SWE_DATASET is not set.
  - SWE_INSTANCE_IDS: comma-separated list of specific instance_ids to run (default: all)
                      Example: "django__django-10097,psf__requests-1724"
  - SWE_SKIP: number of instances to skip before starting (default: 0)
  - SWE_LIMIT: max instances to yield (default: no limit)
  - SWE_RUN_ORACLE: "1"/"true" to use golden patch without agent (default: false)
  - SWE_WORKER_VENV: "1"/"true" to upload the current nexus wheel into the eval sandbox (default: false)
                     Needed when the host has dataset handlers not baked into the runtime image.
  - SWE_RECORD_TRAJECTORY: "1"/"true" to record agent LLM trajectories via chat_tracker (default: true)
                           Results land in each run's RunOutput.processor_results.chat.trajectories. Set "0" for very large sweeps.
  - SWE_TOOLS_MODE: tools mode for chat_tracker (default: branch_last)
  - SWE_AGENT_NAME: agent name (default: swe_agent; ignored if run_oracle=true)
  - SWE_MAX_ITERATIONS: max agent steps/model calls (default: 5). Applies to swe_agent and mini_swe_agent
                        (mini mirrors swe_agent's config surface — this is its only tunable knob).
  - SWE_RUNTIME_BASE_URL: runtime base URL (default: http://hyrl-sandbox.prod.woa.com:8052)
  - SWE_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: gongfeng)
  - SWE_RUNTIME_TAG: runtime tag to use (default: current)
  - SWE_RUNTIME_TIMEOUT: runtime timeout in seconds (default: 1800)
  - SWE_RUNTIME_CPU / SWE_RUNTIME_MEMORY: default runtime resources (default: 2 / 10)
  - GONGFENG_RUNTIME_TOKEN: authentication token for Gongfeng sandbox API (required for gongfeng provider)
  - OPENAI_CLIENT_NAME / OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: used to configure OpenAI client (optional)
  - CLAUDE_CODE_VERSION / CLAUDE_CODE_BINARY_URL / CLAUDE_CODE_BINARY_CHECKSUM /
    CLAUDE_CODE_WORKING_DIR / CLAUDE_CODE_TIMEOUT: used when SWE_AGENT_NAME is
    claude_code or claude_code_interactive
  - SWE_PTY_TOOL_SEARCH: enable progressive tool disclosure for claude_code_interactive (default: false)
  - SWE_PTY_TOOL_SEARCH_MODE: ENABLE_TOOL_SEARCH value when enabled (default: true)
  - SWE_PTY_LOG_FRAMES / SWE_PTY_ROWS / SWE_PTY_COLS / SWE_PTY_PROMPT_SETTLE_SECONDS:
    PTY diagnostics and terminal geometry for claude_code_interactive
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_BENCH, SWEbenchInstance
from nexus.runner.single import RunInput


DEFAULT_SWE_DATASET_PATH = './workspace/swe-bench.jsonl'

# Override resource_spec for specific instances
# Format: {'instance_id': {'cpu': N, 'memory': M}}
INSTANCE_RESOURCE_OVERRIDE = {
    # scikit-learn__scikit-learn-14710 需要进行大量的数学计算, 十分耗cpu
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},
}


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
    """Parse comma-separated list from env var, return as set for O(1) lookup."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _iter_instances_from_file(dataset_path: str) -> Iterator[SWEbenchInstance]:
    """Iterate over instances from a local JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_remote_instance_ids(dataset: str, split: str | None = None) -> Iterator[str]:
    """Iterate over instance IDs from the remote dataset server."""
    from nexus.utils.legacy_dataset_client import list_instances_sync

    instance_ids = list_instances_sync(dataset, split)
    yield from instance_ids


def _iter_instance_ids(
    remote_dataset: str | None, remote_split: str | None, dataset_path: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate over (instance_id, metadata) tuples from remote or local source."""
    if remote_dataset:
        # Remote mode: iterate over instance IDs from server
        for instance_id in _iter_remote_instance_ids(remote_dataset, remote_split):
            if instance_id:
                yield (
                    instance_id,
                    {
                        'dataset': remote_dataset,
                        'split': remote_split or '',
                    },
                )
    else:
        # Local mode: iterate over instances from local file
        for inst in _iter_instances_from_file(dataset_path):
            instance_id = str(inst.get('instance_id', '') or '')
            if instance_id:
                inst_dataset_type = str(inst.get('dataset_type') or DATASET_SWE_BENCH)
                yield (
                    instance_id,
                    {
                        'dataset_path': dataset_path,
                        'repo': inst.get('repo', ''),
                        'dataset_type': inst_dataset_type,
                    },
                )


def _build_openai_clients(*, defer_sensitive_env: bool = False) -> dict[str, dict[str, Any]]:
    client_name = os.getenv('OPENAI_CLIENT_NAME', 'openai')
    if defer_sensitive_env:
        # Batch configs are persisted in SQLite. Keep credentials and private
        # endpoints out of the database; RunInput.from_dict resolves these
        # placeholders from the execution environment when each item starts.
        api_key = '${oc.env:OPENAI_API_KEY,""}'
        base_url = '${oc.env:OPENAI_BASE_URL,""}'
        model = '${oc.env:OPENAI_MODEL,""}'
    else:
        api_key = os.getenv('OPENAI_API_KEY', 'sk-proj-1234567890')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        model = os.getenv('OPENAI_MODEL', 'gpt-4.1')
    return {
        'main': {
            'name': client_name,
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
    }


def _build_agent_config(agent_name: str) -> dict[str, Any]:
    if agent_name in {'claude_code', 'claude_code_interactive'}:
        agent_config: dict[str, Any] = {
            'working_dir': os.getenv('CLAUDE_CODE_WORKING_DIR', '/testbed'),
            'timeout': float(_env_int('CLAUDE_CODE_TIMEOUT') or 3600),
            'version': os.getenv('CLAUDE_CODE_VERSION', 'latest'),
        }
        binary_url = os.getenv('CLAUDE_CODE_BINARY_URL', '').strip()
        if binary_url:
            agent_config['binary_url'] = binary_url
        binary_checksum = os.getenv('CLAUDE_CODE_BINARY_CHECKSUM', '').strip()
        if binary_checksum:
            agent_config['binary_checksum'] = binary_checksum

        if agent_name == 'claude_code_interactive':
            model = os.getenv('CLAUDE_CODE_MODEL') or os.getenv('OPENAI_MODEL') or 'anthropic.claude-opus-4-8'
            extra_env = {
                'ANTHROPIC_MODEL': model,
                'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            }
            if _env_bool('SWE_PTY_TOOL_SEARCH'):
                extra_env['ENABLE_TOOL_SEARCH'] = os.getenv('SWE_PTY_TOOL_SEARCH_MODE', 'true')
            else:
                extra_env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] = '1'

            agent_config.update(
                extra_env=extra_env,
                rows=int(os.getenv('SWE_PTY_ROWS', '50')),
                cols=int(os.getenv('SWE_PTY_COLS', '200')),
                prompt_settle_seconds=float(os.getenv('SWE_PTY_PROMPT_SETTLE_SECONDS', '4')),
                log_frames=_env_bool('SWE_PTY_LOG_FRAMES'),
            )
        return agent_config

    # Default branch for swe_agent and mini_swe_agent: both share the same config
    # surface, so only max_iterations is tunable (via SWE_MAX_ITERATIONS). mini's
    # working_dir is supplied per-instance by swe_task; every other MiniSweAgentInput
    # field keeps its default.
    return {
        'max_iterations': int(os.getenv('SWE_MAX_ITERATIONS', '5')),
    }


def generate_configs() -> Iterator[BatchItem]:
    # Check for remote dataset first (takes precedence over local file)
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)

    skip = _env_int('SWE_SKIP') or 0
    limit = _env_int('SWE_LIMIT')
    instance_ids_filter = _env_list('SWE_INSTANCE_IDS')  # specific instance_ids to run
    run_oracle = _env_bool('SWE_RUN_ORACLE', default=False)
    raise_on_runtime_error = _env_bool('SWE_RAISE_ON_RUNTIME_ERROR', default=False)
    eval_max_attempts = _env_int('SWE_EVAL_MAX_ATTEMPTS') or 1
    runtime_provider_name = os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('SWE_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    runtime_tag = os.getenv('SWE_RUNTIME_TAG', 'current').strip() or 'current'
    runtime_timeout = _env_int('SWE_RUNTIME_TIMEOUT') or 1800.0
    agent_name = os.getenv('SWE_AGENT_NAME', 'swe_agent')
    defer_sensitive_env = agent_name == 'claude_code_interactive'
    runtime_token = (
        '${oc.env:GONGFENG_RUNTIME_TOKEN,""}' if defer_sensitive_env else os.getenv('GONGFENG_RUNTIME_TOKEN', '')
    )
    runtime_cpu = _env_int('SWE_RUNTIME_CPU') or 2
    runtime_memory = _env_int('SWE_RUNTIME_MEMORY') or 10
    rpc_protocol = os.getenv('SWE_RPC_PROTOCOL', 'http')
    worker_venv = _env_bool('SWE_WORKER_VENV', default=False)
    record_trajectory = _env_bool('SWE_RECORD_TRAJECTORY', default=True)
    tools_mode = os.getenv('SWE_TOOLS_MODE', 'branch_last')
    processors = {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': tools_mode}}} if record_trajectory else {}

    runtime_image = (
        runtime_tag if runtime_tag == 'current' else f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}'
    )

    openai_clients = _build_openai_clients(defer_sensitive_env=defer_sensitive_env)

    imports = [
        'nexus.extensions.tasks.swe',
        'nexus.extensions.tasks.swe.tests.mock_swe_agent',
        'nexus.extensions.agents.swe_agent',
        'nexus.extensions.agents.codeact',
        'nexus.extensions.agents.terminus2',
    ]
    if agent_name in {'claude_code', 'claude_code_interactive'}:
        imports.append('nexus.extensions.agents.claude_code')

    runtime_configs: dict[str, dict[str, bool]] = {}
    if worker_venv:
        runtime_configs['eval'] = {'worker_venv': True}
        if agent_name == 'claude_code_interactive':
            runtime_configs['main'] = {'worker_venv': True}

    skipped = 0
    produced = 0

    for instance_id, instance_meta in _iter_instance_ids(remote_dataset, remote_split, dataset_path):
        # Filter by specific instance_ids if provided
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue

        # Skip instances until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        # Build task_config based on source type
        if remote_dataset:
            task_config: dict[str, Any] = {
                'remote_dataset': remote_dataset,
                'split': remote_split or '',
                'instance_id': instance_id,
                'run_oracle': bool(run_oracle),
                'raise_on_runtime_error': bool(raise_on_runtime_error),
                'eval_max_attempts': eval_max_attempts,
            }
        else:
            task_config = {
                'dataset_path': dataset_path,
                'instance_id': instance_id,
                'run_oracle': bool(run_oracle),
                'raise_on_runtime_error': bool(raise_on_runtime_error),
                'eval_max_attempts': eval_max_attempts,
            }

        config: dict[str, Any] = {
            'imports': imports,
            'task_name': 'swe_task',
            'task_config': task_config,
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'api_base_url': runtime_base_url,
                'token': runtime_token,
                'runtime_image': runtime_image,
                'request_timeout': runtime_timeout,
                'resource_spec': {
                    'cpu': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('cpu', runtime_cpu),
                    'memory': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('memory', runtime_memory),
                },
                'rpc_protocol': rpc_protocol,
            },
            'runtime_configs': runtime_configs,
            'processors': processors,
        }

        # If not running oracle, we need an agent + LLM config.
        if not run_oracle:
            config['agent_name'] = agent_name
            config['agent_config'] = _build_agent_config(agent_name)
            config['openai_clients'] = openai_clients
        else:
            config['agent_name'] = None
            config['openai_clients'] = {}

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=instance_id,
            metadata={'instance_id': instance_id, **instance_meta},
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)
    skip = _env_int('SWE_SKIP') or 0
    limit = _env_int('SWE_LIMIT')
    instance_ids_filter = _env_list('SWE_INSTANCE_IDS')

    n = 0
    skipped = 0

    for instance_id, _ in _iter_instance_ids(remote_dataset, remote_split, dataset_path):
        # Filter by specific instance_ids if provided
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue
        # Skip instances until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue
        n += 1
        if limit is not None and n >= limit:
            return limit
    return n


def get_batch_metadata() -> dict[str, Any]:
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)
    instance_ids_filter = _env_list('SWE_INSTANCE_IDS')

    metadata: dict[str, Any] = {
        'task_name': 'swe_task',
        'run_oracle': _env_bool('SWE_RUN_ORACLE', default=False),
        'instance_ids_filter': list(instance_ids_filter) if instance_ids_filter else None,
    }

    agent_name = None if metadata['run_oracle'] else os.getenv('SWE_AGENT_NAME', 'swe_agent')
    if agent_name == 'claude_code_interactive':
        metadata.update(
            agent=agent_name,
            claude_code_version=os.getenv('CLAUDE_CODE_VERSION', 'latest'),
            model=os.getenv('CLAUDE_CODE_MODEL') or os.getenv('OPENAI_MODEL', 'anthropic.claude-opus-4-8'),
            tool_search=_env_bool('SWE_PTY_TOOL_SEARCH'),
        )

    if remote_dataset:
        metadata['dataset'] = remote_dataset
        metadata['split'] = remote_split or ''
    else:
        metadata['dataset_path'] = dataset_path
        metadata['dataset_type_default'] = DATASET_SWE_BENCH

    return metadata
