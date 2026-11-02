"""Shared utilities for rubric batch generators."""

import contextlib
import json
import logging
import os
import sqlite3

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> list[dict]:
    records = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_dataset_instances(dataset_path: str) -> list[dict]:
    path = Path(dataset_path)
    if path.suffix == '.jsonl':
        return load_jsonl(dataset_path)
    with open(dataset_path, encoding='utf-8') as f:
        data = json.loads(f.read())
    return data if isinstance(data, list) else [data]


def load_runs_from_db(db_path: str, status: str = 'success') -> list[dict]:
    """Load runs from a batch SQLite DB, filtered by status.

    Returns normalized records:
    {
        'instance_id': str,
        'task_output': dict,
        'metadata': dict,
    }
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        'SELECT run_id, metadata, task_output FROM runs WHERE status = ? AND task_output IS NOT NULL',
        (status,),
    )

    records = []
    for row in cursor.fetchall():
        run_id = row['run_id']

        metadata = {}
        if row['metadata']:
            with contextlib.suppress(json.JSONDecodeError):
                metadata = json.loads(row['metadata'])

        task_output = {}
        if row['task_output']:
            try:
                task_output = json.loads(row['task_output'])
            except json.JSONDecodeError:
                continue

        instance_id = metadata.get('instance_id', '')
        if not instance_id:
            for suffix in ('_infer', '_rubric', '_run', '_rubric_gen', '_rubric_eval'):
                if run_id.endswith(suffix):
                    instance_id = run_id[: -len(suffix)]
                    break
            if not instance_id:
                instance_id = run_id

        records.append({
            'instance_id': instance_id,
            'task_output': task_output,
            'metadata': metadata,
        })

    conn.close()
    logger.info(f'Loaded {len(records)} runs (status={status}) from DB: {db_path}')
    return records


def load_runs_from_source(path: str) -> list[dict]:
    """Load runs from DB or JSONL, auto-detected by extension."""
    if path.endswith('.db'):
        return load_runs_from_db(path)
    raw = load_jsonl(path)
    records = []
    for r in raw:
        task_output = r.get('task_output', {})
        metadata = r.get('metadata', {})
        instance_id = metadata.get('instance_id') or r.get('instance_id', '')
        records.append({
            'instance_id': instance_id,
            'task_output': task_output,
            'metadata': metadata,
        })
    return records


def env_list(key: str) -> list[str]:
    val = os.environ.get(key, '')
    return [x.strip() for x in val.split(',') if x.strip()] if val else []


def env_int(key: str, default: int = 0) -> int:
    val = os.environ.get(key, '')
    return int(val) if val else default


def require_env(key: str, msg: str | None = None) -> str:
    val = os.environ.get(key, '')
    assert val, msg or f'Environment variable {key} is required'
    return val


def require_path(key: str, msg: str | None = None) -> str:
    val = require_env(key, msg)
    assert Path(val).exists(), f'{key}={val} not found'
    return val


def build_litellm_client(
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    return {
        'name': 'litellm',
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


def build_llm_client_from_env(
    prefix: str = '',
) -> dict[str, Any]:
    """Build litellm client config from env vars.

    Tries {PREFIX}_API_KEY first, falls back to OPENAI_API_KEY, etc.
    Asserts if nothing found.
    """
    api_key = os.environ.get(f'{prefix}_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
    base_url = os.environ.get(f'{prefix}_BASE_URL', '') or os.environ.get('OPENAI_BASE_URL', '')
    model = os.environ.get(f'{prefix}_MODEL', '') or os.environ.get('OPENAI_MODEL', '')

    label = prefix or 'OPENAI'
    assert api_key, f'{label}_API_KEY (or OPENAI_API_KEY) is required'
    assert base_url, f'{label}_BASE_URL (or OPENAI_BASE_URL) is required'
    assert model, f'{label}_MODEL (or OPENAI_MODEL) is required'

    return build_litellm_client(api_key, base_url, model)


LITELLM_IMPORT = 'nexus.extensions.openai_clients.litellm'
AGENT_IMPORT = 'nexus.extensions.tasks.swe.swe_agent_rubrics'

# Task module imports (per-task, avoids triggering all registrations in runtime containers)
AGENTIC_PATCH_BASE = 'nexus.extensions.tasks.swe.rubrics.agentic_patch'
GEN_TASK_IMPORT = f'{AGENTIC_PATCH_BASE}.gen_task'
EVAL_TASK_IMPORT = f'{AGENTIC_PATCH_BASE}.eval_task'
FULL_TASK_IMPORT = f'{AGENTIC_PATCH_BASE}.task'


def build_runtime_provider_config() -> dict[str, Any]:
    """Build runtime_provider_config from env vars.

    Environment variables:
      SWE_RUNTIME_PROVIDER_NAME: provider name (default: gongfeng)
      SWE_RUNTIME_BASE_URL: API base URL
      SWE_RUNTIME_TOKEN: authentication token (required for gongfeng)
      SWE_RUNTIME_IMAGE: runtime image (default: current)
      SWE_RUNTIME_TIMEOUT: request timeout in seconds (default: 7200)
      SWE_RUNTIME_CPU: CPU cores (default: 2)
      SWE_RUNTIME_MEMORY: memory in GB (default: 10)
      SWE_RPC_PROTOCOL: rpc protocol (default: http)
    """
    return {
        'api_base_url': os.environ.get('SWE_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
        'token': os.environ.get('SWE_RUNTIME_TOKEN', ''),
        'runtime_image': os.environ.get('SWE_RUNTIME_IMAGE', 'current'),
        'request_timeout': float(os.environ.get('SWE_RUNTIME_TIMEOUT', '7200')),
        'resource_spec': {
            'cpu': env_int('SWE_RUNTIME_CPU', default=2),
            'memory': env_int('SWE_RUNTIME_MEMORY', default=10),
        },
        'rpc_protocol': os.environ.get('SWE_RPC_PROTOCOL', 'http'),
    }


def build_agent_config() -> dict[str, Any]:
    """Build agent config from env vars.

    Environment variables:
      SWE_MAX_ITERATIONS: max agent iterations (default: 50)
    """
    return {
        'max_iterations': env_int('SWE_MAX_ITERATIONS', default=50),
    }


def build_processors() -> dict[str, Any]:
    """Build default processors config (chat_tracker)."""
    from nexus.runner.single import ProcessorConfig

    return {
        'chat': ProcessorConfig(
            name='chat_tracker',
            args={'tools_mode': 'branch_last'},
        ),
    }
