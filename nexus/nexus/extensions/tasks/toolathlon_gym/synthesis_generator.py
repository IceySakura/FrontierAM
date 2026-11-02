"""Batch generator for self-contained Toolathlon parquet manifests.

The source parquet stores the complete ``RunInput`` in its ``nexus`` column,
including the task/data binding.  A lightweight JSONL manifest is used only as
an iterator because the normal Nexus environment does not depend on pyarrow;
the manifest must preserve the embedded ``nexus`` object byte-for-byte.
"""

from __future__ import annotations

import json
import os
import socket

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.mcp_benchmark.generator import _build_client_config
from nexus.runner.single import RunInput


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _build_openai_clients() -> dict[str, Any]:
    client = _build_client_config('MCP_BENCHMARK_OPENAI', default_block='main')
    client_args = client.get('config', {}).get('client_args', {})
    wsid = client_args.pop('wsid', None)
    if wsid:
        headers = dict(client_args.get('default_headers') or {})
        headers.setdefault('wsid', str(wsid))
        client_args['default_headers'] = headers
    return {'main': client}


def _load_rows() -> list[dict[str, Any]]:
    path = os.getenv('SYNTHESIS_MANIFEST_JSONL')
    if not path:
        raise ValueError('SYNTHESIS_MANIFEST_JSONL is not set')
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'invalid JSONL at {path}:{line_number}') from exc
    skip = _env_int('SYNTHESIS_SKIP', 0) or 0
    limit = _env_int('SYNTHESIS_LIMIT')
    rows = rows[skip:]
    return rows if limit is None else rows[:limit]


def _apply_overrides(nexus: dict[str, Any]) -> None:
    nexus['openai_clients'] = _build_openai_clients()
    nexus.setdefault('processors', {'agent_metrics': {'name': 'main_agent_metrics'}})
    nexus.setdefault('runner', {})['timeout'] = float(os.getenv('TOOLATHLON_GYM_RUN_TIMEOUT', '7200'))

    max_iterations = _env_int('TOOLATHLON_GYM_MAX_ITERATIONS')
    if max_iterations is not None:
        nexus.setdefault('task_config', {})['max_iterations'] = max_iterations

    provider = nexus.setdefault('runtime_provider_config', {})
    overrides = {
        'token': os.getenv('MCP_BENCHMARK_RUNTIME_TOKEN'),
        'api_base_url': os.getenv('MCP_BENCHMARK_RUNTIME_BASE_URL'),
        'purpose': os.getenv('MCP_BENCHMARK_RUNTIME_PURPOSE'),
    }
    provider.update({key: value for key, value in overrides.items() if value})
    cpu = _env_int('MCP_BENCHMARK_RUNTIME_CPU')
    memory = _env_int('MCP_BENCHMARK_RUNTIME_MEMORY')
    if cpu is not None or memory is not None:
        resources = dict(provider.get('resource_spec') or {})
        if cpu is not None:
            resources['cpu'] = cpu
        if memory is not None:
            resources['memory'] = memory
        provider['resource_spec'] = resources

    image = os.getenv('MCP_BENCHMARK_MAIN_IMAGE')
    if image:
        nexus.setdefault('runtime_configs', {}).setdefault('main', {})['image'] = image


def generate_configs() -> Iterator[BatchItem]:
    for row in _load_rows():
        nexus = row['nexus']
        _apply_overrides(nexus)
        env_id = row.get('env_id') or nexus.get('task_config', {}).get('case_id', 'unknown')
        task_index = int(row.get('task_index') or 0)
        yield BatchItem(
            config=RunInput(**nexus),
            run_id=f'bound-{env_id}-t{task_index}',
            metadata={
                'dataset': row.get('dataset', ''),
                'env_id': env_id,
                'task_index': task_index,
                'case_id': nexus.get('task_config', {}).get('case_id', ''),
            },
        )


def get_total_count() -> int:
    return len(_load_rows())


def get_batch_metadata() -> dict[str, Any]:
    rows = _load_rows()
    return {
        'task_type': 'toolathlon_bound_parquet',
        'generated_count': len(rows),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
    }
