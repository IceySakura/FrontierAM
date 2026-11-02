#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.mcp_benchmark.atlas_selection import (
    ATLAS_MODE_ENV,
    atlas_mode_to_server_mode,
    atlas_selection_from_env,
    select_atlas_case_ids,
)
from nexus.extensions.tasks.mcp_benchmark.constants import BENCHMARK_IMAGE_NAME, TEST_ATLAS_YAML
from nexus.extensions.tasks.mcp_benchmark.credentials import WorkspaceKeys
from nexus.extensions.tasks.mcp_benchmark.datasets import collect_cases
from nexus.extensions.tasks.mcp_benchmark.toolathlon_selection import (
    TOOLATHLON_MODE_ENV,
    select_toolathlon_case_ids,
    toolathlon_mode_to_server_mode,
    toolathlon_selection_from_env,
)
from nexus.extensions.tasks.mcp_benchmark.universe_selection import (
    UNIVERSE_MODE_ENV,
    select_universe_case_ids,
    universe_mode_to_server_mode,
    universe_selection_from_env,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput, RunnerConfig


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _env_list(name: str) -> set[str] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    items = {item.strip() for item in raw.split(',') if item.strip()}
    return items or None


def _datasets_env() -> list[str]:
    raw = os.getenv('MCP_BENCHMARK_DATASETS', 'toolathlon,atlas,universe')
    return [item.strip() for item in raw.split(',') if item.strip()]


def _selected_dataset_mode_override(datasets: list[str]) -> tuple[str, str, str] | None:
    if len(datasets) != 1:
        return None

    dataset = datasets[0]
    if dataset == 'atlas':
        mode = (os.getenv(ATLAS_MODE_ENV) or '').strip()
        if not mode:
            return None
        return ATLAS_MODE_ENV, mode, atlas_mode_to_server_mode(mode)
    if dataset == 'toolathlon':
        mode = (os.getenv(TOOLATHLON_MODE_ENV) or '').strip()
        if not mode:
            return None
        return TOOLATHLON_MODE_ENV, mode, toolathlon_mode_to_server_mode(mode)
    if dataset == 'universe':
        mode = (os.getenv(UNIVERSE_MODE_ENV) or '').strip()
        if not mode:
            return None
        return UNIVERSE_MODE_ENV, mode, universe_mode_to_server_mode(mode)
    return None


def _server_modes_env(datasets: list[str]) -> list[str]:
    raw = os.getenv('MCP_BENCHMARK_SERVER_MODES') or os.getenv('MCP_BENCHMARK_SERVER_MODE')
    if raw:
        return [item.strip() for item in raw.split(',') if item.strip()]
    dataset_mode_override = _selected_dataset_mode_override(datasets)
    if dataset_mode_override is not None:
        return [dataset_mode_override[2]]
    return ['api']


def _validate_dataset_mode(*, datasets: list[str], server_modes: list[str]) -> None:
    dataset_mode_override = _selected_dataset_mode_override(datasets)
    if dataset_mode_override is None:
        return

    mode_env, selected_mode, expected_server_mode = dataset_mode_override
    if server_modes != [expected_server_mode]:
        raise ValueError(
            f'{mode_env}={selected_mode} requires MCP_BENCHMARK_SERVER_MODES={expected_server_mode} '
            f'(or leave MCP_BENCHMARK_SERVER_MODES unset and let generator infer it).'
        )


def _apply_dataset_selection_filters(
    cases: list[Any],
    skipped: list[dict[str, Any]],
    *,
    dataset: str,
    selection_from_env: Any,
    select_case_ids: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    filters = selection_from_env()
    if filters is None:
        return cases, skipped

    selected_case_ids = set(select_case_ids(filters, keys=WorkspaceKeys()))
    if not selected_case_ids:
        return [case for case in cases if case.dataset.value != dataset], [
            item for item in skipped if item['dataset'] != dataset
        ]

    filtered_cases = [case for case in cases if case.dataset.value != dataset or case.case_id in selected_case_ids]
    filtered_skipped = [item for item in skipped if item['dataset'] != dataset or item['case_id'] in selected_case_ids]
    return filtered_cases, filtered_skipped


_DEFAULT_OPENAI_CLIENTS: dict[str, Any] | None = None


def _load_default_openai_clients() -> dict[str, Any]:
    global _DEFAULT_OPENAI_CLIENTS
    if _DEFAULT_OPENAI_CLIENTS is None:
        if TEST_ATLAS_YAML.exists():
            data = yaml.safe_load(TEST_ATLAS_YAML.read_text()) or {}
            _DEFAULT_OPENAI_CLIENTS = data.get('openai_clients', {}) or {}
        else:
            _DEFAULT_OPENAI_CLIENTS = {}
    return _DEFAULT_OPENAI_CLIENTS


def _require_env_or_default(*names: str, default: str | None = None) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    if default:
        return default
    joined = "', '".join(names)
    raise ValueError(f"Required environment variable '{joined}' is not set.")


def _client_default(block: str, field: str) -> str | None:
    defaults = _load_default_openai_clients()
    if field == 'name':
        return defaults.get(block, {}).get('name')
    if field == 'base_url':
        return defaults.get(block, {}).get('config', {}).get('client_args', {}).get('base_url')
    if field == 'api_key':
        return defaults.get(block, {}).get('config', {}).get('client_args', {}).get('api_key')
    if field == 'model':
        return defaults.get(block, {}).get('config', {}).get('request_args', {}).get('model')
    return None


def _build_client_config(prefix: str, *, default_block: str, fallback_prefix: str | None = None) -> dict[str, Any]:
    def _env_names(suffix: str) -> tuple[str, ...]:
        names = [f'{prefix}_{suffix}']
        if fallback_prefix:
            names.append(f'{fallback_prefix}_{suffix}')
        return tuple(names)

    name = _require_env_or_default(*_env_names('CLIENTS_NAME'), default=_client_default(default_block, 'name'))
    base_url = _require_env_or_default(*_env_names('BASE_URL'), default=_client_default(default_block, 'base_url'))
    api_key = _require_env_or_default(*_env_names('API_KEY'), default=_client_default(default_block, 'api_key'))
    model = _require_env_or_default(*_env_names('MODEL'), default=_client_default(default_block, 'model'))

    client_args: dict[str, Any] = {'base_url': base_url, 'api_key': api_key}
    wsid = None
    for env_name in _env_names('WSID'):
        wsid = os.getenv(env_name)
        if wsid:
            break
    if wsid:
        client_args['wsid'] = wsid

    request_args: dict[str, Any] = {'model': model}

    # Allow environment variable overrides for generation parameters, with sensible defaults
    # aligned to the upstream Toolathlon baseline configuration.
    # Note: Set MCP_BENCHMARK_OPENAI_TEMPERATURE="" to omit temperature (e.g. for Claude thinking mode)
    temperature = os.getenv('MCP_BENCHMARK_OPENAI_TEMPERATURE', '0.6')
    if temperature:
        request_args['temperature'] = float(temperature)

    top_p = os.getenv('MCP_BENCHMARK_OPENAI_TOP_P', '1')
    if top_p:
        request_args['top_p'] = float(top_p)

    max_tokens_env = os.getenv('MCP_BENCHMARK_OPENAI_MAX_TOKENS', '32768')
    request_args['max_tokens'] = int(max_tokens_env)

    # extra_body for model-specific options (e.g. qwen thinking mode, claude thinking)
    extra_body: dict[str, Any] = {}

    enable_thinking = os.getenv('MCP_BENCHMARK_OPENAI_ENABLE_THINKING', '').lower()
    if enable_thinking in ('1', 'true', 'yes'):
        extra_body['chat_template_kwargs'] = {'enable_thinking': True}

    # Allow arbitrary extra_body via JSON env var (e.g. for Claude thinking config)
    # Example: MCP_BENCHMARK_OPENAI_EXTRA_BODY='{"thinking": {"type": "enabled", "budget_tokens": 10000}}'
    extra_body_json = os.getenv('MCP_BENCHMARK_OPENAI_EXTRA_BODY', '')
    if extra_body_json:
        import json as _json

        extra_body.update(_json.loads(extra_body_json))

    if extra_body:
        request_args['extra_body'] = extra_body

    return {
        'name': name,
        'config': {
            'client_args': client_args,
            'request_args': request_args,
        },
    }


def _build_openai_clients(needs_judge: bool) -> dict[str, dict[str, Any]]:
    clients = {'main': _build_client_config('MCP_BENCHMARK_OPENAI', default_block='main')}
    if needs_judge:
        clients['judge'] = _build_client_config(
            'JUDGE_OPENAI',
            default_block='judge',
            fallback_prefix='MCP_BENCHMARK_OPENAI',
        )
    return clients


def _build_runtime_settings() -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    provider = os.getenv('MCP_BENCHMARK_RUNTIME_PROVIDER_NAME', 'docker')
    main_image = os.getenv('MCP_BENCHMARK_MAIN_IMAGE', BENCHMARK_IMAGE_NAME)
    runtime_image = os.getenv('MCP_BENCHMARK_RUNTIME_IMAGE', 'current')
    request_timeout = float(os.getenv('MCP_BENCHMARK_RUNTIME_REQUEST_TIMEOUT', '1800'))
    runtime_cluster = (
        os.getenv('MCP_BENCHMARK_RUNTIME_PROVIDER_CLUSTER')
        or os.getenv('MCP_BENCHMARK_RUNTIME_CLUSTER')
        or os.getenv('MCP_BENCHMARK_GONGFENG_RUNTIME_PROVIDER')
    )
    runtime_cpu = _env_int('MCP_BENCHMARK_RUNTIME_CPU')
    runtime_memory = _env_int('MCP_BENCHMARK_RUNTIME_MEMORY')

    if provider == 'docker':
        provider_config: dict[str, Any] = {
            'runtime_image': runtime_image,
            'request_timeout': request_timeout,
        }
    else:
        provider_config = {
            'api_base_url': os.getenv('MCP_BENCHMARK_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
            'token': os.getenv('MCP_BENCHMARK_RUNTIME_TOKEN', ''),
            'runtime_image': runtime_image,
            'request_timeout': request_timeout,
            'rpc_protocol': os.getenv('MCP_BENCHMARK_RPC_PROTOCOL', 'http'),
        }
        if runtime_cluster:
            provider_config['runtime_provider'] = runtime_cluster
        resource_spec = {}
        if runtime_cpu is not None:
            resource_spec['cpu'] = runtime_cpu
        if runtime_memory is not None:
            resource_spec['memory'] = runtime_memory
        if resource_spec:
            provider_config['resource_spec'] = resource_spec

    runtime_configs = {
        'main': {
            'image': main_image,
            'working_dir': '/workspace',
        }
    }
    return provider, provider_config, runtime_configs


def _write_generation_report(
    path: str | None, *, generated: list[dict[str, Any]], skipped: list[dict[str, Any]]
) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                'generated_count': len(generated),
                'skipped_count': len(skipped),
                'generated': generated,
                'skipped': skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _collect_filtered_cases() -> tuple[list[Any], list[dict[str, Any]]]:
    datasets = _datasets_env()
    server_modes = _server_modes_env(datasets)
    _validate_dataset_mode(datasets=datasets, server_modes=server_modes)
    case_ids_filter = _env_list('MCP_BENCHMARK_CASE_IDS')
    skip = _env_int('MCP_BENCHMARK_SKIP', 0) or 0
    limit = _env_int('MCP_BENCHMARK_LIMIT')
    max_iterations_override = _env_int('MCP_BENCHMARK_MAX_ITERATIONS')

    cases, skipped = collect_cases(datasets, server_modes=server_modes)
    cases, skipped = _apply_dataset_selection_filters(
        cases,
        skipped,
        dataset='toolathlon',
        selection_from_env=toolathlon_selection_from_env,
        select_case_ids=select_toolathlon_case_ids,
    )
    cases, skipped = _apply_dataset_selection_filters(
        cases,
        skipped,
        dataset='atlas',
        selection_from_env=atlas_selection_from_env,
        select_case_ids=select_atlas_case_ids,
    )
    cases, skipped = _apply_dataset_selection_filters(
        cases,
        skipped,
        dataset='universe',
        selection_from_env=universe_selection_from_env,
        select_case_ids=select_universe_case_ids,
    )
    if case_ids_filter is not None:
        cases = [case for case in cases if case.case_id in case_ids_filter]
        skipped = [item for item in skipped if item['case_id'] in case_ids_filter]

    if max_iterations_override is not None:
        for case in cases:
            case.max_iterations = max_iterations_override

    paged_cases = cases[skip:]
    if limit is not None:
        paged_cases = paged_cases[:limit]

    return paged_cases, skipped


def generate_configs() -> Iterator[BatchItem]:
    cases, skipped = _collect_filtered_cases()
    provider_name, provider_config, runtime_configs = _build_runtime_settings()
    openai_clients = _build_openai_clients(needs_judge=any(case.dataset.value == 'atlas' for case in cases))
    runner_timeout = float(os.getenv('MCP_BENCHMARK_RUN_TIMEOUT', '7200'))

    generated_metadata: list[dict[str, Any]] = []

    for case in cases:
        run_id = f'{case.dataset.value}-{case.server_mode.value}-{case.case_id}'
        generated_metadata.append({
            'dataset': case.dataset.value,
            'case_id': case.case_id,
            'server_mode': case.server_mode.value,
        })
        config = RunInput(
            imports=['nexus.extensions.tasks.mcp_benchmark', 'nexus.extensions.agents.main'],
            task_name='mcp_benchmark',
            task_config=case.to_task_config(),
            runtime_provider_name=provider_name,
            runtime_provider_config=provider_config,
            runtime_configs=runtime_configs,
            agent_name='main_agent',
            openai_clients=openai_clients,
            processors={'agent_metrics': ProcessorConfig(name='main_agent_metrics')},
            runner=RunnerConfig(timeout=runner_timeout),
        )
        metadata = {
            'dataset': case.dataset.value,
            'case_id': case.case_id,
            'server_mode': case.server_mode.value,
            'server_names': case.metadata.get('server_names', []),
            'category': case.metadata.get('category'),
        }
        if case.dataset.value != 'toolathlon' and case.metadata.get('source_path'):
            metadata['source_path'] = case.metadata.get('source_path')
        yield BatchItem(
            config=config,
            run_id=run_id,
            metadata=metadata,
        )

    _write_generation_report(
        os.getenv('MCP_BENCHMARK_GENERATION_REPORT'), generated=generated_metadata, skipped=skipped
    )


def get_total_count() -> int:
    cases, _ = _collect_filtered_cases()
    return len(cases)


def get_batch_metadata() -> dict[str, Any]:
    cases, skipped = _collect_filtered_cases()
    return {
        'task_type': 'mcp_benchmark',
        'generated_count': len(cases),
        'skipped_count': len(skipped),
        'datasets': os.getenv('MCP_BENCHMARK_DATASETS', 'toolathlon,atlas,universe'),
        'server_modes': _server_modes_env(_datasets_env()),
        'toolathlon_mode': os.getenv(TOOLATHLON_MODE_ENV),
        'atlas_mode': os.getenv(ATLAS_MODE_ENV),
        'universe_mode': os.getenv(UNIVERSE_MODE_ENV),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
    }
