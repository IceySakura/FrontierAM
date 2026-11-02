#!/usr/bin/env python3
"""mm_agent task batch generator for `nexus run-batch`.

Philosophy
----------
This generator is intentionally *thin*: almost everything that describes a
single task (agent_name, agent_config, task_config, tool_configs,
runtime_provider_name, runtime_provider_config, runtime_configs, imports)
is read straight out of the dataset's ``nexus`` column. The generator only
owns concerns that are orthogonal to the dataset:

1. Which dataset to read + pagination / ID filtering.
2. Main LLM client (``openai_clients['main']``) construction from env vars.
3. Span processors (chat_tracker + optional trace file).

``${oc.env:...}`` interpolations inside the dataset are resolved by
``RunInput.from_dict`` (OmegaConf). Provide the referenced env vars in the
launch script (e.g. ``MAX_ITERATIONS``, ``GONGFENG_TOKEN``,
``PYTHON_PATH``, ``CODE_INTERPRETER_WORK_DIR``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import socket

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.mm_agent.utils import (
    _build_metadata,
    _is_aws_third_anthropic_model,
    _is_dashscope_ali_qwen_model,
    _is_litellm_azure_openai_responses_model,
    _resolve_openai_responses_reasoning_effort,
    _safe_parse_imports,
)
from nexus.runner.single import RunInput
from nexus.utils.common import serialize_object


logger = logging.getLogger(__name__)


def _parse_nexus_config(value: Any) -> Any:
    """Parse the nexus column value, handling both dict and JSON string formats."""
    result = serialize_object(value)
    if isinstance(result, str):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            result = json.loads(result)
    return result


# ============================================================================
# Dataset Loading Helpers (support parquet + jsonl, stream-friendly)
# ============================================================================

_JSONL_SUFFIXES = {'.jsonl', '.json'}
_PARQUET_SUFFIXES = {'.parquet', '.pq'}


def _detect_dataset_format(path: Path) -> str:
    """Return 'parquet' or 'jsonl' based on file suffix."""
    suffix = path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        return 'parquet'
    if suffix in _JSONL_SUFFIXES:
        return 'jsonl'
    raise ValueError(
        f'Unsupported dataset format: {suffix!r} ({path}). '
        f'Supported suffixes: {sorted(_PARQUET_SUFFIXES | _JSONL_SUFFIXES)}'
    )


def _import_pandas():
    """Lazy-import pandas so JSONL-only runs don't require it.

    Raises a clear actionable error if pandas is missing but the user is
    trying to read a parquet dataset.
    """
    try:
        import pandas as pd  # noqa: PLC0415 - intentional lazy import
    except ImportError as e:
        raise ImportError(
            'pandas is required to read parquet datasets. '
            'Install it with `uv pip install pandas pyarrow` (or switch the dataset to .jsonl).'
        ) from e
    return pd


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows from a JSONL file without loading the whole file into memory."""
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'Skipping malformed JSONL row at {path}:{lineno}: {e}')


def _iter_dataset_rows(path: Path, skip: int, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (row_index, row_dict) from parquet or jsonl, honoring skip/limit."""
    fmt = _detect_dataset_format(path)
    yielded = 0
    if fmt == 'parquet':
        pd = _import_pandas()
        df = pd.read_parquet(path)
        sliced = df.iloc[skip : skip + limit] if limit is not None else df.iloc[skip:]
        for idx, row in sliced.iterrows():
            yield int(idx), row.to_dict()
            yielded += 1
        return

    # jsonl: stream
    for idx, row in enumerate(_iter_jsonl_rows(path)):
        if idx < skip:
            continue
        if limit is not None and yielded >= limit:
            break
        yield idx, row
        yielded += 1


def _count_dataset_rows(path: Path, skip: int, limit: int | None) -> int:
    """Count rows effectively processed, honoring skip/limit. Streams jsonl."""
    fmt = _detect_dataset_format(path)
    if fmt == 'parquet':
        pd = _import_pandas()
        df = pd.read_parquet(path)
        df = df.iloc[skip : skip + limit] if limit is not None else df.iloc[skip:]
        return len(df)
    total = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total += 1
    effective = max(0, total - skip)
    if limit is not None:
        effective = min(effective, limit)
    return effective


# ============================================================================
# Environment Helpers
# ============================================================================


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f"Environment variable '{name}' must be a boolean value, got: {value}")


def _env_thinking_type(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {'enabled', 'disabled'}:
        raise ValueError(f"Environment variable '{name}' must be 'enabled' or 'disabled', got: {value}")
    return normalized


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set. Please set it before running the generator."
        )
    return value


def _parse_instance_ids_filter() -> set[str] | None:
    ids_raw = os.getenv('INSTANCE_IDS')
    if not ids_raw:
        return None
    tokens = [token.strip() for token in re.split(r'[\s,]+', ids_raw) if token.strip()]
    return {token for token in tokens if token} or None


def _parse_run_ids_filter() -> set[str] | None:
    ids_raw = os.getenv('RUN_IDS')
    if not ids_raw:
        return None
    tokens = [token.strip() for token in re.split(r'[\s,]+', ids_raw) if token.strip()]
    return {token for token in tokens if token} or None


def _parse_json_override(env_name: str) -> dict[str, Any]:
    """Parse a JSON object from env var ``env_name``.

    Returns an empty dict if the variable is unset/empty. Raises ValueError if
    the value is not valid JSON or does not decode to a dict.
    """
    raw = os.getenv(env_name)
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Environment variable '{env_name}' must be valid JSON, got: {raw!r} ({e})") from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Environment variable '{env_name}' must decode to a JSON object (dict), got {type(parsed).__name__}"
        )
    return parsed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    - Nested dicts are merged key-by-key.
    - Non-dict values in ``override`` replace the value in ``base``.
    - ``base`` is not mutated; a new dict is returned.
    """
    if not override:
        return dict(base)
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


# ============================================================================
# Main LLM Client Builder (the ONE thing generator still owns)
# ============================================================================


def _build_main_client() -> dict[str, dict[str, Any]]:
    """Build the ``main`` OpenAI client config from task-agnostic env vars.

    Required:
        - ``CLIENTS_NAME`` (e.g. ``taiji`` / ``openai`` / ``infer_proxy``)
        - ``MODEL``
    Optional:
        - ``WSID``, ``OPENAI_MAX_RETRIES``
        - ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` (required when name != 'taiji')

    This is the sole client generator constructs. Judge/extractor/etc. clients
    are either not needed here (reward routing is now dataset-driven via
    ``reward_func``) or expected to come from dataset ``tool_configs`` /
    reward-system plumbing.
    """
    name = _require_env('CLIENTS_NAME')
    model = _require_env('MODEL')
    wsid = os.getenv('WSID')
    max_retries_raw = os.getenv('OPENAI_MAX_RETRIES')

    if name == 'taiji':
        client_args: dict[str, Any] = {'model': model}
    else:
        base_url = _require_env('OPENAI_BASE_URL')
        api_key = _require_env('OPENAI_API_KEY')
        logger.debug(f'Building main client: name={name}, model={model}, base_url={base_url}')
        client_args: dict[str, Any] = {'base_url': base_url, 'api_key': api_key}

    if wsid:
        client_args['wsid'] = wsid
    if max_retries_raw:
        client_args['max_retries'] = int(max_retries_raw)
        logger.debug(f'main client max_retries: {max_retries_raw}')

    temperature = _env_float('TEMPERATURE')
    top_p = _env_float('TOP_P')
    top_k = _env_int('TOP_K')
    max_tokens = _env_int('MAX_TOKENS', 32768)
    reasoning_effort = os.getenv('REASONING_EFFORT')
    interleaved_thinking = _env_bool('INTERLEAVED_THINKING')
    thinking_type = _env_thinking_type('THINKING_TYPE')

    request_args: dict[str, Any] = {'model': model}
    if temperature is not None:
        request_args['temperature'] = temperature
    if top_p is not None:
        request_args['top_p'] = top_p
    if max_tokens is not None:
        request_args['max_tokens'] = max_tokens
    # reasoning_effort is a first-class parameter for both OpenAI responses API
    # and Taiji (TaijiClient reads it from request_args as well).
    if reasoning_effort is not None:
        request_args['reasoning_effort'] = reasoning_effort

    effective_thinking_enabled = False
    if interleaved_thinking is not None:
        effective_thinking_enabled = interleaved_thinking
    elif thinking_type is not None:
        effective_thinking_enabled = thinking_type == 'enabled'

    extra_body: dict[str, Any] = {}
    # top_k is not a standard OpenAI param; pass through extra_body so that
    # vLLM / Taiji / other OpenAI-compatible backends can pick it up.
    if top_k is not None:
        extra_body['top_k'] = top_k
    if reasoning_effort is not None:
        extra_body['reasoning_effort'] = reasoning_effort
    if thinking_type is not None:
        extra_body['thinking'] = {'type': thinking_type}
    request_args['extra_body'] = extra_body

    if _is_aws_third_anthropic_model(model):
        if reasoning_effort:
            request_args['extra_body']['reasoning_effort'] = reasoning_effort
        if temperature is not None:
            logger.debug('Ignoring TEMPERATURE for aws_third_anthropic model')
            request_args.pop('temperature', None)
        if interleaved_thinking is not None:
            request_args['extra_headers'] = {'anthropic-beta': 'interleaved-thinking-2025-05-14'}
    elif _is_dashscope_ali_qwen_model(model):
        if reasoning_effort:
            logger.debug(
                'Ignoring REASONING_EFFORT for DashScope ali_qwen model; '
                'use THINKING_TYPE or INTERLEAVED_THINKING to control enable_thinking'
            )
        request_args['extra_body'] = {'enable_thinking': effective_thinking_enabled}
    elif _is_litellm_azure_openai_responses_model(model):
        responses_reasoning_effort = _resolve_openai_responses_reasoning_effort(model, reasoning_effort)
        if responses_reasoning_effort is not None:
            request_args['reasoning_effort'] = responses_reasoning_effort
        request_args['drop_params'] = True
        if interleaved_thinking is not None:
            logger.debug('Ignoring INTERLEAVED_THINKING for responses model')

    # Infer proxy chat template kwargs pass-through.
    if name == 'infer_proxy':
        eb = request_args.get('extra_body') or {}
        eb['chat_template_kwargs'] = {
            'reasoning_effort': eb.get('reasoning_effort', 'high'),
            'interleaved_thinking': eb.get('interleaved_thinking', True),
        }
        request_args['extra_body'] = eb

    return {
        'main': {
            'name': name,
            'config': {
                'client_args': client_args,
                'request_args': request_args,
            },
        }
    }


def _build_processors() -> dict[str, Any]:
    """chat_tracker always on; optional json_line trace when TRACE_FILE is set."""
    processors: dict[str, Any] = {'chat': {'name': 'chat_tracker'}}
    trace_file = os.getenv('TRACE_FILE')
    if trace_file:
        processors['trace'] = {
            'name': 'json_line',
            'args': {'file_path': trace_file},
        }
        logger.debug(f'Trace processor configured: {trace_file}')
    return processors


# ============================================================================
# Main Generator
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for mm_agent task.

    Reads the dataset, validates required fields, and yields BatchItem
    instances. Each row's ``nexus`` column provides agent_name /
    agent_config / task_config / tool_configs / runtime_provider_* /
    runtime_configs / imports verbatim (with ``${oc.env:...}`` resolved via
    ``RunInput.from_dict``). The generator only injects the main OpenAI
    client and span processors.
    """

    # ------------------------------------------------------------------
    # Env-driven inputs (only what generator truly needs).
    # ------------------------------------------------------------------
    dataset_path = _require_env('DATASET_PATH')
    skip = _env_int('DATA_SKIP', 0) or 0
    limit = _env_int('DATA_LIMIT')
    selected_instance_ids = _parse_instance_ids_filter()
    selected_run_ids = _parse_run_ids_filter()

    openai_clients = _build_main_client()
    processors = _build_processors()

    # Optional per-run overrides (applied to every sample).
    task_config_override = _parse_json_override('TASK_CONFIG_OVERRIDE')
    agent_config_override = _parse_json_override('AGENT_CONFIG_OVERRIDE')
    tool_configs_override = _parse_json_override('TOOL_CONFIGS_OVERRIDE')
    if task_config_override:
        logger.debug(f'TASK_CONFIG_OVERRIDE applied: {task_config_override}')
    if agent_config_override:
        logger.debug(f'AGENT_CONFIG_OVERRIDE applied: {agent_config_override}')
    if tool_configs_override:
        logger.debug(f'TOOL_CONFIGS_OVERRIDE applied: {tool_configs_override}')

    # ------------------------------------------------------------------
    # Dataset loading.
    # ------------------------------------------------------------------
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    logger.debug(f'Loading dataset from: {dataset_path}')
    dataset_format = _detect_dataset_format(dataset_file)
    logger.debug(f'Detected dataset format: {dataset_format}')
    logger.debug(f'Processing items (skip={skip}, limit={limit if limit is not None else "all"})')
    if selected_instance_ids is not None:
        logger.debug('Instance ID filter enabled, target count=%d', len(selected_instance_ids))
    if selected_run_ids is not None:
        logger.debug('Run ID filter enabled, target count=%d', len(selected_run_ids))

    skipped_count = 0
    filtered_count = 0
    generated_count = 0

    for idx, row in _iter_dataset_rows(dataset_file, skip, limit):
        if 'nexus' not in row:
            raise ValueError("Dataset must contain 'nexus' column with task configurations")
        nexus_config = _parse_nexus_config(row['nexus'])
        if not isinstance(nexus_config, dict):
            logger.warning(f'Skipping row {idx}: nexus config is not a dict')
            skipped_count += 1
            continue

        task_name = nexus_config.get('task_name')
        if task_name != 'mm_agent':
            logger.warning(f'Skipping row {idx}: task_name is {task_name!r}, expected "mm_agent"')
            skipped_count += 1
            continue

        task_config = nexus_config.get('task_config') or {}
        task_instance_id = task_config.get('instance_id')

        if selected_instance_ids is not None and (
            not task_instance_id or task_instance_id not in selected_instance_ids
        ):
            filtered_count += 1
            continue

        if 'question' not in task_config or 'answer' not in task_config:
            logger.warning(f'Skipping row {idx}: missing question/answer in task_config')
            skipped_count += 1
            continue

        agent_name = nexus_config.get('agent_name')
        if not agent_name:
            logger.warning(f'Skipping row {idx}: missing agent_name')
            skipped_count += 1
            continue

        imports = _safe_parse_imports(nexus_config.get('imports')) or ['nexus.extensions']

        # Apply overrides (deep merge, override wins on conflicts).
        task_config = _deep_merge(task_config, task_config_override)
        agent_config = _deep_merge(nexus_config.get('agent_config') or {}, agent_config_override)
        tool_configs = _deep_merge(nexus_config.get('tool_configs') or {}, tool_configs_override)

        # Build the RunInput config dict; let OmegaConf resolve env refs.
        run_config_dict: dict[str, Any] = {
            'imports': imports,
            'task_name': task_name,
            'task_config': task_config,
            'tool_configs': tool_configs,
            'agent_name': agent_name,
            'agent_config': agent_config,
            'openai_clients': openai_clients,
            'runtime_provider_name': nexus_config.get('runtime_provider_name'),
            'runtime_provider_config': nexus_config.get('runtime_provider_config') or {},
            'runtime_configs': nexus_config.get('runtime_configs') or {},
            'processors': processors,
        }

        try:
            config = RunInput.from_dict(run_config_dict)
        except Exception as e:
            logger.warning(f'Skipping row {idx}: failed to build RunInput ({e})')
            skipped_count += 1
            continue

        instance_id = task_instance_id or f'mm_agent_{idx:05d}'
        if selected_run_ids is not None and instance_id not in selected_run_ids:
            filtered_count += 1
            continue

        metadata = _build_metadata(instance_id, task_name, agent_name, task_config)

        yield BatchItem(
            config=config,
            run_id=instance_id,
            metadata=metadata,
        )
        generated_count += 1

    if skipped_count > 0:
        logger.warning(f'Skipped {skipped_count} items due to validation errors')
    if filtered_count > 0:
        logger.debug(f'Filtered out {filtered_count} items not in selected ID set')
    if generated_count == 0:
        logger.warning(
            'generate_configs() completed but generated 0 items. Check dataset path, skip/limit, and data validation.'
        )
    else:
        logger.info(f'generate_configs() completed: generated {generated_count} items')


def get_total_count() -> int:
    """Return total instances that will be generated, honoring skip/limit/filters."""
    try:
        dataset_path = _require_env('DATASET_PATH')
    except ValueError:
        return 0

    skip = _env_int('DATA_SKIP', 0) or 0
    limit = _env_int('DATA_LIMIT')
    selected_instance_ids = _parse_instance_ids_filter()
    selected_run_ids = _parse_run_ids_filter()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return 0

    try:
        if selected_instance_ids is None and selected_run_ids is None:
            return _count_dataset_rows(dataset_file, skip, limit)

        count = 0
        for idx, row in _iter_dataset_rows(dataset_file, skip, limit):
            nexus_config = _parse_nexus_config(row.get('nexus'))
            if not isinstance(nexus_config, dict):
                continue
            task_instance_id = (nexus_config.get('task_config') or {}).get('instance_id')
            if selected_instance_ids is not None and task_instance_id not in selected_instance_ids:
                continue
            run_id = task_instance_id or f'mm_agent_{idx:05d}'
            if selected_run_ids is not None and run_id not in selected_run_ids:
                continue
            count += 1
        return count
    except Exception:
        return 0


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata for tracking/debugging."""
    try:
        dataset_path = _require_env('DATASET_PATH')
    except ValueError:
        dataset_path = 'NOT_SET'

    metadata: dict[str, Any] = {
        'task_type': 'mm_agent',
        'dataset_path': dataset_path,
        'skip': _env_int('DATA_SKIP', 0) or 0,
        'limit': _env_int('DATA_LIMIT'),
        'model': os.getenv('MODEL'),
        'clients_name': os.getenv('CLIENTS_NAME'),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
    }

    selected_instance_ids = _parse_instance_ids_filter()
    if selected_instance_ids is not None:
        metadata['instance_id_filter_enabled'] = True
        metadata['instance_id_filter_count'] = len(selected_instance_ids)
    selected_run_ids = _parse_run_ids_filter()
    if selected_run_ids is not None:
        metadata['run_id_filter_enabled'] = True
        metadata['run_id_filter_count'] = len(selected_run_ids)

    if dataset_path != 'NOT_SET':
        dataset_file = Path(dataset_path)
        if dataset_file.exists():
            metadata['dataset_size_mb'] = round(dataset_file.stat().st_size / (1024 * 1024), 2)
            try:
                fmt = _detect_dataset_format(dataset_file)
                metadata['dataset_format'] = fmt
                if fmt == 'parquet':
                    pd = _import_pandas()
                    df = pd.read_parquet(dataset_file)
                    metadata['dataset_rows'] = len(df)
                    metadata['dataset_columns'] = list(df.columns)
                else:
                    rows = 0
                    columns: list[str] = []
                    with dataset_file.open('r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            if rows == 0:
                                with contextlib.suppress(json.JSONDecodeError, ValueError):
                                    columns = list(json.loads(line).keys())
                            rows += 1
                    metadata['dataset_rows'] = rows
                    metadata['dataset_columns'] = columns
            except Exception:
                pass

    return metadata
