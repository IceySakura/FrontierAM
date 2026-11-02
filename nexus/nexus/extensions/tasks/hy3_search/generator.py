#!/usr/bin/env python3
"""HY3 Search task batch generator for `nexus run-batch`."""

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

import pandas as pd

from nexus import BatchItem
from nexus.extensions.tasks.hy3_search.utils import (
    _build_metadata,
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
# Environment Configuration Helpers
# ============================================================================


def _env_int(name: str, default: int | None = None) -> int | None:
    """Get integer from environment variable with fallback."""
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float | None = None) -> float | None:
    """Get float from environment variable with fallback."""
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    """Get boolean from environment variable with fallback."""
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
    """Get thinking type from environment variable with strict validation."""
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {'enabled', 'disabled'}:
        raise ValueError(f"Environment variable '{name}' must be 'enabled' or 'disabled', got: {value}")
    return normalized


def _require_env(name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set. Please set it before running the generator."
        )
    return value


def _parse_instance_ids_filter() -> set[str] | None:
    """Parse instance-id filters from environment variables."""
    ids_raw = os.getenv('SEARCH_INSTANCE_IDS')
    if not ids_raw:
        return None

    tokens = [token.strip() for token in re.split(r'[\s,]+', ids_raw) if token.strip()]
    ids_set = {token for token in tokens if token}
    return ids_set or None


def _build_main_clients() -> dict[str, Any]:
    """Build main clients configuration from environment variables."""

    name = _require_env('SEARCH_OPENAI_CLIENTS_NAME')
    base_url = _require_env('OPENAPI_URL')
    api_key = _require_env('HY_OPENAPI_KEY')
    model = _require_env('SEARCH_OPENAI_MODEL')
    max_retries_raw = os.getenv('SEARCH_AGENT_MAX_RETRIES')
    client_args = {'base_url': base_url, 'api_key': api_key}
    if max_retries_raw:
        client_args['max_retries'] = int(max_retries_raw)
        logger.debug(f'  - max_retries configured {max_retries_raw}')

    request_args: dict[str, Any] = {
        'model': model,
        'extra_headers': {'X-route-env': os.getenv('OPENAPI_ROUTE_ENV')},
        'reasoning': {
            'effort': os.getenv('SEARCH_REASONING_EFFORT'),
            'interleaved_thinking': _env_bool('SEARCH_INTERLEAVED_THINKING'),
        },
        'temperature': _env_float('SEARCH_TEMPERATURE'),
        'max_tokens': _env_int('SEARCH_MAX_TOKENS'),
    }
    if _env_bool('DISABLE_CONTENT_FILTER'):
        request_args['extra_body'] = {'advanced_options': {'content_filter_options': {'disable': True}}}
    logger.debug(
        f'Building client config : name={name}, model={model}, base_url={base_url}, client_args={client_args}, request_args={request_args}'
    )
    return {
        'name': name,
        'config': {
            'client_args': client_args,
            'request_args': request_args,
        },
    }


def _build_openai_clients(eval_mode: str) -> dict[str, dict[str, Any]]:
    """Build OpenAI clients configuration from environment variables.

    Creates clients for search task:
    - main: for agent execution (required)
    - judge: for LLM-based evaluation (only required for llm_judge mode)

    Note: visit tool's extractor LLM uses a custom API (not OpenAI-compatible),
    configured via DEFAULT_JINA_MODEL_* env vars or tool_configs.

    Environment variable prefixes:
    - SEARCH_OPENAI_*: main client
    - JUDGE_OPENAI_*: judge client

    Args:
        eval_mode: Evaluation mode ('exact_match', 'llm_judge', or 'reward_service')

    Returns:
        Dict with client configurations

    Raises:
        ValueError: If required environment variables are missing
    """
    logger.debug(f'Building OpenAI clients for eval_mode={eval_mode}')

    clients = {
        'main': _build_main_clients(),
    }

    # Judge client is only required for llm_judge mode
    if eval_mode == 'llm_judge':
        logger.debug('Judge client required for llm_judge mode')
        clients['judge'] = _build_client_config('JUDGE_OPENAI')
    elif os.getenv('JUDGE_OPENAI_CLIENTS_NAME'):
        # Optional: add judge client if configured even in other modes
        logger.debug('Judge client configured but not required for this eval_mode, adding anyway')
        try:
            clients['judge'] = _build_client_config('JUDGE_OPENAI')
        except ValueError as e:
            logger.debug(f'Judge client config incomplete, skipping: {e}')

    logger.debug(f'OpenAI clients configured: {list(clients.keys())}')
    return clients


def _parse_run_ids_filter() -> set[str] | None:
    """Parse run-id filters from environment variables."""
    ids_raw = os.getenv('SEARCH_RUN_IDS')
    if not ids_raw:
        return None

    tokens = [token.strip() for token in re.split(r'[\s,]+', ids_raw) if token.strip()]
    ids_set = {token for token in tokens if token}
    return ids_set or None


# ============================================================================
# Configuration Builders
# ============================================================================


def _build_client_config(
    env_prefix: str,
    request_args_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single OpenAI client config from prefixed env vars.

    Expects: {PREFIX}_CLIENTS_NAME, {PREFIX}_BASE_URL, {PREFIX}_API_KEY, {PREFIX}_MODEL
    Optional: {PREFIX}_WSID

    Args:
        env_prefix: Environment variable prefix (e.g. 'SEARCH_OPENAI', 'JUDGE_OPENAI', 'EXTRACTOR_OPENAI')

    Returns:
        Client configuration dict with 'name' and 'config' keys

    Raises:
        ValueError: If required env vars are missing
    """
    name = _require_env(f'{env_prefix}_CLIENTS_NAME')
    base_url = _require_env(f'{env_prefix}_BASE_URL')
    api_key = _require_env(f'{env_prefix}_API_KEY')
    model = _require_env(f'{env_prefix}_MODEL')
    wsid = os.getenv(f'{env_prefix}_WSID')
    max_retries_raw = os.getenv(f'{env_prefix}_MAX_RETRIES')

    logger.debug(f'Building client config [{env_prefix}]: name={name}, model={model}, base_url={base_url}')

    client_args = {'base_url': base_url, 'api_key': api_key}
    if wsid:
        client_args['wsid'] = wsid
        logger.debug(f'  - wsid configured for {env_prefix}')
    if max_retries_raw:
        client_args['max_retries'] = int(max_retries_raw)
        logger.debug(f'  - max_retries configured for {env_prefix}: {max_retries_raw}')

    request_args: dict[str, Any] = {'model': model}
    if request_args_overrides:
        request_args.update(request_args_overrides)

    # If client is Infer Proxy, add chat template kwargs.
    if name == 'infer_proxy':
        extra_body = request_args.get('extra_body', {})
        reasoning_effort = extra_body.get('reasoning_effort', 'high')
        interleaved_thinking = extra_body.get('interleaved_thinking', True)
        extra_body['chat_template_kwargs'] = {
            'reasoning_effort': reasoning_effort,
            'interleaved_thinking': interleaved_thinking,
        }

    return {
        'name': name,
        'config': {
            'client_args': client_args,
            'request_args': request_args,
        },
    }


def _build_tool_configs() -> dict[str, Any]:
    """Build tool configurations from environment variables.

    Supports:
    - google_serper_search: Requires Serper API credentials (SEARCH_SERPER_*)
    - visit: Requires Jina API credentials (DEFAULT_JINA_*)

    Returns:
        Dict with tool configurations

    Raises:
        ValueError: If required tool credentials are missing
    """
    configs: dict[str, Any] = {}

    # Google Search (Serper)
    serper_app_id = _require_env('SEARCH_SERPER_APP_ID')
    serper_app_key = _require_env('SEARCH_SERPER_APP_KEY')
    serper_url = os.getenv('SEARCH_SERPER_URL', 'http://trpc-gpt-eval.production.polaris:8080/search')
    configs['google_serper_search'] = {
        'serper_url': serper_url,
        'serper_app_id': serper_app_id,
        'serper_app_key': serper_app_key,
    }

    # Visit tool (Jina) - optional
    # Jina API for webpage reading
    jina_app_id = os.getenv('DEFAULT_JINA_APP_ID')
    jina_app_key = os.getenv('DEFAULT_JINA_APP_KEY')
    jina_url = os.getenv('DEFAULT_JINA_URL', 'http://trpc-gpt-eval.production.polaris:8080/api/jina_ai/search')
    # Extractor LLM for content extraction (custom API, not OpenAI-compatible)
    jina_model_base_url = os.getenv(
        'DEFAULT_JINA_MODEL_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval'
    )
    jina_model = os.getenv('DEFAULT_JINA_MODEL', 'api_doubao_DeepSeek-V3.1-250821')
    jina_model_app_id = os.getenv('DEFAULT_JINA_MODEL_APP_ID')
    jina_model_app_key = os.getenv('DEFAULT_JINA_MODEL_APP_KEY')

    if jina_app_id and jina_app_key:
        visit_config = {
            'jina_url': jina_url,
            'jina_app_id': jina_app_id,
            'jina_app_key': jina_app_key,
            'jina_model_base_url': jina_model_base_url,
            'jina_model': jina_model,
            'jina_model_app_id': jina_model_app_id,
            'jina_model_app_key': jina_model_app_key,
        }
        configs['visit'] = visit_config

    return configs


def _build_processors() -> dict[str, Any]:
    """Build span processors configuration from environment variables.

    chat_tracker is always enabled.
    JsonLineSpanProcessor is enabled when SEARCH_TRACE_FILE is set.

    Returns:
        Dict of processor configurations keyed by processor name
    """
    processors: dict[str, Any] = {
        'chat': {'name': 'chat_tracker'},
    }

    trace_file = os.getenv('SEARCH_TRACE_FILE')
    if trace_file:
        processors['trace'] = {
            'name': 'json_line',
            'args': {'file_path': trace_file},
        }
        logger.debug(f'Trace processor configured: {trace_file}')

    return processors


def _build_runtime_provider_config() -> tuple[str | None, dict[str, Any]]:
    """Build runtime provider configuration from environment variables.

    Runtime provider is optional for search task. Only include if explicitly configured.

    Required environment variables:
        SEARCH_RUNTIME_PROVIDER_NAME: Provider name (e.g., 'gongfeng')
        SEARCH_RUNTIME_BASE_URL: API base URL (e.g., 'http://hyrl-sandbox.prod.woa.com:8052')
        SEARCH_RUNTIME_IMAGE: Main container image (e.g., 'mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155')
        SEARCH_RUNTIME_RUNTIME_IMAGE: Runtime image with /nix environment (e.g., 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest')

    Optional environment variables:
        SEARCH_RUNTIME_TIMEOUT: Request timeout in seconds (default: 1800.0)
        SEARCH_GONGFENG_TOKEN: Authentication token for Gongfeng sandbox API (required for gongfeng provider)

    Returns:
        Tuple of (provider_name, provider_config)
        Returns (None, {}) if no runtime provider is configured
    """
    runtime_provider_name = os.getenv('SEARCH_RUNTIME_PROVIDER_NAME')

    if not runtime_provider_name:
        return None, {}

    runtime_base_url = _require_env('SEARCH_RUNTIME_BASE_URL')
    runtime_image = _require_env('SEARCH_RUNTIME_IMAGE')
    runtime_runtime_image = _require_env('SEARCH_RUNTIME_RUNTIME_IMAGE')
    runtime_timeout = _env_float('SEARCH_RUNTIME_TIMEOUT', 2400.0) or 2400.0

    provider_config = {
        'api_base_url': runtime_base_url,
        'image': runtime_image,
        'runtime_image': runtime_runtime_image,
        'request_timeout': runtime_timeout,
    }

    # Add token for gongfeng provider (required for authentication)
    if runtime_provider_name == 'gongfeng':
        token = _require_env('SEARCH_GONGFENG_TOKEN')
        provider_config['token'] = token
        logger.debug('Gongfeng token configured from SEARCH_GONGFENG_TOKEN')

    return runtime_provider_name, provider_config


# ============================================================================
# Main Generator Functions
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for search task from Parquet dataset.

    Reads the dataset, validates required fields, and yields BatchItem instances
    configured for the search task. Automatically handles pagination via SEARCH_SKIP
    and SEARCH_LIMIT environment variables.

    Yields:
        BatchItem: Configured batch items ready for execution

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is missing required fields or environment variables
    """

    # ========================================================================
    # Load Environment Configuration
    # ========================================================================

    dataset_path = _require_env('SEARCH_DATASET_PATH')
    skip = _env_int('SEARCH_SKIP', 0) or 0
    limit = _env_int('SEARCH_LIMIT')
    selected_instance_ids = _parse_instance_ids_filter()
    selected_run_ids = _parse_run_ids_filter()

    # Evaluation configuration
    eval_mode = os.getenv('SEARCH_EVAL_MODE', 'exact_match')
    judge_use_tool_calling = os.getenv('JUDGE_USE_TOOL_CALLING')
    enable_summary = os.getenv('SEARCH_ENABLE_SUMMARY')
    run_oracle = _env_bool('SEARCH_RUN_ORACLE')
    logger.debug(f'Eval mode: {eval_mode}')
    if run_oracle:
        logger.info(
            'Oracle mode enabled: agent will be bypassed; non-wide eval uses exact_match '
            '(score is typically 1.0 when prediction equals reference)'
        )

    openai_clients = _build_openai_clients(eval_mode)

    # Build shared configurations
    # openai_clients = _build_openai_clients(eval_mode)
    # tool_configs = _build_tool_configs()
    # logger.info(f'Tool configs built: {list(tool_configs.keys())}')
    processors = _build_processors()

    runtime_provider_name, runtime_provider_config = _build_runtime_provider_config()
    if runtime_provider_name:
        logger.debug(f'Runtime provider configured: {runtime_provider_name}')

    reward_service_configured = bool(
        os.getenv('REWARD_SERVICE_URL') and os.getenv('REWARD_SERVICE_TOKEN') and os.getenv('REWARD_SERVICE_ROUTER_ID')
    )
    if reward_service_configured:
        logger.debug(f'Reward service configured: router_id={os.getenv("REWARD_SERVICE_ROUTER_ID")}')

    # ========================================================================
    # Load and Validate Dataset
    # ========================================================================

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    logger.debug(f'Loading dataset from: {dataset_path}')
    df = pd.read_parquet(dataset_file)

    if 'nexus' not in df.columns:
        raise ValueError("Dataset must contain 'nexus' column with task configurations")

    # Apply pagination (efficient slicing)
    df = df.iloc[skip : skip + limit] if limit is not None else df.iloc[skip:]

    logger.debug(
        f'Processing {len(df)} items from dataset (skip={skip}, limit={limit if limit is not None else "all"})'
    )
    if selected_instance_ids is not None:
        logger.debug('Instance ID filter enabled, target count=%d', len(selected_instance_ids))
    if selected_run_ids is not None:
        logger.debug('Run ID filter enabled, target count=%d', len(selected_run_ids))

    # ========================================================================
    # Generate Batch Items
    # ========================================================================

    skipped_count = 0
    filtered_count = 0
    generated_count = 0

    for idx, row in df.iterrows():
        nexus_config = _parse_nexus_config(row['nexus'])

        # Validate required fields
        task_name = nexus_config.get('task_name')
        if not task_name:
            logger.warning(f'Skipping row {idx}: missing task_name')
            skipped_count += 1
            continue

        if task_name not in ('search', 'hy3_search'):
            logger.warning(f'Skipping row {idx}: task_name is "{task_name}", expected "search" or "hy3_search"')
            skipped_count += 1
            continue

        task_name = 'hy3_search'

        # Extract configuration with sensible defaults
        task_config = nexus_config.get('task_config', {})
        if 'history' not in task_config:
            task_config['history'] = []
        task_instance_id = task_config.get('instance_id')

        if selected_instance_ids is not None:
            if not task_instance_id:
                logger.warning(f'Skipping row {idx}: missing task_config.instance_id while ID filter is enabled')
                skipped_count += 1
                continue
            if task_instance_id not in selected_instance_ids:
                filtered_count += 1
                continue

        # Validate required task_config fields
        if 'question' not in task_config or 'answer' not in task_config:
            logger.warning(f'Skipping row {idx}: missing question or answer in task_config')
            skipped_count += 1
            continue

        # Add evaluation mode (env var takes priority over dataset config)
        if eval_mode:
            task_config['eval_mode'] = eval_mode
        if judge_use_tool_calling is not None:
            task_config['judge_use_tool_calling'] = judge_use_tool_calling.lower() in ('true', '1', 'yes')
        if enable_summary is not None:
            task_config['enable_summary'] = enable_summary.lower() in ('true', '1', 'yes')
        if run_oracle is not None:
            task_config['run_oracle'] = run_oracle

        # Validate reward service env vars are configured (config resolved at runtime by evaluator)
        if eval_mode == 'reward_service' and not reward_service_configured:
            logger.warning(
                f'Skipping row {idx}: eval_mode is reward_service but REWARD_SERVICE_TOKEN '
                'and REWARD_SERVICE_ROUTER_ID are not configured'
            )
            skipped_count += 1
            continue
        # SEARCH_AGENT_NAME env var takes priority; fall back to dataset config, then default
        item_agent_name = os.getenv('SEARCH_AGENT_NAME')
        agent_config_base = nexus_config.get('agent_config', {})
        agent_config = {
            **agent_config_base,
        }

        logger.debug(f'task_config: {json.dumps(task_config)}')
        # Get imports (with safe parsing)
        imports = _safe_parse_imports(nexus_config.get('imports'))
        if not imports:
            # Default imports for search task
            imports = ['nexus.extensions']

        # Create RunInput configuration
        config = RunInput(
            imports=imports,
            task_name=task_name,
            task_config=task_config,
            tool_configs={},
            agent_name=item_agent_name,
            agent_config=agent_config,
            openai_clients=openai_clients,
            runtime_provider_name=runtime_provider_name,
            runtime_provider_config=runtime_provider_config,
            processors=processors,
        )

        # Generate unique run identifier
        instance_id = task_instance_id or f'search_{idx:05d}'
        if selected_run_ids is not None and instance_id not in selected_run_ids:
            filtered_count += 1
            continue

        # Build metadata for tracking and analysis
        metadata = _build_metadata(instance_id, task_name, item_agent_name, task_config)

        yield BatchItem(
            config=config,
            run_id=instance_id,
            metadata=metadata,
        )
        generated_count += 1

    # Log statistics
    if skipped_count > 0:
        logger.warning(f'Skipped {skipped_count} items due to validation errors')
    if filtered_count > 0:
        logger.debug(f'Filtered out {filtered_count} items not in selected instance IDs')

    # Warn if no items were generated
    if generated_count == 0:
        logger.warning(
            'generate_configs() completed but generated 0 items. '
            'Check dataset path, skip/limit parameters, and data validation.'
        )
    else:
        logger.info(f'generate_configs() completed successfully - generated {generated_count} items')
        # Log with args to avoid TrimFormatter truncating long f-string messages.
        logger.debug('Example config for first generated item: %s', config.model_dump())


def get_total_count() -> int:
    """Calculate total number of instances to be generated.

    Takes into account SEARCH_SKIP and SEARCH_LIMIT environment variables
    to return the actual number of items that will be processed.

    Returns:
        Total count of instances that will be generated, or 0 if dataset doesn't exist
        or cannot be read
    """

    try:
        dataset_path = _require_env('SEARCH_DATASET_PATH')
    except ValueError:
        return 0

    skip = _env_int('SEARCH_SKIP', 0) or 0
    limit = _env_int('SEARCH_LIMIT')
    selected_instance_ids = _parse_instance_ids_filter()
    selected_run_ids = _parse_run_ids_filter()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return 0

    try:
        df = pd.read_parquet(dataset_file)
        df = df.iloc[skip : skip + limit] if limit is not None else df.iloc[skip:]
        if selected_instance_ids is None:
            if selected_run_ids is None:
                return len(df)
        elif selected_run_ids is None:
            count = 0
            for _, row in df.iterrows():
                nexus_config = _parse_nexus_config(row['nexus'])
                task_instance_id = (nexus_config.get('task_config') or {}).get('instance_id')
                if task_instance_id in selected_instance_ids:
                    count += 1
            return count

        count = 0
        for idx, row in df.iterrows():
            nexus_config = _parse_nexus_config(row['nexus'])
            task_instance_id = (nexus_config.get('task_config') or {}).get('instance_id')
            if selected_instance_ids is not None and task_instance_id not in selected_instance_ids:
                continue
            run_id = f'search_{idx:05d}'
            if selected_run_ids is not None and run_id not in selected_run_ids:
                continue
            if (selected_instance_ids is not None) or (selected_run_ids is not None):
                count += 1
        return count
    except Exception:
        # Silently return 0 on any error (file corruption, permission, etc.)
        return 0


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata for tracking and debugging.

    Gathers information about the dataset, runtime configuration, and environment.
    This metadata is stored in the batch database and useful for:
    - Tracking which dataset version was used
    - Reproducing runs with same configuration
    - Debugging and analysis

    Returns:
        Dictionary containing comprehensive batch configuration and environment info
    """

    try:
        dataset_path = _require_env('SEARCH_DATASET_PATH')
    except ValueError:
        dataset_path = 'NOT_SET'

    # Core configuration
    metadata: dict[str, Any] = {
        'task_type': 'hy3_search',
        'dataset_path': dataset_path,
        'agent_name': os.getenv('SEARCH_AGENT_NAME'),
        'skip': _env_int('SEARCH_SKIP', 0) or 0,
        'limit': _env_int('SEARCH_LIMIT'),
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

    # Add runtime provider info if configured
    runtime_provider_name = os.getenv('SEARCH_RUNTIME_PROVIDER_NAME')
    if runtime_provider_name:
        metadata['runtime_provider'] = runtime_provider_name
        metadata['runtime_tag'] = os.getenv('SEARCH_RUNTIME_TAG', 'latest')
        metadata['runtime_base_url'] = os.getenv('SEARCH_RUNTIME_BASE_URL', 'http://hunyuan.sandbox.test.polaris:8052')

    # Enrich with dataset information if available
    if dataset_path != 'NOT_SET':
        dataset_file = Path(dataset_path)
        if dataset_file.exists():
            metadata['dataset_size_mb'] = round(dataset_file.stat().st_size / (1024 * 1024), 2)

            try:
                df = pd.read_parquet(dataset_file)
                metadata['dataset_rows'] = len(df)
                metadata['dataset_columns'] = list(df.columns)
            except Exception:
                # Don't fail if we can't read dataset metadata
                pass

    return metadata
