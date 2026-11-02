#!/usr/bin/env python3
"""Search task batch generator for `nexus run-batch`."""

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
from nexus.extensions.agents.search_agent.agent import TOOLSET_REQUIRED_TOOLS
from nexus.extensions.tasks.search.config import (
    SearchConfig,
    load_search_config_if_exists,
)
from nexus.extensions.tasks.search.task import WIDESEARCH_EVALUATION_REQUIRED_KEYS
from nexus.extensions.tasks.search.utils import (
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

    # Infer Proxy expects reasoning/interleaved flags only under chat_template_kwargs,
    # not as top-level extra_body keys (avoids duplicate reasoning_effort / thinking).
    if name == 'infer_proxy':
        raw_extra = dict(request_args.get('extra_body') or {})
        reasoning_effort = raw_extra.get('reasoning_effort', 'high')
        interleaved_thinking = raw_extra.get('interleaved_thinking')
        if interleaved_thinking is None:
            interleaved_thinking = _env_bool('SEARCH_INTERLEAVED_THINKING')
        if interleaved_thinking is None:
            interleaved_thinking = True
        request_args['extra_body'] = {
            'chat_template_kwargs': {
                'reasoning_effort': reasoning_effort,
                'interleaved_thinking': interleaved_thinking,
            }
        }

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

    main_model = _require_env('SEARCH_OPENAI_MODEL')
    main_temperature = _env_float('SEARCH_TEMPERATURE')
    main_max_tokens = _env_int('SEARCH_MAX_TOKENS', 32768)
    main_reasoning_effort = os.getenv('SEARCH_REASONING_EFFORT')
    main_interleaved_thinking = _env_bool('SEARCH_INTERLEAVED_THINKING')
    main_thinking_type = _env_thinking_type('SEARCH_THINKING_TYPE')

    main_request_args = _build_main_request_args(
        main_model=main_model,
        main_temperature=main_temperature,
        main_max_tokens=main_max_tokens,
        main_reasoning_effort=main_reasoning_effort,
        main_interleaved_thinking=main_interleaved_thinking,
        main_thinking_type=main_thinking_type,
    )

    clients = {
        'main': _build_client_config('SEARCH_OPENAI', request_args_overrides=main_request_args),
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


def _build_main_request_args(
    main_model: str,
    main_temperature: float | None,
    main_max_tokens: int | None,
    main_reasoning_effort: str | None,
    main_interleaved_thinking: bool | None,
    main_thinking_type: str | None,
) -> dict[str, Any]:
    """Build request_args for SEARCH_OPENAI main client."""
    main_request_args: dict[str, Any] = {}
    if main_temperature is not None:
        main_request_args['temperature'] = main_temperature
    if main_max_tokens is not None:
        main_request_args['max_tokens'] = main_max_tokens

    # Thinking / Reasoning mode logic:
    effective_thinking_enabled = False
    if main_interleaved_thinking is not None:
        effective_thinking_enabled = main_interleaved_thinking
    elif main_thinking_type is not None:
        effective_thinking_enabled = main_thinking_type == 'enabled'

    main_request_args['extra_body'] = {}
    if main_reasoning_effort is not None:
        main_request_args['extra_body']['reasoning_effort'] = main_reasoning_effort

    if main_thinking_type is not None:
        thinking_config: dict[str, Any] = {'type': main_thinking_type}
        if main_thinking_type == 'enabled':
            thinking_config['budget_tokens'] = _env_int('SEARCH_THINKING_BUDGET_TOKENS', 4096)
        main_request_args['extra_body']['thinking'] = thinking_config

    if _is_aws_third_anthropic_model(main_model):
        if main_reasoning_effort:
            main_request_args['extra_body']['reasoning_effort'] = main_reasoning_effort
        if main_temperature is not None:
            logger.debug('Ignoring SEARCH_TEMPERATURE for aws_third_anthropic model')
            main_request_args.pop('temperature', None)
        if main_interleaved_thinking is not None:
            main_request_args['extra_headers'] = {'anthropic-beta': 'interleaved-thinking-2025-05-14'}

        return main_request_args

    if _is_dashscope_ali_qwen_model(main_model):
        if main_reasoning_effort:
            logger.debug(
                'Ignoring SEARCH_REASONING_EFFORT for DashScope ali_qwen model; '
                'use SEARCH_THINKING_TYPE or SEARCH_INTERLEAVED_THINKING to control enable_thinking'
            )
        main_request_args['extra_body'] = {'enable_thinking': effective_thinking_enabled}
        return main_request_args

    if _is_litellm_azure_openai_responses_model(main_model):
        responses_reasoning_effort = _resolve_openai_responses_reasoning_effort(main_model, main_reasoning_effort)
        if responses_reasoning_effort is not None:
            main_request_args['reasoning_effort'] = responses_reasoning_effort
        main_request_args['drop_params'] = True
        if main_interleaved_thinking is not None:
            logger.debug('Ignoring SEARCH_INTERLEAVED_THINKING for responses model')
        return main_request_args

    return main_request_args


def _build_tool_configs() -> dict[str, Any]:
    """Build tool configurations from environment variables.

    Supports:
    - google_serper_search: Requires Serper API credentials (SEARCH_SERPER_*)
    - visit: Requires Jina API credentials (DEFAULT_JINA_*)
    - search_stateful: Requires stateful search credentials (SEARCH_API_*)
    - jupyter_code_interpreter: Optional python_path override (JUPYTER_PYTHON_PATH)

    Returns:
        Dict with tool configurations

    Raises:
        ValueError: If required tool credentials are missing
    """
    configs: dict[str, Any] = {}

    # Jupyter code interpreter - use base image's Python with scientific packages
    # instead of the nix runtime's Python which lacks bs4/PyPDF2/fitz etc.
    jupyter_python_path = os.getenv('JUPYTER_PYTHON_PATH')
    if jupyter_python_path:
        configs['jupyter_code_interpreter'] = {'python_path': jupyter_python_path}
        logger.debug('Jupyter code interpreter configured with python_path: %s', jupyter_python_path)

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
    block_huggingface = _env_bool('DEFAULT_JINA_BLOCK_HUGGINGFACE')

    if jina_app_id and jina_app_key:
        visit_config: dict[str, Any] = {
            'jina_url': jina_url,
            'jina_app_id': jina_app_id,
            'jina_app_key': jina_app_key,
            'jina_model_base_url': jina_model_base_url,
            'jina_model': jina_model,
            'jina_model_app_id': jina_model_app_id,
            'jina_model_app_key': jina_model_app_key,
        }
        if block_huggingface is not None:
            visit_config['block_huggingface'] = block_huggingface
        jina_model_2 = os.getenv('DEFAULT_JINA_MODEL_2')
        if jina_model_2:
            visit_config['jina_model_2'] = jina_model_2
            jina_model_app_id_2 = os.getenv('DEFAULT_JINA_MODEL_APP_ID_2')
            jina_model_app_key_2 = os.getenv('DEFAULT_JINA_MODEL_APP_KEY_2')
            if jina_model_app_id_2:
                visit_config['jina_model_app_id_2'] = jina_model_app_id_2
            if jina_model_app_key_2:
                visit_config['jina_model_app_key_2'] = jina_model_app_key_2
            logger.debug(f'Visit tool dual summary models: [{jina_model}, {jina_model_2}]')
        configs['visit'] = visit_config

    configs['search_stateful'] = _build_search_stateful_tool_config()

    return configs


def _build_search_stateful_tool_config() -> dict[str, Any]:
    """Build nested search_stateful config for science_agent."""
    serper_user = os.getenv('SEARCH_API_USER') or _require_env('SEARCH_SERPER_APP_ID')
    serper_apikey = os.getenv('SEARCH_API_KEY') or _require_env('SEARCH_SERPER_APP_KEY')
    config: dict[str, Any] = {
        'serper': {
            'serper_url': os.getenv('SEARCH_SERPER_URL', 'http://trpc-gpt-eval.production.polaris:8080/search'),
            'serper_app_id': serper_user,
            'serper_app_key': serper_apikey,
        },
        'jina': {
            'jina_url': os.getenv(
                'DEFAULT_JINA_URL', 'http://trpc-gpt-eval.production.polaris:8080/api/jina_ai/search'
            ),
            'provider': os.getenv('DEFAULT_JINA_PROVIDER', 'aws_jina'),
            'jina_app_id': os.getenv('DEFAULT_JINA_APP_ID', serper_user),
            'jina_app_key': os.getenv('DEFAULT_JINA_APP_KEY', serper_apikey),
        },
    }

    if (use_cache := _env_bool('SEARCH_STATEFUL_USE_CACHE')) is not None:
        config['use_cache'] = use_cache
    if (max_search_results := _env_int('SEARCH_STATEFUL_MAX_RESULTS')) is not None:
        config['max_search_results'] = max_search_results
    return config


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


def _build_sandbox_provider_config() -> tuple[str | None, dict[str, Any]]:
    """Build sandbox provider configuration from environment variables.

    Sandbox provider is optional for search task. Only include if explicitly configured.

    Required environment variables:
        SEARCH_SANDBOX_PROVIDER_NAME: Provider name (e.g., 'gongfeng')
        SEARCH_SANDBOX_BASE_URL: API base URL (e.g., 'http://hyrl-sandbox.prod.woa.com:8052')
        SEARCH_SANDBOX_IMAGE: Main container image (e.g., 'mirrors.tencent.com/agentic_data_synthesis/execute-python-tool:latest')
        SEARCH_SANDBOX_RUNTIME_IMAGE: Nexus runtime image with /nix environment (e.g., 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest')

    Optional environment variables:
        SEARCH_SANDBOX_TIMEOUT: Request timeout in seconds (default: 1800.0)
        SEARCH_SANDBOX_TOKEN: Authentication token for sandbox API (e.g., required for gongfeng provider)

    Returns:
        Tuple of (provider_name, provider_config)
        Returns (None, {}) if no sandbox provider is configured
    """
    provider_name = os.getenv('SEARCH_SANDBOX_PROVIDER_NAME')

    if not provider_name:
        return None, {}

    base_url = _require_env('SEARCH_SANDBOX_BASE_URL')
    image = _require_env('SEARCH_SANDBOX_IMAGE')
    nexus_image = _require_env('SEARCH_SANDBOX_RUNTIME_IMAGE')
    timeout = _env_float('SEARCH_SANDBOX_TIMEOUT', 1800.0) or 1800.0

    provider_config: dict[str, Any] = {
        'api_base_url': base_url,
        'image': image,
        'runtime_image': nexus_image,
        'request_timeout': timeout,
    }

    token = os.getenv('SEARCH_SANDBOX_TOKEN')
    if token:
        provider_config['token'] = token
        logger.debug('Sandbox token configured from SEARCH_SANDBOX_TOKEN')

    return provider_name, provider_config


# ============================================================================
# Main Generator Functions
# ============================================================================


def _valid_toolset_versions(_yaml_config: SearchConfig | None = None) -> set[str]:
    """Return all known hardcoded toolset version names.

    The legacy ``yaml_config`` parameter is kept only for call-site compatibility
    and ignored — custom tool combinations now live in ``agent_config.tools``
    directly, not in a separate toolset-version registry.
    """
    return set(TOOLSET_REQUIRED_TOOLS)


def _resolve_system_prompt_template(
    template: str | None,
    *,
    include_finish: bool,
) -> str | None:
    """Fill ``{today_date}`` and ``{finish_instruction}`` placeholders in a prompt template.

    Returns ``None`` when no template is provided, so the agent falls back to its
    built-in default prompt.
    """
    if not template:
        return None

    from datetime import date

    from nexus.extensions.agents.search_agent.prompt import _FINISH_INSTRUCTION, _NO_FINISH_INSTRUCTION

    finish_text = _FINISH_INSTRUCTION if include_finish else _NO_FINISH_INSTRUCTION
    return template.replace('{today_date}', date.today().strftime('%Y-%m-%d')).replace(
        '{finish_instruction}', finish_text
    )


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for search task from Parquet dataset.

    Reads the dataset, validates required fields, and yields BatchItem instances
    configured for the search task. Automatically handles pagination via SEARCH_SKIP
    and SEARCH_LIMIT environment variables.

    Configuration precedence (highest wins) for every per-row mergeable field
    (``task_config``, ``agent_config``, ``agent_name``, ``imports``):
        env overrides  >  YAML  >  parquet row  >  code defaults

    Rationale: the YAML is an *execution config* written by the human running
    the benchmark, while a parquet row only supplies the QA instance (question,
    answer, instance_id, evaluation metadata). When both specify the same knob
    the human's intent wins.

    For top-level self-contained sections (``openai_clients``, ``tool_configs``,
    ``processors``, ``runtime_provider_*``), a YAML value replaces the env-var
    builder output entirely; when YAML is absent, the env-var builders run as
    before. ``SEARCH_CONFIG_PATH`` enables YAML loading; without it, the workflow
    is identical to the legacy env-var-only path.

    Yields:
        BatchItem: Configured batch items ready for execution

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is missing required fields or environment variables
    """

    # ========================================================================
    # Load YAML Configuration (optional)
    #
    # The YAML is the *execution config* for this run; the parquet row provides
    # only the QA instance. Precedence for every per-row mergeable field:
    #     env overrides > YAML > parquet row > code default
    # ========================================================================

    yaml_config = load_search_config_if_exists()
    yaml_task_config: dict[str, Any] = dict(yaml_config.task_config) if yaml_config else {}
    yaml_agent_config: dict[str, Any] = dict(yaml_config.agent_config) if yaml_config else {}

    if yaml_config:
        logger.debug(
            'YAML config loaded: agent_name=%s, agent_config_keys=%s, task_config_keys=%s, '
            'openai_clients=%s, tool_configs=%s, benchmarks=%s',
            yaml_config.agent_name,
            list(yaml_agent_config),
            list(yaml_task_config),
            list(yaml_config.openai_clients),
            list(yaml_config.tool_configs),
            list(yaml_config.benchmarks),
        )

    # ========================================================================
    # Load Environment Configuration
    # ========================================================================

    dataset_path = _require_env('SEARCH_DATASET_PATH')
    skip = _env_int('SEARCH_SKIP', 0) or 0
    limit = _env_int('SEARCH_LIMIT')
    selected_instance_ids = _parse_instance_ids_filter()
    selected_run_ids = _parse_run_ids_filter()

    # agent_name: env > YAML > default. Kept as a scalar (not in the merge dict)
    # because parquet rows are allowed to override it independently.
    agent_name_env = os.getenv('SEARCH_AGENT_NAME')
    agent_name = agent_name_env or (
        yaml_config.agent_name if yaml_config and yaml_config.agent_name else 'search_agent'
    )
    logger.debug(f'Agent name: {agent_name}')

    # Overridable per-row agent_config knobs, resolved from env once up-front so
    # they uniformly participate in the dict-merge as the highest-priority layer.
    agent_env_overrides: dict[str, Any] = {}
    if (v := _env_bool('IS_TRAINING')) is not None:
        agent_env_overrides['is_training'] = v
    if (v := _env_int('SEARCH_MAX_ITERATIONS')) is not None:
        agent_env_overrides['max_iterations'] = v
    if (v := _env_int('SEARCH_AGENT_MAX_LLM_RETRIES')) is not None:
        agent_env_overrides['max_llm_retries'] = v
    toolset_version_env = os.getenv('SEARCH_TOOLSET_VERSION')
    if toolset_version_env is not None:
        all_toolset_versions = _valid_toolset_versions()
        if toolset_version_env not in all_toolset_versions:
            raise ValueError(
                f'Unknown SEARCH_TOOLSET_VERSION: {toolset_version_env!r}. Available: {sorted(all_toolset_versions)}'
            )
        agent_env_overrides['toolset_version'] = toolset_version_env
    if (v := _env_int('SEARCH_KEEP_RECENT_K')) is not None:
        agent_env_overrides['keep_recent_k'] = v
    if (v := _env_int('SEARCH_MAX_CONTEXT_TOKENS')) is not None:
        agent_env_overrides['max_context_tokens'] = v
    if (v := _env_bool('SEARCH_ENABLE_COMPACTION')) is not None:
        agent_env_overrides['enable_compaction'] = v
    logger.debug('Agent env overrides: %s', agent_env_overrides or '(none)')

    # Evaluation knobs: env takes final priority (matches `run-benchmarks --eval-mode` semantics).
    task_env_overrides: dict[str, Any] = {}
    if eval_mode_env := os.getenv('SEARCH_EVAL_MODE'):
        task_env_overrides['eval_mode'] = eval_mode_env
    if (v := _env_bool('SEARCH_ENABLE_SUMMARY')) is not None:
        task_env_overrides['enable_summary'] = v
    if (v := _env_bool('SEARCH_RUN_ORACLE')) is not None:
        task_env_overrides['run_oracle'] = v
    if (v := _env_bool('TRAINING_LENGTH_EXCEEDED')) is not None:
        task_env_overrides['training_length_exceeded'] = v
    # Baseline eval_mode is what YAML declares, else env, else hard default.
    eval_mode = task_env_overrides.get('eval_mode') or yaml_task_config.get('eval_mode') or 'llm_judge'
    run_oracle = task_env_overrides.get('run_oracle', yaml_task_config.get('run_oracle', False))
    logger.debug(f'Eval mode: {eval_mode}')
    if run_oracle:
        logger.info(
            'Oracle mode enabled: agent will be bypassed; non-wide eval uses exact_match '
            '(score is typically 1.0 when prediction equals reference)'
        )

    # OpenAI clients: YAML (self-contained) > env-var builder
    if yaml_config and yaml_config.openai_clients:
        openai_clients = {role: dict(spec) for role, spec in yaml_config.openai_clients.items()}
        logger.debug('OpenAI clients from YAML: %s', list(openai_clients.keys()))
    else:
        openai_clients = _build_openai_clients(eval_mode)

    # Tool configs: YAML (self-contained) > env-var builder
    if yaml_config and yaml_config.tool_configs:
        tool_configs: dict[str, Any] = {k: dict(v) for k, v in yaml_config.tool_configs.items()}
        logger.debug('Tool configs from YAML: %s', list(tool_configs.keys()))
    else:
        tool_configs = _build_tool_configs()
        logger.debug(f'Tool configs from env vars: {list(tool_configs.keys())}')

    # Processors and runtime provider: YAML direct-passthrough when set, else env builder.
    # Note: dict(...) is a shallow copy of the ProcessorConfig mapping; RunInput re-validates
    # the values through pydantic so passing ProcessorConfig instances directly is fine.
    if yaml_config and yaml_config.processors:
        processors = dict(yaml_config.processors)
        logger.debug('Processors from YAML: %s', list(processors.keys()))
    else:
        processors = _build_processors()

    if yaml_config and yaml_config.runtime_provider_name:
        sandbox_provider_name = yaml_config.runtime_provider_name
        sandbox_provider_config = dict(yaml_config.runtime_provider_config)
    else:
        sandbox_provider_name, sandbox_provider_config = _build_sandbox_provider_config()
    if sandbox_provider_name:
        logger.debug(f'Sandbox provider configured: {sandbox_provider_name}')
        logger.debug(
            'Sandbox images: image=%s, runtime_image=%s',
            sandbox_provider_config.get('image'),
            sandbox_provider_config.get('runtime_image'),
        )
        logger.debug('JUPYTER_PYTHON_PATH=%s', os.getenv('JUPYTER_PYTHON_PATH', 'NOT SET'))

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

        if task_name != 'search':
            logger.warning(f'Skipping row {idx}: task_name is "{task_name}", expected "search"')
            skipped_count += 1
            continue

        # task_config merge: parquet row (QA data) ⊕ YAML execution config ⊕ env overrides.
        # The parquet row provides question/answer/instance_id/evaluation; YAML
        # overrides execution knobs like eval_mode; env wins last to match CLI
        # flag semantics (e.g. `--eval-mode`). `eval_mode` is always materialized
        # so downstream validators and SearchTaskInput see an explicit value.
        row_task_config = nexus_config.get('task_config', {}) or {}
        task_config = {**row_task_config, **yaml_task_config, **task_env_overrides}
        task_config.setdefault('eval_mode', eval_mode)

        task_instance_id = task_config.get('instance_id')
        if selected_instance_ids is not None:
            if not task_instance_id:
                logger.warning(f'Skipping row {idx}: missing task_config.instance_id while ID filter is enabled')
                skipped_count += 1
                continue
            if task_instance_id not in selected_instance_ids:
                filtered_count += 1
                continue

        if 'question' not in task_config or 'answer' not in task_config:
            logger.warning(f'Skipping row {idx}: missing question or answer in task_config')
            skipped_count += 1
            continue

        # Validate reward service env vars (config resolved at runtime by evaluator).
        if task_config.get('eval_mode') == 'reward_service' and not reward_service_configured:
            logger.warning(
                f'Skipping row {idx}: eval_mode is reward_service but REWARD_SERVICE_TOKEN '
                'and REWARD_SERVICE_ROUTER_ID are not configured'
            )
            skipped_count += 1
            continue

        # agent_name precedence: env > YAML > parquet row > 'search_agent'.
        # The resolved `agent_name` already encodes env > YAML > default; only
        # use the row value when neither env nor YAML set one explicitly.
        yaml_agent_name = yaml_config.agent_name if yaml_config and yaml_config.agent_name else None
        if agent_name_env or yaml_agent_name:
            item_agent_name = agent_name
        else:
            item_agent_name = nexus_config.get('agent_name', agent_name)

        # WideSearch samples force-strip the finish tool so the agent answers freely.
        evaluation = task_config.get('evaluation')
        is_wide_search_sample = isinstance(evaluation, dict) and WIDESEARCH_EVALUATION_REQUIRED_KEYS.issubset(
            evaluation
        )

        # agent_config merge: code defaults ⊕ parquet row ⊕ YAML ⊕ env overrides.
        # Strip None values from the parquet row so they don't mask real defaults/YAML values
        # during the dict merge (carried over from manna's eb22130a fix).
        row_agent_config = nexus_config.get('agent_config', {}) or {}
        if isinstance(row_agent_config, dict):
            row_agent_config = {k: v for k, v in row_agent_config.items() if v is not None}
        agent_config: dict[str, Any] = {
            'is_training': True,
            'max_iterations': 10,
            'max_llm_retries': 0,
            'finish_tool_names': ['finish'],
            **row_agent_config,
            **yaml_agent_config,
            **agent_env_overrides,
        }
        # Business rule: wide-search samples MUST answer as free text, never via a
        # finish tool — enforced after the merge so no YAML/row setting can break it.
        if is_wide_search_sample:
            agent_config['finish_tool_names'] = []

        # Resolve {today_date} / {finish_instruction} placeholders in system_prompt template.
        if isinstance(agent_config.get('system_prompt'), str):
            agent_config['system_prompt'] = _resolve_system_prompt_template(
                agent_config['system_prompt'],
                include_finish=bool(agent_config.get('finish_tool_names')),
            )

        # imports precedence: YAML > parquet row > ['nexus.extensions']. No env layer for imports.
        if yaml_config and yaml_config.imports:
            imports = list(yaml_config.imports)
        else:
            imports = _safe_parse_imports(nexus_config.get('imports')) or ['nexus.extensions']
        if item_agent_name == 'science_agent' and 'nexus.extensions.agents.science_agent.agent' not in imports:
            imports.append('nexus.extensions.agents.science_agent.agent')

        config = RunInput(
            imports=imports,
            task_name=task_name,
            task_config=task_config,
            tool_configs=tool_configs,
            agent_name=item_agent_name,
            agent_config=agent_config,
            openai_clients=openai_clients,
            runtime_provider_name=sandbox_provider_name,
            runtime_provider_config=sandbox_provider_config,
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
            run_id = task_instance_id or f'search_{idx:05d}'
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
        'task_type': 'search',
        'dataset_path': dataset_path,
        'agent_name': os.getenv('SEARCH_AGENT_NAME', 'search_agent'),
        'is_training': _env_bool('IS_TRAINING', True),
        'max_iterations': _env_int('SEARCH_MAX_ITERATIONS', 10) or 10,
        'skip': _env_int('SEARCH_SKIP', 0) or 0,
        'limit': _env_int('SEARCH_LIMIT'),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
    }
    yaml_config_path = os.getenv('SEARCH_CONFIG_PATH')
    if yaml_config_path:
        metadata['yaml_config_path'] = yaml_config_path
    toolset_version = os.getenv('SEARCH_TOOLSET_VERSION')
    if toolset_version:
        metadata['toolset_version'] = toolset_version
    selected_instance_ids = _parse_instance_ids_filter()
    if selected_instance_ids is not None:
        metadata['instance_id_filter_enabled'] = True
        metadata['instance_id_filter_count'] = len(selected_instance_ids)
    selected_run_ids = _parse_run_ids_filter()
    if selected_run_ids is not None:
        metadata['run_id_filter_enabled'] = True
        metadata['run_id_filter_count'] = len(selected_run_ids)

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
