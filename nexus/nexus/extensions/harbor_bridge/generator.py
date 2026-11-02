"""Generic Harbor Bridge Generator: Supports any harbor-format benchmark.

This generator scans a given dataset directory for harbor-format tasks
(directories containing task.toml) and generates nexus BatchItems.

It follows the same environment variable conventions as nexus's native
terminal task generator (generator_gongfeng.py), so switching between
`terminal_task` and `harbor_bridge_task` requires zero config changes
for shared settings (agent, LLM, runtime).

Harbor-specific env vars (dataset loading):
  - HARBOR_DATASET_PATH: Path to harbor-format dataset (required, comma-separated for multiple)
  - HARBOR_DATASET_NAME: Dataset name for run_id prefix (default: auto-detected from path)
  - HARBOR_ENABLE_DIND: Explicit batch override for environment.enable_dind (optional)
  - HARBOR_MCP_INJECT: Set to off/none to disable task MCP wiring for legacy runs
  - HARBOR_SKIP_INVALID_TASKS: Warn and skip invalid tasks instead of failing validation (default: false)
  - HARBOR_OFFLINE_ALLOWLIST: Comma-separated egress targets allowed for allow_internet=false tasks

Nexus-native env vars (shared with all nexus generators):
  - AGENT_NAME: Agent to use (default: terminus2)
  - MAX_ITERATIONS: Max agent turns (default: 200)
  - AGENT_CONFIG_JSON: Optional JSON object merged into generic agent config
  - OPENAI_API_KEY: LLM API key (default: EMPTY)
  - OPENAI_BASE_URL: LLM endpoint (default: internal litellm proxy)
  - OPENAI_MODEL: Model name (default: gpt-5.4)
  - OPENAI_CLIENT_NAME: OpenAI client type (default: litellm)
  - OPENAI_CACHE_CONTROL_ENABLED: Enable LiteLLM prompt cache markers (default: true for litellm)
  - OPENAI_CACHE_CONTROL_STRATEGY: Cache marker strategy (default: claude_code)
  - OPENAI_LOG_LLM_USAGE: Log token/cache usage (default: true for litellm)
  - OPENAI_MAX_RETRIES: LLM request retries (default: 3)
  - OPENAI_THINKING_TYPE: Claude thinking mode (default: adaptive for Opus 4.8)
  - OPENAI_THINKING_DISPLAY: Claude thinking display mode (default: summarized)
  - OPENAI_THINKING_EFFORT: Claude adaptive-thinking effort (default: xhigh)
  - OPENAI_MAX_TOKENS: Maximum output tokens (default: 128000 for Opus 4.8)
  - CLAUDE_CODE_THINKING_PRESET: Proxy thinking format (default: bedrock-claude for Claude)
  - TEMPERATURE: Sampling temperature (default: None)
  - TASK_NAMES: Comma-separated task name filter (default: all)
  - TASK_LIMIT: Max tasks to generate (default: no limit)
  - TASK_SKIP: Number of tasks to skip (default: 0)
  - TIMEOUT_MULTIPLIER: Multiplier for task-defined timeouts (default: 1.0)
  - GONGFENG_RUNTIME_TOKEN: Auth token for gongfeng sandbox (required)
  - GONGFENG_API_BASE_URL: Sandbox API URL (default: http://hyrl-sandbox.prod.woa.com:8052)
  - GONGFENG_RUNTIME_IMAGE: Nexus runtime image (default: mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.4.8)
  - GONGFENG_RUNTIME_IMAGE_TAG: Legacy runtime image tag override (optional)
  - GONGFENG_REQUEST_TIMEOUT: API request timeout (default: 24000)
  - GONGFENG_STARTUP_JITTER_MAX: Max random seconds before each sandbox acquire (default: 0)
  - GONGFENG_RUNTIME_PROVIDER: Cluster selection (default: gongfeng)
  - GONGFENG_PURPOSE: Runtime purpose: prod, eval, or test (default: backend test)
  - GONGFENG_CPU: CPU cores per sandbox (default: 2)
  - GONGFENG_MEMORY: Memory GB per sandbox (default: 10)
  - RUN_ORACLE: Run oracle solution instead of agent (default: false)
  - RUN_NOP: Skip agent/oracle and verify pristine post-boot state (default: false)

Usage:
  HARBOR_DATASET_PATH=/tmp/terminal-bench-2 \\
  GONGFENG_RUNTIME_TOKEN=xxx \\
  AGENT_NAME=terminus2 \\
  OPENAI_MODEL=gpt-5.4 \\
  uv run nexus run-batch \\
    -g nexus/extensions/harbor_bridge/generator.py \\
    -d ./outputs/eval.db --max-concurrent 10
"""

import json
import logging
import os

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

from nexus import BatchItem
from nexus.extensions.harbor_bridge import compiler as task_compiler
from nexus.extensions.harbor_bridge.discovery import discover_task_dirs
from nexus.extensions.harbor_bridge.identity import TaskIdentity, build_task_identity
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)

# ============ Defaults (aligned with nexus terminal task conventions) ============

DEFAULTS: dict[str, Any] = {
    'agent_name': 'terminus2',
    'max_iterations': 200,
    'openai_client_name': 'noauth_openai',
    'api_base': 'http://llm-api.model-eval.woa.com/v1',
    'api_key': 'EMPTY',
    'model': 'api_azure_openai_gpt-5.4-2026-03-05',
    'temperature': None,
    'openai_cache_control_enabled': True,
    'openai_cache_control_strategy': 'claude_code',
    'openai_log_llm_usage': True,
    'openai_max_retries': 3,
    'openai_claude_thinking_type': 'adaptive',
    'openai_claude_thinking_display': 'summarized',
    'openai_claude_thinking_effort': 'xhigh',
    'openai_claude_max_tokens': 128000,
    'gongfeng_api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
    'gongfeng_runtime_image': 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.4.8',
    'gongfeng_request_timeout': 24000,
    'gongfeng_startup_jitter_max': 0.0,
    'gongfeng_purpose': '',
    'gongfeng_cpu': task_compiler.DEFAULT_GONGFENG_CPU,
    'gongfeng_memory': task_compiler.DEFAULT_GONGFENG_MEMORY_GB,
    'timeout_multiplier': task_compiler.DEFAULT_TIMEOUT_MULTIPLIER,
    'run_oracle': False,
    'run_nop': False,
}


def _append_auth_query(api_key: str, **params: str) -> str:
    """Append or replace routing parameters in an API-key query string."""
    if not api_key:
        return api_key
    base, separator, query = api_key.partition('?')
    query_items = [item for item in query.split('&') if item] if separator else []
    for key, value in params.items():
        if not value:
            continue
        query_items = [item for item in query_items if item.split('=', 1)[0] != key]
        query_items.append(f'{quote(key)}={quote(value)}')
    return f'{base}?{"&".join(query_items)}' if query_items else base


def _make_cache_task_id(run_id: str) -> str:
    """Return a stable task-scoped cache-affinity ID for one RunInput.

    Delegates to the runner's shared derivation so the legacy generator and the
    composed run path (single.py) stay byte-for-byte identical. Imported lazily
    because tests load this module against a stubbed ``nexus.runner.single``.
    """
    from nexus.runner.single import cache_task_id_for

    return cache_task_id_for(run_id)


def _get_env(key: str, default: Any = None) -> Any:
    """Get env var with type coercion (bool/int/float)."""
    value = os.getenv(key)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ('true', '1', 'yes')
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _get_env_bool_optional(key: str) -> bool | None:
    value = os.getenv(key)
    if value is None or value == '':
        return None
    normalized = value.strip().lower()
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(f'{key} must be a boolean, got {value!r}')


def _build_request_args(model: str, temperature: Any) -> dict[str, Any]:
    request_args: dict[str, Any] = {
        'model': model,
        **(({'temperature': temperature}) if temperature is not None else {}),
    }
    max_tokens = os.getenv('OPENAI_MAX_TOKENS', '').strip()
    is_claude_opus_48 = 'claude-opus-4-8' in model.lower()
    disable_thinking = _get_env('OPENAI_DISABLE_THINKING', False)
    if is_claude_opus_48:
        request_args['max_tokens'] = int(max_tokens or DEFAULTS['openai_claude_max_tokens'])
    elif max_tokens:
        request_args['max_tokens'] = int(max_tokens)

    if disable_thinking:
        request_args['thinking'] = {'type': 'disabled'}
    elif is_claude_opus_48:
        thinking_type = os.getenv(
            'OPENAI_THINKING_TYPE',
            DEFAULTS['openai_claude_thinking_type'],
        ).strip()
        thinking: dict[str, str] = {'type': thinking_type}
        thinking_display = os.getenv(
            'OPENAI_THINKING_DISPLAY',
            DEFAULTS['openai_claude_thinking_display'],
        ).strip()
        if thinking_display:
            thinking['display'] = thinking_display
        request_args['thinking'] = thinking

        thinking_effort = os.getenv(
            'OPENAI_THINKING_EFFORT',
            DEFAULTS['openai_claude_thinking_effort'],
        ).strip()
        if thinking_effort:
            request_args['output_config'] = {'effort': thinking_effort}
    return request_args


def _build_openai_client_args(
    *,
    client_name: str,
    api_key: str,
    api_base: str,
    run_id: str,
    cache_control_enabled: bool,
    cache_control_strategy: str,
    log_llm_usage: bool,
    max_retries: int,
) -> tuple[dict[str, Any], str | None]:
    """Build client args and optional task-scoped gateway cache affinity."""
    client_args: dict[str, Any] = {
        'api_key': api_key,
        'base_url': api_base,
    }
    if client_name != 'litellm':
        return client_args, None

    client_args.update({
        'max_retries': max_retries,
        'log_llm_usage': log_llm_usage,
    })
    if not cache_control_enabled:
        return client_args, None
    if not cache_control_strategy:
        raise ValueError('OPENAI_CACHE_CONTROL_STRATEGY cannot be empty when prompt caching is enabled')

    cache_task_id = _make_cache_task_id(run_id)
    client_args.update({
        'api_key': _append_auth_query(api_key, cache_task_id=cache_task_id),
        'cache_control_enabled': True,
        'cache_control_strategy': cache_control_strategy,
    })
    return client_args, cache_task_id


# MCP declarations are normalized by harbor_bridge.compiler and serialized into
# task_config. Harnesses consume that frozen task input instead of re-reading task.toml.


def _parse_offline_allowlist(raw: str) -> tuple[str, ...]:
    """Parse ``HARBOR_OFFLINE_ALLOWLIST``'s comma-separated egress targets."""
    return tuple(target.strip() for target in raw.split(',') if target.strip())


def _build_agent_config(
    workdir: str,
    max_iterations: int,
    task_agent_timeout: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        'max_iterations': max_iterations,
        'working_dir': workdir,
        'timeout': float(task_agent_timeout),
    }
    raw_override = os.getenv('AGENT_CONFIG_JSON', '').strip()
    if raw_override:
        override = json.loads(raw_override)
        if not isinstance(override, dict):
            raise ValueError('AGENT_CONFIG_JSON must decode to a JSON object')
        config.update(override)
    return config


# ============ Harbor-specific helpers ============


def _discover_task_dirs(dataset_paths: list[Path]) -> list[Path]:
    """Discover all valid harbor-format task directories."""
    task_filter_str = os.getenv('TASK_NAMES', '')
    task_filter = {t.strip() for t in task_filter_str.split(',') if t.strip()} if task_filter_str else None
    flat_layout = _get_env('HARBOR_FLAT_TASKS', False)
    workers = max(1, int(os.getenv('HARBOR_LOAD_WORKERS', '1')))
    task_dirs = discover_task_dirs(dataset_paths, flat=flat_layout, workers=workers)
    return [task_dir for task_dir in task_dirs if task_filter is None or task_dir.name in task_filter]


def _apply_task_include(task_dirs: list[Path]) -> list[Path]:
    """Apply the optional HARBOR_TASK_INCLUDE basename filter."""
    include_csv = os.getenv('HARBOR_TASK_INCLUDE', '').strip()
    if not include_csv:
        return task_dirs
    wanted = {name.strip() for name in include_csv.split(',') if name.strip()}
    return [task_dir for task_dir in task_dirs if task_dir.name in wanted]


def _load_valid_tasks(task_dirs: list[Path]) -> list[tuple[Path, dict[str, Any], Any]]:
    """Validate selected tasks before generation, failing unless skipping is explicit."""
    valid_tasks: list[tuple[Path, dict[str, Any], Any]] = []
    failures: list[tuple[Path, Exception]] = []

    def load_one(task_dir: Path) -> tuple[Path, dict[str, Any] | None, Any | None, Exception | None]:
        try:
            loaded = task_compiler.load_harbor_task(task_dir)
            return task_dir, loaded.config, loaded.task, None
        except Exception as exc:
            return task_dir, None, None, exc

    load_workers = max(1, int(os.getenv('HARBOR_LOAD_WORKERS', '1')))
    executor: ThreadPoolExecutor | None = None
    if load_workers == 1 or len(task_dirs) <= 1:
        loaded = map(load_one, task_dirs)
    else:
        executor = ThreadPoolExecutor(max_workers=min(load_workers, len(task_dirs)))
        loaded = executor.map(load_one, task_dirs)

    try:
        for task_dir, config, harbor_task, error in loaded:
            if error is not None:
                failures.append((task_dir, error))
            else:
                assert config is not None
                assert harbor_task is not None
                valid_tasks.append((task_dir, config, harbor_task))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not failures:
        return valid_tasks

    skip_invalid = _get_env('HARBOR_SKIP_INVALID_TASKS', False)
    if skip_invalid:
        for task_dir, exc in failures:
            logger.warning(
                'Skipping invalid Harbor task %s: %s: %s',
                task_dir,
                type(exc).__name__,
                str(exc).replace('\n', ' | '),
            )
        logger.warning('Skipped %d invalid Harbor task(s)', len(failures))
        return valid_tasks

    details = '\n'.join(
        f'  - {task_dir}: {type(exc).__name__}: {str(exc).replace(chr(10), " | ")}' for task_dir, exc in failures
    )
    raise ValueError(
        f'Found {len(failures)} invalid Harbor task(s):\n{details}\n'
        'Fix these tasks or set HARBOR_SKIP_INVALID_TASKS=1 to warn and skip them.'
    )


def _build_task_identities(
    valid_tasks: list[tuple[Path, dict[str, Any], Any]],
    *,
    skip_invalid: bool = False,
) -> dict[Path, TaskIdentity]:
    """Build identities shared with the T3-Harbor Dataset Source."""
    identities: dict[Path, TaskIdentity] = {}
    paths_by_id: dict[str, list[Path]] = {}
    failures: list[tuple[Path, Exception]] = []

    for task_dir, _, _ in valid_tasks:
        try:
            identity = build_task_identity(task_dir, task_dir.name)
        except Exception as exc:
            failures.append((task_dir, exc))
            continue
        identities[task_dir] = identity
        paths_by_id.setdefault(identity.run_id, []).append(task_dir)

    if failures:
        if not skip_invalid:
            details = '\n'.join(
                f'  - {task_dir}: {type(exc).__name__}: {str(exc).replace(chr(10), " | ")}'
                for task_dir, exc in failures
            )
            raise ValueError(f'Failed to build T3-Harbor task identities:\n{details}')
        for task_dir, exc in failures:
            logger.warning(
                'Skipping Harbor task with invalid identity inputs %s: %s: %s',
                task_dir,
                type(exc).__name__,
                str(exc).replace('\n', ' | '),
            )

    collisions = {run_id: paths for run_id, paths in paths_by_id.items() if len(paths) > 1}
    if collisions:
        details = '\n'.join(
            f'  - {run_id}: {", ".join(str(path) for path in paths)}' for run_id, paths in collisions.items()
        )
        raise ValueError(
            'T3-Harbor task IDs and canonical task content are duplicated across dataset roots. '
            'Rename one task directory or select only one copy:\n'
            f'{details}'
        )
    return identities


def _build_run_ids(
    valid_tasks: list[tuple[Path, dict[str, Any], Any]],
) -> dict[Path, str]:
    """Compatibility helper returning only the canonical run IDs."""
    return {task_dir: identity.run_id for task_dir, identity in _build_task_identities(valid_tasks).items()}


def _get_dataset_name(dataset_paths: list[Path]) -> str:
    """Auto-detect dataset name from path."""
    explicit = os.getenv('HARBOR_DATASET_NAME', '')
    if explicit:
        return explicit
    if dataset_paths:
        return dataset_paths[0].name
    return 'harbor-dataset'


def _parse_dataset_paths() -> list[Path]:
    """Parse HARBOR_DATASET_PATH into a list of paths."""
    raw = os.getenv('HARBOR_DATASET_PATH', '')
    if not raw:
        raise ValueError(
            'HARBOR_DATASET_PATH is required. '
            'Set it to a harbor-format dataset directory '
            '(or comma-separated list of directories).'
        )
    return [Path(p.strip()) for p in raw.split(',') if p.strip()]


# ============ Generator (nexus run-batch interface) ============


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItems for nexus run-batch."""
    dataset_paths = _parse_dataset_paths()
    task_dirs = _apply_task_include(_discover_task_dirs(dataset_paths))

    if not task_dirs:
        raise ValueError(
            f'No valid harbor tasks found in: {dataset_paths}. Tasks must have task.toml + instruction.md.'
        )
    valid_tasks = _load_valid_tasks(task_dirs)
    if not valid_tasks:
        raise ValueError(f'No valid Harbor tasks remain after validation: {dataset_paths}')

    # Read nexus-native env vars (same names as generator_gongfeng.py)
    agent_name = _get_env('AGENT_NAME', DEFAULTS['agent_name'])
    max_iterations = _get_env('MAX_ITERATIONS', DEFAULTS['max_iterations'])
    openai_client_name = _get_env('OPENAI_CLIENT_NAME', DEFAULTS['openai_client_name'])
    api_base = _get_env('OPENAI_BASE_URL', DEFAULTS['api_base'])
    api_key = _get_env('OPENAI_API_KEY', DEFAULTS['api_key'])
    model = _get_env('OPENAI_MODEL', DEFAULTS['model'])
    temperature = _get_env('TEMPERATURE', DEFAULTS['temperature'])
    cache_control_enabled = _get_env(
        'OPENAI_CACHE_CONTROL_ENABLED',
        DEFAULTS['openai_cache_control_enabled'],
    )
    cache_control_strategy = str(
        _get_env(
            'OPENAI_CACHE_CONTROL_STRATEGY',
            DEFAULTS['openai_cache_control_strategy'],
        )
    ).strip()
    log_llm_usage = _get_env('OPENAI_LOG_LLM_USAGE', DEFAULTS['openai_log_llm_usage'])
    openai_max_retries = _get_env('OPENAI_MAX_RETRIES', DEFAULTS['openai_max_retries'])
    if openai_max_retries < 0:
        raise ValueError(f'OPENAI_MAX_RETRIES must be >= 0, got {openai_max_retries}')
    timeout_multiplier = _get_env('TIMEOUT_MULTIPLIER', DEFAULTS['timeout_multiplier'])
    if timeout_multiplier <= 0:
        raise ValueError(f'TIMEOUT_MULTIPLIER must be positive, got {timeout_multiplier}')
    run_oracle = _get_env('RUN_ORACLE', DEFAULTS['run_oracle'])
    run_nop = _get_env('RUN_NOP', DEFAULTS['run_nop'])
    if run_oracle and run_nop:
        raise ValueError('RUN_ORACLE and RUN_NOP cannot both be true.')
    mode = 'oracle' if run_oracle else 'nop' if run_nop else 'agent'

    gongfeng_api_base_url = _get_env('GONGFENG_API_BASE_URL', DEFAULTS['gongfeng_api_base_url'])
    gongfeng_runtime_token = _get_env('GONGFENG_RUNTIME_TOKEN', '')
    gongfeng_runtime_image = _get_env('GONGFENG_RUNTIME_IMAGE', DEFAULTS['gongfeng_runtime_image'])
    legacy_runtime_image_tag = os.getenv('GONGFENG_RUNTIME_IMAGE_TAG', '').strip()
    if legacy_runtime_image_tag:
        gongfeng_runtime_image = (
            'current'
            if legacy_runtime_image_tag == 'current'
            else f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{legacy_runtime_image_tag}'
        )
    gongfeng_request_timeout = _get_env('GONGFENG_REQUEST_TIMEOUT', DEFAULTS['gongfeng_request_timeout'])
    gongfeng_startup_jitter_max = _get_env('GONGFENG_STARTUP_JITTER_MAX', DEFAULTS['gongfeng_startup_jitter_max'])
    if gongfeng_startup_jitter_max < 0:
        raise ValueError(f'GONGFENG_STARTUP_JITTER_MAX must be >= 0, got {gongfeng_startup_jitter_max}')
    # None when unset: compile_harbor_task then defaults to gongfeng, or upgrades to suanli for
    # network-isolated tasks. An explicit non-suanli value there conflicts with isolation and raises.
    gongfeng_runtime_provider = os.getenv('GONGFENG_RUNTIME_PROVIDER') or None
    gongfeng_purpose = str(_get_env('GONGFENG_PURPOSE', DEFAULTS['gongfeng_purpose'])).strip()
    if gongfeng_purpose not in {'', 'prod', 'eval', 'test'}:
        raise ValueError(f'GONGFENG_PURPOSE must be prod, eval, or test, got {gongfeng_purpose!r}')
    gongfeng_cpu = _get_env('GONGFENG_CPU', DEFAULTS['gongfeng_cpu'])
    gongfeng_memory = _get_env('GONGFENG_MEMORY', DEFAULTS['gongfeng_memory'])

    if not gongfeng_runtime_token:
        raise ValueError('GONGFENG_RUNTIME_TOKEN is required.')

    # Harbor-specific
    dataset_name = _get_dataset_name(dataset_paths)
    skip_invalid = _get_env('HARBOR_SKIP_INVALID_TASKS', False)
    task_identities = _build_task_identities(valid_tasks, skip_invalid=skip_invalid)
    skip = _get_env('TASK_SKIP', 0)
    limit_val = os.getenv('TASK_LIMIT')
    limit = int(limit_val) if limit_val else None

    offline_allowlist = _parse_offline_allowlist(os.getenv('HARBOR_OFFLINE_ALLOWLIST', ''))
    boot_command_override = os.getenv('HARBOR_BOOT_CMD', '').strip()
    verifier_boot_command_override = os.getenv('HARBOR_VERIFIER_BOOT_CMD', '').strip()

    skipped = 0
    produced = 0

    for task_dir, config, harbor_task in valid_tasks:
        if task_dir not in task_identities:
            continue
        if skipped < skip:
            skipped += 1
            continue

        try:
            compiled = task_compiler.compile_harbor_task(
                task_dir,
                config,
                harbor_task,
                fallback_cpu=gongfeng_cpu,
                fallback_memory_gb=gongfeng_memory,
                timeout_multiplier=timeout_multiplier,
                configured_runtime_provider=gongfeng_runtime_provider,
                enable_dind_override=_get_env_bool_optional('HARBOR_ENABLE_DIND'),
                offline_allowlist=offline_allowlist,
                boot_command_override=boot_command_override,
                verifier_boot_command_override=verifier_boot_command_override,
            )
        except Exception as exc:
            if not skip_invalid:
                raise
            logger.warning(
                'Skipping invalid compiled Harbor task %s: %s: %s',
                task_dir,
                type(exc).__name__,
                str(exc).replace('\n', ' | '),
            )
            continue
        compiled_body = compiled.dataset_body()
        compiled_task_config = dict(compiled_body['task_config'])
        if os.getenv('HARBOR_MCP_INJECT', '').strip().lower() in {'off', 'none'}:
            compiled_task_config['mcp_servers'] = []
        task_agent_timeout = int(compiled_task_config['agent_timeout'])

        identity = task_identities[task_dir]
        run_id = identity.run_id
        openai_client_args, cache_task_id = _build_openai_client_args(
            client_name=openai_client_name,
            api_key=api_key,
            api_base=api_base,
            run_id=run_id,
            cache_control_enabled=bool(cache_control_enabled),
            cache_control_strategy=cache_control_strategy,
            log_llm_usage=bool(log_llm_usage),
            max_retries=openai_max_retries,
        )
        openai_request_args = _build_request_args(model, temperature)

        run_config: dict[str, Any] = {
            **compiled_body,
            'imports': [
                'nexus.extensions.harbor_bridge.task',
                'nexus.extensions.harbor_bridge.noauth_client',
            ],
            'task_name': 'harbor_bridge_task',
            'task_config': {
                **compiled_task_config,
                'mode': mode,
            },
            'runtime_provider_name': 'gongfeng',
            'runtime_provider_config': {
                'api_base_url': gongfeng_api_base_url,
                'token': gongfeng_runtime_token,
                'runtime_image': gongfeng_runtime_image,
                'request_timeout': gongfeng_request_timeout,
                'startup_jitter_max': gongfeng_startup_jitter_max,
                'runtime_provider': compiled.runtime_provider,
                **({'purpose': gongfeng_purpose} if gongfeng_purpose else {}),
            },
            'agent_name': agent_name,
            'agent_config': _build_agent_config(
                str(compiled_task_config['workdir']),
                max_iterations,
                task_agent_timeout,
            ),
            'openai_clients': {
                'main': {
                    'name': openai_client_name,
                    'config': {
                        'client_args': openai_client_args,
                        'request_args': openai_request_args,
                        **(
                            {'enforced_request_args': {'max_tokens': openai_request_args['max_tokens']}}
                            if agent_name.startswith('claude_code') and 'max_tokens' in openai_request_args
                            else {}
                        ),
                    },
                },
            },
        }

        yield BatchItem(
            config=RunInput.model_validate(run_config),
            run_id=run_id,
            metadata={
                'benchmark': dataset_name,
                **compiled.metadata,
                **identity.metadata(task_dir=task_dir, config=config),
                **({'cache_task_id': cache_task_id} if cache_task_id else {}),
            },
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    """Get total number of tasks available."""
    try:
        dataset_paths = _parse_dataset_paths()
    except (FileNotFoundError, ValueError):
        return 0
    task_dirs = _apply_task_include(_discover_task_dirs(dataset_paths))
    valid_tasks = _load_valid_tasks(task_dirs)
    identities = _build_task_identities(
        valid_tasks,
        skip_invalid=_get_env('HARBOR_SKIP_INVALID_TASKS', False),
    )
    limit_val = os.getenv('TASK_LIMIT')
    limit = int(limit_val) if limit_val else None
    skip = _get_env('TASK_SKIP', 0)
    count = max(0, len(identities) - skip)
    if limit is not None:
        return min(count, limit)
    return count


def get_batch_metadata() -> dict[str, Any]:
    """Metadata for the batch run."""
    try:
        dataset_paths = _parse_dataset_paths()
        dataset_name = _get_dataset_name(dataset_paths)
    except ValueError:
        dataset_name = 'unknown'
    client_name = _get_env('OPENAI_CLIENT_NAME', DEFAULTS['openai_client_name'])
    cache_enabled = bool(
        client_name == 'litellm' and _get_env('OPENAI_CACHE_CONTROL_ENABLED', DEFAULTS['openai_cache_control_enabled'])
    )
    return {
        'benchmark': dataset_name,
        'bridge': 'harbor',
        'task_name': 'harbor_bridge_task',
        'agent_name': _get_env('AGENT_NAME', DEFAULTS['agent_name']),
        'model': _get_env('OPENAI_MODEL', DEFAULTS['model']),
        'max_iterations': _get_env('MAX_ITERATIONS', DEFAULTS['max_iterations']),
        'prompt_cache_enabled': cache_enabled,
        **(
            {
                'prompt_cache_strategy': _get_env(
                    'OPENAI_CACHE_CONTROL_STRATEGY',
                    DEFAULTS['openai_cache_control_strategy'],
                )
            }
            if cache_enabled
            else {}
        ),
    }
