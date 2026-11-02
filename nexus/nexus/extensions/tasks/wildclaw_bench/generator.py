#!/usr/bin/env python3
"""WildClawBench batch generator for ``nexus run-batch``.

Generates batch configs for all WildClawBench tasks, supporting automatic task
discovery by category, flexible LLM configuration via environment variables,
and parallel execution.

Usage:
  nexus run-batch -g nexus/extensions/tasks/wildclaw_bench/generator.py \\
      -d ./outputs/wildclaw_bench.db --max-concurrent 4

Environment variables:
  - WILDCLAW_ROOT: Root directory of WildClawBench dataset (required)
  - WILDCLAW_CATEGORY: Category to run (e.g. 01_Productivity_Flow, or 'all')
  - WILDCLAW_MODALITY: Filter by modality – 'all' (default), 'multimodal', or
    'text_only'.  'multimodal' selects tasks whose markdown mentions the
    OpenRouter API for image/multimodal capabilities; 'text_only' selects the
    remaining tasks that do NOT require such capabilities.
  - WILDCLAW_TASK_IDS: Comma-separated task IDs to run (default: all tasks)
  - WILDCLAW_SKIP: Number of tasks to skip (default: 0)
  - WILDCLAW_LIMIT: Maximum number of tasks to run (default: no limit)
  - WILDCLAW_IMAGE: Docker image to use (required, e.g. wildclawbench-ubuntu:v0.4)
  - WILDCLAW_FORCE_REBUILD: "true" to force rebuild the image (default: false)
  - WILDCLAW_TIMEOUT: Override timeout per task in seconds (default: from task frontmatter)
  - WILDCLAW_TIMEOUT_MULTIPLIER: Multiplier for timeout (default: 1.0)
  - OPENCLAW_VERSION: OpenClaw agent version (default: 2026.3.11)
  - OPENCLAW_MODEL: Main agent model (required)
  - OPENCLAW_LITELLM_API_KEY: API key for LLM (required)
  - OPENCLAW_LITELLM_BASE_URL: API base URL (required)
  - OPENCLAW_NPM_REGISTRY: NPM registry URL (optional)
  - OPENCLAW_CONTEXT_WINDOW: Context window size (default: 200000)
  - OPENCLAW_MAX_TOKENS: Maximum output tokens (default: 65536)
  - OPENCLAW_SUPPORTS_IMAGES: Whether model supports image input (default: false)
  - WILDCLAW_REASONING_EFFORT: Reasoning effort (e.g. low/medium/high/none)
"""

from __future__ import annotations

import logging
import os
import re

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nexus import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput


logger = logging.getLogger(__name__)

# All WildClawBench categories (duplicated from task.py because generator.py
# is loaded via importlib file path by the batch runner, not as a package import)
ALL_CATEGORIES = [
    '01_Productivity_Flow',
    '02_Code_Intelligence',
    '03_Social_Interaction',
    '04_Search_Retrieval',
    '05_Creative_Synthesis',
    '06_Safety_Alignment',
]

SERPER_BRIDGE_ENV_VARS = (
    'SERPER_API_KEY',
    'SERPER_API_TYPE',
    'SEARCH_SERPER_URL',
    'SEARCH_SERPER_APP_ID',
    'SEARCH_SERPER_APP_KEY',
    'SERPER_KEY_ID',
)


# ============================================================================
# Environment Configuration Helpers
# ============================================================================


def _env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Get integer from environment variable with fallback."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float = 1.0) -> float:
    """Get float from environment variable with fallback."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str] | None:
    """Parse comma-separated list from environment variable."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return items if items else None


def _env_path(name: str, default: Path | None = None) -> Path | None:
    """Get path from environment variable with fallback."""
    v = os.getenv(name)
    if not v:
        return default
    return Path(v)


# ============================================================================
# Task Discovery and Parsing
# ============================================================================


def _load_task_meta(task_file: Path) -> dict:
    """Load task metadata from YAML frontmatter.

    Args:
        task_file: Path to the task markdown file

    Returns:
        Dictionary containing the parsed frontmatter

    Raises:
        ValueError: If no YAML frontmatter is found
    """
    content = task_file.read_text(encoding='utf-8')
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        raise ValueError(f'No YAML frontmatter in {task_file}')
    return yaml.safe_load(match.group(1))


def _discover_tasks(root_dir: Path, categories: list[str] | None = None) -> list[Path]:
    """Discover task files in the WildClawBench dataset.

    Args:
        root_dir: Root directory of WildClawBench dataset
        categories: Optional list of categories to filter

    Returns:
        Sorted list of task file paths
    """
    tasks_dir = root_dir / 'tasks'
    if not tasks_dir.exists():
        raise ValueError(f'Tasks directory not found: {tasks_dir}')

    target_categories = categories or ALL_CATEGORIES
    task_files: list[Path] = []

    for category in target_categories:
        cat_dir = tasks_dir / category
        if not cat_dir.exists():
            logger.warning('Category directory not found: %s', cat_dir)
            continue
        cat_files = sorted(cat_dir.glob('*task_*.md'))
        task_files.extend(cat_files)
        logger.debug('Category %s: %d task files', category, len(cat_files))

    logger.debug('Total discovered: %d task files', len(task_files))
    return task_files


# Hard-coded set of 25 multimodal tasks that require image/video understanding.
# These tasks mention the OpenRouter API for image understanding or multimodal
# generation capabilities in their markdown description.
# Last updated: 2026-03-27
_MULTIMODAL_TASK_STEMS: frozenset[str] = frozenset({
    '01_Productivity_Flow_task_3_bibtex',
    '01_Productivity_Flow_task_8_real_image_category',
    '02_Code_Intelligence_task_3_jigsaw_puzzle_zh',
    '02_Code_Intelligence_task_4_jigsaw_puzzle_medium_zh',
    '02_Code_Intelligence_task_5_jigsaw_puzzle_hard_zh',
    '02_Code_Intelligence_task_6_benchmark_vlmeval_ocrbench_zh',
    '02_Code_Intelligence_task_7_connect_the_dots_medium_img_zh',
    '02_Code_Intelligence_task_8_link_a_pix_color_zh',
    '02_Code_Intelligence_task_9_link_a_pix_color_easy_zh',
    '02_Code_Intelligence_task_10_acad_homepage_zh',
    '02_Code_Intelligence_task_11_resume_homepage_zh',
    '02_Code_Intelligence_task_12_connect_the_dots_hard_zh',
    '04_Search_Retrieval_task_7_location_search',
    '04_Search_Retrieval_task_9_artwork_search',
    '05_Creative_Synthesis_task_1_match_report',
    '05_Creative_Synthesis_task_2_goal_highlights',
    '05_Creative_Synthesis_task_3_product_poster',
    '05_Creative_Synthesis_task_4_video_notes',
    '05_Creative_Synthesis_task_5_product_launch_video_to_json',
    '05_Creative_Synthesis_task_6_clothing_outfit_to_model_image',
    '05_Creative_Synthesis_task_7_paper_to_poster',
    '05_Creative_Synthesis_task_8_repo_to_homepage',
    '05_Creative_Synthesis_task_9_repo_to_slides',
    '05_Creative_Synthesis_task_10_social_poster_multi_crop',
    '05_Creative_Synthesis_task_11_video_en_to_zh_dub',
})


def _is_multimodal_task(task_file: Path) -> bool:
    """Check whether a task requires multimodal / image understanding.

    Uses the hard-coded ``_MULTIMODAL_TASK_STEMS`` set for O(1) lookup.
    """
    return task_file.stem in _MULTIMODAL_TASK_STEMS


def _filter_by_modality(task_files: list[Path], modality: str) -> list[Path]:
    """Filter task files by modality category.

    Args:
        task_files: List of task file paths
        modality: One of 'all', 'multimodal', or 'text_only'

    Returns:
        Filtered list of task file paths
    """
    if modality == 'all':
        return task_files

    if modality == 'multimodal':
        filtered = [f for f in task_files if _is_multimodal_task(f)]
    elif modality == 'text_only':
        filtered = [f for f in task_files if not _is_multimodal_task(f)]
    else:
        logger.warning("Unknown WILDCLAW_MODALITY '%s', treating as 'all'", modality)
        return task_files

    logger.debug('Modality filter (%s): %d → %d tasks', modality, len(task_files), len(filtered))
    return filtered


def _filter_tasks_by_ids(task_files: list[Path], task_ids: list[str]) -> list[Path]:
    """Filter task files by specified task IDs.

    Args:
        task_files: List of all task file paths
        task_ids: List of task IDs to filter by

    Returns:
        Filtered list of task file paths
    """
    task_map: dict[str, Path] = {}
    for f in task_files:
        task_map[f.stem] = f
        try:
            meta = _load_task_meta(f)
            fid = meta.get('id')
            if fid:
                task_map[fid] = f
        except Exception as e:
            logger.debug('Failed to load metadata for task %s: %s', f.stem, e)
            pass

    filtered = []
    for tid in task_ids:
        if tid not in task_map:
            raise ValueError(f"Task ID '{tid}' not found in tasks directory")
        if task_map[tid] not in filtered:
            filtered.append(task_map[tid])

    return filtered


def _resolve_task_files() -> tuple[Path, list[Path]]:
    """Resolve task files based on environment configuration.

    Returns:
        Tuple of (root_dir, list of task file paths)

    Raises:
        ValueError: If root directory not set or doesn't exist
    """
    root_dir = _env_path('WILDCLAW_ROOT')
    if root_dir is None:
        raise ValueError('WILDCLAW_ROOT environment variable is required')
    if not root_dir.exists():
        raise ValueError(f'WildClawBench root directory not found: {root_dir}')

    # Determine categories
    category_str = os.getenv('WILDCLAW_CATEGORY', 'all')
    categories = None if category_str.lower() == 'all' else [c.strip() for c in category_str.split(',') if c.strip()]

    # Discover tasks
    task_files = _discover_tasks(root_dir, categories)

    # Filter by modality (multimodal vs text-only)
    modality = os.getenv('WILDCLAW_MODALITY', 'all').strip().lower()
    task_files = _filter_by_modality(task_files, modality)

    # Filter by explicit task IDs if provided
    explicit_ids = _env_list('WILDCLAW_TASK_IDS')
    if explicit_ids:
        logger.debug('Using explicit task IDs: %s', explicit_ids)
        task_files = _filter_tasks_by_ids(task_files, explicit_ids)

    # Apply skip/limit
    skip = _env_int('WILDCLAW_SKIP', 0) or 0
    limit = _env_int('WILDCLAW_LIMIT')

    task_files = task_files[skip:]
    if limit is not None:
        task_files = task_files[:limit]

    logger.debug('Resolved %d tasks to run (skip=%d, limit=%s)', len(task_files), skip, limit)
    return root_dir, task_files


# ============================================================================
# Configuration Builders
# ============================================================================


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client configurations from environment variables.

    Returns:
        Dict with 'main' client configuration
    """
    api_key = os.getenv('OPENCLAW_LITELLM_API_KEY', '')
    base_url = os.getenv('OPENCLAW_LITELLM_BASE_URL', '')
    model = os.getenv('OPENCLAW_MODEL', '')
    if not api_key or not base_url or not model:
        raise ValueError(
            'OPENCLAW_LITELLM_API_KEY, OPENCLAW_LITELLM_BASE_URL, and OPENCLAW_MODEL environment variables are required'
        )

    # Build main client request_args
    main_request_args: dict[str, Any] = {
        'model': model,
    }

    # Inject reasoning_effort if specified
    reasoning_effort = os.getenv('WILDCLAW_REASONING_EFFORT', '')
    if reasoning_effort.lower() == 'none':
        reasoning_effort = None
    if reasoning_effort and reasoning_effort.lower() not in ('', 'none'):
        chat_template_kwargs: dict[str, Any] = {
            'reasoning_effort': reasoning_effort,
        }
        # Only enable interleaved_thinking for actual thinking modes (not no_think)
        # if reasoning_effort.lower() != 'no_think':
        #     chat_template_kwargs['interleaved_thinking'] = True
        main_request_args['extra_body'] = {
            'chat_template_kwargs': chat_template_kwargs,
        }

    clients: dict[str, dict[str, Any]] = {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': main_request_args,
            },
        },
    }

    return clients


def _build_env_vars(task_file: Path) -> dict[str, str]:
    """Build extra environment variables for the task container.

    Reads the ## Env section from the task file and resolves the variables
    from the host environment, matching the original docker_utils.start_container().

    Args:
        task_file: Path to the task markdown file

    Returns:
        Dict of environment variable name → value
    """
    content = task_file.read_text(encoding='utf-8')
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not fm_match:
        return {}

    body = fm_match.group(2)
    sections: dict[str, str] = {}
    current_section: str | None = None
    lines: list[str] = []

    for line in body.split('\n'):
        header = re.match(r'^##\s+(.+)$', line)
        if header:
            if current_section is not None:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = header.group(1)
            lines = []
        else:
            lines.append(line)
    if current_section is not None:
        sections[current_section] = '\n'.join(lines).strip()

    def strip_codeblock(raw: str) -> str:
        s = re.sub(r'^```[^\n]*\n?', '', raw.strip())
        return re.sub(r'\n?```$', '', s).strip()

    env_section = strip_codeblock(sections.get('Env', ''))
    env_vars: dict[str, str] = {}

    # NOTE: Proxy variables (HTTP_PROXY_INNER etc.) are intentionally NOT
    # passed here.  The OpenClaw agent communicates with the LLM via a
    # local Socket.IO proxy on 127.0.0.1, so injecting an HTTP proxy into
    # the container causes the requests to be hijacked by the corporate
    # gateway (igate) and fail with 403.  This matches the PinchBench
    # approach which also does not pass proxy env vars.

    # Add BRAVE_API_KEY if set
    brave_key = os.getenv('BRAVE_API_KEY', '')
    if brave_key:
        env_vars['BRAVE_API_KEY'] = brave_key
    # Add Serper bridge configuration if set.
    for var in SERPER_BRIDGE_ENV_VARS:
        value = os.getenv(var, '')
        if value:
            env_vars[var] = value
    # Add GEMINI_API_KEY if set
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    if gemini_key:
        env_vars['GEMINI_API_KEY'] = gemini_key

    # Always inject grading-related env vars if set.
    # Many task ## Env sections only declare OPENROUTER_API_KEY, but the
    # grade() functions inside Automated Checks also read GRADING_MODEL,
    # OPENROUTER_BASE_URL, and OPENROUTER_BASE_URL_COMP.  Without these
    # the LLM Judge calls inside the container fail with 401 or fall back
    # to unreachable default endpoints (openrouter.ai).
    grading_env_vars = [
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL',
        'OPENROUTER_BASE_URL_COMP',
        'GRADING_MODEL',
    ]
    for var in grading_env_vars:
        value = os.getenv(var, '')
        if value:
            env_vars[var] = value

    # Add task-specific env vars
    for line in env_section.splitlines():
        key = line.strip()
        if not key or key.startswith('#'):
            continue
        value = os.getenv(key, '')
        env_vars[key] = value

    return env_vars


def _build_run_input(
    task_file: Path,
    root_dir: Path,
) -> tuple[RunInput, dict]:
    """Build RunInput configuration for a WildClawBench task.

    Args:
        task_file: Path to the task markdown file
        root_dir: Root directory of WildClawBench dataset

    Returns:
        Tuple of (RunInput instance, metadata dict)
    """
    meta = _load_task_meta(task_file)
    task_id = meta.get('id', task_file.stem)

    # Timeout handling
    timeout = _env_int('WILDCLAW_TIMEOUT') or meta.get('timeout_seconds', 300)
    timeout_multiplier = _env_float('WILDCLAW_TIMEOUT_MULTIPLIER', 1.0)
    effective_timeout = int(timeout * timeout_multiplier)

    # Build configurations
    openai_clients = _build_openai_clients()

    # Docker image configuration
    docker_image = os.getenv('WILDCLAW_IMAGE', '')
    if not docker_image:
        raise ValueError('WILDCLAW_IMAGE environment variable is required')

    # Agent configuration
    model = os.getenv('OPENCLAW_MODEL', '')
    if not model:
        raise ValueError('OPENCLAW_MODEL environment variable is required')

    agent_config: dict[str, Any] = {
        'working_dir': '/tmp_workspace',
        'timeout': effective_timeout,
        'version': os.getenv('OPENCLAW_VERSION', '2026.3.11'),
        'model': model,
        'npm_registry': os.getenv('OPENCLAW_NPM_REGISTRY', ''),
        'context_window': _env_int('OPENCLAW_CONTEXT_WINDOW', 200_000) or 200_000,
        'max_tokens': _env_int('OPENCLAW_MAX_TOKENS', 65_536) or 65_536,
        'supports_images': os.getenv('OPENCLAW_SUPPORTS_IMAGES', '').lower() in ('1', 'true', 'yes'),
    }

    # Resolve workspace path for Docker mount
    # The original pipeline mounts workspace_path/exec → /app
    content = task_file.read_text(encoding='utf-8')
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    body = fm_match.group(2) if fm_match else ''

    sections: dict[str, str] = {}
    current: str | None = None
    sec_lines: list[str] = []
    for line in body.split('\n'):
        hdr = re.match(r'^##\s+(.+)$', line)
        if hdr:
            if current is not None:
                sections[current] = '\n'.join(sec_lines).strip()
            current = hdr.group(1)
            sec_lines = []
        else:
            sec_lines.append(line)
    if current is not None:
        sections[current] = '\n'.join(sec_lines).strip()

    def strip_cb(raw: str) -> str:
        s = re.sub(r'^```[^\n]*\n?', '', raw.strip())
        return re.sub(r'\n?```$', '', s).strip()

    raw_workspace = strip_cb(sections.get('Workspace Path', ''))
    wp = Path(raw_workspace)
    if not wp.is_absolute():
        wp = (root_dir / wp).resolve()
    workspace_path = str(wp)

    # The original mounts workspace_path/exec → /app:ro
    exec_dir = str(Path(workspace_path) / 'exec')

    # Build extra env vars for the container
    env_vars = _build_env_vars(task_file)

    # Task config
    task_config: dict[str, Any] = {
        'task_file': str(task_file.resolve()),
        'wildclaw_root': str(root_dir.resolve()),
        'timeout_multiplier': timeout_multiplier,
    }

    if _env_int('WILDCLAW_TIMEOUT') is not None:
        task_config['timeout'] = _env_int('WILDCLAW_TIMEOUT')

    # Build RunInput
    # NOTE: Proxy env vars are intentionally NOT injected into runtime_env.
    # OpenClaw talks to the LLM through a local Socket.IO proxy on 127.0.0.1;
    # injecting HTTP_PROXY causes requests to be hijacked by igate → 403.
    # This matches the PinchBench approach.
    runtime_env: dict[str, str] = {}

    brave_key = os.getenv('BRAVE_API_KEY', '')
    if brave_key:
        runtime_env['BRAVE_API_KEY'] = brave_key
    for var in SERPER_BRIDGE_ENV_VARS:
        value = os.getenv(var, '')
        if value:
            runtime_env[var] = value
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    if gemini_key:
        runtime_env['GEMINI_API_KEY'] = gemini_key

    # Add task-specific env vars (from ## Env section in task markdown)
    runtime_env.update(env_vars)

    # Build extra Docker CLI arguments for volume mounts and environment variables.
    # DockerRuntimeProvider does NOT natively support 'mounts' or 'env' keys;
    # they must be passed as raw docker CLI flags via 'extra_args' in
    # runtime_configs (which feeds into provider.configure() → acquire()).
    extra_args: list[str] = []

    # Volume mount: workspace_path/exec → /app:ro (matching original pipeline)
    extra_args.extend(['-v', f'{exec_dir}:/app:ro'])

    # Explicitly clear proxy env vars that may be baked into the Docker image.
    # Some WildClawBench images (e.g. wildclawbench-ubuntu) have http_proxy /
    # https_proxy hardcoded in their image layers pointing to an internal proxy
    # that is not reachable from this host.  This causes pip install and other
    # network operations inside warmup to fail with "403 Forbidden".
    # Setting them to empty string effectively disables the proxy.
    for proxy_var in [
        'http_proxy',
        'https_proxy',
        'HTTP_PROXY',
        'HTTPS_PROXY',
        'no_proxy',
        'NO_PROXY',
    ]:
        extra_args.extend(['-e', f'{proxy_var}='])

    # Environment variables
    for env_key, env_value in runtime_env.items():
        extra_args.extend(['-e', f'{env_key}={env_value}'])

    # HTTP request_timeout must exceed agent timeout + runtime overhead
    # (OpenClaw npm install, onboard, LLM warm-up, grading, etc.)
    # Add 900s buffer to the effective agent timeout.
    _env_req_timeout = os.getenv('WILDCLAW_REQUEST_TIMEOUT')
    request_timeout = float(_env_req_timeout) if _env_req_timeout else (effective_timeout + 900)

    config = RunInput(
        imports=['nexus.extensions'],
        task_name='wildclaw_bench',
        task_config=task_config,
        runtime_provider_name='docker',
        runtime_provider_config={
            'image': docker_image,
            'runtime_image': 'current',
            'force_rebuild': _env_bool('WILDCLAW_FORCE_REBUILD'),
            'request_timeout': request_timeout,
        },
        runtime_configs={
            'main': {
                'extra_args': extra_args,
            },
        },
        agent_name='openclaw',
        agent_config=agent_config,
        openai_clients=openai_clients,
        processors={
            'chat': ProcessorConfig(name='chat_tracker'),
        },
    )

    # Build metadata
    metadata = {
        'task_id': task_id,
        'task_name': meta.get('name', 'Unknown'),
        'category': meta.get('category', task_file.parent.name),
        'timeout_seconds': effective_timeout,
        'model': model,
        'workspace_path': workspace_path,
    }

    return config, metadata


# ============================================================================
# Main Generator Functions
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for WildClawBench evaluation.

    Reads configuration from environment variables, discovers task files,
    and yields BatchItem instances ready for execution.

    Yields:
        BatchItem: Configured batch items for each task

    Raises:
        ValueError: If task discovery or configuration fails
    """
    root_dir, task_files = _resolve_task_files()

    logger.debug('Generating configs for %d WildClawBench tasks', len(task_files))

    for task_file in task_files:
        try:
            config, metadata = _build_run_input(task_file, root_dir)
            task_id = metadata['task_id']

            yield BatchItem(
                config=config,
                run_id=task_id,
                metadata=metadata,
            )

        except Exception as e:
            logger.error('Failed to generate config for %s: %s', task_file, e)
            raise

    logger.debug('Generated %d batch items for WildClawBench', len(task_files))


def get_total_count() -> int:
    """Return total number of tasks that will be generated.

    Returns:
        Total count of tasks to be generated
    """
    try:
        _, task_files = _resolve_task_files()
        return len(task_files)
    except Exception as e:
        logger.error('Failed to count tasks: %s', e)
        return 0


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata for tracking and debugging.

    Returns:
        Dictionary containing batch configuration and environment info
    """
    root_dir = _env_path('WILDCLAW_ROOT')

    return {
        'task_type': 'wildclaw_bench',
        'root_dir': str(root_dir) if root_dir else '',
        'category': os.getenv('WILDCLAW_CATEGORY', 'all'),
        'modality': os.getenv('WILDCLAW_MODALITY', 'all'),
        'model': os.getenv('OPENCLAW_MODEL', ''),
        'agent_version': os.getenv('OPENCLAW_VERSION', '2026.3.11'),
        'docker_image': os.getenv('WILDCLAW_IMAGE', ''),
        'skip': _env_int('WILDCLAW_SKIP', 0) or 0,
        'limit': _env_int('WILDCLAW_LIMIT'),
        'explicit_task_ids': _env_list('WILDCLAW_TASK_IDS'),
        'timeout': _env_int('WILDCLAW_TIMEOUT'),
        'timeout_multiplier': _env_float('WILDCLAW_TIMEOUT_MULTIPLIER', 1.0),
    }
