#!/usr/bin/env python3
"""BFCL v4 batch generator for `nexus run-batch`.

Yields BatchItem instances for BFCL test cases, loading data from the BFCL
dataset and ground truth files. Configured via environment variables.

Environment variables:
    BFCL_CATEGORIES: Comma-separated categories or collection names (default: 'all_scoring')
    BFCL_SKIP: Number of entries to skip per category (default: 0)
    BFCL_LIMIT: Max entries per category (default: all)
    BFCL_OPENAI_BASE_URL: OpenAI-compatible API base URL (required)
    BFCL_OPENAI_API_KEY: API key (required)
    BFCL_OPENAI_MODEL: Model name (default: 'deepseek-v3.2')
    BFCL_IS_FC_MODEL: Whether the model supports function calling (default: 'true')
    BFCL_TEMPERATURE: Sampling temperature (default: 0.0)
    BFCL_OPENAI_CLIENTS_NAME: OpenAI client registry name (default: 'openai')
"""

from __future__ import annotations

import logging
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bfcl_eval.utils import (
    load_dataset_entry,
    load_ground_truth_entry,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
    populate_test_cases_with_predefined_functions,
)

from nexus import BatchItem
from nexus.extensions.tasks.bfclv4.types import (
    get_category_type,
    parse_categories,
)
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


# ============================================================================
# Environment Helpers
# ============================================================================


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable '{name}' is not set.")
    return value


# ============================================================================
# Configuration Builders
# ============================================================================


def _build_taiji_client_args() -> dict[str, Any]:
    """Build taiji client_args from BFCL_TAIJI_* env vars."""
    model = os.getenv('BFCL_OPENAI_MODEL', 'deepseek-v3.2')
    return {
        'model': model,
        'namespace': os.getenv('BFCL_TAIJI_NS', 'Production'),
        'service': os.getenv('BFCL_TAIJI_SERVICE', 'stream-server-online-sbs-11644'),
        'authorization': os.getenv('BFCL_TAIJI_AUTH', 'Bearer 7auGXNATFSKl7dF'),
        'wsid': os.getenv('BFCL_TAIJI_WSID', '11644'),
        'url_pattern': os.getenv('BFCL_TAIJI_URL_PATTERN', 'openapi'),
        'timeout': float(os.getenv('BFCL_TAIJI_TIMEOUT', '120')),
        'enable_thinking': _env_bool('BFCL_TAIJI_ENABLE_THINKING', False),
    }


def _build_openai_clients() -> dict[str, Any]:
    """Build OpenAI client configuration from BFCL_OPENAI_* env vars."""
    clients_name = os.getenv('BFCL_OPENAI_CLIENTS_NAME', 'openai')
    model = os.getenv('BFCL_OPENAI_MODEL', 'deepseek-v3.2')

    if clients_name == 'taiji':
        client_args = _build_taiji_client_args()
    else:
        base_url = _require_env('BFCL_OPENAI_BASE_URL')
        api_key = _require_env('BFCL_OPENAI_API_KEY')
        client_args = {'base_url': base_url, 'api_key': api_key}

    logger.debug(f'BFCL OpenAI client: name={clients_name}, model={model}')

    return {
        'main': {
            'name': clients_name,
            'config': {
                'client_args': client_args,
                'request_args': {
                    'model': model,
                },
            },
        },
    }


def _build_task_config(
    test_entry: dict[str, Any],
    test_category: str,
    ground_truth: dict[str, Any] | None,
    model_name: str,
    is_fc_model: bool,
    temperature: float,
    openai_client_config: dict[str, str],
) -> dict[str, Any]:
    """Build per-entry task_config dict."""
    return {
        'test_entry': test_entry,
        'test_category': test_category,
        'ground_truth': ground_truth,
        'model_name': model_name,
        'is_fc_model': is_fc_model,
        'temperature': temperature,
        # Embedded OpenAI client config for building sync client in task
        # (needed because the runner only creates clients when agent_name is set)
        '_openai_client_config': openai_client_config,
    }


# ============================================================================
# Main Generator
# ============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate BatchItem instances for BFCL test cases.

    Loads dataset entries and ground truth per category from the BFCL data,
    then yields BatchItem instances configured for the 'bfcl' task.

    Yields:
        BatchItem: Configured batch items ready for execution
    """
    # Parse configuration
    categories_str = os.getenv('BFCL_CATEGORIES', 'all_scoring')
    categories = parse_categories(categories_str)
    skip = _env_int('BFCL_SKIP', 0) or 0
    limit = _env_int('BFCL_LIMIT')
    model_name = os.getenv('BFCL_OPENAI_MODEL', 'deepseek-v3.2')
    is_fc_model = _env_bool('BFCL_IS_FC_MODEL', True)
    temperature = _env_float('BFCL_TEMPERATURE', 0.0) or 0.0

    logger.debug(
        f'BFCL generator: categories={categories}, skip={skip}, limit={limit}, '
        f'model={model_name}, is_fc={is_fc_model}, temp={temperature}'
    )

    # Build shared config
    openai_clients = _build_openai_clients()

    # Build sync OpenAI client config for embedding in task_config (fallback when
    # ctx.openai_clients is not populated). Not needed for taiji — the primary
    # path through ctx.openai_clients handles TaijiClient directly.
    clients_name = os.getenv('BFCL_OPENAI_CLIENTS_NAME', 'openai')
    if clients_name == 'taiji':
        openai_client_config: dict[str, str] = {}
    else:
        base_url = _require_env('BFCL_OPENAI_BASE_URL')
        api_key = _require_env('BFCL_OPENAI_API_KEY')
        openai_client_config = {'base_url': base_url, 'api_key': api_key}

    # Memory categories need a persistent directory for snapshot files.
    # Prereq entries write memory snapshots here; eval entries read them back.
    model_result_dir = Path(os.getenv('BFCL_MODEL_RESULT_DIR', './workspace/bfcl_memory'))
    model_result_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f'BFCL memory result dir: {model_result_dir.resolve()}')
    # Note: model_result_dir must be converted to str before embedding in task_config
    # because PosixPath is not JSON serializable. The BFCL populate function needs
    # Path, but the stored config needs str. We convert in test_entry after populate.

    generated_count = 0

    for category in categories:
        logger.debug(f'Loading BFCL category: {category}')

        # Load dataset entries
        try:
            entries = load_dataset_entry(category)
        except Exception as e:
            logger.error(f'Failed to load dataset for {category}: {e}')
            continue

        # Populate entries with predefined functions (multi_turn / agentic)
        # and initial settings (web_search show_snippet, memory model_result_dir, etc.)
        entries = populate_test_cases_with_predefined_functions(entries)
        entries = populate_initial_settings_for_web_search_test_cases(entries)
        entries = populate_initial_settings_for_memory_test_cases(entries, model_result_dir)

        # Convert PosixPath objects in initial_config to str for JSON serialization.
        # populate_initial_settings_for_memory_test_cases stores Path objects which
        # are not JSON serializable. The BFCL checker reconstructs Path from str.
        for entry in entries:
            ic = entry.get('initial_config', {})
            for _cls_name, cls_config in ic.items():
                if 'model_result_dir' in cls_config:
                    cls_config['model_result_dir'] = str(cls_config['model_result_dir'])

        # Load ground truth (not needed for irrelevance/relevance)
        category_type = get_category_type(category)
        ground_truths: list[dict] | None = None
        if category_type != 'relevance':
            try:
                ground_truths = load_ground_truth_entry(category)
            except Exception as e:
                logger.error(f'Failed to load ground truth for {category}: {e}')
                continue

        # Apply skip/limit per category
        entries_slice = entries[skip:]
        if limit is not None:
            entries_slice = entries_slice[:limit]

        if ground_truths is not None:
            gt_slice = ground_truths[skip:]
            if limit is not None:
                gt_slice = gt_slice[:limit]
        else:
            gt_slice = [None] * len(entries_slice)

        logger.debug(f'  {category}: {len(entries_slice)} entries (total available: {len(entries)})')

        for idx, test_entry in enumerate(entries_slice):
            test_id = test_entry['id']
            ground_truth = gt_slice[idx] if idx < len(gt_slice) else None

            task_config = _build_task_config(
                test_entry=test_entry,
                test_category=category,
                ground_truth=ground_truth,
                model_name=model_name,
                is_fc_model=is_fc_model,
                temperature=temperature,
                openai_client_config=openai_client_config,
            )

            config = RunInput(
                imports=['nexus.extensions'],
                task_name='bfcl',
                task_config=task_config,
                openai_clients=openai_clients,
                runner={'timeout': 1800},  # 30 min per task  # ty: ignore[invalid-argument-type]
            )

            run_id = f'{category}__{test_id}'

            yield BatchItem(
                config=config,
                run_id=run_id,
                metadata={
                    'test_id': test_id,
                    'test_category': category,
                    'category_type': category_type,
                    'model_name': model_name,
                },
            )
            generated_count += 1

    if generated_count == 0:
        logger.warning('BFCL generator produced 0 items. Check categories and data paths.')
    else:
        logger.info(f'BFCL generator complete: {generated_count} items')


def get_total_count() -> int:
    """Calculate total number of test entries to be generated."""
    categories_str = os.getenv('BFCL_CATEGORIES', 'all_scoring')
    skip = _env_int('BFCL_SKIP', 0) or 0
    limit = _env_int('BFCL_LIMIT')

    try:
        categories = parse_categories(categories_str)
    except Exception:
        return 0

    total = 0
    for category in categories:
        try:
            entries = load_dataset_entry(category)
            count = len(entries) - skip
            if limit is not None:
                count = min(count, limit)
            total += max(0, count)
        except Exception:
            continue

    return total


def get_batch_metadata() -> dict[str, Any]:
    """Collect batch-level metadata for tracking."""
    return {
        'task_type': 'bfcl',
        'categories': os.getenv('BFCL_CATEGORIES', 'all_scoring'),
        'model_name': os.getenv('BFCL_OPENAI_MODEL', 'deepseek-v3.2'),
        'is_fc_model': _env_bool('BFCL_IS_FC_MODEL', True),
        'temperature': _env_float('BFCL_TEMPERATURE', 0.0),
        'skip': _env_int('BFCL_SKIP', 0),
        'limit': _env_int('BFCL_LIMIT'),
        'bfcl_eval_version': getattr(__import__('bfcl_eval', fromlist=['']), '__version__', 'unknown'),
    }
