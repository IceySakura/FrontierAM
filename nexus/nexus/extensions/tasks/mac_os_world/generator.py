"""macOSWorld GPT CUA batch generator for ``nexus run-batch``.

This mirrors the OSWorld generator pattern: generated configs keep secret fields
as ``${oc.env:...}`` interpolations, so batch databases store environment
references rather than credential values.

Set MACOSWORLD_DATASET_URI=macosworld/macosworld@v1.0.0 and optionally
MACOSWORLD_DATASET_SPLIT=test (otherwise use the Dataset default split).
MACOSWORLD_CATEGORIES defaults to sys_apps; set all for the entire split.
MACOSWORLD_TASK_IDS selects explicit IDs within the selected categories.
MACOSWORLD_LIMIT defaults to 10; set it to 0 to include all selected tasks.
Sampling (category_first/round_robin) and skip/limit apply after selection.
Authenticate with WEDATA_KEY or the standard Nexus Wedata key file.

Run from the repository root::

    nexus run-batch -g nexus/extensions/tasks/mac_os_world/generator.py \
        -d outputs/macosworld.db --max-concurrent 1

For a single task, compose a runtime YAML with Dataset identity::

    nexus run-single --use docs/examples/configs/mac_os_world/mac_os_world_gptcua_devcloud.yaml \
        --data macosworld/macosworld@v1.0.0 --id <task-id> -o outputs/single.json

MACOSWORLD_TASK_ROOT no longer selects data. The sandbox still needs its
existing benchmark_files and prepared macOS applications.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nexus import BatchItem
from nexus.dataset.item import DatasetItem
from nexus.extensions.tasks.mac_os_world.dataset_source import dataset_index, dataset_uri, item_instruction
from nexus.runner.single import RunInput, RunnerConfig


DEFAULT_SANDBOX_API_BASE_URL = 'http://21.6.221.82:18800'
DEFAULT_SANDBOX_TIMEOUT = 4800
DEFAULT_DEVCLOUD_REQUEST_TIMEOUT = 5400.0
DEFAULT_DEVCLOUD_LEASE_TIME = 5400
DEFAULT_DEVCLOUD_INSTANCE_READY_TIMEOUT = 600.0
DEFAULT_PYAUTOGUI_TIMEOUT = 120.0
DEVCLOUD_APP_ID_INTERPOLATION = '${oc.env:DEVCLOUD_APP_ID}'
DEVCLOUD_APP_TOKEN_INTERPOLATION = '${oc.env:DEVCLOUD_APP_TOKEN}'
GPT_CUA_API_KEY_INTERPOLATION = (
    '${oc.env:GPT_CUA_API_KEY,${oc.env:OPENAI_API_KEY,'
    '${oc.env:MODEL_SERVICE_API_APP_ID,""}:${oc.env:MODEL_SERVICE_API_APP_KEY,""}}}'
)
GPT_CUA_BASE_URL_INTERPOLATION = (
    '${oc.env:GPT_CUA_BASE_URL,${oc.env:OPENAI_BASE_URL,'
    '${oc.env:MODEL_SERVICE_API_BASE_URL,"http://llm-api.model-eval.woa.com/v1"}}}'
)
GPT_CUA_MODEL_INTERPOLATION = '${oc.env:GPT_CUA_MODEL,${oc.env:NEXUS_MODEL,"api_azure_openai_gpt-5.4-2026-03-05"}}'
_FALSE_ENV_VALUES = {'0', 'false', 'no', 'off'}
DEFAULT_SAMPLING = 'category_first'
SAMPLING_MODES = {DEFAULT_SAMPLING, 'round_robin'}
DEFAULT_PATH_MAPPING_MODE = 'active_user_home'
PATH_MAPPING_MODES = {'off', DEFAULT_PATH_MAPPING_MODE, 'symlink_gui_documents'}


@dataclass(frozen=True)
class MacOSWorldBatchTask:
    category: str
    task_id: str
    item: DatasetItem


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _env_set(name: str) -> set[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    items = {item.strip() for item in value.split(',') if item.strip()}
    return items or None


def _probe_commands() -> list[dict[str, Any]]:
    value = os.getenv('MACOSWORLD_PROBE_COMMANDS_JSON')
    if value is None or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise ValueError('MACOSWORLD_PROBE_COMMANDS_JSON must be a JSON list of objects')
    return parsed


def _sampling() -> str:
    value = os.getenv('MACOSWORLD_SAMPLING')
    if value is None or not value.strip():
        return DEFAULT_SAMPLING

    sampling = value.strip().lower()
    if sampling not in SAMPLING_MODES:
        expected = ', '.join(sorted(SAMPLING_MODES))
        raise ValueError(f'Invalid MACOSWORLD_SAMPLING {value!r}; expected one of: {expected}')
    return sampling


def _path_mapping_mode() -> str:
    value = os.getenv('MACOSWORLD_PATH_MAPPING_MODE')
    if value is None or not value.strip():
        return DEFAULT_PATH_MAPPING_MODE

    mode = value.strip().lower()
    if mode not in PATH_MAPPING_MODES:
        expected = ', '.join(sorted(PATH_MAPPING_MODES))
        raise ValueError(f'Invalid MACOSWORLD_PATH_MAPPING_MODE {value!r}; expected one of: {expected}')
    return mode


def _selected_categories(index: dict[str, DatasetItem]) -> list[str]:
    categories = os.getenv('MACOSWORLD_CATEGORIES') or os.getenv('MACOSWORLD_CATEGORY') or 'sys_apps'
    available = {item.body['task_config']['category'] for item in index.values()}
    selected = (
        sorted(available)
        if categories.strip().lower() == 'all'
        else [item.strip() for item in categories.split(',') if item.strip()]
    )
    missing = set(selected) - available
    if missing:
        raise ValueError(f'Categories absent from selected Dataset split: {sorted(missing)}')
    return list(dict.fromkeys(selected))


def _category_tasks(
    index: dict[str, DatasetItem],
    category: str,
    task_ids: set[str] | None,
) -> Iterator[MacOSWorldBatchTask]:
    for task_id, item in sorted(index.items()):
        if item.body['task_config']['category'] != category:
            continue
        if task_ids is not None and task_id not in task_ids:
            continue
        yield MacOSWorldBatchTask(category=category, task_id=task_id, item=item)


def _category_first_tasks(
    index: dict[str, DatasetItem],
    categories: list[str],
    task_ids: set[str] | None,
) -> Iterator[MacOSWorldBatchTask]:
    for category in categories:
        yield from _category_tasks(index, category, task_ids)


def _round_robin_tasks(
    index: dict[str, DatasetItem],
    categories: list[str],
    task_ids: set[str] | None,
) -> Iterator[MacOSWorldBatchTask]:
    task_groups = [list(_category_tasks(index, category, task_ids)) for category in categories]
    positions = [0] * len(task_groups)
    active_indices = [group_index for group_index, tasks in enumerate(task_groups) if tasks]

    while active_indices:
        next_active_indices = []
        for group_index in active_indices:
            yield task_groups[group_index][positions[group_index]]
            positions[group_index] += 1
            if positions[group_index] < len(task_groups[group_index]):
                next_active_indices.append(group_index)
        active_indices = next_active_indices


def _ordered_tasks(
    index: dict[str, DatasetItem],
    categories: list[str],
    task_ids: set[str] | None,
    sampling: str,
) -> Iterator[MacOSWorldBatchTask]:
    if sampling == DEFAULT_SAMPLING:
        return _category_first_tasks(index, categories, task_ids)
    if sampling == 'round_robin':
        return _round_robin_tasks(index, categories, task_ids)
    raise ValueError(f'Unsupported MacOSWorld sampling mode: {sampling}')


def _apply_skip_limit(
    tasks: Iterator[MacOSWorldBatchTask],
    skip: int,
    limit: int | None,
) -> Iterator[MacOSWorldBatchTask]:
    skipped = 0
    produced = 0
    for task in tasks:
        if skipped < skip:
            skipped += 1
            continue
        yield task
        produced += 1
        if limit and produced >= limit:
            return


def _iter_selected_tasks() -> Iterator[MacOSWorldBatchTask]:
    index = dataset_index()

    task_ids = _env_set('MACOSWORLD_TASK_IDS')
    categories = _selected_categories(index)
    if task_ids and task_ids - index.keys():
        raise ValueError(f'Task IDs absent from selected Dataset split: {sorted(task_ids - index.keys())}')
    sampling = _sampling()
    skip = max(_env_int('MACOSWORLD_SKIP', 0) or 0, 0)
    limit = _env_int('MACOSWORLD_LIMIT', 10)

    yield from _apply_skip_limit(
        _ordered_tasks(index, categories, task_ids, sampling),
        skip,
        limit,
    )


def _runtime_provider_config() -> dict[str, Any]:
    return {
        'app_id': DEVCLOUD_APP_ID_INTERPOLATION,
        'app_token': DEVCLOUD_APP_TOKEN_INTERPOLATION,
        'sandbox_api_base_url': os.getenv('DEVCLOUD_SANDBOX_API_BASE_URL', DEFAULT_SANDBOX_API_BASE_URL),
        'lease_time': _env_int('DEVCLOUD_LEASE_TIME', DEFAULT_DEVCLOUD_LEASE_TIME) or DEFAULT_DEVCLOUD_LEASE_TIME,
        'server_port': _env_int('DEVCLOUD_SERVER_PORT', 5000) or 5000,
        'request_timeout': _env_float('DEVCLOUD_REQUEST_TIMEOUT', DEFAULT_DEVCLOUD_REQUEST_TIMEOUT),
        'instance_ready_timeout': _env_float(
            'DEVCLOUD_INSTANCE_READY_TIMEOUT', DEFAULT_DEVCLOUD_INSTANCE_READY_TIMEOUT
        ),
    }


def _openai_clients() -> dict[str, Any]:
    return {
        'main': {
            'name': 'macosworld_openai_responses',
            'config': {
                'client_args': {
                    'api_key': GPT_CUA_API_KEY_INTERPOLATION,
                    'base_url': GPT_CUA_BASE_URL_INTERPOLATION,
                    'ensure_v1_path': True,
                    'max_retries': 5,
                },
                'request_args': {
                    'model': GPT_CUA_MODEL_INTERPOLATION,
                },
            },
        }
    }


def _build_config(task: MacOSWorldBatchTask) -> dict[str, Any]:
    files = task.item.resolved_task_files()
    assert files is not None  # Validated by dataset_index().
    return {
        'imports': [
            'nexus.extensions.tasks.mac_os_world',
            'nexus.extensions.agents.cua.macosworld_cua.gpt',
        ],
        'task_name': 'mac_os_world',
        'task_files': files.model_dump(exclude_none=True),
        'task_config': {
            'wedata_key': '${oc.env:WEDATA_KEY,""}',
            'category': task.category,
            'task_id': task.task_id,
            'language': os.getenv('MACOSWORLD_LANGUAGE', 'en'),
            'environment_language': os.getenv('MACOSWORLD_ENVIRONMENT_LANGUAGE', 'auto'),
            'observation_type': 'screenshot',
            'action_space': os.getenv('MACOSWORLD_ACTION_SPACE', 'pyautogui'),
            'run_pre_command': _env_bool('MACOSWORLD_RUN_PRE_COMMAND', True),
            'execute_agent': _env_bool('MACOSWORLD_RUN_AGENT', True),
            'probe_commands': _probe_commands(),
            'safety_in_process_delay': _env_float('MACOSWORLD_SAFETY_IN_PROCESS_DELAY', 5.0),
            'screenshot_timeout': _env_float('MACOSWORLD_SCREENSHOT_TIMEOUT', 120.0),
            'run_preflight': _env_bool('MACOSWORLD_RUN_PREFLIGHT', True),
            'path_mapping_mode': _path_mapping_mode(),
            'run_evaluator': _env_bool('MACOSWORLD_RUN_EVALUATOR', True),
            'binary_grading': _env_bool('MACOSWORLD_BINARY_GRADING', True),
            'client_password': os.getenv('MACOSWORLD_CLIENT_PASSWORD', 'password'),
        },
        'runtime_provider_name': 'devcloud',
        'runtime_provider_config': _runtime_provider_config(),
        'runtime_configs': {
            'main': {
                'runtime_kind': 'macosworld_devcloud',
                'timeout': _env_int('MACOSWORLD_SANDBOX_TIMEOUT', DEFAULT_SANDBOX_TIMEOUT) or DEFAULT_SANDBOX_TIMEOUT,
            },
        },
        'openai_clients': _openai_clients(),
        'agent_name': 'macosworld_cua_gpt',
        'agent_config': {
            'max_iterations': _env_int('MACOSWORLD_MAX_ITERATIONS', 30) or 30,
            'reasoning_effort': os.getenv('MACOSWORLD_REASONING_EFFORT', 'high'),
            'client_password': os.getenv('MACOSWORLD_CLIENT_PASSWORD', 'password'),
            'sleep_after_execution': _env_float('MACOSWORLD_SLEEP_AFTER_EXECUTION', 1.0),
            'step_pause': _env_float('MACOSWORLD_STEP_PAUSE', 2.0),
            'pyautogui_execution_mode': os.getenv('MACOSWORLD_PYAUTOGUI_EXECUTION_MODE', 'run_command'),
            'pyautogui_timeout': _env_float('MACOSWORLD_PYAUTOGUI_TIMEOUT', DEFAULT_PYAUTOGUI_TIMEOUT),
            'platform_name': 'macos',
        },
        'processors': {
            'chat': {'name': 'chat_tracker'},
        },
        'runner': RunnerConfig(
            timeout=_env_float('MACOSWORLD_RUNNER_TIMEOUT', float(DEFAULT_SANDBOX_TIMEOUT)),
        ),
    }


def generate_configs() -> Iterator[BatchItem]:
    """Yield GPT CUA macOSWorld ``BatchItem`` configs for current env filters."""
    for task in _iter_selected_tasks():
        yield BatchItem(
            config=RunInput.model_validate(_build_config(task)),
            run_id=f'{task.category}__{task.task_id}',
            metadata={
                'category': task.category,
                'task_id': task.task_id,
                'instruction': item_instruction(task.item, os.getenv('MACOSWORLD_LANGUAGE', 'en')),
                'dataset_uri': dataset_uri(),
            },
        )


def get_total_count() -> int:
    """Return generated item count after filters."""
    return sum(1 for _ in _iter_selected_tasks())


def get_batch_metadata() -> dict[str, Any]:
    """Return batch-level metadata for ``nexus run-batch``."""
    return {
        'task_name': 'mac_os_world',
        'agent_name': 'macosworld_cua_gpt',
        'openai_client': 'macosworld_openai_responses',
        'model': os.getenv('GPT_CUA_MODEL') or os.getenv('NEXUS_MODEL') or 'api_azure_openai_gpt-5.4-2026-03-05',
        'dataset_uri': dataset_uri(),
        'dataset_split': os.getenv('MACOSWORLD_DATASET_SPLIT') or None,
        'categories': _selected_categories(dataset_index()),
        'task_ids': sorted(_env_set('MACOSWORLD_TASK_IDS') or []),
        'sampling': _sampling(),
        'skip': max(_env_int('MACOSWORLD_SKIP', 0) or 0, 0),
        'limit': _env_int('MACOSWORLD_LIMIT', 10),
        'language': os.getenv('MACOSWORLD_LANGUAGE', 'en'),
        'run_evaluator': _env_bool('MACOSWORLD_RUN_EVALUATOR', True),
        'run_preflight': _env_bool('MACOSWORLD_RUN_PREFLIGHT', True),
        'path_mapping_mode': _path_mapping_mode(),
        'runner_timeout': _env_float('MACOSWORLD_RUNNER_TIMEOUT', float(DEFAULT_SANDBOX_TIMEOUT)),
        'sandbox_timeout': _env_int('MACOSWORLD_SANDBOX_TIMEOUT', DEFAULT_SANDBOX_TIMEOUT) or DEFAULT_SANDBOX_TIMEOUT,
        'devcloud_lease_time': _env_int('DEVCLOUD_LEASE_TIME', DEFAULT_DEVCLOUD_LEASE_TIME)
        or DEFAULT_DEVCLOUD_LEASE_TIME,
        'devcloud_request_timeout': _env_float('DEVCLOUD_REQUEST_TIMEOUT', DEFAULT_DEVCLOUD_REQUEST_TIMEOUT),
        'max_iterations': _env_int('MACOSWORLD_MAX_ITERATIONS', 30) or 30,
        'reasoning_effort': os.getenv('MACOSWORLD_REASONING_EFFORT', 'high'),
        'pyautogui_execution_mode': os.getenv('MACOSWORLD_PYAUTOGUI_EXECUTION_MODE', 'run_command'),
        'pyautogui_timeout': _env_float('MACOSWORLD_PYAUTOGUI_TIMEOUT', DEFAULT_PYAUTOGUI_TIMEOUT),
    }
