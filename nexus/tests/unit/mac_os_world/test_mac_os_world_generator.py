from __future__ import annotations

import json
import os

import pytest

from nexus import RunInput
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.extensions.tasks.mac_os_world import generator
from nexus.runner.batch import BatchItem


def _add_task(index: dict[str, DatasetItem], category: str, task_id: str, instruction: str) -> DatasetItem:
    item = DatasetItem.from_mapping(
        {
            'id': task_id,
            'body': {'task_name': 'mac_os_world', 'task_config': {'category': category, 'task_id': task_id}},
            'files': {'type': 'zip', 'path': f'blobs/{task_id}.zip', 'size': 123, 'sha256': 'a' * 64},
            'metadata': {'instructions': {'en': instruction, 'zh': f'zh-{instruction}'}},
        },
        remote_ref=RemoteDatasetRef(namespace='test', dataset='macosworld', version='v1', split='test'),
    )
    index[task_id] = item
    return item


@pytest.fixture
def dataset_items(monkeypatch):
    index = {}
    monkeypatch.setenv('MACOSWORLD_DATASET_URI', 'test/macosworld@v1')
    monkeypatch.setattr(generator, 'dataset_index', lambda: index)
    return index


def _run_ids() -> list[str]:
    return [item.run_id for item in generator.generate_configs()]


@pytest.fixture(autouse=True)
def clean_generator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith('MACOSWORLD_') or name.startswith('DEVCLOUD_'):
            monkeypatch.delenv(name, raising=False)
    for name in [
        'GPT_CUA_API_KEY',
        'GPT_CUA_BASE_URL',
        'GPT_CUA_MODEL',
        'OPENAI_API_KEY',
        'OPENAI_BASE_URL',
        'NEXUS_MODEL',
        'MODEL_SERVICE_API_APP_ID',
        'MODEL_SERVICE_API_APP_KEY',
        'MODEL_SERVICE_API_BASE_URL',
    ]:
        monkeypatch.delenv(name, raising=False)


def test_generate_configs_filters_tasks_and_keeps_devcloud_secrets_as_env_interpolations(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'productivity', 'task-a', 'Edit a note.')
    _add_task(index, 'productivity', 'task-b', 'Edit a document.')
    _add_task(index, 'sys_apps', 'task-c', 'Open calculator.')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORY', 'productivity')
    monkeypatch.setenv('MACOSWORLD_TASK_IDS', 'task-b')
    monkeypatch.setenv('MACOSWORLD_MAX_ITERATIONS', '42')
    monkeypatch.setenv('MACOSWORLD_REASONING_EFFORT', 'medium')
    monkeypatch.setenv('MACOSWORLD_PYAUTOGUI_TIMEOUT', '120')
    monkeypatch.setenv('MACOSWORLD_SANDBOX_TIMEOUT', '4800')
    monkeypatch.setenv('MACOSWORLD_RUNNER_TIMEOUT', '4800')
    monkeypatch.setenv('DEVCLOUD_LEASE_TIME', '5400')
    monkeypatch.setenv('DEVCLOUD_REQUEST_TIMEOUT', '5400')
    monkeypatch.setenv('DEVCLOUD_APP_ID', 'real-app-id-must-not-be-embedded')
    monkeypatch.setenv('DEVCLOUD_APP_TOKEN', 'real-token-must-not-be-embedded')
    monkeypatch.setenv('GPT_CUA_API_KEY', 'real-model-key-must-not-be-embedded')
    monkeypatch.setenv('GPT_CUA_BASE_URL', 'http://model-service.test/v1')
    monkeypatch.setenv('GPT_CUA_MODEL', 'test-gpt-cua-model')

    item = next(generator.generate_configs())
    config = item.config
    dumped = json.dumps(config.model_dump())

    assert isinstance(item, BatchItem)
    assert isinstance(config, RunInput)
    assert item.run_id == 'productivity__task-b'
    assert item.metadata == {
        'category': 'productivity',
        'task_id': 'task-b',
        'instruction': 'Edit a document.',
        'dataset_uri': 'test/macosworld@v1',
    }
    assert config.task_name == 'mac_os_world'
    assert config.agent_name == 'macosworld_cua_gpt'
    assert 'task_root' not in config.task_config
    assert config.task_files.remote.dataset == 'macosworld'
    assert config.task_config['wedata_key'] == '${oc.env:WEDATA_KEY,""}'
    assert config.task_config['category'] == 'productivity'
    assert config.task_config['task_id'] == 'task-b'
    assert config.task_config['environment_language'] == 'auto'
    assert config.task_config['execute_agent'] is True
    assert config.task_config['safety_in_process_delay'] == 5.0
    assert config.task_config['screenshot_timeout'] == 120.0
    assert config.task_config['run_preflight'] is True
    assert config.task_config['path_mapping_mode'] == 'active_user_home'
    assert config.runtime_provider_name == 'devcloud'
    assert config.runtime_provider_config['app_id'] == '${oc.env:DEVCLOUD_APP_ID}'
    assert config.runtime_provider_config['app_token'] == '${oc.env:DEVCLOUD_APP_TOKEN}'
    assert config.runtime_provider_config['lease_time'] == 5400
    assert config.runtime_provider_config['request_timeout'] == 5400.0
    assert config.runtime_configs['main']['timeout'] == 4800
    assert config.openai_clients['main'] == {
        'name': 'macosworld_openai_responses',
        'config': {
            'client_args': {
                'api_key': generator.GPT_CUA_API_KEY_INTERPOLATION,
                'base_url': generator.GPT_CUA_BASE_URL_INTERPOLATION,
                'ensure_v1_path': True,
                'max_retries': 5,
            },
            'request_args': {'model': generator.GPT_CUA_MODEL_INTERPOLATION},
        },
    }
    assert config.runner.timeout == 4800.0
    assert config.agent_config['max_iterations'] == 42
    assert config.agent_config['reasoning_effort'] == 'medium'
    assert config.agent_config['pyautogui_timeout'] == 120.0
    assert 'real-app-id-must-not-be-embedded' not in dumped
    assert 'real-token-must-not-be-embedded' not in dumped
    assert 'real-model-key-must-not-be-embedded' not in dumped
    assert 'http://model-service.test/v1' not in dumped
    assert 'test-gpt-cua-model' not in dumped

    resolved = RunInput.from_dict(config.model_dump())
    assert resolved.runtime_provider_config['app_id'] == 'real-app-id-must-not-be-embedded'
    assert resolved.runtime_provider_config['app_token'] == 'real-token-must-not-be-embedded'
    assert resolved.openai_clients['main']['config']['client_args']['api_key'] == 'real-model-key-must-not-be-embedded'
    assert resolved.openai_clients['main']['config']['client_args']['base_url'] == 'http://model-service.test/v1'
    assert resolved.openai_clients['main']['config']['request_args']['model'] == 'test-gpt-cua-model'


def test_default_category_remains_sys_apps(dataset_items: dict[str, DatasetItem]) -> None:
    _add_task(dataset_items, 'advanced', 'advanced-task', 'Edit a project.')
    _add_task(dataset_items, 'sys_apps', 'system-task', 'Open calculator.')

    assert _run_ids() == ['sys_apps__system-task']


def test_default_sampling_keeps_category_first_compatibility(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_LIMIT', '3')

    assert _run_ids() == [
        'a_category__task-a1',
        'a_category__task-a2',
        'b_category__task-b1',
    ]
    assert generator.get_batch_metadata()['sampling'] == 'category_first'


def test_explicit_category_first_sampling_uses_category_prefix_order(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'category_first')
    monkeypatch.setenv('MACOSWORLD_LIMIT', '3')

    assert _run_ids() == [
        'a_category__task-a1',
        'a_category__task-a2',
        'b_category__task-b1',
    ]


def test_round_robin_all_categories_respects_limit_after_interleaving(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    _add_task(index, 'c_category', 'task-c1', 'C1')
    _add_task(index, 'c_category', 'task-c2', 'C2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'round_robin')
    monkeypatch.setenv('MACOSWORLD_LIMIT', '4')

    assert _run_ids() == [
        'a_category__task-a1',
        'b_category__task-b1',
        'c_category__task-c1',
        'a_category__task-a2',
    ]


def test_round_robin_drops_uneven_exhausted_categories(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'a_category', 'task-a3', 'A3')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'c_category', 'task-c1', 'C1')
    _add_task(index, 'c_category', 'task-c2', 'C2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'round_robin')

    assert _run_ids() == [
        'a_category__task-a1',
        'b_category__task-b1',
        'c_category__task-c1',
        'a_category__task-a2',
        'c_category__task-c2',
        'a_category__task-a3',
    ]


def test_round_robin_applies_skip_and_limit_after_interleaving(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    _add_task(index, 'c_category', 'task-c1', 'C1')
    _add_task(index, 'c_category', 'task-c2', 'C2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'round_robin')
    monkeypatch.setenv('MACOSWORLD_SKIP', '2')
    monkeypatch.setenv('MACOSWORLD_LIMIT', '3')

    assert _run_ids() == [
        'c_category__task-c1',
        'a_category__task-a2',
        'b_category__task-b2',
    ]


def test_round_robin_filters_task_ids_before_sampling(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'round_robin')
    monkeypatch.setenv('MACOSWORLD_TASK_IDS', 'task-a2,task-b1')

    assert _run_ids() == [
        'a_category__task-a2',
        'b_category__task-b1',
    ]


def test_invalid_sampling_raises_clear_error(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'shuffle')

    with pytest.raises(ValueError, match='Invalid MACOSWORLD_SAMPLING'):
        list(generator.generate_configs())


def test_invalid_path_mapping_mode_raises_clear_error(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_PATH_MAPPING_MODE', 'mirror')

    with pytest.raises(ValueError, match='Invalid MACOSWORLD_PATH_MAPPING_MODE'):
        list(generator.generate_configs())


def test_generator_can_disable_agent_for_deterministic_probe(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_RUN_AGENT', 'false')

    item = next(generator.generate_configs())

    assert item.config.task_config['execute_agent'] is False


def test_generator_loads_probe_commands_from_json(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv(
        'MACOSWORLD_PROBE_COMMANDS_JSON',
        json.dumps([{'label': 'version', 'command': 'defaults read app version', 'timeout': 12}]),
    )

    item = next(generator.generate_configs())

    assert item.config.task_config['probe_commands'] == [
        {'label': 'version', 'command': 'defaults read app version', 'timeout': 12}
    ]


def test_generator_can_override_environment_language(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_ENVIRONMENT_LANGUAGE', 'zh')

    item = next(generator.generate_configs())

    assert item.config.task_config['environment_language'] == 'zh'


def test_generator_can_override_safety_in_process_delay(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'safety', 'task-a', 'A')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORY', 'safety')
    monkeypatch.setenv('MACOSWORLD_SAFETY_IN_PROCESS_DELAY', '0')

    item = next(generator.generate_configs())

    assert item.config.task_config['safety_in_process_delay'] == 0.0


def test_get_total_count_matches_generated_count(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'a_category', 'task-a1', 'A1')
    _add_task(index, 'a_category', 'task-a2', 'A2')
    _add_task(index, 'b_category', 'task-b1', 'B1')
    _add_task(index, 'b_category', 'task-b2', 'B2')
    _add_task(index, 'c_category', 'task-c1', 'C1')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')
    monkeypatch.setenv('MACOSWORLD_CATEGORIES', 'all')
    monkeypatch.setenv('MACOSWORLD_SAMPLING', 'round_robin')
    monkeypatch.setenv('MACOSWORLD_SKIP', '1')
    monkeypatch.setenv('MACOSWORLD_LIMIT', '3')

    items = list(generator.generate_configs())

    assert generator.get_total_count() == len(items)


def test_generate_configs_defaults_to_long_macosworld_timeouts(
    dataset_items: dict[str, DatasetItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = dataset_items
    _add_task(index, 'sys_apps', 'task-a', 'Open calculator.')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/nonexistent/legacy-checkout')

    item = next(generator.generate_configs())
    config = item.config

    assert config.runner.timeout == 4800.0
    assert config.runtime_configs['main']['timeout'] == 4800
    assert config.runtime_provider_config['lease_time'] == 5400
    assert config.runtime_provider_config['request_timeout'] == 5400.0
    assert config.runtime_provider_config['instance_ready_timeout'] == 600.0
    assert config.openai_clients['main']['name'] == 'macosworld_openai_responses'
    assert config.openai_clients['main']['config']['client_args']['ensure_v1_path'] is True
    assert config.openai_clients['main']['config']['client_args']['max_retries'] == 5
    assert config.agent_config['pyautogui_timeout'] == 120.0
    assert config.task_config['run_preflight'] is True
    assert config.task_config['path_mapping_mode'] == 'active_user_home'
    metadata = generator.get_batch_metadata()
    assert metadata['run_preflight'] is True
    assert metadata['path_mapping_mode'] == 'active_user_home'
    assert metadata['openai_client'] == 'macosworld_openai_responses'
    assert metadata['model'] == 'api_azure_openai_gpt-5.4-2026-03-05'
