from pathlib import Path

import pandas as pd
import pytest

from nexus.extensions.tasks.search import generator as search_generator_module


DATASET_FILENAME = 'dataset.parquet'
SEARCH_TASK_NAME = 'search'
SEARCH_AGENT_NAME = 'search_agent'
RESPONSES_TEST_MODEL = 'openai/responses/api_azure_openai_gpt-5'
REASONING_EFFORT_JSON = '{"effort": "high", "summary": "detailed"}'
SEARCH_TOOLSET_VERSION = 'v2'
DATASET_INSTANCE_ID = 'search_00042'
FILTERED_INSTANCE_ID = 'search_00002'
DEFAULT_IMPORTS = ['nexus.extensions']
DATASET_TABLE_QUESTION = 'Find a table'
DATASET_TABLE_ANSWER = 'company=acme'
NORMAL_QUESTION = 'Find the answer'
NORMAL_ANSWER = '42'
OPENAI_TEST_MODEL = 'gpt-4.1-mini'
INFER_PROXY_TEST_MODEL = 'HY3.0_A20B_sftexp7_bus0331_2.38T_wo_mtp_HF'
LITELLM_QWEN_MODEL = 'openai/api_ali_qwen3.5-plus'
LITELLM_SEED_MODEL = 'openai/api_doubao_doubao-seed-2-0-pro-260215'
LITELLM_DEEPSEEK_MODEL = 'openai/api_naci_default_deepseek-v3.2'
LITELLM_CLAUDE_MODEL = 'openai/api_aws_third_anthropic.claude-sonnet-4-6'
WIDE_SEARCH_EVALUATION = {
    'required': ['company'],
    'unique_columns': ['company'],
    'eval_pipeline': {'company': {'metric': ['exact_match']}},
}


def _make_dataset_path(tmp_path: Path) -> Path:
    dataset_path = tmp_path / DATASET_FILENAME
    dataset_path.write_text('placeholder', encoding='utf-8')
    return dataset_path


def _patch_generator_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_generator_module, '_build_openai_clients', lambda _: {'main': {'config': {}}})
    monkeypatch.setattr(search_generator_module, '_build_tool_configs', lambda: {'google_serper_search': {}})
    monkeypatch.setattr(search_generator_module, '_build_processors', lambda: {'chat': {'name': 'chat_tracker'}})
    monkeypatch.setattr(search_generator_module, '_build_sandbox_provider_config', lambda: (None, {}))


def _set_client_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client_name: str,
    model: str,
    prefix: str = 'SEARCH_OPENAI',
) -> None:
    monkeypatch.setenv(f'{prefix}_CLIENTS_NAME', client_name)
    monkeypatch.setenv(f'{prefix}_BASE_URL', 'http://example.com/openapi')
    monkeypatch.setenv(f'{prefix}_API_KEY', 'test-key')
    monkeypatch.setenv(f'{prefix}_MODEL', model)


def _set_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_client_env(monkeypatch, client_name='litellm', model=LITELLM_DEEPSEEK_MODEL, prefix='JUDGE_OPENAI')


def _patch_single_normal_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([
            {
                'nexus': {
                    'task_name': SEARCH_TASK_NAME,
                    'agent_name': SEARCH_AGENT_NAME,
                    'task_config': {
                        'question': NORMAL_QUESTION,
                        'answer': NORMAL_ANSWER,
                        'instance_id': DATASET_INSTANCE_ID,
                    },
                }
            }
        ]),
    )


def test_build_openai_clients_from_env_with_infer_proxy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_client_env(monkeypatch, client_name='infer_proxy', model=INFER_PROXY_TEST_MODEL)
    monkeypatch.setenv('SEARCH_TEMPERATURE', '0.9')
    monkeypatch.setenv('SEARCH_MAX_TOKENS', '131072')
    monkeypatch.setenv('SEARCH_REASONING_EFFORT', 'high')
    monkeypatch.setenv('SEARCH_INTERLEAVED_THINKING', 'false')
    clients = search_generator_module._build_openai_clients('exact_match')
    request_args = clients['main']['config']['request_args']
    extra = request_args['extra_body']

    assert clients['main']['name'] == 'infer_proxy'
    assert request_args['model'] == INFER_PROXY_TEST_MODEL
    assert request_args['temperature'] == 0.9
    assert request_args['max_tokens'] == 131072
    assert set(extra.keys()) == {'chat_template_kwargs'}
    assert extra['chat_template_kwargs']['reasoning_effort'] == 'high'
    assert extra['chat_template_kwargs']['interleaved_thinking'] is False


def test_build_openai_clients_from_env_with_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_client_env(monkeypatch, client_name='openai', model=OPENAI_TEST_MODEL)
    monkeypatch.setenv('SEARCH_TEMPERATURE', '1')
    monkeypatch.setenv('SEARCH_MAX_TOKENS', '32768')
    monkeypatch.setenv('SEARCH_REASONING_EFFORT', 'high')
    monkeypatch.delenv('SEARCH_INTERLEAVED_THINKING', raising=False)
    monkeypatch.delenv('SEARCH_THINKING_TYPE', raising=False)
    clients = search_generator_module._build_openai_clients('exact_match')

    assert clients == {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'base_url': 'http://example.com/openapi',
                    'api_key': 'test-key',
                },
                'request_args': {
                    'model': OPENAI_TEST_MODEL,
                    'temperature': 1.0,
                    'max_tokens': 32768,
                    'extra_body': {'reasoning_effort': 'high'},
                },
            },
        }
    }


def test_build_openai_clients_from_env_with_litellm_seed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_client_env(monkeypatch, client_name='litellm', model=LITELLM_SEED_MODEL)
    _set_judge_env(monkeypatch)
    monkeypatch.setenv('SEARCH_TEMPERATURE', '1')
    monkeypatch.setenv('SEARCH_MAX_TOKENS', '32768')
    monkeypatch.setenv('SEARCH_REASONING_EFFORT', 'high')
    monkeypatch.delenv('SEARCH_INTERLEAVED_THINKING', raising=False)
    monkeypatch.delenv('SEARCH_THINKING_TYPE', raising=False)

    clients = search_generator_module._build_openai_clients('llm_judge')

    assert clients['main']['name'] == 'litellm'
    assert clients['main']['config']['request_args'] == {
        'model': LITELLM_SEED_MODEL,
        'temperature': 1.0,
        'max_tokens': 32768,
        'extra_body': {'reasoning_effort': 'high'},
    }
    assert clients['judge']['name'] == 'litellm'


def test_build_openai_clients_from_env_with_litellm_qwen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_client_env(monkeypatch, client_name='litellm', model=LITELLM_QWEN_MODEL)
    _set_judge_env(monkeypatch)
    monkeypatch.setenv('SEARCH_MAX_TOKENS', '32768')
    monkeypatch.setenv('SEARCH_REASONING_EFFORT', 'high')
    monkeypatch.setenv('SEARCH_THINKING_TYPE', 'enabled')
    monkeypatch.delenv('SEARCH_TEMPERATURE', raising=False)
    monkeypatch.delenv('SEARCH_INTERLEAVED_THINKING', raising=False)

    clients = search_generator_module._build_openai_clients('llm_judge')

    assert clients['main']['name'] == 'litellm'
    assert clients['main']['config']['request_args'] == {
        'model': LITELLM_QWEN_MODEL,
        'max_tokens': 32768,
        'extra_body': {'enable_thinking': True},
    }
    assert clients['judge']['name'] == 'litellm'


def test_build_openai_clients_from_env_with_litellm_claude_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_client_env(monkeypatch, client_name='litellm', model=LITELLM_CLAUDE_MODEL)
    _set_judge_env(monkeypatch)
    monkeypatch.setenv('SEARCH_TEMPERATURE', '1')
    monkeypatch.setenv('SEARCH_MAX_TOKENS', '32768')
    monkeypatch.setenv('SEARCH_REASONING_EFFORT', 'high')
    monkeypatch.setenv('SEARCH_INTERLEAVED_THINKING', 'true')
    monkeypatch.setenv('SEARCH_THINKING_TYPE', 'enabled')

    clients = search_generator_module._build_openai_clients('llm_judge')

    assert clients['main']['name'] == 'litellm'
    assert clients['main']['config']['request_args'] == {
        'model': LITELLM_CLAUDE_MODEL,
        'max_tokens': 32768,
        'extra_body': {
            'reasoning_effort': 'high',
            'thinking': {'type': 'enabled', 'budget_tokens': 4096},
        },
        'extra_headers': {'anthropic-beta': 'interleaved-thinking-2025-05-14'},
    }
    assert clients['judge']['name'] == 'litellm'


def test_build_main_request_args_for_responses_model_normalizes_reasoning_effort() -> None:
    request_args = search_generator_module._build_main_request_args(
        main_model=RESPONSES_TEST_MODEL,
        main_temperature=0.3,
        main_max_tokens=2048,
        main_reasoning_effort=REASONING_EFFORT_JSON,
        main_interleaved_thinking=True,
        main_thinking_type=None,
    )

    assert request_args['max_tokens'] == 2048
    assert request_args['drop_params'] is True
    assert request_args['reasoning_effort'] == {'effort': 'high', 'summary': 'detailed'}
    assert 'temperature' in request_args


def test_generate_configs_wide_search_sample_disables_finish_and_uses_default_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_TOOLSET_VERSION', SEARCH_TOOLSET_VERSION)
    monkeypatch.delenv('SEARCH_AGENT_NAME', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)

    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([
            {
                'nexus': {
                    'task_name': SEARCH_TASK_NAME,
                    'agent_name': SEARCH_AGENT_NAME,
                    'task_config': {
                        'question': DATASET_TABLE_QUESTION,
                        'answer': DATASET_TABLE_ANSWER,
                        'instance_id': DATASET_INSTANCE_ID,
                        'evaluation': WIDE_SEARCH_EVALUATION,
                    },
                }
            }
        ]),
    )
    _patch_generator_dependencies(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert batch_item.run_id == DATASET_INSTANCE_ID
    assert batch_item.config.task_name == SEARCH_TASK_NAME
    assert batch_item.config.agent_name == SEARCH_AGENT_NAME
    assert batch_item.config.imports == DEFAULT_IMPORTS
    assert batch_item.config.agent_config['finish_tool_names'] == []
    assert batch_item.config.agent_config['toolset_version'] == SEARCH_TOOLSET_VERSION


def test_generate_configs_defaults_is_training_to_true_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('IS_TRAINING', raising=False)
    monkeypatch.delenv('SEARCH_TOOLSET_VERSION', raising=False)
    monkeypatch.delenv('SEARCH_EVAL_MODE', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_single_normal_sample(monkeypatch)
    _patch_generator_dependencies(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert batch_item.config.agent_config['is_training'] is True


def test_generate_configs_does_not_inject_toolset_version_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_TOOLSET_VERSION', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_single_normal_sample(monkeypatch)
    _patch_generator_dependencies(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert 'toolset_version' not in batch_item.config.agent_config


def test_generate_configs_defaults_eval_mode_to_llm_judge_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_EVAL_MODE', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    captured: dict[str, object] = {}

    def fake_build_openai_clients(eval_mode: str) -> dict[str, dict[str, object]]:
        captured['eval_mode'] = eval_mode
        return {'main': {'config': {}}}

    monkeypatch.setattr(search_generator_module, '_build_openai_clients', fake_build_openai_clients)
    monkeypatch.setattr(search_generator_module, '_build_tool_configs', lambda: {'google_serper_search': {}})
    monkeypatch.setattr(search_generator_module, '_build_processors', lambda: {'chat': {'name': 'chat_tracker'}})
    monkeypatch.setattr(search_generator_module, '_build_sandbox_provider_config', lambda: (None, {}))
    _patch_single_normal_sample(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert captured['eval_mode'] == 'llm_judge'
    assert batch_item.config.task_config['eval_mode'] == 'llm_judge'


def test_generate_configs_rejects_unknown_toolset_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_TOOLSET_VERSION', 'invalid_version')
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_single_normal_sample(monkeypatch)
    _patch_generator_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="Unknown SEARCH_TOOLSET_VERSION: 'invalid_version'"):
        next(search_generator_module.generate_configs())


@pytest.mark.parametrize(
    ('env_value', 'expected'),
    [
        ('true', True),
        ('false', False),
    ],
)
def test_generate_configs_propagates_enable_summary_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_value: str,
    expected: bool,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_ENABLE_SUMMARY', env_value)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_single_normal_sample(monkeypatch)
    _patch_generator_dependencies(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert batch_item.config.task_config['enable_summary'] is expected


@pytest.mark.parametrize(
    ('is_training', 'toolset_version', 'eval_mode', 'is_wide_search'),
    [
        (True, 'v1', 'llm_judge', False),
        (True, 'v2', 'llm_judge', False),
        (True, 'v1', 'llm_judge', True),
        (True, 'v2', 'llm_judge', True),
        (False, 'v1', 'llm_judge', False),
        (False, 'v2', 'llm_judge', False),
        (False, 'v2', 'exact_match', False),
        (False, 'v2', 'reward_service', False),
        (False, 'v1', 'llm_judge', True),
        (False, 'v2', 'llm_judge', True),
    ],
    ids=[
        'training-v1-normal',
        'training-v2-normal',
        'training-v1-wide',
        'training-v2-wide',
        'inference-v1-normal',
        'inference-v2-normal',
        'inference-v2-normal-exact-match',
        'inference-v2-normal-reward-service',
        'inference-v1-wide',
        'inference-v2-wide',
    ],
)
def test_generate_configs_matches_documented_configuration_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    is_training: bool,
    toolset_version: str,
    eval_mode: str,
    is_wide_search: bool,
) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_TOOLSET_VERSION', toolset_version)
    monkeypatch.setenv('SEARCH_EVAL_MODE', eval_mode)
    monkeypatch.setenv('IS_TRAINING', str(is_training).lower())
    monkeypatch.setenv('SEARCH_MAX_ITERATIONS', '12')
    monkeypatch.setenv('SEARCH_AGENT_MAX_LLM_RETRIES', '4')
    if eval_mode == 'reward_service':
        monkeypatch.setenv('REWARD_SERVICE_URL', 'https://example.com/reward')
        monkeypatch.setenv('REWARD_SERVICE_TOKEN', 'reward-token')
        monkeypatch.setenv('REWARD_SERVICE_ROUTER_ID', 'reward-router-id')
    monkeypatch.delenv('SEARCH_AGENT_NAME', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)

    evaluation = WIDE_SEARCH_EVALUATION if is_wide_search else None
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([
            {
                'nexus': {
                    'task_name': SEARCH_TASK_NAME,
                    'agent_name': SEARCH_AGENT_NAME,
                    'agent_config': {
                        'max_iterations': 2,
                        'max_llm_retries': 9,
                    },
                    'task_config': {
                        'question': DATASET_TABLE_QUESTION if is_wide_search else NORMAL_QUESTION,
                        'answer': DATASET_TABLE_ANSWER if is_wide_search else NORMAL_ANSWER,
                        'instance_id': DATASET_INSTANCE_ID,
                        **({'evaluation': evaluation} if evaluation else {}),
                    },
                }
            }
        ]),
    )
    _patch_generator_dependencies(monkeypatch)

    batch_item = next(search_generator_module.generate_configs())

    assert batch_item.run_id == DATASET_INSTANCE_ID
    assert batch_item.config.imports == DEFAULT_IMPORTS
    assert batch_item.config.agent_name == SEARCH_AGENT_NAME
    assert batch_item.config.task_name == SEARCH_TASK_NAME
    assert batch_item.config.task_config['eval_mode'] == eval_mode
    assert batch_item.config.agent_config['is_training'] is is_training
    assert batch_item.config.agent_config['toolset_version'] == toolset_version
    assert batch_item.config.agent_config['max_iterations'] == 12
    assert batch_item.config.agent_config['max_llm_retries'] == 4
    assert batch_item.config.agent_config['finish_tool_names'] == ([] if is_wide_search else ['finish'])


def test_get_total_count_respects_filters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_INSTANCE_IDS', FILTERED_INSTANCE_ID)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([
            {'nexus': {'task_config': {'instance_id': 'search_00001'}}},
            {'nexus': {'task_config': {'instance_id': FILTERED_INSTANCE_ID}}},
            {'nexus': {'task_config': {'instance_id': 'search_00003'}}},
        ]),
    )

    assert search_generator_module.get_total_count() == 1


def test_get_total_count_filters_run_ids_by_instance_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.setenv('SEARCH_RUN_IDS', DATASET_INSTANCE_ID)
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([
            {'nexus': {'task_config': {'instance_id': DATASET_INSTANCE_ID}}},
            {'nexus': {'task_config': {'instance_id': 'other_instance'}}},
        ]),
    )

    assert search_generator_module.get_total_count() == 1


def test_get_batch_metadata_reads_dataset_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_AGENT_NAME', SEARCH_AGENT_NAME)
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([{'nexus': {}}, {'nexus': {}}]),
    )

    metadata = search_generator_module.get_batch_metadata()

    assert metadata['task_type'] == SEARCH_TASK_NAME
    assert metadata['dataset_path'] == str(dataset_path)
    assert metadata['agent_name'] == SEARCH_AGENT_NAME
    assert metadata['dataset_rows'] == 2
    assert metadata['dataset_columns'] == ['nexus']


# ============================================================================
# Precedence: env > YAML > parquet row > code default
# (for every per-row mergeable field — task_config, agent_config, agent_name, imports)
# ============================================================================


def _patch_yaml_config(monkeypatch: pytest.MonkeyPatch, yaml_cfg) -> None:
    """Stub load_search_config_if_exists() to return a hand-built SearchConfig."""
    monkeypatch.setattr(search_generator_module, 'load_search_config_if_exists', lambda: yaml_cfg)


def _patch_single_row_with(monkeypatch: pytest.MonkeyPatch, nexus: dict) -> None:
    monkeypatch.setattr(
        search_generator_module.pd,
        'read_parquet',
        lambda _: pd.DataFrame([{'nexus': nexus}]),
    )


def test_task_config_yaml_beats_parquet_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """task_config: YAML value wins over a conflicting parquet row value."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_EVAL_MODE', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    _patch_yaml_config(
        monkeypatch,
        SearchConfig(task_name='search', task_config={'eval_mode': 'llm_judge'}),
    )
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
                'eval_mode': 'exact_match',  # row tries to override — should lose
            },
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.task_config['eval_mode'] == 'llm_judge'


def test_task_config_env_beats_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """task_config: env override (matches `--eval-mode` CLI) wins over YAML."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.setenv('SEARCH_EVAL_MODE', 'reward_service')
    monkeypatch.setenv('REWARD_SERVICE_URL', 'http://example.com')
    monkeypatch.setenv('REWARD_SERVICE_TOKEN', 'token')
    monkeypatch.setenv('REWARD_SERVICE_ROUTER_ID', 'router')
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    _patch_yaml_config(
        monkeypatch,
        SearchConfig(task_name='search', task_config={'eval_mode': 'llm_judge'}),
    )
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
                'eval_mode': 'exact_match',
            },
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.task_config['eval_mode'] == 'reward_service'


def test_agent_config_yaml_beats_parquet_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """agent_config: YAML value wins over a conflicting parquet row value."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_MAX_ITERATIONS', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    _patch_yaml_config(
        monkeypatch,
        SearchConfig(task_name='search', agent_config={'max_iterations': 25}),
    )
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
            },
            'agent_config': {'max_iterations': 99},  # row tries to override — should lose
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.agent_config['max_iterations'] == 25


def test_agent_name_yaml_beats_parquet_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """agent_name: YAML value wins over a conflicting parquet row value."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_AGENT_NAME', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    _patch_yaml_config(
        monkeypatch,
        SearchConfig(task_name='search', agent_name='yaml_agent'),
    )
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'agent_name': 'row_agent',  # row tries to override — should lose
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
            },
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.agent_name == 'yaml_agent'


def test_imports_yaml_beats_parquet_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """imports: YAML value wins over a conflicting parquet row value."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    _patch_yaml_config(
        monkeypatch,
        SearchConfig(task_name='search', imports=['nexus.extensions', 'my.yaml_pkg']),
    )
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'imports': ['my.row_pkg'],  # row tries to override — should lose
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
            },
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.imports == ['nexus.extensions', 'my.yaml_pkg']


def test_precedence_fallback_uses_parquet_row_when_yaml_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When YAML doesn't set a field, parquet row still applies (as fallback)."""
    from nexus.extensions.tasks.search.config import SearchConfig

    dataset_path = _make_dataset_path(tmp_path)
    monkeypatch.setenv('SEARCH_DATASET_PATH', str(dataset_path))
    monkeypatch.delenv('SEARCH_EVAL_MODE', raising=False)
    monkeypatch.delenv('SEARCH_MAX_ITERATIONS', raising=False)
    monkeypatch.delenv('SEARCH_AGENT_NAME', raising=False)
    monkeypatch.delenv('SEARCH_INSTANCE_IDS', raising=False)
    monkeypatch.delenv('SEARCH_RUN_IDS', raising=False)
    _patch_generator_dependencies(monkeypatch)

    # YAML sets no task_config/agent_config/agent_name/imports.
    _patch_yaml_config(monkeypatch, SearchConfig(task_name='search'))
    _patch_single_row_with(
        monkeypatch,
        {
            'task_name': SEARCH_TASK_NAME,
            'agent_name': 'row_agent',
            'imports': ['my.row_pkg'],
            'task_config': {
                'question': NORMAL_QUESTION,
                'answer': NORMAL_ANSWER,
                'instance_id': DATASET_INSTANCE_ID,
                'eval_mode': 'exact_match',
            },
            'agent_config': {'max_iterations': 17},
        },
    )

    batch_item = next(search_generator_module.generate_configs())
    assert batch_item.config.task_config['eval_mode'] == 'exact_match'
    assert batch_item.config.agent_config['max_iterations'] == 17
    assert batch_item.config.agent_name == 'row_agent'
    assert batch_item.config.imports == ['my.row_pkg']
