import json

from pathlib import Path

import pytest

from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_io import write_local_nexus_dataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_items import build_stem_mas_item
from nexus.extensions.tasks.stem_mas.benchmark.eval.profile import BenchmarkProfile
from nexus.extensions.tasks.stem_mas.run_parts import stem_mas_benchmark_run_part
from nexus.runner.compose import compile_batch, deep_merge, parse_ordered_parts
from nexus.runner.single import RunInput


CONFIG_ROOT = Path(__file__).resolve().parents[4] / 'nexus/extensions/tasks/stem_mas/benchmark/eval/config'


def formal_item():
    question = 'theorem sample : True := by\n  sorry'
    return build_stem_mas_item(
        item_id='sample-1',
        dataset_name='minif2f',
        source_index=0,
        question=question,
        prompt=[{'role': 'user', 'content': question}],
        answer='by trivial',
        category='mathematics',
        task_fields={
            'evaluator_type': 'lean4_verifier',
            'lean4_verifier_benchmark': 'minif2f',
            'lean4_verifier_task_id': 'minif2f.1',
            'lean4_verifier_manifest': 'minif2f-v1',
            'lean4_verifier_candidate_kind': 'proof_body',
            'lean4_verifier_candidate_policy': 'proof_body',
        },
    )


def hle_item():
    return build_stem_mas_item(
        item_id='hle-1',
        dataset_name='hle_200',
        source_index=0,
        question='What is 1 + 1?',
        prompt=[{'role': 'user', 'content': 'What is 1 + 1?'}],
        answer='2',
        category='mathematics',
    )


def test_profile_run_part_composes_with_portable_dataset_item() -> None:
    item = formal_item()
    part = stem_mas_benchmark_run_part({'profile': 'math/deepseek_v4_flash/no_agent'})

    run_input = RunInput.from_dict(deep_merge(item.body, part.body, part.merge))

    assert run_input.agent_name == 'single_turn'
    assert run_input.task_config['question'].startswith('theorem sample')
    assert run_input.task_config['lean4_verifier_task_id'] == 'minif2f.1'
    assert run_input.task_config['metadata']['solver_strategy'] == 'no_agent'
    assert run_input.task_config['metadata']['solver_family'] == 'direct'
    assert run_input.task_config['timeout'] == 7200
    assert 'cache_metrics' not in part.body


def test_standard_nexus_compose_loads_dataset_and_registered_profile(tmp_path) -> None:
    dataset_root = write_local_nexus_dataset([formal_item()], tmp_path / 'minif2f', name='minif2f')

    batch = compile_batch(
        parse_ordered_parts(
            [
                'run',
                'batch',
                '--data',
                str(dataset_root),
                '--use',
                'stem_mas_benchmark:profile=math/deepseek_v4_flash/no_agent',
            ]
        )
    )

    assert len(batch) == 1
    item = batch[0]
    assert item.run_id == 'sample-1'
    assert item.metadata['dataset_item_id'] == 'sample-1'
    assert item.config.agent_name == 'single_turn'
    assert item.config.task_config['lean4_verifier_task_id'] == 'minif2f.1'
    assert item.config.task_config['metadata']['solver_strategy'] == 'no_agent'
    assert item.config.task_config['metadata']['solver_family'] == 'direct'


@pytest.mark.parametrize(
    ('profile_name', 'source_row'),
    [
        (
            'hle/judge_deepseek',
            {
                'id': 'hle-case-1',
                'origin_question': 'What is 1 + 1?',
                'ref_answer': '2',
                'subject': 'mathematics',
            },
        ),
        (
            'math/deepseek_v4_flash/no_agent',
            {
                'name': 'sample',
                'formal_statement': 'theorem sample : True := by\n  sorry',
                'header': 'import Mathlib',
            },
        ),
    ],
)
def test_materialized_compose_matches_direct_adapter_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    source_row: dict,
) -> None:
    monkeypatch.delenv('STEM_MAS_DATASET_NAME', raising=False)
    monkeypatch.delenv('STEM_MAS_DATASET_PATH', raising=False)
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(source_row) + '\n', encoding='utf-8')

    profile = BenchmarkProfile.load(CONFIG_ROOT / f'{profile_name}.yaml')
    adapter = profile.build_dataset_adapter(source)
    dataset_item = adapter.load_items()[0]
    part = stem_mas_benchmark_run_part({'profile': profile_name})
    direct = RunInput.from_dict(deep_merge(dataset_item.body, part.body, part.merge))

    dataset_root = write_local_nexus_dataset([dataset_item], tmp_path / 'materialized', name=adapter.name)
    compiled = compile_batch(
        parse_ordered_parts(
            [
                'run',
                'batch',
                '--data',
                str(dataset_root),
                '--use',
                f'stem_mas_benchmark:profile={profile_name}',
            ]
        )
    )[0].config

    assert compiled.model_dump() == direct.model_dump()


def test_materialized_dataset_does_not_serialize_profile_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('SCIENCE_API_KEY', 'private-science-key')
    monkeypatch.setenv('SCIENCE_BASE_URL', 'https://private-model.example/v1')
    source = tmp_path / 'source.jsonl'
    source.write_text(
        json.dumps(
            {
                'name': 'sample',
                'formal_statement': 'theorem sample : True := by\n  sorry',
                'header': 'import Mathlib',
            }
        )
        + '\n',
        encoding='utf-8',
    )
    profile = BenchmarkProfile.load(CONFIG_ROOT / 'math/deepseek_v4_flash/no_agent.yaml')
    item = profile.build_dataset_adapter(source).load_items()[0]

    dataset_root = write_local_nexus_dataset([item], tmp_path / 'materialized', name='minif2f')
    serialized = (dataset_root / 'items/000000/item.yaml').read_text(encoding='utf-8')

    assert 'private-science-key' not in serialized
    assert 'private-model.example' not in serialized
    assert 'openai_clients' not in serialized
    assert 'runtime_configs' not in serialized


@pytest.mark.parametrize(
    ('profile', 'evaluator_type'),
    [
        ('hle/claude_code_v7', 'reward_system'),
        ('hle/codex_v7', 'reward_system'),
        ('hle/judge_deepseek', 'hle'),
        ('hle/judge_gpt_oss', 'hle'),
        ('hle/main_agent_api', 'reward_system'),
        ('hle/main_agent_manual_schema', 'hle'),
    ],
)
def test_all_hle_profiles_compose_without_math_solver_metadata(profile: str, evaluator_type: str) -> None:
    part = stem_mas_benchmark_run_part({'profile': profile})
    run_input = RunInput.from_dict(deep_merge(hle_item().body, part.body, part.merge))

    assert run_input.task_config['question'] == 'What is 1 + 1?'
    assert run_input.task_config['answer'] == '2'
    assert run_input.task_config['evaluator_type'] == evaluator_type
    assert 'solver_strategy' not in run_input.task_config
    assert 'solver_family' not in run_input.task_config['metadata']


@pytest.mark.parametrize(
    'profile',
    [
        'hle/judge_deepseek',
        'hle/judge_gpt_oss',
        'hle/main_agent_manual_schema',
    ],
)
def test_hle_gongfeng_profiles_use_portable_worker_environment(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('STEM_MAS_DATASET_PATH', '/tmp/hle-smoke.jsonl')
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')

    part = stem_mas_benchmark_run_part({'profile': profile})

    assert part.body['runtime_provider_config']['token'] == 'test-token'
    assert part.body['runtime_configs']['main']['environment'] == {
        'UV_CACHE_DIR': '/tmp/nexus-uv-cache',
        'UV_PYTHON_DOWNLOADS': 'never',
    }


@pytest.mark.parametrize(
    'profile',
    [
        'math/a20b_high_260609_v5_step320_192k/codex',
        'math/a20b_high_260609_v5_step320_192k/codex_lean',
        'math/a20b_high_260609_v5_step320_192k/no_agent',
        'math/deepseek_v4_flash/codex',
        'math/deepseek_v4_flash/codex_lean',
        'math/deepseek_v4_flash/no_agent',
        'math/deepseek_v4_flash/official',
    ],
)
def test_all_math_profiles_preserve_dataset_owned_verifier_contract(profile: str) -> None:
    part = stem_mas_benchmark_run_part({'profile': profile})
    run_input = RunInput.from_dict(deep_merge(formal_item().body, part.body, part.merge))

    assert run_input.task_config['evaluator_type'] == 'lean4_verifier'
    assert run_input.task_config['lean4_verifier_benchmark'] == 'minif2f'
    assert run_input.task_config['lean4_verifier_task_id'] == 'minif2f.1'
    assert run_input.task_config['lean4_verifier_manifest'] == 'minif2f-v1'
    assert run_input.task_config['metadata']['solver_strategy']


def test_generic_official_profile_infers_pipeline_from_dataset() -> None:
    inferred = stem_mas_benchmark_run_part({'profile': 'math/deepseek_v4_flash/official'})

    overridden = stem_mas_benchmark_run_part(
        {
            'profile': 'math/deepseek_v4_flash/official',
            'official_pipeline': 'fate_eval',
        }
    )

    assert inferred.body['task_config']['official_solver_pipeline'] == 'minif2f'
    assert overridden.body['task_config']['official_solver_pipeline'] == 'fate_eval'


@pytest.mark.parametrize(
    ('dataset_name', 'pipeline'),
    [('minif2f', 'minif2f'), ('putnam672', 'putnambench'), ('putnam-2025', 'putnambench'), ('fateh', 'fate_eval')],
)
def test_generic_official_profile_tracks_dataset_override(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    pipeline: str,
) -> None:
    monkeypatch.setenv('STEM_MAS_DATASET_NAME', dataset_name)

    part = stem_mas_benchmark_run_part({'profile': 'math/deepseek_v4_flash/official'})

    assert part.body['task_config']['official_solver_pipeline'] == pipeline


def test_profile_run_part_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match='must stay under'):
        stem_mas_benchmark_run_part({'profile': '../outside'})
