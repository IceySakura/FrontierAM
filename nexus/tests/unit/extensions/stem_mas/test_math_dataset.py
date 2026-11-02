import pytest

from pydantic import ValidationError

from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets import build_dataset_adapter
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math import MathDataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math.profiles import build_math_prompt_profile
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math.sources import MathSourceLoader


def test_non_manifest_bound_candidate_kind_is_rejected_by_dataset() -> None:
    with pytest.raises(ValidationError, match='lean4_verifier_candidate_kind'):
        build_dataset_adapter('putnam672', '', lean4_verifier_candidate_kind='full_declaration')


def test_unknown_dataset_option_is_rejected() -> None:
    with pytest.raises(ValidationError, match='unexpected_option'):
        build_dataset_adapter('minif2f', '', unexpected_option=True)


def test_putnam_uses_full_code_prompt_with_manifest_bound_candidate() -> None:
    dataset = build_dataset_adapter('putnam672', '')
    assert isinstance(dataset, MathDataset)

    spec = dataset.benchmark_spec()

    assert spec.prompt_profile == 'putnam_full_code'
    assert spec.candidate_policy == 'target_declaration_proof'


def test_benchmark_prompt_contracts_remain_distinct() -> None:
    minif2f_prompt = build_math_prompt_profile('minif2f', 'minif2f').build_prompt(
        lean_version='v4.30.0',
        header='import Mathlib',
        statement='theorem test : True := by sorry',
    )
    fate_prompt = build_math_prompt_profile('fate_full_code', 'fatem').build_prompt(
        lean_version='v4.16.0',
        header='import Mathlib',
        statement='theorem test : True := by sorry',
    )
    putnam_prompt = build_math_prompt_profile('putnam_full_code', 'putnam672').build_prompt(
        lean_version='v4.27.0',
        header='import Mathlib',
        statement='theorem test : True := sorry',
    )

    assert '<final_proof>' in minif2f_prompt
    assert 'complete Lean code' in fate_prompt
    assert 'exactly one import line: `import Mathlib`' in putnam_prompt


def test_putnam_rejects_noncanonical_statement_field() -> None:
    row = {
        'problem_name': 'putnam_2025_a1',
        'lean4_statement': 'theorem putnam_2025_a1 : True := by sorry',
    }

    with pytest.raises(ValueError, match='unsupported source fields: lean4_statement'):
        MathDataset(path='', name='putnam672').to_dataset_item(row, 0)


def test_math_source_loader_parses_jsonl() -> None:
    rows = MathSourceLoader.load_jsonl_text('memory.jsonl', '{"split":"train","id":1}\n{"split":"test","id":2}')

    assert rows == [{'split': 'train', 'id': 1}, {'split': 'test', 'id': 2}]


def test_runtime_metadata_does_not_duplicate_transport_task_config() -> None:
    source = {
        'name': 'sample',
        'formal_statement': 'theorem sample : True := by sorry',
        'header': 'import Mathlib',
    }

    item = MathDataset(path='', name='minif2f').to_dataset_item(source, 0)

    assert 'task_config' not in item.body['task_config']['metadata']
    assert item.body['task_config']['evaluator_type'] == 'lean4_verifier'


def test_math_adapter_rejects_training_export_schema() -> None:
    row = {
        'prompt': [{'role': 'user', 'content': 'Training prompt'}],
        'reward_model': {'ground_truth': 'training answer'},
    }

    with pytest.raises(ValueError, match='unsupported source fields'):
        MathDataset(path='', name='minif2f').to_dataset_item(row, 0)
