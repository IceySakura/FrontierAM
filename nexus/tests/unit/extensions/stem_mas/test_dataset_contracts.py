from __future__ import annotations

import json

from pathlib import Path

import pytest

from nexus.dataset import load_local_dataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_io import write_local_nexus_dataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets import build_dataset_adapter
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.base import BenchmarkDatasetAdapter
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math import MathDataset


def write_jsonl(tmp_path: Path, row: dict) -> Path:
    return write_jsonl_rows(tmp_path, [row])


def write_jsonl_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / 'dataset.jsonl'
    path.write_text(''.join(f'{json.dumps(row)}\n' for row in rows), encoding='utf-8')
    return path


def formal_math_row(dataset_name: str, task_id: str, statement: str, **metadata: object) -> dict:
    if dataset_name == 'minif2f':
        return {
            'name': task_id,
            'formal_statement': statement,
            'header': 'import Mathlib',
            **metadata,
        }
    if dataset_name in {'putnam672', 'putnam-2025'}:
        return {'problem_name': task_id, 'formal_statement': statement, **metadata}
    return {'id': task_id, 'formal_statement': statement, **metadata}


def test_hle_dataset_item_contract(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        {
            'id': 'hle-case-7',
            'origin_question': 'Which value is prime?',
            'ref_answer': '13',
            'subject': 'mathematics',
        },
    )

    item = build_dataset_adapter('hle_200', path).load_items()[0]

    assert item.model_dump(mode='json', exclude_none=True) == {
        'id': 'hle-case-7',
        'body': {
            'task_name': 'stem_mas',
            'task_config': {
                'task_id': 'hle-case-7',
                'category': 'mathematics',
                'question': 'Which value is prime?',
                'prompt': [{'role': 'user', 'content': 'Which value is prime?'}],
                'answer': '13',
                'metadata': {
                    'index': 0,
                    'task_id': 'hle-case-7',
                    'question': 'Which value is prime?',
                    'answer': '13',
                    'subject': 'mathematics',
                    'source_format': 'hle',
                    'data_source': 'hle_200',
                    'ability': 'mathematics',
                },
            },
        },
        'merge': {},
        'metadata': {'index': 0, 'dataset_name': 'hle_200'},
    }


def test_hle_three_row_identity_contract(tmp_path: Path) -> None:
    rows = [
        {'id': 'hle-case-1', 'origin_question': 'Question 1', 'ref_answer': 'Answer 1'},
        {'id': 7, 'origin_question': 'Question 2', 'ref_answer': 2},
        {'origin_question': 'Question 3', 'ref_answer': 'Answer 3'},
    ]

    items = build_dataset_adapter('hle_200', write_jsonl_rows(tmp_path, rows)).load_items()

    assert [item.id for item in items] == ['hle-case-1', '7', '2']
    assert [item.metadata['index'] for item in items] == [0, 1, 2]
    assert [item.body['task_config']['answer'] for item in items] == ['Answer 1', '2', 'Answer 3']
    for index, item in enumerate(items):
        task_config = item.body['task_config']
        expected_question = f'Question {index + 1}'
        assert task_config['question'] == expected_question
        assert task_config['prompt'] == [{'role': 'user', 'content': expected_question}]
        assert task_config['metadata']['source_format'] == 'hle'
        assert task_config['metadata']['data_source'] == 'hle_200'


def test_hle_dataset_item_round_trips_through_nexus_local_dataset(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path,
        {'id': 'hle-case-1', 'origin_question': 'Question 1', 'ref_answer': 'Answer 1'},
    )
    original = build_dataset_adapter('hle_200', source).load_items()[0]

    dataset_root = write_local_nexus_dataset([original], tmp_path / 'nexus-dataset', name='hle_200')
    loaded = load_local_dataset(dataset_root).load_items()[0]

    assert loaded.model_dump(mode='json', exclude_none=True) == original.model_dump(mode='json', exclude_none=True)


def test_hle_non_string_fields_preserve_legacy_text_semantics(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path,
        {'id': True, 'origin_question': 1, 'ref_answer': False, 'subject': 7},
    )

    item = build_dataset_adapter('hle_200', source).load_items()[0]
    task_config = item.body['task_config']

    assert item.id == 'True'
    assert task_config['question'] == '1'
    assert task_config['answer'] == 'False'
    assert task_config['category'] == '7'


@pytest.mark.parametrize(
    ('dataset_name', 'task_id', 'verifier_benchmark', 'verifier_task_id', 'manifest', 'prompt_marker', 'policy'),
    [
        (
            'minif2f',
            'sample',
            'minif2f',
            'minif2f.sample',
            'minif2f-v2-905eca84',
            '<final_proof>',
            'proof_body',
        ),
        (
            'putnam672',
            'putnam_2025_a1',
            'putnam672',
            'putnam672.putnam_2025_a1',
            'putnambench-c8c8fa28',
            'exactly one import line',
            'target_declaration_proof',
        ),
        (
            'putnam-2025',
            'putnam_2025_a1',
            'putnam672',
            'putnam672.putnam_2025_a1',
            'putnambench-c8c8fa28',
            'exactly one import line',
            'target_declaration_proof',
        ),
        ('fatem', '1', 'fatem', 'fatem.001', 'fate-eval-fe358fb5', 'complete Lean code', 'target_declaration_proof'),
        ('fateh', '1', 'fateh', 'fateh.001', 'fate-eval-fe358fb5', 'complete Lean code', 'target_declaration_proof'),
        ('fatex', '1', 'fatex', 'fatex.001', 'fate-eval-fe358fb5', 'complete Lean code', 'target_declaration_proof'),
    ],
)
def test_formal_math_dataset_item_contract(
    tmp_path: Path,
    dataset_name: str,
    task_id: str,
    verifier_benchmark: str,
    verifier_task_id: str,
    manifest: str,
    prompt_marker: str,
    policy: str,
) -> None:
    path = write_jsonl(
        tmp_path,
        formal_math_row(dataset_name, task_id, f'theorem {task_id} : True := by\n  sorry'),
    )

    item = build_dataset_adapter(dataset_name, path).load_items()[0]
    task_config = item.body['task_config']
    metadata = task_config['metadata']

    assert item.id == task_id
    assert item.metadata == {'index': 0, 'dataset_name': dataset_name}
    assert task_config['evaluator_type'] == 'lean4_verifier'
    assert task_config['lean4_verifier_benchmark'] == verifier_benchmark
    assert task_config['lean4_verifier_task_id'] == verifier_task_id
    assert task_config['lean4_verifier_manifest'] == manifest
    assert task_config['lean4_verifier_candidate_kind'] == 'proof_body'
    assert task_config['lean4_verifier_candidate_policy'] == policy
    assert task_config['answer'] == f'theorem {task_id} : True := by\n  sorry'
    assert prompt_marker in task_config['prompt'][-1]['content']
    assert 'task_config' not in metadata


@pytest.mark.parametrize('dataset_name', ['minif2f', 'putnam672', 'putnam-2025'])
def test_default_hf_source_records_pinned_revision(dataset_name: str) -> None:
    dataset = build_dataset_adapter(dataset_name, '')
    assert isinstance(dataset, MathDataset)
    spec = dataset.benchmark_spec()

    assert spec.hf_revision


@pytest.mark.parametrize(
    ('dataset_name', 'task_ids', 'expected_verifier_ids', 'manifest', 'policy', 'prompt_marker'),
    [
        (
            'minif2f',
            ['sample_1', 'sample_2', 'sample_3'],
            ['minif2f.sample_1', 'minif2f.sample_2', 'minif2f.sample_3'],
            'minif2f-v2-905eca84',
            'proof_body',
            '<final_proof>',
        ),
        (
            'putnam672',
            ['putnam_2025_a1', 'putnam_2025_a2', 'putnam_2025_a3'],
            ['putnam672.putnam_2025_a1', 'putnam672.putnam_2025_a2', 'putnam672.putnam_2025_a3'],
            'putnambench-c8c8fa28',
            'target_declaration_proof',
            'exactly one import line',
        ),
        (
            'putnam-2025',
            ['putnam_2025_a1', 'putnam_2025_a2', 'putnam_2025_a3'],
            ['putnam672.putnam_2025_a1', 'putnam672.putnam_2025_a2', 'putnam672.putnam_2025_a3'],
            'putnambench-c8c8fa28',
            'target_declaration_proof',
            'exactly one import line',
        ),
        (
            'fatem',
            ['1', '2', '3'],
            ['fatem.001', 'fatem.002', 'fatem.003'],
            'fate-eval-fe358fb5',
            'target_declaration_proof',
            'complete Lean code',
        ),
        (
            'fateh',
            ['1', '2', '3'],
            ['fateh.001', 'fateh.002', 'fateh.003'],
            'fate-eval-fe358fb5',
            'target_declaration_proof',
            'complete Lean code',
        ),
        (
            'fatex',
            ['1', '2', '3'],
            ['fatex.001', 'fatex.002', 'fatex.003'],
            'fate-eval-fe358fb5',
            'target_declaration_proof',
            'complete Lean code',
        ),
    ],
)
def test_formal_math_three_row_identity_contract(
    tmp_path: Path,
    dataset_name: str,
    task_ids: list[str],
    expected_verifier_ids: list[str],
    manifest: str,
    policy: str,
    prompt_marker: str,
) -> None:
    rows = [
        formal_math_row(dataset_name, task_id, f'theorem sample_{index} : True := by\n  sorry')
        for index, task_id in enumerate(task_ids, start=1)
    ]

    items = build_dataset_adapter(dataset_name, write_jsonl_rows(tmp_path, rows)).load_items()

    assert [item.id for item in items] == task_ids
    assert [item.metadata['index'] for item in items] == [0, 1, 2]
    assert [item.body['task_config']['lean4_verifier_task_id'] for item in items] == expected_verifier_ids
    for index, item in enumerate(items, start=1):
        task_config = item.body['task_config']
        assert task_config['evaluator_type'] == 'lean4_verifier'
        assert task_config['lean4_verifier_manifest'] == manifest
        assert task_config['lean4_verifier_candidate_kind'] == 'proof_body'
        assert task_config['lean4_verifier_candidate_policy'] == policy
        assert task_config['answer'] == f'theorem sample_{index} : True := by\n  sorry'
        assert prompt_marker in task_config['prompt'][-1]['content']


@pytest.mark.parametrize(
    'dataset_name',
    ['minif2f', 'putnam672', 'putnam-2025', 'fatem', 'fateh', 'fatex'],
)
def test_formal_math_dataset_item_round_trips_through_nexus_local_dataset(
    tmp_path: Path,
    dataset_name: str,
) -> None:
    source = write_jsonl(
        tmp_path,
        formal_math_row(dataset_name, '1', 'theorem sample : True := by\n  sorry'),
    )
    original = build_dataset_adapter(dataset_name, source).load_items()[0]

    dataset_root = write_local_nexus_dataset([original], tmp_path / 'nexus-dataset', name=dataset_name)
    loaded = load_local_dataset(dataset_root).load_items()[0]

    assert loaded.model_dump(mode='json', exclude_none=True) == original.model_dump(mode='json', exclude_none=True)


def test_math_source_metadata_cannot_inject_runtime_configuration(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path,
        {
            'name': 'sample',
            'formal_statement': 'theorem sample : True := by\n  sorry',
            'header': 'import Mathlib',
            'extra_info': {'runtime_configs': {'main': {'api_key': 'must-not-publish'}}},
        },
    )

    with pytest.raises(ValueError, match='unsupported source fields: extra_info'):
        build_dataset_adapter('minif2f', source).load_items()


def test_formal_math_source_cannot_override_dataset_evaluator(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path,
        {
            'name': 'sample',
            'formal_statement': 'theorem sample : True := by\n  sorry',
            'header': 'import Mathlib',
            'evaluator_type': 'none',
        },
    )

    with pytest.raises(ValueError, match='unsupported source fields: evaluator_type'):
        build_dataset_adapter('minif2f', source).load_items()


def test_duplicate_dataset_adapter_registration_is_rejected() -> None:
    class FirstAdapter(BenchmarkDatasetAdapter):
        dataset_names = ('unit_test_duplicate_adapter',)

    assert FirstAdapter.dataset_names == ('unit_test_duplicate_adapter',)
    with pytest.raises(ValueError, match='Duplicate dataset adapter'):
        class SecondAdapter(BenchmarkDatasetAdapter):
            dataset_names = ('unit_test_duplicate_adapter',)
