from pathlib import Path

import pytest

from nexus.dataset import load_local_dataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_io import write_local_nexus_dataset
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_items import build_stem_mas_item


def formal_item(task_id: str = 'sample-1'):
    question = 'theorem sample : True := by\n  sorry'
    return build_stem_mas_item(
        item_id=task_id,
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


def test_portable_dataset_item_excludes_environment_and_run_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_ROUTER_ID', 'must-not-leak')

    item = formal_item()

    task_config = item.body['task_config']
    assert item.id == 'sample-1'
    assert item.body['task_name'] == 'stem_mas'
    assert task_config['evaluator_type'] == 'lean4_verifier'
    assert task_config['lean4_verifier_task_id'] == 'minif2f.1'
    assert 'timeout' not in task_config
    assert 'reward_system_router_id' not in task_config
    assert item.metadata['dataset_name'] == 'minif2f'
    assert 'cache_session_id' not in item.metadata


def test_write_local_nexus_dataset_round_trips(tmp_path: Path) -> None:
    items = [formal_item()]

    root = write_local_nexus_dataset(items, tmp_path / 'dataset', name='minif2f')
    loaded = load_local_dataset(root).load_items()

    assert [item.id for item in loaded] == ['sample-1']
    assert loaded[0].body == items[0].body
    assert loaded[0].metadata == items[0].metadata


def test_write_local_nexus_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    items = [formal_item(), formal_item()]

    with pytest.raises(ValueError, match='Duplicate dataset item id'):
        write_local_nexus_dataset(items, tmp_path / 'dataset', name='minif2f')


def test_write_local_nexus_dataset_force_replaces_existing_dataset(tmp_path: Path) -> None:
    root = write_local_nexus_dataset([formal_item('old')], tmp_path / 'dataset', name='minif2f')

    write_local_nexus_dataset([formal_item('new')], root, name='minif2f', force=True)

    assert [item.id for item in load_local_dataset(root).load_items()] == ['new']


def test_write_local_nexus_dataset_force_rejects_unrelated_directory(tmp_path: Path) -> None:
    root = tmp_path / 'not-a-dataset'
    root.mkdir()
    sentinel = root / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')

    with pytest.raises(FileExistsError, match='Refusing to replace non-dataset directory'):
        write_local_nexus_dataset([formal_item()], root, name='minif2f', force=True)

    assert sentinel.read_text(encoding='utf-8') == 'keep'


def test_dataset_item_rejects_execution_owned_fields() -> None:
    with pytest.raises(ValueError, match='needs_runtime'):
        build_stem_mas_item(
            item_id='sample-1',
            dataset_name='minif2f',
            source_index=0,
            question='question',
            prompt=[{'role': 'user', 'content': 'question'}],
            answer='answer',
            category='mathematics',
            task_fields={'needs_runtime': False},
        )


@pytest.mark.parametrize(
    'metadata',
    [
        {'needs_runtime': False},
        {'client': {'api_key': 'secret'}},
        {'router': {'reward_system_router_id': 'route'}},
    ],
)
def test_dataset_item_rejects_unsupported_metadata(metadata: dict) -> None:
    with pytest.raises(ValueError, match='Dataset metadata contains unsupported fields'):
        build_stem_mas_item(
            item_id='sample-1',
            dataset_name='minif2f',
            source_index=0,
            question='question',
            prompt=[{'role': 'user', 'content': 'question'}],
            answer='answer',
            category='mathematics',
            metadata=metadata,
        )


def test_dataset_item_rejects_environment_interpolation() -> None:
    with pytest.raises(ValueError, match='environment interpolation'):
        build_stem_mas_item(
            item_id='sample-1',
            dataset_name='minif2f',
            source_index=0,
            question='question',
            prompt=[{'role': 'user', 'content': 'question'}],
            answer='answer',
            category='mathematics',
            metadata={'source': '${oc.env:DATASET_SOURCE,"fallback"}'},
        )


def test_dataset_item_allows_source_provenance_url() -> None:
    item = build_stem_mas_item(
        item_id='sample-1',
        dataset_name='minif2f',
        source_index=0,
        question='question',
        prompt=[{'role': 'user', 'content': 'question'}],
        answer='answer',
        category='mathematics',
        metadata={'source_url': 'https://example.test/benchmark.jsonl'},
    )

    assert item.body['task_config']['metadata']['source_url'].endswith('benchmark.jsonl')


def test_dataset_item_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match='prompt must not be empty'):
        build_stem_mas_item(
            item_id='sample-1',
            dataset_name='minif2f',
            source_index=0,
            question='question',
            prompt=[],
            answer='answer',
            category='mathematics',
        )


@pytest.mark.parametrize(('item_id', 'dataset_name'), [('', 'minif2f'), ('   ', 'minif2f'), ('sample-1', '')])
def test_dataset_item_rejects_empty_identity(item_id: str, dataset_name: str) -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        build_stem_mas_item(
            item_id=item_id,
            dataset_name=dataset_name,
            source_index=0,
            question='question',
            prompt=[{'role': 'user', 'content': 'question'}],
            answer='answer',
            category='mathematics',
        )
