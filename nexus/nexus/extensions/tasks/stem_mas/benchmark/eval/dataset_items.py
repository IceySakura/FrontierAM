"""Build portable Nexus DatasetItems for STEM-MAS benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nexus.dataset import DatasetItem


DATASET_TASK_FIELDS = frozenset({
    'evaluator_type',
    'lean4_verifier_benchmark',
    'lean4_verifier_candidate_kind',
    'lean4_verifier_candidate_policy',
    'lean4_verifier_manifest',
    'lean4_verifier_task_id',
})

DATASET_METADATA_FIELDS = frozenset({
    'hf_file',
    'hf_id',
    'hf_revision',
    'source',
    'source_format',
    'source_url',
    'split',
    'subject',
    'tag',
    'tags',
})


def build_stem_mas_item(
    *,
    item_id: str,
    dataset_name: str,
    source_index: int,
    question: str,
    prompt: list[dict[str, Any]],
    answer: str,
    category: str,
    task_fields: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetItem:
    """Create a data-only partial RunInput without environment or solver settings."""

    if not item_id.strip():
        raise ValueError('Dataset item id must not be empty')
    if not dataset_name.strip():
        raise ValueError('Dataset name must not be empty')
    if not prompt:
        raise ValueError('Dataset item prompt must not be empty')

    resolved_task_fields = dict(task_fields or {})
    unsupported = set(resolved_task_fields) - DATASET_TASK_FIELDS
    if unsupported:
        names = ', '.join(sorted(unsupported))
        raise ValueError(f'Dataset task fields contain execution-owned or unsupported fields: {names}')

    task_metadata = dict(metadata or {})
    _validate_metadata_fields(task_metadata, allowed=DATASET_METADATA_FIELDS, path='metadata')
    task_metadata.setdefault('index', source_index)
    task_metadata.setdefault('task_id', item_id)
    task_metadata.setdefault('question', question)
    task_metadata.setdefault('answer', answer)
    task_metadata.setdefault('data_source', dataset_name)
    task_metadata.setdefault('ability', category)

    task_config: dict[str, Any] = {
        'task_id': item_id,
        'category': category,
        'question': question,
        'prompt': prompt,
        'answer': answer,
        'metadata': task_metadata,
    }
    task_config.update(resolved_task_fields)

    dataset_metadata = {
        'index': source_index,
        'dataset_name': dataset_name,
    }
    return DatasetItem(
        id=item_id,
        body={'task_name': 'stem_mas', 'task_config': task_config},
        metadata=dataset_metadata,
    )


def _validate_metadata_fields(value: Mapping[str, Any], *, allowed: frozenset[str], path: str) -> None:
    unsupported = set(value) - allowed
    if unsupported:
        names = ', '.join(sorted(unsupported))
        raise ValueError(f'Dataset {path} contains unsupported fields: {names}')
    _reject_environment_interpolation(value, path=path)


def _reject_environment_interpolation(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            _reject_environment_interpolation(child, path=f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_environment_interpolation(child, path=f'{path}[{index}]')
        return
    if isinstance(value, str) and '${oc.env:' in value:
        raise ValueError(f'Dataset metadata contains environment interpolation at {path}')


__all__ = [
    'DATASET_METADATA_FIELDS',
    'DATASET_TASK_FIELDS',
    'build_stem_mas_item',
]
