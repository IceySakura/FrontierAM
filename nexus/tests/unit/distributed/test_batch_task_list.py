from __future__ import annotations

import hashlib

import pytest

from nexus.distributed.batch.task_list import MAX_BATCH_ITEM_ID_LENGTH, is_valid_batch_item_id, normalize_batch_item_id


@pytest.mark.parametrize(
    'item_id',
    [
        'a',
        'item-000000000001',
        'case_1.part-2',
        'a' * MAX_BATCH_ITEM_ID_LENGTH,
    ],
)
def test_batch_item_id_accepts_canonical_ascii_values(item_id):
    assert is_valid_batch_item_id(item_id)


@pytest.mark.parametrize(
    'item_id',
    [
        None,
        '',
        ' item',
        'item ',
        'Item',
        'café',
        'cafe\u0301',
        '_item',
        'item/slash',
        'a' * (MAX_BATCH_ITEM_ID_LENGTH + 1),
    ],
)
def test_batch_item_id_rejects_noncanonical_values(item_id):
    assert not is_valid_batch_item_id(item_id)


def test_normalize_batch_item_id_preserves_valid_values():
    assert normalize_batch_item_id('case_1.part-2') == 'case_1.part-2'


@pytest.mark.parametrize(
    ('source', 'readable_prefix'),
    [
        ('Mini SWE Agent/Case A', 'mini-swe-agent-case-a'),
        ('_leading', 'leading'),
        ('café', 'caf'),
        ('x' * (MAX_BATCH_ITEM_ID_LENGTH + 1), 'x' * 32),
    ],
)
def test_normalize_batch_item_id_maps_invalid_values_deterministically(source, readable_prefix):
    normalized = normalize_batch_item_id(source)

    assert normalized.startswith(readable_prefix)
    assert normalized.endswith(f'-{hashlib.sha256(source.encode()).hexdigest()[:12]}')
    assert len(normalized) <= MAX_BATCH_ITEM_ID_LENGTH
    assert is_valid_batch_item_id(normalized)
