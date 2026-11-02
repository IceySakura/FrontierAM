"""Numeric trajectory payload representation, ownership, and serialization contracts."""

import asyncio
import copy
import json
import pickle

from typing import Any

import numpy as np
import pytest

from nexus.gateway.trajectory import Chain, TokenAccount, Trajectory, TrajectoryNode
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore


@pytest.mark.parametrize(
    ('name', 'values', 'dtype', 'shape'),
    [
        ('token_ids', [1000, 1001], np.int32, (2,)),
        ('raw_token_ids', [1000, 1001], np.int32, (2,)),
        ('logprobs', [-0.1, -0.2], np.float32, (2,)),
        ('topk_indices', [[1000, 1001], [1002, 1003]], np.int32, (2, 2)),
        ('topk_values', [[-0.1, -0.2], [-0.3, -0.4]], np.float32, (2, 2)),
        ('topk_indices', [], np.int32, (0, 0)),
        ('topk_values', [], np.float32, (0, 0)),
        ('token_ids', [], np.int32, (0,)),
    ],
)
def test_token_account_keeps_arrays_prepared_by_caller(
    name: str, values: Any, dtype: Any, shape: tuple[int, ...]
) -> None:
    prepared = np.array(values, dtype=dtype).reshape(shape)
    account = TokenAccount(**{name: prepared})
    value = getattr(account, name)
    assert value is prepared, 'the account must retain arrays without normalizing or copying them'
    assert value.dtype == dtype, 'the caller supplies the numeric dtype'
    assert value.shape == shape, 'empty top-k fields must remain rank two'
    np.testing.assert_array_equal(value, np.asarray(values, dtype=dtype).reshape(shape))


def test_token_account_views_and_store_refs_share_published_arrays() -> None:
    ids = np.arange(8, dtype=np.int32)
    account = TokenAccount(token_ids=ids, gen_start=2, gen_len=4, logprobs=np.arange(4, dtype=np.float32))
    record = account.build_store_payload_record('arrays', 0)
    assert record.token_ids is ids, 'publishing must not convert arrays back to lists or copy them'
    assert np.shares_memory(account.generated_ids, ids), 'sampled tokens must be an array view'
    assert np.shares_memory(account.glue_ids, ids), 'glue tokens must be an array view'
    assert np.shares_memory(account.trailing_ids, ids), 'trailing tokens must be an array view'
    store = InMemoryTrajectoryRecordStore('arrays')
    [ref] = asyncio.run(store.put_many([record]))
    [payload] = asyncio.run(store.get_many([ref]))
    account.discard_store_payload()
    account.hydrate_store_payload(payload)
    assert account.token_ids is ids, 'in-memory payload hydration must retain the published array'


@pytest.mark.parametrize('name', ['resp_topk_logprob_indice', 'resp_topk_logprob_value'])
@pytest.mark.parametrize('values', [[[1], [2, 3]], [[[1, 2]]]])
def test_topk_rejects_non_rectangular_or_non_matrix_fields(name: str, values: list) -> None:
    with pytest.raises(ValueError, match=rf'{name}.*shape \[gen_len, K\]'):
        TrajectoryAssembler._normalize_token_training({'extra_fields': {name: values}})


def test_array_dump_pickle_and_deepcopy_preserve_exact_values() -> None:
    account = TokenAccount(
        token_ids=np.arange(8192, dtype=np.int32)[::2],
        gen_len=4096,
        logprobs=np.linspace(-10, 0, 4096, dtype=np.float32),
        topk_indices=np.arange(32768, dtype=np.int32).reshape(4096, 8),
        topk_values=np.linspace(-10, 0, 32768, dtype=np.float32).reshape(4096, 8),
    )
    original = Trajectory(
        session_id='array-roundtrip',
        nodes={0: TrajectoryNode(0, 0, None, 'llm', children=[], token_account=account)},
        roots=[0],
        chains={'main': Chain('main', roots=[0])},
    )
    for restored in (
        Trajectory.from_dump(json.loads(json.dumps(original.dump()))),
        pickle.loads(pickle.dumps(original, protocol=5)),
        copy.deepcopy(original),
    ):
        for name in ('token_ids', 'logprobs', 'topk_indices', 'topk_values'):
            expected = getattr(account, name)
            actual = getattr(restored.nodes[0].token_account, name)
            assert actual.dtype == expected.dtype, name
            assert actual.shape == expected.shape, name
            assert actual.tobytes() == expected.tobytes(), name


def test_topk_normalization_copies_arrays_without_recursing_through_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOTE(jerryhrzhou): Patch the assembler's imported copy function rather than the shared module.
    import nexus.gateway.trajectory_assembler as assembler_module

    indices = np.arange(32, dtype=np.int32).reshape(8, 4)
    values = np.linspace(-1, 0, 32, dtype=np.float32).reshape(8, 4)
    training = {
        'prompt_ids': [1, 2],
        'raw_prompt_ids': [1, 2],
        'response_ids': list(range(8)),
        'response_logprobs': [0.1] * 8,
        'extra_fields': {'resp_topk_logprob_indice': indices, 'resp_topk_logprob_value': values},
    }
    copied = []
    original_copy = assembler_module.copy_value

    def record_copy(value: Any) -> Any:
        copied.append(value)
        return original_copy(value)

    monkeypatch.setattr(assembler_module, 'copy_value', record_copy)
    normalized = TrajectoryAssembler._normalize_token_training(training)
    assert all(value is not indices and value is not values for value in copied), 'top-k must bypass recursive copy'
    for name, expected in (('resp_topk_logprob_indice', indices), ('resp_topk_logprob_value', values)):
        actual = normalized['extra_fields'][name]
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected), 'normalization must isolate the producer snapshot'


def test_records_preserve_caller_arrays_without_dtype_conversion() -> None:
    tokens = np.array([1, 2], dtype=np.int32)
    logprobs = np.array([-0.1], dtype=np.float32)
    indices = np.array([[2, 3]], dtype=np.int32)
    values = np.array([[-0.1, -0.2]], dtype=np.float32)
    account = TokenAccount(
        token_ids=tokens, raw_token_ids=tokens, logprobs=logprobs, topk_indices=indices, topk_values=values
    )
    record = account.build_store_payload_record('arrays', 0)
    assert account.raw_token_ids is tokens, 'raw tokens must not be normalized in the account'
    assert record.token_ids is tokens, 'publishing must not change the supplied token array'
    assert record.logprobs is logprobs, 'publishing must not change the supplied logprob array'
    assert record.topk_indices is indices, 'publishing must not change the supplied top-k index array'
    assert record.topk_values is values, 'publishing must not change the supplied top-k value array'


def test_legacy_dump_converts_numeric_lists_at_load_boundary() -> None:
    original = Trajectory(
        session_id='legacy-arrays',
        nodes={0: TrajectoryNode(0, 0, None, 'llm', children=[], token_account=TokenAccount())},
        roots=[0],
        chains={'main': Chain('main', roots=[0])},
    )
    dumped = original.dump()
    dumped['nodes'][0]['token_account'].update(
        token_ids=[1, 2], logprobs=[-0.1], topk_indices=[[2, 3]], topk_values=[[-0.1, -0.2]], gen_start=1, gen_len=1
    )

    account = Trajectory.from_dump(dumped).nodes[0].token_account

    for name, dtype in (
        ('token_ids', np.int32),
        ('logprobs', np.float32),
        ('topk_indices', np.int32),
        ('topk_values', np.float32),
    ):
        value = getattr(account, name)
        assert value.dtype == dtype, 'legacy JSON lists must be normalized before constructing the account'
        np.testing.assert_array_equal(value, np.array(dumped['nodes'][0]['token_account'][name], dtype=dtype))
