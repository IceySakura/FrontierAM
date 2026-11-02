import inspect
import json
import pickle

from dataclasses import asdict, fields

import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis import HybridNodeRef, NodeRecord, NodeRef, SegmentRecord, SegmentRef
from nexus.gateway.trajectory_store_backends.redis_store.types import Serializable
from nexus.gateway.trajectory_store_backends.types import (
    NodePayloadRecord,
    SegmentDataRecord,
)


@pytest.mark.parametrize(('record_type', 'base'), [(NodeRecord, NodePayloadRecord), (SegmentRecord, SegmentDataRecord)])
def test_record_fields_and_constructor_match_gateway_types(record_type, base):
    assert issubclass(record_type, Serializable)
    assert fields(record_type) == fields(base)
    assert inspect.signature(record_type) == inspect.signature(base)


def test_serializable_has_no_data_fields():
    assert vars(Serializable()) == {}


def test_hybrid_ref_preserves_old_ref_structure_and_serialization():
    for cls in (NodeRef, SegmentRef):
        assert [item.name for item in fields(cls)] == ['namespace', 'record_id']
        ref = cls('run', 'node')
        assert pickle.loads(pickle.dumps(ref)) == ref
        assert cls.__module__ == 'nexus.gateway.trajectory_store_backends.redis'
    ref = HybridNodeRef('run', 'node', 'hykvfs://run/' + 'a' * 32, 'hykvfs://run/' + 'b' * 32)
    assert isinstance(ref, NodeRef)
    assert HybridNodeRef(**json.loads(json.dumps(asdict(ref)))) == pickle.loads(pickle.dumps(ref)) == ref
    assert len({ref, pickle.loads(pickle.dumps(ref))}) == 1
    assert ref.blob_urls == {'routed_experts': ref.routed_experts_url, 'multi_modal_items': ref.multi_modal_items_url}


def test_node_payload_record_constructs_with_required_fields() -> None:
    record = NodePayloadRecord(trajectory_id='trajectory-1', node_id=3, token_ids=np.array([10, 11], dtype=np.int32))
    assert record.trajectory_id == 'trajectory-1'
    assert record.node_id == 3
    assert record.token_ids.tolist() == [10, 11]
    assert record.logprobs is None
    assert record.topk_indices is None
    assert record.topk_values is None
    assert record.routed_experts is None
    assert record.multi_modal_items is None


def test_segment_data_record_constructs_with_required_fields() -> None:
    record = SegmentDataRecord('trajectory-1', 'main@3', {'old_log_probs': [0.1]})
    assert record.trajectory_id == 'trajectory-1'
    assert record.segment_id == 'main@3'
    assert record.fields == {'old_log_probs': [0.1]}
