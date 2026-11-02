import json
import os
import pickle
import subprocess
import sys
import zlib

from collections import Counter
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from nexus.gateway.trajectory_store_backends.redis import NodeRecord, RedisTrajectoryRecordStore, SegmentRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.types import Serializable
from tests.unit.gateway.redis_store.payloads import BinaryPayload, JsonPayload, SamplePayload


pytestmark = pytest.mark.integration


def test_custom_codecs_in_mixed_cross_slot_batch_and_new_process(backend_config, monkeypatch):
    from redis.cluster import key_slot

    monkeypatch.setattr(
        'nexus.gateway.trajectory_store_backends.redis_store.client.new_node_id',
        iter(['json{codec-json}', 'binary{codec-binary}', 'pickle{codec-pickle}']).__next__,
    )
    with RedisClient(**backend_config, key_prefix=uuid4().hex) as client:
        payloads = [JsonPayload('跨进程'), BinaryPayload(-42), SamplePayload(text='default')]
        refs = client.put_multi_node(payloads)
        assert all(isinstance(ref, str) for ref in refs)
        assert len({key_slot(client._node_key(ref)) for ref in refs}) == 3
        read_name = 'mget_nonatomic' if callable(getattr(client._r, 'mget_nonatomic', None)) else 'mget'
        read = Mock(wraps=getattr(client._r, read_name))
        monkeypatch.setattr(client._r, read_name, read)
        order = [1, 0, 2, 0]
        assert client.get_multi_node([(refs[i], type(payloads[i])) for i in order]) == [payloads[i] for i in order]
        read.assert_called_once()

        code = """
import json, sys
from tests.unit.gateway.redis_store.payloads import BinaryPayload, JsonPayload, SamplePayload
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
data = json.load(sys.stdin)
refs = data['ids']
with RedisClient(**data['config'], key_prefix=data['prefix']) as client:
    assert client.get_multi_node([
        (refs[1], BinaryPayload), (refs[0], JsonPayload), (refs[2], SamplePayload), (refs[0], JsonPayload),
    ]) == [BinaryPayload(-42), JsonPayload('跨进程'), SamplePayload(text='default'), JsonPayload('跨进程')]
"""
        subprocess.run(
            [sys.executable, '-c', code],
            input=json.dumps({'config': backend_config, 'prefix': client.key_prefix, 'ids': refs}),
            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[4])},
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        client.delete_multi_node(refs)


async def test_trajectory_read_preflight_and_cas_use_record_codecs(backend_config, monkeypatch):
    calls = Counter()

    def encode_node(record):
        calls['node_encode'] += 1
        return json.dumps(vars(record)).encode()

    def decode_node(blob):
        calls['node_decode'] += 1
        return NodeRecord(**json.loads(blob))

    def encode_segment(record):
        calls['segment_encode'] += 1
        # Vary the valid encoding on every call to expose re-encoding of CAS expectations.
        return zlib.compress(
            json.dumps({
                'encoding_sequence': calls['segment_encode'],
                'trajectory_id': record.trajectory_id,
                'segment_id': record.segment_id,
                'fields': {name: value.hex() for name, value in record.fields.items()},
            }).encode()
        )

    def decode_segment(blob):
        calls['segment_decode'] += 1
        data = json.loads(zlib.decompress(blob))
        return SegmentRecord(
            data['trajectory_id'],
            data['segment_id'],
            {name: bytes.fromhex(value) for name, value in data['fields'].items()},
        )

    def unexpected_base_decode(blob):
        raise AssertionError('custom records must not use the base pickle decoder')

    monkeypatch.setattr(NodeRecord, 'serialize', encode_node)
    monkeypatch.setattr(NodeRecord, 'deserialize', staticmethod(decode_node))
    monkeypatch.setattr(SegmentRecord, 'serialize', encode_segment)
    monkeypatch.setattr(SegmentRecord, 'deserialize', staticmethod(decode_segment))
    monkeypatch.setattr(Serializable, 'deserialize', staticmethod(unexpected_base_decode))

    with RedisTrajectoryRecordStore('codecs', {**backend_config, 'key_prefix': uuid4().hex}) as store:
        node, segment, conflict = await store.put_many([
            NodeRecord('t', 1, [1, 2], logprobs=[-0.1]),
            SegmentRecord('t', 's1', {}),
            SegmentRecord('t', 's2', {'reward': 1}),
        ])
        assert await store.get_many([conflict, node, segment, node], fields=['reward', 'token_ids']) == [
            {'reward': 1},
            {'token_ids': [1, 2]},
            {},
            {'token_ids': [1, 2]},
        ]
        with pytest.raises(ValueError, match='conflicting record field'):
            await store.put_fields_many({segment: {'late': 1}, conflict: {'reward': 2}})
        assert await store.get_many([segment]) == [{}]

        cas = store._client.compare_and_set_many

        def compete(updates):
            calls['cas'] += 1
            if calls['cas'] == 1:
                ref, old, _ = updates[0]
                concurrent = SegmentRecord.deserialize(old).with_fields({'other': pickle.dumps(2)})
                store._client._r.set(store._client._node_key(ref), concurrent.serialize())
            return cas(updates)

        monkeypatch.setattr(store._client, 'compare_and_set_many', compete)
        await store.put_fields_many({segment: {'late': 1}})
        assert calls['cas'] == 2
        await store.put_fields_many({segment: {'late': 1}})
        assert calls['cas'] == 2
        assert await store.get_many([segment]) == [{'other': 2, 'late': 1}]
        assert calls['node_encode'] > 0
        assert calls['node_decode'] > 0
        assert calls['segment_encode'] > 0
        assert calls['segment_decode'] > 0
        await store.release_many([node, segment, conflict])
