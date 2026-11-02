import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.blob_codec import decode_blob, encode_blob


torch = pytest.importorskip('torch')


def test_round_trip_keeps_structure_dtypes_and_shapes():
    tensors = {
        'bf16': torch.randn(3, 32, 48).to(torch.bfloat16),
        'transposed': torch.arange(24).reshape(4, 6).t(),
        'empty': torch.empty(0, 5),
        'scalar': torch.tensor(3.5),
        'grad': torch.ones(2, requires_grad=True),  # pickled as usual
    }
    value = {
        'images': [{'tensors': tensors, 'meta': ('a', 1.5, None)}],
        'routed_experts': np.arange(12, dtype=np.int16).reshape(3, 4),
    }
    restored = decode_blob(encode_blob(value))
    item = restored['images'][0]
    for name, expected in tensors.items():
        actual = item['tensors'][name]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)
    assert item['tensors']['grad'].requires_grad and item['meta'] == ('a', 1.5, None)
    np.testing.assert_array_equal(restored['routed_experts'], value['routed_experts'])


def test_decoded_tensors_are_views_of_the_blob():
    blob = encode_blob({'t': torch.arange(1024, dtype=torch.int32)})
    decode_blob(blob)['t'].add_(1)
    assert torch.equal(decode_blob(blob)['t'], torch.arange(1, 1025, dtype=torch.int32))
