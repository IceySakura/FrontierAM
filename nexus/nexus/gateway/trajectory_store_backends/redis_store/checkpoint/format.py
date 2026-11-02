"""Typed Parquet schema, numeric columns and checkpoint blob references."""

from __future__ import annotations

import math

from pathlib import Path
from typing import Any

import numpy as np

from nexus.gateway.types import LOGPROB_DTYPE, TOKEN_ID_DTYPE

from ..file_storage.types import LARGE_FIELDS


ARRAY_DTYPES = {
    'token_ids': np.dtype(TOKEN_ID_DTYPE),
    'logprobs': np.dtype(LOGPROB_DTYPE),
    'topk_indices': np.dtype(TOKEN_ID_DTYPE),
    'topk_values': np.dtype(LOGPROB_DTYPE),
}
PARQUET_BATCH_SIZE = 16
CHECKPOINT_URI_PREFIX = 'ckpt://'


def parse_blob_reference(value: str) -> Path:
    path = Path(value.removeprefix(CHECKPOINT_URI_PREFIX))
    if (
        not value.startswith(CHECKPOINT_URI_PREFIX)
        or path.is_absolute()
        or not path.parts
        or path.parts[0] != 'data'
        or '..' in path.parts
    ):
        raise ValueError(f'invalid checkpoint blob path: {value!r}')
    return path


def build_schema(dump_id: str):
    import pyarrow as pa

    array_types = {
        name: pa.struct([
            ('shape', pa.list_(pa.int64())),
            ('values', pa.large_list(pa.from_numpy_dtype(dtype))),
        ])
        for name, dtype in ARRAY_DTYPES.items()
    }
    return pa.schema(
        [
            ('entry_id', pa.int64()),
            ('kind', pa.string()),
            ('trajectory_id', pa.string()),
            ('node_id', pa.int64()),
            ('segment_id', pa.string()),
            *array_types.items(),
            *((name, pa.string()) for name in LARGE_FIELDS),
            ('segment_fields', pa.map_(pa.string(), pa.binary())),
        ],
        metadata={b'nexus_format': b'trajectory_parquet_v1', b'dump_id': dump_id.encode()},
    )


def encode_array_column(arrays: list[Any], name, arrow_type):
    """Build Arrow buffers from ndarray memory without expanding elements to Python."""
    import pyarrow as pa

    dtype = ARRAY_DTYPES[name]
    for value in arrays:
        if value is not None and (not isinstance(value, np.ndarray) or value.dtype != dtype):
            raise TypeError(f'Node {name} must be an ndarray with dtype {dtype}')
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([0 if value is None else value.size for value in arrays], out=offsets[1:])
    flat_arrays = [value.reshape(-1) for value in arrays if value is not None]
    flat_values = np.concatenate(flat_arrays) if flat_arrays else np.empty(0, dtype=dtype)
    value_type = arrow_type.field('values').type.value_type
    children = [
        pa.array(
            [None if value is None else list(value.shape) for value in arrays], type=arrow_type.field('shape').type
        ),
        pa.LargeListArray.from_arrays(
            offsets,
            pa.array(flat_values, type=value_type),
            mask=pa.array([value is None for value in arrays]),
        ),
    ]
    return pa.StructArray.from_arrays(children, fields=list(arrow_type), mask=pa.array([v is None for v in arrays]))


def decode_array(value, name, *, summaries):
    shape = value['shape'].as_py()
    if summaries:
        return {'dtype': ARRAY_DTYPES[name].str, 'shape': shape, 'elements': math.prod(shape)}
    return value['values'].values.to_numpy(zero_copy_only=False).reshape(shape)
