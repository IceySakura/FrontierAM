"""Encode payload fields with tensors and arrays stored as raw bytes.

Plain ``pickle`` routes every torch tensor through ``torch.save``/``torch.load``:
two full copies plus format parsing under the GIL. Here pickle encodes only the
container structure; the bytes of CPU tensors and numpy arrays follow it, and
decoding wraps them with ``frombuffer`` instead of copying.

Layout: ``u64 meta length | pickle(meta) | arrays``, each array 64-byte aligned.

Decoded arrays are views of the input buffer and keep it alive. Tensors that
share a storage are stored as independent copies.
"""

from __future__ import annotations

import io
import math
import pickle
import struct
import sys

from typing import Any


_LENGTH = struct.Struct('<Q')
_ALIGN = 64


def _align(offset: int) -> int:
    return -(-offset // _ALIGN) * _ALIGN


def _is_raw_array(obj: Any) -> bool:
    """CPU tensors and numpy arrays go out of band; anything else is pickled as usual."""
    torch, numpy = sys.modules.get('torch'), sys.modules.get('numpy')
    if torch is not None and type(obj) is torch.Tensor:
        return obj.device.type == 'cpu' and obj.layout == torch.strided and not obj.requires_grad
    return numpy is not None and type(obj) is numpy.ndarray and not obj.dtype.hasobject


class _Pickler(pickle.Pickler):
    def __init__(self, file, arrays: list):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._arrays = arrays

    def persistent_id(self, obj):  # noqa: D102 - pickle hook
        if _is_raw_array(obj):
            self._arrays.append(obj)
            return len(self._arrays) - 1
        return None


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, arrays: list):
        super().__init__(file)
        self._arrays = arrays

    def persistent_load(self, pid):  # noqa: D102 - pickle hook
        return self._arrays[pid]


def encode_blob(value: Any) -> bytearray:
    structure = io.BytesIO()
    arrays: list = []
    _Pickler(structure, arrays).dump(value)
    table, raw = [], []
    if arrays:
        import numpy  # also backs Tensor.numpy() below
    for array in arrays:
        if isinstance(array, numpy.ndarray):
            table.append(('numpy', array.dtype.str, array.shape))
            raw.append(numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint8))
        else:
            import torch

            table.append(('torch', str(array.dtype).removeprefix('torch.'), tuple(array.shape)))
            # Reinterpret as bytes before .numpy(): numpy has no bfloat16.
            raw.append(array.contiguous().reshape(-1).view(torch.uint8).numpy())
    meta = pickle.dumps({'structure': structure.getvalue(), 'arrays': table}, protocol=pickle.HIGHEST_PROTOCOL)
    offsets, end = [], _LENGTH.size + len(meta)
    for data in raw:
        offsets.append(_align(end))
        end = offsets[-1] + data.nbytes
    out = bytearray(end)
    _LENGTH.pack_into(out, 0, len(meta))
    out[_LENGTH.size : _LENGTH.size + len(meta)] = meta
    view = memoryview(out)
    for data, offset in zip(raw, offsets, strict=True):
        view[offset : offset + data.nbytes] = data
    return out


def decode_blob(blob: Any) -> Any:
    view = memoryview(blob)
    (length,) = _LENGTH.unpack_from(view)
    meta = pickle.loads(view[_LENGTH.size : _LENGTH.size + length])
    arrays, end = [], _LENGTH.size + length
    for kind, dtype, shape in meta['arrays']:
        offset, count = _align(end), math.prod(shape)
        if kind == 'numpy':
            import numpy

            array = numpy.frombuffer(view, numpy.dtype(dtype), count, offset).reshape(shape)
        else:
            import torch

            dtype = getattr(torch, dtype)
            # torch.frombuffer rejects an empty range.
            flat = (
                torch.frombuffer(view, dtype=dtype, count=count, offset=offset)
                if count
                else torch.empty(0, dtype=dtype)
            )
            array = flat.view(shape)
        arrays.append(array)
        end = offset + array.nbytes
    return _Unpickler(io.BytesIO(meta['structure']), arrays).load()
