#!/usr/bin/env python3
"""Thin FAMEXP2 adapter for SciPy's unmodified expm_multiply implementation."""

import json
import struct
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_exact(length):
    result = bytearray()
    while len(result) < length:
        block = stdin.read(length - len(result))
        if not block:
            raise EOFError("short input")
        result.extend(block)
    return bytes(result)


stdout.write(b"READY\n")
stdout.flush()
if read_exact(8) != b"FAMEXP2\n":
    raise ValueError("bad magic")
header = json.loads(read_exact(struct.unpack("<I", read_exact(4))[0]))
n, nnz = header["n"], header["nnz"]
indptr = np.frombuffer(read_exact(8 * (n + 1)), dtype="<u8").astype(np.int64)
indices = np.frombuffer(read_exact(4 * nnz), dtype="<u4").astype(np.int32)
data = np.frombuffer(read_exact(8 * nnz), dtype="<f8")
t = float(header["t"])
L = csr_matrix((data, indices, indptr), shape=(n, n))
trace_a = -t * float(L.diagonal().sum())

for _ in range(header["requests"]):
    b0 = np.frombuffer(read_exact(8 * n), dtype="<f8")
    b1 = np.frombuffer(read_exact(8 * n), dtype="<f8")
    # exp(t*[[-L,b1],[0,0]]) * [b0;1] has the requested vector in its first n entries.
    row_nnz = np.diff(indptr) + 1
    aug_indptr = np.empty(n + 2, dtype=np.int64)
    aug_indptr[0] = 0
    np.cumsum(row_nnz, out=aug_indptr[1 : n + 1])
    aug_indptr[n + 1] = aug_indptr[n]
    aug_indices = np.empty(nnz + n, dtype=np.int32)
    aug_data = np.empty(nnz + n, dtype=np.float64)
    for row in range(n):
        old_lo, old_hi = indptr[row], indptr[row + 1]
        new_lo = aug_indptr[row]
        count = old_hi - old_lo
        aug_indices[new_lo : new_lo + count] = indices[old_lo:old_hi]
        aug_data[new_lo : new_lo + count] = -t * data[old_lo:old_hi]
        aug_indices[new_lo + count] = n
        aug_data[new_lo + count] = t * b1[row]
    operator = csr_matrix((aug_data, aug_indices, aug_indptr), shape=(n + 1, n + 1))
    initial = np.empty(n + 1)
    initial[:n], initial[n] = b0, 1.0
    y = expm_multiply(operator, initial, traceA=trace_a)[:n]
    stdout.write(np.asarray(y, dtype="<f8").tobytes())
    stdout.flush()
