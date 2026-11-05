#!/usr/bin/env python3
"""Thin FAMEXP2 adapter for SciPy's unmodified expm_multiply implementation."""

import json
import struct
import sys

import numpy as np
from scipy.sparse import bmat, csr_matrix
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
minus_tl = (-t * L).tocsr()

for _ in range(header["requests"]):
    b0 = np.frombuffer(read_exact(8 * n), dtype="<f8")
    b1 = np.frombuffer(read_exact(8 * n), dtype="<f8")
    # exp(t*[[-L,b1],[0,0]]) * [b0;1] has the requested vector in its first n entries; built with sparse block
    # operations so adapter cost stays negligible next to expm_multiply itself.
    operator = bmat([[minus_tl, csr_matrix((t * b1).reshape(n, 1))], [None, csr_matrix((1, 1))]], format='csr')
    initial = np.empty(n + 1)
    initial[:n], initial[n] = b0, 1.0
    y = expm_multiply(operator, initial, traceA=trace_a)[:n]
    stdout.write(np.asarray(y, dtype="<f8").tobytes())
    stdout.flush()
