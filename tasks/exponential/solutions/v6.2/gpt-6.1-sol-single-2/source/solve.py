#!/usr/bin/env python3
"""Streaming sparse symmetric exponential propagation."""

import json
import math
import os
import struct
import sys

import numpy as np
from scipy import sparse
from scipy.special import ive


def read_exact(size):
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        block = os.read(0, min(size - offset, 1 << 20))
        if not block:
            raise EOFError("incomplete input")
        view[offset : offset + len(block)] = block
        offset += len(block)
    return result


def write_all(data):
    view = memoryview(data).cast("B")
    while view:
        count = os.write(1, view)
        if count <= 0:
            raise OSError("incomplete output")
        view = view[count:]


def scalar_factors(eigenvalues, t):
    exponential = np.exp(-t * eigenvalues)
    integral = np.full_like(eigenvalues, t)
    nonzero = eigenvalues != 0.0
    integral[nonzero] = (
        -np.expm1(-t * eigenvalues[nonzero]) / eigenvalues[nonzero]
    )
    return exponential, integral


class Propagator:
    def __init__(self, matrix, t, rtol):
        self.matrix = matrix
        self.t = t
        self.n = matrix.shape[0]
        self.mode = "polynomial"

        if t == 0.0:
            self.mode = "identity"
            return

        diagonal = matrix.diagonal()
        rows = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(matrix.indptr))
        is_diagonal = not np.any(
            (matrix.indices != rows) & (matrix.data != 0.0)
        )
        del rows
        if is_diagonal:
            self.mode = "diagonal"
            self.exponential, self.integral = scalar_factors(diagonal, t)
            return

        if self.n <= 64:
            self.mode = "dense"
            eigenvalues, self.eigenvectors = np.linalg.eigh(matrix.toarray())
            self.exponential, self.integral = scalar_factors(eigenvalues, t)
            return

        radius = float(np.max(np.asarray(abs(matrix).sum(axis=1))))
        if radius == 0.0:
            self.mode = "diagonal"
            self.exponential = np.ones(self.n)
            self.integral = np.full(self.n, t)
            return

        self.factor = 2.0 / radius
        z = 0.5 * t * radius
        tolerance = max(1e-18, rtol * 1e-5 / math.sqrt(self.n))
        log_bound = math.log(32.0 / tolerance)
        degree_bound = int(math.ceil(math.sqrt(2.0 * z * log_bound) + log_bound + 8))
        coefficients = ive(np.arange(degree_bound + 1, dtype=np.float64), z)
        coefficients[1:] *= 2.0
        tails = np.cumsum(coefficients[::-1])[::-1]
        negligible = np.flatnonzero(tails <= tolerance)
        if negligible.size:
            coefficients = coefficients[: max(1, int(negligible[0]))]
        self.coefficients = coefficients

    def apply_exponential(self, vector):
        coefficients = self.coefficients
        previous = vector
        result = coefficients[0] * previous
        if len(coefficients) == 1:
            return result

        current = previous - self.factor * self.matrix.dot(previous)
        result += coefficients[1] * current
        for coefficient in coefficients[2:]:
            following = (
                2.0 * current
                - (2.0 * self.factor) * self.matrix.dot(current)
                - previous
            )
            result += coefficient * following
            previous, current = current, following
        return result

    def solve(self, b0, b1):
        if self.mode == "identity":
            return b0.copy()
        if self.mode == "diagonal":
            return self.exponential * b0 + self.integral * b1
        if self.mode == "dense":
            vectors = self.eigenvectors
            return vectors @ (
                self.exponential * (vectors.T @ b0)
                + self.integral * (vectors.T @ b1)
            )
        initial = b0 + self.t * b1
        return self.apply_exponential(initial)


def main():
    write_all(b"READY\n")
    if read_exact(8) != b"FAMEXP2\n":
        raise ValueError("invalid protocol magic")
    length = struct.unpack("<I", read_exact(4))[0]
    metadata = json.loads(read_exact(length))
    n = int(metadata["n"])
    nnz = int(metadata["nnz"])
    indptr = np.frombuffer(read_exact(8 * (n + 1)), dtype="<u8").astype(np.int32)
    indices = np.frombuffer(read_exact(4 * nnz), dtype="<u4").astype(np.int32)
    values = np.frombuffer(read_exact(8 * nnz), dtype="<f8")
    matrix = sparse.csr_matrix((values, indices, indptr), shape=(n, n), copy=False)
    propagator = Propagator(matrix, float(metadata["t"]), float(metadata["rtol"]))

    for _ in range(int(metadata["requests"])):
        b0 = np.frombuffer(read_exact(8 * n), dtype="<f8")
        b1 = np.frombuffer(read_exact(8 * n), dtype="<f8")
        result = np.ascontiguousarray(propagator.solve(b0, b1), dtype="<f8")
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite propagation result")
        write_all(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
