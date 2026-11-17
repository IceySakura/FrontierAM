"""NPZ schema and a minimal CSR implementation used by the evaluator.

The evaluator intentionally does not trust scipy sparse object serialization.  A
problem is a plain NPZ containing the six standard CSR arrays for Q and A and
ordinary float64 vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional real array")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class SparseMatrix:
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        indices = np.asarray(self.indices, dtype=np.int64)
        indptr = np.asarray(self.indptr, dtype=np.int64)
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError("CSR arrays must be one-dimensional")
        if data.size != indices.size or indptr.size != self.shape[0] + 1:
            raise ValueError("inconsistent CSR lengths")
        if tuple(self.shape) != (int(self.shape[0]), int(self.shape[1])):
            raise ValueError("shape must contain two integers")
        if indptr[0] != 0 or indptr[-1] != data.size:
            raise ValueError("invalid CSR indptr")
        if np.any(indptr[1:] < indptr[:-1]) or np.any(indices < 0) or np.any(indices >= self.shape[1]):
            raise ValueError("invalid CSR indices")
        if not np.all(np.isfinite(data)):
            raise ValueError("CSR data must be finite")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "indptr", indptr)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size != self.shape[1]:
            raise ValueError("matvec dimension mismatch")
        result = np.zeros(self.shape[0], dtype=np.float64)
        for row in range(self.shape[0]):
            start, end = self.indptr[row], self.indptr[row + 1]
            result[row] = np.dot(self.data[start:end], x[self.indices[start:end]])
        return result

    def t_matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size != self.shape[0]:
            raise ValueError("transpose matvec dimension mismatch")
        result = np.zeros(self.shape[1], dtype=np.float64)
        for row in range(self.shape[0]):
            start, end = self.indptr[row], self.indptr[row + 1]
            result[self.indices[start:end]] += self.data[start:end] * x[row]
        return result

    def diagonal(self) -> np.ndarray:
        result = np.zeros(min(self.shape), dtype=np.float64)
        for row in range(self.shape[0]):
            start, end = self.indptr[row], self.indptr[row + 1]
            columns = self.indices[start:end]
            matches = np.flatnonzero(columns == row)
            if matches.size:
                result[row] = self.data[start + matches[-1]]
        return result

    def to_dense(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=np.float64)
        for row in range(self.shape[0]):
            start, end = self.indptr[row], self.indptr[row + 1]
            result[row, self.indices[start:end]] += self.data[start:end]
        return result


@dataclass(frozen=True)
class Problem:
    Q: SparseMatrix
    A: SparseMatrix
    c: np.ndarray
    b: np.ndarray
    c0: float
    metadata: dict[str, Any]

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[0]


def _csr_from_npz(data: Any, prefix: str) -> SparseMatrix:
    shape_raw = np.asarray(data[f"{prefix}_shape"])
    if shape_raw.shape != (2,):
        raise ValueError(f"{prefix}_shape must have length two")
    shape = (int(shape_raw[0]), int(shape_raw[1]))
    return SparseMatrix(data[f"{prefix}_data"], data[f"{prefix}_indices"],
                        data[f"{prefix}_indptr"], shape)


def _metadata(raw: Any, name: str) -> dict[str, Any]:
    array = np.asarray(raw)
    if array.ndim != 0 or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    try:
        value = json.loads(str(array))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{name} must contain schema_version=1")
    json.dumps(value, allow_nan=False)
    return value


def load_problem(path: str | Path) -> Problem:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "q_data", "q_indices", "q_indptr", "q_shape",
            "a_data", "a_indices", "a_indptr", "a_shape", "c", "b",
            "metadata_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing problem fields: {missing}")
        Q = _csr_from_npz(data, "q")
        A = _csr_from_npz(data, "a")
        c = _vector(data["c"], "c")
        b = _vector(data["b"], "b")
        metadata = _metadata(data["metadata_json"], "problem metadata")
    if Q.shape[0] != Q.shape[1] or Q.shape[0] != c.size:
        raise ValueError("Q and c dimensions disagree")
    if A.shape[1] != c.size or A.shape[0] != b.size:
        raise ValueError("A, b and c dimensions disagree")
    scale = max(1.0, float(np.max(np.abs(Q.data))) if Q.data.size else 1.0)
    dense_q = Q.to_dense()
    if float(np.max(np.abs(dense_q - dense_q.T))) > 1e-10 * scale:
        raise ValueError("Q must be symmetric")
    c0 = metadata.get("objective_offset", 0.0)
    if isinstance(c0, bool) or not isinstance(c0, (int, float)) or not np.isfinite(c0):
        raise ValueError("objective_offset must be finite")
    return Problem(Q=Q, A=A, c=c, b=b, c0=float(c0), metadata=metadata)


def load_output(path: str | Path, *, expected_n: int | None = None,
                expected_m: int | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("output is missing metadata_json")
        metadata = _metadata(data["metadata_json"], "output metadata")
        result: dict[str, Any] = {"metadata": metadata}
        for key in ("x", "y"):
            if key in data.files:
                result[key] = _vector(data[key], key)
        for key, size in (("x", expected_n), ("y", expected_m)):
            if size is not None and key in result and result[key].size != size:
                raise ValueError(f"{key} has wrong length")
        return result


def save_output(path: str | Path, x: np.ndarray, y: np.ndarray,
                *, status: str = "solved", objective: float | None = None) -> None:
    metadata: dict[str, Any] = {"schema_version": 1, "status": status}
    if objective is not None:
        if not np.isfinite(objective):
            raise ValueError("objective must be finite")
        metadata["objective"] = float(objective)
    np.savez(path, metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":")), dtype="U"),
             x=np.asarray(x, dtype=np.float64), y=np.asarray(y, dtype=np.float64))
