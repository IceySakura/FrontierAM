"""Portable sparse numeric inputs; no solver-specific preprocessing or pickle."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class Problem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    b: np.ndarray
    cones: list
    name: str = ""
    objective_offset: float = 0.0
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.P = sparse.csc_matrix(self.P, dtype=np.float64)
        self.A = sparse.csc_matrix(self.A, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        n, m = self.q.size, self.b.size
        if self.P.shape != (n, n) or self.A.shape != (m, n):
            raise ValueError("Input matrix/vector dimensions differ")
        for matrix in (self.P, self.A):
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            matrix.check_format(full_check=True)
        if not all(np.isfinite(v).all() for v in (self.P.data, self.A.data, self.q, self.b)):
            raise ValueError("Nonfinite input coefficient")
        if not np.isfinite(self.objective_offset):
            raise ValueError("Nonfinite objective constant")
        delta = self.P - self.P.T
        if delta.nnz and np.max(np.abs(delta.data)) > 1e-12 * max(1.0, np.max(np.abs(self.P.data))):
            raise ValueError("P must contain the full symmetric matrix")
        dimension = 0
        for cone in self.cones:
            kind, size = cone["type"], cone["dim"]
            if type(size) is not int or size <= 0:
                raise ValueError("Cone dimensions must be positive integers")
            if kind not in {"zero", "nonnegative", "soc", "exponential", "psd"}:
                raise ValueError(f"Unsupported cone: {kind}")
            if kind == "exponential" and size != 3:
                raise ValueError("Exponential cone dimension must be 3")
            if kind == "soc" and size < 2:
                raise ValueError("SOC dimension must be at least 2")
            dimension += size * (size + 1) // 2 if kind == "psd" else size
        if dimension != m:
            raise ValueError("Cone blocks do not partition all constraint rows")
        return self


def save_problem(problem, path):
    problem.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"q": problem.q, "b": problem.b}
    for prefix, matrix in (("P", problem.P), ("A", problem.A)):
        arrays.update({f"{prefix}_data": matrix.data, f"{prefix}_indices": matrix.indices,
                       f"{prefix}_indptr": matrix.indptr, f"{prefix}_shape": np.array(matrix.shape)})
    arrays["metadata_json"] = np.array(json.dumps({"schema_version": 1, "name": problem.name,
        "cones": problem.cones, "objective_offset": problem.objective_offset,
        "metadata": problem.metadata}, sort_keys=True, ensure_ascii=False))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_problem(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        if meta["schema_version"] != 1:
            raise ValueError("Unsupported input schema")
        matrices = {}
        for prefix in ("P", "A"):
            matrices[prefix] = sparse.csc_matrix((data[f"{prefix}_data"], data[f"{prefix}_indices"],
                data[f"{prefix}_indptr"]), shape=tuple(data[f"{prefix}_shape"]))
        problem = Problem(matrices["P"], data["q"], matrices["A"], data["b"], meta["cones"],
                          meta["name"], meta["objective_offset"], meta["metadata"])
    return problem.validate()
