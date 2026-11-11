"""Sparse conic problem input and result serialization."""

import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class Problem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    b: np.ndarray
    cones: list

    @property
    def n(self):
        return self.q.size

    @property
    def m(self):
        return self.b.size


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        matrices = []
        for key in ("P", "A"):
            mat = sparse.csc_matrix(
                (data[key + "_data"], data[key + "_indices"], data[key + "_indptr"]),
                shape=tuple(data[key + "_shape"]),
            )
            mat.sum_duplicates()
            mat.eliminate_zeros()
            mat.sort_indices()
            matrices.append(mat)
        return Problem(matrices[0], data["q"].copy(), matrices[1],
                       data["b"].copy(), meta["cones"])


def save(path, result):
    arrays = {key: np.asarray(result[key], dtype=np.float64).reshape(-1)
              for key in ("x", "s", "z") if key in result}
    metadata = {"schema_version": 1, "status": result["status"]}
    arrays["metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
    with open(path, "wb") as stream:
        np.savez(stream, **arrays)


def blocks(cones):
    start = 0
    for cone in cones:
        kind, dim = cone["type"], cone["dim"]
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        yield kind, dim, slice(start, start + width)
        start += width
