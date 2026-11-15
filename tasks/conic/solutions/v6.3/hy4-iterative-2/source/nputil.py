"""Small NumPy-only helpers for CSC sparse matrices and NPZ I/O.

Kept deliberately free of scipy imports: importing scipy costs ~0.45 s on this
machine, which is a large fraction of the 1 s per-case budget.
"""
import json
import numpy as np


class CSC:
    """Compressed sparse column matrix with NumPy matvecs."""

    __slots__ = ("data", "indices", "indptr", "shape", "m", "n", "nnz", "col",
                 "_dense", "maxabs")

    def __init__(self, data, indices, indptr, shape):
        self.data = np.asarray(data, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.m, self.n = int(shape[0]), int(shape[1])
        self.nnz = self.data.size
        self.col = None
        self._dense = None
        self.maxabs = float(np.max(np.abs(self.data))) if self.nnz else 0.0

    def columns(self):
        if self.col is None:
            self.col = np.repeat(np.arange(self.n, dtype=np.int64),
                                 np.diff(self.indptr))
        return self.col

    def matvec(self, x):
        out = np.bincount(self.indices, weights=self.data * x[self.columns()],
                          minlength=self.m)
        return out

    def rmatvec(self, y):
        return np.bincount(self.columns(), weights=self.data * y[self.indices],
                           minlength=self.n)

    def dense(self):
        if self._dense is None:
            d = np.zeros((self.m, self.n), dtype=np.float64)
            np.add.at(d, (self.indices, self.columns()), self.data)
            self._dense = d
        return self._dense

    def canonical(self):
        """Return a CSC with duplicates summed and sorted indices."""
        # fast path: already canonical (sorted by (column,row), no duplicates)
        col = self.columns()
        valid = (self.indptr.size == self.n + 1 and self.indptr[-1] == self.nnz)
        if valid and self.nnz > 1:
            key = col.astype(np.int64) * (self.m + 1) + self.indices
            if np.all(np.diff(key) > 0):
                return self
        elif valid and self.nnz <= 1:
            return self
        key = col * self.m + self.indices
        order = np.argsort(key, kind="stable")
        c = col[order]
        r = self.indices[order]
        v = self.data[order]
        if len(v) == 0:
            return CSC(v, r, np.zeros(self.n + 1, dtype=np.int64), (self.m, self.n))
        keep = np.empty(len(v), dtype=bool)
        keep[0] = True
        keep[1:] = (c[1:] != c[:-1]) | (r[1:] != r[:-1])
        starts = np.flatnonzero(keep)
        cs = c[starts]
        rs = r[starts]
        vs = np.add.reduceat(v, starts)
        indptr = np.searchsorted(cs, np.arange(self.n + 1))
        return CSC(vs, rs, indptr, (self.m, self.n))

    def transpose_csc(self):
        """Return the transpose as a CSC matrix (shape n x m)."""
        row = self.columns()
        col = self.indices
        key = col * self.n + row
        order = np.argsort(key, kind="stable")
        indptr = np.searchsorted(col[order], np.arange(self.m + 1))
        return CSC(self.data[order], row[order], indptr, (self.n, self.m))

    def tril(self):
        """Lower-triangular part as a CSC matrix (keeps duplicate structure)."""
        rows = self.indices
        cols = self.columns()
        mask = rows >= cols
        return CSC.from_coo(rows[mask], cols[mask], self.data[mask],
                            (self.m, self.n))

    @staticmethod
    def from_coo(rows, cols, data, shape):
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        data = np.asarray(data, dtype=np.float64)
        if len(data) == 0:
            return CSC(data, rows, np.zeros(shape[1] + 1, dtype=np.int64), shape)
        order = np.lexsort((rows, cols))
        c, r, v = cols[order], rows[order], data[order]
        keep = np.empty(len(v), dtype=bool)
        keep[0] = True
        keep[1:] = (c[1:] != c[:-1]) | (r[1:] != r[:-1])
        starts = np.flatnonzero(keep)
        cs, rs = c[starts], r[starts]
        vs = np.add.reduceat(v, starts)
        indptr = np.searchsorted(cs, np.arange(shape[1] + 1))
        return CSC(vs, rs, indptr, shape)

    @staticmethod
    def vstack(mats):
        total_rows = sum(M.m for M in mats)
        n = mats[0].n
        rows, cols, data = [], [], []
        off = 0
        for M in mats:
            rows.append(M.indices + off)
            cols.append(M.columns())
            data.append(M.data)
            off += M.m
        return CSC.from_coo(np.concatenate(rows), np.concatenate(cols),
                            np.concatenate(data), (total_rows, n))


def load_problem(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        P = CSC(d["P_data"], d["P_indices"], d["P_indptr"],
                tuple(int(v) for v in d["P_shape"]))
        A = CSC(d["A_data"], d["A_indices"], d["A_indptr"],
                tuple(int(v) for v in d["A_shape"]))
        q = np.asarray(d["q"], dtype=np.float64).ravel()
        b = np.asarray(d["b"], dtype=np.float64).ravel()
        cones = meta["cones"]
        offset = float(meta.get("objective_offset", 0.0))
    return P, q, A, b, cones, offset


def save_result(path, status, x=None, s=None, z=None):
    import os
    meta = {"schema_version": 1, "status": status}
    arrays = {"metadata_json": np.array(json.dumps(meta, separators=(",", ":")))}
    if x is not None:
        arrays["x"] = np.asarray(x, dtype=np.float64).ravel()
    if s is not None:
        arrays["s"] = np.asarray(s, dtype=np.float64).ravel()
    if z is not None:
        arrays["z"] = np.asarray(z, dtype=np.float64).ravel()
    tmp = str(path) + ".tmp%d.npz" % os.getpid()
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def cone_layout(cones, m):
    """Return per-row cone metadata and total width."""
    rows_type = []
    total = 0
    for c in cones:
        kind = c["type"]
        dim = int(c["dim"])
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        rows_type.append((kind, dim, total, width))
        total += width
    if total != m:
        raise ValueError("cone widths do not partition b")
    return rows_type
