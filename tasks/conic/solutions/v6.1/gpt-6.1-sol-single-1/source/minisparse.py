"""Small CSC adapter over the installed SciPy numeric extensions."""

import glob
import importlib.util
import sys
import sysconfig
import numpy as np


def extension(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, glob.glob(path)[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BASE = sysconfig.get_paths()["purelib"] + "/scipy"
TOOLS = extension("_sparsetools", _BASE + "/sparse/_sparsetools*.so")
SUPER = extension("_superlu", _BASE + "/sparse/linalg/_dsolve/_superlu*.so")


class CSC:
    def __init__(self, m, n, data=None, rows=None, ptr=None):
        self.m, self.n = int(m), int(n)
        self.data = np.asarray(np.array([]) if data is None else data, dtype=np.float64)
        self.rows = np.asarray(np.array([], np.int32) if rows is None else rows, dtype=np.int32)
        self.ptr = np.zeros(n + 1, np.int32) if ptr is None else np.asarray(ptr, dtype=np.int32)
        self._cols = self._T = None

    @property
    def nnz(self):
        return len(self.data)

    @property
    def cols(self):
        if self._cols is None:
            self._cols = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.ptr))
        return self._cols

    def canonical(self):
        TOOLS.csr_sort_indices(self.n, self.ptr, self.rows, self.data)
        TOOLS.csr_sum_duplicates(self.n, self.m, self.ptr, self.rows, self.data)
        size = int(self.ptr[-1])
        self.rows, self.data = self.rows[:size], self.data[:size]
        self._cols = self._T = None
        return self

    @classmethod
    def coo(cls, m, n, row, col, data):
        row, col, data = np.asarray(row), np.asarray(col), np.asarray(data)
        order = np.argsort(col, kind="stable")
        ptr = np.r_[0, np.cumsum(np.bincount(col, minlength=n))].astype(np.int32)
        return cls(m, n, data[order], row[order], ptr).canonical()

    @classmethod
    def diagonal(cls, data):
        n = len(data)
        return cls(n, n, data, np.arange(n, dtype=np.int32), np.arange(n + 1, dtype=np.int32))

    def mul(self, x):
        y = np.zeros(self.m)
        TOOLS.csc_matvec(self.m, self.n, self.ptr, self.rows, self.data, x, y)
        return y

    def tmul(self, x):
        y = np.zeros(self.n)
        TOOLS.csr_matvec(self.n, self.m, self.ptr, self.rows, self.data, x, y)
        return y

    @property
    def T(self):
        if self._T is None:
            ptr = np.empty(self.m + 1, np.int32)
            rows = np.empty(self.nnz, np.int32)
            data = np.empty(self.nnz)
            TOOLS.csr_tocsc(self.n, self.m, self.ptr, self.rows, self.data,
                            ptr, rows, data)
            self._T = CSC(self.n, self.m, data, rows, ptr)
            self._T._T = self
        return self._T

    def matmul(self, other):
        capacity = TOOLS.csr_matmat_maxnnz(
            other.n, self.m, other.ptr, other.rows, self.ptr, self.rows)
        ptr = np.empty(other.n + 1, np.int32)
        rows, data = np.empty(capacity, np.int32), np.empty(capacity)
        TOOLS.csr_matmat(other.n, self.m,
            other.ptr, other.rows, other.data, self.ptr, self.rows, self.data,
            ptr, rows, data)
        size = int(ptr[-1])
        return CSC(self.m, other.n, data[:size], rows[:size], ptr).canonical()

    def add(self, other):
        ptr = np.empty(self.n + 1, np.int32)
        rows = np.empty(self.nnz + other.nnz, np.int32)
        data = np.empty(self.nnz + other.nnz)
        TOOLS.csr_plus_csr(self.n, self.m,
            self.ptr, self.rows, self.data, other.ptr, other.rows, other.data,
            ptr, rows, data)
        size = int(ptr[-1])
        return CSC(self.m, self.n, data[:size], rows[:size], ptr).canonical()

    def scale(self, row=None, col=None, scalar=1.):
        data = self.data * scalar
        if row is not None:
            data *= row[self.rows]
        if col is not None:
            data *= col[self.cols]
        return CSC(self.m, self.n, data, self.rows, self.ptr)

    def select_rows(self, indices):
        pos = np.full(self.m, -1, dtype=np.int32)
        pos[indices] = np.arange(len(indices))
        use = pos[self.rows] >= 0
        counts = np.bincount(self.cols[use], minlength=self.n)
        ptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
        return CSC(len(indices), self.n, self.data[use], pos[self.rows[use]], ptr).canonical()

    def maxcol(self):
        out = np.zeros(self.n)
        use = np.flatnonzero(np.diff(self.ptr))
        if len(use):
            out[use] = np.maximum.reduceat(np.abs(self.data), self.ptr[use])
        return out

    def maxrow(self):
        return self.T.maxcol()

    def factor(self, order="COLAMD", pivot=.1):
        return SUPER.gstrf(self.n, self.nnz, self.data, self.rows, self.ptr,
            csc_construct_func=lambda *a: None, ilu=False,
            options={"ColPerm": order, "DiagPivotThresh": pivot})


def block_matrix(blocks, row_sizes, col_sizes):
    m, n = sum(row_sizes), sum(col_sizes)
    roff = np.r_[0, np.cumsum(row_sizes)]
    coff = np.r_[0, np.cumsum(col_sizes)]
    counts = np.zeros(n, dtype=np.int32)
    for row in blocks:
        for j, mat in enumerate(row):
            if mat is not None:
                counts[coff[j]:coff[j + 1]] += np.diff(mat.ptr)
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
    cursor = ptr[:-1].copy()
    rows, data = np.empty(int(ptr[-1]), np.int32), np.empty(int(ptr[-1]))
    for i, row in enumerate(blocks):
        for j, mat in enumerate(row):
            if mat is None or mat.nnz == 0:
                continue
            cols = mat.cols
            positions = cursor[coff[j] + cols] + (np.arange(mat.nnz) - mat.ptr[cols])
            rows[positions] = mat.rows + roff[i]
            data[positions] = mat.data
            cursor[coff[j]:coff[j + 1]] += np.diff(mat.ptr)
    return CSC(m, n, data, rows, ptr).canonical()
