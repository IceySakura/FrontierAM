"""Minimal CSC/CSR operations backed by the installed SciPy sparse kernels."""
import os
import numpy as np
from kernels import extension

st = extension("scipy.sparse._sparsetools", "sparse/_sparsetools*.so")
su = extension("scipy.sparse.linalg._dsolve._superlu",
               "sparse/linalg/_dsolve/_superlu*.so")


class Dense:
    def __init__(self, a):
        self.a = a

    def toarray(self):
        return self.a


class Matrix:
    __array_priority__ = 10000

    def __init__(self, arg, shape=None, dtype=np.float64, copy=False, format="csc"):
        self.format = format
        self._diagonal = None
        if isinstance(arg, Matrix):
            source = arg.tocsc() if format == "csc" else arg.tocsr()
            self.shape = source.shape
            self.data = source.data.copy() if copy else source.data
            self.indices = source.indices.copy() if copy else source.indices
            self.indptr = source.indptr.copy() if copy else source.indptr
            self._diagonal = self.data if source._diagonal is not None else None
            if getattr(source, "_canonical", False):
                self._canonical = True
        elif hasattr(arg, "indptr") and hasattr(arg, "indices"):
            source = arg.tocsc() if format == "csc" else arg.tocsr()
            self.shape = source.shape
            self.data = np.asarray(source.data, dtype=np.float64).copy() if copy else np.asarray(source.data, dtype=np.float64)
            self.indices = np.asarray(source.indices, dtype=np.int32)
            self.indptr = np.asarray(source.indptr, dtype=np.int32)
        elif isinstance(arg, tuple) and len(arg) == 2 and all(np.isscalar(x) for x in arg):
            self.shape = tuple(map(int, arg))
            self.data = np.empty(0, dtype=float)
            self.indices = np.empty(0, dtype=np.int32)
            major = self.shape[1] if format == "csc" else self.shape[0]
            self.indptr = np.zeros(major + 1, dtype=np.int32)
        elif isinstance(arg, tuple) and len(arg) == 3:
            self.shape = tuple(map(int, shape))
            self.data = np.asarray(arg[0], dtype=np.float64)
            self.indices = np.asarray(arg[1], dtype=np.int32)
            self.indptr = np.asarray(arg[2], dtype=np.int32)
            end = int(self.indptr[-1])
            self.data, self.indices = self.data[:end], self.indices[:end]
            if copy:
                self.data, self.indices, self.indptr = self.data.copy(), self.indices.copy(), self.indptr.copy()
        elif isinstance(arg, tuple) and len(arg) == 2:
            values, (r, c) = arg
            r, c = np.asarray(r, dtype=np.int32), np.asarray(c, dtype=np.int32)
            values = np.asarray(values, dtype=np.float64)
            if shape is None:
                shape = (int(np.max(r, initial=-1)) + 1, int(np.max(c, initial=-1)) + 1)
            self.shape = tuple(map(int, shape))
            major, minor = (self.shape[1], self.shape[0]) if format == "csc" else self.shape
            self.indptr = np.empty(major + 1, dtype=np.int32)
            self.indices = np.empty(len(values), dtype=np.int32)
            self.data = np.empty(len(values), dtype=float)
            rr, cc = (c, r) if format == "csc" else (r, c)
            st.coo_tocsr(major, minor, len(values), rr, cc, values,
                        self.indptr, self.indices, self.data)
            self.sum_duplicates()
        else:
            a = np.asarray(arg, dtype=float)
            if a.ndim != 2:
                raise ValueError("Expected a two-dimensional matrix")
            r, c = np.nonzero(a)
            other = Matrix((a[r, c], (r, c)), shape=a.shape, format=format)
            self.shape, self.data, self.indices, self.indptr = other.shape, other.data, other.indices, other.indptr

    @classmethod
    def raw(cls, data, indices, indptr, shape, format="csc"):
        return cls((data, indices, indptr), shape=shape, format=format)

    @property
    def nnz(self):
        return int(self.indptr[-1])

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return 2

    @property
    def T(self):
        result = Matrix.raw(self.data, self.indices, self.indptr, self.shape[::-1],
                            "csr" if self.format == "csc" else "csc")
        if hasattr(self, "_dense_cache"):
            result._dense_cache = self._dense_cache.T
        if getattr(self, "_canonical", False):
            result._canonical = True
        return result

    def transpose(self):
        return self.T

    def copy(self):
        return Matrix(self, copy=True, format=self.format)

    def _major_minor(self):
        return (self.shape[1], self.shape[0]) if self.format == "csc" else self.shape

    def sort_indices(self):
        major, minor = self._major_minor()
        if not st.csr_has_sorted_indices(major, self.indptr, self.indices):
            st.csr_sort_indices(major, self.indptr, self.indices, self.data)

    def sum_duplicates(self):
        self.sort_indices()
        major, minor = self._major_minor()
        st.csr_sum_duplicates(major, minor, self.indptr, self.indices, self.data)
        end = int(self.indptr[-1])
        self.data, self.indices = self.data[:end], self.indices[:end]
        self._canonical = True

    def eliminate_zeros(self):
        major, minor = self._major_minor()
        st.csr_eliminate_zeros(major, minor, self.indptr, self.indices, self.data)
        end = int(self.indptr[-1])
        self.data, self.indices = self.data[:end], self.indices[:end]

    def tocsc(self, copy=False):
        if self.format == "csc":
            return self.copy() if copy else self
        return self._convert("csc")

    def tocsr(self, copy=False):
        if self.format == "csr":
            return self.copy() if copy else self
        return self._convert("csr")

    def _convert(self, format):
        major, minor = self._major_minor()
        ip = np.empty(minor + 1, dtype=np.int32)
        ix, da = np.empty(self.nnz, dtype=np.int32), np.empty(self.nnz, dtype=float)
        st.csr_tocsc(major, minor, self.indptr, self.indices, self.data, ip, ix, da)
        return Matrix.raw(da, ix, ip, self.shape, format)

    def toarray(self, order=None):
        major, minor = self._major_minor()
        a = np.zeros((major, minor), dtype=float)
        st.csr_todense(major, minor, self.indptr, self.indices, self.data, a)
        out = a.T if self.format == "csc" else a
        if order == "C":
            return np.array(out, order="C")
        if order == "F":
            return np.array(out, order="F")
        return out

    def diagonal(self):
        major, minor = self._major_minor()
        d = np.zeros(min(major, minor))
        st.csr_diagonal(0, major, minor, self.indptr, self.indices, self.data, d)
        return d

    def max(self, axis=None):
        if axis is None:
            return float(np.max(self.data, initial=0))
        major, minor = self._major_minor()
        major_axis = 0 if self.format == "csc" else 1
        if axis == major_axis:
            out = np.zeros(major)
            counts = np.diff(self.indptr)
            nz = np.flatnonzero(counts)
            if len(nz):
                out[nz] = np.maximum.reduceat(self.data, self.indptr[nz])
            # The caller uses nonnegative coefficients (absolute values).
        else:
            out = np.zeros(minor)
            np.maximum.at(out, self.indices, self.data)
        return Dense(out[None, :] if axis == 0 else out[:, None])

    def __abs__(self):
        return Matrix.raw(np.abs(self.data), self.indices, self.indptr, self.shape, self.format)

    def __neg__(self):
        return Matrix.raw(-self.data, self.indices, self.indptr, self.shape, self.format)

    def __mul__(self, value):
        if not np.isscalar(value):
            return NotImplemented
        return Matrix.raw(self.data * value, self.indices, self.indptr, self.shape, self.format)

    __rmul__ = __mul__

    def __truediv__(self, value):
        return self * (1.0 / value)

    def __imul__(self, value):
        self.data *= value
        if hasattr(self, "_dense_cache"):
            self._dense_cache *= value
        if self._diagonal is not None:
            self._diagonal = self.data
        return self

    def _binop(self, other, subtract=False):
        other = csc_matrix(other)
        a = self.tocsc()
        if a.shape != other.shape:
            raise ValueError("Sparse shape mismatch")
        a.sort_indices()
        other.sort_indices()
        ip = np.empty(a.shape[1] + 1, dtype=np.int32)
        ix = np.empty(a.nnz + other.nnz, dtype=np.int32)
        da = np.empty(len(ix))
        fn = st.csr_minus_csr if subtract else st.csr_plus_csr
        fn(a.shape[1], a.shape[0], a.indptr, a.indices, a.data,
           other.indptr, other.indices, other.data, ip, ix, da)
        return Matrix.raw(da, ix, ip, a.shape)

    def __add__(self, other):
        return self._binop(other)

    def __sub__(self, other):
        return self._binop(other, True)

    def __matmul__(self, other):
        if isinstance(other, Matrix) or hasattr(other, "indptr"):
            a, b = self.tocsc(), csc_matrix(other)
            if a.shape[1] != b.shape[0]:
                raise ValueError("Sparse product shape mismatch")
            if self._diagonal is not None:
                result = b.copy()
                result.data *= self._diagonal[result.indices]
                return result
            if b._diagonal is not None:
                result = a.copy()
                st.csr_scale_rows(a.shape[1], a.shape[0], result.indptr,
                                  result.indices, result.data, b._diagonal)
                return result
            nnz = st.csr_matmat_maxnnz(b.shape[1], a.shape[0],
                                       b.indptr, b.indices, a.indptr, a.indices)
            ip = np.zeros(b.shape[1] + 1, dtype=np.int32)
            ix, da = np.empty(nnz, dtype=np.int32), np.empty(nnz)
            if nnz:
                st.csr_matmat(b.shape[1], a.shape[0],
                              b.indptr, b.indices, b.data,
                              a.indptr, a.indices, a.data, ip, ix, da)
            result = Matrix.raw(da, ix, ip, (a.shape[0], b.shape[1]))
            result.sort_indices()
            return result
        x = np.asarray(other)
        if (x.dtype == np.float64 and os.environ.get("SPARSE_DENSE_CACHE", "1") != "0"
                and self.nnz > .6 * self.shape[0] * self.shape[1]
                and self.shape[0] * self.shape[1] <= 25000000):
            if not hasattr(self, "_dense_cache"):
                self._dense_cache = self.toarray()
            return self._dense_cache @ x
        return self.matvec_sparse(x)

    def matvec_sparse(self, x, scale=None):
        x = np.asarray(x)
        dtype = np.result_type(self.data.dtype, x.dtype)
        data = self.data.astype(dtype, copy=False)
        if scale is not None:
            data = data / scale
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 1:
            out = np.zeros(self.shape[0], dtype=dtype)
            fn = st.csc_matvec if self.format == "csc" else st.csr_matvec
            fn(self.shape[0], self.shape[1], self.indptr, self.indices, data, x, out)
        else:
            out = np.zeros((self.shape[0], x.shape[1]), dtype=dtype)
            fn = st.csc_matvecs if self.format == "csc" else st.csr_matvecs
            fn(self.shape[0], self.shape[1], x.shape[1], self.indptr, self.indices,
               data, np.ascontiguousarray(x).ravel(), out.ravel())
        return out

    def __rmatmul__(self, other):
        a = np.asarray(other)
        return self.T @ a if a.ndim == 1 else (self.T @ a.T).T

    def _major_select(self, idx):
        major, minor = self._major_minor()
        if isinstance(idx, slice):
            if idx == slice(None):
                return self
            start, stop, step = idx.indices(major)
            if step == 1:
                p0, p1 = self.indptr[start], self.indptr[stop]
                ip = self.indptr[start:stop + 1].copy() - p0
                shape = (minor, stop - start) if self.format == "csc" else (stop - start, minor)
                return Matrix.raw(self.data[p0:p1].copy(), self.indices[p0:p1].copy(), ip, shape, self.format)
            idx = np.arange(start, stop, step)
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = np.asarray(idx, dtype=np.int32).ravel()
        counts = self.indptr[idx + 1] - self.indptr[idx]
        ip = np.zeros(len(idx) + 1, dtype=np.int32)
        np.cumsum(counts, out=ip[1:])
        ix, da = np.empty(ip[-1], dtype=np.int32), np.empty(ip[-1])
        st.csr_row_index(len(idx), idx, self.indptr, self.indices, self.data, ix, da)
        shape = (minor, len(idx)) if self.format == "csc" else (len(idx), minor)
        return Matrix.raw(da, ix, ip, shape, self.format)

    def _minor_select(self, idx):
        major, minor = self._major_minor()
        if isinstance(idx, slice):
            if idx == slice(None):
                return self
            idx = np.arange(minor)[idx]
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = np.asarray(idx, dtype=np.int32).ravel()
        offsets = np.zeros(minor, dtype=np.int32)
        ip = np.empty_like(self.indptr)
        st.csr_column_index1(len(idx), idx, major, minor, self.indptr,
                             self.indices, offsets, ip)
        order = np.argsort(idx).astype(np.int32)
        ix, da = np.empty(ip[-1], dtype=np.int32), np.empty(ip[-1])
        st.csr_column_index2(order, offsets, self.nnz, self.indices, self.data, ix, da)
        shape = (len(idx), major) if self.format == "csc" else (major, len(idx))
        result = Matrix.raw(da, ix, ip, shape, self.format)
        result.sort_indices()
        return result

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        r, c = key
        if np.isscalar(r):
            r = [int(r)]
        if np.isscalar(c):
            c = [int(c)]
        major, minor = (c, r) if self.format == "csc" else (r, c)
        result = self._major_select(major)
        return result._minor_select(minor)


def csc_matrix(arg, shape=None, dtype=float, copy=False):
    return Matrix(arg, shape=shape, dtype=dtype, copy=copy, format="csc")


def csr_matrix(arg, shape=None, dtype=float, copy=False):
    return Matrix(arg, shape=shape, dtype=dtype, copy=copy, format="csr")


def diags(values, offsets=0, shape=None, format="csc"):
    if offsets != 0:
        raise ValueError("Only diagonal matrices are needed")
    v = np.asarray(values, dtype=float).ravel()
    n = len(v)
    p = np.arange(n + 1, dtype=np.int32)
    a = Matrix.raw(v.copy(), p[:-1].copy(), p, (n, n), format)
    a._diagonal = a.data
    a._canonical = True
    return a


def tril(a, format="csc"):
    a = csc_matrix(a)
    col = np.repeat(np.arange(a.shape[1], dtype=np.int32), np.diff(a.indptr))
    mask = a.indices >= col
    return Matrix((a.data[mask], (a.indices[mask], col[mask])), shape=a.shape, format=format)


def bmat(blocks, format="csc"):
    nr, nc = len(blocks), len(blocks[0])
    rh, cw = np.zeros(nr, dtype=np.int64), np.zeros(nc, dtype=np.int64)
    for i in range(nr):
        for j in range(nc):
            a = blocks[i][j]
            if a is not None:
                rh[i], cw[j] = a.shape
    ro, co = np.r_[0, np.cumsum(rh)], np.r_[0, np.cumsum(cw)]
    rr, cc, vv = [], [], []
    for i in range(nr):
        for j in range(nc):
            a = blocks[i][j]
            if a is None or not a.nnz:
                continue
            major, minor = a._major_minor()
            ex = np.repeat(np.arange(major, dtype=np.int32), np.diff(a.indptr))
            r, c = (a.indices, ex) if a.format == "csc" else (ex, a.indices)
            rr.append(r + ro[i])
            cc.append(c + co[j])
            vv.append(a.data)
    shape = (int(ro[-1]), int(co[-1]))
    if not vv:
        return Matrix(shape, format=format)
    return Matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                  shape=shape, format=format)


def vstack(blocks, format="csc"):
    return bmat([[a] for a in blocks], format=format)


def hstack(blocks, format="csc"):
    return bmat([blocks], format=format)


def gram_dense(b):
    b = csc_matrix(b)
    a = b.T.tocsc()
    n = b.shape[1]
    nnz = st.csr_matmat_maxnnz(n, n, b.indptr, b.indices, a.indptr, a.indices)
    ip = np.zeros(n + 1, dtype=np.int32)
    ix, da = np.empty(nnz, dtype=np.int32), np.empty(nnz)
    if nnz:
        st.csr_matmat(n, n, b.indptr, b.indices, b.data,
                      a.indptr, a.indices, a.data, ip, ix, da)
    return Matrix.raw(da, ix, ip, (n, n)).toarray()


def splu(a, permc_spec="COLAMD", **kwargs):
    a = csc_matrix(a)
    if not getattr(a, "_canonical", False):
        a.sum_duplicates()
    return su.gstrf(a.shape[0], a.nnz, a.data, a.indices, a.indptr,
                   csc_construct_func=None, ilu=False,
                   options={"ColPerm": permc_spec, "DiagPivotThresh": 1.0})


def issparse(a):
    return isinstance(a, Matrix)
