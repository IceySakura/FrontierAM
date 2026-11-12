"""Reusable symmetric sparse ordering and optional native LDL factors."""

import ctypes
import os

import numpy as np
from scipy import sparse
from numeric import splu


_library = None


def library():
    global _library
    if _library is None:
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "native_ldl.so"))
        ptr_i = ctypes.POINTER(ctypes.c_int)
        ptr_d = ctypes.POINTER(ctypes.c_double)
        lib.sparse_ldl_create.argtypes = [ctypes.c_int, ptr_i, ptr_i]
        lib.sparse_ldl_create.restype = ctypes.c_void_p
        lib.sparse_ldl_destroy.argtypes = [ctypes.c_void_p]
        lib.sparse_ldl_nnz.argtypes = [ctypes.c_void_p]
        lib.sparse_ldl_nnz.restype = ctypes.c_size_t
        lib.sparse_ldl_factor.argtypes = [ctypes.c_void_p, ptr_d, ptr_d]
        lib.sparse_ldl_factor.restype = ctypes.c_int
        lib.sparse_ldl_solve.argtypes = [ctypes.c_void_p, ptr_d, ctypes.c_int]
        lib.csc_positions.argtypes = [ctypes.c_int, ptr_i, ptr_i, ptr_i, ptr_i, ptr_i]
        lib.csc_positions.restype = ctypes.c_int
        _library = lib
    return _library


def ptrd(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class NativeLDL:
    def __init__(self, upper):
        self.lib = library()
        self.ap = np.ascontiguousarray(upper.indptr, dtype=np.int32)
        self.ai = np.ascontiguousarray(upper.indices, dtype=np.int32)
        self.handle = self.lib.sparse_ldl_create(upper.shape[0],
            self.ap.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if not self.handle:
            raise RuntimeError("LDL symbolic factorization exceeded its fill budget")
        self.nnz = self.lib.sparse_ldl_nnz(self.handle)
        self.n = upper.shape[0]

    def matches(self, upper):
        return np.array_equal(self.ap, upper.indptr) and np.array_equal(self.ai, upper.indices)

    def factor(self, upper, signs):
        values = np.ascontiguousarray(upper.data, dtype=np.float64)
        signs = np.ascontiguousarray(signs, dtype=np.float64)
        if self.lib.sparse_ldl_factor(self.handle, ptrd(values), ptrd(signs)):
            raise RuntimeError("LDL pivot sign or magnitude is unreliable")

    def solve(self, rhs):
        result = np.array(rhs, dtype=np.float64, order="F", copy=True)
        self.lib.sparse_ldl_solve(self.handle, ptrd(result), 1 if result.ndim == 1 else result.shape[1])
        return result

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.sparse_ldl_destroy(self.handle)
            self.handle = None


class Permuted:
    def __init__(self, factor, perm):
        self.factor, self.perm = factor, perm

    def solve(self, rhs):
        x = self.factor.solve(rhs[self.perm])
        out = np.empty_like(x)
        out[self.perm] = x
        return out


class SparseFactors:
    def __init__(self, order=None, pivot=0., native=False):
        self.order = order or os.getenv("SPARSE_ORDER", "MMD_AT_PLUS_A")
        self.pivot = pivot
        self.perm, self.native_factor = None, None
        self.source, self.permuted = None, None
        self.upper, self.upper_index = None, None
        self.native = native and pivot == 0. and os.getenv("KKT_NATIVE", "auto") != "0"
        self.force_native = os.getenv("KKT_NATIVE") == "1"
        self.reuse = self.order != "NATURAL" and os.getenv("KKT_REUSE", "1") != "0"

    def __call__(self, K):
        if not self.reuse or self.perm is None:
            factor = splu(K, permc_spec=self.order, diag_pivot_thresh=self.pivot,
                          options={"SymmetricMode": self.order != "COLAMD"})
            if self.reuse:
                self.perm = np.argsort(factor.perm_c)
                if self.native and not self.force_native:
                    self.native = K.shape[0] >= 2000 and factor.nnz <= 20 * K.shape[0]
            return factor
        p = self.perm
        same = self.source is K
        if self.source is not None and not same:
            same = (K.shape == self.source.shape and
                    np.array_equal(K.indptr, self.source.indptr) and
                    np.array_equal(K.indices, self.source.indices))
        if not same or self.permuted is None:
            tagged = sparse.csc_matrix((np.arange(K.nnz, dtype=np.int64), K.indices, K.indptr),
                                      shape=K.shape, copy=False)
            mapping = tagged[p, :][:, p].tocsc()
            mapping.sort_indices()
            self.gather = mapping.data
            self.permuted = sparse.csc_matrix((np.empty(mapping.nnz), mapping.indices, mapping.indptr),
                                               shape=mapping.shape, copy=False)
            self.upper, self.upper_index, self.native_factor = None, None, None
        self.source = K
        Kp = self.permuted
        Kp.data[:] = K.data[self.gather]
        if self.native and Kp.nnz < .08 * Kp.shape[0]**2:
            try:
                if self.upper is None:
                    col = np.repeat(np.arange(Kp.shape[1]), np.diff(Kp.indptr))
                    keep = Kp.indices <= col
                    self.upper_index = np.flatnonzero(keep)
                    ptr = np.r_[0, np.cumsum(np.bincount(col[keep], minlength=Kp.shape[1]))]
                    self.upper = sparse.csc_matrix((np.empty(len(self.upper_index)), Kp.indices[keep], ptr),
                                                   shape=Kp.shape)
                upper = self.upper
                upper.data[:] = Kp.data[self.upper_index]
                if self.native_factor is None:
                    self.native_factor = NativeLDL(upper)
                if not self.force_native and (Kp.shape[0] < 2000 or self.native_factor.nnz > 10 * Kp.shape[0]):
                    self.native = False
                    self.native_factor = None
                elif self.native_factor.nnz <= 4_000_000:
                    self.native_factor.factor(upper, np.sign(Kp.diagonal()))
                    return Permuted(self.native_factor, p)
            except (RuntimeError, OSError):
                self.native_factor = None
        factor = splu(Kp, permc_spec="NATURAL", diag_pivot_thresh=self.pivot,
                      options={"SymmetricMode": self.order != "COLAMD"})
        return Permuted(factor, p)


def positions(K, rows, cols):
    lib = library()
    ap = np.ascontiguousarray(K.indptr, dtype=np.int32)
    ai = np.ascontiguousarray(K.indices, dtype=np.int32)
    rows, cols = (np.ascontiguousarray(v, dtype=np.int32) for v in (rows, cols))
    out = np.empty(len(rows), dtype=np.int32)
    pi = ctypes.POINTER(ctypes.c_int)
    if lib.csc_positions(len(rows), ap.ctypes.data_as(pi), ai.ctypes.data_as(pi),
                         rows.ctypes.data_as(pi), cols.ctypes.data_as(pi), out.ctypes.data_as(pi)):
        raise RuntimeError("KKT numeric update changed its sparsity")
    return out


class AugmentedMatrix:
    def __init__(self, P, E, C, cs, reg):
        n, ne, nc, na = P.shape[0], E.shape[0], cs.size, cs.naux
        Z = sparse.csc_matrix((ne, nc))
        xn, en = sparse.csc_matrix((n, na)), sparse.csc_matrix((ne, na))
        aux = sparse.diags(cs.aux_sign * (1 + reg)) if na else sparse.csc_matrix((0, 0))
        self.K = sparse.bmat([[P + reg * sparse.eye(n), E.T, C.T, xn],
                              [E, -reg * sparse.eye(ne), Z, en],
                              [C, Z.T, -cs.Di - reg * sparse.eye(nc), cs.U],
                              [xn.T, en.T, cs.U.T, aux]], format="csc")
        self.K.sort_indices()
        diag = n + ne + np.arange(nc)
        self.diag = positions(self.K, diag, diag)
        self.reg = reg
        if na:
            rows = cs.U.indices
            cols = np.repeat(np.arange(na), np.diff(cs.U.indptr))
            r, c = n + ne + rows, n + ne + nc + cols
            self.lower = positions(self.K, r, c)
            self.upper = positions(self.K, c, r)
        self.update(cs)

    def update(self, cs):
        self.K.data[self.diag] = -cs.Di.diagonal() - self.reg
        if cs.naux:
            self.K.data[self.lower] = cs.U.data
            self.K.data[self.upper] = cs.U.data
        return self.K
