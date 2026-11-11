"""Cached CSC assembly for the scaled sparse KKT matrix."""
import ctypes as ct
import numpy as np
import fastsparse as sp
from native_ldl import _lib, _ip, _dp, ip, dp

_lib.conic_kkt_create.argtypes = [ct.c_int, ct.c_int, _ip, _ip, _dp, _ip, _ip, _dp]
_lib.conic_kkt_create.restype = ct.c_void_p
_lib.conic_kkt_free.argtypes = [ct.c_void_p]
_lib.conic_kkt_free.restype = None
_lib.conic_kkt_nnz.argtypes = [ct.c_void_p]
_lib.conic_kkt_nnz.restype = ct.c_int64
_lib.conic_kkt_export.argtypes = [ct.c_void_p, _ip, _ip, _dp]
_lib.conic_kkt_export.restype = None
_lib.conic_kkt_update.argtypes = [ct.c_void_p, _dp, _dp]
_lib.conic_kkt_update.restype = None
_lib.conic_kkt_shift.argtypes = [ct.c_void_p, _dp, ct.c_double]
_lib.conic_kkt_shift.restype = None


class Assembly:
    def __init__(self, P, B, diagonal):
        n, m = P.shape[0], B.shape[0]
        self.indptr, self.indices = B.indptr.copy(), B.indices.copy()
        self.handle = _lib.conic_kkt_create(n, m, ip(P.indptr), ip(P.indices), dp(P.data),
                                           ip(B.indptr), ip(B.indices), dp(diagonal))
        if not self.handle:
            raise RuntimeError("KKT assembly memory limit")
        self._free = _lib.conic_kkt_free
        nnz = _lib.conic_kkt_nnz(self.handle)
        ptr = np.empty(n + m + 1, dtype=np.int32)
        indices = np.empty(nnz, dtype=np.int32)
        values = np.empty(nnz, dtype=float)
        _lib.conic_kkt_export(self.handle, ip(ptr), ip(indices), dp(values))
        self.matrix = sp.Matrix.raw(values, indices, ptr, (n + m, n + m))
        self.matrix._canonical = True
        self.update(B)

    def __del__(self):
        if getattr(self, "handle", None):
            self._free(self.handle)
            self.handle = None

    def compatible(self, B):
        return (np.array_equal(B.indptr, self.indptr) and
                np.array_equal(B.indices, self.indices))

    def update(self, B):
        _lib.conic_kkt_update(self.handle, dp(B.data), dp(self.matrix.data))
        if hasattr(self.matrix, "_dense_cache"):
            del self.matrix._dense_cache
        return self.matrix

    def regularized(self, delta):
        K = sp.Matrix.raw(self.matrix.data.copy(), self.matrix.indices,
                          self.matrix.indptr, self.matrix.shape)
        K._canonical = True
        _lib.conic_kkt_shift(self.handle, dp(K.data), delta)
        return K
