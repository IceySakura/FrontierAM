"""Optional reusable symbolic/numeric sparse LDL factorization."""
import ctypes as ct
import os
import numpy as np

_lib = ct.CDLL(os.path.join(os.path.dirname(__file__), "ldl.so"))
_ip = ct.POINTER(ct.c_int)
_dp = ct.POINTER(ct.c_double)
_lib.conic_ldl_create.argtypes = [ct.c_int, _ip, _ip, _ip]
_lib.conic_ldl_create.restype = ct.c_void_p
_lib.conic_ldl_free.argtypes = [ct.c_void_p]
_lib.conic_ldl_free.restype = None
_lib.conic_ldl_factor.argtypes = [ct.c_void_p, _ip, _ip, _dp, ct.c_int, ct.c_double, ct.c_double]
_lib.conic_ldl_factor.restype = ct.c_int
_lib.conic_ldl_solve.argtypes = [ct.c_void_p, _dp, _dp, ct.c_int]
_lib.conic_ldl_solve.restype = ct.c_int
_lib.conic_ldl_nnz.argtypes = [ct.c_void_p]
_lib.conic_ldl_nnz.restype = ct.c_int64


def ip(a):
    return a.ctypes.data_as(_ip)


def dp(a):
    return a.ctypes.data_as(_dp)


class LDL:
    def __init__(self, a, order):
        self.order = np.ascontiguousarray(order, dtype=np.int32)
        self.indptr = a.indptr.copy()
        self.indices = a.indices.copy()
        self.handle = _lib.conic_ldl_create(a.shape[0], ip(self.order), ip(a.indptr), ip(a.indices))
        if not self.handle:
            raise RuntimeError("Sparse LDL symbolic memory limit")
        self._free = _lib.conic_ldl_free
        self.nnz = _lib.conic_ldl_nnz(self.handle)

    def __del__(self):
        if getattr(self, "handle", None):
            self._free(self.handle)
            self.handle = None

    def compatible(self, a):
        return (np.array_equal(self.indptr, a.indptr) and
                np.array_equal(self.indices, a.indices))

    def factor(self, a, positive):
        code = _lib.conic_ldl_factor(self.handle, ip(a.indptr), ip(a.indices),
                                     dp(a.data), positive, 1e-13, 1e-8)
        if code < 0:
            raise RuntimeError("Sparse LDL numeric failure")
        self.clips = code
        return self

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=float)
        result = np.empty_like(b)
        code = _lib.conic_ldl_solve(self.handle, dp(b), dp(result), 1 if b.ndim == 1 else b.shape[1])
        if code:
            raise RuntimeError("Sparse LDL nonfinite solution")
        return result
