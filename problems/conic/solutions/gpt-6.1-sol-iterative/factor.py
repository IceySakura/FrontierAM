"""Reusable symbolic sparse factorization with a portable SuperLU fallback."""
import ctypes
import os

import numpy as np
from scipy.sparse import linalg as sla


class LDL:
    lib = None
    checked = False

    def __init__(self, K, primal, perm):
        if not LDL.checked:
            LDL.checked = True
            path = os.path.join(os.path.dirname(__file__), "libldl.so")
            if os.path.exists(path):
                lib = ctypes.CDLL(path)
                ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
                lib.ldl_create.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, ip]
                lib.ldl_create.restype = ctypes.c_void_p
                lib.ldl_numeric.argtypes = [ctypes.c_void_p, dp]
                lib.ldl_numeric.restype = ctypes.c_int
                lib.ldl_solve.argtypes = [ctypes.c_void_p, dp, dp]
                lib.ldl_free.argtypes = [ctypes.c_void_p]
                LDL.lib = lib
        self.ptr = None
        if LDL.lib is None:
            raise RuntimeError("Native LDL unavailable")
        self.K = K
        self.order = np.argsort(perm).astype(np.int32)
        self.ap = K.indptr.astype(np.int32, copy=False)
        self.ai = K.indices.astype(np.int32, copy=False)
        ip = ctypes.POINTER(ctypes.c_int)
        self.ptr = self.lib.ldl_create(K.shape[0], primal,
                                       self.ap.ctypes.data_as(ip),
                                       self.ai.ctypes.data_as(ip),
                                       self.order.ctypes.data_as(ip))
        if not self.ptr:
            raise MemoryError("Sparse LDL symbolic fill")

    def numeric(self, K):
        dp = ctypes.POINTER(ctypes.c_double)
        if self.lib.ldl_numeric(self.ptr, K.data.ctypes.data_as(dp)) != 0:
            raise ArithmeticError("Sparse LDL numerical factorization")

    def matches(self, K):
        return (K.shape == self.K.shape and np.array_equal(K.indptr, self.ap)
                and np.array_equal(K.indices, self.ai))

    def solve(self, rhs):
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        out = np.empty_like(rhs)
        dp = ctypes.POINTER(ctypes.c_double)
        self.lib.ldl_solve(self.ptr, rhs.ctypes.data_as(dp), out.ctypes.data_as(dp))
        return out

    def __del__(self):
        if self.ptr:
            self.lib.ldl_free(self.ptr)


def factorize(K, primal, previous=None):
    if previous is not None and previous.matches(K):
        previous.numeric(K)
        return previous, previous
    lu = sla.splu(K, permc_spec=os.environ.get("SOLVER_PERM", "MMD_AT_PLUS_A"),
                 diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    if os.environ.get("SOLVER_FACTOR", "superlu") == "ldl":
        try:
            native = LDL(K, primal, lu.perm_c)
            native.numeric(K)
            test = K @ np.ones(K.shape[0])
            v = native.solve(test)
            error = np.max(np.abs(K @ v - test)) / max(1.0, np.max(np.abs(test)))
            if np.isfinite(v).all() and error < 1e-8:
                return native, native
        except (ArithmeticError, RuntimeError, MemoryError):
            pass
    return lu, None
