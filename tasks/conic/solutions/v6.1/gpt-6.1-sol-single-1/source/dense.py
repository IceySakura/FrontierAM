"""Reusable LAPACK factorizations from the pinned NumPy OpenBLAS runtime."""

import ctypes
import glob
import os
import numpy as np


_lib = None


def library():
    global _lib
    if _lib is None:
        paths = glob.glob(os.path.join(os.path.dirname(np.__file__), "..",
                                       "numpy.libs", "*openblas*.so"))
        _lib = ctypes.CDLL(paths[0])
    return _lib


def routine(name):
    return getattr(library(), "scipy_" + name + "_64_")


def addr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class LU:
    def __init__(self, A):
        self.a = np.array(A, dtype=np.float64, order="F")
        self.n = ctypes.c_int64(len(A))
        self.piv = np.empty(len(A), dtype=np.int64)
        info = ctypes.c_int64()
        routine("dgetrf")(ctypes.byref(self.n), ctypes.byref(self.n),
                           addr(self.a), ctypes.byref(self.n), addr(self.piv),
                           ctypes.byref(info))
        if info.value:
            raise np.linalg.LinAlgError("singular dense KKT")

    def solve(self, rhs):
        b = np.array(rhs, dtype=np.float64, order="F", copy=True)
        nrhs = ctypes.c_int64(1 if b.ndim == 1 else b.shape[1])
        info = ctypes.c_int64()
        routine("dgetrs")(ctypes.c_char_p(b"N"), ctypes.byref(self.n),
                           ctypes.byref(nrhs), addr(self.a), ctypes.byref(self.n),
                           addr(self.piv), addr(b), ctypes.byref(self.n),
                           ctypes.byref(info), ctypes.c_size_t(1))
        if info.value:
            raise np.linalg.LinAlgError("dense KKT solve failure")
        return b


class Cholesky:
    def __init__(self, A):
        self.a = np.array(A, dtype=np.float64, order="F")
        self.n = ctypes.c_int64(len(A))
        info = ctypes.c_int64()
        routine("dpotrf")(ctypes.c_char_p(b"L"), ctypes.byref(self.n),
                           addr(self.a), ctypes.byref(self.n), ctypes.byref(info),
                           ctypes.c_size_t(1))
        if info.value:
            raise np.linalg.LinAlgError("nonpositive dense Hessian")

    def solve(self, rhs):
        b = np.array(rhs, dtype=np.float64, order="F", copy=True)
        nrhs = ctypes.c_int64(1 if b.ndim == 1 else b.shape[1])
        info = ctypes.c_int64()
        routine("dpotrs")(ctypes.c_char_p(b"L"), ctypes.byref(self.n),
                           ctypes.byref(nrhs), addr(self.a), ctypes.byref(self.n),
                           addr(b), ctypes.byref(self.n), ctypes.byref(info),
                           ctypes.c_size_t(1))
        if info.value:
            raise np.linalg.LinAlgError("dense Hessian solve failure")
        return b
