import ctypes
import os
import numpy as np

_lib = None


def library():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libconic_ldl.so"))
        ip = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        dp = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        _lib.conic_ldl_symbolic.argtypes = [ctypes.c_int, ip, ip, ip, ip, ip, ip]
        _lib.conic_ldl_symbolic.restype = ctypes.c_int64
        _lib.conic_ldl_numeric.argtypes = [
            ctypes.c_int, ip, ip, dp, ip, ip, ip, dp, dp, dp, ip, ip, ip, ip,
            ctypes.c_double, ctypes.c_double]
        _lib.conic_ldl_numeric.restype = ctypes.c_int
        _lib.conic_ldl_solve.argtypes = [ctypes.c_int, ip, ip, dp, dp, dp]
        _lib.conic_ldl_solve.restype = None
        _lib.conic_csc_sorted.argtypes = [ctypes.c_int, ip, ip, dp]
        _lib.conic_csc_sorted.restype = ctypes.c_int
        _lib.conic_csc_matvec.argtypes = [
            ctypes.c_int, ctypes.c_int, ip, ip, dp, dp, dp, ctypes.c_int]
        _lib.conic_csc_matvec.restype = None
        _lib.conic_csc_transpose.argtypes = [
            ctypes.c_int, ctypes.c_int, ip, ip, dp, ip, ip, dp, ip]
        _lib.conic_csc_transpose.restype = None
    return _lib


class LDL:
    def __init__(self, matrix, sign):
        self.lib = library()
        self.n = matrix.shape[0]
        self.ap = np.asarray(matrix.indptr, dtype=np.int32)
        self.ai = np.asarray(matrix.indices, dtype=np.int32)
        self.lp = np.empty(self.n+1, dtype=np.int32)
        self.parent = np.empty(self.n, dtype=np.int32)
        self.count = np.empty(self.n, dtype=np.int32)
        self.flag = np.empty(self.n, dtype=np.int32)
        self.pattern = np.empty(self.n, dtype=np.int32)
        self.sign = np.asarray(sign, dtype=np.int32)
        size = self.lib.conic_ldl_symbolic(
            self.n, self.ap, self.ai, self.lp, self.parent, self.count, self.flag)
        if size < 0:
            raise MemoryError("Sparse LDL fill exceeds the workspace budget")
        degrees = np.diff(self.lp).astype(np.float64)
        if degrees.dot(degrees) > 25000000:
            raise MemoryError("Dense fill is handled by the blocked LU backend")
        self.li = np.empty(size, dtype=np.int32)
        self.lx = np.empty(size, dtype=np.float64)
        self.d = np.empty(self.n)
        self.y = np.empty(self.n)

    def matches(self, matrix):
        return (matrix.shape[0] == self.n and
                np.array_equal(matrix.indptr, self.ap) and
                np.array_equal(matrix.indices, self.ai))

    def factor(self, matrix):
        count = self.lib.conic_ldl_numeric(
            self.n, self.ap, self.ai, matrix.data, self.lp, self.parent,
            self.li, self.lx, self.d, self.y, self.count, self.pattern, self.flag,
            self.sign, 1e-13, 1e-9)
        if count < 0:
            raise ArithmeticError("Sparse LDL numerical breakdown")
        return count

    def solve(self, rhs, overwrite=False):
        x = np.asarray(rhs, dtype=np.float64)
        if not overwrite:
            x = x.copy()
        self.lib.conic_ldl_solve(self.n, self.lp, self.li, self.lx, self.d, x)
        return x
