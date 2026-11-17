import ctypes
import os

import numpy as np
from scipy import linalg, sparse


class SparseFactor:
    def __init__(self, matrix, permutation, primal):
        path = os.path.join(os.path.dirname(__file__), "sparse_ldl.so")
        self.lib = ctypes.CDLL(path)
        self.lib.ldl_create.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ldl_create.restype = ctypes.c_void_p
        self.lib.ldl_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ldl_numeric.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ldl_numeric.restype = ctypes.c_int
        self.lib.ldl_solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ldl_nonzeros.argtypes = [ctypes.c_void_p]
        self.lib.ldl_nonzeros.restype = ctypes.c_int64
        self.permutation = np.argsort(permutation)
        p = self.permutation
        encoded = sparse.csc_matrix(
            (np.arange(1, matrix.nnz + 1, dtype=float), matrix.indices, matrix.indptr),
            shape=matrix.shape)
        upper = sparse.triu(encoded[p, :][:, p], format="csc")
        self.mapping = upper.data.astype(np.int64) - 1
        self.ap = np.asarray(upper.indptr, dtype=np.int32)
        self.ai = np.asarray(upper.indices, dtype=np.int32)
        self.sign = (np.where(p < primal, 1.0, -1.0) if np.isscalar(primal)
                     else np.asarray(primal, dtype=float)[p])
        self.context = self.lib.ldl_create(matrix.shape[0], self.ap.ctypes.data,
                                           self.ai.ctypes.data)
        if not self.context:
            raise MemoryError("Sparse LDL symbolic fill exceeds memory cap")
        self.nnz = self.lib.ldl_nonzeros(self.context)
        if not self.update(matrix):
            raise RuntimeError("Sparse LDL numerical pivot failure")

    def __del__(self):
        if getattr(self, "context", None):
            self.lib.ldl_destroy(self.context)
            self.context = None

    def update(self, matrix):
        self.values = np.ascontiguousarray(matrix.data[self.mapping])
        return bool(self.lib.ldl_numeric(self.context, self.values.ctypes.data,
                                        self.sign.ctypes.data))

    def solve(self, rhs):
        permuted = np.ascontiguousarray(rhs[self.permutation])
        result = np.empty_like(permuted)
        self.lib.ldl_solve(self.context, permuted.ctypes.data, result.ctypes.data)
        original = np.empty_like(result)
        original[self.permutation] = result
        return original


class DenseFactor:
    def __init__(self, problem, cones):
        self.n = problem.n
        self.zero, self.linear = cones.zero, cones.linear
        self.P = problem.P.toarray()
        self.A = problem.A[self.linear, :].tocsc()
        self.E = problem.A[self.zero, :].toarray()

    def update(self, cones):
        self.h = 1.0 / (cones.linear_inverse + 1e-9)
        normal = (self.A.T @ self.A.multiply(self.h[:, None])).toarray()
        normal += self.P
        normal[np.diag_indices(self.n)] += 1e-9
        self.factor = linalg.cho_factor(normal, lower=True, check_finite=False)
        if self.zero.size:
            self.W = linalg.cho_solve(self.factor, self.E.T, check_finite=False)
            schur = self.E @ self.W
            schur[np.diag_indices(self.zero.size)] += 1e-9
            self.schur = linalg.cho_factor(schur, lower=True, check_finite=False)

    def solve(self, rhs):
        rx, rz = rhs[:self.n], rhs[self.n:]
        body = rx + self.A.T @ (self.h * rz[self.linear])
        x = linalg.cho_solve(self.factor, body, check_finite=False)
        z = np.zeros(rz.size)
        if self.zero.size:
            ze = linalg.cho_solve(self.schur, self.E @ x - rz[self.zero],
                                  check_finite=False)
            x -= self.W @ ze
            z[self.zero] = ze
        z[self.linear] = self.h * (self.A @ x - rz[self.linear])
        return np.r_[x, z]
