"""Optional sparse LU callback for the native Newton solver."""

import ctypes

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def enable(lib):
    I = ctypes.c_int
    L = ctypes.c_int64
    PI = ctypes.POINTER(I)
    PD = ctypes.POINTER(ctypes.c_double)
    MAKE = ctypes.CFUNCTYPE(L,L,I,PI,PI,PD)
    SOLVE = ctypes.CFUNCTYPE(I,L,I,PD)
    DROP = ctypes.CFUNCTYPE(None,L)
    objects = {}
    counter = 0

    @MAKE
    def make(old, n, p, i, v):
        nonlocal counter
        try:
            offsets = np.ctypeslib.as_array(p, (n+1,))
            count = int(offsets[-1])
            values = np.ctypeslib.as_array(v, (count,))
            if old in objects:
                A, mapping, lu = objects.pop(old)
                del lu
            else:
                upper = np.ctypeslib.as_array(i, (count,))
                column = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
                off = upper != column
                X = sparse.coo_matrix((np.r_[np.arange(1,count+1),np.flatnonzero(off)+1],
                    (np.r_[upper,column[off]],np.r_[column,upper[off]])),shape=(n,n)).tocsc()
                mapping = X.data.astype(np.int32)-1
                A = sparse.csc_matrix((np.empty(X.nnz), X.indices, X.indptr),shape=(n,n))
            np.take(values, mapping, out=A.data)
            F = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=.15,
                     options={"Equil": False})
            counter += 1
            objects[counter] = (A, mapping, F)
            return counter
        except Exception:
            return 0

    @SOLVE
    def solve(key, n, ptr):
        try:
            x = np.ctypeslib.as_array(ptr, (n,))
            x[:] = objects[key][2].solve(x)
            return 1
        except Exception:
            return 0

    @DROP
    def drop(key):
        objects.pop(key, None)

    lib.conic_set_lu.argtypes = [MAKE,SOLVE,DROP]
    lib.conic_set_lu.restype = None
    lib.conic_set_lu(make,solve,drop)
    return (make,solve,drop)
