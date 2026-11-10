"""Reusable symbolic sparse LDL factors, with a low-cost symmetric ordering."""
import ctypes
import os
import sys

import numpy as np
from scipy.sparse import linalg as sla

_lib=None


def library():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(os.path.join(os.path.dirname(__file__),"native.so"))
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        _lib.ldl_create.argtypes=[ctypes.c_int,ip,ip,ip,ctypes.c_int]
        _lib.ldl_create.restype=ctypes.c_void_p
        _lib.ldl_factor.argtypes=[ctypes.c_void_p,dp,ctypes.c_int]
        _lib.ldl_factor.restype=ctypes.c_int
        _lib.ldl_solve.argtypes=[ctypes.c_void_p,dp,dp]
        _lib.ldl_residual.argtypes=[ctypes.c_void_p,dp,dp,dp,dp]
        _lib.ldl_free.argtypes=[ctypes.c_void_p]
        _lib.ldl_nnz.argtypes=[ctypes.c_void_p]
        _lib.ldl_nnz.restype=ctypes.c_int64
    return _lib


class NativeLDL:
    def __init__(self,K,permutation=None,quad=False):
        self.lib=library()
        self.handle=None
        self.suspect=False
        self.quad=quad
        if permutation is None:
            order=K.copy()
            order.data[:]=1e-40
            order.setdiag(1.0)
            order.sort_indices()
            ordering=sla.spilu(order,drop_tol=1e-12,fill_factor=1.0,
                drop_rule="basic,area",permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.,options={"SymmetricMode":True})
            permutation=np.argsort(ordering.perm_c)
        self.permutation=np.ascontiguousarray(permutation,dtype=np.int32)
        self.ap=K.indptr
        self.ai=K.indices
        self.handle=self.lib.ldl_create(K.shape[0],self.ap,self.ai,self.permutation,int(quad))
        if not self.handle:
            raise RuntimeError("sparse LDL fill budget")
        if os.environ.get("SOLVER_DEBUG"):
            print("LDL symbolic",K.shape,K.nnz,self.lib.ldl_nnz(self.handle),
                  file=sys.stderr,flush=True)

    def matches(self,K):
        return (len(self.ap)==len(K.indptr) and len(self.ai)==len(K.indices)
                and np.array_equal(self.ap,K.indptr) and np.array_equal(self.ai,K.indices))

    def factor(self,K,enforce_sign=True):
        return bool(self.lib.ldl_factor(self.handle,K.data,int(enforce_sign)))

    def solve(self,rhs):
        out=np.empty_like(rhs)
        self.lib.ldl_solve(self.handle,np.ascontiguousarray(rhs),out)
        return out

    def residual(self,rhs,x,reg):
        out=np.empty_like(rhs)
        self.lib.ldl_residual(self.handle,np.ascontiguousarray(rhs),
                              np.ascontiguousarray(x),reg,out)
        return out

    def __del__(self):
        if getattr(self,"handle",None):
            self.lib.ldl_free(self.handle)
            self.handle=None
