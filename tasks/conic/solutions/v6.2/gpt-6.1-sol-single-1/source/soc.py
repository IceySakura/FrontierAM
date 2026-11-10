"""Native linear-time Lorentz-cone geometry."""
import ctypes
import os

import numpy as np

_lib=None


def library():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(os.path.join(os.path.dirname(__file__),"native.so"))
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        _lib.soc_create.argtypes=[ctypes.c_int,ctypes.c_int,ip]
        _lib.soc_create.restype=ctypes.c_void_p
        _lib.soc_free.argtypes=[ctypes.c_void_p]
        _lib.soc_update.argtypes=[ctypes.c_void_p,dp,dp,dp,dp,dp,dp,ctypes.c_int,ctypes.c_int]
        _lib.soc_correct.argtypes=[ctypes.c_void_p,dp,dp,dp,ctypes.c_double,dp]
        _lib.soc_step.argtypes=[ctypes.c_void_p,dp,dp]
        _lib.soc_step.restype=ctypes.c_double
        _lib.soc_neighborhood.argtypes=[ctypes.c_void_p,dp,dp,ctypes.c_double]
        _lib.soc_neighborhood.restype=ctypes.c_int
    return _lib


class Oracle:
    def __init__(self,ids):
        self.lib=library()
        self.n,self.d=ids.shape
        self.handle=self.lib.soc_create(self.n,self.d,
            np.ascontiguousarray(ids[:,0],dtype=np.int32))
        if not self.handle:
            raise MemoryError("SOC workspace")
        width=self.n*self.d*self.d if self.d<=64 else 0
        self.h=np.empty(width)
        self.hi=np.empty(width)
        self.u=np.empty(self.n*self.d)
        self.diagonal=np.empty(self.n*self.d)
        self.out=np.empty(self.n*self.d)

    def update(self,s,z,dense,inverse):
        self.lib.soc_update(self.handle,s,z,self.h,self.hi,self.u,self.diagonal,
                            int(dense),int(inverse))

    def correct(self,s,ds,dz,sigma_mu):
        self.lib.soc_correct(self.handle,s,ds,dz,sigma_mu,self.out)
        return self.out.reshape(self.n,self.d)

    def step(self,x,dx):
        return self.lib.soc_step(self.handle,x,dx)

    def neighborhood(self,s,z,mu):
        return bool(self.lib.soc_neighborhood(self.handle,s,z,mu))

    def __del__(self):
        if getattr(self,"handle",None):
            self.lib.soc_free(self.handle)
            self.handle=None
