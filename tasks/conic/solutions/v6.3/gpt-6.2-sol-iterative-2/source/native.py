import ctypes as C
from pathlib import Path

import numpy as np


try:
    lib = C.CDLL(str(Path(__file__).with_name('libconic.so')))
    v, i, f = C.c_void_p, C.c_int, C.c_double
    lib.conic_ldl_create.argtypes = (i,v,v,v,v,i,f)
    lib.conic_ldl_create.restype = v
    lib.conic_ldl_factor.argtypes = (v,v)
    lib.conic_ldl_factor.restype = i
    lib.conic_ldl_solve.argtypes = (v,v,v)
    lib.conic_ldl_solve.restype = None
    lib.conic_ldl_free.argtypes = (v,)
    lib.conic_ldl_free.restype = None
    lib.conic_schur.argtypes = (i,i,v,v,v,v,v,v)
    lib.conic_schur.restype = None
    lib.conic_psd_columns.argtypes = lib.conic_schur.argtypes
    lib.conic_psd_columns.restype = None
    lib.conic_project_exp.argtypes = (i,v)
    lib.conic_project_exp.restype = None
    lib.conic_gram.argtypes = (i,i,v,v,v,v,v)
    lib.conic_gram.restype = None
    lib.conic_eq_pivots.argtypes = (i,i,v,v,v,v,v,v,v)
    lib.conic_eq_pivots.restype = i
    lib.conic_scatter.argtypes = (i,v,v,v,v,v)
    lib.conic_scatter.restype = None
    lib.conic_network.argtypes = (i,i,v,v,v,v,v,v,v,v,v,f)
    lib.conic_network.restype = i
    lib.conic_diagonal_qp.argtypes = (i,i,v,v,v,v,v,v,v,v,v,f)
    lib.conic_diagonal_qp.restype = i
    lib.conic_group_rows.argtypes = (i,v,v,v,v,v,v)
    lib.conic_group_rows.restype = i
    lib.conic_band_qp.argtypes = (i,i,v,v,v,v,v,v,v,v,v,v,v,f)
    lib.conic_band_qp.restype = i
    lib.conic_components.argtypes = (i,i,v,v,v,v,v,v,v,v,i,v,v,v,v,v,v,v)
    lib.conic_components.restype = i
    lib.conic_core_qp.argtypes = (i,i,v,v,v,v,v,v,v,v,v,v,v,f,v,v,v,v,v,i,v,v)
    lib.conic_core_qp.restype = i
    lib.conic_core_soc.argtypes = (i,i,v,v,v,v,v,v,v,v,v,v,v,v,f,v,v,v,v,v,i,v,v,v,v)
    lib.conic_core_soc.restype = i
    lib.conic_watchdog.argtypes = (i,)
    lib.conic_watchdog.restype = None
except (OSError, AttributeError):
    lib = None


def address(a):
    return a.ctypes.data


class Ldl:
    def __init__(self,n,ptr,row,perm,sign,maxfill,maxops):
        self.ptr = lib.conic_ldl_create(n,address(ptr),address(row),address(perm),
                                        address(sign),maxfill,maxops)

    def __del__(self):
        if self.ptr:
            lib.conic_ldl_free(self.ptr)

    def factor(self,v):
        return self.ptr and lib.conic_ldl_factor(self.ptr,address(v))

    def solve(self,b):
        return solve_ldl(self.ptr,b)


def solve_ldl(ptr,b):
    v = np.empty_like(b)
    lib.conic_ldl_solve(ptr,address(b),address(v))
    return v


def sdp_schur(d, ptr, ri, rj, val, G):
    n = len(ptr)-1
    H = np.empty((n,n))
    lib.conic_schur(n,d,address(ptr),address(ri),address(rj),address(val),
                    address(np.ascontiguousarray(G)),address(H))
    return H


def sdp_columns(d, ptr, ri, rj, val, Ri):
    n = len(ptr)-1
    H = np.empty((n,d*(d+1)//2))
    lib.conic_psd_columns(n,d,address(ptr),address(ri),address(rj),address(val),
                          address(np.ascontiguousarray(Ri.T)),address(H))
    return H.T


def project_exp(x):
    lib.conic_project_exp(x.shape[0],address(x))
    return x


def gram_into(S,w,G):
    lib.conic_gram(S.shape[0],G.shape[0],address(S.indptr),address(S.indices),
                   address(S.data),address(w),address(G))


def eq_pivots(n,eq,C,A,P):
    out = np.full(len(eq),-1,dtype=np.int32)
    lib.conic_eq_pivots(n,len(eq),address(eq),address(C.indptr),address(C.indices),
                        address(C.data),address(A.indptr),address(P.indptr),address(out))
    return out


def scatter(slot,row,val,w,arr):
    lib.conic_scatter(len(slot),address(slot),address(row),address(val),address(w),address(arr))


def network(A,q,b,lo,hi,seconds):
    x,z = np.empty(A.n),np.empty(A.m)
    ok = lib.conic_network(A.m,A.n,address(A.ptr),address(A.idx),address(A.val),
                           address(q),address(b),address(lo),address(hi),address(x),address(z),seconds)
    return (x,z) if ok else None


def diagonal_qp(At,diag,q,b,eq,seconds):
    x,z = np.empty(At.m),np.empty(At.n)
    ok = lib.conic_diagonal_qp(At.m,At.n,address(At.ptr),address(At.idx),address(At.val),
                              address(diag),address(q),address(b),address(eq),
                              address(x),address(z),seconds)
    return (x,z) if ok else None


def group_rows(At):
    m = At.n
    groups,chosen,mul = np.empty(m,dtype=np.int32),np.empty(m,dtype=np.int32),np.empty(m)
    n = lib.conic_group_rows(m,address(At.ptr),address(At.idx),address(At.val),
                             address(groups),address(chosen),address(mul))
    return (groups,chosen[:n],mul) if 0<n<m else None


def band_qp(At,P,q,b,eq,seconds):
    x,z = np.empty(At.m),np.empty(At.n)
    ok = lib.conic_band_qp(At.m,At.n,address(At.ptr),address(At.idx),address(At.val),
                          address(P.ptr),address(P.idx),address(P.val),
                          address(q),address(b),address(eq),address(x),address(z),seconds)
    return ok,x,z


def components(p,codes):
    A,P=p.A,p.P
    a,b,c=codes
    nx,rs=np.empty(p.n,dtype=np.int32),np.empty(p.m,dtype=np.int32)
    sig=np.empty((2,p.n),dtype=np.uint64)
    n=lib.conic_components(p.n,p.m,address(A.indptr),address(A.indices),address(A.data),
                             address(P.indptr),address(P.indices),address(P.data),
                             address(p.q),address(p.b),len(a),address(a),address(b),address(c),
                             address(nx),address(rs),address(sig[0]),address(sig[1]))
    return (nx,rs,sig[:,:n].T) if n>1 else None


_blas_handle = None
_dense_api = None


def dense_api():
    global _dense_api,_blas_handle
    if _dense_api is None:
        try:
            base=Path(np.__file__).resolve().parent.parent/'scipy.libs'
            version=next(base.glob('libscipy_openblas-*.so'))
            _blas_handle=C.CDLL(str(version))
            _dense_api=tuple(C.cast(getattr(_blas_handle,'scipy_'+n+'_'),C.c_void_p)
                             for n in ('dpotrf','dpotrs','dsytrf','dsytrs','dsyrk'))
        except (OSError,StopIteration,AttributeError):
            _dense_api=(None,)*5
    return _dense_api


def core_qp(At,P,q,b,eq,seconds,plain=False,warm=None):
    x,z = np.empty(At.m),np.empty(At.n)
    ok = lib.conic_core_qp(At.m,At.n,address(At.ptr),address(At.idx),address(At.val),
                          address(P.ptr),address(P.idx),address(P.val),
                          address(q),address(b),address(eq),address(x),address(z),
                          seconds,*dense_api(),int(plain),
                          None if warm is None else address(warm[0]),
                          None if warm is None else address(warm[1]))
    return ok,x,z


def core_soc(p,seconds):
    A=p.A.tocsr()
    P=p.P
    loc=np.array([(a,d) for k,d,a,b in p.K.nonlinear],dtype=np.int32)
    starts,dims=np.ascontiguousarray(loc.T)
    shape=np.array([k=='exponential' for k,d,a,b in p.K.nonlinear],dtype=bool)
    funcs=dense_api()
    svd=None
    if shape.any() and _blas_handle is not None:
        svd=C.cast(_blas_handle.scipy_dgesvd_,C.c_void_p)
    eq=np.zeros(p.m,dtype=bool)
    eq[p.K.eq]=True
    x,z,s=np.empty(p.n),np.empty(p.m),np.empty(p.m)
    ok=lib.conic_core_soc(p.n,p.m,address(A.indptr),address(A.indices),address(A.data),
                          address(P.indptr),address(P.indices),address(P.data),
                          address(p.q),address(p.b),address(eq),address(x),address(z),address(s),
                          seconds,*funcs,len(starts),address(starts),address(dims),address(shape),svd)
    return ok,x,s,z
