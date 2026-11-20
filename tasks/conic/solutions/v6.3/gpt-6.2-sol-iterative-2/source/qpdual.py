"""Small multiplier QP when the primal Hessian is positive definite."""
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lapack
from scipy.sparse.linalg import splu

from model import Layout, Problem, maxabs


def solve(p,run,deadline):
    n,m=p.n,p.m
    if (n<140 or not 1<=m<=155 or m>.16*n or n*m>5_000_000
            or p.K.nonlinear or not p.P.nnz):
        return None
    h=p.P.diagonal()
    mx=maxabs(h)
    if mx==0. or np.min(h)<=1.e-15*mx:
        return None
    if p.P.nnz==n:
        val=1./h
        inv=lambda r:r*val if r.ndim==1 else r*val[:,None]
    elif n<1050 and p.P.nnz>.10*n*n:
        L,info=lapack.dpotrf(p.P.toarray(order='F'),lower=1,overwrite_a=1)
        if info:
            return None
        inv=lambda r:lapack.dpotrs(L,r,lower=1)[0]
    else:
        if p.P.nnz>7*n or p.P.nnz>450000:
            return None
        try:
            L=splu(p.P,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=.05)
            u=L.U.diagonal()
            if np.min(np.abs(u),initial=np.inf)<maxabs(u)*1.e-15:
                return None
            inv=L.solve
        except (ValueError,RuntimeError):
            return None
    if perf_counter()>deadline:
        return None
    q=inv(p.q)
    x=-q
    first=p.optimal(x,np.zeros(m),polish_slack=False)
    if first is not None:
        return first
    if perf_counter()>deadline:
        return None
    Q=inv(p.A.T.toarray(order='F'))
    H=p.A@Q
    H=(H+H.T)*.5
    l=p.K.lpos
    mat=sparse.csc_matrix((-np.ones(len(l)),(np.arange(len(l)),l)),shape=(len(l),m))
    rd=Problem(sparse.csc_matrix(H),p.b+p.A@q,mat,np.zeros(len(l)),
               Layout([('nonnegative',len(l))] if len(l) else []))
    st=run(rd)
    if st is None:
        return None
    if st['status']=='dual_infeasible':
        return p.primal_cert(st['x'])
    if st['status']=='solved':
        z=st['x']
        return p.optimal(inv(-p.q-p.A.T@z),z)
    return None
