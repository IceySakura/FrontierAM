"""Polyhedral outer approximation of exponential-cone LPs."""
from time import perf_counter

import numpy as np
from scipy import sparse

from fastlp import Csc, create, extension
from model import exp_error


def normal(t):
    lo = np.minimum(t,0.)
    z = np.column_stack((-np.exp(lo),(t-1.)*np.exp(lo),np.exp(lo-t)))
    return z / np.maximum(1.e-8,np.max(np.abs(z),axis=1))[:,None]


def transform(m,loc,normals):
    n = len(loc)
    cols = (loc[:,None]+np.arange(3,dtype=np.int32)).reshape(-1)
    return sparse.csr_matrix((normals.reshape(-1),cols,3*np.arange(n+1,dtype=np.int32)),
                             shape=(n,m))


def solve_cuts(p,deadline,point=None):
    if p.P.nnz or p.K.psd or not p.n or p.n>14000 or p.A.nnz>250000:
        return None
    loc = p.K.groups.get(('exponential',3))
    if loc is None or len(loc)>3500 or p.K.nonlinear and len(loc)!=len(p.K.nonlinear):
        return None
    lidx = np.r_[p.K.eq,p.K.lpos]
    leq = len(p.K.eq)
    linear = sparse.csr_matrix((np.ones(len(lidx)),lidx,np.arange(len(lidx)+1,dtype=np.int32)),
                               shape=(len(lidx),p.m))
    ix = loc[:,None]+np.arange(3,dtype=np.int32)
    T = [linear]
    I = np.ones((len(loc),1))
    for row in ((0.,1.,0.),(0.,0.,1.)):
        T.append(transform(p.m,loc,I*np.asarray(row)))
    for a in (-9.,-4.,-2.,-1.,0.,1.,2.,4.,9.,22.):
        T.append(transform(p.m,loc,normal(np.full(len(loc),a))))
    if point is not None:
        s = point[1][ix]
        t = np.clip(np.log(np.maximum(1.e-150,s[:,2]))
                   -np.log(np.maximum(1.e-150,s[:,1])),-35,46)
        T.append(transform(p.m,loc,normal(t)))
    M = sparse.vstack(T,format='csr')
    w = (M@p.A).tocsc()
    u=M@p.b
    lb=np.full(len(u),-np.inf)
    lb[:leq]=u[:leq]
    h=extension('scipy.optimize._highspy._core','optimize/_highspy','_core')
    space = deadline-perf_counter()
    if space<.02:
        return None
    sp=Csc(*w.shape,w.indptr,w.indices,w.data)
    hh=create(h,sp,p.q,lb,u,None,None,None,space,'choose')
    if hh is None:
        return None
    T = [M]
    for k in range(23):
        if perf_counter()>.97*space+deadline-space:
            break
        hh.run()
        st=hh.getModelStatus()
        if st!=h.HighsModelStatus.kOptimal:
            break
        sol=hh.getSolution()
        x=np.asarray(sol.col_value)
        s=p.b-p.A@x
        S=s[ix]
        sc=np.maximum(1.,np.max(np.abs(S),axis=1))
        err=exp_error(*(S/sc[:,None]).T)
        if np.max(err,initial=0.)<9.4e-9:
            z = np.zeros(p.m)
            at = 0
            yd = -np.asarray(sol.row_dual)
            for block in T:
                e = at+block.shape[0]
                z += block.T@yd[at:e]
                at=e
            ans=p.optimal(x,z,s)
            if ans is not None:
                return ans
            if np.max(err,initial=0.)<7.e-10:
                break
        bad=err>6.e-10
        u,v,w=S[bad].T
        t = np.clip(np.log(np.maximum(1.e-150,w))-np.log(np.maximum(1.e-150,v)),-36,50)
        g = transform(p.m,loc[bad],normal(t))
        d = (g@p.A).tocsr()
        m = d.shape[0]
        if not m:
            break
        hi=g@p.b
        status=hh.addRows(m,np.full(m,-np.inf),hi,d.nnz,d.indptr,d.indices,d.data)
        if status==h.HighsStatus.kError:
            break
        T.append(g)
    return None
