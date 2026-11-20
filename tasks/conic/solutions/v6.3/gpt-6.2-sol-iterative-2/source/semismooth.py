"""Semismooth KKT refinement using the derivative of cone projections."""
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, lapack
from scipy.sparse.linalg import splu

from model import Scaled, maxabs, smat, svec, upper


def block_soc(v):
    d = len(v)
    t = np.linalg.norm(v[1:])
    if t<=v[0]:
        return np.eye(d),np.ones(d),v
    if -t>=v[0]:
        return np.eye(d),np.zeros(d),np.zeros(d)
    w=v[1:]/t
    e=w.copy()
    e[0]+=1. if e[0]>=0. else -1.
    T=np.zeros((d,d))
    T[:2,0]=2.**-.5
    T[0,1:]=w*2.**-.5
    T[1,1:]=-T[0,1:]
    T[2:,1:]=np.eye(d-1)[1:]-2.*np.outer(e[1:],e)/(e@e)
    half=.5*(1.+v[0]/t)
    slope=np.r_[1.,0.,np.full(d-2,half)]
    return T,slope,np.r_[t*half,w*t*half]


def block_psd(v,d):
    p,U=eigh(smat(v,d),check_finite=False)
    plus=p.clip(0.)
    i,j,w=upper(d)
    B=(U[i[None,:],i[:,None]]*U[j[None,:],j[:,None]]
       +U[j[None,:],i[:,None]]*U[i[None,:],j[:,None]])*(.5*w[:,None]*w)
    den=p[j]-p[i]
    weights=np.where(p[i]>0.,1.,0.)
    far=den>1.e-14*(1.+maxabs(p))
    weights[far]=(plus[j[far]]-plus[i[far]])/den[far]
    return B,weights,svec((U*plus)@U.T)


def polish(p,point,deadline):
    if (point is None or p.K.exp or not p.K.nonlinear or p.n+p.m>16_000
            or (p.A.nnz>160000) or not p.n
            or any((k=='soc' and d>32) or (k=='psd' and d>10)
                   for k,d,_,_ in p.K.nonlinear)):
        return None
    w=Scaled(p,passes=5,scale_rhs=True)
    A,P,K,q,b=w.A,w.P,w.K,w.q,w.b
    n,m=p.n,p.m
    x=point[0]/w.D
    z=point[2]*w.c/w.E
    aa=m+sum((e-a)*(e-a) for k,d,a,e in K.nonlinear)
    if aa>1_200_000:
        return None
    identity=np.r_[K.eq,K.lpos]
    def funcs(x,z,assemble):
        s=b-A@x
        v=s-z
        p0=np.zeros(m)
        j=np.zeros(m)
        p0[K.lpos]=np.maximum(0.,v[K.lpos])
        j[K.lpos]=v[K.lpos]>0.
        ir=[identity]
        jc=[identity]
        vv=[np.ones(len(identity))]
        for k,d,a,e in K.nonlinear:
            S,h,y=block_soc(v[a:e]) if k=='soc' else block_psd(v[a:e],d)
            j[a:e]=h
            p0[a:e]=y
            if assemble:
                dim=e-a
                r=np.arange(a,e,dtype=np.int32)
                ir.append(np.repeat(r,dim))
                jc.append(np.tile(r,dim))
                vv.append(S.reshape(-1))
        F=s-p0
        g=P@x+q+A.T@z
        if not assemble:
            return max(maxabs(F),maxabs(g))
        T=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(ir),np.concatenate(jc))),
                             shape=(m,m))
        return T,j,F,g

    for i in range(6):
        if perf_counter()>deadline:
            break
        ans=p.optimal(x*w.D,z*w.E/w.c,polish_slack=(i>0))
        if ans is not None:
            return ans
        T,j,F,g=funcs(x,z,True)
        d0=max(maxabs(F),maxabs(g))
        if not np.isfinite(d0):
            break
        T0=T@A
        y=T@F
        free=j<1.-2.e-9
        bound=~free
        zt=np.zeros(m)
        zt[bound]=-y[bound]/j[bound]
        rhsx=-g-T0.T@zt
        h=j[free]/(1.-j[free])
        J=T0[free,:]
        M=sparse.bmat([[P,J.T],[J,-sparse.diags(h)]],format='csc')
        rt=np.r_[rhsx,y[free]/(1.-j[free])]
        sumrows=np.maximum.reduceat(np.r_[np.abs(M.data),0.],M.indptr[:-1])
        d=1./np.sqrt(np.maximum(1.e-7,sumrows))
        M=(M.multiply(d[:,None]).multiply(d[None,:])).tocsc()
        RR=M+sparse.diags(np.r_[np.full(n,9.e-12),np.full(len(h),-9.e-12)],format='csc')
        try:
            if RR.shape[0]<145:
                H,piv,flag=lapack.dsytrf(RR.toarray(order='F'),lower=1,overwrite_a=1)
                if flag:
                    break
                solve=lambda v:lapack.dsytrs(H,piv,v,lower=1)[0]
            else:
                lu=splu(RR,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=.03)
                solve=lu.solve
            v=solve(rt*d)
            err=rt*d-M@v
            if maxabs(err)>1.e-11:
                u=solve(err)
                if maxabs(err-M@u)<maxabs(err):
                    v+=u
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            break
        v*=d
        dx=v[:n]
        zt[free]=v[n:]
        dz=T.T@zt
        for st in (1.,.5,.25,.125):
            xx,zz=x+st*dx,z+st*dz
            dn=funcs(xx,zz,False)
            if dn<d0*(1.-.12*st):
                x,z=xx,zz
                break
        else:
            return None
    return None
