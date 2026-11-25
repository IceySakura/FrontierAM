"""Residual integration with an even Gaussian polynomial control."""
import numpy as np
from scipy import linalg

from common import log, sobol


def gradient_frame(s,residual,d,few=False):
    n = min(22,max(0,(s.remaining-4200)//(2*(d+1))))
    if few and s.remaining>3150:
        n=max(n,min(9,(s.remaining-2300)//(2*(d+1))))
    if n<5:
        return np.eye(d)
    x=s.rng.normal(size=(n,d))
    x*=np.linspace(.67,1.17,n)[:,None]
    x=np.r_[x,-x]
    h=.019
    stencil=np.r_[np.zeros((1,d)),h*np.eye(d)]
    p=(x[:,None,:]+stencil).reshape(-1,d)
    f=residual(p,s.query(p)).reshape(len(x),d+1)
    F=(f[:,1:]-f[:,:1])/h
    G=.5*(F[:n]-F[n:])
    nn=np.linalg.norm(G,axis=1)
    med=np.median(nn)
    G*=np.minimum(1.,5.*med/np.maximum(nn,1e-50))[:,None]
    # The smaller second term keeps axes visible in the unsymmetrized data.
    H=G.T@G + .12*(F[:n]+F[n:]).T@(F[:n]+F[n:])
    return linalg.eigh(H,check_finite=False)[1][:,::-1].copy()


def features(z,n):
    d=z.shape[1]
    k=min(d,9 if n>2200 else 7)
    l=min(k,5)
    q=z*z
    h2=(q-1.)/np.sqrt(2.)
    h3=(z[:,:k]**3-3.*z[:,:k])/np.sqrt(6.)
    h4=(q*q-6.*q+3.)/np.sqrt(24.)
    x=z[:,:k]
    u=q[:,:k]
    h6=(u**3-15.*u*u+45.*u-15.)/np.sqrt(720.)
    h8=(u**4-28.*u**3+210.*u*u-420.*u+105.)/np.sqrt(40320.)
    i,j=np.triu_indices(k,1)
    ii,jj=np.where(~np.eye(l,dtype=bool))
    result = [np.ones((len(z),1)), h2, h4, x[:,i]*x[:,j], h2[:,i]*h2[:,j],
              h3[:,ii]*x[:,jj],h6,h8]
    if n>950:
        ir,jr=np.where(np.triu(np.ones((min(4,k),d),dtype=bool),k=1))
        use=jr>=k
        result.append(z[:,ir[use]]*z[:,jr[use]])
        for i in range(l):
            I,J=np.triu_indices(l,1)
            g=(I!=i)&(J!=i)
            result.append(h2[:,i:i+1]*x[:,I[g]]*x[:,J[g]])
    return np.concatenate(result,axis=1)


def residual_mean(s,residual,x,y,known,shape_controls=(),try_sparse=False):
    q = residual(x,y)
    if np.all(np.isfinite(q)):
        s.best=float(known+np.mean(q))
    d=s.d
    W=gradient_frame(s,residual,d,bool(shape_controls))
    if try_sparse and d==8:
        from sparse_gauss import sparse_mean
        v=sparse_mean(s,residual,W,known)
        if v is not None:
            return v
    n=max(0,s.remaining-340)//2 if getattr(getattr(residual,"__self__",None),"reflect",None) is not None else s.remaining//2
    if n<64:
        return s.best
    parts=[]
    a=n
    mx=2048 if n>=5600 else 1<<max(5,(n//3).bit_length()-1)
    while a>=32 and len(parts)<7:
        k=min(mx, 1<<(a.bit_length()-1))
        parts.append(k)
        a-=k
    batches=[]
    for k in parts:
        if s.elapsed>22.1 or s.remaining<2*k:
            break
        g = sobol(k,d,s.rng)
        g0=g@W.T
        z=np.r_[g0,-g0]
        r=residual(z,s.query(z))
        batches.append((g,.5*(r[:k]+r[k:])))
    if not batches:
        return s.best
    mm=np.array([len(v) for _,v in batches])
    est=np.average([r.mean() for _,r in batches],weights=mm)
    if len(batches)<2 or np.sum(mm)<270:
        return float(known+est)
    mats=[]
    for z,_ in batches:
        F=features(z,np.sum(mm))
        if shape_controls:
            t=z@W.T
            X=[]
            for q in shape_controls:
                A=q.A
                v=np.einsum('ni,ni->n',t@A,t)-np.trace(A)
                X.append(v/max(.01,np.sqrt(2*np.sum(A*A))))
            F=np.c_[F,np.stack(X,axis=1)]
        mats.append(F)
    Gr=np.array([f.T@f for f in mats])
    br=np.array([f.T@r for f,(_,r) in zip(mats,batches)])
    G=Gr.sum(axis=0)
    b=br.sum(axis=0)
    ridge=1.2+.0006*np.sum(mm)
    G.flat[::len(G)+1]+=ridge
    G[0,0]-=ridge
    ds=[]
    rho=[]
    for f,(_,r),g,bb in zip(mats,batches,Gr,br):
        t=linalg.solve(G-g,b-bb,assume_a='pos')
        pred=f@t
        frac=np.std(r-pred)/max(1e-100,np.std(r))
        # In a failed geometry branch, overfitting should not magnify noise.
        share=min(1.,1.3/max(1.3,frac))
        ds.append(share*(t[0]-pred.mean()))
        rho.append(frac)
    v=float(known+est+np.average(ds,weights=mm))
    log("smooth estimate",known,est,"controls",np.average(ds,weights=mm),
        "ratios",np.around(rho,3),"n",np.sum(mm),"value",v)
    return v
