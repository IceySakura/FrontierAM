"""Held-out checked Gaussian kernel control with a known normal mean."""
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

from numerics import log


class NormalKernel:
    def __init__(self, d, narrow):
        a=np.array([.36,.86,1.05,1.34][:min(d,4)])*narrow
        b=np.array([.52,1.13,1.28,1.62,1.95,2.05,2.25,2.40][:min(d,8)])*narrow
        self.widths=(a,b)
        self.scale=np.array([.67,.28])
        self.d=d

    def __call__(self, x, y):
        out=0.
        for s, w in zip(self.scale,self.widths):
            k=len(w)
            out=out+s*np.exp(-.5*cdist(x[:,:k]/w,y[:,:k]/w,metric="sqeuclidean"))
        kk=x[:,1:]@y[:,1:].T
        kk *= .12/max(8,self.d-1)
        kk *= np.exp(-.5*((x[:,0,None]-y[None,:,0])/.63)**2)
        return out+kk

    def mean(self, x):
        return sum(s*np.prod(w/np.sqrt(1+w*w))*
                   np.exp(-.5*np.sum(x[:,:len(w)]**2/(1+w*w),axis=1))
                   for s,w in zip(self.scale,self.widths))


def kernel_update(ctx, fun, R, x, y, nbase, train):
    if ctx.time_left()<18. or len(x)<600:
        return fun,0.
    # The split is also excluded from the fit of the warp coefficients.
    Z=x@R
    r=y-fun(x)
    X,r0=Z[train],r[train]
    inds=np.flatnonzero(~train)
    if np.count_nonzero(inds<nbase)<48:
        return fun,0.
    rr=r[inds]
    wt0=np.where(inds<nbase,1.0,.24*np.exp(-.035*Z[inds,0]**2))
    base=np.sqrt(np.sum(wt0*rr**2)/np.sum(wt0))
    if base<2e-6*np.sqrt(np.mean(y*y)):
        return fun,0.
    W=.65+np.exp(-.055*X[:,0]**2)
    W*=np.where(np.flatnonzero(train)<nbase,1.,.67)
    best=None
    for a in (1.,1.62):
        if best is not None and ctx.time_left()<17.:
            break
        K=NormalKernel(Z.shape[1],a)
        G=K(X,X)
        G.flat[::len(G)+1]+=.055/W
        c=linalg.solve(G,r0,assume_a="pos")
        v=K(Z[inds],X)@c
        err=np.sqrt(np.sum(wt0*(rr-v)**2)/np.sum(wt0))
        if best is None or err<best[0]:
            best=err,K,c
    err,K,c=best
    if err>.96*base:
        return fun,0.
    mu=float(K.mean(X)@c)
    if not np.isfinite(mu) or abs(mu)>.25*max(1e-10,np.std(y)):
        return fun,0.
    log("option kernel",len(X),"heldout",err,base,"mean",mu)

    def corrected(z):
        p=fun(z)
        q=z@R
        for i in range(0,len(z),512):
            j=i+512
            p[i:j]+=K(q[i:j],X)@c
        return p
    return corrected,mu
