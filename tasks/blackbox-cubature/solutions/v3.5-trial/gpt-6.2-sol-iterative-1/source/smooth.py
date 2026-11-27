"""Closed-mean Gaussian kernel controls for smooth residuals."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def sqdist(X, Y):
    D=X@Y.T
    D*=-2.
    D+=np.einsum("ij,ij->i",X,X)[:,None]
    D+=np.einsum("ij,ij->i",Y,Y)
    return np.maximum(D,0.,out=D)


def smooth_kernel(X, Y, V, energy):
    """Fit only if a disjoint part of the pilot confirms residual reduction."""
    n=len(X)
    d=X.shape[1]
    k=min(d, 11, max(3,int(np.sum(energy>.015*max(1e-14,energy[0])))))
    n=min(896,n)
    if n<280 or not np.isfinite(Y).all() or energy[0]<=1e-14:
        return None
    step=max(1,len(X)//n)
    idx=np.arange(0,len(X),step)[:n]
    Y=Y[idx].copy()
    med=np.median(Y)
    s=max(1.e-15, np.quantile(abs(Y-med),.80))
    Y=np.clip(Y,med-3.15*s,med+3.15*s)
    if s<5.e-8:
        return None
    rho=np.maximum(1e-4,energy[:k]/energy[0])**.245
    basis=V[:,:k]*rho
    Z=X[idx]@basis
    nt=n-n//5
    train=Z[:nt]
    dist=sqdist(train,train)
    valid=sqdist(Z[nt:],train)
    res=Y[:nt]-np.mean(Y[:nt])
    peak=max(1e-12,np.mean((Y[nt:]-np.mean(Y[:nt]))**2))
    best=None
    prec=peak*.93
    for ls in (.29,.62,1.05):
        K=np.exp(-.5*ls*dist)
        K.flat[::nt+1]+=.006+ls*.0015
        coef=cho_solve(cho_factor(K,overwrite_a=True,check_finite=False),res,check_finite=False)
        q=np.exp(-.5*ls*valid)@coef
        target=Y[nt:]-np.mean(Y[:nt])
        q-=np.mean(q)
        target-=np.mean(target)
        beta=np.clip(np.dot(q,target)/max(1e-15,np.dot(q,q)),.0,1.06)
        err=np.mean((target-beta*q)**2)
        if err < prec and beta>.14:
            best=ls,beta
            prec=err
    if best is None:
        return None
    ls,beta=best
    K=np.exp(-.5*ls*sqdist(Z,Z))
    K.flat[::n+1]+=.006+ls*.0015
    coef=beta*cho_solve(cho_factor(K,overwrite_a=True,check_finite=False),
                         Y-np.mean(Y),check_finite=False)
    renorm=1+ls*rho*rho
    expect=np.exp(-.5*ls*(Z*Z)@(1/renorm))/np.sqrt(np.prod(renorm))
    mean=float(expect@coef)

    def prediction(x):
        z=x@basis
        ans=np.empty(len(x))
        for i in range(0,len(x),2048):
            block=z[i:i+2048]
            ds=sqdist(block,Z)
            ds*=-.5*ls
            np.exp(ds,out=ds)
            ans[i:i+len(block)]=ds@coef-mean
        return ans

    return prediction
