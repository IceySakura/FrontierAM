"""Payoff controls and conditional sampling for zero-masked integrands."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri

from cubature import Quadratic, net_sizes, normals, ordered_basis, quadratic_cdf, rotation, uniform
from kinks import debug


def _logistic(x, y, beta=None):
    d = x.shape[1]
    if beta is None:
        beta = np.zeros(d+1)
    weight = 1.0/len(x)
    reg = 1.7e-5

    def objective(beta):
        logit = x@beta[:-1]+beta[-1]
        delta = (expit(logit)-y)*weight
        val = np.mean(np.logaddexp(0.,logit)-y*logit) + .5*reg*(beta@beta)
        g = np.r_[x.T@delta,delta.sum()] + reg*beta
        return val,g

    return minimize(objective,beta,jac=True,method="L-BFGS-B",
                    options={"maxiter":80,"ftol":1e-11}).x


def _fit_root_quadratic(x, y):
    n, d = x.shape
    mid = np.sum(x*x,axis=1)-d
    features = np.c_[x,mid,np.ones(n)]
    reg = np.r_[np.full(d, .03*n), .10*n, 1e-8]
    mean_coef = np.linalg.solve(features.T@features+np.diag(reg),features.T@y)
    resid = y-features@mean_coef
    cov = .5*(x.T*resid)@x/n
    vals, W = np.linalg.eigh(cov)
    W = W[:, np.argsort(-abs(vals))[:min(7,d)]]
    w = x@W
    i,j = np.triu_indices(w.shape[1])
    full = np.c_[features, w[:,i]*w[:,j]-(i==j)]
    reg = np.r_[np.full(d,.035*n),.12*n,1e-8,np.full(len(i),.43*n)]
    coef = np.linalg.solve(full.T@full+np.diag(reg),full.T@y)
    Basic = coef[d]*np.eye(d)
    C = np.zeros((w.shape[1],w.shape[1]))
    C[i,j] = coef[d+2:]
    C = .5*(C+C.T)
    A = Basic+W@C@W.T
    b,c = coef[:d],coef[d+1]-np.trace(A)
    # The regression basis was data-derived; use a shrinkage floor in the
    # bracketing width rather than claiming that the training RMS is a bound.
    model = Quadratic(A,b,c)
    err = y-model(x)
    spread = max(np.sqrt(np.mean(err**2)), .018)
    return model, spread


def _train_roots(s, x, v, offset, width, cuts=7):
    def label(t):
        return s.query(x+t[:,None]*v)>0

    n = len(x)
    lo, hi = np.full(n,offset-width),np.full(n,offset+width)
    br1,br2 = label(lo),label(hi)
    ok = ~br1&br2
    lo[br1] -= 1.6
    hi[~br2] += 1.6
    for _ in range(cuts):
        mid = .5*(lo+hi)
        y = label(mid)
        lo = np.where(y,lo,mid)
        hi = np.where(y,mid,hi)
    return .5*(lo+hi), ok


def _digital_block(s, rng, power, U, model, R, w, nsteps):
    d = s.d
    uu = uniform(rng,power,d)
    x = ndtri(uu[:,:-1]) @ R.T
    z = x@U[:,1:].T
    pred = model(x)

    def label(t):
        return s.query(z+t[:,None]*U[:,0]) > 0

    spread = w*np.clip(np.linalg.norm(x,axis=1)/np.sqrt(d-1),.75,1.65)
    a,b = pred-spread,pred+spread
    ok1,ok2 = ~label(a),label(b)
    lo = np.where(ok1,ndtr(a),0.)
    hi = np.where(ok2,ndtr(b),1.)
    for _ in range(nsteps):
        t = .5*(lo+hi)
        y = label(ndtri(np.clip(t,1e-14,1-1e-14)))
        lo = np.where(y,lo,t)
        hi = np.where(y,t,hi)
    base=.5*(lo+hi)
    u=uu[:,-1]
    alpha=.032
    t=np.where(u<alpha,u/alpha,lo+(hi-lo)*(u-alpha)/(1-alpha))
    inv=1/(alpha+np.where((t>=lo)&(t<=hi),(1-alpha)/np.maximum(1e-100,hi-lo),0.))
    y=label(ndtri(np.clip(t,1e-14,1-1e-14)))
    val=1-base+inv*(y.astype(float)-(t>base))
    return np.sum(val-ndtr(-pred)),len(z),np.mean(~ok1 | ~ok2)


def digital(s, rng, zp, yp, amplitude):
    d = s.d
    if s.remaining < 6000:
        return None
    zz = normals(rng,9,d)
    yy = s.query(zz)
    x,y = np.vstack([zp,zz]),np.r_[yp,yy]>0
    if not 0.02 < np.mean(y) < .98:
        return None
    beta = _logistic(x,y)
    for width in (.63,.245):
        mag = np.linalg.norm(beta[:-1])
        if mag < 0.1:
            return None
        v,offset = beta[:-1]/mag,-beta[-1]/mag
        z = normals(rng,8,d)
        tt = rng.uniform(-width,width,len(z)) + offset
        z += (tt-z@v)[:,None]*v
        yy = s.query(z)>0
        x,y = np.vstack([x,z]),np.r_[y,yy]
        beta = _logistic(x,y,beta)
    mag = np.linalg.norm(beta[:-1])
    v, offset = beta[:-1]/mag, -beta[-1]/mag
    if abs(offset)>6 or not np.isfinite(offset):
        return None
    U = rotation(v)
    x = normals(rng,8,d-1)
    eta, good = _train_roots(s,x@U[:,1:].T,v,offset,.62)
    if np.mean(good) < .80:
        return None
    model,spread = _fit_root_quadratic(x,eta)
    B = model.A@model.A + np.outer(model.b,model.b)*.18
    R = ordered_basis(B)
    A = np.zeros((d,d))
    A[1:,1:] = model.A
    exact = quadratic_cdf(Quadratic(A,np.r_[-1.,model.b],model.c))
    margin=max(.068,4.3*spread)
    # Pick how many comparisons fit while retaining the largest digital net.
    nsteps = 3 if s.remaining>=2048*6 else 4
    n = s.remaining//(3+nsteps)
    total, count = 0.,0
    bad = []
    for p in net_sizes(n):
        if s.elapsed>s.limit-4.5 and count >= 512:
            break
        a,b,c = _digital_block(s,rng,p,U,model,R,margin,nsteps)
        total+=a
        count+=b
        bad.append(c)
    debug("digital offset",offset,"root sd",spread,"n",count,"CV",exact,"escapes",bad)
    return amplitude*(exact+total/max(1,count))
