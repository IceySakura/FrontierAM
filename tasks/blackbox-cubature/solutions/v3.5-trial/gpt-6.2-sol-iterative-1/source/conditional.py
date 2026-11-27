"""Defensive one-coordinate importance sampling for option residuals."""

import numpy as np
from scipy.special import ndtr, ndtri

from core import blocks, uniform


def mixture_inverse(u, mu, sd, p0):
    # The inverse of a continuous mixture CDF avoids a component switch in
    # the integration coordinates.
    q = ndtri(np.clip(u, 1e-13, 1-1e-13))
    lo = np.minimum(q, mu+sd*q)
    hi = np.maximum(q, mu+sd*q)
    x = p0*q + (1-p0)*(mu+sd*q)
    for _ in range(22):
        z = (x-mu)/sd
        y = p0*ndtr(x) + (1-p0)*ndtr(z) - u
        lo = np.where(y < 0, x, lo)
        hi = np.where(y > 0, x, hi)
        der = (p0*np.exp(-.5*x*x) + (1-p0)*np.exp(-.5*z*z)/sd)/np.sqrt(2*np.pi)
        p = x-y/np.maximum(der,1e-30)
        x = np.where(abs(y)<2e-14, x, np.where((p>lo)&(p<hi), p, .5*(lo+hi)))
    z = (x-mu)/sd
    weight = 1/(p0+(1-p0)*np.exp(np.clip(.5*(x*x-z*z),-700,700))/sd)
    return x, weight


def payoff_importance(f, X, Y, V, proxy, proxy_mean, fallback):
    xx = X@V[:,0]
    resid = np.maximum(Y,0.) if proxy is None else np.abs(Y-proxy(X))
    ww = (resid / max(1e-16,np.max(resid)))**.72
    w = ww/max(1e-15,np.sum(ww))
    n_eff = 1/max(1e-15,w@w)
    if np.sum(resid>0) < 5 or n_eff < 4:
        return None
    mu = float(np.clip(w@xx,-2.7,2.7))
    sd = float(np.sqrt(w@((xx-mu)**2)))
    sd = np.clip(sd*min(1.,n_eff/38)+max(0.,1-n_eff/38)*.88,.43,1.08)
    if abs(mu) < .16 and sd > .91:
        return None
    p0 = .20 if proxy is None else .28
    acc = []
    weights = []
    for n in blocks(f.remaining):
        if n < 16 or (acc and f.left() < 4.):
            break
        uu = uniform(n,f.d,f.rng)
        t,lw = mixture_inverse(uu[:,0],mu,sd,p0)
        W = ndtri(np.clip(uu,1e-13,1-1e-13))
        W[:,0] = t
        z = W@V.T
        y = f.query(z)
        if proxy is not None:
            y -= proxy(z)
        acc.append(float(np.mean(y*lw)))
        weights.append(n**1.30)
    return proxy_mean+np.average(acc,weights=weights) if acc else None
