"""Nested Gaussian sparse rules for validated low-dimensional residuals."""
from collections import defaultdict
from itertools import combinations, product
import math

import numpy as np
from scipy import special


# Nested symmetric abscissae, in the exp(-x*x) convention. Weights are
# calculated from Gaussian moments, not stored.
_GK = np.array([0., .52403354748695763, .87004089535290285, 1.2247448713915889,
                1.8357079751751868, 2.0232301911005157, 2.2665132620567876,
                2.9592107790638380, 3.6677742159463378, 4.4995993983103881])
_levels = ([0], [0, 3], [0, 1, 3, 5, 7], list(range(10)))


def sparse_weights(d, level):
    a = np.r_[np.sqrt(2)*_GK, -np.sqrt(2)*_GK[1:]]
    prev = np.zeros(len(a))
    diff = []
    for ids in _levels:
        t = a[list(ids)]
        deg = 2*np.arange(len(t))
        V = special.eval_hermitenorm(deg[:, None], t)
        V /= np.sqrt(special.factorial(deg))[:, None]
        g = np.zeros(len(t))
        g[0] = 1.
        w = np.linalg.solve(V, g)
        u = np.zeros(len(a))
        u[0] = w[0]
        for j, f in zip(ids[1:], w[1:]):
            u[j] = u[9+j] = .5*f
        dif = u-prev
        diff.append([(i, x) for i, x in enumerate(dif) if abs(x)>3e-16])
        prev = u
    hi, lo = defaultdict(float), defaultdict(float)
    hi[(0,)*d] = lo[(0,)*d] = 1.
    for k in range(1, min(d, level)+1):
        plans = [v for v in product(range(1, min(level, 3)+1), repeat=k) if sum(v)<=level]
        terms = []
        for P in plans:
            low = sum(P)<level
            for co in product(*(diff[j] for j in P)):
                wi = math.prod(aw for _, aw in co)
                if abs(wi)>3e-17:
                    terms.append((tuple(i for i, _ in co),wi,low))
        for cols in combinations(range(d), k):
            for elt, w, low in terms:
                q = [0]*d
                for j, i in zip(cols, elt):
                    q[j]=i
                q=tuple(q)
                hi[q]+=w
                if low:
                    lo[q]+=w
    ix=[q for q in hi if max(abs(hi[q]),abs(lo[q]))>2e-15]
    pts=a[np.asarray(ix)]
    return pts, np.array([lo[q] for q in ix]), np.array([hi[q] for q in ix])


def sparse_residual(ctx, model, mean, x, y):
    from hard import hermite_ridge_control
    from numerics import log, normal
    d=ctx.d
    lev=4 if d<=9 else 3
    try:
        a, cv, R = hermite_ridge_control(ctx,model,mean,x,y,spend_hessian=True)
        P,WL,WH=sparse_weights(d,lev)
        while len(P)>ctx.remaining-2200 and lev>2:
            lev-=1
            P,WL,WH=sparse_weights(d,lev)
        if len(P)>ctx.remaining-1700:
            return None
        X=P@R.T
        smooth=ctx.query(X)-cv(X)
        lower=a+WL@smooth
        upper=a+WH@smooth
    except (ValueError,np.linalg.LinAlgError):
        return None
    k=1 << max(0,(ctx.remaining//4).bit_length()-1)
    vals=[]
    scat=[]
    for _ in range(min(8,ctx.remaining//(2*k))):
        Z=normal(k,d,ctx.rng)@R.T
        Z=np.vstack([Z,-Z])
        e=ctx.query(Z)-cv(Z)
        vals.append(e.mean())
        scat.append(e.std())
    if not vals:
        return upper
    z=a+np.mean(vals)
    sd=np.std(vals)/np.sqrt(max(1,len(vals)-1))
    delta=abs(upper-lower)
    # Convergence across levels and an independent stochastic estimate
    # protect against a missed branch in the nominally smooth residual.
    err_scale=max(1e-12,sd+.008*np.mean(scat)/np.sqrt(k))
    if abs(z-upper)>4.5*err_scale:
        f=0.
    else:
        f=np.clip(1.1*err_scale/(delta+.12*err_scale),.04,.97)
        if lev>=4 and delta<3.0*err_scale:
            f=max(.73,f)
    log("sparse smooth check",lev,len(WL),z,lower,upper,sd,f)
    return float(z+f*(upper-z))
