"""Cubature branches for payoffs with an exact zero region."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr, ndtri

from core import basis_from, fit_ridge, gaussian, grad_basis, linear_call_mean, net_integral, uniform, unit
from conditional import payoff_importance
from callfit import fitted_call, transverse_features
from barriers import barrier_control
from quadratic import positive_probability


def halfspace(X, indicator):
    # Regularized logistic pilot, a rotation estimator rather than a payoff model.
    d = X.shape[1]
    X1 = np.column_stack((np.ones(len(X)), X))
    yy = np.asarray(indicator, dtype=float)
    pen = .6

    def loss(b):
        q = X1 @ b
        v = np.maximum(q, 0) + np.log1p(np.exp(-np.abs(q))) - yy*q
        p = .5 + .5*np.tanh(.5*q)
        return v.sum() + .5*pen*np.dot(b,b), X1.T @ (p-yy) + pen*b

    x = np.zeros(d+1)
    x[0] = np.log((yy.sum()+1)/(len(yy)-yy.sum()+1))
    b = minimize(loss, x, jac=True, method="L-BFGS-B",
                 options={"maxiter": 80, "ftol": 1e-11}).x
    return -b[0]/max(1e-10, np.linalg.norm(b[1:])), unit(b[1:])


def boundary_model(w, t):
    d=w.shape[1]
    k=min(d,8)
    Q,_=transverse_features(w,k,0)
    X=np.column_stack((np.ones(len(w)),w,Q))
    m=max(16,len(w)//5)
    # Early boundary points come from a scramble independent of the final rule.
    base=fit_ridge(X[m:,:d+1],t[m:],.55)
    model=fit_ridge(X[m:],t[m:],1.20)
    e1=np.mean((t[:m]-X[:m,:d+1]@base)**2)
    e2=np.mean((t[:m]-X[:m]@model)**2)
    if e2 < .94*e1:
        coeff=fit_ridge(X,t,1.12)
        n=np.arange(k)
        i,j=np.triu_indices(k,1)
        A=np.zeros((d,d))
        A[n,n]=coeff[d+1:d+1+k]/np.sqrt(2)
        A[i,j]=A[j,i]=.5*coeff[d+1+k:]
    else:
        coeff=fit_ridge(X[:,:d+1],t,.45)
        A=np.zeros((d,d))
    c=coeff[0]-np.trace(A)
    b=coeff[1:d+1]
    AA=np.zeros((d+1,d+1))
    AA[:d,:d]=-A
    mean=positive_probability(AA,np.r_[-b,1.],-c)
    width=np.clip(.14+4.2*np.sqrt(min(e1,e2)),.22,1.80)
    return A,b,c,mean,width


def digital(f, X, Y):
    pos = Y > 0
    amt = float(np.max(Y))
    if not np.any(pos) or np.all(pos):
        return net_integral(f, np.eye(f.d), fallback=float(Y.mean()))
    t, v = halfspace(X, pos)
    t = np.clip(t, -4.25, 4.25)
    V = basis_from(np.eye(f.d), v)
    z = gaussian(64, f.d-1, f.rng) @ V[:, 1:].T
    # Endpoints test both orientations and detect failures of the graph model.
    a, b = -7.9, 7.9
    low = f.query(z+a*v) > 0
    high = f.query(z+b*v) > 0
    if np.any(low) or not np.all(high):
        def cv(x):
            return amt * (x@v > t)
        return net_integral(f, V, cv, amt*ndtr(-t), float(Y.mean()))

    def roots(z, center, halfwidth, steps, hard):
        lo = center - halfwidth + np.zeros(len(z))
        hi = center + halfwidth + np.zeros(len(z))
        if not hard:
            q0 = f.query(z+lo[:, None]*v) > 0
            q1 = f.query(z+hi[:, None]*v) > 0
            lo[q0] = a
            hi[~q1] = b
        for _ in range(steps):
            q = (lo+hi)*.5
            positive = f.query(z+q[:, None]*v) > 0
            hi[positive] = q[positive]
            lo[~positive] = q[~positive]
        return (lo+hi)*.5

    # Re-estimate the normal from accurately resolved sections, then use
    # principal curvatures of the boundary to order the transverse axes.
    n = 128
    W = gaussian(n, f.d-1, f.rng)
    zz = W @ V[:, 1:].T
    r = roots(zz, t, 1.2, 12, False)
    lin = fit_ridge(np.column_stack((np.ones(n), W)), r, .6)
    v1 = unit(v - V[:, 1:] @ lin[1:])
    resid = r-lin[0]-W @ lin[1:]
    A = zz.T @ (resid[:, None] * zz) / n
    V = basis_from(A @ A, v1)
    bd=zz+r[:,None]*v
    v = v1
    A,b0,c0,mean,width=boundary_model(bd@V[:,1:],bd@v)
    n = min(1024, 1 << (max(1, f.remaining//12).bit_length()-1))
    uv=uniform(n,f.d+1,f.rng)
    ww=ndtri(np.clip(uv[:,:f.d-1],1e-13,1-1e-13))
    Z=ww@V[:,1:].T
    pred=np.einsum("ij,ij->i",ww@A,ww)+ww@b0+c0
    cen=np.clip(pred,-5.5,5.5)
    lo=cen-width
    hi=cen+width
    zero=f.query(Z+lo[:,None]*v)==0
    pos=f.query(Z+hi[:,None]*v)>0
    lo[~zero]=a
    hi[~pos]=b
    for _ in range(min(15,f.remaining//n-2)):
        tmid=.5*(lo+hi)
        yes=f.query(Z+tmid[:,None]*v)>0
        lo[~yes]=tmid[~yes]
        hi[yes]=tmid[yes]
    L,H=ndtr(lo),ndtr(hi)
    q=H-L
    sc=L+q*uv[:,-2]
    tail=(1-q)*uv[:,-1]
    isright=tail>L
    tail+=q*isright
    t0=ndtri(np.clip(sc,1e-14,1-1e-14))
    t1=ndtri(np.clip(tail,1e-14,1-1e-14))
    near=f.query(Z+t0[:,None]*v)>0
    off=f.query(Z+t1[:,None]*v)>0
    inner = ndtr(-hi)+q*near+(1-q)*(off.astype(float)-isright)
    return amt*float(np.clip(mean+np.mean(inner-ndtr(-pred)),0,1))


def call_proxy(X, y, Kmin=0.):
    pos = y > 1e-9
    X1 = np.column_stack((np.ones(pos.sum()), X[pos]))
    Y = y[pos]
    if len(Y) < max(24, int(.7*X.shape[1])):
        return None
    # The strike in discounted payoff units is a nuisance parameter in
    # log(y + strike); solve the linear fit exactly for each trial strike.
    sc = max(np.mean(Y), 1e-9)
    A = X1.T @ X1 + .15*np.eye(X1.shape[1])
    P = solve_pos(A, X1.T)

    def candidate(t, full=False):
        strike = sc*np.exp(t)
        Y1 = np.log(Y + strike)
        b = P @ Y1
        b[0] += np.mean(Y1-X1 @ b)
        pred = sc * np.expm1(np.clip(X1 @ b - np.log(strike), -4, 4)) * np.exp(t)
        loss = np.mean(((pred-Y)/(1+Y/sc))**2) / sc**2
        if full:
            return float(strike), float(b[0]), b[1:]
        return loss

    lo=max(-1.7, np.log(max(Kmin,sc*1e-4)/sc))
    ans = minimize_scalar(candidate, bounds=(lo, max(lo+.03,4.5)), method="bounded",
                          options={"xatol": .04})
    return candidate(ans.x, True)


def solve_pos(a, b):
    from scipy.linalg import solve
    return solve(a, b, assume_a="pos", check_finite=False)


def payoff(f, X, Y):
    unique = np.unique(Y)
    if len(unique) == 2 and unique[0] == 0:
        return digital(f, X, Y)
    pos = Y > 0
    # Additional positive points are sought only when the support is sparse.
    if pos.sum() < 32:
        extra = gaussian(min(1024, f.remaining//4), f.d, f.rng)
        val = f.query(extra)
        X = np.vstack((X, extra))
        Y = np.r_[Y, val]
        pos = Y > 0
    gp = min(32, f.remaining//(5*f.d))
    V, G, ix = grad_basis(f, X, Y, gp, positive=True)
    fallback=float(Y.mean())
    base = call_proxy(X, Y)
    if base is None or np.mean(pos) < .13:
        return net_integral(f, V, fallback=fallback)
    K, c, b = base
    if not (.035 <= np.linalg.norm(b) <= 1.9):
        return net_integral(f, V, fallback=fallback)

    def proxy(x):
        return np.maximum(0, np.exp(np.clip(c+x@b,-100,30))-K)

    fitted = proxy(X)
    corr=np.corrcoef(Y,fitted)[0,1]
    beta = np.clip(np.cov(Y, fitted, ddof=0)[0,1]/(np.var(fitted)+1e-16), 0.0, 1.2)
    mismatch=np.mean((~pos)&(fitted>.16*fallback))
    if corr>.88 and mismatch<.032 and .025<b@V[:,0] and f.remaining>=11750:
        K0,c0,b0=call_proxy(X,Y,max(K,2.8*np.median(Y[pos])))
        cc=fitted_call(f,X,Y,V,K0,c0,b0)
        if cc is not None:
            return cc
    if beta < .10 or not (corr>.58 and np.linalg.norm(b) < 1.04):
        cv, mean = None, 0.
    else:
        cv, mean = lambda x: beta*proxy(x), beta*linear_call_mean(c,b,K)
    if mismatch>.022 and cv is not None and f.remaining>=8192:
        if f.remaining>=10600 and .034 < mismatch < .28 and pos.mean()>.14:
            est=barrier_control(f,X,Y,V,G,K,c,b,proxy,fallback)
            if est is not None:
                return est
        est=payoff_importance(f,X,Y,V,cv,mean,fallback)
        if est is not None:
            return est
    return net_integral(f,V,cv,mean,fallback=fallback)
