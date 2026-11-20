"""Fits for paths with observable one-dimensional update directions."""

import numpy as np


def sym(a):
    return (a+a.T)*0.5


def directions(dx):
    n,d = dx.shape
    if d < 2:
        return None
    lens = np.linalg.norm(dx,axis=1)
    good = lens > max(1e-10,np.sqrt(np.mean(lens*lens))*1e-7)
    if good.sum() < n*.96:
        return None
    zz = dx[good]/lens[good,None]
    u = []
    for v in zz[:min(350,len(zz))]:
        if not u or np.max(np.abs(np.dot(u,v))) < 1-3e-10:
            u.append(v)
            if len(u) > d:
                return None
    if len(u) != d:
        return None
    corr = np.abs(zz @ np.asarray(u).T)
    lab = np.argmax(corr,axis=1)
    ok = np.max(corr,axis=1) > 1-3e-9
    if np.mean(ok) < .9995:
        return None
    labels = np.full(n,-1,dtype=int)
    labels[good] = np.where(ok,lab,-1)
    v = np.array(u)
    for j in range(d):
        z = dx[labels==j]
        if len(z) < max(25, 6*d):
            return None
        vals, vecs = np.linalg.eigh(z.T @ z)
        if vals[-2] > 1.e-8 * vals[-1]:
            return None
        v[j] = vecs[:,-1]
    if np.linalg.cond(v) > 1e5:
        return None
    return v, labels


def lag_check(lab, n, d):
    p = np.bincount(lab[lab>=0],minlength=d).astype(float)
    p /= p.sum()
    val = np.outer(p,p)
    w = n*val
    z = lab[:-1]*d+lab[1:]
    good = (lab[:-1]>=0) & (lab[1:]>=0)
    obs = np.bincount(z[good],minlength=d*d).reshape(d,d)
    stat = np.sum((obs-good.sum()*val)**2/np.maximum(1,w))
    df = max(1.,(d-1)**2)
    return stat < df + 6*np.sqrt(2*df) + 10


def fit(x, v, lab, schedule=None):
    from scipy.optimize import minimize

    n, d = x.shape
    y = np.linalg.solve(v.T, x.T).T
    gms = []
    xs = []
    events = []
    effects = np.zeros(d)
    for j in range(d):
        ok = (lab==j)
        a = y[:-1][ok].copy()
        b = y[1:][ok,j].copy()
        a -= a.mean(axis=0)
        b -= b.mean()
        theta = np.linalg.solve(a.T @ a,a.T @ b)
        effects[j] = np.clip(theta[j],-.90,.97)
        # For a refresh/overrelaxation, transform to the center of the
        # corresponding Gaussian conditional and unit innovation scale.
        f = a.copy()
        f[:,j] = (b-effects[j]*a[:,j])/(1-effects[j])
        xs.append(f)
        gm = f.T @ f
        gms.append(gm)
        events.append(len(f))
    weights = (1-effects)/(1+effects)
    counts = np.array(events,dtype=float)
    gms = np.array(gms)*weights[:,None,None]
    C = sym(y.T @ y/n)
    init = np.linalg.cholesky(np.linalg.solve(C,np.eye(d)))
    i,j = np.tril_indices(d)
    diag = i==j
    size = len(i)
    best = [np.inf,None]

    def obj(a):
        b = np.zeros((d,d))
        b[i,j] = a
        b[i[diag],j[diag]] = np.exp(a[diag])
        F = init @ b
        L = F @ F.T
        dv = np.diag(L)
        dot = np.einsum("ijk,ik->ij",gms,L)
        qu = np.sum(dot*L,axis=1)
        loss = (-counts@np.log(dv) + np.sum(qu/dv))/n
        reg = .0009
        loss += .5*reg*(a@a)
        Q = 2*dot/dv[:,None]
        Q.flat[::d+1] -= counts/dv + qu/dv**2
        Q = sym(Q)
        if loss < best[0]:
            best[:] = [loss, L]
        g = (2 * init.T @ Q @ F / n)[i,j]
        g[diag] *= np.exp(a[diag])
        return loss, g+reg*a

    minimize(obj,np.zeros(size),jac=True,method="L-BFGS-B",
             bounds=[(-2.8,2.5) if b else (-3,3) for b in diag],
             options={"maxiter":68,"maxfun":88,"ftol":1.e-11,"gtol":2.e-6,"maxls":12})
    if best[1] is None:
        return None
    L = best[1]
    model_cov = np.linalg.solve(L,np.eye(d))
    evidence = 0.0
    for j, f in enumerate(xs):
        z = f @ L[j] * np.sqrt(weights[j]/L[j,j])
        size = len(z)
        r2 = z*z
        skew = np.mean(z*r2)/(max(1e-8,np.mean(r2))**1.5)*np.sqrt(size/6.)
        kurt = (np.mean(r2*r2)/max(1e-8,np.mean(r2))**2-3.)*np.sqrt(size/24.)
        # A posterior with state-dependent conditional noise does not
        # get the full precision-based pooling.
        f = f.copy()
        f[:,j] = 0.0
        f -= f.mean(axis=0)
        r = np.sum(f*(f@L),axis=1)
        r -= np.mean(r)
        cor = (r@r2)/max(1e-10,np.linalg.norm(r)*np.linalg.norm(r2-np.mean(r2)))
        evidence += max(abs(skew),1.0)**2 + max(abs(kurt),1.0)**2 + min(250.,size*cor**2)
    mismatch = max(0.0,(evidence-3.6*d)/np.sqrt(7*d))
    strength = np.exp(-.32*mismatch)
    if schedule is None:
        D = np.diag((counts / counts.sum())*(1-effects)/np.diag(L))
        H = 2*model_cov @ np.linalg.solve(D,model_cov)-model_cov
    else:
        # Write one sweep as state and reward equations driven by the
        # per-coordinate conditional innovations.
        p = len(schedule)
        state = np.eye(d)
        noise = np.zeros((d,p))
        reward = np.zeros((d,d))
        reward_noise = np.zeros((d,p))
        for k,j in enumerate(schedule):
            B = np.eye(d)
            B[j] -= (1-effects[j])*L[j]/L[j,j]
            state = B @ state
            noise = B @ noise
            noise[j,k] += np.sqrt((1-effects[j]**2)/L[j,j])
            reward += state
            reward_noise += noise
        z = reward @ np.linalg.solve(np.eye(d)-state,noise) + reward_noise
        H = z @ z.T / p
    return sym(v.T @ H @ v), strength, evidence


def coordinate_covariance(x, dx, diagnostics=None):
    cand = directions(dx)
    if cand is None:
        return None
    v, lab = cand
    n,d = x.shape
    prefix = lab[:min(2048,len(lab))]
    if np.all(prefix>=0):
        for p in range(d,min(3*d,len(prefix)//5)+1):
            if np.mean(prefix[p:]==prefix[:-p]) > .9995 and np.mean(lab[p:]==lab[:-p]) > .998:
                n0 = n//p
                reduced = x[:n0*p].reshape(n0,p,d).mean(axis=1)
                if diagnostics is not None:
                    diagnostics["coordinate_period"] = p
                candidate = fit(x,v,lab,prefix[:p])
                return "periodic", p, (reduced, candidate)
    if not lag_check(lab,n,d):
        return None
    candidate = fit(x,v,lab)
    if candidate is None:
        return None
    h, strength, evidence = candidate
    if diagnostics is not None:
        diagnostics["coordinate_evidence"] = float(evidence)
    return "scan", float(strength), h
