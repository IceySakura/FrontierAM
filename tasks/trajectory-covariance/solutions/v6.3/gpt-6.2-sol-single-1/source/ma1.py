"""Sine-basis Gaussian likelihood for a one-dependent covariance."""

import numpy as np


def sym(a):
    return (a+a.T)*0.5


def floor_cov(a, floor):
    w, v = np.linalg.eigh(sym(a))
    return (v*np.maximum(w,floor))@v.T


def estimate_ma1(x, gamma1):
    from scipy.fft import dst
    from scipy.optimize import minimize

    n, d = x.shape
    k = np.arange(1,n+1,dtype=float)
    a = np.cos((np.pi/(2*(n+1)))*k)**2
    y = dst(x, type=1, axis=0, norm="ortho", workers=1)
    mean = np.zeros(n)
    j = k[::2]
    mean[::2] = np.sqrt(2/(n+1)) / np.tan((np.pi/(2*(n+1)))*j)
    # Keep the frequencies closest to 0 and pi separate, then group
    # frequencies whose endpoint weights differ little.
    upper = n//2
    lower = min(34,max(2,upper))
    edge = np.unique(np.r_[np.arange(lower+1), np.geomspace(lower,upper,100).astype(int)])
    edge = np.unique(np.r_[edge,n-edge])
    count = np.diff(edge)
    m1, m2, sm, freq = [], [], [], []
    for i,j in zip(edge[:-1],edge[1:]):
        f = y[i:j]
        b = mean[i:j]
        m1.append(b @ f)
        m2.append(f.T @ f)
        sm.append(b@b)
        freq.append(a[i:j].mean())
    b = np.array(m1)
    S = np.array(m2)
    s2 = np.array(sm)
    frac = np.array(freq)
    mix = np.array([frac,1-frac])
    I = np.eye(d)

    a0 = I+2*sym(gamma1)
    a1 = I-2*sym(gamma1)
    # Approximate the two endpoint levels from broad local bands before
    # optimizing the exact marginal spectra. These are only initial values.
    init = []
    for end, weight, guess, other in ((0,a,a0,a1), (1,1-a,a1,a0)):
        use = weight > .994
        z = y[use]
        off = np.mean(1-weight[use])
        h = sym((z.T @ z / len(z) - off*other) /(1-off))
        v, u = np.linalg.eigh(guess)
        eps = max(.00008,3.5/n)
        out = v.copy()
        proj = np.diag(u.T @ h @ u)
        low = v < .20
        out[low] = np.maximum(eps, proj[low])
        out[~low] = np.maximum(eps, .55*v[~low]+.45*proj[~low])
        init.append(np.linalg.cholesky(floor_cov((u*out)@u.T,eps)))
    init = np.array(init)
    ii,jj = np.tril_indices(d)
    diag = ii==jj
    size = len(ii)
    best = [np.inf,None]

    def objective(theta):
        p = theta.reshape(2,size)
        L = np.zeros((2,d,d))
        L[:,ii,jj] = p
        L[:,ii[diag],jj[diag]] = np.exp(p[:,diag])
        F = init @ L
        ends = F @ F.transpose(0,2,1)
        cov = (mix.T @ ends.reshape(2,d*d)).reshape(-1,d,d)
        sign, logdet = np.linalg.slogdet(cov)
        if np.any(sign <= 0) or not np.isfinite(logdet).all():
            return 1e6, np.zeros_like(theta)
        inv = np.linalg.solve(cov, np.broadcast_to(I, cov.shape))
        j0 = np.einsum("ijk,i->jk",inv,s2)
        ji = np.linalg.solve(j0, I)
        center = ji @ np.einsum("ijk,ik->j",inv,b)
        sub = b[:,:,None]*center[None,None,:]
        E = S - sub - sub.transpose(0,2,1) + s2[:,None,None]*np.outer(center,center)
        loss = (count@logdet + np.sum(inv*E) + np.linalg.slogdet(j0)[1]) / n
        if loss < best[0]:
            best[:] = [loss, ends.copy()]
        R = count[:,None,None]*inv - inv @ (E + s2[:,None,None]*ji) @ inv
        G = (mix @ R.reshape(-1,d*d)).reshape(2,d,d)
        g = (2*(init.transpose(0,2,1) @ G @ F)/n)[:,ii,jj]
        g[:,diag] *= np.exp(p[:,diag])
        return loss, g.ravel()

    bounds = [(-7.5,4.0) if s else (-10,10) for s in diag]*2
    minimize(objective, np.zeros(2*size), jac=True, method="L-BFGS-B", bounds=bounds,
             options={"maxfun":109, "maxiter":90, "maxls":12, "ftol":6e-11,
                      "gtol":5e-7, "maxcor":9})
    if best[1] is None:
        return a0
    zero, pi = best[1]
    l = np.linalg.cholesky(zero)
    il = np.linalg.solve(l,I)
    k, q = np.linalg.eigh(sym(il @ pi @ il.T))
    spread = np.sqrt(np.maximum(0., k))
    # The zero-frequency level is poorly identified near a spectral
    # zero: its information is proportional to the number of Fourier
    # modes whose frequency is of order sqrt(H(0)/H(pi)).
    noise = ((d+2)*spread + spread.sum())/n
    adjust = 1.0 / (1.0 + .50*np.minimum(1.6,noise))
    r = l @ q
    return sym((r*adjust) @ r.T)
