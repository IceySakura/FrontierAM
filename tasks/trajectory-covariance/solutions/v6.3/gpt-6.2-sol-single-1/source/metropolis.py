"""A conditional Gaussian random-walk model, with goodness-of-fit checks."""

from functools import lru_cache
from time import monotonic

import numpy as np
from numpy.polynomial.legendre import leggauss

from holding import episodes


class BudgetReached(Exception):
    pass


def _sym(a):
    return (a + a.T) * 0.5


@lru_cache(maxsize=8)
def quadrature(n):
    x, w = leggauss(n)
    phi = (1+x)*np.pi/4
    return phi, w*np.pi/4


def joint_move_moments(b):
    theta, w = quadrature(36)
    v = 1 + b[None, :] / (4*np.sin(theta[:, None])**2)
    base = w * np.exp(-0.5 * np.log(v).sum(axis=1)) / np.pi
    return 2*base.sum(), (base[:, None]*b[None, :]/v).sum(axis=0)


def initial_proposal(dx, accept):
    c = _sym(dx.T @ dx / len(dx))
    b, u = np.linalg.eigh(c)
    b = np.maximum(b, 1e-5*b[-1])
    for _ in range(4):
        lo, hi = 0.05, 14.0
        for _ in range(16):
            mid = np.sqrt(lo*hi)
            p, _ = joint_move_moments(b*mid)
            if p > accept:
                lo = mid
            else:
                hi = mid
        b *= mid
        _, e = joint_move_moments(b)
        v = np.diag(u.T @ c @ u)
        b *= np.clip(v * accept/(2*e), .5, 2.0)**0.57
    s = _sym((u*b)@u.T)
    return np.linalg.cholesky(s)


def integration_parts(xf, b, count):
    """Bromwich integral of E[(1-exp(-V(y)+V(x)))_+]."""
    phi, w = quadrature(count)
    size = max(0.08, np.max(b), np.sqrt(np.mean(np.sum(xf*xf,axis=1)))*0.80)
    z = (0.36 + 1j*np.tan(phi)) / size
    cz = 1 / (1 - z[:,None]*b)
    v = 0.5 * z[:,None]**2 * cz
    cent = np.log(cz).sum(axis=1)*0.5
    ew = w / (np.pi * size * np.cos(phi)**2) / (z*(1+z))
    k = np.exp((xf*xf) @ v.T + cent) * ew
    return k, z, cz, v


def fit_rwm(x, max_iter=28, cap=5000):
    from scipy.optimize import minimize

    n, d = x.shape
    deadline = monotonic()+1.65
    y, t = episodes(x)
    m = len(y)
    q = (m-1)/(n-1)
    if d < 2 or m < max(64, 11*d) or not (.024 < q < .954):
        return None
    dl = y[1:] - y[:-1]
    md = 0.5*(y[1:] + y[:-1])
    sumd = dl.T @ dl
    f0 = initial_proposal(dl, q)
    if np.linalg.cond(f0) > 160.0:
        return None
    failures = t-1
    mask = failures > 0
    rejected = y[mask]
    counts = failures[mask]
    if len(counts) > cap:
        # Stratified deterministic integration over the dwell mass.
        inds = np.searchsorted(np.cumsum(counts), (np.arange(cap)+0.5)*counts.sum()/cap)
        sel, mult = np.unique(inds, return_counts=True)
        counts = mult.astype(float)*(counts.sum()/cap)
        rejected = rejected[sel]
    i, j = np.tril_indices(d)
    diag = i == j
    k = len(i)
    size = n-1
    theta0 = np.zeros(2*k+d)
    pscale = float(np.mean(np.diag(sumd))/size)
    strength = min(0.015, 0.033*pscale)
    penalty = np.r_[np.full(k, strength),np.full(k, strength*0.04),np.full(d,strength)]
    dlog = 2*np.log(np.diag(f0)).sum()
    off_i, off_j = np.triu_indices(d)
    best = [np.inf, None]

    def unpack(a):
        g = np.zeros((d,d))
        h = np.zeros((d,d))
        g[i,j] = a[:k]
        h[i,j] = a[k:2*k]
        g[i[diag], j[diag]] = np.exp(a[:k][diag])
        h[i[diag], j[diag]] = np.exp(a[k:2*k][diag])
        F = f0 @ h
        L = g @ g.T
        mu = a[2*k:]
        return g, h, F, L, mu

    def obj(a):
        if monotonic() > deadline:
            raise BudgetReached
        g, h, F, L, mu = unpack(a)
        B = _sym(F.T @ L @ F)
        b, u = np.linalg.eigh(B)
        ir = np.linalg.solve(F, np.eye(d))
        Q = ir.T @ ir
        X = rejected - mu
        Xf = X @ L @ F @ u
        integ, z, cz, v = integration_parts(Xf, b, 32 if d > 3 else 40)
        prob = integ.real.sum(axis=1)
        if not np.isfinite(prob).all() or np.min(prob) < 1e-9:
            return 1e4, np.zeros_like(a)
        prob = np.minimum(prob, 1.0)
        logsen = np.einsum("ij,ij->i", (md-mu)@L, dl)
        active = logsen > 0
        loss = (0.5*np.sum(Q*sumd) + 0.5*(m-1)*(dlog + 2*a[k:2*k][diag].sum())
                + np.sum(logsen[active]) - np.dot(counts, np.log(prob))) / size
        loss += 0.5*np.dot(penalty,a*a)
        if loss < best[0]:
            best[:] = [loss, a.copy()]
        kk = (counts/prob)[:,None] * integ
        grad_x = 2 * Xf * (kk @ v).real
        grad_x = grad_x @ u.T
        grad_b = np.diag((kk.sum(axis=0) @ (z[:,None]*cz/2)).real)
        prod = (Xf[:,off_i]*Xf[:,off_j]).T @ kk
        grad_b[off_i, off_j] += np.sum((prod.T * (z[:,None]**3/2) *
                                       cz[:,off_i] * cz[:,off_j]).real, axis=0)
        grad_b[off_j, off_i] = grad_b[off_i, off_j]
        grad_b = _sym(u @ grad_b @ u.T)
        xy = X.T @ grad_x
        mat = (md[active]-mu).T @ dl[active]
        g_l = _sym(xy @ F.T - mat) + F @ grad_b @ F.T
        g_f = L @ xy + 2 * L @ F @ grad_b + (Q @ sumd @ Q - (m-1)*Q) @ F
        g_mu = L @ (dl[active].sum(axis=0) - F @ grad_x.sum(axis=0))
        g_1 = (2*g_l @ g)[i,j]
        g_2 = (f0.T @ g_f)[i,j]
        g_1[diag] *= g[i[diag], j[diag]]
        g_2[diag] *= h[i[diag], j[diag]]
        grad = -np.r_[g_1,g_2,g_mu]/size + penalty*a
        return loss, grad

    bounds = [(-0.77, 0.73) if item else (-0.87,0.87) for item in diag]
    bounds += [(-0.65, 0.65) if item else (-0.72,0.72) for item in diag]
    bounds += [(-1.1,1.1)]*d
    try:
        minimize(obj, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter":max_iter, "maxfun":max_iter+12,
                          "ftol":7e-9, "gtol":0.00010, "maxcor":8, "maxls":9})
    except BudgetReached:
        pass
    if best[1] is None:
        return None
    g, h, F, L, mu = unpack(best[1])
    return F, L, mu, y, t


def validate(fit):
    """Test predictable duration and accepted-move residual moments."""
    F, L, mu, y, t = fit
    n = t.sum()
    m, d = y.shape
    b, u = np.linalg.eigh(_sym(F.T @ L @ F))
    adj = F @ u
    # Validation uses all the observed episodes. Chunk to bound memory.
    reject, zmean, zsize, zfour = [], [], [], []
    for x in np.array_split(y-mu, max(1, m//3200 + 1)):
        xf = x @ L @ adj
        k, z, cz, _ = integration_parts(xf, b, 40)
        r = np.clip(k.real.sum(axis=1), 1e-7, 1-1e-7)
        q = 1-r
        zz = z[:,None] * cz
        amean = -xf * (k @ zz).real / q[:,None]
        meansq = (xf*xf) @ (zz**2).T
        r2 = meansq + cz.sum(axis=1)
        r4 = r2*r2 + 2*np.sum(cz*cz,axis=1) + 4*(xf*xf) @ (zz**2 * cz).T
        second = (d - (k*r2).real.sum(axis=1)) / q
        fourth = (d*(d+2) - (k*r4).real.sum(axis=1)) / q
        reject.append(r)
        zmean.append(amean)
        zsize.append(second)
        zfour.append(fourth)
    r = np.concatenate(reject)
    mz = np.concatenate(zmean)[:-1]
    q = 1-r
    size = np.concatenate(zsize)[:-1]
    fourth = np.concatenate(zfour)[:-1]
    z = np.linalg.solve(adj, (y[1:]-y[:-1]).T).T
    f = y - mu
    radius = np.einsum("ij,ij->i", f @ L, f)
    p = np.minimum(np.sqrt(radius/d), 3.5)
    scalars = np.c_[np.ones(m), p, p*p, p**3, p**4]
    small = np.c_[np.ones(m), f, f * np.log1p(p)[:,None]]
    pre = np.c_[scalars, small[:, 1:]]
    dur = t-1
    trials = t.copy()
    trials[-1] -= 1
    ne = dur - trials*r
    W = trials*r*q
    drive = pre.T @ ne
    M = pre.T @ (W[:,None]*pre)
    v, u = np.linalg.eigh(_sym(M))
    score = u.T @ drive / np.sqrt(np.maximum(v, v[-1]*1e-6))
    dur_stat = score @ score
    dfs = pre.shape[1] # fitted parameters make this a conservative bound
    norms = np.sum(z*z,axis=1)
    vecs = np.c_[norms-size, norms*norms-fourth]
    hit = 0.0
    for c in vecs.T:
        P = scalars[:-1,:4]
        if not np.isfinite(c).all():
            return 0.0, {}
        score = P.T @ c
        xx = P.T @ ((c*c)[:,None]*P)
        hit += score @ np.linalg.solve(xx + np.eye(len(score))*np.trace(xx)*1e-8,score)
    hu = z-mz
    P = small[:-1]
    v, u = np.linalg.eigh(_sym(P.T @ P))
    P = P @ (u / np.sqrt(np.maximum(v, v[-1]*1e-9)))
    av = np.maximum(np.mean(hu*hu,axis=0), 1e-6)
    vecstat = np.sum((P.T @ hu)**2 / av)
    dfv = P.shape[1]*d
    # Stationary moments implied by the candidate. The marginal check
    # allows considerable sampling variation when relaxation is slow.
    qmodel, dm = joint_move_moments(b)
    direct = (y[1:]-y[:-1]).T @ (y[1:]-y[:-1])/(2*(n-1))
    transform = adj / np.sqrt(b)
    modelD = _sym((transform*dm) @ transform.T)
    invJ = np.linalg.solve(np.linalg.cholesky(direct), np.eye(d))
    change = np.linalg.eigvalsh(_sym(invJ @ modelD @ invJ.T))
    dspread = np.linalg.norm(np.log(np.maximum(1e-8, change)))
    tolerance = (0.10 + 2.8/np.sqrt(max(1, m))) * np.sqrt(d)
    tau = 1 / max(1e-5, np.linalg.eigvalsh(direct)[0])
    xstation = (y - (t @ y/t.sum()))
    # Mardia's radial kurtosis in empirical covariance units.
    norm = np.einsum("ij,ij->i", xstation, xstation)
    kurt = (t @ (norm**2)) / (t.sum()*d*(d+2))
    stationary_shape = max(0.0, kurt - 1.13, 0.90 - kurt)
    score = (.17*max(0, (dur_stat - 1.32*dfs) / np.sqrt(dfs)) +
             .32*max(0, (hit - 11.0) / np.sqrt(8)) +
             .25*max(0, (vecstat - 1.42*dfv) / np.sqrt(dfv)) +
             0.80*max(0, dspread/tolerance-1) +
             2.5*stationary_shape)
    accept_mismatch = abs(qmodel - (m-1)/(n-1)) / max(.013, 0.50*np.sqrt(tau/n))
    score += .38*max(0, accept_mismatch-1)
    evidence = np.exp(-score**1.25)
    return evidence, dict(rwm_duration=float(dur_stat), rwm_jump=float(vecstat),
                          rwm_spread=float(dspread), rwm_q=float(qmodel),
                          rwm_evidence=float(evidence), rwm_radial=float(hit),
                          rwm_model_diff=modelD)


def metropolis_covariance(x, diagnostics=None):
    fit = fit_rwm(x)
    if fit is None:
        return None, 0.0
    F, L, mu, y, t = fit
    evidence, dg = validate(fit)
    if diagnostics is not None:
        diagnostics.update(dg)
    if evidence < .035:
        return None, evidence
    b, u = np.linalg.eigh(_sym(F.T @ L @ F))
    V = (F @ u)/np.sqrt(b)
    from normal_flow import covariance_factor
    eig = covariance_factor(b,degree=3)
    if not np.isfinite(eig).all() or eig.min() <= 0:
        return None, 0.0
    h = (V * eig) @ V.T
    return _sym(h), evidence


def radial_step_fit(y, t, c_init, b_init):
    from scipy.optimize import minimize

    m,d = y.shape
    n = t.sum()-1
    r0 = np.sum(y*y,axis=1)
    rr = r0/c_init
    delta = np.diff(y,axis=0)
    step_energy = .5*np.sum(delta*delta)/n
    uphill = .5*np.sum(np.maximum(np.diff(r0),0))/n
    w = t-1
    vmax = max(1e-5,np.max(rr))
    bin_idx = np.clip((np.sqrt(rr / vmax)*118).astype(int),0,117)
    mass = np.bincount(bin_idx, weights=w, minlength=118)
    rad = np.bincount(bin_idx, weights=w*rr, minlength=118)
    rr = rad[mass>0]/mass[mass>0]
    mass = mass[mass>0]/n
    phi, qw = quadrature(38)
    nominal = max(.08,b_init,b_init*np.sqrt(d)*.8)
    z = (.33+1j*np.tan(phi))/nominal
    # Choose a fixed integration contour that remains inside the
    # moment-generating function domain throughout the fit.
    z[...]=np.minimum(z.real,.54/(b_init*np.exp(1.10)))+1j*z.imag
    qw = qw/(np.pi*nominal*np.cos(phi)**2)/(z*(1+z))
    s_init = c_init*b_init
    move = (m-1)/n
    scale_prior = .00009+.0008*min(1.,step_energy/s_init)

    def obj(logst):
        a,c = logst
        si = np.exp(-a)/s_init
        ci = np.exp(-c)/c_init
        b = b_init*np.exp(a-c)
        a2 = rr*b_init*np.exp(a-2*c)
        cz = 1/(1-z*b)
        k = qw * np.exp(.5*d*np.log(cz) + a2[:,None]*(.5*z*z*cz)[None,:])
        r = k.real.sum(axis=1)
        if r.min() < 1e-7 or not np.isfinite(r).all():
            return 1e4, np.zeros(2)
        wb = .5*z*cz
        wa = .5*z*z*cz
        jr = np.dot(mass/r, ((k@wb).real*d+a2*(k@(z*cz*wa)).real))
        ja = np.dot(mass/r,a2*(k@wa).real)
        loss = .5*move*d*a + si*step_energy + ci*uphill - mass@np.log(r)
        loss += .5*scale_prior*(c*c+.15*a*a)
        g = np.array([.5*move*d-si*step_energy - b*jr-ja, -ci*uphill+b*jr+2*ja])
        g += scale_prior*np.array([.15*a,c])
        return loss,g

    result = minimize(obj,np.zeros(2),jac=True,method="L-BFGS-B",
                      bounds=[(-.45,.45),(-.65,.65)],
                      options={"maxiter":24,"maxfun":32,"ftol":2e-10,"gtol":3e-6})
    a,c = result.x
    return c_init*np.exp(c), b_init*np.exp(a-c)


def common_step_covariance(x, direct, diagnostics=None, use_radial=True):
    """Method of moments for an elliptically scaled Gaussian random walk."""
    y, t = episodes(x)
    n, d = x.shape
    m = len(y)
    q_obs = (m-1)/(n-1)
    if m < max(70, 12*d) or not (.030 < q_obs < .94):
        return None, 0.0
    val, vec = np.linalg.eigh(_sym(direct))
    lval = np.log(np.maximum(1e-9, val))
    spread = np.sum((lval-lval.mean())**2)
    rates = val[:,None]
    corr = (1-rates)*(1-rates.T)
    var = (1+corr)/(1-corr) * (1+np.eye(d)) / n
    expected = var.sum() - np.trace(var)/d
    significance = spread / max(expected, 1e-10)
    if significance > 3.2:
        return None, 0.0
    lo, hi = 1e-5, 40.0
    for _ in range(30):
        mid = np.sqrt(lo*hi)
        b = np.full(d, mid)
        q, _ = joint_move_moments(b)
        if q > q_obs:
            lo = mid
        else:
            hi = mid
    _, dj = joint_move_moments(b)
    C = direct / dj[0]
    cv = val/dj[0]
    if cv[-1] > 2.65 or cv[0] < .38:
        return None, 0.0
    mu = t @ y / n
    if use_radial and .045 < q_obs < .94:
        zz = (y-mu) @ (vec/np.sqrt(val))
        target_c, target_b = radial_step_fit(zz,t,1/dj[0],mid)
        # Retain some information from the time-stationary balance of
        # accepted moves. Shape remains set by the observed increments.
        c0 = (1/dj[0]) * np.exp(.75*np.log(target_c*dj[0]))
        mid *= np.exp(.75*np.log(target_b/mid))
        C = c0*direct
        cv = c0*val
        b = np.full(d,mid)
    L = (vec/cv)@vec.T
    F = (vec*np.sqrt(mid*cv))@vec.T
    evidence, dg = validate((F,L,mu,y,t))
    evidence *= np.exp(-.65*max(0.0, significance-1.0)**1.42)
    # The observed covariance also guards against a wrong overall scale,
    # independent of the evidence for isotropic relaxation.
    trace_sd = np.sqrt(2.0*np.trace(var))/d
    center = np.log(cv).mean()
    evidence *= np.exp(-0.65*max(0., abs(center)/max(.025, trace_sd)-1.25)**1.35)
    if diagnostics is not None:
        diagnostics.update(dg)
        diagnostics.update(rwm_evidence=evidence, rwm_homogeneity=significance, rwm_scale=center)
    if evidence < .04:
        return None, evidence
    from normal_flow import covariance_factor
    h = covariance_factor(b, degree=5 if q_obs < .8 else 3, isotropic=True)[0] * C
    return _sym(h), evidence
