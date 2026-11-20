"""Conditional holding times for trajectories with exact repeated rows."""

import numpy as np


def _sym(a):
    return (a + a.T) * 0.5


def episodes(x):
    first = np.r_[0, np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1]
    lengths = np.diff(np.r_[first, len(x)]).astype(float)
    return x[first], lengths


def holding_fit(x, t, more=True):
    m, d = x.shape
    if m < max(30, 5*d) or t.sum() - m < 12:
        return np.full(m, np.mean(t)), 0.0, 0.0
    # Normalize by the jump distribution rather than the time distribution.
    y = x - x.mean(axis=0)
    lv, u = np.linalg.eigh(_sym(y.T @ y / m))
    lv = np.maximum(lv, max(1e-20, lv[-1] * 1e-9))
    y = y @ (u / np.sqrt(lv))
    r2 = np.sum(y*y, axis=1)
    cap = max(4, np.quantile(r2, 0.993))
    z = y * np.minimum(1.0, np.sqrt(cap / np.maximum(1, r2)))[:, None]
    r = np.sum(z*z, axis=1)
    a = np.log1p(r/d)
    ii, jj = np.triu_indices(d)
    groups = [a[:,None], a[:,None]**2, z, z[:,ii]*z[:,jj]]
    if more:
        groups += [z * a[:,None], z[:,ii]*z[:,jj]*a[:,None]]
    X = np.concatenate([np.ones((m,1)), *groups], axis=1)
    # A geometric episode ends in a success, except the final
    # right-censored episode.
    s = t.copy()
    s[-1] -= 1
    f = t - 1
    fail = min(1 - 1e-6, max(1e-6, f.sum() / s.sum()))
    intercept = np.log(fail/(1-fail))
    # Build approximate independence within candidate blocks using the
    # Fisher metric of the intercept-only fit.
    k = X.shape[1]
    Xr = np.empty_like(X)
    Xr[:,0] = 1
    better = 0.0
    prec = np.zeros(k)
    j0 = 1
    src = 1
    for j, group in enumerate(groups):
        src1 = src + group.shape[1]
        b = X[:,src:src1]
        src = src1
        if j0 > 0:
            prev = Xr[:,:j0]
            b = b - prev @ (prev.T @ (s[:,None]*b) / s.sum())
        val, vec = np.linalg.eigh(_sym(b.T @ (s[:, None]*b)/s.sum()))
        keep = val > max(1e-12, val[-1]*1e-8)
        bv = (b @ (vec[:,keep]/np.sqrt(val[keep])))
        score = bv.T @ (f - fail*s)
        var0 = fail*(1-fail)*s.sum()
        stat = float(score @ score / var0)
        df = len(score)
        amount = max(0.0, 1 - (1.10 if j < 3 else 1.22)*df/max(stat,1e-20))
        if df and amount > 0:
            cols = slice(j0, j0+df)
            Xr[:,cols] = bv
            prec[cols] = var0*((1-amount)/max(0.01,amount)+0.03)
            j0 += df
            better += stat-df
    if j0 == 1:
        return np.full(m, s.sum() / (m-1)), 0.0, 0.0
    Xr = Xr[:,:j0].copy()
    pr = prec[:j0]
    coef = np.r_[intercept, np.zeros(j0-1)]

    def loss(eta, theta):
        return np.dot(s, np.logaddexp(0.0, eta)) - np.dot(f,eta) + 0.5*np.dot(pr,theta*theta)

    eta = Xr @ coef
    obj = loss(eta, coef)
    for _ in range(12):
        p = 0.5*(1+np.tanh(0.5*eta))
        grad = Xr.T @ (s*p-f) + pr*coef
        h = Xr.T @ ((s*p*(1-p))[:,None]*Xr) + np.diag(pr+1e-10)
        step = np.linalg.solve(h, grad)
        if np.max(np.abs(step)) < 2e-6:
            break
        for j in range(8):
            c = coef - (0.5**j)*step
            v = Xr @ c
            a = loss(v, c)
            if a <= obj:
                obj, coef, eta = a, c, v
                break
        else:
            break
    w = 1 + np.exp(np.clip(eta, -16, np.log(max(3, 2.8*np.max(t)))))
    w *= t.sum() / np.sum(w)
    return w, better, float(np.std(np.log(w)))


def weighted_covariance(x, amount=0.8, diagnostics=None):
    """Use expected holding times to de-noise stationary feature moments."""
    y, t = episodes(x)
    w, evidence, hetero = holding_fit(y, t)
    n = len(x)
    eye = np.eye(x.shape[1])
    v = y - (w @ y / w.sum())
    c = v.T @ (w[:,None]*v) / w.sum()
    lam = np.linalg.eigvalsh(_sym(c))
    rel = float(np.max(abs(lam-1)))
    factor = amount * min(1.0, 0.40/max(rel, 1e-8))
    # Very long episodes can be due to unresolved state variation.
    # Do not let a simple conditional model fully replace those moments.
    mx = np.max(t) / n
    if mx > .022:
        factor *= (0.022/mx)**0.6
    c = eye + factor*(c-eye)
    if diagnostics is not None:
        diagnostics.update(hold_fit=evidence, hold_hetero=hetero, hold_weight=factor)
    return _sym(c)


def regenerative_covariance(x, diagnostics=None):
    """A low-variance estimate when the embedded chain is close to independent."""
    y, t = episodes(x)
    m,d = y.shape
    mt = t.mean()
    if not (mt > 1.4 and m > 110):
        return None, 0.0
    mu = y.mean(axis=0)
    z = y-mu
    val,v = np.linalg.eigh(_sym(z.T @ z/m))
    if val[0] <= 1e-10 * val[-1]:
        return None, 0.0
    zz = z @ (v / np.sqrt(val))
    signal = [(m-k)*np.sum((zz[:-k].T @ zz[k:] / (m-k))**2) for k in (1,2,3)]
    df = d*d
    if signal[0] > df + 2.5*np.sqrt(2*df) or sum(signal) > 3*df+2.85*np.sqrt(6*df):
        return None, 0.0
    w, s, het = holding_fit(y,t,more=True)
    if np.max(w) > 0.055*t.sum():
        return None, 0.0
    y = y - w@ y / w.sum()
    h = y.T @ ((2*w*w-w)[:,None]*y)/t.sum()
    # Keep any detectable, short-memory part of the accepted-state
    # rewards, including negative dependence.
    r = w[:,None]*y
    u = r[1:].T @ r[:-1] / m
    k = np.trace(r.T @ r /m)/d
    raw = (m-1)*np.sum(_sym(u/k)**2)
    sh = max(0.,1.0-1.08*d*(d+1)*.5/max(1e-12,raw))
    h += 2*sh*_sym(u)/mt
    confidence = (0.88 if het < .23 else .65)*np.exp(-.34*max(0., (signal[0]-df)/np.sqrt(2*df)))
    if diagnostics is not None:
        diagnostics["regenerative_weight"] = float(confidence)
    return _sym(h), confidence
