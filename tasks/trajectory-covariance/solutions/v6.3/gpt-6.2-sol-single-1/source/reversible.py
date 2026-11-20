"""Energy-form regression for reversible transition operators."""

import numpy as np


def _sym(a):
    return (a + a.T) / 2


def _sqrt(a, inverse=False):
    v, q = np.linalg.eigh(_sym(a))
    v = np.maximum(v, max(1e-30, v[-1] * 1e-10))
    return (q * v**(-0.5 if inverse else 0.5)) @ q.T


def _n2(a):
    return np.sum(a * a)


def feature_groups(x, vals, vect, more=True):
    d = x.shape[1]
    y = x @ vect
    r2 = np.sum(y*y, axis=1)
    cap = max(8, np.quantile(r2, 0.997))
    radius = np.minimum(r2, cap)
    z = y * np.minimum(1.0, np.sqrt(cap / np.maximum(1.0, r2)))[:, None]
    i, j = np.triu_indices(d)
    result = [y * (radius - d)[:, None], z[:, i] * z[:, j], z**3]
    if more and d >= 3:
        # One more smooth radial level; orthogonalization removes the
        # information already captured by the first radial basis.
        result.append(y * np.log1p(radius / d)[:, None])
        # Summed anisotropic radii, without a full cubic tensor.
        lev = np.log(vals)
        lev -= lev.mean()
        if np.std(lev) > 0.35:
            lev /= max(1.0, np.max(np.abs(lev)))
            result.append(z * ((z*z) @ lev)[:, None])
    return result


def covariance_rates(n, dvals, strength):
    """Shrink noisy marginal shape toward a common relaxation-rate model."""
    d = len(dvals)
    if d < 2 or strength <= 0 or dvals[-1] > 1.33 or dvals[0] < 2e-9:
        return np.ones(d), 0.0
    log_rates = np.log(dvals)
    deviation = log_rates - log_rates.mean()
    rates = dvals[:, None]
    dot = (1 - rates)*(1 - rates.T)
    vary = (1 + dot) / (1 - dot) / n
    vary *= 1 + np.eye(d)
    spread = deviation[:, None] - deviation[None, :]
    jac = np.ones((d, d))
    neq = np.abs(spread) > 1e-6
    jac[neq] = (np.sqrt(rates * rates.T)*spread)[neq] / (rates - rates.T)[neq]
    vary *= jac**2
    variance = vary.sum() - np.trace(vary) / d
    ss = _n2(deviation)
    weight = strength * variance / max(ss, 1e-12)
    std = np.sqrt(ss / d)
    if std > 0.42:
        weight *= (0.42 / std)**0.7
    weight = min(0.985, weight)
    ratio = np.exp(weight * deviation)
    return ratio, weight


def semigroup(x, power=1.0, extra=True, diagnostics=None, directional=0.0):
    n, d = x.shape
    nn = n-1
    eye = np.eye(d)
    dx = np.diff(x, axis=0)
    dform = dx.T @ dx / (2*nn)
    dvals, dvecs = np.linalg.eigh(_sym(dform))
    if dvals[0] <= 1e-12 or dvals[-1] >= 2.0001:
        return None
    t = (dvecs / dvals) @ dvecs.T
    base = _sym(2*t - eye)
    gain = np.zeros((d, d))

    # Blocks are used only to estimate the noise in the fitted moments.
    # They do not supply a separate batch-means estimate of the target.
    b = min(28, max(8, n // 250))
    edges = np.linspace(0, nn, b + 1, dtype=int)
    blocks = [slice(i, j) for i, j in zip(edges[:-1], edges[1:])]
    lens = np.diff(edges)
    amount = 0.0
    other = feature_groups(x, dvals, dvecs, more=extra)
    w = np.concatenate(other, axis=1)
    w -= w.mean(axis=0)
    w -= x @ (x.T @ w / n)
    w /= np.maximum(np.sqrt(np.mean(w*w, axis=0)), 1e-12)
    dw = np.diff(w, axis=0)
    target = w.shape[1]
    wx = dw.T @ dx / (2*nn)
    ww = dw.T @ dw / (2*nn)
    cross = wx @ t
    # Residualize in the increment inner product, so the linear
    # features retain their fitted equation after nonlinear shrinkage.
    ww = _sym(ww - cross @ wx.T)
    moments = []
    for sl, l in zip(blocks,lens):
        moment = (dw[sl].T @ dx[sl]) / (2*l)
        moment -= (w[sl].T @ x[sl] / l) @ dform
        moment -= cross @ (dx[sl].T @ dx[sl] / (2*l) - dform)
        moments.append(moment)
    moments = np.asarray(moments)
    wh = _sqrt(base, inverse=True)
    hh = _sqrt(base)
    a0 = 0
    for j, a1 in enumerate(np.cumsum([a.shape[1] for a in other])):
        sl = slice(a0, a1)
        st = _sqrt(ww[sl, sl], inverse=True)
        b1 = st @ wx[sl] @ t
        v = b1 @ wh
        m = st @ (moments[:, sl] - moments[:, sl].mean(axis=0)) @ t @ wh
        noise = _n2(m) / (b*(b-1))
        ss = _n2(v)
        raw = power * max(0.0, 1.0 - (1.35 if j <= 1 else 1.6) * noise / max(ss, 1e-20))
        if raw:
            small_gain = raw*(b1.T @ b1)
            if directional:
                v0 = m.reshape(-1,d)
                noise_h = (v0.T @ v0)/(b*(b-1))
                g0 = _sym(v.T @ v - (1.62 if j <= 1 else 1.78)*noise_h)
                lv, uv = np.linalg.eigh(g0)
                lv = np.clip(lv, 0.0, 0.56)
                denom = max(1e-12,lv.sum())
                gg = (uv * (lv*min(power,raw*ss/denom))) @ uv.T
                small_gain = (1-directional)*small_gain + directional*hh @ gg @ hh
            gain += 2*small_gain
            amount += np.sqrt(raw*ss)
        if a1 < target:
            c = (ww[a1:, sl] @ st) @ st
            wx[a1:] -= c @ wx[sl]
            moments[:, a1:] -= c @ moments[:, sl]
            ww[a1:, a1:] -= c @ ww[sl, a1:]
        a0 = a1
    h = _sym(base + gain)
    if diagnostics is not None:
        diagnostics["poisson"] = amount
    return h, amount
