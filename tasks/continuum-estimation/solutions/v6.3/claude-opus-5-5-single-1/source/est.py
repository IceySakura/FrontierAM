import numpy as np

R_OFF = 1.5  # third exponent is q + R_OFF

# Empirical-Bayes prior: flat over the ranges observed in the public smooth
# examples (min, max, count) with "uniform with unknown bounds" polynomial tails.
TAILS = dict(p=(0.657, 2.632, 8), d=(0.462, 1.848, 7), a=(0.289, 1.488, 7),
             b=(0.120, 0.868, 7), c=(0.052, 0.373, 7), t=(1.137, 2.888, 7))
DEFAULT = dict(
    p=(0.15, 4.0), d=(0.15, 3.5), tails=TAILS,
    hard=dict(t=(0.3, 6.0), a=(-4.0, 4.0), b=(-3.0, 0.0), c=(-2.0, 0.0)),
    soft=dict(p=0.02, d=0.02, t=0.02, a=0.02, b=0.005, c=0.003),
    hp=0.004, kappa=1.0,
)
# Used only when the main prior leaves no posterior mass (data outside the model).
BROAD = dict(
    p=(0.1, 6.0), d=(0.1, 5.0), t=(0.0, 1e3), a=(-1e3, 1e3), b=(-1e3, 1e3), c=(-1e3, 1e3),
    soft=dict(p=0.02, d=0.02, t=0.02, a=0.02, b=0.02, c=0.02),
    hp=0.01, kappa=1.0,
)


def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def box(v, lo, hi, s):
    return sig((v - lo) / s) * sig((hi - v) / s)


def cofactors(fixed_cols, n, pos):
    """fixed_cols: list of (Nq, n) arrays in column order with a hole at `pos`.
    Returns K (Nq, n) with det(M[:, pos]=u) = K @ u."""
    Nq = fixed_cols[0].shape[0]
    M = np.empty((Nq, n, n))
    j = 0
    for k in range(n):
        if k == pos:
            continue
        M[:, :, k] = fixed_cols[j]
        j += 1
    K = np.empty((Nq, n))
    for i in range(n):
        M[:, :, pos] = 0.0
        M[:, i, pos] = 1.0
        K[:, i] = np.linalg.det(M)
    return K


def wmedian(v, w):
    o = np.argsort(v)
    v = v[o]
    cw = np.cumsum(w[o])
    if cw[-1] <= 0 or not np.isfinite(cw[-1]):
        return None
    cw /= cw[-1]
    k = np.searchsorted(cw, 0.5)
    return float(v[min(k, len(v) - 1)])


def tail(v, m, M, n):
    R = M - m
    e = np.maximum(m - v, 0.0) + np.maximum(v - M, 0.0)
    return (R / (R + e)) ** n


def prior_lin(t, a, b, c, P):
    if 'tails' in P:
        T = P['tails']; s = P['soft']
        w = tail(np.abs(t), *T['t']) * tail(np.abs(a), *T['a']) * tail(-b, *T['b']) * tail(-c, *T['c'])
        if 'hard' in P:
            H = P['hard']
            w = w * box(b, H['b'][0], H['b'][1], s['b']) * box(c, H['c'][0], H['c'][1], s['c']) \
                  * box(a, H['a'][0], H['a'][1], s['a']) * box(np.abs(t), H['t'][0], H['t'][1], s['t'])
        return w
    s = P['soft']
    at = np.abs(t)
    return (box(at, P['t'][0], P['t'][1], s['t']) * box(a, P['a'][0], P['a'][1], s['a'])
            * box(b, P['b'][0], P['b'][1], s['b']) * box(c, P['c'][0], P['c'][1], s['c']))


def posterior_samples(x, y, P=DEFAULT, chunk=128):
    """Returns (t_values, weights) over a grid/curve representation of the posterior."""
    n = len(x)
    s = P['soft']
    h = P['hp']
    p_lo, p_hi = P['p'][0] - 4 * s['p'], P['p'][1] + 4 * s['p']
    d_lo, d_hi = P['d'][0] - 4 * s['d'], P['d'][1] + 4 * s['d']
    pg_all = np.arange(p_lo, p_hi + h / 2, h)
    qg = np.arange(p_lo + d_lo, p_hi + d_hi + h / 2, h)
    lx = np.log(x)
    Xq = np.exp(np.outer(qg, lx))          # (Nq, n)
    Xr = np.exp(np.outer(qg + R_OFF, lx))  # (Nq, n)
    one = np.ones_like(Xq)
    Y = np.broadcast_to(y, Xq.shape)
    if n == 4:
        Kdet = cofactors([one, Xq, Xr], 4, 1).T
        Kt = cofactors([Y, Xq, Xr], 4, 1).T
        Kb = cofactors([one, Y, Xr], 4, 1).T
        Kc = cofactors([one, Xq, Y], 4, 1).T
        M = np.empty((len(qg), 4, 4))
        M[:, :, 0] = 1.0; M[:, :, 1] = Y; M[:, :, 2] = Xq; M[:, :, 3] = Xr
        anum = np.linalg.det(M)[None, :]
    elif n == 5:
        K5 = cofactors([one, Xq, Xr, Y], 5, 1)
        dq = 1e-6
        K5p = cofactors([one, np.exp(np.outer(qg + dq, lx)), np.exp(np.outer(qg + dq + R_OFF, lx)), Y], 5, 1)
        K5m = cofactors([one, np.exp(np.outer(qg - dq, lx)), np.exp(np.outer(qg - dq + R_OFF, lx)), Y], 5, 1)
        K5d = ((K5p - K5m) / (2 * dq)).T
        K5 = K5.T
        kh = P['kappa'] * h
    else:
        raise ValueError
    ts, ws = [], []
    for i0 in range(0, len(pg_all), chunk):
        pg = pg_all[i0:i0 + chunk]
        Xp = np.exp(np.outer(pg, lx))      # (Np, n)
        PP, QQ = np.meshgrid(pg, qg, indexing='ij')
        DD = QQ - PP
        pri_nl = box(PP, P['p'][0], P['p'][1], s['p']) * box(DD, P['d'][0], P['d'][1], s['d'])
        if 'tails' in P:
            pri_nl = pri_nl * tail(PP, *P['tails']['p']) * tail(DD, *P['tails']['d'])
        with np.errstate(all='ignore'):
            if n == 4:
                inv = 1.0 / (Xp @ Kdet)
                t = (Xp @ Kt) * inv
                a = anum * inv
                b = (Xp @ Kb) * inv
                c = (Xp @ Kc) * inv
                w = pri_nl * prior_lin(t, a, b, c, P) * np.abs(inv)
                m = np.isfinite(w) & (w > 0) & np.isfinite(t)
                ts.append(t[m]); ws.append(w[m])
            else:
                D = Xp @ K5
                g = np.hypot(Xp @ K5d, (Xp * lx) @ K5)
                dist = np.abs(D) / g
                sel = (dist < 4 * kh) & (pri_nl > 1e-12)
                ii, jj = np.nonzero(sel)
                if len(ii) == 0:
                    continue
                A = np.empty((len(ii), 5, 4))
                A[:, :, 0] = 1.0
                A[:, :, 1] = Xp[ii]
                A[:, :, 2] = Xq[jj]
                A[:, :, 3] = Xr[jj]
                ok = np.all(np.isfinite(A), axis=(1, 2))
                A, ii, jj = A[ok], ii[ok], jj[ok]
                if len(ii) == 0:
                    continue
                beta = np.einsum('kij,j->ki', np.linalg.pinv(A), y)
                t, a, b, c = beta.T
                ker = np.exp(-0.5 * (dist[ii, jj] / kh) ** 2)
                w = pri_nl[ii, jj] * prior_lin(t, a, b, c, P) * ker / g[ii, jj]
                m = np.isfinite(w) & (w > 0) & np.isfinite(t)
                ts.append(t[m]); ws.append(w[m])
    if not ts:
        return np.array([]), np.array([])
    return np.concatenate(ts), np.concatenate(ws)


def estimate(x, y, P=DEFAULT):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(-x); x = x[o]; y = y[o]
    for prior in (P, BROAD):
        t, w = posterior_samples(x, y, prior)
        if len(t) > 0 and np.sum(w) > 0:
            w = w / np.maximum(1.0, np.abs(t))
            z = wmedian(t, w)
            if z is not None and np.isfinite(z):
                return z
    return None


def estimate_any(x, y, P=DEFAULT):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(-x); x = x[o]; y = y[o]
    if len(x) > 5:
        # use the five finest levels
        x = x[-5:]; y = y[-5:]
    if len(x) < 4:
        return None
    return estimate(x, y, P)
