"""Prototype v2: configurable Bayesian continuum-limit estimator."""
import numpy as np


def log_prior_pq(p, q, cfg):
    d = q - p
    if cfg.get('pq_mix', False):
        val = np.full(np.shape(p), cfg.get('pq_background', 0.05), dtype=float)
        for w, pm, ps, dm, ds in cfg['pq_components']:
            val = val + w * np.exp(-0.5 * ((p - pm) / ps) ** 2 - 0.5 * ((d - dm) / ds) ** 2)
        return cfg.get('pq_strength', 1.0) * np.log(np.maximum(val, 1e-300))
    lp = -np.log1p(((p - cfg['p_center']) / cfg['p_width']) ** 2)
    ld = -np.log1p(((d - cfg['d_center']) / cfg['d_width']) ** 2)
    if cfg.get('d2_weight', 0) > 0:
        bump = cfg['d2_weight'] * np.exp(-0.5 * ((d - cfg['d2_center']) / cfg['d2_width']) ** 2)
        ld = np.log(np.exp(ld) + bump + 1e-300)
    return (cfg['p_strength'] * lp + cfg['d_strength'] * ld
            + cfg.get('p_tilt', 0.0) * p + cfg.get('q_tilt', 0.0) * q)


def log_prior_coef(A, B, C, cfg):
    sa, sb, sc = cfg['sa'], cfg['sb'], cfg['sc']
    la = -np.log1p((A / sa) ** 2) * cfg['a_strength']
    lb = -np.log1p(((B - cfg['mb']) / sb) ** 2) * cfg['b_strength']
    lc = -np.log1p(((C - cfg['mc']) / sc) ** 2) * cfg['c_strength']
    if cfg.get('hard_neg_bc', 0):
        bad = (B > 0) | (C > 0)
        lb = np.where(bad, lb - cfg['hard_neg_bc'], lb)
    lb -= cfg['b_pos_pen'] * np.maximum(B, 0.0) / sb
    lc -= cfg['c_pos_pen'] * np.maximum(C, 0.0) / sc
    if cfg.get('ratio_strength', 0) > 0:
        ratio = np.where(np.abs(B) > 1e-12, C / np.where(B == 0, 1e-12, B), -1.0)
        good = (ratio > 0) & (B < 0) & (C < 0)
        pen = -0.5 * ((np.log(np.maximum(ratio, 1e-12)) - cfg['ratio_mu']) / cfg['ratio_sig']) ** 2
        pen = np.where(good, pen, -cfg['ratio_bad'])
        la = la + cfg['ratio_strength'] * pen
    return la + lb + lc


def design(u, p, q):
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    lu = np.log(u)[:, None]
    pf = p.ravel()[None, :]
    qf = q.ravel()[None, :]
    A = np.stack([np.ones((len(u), pf.size)), np.exp(lu * pf), np.exp(lu * qf),
                  np.exp(lu * (qf + 1.5))], axis=-1)
    A = np.moveaxis(A, 0, -2)
    return A.reshape(p.shape + (len(u), 4))


def qr_solve(A, z):
    N, n, k = A.shape
    Q, R = np.linalg.qr(A, mode='complete')
    Qz = np.einsum('nij,ni->nj', Q, np.broadcast_to(z, (N, n)))
    Rk = np.array(R[:, :k, :], copy=True)
    # A tiny relative ridge keeps nearly collinear columns (possible when the
    # x-levels span many orders of magnitude) from making R singular.
    scale = np.max(np.abs(Rk), axis=(1, 2))
    eps = 1e-12 * np.maximum(scale, 1e-300)
    ii = np.arange(k)
    Rk[:, ii, ii] += eps[:, None]
    c = np.linalg.solve(Rk, Qz[:, :k])
    r = Qz[:, k] if n > k else np.zeros(N)
    diagR = np.abs(np.diagonal(Rk, axis1=1, axis2=2))
    logdetR = np.maximum(np.sum(np.log(np.maximum(diagR, 1e-300)), axis=1), -25.0)
    return c, r, logdetR


def residual_batch(u, z, p, q):
    return qr_solve(design(u, p, q), z)[1]


def two_power_residual(u, z, cfg):
    """Smallest relative residual of a two-term power fit.

    For the smooth family the discarded third term is tiny, so this is
    O(1e-4); oscillatory / noisy audit cases leave a much larger residual.
    """
    ps = np.linspace(cfg.get('osc_pmin', 0.3), cfg.get('osc_pmax', 4.0), 18)
    ds = np.linspace(cfg.get('osc_dmin', 0.15), cfg.get('osc_dmax', 3.2), 18)
    P, D = np.meshgrid(ps, ds, indexing='ij')
    Q = P + D
    pf, qf = P.ravel(), Q.ravel()
    lu = np.log(u)[:, None]
    A2 = np.stack([np.ones((len(u), pf.size)), np.exp(lu * pf[None, :]),
                   np.exp(lu * qf[None, :])], axis=-1)
    A2 = np.moveaxis(A2, 0, -2)
    c, _, _ = qr_solve(A2, z)
    res = z[None, :] - np.einsum('nik,nk->ni', A2, c)
    rss = np.mean(res * res, axis=1)
    return float(np.sqrt(np.min(rss)))


def extra_logweight(p, q, cfg):
    # optional factor from putting priors on raw (unscaled) coefficients
    xmax = cfg.get('_xmax', 1.0)
    return cfg.get('xmax_factor', 0.0) * (-(p + 2 * q + 1.5) * np.log(xmax))


def qcurve_fast(u, z, ps, ds, cfg):
    qmin = cfg['pmin'] + cfg['dmin']
    qmax = cfg['pmax'] + cfg['dmax']
    qs = np.arange(qmin, qmax + 1e-9, cfg['dq'])
    ds = np.arange(cfg['dmin'], cfg['dmax'] + 1e-9, cfg['dd'])
    Q = qs[:, None]
    P = Q - ds[None, :]
    valid = (P >= cfg['pmin']) & (P <= cfg['pmax'])
    A = design(u, P, Q)
    nq, nd = P.shape
    cgrid, r, _ = qr_solve(A.reshape(nq * nd, len(u), 4), z)
    cgrid = cgrid.reshape(nq, nd, 4)
    r = r.reshape(nq, nd)
    Pg, Qg = np.broadcast_arrays(P, Q)
    lpgrid = (log_prior_pq(Pg, Qg, cfg)
              + log_prior_coef(cgrid[..., 1], cgrid[..., 2], cgrid[..., 3], cfg))
    quality = float(np.max(lpgrid[valid])) if np.any(valid) else -np.inf
    minres = float(np.min(np.abs(r[valid]))) if np.any(valid) else np.inf
    sgn = np.sign(r)
    ch = (sgn[:, :-1] * sgn[:, 1:] < 0) & valid[:, :-1] & valid[:, 1:]
    iq, idd = np.nonzero(ch)
    if len(iq) == 0:
        return np.empty((0, 2)), (minres, quality)
    qv = qs[iq]
    dlo = ds[idd].copy(); dhi = ds[idd + 1].copy(); rlo = r[iq, idd].copy()
    for _ in range(cfg.get('nbisect', 36)):
        dm = 0.5 * (dlo + dhi)
        rm = residual_batch(u, z, qv - dm, qv)
        same = (np.sign(rm) == np.sign(rlo))
        dlo = np.where(same, dm, dlo); dhi = np.where(same, dhi, dm)
        rlo = np.where(same, rm, rlo)
    droot = 0.5 * (dlo + dhi)
    proot = qv - droot
    AA = design(u, proot, qv)
    c, rr, logdet = qr_solve(AA, z)
    h = cfg.get('hderiv', 1e-4)
    drdp = (residual_batch(u, z, proot + h, qv) - residual_batch(u, z, proot - h, qv)) / (2 * h)
    good = (np.abs(rr) < cfg.get('root_tol', 1e-9)) & (np.abs(drdp) > cfg.get('min_drdp', 1e-7))
    if not np.any(good):
        return np.empty((0, 2)), (minres, quality)
    proot, qv, c, logdet, drdp = proot[good], qv[good], c[good], logdet[good], drdp[good]
    lw = (log_prior_pq(proot, qv, cfg) + log_prior_coef(c[:, 1], c[:, 2], c[:, 3], cfg)
          + extra_logweight(proot, qv, cfg)
          - cfg.get('logdet_factor', 1.0) * logdet
          - cfg.get('curve_jac', 1.0) * np.log(np.abs(drdp)))
    return np.column_stack([c[:, 0], lw]), (minres, quality)


def grid_samples(u, z, ps, ds, cfg):
    P, D = np.meshgrid(ps, ds, indexing='ij')
    Q = P + D
    Pf = P.ravel(); Qf = Q.ravel()
    A = design(u, Pf, Qf)
    c, r, logdetR = qr_solve(A, z)
    lw = (log_prior_pq(Pf, Qf, cfg) + log_prior_coef(c[:, 1], c[:, 2], c[:, 3], cfg)
          + extra_logweight(Pf, Qf, cfg) - cfg.get('det_factor', 1.0) * logdetR)
    good = np.isfinite(lw) & np.isfinite(c[:, 0])
    lp = log_prior_pq(Pf, Qf, cfg) + log_prior_coef(c[:, 1], c[:, 2], c[:, 3], cfg)
    quality = float(np.max(lp[np.isfinite(lp)])) if np.any(np.isfinite(lp)) else -np.inf
    return np.column_stack([c[good, 0], lw[good]]), (0.0, quality)


def estimate(x, y, cfg, return_info=False):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    idx = np.argsort(x); x, y = x[idx], y[idx]
    n = len(x)
    u = x / x[-1]
    y0 = y[0]
    s = np.max(np.abs(y - y0))
    if not np.isfinite(s) or s <= 0:
        return (float(y0), {}) if return_info else float(y0)
    z = (y - y0) / s
    cfg = dict(cfg)
    cfg['_xmax'] = x[-1]
    ps = np.arange(cfg['pmin'], cfg['pmax'] + 1e-9, cfg['dp'])
    ds = np.arange(cfg['dmin'], cfg['dmax'] + 1e-9, cfg['dd'])
    if n == 5:
        samples, (minres, quality) = qcurve_fast(u, z, ps, ds, cfg)
    else:
        samples, (minres, quality) = grid_samples(u, z, ps, ds, cfg)
    osc = two_power_residual(u, z, cfg)
    info = {'minres': minres, 'quality': quality, 'osc': osc, 'nsamples': len(samples)}
    qt = cfg.get('quality_tol', -8.0)
    bad = (len(samples) == 0 or osc > cfg.get('osc_tol', 8e-3)
           or (quality < qt and osc > cfg.get('osc_low', 1e-3)))
    if bad:
        val = robust_estimate(x, y, z, s, y0, cfg)
        info['mode'] = 'robust'
        return (val, info) if return_info else val
    T, lw = samples[:, 0], samples[:, 1]
    ts = cfg.get('t_prior_sigma', 0.0)
    if ts > 0:
        lw = lw - 0.5 * (T / ts) ** 2
    lw = (lw - np.max(lw)) / cfg.get('temperature', 1.0)
    w = np.exp(np.clip(lw, -700, 0))
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        val = robust_estimate(x, y, z, s, y0, cfg)
        info['mode'] = 'robust'
        return (val, info) if return_info else val
    w /= wsum
    agg = cfg.get('agg4', cfg['agg']) if n == 4 else cfg['agg']
    if agg == 'mean':
        That = float(np.dot(w, T))
    else:
        order = np.argsort(T)
        cw = np.cumsum(w[order])
        j = np.searchsorted(cw, 0.5)
        That = float(T[order[min(j, len(T) - 1)]])
    val = y0 + s * That
    if cfg.get('clamp', 0):
        val = float(np.clip(val, y0 - s * cfg['clamp'], y0 + s * cfg['clamp']))
    info['mode'] = 'bayes'
    info['That'] = That
    return (float(val), info) if return_info else (float(val) if np.isfinite(val) else float(y0))


def robust_estimate(x, y, z, s, y0, cfg):
    """Model-free extrapolation, used when the smooth power model does not fit.

    Oscillatory / noisy deviations largely cancel between the two finest
    levels, so a small-x-weighted local average is the safest estimate.
    """
    avg2 = 0.5 * (y[0] + y[1])
    med = float(np.median(y))
    v = 0.9 * avg2 + 0.1 * med
    return float(v) if np.isfinite(v) else float(y[0])


def default_cfg(**kw):
    cfg = dict(
        pmin=0.3, pmax=4.5, dp=0.05,
        dmin=0.2, dmax=4.5, dd=0.02, dq=0.02,
        param='q', npscan=160, nbisect=36,
        p_center=2.15, p_width=1.2, p_strength=1.0,
        d_center=0.65, d_width=0.7, d_strength=1.0,
        d2_weight=0.5, d2_center=1.9, d2_width=0.6,
        p_tilt=0.0, q_tilt=0.0,
        sa=2.0, a_strength=1.0,
        sb=1.0, mb=-0.4, b_strength=1.0, b_pos_pen=1.0,
        sc=1.0, mc=-0.2, c_strength=1.0, c_pos_pen=1.0,
        hard_neg_bc=0.0,
        ratio_strength=0.0, ratio_mu=-1.6, ratio_sig=1.2, ratio_bad=20.0,
        xmax_factor=0.0, det_factor=1.0, logdet_factor=1.0, curve_jac=1.0,
        agg='median', clamp=0.0, temperature=1.0, t_prior_sigma=0.0,
        fit_tol=1e-3, quality_tol=-6.0,
    )
    cfg.update(kw)
    return cfg


def production_cfg():
    """Tuned configuration used by the submitted program."""
    return default_cfg(
        dq=0.025, dd=0.025, dp=0.05, nbisect=34,
        temperature=2.0, hard_neg_bc=5.0, xmax_factor=0.5,
        quality_tol=-8.0,
    )
