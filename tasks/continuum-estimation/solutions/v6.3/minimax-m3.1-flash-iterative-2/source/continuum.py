#!/usr/bin/env python3
"""Continuum estimation from positive discretization levels.

Reads {"x": [...], "y": [...]} and writes {"estimate": <finite float>}.

Model
-----
The observations follow a convergent expansion with unknown positive
convergence orders:

    y(x) = t + a*x^p + b*x^q + c*x^(q+1.5)

(t is the limiting value; the exponents p < q and the third exponent q+1.5
were recovered from the public development pairs).  Six parameters have to be
determined from four or five observations, so the data do not pin the
expansion down to a single point: for five observations the exactly fitting
exponent pairs form a curve, for four observations a two-dimensional set.

Estimation
----------
For every candidate pair of exponents the linear parameters (t, a, b, c) are
obtained by least squares (variable projection).  The exponent pairs that
reproduce the data exactly are located by following the sign changes of the
projected residual (a secant iteration refines each crossing to machine
precision).  The resulting consistent set is weighted with priors on the
exponents and on the linear coefficients, and the limiting value is reported
as the posterior mean (the posterior median is used as a robustness variant).

Everything runs on the bundled NumPy only, in a fraction of a second.
"""
import json
import math
import os
import sys

EPS = 1e-300

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:          # pragma: no cover - the bundled runtime has NumPy
    HAVE_NUMPY = False
    np = None

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
class Cfg:
    # exponent priors (soft: no hard cut-offs, only decaying tails)
    p_lo, p_hi = 0.25, 3.70
    q_lo, q_hi = 0.60, 5.40
    p_edge_s = 0.35
    q_edge_s = 0.40
    gap_mu, gap_sd = 0.90, 0.90
    gap_lo, gap_hi = -0.30, 3.00
    gap_edge_s = 0.30
    gap_flat = False
    # coefficient priors
    a_mu, a_sd = 0.0, 1.60
    b_mu, b_sd = -0.55, 0.60
    c_mu, c_sd = -0.25, 0.45
    b_neg = 1.0
    c_neg = 1.0
    b_neg_s = 0.035
    c_neg_s = 0.035
    # prior on the limiting value itself (the generator keeps |t| bounded)
    t_prior = 0
    t_lo, t_hi = -3.20, 3.20
    t_edge_s = 0.50
    # soft box prior on the coefficients (mixed in with weight box_w)
    box_w = 0.0
    box_s = 0.15
    box_a_lo, box_a_hi = -1.60, 1.60
    box_b_lo, box_b_hi = -0.90, -0.10
    box_c_lo, box_c_hi = -0.45, -0.05
    # search (filled lazily so that the module also imports without NumPy)
    p_grid = None
    q_grid = None
    tol = 1e-9
    refine_iters = 8
    use_k = True
    agg = "median"
    # ensemble over coefficient-prior widths (0 = off)
    ensemble = 0
    ensemble_widths = ((1.20, 0.40, 0.28), (1.60, 0.60, 0.45), (2.40, 0.90, 0.70))
    soft_sigma = 3e-7


def _grids(cfg):
    if cfg.p_grid is None:
        cfg.p_grid = np.arange(0.30, 3.8001, 0.01)
        cfg.q_grid = np.arange(0.55, 5.6001, 0.01)
    return cfg.p_grid, cfg.q_grid


def _soft(v, lo, hi, s):
    """Smooth indicator: 1 inside [lo, hi], decaying exponentially outside."""
    v = np.asarray(v, float)
    with np.errstate(over="ignore"):
        up = 1.0 / (1.0 + np.exp((v - hi) / s))
        dn = 1.0 / (1.0 + np.exp((lo - v) / s))
    return up * dn


def _flat(v, lo, hi, w):
    """Flat inside [lo, hi], smoothly decaying to zero over a width w."""
    t = np.clip((lo - np.asarray(v, float)) / w, 0.0, 1.0)
    a = 1.0 - t * t * (3.0 - 2.0 * t)
    t = np.clip((np.asarray(v, float) - hi) / w, 0.0, 1.0)
    b = 1.0 - t * t * (3.0 - 2.0 * t)
    return a * b


def w_exp(p, q, cfg):
    """Prior density of the exponents.

    Soft (logistic) edges around the plausible ranges of `p`, `q` and of the
    gap `q - p`; no hard cut-offs, so a case outside the core range is
    down-weighted rather than discarded.
    """
    pp = _soft(p, cfg.p_lo, cfg.p_hi, cfg.p_edge_s)
    qq = _soft(q, cfg.q_lo, cfg.q_hi, cfg.q_edge_s)
    gap = q - p
    if cfg.gap_flat:
        gg = _flat(gap, cfg.gap_lo, cfg.gap_hi, cfg.gap_edge_s)
    else:
        gg = np.exp(-0.5 * ((gap - cfg.gap_mu) / cfg.gap_sd) ** 2)
        gg = gg * _soft(gap, cfg.gap_lo, cfg.gap_hi, 0.30)
    return pp * qq * gg


def w_t(t, cfg):
    """Prior density of the limiting value (flat inside the plausible range)."""
    if not cfg.t_prior:
        return 1.0
    return _flat(t, cfg.t_lo, cfg.t_hi, cfg.t_edge_s)


def w_coef(a, b, c, cfg):
    """Prior density of the linear coefficients.

    A Gaussian centred on the values seen in the public development pairs
    (|a| <= 1.5, b in [-0.9, -0.1], c in [-0.4, -0.05]) with deliberately
    generous widths, plus a sharp suppression of the (never observed) case
    b > 0, c > 0.  A soft-box variant is available through `cfg.box_w`.
    """
    wa = np.exp(-0.5 * ((a - cfg.a_mu) / cfg.a_sd) ** 2)
    wb = np.exp(-0.5 * ((b - cfg.b_mu) / cfg.b_sd) ** 2)
    wc = np.exp(-0.5 * ((c - cfg.c_mu) / cfg.c_sd) ** 2)
    if cfg.b_neg:
        wb = wb * np.exp(-cfg.b_neg * np.maximum(b, 0.0) / cfg.b_neg_s)
    if cfg.c_neg:
        wc = wc * np.exp(-cfg.c_neg * np.maximum(c, 0.0) / cfg.c_neg_s)
    if cfg.box_w > 0.0:
        ba = _soft(a, cfg.box_a_lo, cfg.box_a_hi, cfg.box_s)
        bb = _soft(b, cfg.box_b_lo, cfg.box_b_hi, cfg.box_s)
        bc = _soft(c, cfg.box_c_lo, cfg.box_c_hi, cfg.box_s)
        wa = np.exp(np.log(np.maximum(wa, EPS)) + cfg.box_w * np.log(np.maximum(ba, EPS)))
        wb = np.exp(np.log(np.maximum(wb, EPS)) + cfg.box_w * np.log(np.maximum(bb, EPS)))
        wc = np.exp(np.log(np.maximum(wc, EPS)) + cfg.box_w * np.log(np.maximum(bc, EPS)))
    return wa * wb * wc


# --------------------------------------------------------------------------
# variable projection
# --------------------------------------------------------------------------
class Fit:
    """Least squares for  y ~ 1 + a*x^p + b*x^q + c*x^(q+1.5)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.lx = np.log(self.x)
        self.n = len(self.x)
        self.yy = float(np.dot(self.y, self.y))
        self.Sy = float(np.sum(self.y))
        self.yabs = float(np.linalg.norm(self.y))

    def _S(self, exps, w=None):
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.exp(np.asarray(exps, float)[..., None] * self.lx)
        if w is None:
            return e.sum(axis=-1)
        return (e * w).sum(axis=-1)

    def fit_many(self, ps, qs):
        """Least squares for many (p, q) pairs; ps and qs broadcast.

        Returns (resid[..., n], coef[..., 4]).
        """
        ps = np.asarray(ps, float)
        qs = np.asarray(qs, float)
        P, Q = np.broadcast_arrays(ps, qs)
        shape = P.shape
        P = P.ravel()
        Q = Q.ravel()
        y = self.y
        Sp = self._S(P)
        S2p = self._S(2 * P)
        Sy_p = self._S(P, y)
        Sq = self._S(Q)
        Sq15 = self._S(Q + 1.5)
        S2q = self._S(2 * Q)
        S2q15 = self._S(2 * Q + 1.5)
        Sq3 = self._S(2 * Q + 3.0)
        Sy_q = self._S(Q, y)
        Sy_q15 = self._S(Q + 1.5, y)
        Spq = self._S(P + Q)
        Spq15 = self._S(P + Q + 1.5)

        m = P.size
        ATA = np.empty((m, 4, 4))
        ATA[:, 0, 0] = self.n
        ATA[:, 0, 1] = ATA[:, 1, 0] = Sp
        ATA[:, 0, 2] = ATA[:, 2, 0] = Sq
        ATA[:, 0, 3] = ATA[:, 3, 0] = Sq15
        ATA[:, 1, 1] = S2p
        ATA[:, 1, 2] = ATA[:, 2, 1] = Spq
        ATA[:, 1, 3] = ATA[:, 3, 1] = Spq15
        ATA[:, 2, 2] = S2q
        ATA[:, 2, 3] = ATA[:, 3, 2] = S2q15
        ATA[:, 3, 3] = Sq3
        ATy = np.empty((m, 4))
        ATy[:, 0] = self.Sy
        ATy[:, 1] = Sy_p
        ATy[:, 2] = Sy_q
        ATy[:, 3] = Sy_q15

        # exponent pairs that make two columns coincide are degenerate: they
        # are kept out of the linear algebra so the batched solve stays valid
        good = (np.abs(P - Q) > 0.03) & (np.abs(P - Q - 1.5) > 0.03)
        if not good.all():
            ATA[~good] = np.eye(4)
            ATy[~good] = 0.0

        try:
            coef = np.linalg.solve(ATA, ATy[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coef = np.empty_like(ATy)
            for i in range(m):
                c, *_ = np.linalg.lstsq(ATA[i], ATy[i], rcond=None)
                coef[i] = c
        bad = ~np.all(np.isfinite(coef), axis=1)
        if bad.any():
            for i in np.where(bad)[0]:
                c, *_ = np.linalg.lstsq(ATA[i], ATy[i], rcond=None)
                coef[i] = c

        A = np.empty((m, self.n, 4))
        A[:, :, 0] = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            A[:, :, 1] = np.exp(P[:, None] * self.lx[None, :])
            xq = np.exp(Q[:, None] * self.lx[None, :])
        A[:, :, 2] = xq
        with np.errstate(over="ignore", invalid="ignore"):
            A[:, :, 3] = xq * np.exp(1.5 * self.lx[None, :])
        R = np.matmul(A, coef[..., None])[..., 0] - y[None, :]
        return R.reshape(shape + (self.n,)), coef.reshape(shape + (4,))


def _trace(fit, cfg):
    """Exponent pairs that reproduce the observations exactly.

    With five observations the projected residual is a one-dimensional
    function of (p, q); its zero set is a curve.  The curve is followed by
    detecting sign changes of the residual along q for each p and refining
    every crossing with a secant iteration.

    Returns (p, q, coef, k) with k the local growth rate of the residual.
    """
    ps, qs = _grids(cfg)
    chunk = 60
    R_parts = []
    for i0 in range(0, len(ps), chunk):
        pc = ps[i0:i0 + chunk]
        R, _ = fit.fit_many(pc[:, None], qs[None, :])
        R_parts.append(R)
    Rall = np.concatenate(R_parts, axis=0)
    ncomp = Rall.shape[2]

    # The projected residual is a vector; each of its components changes sign
    # where the fit becomes exact.  Using all components makes the crossing
    # detection independent of the slowly rotating residual direction.
    pi_list = []
    qi_list = []
    comp_list = []
    for c in range(ncomp):
        Sc = Rall[:, :, c]
        sgn = np.sign(Sc)
        sgn[sgn == 0] = 1.0
        cross = (sgn[:, :-1] * sgn[:, 1:]) < 0
        pi_c, qi_c = np.where(cross)
        if len(pi_c) == 0:
            continue
        pi_list.append(pi_c)
        qi_list.append(qi_c)
        comp_list.append(np.full(len(pi_c), c))
    if not pi_list:
        return (np.array([]), np.array([]), np.zeros((0, 4)), np.array([]))
    pi_ = np.concatenate(pi_list)
    qi_ = np.concatenate(qi_list)
    comp_idx = np.concatenate(comp_list)
    q0 = qs[qi_]
    q1 = qs[qi_ + 1]
    s0 = Rall[pi_, qi_, comp_idx]
    s1 = Rall[pi_, qi_ + 1, comp_idx]
    pp = ps[pi_]

    qa, qb, sa, sb = q0.copy(), q1.copy(), s0.copy(), s1.copy()
    for _ in range(cfg.refine_iters):
        with np.errstate(divide='ignore', invalid='ignore'):
            qn = qb - sb * (qb - qa) / (sb - sa)
        bad = ~np.isfinite(qn) | (qn < qs[0]) | (qn > qs[-1])
        qn = np.where(bad, qb, qn)
        R, _ = fit.fit_many(pp, qn)
        sn = R[np.arange(len(pp)), comp_idx]
        qa, sa, qb, sb = qb, sb, qn, sn
        if np.all(np.abs(sb) < 1e-18):
            break
    qstar = qb

    R, C = fit.fit_many(pp, qstar)
    nrm = np.linalg.norm(R, axis=1)
    ok = (nrm <= cfg.tol) & (qstar >= qs[0]) & (qstar <= qs[-1])
    if not ok.any():
        return (np.array([]), np.array([]), np.zeros((0, 4)), np.array([]))

    dq = qs[1] - qs[0]
    qi_star = np.clip(np.searchsorted(qs, qstar) - 1, 0, len(qs) - 2)
    k = np.empty(len(qstar))
    for c in range(ncomp):
        m = comp_idx == c
        if not m.any():
            continue
        dSc = np.abs(np.gradient(Rall[:, :, c], dq, axis=1))
        k[m] = dSc[pi_[m], qi_star[m]]
    return pp[ok], qstar[ok], C[ok], k[ok]


def _grid_posterior(fit, cfg, exact):
    """Posterior over exponents when every pair fits (four observations)."""
    ps, qs = _grids(cfg)
    chunk = 60
    ts, ws, nrms = [], [], []
    for i0 in range(0, len(ps), chunk):
        pc = ps[i0:i0 + chunk]
        R, C = fit.fit_many(pc[:, None], qs[None, :])
        nrm = np.linalg.norm(R, axis=2)
        w = w_exp(pc[:, None], qs[None, :], cfg) * w_coef(C[:, :, 1], C[:, :, 2], C[:, :, 3], cfg)
        w = w * w_t(C[:, :, 0], cfg)
        w = np.where(qs[None, :] > pc[:, None], w, 0.0)
        ts.append(C[:, :, 0].ravel())
        ws.append(w.ravel())
        nrms.append(nrm.ravel())
    if not ts:
        return np.array([]), np.array([]), np.array([])
    t = np.concatenate(ts)
    w = np.concatenate(ws)
    nrm = np.concatenate(nrms)
    if exact:
        w = w * np.exp(-0.5 * (nrm / (1e-8 * max(1.0, fit.yabs))) ** 2)
    else:
        # the model does not reproduce the data exactly: keep the exponent
        # pairs that fit best and let their residual decide the weight
        rmin = float(nrm.min()) if np.isfinite(nrm).any() else 0.0
        sigma = max(cfg.soft_sigma, 0.3 * rmin)
        wres = np.exp(-0.5 * (nrm / sigma) ** 2)
        # when the model does not fit, the coefficient priors are unreliable:
        # let the residual decide which exponent pairs are plausible
        w = wres
    return t, w, nrm


def summarize(t, w, agg="mean"):
    """Posterior summary of the limiting value."""
    if len(t) == 0 or not np.all(np.isfinite(w)) or w.sum() <= 0:
        return None
    w = w / w.sum()
    if agg == "mean":
        return float(np.sum(w * t))
    idx = np.argsort(t)
    ts_, ws_ = t[idx], w[idx]
    cw = np.cumsum(ws_)
    if agg == "map":
        k = int(np.argmax(ws_))
        return float(ts_[k])
    if agg == "lmed":
        # optimal for the evaluation loss |z-t|/max(1,|t|): a median with the
        # posterior mass re-weighted by 1/max(1,|t|)
        wl = ws_ / np.maximum(1.0, np.abs(ts_))
        tot = wl.sum()
        if tot > 0:
            wl = wl / tot
            k = int(np.searchsorted(np.cumsum(wl), 0.5))
            return float(ts_[min(k, len(ts_) - 1)])
    if agg == "trim":
        lo = np.interp(0.05, cw, ts_)
        hi = np.interp(0.95, cw, ts_)
        sel = (ts_ >= lo) & (ts_ <= hi)
        if sel.any():
            ww = ws_[sel] / ws_[sel].sum()
            return float(np.sum(ww * ts_[sel]))
    k = int(np.searchsorted(cw, 0.5))
    return float(ts_[min(k, len(ts_) - 1)])


def _poly_extrap(x, y):
    """Pure-python fallback: polynomial extrapolation to x = 0."""
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return float(y[0])
    total = 0.0
    for i in range(n):
        term = float(y[i])
        for j in range(n):
            if i == j:
                continue
            den = x[i] - x[j]
            if den == 0.0:
                term = None
                break
            term *= (-x[j]) / den
        if term is not None:
            total += term
    if total != total or total in (float('inf'), float('-inf')):
        ys = sorted(y)
        total = ys[len(ys) // 2]
    return float(total)


def _estimate_one(x, y, cfg=Cfg, agg=None):
    """Posterior estimate of the limiting value."""
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    n = len(x)
    if not HAVE_NUMPY:
        return _poly_extrap(x, y)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    agg = agg or cfg.agg
    fallback = float(np.median(y))
    est = None
    try:
        fit = Fit(x, y)
        if n >= 5:
            ps, qs, coef, k = _trace(fit, cfg)
            if len(ps) > 0:
                t = coef[:, 0]
                w = w_exp(ps, qs, cfg) * w_coef(coef[:, 1], coef[:, 2], coef[:, 3], cfg)
                w = w * w_t(t, cfg)
                if cfg.use_k:
                    w = w / np.maximum(k, 1e-12)
                if w.sum() <= 0:
                    w = np.ones_like(t)
                est = summarize(t, w, agg)
            if est is None:
                # the three-term model does not reproduce the data: fall back
                # to the exponent pairs that fit best
                t, w, nrm = _grid_posterior(fit, cfg, exact=False)
                est = summarize(t, w, "median")
        else:
            t, w, nrm = _grid_posterior(fit, cfg, exact=True)
            est = summarize(t, w, agg)
            if est is None:
                t, w, nrm = _grid_posterior(fit, cfg, exact=False)
                est = summarize(t, w, "median")
    except Exception:
        est = None
    if est is None or not np.isfinite(est):
        est = fallback
    # sanity guard: the limit of a convergent expansion with these data must
    # stay in a sane range; anything wild is replaced by the sample median
    try:
        bound = 10.0 * (1.0 + float(np.max(np.abs(y))))
        if not np.isfinite(est) or abs(est) > bound:
            est = fallback
    except Exception:
        est = fallback
    if not np.isfinite(est):
        est = 0.0
    return float(est)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def estimate(x, y, cfg=Cfg, agg=None):
    """Posterior estimate of the limiting value.

    With `cfg.ensemble` the estimate is the median of the estimates obtained
    with several coefficient-prior widths; this hedges against a mis-specified
    prior width.
    """
    if cfg.ensemble:
        vals = []
        for a_sd, b_sd, c_sd in cfg.ensemble_widths:
            class C2(Cfg):
                pass
            C2.a_sd, C2.b_sd, C2.c_sd = a_sd, b_sd, c_sd
            try:
                vals.append(_estimate_one(x, y, C2, agg))
            except Exception:
                continue
        if vals:
            return float(np.median(vals))
    return _estimate_one(x, y, cfg, agg)


def _load(path):
    with open(path, "r") as fh:
        obj = json.load(fh)
    x = [float(v) for v in obj["x"]]
    y = [float(v) for v in obj["y"]]
    return x, y


def _write(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"estimate": float(value)}, fh)


def main(argv):
    inp = argv[1] if len(argv) > 1 else "/workspace/inputs/input.json"
    out = argv[2] if len(argv) > 2 else "/workspace/result/output.json"
    try:
        x, y = _load(inp)
    except Exception:
        _write(out, 0.0)
        return 0
    try:
        value = estimate(x, y)
    except Exception:
        value = 0.0
    try:
        if value is None or not math.isfinite(value):
            value = float(np.median(y)) if len(y) else 0.0
    except Exception:
        value = 0.0
    _write(out, value)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
