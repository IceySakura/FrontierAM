#!/usr/bin/env python3
"""Continuum-limit estimation from positive discretization levels.

Model discovered from the public development pairs (residuals at machine
precision on all seven smooth-primary examples):

    y(x) = t + a*x**p + b*x**q + c*x**(q + 3/2)

with 0 < p < q.  The limit sought is t = lim_{x->0} y(x).

Given n observations the model has 6 unknowns (t,a,p,b,q,c), of which
(t,a,b,c) enter linearly once (p,q) are fixed.  Therefore

  * n = 4 : every (p,q) interpolates the data exactly, so the answer is
            governed entirely by the prior on the parameters;
  * n = 5 : the exact-fit condition is the single scalar equation
            det[1, x^p, x^q, x^(q+3/2), y] = 0,
            whose solution set is a curve in the (p,q) plane.

Both cases are handled by the same routine: evaluate the prior-weighted
posterior on a dense set of exactly-fitting (p,q) candidates and report the
posterior median of t (the Bayes estimator for the L1 loss used in scoring).

Only NumPy is required; root isolation/refinement is done in-house.
"""
import json
import math
import os
import sys

import numpy as np

GAP3 = 1.5

# ---------------------------------------------------------------------------
# priors -- smooth ("soft box") log-densities; ranges inferred from the public
# development pairs and deliberately widened so that plausible unseen cases
# are not excluded.
# ---------------------------------------------------------------------------
CFG = dict(
    p=(-0.05, 3.60, 0.30),      # first exponent
    q=(0.40, 5.40, 0.40),       # second exponent
    gap=(0.28, 2.35, 0.22),     # q - p
    t=(0.55, 3.80, 0.45),       # |limit|
    a=(-1.90, 1.90, 0.22),
    b=(-1.10, 0.08, 0.09),
    c=(-0.55, 0.15, 0.07),
    use_jac=False,
    summary='median',
    np5=340,          # p-slices used to isolate manifold roots (n = 5)
    nq=2200,          # q-samples per p-slice
    grid4=(300, 340), # (p, q) grid for n = 4
    bisect_iters=52,
    sigma_rel=1e-9,   # soft likelihood scale (safety valve for inexact data)
    eps_wide=1e-2,    # weight of the wide-likelihood component
    sigma_wide=3e-3,
    refine_rounds=2,
    use_slope=False,   # 1/|dF/dq| manifold measure; measured to hurt (README)
    pool=True,         # one posterior over the union of all candidate sets
    strain_cut=-6.0,   # log-prior-plausibility below which the 3-term model is rejected
)


def _lsig(z):
    return -np.logaddexp(0.0, -np.clip(z, -60.0, 60.0))


def softbox(v, lo, hi, s):
    return _lsig((v - lo) / s) + _lsig((hi - v) / s)


# ---------------------------------------------------------------------------
# model algebra
# ---------------------------------------------------------------------------
def design(x, p, q):
    """A[k,i,:] = [1, x_i^p_k, x_i^q_k, x_i^(q_k+3/2)]"""
    lx = np.log(x)
    A = np.empty((p.size, x.size, 4))
    A[:, :, 0] = 1.0
    A[:, :, 1] = np.exp(np.outer(p, lx))
    A[:, :, 2] = np.exp(np.outer(q, lx))
    A[:, :, 3] = np.exp(np.outer(q + GAP3, lx))
    return A


def lstsq_theta(A, y):
    """Batched least squares.  Returns theta, residual norm, log|det(A^T A)|/2, ok."""
    Q, R = np.linalg.qr(A)
    Qty = np.einsum('kni,n->ki', Q, y)
    d = np.diagonal(R, axis1=1, axis2=2)
    with np.errstate(all='ignore'):
        try:
            th = np.linalg.solve(R, Qty[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            th = np.einsum('kij,kj->ki', np.linalg.pinv(R), Qty)
        rn = np.linalg.norm(np.einsum('knj,kj->kn', A, th) - y, axis=1)
        lad = np.log(np.abs(d) + 1e-300).sum(axis=1)
    good = (np.isfinite(th).all(axis=1) & np.isfinite(rn) & np.isfinite(lad)
            & (np.abs(th).max(axis=1) < 1e12))
    return th, rn, lad, good


def detF(x, y, p, q):
    """Signed residual determinant: zero iff y lies in the model column space."""
    M = design(x, p, q)
    col = np.broadcast_to(y[None, :, None], (p.size, x.size, 1))
    with np.errstate(all='ignore'):
        return np.linalg.det(np.concatenate([M, col], axis=2))


def logprior(p, q, th, cfg):
    lp = (softbox(p, *cfg['p']) + softbox(q, *cfg['q'])
          + softbox(q - p, *cfg['gap']) + softbox(np.abs(th[:, 0]), *cfg['t']))
    for i, k in zip((1, 2, 3), 'abc'):
        lp = lp + softbox(th[:, i], *cfg[k])
    return lp


def loglike(rn, scale, cfg):
    """Robust likelihood: a sharp component for exact fits plus a wide floor."""
    s1 = max(cfg['sigma_rel'] * scale, 1e-300)
    s2 = max(cfg['sigma_wide'] * scale, 1e-300)
    z1 = -0.5 * np.clip((rn / s1) ** 2, 0.0, 1e12)
    z2 = math.log(cfg['eps_wide']) - 0.5 * np.clip((rn / s2) ** 2, 0.0, 1e12)
    return np.logaddexp(z1, z2)


# ---------------------------------------------------------------------------
# candidate sets
# ---------------------------------------------------------------------------
def _bisect(x, y, p, a, b, iters):
    """Vectorised bisection of F(p, .)=0 inside [a,b] (sign change assumed)."""
    fa = detF(x, y, p, a)
    sa = np.sign(fa)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = detF(x, y, p, m)
        same = np.sign(fm) == sa
        a = np.where(same, m, a)
        b = np.where(same, b, m)
        sa = np.where(same, np.sign(fm), sa)
        if not np.any(same):
            break
    return 0.5 * (a + b)


def candidates_n5(x, y, cfg, qlo, qhi, pgrid=None, nq=None):
    """Isolate and refine the exact-fit manifold {(p,q): det F(p,q)=0}."""
    Ps, As, Bs = [], [], []
    nq = nq or cfg['nq']
    pg = (np.linspace(max(cfg['p'][0] + 1e-3, 0.02), cfg['p'][1], cfg['np5'])
          if pgrid is None else pgrid)
    for pv in pg:
        lo = max(qlo, pv + 1e-3)
        if lo >= qhi:
            continue
        zs = np.linspace(lo, qhi, nq)
        F = detF(x, y, np.full_like(zs, pv), zs)
        s = np.sign(F)
        idx = np.where((s[:-1] * s[1:] < 0) & np.isfinite(F[:-1]) & np.isfinite(F[1:]))[0]
        if idx.size:
            Ps.append(np.full(idx.size, pv))
            As.append(zs[idx])
            Bs.append(zs[idx + 1])
    if not Ps:
        return None
    p = np.concatenate(Ps); a = np.concatenate(As); b = np.concatenate(Bs)
    q = _bisect(x, y, p, a, b, cfg['bisect_iters'])
    keep = (q > p + 1e-4) & np.isfinite(q)
    p = p[keep]; q = q[keep]
    if not cfg['use_slope']:
        return p, q
    # |dF/dq| on the manifold: the delta-function Jacobian that turns the
    # 2-d (p,q) integral into an integral along the exact-fit curve.
    h = np.maximum(np.abs(q) * 1e-7, 1e-9)
    with np.errstate(all='ignore'):
        sl = np.abs(detF(x, y, p, q + h) - detF(x, y, p, q - h)) / (2.0 * h)
    sl = np.where(np.isfinite(sl) & (sl > 0), sl, np.nan)
    return p, q, sl


def candidates_n4(x, y, cfg):
    gp, gq = cfg['grid4']
    pg = np.linspace(max(cfg['p'][0] + 1e-3, 0.02), cfg['p'][1], gp)
    qg = np.linspace(cfg['q'][0], cfg['q'][1], gq)
    P, Q = np.meshgrid(pg, qg, indexing='ij')
    m = Q > P + 1e-3
    return P[m], Q[m]


def candidates_n4b(x, y, cfg):
    """Coarse safety grid used if the main grid yields no usable candidate."""
    gp, gq = max(40, cfg['grid4'][0] // 6), max(40, cfg['grid4'][1] // 6)
    pg = np.linspace(0.05, 6.0, gp)
    qg = np.linspace(0.10, 9.0, gq)
    P, Q = np.meshgrid(pg, qg, indexing='ij')
    m = Q > P + 1e-3
    return P[m], Q[m]


# ---------------------------------------------------------------------------
# posterior summary
# ---------------------------------------------------------------------------
def summarize(th, logw, cfg):
    fin = np.isfinite(logw)
    if not np.any(fin):
        return None
    w = np.exp(logw - logw[fin].max())
    w = np.where(np.isfinite(w), w, 0.0)
    s = w.sum()
    if not (s > 0.0) or not np.isfinite(s):
        return None
    t = th[:, 0]
    if cfg['summary'] == 'mean':
        z = float((w * t).sum() / s)
    elif cfg['summary'] == 'map':
        z = float(t[int(np.argmax(w))])
    else:
        o = np.argsort(t)
        cdf = np.cumsum(w[o])
        cdf /= cdf[-1]
        z = float(np.interp(0.5, cdf, t[o]))
    return z if np.isfinite(z) else None


def _global_argmax(x, y, cfg, cands, scale):
    bp = bq = None
    blw = -np.inf
    for p, q, sl in cands:
        if p.size == 0:
            continue
        th, rn, lad, good = lstsq_theta(design(x, p, q), y)
        lw = logprior(p, q, th, cfg) + loglike(rn, scale, cfg)
        if sl is not None:
            lw = lw - np.log(np.maximum(sl, 1e-300))
            good = good & np.isfinite(sl)
        lw[~good] = -np.inf
        if not np.any(np.isfinite(lw)):
            continue
        k = int(np.argmax(lw))
        if lw[k] > blw:
            blw = float(lw[k]); bp = float(p[k]); bq = float(q[k])
    return bp, bq


def posterior_estimate(x, y, cfg):
    n = x.size
    scale = max(1.0, float(np.max(np.abs(y))))
    span = float(np.max(y) - np.min(y))
    scale = max(scale, 1e-3 * span if span > 0 else 1e-300)

    cands = []
    if n == 5:
        c = candidates_n5(x, y, cfg, cfg['q'][0], cfg['q'][1])
        if c is not None:
            if len(c) == 3:
                cands.append((c[0], c[1], c[2]))
            else:
                cands.append((c[0], c[1], None))
    elif n == 4:
        p4, q4 = candidates_n4(x, y, cfg)
        cands.append((p4, q4, None))
    if not cands:
        # outside the specified 4/5 observation range: fall back to the coarse
        # lattice, where the soft likelihood (not exactness) does the selection
        p4, q4 = candidates_n4b(x, y, cfg)
        cands.append((p4, q4, None))

    # local refinement around the best global candidate sharpens the estimate
    try:
        p0, q0 = _global_argmax(x, y, cfg, cands, scale)
        if p0 is not None:
            extra = refine_candidates(x, y, cfg, p0, q0, cfg['refine_rounds'])
            if extra is not None:
                cands.append(extra)
    except Exception:
        pass

    groups = ([[ (p, q, sl) for (p, q, sl) in cands if p.size ]] if cfg['pool']
              else [[c] for c in cands if c[0].size])
    best = None
    for grp in groups:
        if not grp:
            return None
        p = np.concatenate([c[0] for c in grp])
        q = np.concatenate([c[1] for c in grp])
        sl = np.concatenate([c[2] if c[2] is not None else np.ones_like(c[0])
                             for c in grp])
        A = design(x, p, q)
        th, rn, lad, good = lstsq_theta(A, y)
        logw = logprior(p, q, th, cfg) + loglike(rn, scale, cfg) - np.log(sl)
        if cfg['use_jac']:
            logw = logw - lad
        logw[~good] = -np.inf
        if not np.any(np.isfinite(logw)):
            continue
        z = summarize(th, logw, cfg)
        if z is None:
            continue
        mx = float(np.max(logw[np.isfinite(logw)]))
        mass = float(np.exp(logw[np.isfinite(logw)] - mx).sum())
        if best is None or mass > best[1]:
            best = (z, mass, mx)
    if best is None:
        return None
    return best[0], best[2]



def refine_candidates(x, y, cfg, p0, q0, rounds=2):
    """Local zoom around a candidate MAP; returns extra (p, q, slope) candidates."""
    n = x.size
    plo = max(cfg['p'][0] + 1e-3, 0.02)
    dp0 = (cfg['p'][1] - plo) / cfg['np5']
    dq0 = (cfg['q'][1] - cfg['q'][0]) / max(cfg['grid4'][1], cfg['nq'] // 8)
    scale = max(1.0, float(np.max(np.abs(y))))
    out = []
    cp, cq = float(p0), float(q0)
    for it in range(rounds):
        f = 6.0 / (1.7 ** it)
        dp = max(dp0 * f, 1e-5)
        dq = max(dq0 * f * 3.0, 1e-5)
        lo = max(plo, cp - dp); hi = min(cfg['p'][1], cp + dp)
        qlo = max(cfg['q'][0], cq - dq); qhi = min(cfg['q'][1], cq + dq)
        if hi <= lo or qhi <= qlo:
            break
        if n == 5:
            c = candidates_n5(x, y, cfg, qlo, qhi,
                              pgrid=np.linspace(lo, hi, 220), nq=900)
        else:
            pg = np.linspace(lo, hi, 200)
            qg = np.linspace(qlo, qhi, 220)
            P, Q = np.meshgrid(pg, qg, indexing='ij')
            m = Q > P + 1e-4
            c = (P[m], Q[m])
        if c is None or c[0].size == 0:
            break
        p, q = c[0], c[1]
        sl = c[2] if len(c) == 3 else None
        out.append((p, q, sl))
        th, rn, lad, good = lstsq_theta(design(x, p, q), y)
        lw = logprior(p, q, th, cfg) + loglike(rn, scale, cfg)
        if sl is not None:
            lw = lw - np.log(np.maximum(sl, 1e-300))
            good = good & np.isfinite(sl)
        lw[~good] = -np.inf
        if not np.any(np.isfinite(lw)):
            break
        k = int(np.argmax(lw))
        cp, cq = float(p[k]), float(q[k])
    if not out:
        return None
    have_sl = all(o[2] is not None for o in out)
    P = np.concatenate([o[0] for o in out])
    Q = np.concatenate([o[1] for o in out])
    S = np.concatenate([o[2] for o in out]) if have_sl else None
    return P, Q, S

# ---------------------------------------------------------------------------
# two-term fallback for data the three-term model cannot explain plausibly
# ---------------------------------------------------------------------------
def _design2(x, p, q):
    lx = np.log(x)
    A = np.empty((p.size, x.size, 3))
    A[:, :, 0] = 1.0
    A[:, :, 1] = np.exp(np.outer(p, lx))
    A[:, :, 2] = np.exp(np.outer(q, lx))
    return A


def _ls2(A, y):
    Q, R = np.linalg.qr(A)
    Qty = np.einsum('kni,n->ki', Q, y)
    with np.errstate(all='ignore'):
        try:
            th = np.linalg.solve(R, Qty[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            th = np.einsum('kij,kj->ki', np.linalg.pinv(R), Qty)
        rn = np.linalg.norm(np.einsum('knj,kj->kn', A, th) - y, axis=1)
    ok = (np.isfinite(th).all(axis=1) & np.isfinite(rn)
          & (np.abs(th).max(axis=1) < 1e8))
    return th, np.where(ok, rn, np.inf)


def _scan2(x, y, plo, phi, qlo, qhi, gp, gq):
    pg = np.linspace(plo, phi, gp)
    qg = np.linspace(qlo, qhi, gq)
    P, Q = np.meshgrid(pg, qg, indexing='ij')
    m = Q > P + 1e-3
    p = P[m]; q = Q[m]
    th, rn = _ls2(_design2(x, p, q), y)
    return p, q, th, rn


def two_term_fit(x, y, nstart=8, rounds=11, gsize=(150, 180)):
    """Least-squares fit of y = t + a*x**p + b*x**q with multi-start grid zoom.

    Used only when the three-term model is rejected by the prior-strain test;
    a smooth two-power model averages an oscillatory adverse component instead
    of chasing it, which recovers the limit far more accurately there.
    """
    p, q, th, rn = _scan2(x, y, 0.05, 6.5, 0.10, 10.0, *gsize)
    if not np.any(np.isfinite(rn)):
        return None
    best = None
    starts = np.argsort(rn)[:nstart]
    for k in starts:
        cp, cq = float(p[k]), float(q[k])
        wp = 6.5 / gsize[0] * 4.0
        wq = 10.0 / gsize[1] * 4.0
        for _ in range(rounds):
            wp *= 0.4; wq *= 0.4
            pp, qq, tt, rr = _scan2(x, y, max(0.02, cp - wp), cp + wp,
                                    max(0.03, cq - wq), cq + wq, 33, 33)
            j = int(np.argmin(rr))
            cp, cq = float(pp[j]), float(qq[j])
            if np.isfinite(rr[j]) and (best is None or rr[j] < best[0]):
                best = (float(rr[j]), cp, cq, tt[j].copy())
    if best is None:
        return None
    return float(best[3][0])


# ---------------------------------------------------------------------------
# robust fallbacks (used only if the Bayesian path fails)
# ---------------------------------------------------------------------------
def _richardson_scan(x, y):
    """Best single-power fit y = t + a*x^p over a dense p scan (all pairs)."""
    o = np.argsort(x); xs = x[o]; ys = y[o]
    n = xs.size
    best = None
    grid = np.exp(np.linspace(math.log(0.02), math.log(20.0), 1200))
    for i in range(n - 1):
        for j in range(i + 1, n):
            x1, x2, y1, y2 = xs[i], xs[j], ys[i], ys[j]
            if y1 == y2:
                continue
            pp = grid
            with np.errstate(all='ignore'):
                num = y1 * x2 ** pp - y2 * x1 ** pp
                den = x2 ** pp - x1 ** pp
                t = np.where(np.abs(den) > 1e-12, num / den, np.nan)
            t = t[np.isfinite(t)]
            if t.size == 0:
                continue
            v = float(np.median(t))
            if best is None or abs(v - ys[n - 1]) < abs(best - ys[n - 1]):
                best = v
    return best


def _poly_const(x, y):
    o = np.argsort(x); xs = x[o]; ys = y[o]
    deg = xs.size - 1
    try:
        cf = np.polyfit(xs, ys, deg)
        v = float(cf[-1])
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def fallback(x, y):
    for fn in (_poly_const, _richardson_scan):
        try:
            v = fn(x, y)
            if v is not None and math.isfinite(v):
                return v
        except Exception:
            continue
    return float(y[int(np.argmin(x))])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def estimate(x, y, cfg=None):
    cfg = dict(CFG, **(cfg or {}))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    o = np.argsort(x)
    x = x[o]; y = y[o]
    strain = None
    try:
        out = posterior_estimate(x, y, cfg)
        if out is not None:
            z, strain = out
        else:
            z = None
    except Exception:
        z = None
    # The three-term model explains the scored family exactly; when no
    # parameter set both fits the data and stays inside the plausible range the
    # data is from a different (adverse) regime, and a smooth two-power model
    # recovers the limit much better than forcing the three-term form.
    if (z is not None and math.isfinite(z) and strain is not None
            and strain < cfg['strain_cut']):
        try:
            z2 = two_term_fit(x, y)
        except Exception:
            z2 = None
        if z2 is not None and math.isfinite(z2):
            z = z2
    if z is None or not math.isfinite(z):
        z = fallback(x, y)
    if not math.isfinite(z):
        z = float(y[0])
    # guard against absurd values that would score worse than a trivial guess
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo
    if span > 0 and not (lo - 50.0 * span <= z <= hi + 50.0 * span):
        z = fallback(x, y)
        if not math.isfinite(z):
            z = float(y[0])
    return float(z)


def read_input(path):
    with open(path, 'r') as fh:
        obj = json.load(fh)
    x = [float(v) for v in obj['x']]
    y = [float(v) for v in obj['y']]
    return x, y


def write_output(path, value):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    txt = json.dumps({"estimate": float(value)}, allow_nan=False)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(txt)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main(argv):
    inp = argv[1] if len(argv) > 1 else '/workspace/inputs/input.json'
    out = argv[2] if len(argv) > 2 else '/workspace/result/output.json'
    value = 0.0
    try:
        x, y = read_input(inp)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError('length mismatch')
        if not all(math.isfinite(v) and v > 0 for v in x):
            raise ValueError('bad x')
        if not all(math.isfinite(v) for v in y):
            raise ValueError('bad y')
        # de-duplicate defensively (spec guarantees distinct x)
        pairs = {}
        for xi, yi in zip(x, y):
            pairs.setdefault(xi, yi)
        xs = np.array(list(pairs.keys()), float)
        ys = np.array(list(pairs.values()), float)
        if xs.size >= 3:
            value = estimate(xs, ys)
        else:
            value = float(ys[int(np.argmin(xs))])
    except Exception:
        try:
            value = fallback(np.array(x, float), np.array(y, float))
        except Exception:
            value = 0.0
    if not math.isfinite(value):
        value = 0.0
    try:
        write_output(out, value)
    except Exception:
        sys.stderr.write('failed to write output\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
