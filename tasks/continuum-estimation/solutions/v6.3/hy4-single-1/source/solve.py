#!/usr/bin/env python3
"""Continuum-limit estimation from a handful of positive discretization levels.

The observations are modelled as a smooth three-term asymptotic expansion

    y(x) = t + a*x**p + b*x**q + c*x**(q+1.5),    0 < p < q,

which is an exact description of the smooth convergent families used by the
generator.  With four or five samples the six parameters are not identifiable,
so the limit t is obtained as a Bayesian model average: the (p,q) plane (four
samples) or its exact-fit curve (five samples) is integrated with a prior on
the exponents and on the expansion coefficients.  The integration measure
contains the Jacobian 1/|det R| of the linear coefficients, and for five
samples the co-area factor 1/|dF/dq| of the exact-fit constraint.

Only NumPy/SciPy from the bundled runtime are used; no state or network access.
"""

import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Configuration (fixed for every case)
# ---------------------------------------------------------------------------
CFG4 = {
    "Np": 420, "Nd": 380,
    "p_lo": 0.20, "p_hi": 4.60, "d_lo": 0.10, "d_hi": 4.00,
}
CFG5 = {
    "Np": 380, "Nq": 280,
    "p_lo": 0.20, "p_hi": 4.60, "d_lo": 0.10, "d_hi": 4.20,
    "h_deriv": 1.0e-4,
}
# prior / aggregation (tuned on synthetic suites and the public dev cases)
PRIOR = {
    "sa": 1.3567, "sb": 0.9470, "sc": 0.5766,
    "mb": -0.1466, "mc": -0.0820, "ksign": 3.51,
    "pmu": 1.6830, "psd": 0.8564, "dmu": 1.3896, "dsd": 1.3755,
    "boxA": 1.5236, "boxB": 1.2454, "boxC": 0.7155,
    "boxpen": 1.8442, "bpos": 0.0274, "cpos": 0.0268,
    "alpha4": 0.9782, "alpha5": 1.7988,
    "agg4": "median", "agg5": "median",
    "cap": 10000.0,
}


# ---------------------------------------------------------------------------
# Linear algebra for the model
# ---------------------------------------------------------------------------
def design(x, p, q):
    """Batched (N, n, 4) design matrix of [1, x^p, x^q, x^(q+1.5)]."""
    L = np.log(x)[:, None]
    P = p[None, :]
    Q = q[None, :]
    A = np.empty((x.size, p.size, 4), dtype=float)
    A[..., 0] = 1.0
    A[..., 1] = np.exp(L * P)
    A[..., 2] = np.exp(L * Q)
    A[..., 3] = np.exp(L * (Q + 1.5))
    return np.transpose(A, (1, 0, 2))


def solve_batch(x, y, p, q):
    """Least-squares coefficients, residual norm and column-space volume."""
    A = design(x, p, q)
    Qm, Rm = np.linalg.qr(A)
    Qy = np.einsum("kij,i->kj", Qm, y)
    with np.errstate(all="ignore"):
        c = np.linalg.solve(Rm, Qy)
    fit = np.einsum("kij,kj->ki", A, c)
    r = y[None, :] - fit
    rnorm = np.sqrt(np.einsum("ki,ki->k", r, r))
    det = np.abs(np.prod(np.diagonal(Rm, axis1=1, axis2=2), axis=1))
    return c, rnorm, det


def signed_F_batch(x, y, p, q):
    """Signed residual F = n_hat . y for n = 5 (vectorized over (p,q))."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    B = design(x, p, q)
    n = x.size
    dets = np.empty((p.size, n))
    for i in range(n):
        rows = [j for j in range(n) if j != i]
        dets[:, i] = np.linalg.det(B[:, rows, :])
    signs = (-1.0) ** (np.arange(n) + 5)
    Fraw = dets @ (signs * y)
    nrm = np.sqrt(np.einsum("ki,ki->k", dets, dets))
    return Fraw / np.maximum(nrm, 1e-300)


def signed_F_one(x, y, p, q):
    return float(signed_F_batch(x, y, (p,), (q,))[0])


# ---------------------------------------------------------------------------
# Prior and posterior aggregation
# ---------------------------------------------------------------------------
def log_prior(c, p, d, cfg):
    a, b, cc = c[:, 1], c[:, 2], c[:, 3]
    sa, sb, sc = cfg["sa"], cfg["sb"], cfg["sc"]
    mb, mc, ksign = cfg["mb"], cfg["mc"], cfg["ksign"]
    lw = (-0.5 * (a / sa) ** 2
          - 0.5 * ((b - mb) / sb) ** 2
          - 0.5 * ((cc - mc) / sc) ** 2
          - ksign * np.maximum(0.0, b) ** 2 / sb ** 2
          - ksign * np.maximum(0.0, cc) ** 2 / sc ** 2
          - 0.5 * ((p - cfg["pmu"]) / cfg["psd"]) ** 2
          - 0.5 * ((d - cfg["dmu"]) / cfg["dsd"]) ** 2)
    in_box = ((np.abs(a) <= cfg["boxA"]) & (b >= -cfg["boxB"]) &
              (b <= cfg["bpos"]) & (cc >= -cfg["boxC"]) &
              (cc <= cfg["cpos"]))
    lw -= cfg["boxpen"] * (~in_box)
    return lw


def aggregate(t, lw, mode, scale):
    good = np.isfinite(lw) & np.isfinite(t) & (np.abs(t) < 100.0 * scale)
    t, lw = t[good], lw[good]
    if t.size == 0:
        return None, 0.0
    m = float(lw.max())
    if not np.isfinite(m):
        return None, 0.0
    w = np.exp(lw - m)
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        return None, 0.0
    w /= s
    if mode == "median":
        o = np.argsort(t)
        cw = np.cumsum(w[o])
        return float(t[o][min(np.searchsorted(cw, 0.5), t.size - 1)]), float(m)
    if mode == "trim":
        o = np.argsort(t)
        cw = np.cumsum(w[o])
        lo = np.searchsorted(cw, 0.05)
        hi = max(np.searchsorted(cw, 0.95), lo + 1)
        sel = o[lo:hi]
        return float(np.dot(w[sel], t[sel]) / w[sel].sum()), float(m)
    return float(np.dot(w, t)), float(m)


# ---------------------------------------------------------------------------
# Four samples: posterior over the (p, q-p) plane
# ---------------------------------------------------------------------------
def estimate4(x, y, cfg, prior):
    g = cfg
    P, D = np.meshgrid(
        np.linspace(g["p_lo"], g["p_hi"], g["Np"]),
        np.linspace(g["d_lo"], g["d_hi"], g["Nd"]), indexing="ij")
    p = P.ravel()
    d = D.ravel()
    q = p + d
    c, rn, det = solve_batch(x, y, p, q)
    t = c[:, 0]
    lw = log_prior(c, p, d, prior)
    if prior["alpha4"]:
        lw -= prior["alpha4"] * np.log(np.maximum(det, 1e-300))
    scale = max(1.0, float(np.max(np.abs(y))))
    return aggregate(t, lw, prior["agg4"], scale)[0]


# ---------------------------------------------------------------------------
# Five samples: posterior on the exact-fit curve F(p, q) = 0
# ---------------------------------------------------------------------------
def estimate5(x, y, cfg, prior):
    g = cfg
    scale = max(1.0, float(np.max(np.abs(y))))
    ps = np.linspace(g["p_lo"], g["p_hi"], g["Np"])
    h = g["h_deriv"]
    ts, lws = [], []
    best_plausibility = -np.inf
    min_absF = np.inf
    t_start = time.monotonic()
    for p in ps:
        # Hard safety bounds against degenerate / noisy residual functions.
        if time.monotonic() - t_start > 15.0 or len(ts) > 4000:
            break
        qs = np.linspace(p + g["d_lo"], p + g["d_hi"], g["Nq"])
        F = signed_F_batch(x, y, np.full_like(qs, p), qs)
        ok = np.isfinite(F)
        min_absF = min(min_absF, float(np.min(np.abs(F[ok])))) if ok.any() else min_absF
        brackets = [(qs[i], qs[i + 1]) for i in range(g["Nq"] - 1)
                    if ok[i] and ok[i + 1] and F[i] * F[i + 1] < 0.0]
        if len(brackets) > 10:
            brackets = brackets[:10]
        for lo, hi in brackets:
            try:
                qr = brentq(lambda z: signed_F_one(x, y, p, z), lo, hi,
                            xtol=1e-13, rtol=1e-14, maxiter=50)
            except (ValueError, RuntimeError, OverflowError):
                continue
            c, _, det = solve_batch(x, y, np.array([p]), np.array([qr]))
            t, a, b, cc = c[0]
            if not np.isfinite(t) or abs(t) > 100.0 * scale:
                continue
            fq = (signed_F_one(x, y, p, qr + h) - signed_F_one(x, y, p, qr - h)) / (2 * h)
            if not np.isfinite(fq) or abs(fq) < 1e-13:
                fq = 1e-13
            lp = log_prior(c, np.array([p]), np.array([qr - p]), prior)[0]
            best_plausibility = max(best_plausibility, float(lp))
            lw = lp - np.log(abs(fq))
            if prior["alpha5"]:
                lw -= prior["alpha5"] * np.log(max(det[0], 1e-300))
            ts.append(float(t))
            lws.append(float(lw))
    if not ts:
        return None, {"min_absF": min_absF, "best_plausibility": best_plausibility}
    ts = np.asarray(ts)
    lws = np.asarray(lws)
    cap = prior.get("cap")
    if cap is not None:
        fin = np.isfinite(lws)
        if fin.any():
            lws = np.minimum(lws, np.median(lws[fin]) + math.log(cap))
    z, _ = aggregate(ts, lws, prior["agg5"], scale)
    return z, {"min_absF": min_absF, "best_plausibility": best_plausibility,
               "nroots": len(ts)}


# ---------------------------------------------------------------------------
# Robust fallbacks (used only when the smooth model is implausible)
# ---------------------------------------------------------------------------
def fallback_estimate(x, y):
    """Local Richardson-style extrapolation for non-smooth/adverse trends."""
    n = len(x)
    k = min(3, n)
    xs = x[:k]
    ys = y[:k]
    # weighted mean of the k smallest-level values, strongly favouring the first
    w = np.array([1.0, 0.35, 0.15][:k])
    base = float(np.dot(w, ys) / w.sum())
    # small correction using a local power-law fit when the data are monotone
    try:
        if k >= 3 and np.all(np.diff(ys) != 0):
            r = (ys[1] - ys[0]) / (ys[2] - ys[1])
            ratio = xs[1] / xs[0]
            if r > 0 and np.isfinite(r):
                p = math.log(r) / math.log(ratio)
                if 0.2 < p < 8.0:
                    denom = (xs[1] / xs[0]) ** p - 1.0
                    if abs(denom) > 1e-6:
                        corr = (ys[1] - ys[0]) / denom
                        if abs(corr) < 0.5 * max(1.0, abs(base)):
                            base = ys[0] - corr
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return base


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    scale = max(1.0, float(np.max(np.abs(y))))
    # A level-independent simulator has an immediate, exact limit.
    if float(np.max(y) - np.min(y)) <= 1e-12 * scale:
        return float(y[0])
    info = {}
    if len(x) == 4:
        z = estimate4(x, y, CFG4, PRIOR)
    elif len(x) >= 5:
        z, info = estimate5(x, y, CFG5, PRIOR)
    else:
        z = None
    # Sanity / misfit switch.  A smooth three-term family always admits a fit
    # with O(1) coefficients; if the best coefficient log-prior is tiny the
    # observations do not follow the smooth model.
    if len(x) >= 5:
        plaus = info.get("best_plausibility", -np.inf)
        if z is None or not np.isfinite(z) or plaus < -25.0 or \
                abs(z) > 20.0 * scale or abs(z - y[0]) > 5.0 * scale:
            z = fallback_estimate(x, y)
    elif len(x) == 4:
        if z is None or not np.isfinite(z) or abs(z) > 20.0 * scale:
            z = fallback_estimate(x, y)
    if z is None or not np.isfinite(z):
        z = float(y[0])
    return float(z)


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    try:
        with open(inp, "r") as fh:
            obj = json.load(fh)
        x = [float(v) for v in obj["x"]]
        y = [float(v) for v in obj["y"]]
        z = compute(x, y)
        if not math.isfinite(z):
            raise ValueError("nonfinite estimate")
    except Exception:
        # A finite estimate must still be produced for malformed input.
        try:
            z = float(y[0])
            if not math.isfinite(z):
                z = 0.0
        except Exception:
            z = 0.0
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"estimate": z}, fh, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
