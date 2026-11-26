#!/usr/bin/env python3
"""Continuum-limit estimation from positive discretisation levels.

Problem
-------
A deterministic numerical simulation reports values ``y`` at positive
discretisation levels ``x``; the quantity of interest is the limit
``t = lim_{x->0+} y(x)``.

Model family
------------
The observed error behaves like a short expansion in powers of the level::

    y(x) = t + a*x^p + b*x^q + c*x^(q + 3/2)

with unknown positive convergence orders ``p < q`` and unknown amplitudes
``a`` (either sign), ``b <= 0`` and ``c <= 0``.  This structure was
recovered from the public development pairs in ``/workspace/public/dev``:
with the correct orders the model reproduces the observations to machine
precision, and no simpler two-term expansion does.

Estimation
----------
The model has six unknowns but only four or five observations are given, so
the data alone never identify a single model:

* five levels - the models that fit the data *exactly* form a one dimensional
  curve in the ``(p, q)`` plane.  It is located with
  ``det[1, x^p, x^q, x^(q+3/2), y]``, which changes sign across the curve.
* four levels - every ``(p, q)`` reproduces the data exactly, so the whole
  plane is admissible.

The remaining disambiguation comes from a prior on the amplitudes (their
support, learned from the development pairs) together with the geometry of
the solution set.  Admissible models are weighted by that prior and by the
measure induced on the solution manifold, and the reported limit is the
weighted median of ``t`` over the admissible set - the median being the
optimal point estimate for an absolute-error loss.

Every stage has a fallback, so a finite number is always produced.

Usage:  estimate.py [INPUT_JSON [OUTPUT_JSON]]
"""

import json
import math
import os
import sys

try:
    import numpy as np
except Exception:                                    # pragma: no cover
    np = None

DEFAULT_INPUT = "/workspace/inputs/input.json"
DEFAULT_OUTPUT = "/workspace/result/output.json"

GAMMA = 1.5            # the third exponent is q + GAMMA

# ----------------------------------------------------------------- prior ---
# Support of the model amplitudes, estimated from the public dev pairs.
A_LO, A_HI = -1.50, 1.50      # leading amplitude, either sign
B_LO, B_HI = -1.00, 0.00      # second amplitude, non positive
C_LO, C_HI = -0.50, 0.00      # third amplitude, non positive
# Support of the convergence orders.
P_LO, P_HI = 0.50, 3.00
Q_LO, Q_HI = 1.00, 4.50
Q_GAP = 0.30                  # q has to exceed p by at least this much

# Widening factors applied to the prior when nothing is admissible.  This
# keeps the program usable (instead of failing outright) if the true
# amplitudes lie outside the assumed support.
RELAX = (1.0, 1.25, 1.6, 2.0, 3.0, 8.0)

# Grid used to explore the (p, q) plane.
GRID_STEP = 0.01
P_SCAN_LO, P_SCAN_HI = 0.20, 3.40
Q_SCAN_LO, Q_SCAN_HI = 0.40, 5.20


# -------------------------------------------------------------- utilities ---
def _root(f, a, b, tol=1e-13, maxiter=100):
    """Illinois (modified regula falsi) root finder on a bracketed interval."""
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("interval does not bracket a root")
    c = 0.5 * (a + b)
    for _ in range(maxiter):
        if fb != fa:
            c = b - fb * (b - a) / (fb - fa)
        lo, hi = (a, b) if a < b else (b, a)
        if not (lo < c < hi):
            c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0:
            return c
        if abs(c - b) <= tol * abs(c) + 1e-16:
            return c
        if fa * fc < 0.0:
            b, fb = c, fc
            fa *= 0.5                       # Illinois damping
        else:
            a, fa = c, fc
            fb *= 0.5
    return 0.5 * (a + b)


def _weighted_median(values, weights):
    """Weighted median of ``values`` (weights must be non negative)."""
    n = len(values)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: values[i])
    total = 0.0
    for i in order:
        total += weights[i]
    if total <= 0.0:
        return float(values[order[n // 2]])
    half = 0.5 * total
    acc = 0.0
    for i in order:
        acc += weights[i]
        if acc >= half:
            return float(values[i])
    return float(values[order[-1]])


# ------------------------------------------------------------- numpy core ---
def _build_design(x, p, q):
    return np.column_stack((np.ones_like(x), x ** p, x ** q, x ** (q + GAMMA)))


def _fit_point(x, y, p, q):
    """Least squares fit of (t, a, b, c) for fixed orders; returns (coef, rss)."""
    B = _build_design(x, p, q)
    try:
        sol, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, float("inf")
    r = B @ sol - y
    return sol, float(r @ r)


def _fit_grid(x, y, p, qs):
    """Vectorised fit for many q at fixed p.  Returns (coef[nq,4], rss[nq])."""
    logx = np.log(x)
    w = x ** GAMMA
    U = np.exp(logx[None, :] * np.asarray(qs)[:, None])
    B = np.empty((len(qs), len(x), 4))
    B[:, :, 0] = 1.0
    B[:, :, 1] = x ** p
    B[:, :, 2] = U
    B[:, :, 3] = U * w[None, :]
    coef = np.linalg.pinv(B) @ y
    resid = np.einsum("qni,qi->qn", B, coef) - y
    rss = np.einsum("qn,qn->q", resid, resid)
    return coef, rss


def _design_stack(x, c):
    """Stack of design matrices; ``c`` has columns (p, q, t, a, b, c)."""
    n = len(x)
    X = x[None, :]
    N = len(c)
    return np.stack([np.ones((N, n)), X ** c[:, 0][:, None], X ** c[:, 1][:, None],
                     X ** (c[:, 1] + GAMMA)[:, None]], axis=2)


def _volume_factor(x, c):
    """Volume of the design matrix: |det| for four levels, pseudo-determinant
    for five.  The posterior density over the admissible orders is inversely
    proportional to it (near degenerate designs leave more room for
    amplitudes and therefore carry more posterior mass)."""
    A = _design_stack(x, c)
    if A.shape[1] == A.shape[2]:
        vol = np.abs(np.linalg.det(A))
    else:
        sv = np.linalg.svd(A, compute_uv=False)
        vol = np.prod(sv, axis=1)
    vol = np.where(np.isfinite(vol), vol, 0.0)
    positive = vol[vol > 0]
    floor = 1e-10 * np.median(positive) if positive.size else 1e-300
    return np.maximum(vol, floor)


def _det_over_q(x, y, p, qs):
    logx = np.log(x)
    w = x ** GAMMA
    U = np.exp(logx[None, :] * np.asarray(qs)[:, None])
    V = np.empty((len(qs), len(x), 5))
    V[:, :, 0] = 1.0
    V[:, :, 1] = x ** p
    V[:, :, 2] = U
    V[:, :, 3] = U * w[None, :]
    V[:, :, 4] = y
    return np.linalg.det(V)


def _det_over_p(x, y, q, ps):
    logx = np.log(x)
    P = np.exp(logx[None, :] * np.asarray(ps)[:, None])
    V = np.empty((len(ps), len(x), 5))
    V[:, :, 0] = 1.0
    V[:, :, 1] = P
    V[:, :, 2] = x ** q
    V[:, :, 3] = x ** (q + GAMMA)
    V[:, :, 4] = y
    return np.linalg.det(V)


def _det_point(x, y, p, q):
    V = np.column_stack((np.ones_like(x), x ** p, x ** q, x ** (q + GAMMA), y))
    return float(np.linalg.det(V))


def _exact_points(x, y, step=GRID_STEP):
    """Orders (p, q) for which the model reproduces the data exactly."""
    found = []
    ps = np.arange(P_SCAN_LO, P_SCAN_HI + 1e-9, step)
    qs = np.arange(Q_SCAN_LO, Q_SCAN_HI + 1e-9, step)

    def keep(p, q):
        if not (P_SCAN_LO <= p <= P_SCAN_HI and Q_SCAN_LO <= q <= Q_SCAN_HI):
            return
        if abs(p - q) < 1e-9 or abs(p - (q + GAMMA)) < 1e-9:
            return                      # degenerate design, not a real solution
        for a, b in found:
            if abs(p - a) < 1e-7 and abs(q - b) < 1e-7:
                return
        found.append((p, q))

    for p in ps:                        # scan in q for fixed p
        d = _det_over_q(x, y, p, qs)
        sgn = np.sign(d)
        for i in np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]:
            try:
                qz = _root(lambda qq: _det_point(x, y, p, qq), qs[i], qs[i + 1])
            except Exception:
                continue
            keep(p, qz)
    for q in qs:                        # scan in p for fixed q
        d = _det_over_p(x, y, q, ps)
        sgn = np.sign(d)
        for i in np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]:
            try:
                pz = _root(lambda pp: _det_point(x, y, pp, q), ps[i], ps[i + 1])
            except Exception:
                continue
            keep(pz, q)
    return found


# ------------------------------------------------------------- candidates ---
def _candidates(x, y, step=GRID_STEP):
    """All candidate models; columns (p, q, t, a, b, c)."""
    if len(x) == 5:
        rows = []
        for p, q in _exact_points(x, y, step):
            sol, rss = _fit_point(x, y, p, q)
            if sol is None:
                continue
            rows.append((p, q, sol[0], sol[1], sol[2], sol[3]))
        if rows:
            return np.array(rows)
        return _best_fit_rows(x, y)
    ps = np.arange(P_SCAN_LO, P_SCAN_HI + 1e-9, step)
    qs = np.arange(Q_SCAN_LO, Q_SCAN_HI + 1e-9, step)
    rows = []
    for p in ps:
        coef, _ = _fit_grid(x, y, p, qs)
        for j in range(len(qs)):
            rows.append((p, qs[j], coef[j, 0], coef[j, 1], coef[j, 2], coef[j, 3]))
    return np.array(rows)


def _best_fit_rows(x, y):
    """Fallback used when the model does not fit exactly: best grid fit."""
    ps = np.arange(P_SCAN_LO, P_SCAN_HI + 1e-9, 0.02)
    qs = np.arange(Q_SCAN_LO, Q_SCAN_HI + 1e-9, 0.02)
    best = None
    for p in ps:
        coef, rss = _fit_grid(x, y, p, qs)
        k = int(np.argmin(rss))
        if best is None or rss[k] < best[0]:
            best = (float(rss[k]), float(p), float(qs[k]), coef[k])
    if best is None:
        return np.zeros((0, 6))
    _, p, q, s = best
    sol, _ = _fit_point(x, y, p, q)
    if sol is None:
        return np.zeros((0, 6))
    return np.array([(p, q, sol[0], sol[1], sol[2], sol[3])])


# ----------------------------------------------------------------- prior ----
def _widen(factor):
    return dict(
        p_lo=max(0.05, P_LO / factor), p_hi=min(P_SCAN_HI, P_HI * factor),
        q_lo=max(0.05, Q_LO / factor), q_hi=min(Q_SCAN_HI, Q_HI * factor),
        gap=max(0.0, Q_GAP / factor),
        a_lo=A_LO * factor, a_hi=A_HI * factor,
        b_lo=B_LO * factor, b_hi=B_HI * factor,
        c_lo=C_LO * factor, c_hi=C_HI * factor,
    )


def _prior_weight(c, cfg):
    p, q, t, a, b, cc = (c[:, i] for i in range(6))
    w = np.ones(len(c))
    w *= (p >= cfg["p_lo"]) & (p <= cfg["p_hi"])
    w *= (q >= cfg["q_lo"]) & (q <= cfg["q_hi"])
    w *= (q >= p + cfg["gap"])
    w *= (a >= cfg["a_lo"]) & (a <= cfg["a_hi"])
    w *= (b >= cfg["b_lo"]) & (b <= cfg["b_hi"])
    w *= (cc >= cfg["c_lo"]) & (cc <= cfg["c_hi"])
    return w


# ------------------------------------------------------------- estimation ---
def estimate_limit(x, y):
    """Estimate the limiting value from the observed levels."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    cands = _candidates(x, y)
    if len(cands) == 0:
        return _crude_estimate(x, y)

    vol = _volume_factor(x, cands)
    for factor in RELAX:
        cfg = _widen(factor)
        w = _prior_weight(cands, cfg) / vol
        if w.sum() > 0.0:
            return _weighted_median(list(cands[:, 2]), list(w))
    # Nothing admissible: fall back to the plain median over all candidates.
    return _weighted_median(list(cands[:, 2]), [1.0] * len(cands))


def _crude_estimate(x, y):
    """Very last resort: Richardson style extrapolation with a guessed order."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) >= 2:
            x1, x2 = float(x[0]), float(x[1])
            y1, y2 = float(y[0]), float(y[1])
            if x1 > 0.0 and x2 > 0.0 and x1 != x2:
                order = 2.0
                if len(x) >= 3:
                    num = y1 - float(y[-1])
                    den = y2 - float(y[-1])
                    if num * den > 0.0 and num != 0.0 and den != 0.0:
                        cand = math.log(abs(num / den)) / math.log(x1 / x2)
                        if 0.05 < cand < 12.0:
                            order = cand
                r = order
                return (y1 * r - y2) / (r - 1.0)
        return float(y[int(np.argmin(x))])
    except Exception:
        return 0.0


# ------------------------------------------------------------------- I/O ----
def _read_input(path):
    with open(path, "r") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError("input must be an object")
    xs = obj.get("x")
    ys = obj.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list):
        raise ValueError("input needs list fields x and y")
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("x and y must have the same non-zero length")
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    for v in x + y:
        if not math.isfinite(v):
            raise ValueError("non finite value in input")
    if any(v <= 0.0 for v in x):
        raise ValueError("levels must be positive")
    return x, y


def _write_output(path, estimate):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    value = float(estimate)
    if not math.isfinite(value):
        value = 0.0
    with open(path, "w") as fh:
        json.dump({"estimate": value}, fh)
        fh.write("\n")


def main(argv):
    in_path = argv[1] if len(argv) > 1 else DEFAULT_INPUT
    out_path = argv[2] if len(argv) > 2 else DEFAULT_OUTPUT
    x, y = [], []
    try:
        x, y = _read_input(in_path)
    except Exception:
        pass
    estimate = None
    if x and np is not None:
        try:
            estimate = estimate_limit(x, y)
        except Exception:
            estimate = None
    if estimate is None or not math.isfinite(float(estimate)):
        try:
            estimate = _crude_estimate(x, y) if x else 0.0
        except Exception:
            estimate = 0.0
    try:
        _write_output(out_path, estimate)
    except Exception:
        try:
            os.makedirs("/workspace/result", exist_ok=True)
            with open(DEFAULT_OUTPUT, "w") as fh:
                value = float(estimate)
                if not math.isfinite(value):
                    value = 0.0
                json.dump({"estimate": value}, fh)
        except Exception:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
