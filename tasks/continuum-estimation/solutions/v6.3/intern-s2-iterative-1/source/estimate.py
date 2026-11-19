"""Estimate the limiting value t as x -> 0 from positive observations y(x).

Combines several independent extrapolation models and returns a robust
aggregate. Models all run from a single vectorized grid computation:

  1. polynomial extrapolation to x = 0 (degree n-1 Vandermonde)
  2. polynomial of degree n-2 (a non-interpolating second opinion)
  3. local two-point Richardson from the log-log slope of |dy/dx|
  4. two-term power series  t + c1*x^p + c2*x^q  on a log-spaced (p,q) grid,
     with the coefficients solved analytically from the 3x3 normal equations
  5. optional 3-term refinement when the 2-term model beats the polynomial
     by a decisive margin.

Why the "non-degenerate region" matters
---------------------------------------
The grid spans exponents down to pmin = 0.015. For a small exponent, x^p is
nearly constant over the observed x range, so the two-term model collapses onto
a constant and can fit the data exactly (residual == 0). The global argmin of
the grid residual therefore often lives in that p -> 0 region and returns a
t that is essentially a weighted mean of the observations, not the limit.
Exponents below ND_LO are discarded when picking the grid argmin/median; the
low-exponent slots are still computed and used for the residual band only when
they are not the sole survivors (NBAND_MIN guard).
"""

import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# single thread, no BLAS threading
# --------------------------------------------------------------------------
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_NUM_THREADS'] = '1'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'


DEFAULT_INPUT = '/workspace/inputs/input.json'
DEFAULT_OUTPUT = '/workspace/result/output.json'

# Grid: log-spaced exponents.
GRID_N = 160
GRID_PMIN = 0.015
GRID_PMAX = 80.0

BAND = 2.0               # residual band, measured against the grid minimum
P_MIN_FLOOR = 0.1        # candidate exponents below this are discarded
ND_LO = 1.0              # lower exponent floor for the non-degenerate candidates
ND_HI = 12.0             # upper exponent floor for the non-degenerate candidates
NBAND_MIN = 2            # band-derived (and 3-term) candidates need this many
                         # surviving slots; a single survivor is almost always
                         # the degenerate p -> 0 slot
TRIM = 0.25              # fraction of extreme candidates to trim before mean
POLY_N2 = False          # degree-(n-2) polynomial as a second opinion. It is a
                         # genuine second guess, but on the public dev set it
                         # consistently drags the aggregate away from the
                         # power-series answer, so it is left off.
GATE = 0.1               # the 3-term refinement only fires if the 2-term
                         # residual is below this factor of the genuine
                         # (degree n-2) polynomial residual
KC = 3                   # number of seeds kept for the 3-term refinement
DEGEN_FLOOR = 1e-12      # relative to the y range: below this the 2-term grid
                         # has collapsed onto the constant term and its
                         # minimum is meaningless


def _sort(x, y):
    idx = np.argsort(x)
    return x[idx], y[idx]


# --------------------------------------------------------------------------
# two-term power series: vectorized grid via the analytic 3x3 solution
# --------------------------------------------------------------------------
def two_term_grid(x, y, n=GRID_N, pmin=GRID_PMIN, pmax=GRID_PMAX):
    """Return (R, T, c1, c2, xp) over the full n x n grid.

    R and T are n x n; only the off-diagonal entries are finite (the diagonal
    p == q is rank-deficient and left as NaN/inf). c1 and c2 are flat arrays
    indexed by the off-diagonal mask.
    """
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    n_ = len(xs)
    xp = np.logspace(np.log10(pmin), np.log10(pmax), n)

    X = xs[:, None] ** xp[None, :]
    S0 = X.sum(0)
    S2 = (X * X).sum(0)
    Sy = X.T @ ys
    Syy = ys @ ys
    Spq = X.T @ X

    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    mask = ii != jj
    pi, pj = ii[mask], jj[mask]

    a = S0[pi]
    b = S0[pj]
    A = S2[pi]
    B = S2[pj]
    C = Spq[pi, pj]

    b0 = ys.sum()
    b1 = Sy[pi]
    b2 = Sy[pj]

    d00 = A * B - C * C
    d01 = b * C - a * B
    d02 = a * C - b * A
    d11 = n_ * B - b * b
    d12 = a * b - n_ * C
    d22 = n_ * A - a * a

    det = n_ * d00 + a * d01 + b * d02

    with np.errstate(divide='ignore', invalid='ignore'):
        c0 = (d00 * b0 + d01 * b1 + d02 * b2) / det
        c1 = (d01 * b0 + d11 * b1 + d12 * b2) / det
        c2 = (d02 * b0 + d12 * b1 + d22 * b2) / det

    res2 = (Syy
            - 2.0 * (b0 * c0 + b1 * c1 + b2 * c2)
            + (c0 * (n_ * c0 + a * c1 + b * c2)
               + c1 * (a * c0 + A * c1 + C * c2)
               + c2 * (b * c0 + C * c1 + B * c2)))
    res = np.sqrt(np.maximum(res2, 0.0))

    R = np.full((n, n), np.inf)
    T = np.full((n, n), np.nan)
    R[pi, pj] = res
    T[pi, pj] = c0
    return R, T, c1, c2, xp


def _band_vals(res, t, xp, band=BAND, pminf=P_MIN_FLOOR):
    """Candidate t values whose residual lies within `band` of the minimum."""
    ii, jj = np.where(np.isfinite(res))
    rf = res[ii, jj]
    tf = t[ii, jj]
    pxp, qxp = xp[ii], xp[jj]
    rmin = rf.min()
    m = (rf <= band * rmin) & (pxp >= pminf) & (qxp >= pminf)
    return tf[m], rf[m], pxp[m], qxp[m], rmin


def _nondeg(res, t, xp, lo=ND_LO, hi=ND_HI):
    """Slots whose exponents are safely away from the degenerate p -> 0 / huge
    regions, where the basis collapses onto the constant term."""
    ii, jj = np.where(np.isfinite(res))
    rf = res[ii, jj]
    tf = t[ii, jj]
    pxp, qxp = xp[ii], xp[jj]
    m = (pxp >= lo) & (qxp >= lo) & (pxp <= hi) & (qxp <= hi)
    return tf[m], rf[m], pxp[m], qxp[m]


def two_term_candidates(x, y, n=GRID_N, pmin=GRID_PMIN, pmax=GRID_PMAX,
                        band=BAND, pminf=P_MIN_FLOOR):
    """Return a dict of several 2-term derived estimates."""
    res, t, c1, c2, xp = two_term_grid(x, y, n, pmin, pmax)
    vals, rr, pp, qq, rmin = _band_vals(res, t, xp, band, pminf)
    out = {
        'argmin': (vals[np.argmin(rr)] if vals.size else np.nan),
        'median': (np.median(vals) if vals.size else np.nan),
        'weighted': (np.average(vals, weights=1.0 / np.maximum(rr, 1e-300))
                     if vals.size else np.nan),
        'n_band': vals.size,
        'rmin': rmin,
        'xp': xp,
        'res': res,
        't': t,
        'p': pp[np.argmin(rr)] if vals.size else np.nan,
        'q': qq[np.argmin(rr)] if vals.size else np.nan,
    }
    return out


# --------------------------------------------------------------------------
# polynomial extrapolation
# --------------------------------------------------------------------------
def poly_ext(x, y):
    """Extrapolate to x=0 with a degree-(n-1) polynomial.

    Vandermonde [1, x, x^2, ...] in increasing order, so the constant term is
    the value at zero.
    """
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    V = np.vander(xs, N=len(xs), increasing=True)
    return np.linalg.lstsq(V, ys, rcond=None)[0][0]


def poly_lo(x, y):
    """Degree-(n-2) polynomial fit: a non-interpolating second opinion whose
    residual is a usable reference for gating the 3-term refinement."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    n_ = len(xs)
    d = max(1, n_ - 2)
    V = np.vander(xs, N=d + 1, increasing=True)
    co = np.linalg.lstsq(V, ys, rcond=None)[0]
    return co[0], np.linalg.norm(ys - V @ co)


# --------------------------------------------------------------------------
# local dominant order + two-point Richardson
# --------------------------------------------------------------------------
def rich2(x, y):
    """Two-point Richardson from the local log-log slope of |dy/dx|."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    dy = np.gradient(ys, xs)
    s = np.abs(dy)
    # avoid log(0) and sign flips
    good = s > 1e-300
    if good.sum() < 3:
        return np.nan
    xg, yg, dg = xs[good], ys[good], s[good]
    # |dy/dx| ~ x^(p-1) on a log-log plot, so the dominant order is 1 + slope
    p = 1.0 + np.log(dg[:-1] / dg[1:]) / np.log(xg[:-1] / xg[1:])
    p = np.clip(p, 0.1, 30.0)
    ts = yg[1:] + (yg[1:] - yg[:-1]) / ((xg[:-1] / xg[1:]) ** p - 1.0)
    return float(np.median(ts)) if ts.size else np.nan


# --------------------------------------------------------------------------
# 3-term refinement
# --------------------------------------------------------------------------
def three_term(x, y, p, q, r):
    """Fit t + c1*x^p + c2*x^q + c3*x^r by lstsq; return t or NaN."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    A = np.stack([np.ones_like(xs), xs ** p, xs ** q, xs ** r], axis=1)
    co, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return co[0]


def three_term_candidate(x, y, rr, pp, qq, k=5):
    """Refine the best k power-series slots with a third exponent."""
    order = np.argsort(rr[:k])
    out = []
    for s in order[:min(k, len(rr))]:
        # third exponent: a value below both, so it captures the leading term
        for r in (pp[s] * 0.5, (pp[s] + qq[s]) / 3.0, min(pp[s], qq[s]) * 0.6):
            v = three_term(x, y, pp[s], qq[s], r)
            if np.isfinite(v):
                out.append(v)
    return out


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def estimate(x, y, grid_n=GRID_N, band=BAND, pminf=P_MIN_FLOOR,
             nd_lo=ND_LO, nd_hi=ND_HI, nb_min=NBAND_MIN, trim=TRIM,
             gate=GATE, kc=KC):
    """Robust aggregate of several independent extrapolation estimators."""
    vals = []

    # 1. polynomial extrapolation, degree n-1 and n-2
    try:
        v = poly_ext(x, y)
        if np.isfinite(v):
            vals.append(('poly.n-1', float(v)))
        if POLY_N2:
            v, _ = poly_lo(x, y)
            if np.isfinite(v):
                vals.append(('poly.n-2', float(v)))
    except Exception:
        pass

    # 2. two-term power series on the grid
    try:
        tt = two_term_candidates(x, y, n=grid_n, band=band, pminf=pminf)
        nd_vals, nd_rr, nd_pp, nd_qq = _nondeg(tt['res'], tt['t'], tt['xp'],
                                               lo=nd_lo, hi=nd_hi)

        # non-degenerate region: the reliable core of the grid
        if nd_vals.size:
            vals.append(('2term.nd_argmin', float(nd_vals[np.argmin(nd_rr)])))
            vals.append(('2term.nd_median', float(np.median(nd_vals))))
            vals.append(('2term.nd_weighted',
                         float(np.average(
                             nd_vals, weights=1.0 / np.maximum(nd_rr, 1e-300)))))

        # degeneracy guard: when the grid's best residual is (relative to the
        # range of y) essentially zero, the power-series model is not
        # identifiable and the surviving slots collapse onto the constant term.
        # Their t values must not be trusted. The argmin exponent is below
        # nd_lo precisely in that case.
        argmin_p = float(tt['p'])
        yscale = max(1.0, np.ptp(y))
        degenerate = (tt['rmin'] < DEGEN_FLOOR * yscale) and (argmin_p < nd_lo)

        # residual band: usable only when several slots survive (a single
        # survivor is almost always the degenerate p->0 slot; reused below)
        tf = rr = pp = qq = np.array([], dtype=float)
        if not degenerate and tt['n_band'] >= nb_min:
            tf, rr, pp, qq, _ = _band_vals(tt['res'], tt['t'], tt['xp'],
                                           band, pminf)
            if tf.size >= nb_min:
                vals.append(('2term.band_median', float(np.median(tf))))
                vals.append(('2term.band_weighted',
                             float(np.average(
                                 tf, weights=1.0 / np.maximum(rr, 1e-300)))))
        rmin = float(tt['rmin'])

        # 3. local Richardson
        v = rich2(x, y)
        if np.isfinite(v):
            vals.append(('rich2', float(v)))

        # 4. 3-term refinement, only when the 2-term model beats the
        #    polynomial fit by a decisive margin (gated on a genuine residual)
        _, lores = poly_lo(x, y)
        if not degenerate and lores > 1e-15 and rmin < gate * lores \
                and tf.size >= nb_min:
            for v in three_term_candidate(x, y, rr, pp, qq, k=kc):
                if np.isfinite(v):
                    vals.append(('3term', float(v)))
    except Exception:
        pass

    cand = np.array([c[1] for c in vals], dtype=float)
    # robust aggregate: trim the most extreme `trim` fraction then mean
    cand.sort()
    lo = int(len(cand) * trim)
    hi = len(cand) - lo
    if hi > lo:
        return float(np.mean(cand[lo:hi]))
    return float(np.mean(cand)) if cand.size else 0.0


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main(argv):
    in_path = argv[1] if len(argv) > 1 else DEFAULT_INPUT
    out_path = argv[2] if len(argv) > 2 else DEFAULT_OUTPUT

    with open(in_path) as f:
        data = json.load(f)

    x = np.asarray(data['x'], dtype=float)
    y = np.asarray(data['y'], dtype=float)

    est = estimate(x, y)
    if not np.isfinite(est):
        # should never happen: every candidate is gated on finiteness, and the
        # fallback below is a plain mean of finite numbers
        try:
            v = poly_ext(x, y)
            est = float(v) if np.isfinite(v) else 0.0
        except Exception:
            est = 0.0

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'estimate': float(est)}, f)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
