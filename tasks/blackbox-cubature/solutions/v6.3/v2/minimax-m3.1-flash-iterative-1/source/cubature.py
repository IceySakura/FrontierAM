"""Black-box cubature on [0,1]^d under a hard function-evaluation budget.

The integrand is unknown: only its values are available, and every value
costs one unit of the budget.  The method therefore combines cheap global
information with a small zoo of structural models, and lets held-out data
decide which model (if any) may be trusted.

Outline
-------
1.  A scrambled Sobol' pilot supplies a space-filling sample, a fallback
    estimate and the reference point for the profiles.
2.  Every coordinate is scanned (1-D "profiles").  Each profile is
    integrated with an adaptive Chebyshev scheme: a uniform seed grid, then
    bisection of the worst interval with local Chebyshev-Lobatto fits, and
    bisection of detected jumps.  Peaked, kinked and stepped profiles are
    resolved to (nearly) machine precision.
3.  The profiles feed a zoo of structural models, each of which provides an
    analytic integral of the cube plus a point-wise predictor:

    * separable product of the coordinate profiles (peaks, ridges that are
      axis aligned, products of one-dimensional factors),
    * constant plus product of detected steps (checkerboard type functions),
    * a low-order total-degree polynomial (quadratic ridges, smooth trends),
    * a plane wave with an offset (oscillatory integrands, in any direction),
    * an additive per-axis model (sums of one-dimensional ridges),
    * models built on directions discovered in the data (rotated ridges).

4.  A held-out quarter of the uniform points scores every model.  A model
    that reproduces the held-out data is trusted directly, because its
    analytic integral is far more accurate than the quasi-Monte-Carlo average
    of the residual; otherwise it is used as a control variate.  If no model
    is convincing, the plain quasi-Monte-Carlo mean is returned.

Everything is generic: no function family is detected, no integral is known
in advance, and every internal point evaluation is charged to the budget.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["integrate", "Cubature"]


# --------------------------------------------------------------------------
# Chebyshev helpers
# --------------------------------------------------------------------------
def _lobatto(n: int) -> np.ndarray:
    """n Chebyshev-Lobatto nodes on [-1, 1] in ascending order."""
    return -np.cos(np.pi * np.arange(n) / (n - 1))


def _cheb_coeffs(v: np.ndarray) -> np.ndarray:
    """Chebyshev interpolation coefficients of values taken at Lobatto nodes.

    The interpolant is  p(u) = c0/2 + sum_{k>=1} c_k T_k(u)  with the last
    coefficient halved as well.
    """
    v = np.asarray(v, float)
    n = len(v)
    try:
        from scipy.fft import dct
        x = dct(v, type=1)
    except Exception:  # pragma: no cover - pure numpy fallback
        k = np.arange(n)
        j = np.arange(n)
        x = 2.0 * (np.cos(np.pi * np.outer(k, j) / (n - 1)) @ v)
        x[0] *= 0.5
        x[-1] *= 0.5
    x = np.asarray(x, float)
    # The nodes are ascending (u_j = -cos(pi j/(n-1))); the DCT-I is written
    # for the descending order, so the odd coefficients change sign.
    x[1::2] *= -1.0
    return x / (n - 1)


def _int_T(k: int) -> float:
    """int_{-1}^{1} T_k(u) du."""
    if k == 0:
        return 2.0
    if k & 1:
        return 0.0
    return 2.0 / (1.0 - float(k) * float(k))


def _cheb_integral(a: float, b: float, c: np.ndarray) -> float:
    """Integral over [a, b] of the Chebyshev interpolant with coefficients c."""
    n = len(c)
    s = 0.5 * c[0] * _int_T(0)
    for k in range(1, n - 1):
        if (k & 1) == 0:
            s += c[k] * _int_T(k)
    if (n - 1) % 2 == 0:
        s += 0.5 * c[-1] * _int_T(n - 1)
    return 0.5 * (b - a) * s


def _cheb_eval(u: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Evaluate the Chebyshev interpolant at u in [-1, 1]."""
    u = np.clip(np.asarray(u, float), -1.0, 1.0)
    theta = np.arccos(u)
    n = len(c)
    out = np.full(u.shape, 0.5 * c[0])
    for k in range(1, n - 1):
        out += c[k] * np.cos(k * theta)
    if n > 1:
        out += 0.5 * c[-1] * np.cos((n - 1) * theta)
    return out


def _ls_coeffs(u: np.ndarray, v: np.ndarray, deg: int):
    """Least-squares Chebyshev coefficients of values v at arbitrary nodes u."""
    u = np.clip(np.asarray(u, float), -1.0, 1.0)
    theta = np.arccos(u)
    M = np.stack([np.cos(k * theta) for k in range(deg + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(M, np.asarray(v, float), rcond=None)
    return coef


class _Seg:
    """A sub-interval of a 1-D profile with its Chebyshev fit."""

    __slots__ = ("a", "b", "vals", "coeffs", "integral", "err", "lobatto")

    def __init__(self, a, b, t, vals, lobatto=True):
        self.a = float(a)
        self.b = float(b)
        self.vals = np.asarray(vals, float)
        self.lobatto = lobatto
        n = len(self.vals)
        if lobatto:
            self.coeffs = _cheb_coeffs(self.vals)
        else:
            deg = min(n - 1, 6)
            u = 2.0 * (np.asarray(t, float) - 0.5 * (self.a + self.b)) / (self.b - self.a)
            self.coeffs = _ls_coeffs(u, self.vals, deg)
        self.integral = _cheb_integral(self.a, self.b, self.coeffs)
        # A posteriori error indicator: compare with the fit that uses only
        # every second node (a nested, much coarser rule).
        err = 0.0
        if n >= 5:
            te = np.asarray(t, float)[::2]
            ve = self.vals[::2]
            if lobatto:
                ce = _cheb_coeffs(ve)
            else:
                u = 2.0 * (te - 0.5 * (self.a + self.b)) / (self.b - self.a)
                ce = _ls_coeffs(u, ve, min(len(ve) - 1, 3))
            err = abs(self.integral - _cheb_integral(self.a, self.b, ce))
        else:
            err = abs(self.integral) * 1e-3
        self.err = err

    def predict(self, x):
        u = (2.0 * np.asarray(x, float) - self.a - self.b) / (self.b - self.a)
        return _cheb_eval(u, self.coeffs)


# --------------------------------------------------------------------------
# 1-D profile integration with adaptive refinement
# --------------------------------------------------------------------------
class Profile:
    """Adaptive integration of the profile f(x_ref with coordinate i = t)."""

    def __init__(self, dim, direction, x_ref, oracle, m0=17, nloc=7,
                 tol=1e-11, s0=0.0, s1=1.0):
        self.dim = dim
        v = np.asarray(direction, float).ravel()
        nrm = float(np.linalg.norm(v))
        self.dir = v / nrm if nrm > 0 else v
        self.axis = int(np.argmax(np.abs(self.dir))) if len(self.dir) else 0
        self.x_ref = np.asarray(x_ref, float)
        self.oracle = oracle
        self.nloc = nloc
        self.tol = tol
        self.s0 = float(s0)
        self.s1 = float(s1)
        self.s_ref = float(np.dot(self.dir, self.x_ref))
        self.segs: list[_Seg] = []
        self.n_points = 0
        self.jump_at = None
        self.at_ref = None
        self.samples: list[tuple[float, float]] = []
        self._build_initial(m0)

    # -- point helpers ----------------------------------------------------
    def _points(self, ts):
        ts = np.asarray(ts, float).ravel()
        pts = np.repeat(self.x_ref[None, :], len(ts), axis=0)
        pts += (ts - self.s_ref)[:, None] * self.dir[None, :]
        # the evaluator only accepts points inside the domain
        return np.clip(pts, 0.0, 1.0)

    def _eval(self, ts):
        ts = np.asarray(ts, float).ravel()
        if len(ts) == 0:
            return np.zeros(0)
        vals = np.asarray(self.oracle(self._points(ts)), float)
        self.n_points += len(ts)
        for t, v in zip(ts, vals):
            self.samples.append((float(t), float(v)))
        return vals

    # -- construction -----------------------------------------------------
    def _build_initial(self, m0):
        m0 = int(max(7, m0))
        t = np.linspace(self.s0, self.s1, m0)
        vals = self._eval(t)
        if len(vals) < m0:
            self.segs = []
            return
        self.segs = [_Seg(self.s0, self.s1, t, vals, lobatto=False)]
        ref = self._eval([self.s_ref])
        if len(ref):
            self.at_ref = float(ref[0])
        self._maybe_jump()

    def _maybe_jump(self):
        """Detect a step in the initial scan and locate it by bisection."""
        if not self.segs:
            return
        seg = self.segs[0]
        v = seg.vals
        n = len(v)
        if n < 5:
            return
        d = np.abs(np.diff(v))
        scale = float(np.max(v) - np.min(v))
        if scale <= 0:
            return
        k = int(np.argmax(d))
        if d[k] < 0.3 * scale:
            return
        left = v[: k + 1]
        right = v[k + 1:]
        if len(left) < 2 or len(right) < 2:
            return
        # A genuine jump has (nearly) constant levels on both sides, whereas a
        # narrow peak keeps varying across neighbouring nodes.
        if (float(np.max(left) - np.min(left)) > 0.3 * d[k]
                or float(np.max(right) - np.min(right)) > 0.3 * d[k]):
            return
        lo, hi = float(seg.t[0]) if hasattr(seg, "t") else 0.0, 0.0
        # recover the node positions from the sample list
        ts = [t for t, _ in self.samples[:n]]
        lo, hi = float(ts[k]), float(ts[k + 1])
        lv, rv = float(v[k]), float(v[k + 1])
        good = True
        for _ in range(60):
            if hi - lo < 1e-12:
                break
            mid = 0.5 * (lo + hi)
            mv = self._eval([mid])
            if len(mv) == 0:
                break
            mv = float(mv[0])
            if abs(mv - lv) <= abs(mv - rv):
                lo = mid
            else:
                hi = mid
            if abs(mv - lv) > 0.35 * d[k] and abs(mv - rv) > 0.35 * d[k]:
                good = False
                break
        if not good:
            return
        self.jump_at = 0.5 * (lo + hi)
        self._split_at_jump()

    def _split_at_jump(self):
        """Rebuild the profile as two smooth pieces separated by the jump."""
        c = self.jump_at
        nodes = 0.5 * (1.0 + _lobatto(self.nloc))
        tq = self.s0 + (c - self.s0) * nodes
        vl = self._eval(tq)
        if len(vl) < self.nloc:
            self.jump_at = None
            return
        tq = c + (self.s1 - c) * nodes
        vr = self._eval(tq)
        if len(vr) < self.nloc:
            self.jump_at = None
            return
        self.segs = [_Seg(self.s0, c, tq, vl), _Seg(c, self.s1, tq, vr)]

    # -- refinement -------------------------------------------------------
    def refine(self, n_new):
        """Spend at most n_new further points to reduce the profile error."""
        if n_new <= 0 or not self.segs:
            return
        left = int(n_new)
        nodes = 0.5 * (1.0 + _lobatto(self.nloc))
        while left > 0 and self.segs:
            errs = [s.err for s in self.segs]
            k = int(np.argmax(errs))
            seg = self.segs[k]
            if seg.err <= self.tol:
                break
            mid = 0.5 * (seg.a + seg.b)
            if not (seg.a < mid < seg.b):
                seg.err = 0.0
                continue
            ta = seg.a + 0.5 * (seg.b - seg.a) * nodes
            tb = mid + (seg.b - mid) * nodes
            need = np.concatenate([ta, tb])
            got = self._eval(need)
            if len(got) < len(need):
                break
            na = len(ta)
            s1 = _Seg(seg.a, mid, ta, got[:na])
            s2 = _Seg(mid, seg.b, tb, got[na:])
            self.segs[k:k + 1] = [s1, s2]
            left -= len(need)

    # -- results ----------------------------------------------------------
    @property
    def integral(self):
        return float(sum(s.integral for s in self.segs))

    @property
    def error(self):
        return float(sum(s.err for s in self.segs))

    def value(self, t):
        """Exact profile value at t (one further oracle call)."""
        v = self.oracle(self._points([t]))
        if len(v) == 0:
            return float("nan")
        return float(v[0])

    def at(self, proj):
        """Evaluate the Chebyshev profile interpolant at projection values."""
        proj = np.asarray(proj, float)
        out = np.zeros(len(proj))
        for s in self.segs:
            m = (proj >= s.a) & (proj <= s.b)
            if not np.any(m):
                continue
            out[m] = s.predict(proj[m])
        return out

    def predict(self, X):
        """Evaluate the Chebyshev profile interpolant at the projections v.x."""
        return self.at(np.asarray(X, float) @ self.dir)


# --------------------------------------------------------------------------
# structural models
# --------------------------------------------------------------------------
def _poly_index_count(deg, d):
    """Multi-indices of length d whose total degree is at most deg.

    Only admissible indices are generated; enumerating all (deg+1)^d
    combinations and filtering afterwards would explode in higher dimensions.
    """
    out = []

    def rec(prefix, rem, budget):
        if rem == 0:
            out.append(tuple(prefix))
            return
        for k in range(min(deg, budget) + 1):
            prefix.append(k)
            rec(prefix, rem - 1, budget - k)
            prefix.pop()

    rec([], d, deg)
    return out


def _poly_basis(u, idx):
    M = np.empty((u.shape[0], len(idx)))
    for j, t in enumerate(idx):
        col = np.ones(u.shape[0])
        for i, k in enumerate(t):
            if k:
                col = col * np.cos(k * np.arccos(np.clip(u[:, i], -1, 1)))
        M[:, j] = col
    return M


def _poly_integral(idx):
    """Integral over [0,1]^d of the basis monomial."""
    tot = 1.0
    for k in idx:
        tot *= 0.5 * _int_T(k)
    return tot


def _fit_polynomial(X, v, max_deg=6):
    """Least-squares total-degree Chebyshev fit (degree chosen by fit quality)."""
    u = 2.0 * X - 1.0
    n = len(v)
    scale = max(float(np.std(v)), 1e-300)
    best = None
    for deg in range(0, max_deg + 1):
        idx = _poly_index_count(deg, X.shape[1])
        if len(idx) > 0.6 * n:
            break
        M = _poly_basis(u, idx)
        try:
            coef, *_ = np.linalg.lstsq(M, v, rcond=None)
        except Exception:
            continue
        resid = v - M @ coef
        rmse = float(np.sqrt(np.mean(resid ** 2))) / scale
        if best is None or rmse < best[0]:
            best = (rmse, coef, idx)
    if best is None:
        return None
    rmse, coef, idx = best
    return coef, idx, rmse


def _fit_plane_wave(X, v, omega):
    """Fit f = c0 + a cos(omega.x) + b sin(omega.x); returns (coef, rmse)."""
    A = np.stack([np.ones(len(X)), np.cos(X @ omega), np.sin(X @ omega)], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    except Exception:
        return None
    resid = v - A @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2))) / max(float(np.std(v)), 1e-300)
    return coef, rmse


def _wave_integral(omega, coef):
    """Integral over the cube of c0 + a cos(omega.x) + b sin(omega.x)."""
    c0 = float(coef[0])
    a = float(coef[1])
    b = float(coef[2])
    z = np.prod((np.exp(1j * omega) - 1.0) / (1j * omega))
    return c0 + a * float(np.real(z)) + b * float(np.imag(z))


def _estimate_frequency(t, v, wmax=220.0):
    """Dominant angular frequency of a 1-D profile (least squares)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    scale = max(float(np.max(v) - np.min(v)), 1e-300)
    if len(t) < 6:
        return None, np.inf
    best = [np.inf, 1.0]

    def resid(omega):
        # a constant offset is allowed: the profile may be a wave on top of a
        # slowly varying background
        A = np.stack([np.ones(len(t)), np.cos(omega * t), np.sin(omega * t)],
                     axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        except Exception:
            return np.inf
        return float(np.sum((v - A @ coef) ** 2)) / scale ** 2

    grid = np.linspace(0.05, wmax, 900)
    vals = np.array([resid(w) for w in grid])
    k = int(np.argmin(vals))
    best = [float(vals[k]), float(grid[k])]
    step = grid[1] - grid[0]
    lo, hi = max(0.01, grid[k] - step), grid[k] + step
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        for m in (m1, m2):
            r = resid(m)
            if r < best[0]:
                best = [r, m]
        if best[1] <= m1:
            hi = m2
        else:
            lo = m1
    return best[1], best[0]



def _hat_basis(x, nk):
    """Piecewise-linear (hat) basis on [0, 1] with nk intervals.

    Returns (basis matrix, integral of each basis function).
    """
    knots = np.linspace(0.0, 1.0, nk + 1)
    x = np.asarray(x, float)
    B = np.zeros((len(x), nk + 1))
    # interior hats
    for k in range(1, nk):
        left = knots[k] - knots[k - 1]
        right = knots[k + 1] - knots[k]
        a = np.clip((x - knots[k - 1]) / left, 0.0, 1.0)
        b = np.clip((knots[k + 1] - x) / right, 0.0, 1.0)
        B[:, k] = np.minimum(a, b)
    # boundary ramps
    a = np.clip(x / knots[1], 0.0, 1.0)
    B[:, 0] = 1.0 - a
    B[:, nk] = np.clip((x - knots[nk - 1]) / (knots[nk] - knots[nk - 1]), 0.0, 1.0)
    ints = np.zeros(nk + 1)
    ints[0] = 0.5 * knots[1]
    ints[nk] = 0.5 * (knots[nk] - knots[nk - 1])
    for k in range(1, nk):
        ints[k] = 0.5 * (knots[k + 1] - knots[k - 1])
    return B, ints




# --------------------------------------------------------------------------
# directional (ridge) helpers
# --------------------------------------------------------------------------
def _projection_density(v, n=4096):
    """Density of the projection s = v.X for X ~ U[0,1]^d on a uniform grid.

    The density of a projection of the unit cube is a box spline; it is built
    here by repeated moving averages, which is exact for piecewise polynomial
    densities and stable in any dimension.
    """
    v = np.asarray(v, float).ravel()
    nrm = float(np.linalg.norm(v))
    if nrm <= 0:
        return np.zeros(0), np.zeros(0)
    v = v / nrm
    lo = float(np.sum(np.minimum(v, 0.0)))
    hi = float(np.sum(np.maximum(v, 0.0)))
    pad = max((hi - lo) * 1e-6, 1e-12)
    grid = np.linspace(lo - pad, hi + pad, int(n))
    h = grid[1] - grid[0]
    a0, b0 = min(0.0, v[0]), max(0.0, v[0])
    w0 = abs(v[0])
    if w0 < 1e-15:
        # degenerate first factor: shift the grid and continue with the rest
        grid = grid - v[0]
        dens = np.zeros(len(grid))
        cum = np.zeros(len(grid) + 1)
        start = 1
    else:
        dens = np.where((grid >= a0) & (grid <= b0), 1.0 / w0, 0.0)
        # exact cumulative of the first (uniform) factor
        cum = np.clip((grid - a0) / w0, 0.0, 1.0)
        start = 1
    for i in range(start, len(v)):
        w = abs(v[i])
        if w < 1e-15:
            continue
        if v[i] > 0:
            # convolve with the uniform density on [0, w]
            shifted = np.interp(grid - w, grid, cum, left=0.0, right=cum[-1])
            dens = (cum - shifted) / w
        else:
            # convolve with the uniform density on [-w, 0]
            shifted = np.interp(grid + w, grid, cum, left=0.0, right=cum[-1])
            dens = (shifted - cum) / w
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * h)])
    return grid, dens


def _profile_cube_integral(prof, ngrid=2048):
    """Integral of the profile over the cube: int p(v.x) dx."""
    grid, dens = _projection_density(prof.dir, ngrid)
    if len(grid) == 0:
        return 0.0
    vals = prof.at(grid)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return float(np.trapezoid(vals * dens, grid))


def _direction_score(X, y, v, nk=8):
    """R^2 of y explained by a 1-D function of the projection v.x."""
    s = np.asarray(X, float) @ v
    lo = float(s.min())
    hi = float(s.max())
    if hi - lo < 1e-12:
        return 0.0
    u = (s - lo) / (hi - lo)
    B, _ = _hat_basis(u, nk)
    A = np.column_stack([np.ones(len(u)), B])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return 0.0
    resid = y - A @ coef
    ss = float(np.sum((y - np.mean(y)) ** 2))
    if ss <= 0:
        return 0.0
    return 1.0 - float(np.sum(resid ** 2)) / ss


def _discover_directions(X, y, d, rng, n_random=32, keep=3, n_refine=40):
    """Search for directions along which the integrand is nearly one-dimensional."""
    cands = [np.eye(d)[i] for i in range(d)]
    for _ in range(n_random):
        v = rng.normal(size=d)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            cands.append(v / nrm)
    scores = [_direction_score(X, y, v) for v in cands]
    order = list(np.argsort(scores)[::-1])
    out = []
    for k in order[:6]:
        v = np.asarray(cands[k], float)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            continue
        v = v / nrm
        sc = scores[k]
        step = 0.3
        for _ in range(n_refine):
            pert = rng.normal(size=d)
            pert = pert - np.dot(pert, v) * v
            pn = float(np.linalg.norm(pert))
            if pn < 1e-9:
                continue
            cand = v + step * pert / pn
            cand = cand / float(np.linalg.norm(cand))
            s2 = _direction_score(X, y, cand)
            if s2 > sc:
                v, sc = cand, s2
            else:
                step *= 0.93
        out.append((sc, v))
    out.sort(key=lambda z: -z[0])
    # A ridge concentrates the largest samples along its own direction, so the
    # spread of those samples along a candidate direction is a sharp criterion.
    try:
        top = np.argsort(y)[-max(3, len(y) // 10):]
        Xt = X[top]

        def spread(v):
            return float(np.var(Xt @ v))

        extra = []
        for sc, v in out[:2]:
            w = v.copy()
            sw = spread(w)
            step = 0.2
            for _ in range(n_refine):
                pert = rng.normal(size=d)
                pert = pert - np.dot(pert, w) * w
                pn = float(np.linalg.norm(pert))
                if pn < 1e-9:
                    continue
                cand = w + step * pert / pn
                cand = cand / float(np.linalg.norm(cand))
                c2 = spread(cand)
                if c2 < sw:
                    w, sw = cand, c2
                else:
                    step *= 0.93
            extra.append((sc, w))
        out = out + extra
        out.sort(key=lambda z: -z[0])
    except Exception:
        pass
    res = []
    for sc, v in out:
        # skip directions that duplicate an axis or an earlier pick
        dup = False
        for _, w in res:
            if abs(float(np.dot(v, w))) > 0.999:
                dup = True
                break
        if not dup:
            res.append((sc, v))
        if len(res) >= keep:
            break
    return res



# --------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------
def pred_of(models, name):
    """Point-wise predictor of the named model (None if it has none)."""
    for nm, _integ, pred, _verr in models:
        if nm == name:
            return pred
    return None


class Cubature:
    def __init__(self, f_batch, dim, budget, seed=0):
        self.f_batch = f_batch
        self.d = int(dim)
        self.B = int(budget)
        self.seed = int(seed)
        self.used = 0
        self.Xu = []
        self.Vu = []

    # -- evaluation -------------------------------------------------------
    def _eval(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        rem = self.B - self.used
        if rem <= 0:
            return np.zeros(0)
        if len(X) > rem:
            X = X[:rem]
        v = np.asarray(self.f_batch(X), float).ravel()
        self.used += len(X)
        self.Xu.append(X)
        self.Vu.append(v)
        return v

    def _uniform(self, n, rng=0):
        """n uniform quasi-random points (scrambled Sobol', sequential).

        Points are handed out sequentially so that every prefix of the
        sequence is itself a well distributed point set.
        """
        n = int(max(0, n))
        if n == 0:
            return np.zeros((0, self.d))
        return self._sobol(n)

    def _sobol(self, n):
        from scipy.stats import qmc
        if getattr(self, "_eng", None) is None:
            self._eng = qmc.Sobol(d=self.d, scramble=True, seed=self.seed)
        m = 1 << int(math.floor(math.log2(max(n, 1))))
        pts = self._eng.random(m)
        if m < n:
            pts = np.vstack([pts, self._eng.random(n - m)])
        return pts[:n]

    # -- main -------------------------------------------------------------
    def run(self):
        try:
            return self._run()
        except Exception:
            if self.Vu:
                allv = np.concatenate(self.Vu)
                if len(allv):
                    return float(np.mean(allv))
            return 0.0

    def _run(self):
        d = self.d
        B = self.B
        # ---------------- pilot -------------------------------------------
        n_pilot = 256 if B >= 1024 else max(32, 1 << int(math.floor(math.log2(max(B // 4, 16)))))
        n_pilot = int(min(n_pilot, max(16, B - 8)))
        Xp = self._uniform(n_pilot, 0)
        vp = self._eval(Xp)
        if len(vp) == 0:
            return 0.0
        qmc_mean = float(np.mean(vp))
        if not math.isfinite(qmc_mean):
            qmc_mean = 0.0
        x_ref = Xp[int(np.argmax(vp))]
        f_ref = float(np.max(vp))
        var_f = float(np.var(vp))

        # ---------------- profiles ----------------------------------------
        prof_total = int(min(max(32, int(B * 0.65)), max(32, B - n_pilot - 32)))
        axis_share = prof_total
        dir_share = int(max(0, B - n_pilot - prof_total - int(B * 0.25)))
        m0 = 33 if d <= 4 else 17
        nloc = 7
        profiles = []
        for i in range(d):
            try:
                p = Profile(d, np.eye(d)[i], x_ref, self._eval, m0=m0, nloc=nloc)
                profiles.append(p)
            except Exception:
                profiles.append(None)
        used_prof = sum(p.n_points for p in profiles if p is not None)
        spare = max(0, axis_share - used_prof)
        guard = 0
        while spare > 4 and guard < 5000:
            guard += 1
            cand = [(p.error, i) for i, p in enumerate(profiles) if p is not None]
            if not cand:
                break
            cand.sort(key=lambda z: -z[0])
            progressed = False
            for err, i in cand:
                if spare <= 4 or err <= 0:
                    continue
                p = profiles[i]
                before = p.n_points
                p.refine(min(spare, 26))
                spent = p.n_points - before
                spare -= spent
                progressed = progressed or spent > 0
            if not progressed:
                break

        # ---------------- directional (ridge) profiles ----------------------
        dir_profiles = []
        try:
            rng = np.random.default_rng(self.seed + 12345)
            found = _discover_directions(Xp, vp, d, rng, n_random=24, keep=3)
            for _sc, v in found:
                lo_v = float(np.sum(np.minimum(v, 0.0)))
                hi_v = float(np.sum(np.maximum(v, 0.0)))
                if hi_v - lo_v < 1e-9:
                    continue
                centre = 0.5 * np.ones(d)
                dp = Profile(d, v, centre, self._eval, m0=17, nloc=nloc,
                             s0=lo_v, s1=hi_v)
                dir_profiles.append(dp)
            dspare = max(0, dir_share - sum(q.n_points for q in dir_profiles))
            guard = 0
            while dspare > 4 and dir_profiles and guard < 500:
                guard += 1
                cand = sorted(((q.error, i) for i, q in enumerate(dir_profiles)),
                              key=lambda z: -z[0])
                progressed = False
                for err, i in cand:
                    if dspare <= 4 or err <= 0:
                        continue
                    q = dir_profiles[i]
                    before = q.n_points
                    q.refine(min(dspare, 26))
                    spent = q.n_points - before
                    dspare -= spent
                    progressed = progressed or spent > 0
                if not progressed:
                    break
        except Exception:
            dir_profiles = []

        # ---------------- extra uniform points ----------------------------
        n_extra = int(max(0, B - self.used - 4))
        Xe = self._uniform(n_extra, 1)
        ve = self._eval(Xe)
        Xu = np.vstack([Xp, Xe]) if len(Xe) else Xp
        Vu = np.concatenate([vp, ve]) if len(ve) else vp

        # validation split (uniform points only)
        if len(Xu) >= 64:
            n_val = max(16, min(int(0.25 * len(Xu)), len(Xu) // 3))
        else:
            n_val = 0
        if n_val:
            Xtr, ytr = Xu[:-n_val], Vu[:-n_val]
            Xva, yva = Xu[-n_val:], Vu[-n_val:]
        else:
            Xtr, ytr, Xva, yva = Xu, Vu, Xu, Vu

        scale = max(float(np.std(ytr)), 1e-300)
        models = []   # (name, direct_integral, predictor or None, val_err)

        qmc_all = float(np.mean(Vu))
        if not math.isfinite(qmc_all):
            qmc_all = qmc_mean
        models.append(("const", qmc_all, None, 0.0))

        # --- model: separable product of the profiles --------------------
        if all(p is not None and p.segs for p in profiles):
            try:
                ints = np.array([p.integral for p in profiles])
                base = np.array([p.at_ref for p in profiles], float)
                if (np.all(np.isfinite(ints)) and np.all(np.isfinite(base))
                        and np.all(np.abs(base) > 0) and abs(f_ref) > 0):
                    A = f_ref / float(np.prod(base))
                    integ = A * float(np.prod(ints))
                    pred = self._prod_predictor(profiles, A)
                    verr = self._val_err(pred, Xva, yva, scale)
                    models.append(("product", integ, pred, verr))
            except Exception:
                pass

        # --- model: constant + product of detected steps ------------------
        if all(p is not None and p.jump_at is not None for p in profiles):
            try:
                S = np.ones(len(Xtr))
                for i, p in enumerate(profiles):
                    S = S * np.where(Xtr[:, i] < p.jump_at, 1.0, -1.0)
                A = np.stack([np.ones(len(Xtr)), S], axis=1)
                coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
                resid = ytr - A @ coef
                verr = float(np.sqrt(np.mean(resid ** 2))) / scale
                prod = 1.0
                for p in profiles:
                    prod *= (2.0 * p.jump_at - 1.0)
                integ = float(coef[0]) + float(coef[1]) * prod

                def pred(X, _c=coef, _pr=profiles):
                    S = np.ones(len(X))
                    for i, p in enumerate(_pr):
                        S = S * np.where(X[:, i] < p.jump_at, 1.0, -1.0)
                    return _c[0] + _c[1] * S

                verr2 = self._val_err(pred, Xva, yva, scale)
                models.append(("steps", integ, pred, min(verr, verr2)))
            except Exception:
                pass

        # --- model: polynomial -------------------------------------------
        try:
            fit = _fit_polynomial(Xtr, ytr, max_deg=min(6, d + 2))
            if fit is not None:
                coef, idx, verr = fit
                integ = float(sum(c * _poly_integral(t) for c, t in zip(coef, idx)))

                def pred(X, _c=coef, _idx=idx):
                    return _poly_basis(2.0 * X - 1.0, _idx) @ _c

                verr2 = self._val_err(pred, Xva, yva, scale)
                models.append(("poly", integ, pred, max(verr, verr2)))
        except Exception:
            pass

        # --- model: additive (sum of per-axis functions) -------------------
        try:
            nk = 12 if d <= 4 else 8
            design = np.empty((len(Xtr), 1 + d * (nk + 1)))
            design[:, 0] = 1.0
            for i in range(d):
                Bi, _ = _hat_basis(Xtr[:, i], nk)
                design[:, 1 + i * (nk + 1):1 + (i + 1) * (nk + 1)] = Bi
            coef, *_ = np.linalg.lstsq(design, ytr, rcond=None)
            resid = ytr - design @ coef
            verr = float(np.sqrt(np.mean(resid ** 2))) / scale
            integ = float(coef[0])
            _, ints_i = _hat_basis(np.zeros(1), nk)
            for i in range(d):
                integ += float(np.dot(coef[1 + i * (nk + 1):1 + (i + 1) * (nk + 1)], ints_i))

            def pred(X, _c=coef, _nk=nk, _d=d):
                D = np.empty((len(X), 1 + _d * (_nk + 1)))
                D[:, 0] = 1.0
                for i in range(_d):
                    Bi, _ = _hat_basis(X[:, i], _nk)
                    D[:, 1 + i * (_nk + 1):1 + (i + 1) * (_nk + 1)] = Bi
                return D @ _c

            verr2 = self._val_err(pred, Xva, yva, scale)
            models.append(("additive", integ, pred, max(verr, verr2)))
        except Exception:
            pass

        # --- model: plane wave -------------------------------------------
        try:
            omegas = []
            good = True
            for p in profiles:
                if p is None or not p.segs:
                    good = False
                    break
                pts = np.array(p.samples, float)
                if len(pts) < 8:
                    good = False
                    break
                om, r = _estimate_frequency(pts[:, 0], pts[:, 1])
                if om is None or not np.isfinite(om) or r > 0.02:
                    good = False
                    break
                omegas.append(om)
            if good and len(omegas) == d:
                omega = np.array(omegas)
                # The per-axis profiles only reveal |omega_i|; the relative
                # signs are recovered by fitting the joint wave model.
                best = None
                if d <= 10:
                    for mask in range(1 << d):
                        sgn = np.array([1.0 if (mask >> i) & 1 else -1.0
                                        for i in range(d)])
                        w = omega * sgn
                        fw = _fit_plane_wave(Xtr, ytr, w)
                        if fw is None:
                            continue
                        if best is None or fw[1] < best[0]:
                            best = (fw[1], w, fw[0])
                else:
                    sgn = np.ones(d)
                    fw = _fit_plane_wave(Xtr, ytr, omega * sgn)
                    best = (fw[1], omega * sgn, fw[0]) if fw else None
                    for _ in range(3):
                        for i in range(d):
                            sgn2 = sgn.copy()
                            sgn2[i] = -sgn2[i]
                            fw2 = _fit_plane_wave(Xtr, ytr, omega * sgn2)
                            if fw2 is not None and fw2[1] < best[0]:
                                best = (fw2[1], omega * sgn2, fw2[0])
                                sgn = sgn2
                if best is not None:
                    verr, omega, coef = best
                    integ = _wave_integral(omega, coef)

                    def pred(X, _c=coef, _w=omega):
                        return (_c[0] + _c[1] * np.cos(X @ _w)
                                + _c[2] * np.sin(X @ _w))

                    verr2 = self._val_err(pred, Xva, yva, scale)
                    models.append(("wave", integ, pred, max(verr, verr2)))
        except Exception:
            pass

        # --- model: directional ridge structures ---------------------------
        if dir_profiles:
            try:
                ints = [_profile_cube_integral(q) for q in dir_profiles]
                for k in (1, 2, 3):
                    if k > len(dir_profiles):
                        break
                    ps = dir_profiles[:k]
                    bases = np.array([q.at([q.s_ref])[0] for q in ps], float)
                    if not np.all(np.isfinite(bases)) or np.any(np.abs(bases) <= 0):
                        continue
                    if not math.isfinite(f_ref) or abs(f_ref) <= 0:
                        continue
                    A = f_ref / float(np.prod(bases))
                    dirs_k = [q.dir for q in ps]
                    orth = True
                    for a_ in range(len(dirs_k)):
                        for b_ in range(a_ + 1, len(dirs_k)):
                            if abs(float(np.dot(dirs_k[a_], dirs_k[b_]))) > 1e-8:
                                orth = False
                    if orth:
                        integ = A * float(np.prod(ints[:k]))
                    else:
                        # projections are correlated: integrate the model
                        # itself over the uniform sample instead
                        integ = float("nan")

                    def pred(X, _ps=ps, _A=A):
                        out = np.full(len(X), float(_A))
                        for q in _ps:
                            out = out * q.predict(X)
                        return out

                    if not math.isfinite(integ):
                        try:
                            integ = float(np.mean(pred(Xu)))
                        except Exception:
                            continue
                    verr = self._val_err(pred, Xva, yva, scale)
                    models.append(("ridge_prod%d" % k, integ, pred, verr))
                # sum of directional profiles fitted by alternating least
                # squares: each 1-D profile absorbs the contribution of the
                # other directions
                try:
                    nk = 10
                    proj = [Xtr @ q.dir for q in dir_profiles]
                    coef = []
                    for m, q in enumerate(dir_profiles):
                        u = (proj[m] - q.s0) / max(q.s1 - q.s0, 1e-300)
                        B, _ = _hat_basis(u, nk)
                        c, *_ = np.linalg.lstsq(B, ytr, rcond=None)
                        coef.append(c)
                    for _ in range(3):
                        vals = []
                        for m, q in enumerate(dir_profiles):
                            u = (proj[m] - q.s0) / max(q.s1 - q.s0, 1e-300)
                            B, _ = _hat_basis(u, nk)
                            vals.append(B @ coef[m])
                        for m, q in enumerate(dir_profiles):
                            target = ytr - sum(vals[m2] for m2 in range(len(vals))
                                               if m2 != m)
                            u = (proj[m] - q.s0) / max(q.s1 - q.s0, 1e-300)
                            B, _ = _hat_basis(u, nk)
                            c, *_ = np.linalg.lstsq(B, target, rcond=None)
                            coef[m] = c
                    pred_als = []
                    for m, q in enumerate(dir_profiles):
                        u = (proj[m] - q.s0) / max(q.s1 - q.s0, 1e-300)
                        B, _ = _hat_basis(u, nk)
                        pred_als.append(B @ coef[m])
                    fit_als = np.sum(pred_als, axis=0)
                    verr_als = float(np.sqrt(np.mean((ytr - fit_als) ** 2))) / scale
                    integ_als = 0.0
                    for m, q in enumerate(dir_profiles):
                        grid, dens = _projection_density(q.dir, 2048)
                        if len(grid) == 0:
                            integ_als = float("nan")
                            break
                        u = (grid - q.s0) / max(q.s1 - q.s0, 1e-300)
                        B, _ = _hat_basis(u, nk)
                        integ_als += float(np.trapezoid((B @ coef[m]) * dens, grid))

                    def pred_als_fn(X, _ps=dir_profiles, _c=coef, _nk=nk):
                        out = np.zeros(len(X))
                        for m, q in enumerate(_ps):
                            u = (X @ q.dir - q.s0) / max(q.s1 - q.s0, 1e-300)
                            B, _ = _hat_basis(u, _nk)
                            out = out + B @ _c[m]
                        return out

                    if math.isfinite(integ_als):
                        verr_als2 = self._val_err(pred_als_fn, Xva, yva, scale)
                        models.append(("ridge_als", integ_als, pred_als_fn,
                                       max(verr_als, verr_als2)))
                except Exception:
                    pass

                # sum of directional profiles
                cols = np.column_stack([q.predict(Xtr) for q in dir_profiles])
                try:
                    coef, *_ = np.linalg.lstsq(cols, ytr, rcond=None)
                except Exception:
                    coef = None
                if coef is not None and np.all(np.isfinite(coef)):
                    resid = ytr - cols @ coef
                    verr = float(np.sqrt(np.mean(resid ** 2))) / scale
                    integ = float(sum(coef[m] * ints[m]
                                      for m in range(len(dir_profiles))))

                    def pred(X, _ps=dir_profiles, _c=coef):
                        return np.column_stack([q.predict(X) for q in _ps]) @ _c

                    verr2 = self._val_err(pred, Xva, yva, scale)
                    models.append(("ridge_sum", integ, pred, max(verr, verr2)))
            except Exception:
                pass

        # When no structural model reproduces the held-out data the integrand
        # is not separable in the coordinates, so a larger uniform sample is a
        # better use of the remaining budget than more profile work.
        try:
            n_more = int(max(0, B - self.used - 4))
            if n_more > 16:
                Xm = self._uniform(n_more, 2)
                vm = self._eval(Xm)
                if len(vm):
                    Xu = np.vstack([Xu, Xm])
                    Vu = np.concatenate([Vu, vm])
                    qmc_all = float(np.mean(Vu))
                    var_f = float(np.var(Vu))
                    models = [(nm, (qmc_all if nm == "const" else iv),
                               pr, ve) for (nm, iv, pr, ve) in models]
        except Exception:
            pass

        # ---------------- choose the estimate ------------------------------
        return self._select(models, Xtr, ytr, Xva, yva, var_f, scale)

    # -- helpers ----------------------------------------------------------
    def _prod_predictor(self, profiles, A):
        def pred(X):
            out = np.full(len(X), float(A))
            for i, p in enumerate(profiles):
                out = out * p.predict(X)
            return out
        return pred

    def _val_err(self, pred, Xva, yva, scale):
        try:
            r = np.asarray(pred(Xva), float) - yva
            if not np.all(np.isfinite(r)):
                return np.inf
            return float(np.sqrt(np.mean(r ** 2))) / max(scale, 1e-300)
        except Exception:
            return np.inf

    def _select(self, models, Xtr, ytr, Xva, yva, var_f, scale):
        """Pick the most reliable estimate among the candidate models.

        A model whose held-out error is tiny is trusted directly: its
        analytic integral is far more accurate than the quasi-Monte-Carlo
        average of the residual.  Otherwise the model is used as a control
        variate, and only if that is demonstrably better than plain
        quasi-Monte-Carlo.
        """
        n = max(len(Xtr), 1)
        qmc_abs_err = math.sqrt(max(var_f, 0.0) / n)
        scored = []
        for name, integ, pred, verr in models:
            if not math.isfinite(integ):
                continue
            if name == "const" or pred is None:
                scored.append((qmc_abs_err, name, integ, False))
                continue
            if verr <= 1e-6:
                # the analytic integral of a well validated model
                scored.append((verr * max(scale, 1e-300), name, integ, True))
                continue
            try:
                r = np.asarray(pred(Xtr), float) - ytr
                r = r[np.isfinite(r)]
                if len(r) < 8:
                    scored.append((qmc_abs_err, name, integ, False))
                    continue
                err_cv = math.sqrt(float(np.var(r)) / n)
                scored.append((err_cv, name, integ, False))
            except Exception:
                scored.append((qmc_abs_err, name, integ, False))
        if not scored:
            return float(np.mean(ytr))
        scored.sort(key=lambda z: z[0])
        _, name, integ, direct = scored[0]
        out = integ
        if not direct and name != "const":
            try:
                r = np.asarray(pred_of(models, name)(Xtr), float) - ytr
                out = integ + float(np.mean(r))
            except Exception:
                out = integ
        if not math.isfinite(out):
            out = float(np.mean(ytr))
        return float(out)


def integrate(f_batch, dim, budget, seed=0):
    """Estimate the integral of f over [0,1]^dim with at most budget calls.

    ``f_batch`` receives an (n, dim) array of points and must return an
    (n,) array of values.
    """
    return Cubature(f_batch, dim, budget, seed).run()
