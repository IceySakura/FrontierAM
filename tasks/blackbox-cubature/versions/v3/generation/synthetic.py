"""Synthetic hard families for blackbox-cubature v3 (Gaussian measure, canonical coordinates y = Q z).

Every instance is f(y) = B(y) + N(y1, y2):

- B is a full-dimensional smooth background with a closed-form mean beta0, so that recovering the two-dimensional
  structure alone does not give the integral:
      B(y) = beta0 + beta1 (exp(w.y - |w|^2/2) - 1) + beta2 (cos(v.y + theta) - exp(-|v|^2/2) cos(theta)).
- N carries the non-smoothness. In y1 every piece is a quadratic polynomial whose coefficients are smooth (non-monotone)
  functions of y2, so the kink and jump curves are curved, change their number of y1-crossings with y2 (fold points),
  and are not monotone along any single direction (the preintegration monotonicity condition fails):
      H1  alpha s(y2) max(phi(y1, y2), 0)                      non-monotone kink
      H2  alpha s(y2) max(phi1, phi2, 0)                       intersecting kinks, incl. interior kinks phi1 = phi2 > 0
      H3  alpha s(y2) (1 + p1 y1 + p2 y1^2) 1{phi(y1, y2) > 0}  curved jump
  with phi(y1, y2) = a y1^2 + b(y2) y1 + c(y2), b(y2) = b0 + b1 sin(om1 y2 + ps1), c(y2) = c0 + c1 cos(om2 y2 + ps2)
  + c2 y2, and s(y2) = exp(g y2 - g^2/2).

Reference (method A): for fixed y2 the y1-integral is exact (pieces are quadratics, see gauss.poly2_gauss); the outer
y2-integral is adaptive quadrature split at every y2 where the piece structure changes (discriminant and resultant
zeros). Method B (reference_swapped) integrates in the other order with numerically located y2-crossings.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize

from gauss import pdf, poly2_gauss, quadratic_roots, random_unit

FAMILIES = ('H1', 'H2', 'H3')
Y2_RANGE = 10.0


def _coef(rng):
    return {'a': float(rng.choice([-1, 1]) * rng.uniform(0.4, 1.4)),
            'b0': float(rng.uniform(-0.8, 0.8)), 'b1': float(rng.uniform(0.3, 1.2)),
            'om1': float(rng.uniform(0.6, 1.8)), 'ps1': float(rng.uniform(0, 2 * math.pi)),
            'c0': float(rng.uniform(-1.2, 1.2)), 'c1': float(rng.uniform(0.4, 1.5)),
            'om2': float(rng.uniform(0.6, 1.8)), 'ps2': float(rng.uniform(0, 2 * math.pi)),
            'c2': float(rng.uniform(-0.3, 0.3))}


def _bc(q: dict, y2):
    b = q['b0'] + q['b1'] * np.sin(q['om1'] * y2 + q['ps1'])
    c = q['c0'] + q['c1'] * np.cos(q['om2'] * y2 + q['ps2']) + q['c2'] * y2
    return b, c


def _bc_prime(q: dict, y2):
    return q['b1'] * q['om1'] * np.cos(q['om1'] * y2 + q['ps1']), \
        -q['c1'] * q['om2'] * np.sin(q['om2'] * y2 + q['ps2']) + q['c2']


def _phi(q: dict, y1, y2):
    b, c = _bc(q, y2)
    return q['a'] * y1 * y1 + b * y1 + c


def _poly(q: dict, y2: float) -> tuple[float, float, float]:
    b, c = _bc(q, y2)
    return float(c), float(b), float(q['a'])


def sample(family: str, d: int, rng: np.random.Generator) -> dict:
    """Draw one instance; rejection keeps the non-smooth structure inside the bulk of the measure."""
    for _ in range(1000):
        spec = {'family': family, 'd': int(d),
                'beta0': float(rng.uniform(0.5, 1.0)), 'beta1': float(rng.uniform(0.1, 0.4)),
                'beta2': float(rng.uniform(0.1, 0.4)), 'theta': float(rng.uniform(0, 2 * math.pi)),
                'w': (random_unit(rng, d) * rng.uniform(0.2, 0.5)).tolist(),
                'v': (random_unit(rng, d) * rng.uniform(0.5, 1.2)).tolist(),
                'alpha': float(rng.uniform(0.5, 1.5)), 'g': float(rng.uniform(-0.4, 0.4)),
                'phi': _coef(rng)}
        if family == 'H2':
            spec['phi2'] = _coef(rng)
        if family == 'H3':
            spec['p1'], spec['p2'] = float(rng.uniform(-0.4, 0.4)), float(rng.uniform(0.0, 0.3))
        if _acceptable(spec):
            return spec
    raise RuntimeError('rejection sampling failed')


def _acceptable(spec: dict) -> bool:
    y2 = np.linspace(-2.5, 2.5, 501)
    folds = [_disc(spec['phi'], y2)]
    if 'phi2' in spec:
        folds.append(_disc(spec['phi2'], y2))
    if not any(np.any(np.diff(np.sign(f)) != 0) for f in folds):
        return False
    rng = np.random.default_rng(7)
    y = rng.standard_normal((20000, 2))
    pos = _phi(spec['phi'], y[:, 0], y[:, 1]) > 0
    if not 0.15 < pos.mean() < 0.85:
        return False
    if 'phi2' in spec:
        p2 = _phi(spec['phi2'], y[:, 0], y[:, 1])
        p1 = _phi(spec['phi'], y[:, 0], y[:, 1])
        interior = (p1 > 0) & (p2 > 0)
        if not 0.05 < interior.mean() < 0.7:
            return False
    return True


def _disc(q: dict, y2):
    b, c = _bc(q, y2)
    return b * b - 4.0 * q['a'] * c


def background(spec: dict, y: np.ndarray) -> np.ndarray:
    w, v = np.asarray(spec['w']), np.asarray(spec['v'])
    return (spec['beta0']
            + spec['beta1'] * (np.exp(y @ w - 0.5 * float(w @ w)) - 1.0)
            + spec['beta2'] * (np.cos(y @ v + spec['theta']) - math.exp(-0.5 * float(v @ v)) * math.cos(spec['theta'])))


def nonsmooth(spec: dict, y1, y2):
    s = spec['alpha'] * np.exp(spec['g'] * y2 - 0.5 * spec['g'] ** 2)
    p = _phi(spec['phi'], y1, y2)
    fam = spec['family']
    if fam == 'H1':
        return s * np.maximum(p, 0.0)
    if fam == 'H2':
        return s * np.maximum(np.maximum(p, _phi(spec['phi2'], y1, y2)), 0.0)
    if fam == 'H3':
        return s * (1.0 + spec['p1'] * y1 + spec['p2'] * y1 * y1) * (p > 0)
    raise ValueError(fam)


def value(spec: dict, y: np.ndarray) -> np.ndarray:
    return background(spec, y) + nonsmooth(spec, y[:, 0], y[:, 1])


# ---------------------------------------------------------------- reference, method A

def _pieces(spec: dict, y2: float) -> list[tuple[float, float, tuple[float, float, float]]]:
    """Intervals of y1 on which N(., y2)/s(y2) is one quadratic polynomial (non-zero pieces only)."""
    fam = spec['family']
    p1 = _poly(spec['phi'], y2)
    cuts = list(quadratic_roots(p1[2], p1[1], p1[0]))
    polys = [p1]
    if fam == 'H2':
        p2 = _poly(spec['phi2'], y2)
        diff = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        cuts += quadratic_roots(p2[2], p2[1], p2[0]) + quadratic_roots(diff[2], diff[1], diff[0])
        polys.append(p2)
    edges = [-math.inf] + sorted(cuts) + [math.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        if math.isinf(lo) and math.isinf(hi):
            mid = 0.0
        elif math.isinf(hi):
            mid = lo + 1.0
        elif math.isinf(lo):
            mid = hi - 1.0
        else:
            mid = 0.5 * (lo + hi)
        vals = [q[0] + q[1] * mid + q[2] * mid * mid for q in polys]
        if fam == 'H1' and vals[0] > 0:
            out.append((lo, hi, p1))
        elif fam == 'H2':
            j = int(np.argmax(vals))
            if vals[j] > 0:
                out.append((lo, hi, polys[j]))
        elif fam == 'H3' and vals[0] > 0:
            out.append((lo, hi, (1.0, spec['p1'], spec['p2'])))
    return out


def inner_exact(spec: dict, y2: float) -> float:
    """Integral over y1 of N(y1, y2) phi(y1) dy1, exact up to rounding."""
    s = spec['alpha'] * math.exp(spec['g'] * y2 - 0.5 * spec['g'] ** 2)
    return s * sum(poly2_gauss(coef, lo, hi) for lo, hi, coef in _pieces(spec, y2))


def _roots(fn, lim: float, step: float) -> list[float]:
    """Zeros of a smooth vectorized fn on [-lim, lim]: sign changes on a grid, plus pairs of zeros closer than the grid
    step, which hide at grid extrema of fn that lie within the local curvature of zero."""
    grid = np.arange(-lim, lim + step, step)
    vals = np.asarray(fn(grid), dtype=float)
    f1 = lambda t: float(fn(np.array([t]))[0])
    brent = lambda lo, hi: optimize.brentq(f1, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
    out = [brent(grid[i], grid[i + 1]) for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]]
    out += [float(grid[i]) for i in np.nonzero(vals == 0.0)[0]]
    sgn, mid = np.sign(vals), vals[1:-1]
    dl, dr = mid - vals[:-2], vals[2:] - mid
    near = sgn[1:-1] * mid <= np.abs(dr - dl)
    cand = np.nonzero((sgn[1:-1] != 0) & (sgn[:-2] == sgn[1:-1]) & (sgn[2:] == sgn[1:-1])
                      & (sgn[1:-1] * dl < 0) & (sgn[1:-1] * dr > 0) & near)[0] + 1
    for i in cand:
        s = sgn[i]
        res = optimize.minimize_scalar(lambda t: s * f1(t), bounds=(grid[i - 1], grid[i + 1]), method='bounded',
                                       options={'xatol': 1e-15})
        if s * f1(res.x) < 0:
            out += [brent(grid[i - 1], res.x), brent(res.x, grid[i + 1])]
    return out


def _structure_functions(spec: dict):
    """Smooth vectorized functions of y2 whose zeros are the y2 where the piece structure of N(., y2) can change."""
    q1 = spec['phi']
    fns = [lambda t: _disc(q1, t)]
    if spec['family'] == 'H2':
        q2 = spec['phi2']
        a1, a2 = q1['a'], q2['a']

        def diff_disc(t):
            (b1, c1), (b2, c2) = _bc(q1, t), _bc(q2, t)
            return b1 - b2 if a1 == a2 else (b1 - b2) ** 2 - 4 * (a1 - a2) * (c1 - c2)

        def resultant(t):
            (b1, c1), (b2, c2) = _bc(q1, t), _bc(q2, t)
            return (a1 * c2 - a2 * c1) ** 2 - (a1 * b2 - a2 * b1) * (b1 * c2 - b2 * c1)

        fns += [lambda t: _disc(q2, t), diff_disc, resultant]
    return fns


def breakpoints(spec: dict, lim: float = Y2_RANGE, step: float = 2e-3) -> list[float]:
    return sorted({r for fn in _structure_functions(spec) for r in _roots(fn, lim, step)})


def y1_breakpoints(spec: dict, lim: float = Y2_RANGE, step: float = 1e-3) -> list[float]:
    """y1 where the y2-crossing structure changes (method B's counterpart of `breakpoints`): points where a kink or
    jump curve has a vertical tangent, phi = d phi / d y2 = 0, which for phi = a y1^2 + b(y2) y1 + c(y2) means
    a c'^2 - b b' c' + c b'^2 = 0 at y1 = -c'/b'; for H2 also the curve phi1 = phi2 and the points phi1 = phi2 = 0."""
    qs = [spec['phi']] + ([spec['phi2']] if spec['family'] == 'H2' else [])

    def coefs(t):
        out = [(q['a'], *_bc(q, t), *_bc_prime(q, t)) for q in qs]
        if len(out) == 2:
            out.append(tuple(u - v for u, v in zip(out[0], out[1])))
        return out

    pts = []
    for k in range(len(coefs(0.0))):
        def fold(t, k=k):
            a, b, c, bp, cp = coefs(t)[k]
            return a * cp * cp - b * bp * cp + c * bp * bp
        for t in _roots(fold, lim, step):
            _, _, _, bp, cp = coefs(t)[k]
            if bp != 0.0:
                pts.append(float(-cp / bp))
    if len(qs) == 2:
        def resultant(t):
            (a1, b1, c1, _, _), (a2, b2, c2, _, _) = coefs(t)[:2]
            return (a1 * c2 - a2 * c1) ** 2 - (a1 * b2 - a2 * b1) * (b1 * c2 - b2 * c1)
        for t in _roots(resultant, lim, step):
            (a1, b1, c1, _, _), (a2, b2, c2, _, _) = coefs(t)[:2]
            if a2 * b1 - a1 * b2 != 0.0:
                pts.append(float((a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2)))
    return sorted(p for p in set(pts) if -lim < p < lim)


def inner_swapped(spec: dict, y1: float) -> float:
    """Integral over y2 of N(y1, y2) phi(y2) dy2, by quadrature split at numerically located crossings."""
    fns = [lambda t: _phi(spec['phi'], y1, t)]
    if spec['family'] == 'H2':
        fns += [lambda t: _phi(spec['phi2'], y1, t),
                lambda t: _phi(spec['phi'], y1, t) - _phi(spec['phi2'], y1, t)]
    cuts = sorted({r for fn in fns for r in _roots(fn, Y2_RANGE, 1e-3)})
    edges = [-Y2_RANGE] + cuts + [Y2_RANGE]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo > 1e-14:
            total += integrate.quad(lambda t: float(nonsmooth(spec, y1, t) * pdf(t)), lo, hi,
                                    epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    return total


def _fold_quad(g, inner_pts: list[float]) -> tuple[float, float]:
    """Integral of g over [-Y2_RANGE, Y2_RANGE] split at inner_pts. At a structure change the outer integrand behaves
    like smooth + C sqrt(|t - t*|) (jumps) or |t - t*|^{3/2} (kinks), so every piece is split at its midpoint and
    integrated in s with t = t* + s^2 from each breakpoint side, which makes the integrand smooth."""
    cuts = [-Y2_RANGE] + list(inner_pts) + [Y2_RANGE]
    opts = dict(epsabs=1e-15, epsrel=1e-14, limit=400)
    total, err = 0.0, 0.0
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        if hi - lo < 1e-14:
            continue
        lo_fold, hi_fold = i > 0, i < len(cuts) - 2
        mid = 0.5 * (lo + hi) if (lo_fold and hi_fold) else (hi if lo_fold else lo)
        if lo_fold:
            val, e = integrate.quad(lambda s: g(lo + s * s) * 2.0 * s, 0.0, math.sqrt(mid - lo), **opts)
            total, err = total + val, err + e
        if hi_fold:
            val, e = integrate.quad(lambda s: g(hi - s * s) * 2.0 * s, 0.0, math.sqrt(hi - mid), **opts)
            total, err = total + val, err + e
        if not lo_fold and not hi_fold:
            val, e = integrate.quad(g, lo, hi, **opts)
            total, err = total + val, err + e
    return total, err


def reference_swapped(spec: dict) -> dict:
    """Method B: the other integration order, y2-crossings found numerically for every y1, outer y1-integral split
    where those crossings change (independent of method A's pieces and structure functions)."""
    pts = y1_breakpoints(spec)
    total, err = _fold_quad(lambda t: inner_swapped(spec, t) * float(pdf(t)), pts)
    return {'value': spec['beta0'] + total, 'abs_err': err,
            'method': f'B: numeric y2 crossings, adaptive y1 in s = sqrt(|t - t*|) from {len(pts)} structure changes'}


def reference(spec: dict) -> dict:
    """Method A: exact inner y1-integral, adaptive outer y2-integral split at the structure changes."""
    inner_pts = [p for p in breakpoints(spec) if -Y2_RANGE < p < Y2_RANGE]
    total, err = _fold_quad(lambda t: inner_exact(spec, t) * float(pdf(t)), inner_pts)
    return {'value': spec['beta0'] + total, 'abs_err': err, 'method': 'A: exact y1 pieces, adaptive y2 in s = '
            f'sqrt(|t - t*|) from {len(inner_pts)} structure changes'}
