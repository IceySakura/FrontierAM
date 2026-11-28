"""Curved-interface families of blackbox-cubature v3.6, the extension track (DUAL_PLAN 9.18; the original author's
review of 2026-11-27: the hard track is built from quadrics, which a model can recognize and integrate in closed form).

Canonical coordinates y = Q z as on the hard track; every instance is f(y) = B(y) + N(y) with the hard track's smooth,
genuinely d-dimensional background B (synthetic.background, mean beta0). The interface of N is a random smooth function
(random Fourier features of a stationary Gaussian process, FEATURES terms), never a quadric, and the families are spread
over geometry, regularity, monotonicity, structural dimension and rarity:

    family  interface                                     regularity  monotone direction  k      rare
    C1      star-shaped closed surface                    jump        none                3, d   no
    C1t     the same in the tail, amplitude a / p         jump        none                3, d   p in [1e-3, 1e-2]
    C2      star-shaped closed surface                    kink        none                3, d   no
    C3      curved graph x1 = h(x2, x3)                   kink        x1                  3      no
    C4      two curved graphs crossing                    kink        x1                  3      no
    C5      curved graph in the tail, amplitude a / p     jump        x1                  3      p in [1e-3, 1e-2]
    C6      ridge of width eps along a curved graph       smooth      -                   3      no

Star bodies live in the first k coordinates: Omega = {c + r u : |u| = 1, 0 <= r < R(u)} with the centre c in span(e1, e2)
and R(u) = r0 exp(rho(s u1, s u2)), rho a random-feature function and s = sqrt(k / 2), so that the shape still varies
over the directions that carry the mass when k is large. The surface is smooth and closed, in general neither convex
nor a quadric; no direction is monotone. r0 is set so that P(Z in Omega) = p. C1 carries the smooth amplitude
alpha exp(w.y - |w|^2/2) (w zero on coordinates 3..k, so E N = alpha P(Z + w in Omega)); C2 is alpha (1 - r^2/R(u)^2)^+.
Graph families: h, h1, h2 random-feature functions of v = (x2, x3); C3 alpha (x1 - h)^+; C4 alpha max(x1 - h1,
s (x1 - h2), 0), s in [1.5, 3], with an interior kink at f > 0 where h2 > h1 (zero detection fails); C5 (a / p) 1{x1 > h};
C6 alpha exp(-(x1 - h)^2 / (2 eps^2)), eps in [0.1, 0.3].

References, two routes each:
  stars   polar coordinates around the (shifted) centre: R depends on u only through (u1, u2) = sin psi (cos theta,
          sin theta), so E N is a (psi, theta, r) integral with weight sin psi cos^(k-3) psi. Route A: tensor
          Gauss-Legendre x trapezoid x Gauss-Legendre (spectral), tolerance from a resolution doubling; route B: adaptive
          Gauss-Kronrod in (theta, psi) with another radial rule. Pointwise check (check_pointwise): the same code with
          rho = 0 against SciPy's non-central chi-square (and E[T 1{T < a}] = k F_{k+2}(a) + lambda F_{k+4}(a)).
  graphs  E N = E G(V), V = (x2, x3) ~ N(0, I_2), with the x1-integral G in closed form. Route A: tensor Gauss-Hermite;
          route B: adaptive Gauss-Kronrod over the plane. C4, whose G has a kink along h1 = h2: nested adaptive
          quadrature with the kink as breakpoints, x2 outer (A) and x3 outer (B). Pointwise check: G against adaptive
          quadrature in x1.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special

import synthetic
from gauss import pdf

FAMILIES = ('C1', 'C1t', 'C2', 'C3', 'C4', 'C5', 'C6')
STAR = ('C1', 'C1t', 'C2')
FEATURES = 8
SPAN = 10.0  # graph-family quadrature over [-SPAN, SPAN]^2: the Gaussian mass outside is below 1e-22
# C4's two nested routes (kink breakpoints, tangencies left to the adaptivity) do not reach the 1e-13 they request:
# on the v3.6 sets they are up to 2.3e-13 relative apart
C4_TOL = 1e-12


def k_levels(family: str, d: int) -> list[int]:
    return [3, d] if family in STAR else [3]


# ---------------------------------------------------------------- random-feature functions

def _features(rng, amplitude: float, offset: float = 0.0) -> dict:
    kappa = rng.uniform(0.6, 1.2)
    return {'h0': float(offset), 'omega': (rng.standard_normal((FEATURES, 2)) * kappa).tolist(),
            'phase': rng.uniform(0.0, 2 * math.pi, FEATURES).tolist(),
            'amp': (rng.standard_normal(FEATURES) * math.sqrt(2.0 / FEATURES) * amplitude).tolist()}


def feature(fn: dict, v: np.ndarray) -> np.ndarray:
    """h0 + sum_j amp_j cos(omega_j . v + phase_j) for v of shape (..., 2)."""
    return fn['h0'] + np.cos(v @ np.asarray(fn['omega']).T + np.asarray(fn['phase'])) @ np.asarray(fn['amp'])


def radius(spec: dict, u12: np.ndarray) -> np.ndarray:
    return spec['r0'] * np.exp(feature(spec['rho'], spec['s'] * u12))


# ---------------------------------------------------------------- sampling

def _background(spec: dict, d: int, rng) -> None:
    spec.update(beta0=float(rng.uniform(1.0, 2.0)), beta1=float(rng.uniform(0.2, 0.5)), c0=float(rng.uniform(0.3, 0.6)),
                om=rng.uniform(0.5, 1.5, d).tolist(), ph=rng.uniform(0.0, 2 * math.pi, d).tolist(),
                bg_seed=int(rng.integers(2 ** 31)))


def _unit2(rng) -> np.ndarray:
    t = rng.uniform(0.0, 2 * math.pi)
    return np.array([math.cos(t), math.sin(t)])


def sample(family: str, d: int, k: int, rng: np.random.Generator) -> dict:
    spec = {'family': family, 'd': int(d), 'k': int(k)}
    _background(spec, d, rng)
    if family in STAR:
        tail = family == 'C1t'
        spec.update(c=(_unit2(rng) * rng.uniform(1.5, 3.0) if tail else _unit2(rng) * rng.uniform(0.0, 1.5)).tolist(),
                    rho=_features(rng, rng.uniform(0.15, 0.3)), s=math.sqrt(k / 2.0), w=[0.0] * d)
        p = 10 ** rng.uniform(-3, -2) if tail else rng.uniform(0.15, 0.6) if family == 'C1' else rng.uniform(0.2, 0.6)
        spec.update(p=float(p), r0=_solve_r0(spec, p))
        if family == 'C1':
            free = np.r_[0:2, k:d]
            w = np.zeros(d)
            w[free] = rng.standard_normal(len(free))
            spec.update(alpha=float(rng.uniform(0.5, 1.5)), w=(w / np.linalg.norm(w) * rng.uniform(0.1, 0.5)).tolist())
        elif family == 'C1t':
            spec.update(alpha=float(rng.uniform(0.5, 1.5) / p))
        else:
            spec.update(alpha=float(rng.uniform(1.0, 3.0)))
    elif family == 'C3':
        spec.update(h=_features(rng, rng.uniform(0.5, 1.0), rng.uniform(-0.5, 0.5)), alpha=float(rng.uniform(0.5, 1.5)))
    elif family == 'C4':
        spec.update(h1=_features(rng, rng.uniform(0.5, 1.0), rng.uniform(-0.5, 0.5)),
                    h2=_features(rng, rng.uniform(0.5, 1.0), rng.uniform(-0.5, 0.5)),
                    slope=float(rng.uniform(1.5, 3.0)), alpha=float(rng.uniform(0.5, 1.5)))
    elif family == 'C5':
        h = _features(rng, rng.uniform(0.5, 1.0))
        p = 10 ** rng.uniform(-3, -2)
        h['h0'] = _solve_h0(h, p)
        spec.update(h=h, p=float(p), alpha=float(rng.uniform(0.5, 1.5) / p))
    elif family == 'C6':
        spec.update(h=_features(rng, rng.uniform(0.5, 1.0), rng.uniform(-1.0, 1.0)),
                    eps=float(10 ** rng.uniform(-1.0, math.log10(0.3))), alpha=float(rng.uniform(2.0, 5.0)))
    else:
        raise ValueError(family)
    return spec


# ---------------------------------------------------------------- evaluation

def nonsmooth(spec: dict, y: np.ndarray) -> np.ndarray:
    fam = spec['family']
    if fam in STAR:
        xi = y[:, :spec['k']].copy()
        xi[:, :2] -= np.asarray(spec['c'])
        r = np.linalg.norm(xi, axis=1)
        big_r = radius(spec, xi[:, :2] / np.where(r > 0, r, 1.0)[:, None])
        if fam == 'C2':
            return spec['alpha'] * np.maximum(1.0 - np.square(r / big_r), 0.0)
        amp = spec['alpha'] * (np.exp(y @ np.asarray(spec['w']) - 0.5 * np.dot(spec['w'], spec['w']))
                               if fam == 'C1' else 1.0)
        return amp * (r < big_r)
    x1, v = y[:, 0], y[:, 1:3]
    if fam == 'C3':
        return spec['alpha'] * np.maximum(x1 - feature(spec['h'], v), 0.0)
    if fam == 'C4':
        return spec['alpha'] * np.maximum(np.maximum(x1 - feature(spec['h1'], v),
                                                     spec['slope'] * (x1 - feature(spec['h2'], v))), 0.0)
    if fam == 'C5':
        return spec['alpha'] * (x1 > feature(spec['h'], v))
    return spec['alpha'] * np.exp(-0.5 * np.square((x1 - feature(spec['h'], v)) / spec['eps']))


def value(spec: dict, y: np.ndarray) -> np.ndarray:
    return synthetic.background(spec, y) + nonsmooth(spec, y)


# ---------------------------------------------------------------- star bodies: polar coordinates around the centre

def _star_setup(spec: dict) -> tuple[int, np.ndarray, float]:
    """k, the centre of the measure-shifted body, and log of (2 pi)^(-k/2) |S^(k-3)| exp(-|c'|^2 / 2)."""
    k = spec['k']
    cp = np.asarray(spec['c'], float) - (np.asarray(spec['w'][:2]) if spec['family'] == 'C1' else 0.0)
    log_k = (-0.5 * k * math.log(2 * math.pi) + math.log(2.0) + 0.5 * (k - 2) * math.log(math.pi)
             - special.gammaln(0.5 * (k - 2)) - 0.5 * float(cp @ cp))
    return k, cp, log_k


def _radial(spec: dict, k: int, cp: np.ndarray, log_k: float, psi, theta, nodes, weights, rho: bool = True):
    """sin psi cos^(k-3) psi times the radial integral from 0 to R(u) of A(r) phi_k(c' + r u) r^(k-1), on a grid of
    (psi, theta) (any broadcastable shapes), with the radial Gauss-Legendre rule (nodes, weights) on [-1, 1]."""
    psi, theta = np.broadcast_arrays(np.asarray(psi, float), np.asarray(theta, float))
    u12 = np.sin(psi)[..., None] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    big_r = radius(spec, u12) if rho else np.full(psi.shape, float(spec['r0']))
    t = u12 @ cp
    r = 0.5 * big_r[..., None] * (nodes + 1.0)
    f = np.exp(log_k + (k - 1) * np.log(r) - r * t[..., None] - 0.5 * r * r)
    if spec['family'] == 'C2':
        f = f * (1.0 - np.square(r / big_r[..., None]))
    angular = np.sin(psi) * (np.cos(psi) ** (k - 3) if k > 3 else 1.0)
    return angular * 0.5 * big_r * (f @ weights)


def _star_tensor(spec: dict, n_psi: int, n_theta: int, n_r: int, rho: bool = True) -> float:
    k, cp, log_k = _star_setup(spec)
    x, wx = np.polynomial.legendre.leggauss(n_psi)
    psi, w_psi = 0.25 * math.pi * (x + 1.0), 0.25 * math.pi * wx
    theta = np.arange(n_theta) * (2 * math.pi / n_theta)
    nodes, weights = np.polynomial.legendre.leggauss(n_r)
    rows = [_radial(spec, k, cp, log_k, psi[i:i + 16, None], theta[None, :], nodes, weights, rho).sum(axis=1)
            for i in range(0, n_psi, 16)]
    return float(w_psi @ np.concatenate(rows) * (2 * math.pi / n_theta))


def _star_adaptive(spec: dict, n_r: int = 150) -> float:
    k, cp, log_k = _star_setup(spec)
    nodes, weights = np.polynomial.legendre.leggauss(n_r)

    def over_psi(theta):
        return integrate.quad(lambda p: float(_radial(spec, k, cp, log_k, p, theta, nodes, weights)), 0.0, math.pi / 2,
                              epsabs=1e-16, epsrel=1e-13, limit=300)[0]

    return integrate.quad(over_psi, 0.0, 2 * math.pi, epsabs=1e-16, epsrel=1e-13, limit=300)[0]


def _solve_r0(spec: dict, p: float) -> float:
    """r0 with P(Z in Omega) = p (the unweighted body around c)."""
    probe = {**spec, 'family': 'C1t'}
    k = spec['k']

    def gap(log_r0):
        return math.log(max(_star_tensor({**probe, 'r0': math.exp(log_r0)}, 40, 64, 48), 1e-300)) - math.log(p)

    hi = math.log(math.sqrt(k) + 3.5 + 12.0)
    return float(math.exp(optimize.brentq(gap, math.log(1e-3), hi, xtol=1e-12)))


# ---------------------------------------------------------------- graph families: closed-form x1 integral

def _psi_excess(t):
    """E (X - t)^+ for X ~ N(0, 1)."""
    return pdf(t) - t * special.ndtr(-t)


def preintegrated(spec: dict, v: np.ndarray) -> np.ndarray:
    """G(v) = E over x1 of N(x1, v) (closed form), v of shape (..., 2)."""
    fam, alpha = spec['family'], spec['alpha']
    if fam == 'C3':
        return alpha * _psi_excess(feature(spec['h'], v))
    if fam == 'C5':
        return alpha * special.ndtr(-feature(spec['h'], v))
    if fam == 'C6':
        e2 = spec['eps'] ** 2
        return alpha * math.sqrt(e2 / (1.0 + e2)) * np.exp(-0.5 * np.square(feature(spec['h'], v)) / (1.0 + e2))
    # N = (x1 - h1)^+ below the crossing xs of the two lines and s (x1 - h2)^+ above it; where h2 <= h1 only the
    # second piece is ever positive
    h1, h2, s = feature(spec['h1'], v), feature(spec['h2'], v), spec['slope']
    xs = (s * h2 - h1) / (s - 1.0)
    between = np.where(h1 >= 0.0, special.ndtr(-h1) - special.ndtr(-xs),
                       np.where(xs <= 0.0, special.ndtr(xs) - special.ndtr(h1), 1.0 - special.ndtr(h1) - special.ndtr(-xs)))
    crossed = pdf(h1) - pdf(xs) - h1 * between + s * (pdf(xs) - h2 * special.ndtr(-xs))
    return alpha * np.where(h2 > h1, crossed, s * _psi_excess(h2))


def _hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


def _graph_tensor(spec: dict, n: int) -> float:
    x, w = _hermite(n)
    v = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    return float(w @ preintegrated(spec, v) @ w)


def _graph_adaptive(spec: dict) -> float:
    def inner(a):
        return integrate.quad(lambda b: float(preintegrated(spec, np.array([a, b]))) * float(pdf(b)), -SPAN, SPAN,
                              epsabs=1e-17, epsrel=1e-13, limit=300)[0]

    return integrate.quad(lambda a: inner(a) * float(pdf(a)), -SPAN, SPAN, epsabs=1e-17, epsrel=1e-13, limit=300)[0]


def _crossings(fun, lo: float = -SPAN, hi: float = SPAN, n: int = 4001) -> list[float]:
    x = np.linspace(lo, hi, n)
    y = fun(x)
    idx = np.flatnonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)
    return [optimize.brentq(fun, x[i], x[i + 1], xtol=1e-15) for i in idx]


def _crease_nested(spec: dict, outer_axis: int) -> float:
    """C4: nested adaptive quadrature over the plane with the kink of G (h1 = h2) as breakpoints of the inner integral."""
    def plane(a, b):
        b = np.asarray(b, float)
        pair = [np.full(b.shape, a), b] if outer_axis == 0 else [b, np.full(b.shape, a)]
        return np.stack(pair, axis=-1)

    def inner(a):
        cuts = [-SPAN, *_crossings(lambda b: feature(spec['h1'], plane(a, b)) - feature(spec['h2'], plane(a, b))), SPAN]
        return sum(integrate.quad(lambda b: float(preintegrated(spec, plane(a, b))) * float(pdf(b)), lo, hi,
                                  epsabs=1e-17, epsrel=1e-13, limit=300)[0] for lo, hi in zip(cuts[:-1], cuts[1:]))

    return integrate.quad(lambda a: inner(a) * float(pdf(a)), -SPAN, SPAN, epsabs=1e-17, epsrel=1e-13, limit=400)[0]


def _solve_h0(h: dict, p: float) -> float:
    """h0 with E Phi(-(h0 + h~(V))) = p."""
    x, w = _hermite(64)
    v = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    base = feature({**h, 'h0': 0.0}, v)

    def gap(h0):
        return math.log(max(float(w @ special.ndtr(-(h0 + base)) @ w), 1e-300)) - math.log(p)

    return float(optimize.brentq(gap, -5.0, 15.0, xtol=1e-12))


# ---------------------------------------------------------------- references and checks

def _route(method: str, val: float, tol: float) -> dict:
    return {'method': method, 'value': float(val), 'abs_err': max(float(tol), 1e-15 * max(1.0, abs(val)))}


def reference(spec: dict) -> dict:
    """Route A."""
    fam = spec['family']
    if fam in STAR:
        lo, hi = (spec['alpha'] * _star_tensor(spec, n, 5 * n // 3, n) for n in (96, 192))
        return _route('A: polar coordinates around the centre, tensor Gauss-Legendre x trapezoid x Gauss-Legendre '
                      '(192 x 320 x 192)', spec['beta0'] + hi, abs(hi - lo))
    if fam == 'C4':
        val = spec['beta0'] + _crease_nested(spec, 0)
        return _route('A: closed-form x1 integral, nested adaptive Gauss-Kronrod (x2 outer) with the kink as '
                      'breakpoints', val, C4_TOL * max(1.0, abs(val)))
    lo, hi = _graph_tensor(spec, 200), _graph_tensor(spec, 300)
    return _route('A: closed-form x1 integral, tensor Gauss-Hermite (300 x 300)', spec['beta0'] + hi, abs(hi - lo))


def reference_second(spec: dict) -> dict:
    """Route B."""
    fam = spec['family']
    if fam in STAR:
        return _route('B: polar coordinates around the centre, adaptive Gauss-Kronrod in (theta, psi), radial '
                      'Gauss-Legendre (150)', spec['beta0'] + spec['alpha'] * _star_adaptive(spec), 1e-13)
    if fam == 'C4':
        val = spec['beta0'] + _crease_nested(spec, 1)
        return _route('B: closed-form x1 integral, nested adaptive Gauss-Kronrod (x3 outer) with the kink as '
                      'breakpoints', val, C4_TOL * max(1.0, abs(val)))
    return _route('B: closed-form x1 integral, adaptive Gauss-Kronrod over the plane',
                  spec['beta0'] + _graph_adaptive(spec), 1e-13)


def check_pointwise(spec: dict, n: int = 6, seed: int = 0) -> dict:
    """Stars: the polar-coordinate code with rho = 0 against SciPy's non-central chi-square law of |Z - c'|^2 (relative
    error). Graphs: the closed-form x1 integral G(v) against adaptive quadrature in x1 at n random v (relative error)."""
    fam = spec['family']
    if fam in STAR:
        k, cp, _ = _star_setup(spec)
        a, lam = spec['r0'] ** 2, float(cp @ cp)
        law = (lambda df: special.chndtr(a, df, lam)) if lam > 0 else (lambda df: special.chdtr(df, a))
        exact = law(k) if fam != 'C2' else law(k) - (k * law(k + 2) + lam * law(k + 4)) / a
        return {'reduction': abs(_star_tensor(spec, 192, 256, 192, rho=False) - exact) / exact}
    rng = np.random.default_rng(seed)
    worst = 0.0
    for v in rng.standard_normal((n, 2)) * 1.5:
        g = float(preintegrated(spec, v[None, :])[0])
        y = lambda x: float(nonsmooth(spec, np.r_[x, v][None, :])[0])  # noqa: E731
        if fam == 'C4':
            a, b = float(feature(spec['h1'], v)), float(feature(spec['h2'], v))
            cuts = sorted({a, b, (spec['slope'] * b - a) / (spec['slope'] - 1.0)})
        else:
            cuts = [float(feature(spec['h'], v))]
        cuts = [-12.0, *[c for c in cuts if -12.0 < c < 12.0], 12.0]
        q = sum(integrate.quad(lambda x: y(x) * float(pdf(x)), lo, hi, epsabs=1e-17, epsrel=1e-13, limit=300)[0]
                for lo, hi in zip(cuts[:-1], cuts[1:]))
        worst = max(worst, abs(g - q) / max(abs(g), 1e-300))
    return {'closed_form': worst}
