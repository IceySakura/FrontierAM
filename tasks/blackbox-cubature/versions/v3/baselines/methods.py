"""Black-box baselines and the structure-aware oracle for blackbox-cubature v3 (milestones M2, M2b).

Every black-box method sees only a query-counting oracle f (points in R^d as rows) with a budget; probes, finite
differences and line searches all count. Methods return (estimate, note).

- plain_mc: independent N(0, I) points, the whole budget (zero anchor, as in v2).
- plain_rqmc: scrambled Sobol' points mapped to N(0, I), the whole budget.
- as_rqmc: GPCA / active-subspace RQMC (Xiao and Wang 2019; Liu and Owen 2023): forward-difference gradients give the
  eigenvectors of C = E[grad f grad f^T]; the main phase is one scrambled Sobol' net in that basis, leading eigenvector
  first, of the largest power-of-two size that fits (nets are balanced only in blocks of 2^m); the queries left over
  buy more gradients.
- as_preint: the same pilot, then preintegration along the leading eigenvector with numerical smoothing on the
  largest power-of-two number of lines the budget affords (see line_integrals).
- strong_preint / strong_rqmc: the literature pipeline in black-box form (DUAL_PLAN section 9.1): optimal-drift
  importance sampling (Glasserman, Heidelberger, Shahabuddin 1999; with preintegration, Chen, Xu, Du, Wang 2025),
  validated on fresh points; the active subspace / GPCA basis (Liu and Owen 2023; Xiao and Wang 2019) from
  finite-difference gradients plus the Stein mean gradient E[f(Z) Z]; then preintegration along the leading direction
  with numerical smoothing (Bayer, Ben Hammouda, Tempone 2023) and GPCA order for the other variables (Xiao and Wang
  2018), or one GPCA-RQMC net. Without a zero region the boundaries cannot be located from function values and the
  preint mode also uses the net.
- strong: the frozen reference, the method named by STRONG. On the prototype set (11-24) GPCA-RQMC beat both
  pipeline modes once every query counts, so the reference is as_rqmc; the pipeline stays as a documented variant.
- oracle_preint (diagnostic, reads the instance's canonical structure): preintegration along a structural direction
  with the line integral in closed form, one query per line, scrambled Sobol' over the complement ordered by the known
  structure. It stands for the accuracy of preintegration when directions and kinks cost nothing.
- oracle_split (diagnostic, synthetic families only): the non-smooth part exactly, the smooth background by
  GPCA-RQMC with the whole budget; the ceiling for methods that separate the non-smooth structure from the rest.

Line integrals (numerical smoothing without access to the kink function): each line is probed at Gauss-Hermite
nodes; where the probes switch between zero and non-zero values the boundary is bracketed, bisected until two
non-zero points are known and then refined by secant steps along the non-zero branch (the Newton step of numerical
smoothing, since the payoff is the positive part of a smooth function there); a flat non-zero side (a jump) keeps
bisecting. Every non-zero piece is integrated by the Gauss rule of the normal density restricted to it; lines
without a boundary keep the Gauss-Hermite value.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'generation'))
import options  # noqa: E402
import paths  # noqa: E402
import synthetic  # noqa: E402

SQRT2PI = math.sqrt(2.0 * math.pi)
FD_STEP = 0.05
PILOT_FRACTION = 1 / 8
PILOT_GRADIENTS = 64
ODIS_FRACTION = 1 / 16
ODIS_START = 128
ODIS_CAP = 4.0
ODIS_CHECK = 256
STRONG = 'as_rqmc'
PROBES = 8
BISECT = 14
PIECE_NODES = 6
U_MAX = 8.5
CHUNK = 128
ZERO_RANGE = (0.05, 0.95)
LOGNORMAL = ('basket_call', 'asian_call', 'basket_asian_call', 'asian_spread')


class BudgetExceeded(RuntimeError):
    pass


class NotApplicable(RuntimeError):
    pass


class Counted:
    """Query-counting black box: every evaluated point counts, repeats included."""

    def __init__(self, f, d: int, budget: int):
        self.f, self.d, self.budget, self.used = f, d, budget, 0

    def __call__(self, z) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if self.used + len(z) > self.budget:
            raise BudgetExceeded(f'{self.used} + {len(z)} > {self.budget}')
        self.used += len(z)
        return self.f(z)

    @property
    def remaining(self) -> int:
        return self.budget - self.used


class Shifted:
    """Importance sampling with the drift mu: f(z + mu) exp(-mu.z - |mu|^2/2) has the same N(0, I) expectation as f;
    one query of f per point."""

    def __init__(self, f: Counted, mu: np.ndarray):
        self.f, self.mu, self.half = f, mu, 0.5 * float(mu @ mu)

    def __call__(self, z) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.f(z + self.mu) * np.exp(-z @ self.mu - self.half)

    budget = property(lambda self: self.f.budget)
    used = property(lambda self: self.f.used)
    remaining = property(lambda self: self.f.remaining)


def sobol_normal(d: int, n: int, seed: int) -> np.ndarray:
    """First n points of a scrambled Sobol' sequence mapped to N(0, I_d); n need not be a power of two."""
    sampler = stats.qmc.Sobol(d, scramble=True, rng=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = sampler.random(n)
    return special.ndtri(np.clip(u, 1e-16, 1 - 1e-16))


def pow2(n: int) -> int:
    """Largest power of two <= n (n >= 1)."""
    return 1 << (int(n).bit_length() - 1)


# ---------------------------------------------------------------- building blocks

def gradients(f, d: int, n: int, seed: int, points: bool = False):
    """Forward-difference gradients (step FD_STEP) at n scrambled Sobol' points, f at those points (and the points)."""
    z = sobol_normal(d, n, seed)
    vals = f(np.concatenate([z] + [z + FD_STEP * e for e in np.eye(d)])).reshape(d + 1, n)
    grads = (vals[1:] - vals[0]).T / FD_STEP
    return (grads, vals[0], z) if points else (grads, vals[0])


def as_basis(grads: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Eigenvectors of the gradient covariance (plus extra extra^T if given), largest eigenvalue first."""
    c = grads.T @ grads / len(grads)
    if extra is not None:
        c = c + np.outer(extra, extra)
    eig, vec = np.linalg.eigh(c)
    return vec[:, np.argsort(eig)[::-1]]


def _pilot_size(f, d: int) -> int:
    return max(1, min(PILOT_GRADIENTS, int(f.budget * PILOT_FRACTION) // (d + 1)))


def _rqmc_in_basis(f, d: int, seed: int, grads: np.ndarray):
    """GPCA-RQMC main phase: one scrambled Sobol' net of the largest power-of-two size that fits in the remaining
    budget; the queries left over before it buy more gradients."""
    main = pow2(f.remaining)
    extra = (f.remaining - main) // (d + 1)
    if extra:
        grads = np.vstack([grads, gradients(f, d, extra, seed + 2)[0]])
    est = float(f(sobol_normal(d, main, seed) @ as_basis(grads).T).mean())
    return est, f'rqmc, {main} points, {len(grads)} gradients'


def odis_drift(f: Counted, d: int, seed: int) -> np.ndarray:
    """Optimal drift mu* = argmax f(z) phi(z) by forward-difference gradient ascent on log f(z) - |z|^2/2 from the best
    point of a scrambled Sobol' start sample (budget share ODIS_FRACTION). Used only for rare events (positive values
    on less than 20% of the start sample, or a spike above 20 times its median); kept only if |mu| <= ODIS_CAP and, on
    ODIS_CHECK fresh shifted points, the importance-sampling variance is below a quarter of the plain variance
    estimated from the same points; otherwise zero."""
    budget = int(f.budget * ODIS_FRACTION)
    n0 = min(ODIS_START, budget // 2)
    if n0 < 1:
        return np.zeros(d)
    z = sobol_normal(d, n0, seed + 3)
    fz = f(z)
    rare = np.mean(fz > 0) < 0.2 or np.max(fz) > 20.0 * max(float(np.median(np.abs(fz))), 1e-300)
    if not np.any(fz > 0) or not rare:
        return np.zeros(d)
    score = lambda x, fx: math.log(fx) - 0.5 * float(x @ x)
    i = int(np.argmax(np.where(fz > 0, np.log(np.maximum(fz, 1e-300)) - 0.5 * np.sum(z * z, axis=1), -np.inf)))
    x, fx = z[i], float(fz[i])
    sx, used = score(x, fx), n0
    while used + d + 3 + ODIS_CHECK <= budget:
        fp = f(x + FD_STEP * np.eye(d))
        used += d
        g = (np.log(np.maximum(fp, 1e-300)) - math.log(fx)) / FD_STEP - x
        norm = float(np.linalg.norm(g))
        if norm < 1e-6:
            break
        step, moved = g / max(1.0, norm), False
        for _ in range(3):
            xn = x + step
            fn = float(f(xn[None, :])[0])
            used += 1
            if fn > 0 and score(xn, fn) > sx:
                x, fx, sx, moved = xn, fn, score(xn, fn), True
                break
            step = 0.5 * step
        if not moved:
            break
    if float(np.linalg.norm(x)) > ODIS_CAP:
        return np.zeros(d)
    zc = sobol_normal(d, ODIS_CHECK, seed + 5)
    raw = f(zc + x)
    w = np.exp(-zc @ x - 0.5 * float(x @ x))
    shifted = raw * w
    plain_var = float(np.mean(raw * raw * w) - np.mean(shifted) ** 2)
    return x if np.var(shifted) < 0.25 * plain_var else np.zeros(d)


def truncated_gauss(a: np.ndarray, b: np.ndarray, n: int = PIECE_NODES, m: int = 64):
    """n-point Gauss rules of the normal density restricted to [a_s, b_s], one per row: discretized Stieltjes
    procedure on m Gauss-Legendre points of each interval, then the eigen-decomposition of the Jacobi matrix."""
    xi, om = np.polynomial.legendre.leggauss(m)
    half = ((b - a) / 2.0)[:, None]
    x = ((a + b) / 2.0)[:, None] + half * xi
    w = half * om * np.exp(-0.5 * x * x) / SQRT2PI
    alpha, beta = np.zeros((len(a), n)), np.zeros((len(a), n))
    p_prev, p = np.zeros_like(x), np.ones_like(x)
    norm_prev, norm = None, w.sum(axis=1)
    mass = norm.copy()
    for k in range(n):
        alpha[:, k] = (w * x * p * p).sum(axis=1) / norm
        if k:
            beta[:, k] = norm / norm_prev
        p_prev, p = p, (x - alpha[:, k:k + 1]) * p - beta[:, k:k + 1] * p_prev
        norm_prev, norm = norm, (w * p * p).sum(axis=1)
    jac = np.zeros((len(a), n, n))
    idx = np.arange(n)
    jac[:, idx, idx] = alpha
    jac[:, idx[1:], idx[:-1]] = jac[:, idx[:-1], idx[1:]] = np.sqrt(beta[:, 1:])
    nodes, vecs = np.linalg.eigh(jac)
    return nodes, mass[:, None] * vecs[:, 0, :] ** 2


def _locate(f, theta, base, a, b, fb):
    """Boundary between a zero point a and a non-zero point b (value fb) on each line, see the module docstring."""
    b2, fb2 = np.full_like(b, np.nan), np.full_like(fb, np.nan)
    active = np.ones(len(b), dtype=bool)
    last = np.full_like(b, np.nan)
    for _ in range(BISECT):
        i = np.nonzero(active)[0]
        if len(i) == 0:
            break
        slope = np.isfinite(fb2[i]) & (np.abs(fb[i] - fb2[i]) > 1e-12 * np.abs(fb[i]))
        with np.errstate(divide='ignore', invalid='ignore'):
            sec = b[i] - fb[i] * (b[i] - b2[i]) / (fb[i] - fb2[i])
        lo, hi = np.minimum(a[i], b[i]), np.maximum(a[i], b[i])
        use = slope & (sec > lo) & (sec < hi)
        r = np.where(use, sec, 0.5 * (a[i] + b[i]))
        fr = f(base[i] + r[:, None] * theta)
        zero = fr == 0.0
        a[i] = np.where(zero, r, a[i])
        b2[i], fb2[i] = np.where(zero, b2[i], b[i]), np.where(zero, fb2[i], fb[i])
        b[i], fb[i] = np.where(zero, b[i], r), np.where(zero, fb[i], fr)
        done = (np.abs(b[i] - a[i]) < 1e-10) | (use & (np.abs(r - last[i]) < 1e-10))
        last[i] = np.where(use, r, np.nan)
        active[i[done]] = False
    slope = np.isfinite(fb2) & (np.abs(fb - fb2) > 1e-12 * np.abs(fb))
    with np.errstate(divide='ignore', invalid='ignore'):
        sec = b - fb * (b - b2) / (fb - fb2)
    inside = slope & (sec >= np.minimum(a, b)) & (sec <= np.maximum(a, b))
    return np.where(inside, sec, 0.5 * (a + b))


def line_integrals(f, theta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """E over u ~ N(0, 1) of f(base + u theta) for every row of base (numerical smoothing, see the module docstring).
    Refinement stops at the budget; lines left over keep their Gauss-Hermite value."""
    n, d = base.shape
    x, w = np.polynomial.hermite_e.hermegauss(PROBES)
    vals = f((base[:, None, :] + x[None, :, None] * theta).reshape(-1, d)).reshape(n, PROBES)
    out = vals @ (w / SQRT2PI)
    zero = vals == 0.0
    out[zero.all(axis=1)] = 0.0
    switch = zero[:, :-1] != zero[:, 1:]
    flips = switch.sum(axis=1)
    pieces = np.where(zero[:, 0], (flips + 1) // 2, flips // 2 + 1)
    cost = np.where(flips > 0, BISECT * flips + PIECE_NODES * pieces, 0)
    refine = (flips > 0) & (np.cumsum(cost) <= f.remaining)
    li, ki = np.nonzero(switch & refine[:, None])
    if len(li) == 0:
        return out
    lo_zero = zero[li, ki]
    a = np.where(lo_zero, x[ki], x[ki + 1])
    b = np.where(lo_zero, x[ki + 1], x[ki])
    fb = np.where(lo_zero, vals[li, ki + 1], vals[li, ki])
    bnd = _locate(f, theta, base[li], a, b, fb)
    seg_line, seg_a, seg_b = [], [], []
    for i in np.nonzero(refine)[0]:
        cuts = np.concatenate([[-U_MAX], np.sort(bnd[li == i]), [U_MAX]])
        for j in range(len(cuts) - 1):
            if zero[i, 0] == (j % 2 == 1):
                seg_line.append(i)
                seg_a.append(cuts[j])
                seg_b.append(cuts[j + 1])
    seg_line = np.array(seg_line)
    u, wts = truncated_gauss(np.array(seg_a), np.array(seg_b))
    pv = f((base[seg_line][:, None, :] + u[:, :, None] * theta).reshape(-1, d)).reshape(len(seg_line), PIECE_NODES)
    res = np.zeros(n)
    np.add.at(res, seg_line, (pv * wts).sum(axis=1))
    out[refine] = res[refine]
    return out


def _preint(f, d: int, seed: int, basis: np.ndarray):
    """Lines along basis[:, 0] at scrambled Sobol' points of the complement (GPCA order); the line count is the largest
    power of two affordable at the cost per line measured on the first chunk."""
    theta, comp = basis[:, 0], basis[:, 1:]
    reserve = PROBES + 2 * (BISECT + PIECE_NODES)
    cand = sobol_normal(d - 1, max(1, f.remaining // PROBES), seed) @ comp.T
    start = f.used
    k = min(CHUNK, len(cand), max(1, f.remaining // reserve))
    vals = list(line_integrals(f, theta, cand[:k]))
    target = min(len(cand), pow2(max(1, int((f.budget - start) / (1.1 * max(f.used - start, 1) / k)))))
    while len(vals) < target:
        k = min(CHUNK, target - len(vals), f.remaining // reserve)
        if k <= 0:
            break
        vals += list(line_integrals(f, theta, cand[len(vals):len(vals) + k]))
    n = pow2(len(vals))
    return float(np.mean(vals[:n])), f'preint, {n} lines'


# ---------------------------------------------------------------- black-box methods

def plain_mc(f: Counted, d: int, seed: int, info=None):
    z = np.random.default_rng([seed, 99]).standard_normal((f.remaining, d))
    return float(f(z).mean()), 'mc'


def plain_rqmc(f: Counted, d: int, seed: int, info=None):
    return float(f(sobol_normal(d, f.remaining, seed)).mean()), 'rqmc'


def as_rqmc(f: Counted, d: int, seed: int, info=None):
    return _rqmc_in_basis(f, d, seed, gradients(f, d, _pilot_size(f, d), seed + 1)[0])


def as_preint(f: Counted, d: int, seed: int, info=None):
    grads, _ = gradients(f, d, _pilot_size(f, d), seed + 1)
    est, note = _preint(f, d, seed, as_basis(grads))
    return est, f'{note}, {len(grads)} gradients'


def _pipeline(f: Counted, d: int, seed: int, mode: str):
    """Mode 'preint': ODIS (when validated) with preintegration and numerical smoothing, the combination of Chen, Xu,
    Du, Wang 2025 (when the pilot sees a zero region). Mode 'rqmc': one GPCA-RQMC net (Xiao and Wang 2019), without
    importance sampling, whose exponential weights an unsmoothed jump does not tolerate. Both use the active subspace
    from finite-difference gradients plus the Stein mean gradient E[f(Z) Z], which also sees jumps."""
    mu = odis_drift(f, d, seed) if mode == 'preint' else np.zeros(d)
    g = Shifted(f, mu) if np.any(mu) else f
    grads, base, z = gradients(g, d, _pilot_size(f, d), seed + 1, points=True)
    basis = as_basis(grads, (base[:, None] * z).mean(axis=0))
    tag = f'ODIS |mu| = {float(np.linalg.norm(mu)):.2f}'
    if mode == 'preint' and ZERO_RANGE[0] <= float(np.mean(base == 0.0)) <= ZERO_RANGE[1]:
        est, note = _preint(g, d, seed, basis)
        return est, f'{note}, {tag}'
    main = pow2(g.remaining)
    return float(g(sobol_normal(d, main, seed) @ basis.T).mean()), f'rqmc, {main} points, {tag}'


def strong_preint(f: Counted, d: int, seed: int, info=None):
    return _pipeline(f, d, seed, 'preint')


def strong_rqmc(f: Counted, d: int, seed: int, info=None):
    return _pipeline(f, d, seed, 'rqmc')


def strong(f: Counted, d: int, seed: int, info=None):
    return METHODS[STRONG](f, d, seed, info)


# ---------------------------------------------------------------- structure-aware oracle (diagnostic)

def _oracle_synthetic(spec: dict, n: int, seed: int) -> np.ndarray:
    """Preintegrate along y_0, the first coordinate of the non-smooth subspace: the non-smooth part along the line is
    a piecewise quadratic (or an interval indicator) in closed form, the background by 32-point Gauss-Hermite; Sobol'
    coordinates in natural order (the rest of the non-smooth subspace first)."""
    d, fam, forms = spec['d'], spec['family'], spec['forms']
    p = np.hstack([np.zeros((n, 1)), sobol_normal(d - 1, n, seed)])
    gx, gw = np.polynomial.hermite_e.hermegauss(32)
    bg = sum(wk / SQRT2PI * synthetic.background(spec, p + xk * np.eye(d)[0]) for xk, wk in zip(gx, gw))
    f0 = forms[0]
    a0, b0 = f0['idx']
    m0 = np.asarray(f0['m'])
    lam, mu = f0['lam'], m0[0]
    r2 = np.sum((p[:, a0 + 1:b0] - m0[1:]) ** 2, axis=1)

    def moments(lo, hi):
        lo, hi = np.minimum(lo, hi), hi
        m0_ = special.ndtr(hi) - special.ndtr(lo)
        plo, phi = np.exp(-0.5 * lo * lo) / SQRT2PI, np.exp(-0.5 * hi * hi) / SQRT2PI
        return m0_, plo - phi, m0_ + lo * plo - hi * phi

    def inside(gap):
        """Integral over |u - mu| < sqrt(gap / lam) of (gap - lam (u - mu)^2) phi(u); zero where gap <= 0."""
        rho = np.sqrt(np.maximum(gap, 0.0) / lam)
        q0, q1, q2 = moments(mu - rho, mu + rho)
        return np.where(gap > 0, gap * q0 - lam * (q2 - 2 * mu * q1 + mu * mu * q0), 0.0)

    if fam == 'H1':
        cconst = lam * r2 - synthetic.form_value(forms[1], p) - spec['c']
        full = lam * (1 + mu * mu) + cconst
        ns = spec['alpha'] * (full + inside(-cconst))
    elif fam == 'H2':
        floor = np.maximum(spec['c2'] - synthetic.form_value(forms[1], p), 0.0)
        ns = spec['alpha'] * (floor + inside(spec['c1'] - lam * r2 - floor))
    else:
        w = np.asarray(spec['w'])
        rho = np.sqrt(np.maximum(spec['c'] / lam - r2, 0.0))
        q0 = special.ndtr(mu + rho - w[0]) - special.ndtr(mu - rho - w[0])
        ns = spec['alpha'] * np.exp(p @ w - 0.5 * w @ w + 0.5 * w[0] ** 2) * q0
    return bg + ns


def _oracle_paths(spec: dict, n: int, seed: int) -> np.ndarray:
    fam = spec['family']
    if fam.startswith('rainbow') or fam == 'digital_asian':
        theta, a = paths.scaling_direction(spec)
        f = paths.scaled_value(spec, theta, a, paths.pca_complement(spec, theta))
    else:
        theta = paths.monotone_direction(spec)
        f = paths.monotone_value(spec, theta, paths.pca_complement(spec, theta))
    return f(sobol_normal(spec['d'] - 1, n, seed))


def _oracle_lognormal(spec: dict, n: int, seed: int) -> np.ndarray:
    pos, neg = options.lognormal_terms(spec)
    theta = options.directions(spec, pos, neg)[0][1]
    basis = options._ordered_complement(spec, theta, pos, neg)
    return options.preintegrated(spec, theta, basis, pos, neg)(sobol_normal(spec['d'] - 1, n, seed))


def oracle_split(f: Counted, d: int, seed: int, info: dict):
    spec = info['spec']
    if spec['family'] not in synthetic.FAMILIES:
        raise NotApplicable(spec['family'])
    from gauss import rotation
    q = rotation(d, spec['rotation_seed'])
    bg = Counted(lambda z: synthetic.background(spec, z @ q.T), d, f.budget)
    est, note = as_rqmc(bg, d, seed)
    f.used = bg.used
    return est + synthetic.reference(spec)['value'] - spec['beta0'], f'split oracle, background {note}'


def oracle_preint(f: Counted, d: int, seed: int, info: dict):
    spec, n = info['spec'], f.budget
    fam = spec['family']
    f.used = n
    if fam in synthetic.FAMILIES:
        return float(_oracle_synthetic(spec, n, seed).mean()), 'oracle, exact lines along y_0'
    if fam in paths.FAMILIES:
        return float(_oracle_paths(spec, n, seed).mean()), 'oracle, closed-form preintegration'
    return float(_oracle_lognormal(spec, n, seed).mean()), 'oracle, closed-form preintegration'


METHODS = {'plain_mc': plain_mc, 'plain_rqmc': plain_rqmc, 'as_rqmc': as_rqmc, 'as_preint': as_preint, 'strong_preint': strong_preint,
           'strong_rqmc': strong_rqmc, 'strong': strong, 'oracle_preint': oracle_preint, 'oracle_split': oracle_split}
