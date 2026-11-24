"""Black-box baselines and the structure-aware oracle for blackbox-cubature v3 (milestone M2).

Every black-box method sees only a query-counting oracle f (points in R^d as rows) with a budget; probes, finite
differences and line searches all count. Methods return (estimate, note).

- plain_rqmc: scrambled Sobol' points mapped to N(0, I), the whole budget (zero anchor).
- as_rqmc: forward-difference gradients give the active-subspace basis (eigenvectors of the gradient covariance, Liu
  and Owen 2023); the main phase is one scrambled Sobol' net in that basis, leading eigenvector first, of the largest
  power-of-two size that fits after a pilot of PILOT_FRACTION of the budget (scrambled nets are balanced only in
  blocks of 2^m); the queries left between pilot and net buy more gradients.
- as_preint: the same pilot, then preintegration along the leading eigenvector with numerical smoothing, on the largest
  power-of-two number of lines the budget affords: each line is
  probed at Gauss-Hermite nodes; where the probes switch between zero and non-zero values the boundary is located by
  bisection and every non-zero piece is integrated by the Gauss rule of the normal density restricted to that piece
  (zero-boundary detection standing in for the root finding of Bayer, Ben Hammouda and Tempone 2023); lines without a
  boundary keep the Gauss-Hermite value.
- strong: the frozen reference, chosen on the prototype set: plain_rqmc for d <= 3 (rotation gains nothing there and
  halving the net loses), as_rqmc otherwise. as_preint stays as a documented variant: each line costs about 28
  queries, so it affords about 1/32 of the oracle's lines and loses to as_rqmc on most families.
- preint_gh (diagnostic, black-box, d <= 4 only): as_preint's lines on a tensor Gauss-Hermite grid of the complement
  instead of Sobol' points, as many nodes per dimension as the budget affords; it shows the accuracy a deterministic
  outer rule reaches in low dimension when every line has a single smooth boundary.
- oracle_preint (diagnostic, reads the instance's canonical structure): preintegration along the structural direction
  with the line integral in closed form, one query per line, scrambled Sobol' over the complement ordered by the known
  structure. It stands for the accuracy reachable when directions and kinks cost nothing.
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
import synthetic  # noqa: E402

SQRT2PI = math.sqrt(2.0 * math.pi)
FD_STEP = 0.05
PILOT_FRACTION = 1 / 8
PILOT_GRADIENTS = 64
PROBES = 8
BISECT = 14
PIECE_NODES = 6
U_MAX = 8.5
CHUNK = 128
LOW_D = 3
GH_MAX_NODES = 120
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


def sobol_normal(d: int, n: int, seed: int) -> np.ndarray:
    """First n points of a scrambled Sobol' sequence mapped to N(0, I_d); n need not be a power of two."""
    sampler = stats.qmc.Sobol(d, scramble=True, rng=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = sampler.random(n)
    return special.ndtri(np.clip(u, 1e-16, 1 - 1e-16))


# ---------------------------------------------------------------- black-box methods

def plain_rqmc(f: Counted, d: int, seed: int, info=None):
    return float(f(sobol_normal(d, f.remaining, seed)).mean()), 'rqmc'


def pow2(n: int) -> int:
    """Largest power of two <= n (n >= 1): scrambled Sobol' points are balanced only in blocks of 2^m."""
    return 1 << (int(n).bit_length() - 1)


def gradients(f: Counted, d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward-difference gradients (step FD_STEP) at n scrambled Sobol' points, and f at those points."""
    z = sobol_normal(d, n, seed)
    vals = f(np.concatenate([z] + [z + FD_STEP * e for e in np.eye(d)])).reshape(d + 1, n)
    return (vals[1:] - vals[0]).T / FD_STEP, vals[0]


def as_basis(grads: np.ndarray) -> np.ndarray:
    """Eigenvectors of the gradient covariance, largest eigenvalue first (active-subspace basis)."""
    eig, vec = np.linalg.eigh(grads.T @ grads / len(grads))
    return vec[:, np.argsort(eig)[::-1]]


def _pilot_size(f: Counted, d: int) -> int:
    return max(1, min(PILOT_GRADIENTS, int(f.budget * PILOT_FRACTION) // (d + 1)))


def _rqmc_in_basis(f: Counted, d: int, seed: int, grads: np.ndarray):
    """Main phase of as_rqmc: one scrambled Sobol' net of the largest power-of-two size that fits; the queries left over
    before it buy more gradients."""
    main = pow2(f.remaining)
    extra = (f.remaining - main) // (d + 1)
    if extra:
        grads = np.vstack([grads, gradients(f, d, extra, seed + 2)[0]])
    est = float(f(sobol_normal(d, main, seed) @ as_basis(grads).T).mean())
    return est, f'rqmc, {main} points, {len(grads)} gradients'


def as_rqmc(f: Counted, d: int, seed: int, info=None):
    return _rqmc_in_basis(f, d, seed, gradients(f, d, _pilot_size(f, d), seed + 1)[0])


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


def line_integrals(f: Counted, theta: np.ndarray, base: np.ndarray) -> np.ndarray:
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
    lo, hi, lo_zero = x[ki], x[ki + 1], zero[li, ki]
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        same = (f(base[li] + mid[:, None] * theta) == 0.0) == lo_zero
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    bnd = 0.5 * (lo + hi)
    seg_line, seg_a, seg_b = [], [], []
    for i in np.nonzero(refine)[0]:
        cuts = np.concatenate([[-U_MAX], bnd[li == i], [U_MAX]])
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


def _preint(f: Counted, d: int, seed: int, basis: np.ndarray):
    """Lines along basis[:, 0] at scrambled Sobol' points of the complement; the line count is the largest power of two
    affordable at the cost per line measured on the first chunk."""
    theta, comp = basis[:, 0], basis[:, 1:]
    reserve = PROBES + 2 * (BISECT + PIECE_NODES)
    cand = sobol_normal(d - 1, max(1, f.remaining // PROBES), seed) @ comp.T
    start = f.used
    k = min(CHUNK, len(cand), max(1, f.remaining // reserve))
    vals = list(line_integrals(f, theta, cand[:k]))
    target = min(len(cand), pow2(max(1, int((f.budget - start) / (1.1 * (f.used - start) / k)))))
    while len(vals) < target:
        k = min(CHUNK, target - len(vals), f.remaining // reserve)
        if k <= 0:
            break
        vals += list(line_integrals(f, theta, cand[len(vals):len(vals) + k]))
    n = pow2(len(vals))
    return float(np.mean(vals[:n])), f'preint, {n} lines'


def as_preint(f: Counted, d: int, seed: int, info=None):
    grads, _ = gradients(f, d, _pilot_size(f, d), seed + 1)
    est, note = _preint(f, d, seed, as_basis(grads))
    return est, f'{note}, {len(grads)} gradients'


def strong(f: Counted, d: int, seed: int, info=None):
    return plain_rqmc(f, d, seed) if d <= LOW_D else as_rqmc(f, d, seed)


def preint_gh(f: Counted, d: int, seed: int, info=None):
    if not 2 <= d <= 4:
        raise NotApplicable(f'd = {d}')
    grads, _ = gradients(f, d, _pilot_size(f, d), seed + 1)
    basis = as_basis(grads)
    lines = f.remaining / (1.25 * (PROBES + BISECT + PIECE_NODES))
    n = min(GH_MAX_NODES, int(lines ** (1.0 / (d - 1))))
    x, w = np.polynomial.hermite_e.hermegauss(n)
    v = np.stack([g.ravel() for g in np.meshgrid(*([x] * (d - 1)), indexing='ij')], axis=1)
    weights = np.ones(1)
    for _ in range(d - 1):
        weights = np.outer(weights, w / SQRT2PI).ravel()
    vals = line_integrals(f, basis[:, 0], v @ basis[:, 1:].T)
    return float(vals @ weights), f'preint, Gauss-Hermite {n}^{d - 1} lines, {len(grads)} gradients'


# ---------------------------------------------------------------- structure-aware oracle (diagnostic)

def _orthonormal(vectors: list[np.ndarray], k: int) -> np.ndarray:
    out = []
    for v in vectors:
        r = v - sum((q @ v) * q for q in out)
        if np.linalg.norm(r) > 1e-10:
            out.append(r / np.linalg.norm(r))
        if len(out) == k:
            break
    return np.column_stack(out)


def _oracle_synthetic(spec: dict, n: int, seed: int) -> np.ndarray:
    """Preintegrate along y1: exact inner integral of the non-smooth part, closed form for the background ridges;
    Sobol' coordinates along y2, then the background directions, then the rest."""
    d = spec['d']
    w, v = np.asarray(spec['w']), np.asarray(spec['v'])
    eye = np.eye(d)
    wp, vp = w.copy(), v.copy()
    wp[:2] = vp[:2] = 0.0
    basis = _orthonormal([eye[1], wp, vp] + [eye[k] for k in range(2, d)], d - 1)
    y = sobol_normal(d - 1, n, seed) @ basis.T
    bg = (spec['beta0'] + spec['beta1'] * (np.exp(y @ w + 0.5 * w[0] ** 2 - 0.5 * w @ w) - 1.0)
          + spec['beta2'] * (math.exp(-0.5 * v[0] ** 2) * np.cos(y @ v + spec['theta'])
                             - math.exp(-0.5 * v @ v) * math.cos(spec['theta'])))
    return bg + np.array([synthetic.inner_exact(spec, t) for t in y[:, 1]])


def _oracle_conditional(spec: dict, n: int, seed: int) -> np.ndarray:
    """Condition on the last Gaussian of the Cholesky factorisation (closed-form lognormal call, put or probability)."""
    order, base, load = options._conditional_setup(spec, list(range(spec['L'])))
    L = len(order)
    v = sobol_normal(L - 1, n, seed)
    known = np.exp(base[:L - 1] + v @ load[:L - 1, :L - 1].T)
    mu_last = base[L - 1] + v @ load[L - 1, :L - 1]
    vals = [options._inner(spec, order, known[i], mu_last[i], load[L - 1, L - 1]) for i in range(n)]
    return math.exp(-spec['r'] * spec['T']) * np.array(vals)


def _oracle_lognormal(spec: dict, n: int, seed: int) -> np.ndarray:
    pos, neg = options.lognormal_terms(spec)
    theta = options.directions(spec, pos, neg)[0][1]
    basis = options._ordered_complement(spec, theta, pos, neg)
    return options.preintegrated(spec, theta, basis, pos, neg)(sobol_normal(spec['d'] - 1, n, seed))


def oracle_preint(f: Counted, d: int, seed: int, info: dict):
    spec, n = info['spec'], f.budget
    fam = spec['family']
    f.used = n
    if fam in synthetic.FAMILIES:
        return float(_oracle_synthetic(spec, n, seed).mean()), 'oracle, exact y1 line'
    if fam == 'geo_basket_call':
        return options.geo_basket_closed(spec), 'oracle, single direction (exact)'
    if fam in LOGNORMAL:
        return float(_oracle_lognormal(spec, n, seed).mean()), 'oracle, closed-form preintegration'
    return float(_oracle_conditional(spec, n, seed).mean()), 'oracle, Cholesky conditioning'


METHODS = {'plain_rqmc': plain_rqmc, 'as_rqmc': as_rqmc, 'as_preint': as_preint, 'strong': strong,
           'preint_gh': preint_gh, 'oracle_preint': oracle_preint}
