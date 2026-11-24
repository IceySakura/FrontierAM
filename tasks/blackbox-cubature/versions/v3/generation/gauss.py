"""Gaussian helpers for the v3 generators: truncated moments, bivariate normal CDF, rotations, RQMC.

All integrals are against the standard normal density phi. Tail differences are taken on the side where they do not
cancel, so the truncated moments stay accurate to a few ulp also far in the tails.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats

SQRT2PI = math.sqrt(2.0 * math.pi)


def pdf(x):
    return np.exp(-0.5 * np.square(x)) / SQRT2PI


def cdf(x):
    return special.ndtr(x)


def mass(lo: float, hi: float) -> float:
    """Phi(hi) - Phi(lo) without cancellation."""
    if lo >= hi:
        return 0.0
    if lo >= 0.0:
        return float(special.ndtr(-lo) - special.ndtr(-hi))
    if hi <= 0.0:
        return float(special.ndtr(hi) - special.ndtr(lo))
    return float(1.0 - special.ndtr(lo) - special.ndtr(-hi))


def _x_pdf(x: float) -> float:
    return 0.0 if math.isinf(x) else x * float(pdf(x))


def truncated_moments(lo: float, hi: float) -> tuple[float, float, float]:
    """(M0, M1, M2) with Mk = integral over [lo, hi] of y^k phi(y) dy; lo, hi may be infinite."""
    if lo >= hi:
        return 0.0, 0.0, 0.0
    m0 = mass(lo, hi)
    plo = 0.0 if math.isinf(lo) else float(pdf(lo))
    phi = 0.0 if math.isinf(hi) else float(pdf(hi))
    m1 = plo - phi
    m2 = m0 + _x_pdf(lo) - _x_pdf(hi)
    return m0, m1, m2


def poly2_gauss(coef: tuple[float, float, float], lo: float, hi: float) -> float:
    """Integral over [lo, hi] of (c0 + c1 y + c2 y^2) phi(y) dy."""
    m0, m1, m2 = truncated_moments(lo, hi)
    return coef[0] * m0 + coef[1] * m1 + coef[2] * m2


def quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of a y^2 + b y + c (a may be 0), sorted, with the cancellation-free formula."""
    if a == 0.0:
        return [] if b == 0.0 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    s = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(s, b)) if b != 0.0 else -0.5 * s
    roots = [q / a, c / q] if q != 0.0 else [0.0, 0.0]
    return sorted(roots)


def bvn_cdf(h: float, k: float, rho: float) -> float:
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho, via Owen's T function."""
    if abs(rho) < 1e-15:
        return float(special.ndtr(h) * special.ndtr(k))
    eps = 1e-13
    h = h if abs(h) > eps else math.copysign(eps, h if h != 0 else 1.0)
    k = k if abs(k) > eps else math.copysign(eps, k if k != 0 else 1.0)
    s = math.sqrt(max(1.0 - rho * rho, 0.0))
    if s == 0.0:
        return float(special.ndtr(min(h, k))) if rho > 0 else max(0.0, float(special.ndtr(h) - special.ndtr(-k)))
    ah = (k - rho * h) / (h * s)
    ak = (h - rho * k) / (k * s)
    beta = 0.0 if h * k > 0 else 0.5
    return float(0.5 * special.ndtr(h) + 0.5 * special.ndtr(k)
                 - special.owens_t(h, ah) - special.owens_t(k, ak) - beta)


def bvn_cdf_quad(h: float, k: float, rho: float) -> float:
    """Independent check of bvn_cdf: one-dimensional integral of phi(x) Phi((k - rho x)/sqrt(1 - rho^2))."""
    s = math.sqrt(1.0 - rho * rho)
    val, _ = integrate.quad(lambda x: float(pdf(x) * special.ndtr((k - rho * x) / s)), -np.inf, h,
                            epsabs=1e-15, epsrel=1e-13, limit=400)
    return val


def rotation(d: int, seed: int) -> np.ndarray:
    """Haar-distributed orthogonal d x d matrix, unique for a seed (QR with positive diagonal of R)."""
    rng = np.random.default_rng([int(seed), 20261124])
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def random_unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def rqmc_normal(d: int, m: int, seed: int) -> np.ndarray:
    """2^m scrambled Sobol' points mapped to N(0, I_d)."""
    u = stats.qmc.Sobol(d, scramble=True, rng=np.random.default_rng(int(seed))).random_base2(m)
    return special.ndtri(np.clip(u, 1e-16, 1 - 1e-16))


def rqmc_estimate(f, d: int, m: int, replicates: int, seed: int, chunk: int = 1 << 16) -> tuple[float, float]:
    """Mean of `replicates` independent scrambled-Sobol' estimates of E f(Z) and its standard error."""
    estimates = []
    for rep in range(replicates):
        sampler = stats.qmc.Sobol(d, scramble=True, rng=np.random.default_rng([int(seed), rep]))
        total, count = 0.0, 0
        remaining = 1 << m
        while remaining > 0:
            take = min(chunk, remaining)
            u = sampler.random(take)
            z = special.ndtri(np.clip(u, 1e-16, 1 - 1e-16))
            total += float(np.sum(f(z)))
            count += take
            remaining -= take
        estimates.append(total / count)
    est = np.asarray(estimates)
    return float(est.mean()), float(est.std(ddof=1) / math.sqrt(len(est)))
