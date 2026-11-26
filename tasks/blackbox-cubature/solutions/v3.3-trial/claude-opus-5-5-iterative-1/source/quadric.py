"""Quadric surfaces: boundary-point fitting and exact Gaussian expectations of quadric indicators.

q(z) = z^T M z + m^T z + m0. For Z ~ N(0, I_d), Q = q(Z + mu) is a generalised chi-square variable; its CDF at 0
is computed by Gil-Pelaez (Imhof) inversion of the characteristic function. E[Z 1{q(Z) < 0}] is the gradient of
mu -> P(q(Z + mu) < 0) at mu = 0 (Gaussian integration by parts), computed by central differences.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate


def quad_features(Z):
    d = Z.shape[1]
    iu = np.triu_indices(d)
    return np.hstack([Z[:, iu[0]] * Z[:, iu[1]], Z, np.ones((len(Z), 1))])


def coef_to_quadric(coef, d):
    iu = np.triu_indices(d)
    nq = len(iu[0])
    M = np.zeros((d, d))
    M[iu] = coef[:nq]
    M = 0.5 * (M + M.T)          # off-diagonal coefficients multiply z_i z_j once
    return M, coef[nq:nq + d].copy(), float(coef[nq + d])


def fit_quadric(P):
    """Null vector of the feature matrix of boundary points P (n, d). Returns (M, m, m0, gap) where gap is the
    ratio of the smallest to the second smallest singular value (tiny for an exact quadric)."""
    d = P.shape[1]
    Phi = quad_features(P)
    nrm = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi = Phi / nrm
    _, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    coef = Vt[-1]
    M, m, m0 = coef_to_quadric(coef, d)
    gap = S[-1] / max(S[-2], 1e-300)
    s = math.sqrt((M * M).sum() + m @ m + m0 * m0)
    return M / s, m / s, m0 / s, gap


def q_eval(Z, M, m, m0):
    return np.einsum('ij,jk,ik->i', Z, M, Z) + Z @ m + m0


def _canon(M, m, m0, tol=1e-12):
    """Q = sum lam_i (x_i + delta_i)^2 + sigma * xi + c with x, xi iid N(0,1)."""
    lam, U = np.linalg.eigh(M)
    b = U.T @ m
    scale = max(np.abs(lam).max(), 1e-300)
    act = np.abs(lam) > tol * scale
    la = lam[act]
    de = b[act] / (2 * la)
    sigma = math.sqrt(float((b[~act] ** 2).sum()))
    c = m0 - float((b[act] ** 2 / (4 * la)).sum())
    return la, de, sigma, c


def prob_negative(M, m, m0):
    """P(q(Z) < 0), Z ~ N(0, I)."""
    la, de, sigma, c = _canon(M, m, m0)
    if len(la) == 0:
        from scipy.special import ndtr
        return float(ndtr(-c / sigma)) if sigma > 0 else float(c < 0)
    la2 = la * la
    d2 = de * de

    def integrand(u):
        if u == 0.0:
            # limit of Im(phi(u))/u = E[Q]
            return float((la * (1 + d2)).sum() + c)
        z = 1 - 2j * la * u
        logphi = 1j * u * c - 0.5 * sigma * sigma * u * u + np.sum(-0.5 * np.log(z) + 1j * la * d2 * u / z)
        return float(np.exp(logphi).imag / u)

    # upper limit where |phi| is negligible
    def absphi(u):
        return math.exp(-0.5 * sigma * sigma * u * u + float(np.sum(-0.25 * np.log1p(4 * la2 * u * u)
                                                                   - 2 * la2 * d2 * u * u / (1 + 4 * la2 * u * u))))
    U = 1.0 / max(np.abs(la).max(), 1e-300)
    while absphi(U) / U > 1e-16 and U < 1e12:
        U *= 2
    tot = 0.0
    a = 0.0
    seg = 1.0 / max(np.abs(la).max(), 1e-300)
    while a < U:
        b = min(U, a + seg)
        val, _ = integrate.quad(integrand, a, b, limit=200, epsabs=1e-15, epsrel=1e-12)
        tot += val
        a = b
        seg *= 1.5
    p = 0.5 - tot / math.pi
    return min(1.0, max(0.0, p))


def indicator_moments(M, m, m0, h=1e-4):
    """P(q(Z) < 0) and E[Z 1{q(Z) < 0}] (gradient of the shifted probability)."""
    d = len(m)
    p0 = prob_negative(M, m, m0)
    g = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        # q(z + e) = z^T M z + (m + 2 M e)^T z + q(e)
        pp = prob_negative(M, m + 2 * M @ e, float(e @ M @ e + m @ e + m0))
        pm = prob_negative(M, m - 2 * M @ e, float(e @ M @ e - m @ e + m0))
        g[i] = (pp - pm) / (2 * h)
    return p0, g


def indicator_moments_cond(M, m, m0, n=1 << 18, seed=0):
    """P(q(Z) < 0) and E[Z 1{q(Z) < 0}] by conditional RQMC: analytic in the most curved eigen-direction, scrambled
    Sobol' over the other active eigen-directions plus one Gaussian collecting the linear terms of the null space."""
    from scipy.special import ndtr, ndtri
    from scipy.stats import qmc
    d = len(m)
    lam, U = np.linalg.eigh(M)
    b = U.T @ m
    scale = max(np.abs(lam).max(), 1e-300)
    act = np.nonzero(np.abs(lam) > 1e-10 * scale)[0]
    null = np.setdiff1d(np.arange(d), act)
    istar = act[np.argmax(np.abs(lam[act]))]
    others = [i for i in act if i != istar]
    sig = math.sqrt(float((b[null] ** 2).sum())) if len(null) else 0.0
    dim = len(others) + (1 if sig > 0 else 0)
    if dim > 0:
        s = qmc.Sobol(dim, scramble=True, seed=seed)
        X = ndtri(np.clip(s.random(n), 1e-15, 1 - 1e-15))
    else:
        X = np.zeros((1, 0))
    no = len(others)
    Xo = X[:, :no]
    xi = X[:, no] if sig > 0 else np.zeros(len(X))
    C = m0 + sig * xi
    if no:
        C = C + (Xo * Xo) @ lam[others] + Xo @ b[others]
    C = np.asarray(C, float) * np.ones(len(X))
    ls, bs = lam[istar], b[istar]
    D = bs * bs - 4 * ls * C
    pos = D > 0
    sq = np.sqrt(np.where(pos, D, 0.0))
    lo = (-bs - sq) / (2 * abs(ls)) if ls > 0 else (bs - sq) / (2 * abs(ls))
    hi = (-bs + sq) / (2 * abs(ls)) if ls > 0 else (bs + sq) / (2 * abs(ls))
    if ls < 0:   # roots of ls x^2 + bs x + C = 0 are (-bs -+ sq)/(2 ls)
        r1 = (-bs - sq) / (2 * ls)
        r2 = (-bs + sq) / (2 * ls)
        lo, hi = np.minimum(r1, r2), np.maximum(r1, r2)
    pdf = lambda t: np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    if ls > 0:
        mass = np.where(pos, ndtr(hi) - ndtr(lo), 0.0)
        m1 = np.where(pos, pdf(lo) - pdf(hi), 0.0)
    else:
        mass = np.where(pos, 1.0 - (ndtr(hi) - ndtr(lo)), np.where(C < 0, 1.0, 0.0))
        m1 = np.where(pos, -(pdf(lo) - pdf(hi)), 0.0)
    p = float(mass.mean())
    Ex = np.zeros(d)
    Ex[istar] = float(m1.mean())
    for j, i in enumerate(others):
        Ex[i] = float((Xo[:, j] * mass).mean())
    if sig > 0:
        exi = float((xi * mass).mean())
        Ex[null] = b[null] / sig * exi
    return p, U @ Ex


def prob_negative_fast(M, m, m0, h=0.004):
    """P(q(Z) < 0) by Gil-Pelaez inversion in the variable t = log u (trapezoidal rule, vectorised)."""
    la, de, sigma, c = _canon(M, m, m0)
    if len(la) == 0:
        from scipy.special import ndtr
        return float(ndtr(-c / sigma)) if sigma > 0 else float(c < 0)
    smax = np.abs(la).max()
    smin = np.abs(la).min()
    t = np.arange(math.log(1e-9 / smax), math.log(1e6 / smin) + h, h)
    u = np.exp(t)
    z = 1 - 2j * np.outer(u, la)
    logphi = 1j * u * c - 0.5 * sigma * sigma * u * u + np.sum(-0.5 * np.log(z) + 1j * (la * de * de) * u[:, None] / z,
                                                               axis=1)
    val = np.exp(logphi).imag          # integrand of int Im(phi(u))/u du after du = u dt
    p = 0.5 - h * val.sum() / math.pi
    return min(1.0, max(0.0, float(p)))


def indicator_moments_fast(M, m, m0, step=1e-4):
    d = len(m)
    p0 = prob_negative_fast(M, m, m0)
    g = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = step
        pp = prob_negative_fast(M, m + 2 * M @ e, float(e @ M @ e + m @ e + m0))
        pm = prob_negative_fast(M, m - 2 * M @ e, float(e @ M @ e - m @ e + m0))
        g[i] = (pp - pm) / (2 * step)
    return p0, g


def _canon_all(M, m, m0):
    la, de, sigma, c = _canon(M, m, m0)
    return la, de, sigma, c


def abs_mean_fast(M, m, m0, h=0.004):
    """E|q(Z)| = (2/pi) int_0^inf (1 - Re phi(u)) / u^2 du, trapezoidal rule in t = log u (tail beyond the last node
    integrated with Re phi = 0)."""
    la, de, sigma, c = _canon(M, m, m0)
    if len(la) == 0:
        # Q = c + sigma xi
        from scipy.special import ndtr
        if sigma == 0:
            return abs(c)
        a = c / sigma
        return float(sigma * 2 * math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) + c * (1 - 2 * ndtr(-a)))
    smax = np.abs(la).max()
    smin = np.abs(la).min()
    t = np.arange(math.log(1e-7 / max(smax, 1e-300)), math.log(1e7 / max(smin, 1e-300)) + h, h)
    u = np.exp(t)
    z = 1 - 2j * np.outer(u, la)
    logphi = 1j * u * c - 0.5 * sigma * sigma * u * u + np.sum(-0.5 * np.log(z) + 1j * (la * de * de) * u[:, None] / z,
                                                               axis=1)
    val = (1 - np.exp(logphi).real) / u      # (1 - Re phi) / u^2 * du, du = u dt
    tail = 1.0 / u[-1]                       # int_{U}^inf du / u^2
    return float(2 / math.pi * (h * val.sum() + tail))


def pos_part_moments(M, m, m0, step=1e-4):
    """E[max(q(Z), 0)] and E[Z max(q(Z), 0)] (central differences in the shift)."""
    d = len(m)

    def epos(MM, mm, mm0):
        EQ = float(np.trace(MM)) + mm0
        return 0.5 * (EQ + abs_mean_fast(MM, mm, mm0))
    e0 = epos(M, m, m0)
    g = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = step
        pp = epos(M, m + 2 * M @ e, float(e @ M @ e + m @ e + m0))
        pm = epos(M, m - 2 * M @ e, float(e @ M @ e - m @ e + m0))
        g[i] = (pp - pm) / (2 * step)
    return e0, g
