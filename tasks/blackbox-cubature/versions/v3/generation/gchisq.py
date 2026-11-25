"""Generalized chi-square Q = sum_i lam_i (Z_i + delta_i)^2, lam_i > 0, for the anisotropic synthetic families of
blackbox-cubature v3.1 (DUAL_PLAN section 9.9).

Ruben (1962) expands the law as a mixture of scaled central chi-square laws,
    P(Q <= x) = sum_k a_k P(chi2_{n+2k} <= x / beta),
    a_0 = exp(-sum delta^2 / 2) prod (beta / lam)^{1/2},  a_k = (1/k) sum_{r<k} g_{k-r} a_r,
    g_k = (1/2) sum (1 - beta/lam)^k + (k beta / 2) sum (delta^2 / lam) (1 - beta/lam)^{k-1},
so partial expectations and the density follow term by term. beta = min lam keeps every a_k >= 0 (the missing mass
bounds the truncation error, here 2e-15); beta = 0.9 min lam gives a second expansion with other coefficients and
truncation (Ruben's faster beta = 2 lam_min lam_max / (lam_min + lam_max) has signed coefficients and loses about
three digits to cancellation). The two expansions are the two reference routes; Imhof's inversion of the
characteristic function (Gil-Pelaez) is the independent pointwise check.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special


class Ruben:
    """Ruben's mixture for one form; mode 'min' (beta = min lam) or 'scaled' (beta = 0.9 min lam)."""

    def __init__(self, lam, delta2, mode: str = 'min', tol: float = 2e-15, kmax: int = 20000):
        lam, d2 = np.asarray(lam, float), np.asarray(delta2, float)
        self.n, self.lam, self.d2 = len(lam), lam, d2
        lo, hi = lam.min(), lam.max()
        self.beta = float({'min': lo, 'scaled': 0.9 * lo}[mode])
        r = 1.0 - self.beta / lam
        a = [math.exp(-0.5 * d2.sum() + 0.5 * np.log(self.beta / lam).sum())]
        g, rk = [], np.ones(self.n)
        for k in range(1, kmax):
            g.append(0.5 * float((rk * r).sum()) + 0.5 * k * self.beta * float(((d2 / lam) * rk).sum()))
            rk = rk * r
            a.append(float(np.dot(g[::-1], a)) / k)
            # the missing mass bounds the truncation; once the terms are negligible what remains is rounding
            if 1.0 - math.fsum(a) < tol or (k > 10 and a[-1] < 1e-20 and a[-1] <= a[-2]):
                break
        else:
            raise RuntimeError('Ruben series did not converge')
        self.a = np.asarray(a)
        self.nu = self.n + 2 * np.arange(len(self.a))
        self.mean = float((lam * (1.0 + d2)).sum())
        self.var = float((2 * lam ** 2 * (1.0 + 2 * d2)).sum())

    def cdf(self, x: float) -> float:
        return float(self.a @ special.gammainc(0.5 * self.nu, 0.5 * x / self.beta)) if x > 0 else 0.0

    def sf(self, x: float) -> float:
        return float(self.a @ special.gammaincc(0.5 * self.nu, 0.5 * x / self.beta)) if x > 0 else 1.0

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        t, h = 0.5 * x / self.beta, 0.5 * self.nu
        return float(self.a @ np.exp((h - 1) * math.log(t) - t - special.gammaln(h))) * 0.5 / self.beta

    def excess(self, s: float) -> float:
        """E[(Q - s)^+]."""
        if s <= 0:
            return self.mean - s
        t = 0.5 * s / self.beta
        return float(self.a @ (self.beta * self.nu * special.gammaincc(0.5 * self.nu + 1, t)
                               - s * special.gammaincc(0.5 * self.nu, t)))

    def shortfall(self, b: float) -> float:
        """E[(b - Q)^+]."""
        if b <= 0:
            return 0.0
        t = 0.5 * b / self.beta
        return float(self.a @ (b * special.gammainc(0.5 * self.nu, t)
                               - self.beta * self.nu * special.gammainc(0.5 * self.nu + 1, t)))


def _phase(lam, d2, u):
    lu = np.multiply.outer(u, lam)
    return 0.5 * (np.arctan(lu) + d2 * lu / (1 + lu * lu)).sum(axis=-1)


def _rho(lam, d2, u):
    lu2 = np.multiply.outer(u, lam) ** 2
    return np.exp(0.25 * np.log1p(lu2).sum(axis=-1) + 0.5 * (d2 * lu2 / (1 + lu2)).sum(axis=-1))


def imhof_sf(lam, delta2, x: float) -> float:
    """P(Q > x) by Imhof's formula; [0, U] by adaptive quadrature, the tail by QUADPACK's Fourier integral."""
    lam, d2 = np.asarray(lam, float), np.asarray(delta2, float)
    w, U = 0.5 * x, 60.0 / lam.min()
    head = integrate.quad(lambda u: math.sin(_phase(lam, d2, u) - w * u) / (u * _rho(lam, d2, u)) if u > 0 else
                          float((0.5 * lam * (1 + d2)).sum() - w), 0.0, U, epsabs=1e-16, epsrel=1e-13, limit=4000)[0]
    amp_s = lambda u: math.sin(_phase(lam, d2, u)) / (u * _rho(lam, d2, u))
    amp_c = lambda u: math.cos(_phase(lam, d2, u)) / (u * _rho(lam, d2, u))
    return 0.5 + (head + _tail(amp_s, amp_c, w, U, -1.0)) / math.pi


def imhof_pdf(lam, delta2, x: float) -> float:
    """Density by the Gil-Pelaez inversion (1 / 2 pi) int_0^inf cos(theta(u)) / rho(u) du."""
    lam, d2 = np.asarray(lam, float), np.asarray(delta2, float)
    w, U = 0.5 * x, 60.0 / lam.min()
    head = integrate.quad(lambda u: math.cos(_phase(lam, d2, u) - w * u) / _rho(lam, d2, u), 0.0, U,
                          epsabs=1e-16, epsrel=1e-13, limit=4000)[0]
    amp_c = lambda u: math.cos(_phase(lam, d2, u)) / _rho(lam, d2, u)
    amp_s = lambda u: math.sin(_phase(lam, d2, u)) / _rho(lam, d2, u)
    return (head + _tail(amp_c, amp_s, w, U, 1.0)) / (2 * math.pi)


def _tail(amp_cos, amp_sin, w: float, U: float, sign: float) -> float:
    """int_U^inf amp_cos(u) cos(w u) + sign amp_sin(u) sin(w u) du: QUADPACK's Fourier integral when the tail holds
    many periods, plain adaptive quadrature of the infinite range otherwise (QAWF is unreliable as w -> 0)."""
    if w * U > 50.0:
        return (integrate.quad(amp_cos, U, np.inf, weight='cos', wvar=w, epsabs=1e-16, limlst=500)[0]
                + sign * integrate.quad(amp_sin, U, np.inf, weight='sin', wvar=w, epsabs=1e-16, limlst=500)[0])
    return integrate.quad(lambda u: amp_cos(u) * math.cos(w * u) + sign * amp_sin(u) * math.sin(w * u), U, np.inf,
                          epsabs=1e-16, epsrel=1e-13, limit=4000)[0]


def direct2(lam, delta2, x: float) -> tuple[float, float]:
    """(P(Q > x), density at x) for two terms by conditioning on the first: with z = -delta_1 + sqrt(x / lam_1) sin t,
    the second term's law gives closed forms and the square-root endpoint behaviour cancels. Independent of the
    characteristic function and of Ruben's mixture."""
    (l1, l2), (e1, e2) = np.asarray(lam, float), np.sqrt(np.asarray(delta2, float))
    if x <= 0:
        return 1.0, 0.0
    a = math.sqrt(x / l1)

    def parts(t):
        z = -e1 + a * math.sin(t)
        q = math.sqrt(x / l2) * abs(math.cos(t))           # sqrt(r / lam_2), r = x - lam_1 (z + e1)^2
        phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        inner = special.ndtr(q - e2) - special.ndtr(-q - e2)
        dens = (math.exp(-0.5 * (q - e2) ** 2) + math.exp(-0.5 * (q + e2) ** 2)) / math.sqrt(2 * math.pi)
        return phi * inner * a * math.cos(t), phi * dens * a / (2 * math.sqrt(l2 * x))

    cdf = integrate.quad(lambda t: parts(t)[0], -math.pi / 2, math.pi / 2, epsabs=1e-16, epsrel=1e-14, limit=400)[0]
    pdf = integrate.quad(lambda t: parts(t)[1], -math.pi / 2, math.pi / 2, epsabs=1e-16, epsrel=1e-14, limit=400)[0]
    return 1.0 - cdf, pdf
