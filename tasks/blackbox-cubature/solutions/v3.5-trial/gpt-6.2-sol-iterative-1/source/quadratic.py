"""Gaussian probabilities and first moments of quadratic forms."""

import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import eigh
from scipy.special import ndtr


def quad_values(x, A, b, c):
    return np.einsum("ij,ij->i", x @ A, x) + x @ b + c


def diagonal(A, b):
    lam, V = eigh(A, check_finite=False)
    ix = np.argsort(-np.abs(lam))
    return lam[ix], (V.T @ b)[ix], V[:, ix]


def ellipse2(lam, b, c):
    shifts = b / (2 * lam)
    rr = -c + np.dot(shifts * shifts, lam)
    if rr <= 0:
        return 0.
    r = np.sqrt(rr/lam)
    lb = max(-11., -r[0]-shifts[0])
    ub = min(11., r[0]-shifts[0])
    if lb >= ub:
        return 0.

    def pdf(x):
        width = r[1]*np.sqrt(max(0., 1-((x+shifts[0])/r[0])**2))
        u, l = width-shifts[1], -width-shifts[1]
        mass = ndtr(-l)-ndtr(-u) if l >= 0 else ndtr(u)-ndtr(l)
        return np.exp(-.5*x*x)*mass

    return quad(pdf, lb, ub, epsabs=2e-12, epsrel=1e-10, limit=100)[0]/math.sqrt(2*math.pi)


def positive_probability_diag(lam, b, c, accuracy=5e-10):
    if not len(lam):
        return float(c > 0)
    scale = np.sqrt(b @ b + 2*(lam @ lam))
    if not np.isfinite(scale) or scale < 1e-15:
        return float(c+lam.sum() > 0)
    lam, b, c = lam/scale, b/scale, c/scale
    live = (abs(lam) > 2e-12) | (abs(b) > 2e-12)
    lam, b = lam[live], b[live]
    if len(lam) == 2 and np.min(lam) > 0:
        return 1-ellipse2(lam, b, c)
    if len(lam) == 2 and np.max(lam) < 0:
        return ellipse2(-lam, -b, -c)
    if len(lam) == 1:
        a, z = lam[0], b[0]
        if abs(a) < 1e-12:
            return ndtr(c/abs(z))
        D = z*z-4*a*c
        if D <= 0:
            return float(a > 0)
        u, v = sorted(((-z-np.sqrt(D))/(2*a), (-z+np.sqrt(D))/(2*a)))
        m = ndtr(-u)-ndtr(-v) if u >= 0 else ndtr(v)-ndtr(u)
        return 1-m if a > 0 else m
    ll, bb = -2j*lam, -.5*b*b

    def logcf(t):
        a = 1 + t*ll
        return 1j*t*c + np.sum(bb*(t*t)/a - .5*np.log(a))

    def integrand(t):
        if t < 1e-10:
            return c + lam.sum()
        return np.imag(np.exp(logcf(t)))/t

    # Use the asymptotic carrier frequency for slowly decaying transforms.
    T = 16.
    for _ in range(8):
        if logcf(T).real < np.log(accuracy * .16):
            break
        T *= 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val = quad(integrand, 0, min(16., T), epsabs=accuracy, limit=160)[0]
        t = 16.
        while t < T:
            val += quad(integrand, t, 2*t, epsabs=accuracy, limit=140)[0]
            t *= 2
    return np.clip(.5+val/math.pi, 0, 1)


def positive_probability(A, b, c):
    lam, b, _ = diagonal(A, b)
    return positive_probability_diag(lam,b,c)


def positive_part(A, b, c):
    lam, b, _ = diagonal(A, b)
    scale = np.sqrt(b @ b + 2*(lam @ lam))
    if scale < 1e-15:
        return max(0., c+lam.sum())
    lam, b, c = lam/scale, b/scale, c/scale
    ll, bb = -2j*lam, -.5*b*b

    def logcf(t):
        u = 1 + t*ll
        return 1j*t*c + np.sum(bb*(t*t)/u - .5*np.log(u))

    def small(t):
        if t < 1e-7:
            return .5*(1+(c+lam.sum())**2)
        return -np.real(np.expm1(logcf(t)))/(t*t)

    def tail(t):
        return -np.real(np.exp(logcf(t)))/(t*t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val = quad(small, 0, 12, epsabs=2e-9, limit=100)[0]+1/12
        t = 12.
        for _ in range(8):
            if logcf(t).real-np.log(t) < -22.:
                break
            val += quad(tail, t, 2*t, epsabs=1e-10, limit=85)[0]
            t *= 2
    return scale * max(0., .5*(c+lam.sum())+val/math.pi)


def tilted_normal(A, b, c, max_shift=1.5):
    """Exponential tilt towards a quadratic sublevel boundary."""
    from scipy.optimize import brentq
    lam, bl, V = diagonal(A, b)
    nn = np.sqrt(2*lam@lam+bl@bl)
    if not np.isfinite(nn) or nn < 1e-12:
        return np.zeros(len(b)), np.ones(len(b)), V

    def mean(t):
        sd2 = 1/(1+2*t*lam)
        mu = -t*bl*sd2
        return lam@(sd2+mu*mu)+bl@mu+c

    t = 0.
    if mean(0) > 0:
        hi = min(14./nn, -.498/min(-1e-18,lam[-1],min(lam)))
        if mean(hi) < 0:
            t = brentq(mean, 0., hi, xtol=1e-9)
        else:
            t = hi
    sd2 = 1/(1+2*t*lam)
    mu = -t*bl*sd2
    if np.linalg.norm(mu) > max_shift:
        adj = max_shift/np.linalg.norm(mu)
        mu *= adj
        sd2 = (1-adj)+adj*sd2
    return mu, np.sqrt(np.clip(sd2, .08, 3.)), V
