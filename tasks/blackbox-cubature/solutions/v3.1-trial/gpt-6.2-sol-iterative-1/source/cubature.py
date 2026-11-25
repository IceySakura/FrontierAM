"""Gaussian nets, rotations, and small analytic controls."""

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
import warnings


def uniform(rng, power, d):
    sampler = qmc.Sobol(d, scramble=True, rng=rng, bits=40)
    return np.clip(sampler.random_base2(power), 0.5**41, 1 - 0.5**41)


def normals(rng, power, d):
    return ndtri(uniform(rng, power, d))


def net_sizes(n):
    while n:
        p = n.bit_length() - 1
        yield p
        n -= 1 << p


def norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-50)


def rotation(v):
    """Orthogonal matrix with v, a unit vector, in its first column."""
    w = -np.copy(v)
    w[0] += 1
    n = w @ w
    return np.eye(len(v)) - (2 / max(n, 1e-30)) * np.outer(w, w)


def ordered_basis(B, g=None):
    if g is None or np.linalg.norm(g) < 1e-20:
        return np.linalg.eigh(B)[1][:, ::-1]
    U = rotation(g / np.linalg.norm(g))
    _, V = np.linalg.eigh(U[:, 1:].T @ B @ U[:, 1:])
    U[:, 1:] = U[:, 1:] @ V[:, ::-1]
    return U


class Quadratic:
    def __init__(self, A, b, c):
        self.A, self.b, self.c = A, b, float(c)

    @classmethod
    def centered(cls, A, g, p, c=0.0):
        a = A @ p
        return cls(A, g - 2*a, c + p @ a - g @ p)

    def __call__(self, z):
        return np.sum((z @ self.A) * z, axis=-1) + z @ self.b + self.c

    def scaled(self, value):
        return Quadratic(value*self.A, value*self.b, value*self.c)

    def shifted(self, v):
        return Quadratic(self.A, self.b+2*self.A@v, self(v))

    def spectrum(self):
        lam, V = np.linalg.eigh(self.A)
        return lam, V.T @ self.b, self.c, V

    def mean(self):
        return self.c + np.trace(self.A)


def quadratic_positive_diag(lam, b, c, cap=4096.0, tol=4e-9):
    """E max(q, 0) in a diagonal representation."""
    mu = c + lam.sum()
    sd2 = 2*(lam @ lam) + b @ b
    if sd2 < 1e-24:
        return max(mu, 0)
    scale = max(np.sqrt(sd2), abs(mu))
    lam, b, c, mu = lam/scale, b/scale, c/scale, mu/scale
    bb = -0.5*b*b
    l2 = 2j*lam

    def integrand(t):
        if t < 2e-7:
            return 0.5*(mu*mu + sd2/scale**2)
        a = 1 - t*l2
        f = (1j*c)*t - 0.5*np.log(a).sum() + t*t*np.sum(bb/a)
        return -np.expm1(f).real / (t*t)

    # The unmodelled part of the tail here is only the CF term.
    cut = [0.,8.,64.,512.,cap] if cap>512 else [0.,8.,64.,cap]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        v = sum(quad(integrand, a, b, epsabs=tol, epsrel=5e-8, limit=140)[0]
                for a,b in zip(cut[:-1],cut[1:]))
    return scale*(0.5*mu + (v + 1/cap)/np.pi)


def quadratic_positive(control):
    """E max(q, 0); Fourier inversion of the absolute first moment."""
    lam,b,c,_ = control.spectrum()
    return quadratic_positive_diag(lam,b,c)


def quadratic_cdf(control):
    """Gaussian P(q <= 0), exponentially tilted Laplace inversion."""
    from scipy.optimize import brentq

    lam, b, c, _ = control.spectrum()
    mu = c + lam.sum()
    sd = np.sqrt(2*(lam @ lam) + b @ b)
    if sd < 1e-24:
        return float(mu <= 0)
    flip = mu < 0
    fac = (-1 if flip else 1)/sd
    c, lam, b = c*fac, lam*fac, b*fac
    # Tiny fitted eigenvalues outside the active space do not belong in the inversion.
    ll = abs(lam) > 1e-10*max(np.max(abs(lam)), 1e-20)
    orthvar = b[~ll] @ b[~ll]
    lam, bb = lam[ll], b[ll]**2

    def log_mgf(t):
        den = 1-2*t*lam
        return t*c + 0.5*(t*t*(orthvar+(bb/den).sum()) - np.log(den).sum())

    def derivative(t):
        den = 1-2*t*lam
        return c + (lam/den).sum() + t*orthvar + (t*bb*(1-t*lam)/(den*den)).sum()

    # A fixed modest tilt near the centre avoids a nearly singular integrand.
    lim = 0.499 / max(-float(lam.min(initial=0)), 1e-4)
    end = -min(lim, 32.0)
    if derivative(end) < 0:
        shift = brentq(derivative, end, 0.0, xtol=1e-8)
    else:
        shift = end
    shift = min(-min(lim, 0.15), shift)
    kap = float(log_mgf(shift))
    if kap < -50:
        return float(flip)

    def integrand(t):
        w = shift + 1j*t
        return (-np.exp(log_mgf(w)-kap)/w).real

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val = sum(quad(integrand, a, b, epsabs=1e-8, epsrel=1e-7, limit=240)[0]
                  for a, b in ((0, 8), (8, 64), (64, 512), (512, 8192)))
    val *= np.exp(kap)/np.pi
    return float(np.clip(1-val if flip else val, 0, 1))


def grad_batch(session, x, step=2e-5, central=False):
    d = session.d
    e = step*np.eye(d)
    arrays = [x[:, None, :] + e]
    arrays.append(x[:, None, :] - e if central else x[:, None, :])
    vals = session.query(np.concatenate(arrays, axis=1)).reshape(len(x), -1)
    return (vals[:, :d] - vals[:, d:]) / (step*(1+central))


def quadrature_control(session, rng, basis, control=None, exact=0.0, antithetic=False):
    total = 0.0
    count = 0
    for p in net_sizes(session.remaining):
        if session.elapsed > session.limit-4.5 and count >= 1024:
            break
        if antithetic and p > 0:
            a = normals(rng, p-1, session.d) @ basis.T
            z = np.concatenate((a, -a))
        else:
            z = normals(rng, p, session.d) @ basis.T
        y = session.query(z)
        if control is not None:
            y -= control(z)
        total += y.sum()
        count += len(y)
    return exact + total/max(count, 1)
