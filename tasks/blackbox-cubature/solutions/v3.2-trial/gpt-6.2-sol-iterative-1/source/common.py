"""Numerical primitives for black-box Gaussian cubature."""

import math
import sys
import time
import warnings

import numpy as np
from scipy.linalg import eigh, solve
from scipy.special import ndtri
from scipy.stats import qmc


def report(*args):
    # Only called on diagnostic or recovery paths.
    print(*args, file=sys.stderr, flush=True)


class Work:
    def __init__(self, session):
        self.s = session
        self.d = session.d
        self.rng = np.random.default_rng([session.seed % 2**64, 902137])
        self.start = time.monotonic()
        self.limit = float(session.init.get('time_limit_s', 30))
        self.pilot_x = self.pilot_y = None
        self.safe = None

    @property
    def left(self):
        return self.s.remaining

    def elapsed(self):
        return time.monotonic() - self.start

    def ask(self, x):
        if self.elapsed() > max(3.,self.limit-5.):
            raise ArithmeticError('wall-clock reserve')
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if len(x) > self.left:
            raise ArithmeticError('insufficient cubature budget')
        if not np.all(np.isfinite(x)):
            raise ArithmeticError('could not construct finite query coordinates')
        out = self.s.query(np.clip(x, -40, 40))
        if not np.all(np.isfinite(out)):
            raise ArithmeticError('non-finite function values')
        return out

    def uniforms(self, n, d):
        if n == 0:
            return np.empty((0, d))
        sobol = qmc.Sobol(d, seed=self.rng, scramble=True, bits=30)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return sobol.random(n)

    def normal(self, n, d=None):
        return gauss(self.uniforms(n, self.d if d is None else d))

    def pilot(self, n):
        self.pilot_x = self.normal(n)
        self.pilot_y = self.ask(self.pilot_x)
        self.safe = float(np.mean(self.pilot_y))
        return self.pilot_x, self.pilot_y


def gauss(u):
    return ndtri(np.clip(u, 1.e-15, 1-1.e-15))


def unit(x):
    norm = np.linalg.norm(x)
    if norm < 1.e-18:
        out = np.zeros_like(x)
        out[0] = 1
        return out
    return x / norm


def perp(v):
    """Orthonormal complement with a well-conditioned Householder reflection."""
    ix = int(np.argmax(np.abs(v)))
    a = np.array(v, copy=True)
    a[ix] += math.copysign(1., a[ix])
    Q = np.eye(len(a)) - (2/np.dot(a, a))*np.outer(a, a)
    return Q[:, np.arange(len(a)) != ix]


def eigenbasis(H, absolute=False):
    val, vec = eigh((H + H.T) * .5, check_finite=False)
    return vec[:, np.argsort(np.abs(val) if absolute else val)[::-1]]


def first_basis(v, M=None):
    v = unit(v)
    other = perp(v)
    if M is not None and len(v) > 1:
        other = other @ eigenbasis(other.T @ M @ other)
    return np.column_stack([v, other])


def ridge(X, y, penal, center=False):
    if center:
        xm = X.mean(axis=0)
        m = y.mean(axis=0)
        b = ridge(X-xm, y-m, penal)
        return m-xm@b, b
    gram = X.T @ X
    gram.flat[::len(gram)+1] += penal
    return solve(gram, X.T @ y, assume_a='pos', check_finite=False)


def gradient(w, x, h=.001):
    x = np.atleast_2d(x)
    E = h * np.eye(w.d)
    X = x[:, None, :] + E[None, :, :]
    raw = w.ask(np.concatenate([x, X.reshape(-1, w.d)]))
    return raw[:len(x)], (raw[len(x):].reshape(-1, w.d) - raw[:len(x), None]) / h


def jet(w, x, h=.001, frame=None):
    """A second-order jet from a forward triangular stencil."""
    d = w.d
    i, j = np.triu_indices(d, 1)
    D = np.eye(d) if frame is None else frame.T
    e = h * D
    out = w.ask(x + np.concatenate([np.zeros((1, d)), e, 2*e, e[i]+e[j]]))
    a, b = out[1:d+1], out[d+1:2*d+1]
    g = (-3*out[0]+4*a-b)/(2*h)
    H = np.empty((d, d))
    np.fill_diagonal(H, (out[0]-2*a+b)/h**2)
    H[i, j] = H[j, i] = (out[2*d+1:]-a[i]-a[j]+out[0])/h**2
    return out[0], D.T @ g, D.T @ H @ D


def moment_rotation(x, y, grad=None):
    c = y - np.mean(y)
    n = len(c)
    g = c @ x / n
    H = (x.T * c) @ x / n
    cov = H @ H
    if grad is not None and len(grad):
        cov *= .07
        cov += grad.T @ grad / len(grad)
    cov += 1.8 * np.outer(g, g)
    return eigenbasis(cov)


def rqm(w, basis=None, n=None):
    n = w.left if n is None else min(n, w.left)
    if n < 32:
        return w.safe if w.safe is not None else 0.
    z = w.normal(n)
    y = w.ask(z if basis is None else z@basis.T)
    return float(np.mean(y))


def normal_moments(a, b, max_order=4):
    """Unnormalized t^j Gaussian integrals over [a,b] (finite endpoints)."""
    from scipy.special import ndtr
    phi_a = np.exp(-a*a/2)/math.sqrt(2*math.pi)
    phi_b = np.exp(-b*b/2)/math.sqrt(2*math.pi)
    M = [np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))]
    if max_order:
        M.append(phi_a-phi_b)
    for k in range(2, max_order+1):
        M.append(a**(k-1)*phi_a-b**(k-1)*phi_b+(k-1)*M[k-2])
    return M
