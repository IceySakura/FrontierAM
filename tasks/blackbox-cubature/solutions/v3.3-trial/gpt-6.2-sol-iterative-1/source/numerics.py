"""Gaussian rules, polynomial models, and integration of quadratic surrogates."""
import math
import warnings

import numpy as np
from scipy import integrate, linalg, special
from scipy.stats import qmc


FLOAT_EPS = np.finfo(float).eps


def log(*args):
    import os
    if os.environ.get("CUBATURE_DEBUG") == "1":
        import sys
        print(*args, file=sys.stderr, flush=True)


def sobol(n, d, rng):
    if n <= 0:
        return np.empty((0, d))
    seq = qmc.Sobol(d, scramble=True, seed=int(rng.integers(2**31)))
    return seq.random_base2((int(n) - 1).bit_length())[:n]


def normal(n, d, rng):
    return special.ndtri(np.clip(sobol(n, d, rng), 1e-15, 1 - 1e-15))


def orthogonal_to(V, d=None):
    if d is None:
        d = len(V)
    if V.ndim == 1:
        V = V[:, None]
    if V.shape[1] == 0:
        return np.eye(d)
    O, _ = np.linalg.qr(V, mode="complete")
    sign = np.sign(np.sum(O[:, :V.shape[1]] * V, axis=0))
    O[:, :V.shape[1]] *= np.where(sign == 0, 1., sign)
    return O


def sym(x):
    return (x + x.T) * .5


class Quad:
    """c + b dot x + x dot A x."""
    def __init__(self, A, b, c):
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        self.c = float(c)

    def __call__(self, x):
        return np.sum((x @ self.A) * x, axis=-1) + x @ self.b + self.c

    def grad(self, x):
        return 2 * (x @ self.A) + self.b

    def scaled(self, v):
        return Quad(self.A * v, self.b * v, self.c * v)

    def plus(self, q):
        return Quad(self.A + q.A, self.b + q.b, self.c + q.c)

    def restrict(self, V):
        return Quad(V.T @ self.A @ V, V.T @ self.b, self.c)

    def mean(self):
        return self.c + np.trace(self.A)

    def sd(self):
        return np.sqrt(2 * np.sum(self.A**2) + self.b @ self.b)

    @classmethod
    def at(cls, A, g, x, f):
        return cls(A, g - 2 * A @ x, f - g @ x + x @ A @ x)


def fd_gradient(query, x, h=1e-4):
    x = np.atleast_2d(x)
    E = h * np.eye(x.shape[1])
    vals = query((x[:, None] + np.vstack([E, -E])[None]).reshape(-1, x.shape[1]))
    p, m = np.split(vals.reshape(len(x), -1), 2, axis=1)
    return (p-m) / (2*h)


def quad_stencil(d, k=None):
    if k is None:
        k = d
    I = np.eye(d)
    i, j = np.triu_indices(d, 1)
    ok = i < k
    i, j = i[ok], j[ok]
    stencil = np.vstack([np.zeros((1, d)), I, -I[:k], I[i] + I[j]])
    return stencil, i, j


def unpack_stencil(Y, h, d, k, i, j):
    f = Y[0]
    yp, ym = Y[1:1+d], Y[1+d:1+d+k]
    D = np.zeros((d, k))
    D[np.arange(k), np.arange(k)] = (yp[:k] + ym - 2 * f) / h**2
    D[j, i] = (Y[1+d+k:] - yp[i] - yp[j] + f) / h**2
    ok = j < k
    D[i[ok], j[ok]] = D[j[ok], i[ok]]
    g = (yp-f)/h
    g[:k] = (yp[:k]-ym)/(2*h)
    return f, g, D


def hessian(query, x, scale=.003, R=None):
    d = len(x)
    S, i, j = quad_stencil(d)
    Y = query(x + scale * (S if R is None else S @ R.T))
    f, g, D = unpack_stencil(Y, scale, d, d, i, j)
    if R is not None:
        g, D = R @ g, R @ D @ R.T
    return f, g, D


def rect_gaussian_regression(x, y, rank_quad=0, R=None, ridge=1e-5):
    """Gaussian moment features; quadratic terms span leading directions only."""
    d = x.shape[1]
    R = np.eye(d) if R is None else R
    i, j = np.triu_indices(rank_quad, 1)

    def feat(z):
        a = z @ R
        return np.concatenate([np.ones((len(a), 1)), a,
                               (a**2-1) / math.sqrt(2),
                               a[:, i] * a[:, j]], axis=1)

    X = feat(x)
    A = X.T @ X
    A.flat[::len(A)+1] += ridge * len(y)
    c = linalg.solve(A, X.T @ y, assume_a="pos")
    return float(c[0]), lambda z: feat(z) @ c


def positive_mean(q):
    """Quadratic positive-part mean via a characteristic function integral."""
    if q.sd() <= 1e-14:
        return max(0., q.mean())
    lam, V = np.linalg.eigh(q.A)
    scale = max(q.sd(), abs(q.mean()) / 3)
    lam, b, c = lam/scale, (V.T @ q.b)/scale, q.c/scale
    mean = q.mean()/scale

    def fn(t):
        L = 1 - 2j*t*lam
        log_cf = 1j*t*c - .5*np.log(L).sum() - .5*t*t*np.sum(b*b/L)
        return -np.expm1(log_cf).real / (t*t)

    # The nonoscillating 1/t^2 term is integrated analytically in the tail.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        a = integrate.quad(fn, 0, 12, epsabs=2e-11, limit=180)[0]
        a += integrate.quad(lambda t: fn(t)-1/(t*t), 12, np.inf,
                            epsabs=2e-10, limit=550)[0] + 1/12
    return max(0., scale * (mean*.5 + a/math.pi))


class GammaSeries:
    """Positive quadratic-form CDF using a positive mixture of gamma laws."""
    def __init__(self, lam, mu, terms=2400):
        lam, mu = np.asarray(lam), np.asarray(mu)
        self.beta = np.min(lam)
        rho = np.clip(1-self.beta/lam, 0, 1)
        c = np.exp(.5 * np.log(self.beta/lam).sum() - .5 * np.sum(mu*mu))
        k = np.arange(1, terms+1)
        L = .5 * np.sum((rho[None]**(k[:, None]-1)) *
                        (rho[None] + k[:, None] * (mu*mu*(1-rho))[None]), axis=1)
        a = np.empty(terms+1)
        a[0] = c
        total = c
        for i in range(1, terms+1):
            a[i] = L[:i] @ a[i-1::-1] / i
            total += a[i]
            if total >= 1-5e-14:
                a = a[:i+1]
                break
        self.a = a
        self.shapes = np.arange(len(a)) + len(lam)/2
        self.mass = total

    def cdf(self, value):
        xx = np.maximum(np.asarray(value)/(2*self.beta), 0.)
        return special.gammainc(self.shapes, xx[..., None]) @ self.a


def ellipsoid_prob(A, b, c, tilt=None, rel=0.):
    """P(x'Ax+b'x+c < 0) under N(tilt,I), for positive semidefinite A."""
    e, V = np.linalg.eigh(A)
    keep = e > max(float(e[-1])*max(1e-10, rel), 1e-15)
    if not np.any(keep) or e[0] < -max(e[-1]*1.e-7, 1e-12):
        raise ValueError("not a positive quadratic form")
    e, V = e[keep], V[:, keep]
    b0 = V.T @ b
    mu = b0/(2*e)
    r = np.sum(e*mu**2)-c
    if r <= 0:
        return 0.
    if tilt is not None:
        mu = mu + V.T @ tilt
    s = GammaSeries(e, mu)
    p = float(s.cdf(r))
    if s.mass < 1-3e-10 and p < .99:
        # When the quadratic spectrum is poorly conditioned, invert the CF.
        cs = min(r, np.sqrt(np.sum((2 + 4*mu**2)*e**2)))
        e, r = e/cs, r/cs
        nonc = (mu*mu)*e

        def cf(t):
            l = 1-2j*t*e
            return np.exp(-1j*t*r - .5*np.log(l).sum() + 1j*t*np.sum(nonc/l)).imag/t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            p = .5-integrate.quad(cf, 0, np.inf, epsabs=2e-10, limit=1200)[0]/math.pi
    return float(np.clip(p, 0, 1))


def neg_mixture(q, cutoff=3e-5):
    e, V = np.linalg.eigh(-q.A)
    keep = e > max(e[-1]*cutoff, 2e-7)
    if not np.any(keep):
        raise ValueError("no active curvature")
    V, e = V[:, keep], e[keep]
    b = V.T @ q.b
    center = b/(2*e)
    m = q.c + np.sum(e*center**2)
    return m, GammaSeries(e, center), V


def max_mean(qs, rng):
    if not qs:
        return 0.
    if len(qs) == 1:
        q = qs[0]
        return positive_mean(q)
    try:
        models = [neg_mixture(q) for q in qs]
        if all(np.linalg.norm(a[2].T @ b[2], ord=2) < 2e-3 for i, a in enumerate(models)
               for b in models[:i]):
            t, w = special.roots_legendre(160)
            m = max(0., *(a[0] for a in models))
            t = m*(t+1)*.5
            p = np.ones(len(t))
            for top, model, _ in models:
                p *= 1-model.cdf(top-t)
            return float((1-p) @ w * m*.5)
    except (ValueError, np.linalg.LinAlgError):
        pass
    # Independent cheap surrogate-only integration when its pieces overlap.
    es = []
    d = len(qs[0].b)
    AA = sum(q.A @ q.A + .25 * np.outer(q.b, q.b) for q in qs)
    _, R = np.linalg.eigh(AA)
    for _ in range(6):
        z = normal(2**16, d, rng) @ R[:, ::-1].T
        es.append(np.maximum(0., np.max([q(z) for q in qs], axis=0)).mean())
    return float(np.mean(es))


def balanced_blocks(n, d, rng):
    """Distribute a non-power-of-two budget across complete scrambled nets."""
    n = int(n)
    if n <= 0:
        return []
    k = 1 << max(0, (n//2).bit_length()-1)
    out = []
    while n:
        m = min(k, 1 << (n.bit_length()-1))
        if m < 64 and out:
            break
        out.append(normal(m, d, rng))
        n -= m
    return out


def balanced_rule(n, d, rng):
    out = balanced_blocks(n,d,rng)
    return np.vstack(out) if out else np.empty((0,d))
