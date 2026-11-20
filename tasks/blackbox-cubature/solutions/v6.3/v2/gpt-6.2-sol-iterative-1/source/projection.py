"""One-dimensional projections and integration of periodic functions."""

from decimal import Decimal, localcontext
import math

import numpy as np
from scipy import fft
from numpy.polynomial.chebyshev import chebval

from core import agrees, gauss_extended
from ridges import Line, fit_wave, spectral_line


def face_periods(s, n=3):
    d = s.d
    if s.left < 2*n*d+350:
        return np.zeros(d, dtype=bool)
    z = .06+.88*s.design(n)
    q = np.broadcast_to(z, (d, n, d)).copy()
    j = np.arange(d)
    q[j, :, j] = 0.
    p = q.copy()
    p[j, :, j] = 1.
    y = (s.ask(np.vstack((q.reshape(-1, d), p.reshape(-1, d))))/s.scale).reshape(2, d, n)
    return np.all(np.abs(y[0]-y[1]) < 9.e-12*(1.+np.abs(y[0])+np.abs(y[1])), axis=1)


def nearest_prime(n):
    k = min(max(3, int(n)), 100003)
    if not k & 1:
        k -= 1
    while k > 3:
        if all(k % j for j in range(3, math.isqrt(k)+1, 2)):
            break
        k -= 2
    return k


def lattice(s, n, d=None):
    d = s.d if d is None else d
    gen = np.empty(d, dtype=np.int64)
    used = set()
    for j in range(d):
        while True:
            g = int(s.rng.integers(1, n))
            if g not in used or len(used) >= n-1:
                break
        gen[j] = g
        used.add(g)
        used.add(n-g)
    v = np.arange(n, dtype=np.int64)[:, None]*gen
    return (v/n % 1.+s.rng.random(d)) % 1.


def periodic_mean(s):
    if s.left < 450:
        return None
    # Independent prime lattices are unlikely to share an alias. Agreement
    # then gives an internal check on the constant Fourier coefficient.
    val = []
    for multiplier in (.29, .36, .48, .54, .89):
        n = nearest_prime(max(31, min(s.left, (s.left-100)*multiplier)))
        if n < 31 or n > s.left:
            break
        y = s.ask(lattice(s, n))/s.scale
        val.append(math.fsum(y)/len(y))
        if len(val) >= 3:
            a = np.sort(val)
            dif = a[2:]-a[:-2]
            j = int(np.argmin(dif))
            if dif[j] < 3.e-14*(1.+abs(a[j]+a[j+2])):
                return float(np.mean(a[j:j+3]))
    if val:
        # Periodic but not sparse in frequency: keep the separate random
        # shifts, using a robust center to suppress an occasional alias.
        return float(np.median(val))
    return None


def get_direction(s):
    d = s.d
    if s.left < 24*d+170 or not 2 <= d <= 10:
        return None
    anchors = .14+.72*s.design(4)
    h = .000073
    st = np.array([-3., -2., -1., 1., 2., 3.])
    ww = np.array([-1., 9., -45., 45., -9., 1.])/(60*h)
    p = (anchors[:, None, None, :]+st[None, None, :, None]*h*np.eye(d)[None, :, None, :])
    f = (s.ask(p.reshape(-1, d))/s.scale).reshape(len(anchors), d, 6)
    g = f@ww
    norm = np.linalg.norm(g, axis=1)
    k = int(np.argmax(norm))
    good = norm > max(2.e-6, 2.e-4*norm[k])
    if np.count_nonzero(good) < 2:
        return None
    i = int(np.argmax(np.abs(g[k])))
    ratio = g[k]/g[k, i]
    res = g[good]-g[good, i, None]*ratio
    if np.max(np.abs(res)/norm[good, None]) > 4.e-7:
        return None
    jj = np.arange(d) != i
    ii = np.argsort(-norm)[:2]
    ii = ii[good[ii]]
    a = anchors[ii]
    b = s.ask(a)/s.scale
    delta = np.array([.091, -.103])[:len(a)]
    p0 = a[:, None]+delta[:, None, None]*np.eye(d)[None, jj]
    jac = delta*g[ii, i]
    for _ in range(3):
        p = p0.copy()
        p[:, :, i] -= delta[:, None]*ratio[None, jj]
        err = (s.ask(p.reshape(-1, d))/s.scale).reshape(len(a), -1)-b[:, None]
        ratio[jj] += (jac@err)/(jac@jac)
        if np.max(np.abs(err), initial=0.) < 5.e-15:
            break
    if np.max(np.abs(ratio)) > 1.000001:
        return None
    return ratio


class Projection:
    """Distribution of a weighted sum of independent uniform variables."""

    def __init__(self, v):
        v = np.asarray(v, dtype=float)
        self.v = v
        self.total = np.sum(np.abs(v))
        self.d = np.count_nonzero(np.abs(v) > 8.e-16*self.total)
        self.positive = v >= 0.
        self._setup(np.abs(v[np.abs(v) > 8.e-16*self.total]))

    def restrict(self, x):
        return np.clip(.5+((x-.5)@self.v)/self.total, 0., 1.)

    def lift(self, t):
        return np.where(self.positive, t[..., None], 1.-t[..., None])

    def _setup(self, a):
        d = len(a)
        p = d-1
        if d == 1:
            self.edges = np.array([0., .5])
            self.coeffs = np.ones((1, 1))
            return
        with localcontext() as ctx:
            ctx.prec = max(64, 22+16*d)
            z = [Decimal.from_float(float(q)) for q in a]
            total = sum(z)
            z = [q/total for q in z]
            terms = [(Decimal(0), 1)]
            half = Decimal(".5")
            for w in z:
                terms += [(u+w, -s) for u, s in terms if u+w < half]
            grouped = {}
            for u, s in terms:
                grouped[u] = grouped.get(u, 0)+s
            b = sorted(u for u, c in grouped.items() if c)
            b.append(half)
            den = math.factorial(p)*math.prod(z)
            coeff = []
            edges = [0.]
            for i, (l, r) in enumerate(zip(b, b[1:])):
                if r-l < Decimal("1e-17"):
                    continue
                delta = [l-u for u in b[:i+1]]
                mult = [grouped[u] for u in b[:i+1]]
                row = []
                for k in range(p+1):
                    m = p-k
                    w = sum(s*t**m for t, s in zip(delta, mult) if t) if m else sum(mult)
                    row.append(np.longdouble(str(w*math.comb(p, k)*(r-l)**k/den)))
                edges[-1] = np.longdouble(str(l))
                edges.append(np.longdouble(str(r)))
                coeff.append(row)
        self.edges = np.array(edges)
        self.coeffs = np.array(coeff)

    def density(self, t):
        t = np.minimum(t, 1.-t)
        i = np.clip(np.searchsorted(self.edges, t, side="right")-1, 0, len(self.coeffs)-1)
        z = (t-self.edges[i])/(self.edges[i+1]-self.edges[i])
        return np.maximum(0., np.polynomial.polynomial.polyval(z, self.coeffs[i].T, tensor=False))

    def integrate(self, c):
        n = max(8, (len(c)+self.d+1)//2)
        z, w = gauss_extended(n)
        a, b = self.edges[:-1], self.edges[1:]
        t = (a[:, None]+(b-a)[:, None]*z).ravel()
        density = self.coeffs@np.array([z**k for k in range(self.d)])
        W = (density*(b-a)[:, None]*w).ravel()
        # Use symmetry of the projected measure to integrate only even
        # Chebyshev terms, including the constant exactly.
        c = c.copy()
        c[1::2] = 0.
        return float((W@chebval(2.*t-1., c))/np.sum(W))


def refine_fourier(s, proj, c, moment):
    """Prefer closed-form Fourier moments after a successful spectral fit."""
    if s.time_left() < 2.5:
        return moment
    n = min(321, max(97, len(c)))
    t = np.linspace(0., 1., n)
    f = np.asarray(chebval(2.*t-1., c), dtype=float)
    line = Line(proj.lift(np.array(0.)), np.where(proj.positive, 1., -1.), t, f)
    found = spectral_line(line)
    if found is None:
        return moment
    frequencies, order = found
    frequencies = (frequencies/proj.total)[:, None]*proj.v
    x, y = s.data()
    m = fit_wave(x, y, frequencies, order, max_nfev=34, exact=True)
    if m is None or not agrees(m.predict(x), y, 3.e-13, 4.e-14):
        return moment
    if abs(m.mean-moment) > 3.e-12*(1.+abs(m.mean)):
        return moment
    return m.mean


def try_projection(s, nbase):
    v = get_direction(s)
    if v is None:
        return None
    proj = Projection(v)
    x, y = s.data()
    x = x[:nbase]
    y = y[:nbase]
    t = proj.restrict(x)
    ii = np.unique(np.r_[np.arange(min(16, nbase)), np.argsort(-np.abs(y))[:7]])
    target = s.ask(proj.lift(t[ii]))/s.scale
    if not agrees(target, y[ii], 1.2e-10, 1.e-11):
        return None
    # A nested Chebyshev sequence resolves arbitrary smooth profiles, not
    # only a prescribed number of sine waves.
    n = 64
    yy = s.ask(proj.lift(.5+.5*np.cos(np.arange(n+1)*np.pi/n)))/s.scale
    best = float("inf")
    coeff = None
    while True:
        c = fft.dct(yy.astype(np.longdouble), type=1)/n
        c[0] *= .5
        c[-1] *= .5
        err = np.max(np.abs(chebval(2.*t-1., c)-y))
        tail = np.max(np.abs(c[int(n*.80):]))
        score = err+tail
        if score < best:
            best, coeff = score, c
        if agrees(chebval(2.*t-1., c), y, 3.e-11, 2.e-12) and tail < 1.8e-12:
            mean = proj.integrate(c)
            return refine_fourier(s, proj, c, mean)
        if s.left < n+36 or s.time_left() < 2.5 or n >= 1024:
            break
        n *= 2
        vals = np.empty(n+1)
        vals[::2] = yy
        vals[1::2] = s.ask(proj.lift(.5+.5*np.cos(np.arange(1, n, 2)*np.pi/n)))/s.scale
        yy = vals
    if coeff is not None:
        if best < 5.e-6 or s.left < 130:
            return proj.integrate(coeff)
    return None
