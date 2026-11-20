"""Optional noninteger truncated-power ridges for continuous cusps."""

import itertools
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
from scipy import optimize

from core import Model, agrees, legendre_features, least
from ridges import connecting_line, central_line


@dataclass
class Cusp:
    n: np.ndarray
    b: float
    p: float


def power_moment(normal, b, p):
    with localcontext() as ctx:
        a = [Decimal.from_float(abs(float(z))) for z in normal if abs(z) > 2.e-13]
        ctx.prec = 30+12*len(a)
        shift = -Decimal.from_float(float(b))-sum(a)/2
        power = Decimal.from_float(float(p))
        terms = [(shift, 1)]
        for k in a:
            terms = [(r+k, c) for r, c in terms]+[(r, -c) for r, c in terms]
        if p > 0. and abs(round(p)-p) < 3.e-15:
            power = Decimal(int(round(p)))
        top = sum(c*x**(power+len(a)) for x, c in terms if x > 0)
        bottom = math.prod(a)*math.prod(power+j for j in range(1, len(a)+1))
        return np.longdouble(str(top/bottom))


def rough_regions(line, y):
    k = 4
    dif = np.abs(np.diff(y, n=k))
    if len(dif) < 16:
        return []
    largest = np.max(dif)
    if largest < 2.e-7*max(np.ptp(y), .005):
        return []
    floor = max(5.*np.median(dif), .09*largest, 1.e-9*np.max(np.abs(line.y)))
    if largest <= floor:
        return []
    ranked = np.argsort(-dif)
    chosen = []
    for j in ranked:
        i = j+2
        if dif[j] <= floor:
            break
        if all(abs(i-k) >= 5 for k in chosen):
            chosen.append(int(i))
    return chosen[:4]


def local_cusp(t, y, j):
    lo, hi = max(0, j-7), min(len(y), j+8)
    if hi-lo < 11:
        return None
    dt = t[1]-t[0]
    x = (t[lo:hi]-t[j])/dt
    y = y[lo:hi]
    f = np.column_stack([x**k for k in range(4)])
    u = f@least(f, y)-y
    norm = np.linalg.norm(u)
    if norm < 1.e-8*max(.001, np.max(np.abs(y))):
        return None
    def fun(z):
        a, p = z
        k = x-a
        g = np.column_stack((f, np.maximum(k, 0.)**p, np.maximum(-k, 0.)**p))
        return (g@least(g, y)-y)/norm
    best, params = float("inf"), None
    for ini in (.62, 1.35, 2.5, 3.35):
        sol = optimize.least_squares(fun, [0., ini],
                                     bounds=([-2.2, .18], [2.2, 4.6]),
                                     max_nfev=32, ftol=1.e-8, xtol=1.e-8, gtol=1.e-8)
        e = np.linalg.norm(sol.fun)
        if e < best:
            best, params = e, sol.x
        if best < 2.e-6:
            break
    if best > .075 or params is None:
        return None
    return params[0], params[1], j, lo, hi


def orient_cusp(s, line, mark, prev):
    a, p, j, lo, hi = mark
    n = hi-lo
    d = s.d
    if s.left < d*n+190:
        return None
    t = line.t[lo:hi]
    u = line.points(t)
    dt = line.t[1]-line.t[0]
    left, right = u.min(axis=0), 1.-u.max(axis=0)
    h = np.where(right > left, np.minimum(right*.75, .19*dt),
                 -np.minimum(left*.75, .19*dt))
    if np.min(np.abs(h)) < 2.e-6:
        return None
    xx = (u[None]+(np.eye(d)*h[:, None])[:, None]).reshape(-1, d)
    yy = s.ask(xx)/s.scale
    if prev is not None:
        yy -= prev.predict(xx)
    yy = yy.reshape(d, n)
    x = (t-line.t[j])/dt
    f = np.column_stack([x**k for k in range(4)])
    v = np.zeros(d)
    for k, y in enumerate(yy):
        scale = np.linalg.norm(f@least(f, y)-y)
        def error(z):
            q = x-z
            g = np.column_stack((f, np.maximum(q, 0.)**p, np.maximum(-q, 0.)**p))
            return np.linalg.norm(g@least(g, y)-y)
        z = optimize.minimize_scalar(error, bounds=(a-1.8, a+1.8), method="bounded",
                                     options={"maxiter": 38, "xatol": 1.e-8})
        if z.fun > .15*scale or abs(z.x-a) > 1.76:
            return None
        v[k] = -(z.x-a)*dt/h[k]
    if abs(v@line.v-1.) > .38 or np.max(np.abs(v)) < 1.e-5:
        return None
    v /= v[np.argmax(np.abs(v))]
    b = (line.points(line.t[j]+a*dt)-.5)@v
    return Cusp(v, float(b), float(p))


def fit_cusps(s, cusps, order=1):
    x, y = s.data()
    t = x-.5
    d, m = s.d, len(cusps)
    n = np.array([a.n for a in cusps])
    b = np.array([a.b for a in cusps])
    p = np.array([a.p for a in cusps])
    fixed = np.argmax(np.abs(n), axis=1)
    free = np.arange(d)[None, :] != fixed[:, None]
    F = legendre_features(x, order)
    f = F.shape[1]

    def basis(n, b, p):
        z = t@n.T-b
        l = np.maximum(-z, 0.)
        r = np.maximum(z, 0.)
        return np.column_stack((F, l**p, r**p)), l, r
    G = basis(n, b, p)[0]
    c = least(G, y)
    sn = m*(d-1)
    sh = sn+2*m
    init = np.r_[n[free], b, p, c]
    def func(z, jac=False, unpack=False):
        ns = n.copy()
        ns[free] = z[:sn]
        bs = z[sn:sn+m]
        ps = z[sn+m:sh]
        cs = z[sh:]
        if unpack:
            return ns, bs, ps, cs
        h, l, r = basis(ns, bs, ps)
        if not jac:
            return h@cs-y
        lc = cs[f:f+m]
        rc = cs[f+m:f+2*m]
        L, R = np.maximum(l, 1.e-13), np.maximum(r, 1.e-13)
        dq = ps*(rc*(r > 0.)*R**(ps-1.)-lc*(l > 0.)*L**(ps-1.))
        dp = lc*h[:, f:f+m]*np.log(L)+rc*h[:, f+m:f+2*m]*np.log(R)
        return np.column_stack(((dq[:, :, None]*t[:, None, :])[:, free], -dq, dp, h))
    # Refine directions and exponents together once there is a useful start.
    if (len(init) < min(130, len(x)) and s.time_left() > 5.
            and np.max(np.abs(c)) < 300.):
        low = np.r_[np.full(sn, -1.45), np.full(m, -s.d*.55),
                    np.full(m, .17), np.full(len(c), -800.)]
        up = np.r_[np.full(sn, 1.45), np.full(m, s.d*.55),
                   np.full(m, 4.7), np.full(len(c), 800.)]
        opt = optimize.least_squares(func, np.clip(init, low+1.e-10, up-1.e-10),
                                     jac=lambda z: func(z, True),
                                     bounds=(low, up), x_scale="jac", max_nfev=66,
                                     ftol=5.e-12, xtol=3.e-12, gtol=4.e-13)
        if np.linalg.norm(opt.fun) < np.linalg.norm(G@c-y):
            n, b, p, c = func(opt.x, unpack=True)
    def predict(z):
        q = (z-.5)@n.T-b
        return np.column_stack((legendre_features(z, order),
                                np.maximum(-q, 0.)**p, np.maximum(q, 0.)**p))@c
    def mean():
        res = np.longdouble(c[0])
        for j in range(m):
            if abs(c[f+j]) > 1.e-14:
                res += np.longdouble(c[f+j])*power_moment(-n[j], -b[j], p[j])
            if abs(c[f+m+j]) > 1.e-14:
                res += np.longdouble(c[f+m+j])*power_moment(n[j], b[j], p[j])
        return float(res)
    model = Model(predict, float("nan"), "cusp ridges")
    model.integral = mean
    refined = [Cusp(n[j], float(b[j]), float(p[j])) for j in range(m)]
    return model, refined


def same(a, b):
    return min(np.linalg.norm(a.n-b.n)+abs(a.b-b.b),
               np.linalg.norm(a.n+b.n)+abs(a.b+b.b)) < .026


def try_cusps(s, line):
    if s.d > 8 or s.left < 530 or not rough_regions(line, line.y):
        return None
    cusps = []
    model = None
    visited = []
    for step in range(5):
        if s.left < 280 or s.time_left() < 6.:
            break
        residual = line.y-model.predict(line.points(line.t)) if model else line.y
        new = []
        for j in rough_regions(line, residual):
            attempt = local_cusp(line.t, residual, j)
            if attempt is None:
                continue
            c = orient_cusp(s, line, attempt, model)
            if c is not None and not any(same(c, a) for a in cusps+new):
                new.append(c)
            if len(new) >= 2 or s.left < 320:
                break
        if new:
            model, cusps = fit_cusps(s, cusps+new, 1)
            x, y = s.data()
            e = model.predict(x)-y
            if np.max(np.abs(e)) < 1.5e-7:
                if s.left < 54:
                    break
                zz = s.design(54)
                yy = s.ask(zz)/s.scale
                if agrees(model.predict(zz), yy, 3.e-9, 8.e-10):
                    model.mean = model.integral()
                    return model
        if s.left < 260 or len(cusps) > 6:
            break
        if model is None:
            if step >= 1:
                break
            line = central_line(s, 65, attempt=1)
        else:
            x, y = s.data()
            resid = y-model.predict(x)
            w = np.abs(resid).copy()
            for v in visited:
                w *= 1.-.9*np.exp(-np.sum((x-v)**2, axis=1)/.018)
            top = x[np.argmax(w)]
            visited.append(top.copy())
            other = s.rng.uniform(.17, .83, s.d)
            line = connecting_line(s, top, other, 65)
    if model is not None and s.left >= 132 and s.time_left() > 3.:
        x, y = s.data()
        e = model.predict(x)-y
        if np.linalg.norm(e) < .18*np.linalg.norm(y-y.mean()):
            model.mean = model.integral()
            if np.isfinite(model.mean) and np.min(y)-1.e-8 <= model.mean <= np.max(y)+1.e-8:
                s.hint = model
    return None
