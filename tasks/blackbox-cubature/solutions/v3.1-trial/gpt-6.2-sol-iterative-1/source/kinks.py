"""Local patch reconstruction for continuous piecewise quadratics."""

import itertools
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import ndtr

from cubature import Quadratic, normals, ordered_basis, quadratic_positive, rotation


def debug(*args):
    if os.environ.get("CUBATURE_DEBUG"):
        print(*args, file=sys.stderr, flush=True)


class Atlas:
    def __init__(self, session, rng, count=32, size=25):
        d = session.d
        a, b = rng.normal(size=(2, count, d))
        self.v = a-b
        self.v /= np.linalg.norm(self.v, axis=1, keepdims=True)
        # Some lines probe a typical transverse fibre; others pass closer to zero.
        self.base = (a+b) / np.sqrt(2)
        self.base[:count//2] *= 0.67
        self.t = np.linspace(-1, 1, size) * (3.0 + 0.42*np.sqrt(d))
        self.h = self.t[1]-self.t[0]
        self.z = self.base[:, None, :] + self.t[None, :, None]*self.v[:, None, :]
        self.y = session.query(self.z).reshape(count, size)
        self.scale = max(1e-9, np.quantile(abs(self.y), 0.65))

    def candidates(self, residual=None):
        f = self.y if residual is None else residual
        dy = abs(np.diff(f, n=4, axis=1))
        accept = ((dy >= maximum_filter1d(dy, size=4, axis=1)) &
                  (dy > 1.0e-5*self.scale))
        accept[:, :2] = False
        accept[:, -2:] = False
        a, b = np.where(accept)
        order = np.argsort(-dy[a, b])
        # Return the centre of the fourth difference.
        return [(int(i), int(j)+2) for i, j in zip(a[order], b[order])]

    def score(self, residual):
        q = np.abs(np.diff(residual, n=4, axis=1))
        return np.mean(q*q)


def crossing(session, atlas, line, ind):
    """Find a root of the difference of two one-sided extrapolants."""
    h = atlas.h
    tm = atlas.t[ind]
    v, x = atlas.v[line], atlas.base[line]
    ts = np.linspace(1.24, 3.12, 7)
    offsets = np.r_[-ts, ts]
    y = session.query(x + (tm+h*offsets[:, None])*v)
    a = np.polynomial.polynomial.polyfit(-ts, y[:7], 5)
    b = np.polynomial.polynomial.polyfit(ts, y[7:], 5)
    full = b-a
    if max(abs(y)) < 1e-30:
        return None
    roots = np.polynomial.polynomial.polyroots(full)
    roots = roots.real[(abs(roots.imag) < 1e-6) & (abs(roots.real) < 0.98)]
    if not len(roots):
        return None
    rr = min(roots, key=abs)
    p = x + (tm+h*rr)*v

    # Refinement checks that both smooth pieces continue to the same point.
    xl = h*0.13
    offs = np.arange(1, 7)*0.23
    xx = np.r_[-offs, offs]
    for _ in range(2):
        Y = session.query(p + (xl*xx[:, None])*v)
        left = np.polynomial.polynomial.polyfit(-offs, Y[:6], 4)
        right = np.polynomial.polynomial.polyfit(offs, Y[6:], 4)
        f = right-left
        if abs(f[1]) < 2e-6*atlas.scale*xl:
            return None
        dt = -f[0]/f[1]
        if abs(dt) > 0.19 or max(abs(f[3:])) > max(1e-9*atlas.scale, .008*abs(f[1])):
            return None
        p = p+xl*dt*v
        xl *= 0.31
    deriv = (f[1]+2*f[2]*dt)/(xl/0.31)
    return p, v, deriv


def local_quadratic(session, info, reserve):
    """Matched two-sided patch derivatives, with optional Nyström compression."""
    p, v, slope = info
    d = session.d
    if session.remaining < reserve + 12*d:
        return None
    a0, h0 = 0.0003, 4e-6
    xs = np.array([p-a0*v, p+a0*v])
    pts = np.concatenate((xs[:, None, :], xs[:, None, :]+h0*np.eye(d)), axis=1)
    yy = session.query(pts).reshape(2, d+1)
    dif = (yy[1, 1:]-yy[1, 0] - yy[0, 1:]+yy[0, 0])/h0
    if abs(dif @ v-slope) > 0.05*abs(slope) or np.linalg.norm(dif) < 1e-12:
        return None
    n = np.copysign(1.,slope)*dif/np.linalg.norm(dif)
    U = rotation(n)
    a = 8e-5
    h = np.full(d, 1.7e-3)
    h[0] = 2.5e-5
    dq = h[:, None]*U.T
    centers = p + (a*np.array([-1., 1.])[:, None])*n
    st = np.concatenate([np.zeros((1, d)), dq, 2*dq], axis=0)
    z = centers[:, None, :]+st
    vals = session.query(z).reshape(2, len(st))
    steps = vals[:, 1:1+d]-vals[:, :1]
    diag = (vals[:, 1+d:]-vals[:, :1]-2*steps)/(h*h)
    H = np.full((2, d, d), np.nan)
    ids = np.arange(d)
    H[:, ids, ids] = diag

    def fill(i, j):
        ij = np.isnan(H[0, i, j])
        i, j = i[ij], j[ij]
        if not len(i):
            return True
        if 2*len(i) + reserve > session.remaining:
            return False
        x = centers[:, None, :] + (dq[i]+dq[j])[None]
        y = session.query(x).reshape(2, -1)
        hh = (y - vals[:, :1] - steps[:, i] - steps[:, j])/(h[i]*h[j])
        H[:, i, j] = hh
        H[:, j, i] = hh
        return True

    delta = None
    for k in (min(d, 6), min(d, 11), min(d, 20), d):
        i, j = np.triu_indices(d, 1)
        mask = i < k
        if not fill(i[mask], j[mask]):
            break
        B = H[1]-H[0]
        if k == d:
            delta = B
            break
        r = k-2
        rec = B[:, :r]
        vals2, vec = np.linalg.eigh(rec[:r])
        good = abs(vals2) > max(2e-6, 1.5e-4*np.max(abs(vals2)))
        if not np.any(good):
            continue
        C = rec @ vec[:, good]
        candidate = (C/vals2[good]) @ C.T
        scale = max(np.max(abs(B[:, :k])), 1e-4)
        residual = max(np.max(abs(B[:, r:k]-candidate[:, r:k])),
                       np.max(abs(np.diag(B)-np.diag(candidate))))
        if residual < max(1.5e-6, 0.00015*scale):
            delta = candidate
            break

    if delta is None or not np.isfinite(delta).all():
        return None
    grad = (steps[1]-steps[0])/h - 0.5*h*np.diag(delta) - a*(H[1]+H[0])[:, 0]
    if (np.linalg.norm(U@grad-dif) > 0.02*np.linalg.norm(dif) or
            np.linalg.norm(delta) > 200*np.linalg.norm(dif)):
        return None
    return Quadratic.centered(0.5*U@delta@U.T, U@grad, p), p, U, H


class Pieces:
    def __init__(self, qs, kind, sign=1):
        self.qs, self.kind, self.sign = qs, kind, sign

    def __call__(self, z):
        q = [p(z) for p in self.qs]
        if self.kind == "sum":
            return sum(s*np.maximum(0, s*u) for u, s in zip(q, self.sign))
        return self.sign*np.maximum.reduce([np.zeros_like(q[0]), *q])

    def rotation_matrix(self):
        d = len(self.qs[0].b)
        B = np.zeros((d, d))
        for q in self.qs:
            B += (4*q.A@q.A + np.outer(q.b, q.b))
        return B

    def canonicalize(self,z,y):
        """Choose the polynomial origin that leaves the smallest pilot remainder."""
        variants=[self]
        if self.kind == "sum":
            for ss in itertools.product([1.,-1.],repeat=len(self.qs)):
                variants.append(Pieces([q.scaled(s) for q,s in zip(self.qs,ss)],"sum",self.sign))
        else:
            p=[Quadratic(np.zeros_like(self.qs[0].A),np.zeros_like(self.qs[0].b),0.),*self.qs]
            for j in range(1,len(p)):
                pp=[Quadratic(q.A-p[j].A,q.b-p[j].b,q.c-p[j].c) for i,q in enumerate(p) if i!=j]
                variants.append(Pieces(pp,"max",self.sign))
        return min(variants,key=lambda model: np.std(y-model(z)))

    def integral(self, rng, power=17):
        if self.kind == "sum":
            return sum(s*quadratic_positive(q.scaled(s)) for q, s in zip(self.qs,self.sign))
        # The individual positive parts have known integrals; condition the overlap.
        qs = self.qs
        d = len(qs[0].b)
        exact = sum(quadratic_positive(q) for q in qs)
        basis = ordered_basis(self.rotation_matrix())
        v = basis[:, 0]
        a = [v @ q.A @ v for q in qs]
        ds = [2*q.A@v for q in qs]
        total, n = 0.0, 0
        for _ in range(4):
            z = normals(rng, power-2, d-1) @ basis[:, 1:].T
            b = [z @ j + q.b @ v for j, q in zip(ds, qs)]
            c = [q(z) for q in qs]
            overlap = conditional_overlap(a, b, c)
            total += overlap.sum()
            n += len(z)
        return self.sign*(exact-total/n)


def _roots(a, b, c):
    if abs(a) < 1e-12:
        ok = abs(b) > 1e-12
        return np.where(ok, -c/np.where(ok, b, 1), 12), np.full_like(b, 12.)
    disc = b*b-4*a*c
    x = -0.5*(b+np.copysign(np.sqrt(np.maximum(0,disc)), b))
    return np.where(disc > 0, x/a, 12.), np.where((disc > 0) & (abs(x)>1e-30),
                                                  c/np.where(abs(x)>1e-30,x,1),12.)


def conditional_overlap(a, b, c):
    """Integrate min(q1_+, q2_+) along a standard normal coordinate."""
    r = [*_roots(a[0],b[0],c[0]), *_roots(a[1],b[1],c[1]),
         *_roots(a[0]-a[1],b[0]-b[1],c[0]-c[1])]
    t = np.sort(np.clip(np.stack(r,axis=1), -12, 12), axis=1)
    t = np.c_[np.full(len(t),-12.), t, np.full(len(t),12.)]
    mid = 0.5*(t[:, 1:]+t[:, :-1])
    q1 = c[0][:, None]+mid*(b[0][:, None]+a[0]*mid)
    q2 = c[1][:, None]+mid*(b[1][:, None]+a[1]*mid)
    pick = (q1 < q2)
    good = (q1 > 0) & (q2 > 0)
    phi = np.exp(-0.5*t*t)/np.sqrt(2*np.pi)
    m0 = np.diff(ndtr(t),axis=1)
    m1 = -np.diff(phi,axis=1)
    m2 = m0 - np.diff(t*phi,axis=1)
    return np.sum(good*(np.where(pick,c[0][:,None],c[1][:,None])*m0 +
                        np.where(pick,b[0][:,None],b[1][:,None])*m1 +
                        np.where(pick,a[0],a[1])*m2),axis=1)


def fit_kinks(session, atlas):
    reserve = 4900
    z = atlas.z.reshape(-1, session.d)
    found, controls, costs = [], [], []
    residual = atlas.y.copy()
    baseline = best = atlas.score(residual)
    best_model, best_patch = None, None
    for stage in range(3):
        if session.remaining <= reserve + 8*session.d:
            break
        attempts = 0
        previous = best
        for i, j in atlas.candidates(residual):
            if attempts >= 13:
                break
            attempts += 1
            info = crossing(session, atlas, i, j)
            if info is None:
                continue
            p, v, slope = info
            if any(abs(q(p)) < 3e-4*max(abs(slope),.001) and
                   abs(abs((2*q.A@p+q.b)@v)-abs(slope)) < .015*abs(slope)
                   for q in found):
                continue
            result = local_quadratic(session, info, reserve)
            if result is None:
                continue
            q, p, U, H = result
            found.append(q)
            controls.append(1 if slope > 0 else -1)
            costs.append(result)
            debug("kink", len(found), "after", session.used, "slope", slope)
            break
        else:
            break
        if not found or len(found) <= stage:
            break
        models = [Pieces(found.copy(), "sum", controls.copy())]
        if len(found) == 2:
            for s, t, u in itertools.product([1., -1.], repeat=3):
                models.append(Pieces([found[0].scaled(s), found[1].scaled(t)], "max", u))
        for model in models:
            r = atlas.y-model(z).reshape(atlas.y.shape)
            j = atlas.score(r)
            if j < best:
                best, best_model, residual = j, model, r
                best_patch = costs[0]
        debug("atlas residual ratio", best/max(baseline, 1e-100))
        if best < 0.0007*baseline or best == previous and len(found) >= 2:
            break
    if best > 0.35*baseline:
        return None, None
    return best_model, best_patch
