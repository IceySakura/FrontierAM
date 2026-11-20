"""Outer approximation for compact conic LPs with difficult cone faces."""

import math
import os
import sys
import time

import numpy as np
from scipy import sparse


class Rows:
    def __init__(self, m):
        self.m = m
        self.p = [0]
        self.i = []
        self.v = []

    def add(self, offset, value):
        for j, val in enumerate(value):
            if val != 0:
                self.i.append(offset + j)
                self.v.append(val)
        self.p.append(len(self.i))

    def addone(self, offset):
        self.i.append(offset)
        self.v.append(1.)
        self.p.append(len(self.i))

    def make(self):
        return sparse.csr_matrix((np.array(self.v, dtype=np.float64),
                                  np.array(self.i, dtype=np.int32),
                                  np.array(self.p, dtype=np.int32)),
                                 shape=(len(self.p)-1, self.m))


def exp_normal(t):
    t = float(np.clip(t, -680., 680.))
    a = math.exp(-t)
    scale = max(1., abs(t - 1.), a)
    return (-1. / scale, (t - 1.) / scale, a / scale)


def blocks(model):
    off = 0
    lin, nonlinear = [], []
    for k, d in zip(model.kinds, model.dims):
        k, d = int(k), int(d)
        w = d * (d + 1) // 2 if k == 3 else d
        if k <= 1:
            lin.append((k, off, w))
        else:
            ids = None if k != 3 else np.tril_indices(d)
            nonlinear.append((k, off, w, d, ids))
        off += w
    return lin, nonlinear


def seed_rows(model, lin, nonlin):
    r = Rows(model.m)
    eq = []
    for k, off, w in lin:
        for j in range(w):
            r.addone(off+j)
        eq.extend([k == 0] * w)
    for k, off, w, d, ids in nonlin:
        if k == 4:
            r.addone(off+1)
            r.addone(off+2)
            for t in (-8., -3., 0., 3., 8.):
                r.add(off, exp_normal(t))
            b = model.b[off:off+3]
            if b[1] > 0 and b[2] > 0:
                r.add(off, exp_normal(math.log(b[2])-math.log(b[1])))
        elif k == 2:
            r.addone(off)
            if d <= 18:
                v = np.zeros(d)
                v[0] = 1.
                for j in range(1, d):
                    v[j] = 1.
                    r.add(off, v)
                    v[j] = -1.
                    r.add(off, v)
                    v[j] = 0.
        else:
            for j in range(d):
                r.addone(off+j*(j+3)//2)
    eq.extend([False] * (len(r.p)-1-len(eq)))
    return r.make(), np.asarray(eq)


def violated_rows(model, nonlin, s, known, threshold):
    r = Rows(model.m)
    worst = 0.

    def take(off, v, key):
        if key not in known:
            r.add(off, v)
            known.add(key)

    for k, off, w, d, ids in nonlin:
        v = s[off:off+w]
        scale = max(1., np.max(np.abs(v), initial=0.))
        u = v/scale
        if k == 2:
            length = float(np.sqrt(u[1:] @ u[1:]))
            gap = length-u[0]
            worst = max(worst, gap)
            if gap > threshold and length > 0.:
                t = -u[1:]/length
                take(off, np.r_[1., t], (off, np.round(t, 12).tobytes()))
        elif k == 4:
            a, b, c = u
            closure = max(abs(b), a, -c, 0.)
            if b > 0 and c > 0:
                t = math.log(c)-math.log(b)
                gap = min(closure, a-b*t)
            elif b > 0:
                gap = closure
                t = min(-26., a/b-1.)
            elif a > 0:
                gap = closure
                t = max(10., math.log(max(1e-300, c))-math.log(a)+math.log(4.))
            else:
                gap = closure
                t = 0.
            worst = max(worst, gap)
            if gap > threshold:
                val = exp_normal(t)
                take(off, val, (off, round(max(-680., min(680., t)), 10)))
                if b < 0:
                    take(off, (0., 1., 0.), (off, "second"))
                if c < 0:
                    take(off, (0., 0., 1.), (off, "third"))
        else:
            i, j = ids
            M = np.empty((d, d))
            z = u*np.where(i == j, 1., 1./math.sqrt(2.))
            M[i,j] = M[j,i] = z
            eig, vect = np.linalg.eigh(M)
            worst = max(worst, -eig[0])
            for a in range(min(5, d)):
                if eig[a] >= -threshold:
                    break
                g = vect[:,a]
                v = g[i]*g[j]*np.where(i == j, 1., math.sqrt(2.))
                take(off, v, (off, np.round(v, 12).tobytes()))
    return r.make(), worst


def optimal_outer(model, budget, highs_new, highs_model, highs_hessian=None):
    start = time.monotonic()
    until = start+budget
    n, m = model.n, model.m
    if n == 0 or n+m > 58000 or model.a[2].size > 650000:
        return 0
    x, s, z = model.results
    lp = model.p[2].size == 0
    lin, nonlin = blocks(model)
    if not nonlin or any(k == 3 and d > 22 for k, _, _, d, _ in nonlin):
        return 0
    a = sparse.csc_matrix((model.a[2], model.a[1], model.a[0]), shape=(m, n))
    csr = a.tocsr()
    cscale = np.ones(n)
    nz = model.a[0][:-1] < model.a[0][1:]
    cscale[nz] = np.maximum.reduceat(np.abs(model.a[2]), model.a[0][:-1][nz])
    global_b = max(1., np.max(np.abs(model.b), initial=0.))
    bounds = np.clip(2e4*global_b/(np.maximum(1e-10, cscale)),1e3,1e20)
    initial_bounds = bounds.copy()
    first, eq = seed_rows(model, lin, nonlin)
    if first.shape[0] > 140000:
        return 0
    seed_a = first @ csr
    seed_a.sum_duplicates()
    seed_b = first @ model.b

    def open_lp(X, b, bound):
        h = highs_new(max(.05, until-time.monotonic()), "simplex" if lp else None, 1e-10)
        A = X.tocsc()
        if int(highs_model(h, model.q, -bound, bound, np.where(eq,b,-np.inf), b,
                           (A.indptr,A.indices,A.data))) < 0:
            return None
        if not lp:
            if highs_hessian is None or int(highs_hessian(h, model)) < 0:
                return None
        return h

    h = open_lp(seed_a, seed_b, bounds)
    if h is None:
        return 0
    transforms = [first]
    asparse = [seed_a]
    rhs = [seed_b]
    known = set()
    direction = False
    unboxed = False
    ids = np.arange(n, dtype=np.int32)

    def lift(y):
        z.fill(0.)
        off=0
        for t in transforms:
            nr = t.shape[0]
            z[:] += t.T @ y[off:off+nr]
            off += nr

    def certificate():
        if time.monotonic() > until:
            return 0
        R = sparse.vstack(asparse, format="csc")
        b = np.concatenate(rhs)
        scale = np.max(np.abs(b), initial=0.)
        if scale == 0:
            return 0
        signs = np.zeros(b.size)
        signs[:eq.size] = -eq.astype(np.float64)
        h2 = highs_new(max(.07, until-time.monotonic()), "simplex", 1e-10)
        t = np.zeros(n)
        if int(highs_model(h2, b/scale, signs, np.ones(b.size), t, t,
                     (R.indptr, R.indices, R.data/(np.max(np.abs(R.data), initial=0.) or 1.)), True)) < 0:
            return 0
        h2.run()
        sol = h2.getSolution()
        if sol.value_valid:
            lift(np.asarray(sol.col_value))
            return model.check(2)
        return 0

    for it in range(80):
        left = until-time.monotonic()
        if left <= .012:
            break
        h.setOptionValue("time_limit", max(.02, left))
        h.run()
        stat = int(h.getModelStatus())
        if stat == 8 and not direction:
            if not unboxed:
                h.changeColsBounds(n, ids, np.full(n, -np.inf), np.full(n, np.inf))
                unboxed = True
                continue
            return certificate()
        if stat not in (6, 7, 13, 14, 10):
            break
        if stat == 10 and lp and not direction:
            direction = True
            R = sparse.vstack(asparse, format="csc")
            bounds = np.minimum(initial_bounds/(2e4*global_b),1e9)
            eq = np.pad(eq, (0,R.shape[0]-eq.size))
            h = open_lp(R, np.zeros(R.shape[0]), bounds)
            if h is None:
                return 0
            continue
        vals = h.getSolution()
        if not vals.value_valid:
            break
        x[:] = vals.col_value
        s[:] = -(csr @ x)
        if not direction:
            s[:] += model.b
        cut, defect = violated_rows(model, nonlin, s, known, 3.5e-9)
        if defect < 1.05e-8 or cut.shape[0] == 0:
            if direction:
                if model.check(3):
                    return 3
            elif vals.dual_valid:
                lift(-np.asarray(vals.row_dual))
                if model.check(17):
                    return 1
                maxbound = np.max(np.abs(x)/bounds, initial=0.)
                if not unboxed and maxbound > .98:
                    if lp and model.check(3):
                        return 3
                    bounds *= 20
                    if np.max(bounds, initial=0.) > 1e23:
                        h.changeColsBounds(n, ids, np.full(n, -np.inf), np.full(n, np.inf))
                        unboxed = True
                    else:
                        h.changeColsBounds(n, ids, -bounds, bounds)
                    continue
        if not cut.shape[0]:
            break
        M = cut @ csr
        B = cut @ model.b
        transforms.append(cut)
        asparse.append(M)
        rhs.append(B)
        nr = M.shape[0]
        ub = np.zeros(nr) if direction else B
        code = h.addRows(nr, np.full(nr,-np.inf), ub, M.nnz, M.indptr, M.indices, M.data)
        if int(code)<0:
            break
        if os.environ.get("CONIC_TRACE"):
            print(f"outer {it}: status {stat}, residual {defect:.3g}, {nr} rows",file=sys.stderr)
    return 0
