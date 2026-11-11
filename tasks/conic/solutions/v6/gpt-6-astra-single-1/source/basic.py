"""Small exact cases and immediate recession/Farkas directions."""

import math

import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm, smat, svec


def solve(p):
    if norm(p.q) == 0:
        r = {"status": "solved", "x": np.zeros(p.n), "s": p.b.copy(), "z": np.zeros(p.m)}
        if acceptable(p, r):
            return r
    free = (np.diff(p.A.indptr) == 0) & (np.diff(p.P.indptr) == 0) & (p.q != 0)
    if np.any(free):
        j = np.flatnonzero(free)[0]
        x = np.zeros(p.n)
        x[j] = -np.sign(p.q[j])
        r = {"status": "dual_infeasible", "x": x, "s": np.zeros(p.m)}
        if acceptable(p, r):
            return r
    if p.m == 0:
        if p.P.nnz:
            x = sla.lsmr(p.P, -p.q, atol=1e-14, btol=1e-14, maxiter=4000)[0]
        else:
            x = np.zeros(p.n)
        r = {"status": "solved", "x": x, "s": np.zeros(0), "z": np.zeros(0)}
        if acceptable(p, r):
            return r
        residual = p.P @ x + p.q
        r = {"status": "dual_infeasible", "x": -residual, "s": np.zeros(0)}
        if acceptable(p, r):
            return r
        return None
    empty = np.bincount(p.A.indices, minlength=p.m) == 0
    for kind, dim, sl in blocks(p.cones):
        if kind in ("zero", "nonnegative"):
            mask = empty[sl] & ((p.b[sl] != 0) if kind == "zero" else (p.b[sl] < 0))
            if not np.any(mask):
                continue
            candidates = np.flatnonzero(mask)
            j = candidates[np.argmax(np.abs(p.b[sl][candidates]))] + sl.start
            z = np.zeros(p.m)
            z[j] = -np.sign(p.b[j]) if kind == "zero" else 1.0
        elif np.all(empty[sl]):
            v = p.b[sl]
            z = np.zeros(p.m)
            if kind == "soc":
                length = np.linalg.norm(v[1:])
                if v[0] >= length:
                    continue
                z[sl.start] = 1
                if length:
                    z[sl.start+1:sl.stop] = -v[1:] / length
            elif kind == "psd":
                ev, U = linalg.eigh(smat(v, dim), subset_by_index=[0, 0], check_finite=False)
                if ev[0] >= 0:
                    continue
                z[sl] = svec(np.outer(U[:, 0], U[:, 0]))
            else:
                a, b, c = v
                if b < 0:
                    z[sl.start+1] = 1
                elif c < 0:
                    z[sl.start+2] = 1
                else:
                    if b > 0 and c > 0:
                        r = math.log(c) - math.log(b)
                    elif b > 0:
                        r = np.clip(a / b, -100, 100)
                    elif a > 0:
                        r = max(1.0, math.log(max(1.0, c / a)) + 2)
                    else:
                        continue
                    w = np.array([-1.0, r - 1, math.exp(float(np.clip(-r, -700, 700)))])
                    z[sl] = w / norm(w)
        else:
            continue
        r = {"status": "primal_infeasible", "z": z}
        if acceptable(p, r):
            return r
    return None
