"""Refine semidefinite Farkas separators by affine/cone projections."""
import os
import sys
import time

import numpy as np
from scipy import linalg, sparse

from solver import norm, smat, svec, quality


def refine_primal(p, z, seconds=50.0):
    if p.n > 1500 or p.n * p.m > 22000000:
        return None
    if any(k == "exponential" for k, _, _, _ in p.blocks):
        return None
    bn, aa = norm(p.b), norm(p.A.data) or 1.0
    if bn == 0 or not np.isfinite(z).all() or norm(z) == 0:
        return None
    z = z / norm(z)
    beta = -(p.b / bn) @ z
    if beta <= 1e-7 or norm(p.A.T @ z) / aa / beta > 0.1:
        return None
    z = z / beta
    C = sparse.vstack((p.A.T / aa, sparse.csr_matrix((p.b / bn)[None, :])),
                      format="csr")
    target = np.r_[np.zeros(p.n), -1.0]
    started = time.monotonic()
    try:
        rownorm = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
        scale = 1.0 / np.maximum(rownorm, 1e-10)
        CT = C.multiply(scale[:, None]).T.toarray()
        Q, R, perm = linalg.qr(CT, mode="economic", pivoting=True, check_finite=False)
        rank = int(np.count_nonzero(np.abs(np.diag(R)) >
                                    1e-12 * max(1.0, norm(R))))
        if not rank:
            return None
        Q = np.ascontiguousarray(Q[:, :rank])
        wanted = (target * scale)[perm]
        coeff = linalg.solve_triangular(R[:rank, :rank].T, wanted[:rank],
                                         lower=True, check_finite=False)
        particular = Q @ coeff
        if norm(C @ particular - target) > 5e-8:
            return None
    except (ValueError, np.linalg.LinAlgError, MemoryError):
        return None

    def affine(v):
        return v - Q @ (Q.T @ v) + particular

    def cone(v):
        out = v.copy()
        for k, d, a, b in p.blocks:
            if k == "nonnegative":
                out[a:b] = np.maximum(v[a:b], 0.0)
            elif k == "soc":
                vv = v[a:b]
                radius = np.linalg.norm(vv[1:])
                if radius <= -vv[0]:
                    out[a:b] = 0
                elif radius > vv[0]:
                    out[a] = 0.5 * (radius + vv[0])
                    out[a+1:b] = out[a] * vv[1:] / radius
            elif k == "psd":
                w, U = linalg.eigh(smat(v[a:b], d), check_finite=False)
                out[a:b] = svec((U * np.maximum(w, 0.0)) @ U.T)
        return out

    w = z.copy()
    best = np.inf
    for it in range(6000):
        if time.monotonic() - started > seconds:
            break
        v = cone(w)
        u = affine(2 * v - w)
        w += 1.6 * (u - v)
        if it % 5 == 0:
            err = norm(C @ v - target)
            if err < best:
                best = err
            if err < 9e-9:
                r = {"status": "primal_infeasible", "z": v}
                if quality(p, r):
                    return r
            if os.environ.get("SOLVER_DEBUG") and it % 250 == 0:
                print("certificate projection", it, err, best, file=sys.stderr)
    return None
