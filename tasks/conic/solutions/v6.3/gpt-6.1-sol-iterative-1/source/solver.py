#!/usr/bin/env python3
"""Offline primal-dual solver for sparse quadratic conic programs."""

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
if os.environ.get("SOLVER_SCIPY"):
    from scipy import sparse as sp
    from scipy import linalg as la
    from scipy.sparse.linalg import splu
else:
    import fastsparse as sp
    import fastla as la
    from fastsparse import splu

warnings.filterwarnings("ignore", category=RuntimeWarning)
SQRT2 = math.sqrt(2.0)
VERBOSE = bool(os.environ.get("SOLVER_VERBOSE"))
PACKING = {}


def log(*args):
    if VERBOSE:
        print(*args, file=sys.stderr, flush=True)


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def load(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        mats = []
        for k in ("P", "A"):
            a = sp.csc_matrix((d[k + "_data"], d[k + "_indices"],
                               d[k + "_indptr"]), shape=tuple(d[k + "_shape"]))
            a.sum_duplicates()
            a.eliminate_zeros()
            a.sort_indices()
            mats.append(a)
        q, b = d["q"].copy(), d["b"].copy()
    return mats[0], q, mats[1], b, meta["cones"]


def save(path, result):
    meta = {"schema_version": 1, "status": result["status"]}
    arrays = {k: np.asarray(result[k], dtype=np.float64)
              for k in ("x", "s", "z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(meta, allow_nan=False))
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def packing(d):
    if d not in PACKING:
        c, r = np.tril_indices(d)
        PACKING[d] = r, c, np.where(r == c, 1.0, SQRT2)
    return PACKING[d]


def smat(v, d):
    a = np.zeros((d, d), dtype=float)
    r, c, scale = packing(d)
    a[r, c] = v / scale
    a[c, r] = a[r, c]
    return a


def svec(a):
    r, c, scale = packing(a.shape[0])
    return a[r, c] * scale


class Cones:
    def __init__(self, cones, m):
        self.m = m
        self.blocks = []
        eq, pos, exp = [], [], []
        soc = {}
        self.psd = []
        off = 0
        self.degree = 0
        for c in cones:
            kind, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if kind == "psd" else d
            rows = np.arange(off, off + w, dtype=np.int32)
            self.blocks.append((kind, d, rows))
            if kind == "zero":
                eq.append(rows)
            elif kind == "nonnegative":
                pos.append(rows)
                self.degree += d
            elif kind == "soc":
                soc.setdefault(d, []).append(rows)
                self.degree += 1
            elif kind == "exponential":
                exp.append(rows)
                self.degree += 3
            elif kind == "psd":
                self.psd.append((d, rows))
                self.degree += d
            off += w
        self.eq = np.concatenate(eq) if eq else np.empty(0, dtype=np.int32)
        self.pos = np.concatenate(pos) if pos else np.empty(0, dtype=np.int32)
        self.exp = np.array(exp, dtype=np.int32).reshape(-1, 3)
        self.soc = {d: np.array(rows, dtype=np.int32)
                    for d, rows in soc.items()}
        self.neq = len(self.eq)
        if len(self.eq):
            active = np.ones(m, dtype=bool)
            active[self.eq] = False
            self.active = np.flatnonzero(active).astype(np.int32)
        else:
            self.active = np.arange(m, dtype=np.int32)

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.pos] = z[self.pos] = 1.0
        for rows in self.soc.values():
            s[rows[:, 0]] = z[rows[:, 0]] = 1.0
        if self.exp.size:
            s[self.exp] = np.array([-1.0, 1.0, 2.0])
            z[self.exp] = np.array([-1.0, 0.0, 1.0])
        for d, rows in self.psd:
            s[rows] = z[rows] = svec(np.eye(d))
        return s, z

    def defect(self, v, dual=False):
        worst = 0.0
        if not dual:
            worst = norm(v[self.eq])
            if worst > 0:
                worst = min(1.0, worst)
        for kind, d, rows in self.blocks:
            if kind == "zero":
                continue
            u = v[rows]
            u = u / max(1.0, norm(u))
            if kind == "nonnegative":
                err = max(0.0, -float(np.min(u)))
            elif kind == "soc":
                err = max(0.0, float(np.linalg.norm(u[1:])) - u[0])
            elif kind == "exponential":
                a, b, c = (-u[1], -u[0], math.e * u[2]) if dual else u
                boundary = max(abs(b), max(a, 0.0), max(-c, 0.0))
                if b > 0 and c > 0:
                    err = min(boundary, max(0.0, a - b * (math.log(c) - math.log(b))))
                else:
                    err = boundary
            else:
                err = max(0.0, -float(la.eigh(smat(u, d), eigvals_only=True,
                                   subset_by_index=[0, 0], check_finite=False)[0]))
            worst = max(worst, err)
        return worst

    def max_step(self, s, ds, dual=False, cap=1.0):
        alpha = float(cap)
        p, dp = s[self.pos], ds[self.pos]
        mask = dp < 0
        if np.any(mask):
            alpha = min(alpha, float(np.min(-p[mask] / dp[mask])))
        for rows in self.soc.values():
            a, b = s[rows], ds[rows]
            aa = b[:, 0] ** 2 - np.sum(b[:, 1:] ** 2, axis=1)
            bb = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
            cc = a[:, 0] ** 2 - np.sum(a[:, 1:] ** 2, axis=1)
            disc = np.maximum(0.0, bb * bb - aa * cc)
            root = np.sqrt(disc)
            denom = -bb + root
            # The smaller positive root, using a cancellation-free form.
            val = np.where(denom > 0, cc / denom, np.inf)
            mask = (aa < 0) | (bb < 0)
            if np.any(mask):
                alpha = min(alpha, float(np.min(val[mask])))
            mask = b[:, 0] < 0
            if np.any(mask):
                alpha = min(alpha, float(np.min(-a[mask, 0] / b[mask, 0])))
        for d, rows in self.psd:
            S, D = smat(s[rows], d), smat(ds[rows], d)
            L = la.cholesky(S, lower=True, check_finite=False)
            Q = la.solve_triangular(L, D, lower=True, check_finite=False)
            Q = la.solve_triangular(L, Q.T, lower=True, check_finite=False).T
            mineig = la.eigh(Q, eigvals_only=True, subset_by_index=[0, 0],
                             check_finite=False)[0]
            if mineig < 0:
                alpha = min(alpha, -1.0 / mineig)
        if self.exp.size:
            V = s[self.exp].copy()
            D = ds[self.exp].copy()
            if dual:
                V = np.stack((-V[:, 1], -V[:, 0], math.e * V[:, 2]), axis=1)
                D = np.stack((-D[:, 1], -D[:, 0], math.e * D[:, 2]), axis=1)
            for k in (1, 2):
                mask = D[:, k] < 0
                if np.any(mask):
                    alpha = min(alpha, float(np.min(-V[mask, k] / D[mask, k])))
            def feasible(t):
                T = V + t * D
                y, z = T[:, 1], T[:, 2]
                return bool(np.all(y > 0) and np.all(z > 0) and
                            np.all(y * (np.log(z) - np.log(y)) - T[:, 0] > 0))
            if not feasible(alpha):
                lo, hi = 0.0, alpha
                fine = os.environ.get("EXP_STEP_FINE", "0") == "1"
                for _ in range(32 if fine else 40):
                    mid = 0.5 * (lo + hi)
                    if feasible(mid):
                        lo = mid
                    else:
                        hi = mid
                    if not fine and hi - lo <= 1e-4 * max(lo, 1e-16):
                        break
                alpha = lo
        return max(0.0, min(float(cap), alpha))


def exp_barrier(V):
    """Gradient and Hessian of the degree-three exponential-cone barrier."""
    x, y, z = V.T
    ell = np.log(z) - np.log(y)
    r = y * ell - x
    t = np.stack((-np.ones_like(x), ell - 1.0, y / z), axis=1)
    g = -t / r[:, None]
    g[:, 1] -= 1.0 / y
    g[:, 2] -= 1.0 / z
    H = np.einsum("bi,bj->bij", t, t) / r[:, None, None] ** 2
    H[:, 1, 1] += 1.0 / (r * y) + 1.0 / y ** 2
    H[:, 1, 2] -= 1.0 / (r * z)
    H[:, 2, 1] -= 1.0 / (r * z)
    H[:, 2, 2] += y / (r * z ** 2) + 1.0 / z ** 2
    return g, H


def exp_third(V, D):
    x, y, z = V.T
    dx, dy, dz = D.T
    ell = np.log(z) - np.log(y)
    r = y * ell - x
    t = np.stack((-np.ones_like(x), ell - 1.0, y / z), axis=1)
    dt = np.stack((np.zeros_like(x), -dy / y + dz / z,
                   dy / z - y * dz / z ** 2), axis=1)
    ddt = np.stack((np.zeros_like(x), dy ** 2 / y ** 2 - dz ** 2 / z ** 2,
                    -2 * dy * dz / z ** 2 + 2 * y * dz ** 2 / z ** 3), axis=1)
    dr = np.sum(t * D, axis=1)
    ddr = np.sum(dt * D, axis=1)
    v = (-ddt / r[:, None] + 2 * dt * (dr / r ** 2)[:, None] +
         t * (ddr / r ** 2 - 2 * dr ** 2 / r ** 3)[:, None])
    v[:, 1] -= 2 * dy ** 2 / y ** 3
    v[:, 2] -= 2 * dz ** 2 / z ** 3
    return v


def exp_geometry(Z, mu=1.0):
    T = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, math.e]])
    V = Z.astype(np.longdouble) @ T.astype(np.longdouble).T
    x, y, z = V.T
    ell = np.log(z) - np.log(y)
    r = y * ell - x
    t = np.stack((-np.ones_like(x), ell - 1.0, y / z), axis=1)
    g = -t / r[:, None]
    g[:, 1] -= 1 / y
    g[:, 2] -= 1 / z
    F = np.zeros((len(Z), 3, 4), dtype=np.longdouble)
    F[:, :, 0] = t / r[:, None]
    F[:, 1, 1] = 1 / np.sqrt(r * y)
    F[:, 2, 1] = -(y / z) / np.sqrt(r * y)
    F[:, 1, 2] = 1 / y
    F[:, 2, 3] = 1 / z
    F = np.asarray(np.sqrt(mu) * (T.T[None] @ F), dtype=float)
    _, R = np.linalg.qr(F.transpose(0, 2, 1), mode="reduced")
    return np.asarray(g @ T, dtype=float), R.transpose(0, 2, 1)


def exp_centrality(cones, s, z, mu):
    Z = z[cones.exp]
    V = np.stack((-Z[:, 1], -Z[:, 0], math.e * Z[:, 2]), axis=1)
    if np.any(V[:, 1:] <= 0):
        return np.inf
    if np.any(V[:, 1] * (np.log(V[:, 2]) - np.log(V[:, 1])) - V[:, 0] <= 0):
        return np.inf
    g, L = exp_geometry(Z)
    r = s[cones.exp] / mu + g
    w = np.linalg.solve(L, r[:, :, None])[:, :, 0]
    return float(np.max(np.linalg.norm(w, axis=1)))


class Scaling:
    def __init__(self, cones, s, z, mu):
        self.c = cones
        self.s, self.z, self.mu = s, z, mu
        self.socs, self.large_socs, self.psds = [], [], []
        p = cones.pos
        self.diag = s[p] / z[p]
        self.expH = None
        self.expg = None
        self.expR = None
        for d, rows in cones.soc.items():
            S, Z = s[rows], z[rows]
            sd = np.sqrt(np.maximum(S[:, 0] ** 2 - np.sum(S[:, 1:] ** 2, axis=1), 1e-300))
            zd = np.sqrt(np.maximum(Z[:, 0] ** 2 - np.sum(Z[:, 1:] ** 2, axis=1), 1e-300))
            beta = np.sqrt(sd / zd)
            a = S / sd[:, None]
            b = Z / zd[:, None]
            b[:, 1:] *= -1
            v = (a + b) / np.sqrt(2.0 * (1.0 + np.sum(S * Z, axis=1) / (sd * zd)))[:, None]
            root = np.sqrt(2 * (v[:, 0] + 1.0))
            v[:, 0] = (v[:, 0] + 1.0) / root
            v[:, 1:] /= root[:, None]
            if d > 64:
                lam = self.soc_mult(v, beta, Z)
                self.large_socs.append((rows, beta, v, lam))
                continue
            J = np.ones(d)
            J[0] = -1
            W = beta[:, None, None] * (2 * np.einsum("bi,bj->bij", v, v) + np.diag(J)[None, :, :])
            Wi = W.copy() / beta[:, None, None] ** 2
            Wi[:, 0, 1:] *= -1
            Wi[:, 1:, 0] *= -1
            H = W @ W
            lam = np.einsum("bij,bj->bi", W, Z)
            self.socs.append((rows, W, Wi, H, lam))
        if cones.exp.size:
            Z = z[cones.exp]
            self.expg, L = exp_geometry(Z, mu)
            S = s[cones.exp]
            v = np.asarray(np.einsum("bji,bj->bi", L.astype(np.longdouble),
                                    Z.astype(np.longdouble)), dtype=float)
            v /= np.linalg.norm(v, axis=1)[:, None]
            axes = np.eye(3)[np.argmin(np.abs(v), axis=1)]
            u = np.cross(v, axes)
            u /= np.linalg.norm(u, axis=1)[:, None]
            w = np.cross(v, u)
            sz = np.sum(S * Z, axis=1)
            F = np.stack((np.einsum("bij,bj->bi", L, u),
                          np.einsum("bij,bj->bi", L, w), S / np.sqrt(sz)[:, None]), axis=2)
            _, R = np.linalg.qr(F.transpose(0, 2, 1), mode="reduced")
            L = R.transpose(0, 2, 1)
            self.expH = L @ L.transpose(0, 2, 1)
            self.expR = np.linalg.inv(L)
        for d, rows in cones.psd:
            S, Z = smat(s[rows], d), smat(z[rows], d)
            L = la.cholesky(Z, lower=True, check_finite=False)
            vals, U = la.eigh(L.T @ S @ L, check_finite=False)
            if np.min(vals) <= 0:
                raise la.LinAlgError("noninterior PSD scaling")
            lam = np.sqrt(vals)
            r = la.solve_triangular(L.T, U * np.sqrt(lam), lower=False, check_finite=False)
            ri = (U.T @ L.T) / np.sqrt(lam)[:, None]
            T, Ti = r @ r.T, ri.T @ ri
            self.psds.append((d, rows, r, ri, T, Ti, lam))

    @staticmethod
    def soc_mult(v, beta, x, inv=False):
        u = v.copy()
        if inv:
            u[:, 1:] *= -1
        if x.ndim == 2:
            jx = x.copy()
            jx[:, 1:] *= -1
            out = 2 * u * np.sum(u * x, axis=1)[:, None] - jx
            return out / beta[:, None] if inv else out * beta[:, None]
        jx = x.copy()
        jx[:, 1:] *= -1
        out = 2 * u[:, :, None] * np.einsum("bi,bij->bj", u, x)[:, None, :] - jx
        return out / beta[:, None, None] if inv else out * beta[:, None, None]

    @staticmethod
    def jordan(a, b):
        out = a[:, :1] * b + b[:, :1] * a
        out[:, 0] = np.sum(a * b, axis=1)
        return out

    @staticmethod
    def jordan_inv(lam, v):
        det = lam[:, 0] ** 2 - np.sum(lam[:, 1:] ** 2, axis=1)
        out = np.empty_like(v)
        out[:, 0] = (lam[:, 0] * v[:, 0] - np.sum(lam[:, 1:] * v[:, 1:], axis=1)) / det
        out[:, 1:] = (v[:, 1:] - out[:, :1] * lam[:, 1:]) / lam[:, :1]
        return out

    def rhs(self, sigma, ds=None, dz=None):
        s, z, c, mu = self.s, self.z, self.c, self.mu
        out = -s.copy()
        out[c.eq] = 0
        p = c.pos
        out[p] += sigma * mu / z[p]
        if ds is not None:
            out[p] -= ds[p] * dz[p] / z[p]
        for rows, W, Wi, H, lam in self.socs:
            v = np.zeros_like(lam)
            v[:, 0] = sigma * mu
            if ds is not None:
                dss = np.einsum("bij,bj->bi", Wi, ds[rows])
                dzz = np.einsum("bij,bj->bi", W, dz[rows])
                v -= self.jordan(dss, dzz)
            v = self.jordan_inv(lam, v)
            out[rows] += np.einsum("bij,bj->bi", W, v)
        for rows, beta, u, lam in self.large_socs:
            v = np.zeros_like(lam)
            v[:, 0] = sigma * mu
            if ds is not None:
                dss = self.soc_mult(u, beta, ds[rows], inv=True)
                dzz = self.soc_mult(u, beta, dz[rows])
                v -= self.jordan(dss, dzz)
            v = self.jordan_inv(lam, v)
            out[rows] += self.soc_mult(u, beta, v)
        if c.exp.size:
            out[c.exp] -= sigma * mu * self.expg
            if dz is not None:
                Z, dZ = z[c.exp], dz[c.exp]
                trans = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0],
                                  [0.0, 0.0, math.e]])
                V, dV = Z @ trans.T, dZ @ trans.T
                third = exp_third(V, dV) @ trans
                out[c.exp] += np.einsum("bij,bj->bi", self.expH, dZ) - 0.5 * mu * third
        for d, rows, r, ri, T, Ti, lam in self.psds:
            v = np.diag(sigma * mu / lam)
            if ds is not None:
                dss = ri @ smat(ds[rows], d) @ ri.T
                dzz = r.T @ smat(dz[rows], d) @ r
                v -= (dss @ dzz + dzz @ dss) / (lam[:, None] + lam[None, :])
            out[rows] += svec(r @ v @ r.T)
        return out

    def apply(self, v, inv=False):
        c = self.c
        out = np.zeros_like(v)
        p = c.pos
        if v.ndim == 2:
            out[p] = v[p] * (1 / self.diag if inv else self.diag)[:, None]
        else:
            out[p] = v[p] * (1 / self.diag if inv else self.diag)
        for rows, W, Wi, H, lam in self.socs:
            M = Wi @ Wi if inv else H
            if v.ndim == 1:
                out[rows] = np.einsum("bij,bj->bi", M, v[rows])
            else:
                out[rows] = np.einsum("bij,bjk->bik", M, v[rows])
        for rows, beta, u, lam in self.large_socs:
            a = self.soc_mult(u, beta, v[rows], inv=inv)
            out[rows] = self.soc_mult(u, beta, a, inv=inv)
        if c.exp.size:
            M = self.expR.transpose(0, 2, 1) @ self.expR if inv else self.expH
            if v.ndim == 1:
                out[c.exp] = np.einsum("bij,bj->bi", M, v[c.exp])
            else:
                out[c.exp] = M @ v[c.exp]
        for d, rows, r, ri, T, Ti, lam in self.psds:
            M = Ti if inv else T
            if v.ndim == 1:
                out[rows] = svec(M @ smat(v[rows], d) @ M)
            else:
                for j in range(v.shape[1]):
                    out[rows, j] = svec(M @ smat(v[rows, j], d) @ M)
        return out

    def sparse_matrix(self):
        c = self.c
        rows, cols, vals = [c.pos], [c.pos], [self.diag]
        for idx, W, Wi, H, lam in self.socs:
            d = idx.shape[1]
            rows.append(np.broadcast_to(idx[:, :, None], H.shape).ravel())
            cols.append(np.broadcast_to(idx[:, None, :], H.shape).ravel())
            vals.append(H.ravel())
        if c.exp.size:
            idx, H = c.exp, self.expH
            rows.append(np.broadcast_to(idx[:, :, None], H.shape).ravel())
            cols.append(np.broadcast_to(idx[:, None, :], H.shape).ravel())
            vals.append(H.ravel())
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(c.m, c.m))

    def inverse_sqrt_sparse(self):
        if hasattr(self, "_inverse_sqrt"):
            return self._inverse_sqrt
        c = self.c
        if not self.socs and not c.exp.size and not self.psds:
            diagonal = np.ones(c.m)
            diagonal[c.pos] = 1 / np.sqrt(self.diag)
            for rows, beta, u, lam in self.large_socs:
                diagonal[rows] = 1 / beta[:, None]
                diagonal[rows[:, 0]] *= -1
            self._inverse_sqrt = sp.diags(diagonal, format="csc")
            return self._inverse_sqrt
        rows = [c.pos, c.eq]
        cols = [c.pos, c.eq]
        vals = [1.0 / np.sqrt(self.diag), np.ones(len(c.eq))]
        for idx, beta, u, lam in self.large_socs:
            diagonal = np.ones_like(u) / beta[:, None]
            diagonal[:, 0] *= -1
            rows.append(idx.ravel())
            cols.append(idx.ravel())
            vals.append(diagonal.ravel())
        for idx, W, Wi, H, lam in self.socs:
            rows.append(np.broadcast_to(idx[:, :, None], Wi.shape).ravel())
            cols.append(np.broadcast_to(idx[:, None, :], Wi.shape).ravel())
            vals.append(Wi.ravel())
        if c.exp.size:
            M = self.expR
            idx = c.exp
            rows.append(np.broadcast_to(idx[:, :, None], M.shape).ravel())
            cols.append(np.broadcast_to(idx[:, None, :], M.shape).ravel())
            vals.append(M.ravel())
        self._inverse_sqrt = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(c.m, c.m))
        return self._inverse_sqrt

    def inverse_sqrt(self, v, dual=False):
        out = self.inverse_sqrt_sparse().T @ v if dual else self.inverse_sqrt_sparse() @ v
        for rows, beta, u, lam in self.large_socs:
            out[rows] = self.soc_mult(u, beta, v[rows], inv=True)
        for d, rows, r, ri, T, Ti, lam in self.psds:
            M = ri.T if dual else ri
            if v.ndim == 1:
                out[rows] = svec(M @ smat(v[rows], d) @ M.T)
            else:
                for j in range(v.shape[1]):
                    out[rows, j] = svec(M @ smat(v[rows, j], d) @ M.T)
        return out

    def quadratic(self, z):
        c = self.c
        value = float(np.sum(self.diag * z[c.pos] ** 2))
        for rows, W, Wi, H, lam in self.socs:
            v = np.einsum("bij,bj->bi", W, z[rows])
            value += float(np.sum(v * v))
        for rows, beta, u, lam in self.large_socs:
            v = self.soc_mult(u, beta, z[rows])
            value += float(np.sum(v * v))
        if c.exp.size:
            L = np.linalg.inv(self.expR)
            v = np.einsum("bji,bj->bi", L, z[c.exp])
            value += float(np.sum(v * v))
        for d, rows, r, ri, T, Ti, lam in self.psds:
            v = r.T @ smat(z[rows], d) @ r
            value += float(np.sum(v * v))
        return value


def equilibrate(P, q, A, b, cones):
    n, m = len(q), len(b)
    D, E, cost = np.ones(n), np.ones(m), 1.0
    P, A, q, b = P.copy(), A.copy(), q.copy(), b.copy()
    for _ in range(6):
        pn = np.asarray(abs(P).max(axis=0).toarray()).ravel() if P.nnz else np.zeros(n)
        an = np.asarray(abs(A).max(axis=0).toarray()).ravel() if A.nnz else np.zeros(n)
        dn = 1.0 / np.sqrt(np.clip(np.maximum(pn, an), 1e-4, 1e4))
        rn = np.asarray(abs(A).max(axis=1).toarray()).ravel() if A.nnz else np.zeros(m)
        for kind, d, rows in cones.blocks:
            if kind not in ("zero", "nonnegative"):
                rn[rows] = np.max(rn[rows])
        en = 1.0 / np.sqrt(np.clip(rn, 1e-4, 1e4))
        D *= dn
        E *= en
        A = sp.diags(en, format="csc") @ A @ sp.diags(dn, format="csc")
        P = sp.diags(dn, format="csc") @ P @ sp.diags(dn, format="csc")
        q *= dn
        b *= en
        c = 1.0 / max(1e-4, norm(q), float(np.mean(pn)) if pn.size else 0)
        c = min(1e4, c)
        P *= c
        q *= c
        cost *= c
    return P.tocsc(), q, A.tocsc(), b, D, E, cost


def solved_metrics(P, q, A, b, x, s, z, exact=False):
    if exact and hasattr(P, "matvec_sparse"):
        Ax, Px, Atz = A.matvec_sparse(x), P.matvec_sparse(x), A.T.matvec_sparse(z)
    else:
        Ax, Px, Atz = A @ x, P @ x, A.T @ z
    rp = norm(Ax + s - b) / (1 + max(norm(Ax), norm(s), norm(b)))
    rd = norm(Px + q + Atz) / (1 + max(norm(Px), norm(q), norm(Atz)))
    xpx = float(x @ Px)
    primal = 0.5 * xpx + float(q @ x)
    dual = -0.5 * xpx - float(b @ z)
    gap = abs(primal - dual) / max(1.0, abs(primal), abs(dual))
    if not all(math.isfinite(v) for v in (rp, rd, primal, dual, gap)):
        return np.inf, np.inf, np.inf
    return rp, rd, gap


def checked(P, q, A, b, cones, result, tol=8e-9):
    try:
        if any(not np.all(np.isfinite(v)) for k, v in result.items() if k in ("x", "s", "z")):
            return False
        st = result["status"]
        if st == "solved":
            x, s, z = result["x"], result["s"], result["z"]
            rp, rd, gap = solved_metrics(P, q, A, b, x, s, z, exact=True)
            if max(rp, rd, gap) > tol:
                return False
            return max(cones.defect(s), cones.defect(z, True)) <= tol
        if st == "primal_infeasible":
            z = result["z"]
            if norm(z) == 0 or norm(b) == 0:
                return False
            z = z / norm(z)
            margin = -float((b / norm(b)) @ z)
            if margin <= 1e-8:
                return False
            az = (A.T.matvec_sparse(z, scale=(norm(A.data) or 1)) if hasattr(A, "matvec_sparse")
                  else (A / (norm(A.data) or 1)).T @ z)
            return max(norm(az),
                       cones.defect(z, True)) <= tol * margin
        if st == "dual_infeasible":
            x, s = result["x"], result["s"]
            if norm(x) == 0 or norm(q) == 0:
                return False
            s, x = s / norm(x), x / norm(x)
            margin = -float((q / norm(q)) @ x)
            if margin <= 1e-8:
                return False
            scale = norm(A.data) or 1
            if hasattr(P, "matvec_sparse"):
                px = P.matvec_sparse(x)
                if norm(px) / (norm(P.data) or 1) > 1.05 * tol * margin:
                    return False
                px = P.matvec_sparse(x, scale=(norm(P.data) or 1))
                ax = A.matvec_sparse(x, scale=scale)
            else:
                px, ax = (P / (norm(P.data) or 1)) @ x, (A / scale) @ x
            return max(norm(px),
                       norm(ax + s / scale),
                       cones.defect(s / scale)) <= tol * margin
    except (ValueError, FloatingPointError, OverflowError, la.LinAlgError):
        return False
    return False


class KKTSolver:
    def __init__(self, P, A, scaling, regularization=1e-10, allow_dense=True, allow_normal=True):
        self.P, self.A, self.sc = P, A, scaling
        n, m = P.shape[0], A.shape[0]
        self.n, self.m = n, m
        refine_mode = os.environ.get("ADAPTIVE_REFINE", "auto")
        adaptive = refine_mode == "1" or (refine_mode == "auto" and n + m >= 10000)
        self.linear_tolerance = (min(1e-7, max(1e-14, 1e-3 * scaling.mu))
                                 if adaptive else 2e-15)
        self.schur = bool(scaling.psds)
        self.qr = False
        self.diagonal = False
        self.dense_box = False
        self.native = False
        self.assembly = None
        self.conic_dense = False
        self.cs_operator = False
        self.fallback = None
        self.reg = regularization
        c = scaling.c
        if (allow_normal and not c.soc and not c.exp.size and not c.psd and
                not os.environ.get("KKT_FULL") and len(c.eq) <= min(1200, n // 2)):
            if not hasattr(A, "_singleton_data"):
                pc = np.diff(P.indptr)
                pdiag = np.all(pc <= 1) and np.array_equal(P.indices, np.flatnonzero(pc))
                Ap = A[c.pos].tocsr()
                rowcounts = np.diff(Ap.indptr)
                A._bound_rows_only = bool(np.all(rowcounts <= 1))
                if pdiag and np.all(rowcounts <= 1):
                    pr = np.flatnonzero(rowcounts)
                    E = A[c.eq].tocsc()
                    denseE = (E.toarray() if len(c.eq) and n * len(c.eq) <= 20000000
                              and E.nnz > .15 * n * len(c.eq) else None)
                    A._singleton_data = (P.diagonal(), E, pr,
                                          Ap.indices.copy(), Ap.data.copy(), denseE)
                else:
                    A._singleton_data = None
            data = A._singleton_data
            if data is not None:
                pd, E, pr, cols, vals, denseE = data
                g = pd + np.bincount(cols, weights=vals ** 2 / scaling.diag[pr], minlength=n)
                if np.all(g > 0) and np.isfinite(g).all():
                    self.dg, self.deq, self.dpr, self.dcols, self.dvals = g, E, pr, cols, vals
                    try:
                        if len(c.eq):
                            if denseE is not None:
                                W = denseE / np.sqrt(g)[None, :]
                                S = W @ W.T
                                self.ds = 1 / np.sqrt(np.diag(S))
                                S *= self.ds[:, None] * self.ds[None, :]
                                self.dfactor = la.cho_factor(S, lower=True, check_finite=False)
                                self.dense_d = True
                            else:
                                S = E @ sp.diags(1 / g) @ E.T
                                self.ds = 1 / np.sqrt(S.diagonal())
                                S = sp.diags(self.ds) @ S @ sp.diags(self.ds)
                                if len(c.eq) <= 250 or S.nnz > .12 * len(c.eq) ** 2:
                                    Sd = S.toarray()
                                    try:
                                        self.dfactor = la.cho_factor(Sd, lower=True, check_finite=False)
                                    except la.LinAlgError:
                                        Sd[np.diag_indices_from(Sd)] += 1e-12
                                        self.dfactor = la.cho_factor(Sd, lower=True, check_finite=False)
                                    self.dense_d = True
                                else:
                                    self.dfactor = splu(S.tocsc())
                                    self.dense_d = False
                        self.diagonal = True
                        return
                    except (RuntimeError, ValueError, la.LinAlgError):
                        pass
        if (allow_dense and os.environ.get("DENSE_QP", "1") != "0" and
                not c.soc and not c.exp.size and not c.psd and
                450 <= n <= 5000 and len(c.eq) <= 50 and P.nnz > .12 * n * n):
            try:
                if not hasattr(A, "_dense_bound_data"):
                    Ap = A[c.pos].tocsr()
                    rc = np.diff(Ap.indptr)
                    A._dense_bound_data = ((P.toarray(), A[c.eq].tocsc(),
                                            np.flatnonzero(rc), Ap.indices.copy(), Ap.data.copy())
                                           if np.all(rc <= 1) else None)
                data = A._dense_bound_data
                if data is not None:
                    Pd, E, pr, cols, vals = data
                    g = np.bincount(cols, weights=vals ** 2 / scaling.diag[pr], minlength=n)
                    G = Pd.copy()
                    delta = max(1e-11, 1e-12 * max(norm(G), norm(g)))
                    G[np.diag_indices(n)] += g + delta
                    self.boxFactor = la.cho_factor(G, lower=True, check_finite=False)
                    if len(c.eq):
                        self.boxV = la.cho_solve(self.boxFactor, E.T.toarray(), check_finite=False)
                        S = E @ self.boxV
                        self.ds = 1 / np.sqrt(np.diag(S))
                        S *= self.ds[:, None] * self.ds[None, :]
                        self.boxEqFactor = la.cho_factor(S, lower=True, check_finite=False)
                    self.deq, self.dpr, self.dcols, self.dvals = E, pr, cols, vals
                    self.dense_box = True
                    self.diagonal = True
                    return
            except (ValueError, RuntimeError, la.LinAlgError):
                pass
        if (allow_dense and os.environ.get("LARGE_SOC_SCHUR", "1") != "0" and
                scaling.large_socs and not scaling.psds and n <= 1000 and len(c.eq) <= 30):
            try:
                self.csrows = c.active
                self.csR = scaling.inverse_sqrt_sparse()
                B0 = (self.csR @ A)[self.csrows].tocsc()
                if B0.nnz > .12 * B0.shape[0] * n and B0.shape[0] * n <= 20000000:
                    denseB = B0.toarray()
                    G = denseB.T @ denseB
                else:
                    G = sp.gram_dense(B0) if hasattr(sp, "gram_dense") else (B0.T @ B0).toarray()
                if P.nnz:
                    G += P.toarray()
                vglobal = np.zeros(m)
                for rows, beta, u, lam in scaling.large_socs:
                    for j, block in enumerate(rows):
                        wi = -2 * u[j, 0] * u[j]
                        wi[0] = 2 * u[j, 0] ** 2 - 1
                        vglobal[block] = wi
                        h = A.T @ vglobal
                        head = A[block[0]].toarray().ravel()
                        G += (2.0 / beta[j] ** 2) * (np.outer(h, h) - np.outer(head, head))
                        vglobal[block] = 0
                self.csE = A[c.eq].tocsc()
                delta = max(1e-11, 1e-13 * norm(G))
                G[np.diag_indices(n)] += delta
                self.csFactor = la.cho_factor(G, lower=True, check_finite=False)
                if len(c.eq):
                    self.csV = la.cho_solve(self.csFactor, self.csE.T.toarray(), check_finite=False)
                    S = self.csE @ self.csV
                    self.csEqFactor = la.cho_factor(S, lower=True, check_finite=False)
                self.cs_operator = True
                self.conic_dense = True
                return
            except (ValueError, RuntimeError, la.LinAlgError):
                pass
        curved_schur = bool((c.soc or c.exp.size) and 450 <= n <= 1800)
        general_qp_schur = bool(
            os.environ.get("GENERAL_QP", "1") != "0" and P.nnz and not c.soc and not c.exp.size
            and 50 <= n <= 4000 and
            (P.nnz > .10 * n * n or
             (n <= 1200 and m > 2*n and not getattr(A, "_bound_rows_only", True))))
        if (allow_dense and os.environ.get("CONIC_SCHUR", "1") != "0" and not scaling.psds and
                not scaling.large_socs and (curved_schur or general_qp_schur) and len(c.eq) <= 50):
            try:
                self.csrows = c.active
                self.csR = scaling.inverse_sqrt_sparse()
                self.csB = (self.csR @ A)[self.csrows].tocsc()
                self.csE = A[c.eq].tocsc()
                B = self.csB
                if B.nnz > .12 * B.shape[0] * n and B.shape[0] * n <= 20000000:
                    bd = B.toarray()
                    G = bd.T @ bd
                else:
                    G = sp.gram_dense(B) if hasattr(sp, "gram_dense") else (B.T @ B).toarray()
                if P.nnz:
                    G += P.toarray()
                if n > 600 and np.count_nonzero(G) < .08 * n * n:
                    raise ValueError("Use sparse factorization for a sparse Schur matrix")
                diagonal = np.arange(n)
                delta = max(1e-11, 1e-12 * norm(G))
                G[diagonal, diagonal] += delta
                self.csFactor = la.cho_factor(G, lower=True, check_finite=False)
                if len(c.eq):
                    self.csV = la.cho_solve(self.csFactor, self.csE.T.toarray(), check_finite=False)
                    S = self.csE @ self.csV
                    self.csEqFactor = la.cho_factor(S, lower=True, check_finite=False)
                self.conic_dense = True
                return
            except (ValueError, RuntimeError, la.LinAlgError):
                pass
        if not self.schur:
            self.R = scaling.inverse_sqrt_sparse()
            B = self.R @ A
            I = np.ones(m)
            I[scaling.c.eq] = 0
            self.extra = sum(2 * len(rows) for rows, *_ in scaling.large_socs)
            if self.extra:
                Kbase = sp.bmat([[P, B.T], [B, -sp.diags(I)]], format="csc")
                ur, uc, uv = [], [], []
                col = 0
                for idx, beta, v, lam in scaling.large_socs:
                    for k, rows in enumerate(idx):
                        w = v[k].copy()
                        w[1:] *= -1
                        h = np.asarray(A[rows].T @ w).ravel() * (2.0 / beta[k])
                        hi = np.flatnonzero(h)
                        ur.extend((hi, n + rows))
                        uc.extend((np.full(len(hi), col), np.full(len(rows), col + 1)))
                        uv.extend((h[hi], w))
                        col += 2
                U = sp.csc_matrix((np.concatenate(uv), (np.concatenate(ur), np.concatenate(uc))),
                                  shape=(n + m, self.extra))
                er = np.arange(self.extra)
                C = sp.csc_matrix((-np.ones(self.extra), (er, er ^ 1)),
                                  shape=(self.extra, self.extra))
                K = sp.bmat([[Kbase, U], [U.T, C]], format="csc")
            else:
                K = None
                if (os.environ.get("KKT_ASSEMBLY", "1") != "0" and
                        hasattr(sp, "Matrix") and n + m >= 500):
                    try:
                        from native_kkt import Assembly
                        cached = getattr(A, "_kkt_assembly", None)
                        if cached is None or not cached.compatible(B):
                            cached = Assembly(P, B, I)
                            A._kkt_assembly = cached
                        self.assembly = cached
                        K = cached.update(B)
                    except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
                        pass
                if K is None:
                    K = sp.bmat([[P, B.T], [B, -sp.diags(I)]], format="csc")
            self.K = K
            use_native = (os.environ.get("SOLVER_LDL", "1") != "0" and not self.extra
                          and K.shape[0] >= 500 and norm(K.data) < 1e6)
            if use_native:
                try:
                    from native_ldl import LDL
                    reg = 1e-8
                    Kr = self.regularized(reg)
                    cached = getattr(A, "_ldl_factor", None)
                    pattern = getattr(A, "_ldl_pattern", None)
                    compatible = (pattern is not None and
                                  np.array_equal(pattern[0], Kr.indptr) and
                                  np.array_equal(pattern[1], Kr.indices))
                    if compatible:
                        if cached is None or not cached.compatible(Kr):
                            cached = LDL(Kr, A._ldl_order)
                            A._ldl_factor = cached
                        cached.factor(Kr, n)
                        self.lu = cached
                        self.native = True
                        return
                    self.lu = splu(self.regularized(regularization), permc_spec="MMD_AT_PLUS_A")
                    A._ldl_order = np.argsort(self.lu.perm_c).astype(np.int32)
                    A._ldl_pattern = (Kr.indptr.copy(), Kr.indices.copy())
                    A._ldl_factor = None
                    return
                except (ImportError, OSError, RuntimeError, ValueError):
                    pass
            reg = regularization
            for _ in range(5):
                try:
                    self.lu = splu(self.regularized(reg), permc_spec="COLAMD")
                    break
                except RuntimeError:
                    reg *= 100
            else:
                raise la.LinAlgError("KKT factorization")
        else:
            eq = scaling.c.eq
            eq = eq[np.diff(A[eq].tocsr().indptr) != 0]
            self.eq = eq
            self.Aeq = A[eq].tocsc()
            if len(eq) <= n and n <= 2000 and m * n <= 30000000:
                self.qr = True
                if not hasattr(A, "_psd_pre"):
                    if len(eq):
                        Qe, Re, ep = la.qr(self.Aeq.T.toarray(), mode="full",
                                           pivoting=True, check_finite=False)
                        Q1, Q2 = Qe[:, :len(eq)], Qe[:, len(eq):]
                        Re = Re[:len(eq)]
                        if np.min(np.abs(np.diag(Re))) < 1e-13 * max(1.0, norm(Re)):
                            self.qr = False
                    else:
                        Q1, Q2, Re, ep = None, None, None, None
                    if P.nnz:
                        if (P - sp.diags(P.diagonal())).nnz == 0:
                            Ps = np.diag(np.sqrt(np.maximum(0, P.diagonal())))
                        else:
                            vals, vecs = la.eigh(P.toarray(), check_finite=False)
                            Ps = np.sqrt(np.maximum(0, vals))[:, None] * vecs.T
                    else:
                        Ps = np.empty((0, n))
                    A._psd_pre = (Q1, Q2, Re, ep, Ps, self.qr)
                self.Q1, self.Q2, self.Re, self.ep, self.Ps, self.qr = A._psd_pre
            if self.qr:
                self.arows = scaling.c.active
                self.B = scaling.inverse_sqrt(A.toarray())[self.arows]
                B = self.B if self.Q2 is None else self.B @ self.Q2
                PB = self.Ps if self.Q2 is None else self.Ps @ self.Q2
                self.nulln = B.shape[1]
                B = np.vstack((B, PB))
                self.Q, self.R, self.perm = la.qr(B, mode="economic", pivoting=True,
                                                 check_finite=False)
                if self.R.shape != (self.nulln, self.nulln):
                    raise la.LinAlgError("underdetermined PSD Schur system")
                return
            H = np.array(P.toarray(), order="F")
            # Non-PSD blocks are small; their contribution remains sparse.
            mask = np.ones(m, dtype=bool)
            mask[eq] = False
            for d, rows, *_ in scaling.psds:
                mask[rows] = False
            linrows = np.flatnonzero(mask)
            if linrows.size:
                SA = scaling.apply(A.toarray(), inv=True)
                for d, rows, *_ in scaling.psds:
                    SA[rows] = 0
                H += np.asarray(A.T @ SA)
            for d, rows, r, ri, T, Ti, lam in scaling.psds:
                block = A[rows].tocsc()
                active = np.flatnonzero(np.diff(block.indptr))
                if active.size == 0:
                    continue
                B = np.empty((d * d, active.size), order="F")
                for jj, j in enumerate(active):
                    v = block[:, j].toarray().ravel()
                    M = ri @ smat(v, d) @ ri.T
                    B[:, jj] = M.ravel()
                H[np.ix_(active, active)] += B.T @ B
            ne = len(eq)
            K = np.zeros((n + ne, n + ne), order="F")
            K[:n, :n] = H
            K[:n, n:] = self.Aeq.T.toarray()
            K[n:, :n] = self.Aeq.toarray()
            self.K = K
            diag = np.arange(n + ne)
            Kreg = K.copy()
            Kreg[diag, diag] += np.r_[np.full(n, regularization), np.full(ne, -regularization)]
            self.lu = la.lu_factor(Kreg, check_finite=False)

    def solve(self, rx, rz):
        n, m, sc = self.n, self.m, self.sc
        if self.fallback is not None:
            return self.fallback.solve(rx, rz)
        if self.conic_dense:
            rb = (sc.inverse_sqrt(rz) if self.cs_operator else self.csR @ rz)[self.csrows]
            re = rz[sc.c.eq]
            dx, u, ze = self.solve_conic_dense(rx, rb, re)
            last = np.inf
            for _ in range(7):
                r1 = rx - self.P @ dx - self.conic_transpose(u) - self.csE.T @ ze
                r2 = rb - self.conic_product(dx) + u
                r3 = re - self.csE @ dx
                eq_scale = max(1e-300, norm(re),
                               getattr(self.A, "_eq_tolerance", 3e-16) / 3e-10)
                error = max(norm(r1) / max(1e-6, norm(rx)),
                            norm(r2) / max(1e-6, norm(rb)),
                            norm(r3) / eq_scale)
                if error <= 3e-14:
                    break
                if error >= last * .95:
                    break
                last = error
                xx, uu, ee = self.solve_conic_dense(r1, r2, r3)
                dx += xx
                u += uu
                ze += ee
            if error > 3e-10:
                self.fallback = KKTSolver(self.P, self.A, sc, self.reg, allow_dense=False)
                return self.fallback.solve(rx, rz)
            zbar = np.zeros(m)
            zbar[self.csrows] = u
            dz = sc.inverse_sqrt(zbar, dual=True) if self.cs_operator else self.csR.T @ zbar
            dz[sc.c.eq] = ze
            return dx, dz
        if self.diagonal:
            dx, dz = self.solve_diagonal(rx, rz)
            last = np.inf
            for _ in range(8 if self.dense_box else 4):
                rr1 = rx - self.P @ dx - self.A.T @ dz
                rr2 = rz - self.A @ dx + sc.apply(dz)
                error = max(norm(rr1), norm(rr2))
                if error >= last * .9 or error <= 3e-15 * max(1e-8, norm(rx), norm(rz)):
                    break
                last = error
                ddx, ddz = self.solve_diagonal(rr1, rr2)
                dx += ddx
                dz += ddz
            rr1 = rx - self.P @ dx - self.A.T @ dz
            rr2 = rz - self.A @ dx + sc.apply(dz)
            if max(norm(rr1), norm(rr2)) > 2e-10 * max(1e-6, norm(rx), norm(rz)):
                self.fallback = KKTSolver(self.P, self.A, sc, self.reg,
                                          allow_dense=False, allow_normal=False)
                return self.fallback.solve(rx, rz)
            return dx, dz
        if not self.schur:
            rhs = np.r_[rx, sc.inverse_sqrt(rz), np.zeros(self.extra)]
            v = self.lu.solve(rhs)
            for _ in range(5):
                r = rhs - self.K @ v
                if norm(r) <= self.linear_tolerance * max(1e-8, norm(rhs)):
                    break
                dv = self.lu.solve(r)
                v += dv
            if self.native and norm(rhs - self.K @ v) > max(1e-11, 5*self.linear_tolerance) * max(1e-6, norm(rhs)):
                self.lu = splu(self.regularized(self.reg))
                self.native = False
                v = self.lu.solve(rhs)
                for _ in range(5):
                    r = rhs - self.K @ v
                    if norm(r) <= self.linear_tolerance * max(1e-8, norm(rhs)):
                        break
                    v += self.lu.solve(r)
            return v[:n], sc.inverse_sqrt(v[n:n + m], dual=True)
        if self.qr:
            if len(self.eq):
                t0 = la.solve_triangular(self.Re.T, rz[self.eq[self.ep]], lower=True,
                                         check_finite=False)
                x0 = self.Q1 @ t0
                xn = self.Q2.T @ rx
            else:
                x0 = np.zeros(n)
                xn = rx
            rhs = np.r_[sc.inverse_sqrt(rz)[self.arows] - self.B @ x0,
                        -self.Ps @ x0]
            a = self.Q.T @ rhs
            t = la.solve_triangular(self.R.T, xn[self.perm], lower=True, check_finite=False)
            dxp = la.solve_triangular(self.R, a + t, lower=False, check_finite=False)
            dxn = np.empty(self.nulln)
            dxn[self.perm] = dxp
            dx = x0 + (dxn if self.Q2 is None else self.Q2 @ dxn)
            u = self.Q @ t - (rhs - self.Q @ a)
            v = np.zeros(m)
            v[self.arows] = u[:len(self.arows)]
            dz = sc.inverse_sqrt(v, dual=True)
            if len(self.eq):
                r = rx - self.P @ dx - self.A.T @ dz
                ze = la.solve_triangular(self.Re, self.Q1.T @ r, lower=False, check_finite=False)
                dz[self.eq[self.ep]] = ze
            return dx, dz
        hz = sc.apply(rz, inv=True)
        rhs = np.r_[rx + self.A.T @ hz, rz[self.eq]]
        v = la.lu_solve(self.lu, rhs, check_finite=False)
        for _ in range(3):
            r = rhs - self.K @ v
            if norm(r) <= 1e-12 * max(1.0, norm(rhs)):
                break
            v += la.lu_solve(self.lu, r, check_finite=False)
        dx = v[:n]
        dz = sc.apply(self.A @ dx - rz, inv=True)
        dz[self.eq] = v[n:]
        return dx, dz

    def regularized(self, delta):
        if self.assembly is not None:
            return self.assembly.regularized(delta)
        diagonal = np.r_[np.full(self.n, delta), np.full(self.m, -delta),
                         np.full(self.extra, delta)]
        return self.K + sp.diags(diagonal, format="csc")

    def solve_diagonal(self, rx, rz):
        c, h = self.sc.c, self.sc.diag
        pr, cols, vals = self.dpr, self.dcols, self.dvals
        rr = rx + np.bincount(cols, weights=vals * rz[c.pos[pr]] / h[pr], minlength=self.n)
        if self.dense_box:
            t = la.cho_solve(self.boxFactor, rr, check_finite=False)
            if len(c.eq):
                rhs = self.ds * (self.deq @ t - rz[c.eq])
                ze = self.ds * la.cho_solve(self.boxEqFactor, rhs, check_finite=False)
                dx = t - self.boxV @ ze
            else:
                dx, ze = t, np.empty(0)
        elif len(c.eq):
            g = self.dg
            rhs = self.ds * (self.deq @ (rr / g) - rz[c.eq])
            ze = (la.cho_solve(self.dfactor, rhs, check_finite=False) if self.dense_d
                  else self.dfactor.solve(rhs))
            ze *= self.ds
            dx = (rr - self.deq.T @ ze) / g
        else:
            dx, ze = rr / self.dg, np.empty(0)
        dz = np.zeros(self.m)
        dz[c.pos] = -rz[c.pos] / h
        dz[c.pos[pr]] += vals * dx[cols] / h[pr]
        dz[c.eq] = ze
        return dx, dz

    def solve_conic_dense(self, rx, rb, re):
        rhs = rx + self.conic_transpose(rb)
        v = la.cho_solve(self.csFactor, rhs, check_finite=False)
        if len(re):
            ze = la.cho_solve(self.csEqFactor, self.csE @ v - re, check_finite=False)
            dx = v - self.csV @ ze
        else:
            dx, ze = v, np.empty(0)
        u = self.conic_product(dx) - rb
        return dx, u, ze

    def conic_product(self, x):
        if not self.cs_operator:
            return self.csB @ x
        return self.sc.inverse_sqrt(self.A @ x)[self.csrows]

    def conic_transpose(self, z):
        if not self.cs_operator:
            return self.csB.T @ z
        v = np.zeros(self.m)
        v[self.csrows] = z
        return self.A.T @ self.sc.inverse_sqrt(v, dual=True)


def interior(P0, q0, A0, b0, cones, max_iter=150):
    P0, A0 = sp.csc_matrix(P0), sp.csc_matrix(A0)
    P, q, A, b, D, E, cost = equilibrate(P0, q0, A0, b0, cones)
    n, m = len(q), len(b)
    x = np.zeros(n)
    s, z = cones.initial()
    tau, kappa = 1.0, 1.0
    nu = cones.degree + 1
    eq_base_tolerance = (1e-11 * (1 + norm(b0)) * np.min(E[cones.eq])
                         if len(cones.eq) else 0.0)
    start = time.monotonic()
    best = None
    besterr = np.inf
    fast_state = None
    state_mode = os.environ.get("FAST_STATE", "auto")
    if (hasattr(sp, "Matrix") and state_mode != "0" and
            (state_mode == "1" or n + m >= 10000) and
            not (n <= 5000 and P.nnz > .3 * n * n)):
        try:
            from native_state import State
            fast_state = State(P, q, A, b, D, E, cost,
                               (norm(A0.data) or 1.0, norm(P0.data) or 1.0,
                                norm(q0), norm(b0)))
        except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
            pass
    for it in range(max_iter):
        try:
            if fast_state is not None and not fast_state.compute(x, s, z, tau):
                fast_state = None
            candidate = None
            if fast_state is not None:
                values = fast_state.values
                rp0, rd0, gap0 = map(float, values[:3])
            else:
                xx, ss, zz = D * x / tau, s / (E * tau), E * z / (cost * tau)
                rp0, rd0, gap0 = solved_metrics(P0, q0, A0, b0, xx, ss, zz)
                candidate = {"status": "solved", "x": xx, "s": ss, "z": zz}
            err = max(rp0, rd0, gap0)
            if candidate is None and (err <= 7e-9 or (err < 1e-6 and err < besterr)):
                candidate = {"status": "solved", "x": D * x / tau,
                             "s": s / (E * tau), "z": E * z / (cost * tau)}
            if err < besterr:
                besterr = err
                if candidate is not None:
                    best = candidate
            log(it, "res", "%.2e %.2e %.2e" % (rp0, rd0, gap0),
                "tau", "%.2e" % tau, "kap", "%.2e" % kappa,
                "sec", "%.3f" % (time.monotonic() - start))
            if err <= 7e-9 and checked(P0, q0, A0, b0, cones, candidate):
                return candidate
            if it > 3:
                if fast_state is None or (values[7] > 1e-8 and values[9] < 1e-7 * values[7]):
                    cand = {"status": "primal_infeasible", "z": E * z}
                    if checked(P0, q0, A0, b0, cones, cand):
                        return cand
                if fast_state is None or (values[8] > 1e-8 and
                        max(values[10], values[11]) < 1e-7 * values[8]):
                    cand = {"status": "dual_infeasible", "x": D * x, "s": s / E}
                    if checked(P0, q0, A0, b0, cones, cand):
                        return cand
            if fast_state is not None:
                Px, rp, rd = fast_state.Px, fast_state.rp, fast_state.rd
                xPx = float(values[3])
                rg = float(values[4] + values[5] + xPx / tau + kappa)
                mu = float(values[6] + tau * kappa) / nu
            else:
                Px = P @ x
                rp, rd = A @ x + s - b * tau, Px + q * tau + A.T @ z
                xPx = float(x @ Px)
                rg = float(q @ x + b @ z + xPx / tau + kappa)
                mu = float(s @ z + tau * kappa) / nu
            if not np.isfinite(mu) or mu <= 0 or tau <= 0:
                break
            sc = Scaling(cones, s, z, mu)
            A._eq_tolerance = eq_base_tolerance * tau
            kkt = KKTSolver(P, A, sc)
            wx, wz = kkt.solve(q, -b)
            Ux = q + 2 * Px / tau
            wpx = wx + x / tau
            denom = -max(0.0, float(wpx @ (P @ wpx))) - sc.quadratic(wz) - kappa / tau

            def direction(rc, rt, residual_scale=1.0, quadratic_correction=0.0):
                vx, vz = kkt.solve(-residual_scale * rd, -residual_scale * rp - rc)
                dt = (-residual_scale * rg - quadratic_correction - rt / tau -
                      float(Ux @ vx + b @ vz)) / denom
                dx, dz = vx - wx * dt, vz - wz * dt
                ds = -residual_scale * rp - A @ dx + b * dt
                ds[cones.eq] = 0
                dk = (rt - kappa * dt) / tau
                return dx, ds, dz, dt, dk

            if err < 1e-4:
                dtrial = direction(sc.rhs(0.0), -tau * kappa)
                dx, ds, dz, dt, dk = dtrial
                if tau + dt > 1e-14 * tau:
                    xt = D * (x + dx) / (tau + dt)
                    st = (s + ds) / (E * (tau + dt))
                    zt = E * (z + dz) / (cost * (tau + dt))
                    trial = {"status": "solved", "x": xt, "s": st, "z": zt}
                    if checked(P0, q0, A0, b0, cones, trial):
                        return trial
            if cones.exp.size and os.environ.get("EXP_PC", "") != "1":
                theta = exp_centrality(cones, s, z, mu)
                sigma = 1.0 if theta > 0.4 else float(os.environ.get("EXP_SIGMA", "0.2"))
                dx, ds, dz, dt, dk = direction(sc.rhs(sigma), sigma * mu - tau * kappa, 1 - sigma)
                alpha = min(cones.max_step(s, ds), cones.max_step(z, dz, dual=True))
                if dt < 0:
                    alpha = min(alpha, -tau / dt)
                if dk < 0:
                    alpha = min(alpha, -kappa / dk)
                alpha *= 0.95
                for _ in range(25):
                    sn, zn = s + alpha * ds, z + alpha * dz
                    tn, kn = tau + alpha * dt, kappa + alpha * dk
                    mun = float(sn @ zn + tn * kn) / nu
                    if mun > 0:
                        thetan = exp_centrality(cones, sn, zn, mun)
                        if thetan <= 0.9 or thetan < (1 - 0.01 * alpha) * theta:
                            break
                    alpha *= 0.5
                log("exp", "theta", "%.2e" % theta, "sigma", sigma, "alpha", "%.2e" % alpha)
                if alpha < 1e-12:
                    break
                x += alpha * dx
                s += alpha * ds
                z += alpha * dz
                tau += alpha * dt
                kappa += alpha * dk
                if tau > 10 or tau < 0.1:
                    x /= tau
                    s /= tau
                    z /= tau
                    kappa /= tau
                    tau = 1.0
                continue
            rc = sc.rhs(0.0)
            da = direction(rc, -tau * kappa)
            dx, ds, dz, dt, dk = da

            def step(ds, dz, dt, dk):
                a = min(cones.max_step(s, ds), cones.max_step(z, dz, dual=True))
                if dt < 0:
                    a = min(a, -tau / dt)
                if dk < 0:
                    a = min(a, -kappa / dk)
                return a

            alpha = step(ds, dz, dt, dk)
            mua = float((s + alpha * ds) @ (z + alpha * dz) +
                        (tau + alpha * dt) * (kappa + alpha * dk)) / nu
            sigma = min(0.99, max(0.0, mua / mu) ** 3)
            if cones.exp.size:
                sigma = max(0.1, sigma)
            rc = sc.rhs(sigma, ds, dz)
            correction = 0.0
            corrector_mode = os.environ.get("QP_CORRECTOR", "auto")
            if P.nnz and (corrector_mode == "1" or
                          (corrector_mode == "auto" and n >= 450 and P.nnz > .12 * n * n)):
                dbar = dx - x * (dt / tau)
                correction = max(0.0, float(dbar @ (P @ dbar)) / tau)
            dc = direction(rc, sigma * mu - tau * kappa - dt * dk, 1 - sigma, correction)
            dx, ds, dz, dt, dk = dc
            alpha = min(1.0, 0.99 * step(ds, dz, dt, dk))
            if alpha < 1e-12 or not np.isfinite(alpha):
                break
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kappa += alpha * dk
            if max(norm(x), norm(s), norm(z), tau, kappa) > 1e12 or tau < 1e-12:
                scale = max(norm(s), norm(z), tau, kappa)
                x /= scale
                s /= scale
                z /= scale
                tau /= scale
                kappa /= scale
        except (la.LinAlgError, RuntimeError, FloatingPointError, ValueError,
                ZeroDivisionError, OverflowError) as e:
            log("IPM stopped:", type(e).__name__, str(e))
            break
    if best is not None and checked(P0, q0, A0, b0, cones, best, 9.5e-9):
        return best
    return {"status": "unresolved"}


def make_highs(P, q, A, lower, upper, tol=1e-9, solver=None):
    from kernels import highs
    h = highs()
    model = h.HighsModel()
    lp = model.lp_
    m, n = A.shape
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_, lp.col_upper_ = np.full(n, -np.inf), np.full(n, np.inf)
    lp.row_lower_, lp.row_upper_ = lower, upper
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    if P.nnz:
        T = sp.tril(P, format="csc")
        model.hessian_.dim_ = n
        model.hessian_.format_ = h.HessianFormat.kTriangular
        model.hessian_.start_, model.hessian_.index_, model.hessian_.value_ = T.indptr, T.indices, T.data
    hs = h._Highs()
    hs.setOptionValue("output_flag", False)
    hs.setOptionValue("threads", 1)
    hs.setOptionValue("parallel", "off")
    hs.setOptionValue("primal_feasibility_tolerance", tol)
    hs.setOptionValue("dual_feasibility_tolerance", tol)
    hs.setOptionValue("ipm_optimality_tolerance", min(tol, 1e-10))
    hs.setOptionValue("small_matrix_value", 1e-12)
    if solver:
        hs.setOptionValue("solver", solver)
    hs.passModel(model)
    return hs


def separable(P, q, A, b, cones):
    n, m = len(q), len(b)
    pc = np.diff(P.indptr)
    if np.any(pc > 1) or not np.array_equal(P.indices, np.flatnonzero(pc)):
        return None
    R = A.tocsr()
    rc = np.diff(R.indptr)
    if np.any(rc > 1):
        return None
    eq = np.zeros(m, dtype=bool)
    eq[cones.eq] = True
    empty = np.flatnonzero((rc == 0) & ((eq & (b != 0)) | (~eq & (b < 0))))
    if empty.size:
        z = np.zeros(m)
        i = empty[0]
        z[i] = -np.sign(b[i]) if eq[i] else 1
        out = {"status": "primal_infeasible", "z": z}
        return out if checked(P, q, A, b, cones, out) else None
    rows = np.flatnonzero(rc)
    cols = R.indices
    vals = R.data
    bounds = b[rows] / vals
    islower = (vals < 0) | eq[rows]
    isupper = (vals > 0) | eq[rows]
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(lower, cols[islower], bounds[islower])
    np.minimum.at(upper, cols[isupper], bounds[isupper])
    lrow, urow = np.full(n, m, dtype=np.int64), np.full(n, m, dtype=np.int64)
    lm = islower & (bounds == lower[cols])
    um = isupper & (bounds == upper[cols])
    np.minimum.at(lrow, cols[lm], rows[lm])
    np.minimum.at(urow, cols[um], rows[um])
    bad = np.flatnonzero(lower > upper)
    if bad.size:
        i = bad[0]
        z = np.zeros(m)
        li, ui = lrow[i], urow[i]
        z[li] = -1 / R.data[R.indptr[li]]
        z[ui] = 1 / R.data[R.indptr[ui]]
        out = {"status": "primal_infeasible", "z": z}
        return out if checked(P, q, A, b, cones, out) else None
    p = P.diagonal()
    unbounded = np.flatnonzero((p == 0) & (((q > 0) & ~np.isfinite(lower)) |
                                         ((q < 0) & ~np.isfinite(upper))))
    if unbounded.size:
        x = np.zeros(n)
        i = unbounded[0]
        x[i] = -np.sign(q[i])
        out = {"status": "dual_infeasible", "x": x, "s": -A @ x}
        return out if checked(P, q, A, b, cones, out) else None
    x = np.zeros(n)
    positive = p > 0
    x[positive] = -q[positive] / p[positive]
    x[(p == 0) & (q > 0)] = lower[(p == 0) & (q > 0)]
    x[(p == 0) & (q < 0)] = upper[(p == 0) & (q < 0)]
    np.clip(x, lower, upper, out=x)
    grad = p * x + q
    z = np.zeros(m)
    lm = (grad > 0) & (lrow < m)
    um = (grad < 0) & (urow < m)
    z[lrow[lm]] = -grad[lm] / R.data[R.indptr[lrow[lm]]]
    z[urow[um]] = -grad[um] / R.data[R.indptr[urow[um]]]
    out = {"status": "solved", "x": x, "s": b - A @ x, "z": z}
    return out if checked(P, q, A, b, cones, out) else None


def polish_poly(P, q, A, b, cones, x, z):
    s = b - A @ x
    active = (np.abs(s) <= 1e-6 * (1 + norm(b))) | (z > 1e-9)
    active[cones.eq] = True
    rows = np.flatnonzero(active)
    Ae = A[rows].tocsc()
    n, ne = len(q), len(rows)
    K = sp.bmat([[P, Ae.T], [Ae, sp.csc_matrix((ne, ne))]], format="csc")
    scale = max(1.0, norm(K.data))
    reg = 1e-11 * scale
    R = sp.diags(np.r_[np.full(n, reg), np.full(ne, -reg)], format="csc")
    try:
        lu = splu(K + R)
        zz = z[rows].copy()
        xx = x.copy()
        for _ in range(8):
            rd = P @ xx + q + Ae.T @ zz
            rp = Ae @ xx - b[rows]
            rhs = -np.r_[rd, rp]
            v = lu.solve(rhs)
            for _ in range(3):
                rr = rhs - K @ v
                if norm(rr) <= 1e-14 * max(norm(rhs), 1e-8):
                    break
                v += lu.solve(rr)
            xx += v[:n]
            zz += v[n:]
            znew = np.zeros(len(b))
            znew[rows] = zz
            out = {"status": "solved", "x": xx.copy(), "s": b - A @ xx, "z": znew}
            if checked(P, q, A, b, cones, out):
                return out
            if norm(v) < 1e-14 * max(1.0, norm(xx)):
                break
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    return None


def equality_quadratic(P, q, A, b, cones):
    """Solve equality-only QPs directly, including regularized ray recovery."""
    n, m = len(q), len(b)
    if len(cones.pos) or n + m == 0:
        return None
    K = sp.bmat([[P, A.T], [A, sp.csc_matrix((m, m))]], format="csc") if m else P
    rhs = np.r_[-q, b]
    scale = max(norm(K.data), 1e-100)
    dense = bool(P.nnz > .12 * n * n and 0 < n <= 5000 and m <= 50)
    Pd, E = (P.toarray(), A.toarray()) if dense else (None, None)
    for delta in (0.0, scale * 1e-11):
        try:
            if dense:
                H = Pd.copy()
                H[np.diag_indices(n)] += delta
                pf = la.cho_factor(H, lower=True, check_finite=False)
                if m:
                    V = la.cho_solve(pf, E.T, check_finite=False)
                    S = E @ V
                    S[np.diag_indices(m)] += delta
                    ef = la.cho_factor(S, lower=True, check_finite=False)
                def solve(rhs):
                    t = la.cho_solve(pf, rhs[:n], check_finite=False)
                    if m:
                        z = la.cho_solve(ef, E @ t - rhs[n:], check_finite=False)
                        return np.r_[t - V @ z, z]
                    return t
            else:
                M = K + sp.diags(np.r_[np.full(n, delta), np.full(m, -delta)]) if delta else K
                lu = splu(M.tocsc())
                solve = lu.solve
            v = solve(rhs)
            for _ in range(8):
                out = {"status": "solved", "x": v[:n].copy(), "s": np.zeros(m), "z": v[n:].copy()}
                if checked(P, q, A, b, cones, out):
                    return out
                if m:
                    cert = {"status": "primal_infeasible", "z": v[n:].copy()}
                    if checked(P, q, A, b, cones, cert):
                        return cert
                cert = {"status": "dual_infeasible", "x": v[:n].copy(), "s": np.zeros(m)}
                if checked(P, q, A, b, cones, cert):
                    return cert
                residual = rhs - K @ v
                if norm(residual) <= 1e-15 * max(1.0, norm(rhs)):
                    break
                v += solve(residual)
        except (RuntimeError, ValueError, la.LinAlgError, FloatingPointError):
            pass
    return None


def polyhedral(P, q, A, b, cones):
    P, A = sp.csc_matrix(P), sp.csc_matrix(A)
    out = separable(P, q, A, b, cones)
    if out is not None:
        return out
    if not len(cones.pos) and os.environ.get("DIRECT_EQ", "1") != "0":
        out = equality_quadratic(P, q, A, b, cones)
        if out is not None:
            return out
    if (P.nnz and len(cones.eq) == 1 and len(q) >= 1000 and
            os.environ.get("SINGLE_EQ", "1") != "0"):
        from single_eq import solve as solve_single_eq
        out = solve_single_eq(P, q, A, b, cones,
                             lambda result: checked(P, q, A, b, cones, result))
        if out is not None:
            return out
    n = len(q)
    prefer_ipm = (P.nnz and n >= 400 and P.nnz < max(20 * n, .2 * n * n)
                  and np.count_nonzero(P.diagonal()) >= .25 * n)
    if (P.nnz and not prefer_ipm and n >= 400 and
            len(cones.eq) <= min(1200, n // 2)):
        pc = np.diff(P.indptr)
        if np.all(pc <= 1) and np.array_equal(P.indices, np.flatnonzero(pc)):
            bounds = A[cones.pos].tocsr()
            if np.all(np.diff(bounds.indptr) <= 1):
                coverage = np.bincount(bounds.indices, minlength=n) > 0
                prefer_ipm = bool(np.all(coverage | (P.diagonal() > 0)))
    if (P.nnz > .12 * n * n and 450 <= n <= 5000 and len(cones.eq) <= 50
            and os.environ.get("DENSE_QP", "1") != "0"):
        bounds = A[cones.pos].tocsr()
        if np.all(np.diff(bounds.indptr) <= 1):
            prefer_ipm = True
    if (P.nnz and os.environ.get("GENERAL_QP", "1") != "0" and len(cones.eq) <= 50
            and 50 <= n <= 4000):
        if P.nnz > .10 * n * n:
            prefer_ipm = True
        elif n <= 1200 and len(b) > 2*n:
            bounds = A[cones.pos].tocsr()
            if np.any(np.diff(bounds.indptr) > 1):
                prefer_ipm = True
    if P.nnz and (os.environ.get("QP_IPM") or
                  (prefer_ipm and not os.environ.get("QP_HIGHS"))):
        out = interior(P, q, A, b, cones)
        if out["status"] != "unresolved":
            return out
    lower = np.full(len(b), -np.inf)
    lower[cones.eq] = b[cones.eq]
    method = os.environ.get("LP_METHOD") if not P.nnz else None
    if method is None and not P.nnz and n >= 2000 and A.nnz >= 10000:
        ne = len(cones.eq)
        choose_custom = False
        if ne and ne <= min(1200, n // 2):
            bounds = A[cones.pos].tocsr()
            if np.all(np.diff(bounds.indptr) <= 1):
                covered = np.bincount(bounds.indices, minlength=n) > 0
                E = A[cones.eq]
                choose_custom = bool(np.all(covered) and E.nnz > 2.5 * n)
        if choose_custom:
            result = interior(P, q, A, b, cones)
            if result["status"] != "unresolved":
                return result
        method = "ipm"
    if (method is None and not P.nnz and n >= 250 and
            os.environ.get("LP_MID_IPM", "1") != "0"):
        method = "ipm"
    if method == "custom":
        return interior(P, q, A, b, cones)
    hs = make_highs(P, q, A, lower, b, tol=1e-10, solver=method)
    if method == "ipm" and os.environ.get("LP_NO_CROSSOVER"):
        hs.setOptionValue("run_crossover", "off")
    hs.run()
    log("HiGHS", hs.modelStatusToString(hs.getModelStatus()))
    sol = hs.getSolution()
    if sol.value_valid and sol.dual_valid:
        x = np.array(sol.col_value)
        z = -np.array(sol.row_dual)
        s = b - A @ x
        out = {"status": "solved", "x": x, "s": s, "z": z}
        if checked(P, q, A, b, cones, out):
            return out
        log("HiGHS residuals", solved_metrics(P, q, A, b, x, s, z))
        out = polish_poly(P, q, A, b, cones, x, z)
        if out is not None:
            return out
    status = hs.modelStatusToString(hs.getModelStatus()).lower()
    n, m = len(q), len(b)
    if "infeasible" in status:
        # A bounded separating-hyperplane LP recovers a normalized Farkas ray.
        B = A.T.tocsc()
        H = make_highs(sp.csc_matrix((m, m)), b / (norm(b) or 1), B,
                       np.zeros(n), np.zeros(n), tol=1e-10)
        lb, ub = np.zeros(m), np.ones(m)
        lb[cones.eq] = -1
        H.changeColsBounds(m, np.arange(m, dtype=np.int32), lb, ub)
        H.run()
        z = np.array(H.getSolution().col_value)
        out = {"status": "primal_infeasible", "z": z}
        if checked(P, q, A, b, cones, out):
            return out
    if "unbounded" in status or "infeasible" in status:
        B = sp.vstack((A, P), format="csc") if P.nnz else A
        lo = np.full(B.shape[0], -np.inf)
        hi = np.zeros(B.shape[0])
        lo[cones.eq] = 0
        if P.nnz:
            lo[m:] = 0
        H = make_highs(sp.csc_matrix((n, n)), q / (norm(q) or 1),
                       B, lo, hi, tol=1e-10)
        H.changeColsBounds(n, np.arange(n, dtype=np.int32), -np.ones(n), np.ones(n))
        H.run()
        x = np.array(H.getSolution().col_value)
        out = {"status": "dual_infeasible", "x": x, "s": -A @ x}
        if checked(P, q, A, b, cones, out):
            return out
    return interior(P, q, A, b, cones)


def cone_presolve(A, b, cones):
    """Exact reductions of affine cone sections, including adjoint recovery."""
    counts = np.diff(A.tocsr().indptr)
    records = []
    changed = False
    for kind, d, rows in cones.blocks:
        rec = ("identity", kind, d, rows, None)
        if kind in ("zero", "nonnegative"):
            empty = rows[counts[rows] == 0]
            bad = empty[b[empty] != 0] if kind == "zero" else empty[b[empty] < 0]
            if bad.size:
                z = np.zeros(len(b))
                z[bad[0]] = -np.sign(b[bad[0]]) if kind == "zero" else 1.0
                return None, {"status": "primal_infeasible", "z": z}
            kept = rows[counts[rows] != 0]
            if len(kept) != len(rows):
                rec = ("select", kind, len(kept), kept, None)
                changed = True
        elif kind == "soc":
            tail = rows[1:]
            varying = tail[counts[tail] != 0]
            if len(varying) == 0:
                t = float(np.linalg.norm(b[tail]))
                g = np.r_[1.0, -b[tail] / t] if t else np.r_[1.0, np.zeros(d - 1)]
                rec = ("linear", "nonnegative", 1, rows, (g, np.array([1.0]), rows[:1]))
                changed = True
            elif d == 2:
                rec = ("soc2", "nonnegative", 2, rows, None)
                changed = True
            else:
                fixed = tail[counts[tail] == 0]
                if len(fixed) > 1:
                    t = float(np.linalg.norm(b[fixed]))
                    kept = np.r_[rows[:1], varying]
                    w = b[fixed] / t if t else np.empty(0)
                    nd = len(kept) + bool(t)
                    rec = ("compress", "soc", nd, kept, (fixed, w))
                    changed = True
        elif kind == "psd":
            r, c, scale = packing(d)
            diag = r == c
            dr, off = rows[diag], rows[~diag]
            if not np.any(counts[off]):
                if not np.any(b[off]):
                    rec = ("select", "nonnegative", d, dr, None)
                    changed = True
                else:
                    M = A[dr].tocsr()
                    k = M.indptr[1]
                    same = (np.all(np.diff(M.indptr) == k) and
                            np.array_equal(M.indices, np.tile(M.indices[:k], d)) and
                            np.array_equal(M.data, np.tile(M.data[:k], d)))
                    if same:
                        val, vec = la.eigh(smat(b[rows], d), subset_by_index=[0, 0],
                                           check_finite=False)
                        g = svec(np.outer(vec[:, 0], vec[:, 0]))
                        rec = ("linear", "nonnegative", 1, rows,
                               (g, np.ones(d), dr))
                        changed = True
        elif kind == "exponential":
            u, v, w = rows
            if counts[v] == 0 and b[v] > 0:
                if counts[u] == 0 and abs(b[u] / b[v]) < 300:
                    t = b[u] / b[v]
                    g = np.array([-math.exp(t), math.exp(t) * (t - 1), 1.0])
                    rec = ("linear", "nonnegative", 1, rows,
                           (g, np.array([1.0]), rows[2:3]))
                    changed = True
                elif counts[w] == 0 and b[w] > 0:
                    t = math.log(b[w]) - math.log(b[v])
                    g = np.array([-1.0, t - 1.0, b[v] / b[w]])
                    rec = ("linear", "nonnegative", 1, rows,
                           (g, np.array([-1.0]), rows[:1]))
                    changed = True
        records.append(rec)
    if not changed:
        return None, None
    tr, tc, tv, fr, fc, fv = [], [], [], [], [], []
    blocks = []
    position = 0
    for how, kind, d, rows, extra in records:
        if not d:
            continue
        width = d * (d + 1) // 2 if kind == "psd" else d
        newrows = np.arange(position, position + width)
        if blocks and blocks[-1]["type"] == kind and kind in ("zero", "nonnegative"):
            blocks[-1]["dim"] += d
        else:
            blocks.append({"type": kind, "dim": d})
        if how in ("identity", "select"):
            tr.append(newrows)
            tc.append(rows)
            tv.append(np.ones(width))
            fr.append(rows)
            fc.append(newrows)
            fv.append(np.ones(width))
        elif how == "linear":
            g, p, prows = extra
            tr.append(np.full(len(rows), position))
            tc.append(rows)
            tv.append(g)
            fr.append(prows)
            fc.append(np.full(len(prows), position))
            fv.append(p)
        elif how == "soc2":
            tr.append(np.repeat(newrows, 2))
            tc.append(np.tile(rows, 2))
            tv.append(np.array([1.0, 1.0, 1.0, -1.0]))
            fr.append(np.repeat(rows, 2))
            fc.append(np.tile(newrows, 2))
            fv.append(np.array([.5, .5, .5, -.5]))
        else:
            fixed, weights = extra
            k = len(rows)
            tr.append(newrows[:k])
            tc.append(rows)
            tv.append(np.ones(k))
            fr.append(rows)
            fc.append(newrows[:k])
            fv.append(np.ones(k))
            if len(weights):
                tr.append(np.full(len(fixed), position + k))
                tc.append(fixed)
                tv.append(weights)
                fr.append(fixed)
                fc.append(np.full(len(fixed), position + k))
                fv.append(weights)
        position += width
    def matrix(rr, cc, vv, shape):
        if not rr:
            return sp.csc_matrix(shape)
        return sp.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                             shape=shape)
    T = matrix(tr, tc, tv, (position, len(b)))
    F = matrix(fr, fc, fv, (len(b), position))
    An = (T @ A).tocsc()
    An.eliminate_zeros()
    bn = T @ b
    return (An, bn, blocks, T, F, b - F @ bn), None


class DiagonalEquations:
    def __init__(self, diagonal):
        self.diagonal = diagonal

    def solve(self, rhs, trans="N"):
        return rhs / (self.diagonal[:, None] if np.ndim(rhs) == 2 else self.diagonal)


def singleton_equalities(P, q, A, b, cones):
    n, ne = len(q), len(cones.eq)
    if not ne or not (cones.soc or cones.exp.size):
        return None
    E = A[cones.eq].tocsc()
    candidates = np.flatnonzero((np.diff(E.indptr) == 1) & (np.diff(P.indptr) == 0))
    if not len(candidates):
        return None
    localrows = E.indices[E.indptr[candidates]]
    coefficients = E.data[E.indptr[candidates]]
    rownorm = np.asarray(abs(E).max(axis=1).toarray()).ravel()
    score = ((1 + np.diff(A.indptr)[candidates]) *
             rownorm[localrows] / np.maximum(np.abs(coefficients), 1e-300))
    order = np.lexsort((candidates, score, localrows))
    sortedrows = localrows[order]
    chosen = order[np.r_[True, sortedrows[1:] != sortedrows[:-1]]]
    pivot, lr, coeff = candidates[chosen], localrows[chosen], coefficients[chosen]
    if len(pivot) < max(4, n * .025):
        return None
    eliminated = cones.eq[lr]
    free_mask = np.ones(n, dtype=bool)
    free_mask[pivot] = False
    free = np.flatnonzero(free_mask)
    Er = E[lr][:, free].tocsc()
    col = np.repeat(np.arange(len(free), dtype=np.int32), np.diff(Er.indptr))
    R = sp.csc_matrix((np.r_[-Er.data / coeff[Er.indices], np.ones(len(free))],
                       (np.r_[pivot[Er.indices], free], np.r_[col, np.arange(len(free))])),
                      shape=(n, len(free)))
    keep_mask = np.ones(len(b), dtype=bool)
    keep_mask[eliminated] = False
    keep = np.flatnonzero(keep_mask)
    An = (A @ R)[keep].tocsc()
    if An.nnz > max(3 * A.nnz, A.nnz + 500):
        return None
    x0 = np.zeros(n)
    x0[pivot] = b[eliminated] / coeff
    bn = (b - A @ x0)[keep]
    Pn = P[free][:, free].tocsc()
    qn = R.T @ q
    blocks = []
    for kind, dim, rows in cones.blocks:
        if kind == "zero":
            dim = int(np.sum(keep_mask[rows]))
        if dim:
            blocks.append({"type": kind, "dim": dim})
    return Pn, qn, An, bn, blocks, R, x0, pivot, DiagonalEquations(coeff), keep, eliminated


def equality_presolve(P, q, A, b, cones):
    n, ne = len(q), len(cones.eq)
    if os.environ.get("EQ_SINGLETON", "1") != "0":
        singleton = singleton_equalities(P, q, A, b, cones)
        if singleton is not None:
            return singleton
    if not ne or ne > n or n > 800 or ne > 600:
        return None
    E = A[cones.eq].tocsc()
    counts = np.diff(A.indptr) + np.diff(P.indptr)
    weights = 1.0 / (1.0 + counts)
    dense = E.toarray()
    _, r, perm = la.qr(dense * weights, mode="economic", pivoting=True, check_finite=False)
    if np.min(np.abs(np.diag(r))) <= 1e-11 * max(norm(r), 1e-100):
        return None
    pivot = perm[:ne]
    free = np.sort(perm[ne:])
    try:
        lu = splu(E[:, pivot].tocsc())
        coeff = -lu.solve(E[:, free].toarray())
        if coeff.size:
            threshold = 1e-14 * np.maximum(np.max(np.abs(coeff), axis=0), 1e-100)
            coeff[np.abs(coeff) < threshold] = 0
        cr, cc = np.nonzero(coeff)
        R = sp.csc_matrix((np.r_[coeff[cr, cc], np.ones(len(free))],
                           (np.r_[pivot[cr], free], np.r_[cc, np.arange(len(free))])),
                          shape=(n, len(free)))
        keep = cones.active
        An = (A @ R)[keep].tocsc()
        if An.nnz > max(3 * A.nnz, A.nnz + 300):
            return None
        Pn = (R.T @ P @ R).tocsc()
        Pn = (Pn + Pn.T) * 0.5
        if Pn.nnz > max(4 * P.nnz, P.nnz + 500):
            return None
        x0 = np.zeros(n)
        x0[pivot] = lu.solve(b[cones.eq])
        qn = R.T @ (P @ x0 + q)
        bn = (b - A @ x0)[keep]
        if An.nnz:
            rownorm = np.asarray(abs(A).max(axis=1).toarray()).ravel()[keep]
            threshold = 1e-13 * rownorm[An.indices] * max(1.0, norm(R.data))
            An.data[np.abs(An.data) < threshold] = 0
            An.eliminate_zeros()
        Pn.eliminate_zeros()
        blocks = [{"type": kind, "dim": d} for kind, d, _ in cones.blocks if kind != "zero"]
        return Pn, qn, An, bn, blocks, R, x0, pivot, lu, keep, cones.eq
    except (RuntimeError, ValueError, la.LinAlgError):
        return None


def recover_equalities(out, reduction, P, q, A, b, cones):
    if out["status"] == "unresolved":
        return None
    _, _, _, _, _, R, x0, pivot, lu, keep, eliminated = reduction
    out = dict(out)
    if "x" in out:
        out["x"] = R @ out["x"]
        if out["status"] == "solved":
            out["x"] += x0
    if "s" in out:
        s = np.zeros(len(b))
        s[keep] = out["s"]
        out["s"] = s
    if "z" in out:
        z = np.zeros(len(b))
        z[keep] = out["z"]
        g = A.T @ z
        if out["status"] == "solved":
            g += P @ out["x"] + q
        z[eliminated] = lu.solve(-g[pivot], trans="T")
        out["z"] = z
    return out if checked(P, q, A, b, cones, out, 9.5e-9) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = {"status": "unresolved"}
    try:
        P, q, A, b, blocks = load(args.input)
        out = solve_model(P, q, A, b, blocks)
    except Exception as e:
        log("Solver exception", type(e).__name__, str(e))
        if VERBOSE:
            import traceback
            traceback.print_exc()
    save(args.output, out)


def solve_model(P, q, A, b, blocks, depth=0):
    P, A = sp.csc_matrix(P), sp.csc_matrix(A)
    q, b = np.asarray(q, dtype=float), np.asarray(b, dtype=float)
    cones = Cones(blocks, len(b))
    inactive = (np.diff(A.indptr) == 0) & (np.diff(P.indptr) == 0)
    free = inactive & (q != 0)
    if np.any(free):
        x = np.zeros(len(q))
        x[free] = -np.sign(q[free])
        out = {"status": "dual_infeasible", "x": x, "s": np.zeros(len(b))}
        if checked(P, q, A, b, cones, out):
            return out
    reduced, certificate = cone_presolve(A, b, cones)
    if certificate is not None and checked(P, q, A, b, cones, certificate):
        return certificate
    A1, b1, blocks1 = (A, b, blocks) if reduced is None else reduced[:3]
    keep = ~(inactive & (q == 0))
    P1, q1 = P, q
    if not np.all(keep):
        P1, q1, A1 = P[keep][:, keep].tocsc(), q[keep], A1[:, keep].tocsc()
    c1 = cones if reduced is None else Cones(blocks1, len(b1))
    eqred = equality_presolve(P1, q1, A1, b1, c1) if depth < 3 else None
    out = None
    if eqred is not None:
        child = solve_model(*eqred[:5], depth=depth + 1)
        out = recover_equalities(child, eqred, P1, q1, A1, b1, c1)
    if out is None:
        if not c1.soc and not c1.psd and not c1.exp.size:
            out = polyhedral(P1, q1, A1, b1, c1)
        else:
            if c1.soc and os.environ.get("SOC_LS", "1") != "0":
                from soc_ls import solve as solve_soc_ls
                out = solve_soc_ls(P1, q1, A1, b1, c1, la,
                                   lambda result: checked(P1, q1, A1, b1, c1, result))
            if c1.exp.size and os.environ.get("GP_DIRECT", "1") != "0":
                from gp import solve as solve_gp
                out = solve_gp(P1, q1, A1, b1, c1, la,
                               lambda result: checked(P1, q1, A1, b1, c1, result))
            if out is None:
                out = interior(P1, q1, A1, b1, c1)
    if out["status"] == "unresolved":
        return out
    if "x" in out and not np.all(keep):
        xx = np.zeros(len(q))
        xx[keep] = out["x"]
        out["x"] = xx
    if reduced is not None:
        _, _, _, T, F, offset = reduced
        if "z" in out:
            out["z"] = T.T @ out["z"]
        if "s" in out:
            out["s"] = F @ out["s"]
            if out["status"] == "solved":
                out["s"] += offset
    if checked(P, q, A, b, cones, out, 9.5e-9):
        return out
    if reduced is not None:
        log("Presolve recovery did not meet original-coordinate tolerance")
        return interior(P, q, A, b, cones)
    return {"status": "unresolved"}


if __name__ == "__main__":
    main()
