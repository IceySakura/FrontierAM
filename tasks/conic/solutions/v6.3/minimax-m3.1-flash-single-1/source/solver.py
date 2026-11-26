#!/usr/bin/env python3
"""Primal-dual interior-point conic solver.

Solves
    min_{x,s}  1/2 x' P x + q' x + c0
    s.t.       A x + s = b,   s in K

where K is a product of zero / nonnegative / second-order / exponential /
semidefinite cones.  The Newton system uses the dual log-barrier of each
cone, which gives a single symmetric quasi-definite system in (x, z):

    [ P    A' ] [dx]   [ -r_d    ]
    [ A   -W  ] [dz] = [ r_p+r_c ]

with W = mu * Hess(dual barrier)(z) and r_c = s + mu*grad(dual barrier)(z).
The exponential cone is handled through the linear map T(z) = (-z2,-z1,e z3)
which maps the dual cone onto the primal cone.
"""

import json
import math
import os
import sys
import time

import numpy as np

START = time.time()
SQ2 = math.sqrt(2.0)
E_ = math.e


# --------------------------------------------------------------------------
# cone implementations
# --------------------------------------------------------------------------
class ZeroCone:
    kind = "zero"

    def __init__(self, dim):
        self.dim = dim
        self.rows = dim
        self.degree = 0

    def primal_interior(self, v):
        return np.zeros(self.rows)

    def dual_grad(self, z):
        return np.zeros(self.rows)

    def dual_hess(self, z):
        return None  # zero block

    def dual_barrier_scale(self, s, z):
        return 0.0

    def step(self, s, ds):
        return np.inf

    def dual_step(self, z, dz):
        return np.inf

    def pairing(self, s, z):
        return 0.0

    def defect(self, v):
        return float(np.max(np.abs(v))) if v.size else 0.0

    def dual_defect(self, v):
        return 0.0


class NonNegCone:
    kind = "nonnegative"

    def __init__(self, dim):
        self.dim = dim
        self.rows = dim
        self.degree = dim

    def primal_interior(self, v):
        return np.maximum(v, 0.0) + 1.0

    def dual_grad(self, z):
        return -1.0 / z

    def dual_hess(self, z):
        return 1.0 / (z * z)  # diagonal

    def dual_barrier_scale(self, s, z):
        return 1.0

    def step(self, s, ds):
        neg = ds < 0
        if not np.any(neg):
            return np.inf
        return float(np.min(-s[neg] / ds[neg]))

    def dual_step(self, z, dz):
        neg = dz < 0
        if not np.any(neg):
            return np.inf
        return float(np.min(-z[neg] / dz[neg]))

    def pairing(self, s, z):
        return float(np.dot(s, z))

    def defect(self, v):
        return float(max(0.0, -np.min(v))) if v.size else 0.0

    def dual_defect(self, v):
        return self.defect(v)


class SocCone:
    kind = "soc"

    def __init__(self, dim):
        self.dim = dim
        self.rows = dim
        self.degree = 2

    def primal_interior(self, v):
        t = v[0]
        nv = float(np.linalg.norm(v[1:]))
        out = v.copy()
        out[0] = max(t, nv) + 1.0
        return out

    def dual_grad(self, z):
        tau = z[0]
        u = z[1:]
        d = tau * tau - float(np.dot(u, u))
        g = np.empty(self.rows)
        g[0] = -2.0 * tau / d
        g[1:] = 2.0 * u / d
        return g

    def dual_hess(self, z):
        # Scaling block of the second-order cone.  Using the arrow matrix
        # Psi(z) = [[tau, u'],[u, tau I]] and G = Psi(z)^-1, the linearised
        # central-path relation  Psi(s) = mu * G  gives the operator
        #   dArrow  ->  G * dArrow * G ,
        # which is the vectorised analogue of the Z^-1 dZ Z^-1 block used for
        # the semidefinite cone.
        tau = z[0]
        u = z[1:]
        d = tau * tau - float(np.dot(u, u))
        d = max(d, 1e-300)
        a = tau / d
        b = -u / d
        h = np.zeros((self.rows, self.rows))
        h[0, 0] = a * a + float(np.dot(b, b))
        h[0, 1:] = 2.0 * a * a * b
        h[1:, 0] = 2.0 * a * a * b
        h[1:, 1:] = a * a * np.eye(self.rows - 1) + np.outer(b, b)
        return h

    def dual_barrier_scale(self, s, z):
        return 1.0

    def step(self, s, ds):
        t, v = s[0], s[1:]
        dt, dv = ds[0], ds[1:]
        a = dt * dt - float(np.dot(dv, dv))
        b = 2.0 * (t * dt - float(np.dot(v, dv)))
        c = t * t - float(np.dot(v, v))
        if c <= 0:
            return 0.0
        roots = []
        if abs(a) < 1e-300:
            if abs(b) > 1e-300:
                roots = [-c / b]
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        pos = [r for r in roots if r > 0]
        if not pos:
            return np.inf
        return float(min(pos))

    def dual_step(self, z, dz):
        return self.step(z, dz)

    def pairing(self, s, z):
        return float(np.dot(s, z))

    def defect(self, v):
        if v.size == 0:
            return 0.0
        nrm = float(np.max(np.abs(v)))
        u = v / max(1.0, nrm)
        return float(max(0.0, np.linalg.norm(u[1:]) - u[0]))

    def dual_defect(self, v):
        return self.defect(v)


class ExpCone:
    kind = "exponential"

    def __init__(self, dim):
        self.dim = dim
        self.rows = 3
        self.degree = 1

    @staticmethod
    def _g(u, v, w):
        if v <= 0 or w <= 0:
            return -1.0
        return v * math.log(w / v) - u

    def primal_interior(self, v):
        u, v_, w = float(v[0]), float(v[1]), float(v[2])
        out = np.array(v, dtype=float)
        if v_ <= 0:
            v_ = 1.0
        if w <= 0:
            w = 1.0
        g = self._g(u, v_, w)
        if g <= 0:
            w = v_ * math.exp(min(u / v_, 50.0)) * 1.5 + 1.0
        out[0], out[1], out[2] = u, v_, w
        return out

    def _T(self, z):
        return np.array([-z[1], -z[0], E_ * z[2]])

    def dual_grad(self, z):
        y = self._T(z)
        u, v, w = y
        g = max(self._g(u, v, w), 1e-12)
        L = math.log(max(w, 1e-300) / max(v, 1e-300))
        # grad of phi(T(z)) pulled back
        dphi_u = 1.0 / g
        dphi_v = (1.0 - L) / g
        dphi_w = -v / (w * g)
        return np.array([-dphi_v, -dphi_u, E_ * dphi_w])

    def dual_hess(self, z):
        y = self._T(z)
        u, v, w = y
        g = self._g(u, v, w)
        g = max(g, 1e-12)
        L = math.log(max(w, 1e-300) / max(v, 1e-300))
        # hessian of phi(u,v,w) = -log(g)
        dg = np.array([-1.0, L - 1.0, v / w])
        d2g = np.zeros((3, 3))
        d2g[1, 1] = -1.0 / v
        d2g[1, 2] = 1.0 / w
        d2g[2, 1] = 1.0 / w
        d2g[2, 2] = -v / (w * w)
        hphi = -d2g / g + np.outer(dg, dg) / (g * g)
        D = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, E_]])
        return D @ hphi @ D

    def dual_barrier_scale(self, s, z):
        return 1.0

    def _inside(self, p):
        u, v, w = float(p[0]), float(p[1]), float(p[2])
        if v <= 0 or w <= 0:
            return False
        return self._g(u, v, w) > 0

    def step(self, s, ds):
        a = 1.0
        for _ in range(80):
            if self._inside(s + a * ds):
                return a
            a *= 0.5
        return 0.0

    def dual_step(self, z, dz):
        a = 1.0
        for _ in range(80):
            if self._inside(self._T(z) + a * self._T(dz)):
                return a
            a *= 0.5
        return 0.0

    def pairing(self, s, z):
        return float(np.dot(s, z))

    @staticmethod
    def _E(a, b, c):
        B = max(abs(b), max(a, 0.0), max(-c, 0.0))
        if b > 0 and c > 0:
            t = a - b * (math.log(c) - math.log(b))
            return min(B, max(t, 0.0))
        return B

    def defect(self, v):
        return self._E(float(v[0]), float(v[1]), float(v[2]))

    def dual_defect(self, v):
        return self._E(-float(v[1]), -float(v[0]), E_ * float(v[2]))


class PsdCone:
    kind = "psd"

    def __init__(self, dim):
        self.dim = dim
        self.d = dim
        self.rows = dim * (dim + 1) // 2
        self.degree = dim
        self._idx = self._build_idx(dim)
        pos = []
        for j in range(dim):
            for i in range(j + 1):
                pos.append((i, j))
        self._pos = np.array(pos, dtype=np.int64)

    @staticmethod
    def _build_idx(d):
        idx = np.zeros((d, d), dtype=np.int64)
        k = 0
        for j in range(d):
            for i in range(j + 1):
                idx[i, j] = k
                idx[j, i] = k
                k += 1
        return idx

    def smat(self, v):
        d = self.d
        M = np.zeros((d, d))
        flat = v[self._idx]
        M = flat.reshape(d, d).copy()
        off = ~np.eye(d, dtype=bool)
        M[off] /= SQ2
        return M

    def svec(self, M):
        i = self._pos[:, 0]
        j = self._pos[:, 1]
        out = M[i, j].copy()
        off = i != j
        out[off] *= SQ2
        return out

    def primal_interior(self, v):
        M = self.smat(v)
        w = np.linalg.eigvalsh(M)
        lam = float(np.min(w))
        scale = max(1.0, float(np.max(np.abs(w))))
        M = M + (max(0.0, -lam) + 1e-2 * scale) * np.eye(self.d)
        return self.svec(M)

    def dual_grad(self, z):
        M = self.smat(z)
        return -self.svec(np.linalg.inv(M))

    def dual_hess(self, z):
        M = self.smat(z)
        G = np.linalg.inv(M)
        d = self.d
        W = np.zeros((self.rows, self.rows))
        k = 0
        for j in range(d):
            for i in range(j + 1):
                if i == j:
                    outer = np.outer(G[:, i], G[:, i])
                else:
                    outer = (np.outer(G[:, i], G[:, j]) + np.outer(G[:, j], G[:, i])) / SQ2
                W[:, k] = self.svec(outer)
                k += 1
        return W

    def _basis(self, k):
        d = self.d
        i, j = int(self._pos[k, 0]), int(self._pos[k, 1])
        E = np.zeros((d, d))
        if i == j:
            E[i, j] = 1.0
        else:
            E[i, j] = 1.0 / SQ2
            E[j, i] = 1.0 / SQ2
        return E

    def dual_barrier_scale(self, s, z):
        return 1.0

    def step(self, s, ds):
        S = self.smat(s)
        dS = self.smat(ds)
        a = 1.0
        for _ in range(80):
            w = np.linalg.eigvalsh(S + a * dS)
            if w[0] >= -1e-12:
                return a
            a *= 0.5
        return 0.0

    def dual_step(self, z, dz):
        return self.step(z, dz)

    def pairing(self, s, z):
        return float(np.dot(s, z))

    def defect(self, v):
        M = self.smat(v)
        w = np.linalg.eigvalsh(M)
        return float(max(0.0, -w[0]))

    def dual_defect(self, v):
        return self.defect(v)


def make_cone(spec):
    t, d = spec["type"], int(spec["dim"])
    if t == "zero":
        return ZeroCone(d)
    if t == "nonnegative":
        return NonNegCone(d)
    if t == "soc":
        return SocCone(d)
    if t == "exponential":
        return ExpCone(d)
    if t == "psd":
        return PsdCone(d)
    raise ValueError("unsupported cone type " + t)


# --------------------------------------------------------------------------
# sparse matrix helpers (no scipy needed for the dense path)
# --------------------------------------------------------------------------
class Csc:
    __slots__ = ("data", "indices", "indptr", "shape")

    def __init__(self, data, indices, indptr, shape):
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))

    def to_dense(self):
        n, m = self.shape
        M = np.zeros((n, m))
        counts = np.diff(self.indptr)
        cols = np.repeat(np.arange(m), counts)
        M[self.indices, cols] = self.data
        return M

    def matvec(self, x):
        n, m = self.shape
        counts = np.diff(self.indptr)
        vals = self.data * np.repeat(x, counts)
        return np.bincount(self.indices, weights=vals, minlength=n)[:n]


def load_instance(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        P = Csc(d["P_data"], d["P_indices"], d["P_indptr"], d["P_shape"])
        A = Csc(d["A_data"], d["A_indices"], d["A_indptr"], d["A_shape"])
        q = np.asarray(d["q"], dtype=float).ravel()
        b = np.asarray(d["b"], dtype=float).ravel()
    return meta, P, A, q, b


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class Solver:
    def __init__(self, meta, P, A, q, b, deadline=None):
        self.meta = meta
        self.P = P
        self.A = A
        self.q = q
        self.b = b
        self.n = q.size
        self.m = b.size
        self.deadline = deadline
        # adaptive self-imposed budget: instances whose reference run was fast
        # get a ~1 s limit, harder instances get an 18 s limit.  Escalate the
        # budget only while progress is being made.
        base = START if deadline is None else min(deadline, START)
        self.phases = [base + 0.75, base + 2.60, base + 15.00]
        self.phase = 0
        self.cones = []
        off = 0
        for spec in meta["cones"]:
            c = make_cone(spec)
            self.cones.append((c, off))
            off += c.rows
        assert off == self.m, "cone rows do not match m"
        self.small = (self.n + self.m) <= 1500
        self.dense = None
        if self.small:
            self.Ad = A.to_dense()
            self.Pd = P.to_dense()
        else:
            import scipy.sparse as sp  # noqa
            import scipy.sparse.linalg as spla  # noqa
            self.sp = sp
            self.spla = spla
            self.As = sp.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)
            self.Ps = sp.csc_matrix((P.data, P.indices, P.indptr), shape=P.shape)
        self.status = "unresolved"
        self.x = None
        self.s = None
        self.z = None

    # -- linear algebra ----------------------------------------------------
    def _assemble(self, W):
        """Return (matrix, rhs builder) for the Newton system."""
        n, m = self.n, self.m
        if self.small:
            K = np.zeros((n + m, n + m))
            K[:n, :n] = self.Pd
            K[:n, n:] = self.Ad.T
            K[n:, :n] = self.Ad
            K[n:, n:] = -W
            return K
        sp = self.sp
        rows = []
        cols = []
        vals = []
        diag = np.zeros(n + m)
        P = self.Ps
        coo = P.tocoo()
        rows.append(coo.row)
        cols.append(coo.col)
        vals.append(coo.data)
        diag[:n] = 0.0
        A = self.As.tocoo()
        rows.append(A.row)
        cols.append(A.col + n)
        vals.append(A.data)
        rows.append(A.col + n)
        cols.append(A.row)
        vals.append(A.data)
        # W blocks
        for (c, off) in self.cones:
            if c.kind == "zero":
                continue
            h = c.dual_hess(self.z[off:off + c.rows])
            if c.kind == "nonnegative":
                vals.append(-h)
                rows.append(np.arange(off, off + c.rows))
                cols.append(np.arange(off, off + c.rows) + n)
            else:
                rr = np.repeat(np.arange(off, off + c.rows), c.rows)
                cc = np.tile(np.arange(off, off + c.rows), c.rows)
                rows.append(rr + n)
                cols.append(cc + n)
                vals.append(-h.ravel())
        K = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n + m, n + m))
        return K

    def _solve(self, W, rhs):
        n, m = self.n, self.m
        if self.small:
            K = np.zeros((n + m, n + m))
            K[:n, :n] = self.Pd
            K[:n, n:] = self.Ad.T
            K[n:, :n] = self.Ad
            K[n:, n:] = -W
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            return sol[:n], sol[n:]
        K = self._assemble(W)
        try:
            lu = self.spla.splu(K.tocsc())
            sol = lu.solve(rhs)
        except Exception:
            sol = np.linalg.lstsq(K.toarray(), rhs, rcond=None)[0]
        return sol[:n], sol[n:]

    # -- helpers ------------------------------------------------------------
    def _matvec_A(self, x):
        if self.small:
            return self.Ad @ x
        return self.As @ x

    def _matvec_AT(self, y):
        if self.small:
            return self.Ad.T @ y
        return self.As.T @ y

    def _matvec_P(self, x):
        if self.small:
            return self.Pd @ x
        return self.Ps @ x

    def _grad_blocks(self):
        """Per cone: gradient of dual barrier and hessian block."""
        grads = []
        hess = []
        for (c, off) in self.cones:
            zb = self.z[off:off + c.rows]
            grads.append(c.dual_grad(zb))
            hess.append(c.dual_hess(zb))
        return grads, hess

    def _W_block(self, hess, mus):
        """Assemble the (m x m) block diagonal matrix W = mu * hess."""
        m = self.m
        if self.small:
            W = np.zeros((m, m))
        else:
            W = None
        for i, ((c, off)) in enumerate(self.cones):
            h = hess[i]
            if c.kind == "zero":
                continue
            mu = mus[i]
            if c.kind == "nonnegative":
                d = mu * h
                if self.small:
                    W[off:off + c.rows, off:off + c.rows] = np.diag(d)
                else:
                    pass
            else:
                blk = mu * h
                if self.small:
                    W[off:off + c.rows, off:off + c.rows] = blk
        return W

    def _diag_W(self, hess, mus):
        """Diagonal-only representation (used by the sparse assembler)."""
        return None

    # -- main ---------------------------------------------------------------
    def run(self):
        n, m, A, q, b = self.n, self.m, self.A, self.q, self.b
        # ---- initialization
        x = np.zeros(n)
        try:
            if self.small:
                xls = np.linalg.lstsq(self.Ad, b, rcond=None)[0]
                x = xls
            else:
                from scipy.sparse.linalg import lsqr
                xls = lsqr(self.As, b, atol=1e-12, btol=1e-12, iter_lim=200)[0]
                x = xls
        except Exception:
            x = np.zeros(n)
        s = b - self._matvec_A(x)
        self.x = x
        self.s = s
        # push s into the cone interiors
        for (c, off) in self.cones:
            s[off:off + c.rows] = c.primal_interior(s[off:off + c.rows])
        self.s = s
        # dual variables from the primal barrier
        z = np.zeros(m)
        mu0 = 1.0
        for (c, off) in self.cones:
            sb = s[off:off + c.rows]
            if c.kind == "zero":
                z[off:off + c.rows] = 0.0
            elif c.kind == "nonnegative":
                z[off:off + c.rows] = mu0 / np.maximum(sb, 1e-12)
            elif c.kind == "soc":
                t = sb[0]
                v = sb[1:]
                D = max(t * t - float(np.dot(v, v)), 1e-12)
                z[off] = 2.0 * mu0 * t / D
                z[off + 1:off + c.rows] = -2.0 * mu0 * v / D
            elif c.kind == "exponential":
                u, vv, w = sb
                g = max(ExpCone._g(u, vv, w), 1e-12)
                L = math.log(max(w, 1e-300) / max(vv, 1e-300))
                z[off] = -mu0 / g
                z[off + 1] = mu0 * (L - 1.0) / g
                z[off + 2] = mu0 * vv / (w * g)
            else:  # psd
                M = c.smat(sb)
                try:
                    G = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    G = np.linalg.pinv(M)
                z[off:off + c.rows] = mu0 * c.svec(G)
        self.z = z

        if not np.all(np.isfinite(self.s)) or not np.all(np.isfinite(self.z)):
            self.z = np.nan_to_num(self.z, nan=1.0, posinf=1.0, neginf=-1.0)

        status = "unresolved"
        best = None
        maxit = 200
        gap_hist = []
        for it in range(maxit):
            if time.time() > self.phases[self.phase]:
                # escalate the budget only if the iteration still makes
                # progress; otherwise stop early (the instance is out of reach)
                if gap_hist and gap_hist[-1] > 0.9 * gap_hist[0]:
                    break
                self.phase += 1
                if self.phase >= len(self.phases):
                    break
            grads, hess = self._grad_blocks()
            mus = []
            for i, ((c, off)) in enumerate(self.cones):
                sb = self.s[off:off + c.rows]
                zb = self.z[off:off + c.rows]
                pair = c.pairing(sb, zb)
                if c.degree > 0:
                    mus.append(max(pair / c.degree, 1e-14))
                else:
                    mus.append(0.0)
            # residuals
            Ax = self._matvec_A(self.x)
            r_p = self.b - Ax - self.s
            r_d = self._matvec_P(self.x) + q + self._matvec_AT(self.z)
            # ---- certificate checks
            cert = self._check_certs()
            if cert is not None:
                return cert
            # ---- affine (predictor) step
            W = self._W_block(hess, mus)
            rhs_aff = np.concatenate([-r_d, r_p + self.s])
            dx, dz = self._solve(W, rhs_aff)
            ds = r_p - self._matvec_A(dx)
            a_s = self._max_step(self.s, ds)
            a_z = self._max_step(self.z, dz)
            a_aff = min(1.0, 0.99 * a_s, 0.99 * a_z)
            # predicted mu
            s_aff = self.s + a_aff * ds
            z_aff = self.z + a_aff * dz
            sigma = []
            for i, ((c, off)) in enumerate(self.cones):
                if c.degree == 0:
                    sigma.append(0.0)
                    continue
                sb = s_aff[off:off + c.rows]
                zb = z_aff[off:off + c.rows]
                mu_aff = c.pairing(sb, zb) / c.degree
                r = mu_aff / mus[i] if mus[i] > 0 else 0.0
                sigma.append(min(1.0, max(1e-8, r ** 3)))
            mu_new = [mus[i] * sigma[i] for i in range(len(mus))]
            # ---- corrector (with the second order term of the affine step)
            r_c = np.zeros(m)
            for i, ((c, off)) in enumerate(self.cones):
                if c.kind == "zero":
                    continue
                r_c[off:off + c.rows] = self.s[off:off + c.rows] + mu_new[i] * grads[i]
            # second order correction: difference between the true residual of
            # the complementarity function at the affine point and its linear
            # model (Mehrotra's corrector).
            for i, ((c, off)) in enumerate(self.cones):
                if c.kind == "zero" or mu_new[i] <= 0:
                    continue
                sl = slice(off, off + c.rows)
                z_aff = self.z[sl] + a_aff * dz[sl]
                g2 = c.dual_grad(z_aff)
                hz = hess[i]
                if c.kind == "nonnegative":
                    lin = hz * dz[sl]
                else:
                    lin = hz @ dz[sl]
                true = self.s[sl] + a_aff * ds[sl] + mu_new[i] * g2
                lin_model = r_c[sl] + a_aff * ds[sl] + mu_new[i] * lin
                r_c[sl] = r_c[sl] + (true - lin_model)
            Wc = self._W_block(hess, mus)
            rhs = np.concatenate([-r_d, r_p + r_c])
            dx, dz = self._solve(Wc, rhs)
            ds = r_p - self._matvec_A(dx)
            a_s = self._max_step(self.s, ds)
            a_z = self._max_dual_step(self.z, dz)
            a_p = min(1.0, 0.99 * a_s)
            a_d = min(1.0, 0.99 * a_z)
            if a_p <= 1e-12 and a_d <= 1e-12:
                break
            self.x = self.x + a_p * dx
            self.s = self.s + a_p * ds
            self.z = self.z + a_d * dz
            if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.s))
                    and np.all(np.isfinite(self.z))):
                break
            gap_hist.append(float(self.s @ self.z))
            if len(gap_hist) > 8:
                gap_hist.pop(0)
            if self._check_converged():
                status = "solved"
                break
        # final certificate / convergence check
        cert = self._check_certs()
        if cert is not None:
            return cert
        if self._check_converged():
            status = "solved"
        self.status = status
        return status

    def _max_step(self, v, dv):
        a = np.inf
        for (c, off) in self.cones:
            if c.kind == "zero":
                continue
            st = c.step(v[off:off + c.rows], dv[off:off + c.rows])
            if st < a:
                a = st
        return a

    def _max_dual_step(self, v, dv):
        a = np.inf
        for (c, off) in self.cones:
            if c.kind == "zero":
                continue
            st = c.dual_step(v[off:off + c.rows], dv[off:off + c.rows])
            if st < a:
                a = st
        return a

    def _cone_defects(self, primal=True):
        # exactly the normalisation used by the checker:
        #   u = v_block / max(1, ||v_block||_inf)
        worst = 0.0
        for (c, off) in self.cones:
            vb = self.s[off:off + c.rows] if primal else self.z[off:off + c.rows]
            if vb.size == 0:
                continue
            nrm = float(np.linalg.norm(vb, np.inf))
            u = vb / max(1.0, nrm)
            d = c.defect(u) if primal else c.dual_defect(u)
            if d > worst:
                worst = d
        return worst

    def _check_converged(self):
        Ax = self._matvec_A(self.x)
        r_p = float(np.linalg.norm(Ax + self.s - self.b))
        scale_p = 1.0 + max(float(np.linalg.norm(Ax, np.inf)),
                            float(np.linalg.norm(self.s, np.inf)),
                            float(np.linalg.norm(self.b, np.inf)))
        if r_p / scale_p > 1e-9:
            return False
        Px = self._matvec_P(self.x)
        ATz = self._matvec_AT(self.z)
        r_d = float(np.linalg.norm(Px + self.q + ATz))
        scale_d = 1.0 + max(float(np.linalg.norm(Px, np.inf)),
                            float(np.linalg.norm(self.q, np.inf)),
                            float(np.linalg.norm(ATz, np.inf)))
        if r_d / scale_d > 1e-9:
            return False
        if self._cone_defects(True) > 1e-9:
            return False
        if self._cone_defects(False) > 1e-9:
            return False
        p0 = 0.5 * float(self.x @ (self._matvec_P(self.x))) + float(self.q @ self.x)
        d0 = -0.5 * float(self.x @ (self._matvec_P(self.x))) - float(self.b @ self.z)
        gap = abs(p0 - d0)
        if gap > 1e-9 and gap > 1e-9 * max(abs(p0), abs(d0)):
            return False
        return True

    def _check_certs(self):
        """Try to certify primal / dual infeasibility."""
        # primal infeasibility: z with A'z ~ 0, b'z < 0, z in K*
        zn = float(np.linalg.norm(self.z))
        bn = float(np.linalg.norm(self.b))
        if zn > 0 and bn > 0:
            zh = self.z / zn
            beta = -float((self.b / bn) @ zh)
            if beta > 1e-8:
                a = self._max_abs_A()
                ATzh = self._matvec_AT(zh)
                if a > 0:
                    r = float(np.linalg.norm(ATzh)) / (a * beta)
                else:
                    r = 0.0
                d = self._dual_defect_scaled(zh)
                if r <= 1e-9 and d <= 1e-9 * beta:
                    self.status = "primal_infeasible"
                    return "primal_infeasible"
        # dual infeasibility: x with Px ~ 0, Ax + s ~ 0, s in K, q'x < 0
        xn = float(np.linalg.norm(self.x))
        qn = float(np.linalg.norm(self.q))
        if xn > 0 and qn > 0:
            xh = self.x / xn
            sh = self.s / xn
            alpha = -float((self.q / qn) @ xh)
            if alpha > 1e-8:
                h = self._max_abs_P()
                a = self._max_abs_A()
                r1 = (float(np.linalg.norm(self._matvec_P(xh))) / (h * alpha)) if h > 0 else 0.0
                if a > 0:
                    r2 = float(np.linalg.norm(self._matvec_A(xh) + sh / a)) / alpha
                    sv = sh / a
                else:
                    r2 = 0.0
                    sv = sh
                d = self._primal_defect_scaled(sv)
                if r1 <= 1e-9 and r2 <= 1e-9 and d <= 1e-9 * alpha:
                    self.status = "dual_infeasible"
                    return "dual_infeasible"
        return None

    def _dual_defect_scaled(self, v):
        worst = 0.0
        for (c, off) in self.cones:
            vb = v[off:off + c.rows]
            nrm = float(np.linalg.norm(vb, np.inf))
            u = vb / max(1.0, nrm)
            d = c.dual_defect(u)
            worst = max(worst, d)
        return worst

    def _primal_defect_scaled(self, v):
        worst = 0.0
        for (c, off) in self.cones:
            vb = v[off:off + c.rows]
            nrm = float(np.linalg.norm(vb, np.inf))
            u = vb / max(1.0, nrm)
            d = c.defect(u)
            worst = max(worst, d)
        return worst

    def _max_abs_A(self):
        if self.small:
            return float(np.max(np.abs(self.Ad))) if self.Ad.size else 1.0
        return float(np.max(np.abs(self.A.data))) if self.A.data.size else 1.0

    def _max_abs_P(self):
        if self.P.data.size == 0:
            return 1.0
        return float(np.max(np.abs(self.P.data)))


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_result(path, status, x=None, s=None, z=None, p0=None, d0=None,
                 objective_offset=0.0):
    meta = {"schema_version": 1, "status": status}
    if status == "solved" and p0 is not None and d0 is not None:
        meta["objective_primal"] = float(p0 + objective_offset)
        meta["objective_dual"] = float(d0 + objective_offset)
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass
    out = {"metadata_json": np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False))}
    if status == "solved":
        out["x"] = np.asarray(x, dtype=float).ravel()
        out["s"] = np.asarray(s, dtype=float).ravel()
        out["z"] = np.asarray(z, dtype=float).ravel()
    elif status == "primal_infeasible":
        out["z"] = np.asarray(z, dtype=float).ravel()
    elif status == "dual_infeasible":
        out["x"] = np.asarray(x, dtype=float).ravel()
        out["s"] = np.asarray(s, dtype=float).ravel()
    np.savez(path, **out)


def main(argv):
    inp = outp = None
    i = 1
    while i < len(argv):
        if argv[i] == "--input":
            inp = argv[i + 1]
            i += 2
        elif argv[i] == "--output":
            outp = argv[i + 1]
            i += 2
        else:
            i += 1
    if inp is None or outp is None:
        sys.stderr.write("usage: solver.py --input IN --output OUT\n")
        return 2
    limit = float(os.environ.get("SOLVER_TIME_LIMIT", "0") or 0)
    deadline = (START + limit * 0.92) if limit > 0 else None
    try:
        meta, P, A, q, b = load_instance(inp)
        solver = Solver(meta, P, A, q, b, deadline)
        status = solver.run()
    except Exception as exc:  # pragma: no cover
        sys.stderr.write("solver error: %r\n" % (exc,))
        write_result(outp, "unresolved")
        return 0
    c0 = float(meta.get("objective_offset", 0.0) or 0.0)
    if status == "solved":
        Px = solver._matvec_P(solver.x)
        p0 = 0.5 * float(solver.x @ Px) + float(solver.q @ solver.x)
        d0 = -0.5 * float(solver.x @ Px) - float(solver.b @ solver.z)
        write_result(outp, "solved", solver.x, solver.s, solver.z, p0, d0, c0)
    elif status == "primal_infeasible":
        write_result(outp, "primal_infeasible", z=solver.z)
    elif status == "dual_infeasible":
        write_result(outp, "dual_infeasible", x=solver.x, s=solver.s)
    else:
        write_result(outp, "unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
