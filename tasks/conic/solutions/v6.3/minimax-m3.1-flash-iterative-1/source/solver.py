#!/usr/bin/env python3
"""Conic interior-point solver for

    minimize    1/2 x' P x + q' x + c0
    subject to  A x + s = b,   s in K

K is a product of zero / non-negative / second-order / semidefinite /
exponential cones.  Pure Python + NumPy/SciPy, single threaded.

Usage:  solver.py --input INPUT.npz --output OUTPUT.npz
"""

import json
import math
import os
import sys
import time

# the clock starts before the (comparatively expensive) NumPy import so that
# the internal time budget covers the whole process
T0 = time.time()

import numpy as np

# SciPy's import pulls in numpy.testing (only used by its own test helpers);
# replacing it with a stub saves ~0.15 s of process start-up, which matters
# because most instances have a one second budget.
try:
    import types as _types

    def _np_testing_stub(*args, **kwargs):
        return None

    _stub = _types.ModuleType("numpy.testing")
    _stub.__getattr__ = lambda name: _np_testing_stub
    sys.modules["numpy.testing"] = _stub
    np.testing = _stub
except Exception:
    pass

SQRT2 = math.sqrt(2.0)

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def psd_smat(v, d):
    """svec (upper triangle, column wise, off diagonal * sqrt2) -> dense matrix."""
    M = np.zeros((d, d))
    pos = 0
    for j in range(d):
        vals = v[pos:pos + j + 1].copy()
        vals[:j] /= SQRT2
        M[:j + 1, j] = vals
        M[j, :j + 1] = vals
        pos += j + 1
    return M


def psd_svec(M):
    """dense symmetric matrix -> svec."""
    d = M.shape[0]
    v = np.empty(d * (d + 1) // 2)
    pos = 0
    for j in range(d):
        vals = M[:j + 1, j].copy()
        vals[:j] *= SQRT2
        v[pos:pos + j + 1] = vals
        pos += j + 1
    return v


def _as_float_array(a):
    return np.asarray(a, dtype=np.float64)


# ----------------------------------------------------------------------------
# problem container
# ----------------------------------------------------------------------------


def csc_rows(indices, indptr, ncols):
    """Column index of every stored entry of a CSC matrix."""
    counts = np.diff(indptr)
    return np.repeat(np.arange(ncols, dtype=np.int64), counts)


def csc_to_dense(data, indices, indptr, shape):
    rows, cols = shape
    D = np.zeros((rows, cols))
    if len(data):
        counts = np.diff(indptr)
        rowidx = np.repeat(np.arange(cols), counts)
        np.add.at(D, (indices.astype(np.int64), rowidx), data)
    return D

def _dense_from_raw(data, indices, indptr, shape):
    """Dense matrix from the raw CSC arrays."""
    D = np.zeros(shape)
    cols = np.repeat(np.arange(shape[1]), np.diff(indptr))
    np.add.at(D, (indices, cols), data)
    return D


def raw_from_dense(D):
    """CSC arrays (data, rows, cols, indptr, shape) of a dense matrix."""
    D = np.asarray(D, dtype=np.float64)
    rows, cols = np.nonzero(D)
    order = np.argsort(cols, kind="stable")
    rows = rows[order]
    cols = cols[order]
    data = D[rows, cols]
    counts = np.bincount(cols, minlength=D.shape[1])
    indptr = np.concatenate(([0], np.cumsum(counts)))
    return data, rows.astype(np.int64), cols.astype(np.int64), indptr.astype(np.int64), D.shape


def _problem_from_dense(P, q, A, b, cones):
    """Build a Problem from dense matrices (no SciPy import needed)."""
    raw = {}
    for name, D in (("P", P), ("A", A)):
        data, rows, cols, indptr, shape = raw_from_dense(D)
        raw[name + "_data"] = data
        raw[name + "_indices"] = rows
        raw[name + "_indptr"] = indptr
        raw[name + "_shape"] = shape
    return Problem(None, q, None, b, cones, 0.0, raw=raw)


class Problem(object):
    """Container for the conic problem; keeps the raw CSC arrays so that
    small instances can be handled without importing SciPy."""

    def __init__(self, P, q, A, b, cones, offset=0.0, raw=None):
        self.q = _as_float_array(q).reshape(-1)
        self.b = _as_float_array(b).reshape(-1)
        self.n = self.q.size
        self.m = self.b.size
        self.cones = cones
        self.offset = float(offset)
        self.raw = raw
        self._P = None
        self._A = None
        if raw is None:
            from scipy import sparse
            self._P = sparse.csc_matrix(P)
            self._A = sparse.csc_matrix(A)

    @property
    def P(self):
        if self._P is None:
            from scipy import sparse
            self._P = sparse.csc_matrix((self.raw["P_data"], self.raw["P_indices"],
                                         self.raw["P_indptr"]), shape=self.raw["P_shape"])
        return self._P

    @property
    def A(self):
        if self._A is None:
            from scipy import sparse
            self._A = sparse.csc_matrix((self.raw["A_data"], self.raw["A_indices"],
                                         self.raw["A_indptr"]), shape=self.raw["A_shape"])
        return self._A

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            raw = {}
            for prefix in ("P", "A"):
                raw[prefix + "_data"] = np.asarray(data[f"{prefix}_data"], dtype=np.float64)
                raw[prefix + "_indices"] = np.asarray(data[f"{prefix}_indices"], dtype=np.int64)
                raw[prefix + "_indptr"] = np.asarray(data[f"{prefix}_indptr"], dtype=np.int64)
                raw[prefix + "_shape"] = tuple(int(v) for v in data[f"{prefix}_shape"])
            q = _as_float_array(data["q"]).reshape(-1)
            b = _as_float_array(data["b"]).reshape(-1)
        return Problem(None, q, None, b, meta["cones"],
                       meta.get("objective_offset", 0.0), raw=raw)


# ----------------------------------------------------------------------------
# cone block descriptors
# ----------------------------------------------------------------------------


class Block(object):
    __slots__ = ("kind", "dim", "row0", "width")

    def __init__(self, kind, dim, row0):
        self.kind = kind
        self.dim = dim
        self.row0 = row0
        self.width = dim * (dim + 1) // 2 if kind == "psd" else dim


def build_blocks(cones, m):
    blocks = []
    row = 0
    for c in cones:
        kind = c["type"]
        dim = int(c["dim"])
        b = Block(kind, dim, row)
        row += b.width
        blocks.append(b)
    if row != m:
        raise ValueError("cone blocks do not cover m rows")
    return blocks


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------


class Solver(object):
    def __init__(self, prob, deadline=None, max_iter=200, verbose=False):
        self.prob = prob
        self.b = prob.b
        self.n = prob.n
        self.m = prob.m
        # decide whether a pure NumPy (dense) path is affordable; this avoids
        # the ~0.45 s import of SciPy for the many small instances.
        # The objective is scaled by a positive constant: this does not change
        # the minimiser but keeps the Newton steps balanced with respect to the
        # constraint data, which may live on a very different scale.
        qscale = float(np.max(np.abs(prob.q), initial=0.0))
        pscale = 0.0
        if prob.raw is not None:
            pscale = float(np.max(np.abs(prob.raw["P_data"]), initial=0.0))
        self.obj_scale = 1.0 / max(1.0, qscale, pscale)
        self.q = prob.q * self.obj_scale
        self.dense = False
        self.Ad = None
        self.Pd = None
        self.Arow = None
        self.Acol = None
        self.Aval = None
        self.Prow = None
        self.Pcol = None
        self.Pval = None
        try:
            raw = prob.raw
            if raw is not None:
                mA, nA = raw["A_shape"]
                mP, nP = raw["P_shape"]
                nz = len(raw["A_indices"]) + len(raw["P_indices"])
                # the dense (NumPy only) path avoids the ~0.3 s SciPy import and
                # pays off while the quadratic algebra stays affordable
                if nA <= 1200 and mA * nA <= 4e6 \
                        and mA * nA * nA <= 4e8 \
                        and mP * nP <= 4e6 \
                        and nz <= 4e6 and nz * max(nA, 1) <= 4e8 \
                        and os.environ.get("CONIC_FORCE_SPARSE") != "1":
                    self.Arow = raw["A_indices"].astype(np.int64)
                    self.Acol = csc_rows(raw["A_indices"], raw["A_indptr"], nA)
                    self.Aval = raw["A_data"]
                    self.Prow = raw["P_indices"].astype(np.int64)
                    self.Pcol = csc_rows(raw["P_indices"], raw["P_indptr"], nP)
                    self.Pval = raw["P_data"] * self.obj_scale
                    self._A_indptr = raw["A_indptr"]
                    # row grouped structure of A with duplicate entries summed
                    keys = self.Arow * nA + self.Acol
                    uniq, inv = np.unique(keys, return_inverse=True)
                    vals = np.bincount(inv, weights=self.Aval)
                    rows_u = uniq // nA
                    cols_u = uniq % nA
                    order = np.argsort(rows_u, kind="stable")
                    self._A_csr_vals = vals[order]
                    self._A_csr_cols = cols_u[order]
                    counts = np.bincount(rows_u, minlength=mA)
                    self._A_csr_indptr = np.concatenate(([0], np.cumsum(counts)))
                    self.dense = True
        except Exception as exc:
            if os.environ.get("CONIC_DEBUG_DENSE"):
                import traceback
                traceback.print_exc()
            self.dense = False
        if self.dense:
            self.P = None
            self.A = None
            self.PT = None
            self.AT = None
            self.Pd = None
            if prob.raw is not None:
                self.Pd = csc_to_dense(prob.raw["P_data"] * self.obj_scale,
                                       prob.raw["P_indices"], prob.raw["P_indptr"],
                                       prob.raw["P_shape"])
        else:
            from scipy import sparse
            self.P = sparse.csr_matrix(prob.P)
            self.P.data = self.P.data * self.obj_scale
            self.A = sparse.csr_matrix(prob.A)
            self.PT = sparse.csr_matrix(prob.P.T)
            self.AT = sparse.csr_matrix(prob.A.T)
        self.blocks = build_blocks(prob.cones, prob.m)
        self.deadline = deadline
        self.max_iter = max_iter
        self.verbose = verbose
        self.status = "unresolved"

        # rows grouped by cone type
        self.zero_rows = []
        self.nn_rows = []
        self.soc_blocks = []
        self.exp_blocks = []
        self.psd_blocks = []
        for b in self.blocks:
            if b.kind == "zero":
                self.zero_rows.append(b)
            elif b.kind == "nonnegative":
                self.nn_rows.append(b)
            elif b.kind == "soc":
                self.soc_blocks.append(b)
            elif b.kind == "exponential":
                self.exp_blocks.append(b)
            elif b.kind == "psd":
                self.psd_blocks.append(b)
        zr = np.concatenate([np.arange(b.row0, b.row0 + b.width)
                             for b in self.zero_rows]) if self.zero_rows \
            else np.zeros(0, dtype=int)
        # rows whose constraint is trivially satisfied (zero matrix row and zero
        # right hand side) carry no information and would make the saddle point
        # system singular, so they are dropped.
        if self.dense:
            row_nz = np.zeros(self.m, dtype=bool)
            if len(self.Arow):
                row_nz[self.Arow] = True
        else:
            row_nz = np.asarray(self.A.getnnz(axis=1) > 0).ravel()
        keep = []
        for r in zr:
            if not (not row_nz[r] and self.b[r] == 0.0):
                keep.append(r)
        self.zero_row_idx = np.array(keep, dtype=int)
        self.nn_row_idx = np.concatenate([np.arange(b.row0, b.row0 + b.width)
                                          for b in self.nn_rows]) \
            if self.nn_rows else np.zeros(0, dtype=int)
        # dense A sub matrices needed by the non-diagonal cones
        if self.dense:
            self.A_soc = [self._dense_block(b.row0, b.width) for b in self.soc_blocks]
            self.A_exp = [self._dense_block(b.row0, b.width) for b in self.exp_blocks]
            self.A_psd = [self._dense_block(b.row0, b.width) for b in self.psd_blocks]
            self._A0_dense = (self._dense_rows(self.zero_row_idx)
                              if self.zero_row_idx.size else None)
        else:
            self.A_soc = [self.A[b.row0:b.row0 + b.width, :].toarray() for b in self.soc_blocks]
            self.A_exp = [self.A[b.row0:b.row0 + b.width, :].toarray() for b in self.exp_blocks]
            self.A_psd = [self.A[b.row0:b.row0 + b.width, :].toarray() for b in self.psd_blocks]

    # -- matrix helpers -----------------------------------------------------
    def _dense_block(self, row0, width):
        """Dense (width x n) block of A starting at row row0."""
        D = np.zeros((width, self.n))
        rows = self.Arow
        sel = (rows >= row0) & (rows < row0 + width)
        if np.any(sel):
            np.add.at(D, (rows[sel] - row0, self.Acol[sel]), self.Aval[sel])
        return D

    def _dense_A(self):
        """Dense copy of A (used for the least squares start only)."""
        D = np.zeros((self.m, self.n))
        np.add.at(D, (self.Arow, self.Acol), self.Aval)
        return D

    def _dense_A_cached(self):
        """Dense copy of A, cached across iterations."""
        D = getattr(self, "_Adense", None)
        if D is None:
            D = self._dense_A()
            self._Adense = D
        return D

    def _dense_rows(self, row_idx):
        """Dense (len(row_idx) x n) sub matrix of A for the given rows."""
        D = np.zeros((len(row_idx), self.n))
        rows = self.Arow
        pos = np.full(self.m, -1, dtype=np.int64)
        pos[row_idx] = np.arange(len(row_idx))
        sel = pos[rows] >= 0
        if np.any(sel):
            np.add.at(D, (pos[rows[sel]], self.Acol[sel]), self.Aval[sel])
        return D

    def mat_A(self, v):
        if self.dense:
            return np.bincount(self.Arow, weights=self.Aval * v[self.Acol],
                               minlength=self.m)
        return self.A @ v

    def mat_AT(self, v):
        if self.dense:
            return np.bincount(self.Acol, weights=self.Aval * v[self.Arow],
                               minlength=self.n)
        return self.AT @ v

    def mat_P(self, v):
        if self.dense:
            return np.bincount(self.Prow, weights=self.Pval * v[self.Pcol],
                               minlength=self.n)
        return self.P @ v

    # -- time ---------------------------------------------------------------
    def out_of_time(self, reserve=0.05):
        return self.deadline is not None and time.time() > self.deadline - reserve

    # -- initial points -----------------------------------------------------
    def initial_points(self):
        n, m = self.n, self.m
        x = np.zeros(n)
        # least squares starting point for x (helps a lot on well posed problems)
        q = self.q
        if self.dense:
            try:
                if len(self.Pval):
                    M = np.zeros((n, n))
                    np.add.at(M, (self.Prow, self.Pcol), self.Pval)
                    Mp = M.T @ M
                    rhs = -(M.T @ q)
                    x = np.linalg.solve(Mp + 1e-12 * np.eye(n), rhs)
                elif len(self.Aval) and self.m * n * n <= 1e8:
                    # least squares starting point
                    Ad = self._dense_A()
                    x = np.linalg.lstsq(Ad, self.b, rcond=None)[0]
                    del Ad
                elif len(self.Aval):
                    Mp = self.mat_AT(self.mat_A(np.eye(n)))
                    rhs = self.mat_AT(self.b)
                    x = np.linalg.solve(Mp + 1e-12 * np.eye(n), rhs)
            except Exception:
                x = np.zeros(n)
        else:
            if self.P.nnz:
                try:
                    from scipy.sparse.linalg import lsqr
                    sol = lsqr(self.P, -q, atol=1e-12, btol=1e-12, iter_lim=10)
                    x = sol[0]
                except Exception:
                    x = np.zeros(n)
            elif self.A.nnz:
                try:
                    from scipy.sparse.linalg import lsqr
                    sol = lsqr(self.A, self.b, atol=1e-12, btol=1e-12, iter_lim=10)
                    x = sol[0]
                except Exception:
                    x = np.zeros(n)
        if not np.all(np.isfinite(x)):
            x = np.zeros(n)

        scale = max(1.0, float(np.max(np.abs(self.b), initial=0.0)),
                    float(np.max(np.abs(q), initial=0.0)))
        s = np.zeros(m)
        z = np.zeros(m)
        Ax = self.mat_A(x)
        resid = self.b - Ax
        for b in self.blocks:
            r0, w = b.row0, b.width
            if b.kind == "zero":
                s[r0:r0 + w] = 0.0
                z[r0:r0 + w] = 0.0
            elif b.kind == "nonnegative":
                blk = resid[r0:r0 + w]
                bsc = np.maximum(1.0, np.abs(self.b[r0:r0 + w]))
                s[r0:r0 + w] = np.maximum(blk, 0.1 * bsc)
                z[r0:r0 + w] = 0.1 * bsc
            elif b.kind == "soc":
                blk = resid[r0:r0 + w]
                v = blk[1:]
                rho = float(np.linalg.norm(v))
                bsc = max(1.0, float(np.max(np.abs(self.b[r0:r0 + w]))))
                t_s = max(blk[0], rho + 0.1 * bsc)
                s[r0] = t_s
                s[r0 + 1:r0 + w] = v
                if rho > 1e-14:
                    u = v / rho
                else:
                    u = np.zeros(w - 1)
                    u[0] = 1.0
                # complementary (central path like) dual point: the vector part
                # is antiparallel to the slack and the Jordan product is mu
                mu0 = 0.5 * bsc * bsc / max(w, 1)
                den = t_s * t_s - rho * rho
                t_z = mu0 * t_s / den if den > 1e-12 else scale
                t_z = min(max(t_z, 1e-6 * bsc), 10.0 * bsc)
                z[r0] = t_z
                z[r0 + 1:r0 + w] = -u * (t_z * rho / max(t_s, 1e-12))
            elif b.kind == "exponential":
                z[r0:r0 + 3] = np.array([-1.0, 0.0, 1.0]) * scale
                g = -exp_dual_grad(z[r0:r0 + 3])
                s[r0:r0 + 3] = (scale / max(float(np.max(np.abs(g))), 1e-12)) * g
            elif b.kind == "psd":
                M = psd_smat(resid[r0:r0 + w], b.dim)
                M = 0.5 * (M + M.T)
                shift = float(np.max(np.abs(M), initial=0.0)) + 0.5 * scale
                M = M + shift * np.eye(b.dim)
                z[r0:r0 + w] = psd_svec(M)
                try:
                    Zinv = np.linalg.inv(psd_smat(z[r0:r0 + w], b.dim))
                except np.linalg.LinAlgError:
                    Zinv = np.eye(b.dim)
                g = psd_svec(Zinv)
                s[r0:r0 + w] = (scale / max(float(np.max(np.abs(g))), 1e-12)) * g
        return x, s, z

    # -- complementarity measure -------------------------------------------
    def measure(self, s, z):
        """Complementarity level per cone family."""
        mu = {"nn": 0.0, "soc": 0.0, "bar": 0.0}
        if len(self.nn_row_idx):
            mu["nn"] = float(np.mean(s[self.nn_row_idx] * z[self.nn_row_idx]))
        lvl = 0.0
        for b in self.soc_blocks:
            r0, w = b.row0, b.width
            sb = s[r0:r0 + w]
            zb = z[r0:r0 + w]
            prod = np.empty(w)
            prod[0] = sb[0] * zb[0] + float(sb[1:] @ zb[1:])
            prod[1:] = sb[0] * zb[1:] + zb[0] * sb[1:]
            lvl = max(lvl, float(np.max(np.abs(prod))))
        mu["soc"] = lvl
        tot = 0.0
        cnt = 0
        for b in self.blocks:
            if b.kind == "zero":
                continue
            r0, w = b.row0, b.width
            if b.kind == "soc":
                # the Jordan product of the cone (its trace)
                sb = s[r0:r0 + w]
                zb = z[r0:r0 + w]
                tot += float(sb[0] * zb[0] + sb[1:] @ zb[1:])
            else:
                tot += float(s[r0:r0 + w] @ z[r0:r0 + w])
            cnt += w
        mu["all"] = tot / cnt if cnt else 0.0
        mu["bar"] = mu["all"]
        return mu

    def gap(self, s, z):
        tot = 0.0
        for b in self.blocks:
            if b.kind == "zero":
                continue
            r0, w = b.row0, b.width
            tot += float(s[r0:r0 + w] @ z[r0:r0 + w])
        return tot

    # -- scaling: build W^{-1} and the right hand side g --------------------
    def build_scaling(self, s, z, mu, sigma, corr=None, dual=False, sigma_blocks=None):
        mu_bar = mu["bar"] if isinstance(mu, dict) else mu
        """Return (g, parts) where parts describes W^{-1} restricted to the
        rows of the non-zero cones.  g holds the right hand side of the
        linearised complementarity  ds = g - W dz."""
        g = np.zeros(self.m)
        parts = {"nn": None, "soc": [], "exp": [], "psd": []}

        if len(self.nn_row_idx):
            sr = s[self.nn_row_idx]
            zr = z[self.nn_row_idx]
            parts["nn"] = (self.nn_row_idx, zr / sr)
            mu_nn = mu["nn"] if isinstance(mu, dict) else mu
            tgt = sigma * mu_nn
            if corr is not None:
                tgt = tgt + corr[self.nn_row_idx]
            g[self.nn_row_idx] = (tgt - sr * zr) / zr

        for i, b in enumerate(self.soc_blocks):
            r0, w = b.row0, b.width
            zb = z[r0:r0 + w]
            sb = s[r0:r0 + w]
            sig = sigma
            if sigma_blocks is not None:
                sig = sigma_blocks.get(id(b), sigma)
            Winv, gblk = soc_scaling(sb, zb, sig, corr, r0, w)
            parts["soc"].append((i, b, Winv))
            g[r0:r0 + w] = gblk

        for i, b in enumerate(self.exp_blocks):
            r0 = b.row0
            zb = z[r0:r0 + 3]
            sb = s[r0:r0 + 3]
            grad = -exp_dual_grad(zb)
            gg = float(grad @ grad)
            mub = float(sb @ grad) / gg if gg > 0 else mu_bar
            if not np.isfinite(mub) or mub <= 0:
                mub = max(mu_bar, 1e-12)
            Winv = exp_dual_hessian_inv(zb) / mub
            if not np.all(np.isfinite(Winv)):
                Winv = np.eye(3)
            nrm = float(np.max(np.abs(Winv)))
            if nrm > 1e6:
                Winv = Winv * (1e6 / nrm)
            parts["exp"].append((i, b, Winv))
            g[r0:r0 + 3] = sigma * mub * grad - sb

        for i, b in enumerate(self.psd_blocks):
            r0, w = b.row0, b.width
            zb = z[r0:r0 + w]
            sb = s[r0:r0 + w]
            Z = psd_smat(zb, b.dim)
            try:
                Zinv = np.linalg.inv(Z)
            except np.linalg.LinAlgError:
                Zinv = np.linalg.pinv(Z)
            grad = psd_svec(Zinv)
            gg = float(grad @ grad)
            mub = float(sb @ grad) / gg if gg > 0 else mu_bar
            if not np.isfinite(mub) or mub <= 0:
                mub = max(mu_bar, 1e-12)
            Winv = psd_map_matrix(Z) / mub
            if not np.all(np.isfinite(Winv)):
                Winv = np.eye(w)
            parts["psd"].append((i, b, Winv))
            g[r0:r0 + w] = sigma * mub * grad - sb
        return g, parts

    # -- KKT solve ----------------------------------------------------------
    def _dense_M(self, parts):
        """Assemble M = P + A' W^-1 A exploiting the sparsity of A.

        The rows of A are processed one by one: the contribution of row k is
        the rank one update (W^-1 a_k) a_k'.
        """
        n = self.n
        M = np.zeros((n, n))
        if len(self.Pval):
            np.add.at(M, (self.Prow, self.Pcol), self.Pval)
        dscale = np.zeros(self.m)
        if parts["nn"] is not None:
            idx, dinv = parts["nn"]
            dscale[idx] = dinv
        if self.m * n <= 3e6 and self.m * n * n <= 4e8:
            # dense BLAS product is faster than the row by row assembly
            Ad = self._dense_A_cached()
            M += Ad.T @ (dscale[:, None] * Ad)
        else:
            indptr = self._A_csr_indptr
            cols = self._A_csr_cols
            vals = self._A_csr_vals
            for k in range(self.m):
                lo = indptr[k]
                hi = indptr[k + 1]
                if hi <= lo:
                    continue
                c = cols[lo:hi]
                v = vals[lo:hi]
                w = dscale[k] * v
                M[np.ix_(c, c)] += np.outer(w, v)
        # blocks with a dense scaling matrix (second order, exponential and
        # semidefinite cones) contribute separately
        for blocks, amat in ((parts["soc"], self.A_soc), (parts["exp"], self.A_exp),
                            (parts["psd"], self.A_psd)):
            for (i, b, Winv) in blocks:
                Ab = amat[i]
                M += Ab.T @ (Winv @ Ab)
        return M

    def prepare_dense(self, parts, rp, rd):
        """Assemble the Newton matrix once; the predictor and the corrector
        step share it."""
        self._Md = self._dense_M(parts)
        self._base_d = -rd - self.mat_AT(self._apply_Winv_dense(parts, rp))
        return self

    def _apply_Winv_dense(self, parts, v):
        m = self.m
        y = np.zeros(m)
        if parts["nn"] is not None:
            idx, dinv = parts["nn"]
            y[idx] = dinv * v[idx]
        for (i, b, Winv) in parts["soc"]:
            r0, w = b.row0, b.width
            y[r0:r0 + w] = Winv @ v[r0:r0 + w]
        for (i, b, Winv) in parts["exp"]:
            r0 = b.row0
            y[r0:r0 + 3] = Winv @ v[r0:r0 + 3]
        for (i, b, Winv) in parts["psd"]:
            r0, w = b.row0, b.width
            y[r0:r0 + w] = Winv @ v[r0:r0 + w]
        return y

    def solve_step_dense(self, parts, g, rp, rd):
        """NumPy-only Newton solve (no SciPy) for small and medium problems."""
        m = self.m
        M = self._Md
        r1 = self._base_d - self.mat_AT(self._apply_Winv_dense(parts, g))
        nz = self.zero_row_idx.size
        if nz:
            A0 = self.mat_AT(np.eye(nz)).T if False else None
            A0 = self._A0_dense
            K = np.zeros((self.n + nz, self.n + nz))
            K[:self.n, :self.n] = M
            K[:self.n, self.n:] = A0.T
            K[self.n:, :self.n] = A0
            rhs_vec = np.concatenate([r1, -rp[self.zero_row_idx]])
            try:
                sol = np.linalg.solve(K, rhs_vec)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs_vec, rcond=None)[0]
            dx = sol[:self.n]
            dz = np.zeros(self.m)
            dz[self.zero_row_idx] = sol[self.n:]
        else:
            try:
                dx = np.linalg.solve(M, r1)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(M, r1, rcond=None)[0]
            dz = np.zeros(self.m)
        # recover dz for the non-zero cones
        if parts["nn"] is not None:
            idx, dinv = parts["nn"]
            dz[idx] = dinv * (self.mat_A(dx)[idx] + (rp + g)[idx])
        for (i, b, Winv) in parts["soc"]:
            r0, w = b.row0, b.width
            dz[r0:r0 + w] = Winv @ (self.mat_A(dx)[r0:r0 + w] + (rp + g)[r0:r0 + w])
        for (i, b, Winv) in parts["exp"]:
            r0 = b.row0
            dz[r0:r0 + 3] = Winv @ (self.mat_A(dx)[r0:r0 + 3] + (rp + g)[r0:r0 + 3])
        for (i, b, Winv) in parts["psd"]:
            r0, w = b.row0, b.width
            dz[r0:r0 + w] = Winv @ (self.mat_A(dx)[r0:r0 + w] + (rp + g)[r0:r0 + w])
        ds = -rp - self.mat_A(dx)
        return dx, dz, ds

    def prepare_step(self, parts, rp, rd):
        """Assemble and factorise the Newton matrix.

        The predictor (affine) and the corrector step use the same matrix --
        the scaling only depends on the current iterate -- so one
        factorisation per iteration is enough.
        """
        if self.dense:
            return self
        from scipy import sparse
        from scipy.sparse.linalg import splu
        M = self.P.copy()
        # right hand side for a given g:  base - A' W^-1 g
        base = -rd.astype(np.float64).copy()
        self._op_blocks = []
        self._nn_op = None
        if parts["nn"] is not None:
            idx, dinv = parts["nn"]
            d = np.zeros(self.m)
            d[idx] = dinv
            Dinv = sparse.diags(d)
            M = M + (self.AT @ Dinv @ self.A)
            base = base - self.AT @ (Dinv @ rp)
            self._nn_op = (idx, dinv, self.AT[:, idx])
        for kind, blocks, amat in (("soc", parts["soc"], self.A_soc),
                                   ("exp", parts["exp"], self.A_exp),
                                   ("psd", parts["psd"], self.A_psd)):
            for (i, b, Winv) in blocks:
                Ab = amat[i]
                M = M + sparse.csr_matrix(Ab.T @ Winv @ Ab)
                base = base - Ab.T @ (Winv @ rp[b.row0:b.row0 + b.width])
                self._op_blocks.append((kind, i, b, Winv))
        if self.zero_row_idx.size:
            A0 = self.A[self.zero_row_idx, :]
            K = sparse.bmat([[M, A0.T], [A0, None]], format="csc")
            self._K = K
            try:
                self._lu = splu(K)
            except Exception:
                reg = sparse.identity(K.shape[0], format="csc") * \
                    (1e-10 * max(1.0, abs(M.diagonal().max())))
                self._lu = splu(K + reg)
        else:
            self._K = None
            self._lu = splu(M.tocsc())
        self._base = base
        self._rp0 = -rp[self.zero_row_idx] if self.zero_row_idx.size else None
        return self

    def solve_prepared(self, parts, g, rp, rd):
        """Solve the prepared Newton system for the right hand side given by g."""
        if self.dense:
            return self.solve_step_dense(parts, g, rp, rd)
        rhs = rp + g
        r1 = self._base.copy()
        if self._nn_op is not None:
            idx, dinv, ATidx = self._nn_op
            r1 = r1 - ATidx @ (dinv * g[idx])
        for (kind, i, b, Winv) in self._op_blocks:
            r0, w = b.row0, b.width
            if kind == "soc":
                Ab = self.A_soc[i]
            elif kind == "exp":
                Ab = self.A_exp[i]
            else:
                Ab = self.A_psd[i]
            r1 = r1 - Ab.T @ (Winv @ g[r0:r0 + w])
        if self._K is not None:
            rhs_vec = np.concatenate([r1, self._rp0])
            sol = self._lu.solve(rhs_vec)
            dx = sol[:self.n]
            dz = np.zeros(self.m)
            dz[self.zero_row_idx] = sol[self.n:]
        else:
            dx = self._lu.solve(r1)
            dz = np.zeros(self.m)
        # recover dz for the non-zero cones
        if parts["nn"] is not None:
            idx, dinv = parts["nn"]
            dz[idx] = dinv * (np.asarray(self.A[idx, :] @ dx).ravel() + rhs[idx])
        for (i, b, Winv) in parts["soc"]:
            Ab = self.A_soc[i]
            dz[b.row0:b.row0 + b.width] = Winv @ (Ab @ dx + rhs[b.row0:b.row0 + b.width])
        for (i, b, Winv) in parts["exp"]:
            Ab = self.A_exp[i]
            dz[b.row0:b.row0 + b.width] = Winv @ (Ab @ dx + rhs[b.row0:b.row0 + b.width])
        for (i, b, Winv) in parts["psd"]:
            Ab = self.A_psd[i]
            dz[b.row0:b.row0 + b.width] = Winv @ (Ab @ dx + rhs[b.row0:b.row0 + b.width])
        ds = -rp - self.A @ dx
        return dx, dz, ds

    def solve_step(self, parts, g, rp, rd):
        if self.dense:
            return self.solve_step_dense(parts, g, rp, rd)
        self.prepare_step(parts, rp, rd)
        return self.solve_prepared(parts, g, rp, rd)

    def max_step(self, s, z, ds, dz):
        alpha_s = 1.0
        alpha_z = 1.0
        for b in self.blocks:
            r0, w = b.row0, b.width
            kind = b.kind
            if kind == "zero":
                continue
            if kind == "nonnegative":
                a = _frac_nonneg(s[r0:r0 + w], ds[r0:r0 + w])
                bstep = _frac_nonneg(z[r0:r0 + w], dz[r0:r0 + w])
                alpha_s = min(alpha_s, a)
                alpha_z = min(alpha_z, bstep)
            elif kind == "soc":
                # a more conservative step keeps a margin to the cone boundary,
                # which the ratio based scaling of this cone is sensitive to
                a = 0.85 * _frac_soc(s[r0:r0 + w], ds[r0:r0 + w])
                bstep = 0.85 * _frac_soc(z[r0:r0 + w], dz[r0:r0 + w])
                alpha_s = min(alpha_s, a)
                alpha_z = min(alpha_z, bstep)
            elif kind == "exponential":
                a = _frac_exp(s[r0:r0 + w], ds[r0:r0 + w])
                bstep = _frac_exp_dual(z[r0:r0 + w], dz[r0:r0 + w])
                alpha_s = min(alpha_s, a)
                alpha_z = min(alpha_z, bstep)
            elif kind == "psd":
                a = _frac_psd(s[r0:r0 + w], ds[r0:r0 + w], b.dim)
                bstep = _frac_psd(z[r0:r0 + w], dz[r0:r0 + w], b.dim)
                alpha_s = min(alpha_s, a)
                alpha_z = min(alpha_z, bstep)
        return alpha_s, alpha_z

    # -- main loop ----------------------------------------------------------
    def run(self):
        x, s, z = self.initial_points()
        best = None
        stale = 0
        grow = 0
        last_gap = 0.0
        gap0 = None
        mu = self.measure(s, z)
        if not isinstance(mu, dict):
            mu = {"nn": mu, "soc": mu, "bar": mu, "all": mu}
        if mu["all"] <= 0:
            mu = {"nn": 1.0, "soc": 1.0, "bar": 1.0, "all": 1.0}
        for it in range(self.max_iter):
            if self.out_of_time():
                break
            Ax = self.mat_A(x)
            Px = self.mat_P(x)
            ATz = self.mat_AT(z)
            rp = Ax + s - self.b
            rd = Px + self.q + ATz
            rp_norm = float(np.max(np.abs(rp), initial=0.0))
            rd_norm = float(np.max(np.abs(rd), initial=0.0))
            den_p = 1.0 + max(float(np.max(np.abs(Ax), initial=0.0)),
                              float(np.max(np.abs(s), initial=0.0)),
                              float(np.max(np.abs(self.b), initial=0.0)))
            den_d = 1.0 + max(float(np.max(np.abs(Px), initial=0.0)),
                              float(np.max(np.abs(self.q), initial=0.0)),
                              float(np.max(np.abs(ATz), initial=0.0)))
            xPx = float(x @ Px)
            p0 = 0.5 * xPx + float(self.q @ x)
            d0 = -0.5 * xPx - float(self.b @ z)
            gap = abs(p0 - d0) / self.obj_scale
            scale = max(abs(p0), abs(d0)) / self.obj_scale
            gap_rel = gap / scale if scale > 0 else 0.0
            if self.verbose:
                print("it %d rp %.3e rd %.3e gap %.3e mu %.3e" %
                      (it, rp_norm / den_p, rd_norm / den_d, gap, mu["all"]), file=sys.stderr)
            if rp_norm / den_p <= 5e-9 and rd_norm / den_d <= 5e-9 and \
                    (gap <= 5e-9 or gap_rel <= 5e-9):
                self.status = "solved"
                return x, s, z
            cur = rp_norm / den_p + rd_norm / den_d
            if best is None or cur < best[3]:
                best = (x.copy(), s.copy(), z.copy(), cur)
                stale = 0
            else:
                stale += 1
                # the iteration is diverging: give up early so that the
                # infeasibility searches can use the remaining time
                if stale > 8 and cur > 1e4 * max(best[3], 1e-12):
                    break
            # a rapidly growing objective means the problem is unbounded
            if gap0 is None:
                gap0 = max(gap, 1e-12)
            if gap > 1.5 * last_gap and gap > 1e3 * gap0:
                grow += 1
            else:
                grow = 0
            last_gap = gap
            if grow > 6:
                break

            # ---- predictor (affine) step
            g_aff, parts_aff = self.build_scaling(s, z, mu, 0.0, None)
            try:
                if self.dense:
                    self.prepare_dense(parts_aff, rp, rd)
                else:
                    self.prepare_step(parts_aff, rp, rd)
                dx, dz, ds = self.solve_prepared(parts_aff, g_aff, rp, rd)
            except Exception as exc:
                if os.environ.get("CONIC_DEBUG"):
                    import traceback
                    traceback.print_exc()
                break
            a_s, a_z = self.max_step(s, z, ds, dz)
            alpha = min(1.0, 0.99 * min(a_s, a_z))
            mu_aff = self.measure(s + alpha * ds, z + alpha * dz)
            if mu["all"] <= 0 or not np.isfinite(mu_aff["all"]) or mu_aff["all"] <= 0:
                sigma = 0.1
            else:
                sigma = (mu_aff["all"] / mu["all"]) ** 3
                sigma = min(1.0, max(1e-8, sigma))

            # ---- corrector step
            sigma_blocks = None
            if os.environ.get("CONIC_PER_BLOCK_SIGMA") == "1" and self.soc_blocks:
                sigma_blocks = {}
                for b in self.soc_blocks:
                    r0, w = b.row0, b.width
                    sb = s[r0:r0 + w]
                    zb = z[r0:r0 + w]
                    lvl = float(sb[0] * zb[0] + sb[1:] @ zb[1:])
                    sn = sb + alpha * ds[r0:r0 + w]
                    zn = zb + alpha * dz[r0:r0 + w]
                    lvl2 = float(sn[0] * zn[0] + sn[1:] @ zn[1:])
                    if lvl > 1e-14 and lvl2 > 0:
                        sigma_blocks[id(b)] = min(1.0, max(1e-8, (lvl2 / lvl) ** 3))
            corr = np.zeros(self.m)
            if len(self.nn_row_idx):
                # second order term of Mehrotra's corrector; scaled by the
                # (fractional) length of the predictor step to keep it moderate
                pexp = float(os.environ.get("CONIC_CORR_EXP", "2"))
                corr[self.nn_row_idx] = (alpha ** pexp) * ds[self.nn_row_idx] * dz[self.nn_row_idx]
            for b in self.soc_blocks:
                r0, w = b.row0, b.width
                a = ds[r0:r0 + w]
                c = dz[r0:r0 + w]
                sc = alpha * alpha
                corr[r0] = sc * (a[0] * c[0] + float(a[1:] @ c[1:]))
                corr[r0 + 1:r0 + w] = sc * (a[0] * c[1:] + c[0] * a[1:])
            g, parts = self.build_scaling(s, z, mu, sigma, corr, sigma_blocks=sigma_blocks)
            try:
                dx, dz, ds = self.solve_prepared(parts, g, rp, rd)
            except Exception as exc:
                if os.environ.get("CONIC_DEBUG"):
                    import traceback
                    traceback.print_exc()
                break
            a_s, a_z = self.max_step(s, z, ds, dz)
            alpha = min(1.0, float(os.environ.get("CONIC_STEP", "0.995")) * min(a_s, a_z))
            if not np.isfinite(alpha) or alpha <= 0:
                break
            x = x + alpha * dx
            s = s + alpha * ds
            z = z + alpha * dz
            # keep the zero cone slack exactly zero
            if self.zero_row_idx.size:
                s[self.zero_row_idx] = 0.0
            new_mu = self.measure(s, z)
            if not np.isfinite(new_mu["all"]):
                break
            mu = new_mu
        if best is not None:
            return best[0], best[1], best[2]
        return x, s, z


# ----------------------------------------------------------------------------
# cone specific helper functions
# ----------------------------------------------------------------------------


def exp_dual_grad(z):
    """gradient of the concave dual barrier of the exp cone (points into K)."""
    p = -z[0]
    w = z[2]
    F = exp_dual_slack(z)
    if F <= 0 or p <= 0 or w <= 0:
        return np.zeros(3)
    g = np.empty(3)
    g[0] = -(1.0 - math.log(math.e * w / p)) / F - 1.0 / z[0]
    g[1] = -1.0 / F
    g[2] = -(p / w) / F - 1.0 / w
    return g


def _frac_psd(v, dv, d):
    """step length keeping v + a dv positive definite (isometric svec)."""
    try:
        S = psd_smat(v, d)
        L = np.linalg.cholesky(S)
        D = psd_smat(dv, d)
        T = np.linalg.solve(L, D)
        T = np.linalg.solve(L, T.T).T
        T = 0.5 * (T + T.T)
        lam = np.linalg.eigvalsh(T)
        lo = float(np.min(lam))
        if lo >= 0:
            return 1.0
        return float(min(1.0, -1.0 / lo))
    except Exception:
        return 1e-6


def soc_barrier_grad(z):
    """-grad phi(z) for the second order cone barrier (points into K)."""
    z0 = z[0]
    vec = z[1:]
    D = z0 * z0 - float(vec @ vec)
    if D <= 1e-300:
        D = 1e-300
    g = np.empty_like(z)
    g[0] = z0
    g[1:] = -vec
    return (2.0 / D) * g


def soc_scaling(sb, zb, sigma, corr, r0, w):
    """Scaling matrix and right hand side for one second order cone block.

    The Jordan structure of the cone gives the exact linearisation
    Z ds + S dz = rhs with Z = [[z0, zv'], [zv, z0 I]] and
    S = [[s0, sv'], [sv, s0 I]].  The inverse of S is regularised with the
    current complementarity level, which keeps the coupling matrix bounded
    when the slack approaches the boundary of the cone.
    """
    s0 = float(sb[0])
    sv = sb[1:]
    z0 = float(zb[0])
    zv = zb[1:]
    d = w - 1
    Zm = np.zeros((w, w))
    Zm[0, 0] = z0
    Zm[0, 1:] = zv
    Zm[1:, 0] = zv
    Zm[1:, 1:] = z0 * np.eye(d)
    Sm = np.zeros((w, w))
    Sm[0, 0] = s0
    Sm[0, 1:] = sv
    Sm[1:, 0] = sv
    Sm[1:, 1:] = s0 * np.eye(d)
    Ds = s0 * s0 - float(sv @ sv)
    if Ds <= 0:
        Ds = 1e-300
    prod = Zm @ sb               # the Jordan product s o z
    p0 = float(prod[0])
    lam = 0.01 * abs(p0)
    ds_reg = Ds + lam
    Sinv = np.zeros((w, w))
    Sinv[0, 0] = s0
    Sinv[0, 1:] = -sv
    Sinv[1:, 0] = -sv
    Sinv[1:, 1:] = s0 * np.eye(d)
    Sinv /= ds_reg
    Winv = Sinv @ Zm              # symmetric positive definite
    rhs = (sigma - 1.0) * prod
    if corr is not None:
        rhs = rhs + corr[r0:r0 + w]
    try:
        gblk = np.linalg.solve(Zm, rhs)
    except np.linalg.LinAlgError:
        gblk = rhs
    return Winv, gblk


def _frac_nonneg(v, dv):
    """largest alpha in (0,1] with v + alpha dv >= 0 (componentwise)."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _frac_soc(v, dv):
    """largest alpha with v + alpha dv in the second order cone."""
    t0 = v[0] + dv[0] * 0.0
    a = float(v @ v) - 0.0
    # quadratic: (v0 + a dv0)^2 - ||v1 + a dv1||^2 > 0
    qa = dv[0] * dv[0] - float(dv[1:] @ dv[1:])
    qb = 2.0 * (v[0] * dv[0] - float(v[1:] @ dv[1:]))
    qc = v[0] * v[0] - float(v[1:] @ v[1:])
    if qa == 0.0:
        if qb == 0.0:
            return 1.0
        root = -qc / qb
        return float(min(1.0, root)) if root > 0 else 1e-12
    disc = qb * qb - 4.0 * qa * qc
    if disc <= 0.0:
        return 1.0 if qa > 0.0 else 1e-12
    sq = math.sqrt(disc)
    r1 = (-qb + sq) / (2.0 * qa)
    r2 = (-qb - sq) / (2.0 * qa)
    roots = [r for r in (r1, r2) if r > 1e-16]
    if not roots:
        if qc <= 0:
            return 1e-12
        if qa > 0:
            # the cone inequality holds for every alpha >= 0
            return 1.0
        return 1e-12
    r = min(roots)
    return float(min(1.0, r))


def _frac_exp(v, dv):
    """step length keeping v + a dv inside the primal exponential cone."""
    alpha = 1.0
    for i in (1, 2):
        if dv[i] < 0:
            alpha = min(alpha, -v[i] / dv[i])
    alpha = max(alpha, 0.0)
    alpha = min(1.0, alpha)
    for _ in range(60):
        w = v + alpha * dv
        if w[1] > 0 and w[2] > 0 and w[1] * math.exp(min(50.0, w[0] / w[1])) <= w[2]:
            return alpha
        alpha *= 0.5
        if alpha < 1e-14:
            break
    return max(alpha, 1e-14)


def _frac_exp_dual(v, dv):
    """step length keeping v + a dv inside the dual exponential cone.

    The dual cone requires u <= 0 and w >= 0, so the first component is
    bounded from above and the third one from below.
    """
    alpha = 1.0
    if dv[0] > 0:
        alpha = min(alpha, -v[0] / dv[0])
    if dv[2] < 0:
        alpha = min(alpha, -v[2] / dv[2])
    alpha = max(alpha, 0.0)
    alpha = min(1.0, alpha)
    for _ in range(60):
        w = v + alpha * dv
        if exp_dual_slack(w) > 0 and w[0] < 0 and w[2] > 0:
            return alpha
        alpha *= 0.5
        if alpha < 1e-14:
            break
    return max(alpha, 1e-14)


def exp_dual_slack(z):
    """F(z) = z2 + p log(e z3 / p),  p = -z1  (>0 for the dual cone interior)."""
    p = -z[0]
    if p <= 0 or z[2] <= 0:
        return -1.0
    return z[1] + p * math.log(math.e * z[2] / p)


def exp_dual_hessian_inv(z):
    """Inverse of -Hessian of the concave dual barrier of the exp cone."""
    p = -z[0]
    w = z[2]
    F = exp_dual_slack(z)
    if F <= 0 or p <= 0 or w <= 0:
        return np.zeros((3, 3))
    gF = np.array([1.0 - math.log(math.e * w / p), 1.0, p / w])
    H = np.zeros((3, 3))
    H[0, 0] = -1.0 / p
    H[0, 2] = -1.0 / w
    H[2, 0] = -1.0 / w
    H[2, 2] = -p / (w * w)
    M3 = H / F - np.outer(gF, gF) / (F * F)
    M3[0, 0] -= 1.0 / (z[0] * z[0])
    M3[2, 2] -= 1.0 / (z[2] * z[2])
    try:
        # the scaling uses -Hess(B) (positive definite for a concave barrier)
        return np.linalg.inv(-M3)
    except np.linalg.LinAlgError:
        return np.zeros((3, 3))


def psd_map_matrix(Z):
    """Dense matrix B with B svec(U) = svec(Z U Z)  (isometric svec)."""
    d = Z.shape[0]
    w = d * (d + 1) // 2
    B = np.zeros((w, w))
    # columns: Z E_kl Z = outer(Z[:,k], Z[:,l]) symmetrised
    cols = []
    # the svec ordering is column wise: for every column l the entries of the
    # rows 0..l appear in that order
    for l in range(d):
        for k in range(l + 1):
            if k == l:
                M = np.outer(Z[:, k], Z[k, :])
            else:
                # the svec basis element is (e_k e_l' + e_l e_k')/sqrt(2)
                M = (np.outer(Z[:, k], Z[l, :]) +
                     np.outer(Z[:, l], Z[k, :])) / SQRT2
            cols.append(psd_svec(M))
    C = np.array(cols).T  # w x w, C[:, idx] = svec(Z E_kl Z)
    return C




# ----------------------------------------------------------------------------
# infeasibility certificates
# ----------------------------------------------------------------------------


def _identity_cones(cones, m, n, kind_swap=False):
    """Cone list for an auxiliary problem whose variable lives in the cones."""
    return cones


def _dense_A_of(prob):
    if prob.raw is not None:
        return _dense_from_raw(prob.raw["A_data"], prob.raw["A_indices"],
                               prob.raw["A_indptr"], prob.raw["A_shape"])
    return np.asarray(prob.A.toarray())


def _dense_P_of(prob):
    if prob.raw is not None:
        return _dense_from_raw(prob.raw["P_data"], prob.raw["P_indices"],
                               prob.raw["P_indptr"], prob.raw["P_shape"])
    return np.asarray(prob.P.toarray())


def solve_dual_infeasible(prob, deadline):
    """Find x with P x = 0, q'x < 0, A x + s = 0 with s in K (||x||_inf <= 1)."""
    n, m = prob.n, prob.m
    Pm = _dense_P_of(prob)
    Am = _dense_A_of(prob)
    cones = [{"type": "zero", "dim": n}]
    G = [Pm]
    rhs = [np.zeros(n)]
    off = 0
    for c in prob.cones:
        kind = c["type"]
        dim = int(c["dim"])
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        cones.append({"type": kind, "dim": dim})
        G.append(Am[off:off + width, :])
        rhs.append(np.zeros(width))
        off += width
    # infinity norm bound |x_i| <= 1 through non-negative slacks
    cones.append({"type": "nonnegative", "dim": n})
    G.append(np.eye(n))
    rhs.append(np.ones(n))
    cones.append({"type": "nonnegative", "dim": n})
    G.append(-np.eye(n))
    rhs.append(np.ones(n))
    aux = _problem_from_dense(np.zeros((n, n)), prob.q, np.vstack(G),
                               np.concatenate(rhs), cones)
    solver = Solver(aux, deadline=deadline, max_iter=300)
    x, sl, z = solver.run()
    if solver.status != "solved":
        return None
    return x, sl


def solve_primal_infeasible(prob, deadline):
    """Find z in K* with A'z = 0 and b'z < 0 (||z||_inf <= 1)."""
    n, m = prob.n, prob.m
    Am = _dense_A_of(prob)
    cones = [{"type": "zero", "dim": n}]
    G_blocks = [Am.T]
    rhs = [np.zeros(n)]
    q = prob.b.copy()
    off = 0
    for c in prob.cones:
        kind = c["type"]
        dim = int(c["dim"])
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        if kind == "exponential":
            return None            # dual exponential cone not supported here
        cones.append({"type": kind, "dim": dim})
        # the slack of the block is the block of z itself (z has to lie in K*)
        sel = np.zeros((width, m))
        sel[:, off:off + width] = -np.eye(width)
        G_blocks.append(sel)
        rhs.append(np.zeros(width))
        off += width
    # infinity norm bound |z_i| <= 1 through non-negative slacks
    cones.append({"type": "nonnegative", "dim": m})
    G_blocks.append(np.eye(m))
    rhs.append(np.ones(m))
    cones.append({"type": "nonnegative", "dim": m})
    G_blocks.append(-np.eye(m))
    rhs.append(np.ones(m))
    aux = _problem_from_dense(np.zeros((m, m)), q, np.vstack(G_blocks),
                               np.concatenate(rhs), cones)
    solver = Solver(aux, deadline=deadline, max_iter=300)
    x, sl, z = solver.run()
    if solver.status != "solved":
        return None
    return x, sl


# ----------------------------------------------------------------------------
# certificate validation (mirrors the independent checker)
# ----------------------------------------------------------------------------


def _cone_defect(v, cones, dual):
    """Maximum cone defect of the vector v over the cone list."""
    worst = 0.0
    start = 0
    for c in cones:
        kind = c["type"]
        dim = int(c["dim"])
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        block = v[start:start + width]
        start += width
        u = block / max(1.0, float(np.max(np.abs(block), initial=0.0)))
        if kind == "zero":
            viol = 0.0 if dual else float(np.max(np.abs(u), initial=0.0))
        elif kind == "nonnegative":
            viol = max(0.0, -float(np.min(u))) if u.size else 0.0
        elif kind == "soc":
            viol = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
        elif kind == "exponential":
            if dual:
                viol = _exp_defect(np.array([-u[1], -u[0], math.e * u[2]]))
            else:
                viol = _exp_defect(u)
        else:
            M = psd_smat(u, dim)
            try:
                ev = np.linalg.eigvalsh(M)
                viol = max(0.0, -float(ev[0]))
            except np.linalg.LinAlgError:
                return float("inf")
        if not np.isfinite(viol):
            return float("inf")
        worst = max(worst, viol)
    return worst


def _exp_defect(u):
    a, b, c = float(u[0]), float(u[1]), float(u[2])
    boundary = max(abs(b), max(a, 0.0), max(-c, 0.0))
    if b > 0.0 and c > 0.0:
        interior = max(0.0, a - b * (math.log(c) - math.log(b)))
        return min(interior, boundary)
    return boundary


def validate_primal_infeasible(prob, z, tol=1e-8):
    z = np.asarray(z, dtype=np.float64)
    nrm = float(np.max(np.abs(z), initial=0.0))
    bnorm = float(np.max(np.abs(prob.b), initial=0.0))
    if nrm <= 0 or bnorm <= 0:
        return False
    zh = z / nrm
    margin = -float((prob.b / bnorm) @ zh)
    if margin <= tol:
        return False
    ascale = float(np.max(np.abs(prob.A.data), initial=0.0)) or 1.0
    A = prob.A.copy()
    A.data = A.data / ascale
    lin = float(np.max(np.abs(A.T @ zh), initial=0.0))
    if lin / margin > tol:
        return False
    if _cone_defect(zh, prob.cones, True) / margin > tol:
        return False
    return True


def validate_dual_infeasible(prob, x, s, tol=1e-8):
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    nrm = float(np.max(np.abs(x), initial=0.0))
    qnorm = float(np.max(np.abs(prob.q), initial=0.0))
    if nrm <= 0 or qnorm <= 0:
        return False
    xh = x / nrm
    sh = s / nrm
    margin = -float((prob.q / qnorm) @ xh)
    if margin <= tol:
        return False
    pscale = float(np.max(np.abs(prob.P.data), initial=0.0)) or 1.0
    ascale = float(np.max(np.abs(prob.A.data), initial=0.0)) or 1.0
    P = prob.P.copy()
    P.data = P.data / pscale
    A = prob.A.copy()
    A.data = A.data / ascale
    quad = float(np.max(np.abs(P @ xh), initial=0.0))
    if quad / margin > tol:
        return False
    lin = float(np.max(np.abs(A @ xh + sh / ascale), initial=0.0))
    if lin / margin > tol:
        return False
    if _cone_defect(sh / ascale, prob.cones, False) / margin > tol:
        return False
    return True


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def write_result(path, status, x=None, s=None, z=None, obj_p=None, obj_d=None):
    meta = {"schema_version": 1, "status": status}
    if status == "solved":
        if obj_p is not None and np.isfinite(obj_p):
            meta["objective_primal"] = float(obj_p)
        if obj_d is not None and np.isfinite(obj_d):
            meta["objective_dual"] = float(obj_d)
    arrays = {"metadata_json": np.array(json.dumps(meta, sort_keys=True))}
    if x is not None:
        arrays["x"] = np.ascontiguousarray(x, dtype=np.float64)
    if s is not None:
        arrays["s"] = np.ascontiguousarray(s, dtype=np.float64)
    if z is not None:
        arrays["z"] = np.ascontiguousarray(z, dtype=np.float64)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main(argv):
    inp = out = None
    i = 1
    while i < len(argv):
        if argv[i] == "--input":
            inp = argv[i + 1]
            i += 2
        elif argv[i] == "--output":
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    if inp is None or out is None:
        print("usage: solver.py --input IN --output OUT", file=sys.stderr)
        return 2
    try:
        prob = Problem.load(inp)
    except Exception as exc:  # pragma: no cover
        print("input error: %r" % (exc,), file=sys.stderr)
        write_result(out, "unresolved")
        return 0
    n, m = prob.n, prob.m
    if prob.raw is not None:
        nnz = len(prob.raw["A_indices"]) + len(prob.raw["P_indices"])
    else:
        nnz = prob.A.nnz + prob.P.nnz
    budget = float(os.environ.get("CONIC_TIME_LIMIT", "0") or 0.0)
    if budget <= 0:
        # total wall clock target for the whole process (startup included)
        total = min(8.0, 0.82 + 3.0e-4 * (n + m) + 3.0e-6 * nnz)
        budget = max(0.05, total - (time.time() - T0))
    deadline = time.time() + budget
    status = "unresolved"
    x = s = z = None
    try:
        solver = Solver(prob, deadline=deadline - 0.15 * budget, max_iter=500,
                        verbose=os.environ.get("CONIC_VERBOSE") == "1")
        x, s, z = solver.run()
        status = solver.status
        if status == "solved":
            alpha = getattr(solver, "obj_scale", 1.0)
            z = z / alpha
            xPx = float(x @ (solver.mat_P(x) / alpha))
            p0 = 0.5 * xPx + float(prob.q @ x)
            d0 = -0.5 * xPx - float(prob.b @ z)
            if not (np.isfinite(p0) and np.isfinite(d0)):
                status = "unresolved"
            else:
                write_result(out, "solved", x, s, z, p0 + prob.offset, d0 + prob.offset)
                return 0
        # the problem did not solve: look for infeasibility certificates
        remain = deadline - time.time()
        if remain > 0.15:
            try:
                res = solve_dual_infeasible(prob, time.time() + 0.45 * remain)
            except Exception:
                res = None
            if res is not None:
                xc, sl = res
                if validate_dual_infeasible(prob, xc, sl[prob.n:prob.n + prob.m]):
                    write_result(out, "dual_infeasible", x=xc, s=sl[prob.n:prob.n + prob.m])
                    return 0
        remain = deadline - time.time()
        if remain > 0.15:
            try:
                res = solve_primal_infeasible(prob, time.time() + 0.9 * remain)
            except Exception:
                res = None
            if res is not None:
                zc = res[0]
                if validate_primal_infeasible(prob, zc):
                    write_result(out, "primal_infeasible", z=zc)
                    return 0
    except Exception as exc:
        print("solver error: %r" % (exc,), file=sys.stderr)
        status = "unresolved"
    # fallback: unresolved
    try:
        if x is not None and s is not None and z is not None:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        write_result(out, "unresolved", x, s, z)
    except Exception:
        write_result(out, "unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
