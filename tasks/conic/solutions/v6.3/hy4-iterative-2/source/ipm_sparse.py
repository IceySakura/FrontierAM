"""Sparse quasi-definite primal-dual interior-point solver.

Used when a dense representation of A or the normal matrix is too large.
The KKT system is written in the (never-squared) quasi-definite form

    [ P + dI      A'      ] [dx]   [ -r_d                 ]
    [ A       -(Hinv + dI)] [dz] = [ -r_p + Hinv * r_c    ]
    ds = -Hinv (r_c + dz)

with Hinv = H^{-1} on conic rows and Hinv = 0 on zero-cone rows (so the
zero-cone equations A_0 dx = -r_p,0 are enforced exactly).  H is the
Nesterov-Todd scaling for symmetric cones and mu*nabla^2 F(s) for K_exp.
"""
import math
import time

import numpy as np

import ipm
from verify import check_solved


class HInv:
    """H^{-1} applications and dense diagonal blocks for the (2,2) KKT part."""

    def __init__(self, ops, s, z, mu, exp_mu=None):
        self.ops = ops
        self.m = ops.m
        self.mu = mu
        self.diag = np.zeros(ops.m)
        self.diag[ops.nn_idx] = s[ops.nn_idx] / np.maximum(z[ops.nn_idx], 1e-300)
        self.soc = []
        for dim, idx in ops.soc:
            S, Z = s[idx], z[idx]
            ss = np.sqrt(np.maximum(S[:, 0] ** 2 - np.einsum("ij,ij->i", S[:, 1:], S[:, 1:]), 1e-300))
            zz = np.sqrt(np.maximum(Z[:, 0] ** 2 - np.einsum("ij,ij->i", Z[:, 1:], Z[:, 1:]), 1e-300))
            sb, zb = S / ss[:, None], Z / zz[:, None]
            gamma = np.sqrt(np.maximum((1.0 + np.einsum("ij,ij->i", sb, zb)) / 2.0, 1e-300))
            Js = sb.copy()
            Js[:, 1:] *= -1.0
            U = (Js + zb) / (2.0 * gamma)[:, None]      # U = J w_bar
            wbar = U.copy()
            wbar[:, 1:] *= -1.0                          # w_bar = J U
            # H = (zz/ss) (2 U U' - J); Hinv = (ss/zz) (2 wbar wbar' - J)
            self.soc.append((idx, dim, ss / zz, wbar))
        self.psd = []
        for dim, idx, rows, cols, scales, inv_scales in ops.psd:
            Ws = []
            for k in range(idx.shape[0]):
                S = ipm._smat(s[idx[k]], dim, rows, cols, scales)
                Z = ipm._smat(z[idx[k]], dim, rows, cols, scales)
                try:
                    ev, U = np.linalg.eigh(S)
                    ev = np.maximum(ev, 1e-300)
                    Sh = (U * np.sqrt(ev)) @ U.T
                    Shi = (U * (1.0 / np.sqrt(ev))) @ U.T
                    T = 0.5 * (Sh @ Z @ Sh + (Sh @ Z @ Sh).T)
                    lam, V = np.linalg.eigh(T)
                    lam = np.maximum(lam, 1e-300)
                    R = Shi @ (V * np.sqrt(lam)) @ V.T @ Shi   # H[M] = R M R
                    Ws.append(np.linalg.inv(R))
                except np.linalg.LinAlgError:
                    Ws.append(np.eye(dim))
            self.psd.append((idx, dim, rows, cols, scales, inv_scales, Ws))
        self.exp = None
        if ops.exp is not None:
            G = ipm._exp_hessian(s[ops.exp]) * (mu if exp_mu is None else exp_mu)
            self.exp = np.linalg.inv(G)
        self.zero = ops.zero_idx

    def apply(self, v):
        out = np.zeros(self.m)
        ops = self.ops
        out[ops.nn_idx] = self.diag[ops.nn_idx] * v[ops.nn_idx]
        for idx, dim, fac, wbar in self.soc:
            V = v[idx]
            Jv = V.copy()
            Jv[:, 1:] *= -1.0
            wv = np.einsum("ij,ij->i", wbar, V)
            out[idx] = fac[:, None] * (2.0 * wv[:, None] * wbar - Jv)
        for idx, dim, rows, cols, scales, inv_scales, Ws in self.psd:
            for k in range(idx.shape[0]):
                M = ipm._smat(v[idx[k]], dim, rows, cols, scales)
                out[idx[k]] = ipm._mat_svec(Ws[k] @ M @ Ws[k], rows, cols, inv_scales)
        if self.exp is not None:
            out[ops.exp] = np.einsum("kij,kj->ki", self.exp, v[ops.exp])
        out[self.zero] = 0.0
        return out

    def dense_blocks(self):
        """COO entries of -Hinv restricted to non-diagonal cone blocks."""
        R, C, V = [], [], []
        Jsign = None
        for idx, dim, fac, wbar in self.soc:
            k = idx.shape[0]
            J = np.eye(dim)
            J[1:, 1:] *= -1.0
            blocks = 2.0 * wbar[:, :, None] * wbar[:, None, :] - J[None, :, :]
            blocks *= -fac[:, None, None]
            rr = idx[:, :, None] * np.ones((1, 1, dim), dtype=int)
            cc = idx[:, None, :] * np.ones((1, dim, 1), dtype=int)
            R.append(rr.ravel()); C.append(cc.ravel()); V.append(blocks.ravel())
        for idx, dim, rows, cols, scales, inv_scales, Ws in self.psd:
            w = dim * (dim + 1) // 2
            if w > 2000:
                continue
            for k in range(idx.shape[0]):
                D = np.zeros((w, w))
                for j in range(w):
                    e = np.zeros(w)
                    e[j] = 1.0
                    M = ipm._smat(e, dim, rows, cols, scales)
                    D[:, j] = -ipm._mat_svec(Ws[k] @ M @ Ws[k], rows, cols, inv_scales)
                ii = idx[k]
                R.append(np.repeat(ii, w)); C.append(np.tile(ii, w)); V.append(D.ravel())
        if self.exp is not None:
            idx = self.ops.exp
            k = idx.shape[0]
            rr = idx[:, :, None] * np.ones((1, 1, 3), dtype=int)
            cc = idx[:, None, :] * np.ones((1, 3, 1), dtype=int)
            R.append(rr.ravel()); C.append(cc.ravel()); V.append((-self.exp).ravel())
        if not R:
            return np.empty(0, int), np.empty(0, int), np.empty(0)
        return (np.concatenate(R).astype(np.int64), np.concatenate(C).astype(np.int64),
                np.concatenate(V))


def solve(P, q, A, b, layout, deadline, maxit=120, exp_factor=0.0, verbose=False):
    if len(q) + len(b) > 12000 or A.nnz > 300_000:
        return {"status": "unresolved", "best": None}
    from scipy import sparse as sp
    from scipy.sparse import linalg as spla

    n, m = len(q), len(b)
    ops = ipm.ConeOps(layout, m)
    zi = ops.zero_idx
    As = sp.csc_matrix((A.data, A.indices, A.indptr), shape=(m, n))
    Ps = sp.csc_matrix((P.data, P.indices, P.indptr), shape=(n, n))
    At = As.T.tocsc()
    scale = max(1.0, float(np.max(np.abs(q))) if n else 1.0,
                float(np.max(np.abs(b))) if m else 1.0, A.maxabs, P.maxabs)
    mu0 = scale
    x = np.zeros(n)
    s, z = ops.init(b - As @ x, mu0)
    hybrid = exp_factor > 0.0 and ops.exp is not None
    mu_exp = mu0
    exp_rows = ops.exp.ravel() if hybrid else None
    best, best_score, last_improve = None, np.inf, 0
    I_n = sp.identity(n, format="csc")
    I_m = sp.identity(m, format="csc")
    for it in range(maxit):
        if time.monotonic() > deadline:
            break
        Ax = As @ x
        Px = Ps @ x
        Atz = At @ z
        rp = Ax + s - b
        rd = Px + q + Atz
        mu = max(float(s[ops.conic_idx] @ z[ops.conic_idx]) / ops.nu, 1e-300)
        inf_ = lambda v: float(np.max(np.abs(v), initial=0.0))
        rp_rel = inf_(rp) / (1.0 + max(inf_(Ax), inf_(s), inf_(b)))
        rd_rel = inf_(rd) / (1.0 + max(inf_(Px), inf_(q), inf_(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        gap = abs(p0 - d0)
        gaprel = gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) else 0.0
        score = max(rp_rel, rd_rel, min(gap, gaprel))
        if score < 1.5e-8:
            ok, _ = check_solved(P, q, A, b, layout, x, s, z)
            if ok:
                return dict(status="solved", x=x, s=s, z=z, it=it)
        if score < best_score * (1.0 - 1e-3):
            best_score = score
            last_improve = it
            best = (x.copy(), s.copy(), z.copy())
        elif it - last_improve > 22:
            break
        w = ops.W(s)
        hinv = HInv(ops, s, z, mu, exp_mu=exp_factor * mu_exp if hybrid else None)
        br, bc, bv = hinv.dense_blocks()
        diag22 = -hinv.diag.copy()
        diag22[zi] = 0.0
        bottom = sp.diags(diag22, format="csc")
        if len(bv):
            bottom = bottom + sp.coo_matrix((bv, (br, bc)), shape=(m, m)).tocsc()
        K0 = sp.bmat([[Ps, At], [As, bottom]], format="csc")
        # dual-only static regularization; iterative refinement below restores
        # the exact (unregularized) KKT solution.
        delta = np.concatenate((np.zeros(n), np.full(m, -1e-9 * scale)))
        delta[n + zi] = -1e-11 * scale
        K = K0 + sp.diags(delta, format="csc")
        rcc = z.copy()
        rcc[zi] = 0.0
        if hybrid:
            rcc[exp_rows] = (1.0 - exp_factor) * z[exp_rows]
        rhs1 = np.concatenate((-rd, -rp + hinv.apply(rcc)))
        try:
            lu = spla.splu(K)
            y = lu.solve(rhs1)
            for _ in range(3):
                y += lu.solve(rhs1 - K0 @ y)
        except (RuntimeError, ValueError):
            break
        dx, dz = y[:n], y[n:]
        ds = -rp - As @ dx       # exact primal step, avoids Hinv cancellation
        ds[zi] = 0.0
        ap0 = ops.maxstep(s, ds, False)
        ad0 = ops.maxstep(z, dz, True)
        aaff = min(1.0, 0.90 * min(ap0, ad0))
        mu_aff = max(float((s[ops.conic_idx] + aaff * ds[ops.conic_idx]) @
                           (z[ops.conic_idx] + aaff * dz[ops.conic_idx])) / ops.nu, 0.0)
        sigma = min(0.99, max(1e-4, (mu_aff / mu) ** 3))
        if hybrid:
            rcc = z.copy()
            rcc[zi] = 0.0
            rcc[exp_rows] = (1.0 - exp_factor) * z[exp_rows]
            # reduce mu for the exp central-ray target
            rcc[exp_rows] = (1.0 - exp_factor * sigma) * z[exp_rows]
        else:
            rcc = z - sigma * mu * w
            rcc[zi] = 0.0
        rhs2 = np.concatenate((-rd, -rp + hinv.apply(rcc)))
        try:
            y2 = lu.solve(rhs2)
            for _ in range(3):
                y2 += lu.solve(rhs2 - K0 @ y2)
        except (RuntimeError, ValueError):
            break
        dx, dz = y2[:n], y2[n:]
        ds = -rp - As @ dx
        ds[zi] = 0.0
        ap = ops.maxstep(s, ds, False)
        ad = ops.maxstep(z, dz, True)
        alpha = min(ap, ad)
        alpha_p = min(1.0, 0.95 * ap)
        alpha_d = min(1.0, 0.95 * ad)
        if alpha < 1e-7:
            break
        x += alpha_p * dx
        s += alpha_p * ds
        z += alpha_d * dz
        s[zi] = 0.0
        if hybrid:
            mu_exp *= exp_factor
        if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
            break
        if verbose:
            print("sp", it, "mu", mu, "a", alpha, "rp", rp_rel, "rd", rd_rel,
                  "gap", gaprel, flush=True)
    if best is not None:
        ok, _ = check_solved(P, q, A, b, layout, best[0], best[1], best[2])
        if ok:
            return dict(status="solved", x=best[0], s=best[1], z=best[2], it=-1)
    return dict(status="unresolved", x=x, s=s, z=z, best=best, it=it)


def solve_diag_sparse(P, q, A, b, layout, deadline, maxit=120, verbose=False):
    """Sparse normal-equations IPM for problems with only zero/nonneg cones."""
    if len(q) + len(b) > 10000 or A.nnz > 300_000:
        return {"status": "unresolved", "best": None}
    from scipy import sparse as sp
    from scipy.sparse import linalg as spla

    n, m = len(q), len(b)
    ops = ipm.ConeOps(layout, m)
    zi = ops.zero_idx
    m0 = zi.size
    ni = ops.nn_idx
    As = sp.csc_matrix((A.data, A.indices, A.indptr), shape=(m, n))
    Ps = sp.csc_matrix((P.data, P.indices, P.indptr), shape=(n, n))
    At = As.T.tocsc()
    A0 = As[zi, :] if m0 else None
    scale = max(1.0, float(np.max(np.abs(q))) if n else 1.0,
                float(np.max(np.abs(b))) if m else 1.0, A.maxabs, P.maxabs)
    reg = 1e-12 * scale
    x = np.zeros(n)
    s, z = ops.init(b - As @ x, scale)
    best, best_score = None, np.inf
    In = sp.identity(n, format="csc")
    for it in range(maxit):
        if time.monotonic() > deadline:
            break
        Ax = As @ x
        Px = Ps @ x
        Atz = At @ z
        rp = Ax + s - b
        rd = Px + q + Atz
        mu = max(float(s[ops.conic_idx] @ z[ops.conic_idx]) / ops.nu, 1e-300)
        inf_ = lambda v: float(np.max(np.abs(v), initial=0.0))
        rp_rel = inf_(rp) / (1.0 + max(inf_(Ax), inf_(s), inf_(b)))
        rd_rel = inf_(rd) / (1.0 + max(inf_(Px), inf_(q), inf_(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        gap = abs(p0 - d0)
        gaprel = gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) else 0.0
        score = max(rp_rel, rd_rel, min(gap, gaprel))
        if score < 1.5e-8:
            ok, _ = check_solved(P, q, A, b, layout, x, s, z)
            if ok:
                return dict(status="solved", x=x, s=s, z=z, it=it)
        if score < best_score * (1.0 - 1e-3):
            best_score = score
            last_improve = it
            best = (x.copy(), s.copy(), z.copy())
        elif it - last_improve > 22:
            break
        w = ops.W(s)
        h = z[ni] / np.maximum(s[ni], 1e-300)
        An = As[ni, :]
        M = (Ps + An.T @ sp.diags(h) @ An).tocsc()
        rcc = z.copy()
        rcc[zi] = 0.0
        Hrp = np.zeros(m)
        Hrp[ni] = h * rp[ni]
        if m0:
            K = sp.bmat([[M + reg * In, A0.T],
                         [A0, -reg * sp.identity(m0, format="csc")]], format="csc")
            rhs_m = -rd - At @ Hrp + At @ rcc
            rhs1 = np.concatenate((rhs_m, -rp[zi]))
        else:
            K = (M + reg * In).tocsc()
            rhs1 = -rd - At @ Hrp + At @ rcc
        try:
            lu = spla.splu(K)
            y = lu.solve(rhs1)
            y += lu.solve(rhs1 - K @ y)
        except (RuntimeError, ValueError):
            break
        dx = y[:n]
        dz0 = y[n:]
        Adx = As @ dx
        dz = np.zeros(m)
        dz[ni] = h * (Adx[ni] + rp[ni]) - rcc[ni]
        ds = -rp - Adx
        ds[zi] = 0.0
        if m0:
            dz[zi] = dz0
        ap0 = ops.maxstep(s, ds, False)
        ad0 = ops.maxstep(z, dz, True)
        aaff = min(1.0, 0.90 * min(ap0, ad0))
        mu_aff = max(float((s[ops.conic_idx] + aaff * ds[ops.conic_idx]) @
                           (z[ops.conic_idx] + aaff * dz[ops.conic_idx])) / ops.nu, 0.0)
        sigma = min(0.99, max(1e-4, (mu_aff / mu) ** 3))
        rcc = z - sigma * mu * w
        rcc[zi] = 0.0
        if m0:
            rhs_m = -rd - At @ Hrp + At @ rcc
            rhs2 = np.concatenate((rhs_m, -rp[zi]))
        else:
            rhs2 = -rd - At @ Hrp + At @ rcc
        try:
            y2 = lu.solve(rhs2)
            y2 += lu.solve(rhs2 - K @ y2)
        except (RuntimeError, ValueError):
            break
        dx = y2[:n]
        dz0 = y2[n:]
        Adx = As @ dx
        dz = np.zeros(m)
        dz[ni] = h * (Adx[ni] + rp[ni]) - rcc[ni]
        if m0:
            dz[zi] = dz0
        ds = -rp - Adx
        ds[zi] = 0.0
        ap = ops.maxstep(s, ds, False)
        ad = ops.maxstep(z, dz, True)
        alpha = min(1.0, 0.90 * min(ap, ad))
        if alpha < 1e-7:
            break
        x += alpha * dx
        s += alpha * ds
        z += alpha * dz
        s[zi] = 0.0
        if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
            break
        if verbose:
            print("ds", it, "mu", mu, "a", alpha, "rp", rp_rel, "rd", rd_rel, flush=True)
    if best is not None:
        ok, _ = check_solved(P, q, A, b, layout, best[0], best[1], best[2])
        if ok:
            return dict(status="solved", x=best[0], s=best[1], z=best[2], it=-1)
    return dict(status="unresolved", x=x, s=s, z=z, best=best, it=it)
