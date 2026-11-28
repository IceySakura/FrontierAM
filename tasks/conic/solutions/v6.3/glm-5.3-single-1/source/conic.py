#!/usr/bin/env python3
"""General convex conic solver: LP / QP / SOCP / SDP / exponential cones.

Problem:  min 0.5 x'Px + q'x   s.t.  Ax + s = b,  s in K.

Primal-dual interior point method.  Each cone block uses its logarithmically
homogeneous self-concordant barrier psi; the central path is z = -mu*grad psi(s).
The Newton system is reduced to a symmetric quasi-definite augmented system

   [ P + dp I     A'       ] [dx]   [ -rd            ]
   [ A         -(W + dd I) ] [dz] = [ -rp + Ms^-1 rc ]

solved with a sparse LU plus iterative refinement.  Mehrotra style
predictor/corrector, primal and dual step lengths, and full self-verification
against the independent acceptance conditions before reporting a status.
"""
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

T0 = time.perf_counter()
SQRT2 = math.sqrt(2.0)
TOL = 1e-8
AIM = 5e-10
INF = float("inf")


# --------------------------------------------------------------------- utils
def norm(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


def smat(v, triu, offd, d):
    M = np.zeros((d, d))
    r, c = triu
    M[r, c] = v * np.where(offd, 1.0 / SQRT2, 1.0)
    M[c, r] = M[r, c]
    return M


def svec(M, triu, offd):
    r, c = triu
    return M[r, c] * np.where(offd, SQRT2, 1.0)


class Blk(object):
    __slots__ = ("kind", "start", "dim", "w", "deg", "triu", "offd")


def build_blocks(conelist):
    blocks = []
    pos = 0
    for c in conelist:
        b = Blk()
        b.kind = c["type"]
        b.dim = int(c["dim"])
        if b.kind == "psd":
            b.w = b.dim * (b.dim + 1) // 2
            b.deg = float(b.dim)
        else:
            b.w = b.dim
            b.deg = float(b.dim if b.kind == "nonnegative" else
                          (2.0 if b.kind in ("soc", "exponential") else 0.0))
        b.start = pos
        pos += b.w
        if b.kind == "psd":
            dd = b.dim
            rr = np.array([ri for ci in range(dd) for ri in range(ci + 1)], dtype=np.int64)
            cc = np.array([ci for ci in range(dd) for ri in range(ci + 1)], dtype=np.int64)
            b.triu = (rr, cc)
            b.offd = (rr != cc)
        blocks.append(b)
    return blocks, pos


def interior_point(blocks, m):
    s0 = np.zeros(m)
    z0 = np.zeros(m)
    for b in blocks:
        i, w, d = b.start, b.w, b.dim
        if b.kind == "nonnegative":
            s0[i:i + w] = 1.0
            z0[i:i + w] = 1.0
        elif b.kind == "soc":
            s0[i] = 1.0
            z0[i] = 1.0
        elif b.kind == "exponential":
            s0[i:i + 3] = (-1.0, 1.0, 2.0)
            z0[i:i + 3] = (-1.0, 0.0, 1.0)
        elif b.kind == "psd":
            idx = np.arange(d)
            dp = i + idx + idx * (idx + 1) // 2
            s0[dp] = 1.0
            z0[dp] = 1.0
    return s0, z0


# ------------------------------------------------------------ cone algebra
def block_data(b, s, z):
    """Return (W, grad, mu) for a dense block (soc/exp/psd)."""
    i, w, d = b.start, b.w, b.dim
    sb = s[i:i + w]
    zb = z[i:i + w]
    ip = float(sb @ zb)
    mu = ip / b.deg if b.deg > 0 else 1.0
    if not (mu > 0) or not np.isfinite(mu):
        mu = 1e-300
    if b.kind == "soc":
        t = sb[0]
        u = sb[1:]
        g = t * t - float(u @ u)
        if not (g > 0) or not np.isfinite(g):
            g = 1e-300
        grad = np.empty(w)
        grad[0] = -2.0 * t / g
        grad[1:] = 2.0 * u / g
        W = np.outer(sb, sb)
        dg = np.full(w, 0.5 * g)
        dg[0] = -0.5 * g
        W[np.diag_indices(w)] += dg
        W = W / mu
    elif b.kind == "exponential":
        uu, vv, ww = float(sb[0]), float(sb[1]), float(sb[2])
        a = uu / vv
        E = math.exp(a) if a < 700 else math.exp(700)
        r = ww - vv * E
        if not (r > 0) or not np.isfinite(r):
            r = 1e-300
        gr = np.array([-E, -E * (1.0 - a), 1.0])
        grad = -gr / r
        grad[1] -= 1.0 / vv
        H = np.outer(gr, gr) / (r * r)
        Hr = np.zeros((3, 3))
        Hr[0, 0] = -E / vv
        Hr[0, 1] = Hr[1, 0] = uu * E / (vv * vv)
        Hr[1, 1] = -(a * a * E) / vv
        H = H - Hr / r
        H[1, 1] += 1.0 / (vv * vv)
        W = np.linalg.inv(H) / mu
    else:  # psd
        S = smat(sb, b.triu, b.offd, d)
        S = 0.5 * (S + S.T)
        Si = np.linalg.inv(S)
        grad = -svec(Si, b.triu, b.offd)
        W = np.empty((w, w))
        r, c = b.triu
        for j in range(w):
            r1, c1 = r[j], c[j]
            if r1 == c1:
                M = np.outer(S[:, r1], S[:, r1])
            else:
                M = (np.outer(S[:, r1], S[:, c1]) +
                     np.outer(S[:, c1], S[:, r1])) / SQRT2
            W[:, j] = svec(M, b.triu, b.offd)
        W = W / mu
    return W, grad, mu


def max_step_block(b, s, ds):
    """Largest alpha with s + alpha*ds strictly inside the cone (cap 1)."""
    i, w = b.start, b.w
    k = b.kind
    if k == "nonnegative":
        neg = ds < 0
        if not neg.any():
            return 1.0
        return min(1.0, float(np.min(-s[neg] / ds[neg])))
    if k == "soc":
        t, u = s[0], s[1:]
        dt, du = ds[0], ds[1:]
        g0 = t * t - float(u @ u)
        g1 = 2.0 * (t * dt - float(u @ du))
        g2 = dt * dt - float(du @ du)
        lim = 1.0
        if dt < 0:
            lim = min(lim, t / (-dt))
        if g1 < 0 or g2 < 0:
            if abs(g2) < 1e-300:
                if g1 < 0:
                    lim = min(lim, -g0 / g1)
            else:
                disc = g1 * g1 - 4.0 * g2 * g0
                if disc >= 0:
                    sq = math.sqrt(disc)
                    cand = [x for x in ((-g1 + sq) / (2 * g2), (-g1 - sq) / (2 * g2)) if x > 0]
                    if cand:
                        lim = min(lim, min(cand))
                else:
                    if g2 < 0:
                        lim = min(lim, -g1 / (2 * g2)) if g1 < 0 else 1.0
        return max(min(1.0, lim), 1e-12)
    if k == "exponential":
        u0, v0, w0 = s[0], s[1], s[2]
        du, dv, dw = ds[0], ds[1], ds[2]

        def ok(a):
            v = v0 + a * dv
            if v <= 0:
                return False
            wv = w0 + a * dw
            if wv <= 0:
                return False
            uu = u0 + a * du
            ex = uu / v
            if ex > 700:
                return False
            return v * math.exp(ex) < wv
        if ok(1.0):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return max(lo, 1e-12)
    # psd
    d = b.dim
    S = smat(s, b.triu, b.offd, d)
    S = 0.5 * (S + S.T)
    dS = smat(ds, b.triu, b.offd, d)
    dS = 0.5 * (dS + dS.T)
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L, dS)
        V = np.linalg.solve(L, X.T).T
        V = 0.5 * (V + V.T)
        ev = np.linalg.eigvalsh(V)
    except np.linalg.LinAlgError:
        return 1e-9
    lmin = float(ev[0])
    if lmin >= 0:
        return 1.0
    return max(min(1.0, -1.0 / lmin * 0.999999), 1e-12)


def max_step_dual_block(b, z, dz):
    """Largest alpha with z + alpha*dz in the interior of the dual cone."""
    k = b.kind
    i, w = b.start, b.w
    if k in ("nonnegative", "psd"):
        # dual cone is the same for nonnegative and psd
        return max_step_block(b, z, dz)
    if k == "zero":
        return 1.0
    if k == "soc":
        return max_step_block(b, z, dz)
    # exponential: a<0 and -a*exp(b/a) < e*c
    a0, b0, c0 = z[0], z[1], z[2]
    da, db, dc = dz[0], dz[1], dz[2]

    def ok(al):
        a = a0 + al * da
        if a >= 0:
            return False
        ex = (b0 + al * db) / a
        if ex > 700:
            return False
        c = c0 + al * dc
        return -a * math.exp(ex) < math.e * c
    if ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return max(lo, 1e-12)


def cone_defect(blocks, v, dual):
    """Max normalized cone defect (mirrors the independent checker)."""
    worst = 0.0
    for b in blocks:
        i, w, d = b.start, b.w, b.dim
        blk = v[i:i + w]
        nrm = norm(blk)
        u = blk / max(1.0, nrm) if nrm > 0 else blk
        k = b.kind
        if k == "zero":
            if not dual:
                worst = max(worst, norm(u))
        elif k == "nonnegative":
            worst = max(worst, max(0.0, -float(np.min(u))) if w else 0.0)
        elif k == "soc":
            worst = max(worst, max(0.0, float(np.linalg.norm(u[1:])) - float(u[0])))
        elif k == "exponential":
            if dual:
                uu = (-u[1], -u[0], math.e * u[2])
            else:
                uu = (float(u[0]), float(u[1]), float(u[2]))
            worst = max(worst, exp_defect(*uu))
        else:
            M = smat(u, b.triu, b.offd, d)
            M = 0.5 * (M + M.T)
            try:
                ev = np.linalg.eigvalsh(M)
                worst = max(worst, max(0.0, -float(ev[0])))
            except np.linalg.LinAlgError:
                worst = max(worst, 1.0)
    return worst


def exp_defect(a, b, c):
    boundary = max(abs(b), max(a, 0.0), max(-c, 0.0))
    if b > 0.0 and c > 0.0:
        interior = max(0.0, a - b * (math.log(c) - math.log(b)))
        return min(interior, boundary)
    return boundary


# ------------------------------------------------------------------- solver
class KKT(object):
    """Sparse augmented system with a fixed pattern; only W changes."""

    def __init__(self, P, A, blocks, m, n, rowkind):
        self.n = n
        self.m = m
        N = n + m
        self.N = N
        Pc = P.tocoo()
        Ac = A.tocoo()
        self.nn = np.flatnonzero(rowkind == 1)
        self.zz = np.flatnonzero(rowkind == 0)
        self.dense = [b for b in blocks if b.kind in ("soc", "psd", "exponential")]
        wr = []
        wc = []
        if self.nn.size:
            wr.append(self.nn + n)
            wc.append(self.nn + n)
        if self.zz.size:
            wr.append(self.zz + n)
            wc.append(self.zz + n)
        for b in self.dense:
            idx = np.arange(b.w)
            wr.append(np.tile(idx, b.w) + n + b.start)
            wc.append(np.repeat(idx, b.w) + n + b.start)
        wrows = np.concatenate(wr) if wr else np.zeros(0, dtype=np.int64)
        wcols = np.concatenate(wc) if wc else np.zeros(0, dtype=np.int64)
        rows = np.concatenate([Pc.row, np.arange(n), n + Ac.row, Ac.col, wrows])
        cols = np.concatenate([Pc.col, np.arange(n), Ac.col, n + Ac.row, wcols])
        vals = np.concatenate([Pc.data, np.zeros(n), Ac.data, Ac.data,
                               np.zeros(wrows.size)])
        K = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        K.sum_duplicates()
        K.sort_indices()
        self.K = K
        self.static = K.data.copy()
        self.nnzW = wrows.size
        # position lookup via sorted keys
        colof = np.repeat(np.arange(N), np.diff(K.indptr))
        keys = colof * (N + 2) + K.indices
        self.wpos = np.searchsorted(keys, wcols * (N + 2) + wrows)
        drows = np.arange(m) + n
        self.dpos = np.searchsorted(keys, drows * (N + 2) + drows)
        self.xdpos = np.searchsorted(keys, np.arange(n) * (N + 2) + np.arange(n))
        self.keys_ok = True

    def factor(self, wvals, dp, dd):
        K = self.K
        K.data = self.static.copy()
        K.data[self.wpos] = -wvals
        K.data[self.dpos] -= dd
        K.data[self.xdpos] += dp
        try:
            self.lu = splu(K, permc_spec="MMD_AT_PLUS_A",
                           diag_pivot_thresh=0.1, options={"SymmetricMode": True})
            return True
        except Exception:
            try:
                self.lu = splu(K)
                return True
            except Exception:
                self.lu = None
                return False

    def solve(self, P, A, rhs_x, rhs_z, Wapply, nref=5):
        n, m = self.n, self.m
        rhs = np.concatenate([rhs_x, rhs_z])
        sol = self.lu.solve(rhs)
        if sol is None or not np.all(np.isfinite(sol)):
            return None, None
        for _ in range(nref):
            dx = sol[:n]
            dz = sol[n:]
            e1 = P @ dx + A.T @ dz - rhs_x
            e2 = A @ dx - Wapply(dz) - rhs_z
            corr = self.lu.solve(np.concatenate([-e1, -e2]))
            if corr is None or not np.all(np.isfinite(corr)):
                break
            sol = sol + corr
            if norm(corr) <= 1e-13 * max(1.0, norm(sol)):
                break
        return sol[:n], sol[n:]


def solve_problem(P, q, A, b, blocks, m, n, nu, deadline, maxit=120, verbose=False):
    rowkind = np.zeros(m, dtype=np.int8)
    for bk in blocks:
        k = {"zero": 0, "nonnegative": 1, "soc": 2, "exponential": 3, "psd": 4}[bk.kind]
        rowkind[bk.start:bk.start + bk.w] = k
    s, z = interior_point(blocks, m)
    x = np.zeros(n)
    dense = [bk for bk in blocks if bk.kind in ("soc", "psd", "exponential")]
    nn = np.flatnonzero(rowkind == 1)
    zz = np.flatnonzero(rowkind == 0)
    kkt = KKT(P, A, blocks, m, n, rowkind)
    At = A.T.tocsc()
    nb = norm(b)
    best = None
    best_score = None
    status = "unresolved"
    mu_prev = None
    for it in range(maxit):
        if time.perf_counter() > deadline:
            break
        Ax = A @ x
        r_p = Ax + s - b
        Px = P @ x
        Atz = At @ z
        r_d = Px + q + Atz
        mu = float(s @ z) / nu if nu > 0 else 0.0
        # ---------- progress / convergence
        nAx, ns, nPx, nq, nAtz = norm(Ax), norm(s), norm(Px), norm(q), norm(Atz)
        rp_n = norm(r_p) / (1.0 + max(nAx, ns, nb))
        rd_n = norm(r_d) / (1.0 + max(nPx, nq, nAtz))
        xPx = float(x @ Px)
        qx = float(q @ x)
        bz = float(b @ z)
        p0 = 0.5 * xPx + qx
        d0 = -0.5 * xPx - bz
        gap = abs(p0 - d0)
        sc = max(abs(p0), abs(d0))
        gap_n = gap / sc if sc > 0 else 0.0
        if verbose:
            sys.stderr.write("it %d mu %.2e rp %.2e rd %.2e gap %.2e\n" %
                             (it, mu, rp_n, rd_n, min(gap, gap_n)))
        score = max(rp_n, rd_n, min(gap, gap_n))
        if score <= 0.4 * TOL:
            if (cone_defect(blocks, s, False) <= 0.4 * TOL and
                    cone_defect(blocks, z, True) <= 0.4 * TOL):
                status = "solved"
                break
        if best_score is None or score < best_score:
            best_score = score
            best = (x.copy(), s.copy(), z.copy())
        if mu <= 0 or not np.isfinite(mu):
            break
        # ---------- cone data
        Wd = np.zeros(m)
        grad = np.zeros(m)
        mus = {}
        Wmats = {}
        ok = True
        for bk in dense:
            try:
                W, gr, mu_b = block_data(bk, s, z)
            except Exception:
                ok = False
                break
            Wmats[bk] = W
            mus[bk] = mu_b
            grad[bk.start:bk.start + bk.w] = gr
        if not ok:
            break
        if nn.size:
            sn = s[nn]
            zn = z[nn]
            muv = sn * zn
            muv = np.where(muv > 0, muv, 1e-300)
            Wd[nn] = sn * sn / muv
            grad[nn] = -1.0 / sn
        wvals = []
        if nn.size:
            wvals.append(Wd[nn])
        if zz.size:
            wvals.append(np.zeros(zz.size))
        for bk in dense:
            wvals.append(Wmats[bk].ravel(order="F"))
        wvals = np.concatenate(wvals) if wvals else np.zeros(0)
        scale = max(1.0, norm(P.data) if P.nnz else 0.0,
                    norm(A.data) if A.nnz else 0.0)
        dp = 1e-10 * scale
        dd = 1e-10 * scale
        if not kkt.factor(wvals, dp, dd):
            break

        def Wapply(v):
            out = np.zeros(m)
            if nn.size:
                out[nn] = Wd[nn] * v[nn]
            for bk in dense:
                i = bk.start
                out[i:i + bk.w] = Wmats[bk] @ v[i:i + bk.w]
            return out

        def msinv_rc(mu_tar, dsa=None, dza=None):
            out = np.zeros(m)
            if nn.size:
                val = s[nn] - mu_tar / z[nn]
                if dsa is not None:
                    val = val + dsa[nn] * dza[nn] / z[nn]
                out[nn] = val
            for bk in dense:
                i, w = bk.start, bk.w
                rc = z[i:i + w] + mu_tar * grad[i:i + w]
                out[i:i + w] = Wmats[bk] @ rc
            return out

        def make_ds(dx, dz, mu_tar, dsa=None, dza=None):
            ds = np.zeros(m)      # zero-cone rows keep s == 0 exactly
            if nn.size:
                ds[nn] = -s[nn] + mu_tar / z[nn] - (s[nn] / z[nn]) * dz[nn]
                if dsa is not None:
                    ds[nn] -= dsa[nn] * dza[nn] / z[nn]
            for bk in dense:
                i, w = bk.start, bk.w
                rc = z[i:i + w] + mu_tar * grad[i:i + w]
                ds[i:i + w] = -Wmats[bk] @ (rc + dz[i:i + w])
            return ds

        # ---------- affine (predictor)
        rc_aff = msinv_rc(0.0)
        dx_a, dz_a = kkt.solve(P, A, -r_d, -r_p + rc_aff, Wapply)
        if dx_a is None:
            break
        ds_a = make_ds(dx_a, dz_a, 0.0)
        ap = 1.0
        ad = 1.0
        for bk in blocks:
            if bk.kind == "zero":
                continue
            i, w = bk.start, bk.w
            ap = min(ap, max_step_block(bk, s[i:i + w], ds_a[i:i + w]))
            ad = min(ad, max_step_dual_block(bk, z[i:i + w], dz_a[i:i + w]))
        ap = min(1.0, ap)
        ad = min(1.0, ad)
        mu_aff = float((s + ap * ds_a) @ (z + ad * dz_a)) / nu if nu > 0 else 0.0
        sig = (mu_aff / mu) ** 3 if mu > 0 and mu_aff > 0 else 0.1
        sig = min(max(sig, 0.05), 0.9)
        # ---------- corrector
        mu_tar = sig * mu
        rc_c = msinv_rc(mu_tar, ds_a, dz_a)
        dx, dz = kkt.solve(P, A, -r_d, -r_p + rc_c, Wapply)
        if dx is None:
            break
        ds = make_ds(dx, dz, mu_tar, ds_a, dz_a)
        ap = 1.0
        ad = 1.0
        for bk in blocks:
            if bk.kind == "zero":
                continue
            i, w = bk.start, bk.w
            ap = min(ap, max_step_block(bk, s[i:i + w], ds[i:i + w]))
            ad = min(ad, max_step_dual_block(bk, z[i:i + w], dz[i:i + w]))
        tau = 0.995 if min(rp_n, rd_n) < 1e-6 else 0.98
        ap = min(1.0, tau * ap)
        ad = min(1.0, tau * ad)
        if ap <= 1e-12 or ad <= 1e-12:
            break
        x = x + ap * dx
        s = s + ap * ds
        z = z + ad * dz
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(z)) and
                np.all(np.isfinite(x))):
            break
    if status != "solved" and best is not None:
        x, s, z = best
        status = "solved"          # run() re-verifies and downgrades if needed
    return status, x, s, z


# ----------------------------------------------------------- verification
def verify(P, q, A, b, blocks, x, s, z, tol=TOL):
    Ax = A @ x
    Px = P @ x
    Atz = A.T @ z
    if not (np.all(np.isfinite(Ax)) and np.all(np.isfinite(Px)) and
            np.all(np.isfinite(Atz)) and np.all(np.isfinite(x)) and
            np.all(np.isfinite(s)) and np.all(np.isfinite(z))):
        return False, {}
    nb, nq = norm(b), norm(q)
    rp = norm(Ax + s - b) / (1.0 + max(norm(Ax), norm(s), nb))
    rd = norm(Px + q + Atz) / (1.0 + max(norm(Px), nq, norm(Atz)))
    xPx = float(x @ Px)
    p0 = 0.5 * xPx + float(q @ x)
    d0 = -0.5 * xPx - float(b @ z)
    gap = abs(p0 - d0)
    sc = max(abs(p0), abs(d0))
    gn = gap / sc if sc > 0 else 0.0
    cd_s = cone_defect(blocks, s, False)
    cd_z = cone_defect(blocks, z, True)
    okk = (rp <= tol and rd <= tol and cd_s <= tol and cd_z <= tol and
           (gap <= tol or gn <= tol))
    return okk, {"rp": rp, "rd": rd, "gap": gap, "gn": gn, "cds": cd_s, "cdz": cd_z,
                 "p0": p0, "d0": d0}


def run(path_in, path_out, budget=120.0, verbose=False):
    with np.load(path_in, allow_pickle=False) as d:
        q = np.ascontiguousarray(d["q"], dtype=np.float64)
        b = np.ascontiguousarray(d["b"], dtype=np.float64)
        P = sp.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]),
                          shape=tuple(d["P_shape"]))
        A = sp.csc_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                          shape=tuple(d["A_shape"]))
        meta = json.loads(str(d["metadata_json"]))
    P = P.astype(np.float64)
    A = A.astype(np.float64)
    P.sum_duplicates()
    A.sum_duplicates()
    n = q.size
    m = b.size
    blocks, mm = build_blocks(meta["cones"])
    if mm != m:
        raise ValueError("cone dimension mismatch")
    offset = float(meta.get("objective_offset", 0.0) or 0.0)
    nu = sum(bk.deg for bk in blocks)
    status, x, s, z = solve_problem(P, q, A, b, blocks, m, n, nu,
                                    T0 + budget, verbose=verbose)
    info = {}
    okk = False
    if status == "solved":
        okk, info = verify(P, q, A, b, blocks, x, s, z)
        if not okk:
            status = "unresolved"
    out = {"schema_version": 1, "status": status}
    arrays = {}
    if status == "solved":
        out["objective_primal"] = float(info["p0"] + offset)
        out["objective_dual"] = float(info["d0"] + offset)
        arrays = {"x": x, "s": s, "z": z}
    elif status == "primal_infeasible":
        arrays = {"z": z}
    elif status == "dual_infeasible":
        arrays = {"x": x, "s": s}
    else:
        arrays = {"x": np.zeros(n), "s": np.zeros(m), "z": np.zeros(m)}
    arrays["metadata_json"] = np.array(json.dumps(out))
    with open(path_out, "wb") as fh:
        np.savez(fh, **arrays)
    return status, info


def main():
    a = sys.argv[1:]
    inp = out = None
    i = 0
    while i < len(a):
        if a[i] == "--input":
            inp = a[i + 1]
            i += 2
        elif a[i] == "--output":
            out = a[i + 1]
            i += 2
        elif a[i] == "--verbose":
            i += 1
        else:
            i += 1
    verbose = "--verbose" in a
    budget = 100.0
    try:
        status, info = run(inp, out, budget, verbose)
    except Exception as exc:  # keep a valid output file on any failure
        arrays = {"metadata_json": np.array(json.dumps(
            {"schema_version": 1, "status": "unresolved"}))}
        try:
            with open(out, "wb") as fh:
                np.savez(fh, **arrays)
        except Exception:
            pass
        sys.stderr.write("error: %r\n" % (exc,))
        return 0
    if verbose:
        sys.stderr.write("%s %s\n" % (status, info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
