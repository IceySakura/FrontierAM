#!/usr/bin/env python3
"""High accuracy conic optimizer for

    min  1/2 x'Px + q'x + c0    s.t.  Ax + s = b,  s in K

K = product of zero / nonnegative / second-order / exponential / PSD cones.

Method: infeasible-start primal-dual path-following interior point method with a
Mehrotra style predictor-corrector step.  Every cone family is handled through
the gradient and Hessian of its standard self-concordant barrier

    nonnegative  -sum log s_i                        (nu = 1 per row)
    soc          -log(t^2 - |v|^2)                   (nu = 2)
    exponential  -log v - log w - log(v log(w/v) - u)(nu = 3)
    psd          -logdet S                           (nu = order)

so the algorithm is one uniform damped Newton scheme on

    F1 = Px + q + A^T z = 0,  F2 = Ax + s - b = 0,  F3 = z + mu grad phi(s) = 0

and only the per block linear algebra differs.  Linear systems are reduced to a
dense Schur complement assembled with sparsity exploiting helpers from
libfast.so and factorized with numpy/LAPACK.  scipy is deliberately not imported
(interpreter start-up is charged against the per instance time limit).
"""
import ctypes
import json
import math
import os
import sys
import time

import numpy as np

T0 = time.perf_counter()
HERE = os.path.dirname(os.path.abspath(__file__))
SQ2 = math.sqrt(2.0)
EE = math.e
REG_P = 1e-11

# --------------------------------------------------------------------------
# optional C helpers (built by build.sh); numpy fallbacks otherwise
# --------------------------------------------------------------------------
_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)


def _load_lib():
    for name in ("libfast.so", "fast.so"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(p)
                sig = {
                    "csr_mv": [_dp, _ip, _ip, _dp, _dp, ctypes.c_int],
                    "csc_mvT": [_dp, _ip, _ip, _dp, _dp, ctypes.c_int],
                    "schur_diag": [_dp, _ip, _ip, _dp, ctypes.c_int, ctypes.c_int, _dp, ctypes.c_int],
                    "schur_gen": [_dp, _ip, _ip, ctypes.c_int, ctypes.c_int, _dp, _dp, ctypes.c_int],
                }
                for f, s in sig.items():
                    getattr(lib, f).argtypes = s
                    getattr(lib, f).restype = None
                return lib
            except Exception:
                pass
    return None


LIB = _load_lib()


def csr_mv(d, i, p, x, out, m):
    if LIB is not None:
        LIB.csr_mv(d.ctypes.data_as(_dp), i.ctypes.data_as(_ip), p.ctypes.data_as(_ip),
                   x.ctypes.data_as(_dp), out.ctypes.data_as(_dp), m)
    else:
        out[:] = np.bincount(rowexp(p, m), weights=d * x[i], minlength=m)


def csc_mvT(d, i, p, x, out, n):
    if LIB is not None:
        LIB.csc_mvT(d.ctypes.data_as(_dp), i.ctypes.data_as(_ip), p.ctypes.data_as(_ip),
                    x.ctypes.data_as(_dp), out.ctypes.data_as(_dp), n)
    else:
        cols = np.repeat(np.arange(n), np.diff(p))
        out[:] = np.bincount(cols, weights=d * x[i], minlength=n)


def rowexp(p, m):
    return np.repeat(np.arange(m, dtype=np.int64), np.diff(p))


def schur_diag(d, i, p, w, r0, r1, M, n):
    if LIB is not None:
        LIB.schur_diag(d.ctypes.data_as(_dp), i.ctypes.data_as(_ip), p.ctypes.data_as(_ip),
                       w.ctypes.data_as(_dp), r0, r1, M.ctypes.data_as(_dp), n)
    else:
        for r in range(r0, r1):
            s, e = p[r], p[r + 1]
            if e == s or w[r] == 0.0:
                continue
            j = i[s:e]
            M[np.ix_(j, j)] += w[r] * np.outer(d[s:e], d[s:e])


def schur_gen(d, i, p, r0, k, H, M, n):
    if LIB is not None:
        LIB.schur_gen(d.ctypes.data_as(_dp), i.ctypes.data_as(_ip), p.ctypes.data_as(_ip),
                      r0, k, H.ctypes.data_as(_dp), M.ctypes.data_as(_dp), n)
    else:
        Hc = np.ascontiguousarray(H)
        rows = [np.arange(p[r0 + a], p[r0 + a + 1]) for a in range(k)]
        for a in range(k):
            if rows[a].size == 0:
                continue
            ja = i[rows[a]]
            da = d[rows[a]]
            for b in range(a + 1):
                if rows[b].size == 0:
                    continue
                h = Hc[a, b] if a == b else 2.0 * Hc[a, b]
                if h == 0.0:
                    continue
                jb = i[rows[b]]
                blk = h * np.outer(da, d[rows[b]])
                M[np.ix_(ja, jb)] += blk
                if a != b:
                    M[np.ix_(jb, ja)] += blk.T


def svec_maps(d):
    """(scatter maps) for svec -> matrix and (gather maps) for matrix -> svec."""
    p, rr, cc, ff = [], [], [], []
    gr, gc, gf = [], [], []
    pos = 0
    for c in range(d):
        for r in range(c + 1):
            f = 1.0 if r == c else 1.0 / SQ2
            p.append(pos + r)
            rr.append(r)
            cc.append(c)
            ff.append(f)
            if r < c:
                p.append(pos + r)
                rr.append(c)
                cc.append(r)
                ff.append(f)
            gr.append(r)
            gc.append(c)
            gf.append(SQ2 if r < c else 1.0)
        pos += c + 1
    return ((np.array(p, dtype=np.int64), np.array(rr, dtype=np.int64),
             np.array(cc, dtype=np.int64), np.array(ff)),
            (np.array(gr, dtype=np.int64), np.array(gc, dtype=np.int64), np.array(gf)))


def dense_from_csr(d, i, p, m, n):
    rows = rowexp(p, m)
    flat = rows * n + i.astype(np.int64)
    return np.bincount(flat, weights=d, minlength=m * n).reshape(m, n)


def row_slice(d, i, p, r0, r1):
    return d[p[r0]:p[r1]], i[p[r0]:p[r1]], p[r0:r1 + 1] - p[r0]


# --------------------------------------------------------------------------
# cone block groups
# --------------------------------------------------------------------------
class Group:
    """All blocks of one (type,dim) family; rows are contiguous."""

    def __init__(self, kind, dim, nblk, start, nz, n):
        self.kind = kind
        self.dim = dim
        if kind == "psd":
            self.width = dim * (dim + 1) // 2
        elif kind == "nonnegative":
            self.width = 1
        else:
            self.width = dim
        self.nblk = nblk
        self.start = start
        self.stop = start + nblk * self.width
        self.a = start - nz
        self.b = self.stop - nz
        self.n = n
        self.nu = {"nonnegative": nblk, "soc": 2 * nblk,
                   "exponential": 3 * nblk}.get(kind, dim * nblk)
        if kind == "psd":
            (self.pi, self.ri, self.ci, self.fi), (self.gr, self.gc, self.gf) = svec_maps(dim)
        self.sg = None
        self.c2_full = None

    def prepare(self, sC):
        self.sg = sC[self.a:self.b].reshape(self.nblk, self.width)
        k = self.kind
        if k == "nonnegative":
            self.c1 = 1.0 / self.sg[:, 0]
            self.c2 = self.c1 * self.c1
        elif k == "soc":
            self.tt = np.ascontiguousarray(self.sg[:, 0])
            self.vv = np.ascontiguousarray(self.sg[:, 1:])
            self.g = self.tt * self.tt - np.einsum("ij,ij->i", self.vv, self.vv)
            self.ig = 1.0 / self.g
        elif k == "exponential":
            self.u = self.sg[:, 0]
            self.v = self.sg[:, 1]
            self.w = self.sg[:, 2]
            self.lw = np.log(self.w) - np.log(self.v)
            self.tl = self.v * self.lw - self.u
            self.itl = 1.0 / self.tl
        else:
            self.S = self.mat(self.sg)
            self.Si = np.linalg.inv(self.S)
        return self.sg

    def mat(self, sv):
        S = np.zeros((sv.shape[0], self.dim, self.dim))
        S[:, self.ri, self.ci] = sv[:, self.pi] * self.fi
        return S

    def svec(self, S):
        return S[:, self.gr, self.gc] * self.gf

    def interior(self):
        v = np.zeros((self.nblk, self.width))
        k = self.kind
        if k == "nonnegative":
            v.fill(1.0)
        elif k == "soc":
            v[:, 0] = 1.0
        elif k == "exponential":
            v[:] = np.array([-1.0, 1.0, 1.0])
        else:
            pos = 0
            for c in range(self.dim):
                v[:, pos + c] = 1.0
                pos += c + 1
        return v

    def grad(self, out):
        k = self.kind
        if k == "nonnegative":
            out[:, 0] = -self.c1
        elif k == "soc":
            out[:, 0] = -2.0 * self.tt * self.ig
            out[:, 1:] = (2.0 * self.ig)[:, None] * self.vv
        elif k == "exponential":
            out[:, 0] = self.itl
            out[:, 1] = -1.0 / self.v - (self.lw - 1.0) * self.itl
            out[:, 2] = -1.0 / self.w - (self.v / self.w) * self.itl
        else:
            out[:] = self.svec(-self.Si)

    def hess_apply(self, vg, out):
        k = self.kind
        if k == "nonnegative":
            np.multiply(self.c2[:, None], vg, out=out)
        elif k == "soc":
            w0 = vg[:, 0]
            wv = vg[:, 1:]
            ud = self.tt * w0 - np.einsum("ij,ij->i", self.vv, wv)
            g2 = self.g * self.g
            out[:, 0] = -2.0 * w0 * self.ig + 4.0 * self.tt * ud / g2
            out[:, 1:] = (2.0 * self.ig)[:, None] * wv - (4.0 / g2)[:, None] * (self.vv * ud[:, None])
        elif k == "exponential":
            out[:] = np.einsum("bij,bj->bi", self.hess_dense(), vg)
        else:
            out[:] = self.svec(self.Si @ self.mat(vg) @ self.Si)

    def hess_dense(self):
        k = self.kind
        w = self.width
        H = np.zeros((self.nblk, w, w))
        if k == "soc":
            J = np.ones(w)
            J[1:] = -1.0
            ii = np.arange(w)
            H[:, ii, ii] = (-2.0 * self.ig)[:, None] * J[None, :]
            u = np.empty((self.nblk, w))
            u[:, 0] = self.tt
            u[:, 1:] = -self.vv
            H += (4.0 / (self.g * self.g))[:, None, None] * u[:, :, None] * u[:, None, :]
        elif k == "exponential":
            v, wv, tl, lw = self.v, self.w, self.tl, self.lw
            gv = lw - 1.0
            gw = v / wv
            H[:, 0, 0] = 1.0 / (v * v)
            H[:, 1, 1] = 1.0 / (v * v) + 1.0 / (v * tl)
            H[:, 2, 2] = 1.0 / (wv * wv) + v / (tl * wv * wv)
            H[:, 1, 2] = -1.0 / (wv * tl)
            H[:, 2, 1] = H[:, 1, 2]
            gu = np.full(self.nblk, -1.0)
            gt = np.stack([gu, gv, gw], axis=1)
            H += (gt[:, :, None] * gt[:, None, :]) * (1.0 / (tl * tl))[:, None, None]
        return H

    def schur(self, M, mu, A):
        k = self.kind
        if k == "nonnegative":
            self.c2_full[self.start:self.stop] = mu * self.c2
            schur_diag(A[0], A[1], A[2], self.c2_full, self.start, self.stop, M, self.n)
        elif k == "psd":
            for bi in range(self.nblk):
                Zm = self.Si[bi] @ self.Ym[bi] @ self.Si[bi]
                Gb = mu * (self.Y[bi] @ self.svec(Zm).T)
                cols = self.cols[bi]
                M[np.ix_(cols, cols)] += Gb + Gb.T - np.diag(np.diag(Gb))
        else:
            Hd = mu * self.hess_dense()
            for bi in range(self.nblk):
                schur_gen(A[0], A[1], A[2], self.start + bi * self.width, self.width,
                          np.ascontiguousarray(Hd[bi]), M, self.n)

    def max_step(self, base, dg, dual=False):
        """largest alpha with base + alpha*dg inside the (open) cone.

        base/dg are (nblk,width) arrays; for dual=True they belong to the dual
        cone (only the exponential cone needs a coordinate transform).
        """
        k = self.kind
        if k == "exponential" and dual:
            base = np.stack([-base[:, 1], -base[:, 0], EE * base[:, 2]], axis=1)
            dg = np.stack([-dg[:, 1], -dg[:, 0], EE * dg[:, 2]], axis=1)
        if k == "nonnegative":
            sb = base[:, 0]
            dd = dg[:, 0]
            neg = dd < 0.0
            return float(np.min(-sb[neg] / dd[neg])) if np.any(neg) else np.inf
        if k == "soc":
            tt, vv = base[:, 0], base[:, 1:]
            dt, dv = dg[:, 0], dg[:, 1:]
            a1 = np.full(self.nblk, np.inf)
            neg = dt < 0.0
            if np.any(neg):
                a1[neg] = -tt[neg] / dt[neg]
            A = dt * dt - np.einsum("ij,ij->i", dv, dv)
            B = 2.0 * (tt * dt - np.einsum("ij,ij->i", vv, dv))
            C = tt * tt - np.einsum("ij,ij->i", vv, vv)
            a2 = np.full(self.nblk, np.inf)
            disc = B * B - 4.0 * A * C
            ok = disc > 0.0
            if np.any(ok):
                sq = np.sqrt(disc[ok])
                Ao, Bo, Co = A[ok], B[ok], C[ok]
                r1 = (-Bo - sq) / (2.0 * Ao)
                r2 = (-Bo + sq) / (2.0 * Ao)
                lo = np.minimum(r1, r2)
                hi = np.maximum(r1, r2)
                # A < 0: positive between the roots -> take the larger one.
                # A > 0: positive outside the roots -> take the smaller positive one.
                val = np.where(Ao < 0.0, hi, np.where(lo > 0.0, lo, np.inf))
                a2[ok] = val
            lin = (~ok) & (B < 0.0)
            if np.any(lin):
                a2[lin] = -C[lin] / B[lin]
            return float(min(np.min(a1), np.min(a2)))
        if k == "exponential":
            u, v, w = base[:, 0], base[:, 1], base[:, 2]
            du, dv, dw = dg[:, 0], dg[:, 1], dg[:, 2]
            a = np.full(self.nblk, 1.006)
            for arr, darr in ((v, dv), (w, dw)):
                neg = darr < 0.0
                if np.any(neg):
                    a[neg] = np.minimum(a[neg], -arr[neg] / darr[neg])
            lo = np.zeros(self.nblk)
            hi = np.minimum(a, 1.006)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                vv = v + mid * dv
                ww = w + mid * dw
                with np.errstate(divide="ignore", invalid="ignore"):
                    tv = vv * (np.log(np.maximum(ww, 1e-300)) - np.log(np.maximum(vv, 1e-300))) - (u + mid * du)
                good = (vv > 0) & (ww > 0) & np.isfinite(tv) & (tv > 0.0)
                lo = np.where(good, mid, lo)
                hi = np.where(good, hi, mid)
            return float(np.min(lo)) if self.nblk else np.inf
        # psd: base + alpha*dg must stay positive definite
        try:
            L = np.linalg.cholesky(self.mat(base))
        except np.linalg.LinAlgError:
            return 0.0
        Li = np.linalg.inv(L)
        Mm = Li @ self.mat(dg) @ np.swapaxes(Li, 1, 2)
        Mm = 0.5 * (Mm + np.swapaxes(Mm, 1, 2))
        ev = np.linalg.eigvalsh(Mm)[:, 0]
        neg = ev < 0.0
        return float(np.min(-1.0 / ev[neg])) if np.any(neg) else np.inf


def initial_point(model, delta_rel=1e-2):
    """s0 close to b but strictly inside K (cone projection + per block margin)."""
    bp = model.bp
    sC = np.zeros(model.mc)
    for G in model.groups:
        bg = bp[G.start:G.stop].reshape(G.nblk, G.width)
        sc = np.max(np.abs(bg), axis=1) if G.nblk else np.zeros(0)
        dd = np.maximum(delta_rel * sc, 1e-9)
        k = G.kind
        if k == "nonnegative":
            v = np.maximum(bg, 0.0) + dd[:, None]
        elif k == "soc":
            t = bg[:, 0]
            vv = bg[:, 1:]
            nv = np.sqrt(np.einsum("ij,ij->i", vv, vv))
            out = np.zeros_like(bg)
            ins = nv <= t
            out[ins] = bg[ins]
            mid = (~ins) & (nv > -t)
            if np.any(mid):
                c = 0.5 * (1.0 + t[mid] / nv[mid])
                out[mid, 0] = c * nv[mid]
                out[mid, 1:] = c[:, None] * vv[mid]
            out[:, 0] += dd
            v = out
        elif k == "exponential":
            vn = np.abs(bg[:, 1]) + dd
            un = np.minimum(bg[:, 0], 0.0)
            wn = np.maximum(bg[:, 2], vn * np.exp(np.clip(un / vn, -50.0, 50.0))) + dd
            v = np.stack([un, vn, wn], axis=1)
        else:
            S = G.mat(bg)
            ev, V = np.linalg.eigh(S)
            ev = np.maximum(ev, dd[:, None])
            v = G.svec((V * ev[:, None, :]) @ np.swapaxes(V, 1, 2))
        sC[G.a:G.b] = v.ravel()
    return sC


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Model:
    def __init__(self, raw, do_scale=True):
        n = int(raw["Ps"][0])
        m = int(raw["As"][0])
        self.n, self.m = n, m
        self.raw_cones = raw["cones"]
        # block layout in original row order
        blocks = []
        r = 0
        for c in raw["cones"]:
            t = c["type"]
            d = int(c["dim"])
            w = d * (d + 1) // 2 if t == "psd" else d
            blocks.append((t, d, w, r))
            r += w
        if r != m:
            raise ValueError("cones do not partition rows")
        self.blocks = blocks
        rank = {"zero": 0, "nonnegative": 1, "soc": 2, "exponential": 3, "psd": 4}
        order = sorted(range(len(blocks)), key=lambda i: (rank[blocks[i][0]], blocks[i][1], blocks[i][3]))
        parts = [np.arange(blocks[i][3], blocks[i][3] + blocks[i][2]) for i in order if blocks[i][0] == "zero"]
        nz = sum(blocks[i][2] for i in order if blocks[i][0] == "zero")
        gl = []
        cur = nz
        i = 0
        while i < len(order):
            t, d, w, r0 = blocks[order[i]]
            if t == "zero":
                i += 1
                continue
            j = i
            rows = []
            nb = 0
            while j < len(order) and blocks[order[j]][0] == t and blocks[order[j]][1] == d:
                _, _, ww, rr = blocks[order[j]]
                rows.append(np.arange(rr, rr + ww))
                nb += 1
                j += 1
            parts.append(np.concatenate(rows))
            gl.append((t, d, nb * w if t == "nonnegative" else nb, cur))
            cur += nb * w
            i = j
        perm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        self.perm = perm.astype(np.int64)
        self.nz = nz
        self.mc = m - nz
        # groups: for nonnegative, width 1 per row so nblk = number of rows
        self.groups = []
        for (t, d, nb, st) in gl:
            self.groups.append(Group(t, d, nb, st, nz, n))
        self.nu = sum(g.nu for g in self.groups)

        # ---------------- scaling (original row order) ----------------
        Ad, Ai, Ap = raw["Ad"], raw["Ai"].astype(np.int64), raw["Ap"]
        Pd, Pi, Pp = raw["Pd"], raw["Pi"].astype(np.int64), raw["Pp"]
        acol = np.repeat(np.arange(n, dtype=np.int64), np.diff(Ap)) if Ad.size else np.zeros(0, dtype=np.int64)
        pcol = np.repeat(np.arange(n, dtype=np.int64), np.diff(Pp)) if Pd.size else np.zeros(0, dtype=np.int64)
        q = raw["q"].astype(np.float64).copy()
        b = raw["b"].astype(np.float64).copy()
        Dc = np.ones(n)
        Dr = np.ones(m)
        Ad = Ad.astype(np.float64).copy()
        Pd = Pd.astype(np.float64).copy()
        if do_scale:
            for _ in range(3):
                if time.perf_counter() - T0 > 0.30:
                    break
                cn = np.zeros(n)
                if Ad.size:
                    cn += np.bincount(acol, weights=np.abs(Ad), minlength=n)
                if Pd.size:
                    cn += np.bincount(pcol, weights=np.abs(Pd), minlength=n)
                s1 = np.where(cn > 1e-30, 1.0 / np.sqrt(np.maximum(cn, 1e-30)), 1.0)
                s1[~np.isfinite(s1)] = 1.0
                s1 = np.clip(s1, 1e-8, 1e8)
                Dc *= s1
                if Ad.size:
                    Ad *= s1[acol]
                if Pd.size:
                    Pd *= s1[pcol] * s1[Pi]
                q *= s1
                rn = np.zeros(m)
                if Ad.size:
                    rn = np.bincount(Ai, weights=np.abs(Ad), minlength=m)
                s2 = np.where(rn > 1e-30, 1.0 / np.sqrt(np.maximum(rn, 1e-30)), 1.0)
                s2[~np.isfinite(s2)] = 1.0
                s2 = np.clip(s2, 1e-8, 1e8)
                for (t, d, w, r0) in blocks:
                    if w > 1:
                        s2[r0:r0 + w] = np.max(s2[r0:r0 + w])
                Dr *= s2
                if Ad.size:
                    Ad *= s2[Ai]
                b *= s2
        gam = 1.0 / max(1.0, float(np.max(np.abs(Pd))) if Pd.size else 0.0,
                        float(np.max(np.abs(q))) if q.size else 0.0)
        self.gamma = gam
        Pd = Pd * gam
        q = q * gam
        self.Dc, self.Dr = Dc, Dr

        # ---------------- permuted storage ----------------
        invp = np.empty(m, dtype=np.int64)
        invp[perm] = np.arange(m)
        newrows = invp[Ai]
        # CSC in permuted rows: only the row index array changes
        self.Acsc = (Ad, np.ascontiguousarray(newrows.astype(np.int32)),
                     np.ascontiguousarray(Ap.astype(np.int32)))
        if Ad.size:
            o = np.argsort(newrows, kind="stable")
            cptr = np.zeros(m + 1, dtype=np.int64)
            np.cumsum(np.bincount(newrows, minlength=m), out=cptr[1:])
            csr = (np.ascontiguousarray(Ad[o]),
                   np.ascontiguousarray(acol[o].astype(np.int32)),
                   np.ascontiguousarray(cptr.astype(np.int32)))
        else:
            csr = (np.zeros(0), np.zeros(0, dtype=np.int32),
                   np.zeros(m + 1, dtype=np.int32))
        self.Acsr = csr
        self.bp = np.ascontiguousarray(b[perm])
        self.q = np.ascontiguousarray(q)
        # dense P (scaled)
        P = np.zeros((n, n))
        if Pd.size:
            flat = Pi * n + pcol
            np.add.at(P.reshape(-1), flat, Pd)
        self.P = 0.5 * (P + P.T)
        self.Pi, self.pcol = Pi, pcol
        # dense A of the equality (zero cone) rows
        self.Az = np.zeros((0, n))
        if nz:
            d_, i_, p_ = row_slice(*self.Acsr, 0, nz)
            self.Az = dense_from_csr(d_, i_, p_, nz, n)
        self.Drp = Dr[perm]
        # psd dense column blocks
        for G in self.groups:
            if G.kind != "psd":
                continue
            G.cols, G.Y, G.Ym = [], [], []
            for bi in range(G.nblk):
                r0 = G.start + bi * G.width
                d_, i_, p_ = row_slice(*self.Acsr, r0, r0 + G.width)
                cols = np.unique(i_).astype(np.int64)
                Ab = np.zeros((G.width, cols.size))
                if cols.size:
                    cm = -np.ones(n, dtype=np.int64)
                    cm[cols] = np.arange(cols.size)
                    rl = np.repeat(np.arange(G.width, dtype=np.int64), np.diff(p_))
                    Ab[rl, cm[i_]] = d_
                G.cols.append(cols)
                Y = np.ascontiguousarray(Ab.T)
                G.Y.append(Y)
                G.Ym.append(G.mat(Y))
        for G in self.groups:
            if G.kind == "nonnegative":
                G.c2_full = np.zeros(m)
        self.Axbuf = np.zeros(m)
        self.t1 = np.zeros(n)
        self.sf = np.zeros(m)
        self.zf = np.zeros(m)
        self.tm = np.zeros(m)

    # ---------------- linear algebra ----------------
    def build_M(self, mu):
        n = self.n
        M = self.P.copy()
        di = np.diag_indices(n)
        M[di] += REG_P * (1.0 + np.abs(M[di]))
        for G in self.groups:
            G.schur(M, mu, self.Acsr)
        return M

    def apply_W(self, out, v, mu):
        for G in self.groups:
            G.hess_apply(v[G.a:G.b].reshape(G.nblk, G.width), out[G.a:G.b].reshape(G.nblk, G.width))
            out[G.a:G.b] *= mu

    def kkt_solve(self, M, F1, F2C, F2Z, tcC, mu, wbuf):
        n, nz = self.n, self.nz
        self.tm[:nz] = 0.0
        self.tm[nz:] = tcC
        self.t1[:] = 0.0
        csc_mvT(*self.Acsc, self.tm, self.t1, n)
        rhs1 = -F1 + self.t1
        if self.mc:
            self.apply_W(wbuf, F2C, mu)
            self.tm[:nz] = 0.0
            self.tm[nz:] = wbuf
            self.t1[:] = 0.0
            csc_mvT(*self.Acsc, self.tm, self.t1, n)
            rhs1 -= self.t1
        if nz:
            # quasi-definite augmented system, solved as one LU: avoids forming
            # M^{-1} explicitly, which amplifies the primal regularisation when
            # P = 0 and A_c has fewer rows than columns.
            K = np.zeros((n + nz, n + nz))
            K[:n, :n] = M
            K[:n, n:] = self.Az.T
            K[n:, :n] = self.Az
            dd = 1e-11 * (1.0 + float(np.max(np.abs(M), initial=0.0)))
            K[np.arange(n, n + nz), np.arange(n, n + nz)] = -dd
            rhs = np.empty(n + nz)
            rhs[:n] = rhs1
            rhs[n:] = -F2Z
            sol = np.linalg.solve(K, rhs)
            dx = sol[:n]
            dzz = sol[n:]
        else:
            dzz = np.zeros(0)
            dx = np.linalg.solve(M, rhs1)
        dAx = self.Axbuf
        dAx[:] = 0.0
        csr_mv(*self.Acsr, dx, dAx, self.m)
        dsC = -F2C - dAx[nz:]
        wds = wbuf
        self.apply_W(wds, dsC, mu)
        dzC = -tcC - wds
        return dx, dsC, dzz, dzC

    def residuals(self, x, sC, zz, zC):
        n, m, nz = self.n, self.m, self.nz
        Ax = self.Axbuf
        Ax[:] = 0.0
        csr_mv(*self.Acsr, x, Ax, m)
        sf = self.sf
        sf[:nz] = 0.0
        sf[nz:] = sC
        F2 = Ax + sf - self.bp
        zf = self.zf
        zf[:nz] = zz
        zf[nz:] = zC
        self.t1[:] = 0.0
        csc_mvT(*self.Acsc, zf, self.t1, n)
        F1 = self.P @ x + self.q + self.t1 if n else np.zeros(0)
        return F1, np.array(F2[:nz]), np.array(F2[nz:])

    def metrics(self, x, sC, zz, zC):
        n, m, nz = self.n, self.m, self.nz
        Ax = self.Axbuf
        Ax[:] = 0.0
        csr_mv(*self.Acsr, x, Ax, m)
        Drp = self.Drp
        gam = self.gamma
        Axs = Ax / Drp
        sf = self.sf
        sf[:nz] = 0.0
        sf[nz:] = sC
        sos = sf / Drp
        bos = self.bp / Drp
        rp = np.max(np.abs(Axs + sos - bos), initial=0.0)
        rpd = 1.0 + max(np.max(np.abs(Axs), initial=0.0), np.max(np.abs(sos), initial=0.0),
                        np.max(np.abs(bos), initial=0.0))
        zf = self.zf
        zf[:nz] = zz
        zf[nz:] = zC
        self.t1[:] = 0.0
        csc_mvT(*self.Acsc, zf, self.t1, n)
        sc = gam * self.Dc
        Pxx = self.P @ x if n else np.zeros(0)
        Pxo = Pxx / sc
        qo = self.q / sc
        Atzo = self.t1 / sc
        rd = np.max(np.abs(Pxo + qo + Atzo), initial=0.0)
        rdd = 1.0 + max(np.max(np.abs(Pxo), initial=0.0), np.max(np.abs(qo), initial=0.0),
                        np.max(np.abs(Atzo), initial=0.0))
        xPx = float(x @ Pxx) / gam
        qx = float(self.q @ x) / gam
        bz = float(self.bp @ zf) / gam
        p0 = 0.5 * xPx + qx
        d0 = -0.5 * xPx - bz
        gap = abs(p0 - d0)
        mg = max(abs(p0), abs(d0))
        return rp / rpd, rd / rdd, gap, (gap / mg if mg > 0.0 else 0.0)

    def to_orig(self, x, sC, zz, zC):
        m, nz = self.m, self.nz
        sf = np.empty(m)
        sf[:nz] = 0.0
        sf[nz:] = sC
        zf = np.empty(m)
        zf[:nz] = zz
        zf[nz:] = zC
        so = np.empty(m)
        so[self.perm] = sf / self.Drp
        zo = np.empty(m)
        zo[self.perm] = zf * self.Drp / self.gamma
        return self.Dc * x, so, zo


# --------------------------------------------------------------------------
# interior point driver
# --------------------------------------------------------------------------
def ipm(model, budget, maxit=200, tol=1e-10, verbose=False, tau=0.98, sig_min=1e-8, sig_max=0.5, res_floor=1.0, init_mode='e'):
    n, m, nz, mc = model.n, model.m, model.nz, model.mc
    groups = model.groups
    nu = model.nu
    best = {"score": np.inf, "iters": 0}
    if n == 0:
        return best, False
    x = np.zeros(n)
    zC = np.zeros(mc)
    sig_z = max(1.0, float(np.max(np.abs(model.q), initial=0.0)))
    sig_s = max(1.0, float(np.max(np.abs(model.bp), initial=0.0)))
    if init_mode == "e":
        sC = np.zeros(mc)
        for G in groups:
            sC[G.a:G.b] = G.interior().ravel()
        sC *= sig_s
    else:
        sC = initial_point(model, 1e-2 if init_mode == "proj" else 1e-3)
    mu0 = sig_s * sig_z
    if init_mode == "zcanon":
        ev = np.zeros(mc)
        for G in groups:
            ev[G.a:G.b] = G.interior().ravel()
        for G in groups:
            G.prepare(ev)
            og = np.empty((G.nblk, G.width))
            G.grad(og)
            zC[G.a:G.b] = (-sig_z * og).ravel()
    else:
        for G in groups:
            G.prepare(sC)
            og = np.empty((G.nblk, G.width))
            G.grad(og)
            zC[G.a:G.b] = (-mu0 * og).ravel()
    zz = np.zeros(nz)
    mu_floor = 1e-14 * max(1.0, sig_s * sig_z)
    wbuf = np.zeros(mc)
    wbuf2 = np.zeros(mc)

    def step_s(ds, base=None):
        base = sC if base is None else base
        a = np.inf
        for G in groups:
            a = min(a, G.max_step(base[G.a:G.b].reshape(G.nblk, G.width),
                                  ds[G.a:G.b].reshape(G.nblk, G.width), False))
        return a

    def step_z(dz, base=None):
        base = zC if base is None else base
        a = np.inf
        for G in groups:
            a = min(a, G.max_step(base[G.a:G.b].reshape(G.nblk, G.width),
                                  dz[G.a:G.b].reshape(G.nblk, G.width), True))
        return a

    def merit(alpha, ds, dz, mu_t, nF1, nF2, sc1, sc2, sc3):
        v1 = (1.0 - alpha) * nF1 / sc1
        v2 = (1.0 - alpha) * nF2 / sc2
        if alpha == 0.0:
            sn, zn = sC, zC
        else:
            sn = sC + alpha * ds
            zn = zC + alpha * dz
        f3 = 0.0
        for G in groups:
            G.prepare(sn)
            og = np.empty((G.nblk, G.width))
            G.grad(og)
            f3 = max(f3, float(np.max(np.abs(zn[G.a:G.b].reshape(G.nblk, G.width) + mu_t * og), initial=0.0)))
        return max(v1, v2, f3 / sc3)

    stalled = 0
    rel0 = [1.0]
    sc1_ref = sc2_ref = 1.0
    worse = 0
    prev_score = np.inf
    for it in range(maxit):
        if time.perf_counter() > budget:
            why = "budget"; break
        for G in groups:
            G.prepare(sC)
        F1, F2Z, F2C = model.residuals(x, sC, zz, zC)
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2Z)) and np.all(np.isfinite(F2C))):
            why = "nonfinite F"; break
        sz = float(np.dot(sC, zC)) if mc else 0.0
        nF1 = float(np.max(np.abs(F1), initial=0.0))
        nF2Z = float(np.max(np.abs(F2Z), initial=0.0))
        nF2C = float(np.max(np.abs(F2C), initial=0.0))
        nF2 = max(nF2Z, nF2C)
        # infeasible-start duality measure: include the residual product so that
        # mu cannot collapse while the linear residuals are still large
        resprod = nF1 * nF2
        # mu must stay consistent with the iterate (z ~ -mu grad phi(s)), otherwise the
        # Newton system is mis-scaled.  The *target* mu_t is floored by the residual
        # level so that centrality is not chased while infeasibility is still large.
        mu = (sz / nu if nu else 0.0)
        mu = max(mu, mu_floor)
        rp, rd, gap, gapr = model.metrics(x, sC, zz, zC)
        score = max(rp, rd, min(gap, gapr))
        if np.isfinite(score) and score < best["score"]:
            xo, so, zo = model.to_orig(x, sC, zz, zC)
            best = {"score": score, "x": xo, "s": so, "z": zo, "rp": rp, "rd": rd,
                    "gap": gap, "gapr": gapr, "mu": mu, "iters": it,
                    "sC": sC.copy(), "zC": zC.copy(), "zz": zz.copy(), "x_t": x.copy()}
        if score <= tol:
            best["iters"] = it
            return best, True
        if False:
            why = "mu tiny score=%.3e" % score; break
        sc1 = max(1.0, nF1)
        sc2 = max(1.0, nF2)
        if it == 0:
            sc1_ref, sc2_ref = sc1, sc2
        sc3 = max(1.0, float(np.max(np.abs(zC), initial=0.0)), sig_z)
        M = model.build_M(mu)
        try:
            dx_a, ds_a, dzz_a, dz_a = model.kkt_solve(M, F1, F2C, F2Z, zC.copy(), mu, wbuf)
        except np.linalg.LinAlgError as e:
            why = "affine LinAlg %s" % e; break
        if not (np.all(np.isfinite(dx_a)) and np.all(np.isfinite(ds_a)) and np.all(np.isfinite(dz_a))):
            why = "nonfinite aff"; break
        ap = tau * min(1.0, step_s(ds_a))
        ad = tau * min(1.0, step_z(dz_a))
        if nu:
            mu_aff = max(float(np.dot(sC + ap * ds_a, zC + ad * dz_a)), 0.0) / nu
        else:
            mu_aff = 0.0
        sig = (mu_aff / mu) ** 2 if mu > 0 else 0.0
        sig = min(max(sig, sig_min), sig_max)
        mu_res = res_floor * (1.0 - ap) * (1.0 - ad) * resprod / nu if nu else 0.0
        mu_t = max(sig * mu, mu_res, mu_floor)
        # Mehrotra combined target.  With W = mu*H(s) fixed, the Newton equation
        # for the perturbed system at mu_t reads
        #     tc = z + mu_t*grad phi(s_a) - W*(s_a - s),   s_a = s + alpha_p ds_a
        # which is the exact linearisation plus the second order remainder.
        s_aff = sC + ap * ds_a
        tcC = np.empty(mc)
        wds = np.zeros(mc)
        model.apply_W(wds, ap * ds_a, mu)
        for G in groups:
            G.prepare(s_aff)
            og = np.empty((G.nblk, G.width))
            G.grad(og)
            tcC[G.a:G.b] = (zC[G.a:G.b].reshape(G.nblk, G.width) + mu_t * og).ravel() - wds[G.a:G.b]
        for G in groups:
            G.prepare(sC)
        try:
            dx, ds, dzz, dz = model.kkt_solve(M, F1, F2C, F2Z, tcC, mu, wbuf2)
        except np.linalg.LinAlgError as e:
            why = "comb LinAlg %s" % e; break
        if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(ds)) and np.all(np.isfinite(dz))):
            why = "nonfinite comb"; break
        stp_s = step_s(ds)
        stp_z = step_z(dz)
        amax = tau * min(stp_s, stp_z)
        amax = min(1.0, amax) if np.isfinite(amax) else 1.0
        alpha = amax
        s_try = sC + alpha * ds
        z_try = zC + alpha * dz
        while not (np.all(np.isfinite(s_try)) and np.all(np.isfinite(z_try))
                   and np.all(np.isfinite(alpha * dx))):
            alpha *= 0.5
            if alpha < 1e-18:
                break
            s_try = sC + alpha * ds
            z_try = zC + alpha * dz
        if alpha < 1e-18:
            why = "no finite step"; break
        x = x + alpha * dx
        sC = s_try
        zC = z_try
        if nz:
            zz = zz + alpha * dzz
        # divergence guard
        if score < prev_score:
            worse = 0
        else:
            worse += 1
        prev_score = score
        if worse >= 6:
            why = "no progress score=%.3e" % score
            break
        if verbose:
            print(f"it {it:3d} mu {mu:.3e} score {score:.3e} rp {rp:.2e} rd {rd:.2e} gap {gap:.2e} gapr {gapr:.2e} alpha {alpha:.3f} amax {amax:.3e} st_s {stp_s:.3e} st_z {stp_z:.3e} sz {sz:.3e}", file=sys.stderr)
    if verbose:
        print("break reason:", locals().get('why','maxit'), file=sys.stderr)
    return best, False
