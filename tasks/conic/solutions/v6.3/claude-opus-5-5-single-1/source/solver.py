#!/usr/bin/env python3
"""Homogeneous self-dual embedding interior-point solver for conic QPs.
min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K  (zero, nonneg, soc, exp, psd)."""
import os, sys, json, time, math
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

T0 = time.time()
EPS = 1e-8
DEBUG = bool(os.environ.get('SOLVER_DEBUG'))
SQ2 = math.sqrt(2.0)
SOC_DENSE_MAX = 48
REUSE_ORDER = not os.environ.get('SOLVER_NO_REUSE')


def _load_superlu():
    try:
        import importlib.util, glob
        spec = importlib.util.find_spec('scipy')
        d = os.path.join(spec.submodule_search_locations[0], 'sparse', 'linalg', '_dsolve')
        f = glob.glob(os.path.join(d, '_superlu*.so'))[0]
        s2 = importlib.util.spec_from_file_location('scipy.sparse.linalg._dsolve._superlu', f)
        mod = importlib.util.module_from_spec(s2)
        s2.loader.exec_module(mod)
        return mod
    except Exception:
        from scipy.sparse.linalg._dsolve import _superlu
        return _superlu


SLU = _load_superlu()


def _load_flapack():
    try:
        import importlib.util, glob
        spec = importlib.util.find_spec('scipy')
        d = os.path.join(spec.submodule_search_locations[0], 'linalg')
        f = glob.glob(os.path.join(d, '_flapack*.so'))[0]
        s2 = importlib.util.spec_from_file_location('scipy.linalg._flapack', f)
        mod = importlib.util.module_from_spec(s2)
        s2.loader.exec_module(mod)
        return mod
    except Exception:
        from scipy.linalg import _flapack
        return _flapack


LAPACK = _load_flapack()


class Mat:
    """Minimal COO/CSC-like sparse matrix with duplicates summed."""
    def __init__(self, m, n, rows, cols, vals):
        self.m, self.n = int(m), int(n)
        if len(vals):
            key = cols.astype(np.int64) * max(self.m, 1) + rows.astype(np.int64)
            if np.any(np.diff(key) <= 0):
                u, inv = np.unique(key, return_inverse=True)
                vals = np.bincount(inv, weights=vals, minlength=len(u))
                rows = (u % max(self.m, 1)); cols = u // max(self.m, 1)
        self.r = np.asarray(rows, dtype=np.int64)
        self.c = np.asarray(cols, dtype=np.int64)
        self.v = np.asarray(vals, dtype=np.float64)

    def mv(self, x):
        if self.v.size == 0:
            return np.zeros(self.m)
        return np.bincount(self.r, weights=self.v * x[self.c], minlength=self.m)

    def tmv(self, y):
        if self.v.size == 0:
            return np.zeros(self.n)
        return np.bincount(self.c, weights=self.v * y[self.r], minlength=self.n)

    def absmax(self):
        return float(np.max(np.abs(self.v))) if self.v.size else 0.0


def inf_norm(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


# ---------------------------------------------------------------- PSD helpers
_svec_cache = {}


def svec_idx(d):
    if d not in _svec_cache:
        I, J = [], []
        for j in range(d):
            for i in range(j + 1):
                I.append(i); J.append(j)
        I = np.array(I); J = np.array(J)
        al = np.where(I == J, 1.0, SQ2)
        _svec_cache[d] = (I, J, al)
    return _svec_cache[d]


def smat(v, d):
    I, J, al = svec_idx(d)
    X = np.zeros((d, d))
    X[I, J] = v / al
    X[J, I] = v / al
    return X


def svec(X, d):
    I, J, al = svec_idx(d)
    return X[I, J] * al


def smat_b(V, d):
    I, J, al = svec_idx(d)
    X = np.zeros((V.shape[0], d, d))
    W = V / al[None, :]
    X[:, I, J] = W
    X[:, J, I] = W
    return X


def svec_b(X, d):
    I, J, al = svec_idx(d)
    return X[:, I, J] * al[None, :]


# ---------------------------------------------------------------- exp cone helpers
EXP_C = np.array([-1.051383945322714, 0.556409619469370, 1.258967884768947])


def exp_primal_ok(S):
    u, v, w = S[:, 0], S[:, 1], S[:, 2]
    ok = (v > 0) & (w > 0)
    with np.errstate(all='ignore'):
        r = v * np.log(np.where(ok, w / np.where(v > 0, v, 1), 1.0)) - u
    return ok & (r > 0) & np.isfinite(r)


def exp_dual_ok(Z):
    u, v, w = Z[:, 0], Z[:, 1], Z[:, 2]
    ok = (u < 0) & (w > 0)
    with np.errstate(all='ignore'):
        l = np.log(np.where(ok, -w / np.where(u < 0, u, -1), 1.0))
        r = v - u - u * l
    return ok & (r > 0) & np.isfinite(r)


def exp_dual_grad_hess(Z):
    u, v, w = Z[:, 0], Z[:, 1], Z[:, 2]
    l = np.log(-w / u)
    r = v - u - u * l
    gr = np.stack([-l, np.ones_like(u), -u / w], axis=1)
    g = -gr / r[:, None]
    g[:, 0] -= 1.0 / u
    g[:, 2] -= 1.0 / w
    H = gr[:, :, None] * gr[:, None, :] / (r * r)[:, None, None]
    H[:, 0, 0] -= 1.0 / (u * r)
    H[:, 0, 2] += 1.0 / (w * r)
    H[:, 2, 0] += 1.0 / (w * r)
    H[:, 2, 2] -= u / (w * w * r)
    H[:, 0, 0] += 1.0 / (u * u)
    H[:, 2, 2] += 1.0 / (w * w)
    return g, H


# ---------------------------------------------------------------- cone structure
class Cones:
    def __init__(self, cones, m):
        zero, nn, soc, exps, psd = [], [], [], [], []
        pos = 0
        for c in cones:
            t = c['type']; d = int(c['dim'])
            if t == 'zero':
                zero.append(np.arange(pos, pos + d)); pos += d
            elif t == 'nonnegative':
                nn.append(np.arange(pos, pos + d)); pos += d
            elif t == 'soc':
                soc.append((pos, d)); pos += d
            elif t == 'exponential':
                exps.append(pos); pos += 3
            elif t == 'psd':
                psd.append((pos, d)); pos += d * (d + 1) // 2
            else:
                raise ValueError('unknown cone ' + t)
        assert pos == m
        self.m = m
        self.zero = np.concatenate(zero) if zero else np.zeros(0, np.int64)
        self.nn = np.concatenate(nn) if nn else np.zeros(0, np.int64)
        self.nsoc = len(soc)
        if soc:
            self.soc_idx = np.concatenate([np.arange(p, p + d) for p, d in soc])
            dims = np.array([d for p, d in soc])
            self.soc_dims = dims
            self.soc_off = np.concatenate([[0], np.cumsum(dims)[:-1]])
            self.soc_seg = np.repeat(np.arange(len(soc)), dims)
            self.soc_tail = np.ones(self.soc_idx.size, bool); self.soc_tail[self.soc_off] = False
            self.soc_J = np.where(self.soc_tail, -1.0, 1.0)
            # dense block pattern (small cones), diagonal + expansion (large cones)
            pi, qi = [], []
            self.soc_large = dims > SOC_DENSE_MAX
            self.large_list = []
            for k, (p, d) in enumerate(soc):
                o = self.soc_off[k]
                a = np.arange(o, o + d)
                if self.soc_large[k]:
                    pi.append(a); qi.append(a)
                    self.large_list.append(k)
                else:
                    pi.append(np.repeat(a, d)); qi.append(np.tile(a, d))
            self.soc_pi = np.concatenate(pi); self.soc_qi = np.concatenate(qi)
            self.soc_pseg = self.soc_seg[self.soc_pi]
            self.soc_pdiag = np.where(self.soc_pi == self.soc_qi, self.soc_J[self.soc_pi], 0.0)
            self.soc_plarge = self.soc_large[self.soc_pseg]
            # expansion pattern: rows (z-index) and extra variable index j
            er, ec = [], []
            for j, k in enumerate(self.large_list):
                o = self.soc_off[k]; d = dims[k]
                er.append(np.arange(o, o + d)); ec.append(np.full(d, j))
            if er:
                self.ext_pos = np.concatenate(er)          # positions in soc arrays
                self.ext_j = np.concatenate(ec)
                self.ext_heads = self.soc_off[self.large_list]
            self.nlarge = len(self.large_list)
        else:
            self.nlarge = 0
        self.exp = np.array(exps, dtype=np.int64)
        self.nexp = len(exps)
        if exps:
            self.exp_idx = (self.exp[:, None] + np.arange(3)[None, :])
            base = self.exp_idx
            self.exp_r = np.repeat(base, 3, axis=1).ravel()
            self.exp_c = np.tile(base, (1, 3)).ravel()
        self.psd = psd
        groups = {}
        for i, (p, d) in enumerate(psd):
            groups.setdefault(d, []).append(i)
        self.psd_groups = []
        for d, ids in groups.items():
            t = d * (d + 1) // 2
            idx = np.array([psd[i][0] for i in ids])[:, None] + np.arange(t)[None, :]
            self.psd_groups.append((d, np.array(ids), idx))
        self.nu = len(self.nn) + self.nsoc + 3 * self.nexp + sum(d for p, d in psd)
        # H pattern (rows/cols within z block)
        hr = [self.zero, self.nn]; hc = [self.zero, self.nn]
        if self.nsoc:
            hr.append(self.soc_idx[self.soc_pi]); hc.append(self.soc_idx[self.soc_qi])
        if self.nexp:
            hr.append(self.exp_r); hc.append(self.exp_c)
        for p, d in psd:
            t = d * (d + 1) // 2
            a = np.arange(p, p + t)
            hr.append(np.repeat(a, t)); hc.append(np.tile(a, t))
        self.hr = np.concatenate(hr).astype(np.int64)
        self.hc = np.concatenate(hc).astype(np.int64)

    # segment sum for soc
    def ssum(self, v):
        return np.add.reduceat(v, self.soc_off)

    def unit(self, v=None):
        e = np.zeros(self.m)
        e[self.nn] = 1.0
        if self.nsoc:
            e[self.soc_idx[self.soc_off]] = 1.0
        if self.nexp:
            e[self.exp_idx] = EXP_C
        for p, d in self.psd:
            e[p:p + d * (d + 1) // 2] = svec(np.eye(d), d)
        return e

    def shift_into(self, s):
        """s + (1+a) e for symmetric cones if s not interior (CVXOPT style)."""
        s = s.copy()
        a = -np.inf
        if self.nn.size:
            a = max(a, -np.min(s[self.nn]))
        if self.nsoc:
            x = s[self.soc_idx]
            x0 = x[self.soc_off]
            t = np.sqrt(self.ssum(x * x * self.soc_tail))
            a = max(a, float(np.max(t - x0)))
        for p, d in self.psd:
            t = d * (d + 1) // 2
            a = max(a, -np.linalg.eigvalsh(smat(s[p:p + t], d))[0])
        if a == -np.inf:
            a = 0.0
        if a >= -1e-8:
            e = self.unit()
            if self.nexp:
                e[self.exp_idx] = 0.0
            s += (1.0 + a) * e
        if self.nexp:
            s[self.exp_idx] = EXP_C
        s[self.zero] = 0.0
        return s


class Scaling:
    """NT scaling for symmetric cones and dual scaling for exponential cones."""
    def __init__(self, K, s, z, mu, need_h=True):
        self.K = K
        self.s, self.z, self.mu = s, z, mu
        m = K.m
        hv = []
        hv.append(np.zeros(K.zero.size))
        nn = K.nn
        self.nn_w = np.sqrt(s[nn] / z[nn])
        self.nn_l = np.sqrt(s[nn] * z[nn])
        hv.append(s[nn] / z[nn])
        if K.nsoc:
            ss = s[K.soc_idx]; zz = z[K.soc_idx]; off = K.soc_off; seg = K.soc_seg
            s0 = ss[off]; z0 = zz[off]
            sn = np.sqrt(K.ssum(ss * ss * K.soc_tail)); zn = np.sqrt(K.ssum(zz * zz * K.soc_tail))
            sJs = (s0 - sn) * (s0 + sn); zJz = (z0 - zn) * (z0 + zn)
            sJs = np.maximum(sJs, 1e-300); zJz = np.maximum(zJz, 1e-300)
            sb = ss / np.sqrt(sJs)[seg]; zb = zz / np.sqrt(zJz)[seg]
            g = np.sqrt(np.maximum((1.0 + K.ssum(sb * zb)) / 2.0, 1e-300))
            w = (sb + K.soc_J * zb) / (2.0 * g)[seg]
            self.soc_w = w
            self.soc_eta = (sJs / zJz) ** 0.25
            eta2 = self.soc_eta ** 2
            hv.append(np.where(K.soc_plarge, eta2[K.soc_pseg],
                               eta2[K.soc_pseg] * (2.0 * w[K.soc_pi] * w[K.soc_qi] - K.soc_pdiag)))
            if K.nlarge:
                lg = np.array(K.large_list)
                self.ext_w = -SQ2 * self.soc_eta[K.soc_seg[K.ext_pos]] * w[K.ext_pos]
                self.ext_h = SQ2 * self.soc_eta[lg]
            self.soc_l = self.soc_W(zz)
        if K.nexp:
            Z = z[K.exp_idx]
            g, H = exp_dual_grad_hess(Z)
            self.exp_g = g
            self.exp_H = mu * H
            hv.append(self.exp_H.ravel())
        self.psd = []
        self.psd_H = []
        for p, d in K.psd:
            t = d * (d + 1) // 2
            S = smat(s[p:p + t], d); Z = smat(z[p:p + t], d)
            Ls = np.linalg.cholesky(S); Lz = np.linalg.cholesky(Z)
            U, lam, Vt = np.linalg.svd(Lz.T @ Ls)
            R = Ls @ Vt.T / np.sqrt(lam)[None, :]
            Ri = (U.T @ Lz.T) / np.sqrt(lam)[:, None]
            G = R @ R.T
            I, J, al = svec_idx(d)
            if need_h:
                Hb = (np.outer(al, al) / 2.0) * (G[np.ix_(I, I)] * G[np.ix_(J, J)] + G[np.ix_(I, J)] * G[np.ix_(J, I)])
                hv.append(Hb.ravel())
                self.psd_H.append(Hb)
            self.psd.append((R, Ri, lam))
        self.hv = np.concatenate(hv) if need_h else None
        self.gG = []; self.gGi = []
        for d, ids, idx in K.psd_groups:
            self.gG.append(np.stack([self.psd[i][0] @ self.psd[i][0].T for i in ids]))
            self.gGi.append(np.stack([self.psd[i][1].T @ self.psd[i][1] for i in ids]))

    def ensure_hv(self):
        if self.hv is None:
            o = Scaling(self.K, self.s, self.z, self.mu, need_h=True)
            self.hv = o.hv; self.psd_H = o.psd_H
        return self.hv

    def Hinv_mul(self, v):
        """H^{-1} v on all non-zero-cone rows (zero rows -> 0)."""
        K = self.K
        out = np.zeros_like(v)
        out[K.nn] = v[K.nn] / (self.nn_w ** 2)
        if K.nsoc:
            vv = v[K.soc_idx]
            out[K.soc_idx] = self.soc_W(self.soc_W(vv, inv=True), inv=True)
        if K.nexp:
            V = v[K.exp_idx]
            out[K.exp_idx] = np.linalg.solve(self.exp_H, V[:, :, None])[:, :, 0]
        for (d, ids, idx), Gi in zip(K.psd_groups, self.gGi):
            out[idx] = svec_b(Gi @ smat_b(v[idx], d) @ Gi, d)
        return out

    def Winv_mat(self, V):
        """W^{-T} V (so that V' H^{-1} V = (W^{-T}V)'(W^{-T}V)); zero rows -> 0."""
        K = self.K
        out = np.zeros_like(V)
        out[K.nn] = V[K.nn] / self.nn_w[:, None]
        if K.nsoc:
            for j in range(V.shape[1]):
                vv = V[K.soc_idx, j]
                if np.any(vv):
                    out[K.soc_idx, j] = self.soc_W(vv, inv=True)
        if K.nexp:
            L = np.linalg.cholesky(self.exp_H)
            X = V[K.exp_idx]
            out[K.exp_idx] = np.linalg.solve(L, X)
        for (p, d), (R, Ri, lam) in zip(K.psd, self.psd):
            t = d * (d + 1) // 2
            I, J, al = svec_idx(d)
            blk = V[p:p + t]
            nzc = np.nonzero(np.any(blk != 0, axis=0))[0]
            for c0 in range(0, len(nzc), 256):
                cc = nzc[c0:c0 + 256]
                X = np.zeros((len(cc), d, d))
                bt = blk[:, cc].T / al[None, :]
                X[:, I, J] = bt
                X[:, J, I] = bt
                Y = Ri @ X @ Ri.T
                out[p:p + t, cc] = (Y[:, I, J] * al[None, :]).T
        return out

    def Hinv_mat(self, V):
        """H^{-1} V for dense V (m x k)."""
        K = self.K
        out = np.zeros_like(V)
        out[K.nn] = V[K.nn] / (self.nn_w ** 2)[:, None]
        if K.nsoc:
            for j in range(V.shape[1]):
                vv = V[K.soc_idx, j]
                if np.any(vv):
                    out[K.soc_idx, j] = self.soc_W(self.soc_W(vv, inv=True), inv=True)
        if K.nexp:
            Hi = np.linalg.inv(self.exp_H)
            X = V[K.exp_idx]  # k x 3 x ncol
            out[K.exp_idx] = np.einsum('kij,kjc->kic', Hi, X)
        for (p, d), (R, Ri, lam) in zip(K.psd, self.psd):
            t = d * (d + 1) // 2
            Gi = Ri.T @ Ri
            I, J, al = svec_idx(d)
            blk = V[p:p + t]
            nzc = np.nonzero(np.any(blk != 0, axis=0))[0]
            for c0 in range(0, len(nzc), 256):
                cc = nzc[c0:c0 + 256]
                X = np.zeros((len(cc), d, d))
                bt = blk[:, cc].T / al[None, :]
                X[:, I, J] = bt
                X[:, J, I] = bt
                Y = Gi @ X @ Gi
                out[p:p + t, cc] = (Y[:, I, J] * al[None, :]).T
        return out

    def soc_W(self, v, inv=False):
        K = self.K; off = K.soc_off; seg = K.soc_seg; w = self.soc_w
        w0 = w[off]; v0 = v[off]
        w1v1 = K.ssum(w * v * K.soc_tail)
        out = np.empty_like(v)
        if not inv:
            c = v0 + w1v1 / (1.0 + w0)
            out[:] = (v + c[seg] * w) * self.soc_eta[seg]
            out[off] = self.soc_eta * (w0 * v0 + w1v1)
        else:
            c = -v0 + w1v1 / (1.0 + w0)
            out[:] = (v + c[seg] * w) / self.soc_eta[seg]
            out[off] = (w0 * v0 - w1v1) / self.soc_eta
        return out

    def Hmul(self, v):
        """H v."""
        K = self.K
        out = np.zeros_like(v)
        out[K.nn] = (self.nn_w ** 2) * v[K.nn]
        if K.nsoc:
            vv = v[K.soc_idx]
            out[K.soc_idx] = self.soc_W(self.soc_W(vv))
        if K.nexp:
            V = v[K.exp_idx]
            out[K.exp_idx] = np.einsum('kij,kj->ki', self.exp_H, V)
        for (d, ids, idx), G in zip(K.psd_groups, self.gG):
            out[idx] = svec_b(G @ smat_b(v[idx], d) @ G, d)
        return out

    def ds_vec(self, s, z, mu, sigma, dsa=None, dza=None):
        """Right-hand side ds for the complementarity block."""
        K = self.K
        ds = np.zeros(K.m)
        sm = sigma * mu
        # nonneg
        nn = K.nn
        l = self.nn_l
        v = l * l - sm
        if dsa is not None:
            v = v + (dsa[nn] / self.nn_w) * (dza[nn] * self.nn_w)
        ds[nn] = self.nn_w * v / l
        if K.nsoc:
            off = K.soc_off; seg = K.soc_seg
            l = self.soc_l
            # l o l
            v = np.empty_like(l)
            v[:] = 2.0 * l[off][seg] * l
            v[off] = K.ssum(l * l)
            v[off] -= sm
            if dsa is not None:
                a = self.soc_W(dsa[K.soc_idx], inv=True)
                bb = self.soc_W(dza[K.soc_idx])
                c = a[off][seg] * bb + bb[off][seg] * a
                c[off] = K.ssum(a * bb)
                v = v + c
            # l \ v
            l0 = l[off]; v0 = v[off]
            l1v1 = K.ssum(l * v * K.soc_tail)
            ln2 = K.ssum(l * l * K.soc_tail)
            x0 = (l0 * v0 - l1v1) / ((l0 - np.sqrt(ln2)) * (l0 + np.sqrt(ln2)))
            x = (v - x0[seg] * l) / l0[seg]
            x[off] = x0
            ds[K.soc_idx] = self.soc_W(x)
        if K.nexp:
            ds[K.exp_idx] = s[K.exp_idx] + sm * self.exp_g
        for (p, d), (R, Ri, lam) in zip(K.psd, self.psd):
            t = d * (d + 1) // 2
            V = np.diag(lam * lam - sm)
            if dsa is not None:
                A1 = Ri @ smat(dsa[p:p + t], d) @ Ri.T
                B1 = R.T @ smat(dza[p:p + t], d) @ R
                V = V + 0.5 * (A1 @ B1 + B1 @ A1)
            X = 2.0 * V / (lam[:, None] + lam[None, :])
            ds[p:p + t] = svec(R @ X @ R.T, d)
        return ds


def step_sym(K, s, ds, amax):
    """max step keeping s + a ds in symmetric cones (no exp)."""
    a = amax
    nn = K.nn
    if nn.size:
        d = ds[nn]
        neg = d < 0
        if np.any(neg):
            a = min(a, float(np.min(-s[nn][neg] / d[neg])))
    if K.nsoc:
        x = s[K.soc_idx]; y = ds[K.soc_idx]; off = K.soc_off
        x0 = x[off]; y0 = y[off]
        xn2 = K.ssum(x * x * K.soc_tail); yn2 = K.ssum(y * y * K.soc_tail)
        xy = K.ssum(x * y * K.soc_tail)
        A_ = y0 * y0 - yn2
        B_ = 2.0 * (x0 * y0 - xy)
        C_ = np.maximum(x0 * x0 - xn2, 0.0)
        D_ = B_ * B_ - 4.0 * A_ * C_
        res = np.full(A_.shape, np.inf)
        with np.errstate(all='ignore'):
            sq = np.sqrt(np.maximum(D_, 0.0))
            t = -0.5 * (B_ + np.where(B_ >= 0, sq, -sq))
            r1 = np.where(t != 0, C_ / t, np.inf)
            r2 = np.where(A_ != 0, t / A_, np.inf)
            r1 = np.where(r1 > 0, r1, np.inf); r2 = np.where(r2 > 0, r2, np.inf)
            cand = np.minimum(r1, r2)
            skip = ((A_ > 0) & (B_ > 0)) | (D_ < 0)
            cand = np.where(skip, np.inf, cand)
            # linear case
            lin = (A_ == 0)
            cand = np.where(lin, np.where(B_ < 0, -C_ / np.where(B_ < 0, B_, -1), np.inf), cand)
            cand = np.where(C_ == 0, np.where(A_ >= 0, np.inf, 0.0), cand)
        res = cand
        # also must keep x0 + a y0 >= 0
        with np.errstate(all='ignore'):
            r0 = np.where(y0 < 0, -x0 / y0, np.inf)
        res = np.minimum(res, r0)
        if res.size:
            a = min(a, float(np.min(res)))
    for d, ids, idx in K.psd_groups:
        S = smat_b(s[idx], d); D = smat_b(ds[idx], d)
        try:
            L = np.linalg.cholesky(S)
            Li = np.linalg.inv(L)
            M = Li @ D @ np.swapaxes(Li, 1, 2)
            lm = float(np.min(np.linalg.eigvalsh((M + np.swapaxes(M, 1, 2)) / 2)[:, 0]))
            if lm < 0:
                a = min(a, -1.0 / lm)
        except np.linalg.LinAlgError:
            a = 0.0
    return max(a, 0.0)


# ---------------------------------------------------------------- KKT
class KKT:
    def __init__(self, n, m, P, A, K):
        N0 = n + m
        N = N0 + 2 * K.nlarge
        self.n, self.m, self.N, self.N0 = n, m, N, N0
        rows = [P.r, A.r + n, A.c, K.hr + n, np.arange(N)]
        cols = [P.c, A.c, A.r + n, K.hc + n, np.arange(N)]
        if K.nlarge:
            zr = n + K.soc_idx[K.ext_pos]
            y1 = N0 + 2 * K.ext_j
            zh = n + K.soc_idx[K.ext_heads]
            y2 = N0 + 2 * np.arange(K.nlarge) + 1
            rows += [zr, y1, zh, y2]
            cols += [y1, zr, y2, zh]
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        self.rows_all = rows; self.cols_all = cols
        self._structure(rows, cols, N)
        self.perm = None
        self.base = np.concatenate([P.v, A.v, A.v])
        self.P, self.A, self.K = P, A, K
        self.lu = None
        self.cp = 'MMD_AT_PLUS_A'
        self.tuned = False
        self.maxcol = int(np.max(np.diff(self.indptr))) if N else 0

    def _structure(self, rows, cols, N):
        key = cols * N + rows
        u, inv = np.unique(key, return_inverse=True)
        self.inv = inv
        self.nnz = len(u)
        self.indices = (u % N).astype(np.intc)
        colc = np.bincount((u // N).astype(np.int64), minlength=N)
        self.indptr = np.concatenate([[0], np.cumsum(colc)]).astype(np.intc)

    def _gstrf(self, data, cp, thresh):
        opts = dict(DiagPivotThresh=thresh, ColPerm=cp, SymmetricMode=True)
        return SLU.gstrf(self.N, self.nnz, data, self.indices, self.indptr,
                         csc_construct_func=lambda *a: None, ilu=False, options=opts)

    def factor(self, hv, reg, sc=None):
        n, m = self.n, self.m
        dg = np.concatenate([np.full(n, reg), np.full(m, -reg)])
        ext = []
        if self.K.nlarge:
            dg = np.concatenate([dg, np.tile([1.0, -1.0], self.K.nlarge)])
            ext = [sc.ext_w, sc.ext_w, sc.ext_h, sc.ext_h]
        vals = np.concatenate([self.base, -hv, dg] + ext)
        data = np.bincount(self.inv, weights=vals, minlength=self.nnz)
        self.lu_perm = False
        if self.perm is not None:
            try:
                self.lu = self._gstrf(data, 'NATURAL', 0.0)
                self.lu_perm = True
                return
            except Exception:
                self.perm = None
                self._structure(self.rows_all, self.cols_all, self.N)
                data = np.bincount(self.inv, weights=vals, minlength=self.nnz)
                self.tuned = True
        t = time.time()
        try:
            self.lu = self._gstrf(data, self.cp, 0.0)
        except Exception:
            self.lu = self._gstrf(data, self.cp, 0.1)
            self.tuned = True
        el = time.time() - t
        if not self.tuned:
            self.tuned = True
            if el > 0.02 and self.maxcol > max(200, 0.25 * self.N):
                t = time.time()
                try:
                    lu2 = self._gstrf(data, 'COLAMD', 0.0)
                    if time.time() - t < 0.7 * el:
                        self.cp = 'COLAMD'; self.lu = lu2
                except Exception:
                    pass
            if REUSE_ORDER:
                # fix the symmetric ordering for later factorizations
                pc = np.asarray(self.lu.perm_c)
                pinv = pc.astype(np.int64)
                self.perm = np.argsort(pinv)
                self.pinv = pinv
                self._structure(pinv[self.rows_all], pinv[self.cols_all], self.N)

    def lsolve(self, rhs):
        if self.N > self.N0:
            rhs = np.concatenate([rhs, np.zeros(self.N - self.N0)])
        if self.perm is not None and self.lu_perm:
            y = np.empty(self.N)
            y[self.pinv] = rhs
            x = self.lu.solve(y)[self.pinv]
        else:
            x = self.lu.solve(rhs)
        return x[:self.N0]

    def solve(self, rhs, scaling, nref=8):
        n = self.n
        x = self.lsolve(rhs)
        if not np.all(np.isfinite(x)):
            return x
        bn = inf_norm(rhs)
        for _ in range(nref):
            r = rhs - self.mul(x, scaling)
            rn = inf_norm(r)
            if rn <= 1e-13 * (1 + bn) or not np.isfinite(rn):
                break
            dx = self.lsolve(r)
            xn = x + dx
            rn2 = inf_norm(rhs - self.mul(xn, scaling))
            if rn2 >= rn:
                break
            x = xn
        return x

    def mul(self, v, scaling):
        n = self.n
        x = v[:n]; z = v[n:]
        top = self.P.mv(x) + self.A.tmv(z)
        bot = self.A.mv(x) - scaling.Hmul(z)
        return np.concatenate([top, bot])


class DenseKKT(KKT):
    """Reduced system: eliminate dz on cone rows, dense LU on (n + m0)."""
    def __init__(self, n, m, P, A, K):
        self.n, self.m = n, m
        self.P, self.A, self.K = P, A, K
        Ad = np.zeros((m, n))
        np.add.at(Ad, (A.r, A.c), A.v)
        self.Ad = Ad
        Pd = np.zeros((n, n))
        np.add.at(Pd, (P.r, P.c), P.v)
        self.Pd = Pd
        self.z0 = K.zero
        self.A0 = Ad[K.zero]
        self.sc = None
        self.full = False
        self.N = n + m

    def factor_dense(self, sc, reg):
        n = self.n; m0 = self.z0.size
        self.sc = sc; self.reg = reg
        self.qr = False
        if not self.full and m0 == 0 and self.P.v.size == 0 and np.all(np.isfinite(sc.s)):
            Wt = sc.Winv_mat(self.Ad)
            Wt = np.vstack([Wt, math.sqrt(reg) * np.eye(n)])
            Rq = np.linalg.qr(Wt, mode='r')
            self.Rq = np.ascontiguousarray(Rq)
            self.qr = True
            return
        if self.full:
            N = self.N
            Kd = np.zeros((N, N))
            Kd[:n, :n] = self.Pd
            Kd[np.arange(n), np.arange(n)] += reg
            Kd[n:, :n] = self.Ad
            Kd[:n, n:] = self.Ad.T
            hv = sc.ensure_hv()
            npsd = sum(len(Hb.ravel()) for Hb in sc.psd_H)
            nh = hv.size - npsd
            np.add.at(Kd, (n + self.K.hr[:nh], n + self.K.hc[:nh]), -hv[:nh])
            for (p, d), Hb in zip(self.K.psd, sc.psd_H):
                t = Hb.shape[0]
                Kd[n + p:n + p + t, n + p:n + p + t] -= Hb
            Kd[np.arange(n, N), np.arange(n, N)] -= reg
            self.lu, self.piv, info = LAPACK.dgetrf(Kd)
            return
        HA = sc.Hinv_mat(self.Ad)
        HA[self.z0] = 0.0
        M = self.Pd + self.Ad.T @ HA
        M[np.diag_indices(n)] += reg
        Kd = np.zeros((n + m0, n + m0))
        Kd[:n, :n] = M
        Kd[n:, :n] = self.A0
        Kd[:n, n:] = self.A0.T
        Kd[np.arange(n, n + m0), np.arange(n, n + m0)] = -reg
        self.lu, self.piv, info = LAPACK.dgetrf(Kd)

    def lusolve(self, rhs):
        n = self.n; sc = self.sc
        if self.full:
            sol, info = LAPACK.dgetrs(self.lu, self.piv, rhs)
            return sol
        if self.qr:
            r1 = rhs[:n]; r2 = rhs[n:]
            u = sc.Hinv_mul(r2)
            top = r1 + self.A.tmv(u)
            y, info = LAPACK.dtrtrs(self.Rq, top, lower=0, trans=1)
            x, info = LAPACK.dtrtrs(self.Rq, y, lower=0, trans=0)
            z = sc.Hinv_mul(self.A.mv(x) - r2)
            return np.concatenate([x, z])
        r1 = rhs[:n]; r2 = rhs[n:]
        r2k = r2.copy(); r2k[self.z0] = 0.0
        u = sc.Hinv_mul(r2k)
        top = r1 + self.A.tmv(u)
        bot = r2[self.z0]
        sol, info = LAPACK.dgetrs(self.lu, self.piv, np.concatenate([top, bot]))
        x = sol[:n]
        z = sc.Hinv_mul(self.A.mv(x) - r2k)
        z[self.z0] = sol[n:]
        return np.concatenate([x, z])

    def solve(self, rhs, scaling, nref=8):
        x = self.solve1(rhs, scaling, nref)
        res = inf_norm(rhs - self.mul(x, scaling))
        if not self.full and not (res <= 1e-9 * (1 + inf_norm(rhs))) and self.N <= 3500:
            if DEBUG: print('   switch to full KKT', res, file=sys.stderr)
            self.full = True
            self.factor_dense(self.sc, self.reg)
            x = self.solve1(rhs, scaling, nref)
        return x

    def solve1(self, rhs, scaling, nref=8):
        x = self.lusolve(rhs)
        if not np.all(np.isfinite(x)):
            return x
        bn = inf_norm(rhs)
        for _ in range(nref):
            r = rhs - self.mul(x, scaling)
            rn = inf_norm(r)
            if rn <= 1e-13 * (1 + bn) or not np.isfinite(rn):
                break
            xn = x + self.lusolve(r)
            rn2 = inf_norm(rhs - self.mul(xn, scaling))
            if rn2 >= rn:
                break
            x = xn
        if DEBUG: print('      ref', inf_norm(rhs - self.mul(x, scaling)), bn, file=sys.stderr)
        return x


# ---------------------------------------------------------------- main solver
def ruiz(P, A, q, b, K, n, m, iters=15):
    D = np.ones(n); E = np.ones(m)
    Pv = P.v.copy(); Av = A.v.copy()
    # block lists for E rectification
    blocks = []
    if K.nsoc:
        blocks.append(('seg', K.soc_idx, K.soc_off, K.soc_seg))
    for it in range(iters):
        cn = np.zeros(n)
        if Pv.size:
            np.maximum.at(cn, P.c, np.abs(Pv))
        if Av.size:
            np.maximum.at(cn, A.c, np.abs(Av))
        rn = np.zeros(m)
        if Av.size:
            np.maximum.at(rn, A.r, np.abs(Av))
        # rectify cones: use max over block for non-separable cones
        if K.nsoc:
            v = rn[K.soc_idx]
            mx = np.maximum.reduceat(v, K.soc_off)
            rn[K.soc_idx] = mx[K.soc_seg]
        if K.nexp:
            v = rn[K.exp_idx]
            rn[K.exp_idx] = v.max(axis=1)[:, None]
        for p, d in K.psd:
            t = d * (d + 1) // 2
            rn[p:p + t] = rn[p:p + t].max()
        cn = np.where(cn == 0, 1.0, cn); rn = np.where(rn == 0, 1.0, rn)
        dd = np.clip(1.0 / np.sqrt(cn), 1e-4, 1e4)
        ee = np.clip(1.0 / np.sqrt(rn), 1e-4, 1e4)
        if np.all(np.abs(dd - 1) < 1e-3) and np.all(np.abs(ee - 1) < 1e-3):
            break
        D *= dd; E *= ee
        Pv = Pv * dd[P.r] * dd[P.c]
        Av = Av * ee[A.r] * dd[A.c]
    D = np.clip(D, 1e-6, 1e6); E = np.clip(E, 1e-6, 1e6)
    Pv = P.v * D[P.r] * D[P.c]
    Av = A.v * E[A.r] * D[A.c]
    qs = q * D
    # cost scaling
    pn = 0.0
    if Pv.size:
        cn = np.zeros(n); np.maximum.at(cn, P.c, np.abs(Pv)); pn = float(np.mean(cn))
    c = 1.0 / max(1.0, pn, inf_norm(qs))
    c = min(max(c, 1e-4), 1e4)
    return D, E, c, Pv, Av


def solve(data, deadline):
    meta = json.loads(str(data['metadata_json']))
    q0 = np.asarray(data['q'], dtype=np.float64)
    b0 = np.asarray(data['b'], dtype=np.float64)
    n = q0.size; m = b0.size

    def load(prefix, shape):
        ind = np.asarray(data[prefix + '_indices'], dtype=np.int64)
        ptr = np.asarray(data[prefix + '_indptr'], dtype=np.int64)
        val = np.asarray(data[prefix + '_data'], dtype=np.float64)
        cols = np.repeat(np.arange(shape[1]), np.diff(ptr))
        return Mat(shape[0], shape[1], ind, cols, val)

    P0 = load('P', (n, n)); A0 = load('A', (m, n))
    K = Cones(meta['cones'], m)
    a_sc = A0.absmax() or 1.0
    h_sc = P0.absmax() or 1.0
    bn0 = inf_norm(b0); qn0 = inf_norm(q0)

    D, E, c, Pv, Av = ruiz(P0, A0, q0, b0, K, n, m)
    P = Mat(n, n, P0.r, P0.c, Pv); A = Mat(m, n, A0.r, A0.c, Av)
    q = c * D * q0; Pv *= c; P.v = Pv * 1.0
    b = E * b0

    # original-space checks
    def check_solved(x, s, z):
        Ax = A0.mv(x); Px = P0.mv(x); Atz = A0.tmv(z)
        rp = inf_norm(Ax + s - b0) / (1 + max(inf_norm(Ax), inf_norm(s), bn0))
        rd = inf_norm(Px + q0 + Atz) / (1 + max(inf_norm(Px), qn0, inf_norm(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q0 @ x); d0 = -0.5 * xPx - float(b0 @ z)
        gap = abs(p0 - d0)
        den = max(abs(p0), abs(d0))
        rg = min(gap, gap / den if den > 0 else 0.0)
        return rp, rd, rg

    def check_pinf(z):
        zn = inf_norm(z)
        if zn <= 0 or bn0 <= 0:
            return np.inf
        zh = z / zn
        beta = -float((b0 / bn0) @ zh)
        if beta <= 1e-7:
            return np.inf
        return inf_norm(A0.tmv(zh) / a_sc) / beta

    def check_dinf(x, s):
        xn = inf_norm(x)
        if xn <= 0 or qn0 <= 0:
            return np.inf
        xh = x / xn; sh = s / xn
        al = -float((q0 / qn0) @ xh)
        if al <= 1e-7:
            return np.inf
        return max(inf_norm(P0.mv(xh)) / h_sc, inf_norm(A0.mv(xh) / a_sc + sh / a_sc)) / al

    dense = bool(K.psd) and (n + K.zero.size) <= 3000 and m * n <= 4e7
    kkt = DenseKKT(n, m, P, A, K) if dense else KKT(n, m, P, A, K)
    nu = K.nu
    # ---- initial point
    if K.nexp == 0 and not dense:
        sc0 = type('S', (), {})()
        hv0 = np.ones(K.hr.size)
        # identity H pattern: need diag ones only
        hv0 = np.where(K.hr == K.hc, 1.0, 0.0)
        hv0[:K.zero.size] = 0.0
        class _I:
            def Hmul(self_, v):
                out = v.copy(); out[K.zero] = 0.0; return out
        I_ = _I()
        if K.nlarge:
            I_.ext_w = np.where(K.soc_tail[K.ext_pos], 0.0, -SQ2)
            I_.ext_h = np.full(K.nlarge, SQ2)
        kkt.factor(hv0, 1e-8, I_)
        sol = kkt.solve(np.concatenate([-q, b]), I_)
        x = sol[:n]; zt = sol[n:]
        s = K.shift_into(-zt)
        z = K.shift_into(zt)
        z[K.zero] = zt[K.zero]
        s[K.zero] = 0.0
    else:
        x = np.zeros(n)
        s = K.unit(); z = K.unit()
    tau = 1.0; kappa = 1.0

    status = 'unresolved'; best = None
    maxit = 200
    for it in range(maxit):
        Px = P.mv(x)
        rx = Px + A.tmv(z) + q * tau
        rz = A.mv(x) + s - b * tau
        xPx = float(x @ Px)
        rt = float(q @ x + b @ z) + kappa + xPx / tau
        mu = (float(s @ z) + tau * kappa) / (nu + 1)
        # termination checks in original space
        xo = D * x / tau; so = s / E / tau; zo = E * z / (c * tau)
        rp, rd, rg = check_solved(xo, so, zo)
        if max(rp, rd, rg) <= 0.95 * EPS:
            return 'solved', xo, so, zo
        pc = check_pinf(E * z)
        if pc <= 0.95 * EPS:
            return 'primal_infeasible', None, None, E * z
        dc = check_dinf(D * x, s / E)
        if dc <= 0.95 * EPS:
            return 'dual_infeasible', D * x, s / E, None
        if DEBUG:
            print('it %3d mu %.2e tau %.2e kap %.2e rp %.2e rd %.2e rg %.2e pc %.2e dc %.2e' % (it, mu, tau, kappa, rp, rd, rg, pc, dc), file=sys.stderr)
        if time.time() > deadline:
            break
        # scaling and factorization
        try:
            sc = Scaling(K, s, z, mu, need_h=not dense)
        except np.linalg.LinAlgError:
            break
        if dense:
            kkt.factor_dense(sc, 1e-8)
        else:
            kkt.factor(sc.hv, 1e-8, sc)
        xi = x / tau
        # constant system
        sol2 = kkt.solve(np.concatenate([-q, b]), sc)
        x2 = sol2[:n]; z2 = sol2[n:]
        Pxi = Px / tau
        qq = q + 2.0 * Pxi
        xiPxi = xPx / tau / tau

        def direction(dx, dz, dt, ds, dk):
            sol1 = kkt.solve(np.concatenate([-dx, ds - dz]), sc)
            x1 = sol1[:n]; z1 = sol1[n:]
            num = -dt + dk / tau - float(qq @ x1) - float(b @ z1)
            den = float(qq @ x2) + float(b @ z2) - kappa / tau - xiPxi
            dtau = num / den
            Dx = x1 + dtau * x2; Dz = z1 + dtau * z2
            Ds = -ds - sc.Hmul(Dz)
            Ds[K.zero] = 0.0
            Dk = -(dk + kappa * dtau) / tau
            return Dx, Dz, Ds, dtau, Dk

        def steplen(Dx, Dz, Ds, dtau, Dk, amax=1.0):
            a = amax
            if dtau < 0: a = min(a, -tau / dtau)
            if Dk < 0: a = min(a, -kappa / Dk)
            a = step_sym(K, s, Ds, a)
            a = step_sym(K, z, Dz, a)
            return a

        def exp_backtrack(a, Dz, Ds, dtau, Dk, center=True):
            if not K.nexp:
                return a
            ei = K.exp_idx
            for _ in range(60):
                Sn = s[ei] + a * Ds[ei]; Zn = z[ei] + a * Dz[ei]
                if np.all(exp_primal_ok(Sn)) and np.all(exp_dual_ok(Zn)):
                    if not center:
                        return a
                    sn = s + a * Ds; zn = z + a * Dz
                    mun = (float(sn @ zn) + (tau + a * dtau) * (kappa + a * Dk)) / (nu + 1)
                    prod = np.sum(Sn * Zn, axis=1)
                    if np.all(prod >= 0.1 * 3 * mun) or a < 1e-7:
                        return a
                a *= 0.8
            return 0.0

        # affine step
        ds_a = sc.ds_vec(s, z, mu, 0.0)
        ds_a[K.zero] = 0.0
        if K.nexp:
            ds_a[K.exp_idx] = s[K.exp_idx]
        Dx, Dz, Ds, dtau, Dk = direction(rx, rz, rt, ds_a, tau * kappa)
        if not (np.all(np.isfinite(Dx)) and np.all(np.isfinite(Dz))):
            break
        aa = steplen(Dx, Dz, Ds, dtau, Dk)
        aa = exp_backtrack(aa, Dz, Ds, dtau, Dk, center=False)
        if DEBUG: print('   aff', aa, file=sys.stderr)
        sigma = (1.0 - aa) ** 3
        sigma = min(max(sigma, 1e-8), 1.0)
        if K.nexp:
            sigma = max(sigma, 0.1 * 0)  # placeholder
        # combined step
        ds_c = sc.ds_vec(s, z, mu, sigma, Ds, Dz)
        ds_c[K.zero] = 0.0
        dk_c = tau * kappa + dtau * Dk - sigma * mu
        Dx, Dz, Ds, dtau, Dk = direction((1 - sigma) * rx, (1 - sigma) * rz, (1 - sigma) * rt, ds_c, dk_c)
        if not (np.all(np.isfinite(Dx)) and np.all(np.isfinite(Dz))):
            break
        a = steplen(Dx, Dz, Ds, dtau, Dk)
        a = min(1.0, 0.99 * a)
        a = exp_backtrack(a, Dz, Ds, dtau, Dk, center=True)
        if (K.nexp and a < 0.2) or a < 1e-3:
            for sig2 in (max(sigma, 0.3), 0.7):
                ds_c = sc.ds_vec(s, z, mu, sig2)
                ds_c[K.zero] = 0.0
                dk_c = tau * kappa - sig2 * mu
                c2 = direction((1 - sig2) * rx, (1 - sig2) * rz, (1 - sig2) * rt, ds_c, dk_c)
                a2 = min(1.0, 0.99 * steplen(*c2))
                a2 = exp_backtrack(a2, c2[1], c2[2], c2[3], c2[4], center=True)
                if a2 > a:
                    a = a2; Dx, Dz, Ds, dtau, Dk = c2; sigma = sig2
                if a >= 0.2:
                    break
        if DEBUG: print('   comb', a, sigma, file=sys.stderr)
        if a < 1e-10:
            break
        x = x + a * Dx; s = s + a * Ds; z = z + a * Dz
        tau = tau + a * dtau; kappa = kappa + a * Dk
        s[K.zero] = 0.0
    return 'unresolved', None, None, None


def write(path, status, x=None, s=None, z=None, obj=None):
    meta = {'schema_version': 1, 'status': status}
    arrs = {'metadata_json': np.array(json.dumps(meta))}
    for k, v in (('x', x), ('s', s), ('z', z)):
        if v is not None:
            arrs[k] = np.asarray(v, dtype=np.float64)
    with open(path, 'wb') as f:
        np.savez(f, **arrs)


def main():
    args = sys.argv[1:]
    inp = out = None
    for i, a in enumerate(args):
        if a == '--input': inp = args[i + 1]
        elif a == '--output': out = args[i + 1]
        elif a.startswith('--input='): inp = a.split('=', 1)[1]
        elif a.startswith('--output='): out = a.split('=', 1)[1]
    tl = float(os.environ.get('SOLVER_TIME_BUDGET', '0') or 0)
    deadline = T0 + (tl if tl > 0 else 250.0)
    try:
        with np.load(inp, allow_pickle=False) as data:
            d = {k: data[k] for k in data.files}
        status, x, s, z = solve(d, deadline)
    except Exception as e:
        import traceback; traceback.print_exc()
        status, x, s, z = 'unresolved', None, None, None
    if status == 'solved':
        write(out, status, x, s, z)
    elif status == 'primal_infeasible':
        write(out, status, z=z)
    elif status == 'dual_infeasible':
        write(out, status, x=x, s=s)
    else:
        write(out, 'unresolved')


if __name__ == '__main__':
    main()
