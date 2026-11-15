"""Primal-dual interior-point method for the five cone families.

The complementarity scaling is built from the Hessian G = nabla^2 F(s) of a
log-homogeneous barrier F for each cone (F = -sum log s for R_+,
-log(t^2-||v||^2) for SOC, -log det X for PSD, and the standard 3-barrier for
K_exp).  For such barriers Euler's identity gives G s = -grad F(s) in int K*,
so H = mu G is a consistent primal-dual scaling for *every* cone, including
the nonsymmetric exponential cone.  The Newton system is reduced to a
quasi-definite saddle system in x and the multipliers of the zero cone:

    (P + A_c' H A_c) dx + A_0' dz_0 = -r_d - A_c' H r_p,c + A_c' r_c
    A_0 dx                          = -r_p,0
    dz_c = H (A_c dx + r_p,c) - r_c,   ds_c = -r_p,c - A_c dx

with r_c = z_c - sigma*mu*(-grad F(s_c)).  Mehrotra-style affine scaling sets
sigma.  Only H (never its inverse) is needed.
"""
import math
import time
import numpy as np

from verify import check_solved, psd_index
import linsolve


def _smat(u, d, rows, cols, scales):
    mat = np.zeros((d, d))
    vals = u * scales
    mat[rows, cols] = vals
    mat[cols, rows] = vals
    return mat


def _mat_svec(mat, rows, cols, inv_scales):
    return mat[rows, cols] * inv_scales


def _exp_interior(U, V, W):
    ok = (V > 0) & (W > 0)
    lv = np.log(np.maximum(V, 1e-300))
    lw = np.log(np.maximum(W, 1e-300))
    psi = V * (lw - lv) - U
    return ok & (psi > 0)


# T maps a dual exponential triple to a primal one: z in Kexp* iff T z in Kexp.
T_EXP = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., math.e]])
T_EXP_INV = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 1.0 / math.e]])


def _exp_hessian(S):
    """Hessian G = nabla^2 F at exp triples S (k,3), as (k,3,3)."""
    Uv, Vv, Wc = S[:, 0], S[:, 1], S[:, 2]
    L = np.log(Wc) - np.log(Vv)
    psi = Vv * L - Uv
    g = np.stack((-np.ones_like(Uv), L - 1.0, Vv / Wc), axis=1)
    ip = 1.0 / psi
    G = np.einsum("ki,kj,k->kij", g, g, ip ** 2)
    G[:, 1, 1] += 1.0 / (Vv * psi) + 1.0 / Vv ** 2
    G[:, 1, 2] += -1.0 / (Wc * psi)
    G[:, 2, 1] += -1.0 / (Wc * psi)
    G[:, 2, 2] += Vv / (Wc ** 2 * psi) + 1.0 / Wc ** 2
    return G


def _exp_nt_hessian(S, Z):
    """Generalized NT scaling: H = nabla^2 F(w) with H s = z, w in int K_exp.

    Solved by a small damped Newton system on the 3-dimensional scaling point.
    Falls back to the primal Hessian if no interior scaling point is found.
    """
    k = S.shape[0]
    w = S.copy()
    eps = 1e-6
    okw = np.ones(k, dtype=bool)
    for _ in range(12):
        if not np.isfinite(w).all():
            break
        G = _exp_hessian(w)
        f = np.einsum("kij,kj->ki", G, S) - Z
        if np.max(np.abs(f)) < 1e-10:
            break
        J = np.empty((k, 3, 3))
        for j in range(3):
            wp = w.copy(); wp[:, j] += eps
            wm = w.copy(); wm[:, j] -= eps
            J[:, :, j] = (np.einsum("kij,kj->ki", _exp_hessian(wp), S)
                          - np.einsum("kij,kj->ki", _exp_hessian(wm), S)) / (2.0 * eps)
        try:
            step = np.linalg.solve(J, -f[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        alpha = np.ones(k)
        for _ in range(25):
            wn = w + alpha[:, None] * step
            good = _exp_interior(wn[:, 0], wn[:, 1], wn[:, 2])
            if good.all():
                break
            alpha[~good] *= 0.5
        w = w + alpha[:, None] * step
        okw &= _exp_interior(w[:, 0], w[:, 1], w[:, 2])
    try:
        H = _exp_hessian(w)
        good = (okw & np.isfinite(H).all(axis=(1, 2))
                & (np.linalg.eigvalsh(0.5 * (H + np.transpose(H, (0, 2, 1))))[:, 0] > 1e-14))
    except np.linalg.LinAlgError:
        H = np.zeros((k, 3, 3))
        good = np.zeros(k, dtype=bool)
    if not good.all():
        Hp = _exp_hessian(S)
        H[~good] = Hp[~good]
    return 0.5 * (H + np.transpose(H, (0, 2, 1)))


def _geomean(A, B):
    """Geometric mean A # B of batched SPD 3x3 matrices."""
    ev, U = np.linalg.eigh(A)
    ev = np.maximum(ev, 1e-300)
    Ah = np.einsum("kij,kj,klj->kil", U, np.sqrt(ev), U)
    Ahi = np.einsum("kij,kj,klj->kil", U, 1.0 / np.sqrt(ev), U)
    C = Ahi @ B @ Ahi
    C = 0.5 * (C + np.transpose(C, (0, 2, 1)))
    lam, V = np.linalg.eigh(C)
    lam = np.maximum(lam, 1e-300)
    root = np.einsum("kij,kj,klj->kil", V, np.sqrt(lam), V)
    return Ah @ root @ Ah


class ConeOps:
    def __init__(self, layout, m):
        self.m = m
        zeros, nns = [], []
        soc = {}
        psd = {}
        exps = []
        nu = 0
        for kind, dim, start, width in layout:
            if kind == "zero":
                zeros.append(np.arange(start, start + width))
            elif kind == "nonnegative":
                nns.append(np.arange(start, start + width))
                nu += dim
            elif kind == "soc":
                soc.setdefault(dim, []).append(start)
                nu += 2
            elif kind == "psd":
                psd.setdefault(dim, []).append(start)
                nu += dim
            else:
                exps.append(start)
                nu += 3
        self.zero_idx = np.concatenate(zeros).astype(np.int64) if zeros else np.empty(0, np.int64)
        self.nn_idx = np.concatenate(nns).astype(np.int64) if nns else np.empty(0, np.int64)
        self.soc = []
        for dim, starts in soc.items():
            idx = np.array(starts, dtype=np.int64)[:, None] + np.arange(dim)[None, :]
            self.soc.append((dim, idx))
        self.psd = []
        for dim, starts in psd.items():
            w = dim * (dim + 1) // 2
            idx = np.array(starts, dtype=np.int64)[:, None] + np.arange(w)[None, :]
            rows, cols, scales = psd_index(dim)
            self.psd.append((dim, idx, rows, cols, scales, 1.0 / scales))
        self.exp = (np.array(exps, dtype=np.int64)[:, None] + np.arange(3)[None, :]
                    ) if exps else None
        self.nu = nu
        self.conic_idx = np.setdiff1d(np.arange(m), self.zero_idx, assume_unique=True)
        self.has_nonsym = bool(self.exp is not None)

    def init(self, sbar, mu0):
        """Interior primal point near sbar = b - A x0, dual on its central ray."""
        s = np.zeros(self.m)
        ni = self.nn_idx
        if ni.size:
            v = sbar[ni]
            shift = max(0.0, -1.5 * float(v.min())) + 1.0 + 0.5 * float(np.max(np.abs(v)))
            s[ni] = v + shift
        for dim, idx in self.soc:
            B = sbar[idx]
            vnorm = np.linalg.norm(B[:, 1:], axis=1)
            delta = np.maximum(1.0, vnorm - B[:, 0] + 1.0)
            S = B.copy()
            S[:, 0] += delta
            s[idx] = S
        for dim, idx, rows, cols, scales, inv_scales in self.psd:
            for k in range(idx.shape[0]):
                M = _smat(sbar[idx[k]], dim, rows, cols, scales)
                try:
                    lmin = float(np.linalg.eigvalsh(M)[0])
                except np.linalg.LinAlgError:
                    lmin = -np.linalg.norm(M)
                delta = max(1.0, -lmin + 1.0)
                M += delta * np.eye(dim)
                s[idx[k]] = _mat_svec(M, rows, cols, inv_scales)
        if self.exp is not None:
            B = sbar[self.exp].copy()
            delta = np.maximum(1.0, np.maximum(-B[:, 1], -B[:, 2] / math.e) + 1.0)
            for _ in range(40):
                V = B[:, 1] + delta
                W = B[:, 2] + math.e * delta
                U = B[:, 0]
                psi = V * (np.log(np.maximum(W, 1e-300)) - np.log(np.maximum(V, 1e-300))) - U
                bad = ~((V > 0) & (W > 0) & (psi > 0))
                if not np.any(bad):
                    break
                delta[bad] *= 2.0
            S = B.copy()
            S[:, 1] += delta
            S[:, 2] += math.e * delta
            s[self.exp] = S
        z = mu0 * self.W(s)
        s[self.zero_idx] = 0.0
        z[self.zero_idx] = 0.0
        return s, z

    def W(self, s):
        """w = -grad F(s) = G s, in int K*."""
        w = np.zeros(self.m)
        w[self.nn_idx] = 1.0 / s[self.nn_idx]
        for dim, idx in self.soc:
            S = s[idx]
            delta = S[:, 0] ** 2 - np.einsum("ij,ij->i", S[:, 1:], S[:, 1:])
            U = S.copy()
            U[:, 1:] *= -1.0
            w[idx] = (2.0 / delta)[:, None] * U
        for dim, idx, rows, cols, scales, inv_scales in self.psd:
            for k in range(idx.shape[0]):
                X = _smat(s[idx[k]], dim, rows, cols, scales)
                R = np.linalg.inv(X)
                w[idx[k]] = _mat_svec(R, rows, cols, inv_scales)
        if self.exp is not None:
            S = s[self.exp]
            U, V, Wc = S[:, 0], S[:, 1], S[:, 2]
            L = np.log(Wc) - np.log(V)
            psi = V * L - U
            g0 = -np.ones_like(U)
            g1 = L - 1.0
            g2 = V / Wc
            w[self.exp] = np.stack((g0 / psi,
                                    g1 / psi + 1.0 / V,
                                    g2 / psi + 1.0 / Wc), axis=1)
        w[self.zero_idx] = 0.0
        return w

    def prepare(self, s, z, mu, exp_mu=None):
        return HState(self, s, z, mu, exp_mu=exp_mu)

    def prepare_primal(self, s, mu):
        return HState(self, s, None, mu, primal=True)

    def barrier(self, s):
        """Sum of log-homogeneous barrier values (finite iff strictly interior)."""
        val = 0.0
        ni = self.nn_idx
        if ni.size:
            v = s[ni]
            if np.any(v <= 0):
                return np.inf
            val -= float(np.sum(np.log(v)))
        for dim, idx in self.soc:
            S = s[idx]
            delta = S[:, 0] ** 2 - np.einsum("ij,ij->i", S[:, 1:], S[:, 1:])
            if np.any(delta <= 0) or np.any(S[:, 0] <= 0):
                return np.inf
            val -= float(np.sum(np.log(delta)))
        for dim, idx, rows, cols, scales, inv_scales in self.psd:
            for k in range(idx.shape[0]):
                X = _smat(s[idx[k]], dim, rows, cols, scales)
                sign, ld = np.linalg.slogdet(X)
                if sign <= 0:
                    return np.inf
                val -= float(ld)
        if self.exp is not None:
            S = s[self.exp]
            U, V, Wc = S[:, 0], S[:, 1], S[:, 2]
            psi = V * (np.log(np.maximum(Wc, 1e-300)) - np.log(np.maximum(V, 1e-300))) - U
            if np.any(V <= 0) or np.any(Wc <= 0) or np.any(psi <= 0):
                return np.inf
            val -= float(np.sum(np.log(psi) + np.log(V) + np.log(Wc)))
        return val

    def maxstep(self, s, ds, dual=False, cap=1.0):
        best = cap
        if dual and self.exp is not None:
            s, ds = _dual_transform(s, ds, self.exp)
        elif not dual and self.exp is not None:
            pass
        ni = self.nn_idx
        if ni.size:
            d = ds[ni]
            v = s[ni]
            neg = d < 0
            if np.any(neg):
                cand = -v[neg] / d[neg]
                best = min(best, float(cand.min()))
        for dim, idx in self.soc:
            S, D = s[idx], ds[idx]
            t, v = S[:, 0], S[:, 1:]
            dt, dv = D[:, 0], D[:, 1:]
            delta = t * t - np.einsum("ij,ij->i", v, v)
            a = dt * dt - np.einsum("ij,ij->i", dv, dv)
            b = 2.0 * (t * dt - np.einsum("ij,ij->i", v, dv))
            c = delta
            # find first positive root in (0,1] of a*x^2+b*x+c
            al = _quad_maxstep(a, b, c, t, dt)
            if al < best:
                best = al
        for dim, idx, rows, cols, scales, inv_scales in self.psd:
            for k in range(idx.shape[0]):
                S = _smat(s[idx[k]], dim, rows, cols, scales)
                D = _smat(ds[idx[k]], dim, rows, cols, scales)
                try:
                    L = np.linalg.cholesky(S)
                    Linv = np.linalg.solve(L, np.eye(dim))
                    B = Linv @ D @ Linv.T
                    lam = float(np.linalg.eigvalsh(B)[0])
                except np.linalg.LinAlgError:
                    return 0.0
                if lam < 0:
                    best = min(best, -1.0 / lam)
        if self.exp is not None:
            # in dual case s,ds already transformed; else plain exp triples
            if dual:
                idx = self.exp
                S = s[idx]
                D = ds[idx]
            else:
                S = s[self.exp]
                D = ds[self.exp]
            al = _exp_maxstep(S, D)
            best = min(best, al)
        return max(0.0, best)


def _dual_transform(s, ds, exp_idx):
    """Map exp-cone dual triples into primal exp coordinates (values only)."""
    T = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., math.e]])
    s2 = s.copy()
    ds2 = ds.copy()
    s2[exp_idx] = s[exp_idx] @ T.T
    ds2[exp_idx] = ds[exp_idx] @ T.T
    return s2, ds2


def _quad_maxstep(a, b, c, t, dt):
    """Largest alpha<=1 with (t+a dt)^2-||v+a dv||^2>0 and t+a dt>0."""
    disc = b * b - 4.0 * a * c
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    r1 = (-b - sqrt_d) / (2.0 * np.where(a == 0, 1e-300, a))
    r2 = (-b + sqrt_d) / (2.0 * np.where(a == 0, 1e-300, a))
    root = np.where(np.isfinite(r1) & (r1 > 0), r1, np.inf)
    alt = np.where(np.isfinite(r2) & (r2 > 0), r2, np.inf)
    root = np.minimum(root, alt)
    # linear case a == 0: c + b*alpha = 0
    lin = np.where(b < 0, -c / np.where(b == 0, 1e-300, b), np.inf)
    root = np.where(a == 0, lin, root)
    cap = np.where(dt < 0, -t / np.where(dt == 0, 1e-300, dt), np.inf)
    return float(np.minimum(1.0, np.min(np.minimum(root, cap))))


def _exp_maxstep(S, D):
    k = S.shape[0]
    lo = np.zeros(k)
    hi = np.ones(k)
    ok1 = _exp_interior(S[:, 0] + D[:, 0], S[:, 1] + D[:, 1], S[:, 2] + D[:, 2])
    lo[ok1] = 1.0
    if ok1.all():
        return 1.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ok = _exp_interior(S[:, 0] + mid * D[:, 0],
                           S[:, 1] + mid * D[:, 1],
                           S[:, 2] + mid * D[:, 2])
        lo[ok] = mid[ok]
        hi[~ok] = mid[~ok]
    return float(lo.min())


class HState:
    """Per-iteration scaling H, with H*s = z on symmetric cones (NT scaling).

    For the exponential cone no self-scaled barrier exists; there we use
    H = mu * nabla^2 F(s), for which H*s = -mu*grad F(s).
    """

    def __init__(self, ops, s, z, mu, primal=False, exp_mu=None):
        self.ops = ops
        self.mu = mu
        self.m = ops.m
        if primal:
            # H = mu * nabla^2 F(s): the primal barrier (central-ray) scaling.
            self.hnn = mu / s[ops.nn_idx] ** 2
            self.soc = []
            for dim, idx in ops.soc:
                S = s[idx]
                delta = S[:, 0] ** 2 - np.einsum("ij,ij->i", S[:, 1:], S[:, 1:])
                U = S.copy()
                U[:, 1:] *= -1.0
                self.soc.append((idx, dim, mu * 4.0 / delta ** 2, U, -mu * 2.0 / delta))
            self.psd = []
            for dim, idx, rows, cols, scales, inv_scales in ops.psd:
                Rs = []
                for k in range(idx.shape[0]):
                    X = _smat(s[idx[k]], dim, rows, cols, scales)
                    try:
                        Rs.append(np.linalg.inv(X))
                    except np.linalg.LinAlgError:
                        Rs.append(np.linalg.pinv(X))
                self.psd.append((idx, dim, rows, cols, scales, inv_scales, Rs, mu))
            self.exp = (_exp_hessian(s[ops.exp]) * (mu if exp_mu is None else exp_mu)
                        if ops.exp is not None else None)
            self.primal = True
            return
        self.primal = False
        # nonnegative: H = diag(z/s)
        self.hnn = z[ops.nn_idx] / s[ops.nn_idx]
        # SOC: H v = hscale * (2 u (u.v) - J v),  u = J w_bar
        self.soc = []
        for dim, idx in ops.soc:
            S, Z = s[idx], z[idx]
            ss = np.sqrt(np.maximum(S[:, 0] ** 2 - np.einsum("ij,ij->i", S[:, 1:], S[:, 1:]), 1e-300))
            zz = np.sqrt(np.maximum(Z[:, 0] ** 2 - np.einsum("ij,ij->i", Z[:, 1:], Z[:, 1:]), 1e-300))
            sb = S / ss[:, None]
            zb = Z / zz[:, None]
            gamma = np.sqrt(np.maximum((1.0 + np.einsum("ij,ij->i", sb, zb)) / 2.0, 1e-300))
            Js = sb.copy()
            Js[:, 1:] *= -1.0
            U = (Js + zb) / (2.0 * gamma)[:, None]     # J w_bar
            self.soc.append((idx, dim, zz / ss, U))
        # PSD: H[M] = R M R with R = W^{-1} from the NT (geometric mean) point
        self.psd = []
        for dim, idx, rows, cols, scales, inv_scales in ops.psd:
            Rs = []
            for k in range(idx.shape[0]):
                S = _smat(s[idx[k]], dim, rows, cols, scales)
                Z = _smat(z[idx[k]], dim, rows, cols, scales)
                try:
                    ev, U = np.linalg.eigh(S)
                    ev = np.maximum(ev, 1e-300)
                    Sh = (U * np.sqrt(ev)) @ U.T
                    Shi = (U * (1.0 / np.sqrt(ev))) @ U.T
                    T = Sh @ Z @ Sh
                    T = 0.5 * (T + T.T)
                    lam, V = np.linalg.eigh(T)
                    lam = np.maximum(lam, 1e-300)
                    Rs.append(Shi @ (V * np.sqrt(lam)) @ V.T @ Shi)
                except np.linalg.LinAlgError:
                    Rs.append(np.eye(dim))
            self.psd.append((idx, dim, rows, cols, scales, inv_scales, Rs, 1.0))
        # exponential: generalized NT scaling (H s = z) or primal Hessian
        self.exp = None
        if ops.exp is not None:
            if getattr(ops, "exp_nt", False):
                self.exp = _exp_nt_hessian(s[ops.exp], z[ops.exp])
            else:
                self.exp = _exp_hessian(s[ops.exp]) * (mu if exp_mu is None else exp_mu)

    def apply(self, v):
        out = np.zeros(self.m)
        ops = self.ops
        out[ops.nn_idx] = self.hnn * v[ops.nn_idx]
        if self.primal:
            for idx, dim, c1, U, c2 in self.soc:
                V = v[idx]
                udv = np.einsum("ij,ij->i", U, V)
                Jv = V.copy()
                Jv[:, 1:] *= -1.0
                out[idx] = c1[:, None] * udv[:, None] * U + c2[:, None] * Jv
        else:
            for idx, dim, hs, U in self.soc:
                V = v[idx]
                udv = np.einsum("ij,ij->i", U, V)
                Jv = V.copy()
                Jv[:, 1:] *= -1.0
                out[idx] = hs[:, None] * (2.0 * udv[:, None] * U - Jv)
        for idx, dim, rows, cols, scales, inv_scales, Rs, fac in self.psd:
            for k in range(idx.shape[0]):
                M = _smat(v[idx[k]], dim, rows, cols, scales)
                out[idx[k]] = fac * _mat_svec(Rs[k] @ M @ Rs[k], rows, cols, inv_scales)
        if self.exp is not None:
            out[ops.exp] = np.einsum("kij,kj->ki", self.exp, v[ops.exp])
        out[ops.zero_idx] = 0.0
        return out

    def apply_cols(self, Ad):
        """H @ A for a dense (m,n) matrix A."""
        out = np.zeros_like(Ad)
        ops = self.ops
        out[ops.nn_idx, :] = self.hnn[:, None] * Ad[ops.nn_idx, :]
        if self.primal:
            for idx, dim, c1, U, c2 in self.soc:
                V = Ad[idx, :]
                udv = np.einsum("kd,kdn->kn", U, V)
                Jv = V.copy()
                Jv[:, 1:, :] *= -1.0
                out[idx, :] = (c1[:, None, None] * U[:, :, None] * udv[:, None, :]
                               + c2[:, None, None] * Jv)
        else:
            for idx, dim, hs, U in self.soc:
                V = Ad[idx, :]
                udv = np.einsum("kd,kdn->kn", U, V)
                Jv = V.copy()
                Jv[:, 1:, :] *= -1.0
                out[idx, :] = hs[:, None, None] * (2.0 * U[:, :, None] * udv[:, None, :] - Jv)
        for idx, dim, rows, cols, scales, inv_scales, Rs, fac in self.psd:
            for k in range(idx.shape[0]):
                blk = Ad[idx[k], :]
                nz = np.flatnonzero(np.any(blk != 0.0, axis=0))
                R = Rs[k]
                for j in nz:
                    M = _smat(blk[:, j], dim, rows, cols, scales)
                    out[idx[k], j] = fac * _mat_svec(R @ M @ R, rows, cols, inv_scales)
        if self.exp is not None:
            out[ops.exp, :] = np.einsum("kij,kjn->kin", self.exp, Ad[ops.exp, :])
        out[ops.zero_idx, :] = 0.0
        return out


def _lin_solve(K, rhs, refine=2):
    """Solve K y = rhs with diagonal equilibration and iterative refinement."""
    diag = np.abs(np.diag(K))
    d = 1.0 / np.sqrt(np.maximum(diag, 1e-12))
    Ks = K * d[:, None] * d[None, :]
    bs = rhs * d
    try:
        lu = np.linalg.solve(Ks, bs)
        for _ in range(refine):
            r = bs - Ks @ lu
            lu += np.linalg.solve(Ks, r)
        return lu * d
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Ks, bs, rcond=None)[0] * d



def _normal_matrix(Pd, A, HA, n):
    """P + A' H A, exploiting the sparsity of A when worthwhile."""
    nnz = A.nnz
    if nnz and nnz * n <= 25_000_000 and n <= 4000:
        rows = A.indices
        cols = A.columns()
        tmp = A.data[:, None] * HA[rows, :]
        idx = cols[:, None] * n + np.arange(n)[None, :]
        return Pd + np.bincount(idx.ravel(), weights=tmp.ravel(),
                                minlength=n * n).reshape(n, n)
    return Pd + A.dense().T @ HA


def solve(P, q, A, b, layout, deadline, tol_target=2e-9, maxit=120,
          x0=None, mu0=None, verbose=False, exp_factor=0.0, exp_nt=False):
    """Solve min .5x'Px+q'x s.t. Ax+s=b, s in K.  Returns dict."""
    n, m = len(q), len(b)
    ops = ConeOps(layout, m)
    ops.exp_nt = exp_nt and ops.exp is not None
    if ops.nu == 0:
        # pure equality-constrained QP: solve directly (no barrier needed)
        return _solve_equality(P, q, A, b, deadline)
    Ad = A.dense()
    Pd = P.dense()
    At = Ad.T.copy()
    zero_idx = ops.zero_idx
    m0 = zero_idx.size
    A0 = Ad[zero_idx, :] if m0 else None
    if mu0 is None:
        scale = max(1.0, float(np.max(np.abs(q))) if n else 1.0,
                    float(np.max(np.abs(b))) if m else 1.0)
        mu0 = scale
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    if x0 is None and m0 and m0 * n <= 4_000_000:
        try:
            x[:] = np.linalg.lstsq(A0, b[zero_idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    s, z = ops.init(b - Ad @ x, mu0)
    hybrid = exp_factor > 0.0 and ops.exp is not None
    mu_exp = mu0
    exp_rows = ops.exp.ravel() if hybrid else None
    # zero rows: s must stay zero; z free. start z_zero = 0
    best = None
    best_score = np.inf
    last_improve = 0
    stalled = 0
    data_scale = max(1.0, float(np.max(np.abs(q))) if n else 1.0,
                     float(np.max(np.abs(b))) if m else 1.0,
                     A.maxabs, P.maxabs)
    reg = 1e-12 * data_scale
    for it in range(maxit):
        if time.monotonic() > deadline:
            break
        Ax = Ad @ x
        Px = Pd @ x
        Atz = At @ z
        rp = Ax + s - b
        rd = Px + q + Atz
        szi = float(s[ops.conic_idx] @ z[ops.conic_idx])
        mu = max(szi / ops.nu, 1e-300)
        w = ops.W(s)                 # -grad F(s), so mu*w = H*s
        rcc_full = z.copy()          # affine (sigma=0) target: r_c = z
        rcc_full[zero_idx] = 0.0     # zero cone has no complementarity condition
        if hybrid:
            # Newton target consistent with z_new = fac*mu_exp*W(s_new)
            rcc_full[exp_rows] = (1.0 - exp_factor) * z[exp_rows]
        # cheap residuals; the full (cone-aware) check only near convergence
        inf_ = lambda v: float(np.max(np.abs(v), initial=0.0))
        rp_rel = inf_(rp) / (1.0 + max(inf_(Ax), inf_(s), inf_(b)))
        rd_rel = inf_(rd) / (1.0 + max(inf_(Px), inf_(q), inf_(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        gap = abs(p0 - d0)
        gaprel = gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) else 0.0
        score = max(rp_rel, rd_rel, min(gap, gaprel))
        diag = dict(rp=rp_rel, rd=rd_rel, gap=gap, gaprel=gaprel)
        if score < 1.5e-8:
            ok, _ = check_solved(P, q, A, b, layout, x, s, z, tol=1e-8)
            if ok:
                return dict(status="solved", x=x, s=s, z=z, it=it)
        if score < best_score:
            best_score = score
            best = (x.copy(), s.copy(), z.copy())
        state = ops.prepare(s, z, mu,
                            exp_mu=exp_factor * mu_exp if hybrid else None)
        HA = state.apply_cols(Ad)
        M = _normal_matrix(Pd, A, HA, n)
        M = 0.5 * (M + M.T)
        Hrp = state.apply(rp)
        rhs_m = -rd - At @ Hrp + At @ rcc_full
        if m0:
            K = np.empty((n + m0, n + m0))
            K[:n, :n] = M
            K[:n, n:] = A0.T
            K[n:, :n] = A0
            K[n:, n:] = 0.0
            d0 = np.arange(n)
            d1 = n + np.arange(m0)
            K[d0, d0] += reg
            K[d1, d1] -= reg
            rhs = np.concatenate((rhs_m, -rp[zero_idx]))
        else:
            K = M.copy()
            K.flat[::n + 1] += reg
            rhs = rhs_m
        try:
            lu = linsolve.LU(K)
            y = lu.refine(rhs, 2)
        except np.linalg.LinAlgError:
            lu = None
            y = _lin_solve(K, rhs)
        dx = y[:n]
        dz0 = y[n:]
        Adx = Ad @ dx
        dz = state.apply(Adx + rp) - rcc_full
        ds = -rp - Adx
        ds[zero_idx] = 0.0
        if m0:
            dz[zero_idx] = dz0
        # affine step length -> sigma
        ap0 = ops.maxstep(s, ds, dual=False)
        ad0 = ops.maxstep(z, dz, dual=True)
        aaff = min(1.0, 0.90 * min(ap0, ad0))
        mu_aff = max(float((s[ops.conic_idx] + aaff * ds[ops.conic_idx]) @
                           (z[ops.conic_idx] + aaff * dz[ops.conic_idx])) / ops.nu, 0.0)
        if ops.exp_nt:
            sigma = min(0.5, max(1e-4, (mu_aff / mu) ** 2)) if mu > 0 else 0.1
        else:
            sigma = min(0.99, max(1e-4, (mu_aff / mu) ** 3)) if mu > 0 else 0.1
        rcc_full = z - sigma * mu * w
        rcc_full[zero_idx] = 0.0
        if hybrid:
            rcc_full[exp_rows] = (1.0 - exp_factor) * z[exp_rows]
        rhs_m = -rd - At @ Hrp + At @ rcc_full
        if m0:
            rhs2 = np.concatenate((rhs_m, -rp[zero_idx]))
        else:
            rhs2 = rhs_m
        if lu is not None:
            y2 = lu.refine(rhs2, 2)
        else:
            y2 = _lin_solve(K, rhs2)
        dx = y2[:n]
        dz0 = y2[n:]
        Adx = Ad @ dx
        dz = state.apply(Adx + rp) - rcc_full
        ds = -rp - Adx
        ds[zero_idx] = 0.0
        if m0:
            dz[zero_idx] = dz0
        ap = ops.maxstep(s, ds, dual=False)
        ad = ops.maxstep(z, dz, dual=True)
        alpha = min(1.0, 0.90 * min(ap, ad))
        if alpha < 1e-6:
            stalled += 1
            if stalled > 3:
                break
        else:
            stalled = 0
        x += alpha * dx
        s += alpha * ds
        z += alpha * dz
        s[zero_idx] = 0.0
        if hybrid:
            mu_exp *= exp_factor
        if verbose:
            print("it", it, "mu", mu, "sigma", sigma, "alpha", alpha,
                  "rp", diag.get("rp"), "rd", diag.get("rd"), "gap", diag.get("gaprel"),
                  flush=True)
        if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
            break
    if best is not None:
        ok, _ = check_solved(P, q, A, b, layout, best[0], best[1], best[2], tol=1e-8)
        if ok:
            return dict(status="solved", x=best[0], s=best[1], z=best[2], it=-1)
    return dict(status="unresolved", x=x, s=s, z=z, best=best, it=maxit)


def solve_barrier(P, q, A, b, layout, deadline, mu0=None, maxit=400,
                  verbose=False, mu_factor=0.25, inner_max=12):
    """Primal barrier path-following with inner Newton iterations.

    z_c = mu*(-grad F(s_c)) is kept exactly on the central ray, so the dual
    iterate can never leave the (possibly nonsymmetric) dual cone.  For fixed
    mu we take damped Newton steps on the barrier subproblem until the Newton
    decrement is small, then reduce mu.
    """
    n, m = len(q), len(b)
    ops = ConeOps(layout, m)
    Ad, Pd, At = A.dense(), P.dense(), A.dense().T.copy()
    zi = ops.zero_idx
    m0 = zi.size
    A0 = Ad[zi, :] if m0 else None
    scale = max(1.0, float(np.max(np.abs(q))) if n else 1.0,
                float(np.max(np.abs(b))) if m else 1.0, A.maxabs, P.maxabs)
    mu = mu0 if mu0 is not None else scale
    reg = 1e-12 * scale
    x = np.zeros(n)
    if m0:
        try:
            x[:] = np.linalg.lstsq(A0, b[zi], rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    sbar = b - Ad @ x
    s0, _ = ops.init(sbar, mu)
    s = s0.copy()
    s[zi] = 0.0
    z = np.zeros(m)
    z[ops.conic_idx] = (mu * ops.W(s))[ops.conic_idx]
    best, best_score = None, np.inf
    iters = 0
    for outer in range(60):
        for inner in range(inner_max):
            iters += 1
            if time.monotonic() > deadline:
                return _barrier_return(P, q, A, b, layout, x, s, z, best, iters)
            rp = Ad @ x + s - b
            rd = Pd @ x + q + At @ z
            ok, diag = check_solved(P, q, A, b, layout, x, s, z, tol=1e-8)
            if ok:
                return dict(status="solved", x=x, s=s, z=z, it=iters)
            score = max(diag.get("rp", 1), diag.get("rd", 1),
                        min(1.0, min(diag.get("gap", 1), diag.get("gaprel", 1))))
            if score < best_score:
                best_score = score
                best = (x.copy(), s.copy(), z.copy())
            state = ops.prepare_primal(s, mu)
            HA = state.apply_cols(Ad)
            M = _normal_matrix(Pd, A, HA, n)
            M = 0.5 * (M + M.T)
            Hrp = state.apply(rp)
            rhs_m = -rd - At @ Hrp
            if m0:
                K = np.empty((n + m0, n + m0))
                K[:n, :n] = M
                K[:n, n:] = A0.T
                K[n:, :n] = A0
                K[n:, n:] = 0.0
                d0 = np.arange(n)
                d1 = n + np.arange(m0)
                K[d0, d0] += reg
                K[d1, d1] -= reg
                rhs = np.concatenate((rhs_m, -rp[zi]))
            else:
                K = M.copy()
                K[np.arange(n), np.arange(n)] += reg
                rhs = rhs_m
            y = linsolve.solve(K, rhs, 2)
            dx = y[:n]
            dz0 = y[n:]
            ds = -rp - Ad @ dx
            f0 = (0.5 * float(x @ (Pd @ x)) + float(q @ x)
                  + mu * ops.barrier(s) + float(z @ rp))
            gd = float(rd @ dx)
            decr = max(0.0, -gd)
            alpha = min(1.0, 0.99 * ops.maxstep(s, ds, dual=False))
            okstep = False
            for _ in range(35):
                xt = x + alpha * dx
                st = s + alpha * ds
                st[zi] = 0.0
                ft = (0.5 * float(xt @ (Pd @ xt)) + float(q @ xt)
                      + mu * ops.barrier(st) + float(z @ (Ad @ xt + st - b)))
                if np.isfinite(ft) and ft <= f0 + 1e-4 * alpha * gd + 1e-13 * max(1.0, abs(f0)):
                    okstep = True
                    break
                alpha *= 0.5
            if not okstep or alpha < 1e-9:
                break
            x += alpha * dx
            s = s + alpha * ds
            s[zi] = 0.0
            if m0:
                z[zi] += alpha * dz0
            z[ops.conic_idx] = (mu * ops.W(s))[ops.conic_idx]
            if verbose:
                print("b", iters, "mu", mu, "a", alpha, "rp", diag.get("rp"),
                      "rd", diag.get("rd"), "decr", decr, flush=True)
            if decr < 1e-10 * max(1.0, scale) and alpha > 0.5:
                break
            if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
                return _barrier_return(P, q, A, b, layout, x, s, z, best, iters)
        mu *= mu_factor
        if mu < 1e-15 * scale:
            # final centering steps at tiny mu
            mu = 1e-15 * scale
    return _barrier_return(P, q, A, b, layout, x, s, z, best, iters)


def _barrier_return(P, q, A, b, layout, x, s, z, best, iters):
    if best is not None:
        ok, _ = check_solved(P, q, A, b, layout, best[0], best[1], best[2], tol=1e-8)
        if ok:
            return dict(status="solved", x=best[0], s=best[1], z=best[2], it=iters)
    return dict(status="unresolved", x=x, s=s, z=z, best=best, it=iters)


def _solve_equality(P, q, A, b, deadline):
    """No conic rows at all: pure linear system KKT."""
    n, m = len(q), len(b)
    Ad, Pd = A.dense(), P.dense()
    K = np.empty((n + m, n + m))
    K[:n, :n] = Pd
    K[:n, n:] = Ad.T
    K[n:, :n] = Ad
    K[n:, n:] = 0.0
    reg = 1e-12 * max(1.0, np.max(np.abs(K)))
    K[np.arange(n), np.arange(n)] += reg
    K[n + np.arange(m), n + np.arange(m)] -= reg
    rhs = np.concatenate((-q, b))
    try:
        y = np.linalg.solve(K, rhs)
        y += np.linalg.solve(K, rhs - K @ y)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(K, rhs, rcond=None)[0]
    x, z = y[:n], y[n:]
    s = b - Ad @ x
    return dict(status="solved", x=x, s=s, z=z, it=0)
