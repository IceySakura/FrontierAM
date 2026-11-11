"""Cone utilities: NT scaling (nonnegative, SOC, PSD), primal-dual scaling for the
exponential cone, step lengths and Mehrotra / higher-order corrections.

All routines are vectorised over cones of the same kind (and SOC dimension)."""

import math
import numpy as np
from scipy import linalg as dla

SQ2 = math.sqrt(2.0)
EXP_CENTRAL = np.array([-1.051383945322714, 0.556409619469370, 1.258967884768947])


# ----------------------------------------------------------------------------- PSD helpers
class PSDIndex:
    def __init__(self, d):
        self.d = d
        I, J = [], []
        for j in range(d):
            for i in range(j + 1):
                I.append(i)
                J.append(j)
        self.I = np.array(I, dtype=np.int64)
        self.J = np.array(J, dtype=np.int64)
        self.N = len(I)
        self.f = np.where(self.I == self.J, 1.0, SQ2)      # svec multipliers
        self.fi = np.where(self.I == self.J, 1.0, 1.0 / SQ2)
        self.diag = np.where(self.I == self.J)[0]

    def smat(self, v):
        M = np.zeros((self.d, self.d))
        vals = v * self.fi
        M[self.I, self.J] = vals
        M[self.J, self.I] = vals
        return M

    def svec(self, M):
        return 0.5 * (M[self.I, self.J] + M[self.J, self.I]) * self.f

    def kron_s(self, G):
        """Matrix of v -> svec(G smat(v) G) (symmetric Kronecker product)."""
        I, J = self.I, self.J
        H = G[np.ix_(I, I)] * G[np.ix_(J, J)]
        H += G[np.ix_(I, J)] * G[np.ix_(J, I)]
        fh = np.where(I == J, 1.0 / SQ2, 1.0)
        H *= fh[:, None]
        H *= fh[None, :]
        return H


def _min_eig_sym(M):
    try:
        return float(dla.eigh(M, eigvals_only=True, subset_by_index=[0, 0], check_finite=False)[0])
    except Exception:
        return float(np.linalg.eigvalsh(M)[0])


# ----------------------------------------------------------------------------- exponential cone
def wright_omega(x):
    """Solve w + log(w) = x (vectorised, x real)."""
    x = np.asarray(x, dtype=float)
    w = np.where(x > 1.0, x - np.log(np.maximum(x, 1.0)), np.maximum(np.exp(np.minimum(x, 1.0) - 1.0), (x + 1.0) / 2.0))
    w = np.maximum(w, 1e-300)
    for _ in range(50):
        f = w + np.log(w) - x
        wn = w - f / (1.0 + 1.0 / w)
        wn = np.where(wn <= 0, w * 0.5, wn)
        if np.all(np.abs(wn - w) <= 1e-15 * np.maximum(1.0, np.abs(w))):
            w = wn
            break
        w = wn
    # one Halley polish
    f = w + np.log(w) - x
    fp = 1.0 + 1.0 / w
    fpp = -1.0 / (w * w)
    w = w - 2 * f * fp / (2 * fp * fp - f * fpp)
    return w


def exp_dual_grad_hess(Z):
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
    l = np.log(-z3 / z1)
    psi = z2 - z1 - z1 * l
    g = np.empty_like(Z)
    g[:, 0] = l / psi - 1.0 / z1
    g[:, 1] = -1.0 / psi
    g[:, 2] = (z1 / psi - 1.0) / z3
    H = np.empty((Z.shape[0], 3, 3))
    p2 = psi * psi
    H[:, 0, 0] = l * l / p2 - 1.0 / (z1 * psi) + 1.0 / (z1 * z1)
    H[:, 0, 1] = -l / p2
    H[:, 0, 2] = (l * z1 / z3) / p2 + 1.0 / (z3 * psi)
    H[:, 1, 1] = 1.0 / p2
    H[:, 1, 2] = -z1 / (z3 * p2)
    H[:, 2, 2] = (z1 * z1) / (z3 * z3 * p2) - z1 / (z3 * z3 * psi) + 1.0 / (z3 * z3)
    H[:, 1, 0] = H[:, 0, 1]
    H[:, 2, 0] = H[:, 0, 2]
    H[:, 2, 1] = H[:, 1, 2]
    return g, H, psi, l


def exp_dual_third(Z, U, V, psi, l):
    """grad^3 f*(z)[u, v] for the dual exponential barrier."""
    z1, z3 = Z[:, 0], Z[:, 2]
    dpsi = np.stack([-l, np.ones_like(l), -z1 / z3], axis=1)
    # second derivatives of psi: psi11 = 1/z1, psi13 = -1/z3, psi33 = z1/z3^2
    def hpsi(X):
        r = np.zeros_like(X)
        r[:, 0] = X[:, 0] / z1 - X[:, 2] / z3
        r[:, 2] = -X[:, 0] / z3 + X[:, 2] * z1 / (z3 * z3)
        return r
    du = np.sum(dpsi * U, axis=1)
    dv = np.sum(dpsi * V, axis=1)
    Hu = hpsi(U)
    Hv = hpsi(V)
    uHv = np.sum(U * Hv, axis=1)
    T = np.zeros_like(U)
    T[:, 0] = -U[:, 0] * V[:, 0] / (z1 * z1) + U[:, 2] * V[:, 2] / (z3 * z3)
    T[:, 2] = (U[:, 0] * V[:, 2] + U[:, 2] * V[:, 0]) / (z3 * z3) - 2 * z1 * U[:, 2] * V[:, 2] / (z3 ** 3)
    p = psi[:, None]
    res = (Hu * dv[:, None] + dpsi * uHv[:, None]) / p ** 2 - 2 * (du * dv)[:, None] * dpsi / p ** 3 \
        - T / p + du[:, None] * Hv / p ** 2
    res[:, 0] += -2.0 * U[:, 0] * V[:, 0] / z1 ** 3
    res[:, 2] += -2.0 * U[:, 2] * V[:, 2] / z3 ** 3
    return res


def exp_primal_grad(S):
    s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]
    om = wright_omega(1.0 - s1 / s2 - np.log(s2 / s3))
    g = np.empty_like(S)
    g[:, 0] = 1.0 / ((om - 1.0) * s2)
    g[:, 1] = g[:, 0] + g[:, 0] * np.log(om * s2 / s3) - 1.0 / s2
    g[:, 2] = om / ((1.0 - om) * s3)
    return g


def exp_primal_feasible(S):
    s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]
    with np.errstate(all='ignore'):
        ok = (s2 > 0) & (s3 > 0)
        val = s2 * np.log(s3 / s2) - s1
        ok &= val > 0
    return ok


def exp_dual_feasible(Z):
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
    with np.errstate(all='ignore'):
        ok = (z1 < 0) & (z3 > 0)
        val = z2 - z1 - z1 * np.log(-z3 / z1)
        ok &= val > 0
    return ok


# ----------------------------------------------------------------------------- SOC helpers
def soc_res(X):
    t = X[:, 0]
    nr = np.linalg.norm(X[:, 1:], axis=1)
    return (t - nr) * (t + nr)


def soc_jprod(U, V):
    R = np.empty_like(U)
    R[:, 0] = np.sum(U * V, axis=1)
    R[:, 1:] = U[:, :1] * V[:, 1:] + V[:, :1] * U[:, 1:]
    return R


def soc_jdiv(L, W):
    """Solve L o Y = W."""
    l0 = L[:, 0]
    det = soc_res(L)
    y0 = (l0 * W[:, 0] - np.sum(L[:, 1:] * W[:, 1:], axis=1)) / det
    Y = np.empty_like(W)
    Y[:, 0] = y0
    Y[:, 1:] = (W[:, 1:] - y0[:, None] * L[:, 1:]) / l0[:, None]
    return Y


def soc_step(X, D):
    """Largest alpha with X + alpha*D in SOC (X interior). Returns per-cone array (inf if unbounded)."""
    a = D[:, 0] ** 2 - np.sum(D[:, 1:] ** 2, axis=1)
    b = 2.0 * (X[:, 0] * D[:, 0] - np.sum(X[:, 1:] * D[:, 1:], axis=1))
    c = np.maximum(soc_res(X), 0.0)
    out = np.full(X.shape[0], np.inf)
    disc = b * b - 4 * a * c
    with np.errstate(all='ignore'):
        sq = np.sqrt(np.maximum(disc, 0.0))
        # roots of a t^2 + b t + c
        q = -0.5 * (b + np.copysign(sq, b))
        r1 = np.where(q != 0, c / q, np.inf)
        r2 = np.where(a != 0, q / a, np.inf)
    r1 = np.where(np.isfinite(r1) & (r1 > 0), r1, np.inf)
    r2 = np.where(np.isfinite(r2) & (r2 > 0), r2, np.inf)
    lin = np.where((a == 0) & (b < 0), -c / np.where(b < 0, b, -1.0), np.inf)
    cand = np.minimum(np.minimum(r1, r2), lin)
    cand = np.where(disc < 0, np.inf, cand)
    out = np.minimum(out, cand)
    # head must stay positive too
    with np.errstate(all='ignore'):
        th = np.where(D[:, 0] < 0, -X[:, 0] / D[:, 0], np.inf)
    out = np.minimum(out, th)
    return out


class Cones:
    """Holds the cone partition and per-iteration scaling state."""

    def __init__(self, cones, m):
        zero, nn, exp_rows, psd = [], [], [], []
        soc = {}
        pos = 0
        for c in cones:
            t, d = c["type"], int(c["dim"])
            if t == "zero":
                zero.append((pos, d)); pos += d
            elif t == "nonnegative":
                nn.append((pos, d)); pos += d
            elif t == "soc":
                soc.setdefault(d, []).append(pos); pos += d
            elif t == "exponential":
                exp_rows.append(pos); pos += 3
            elif t == "psd":
                N = d * (d + 1) // 2
                psd.append((pos, d)); pos += N
            else:
                raise ValueError("unknown cone " + t)
        if pos != m:
            raise ValueError("cone dims mismatch")
        self.m = m
        self.zero_rows = np.concatenate([np.arange(p, p + d) for p, d in zero]) if zero else np.zeros(0, np.int64)
        self.nn_rows = np.concatenate([np.arange(p, p + d) for p, d in nn]) if nn else np.zeros(0, np.int64)
        self.soc_groups = []
        for d in sorted(soc):
            starts = np.array(soc[d], dtype=np.int64)
            self.soc_groups.append((d, starts[:, None] + np.arange(d)[None, :]))
        self.exp_rows = (np.array(exp_rows, dtype=np.int64)[:, None] + np.arange(3)[None, :]) if exp_rows else np.zeros((0, 3), np.int64)
        self.psd = []
        self.psd_idx = {}
        for p, d in psd:
            if d not in self.psd_idx:
                self.psd_idx[d] = PSDIndex(d)
            N = d * (d + 1) // 2
            self.psd.append((d, np.arange(p, p + N)))
        self.nu = len(self.nn_rows) + sum(g[1].shape[0] for g in self.soc_groups) + 3 * self.exp_rows.shape[0] \
            + sum(d for d, _ in self.psd)
        self.has_exp = self.exp_rows.shape[0] > 0
        # block-uniform scaling groups (rows that must share one equilibration factor)
        blocks = []
        for d, R in self.soc_groups:
            blocks.extend(list(R))
        blocks.extend(list(self.exp_rows))
        for d, r in self.psd:
            blocks.append(r)
        self.uniform_blocks = blocks

    # ------------------------------------------------------------------ basic
    def identity(self, exp_central=True):
        e = np.zeros(self.m)
        e[self.nn_rows] = 1.0
        for d, R in self.soc_groups:
            e[R[:, 0]] = 1.0
        if self.exp_rows.shape[0]:
            e[self.exp_rows] = EXP_CENTRAL[None, :]
        for d, r in self.psd:
            e[r[self.psd_idx[d].diag]] = 1.0
        return e

    def uniformize(self, v):
        """Replace entries of blocks requiring uniform scaling by the block mean."""
        v = v.copy()
        for d, R in self.soc_groups:
            v[R] = np.mean(v[R], axis=1, keepdims=True)
        if self.exp_rows.shape[0]:
            R = self.exp_rows
            v[R] = np.mean(v[R], axis=1, keepdims=True)
        for d, r in self.psd:
            v[r] = np.mean(v[r])
        return v

    def min_margin(self, v):
        """Smallest 'eigenvalue' of v over symmetric cones (for initial shift)."""
        mins = [np.inf]
        if len(self.nn_rows):
            mins.append(np.min(v[self.nn_rows]))
        for d, R in self.soc_groups:
            X = v[R]
            mins.append(np.min(X[:, 0] - np.linalg.norm(X[:, 1:], axis=1)))
        for d, r in self.psd:
            mins.append(_min_eig_sym(self.psd_idx[d].smat(v[r])))
        return float(min(mins))

    def shift_init(self, s, z):
        e = self.identity()
        a = self.min_margin(s)
        if a < 1.0:
            s = s + (1.0 - a) * e
        a = self.min_margin(z)
        if a < 1.0:
            z = z + (1.0 - a) * e
        s[self.zero_rows] = 0.0
        return s, z

    # ------------------------------------------------------------------ scaling
    def update_scaling(self, s, z, mu):
        st = {}
        if len(self.nn_rows):
            sn, zn = s[self.nn_rows], z[self.nn_rows]
            st["nn_H"] = sn / zn
            st["nn_lam"] = np.sqrt(sn * zn)
        socs = []
        for d, R in self.soc_groups:
            S, Z = s[R], z[R]
            sres, zres = soc_res(S), soc_res(Z)
            sres = np.maximum(sres, 1e-300)
            zres = np.maximum(zres, 1e-300)
            Sn = S / np.sqrt(sres)[:, None]
            Zn = Z / np.sqrt(zres)[:, None]
            gamma = np.sqrt(np.maximum((1.0 + np.sum(Sn * Zn, axis=1)) / 2.0, 1e-300))
            JZ = -Zn
            JZ[:, 0] = Zn[:, 0]
            w = (Sn + JZ) / (2.0 * gamma)[:, None]
            # renormalise w so that w0^2 - |w1|^2 = 1
            w1n2 = np.sum(w[:, 1:] ** 2, axis=1)
            w[:, 0] = np.sqrt(1.0 + w1n2)
            eta = (sres / zres) ** 0.25
            lam = self._soc_W(w, eta, Z)
            H = 2.0 * w[:, :, None] * w[:, None, :]
            H[:, 0, 0] -= 1.0
            idx = np.arange(1, d)
            H[:, idx, idx] += 1.0
            H *= (eta ** 2)[:, None, None]
            socs.append(dict(w=w, eta=eta, lam=lam, H=H))
        st["soc"] = socs
        if self.exp_rows.shape[0]:
            S, Z = s[self.exp_rows], z[self.exp_rows]
            g, Hd, psi, l = exp_dual_grad_hess(Z)
            st["exp_grad"], st["exp_Hd"], st["exp_psi"], st["exp_l"] = g, Hd, psi, l
            st["exp_H"] = self._exp_pd_scaling(S, Z, g, Hd, mu)
        psds = []
        for d, r in self.psd:
            ix = self.psd_idx[d]
            Smat, Zmat = ix.smat(s[r]), ix.smat(z[r])
            Ls = np.linalg.cholesky(Smat)
            Lz = np.linalg.cholesky(Zmat)
            U, lam, Vt = np.linalg.svd(Lz.T @ Ls)
            sl = np.sqrt(lam)
            R = (Ls @ Vt.T) / sl[None, :]
            Rinv = (U.T @ Lz.T) / sl[:, None]
            psds.append(dict(R=R, Rinv=Rinv, lam=lam))
        st["psd"] = psds
        self.st = st
        return st

    def _soc_W(self, w, eta, V):
        w0 = w[:, 0]
        w1 = w[:, 1:]
        dot = np.sum(w1 * V[:, 1:], axis=1)
        out = np.empty_like(V)
        out[:, 0] = w0 * V[:, 0] + dot
        out[:, 1:] = V[:, 1:] + (V[:, 0] + dot / (1.0 + w0))[:, None] * w1
        return out * eta[:, None]

    def _soc_Winv(self, w, eta, V):
        w0 = w[:, 0]
        w1 = w[:, 1:]
        dot = np.sum(w1 * V[:, 1:], axis=1)
        out = np.empty_like(V)
        out[:, 0] = w0 * V[:, 0] - dot
        out[:, 1:] = V[:, 1:] + (-V[:, 0] + dot / (1.0 + w0))[:, None] * w1
        return out / eta[:, None]

    def _exp_pd_scaling(self, S, Z, g, Hd, mu_glob):
        k = S.shape[0]
        H = mu_glob * Hd.copy()
        try:
            st_ = -g                       # shadow primal  (in K)
            zt = -exp_primal_grad(S)       # shadow dual    (in K*)
        except Exception:
            return H
        mu = np.sum(S * Z, axis=1) / 3.0
        mut = np.sum(st_ * zt, axis=1) / 3.0
        de1 = mu * mut - 1.0
        Hz = np.einsum('kij,kj->ki', Hd, zt)
        de2 = np.sum(zt * Hz, axis=1) - 3.0 * mut * mut
        ok = (np.abs(de1) > 1.5e-8) & (np.abs(de2) > 1e-15) & (mu > 0) & np.all(np.isfinite(zt), axis=1)
        if not np.any(ok):
            return H
        H0 = mu[:, None, None] * Hd
        # S (Z^T S)^{-1} S^T
        a = 1.0 / (3.0 * de1)
        T1 = (mut[:, None, None] * S[:, :, None] * S[:, None, :]
              - (S[:, :, None] * st_[:, None, :] + st_[:, :, None] * S[:, None, :])
              + mu[:, None, None] * st_[:, :, None] * st_[:, None, :]) * a[:, None, None]
        # H0 - H0 Z (Z^T H0 Z)^{-1} Z^T H0
        Zm = np.stack([Z, zt], axis=2)  # k,3,2
        H0Z = np.einsum('kij,kjl->kil', H0, Zm)  # k,3,2
        M = np.einsum('kji,kjl->kil', Zm, H0Z)   # k,2,2
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        ok &= np.abs(det) > 1e-300
        det = np.where(ok, det, 1.0)
        Mi = np.empty_like(M)
        Mi[:, 0, 0] = M[:, 1, 1] / det
        Mi[:, 1, 1] = M[:, 0, 0] / det
        Mi[:, 0, 1] = -M[:, 0, 1] / det
        Mi[:, 1, 0] = -M[:, 1, 0] / det
        T2 = H0 - np.einsum('kil,klm,kjm->kij', H0Z, Mi, H0Z)
        Hpd = T1 + T2
        Hpd = 0.5 * (Hpd + np.transpose(Hpd, (0, 2, 1)))
        # check positive definiteness
        try:
            ev = np.linalg.eigvalsh(Hpd)
            ok &= ev[:, 0] > 0
        except Exception:
            ok[:] = False
        ok &= np.all(np.isfinite(Hpd.reshape(k, -1)), axis=1)
        H[ok] = Hpd[ok]
        return H

    # ------------------------------------------------------------------ operators
    def H_blocks(self):
        """Values of H block entries in the KKT order used by kkt.py."""
        st = self.st
        parts = [np.zeros(len(self.zero_rows))]
        if len(self.nn_rows):
            parts.append(st["nn_H"])
        for g in st["soc"]:
            parts.append(g["H"].ravel())
        if self.exp_rows.shape[0]:
            parts.append(st["exp_H"].ravel())
        return parts

    def apply_H(self, v):
        st = self.st
        out = np.zeros_like(v)
        if len(self.nn_rows):
            out[self.nn_rows] = st["nn_H"] * v[self.nn_rows]
        for (d, R), g in zip(self.soc_groups, st["soc"]):
            out[R] = np.einsum('kij,kj->ki', g["H"], v[R])
        if self.exp_rows.shape[0]:
            R = self.exp_rows
            out[R] = np.einsum('kij,kj->ki', st["exp_H"], v[R])
        for (d, r), g in zip(self.psd, st["psd"]):
            ix = self.psd_idx[d]
            G = g["R"] @ g["R"].T
            out[r] = ix.svec(G @ ix.smat(v[r]) @ G)
        return out

    def psd_H_dense(self, i):
        d, r = self.psd[i]
        g = self.st["psd"][i]
        G = g["R"] @ g["R"].T
        return self.psd_idx[d].kron_s(G)

    def psd_Hinv_dense(self, i, rows_sub=None):
        d, r = self.psd[i]
        g = self.st["psd"][i]
        Gi = g["Rinv"].T @ g["Rinv"]
        ix = self.psd_idx[d]
        if rows_sub is None:
            return ix.kron_s(Gi)
        I, J = ix.I[rows_sub], ix.J[rows_sub]
        H = Gi[np.ix_(I, I)] * Gi[np.ix_(J, J)]
        H += Gi[np.ix_(I, J)] * Gi[np.ix_(J, I)]
        fh = np.where(I == J, 1.0 / SQ2, 1.0)
        H *= fh[:, None]
        H *= fh[None, :]
        return H

    # ------------------------------------------------------------------ Newton RHS pieces
    def ds_combined(self, s, z, dsa, dza, sigma, mu):
        """d_s for the combined (predictor-corrector) step."""
        st = self.st
        out = np.zeros(self.m)
        smu = sigma * mu
        if len(self.nn_rows):
            r = self.nn_rows
            out[r] = (s[r] * z[r] + dsa[r] * dza[r] - smu) / z[r]
        for (d, R), g in zip(self.soc_groups, st["soc"]):
            w, eta, lam = g["w"], g["eta"], g["lam"]
            a1 = self._soc_Winv(w, eta, dsa[R])
            b1 = self._soc_W(w, eta, dza[R])
            T = soc_jprod(lam, lam) + soc_jprod(a1, b1)
            T[:, 0] -= smu
            Y = soc_jdiv(lam, T)
            out[R] = self._soc_W(w, eta, Y)
        if self.exp_rows.shape[0]:
            R = self.exp_rows
            Z = z[R]
            g = st["exp_grad"]
            Hd = st["exp_Hd"]
            eta = np.zeros_like(Z)
            try:
                v = np.linalg.solve(Hd, dsa[R][:, :, None])[:, :, 0]
                eta = -0.5 * exp_dual_third(Z, dza[R], v, st["exp_psi"], st["exp_l"])
                eta[~np.all(np.isfinite(eta), axis=1)] = 0.0
            except Exception:
                eta = np.zeros_like(Z)
            out[R] = s[R] + smu * g + eta
        for (d, r), g in zip(self.psd, st["psd"]):
            ix = self.psd_idx[d]
            Rm, Ri, lam = g["R"], g["Rinv"], g["lam"]
            A1 = Ri @ ix.smat(dsa[r]) @ Ri.T
            B1 = Rm.T @ ix.smat(dza[r]) @ Rm
            C = 0.5 * (A1 @ B1 + B1 @ A1)
            C[np.diag_indices(d)] += lam * lam - smu
            Y = 2.0 * C / (lam[:, None] + lam[None, :])
            out[r] = ix.svec(Rm @ Y @ Rm.T)
        return out

    # ------------------------------------------------------------------ step length
    def step_length(self, s, ds, z, dz, amax=1.0):
        a = amax
        if len(self.nn_rows):
            r = self.nn_rows
            for v, dv in ((s[r], ds[r]), (z[r], dz[r])):
                neg = dv < 0
                if np.any(neg):
                    a = min(a, float(np.min(-v[neg] / dv[neg])))
        for d, R in self.soc_groups:
            a = min(a, float(np.min(soc_step(s[R], ds[R]), initial=np.inf)))
            a = min(a, float(np.min(soc_step(z[R], dz[R]), initial=np.inf)))
        for (d, r), g in zip(self.psd, self.st["psd"]):
            ix = self.psd_idx[d]
            Rm, Ri, lam = g["R"], g["Rinv"], g["lam"]
            il = 1.0 / np.sqrt(lam)
            for M in (Ri @ ix.smat(ds[r]) @ Ri.T, Rm.T @ ix.smat(dz[r]) @ Rm):
                M = M * il[:, None] * il[None, :]
                M = 0.5 * (M + M.T)
                e = _min_eig_sym(M)
                if e < 0:
                    a = min(a, -1.0 / e)
        if self.exp_rows.shape[0]:
            R = self.exp_rows
            S, DS, Z, DZ = s[R], ds[R], z[R], dz[R]
            a_try = a
            for _ in range(80):
                if np.all(exp_primal_feasible(S + a_try * DS)) and np.all(exp_dual_feasible(Z + a_try * DZ)):
                    break
                a_try *= 0.8
            else:
                a_try = 0.0
            a = a_try
        return a

    def centrality_min(self, s, z):
        """Smallest per-cone sqrt(det(s) det(z)) over nonnegative and SOC cones."""
        v = np.inf
        if len(self.nn_rows):
            r = self.nn_rows
            v = min(v, float(np.min(s[r] * z[r])))
        for d, R in self.soc_groups:
            a = soc_res(s[R])
            b = soc_res(z[R])
            if np.any(a <= 0) or np.any(b <= 0):
                return -1.0
            v = min(v, float(np.min(np.sqrt(a) * np.sqrt(b))))
        return v

    # ------------------------------------------------------------------ membership/defects
    def interior_ok(self, s, z):
        try:
            if len(self.nn_rows) and (np.min(s[self.nn_rows]) <= 0 or np.min(z[self.nn_rows]) <= 0):
                return False
            for d, R in self.soc_groups:
                if np.min(soc_res(s[R])) <= 0 or np.min(soc_res(z[R])) <= 0:
                    return False
            if self.exp_rows.shape[0]:
                R = self.exp_rows
                if not (np.all(exp_primal_feasible(s[R])) and np.all(exp_dual_feasible(z[R]))):
                    return False
            for d, r in self.psd:
                ix = self.psd_idx[d]
                np.linalg.cholesky(ix.smat(s[r]))
                np.linalg.cholesky(ix.smat(z[r]))
        except Exception:
            return False
        return True
