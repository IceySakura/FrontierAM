"""Product-cone algebra and Nesterov--Todd scaling."""
import math
import os
import numpy as np
from scipy import sparse, linalg

SQRT2 = math.sqrt(2.0)


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def exp_barrier(v):
    a, b, c = v.T
    ell = np.log(c) - np.log(b)
    psi = b * ell - a
    p = np.stack((-np.ones_like(a), ell - 1, b / c), axis=1)
    g = -p / psi[:, None]
    g[:, 1] -= 1 / b
    g[:, 2] -= 1 / c
    H = p[:, :, None] * p[:, None, :] / psi[:, None, None] ** 2
    H[:, 1, 1] += 1 / (b * psi) + 1 / b ** 2
    H[:, 1, 2] -= 1 / (c * psi)
    H[:, 2, 1] -= 1 / (c * psi)
    H[:, 2, 2] += b / (c ** 2 * psi) + 1 / c ** 2
    return g, H


def exp_barrier_inverse(v):
    a, b, c = v.T
    ell = np.log(c) - np.log(b)
    psi = b * ell - a
    den = 2 * b + psi
    B = np.empty((len(a), 2, 2))
    B[:, 0, 0] = b*b * (b+psi) / den
    B[:, 0, 1] = B[:, 1, 0] = b*b*c / den
    B[:, 1, 1] = c*c * (b+psi) / den
    p = np.stack((ell-1, b/c), axis=1)
    t = np.einsum("kij,kj->ki", B, p)
    H = np.empty((len(a), 3, 3))
    H[:, 0, 0] = psi**2 + np.sum(p*t, axis=1)
    H[:, 0, 1:] = t
    H[:, 1:, 0] = t
    H[:, 1:, 1:] = B
    return H


def exp_centrality(s, z, mu):
    if not len(s):
        return 0.
    v = z @ EXP_T.T
    a, b, c = v.T
    if np.any(b <= 0) or np.any(c <= 0) or mu <= 0:
        return float("inf")
    ell = np.log(c) - np.log(b)
    psi = b * ell - a
    if np.any(psi <= 0):
        return float("inf")
    p = np.stack((-np.ones_like(a), ell-1, b/c), axis=1)
    g = -p / psi[:, None]
    g[:, 1] -= 1/b
    g[:, 2] -= 1/c
    r = (s + mu * (g @ EXP_T)) @ EXP_TI
    hi = exp_barrier_inverse(v)
    delta = np.einsum("ki,kij,kj->k", r, hi, r) / mu**2
    return float(np.max(delta, initial=0.))


def exp_inside(v):
    a, b, c = v.T
    if np.any(b <= 0) or np.any(c <= 0):
        return False
    return bool(np.all(b * (np.log(c) - np.log(b)) - a > 0))


EXP_T = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., math.e]])
EXP_TI = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 1/math.e]])


def q_apply(w, a):
    r = 2 * w * np.sum(w * a, axis=1)[:, None] + a
    r[:, 0] -= 2 * a[:, 0]
    return r


def jordan(a, b):
    r = a[:, :1] * b + b[:, :1] * a
    r[:, 0] = np.sum(a * b, axis=1)
    return r


def inv_jordan(a, b):
    det = a[:, 0] ** 2 - np.sum(a[:, 1:] ** 2, axis=1)
    t = (a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)) / det
    r = (b - a * t[:, None]) / a[:, :1]
    r[:, 0] = t
    return r


class ConeProduct:
    def __init__(self, cones):
        self.spec = cones
        self.exp_mode = os.environ.get("SOLVER_EXP_SCALING", "bfgs")
        self.blocks = []
        self.soc = {}
        self.psd = []
        exp = []
        zeros, nonneg = [], []
        nn_lengths = []
        encodings = {}
        self.degree = 0
        start = 0
        for c in cones:
            k, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if k == "psd" else d
            sl = slice(start, start + w)
            self.blocks.append((k, d, sl))
            if k == "zero":
                zeros.append(sl)
            elif k == "nonnegative":
                nonneg.append(sl)
                nn_lengths.append(w)
                self.degree += d
            elif k == "soc":
                self.soc.setdefault(d, []).append(np.arange(start, start + w))
                self.degree += 1
            elif k == "exponential":
                exp.append(np.arange(start, start + 3))
                self.degree += 3
            else:
                if d not in encodings:
                    ii, jj = np.triu_indices(d)
                    order = np.argsort(jj * d + ii)
                    ii, jj = ii[order], jj[order]
                    weight = np.where(ii == jj, 1., SQRT2)
                    encodings[d] = ii, jj, weight
                ii, jj, weight = encodings[d]
                self.psd.append((sl, d, ii, jj, weight))
                self.degree += d
            start += w
        self.m = start
        self.zero = np.zeros(start, dtype=bool)
        self.nonnegative = np.zeros(start, dtype=bool)
        for sl in zeros:
            self.zero[sl] = True
        for sl in nonneg:
            self.nonnegative[sl] = True
        self.zi = np.flatnonzero(self.zero)
        self.ni = np.flatnonzero(self.nonnegative)
        self.nn_starts = np.r_[0, np.cumsum(nn_lengths)[:-1]].astype(int) if nn_lengths else np.empty(0, int)
        self.ei = np.asarray(exp, dtype=int).reshape(-1, 3)
        self.soc = {d: np.asarray(v) for d, v in self.soc.items()}
        self.small_psd = bool(self.psd) and max(p[1] for p in self.psd) <= 12
        if os.environ.get("SOLVER_PSD_KKT") == "reduced":
            self.small_psd = False

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.ni] = z[self.ni] = 1.
        for ids in self.soc.values():
            s[ids[:, 0]] = z[ids[:, 0]] = 1.
        if self.ei.size:
            if self.exp_mode == "primal":
                s[self.ei] = np.array([-1., 1., 1.])
                g, _ = exp_barrier(s[self.ei])
                z[self.ei] = -g
            else:
                z[self.ei] = np.array([-1., 0., 1.])
                g, _ = exp_barrier(z[self.ei] @ EXP_T.T)
                s[self.ei] = -(g @ EXP_T)
        for sl, d, ii, jj, weight in self.psd:
            v = np.zeros(ii.size)
            v[ii == jj] = 1.
            s[sl] = z[sl] = v
        return s, z

    def mat(self, v, block):
        sl, d, ii, jj, weight = block
        M = np.zeros((d, d))
        M[ii, jj] = v[sl] / weight
        M[jj, ii] = M[ii, jj]
        return M

    def vec(self, M, block):
        _, _, ii, jj, weight = block
        return M[ii, jj] * weight

    def violation(self, v, dual=False):
        worst = 0.
        if not dual and self.zi.size:
            a = norm(v[self.zi])
            worst = a / max(1., a)
        if self.ni.size:
            a = v[self.ni]
            size = np.maximum(1., np.maximum.reduceat(np.abs(a), self.nn_starts))
            err = -np.minimum.reduceat(a, self.nn_starts) / size
            worst = max(worst, float(np.max(err, initial=0.)))
        for ids in self.soc.values():
            a = v[ids]
            a = a / np.maximum(1., np.max(np.abs(a), axis=1))[:, None]
            err = np.linalg.norm(a[:, 1:], axis=1) - a[:, 0]
            worst = max(worst, float(np.max(err, initial=0.)))
        if self.ei.size:
            a = v[self.ei]
            a = a / np.maximum(1., np.max(np.abs(a), axis=1))[:, None]
            if dual:
                a = a @ EXP_T.T
            x, y, z = a.T
            err = np.maximum.reduce((np.abs(y), np.maximum(x, 0.), np.maximum(-z, 0.)))
            mask = (y > 0) & (z > 0)
            interior = np.maximum(0., x[mask] - y[mask]*(np.log(z[mask])-np.log(y[mask])))
            err[mask] = np.minimum(err[mask], interior)
            worst = max(worst, float(np.max(err, initial=0.)))
        for block in self.psd:
            sl, d, ii, jj, weight = block
            a = v[sl]
            a = a / max(1., norm(a))
            if not np.any(a[ii != jj]):
                ev = np.min(a[ii == jj])
            else:
                M = np.zeros((d, d))
                M[ii, jj] = a / weight
                M[jj, ii] = M[ii, jj]
                lower = np.diag(M) - (np.sum(np.abs(M), axis=1) - np.abs(np.diag(M)))
                if np.min(lower) >= 0:
                    ev = 0.
                else:
                    ev = linalg.eigh(M, subset_by_index=[0, 0], eigvals_only=True,
                                     check_finite=False, driver="evr")[0]
            err = max(0., -float(ev))
            worst = max(worst, err)
        return worst

    def row_scales(self, scales):
        for ids in self.soc.values():
            scales[ids] = np.min(scales[ids], axis=1)[:, None]
        if self.ei.size:
            scales[self.ei] = np.min(scales[self.ei], axis=1)[:, None]
        for sl, *_ in self.psd:
            scales[sl] = np.min(scales[sl])
        return scales

    def scaling(self, s, z, mu):
        return Scaling(self, s, z, mu)


class Scaling:
    def __init__(self, K, s, z, mu):
        self.K, self.s, self.z, self.mu = K, s, z, mu
        self.soc = []
        self.psd = []
        self.rows, self.cols, self.vals = [], [], []
        self.irows, self.icols, self.ivals = [], [], []
        self.ur, self.uc, self.uv = [], [], []
        aux = 0
        ni = K.ni
        self.add(ni, ni, s[ni] / z[ni], z[ni] / s[ni])
        for d, ids in K.soc.items():
            ss, zz = s[ids], z[ids]
            ns = np.linalg.norm(ss[:, 1:], axis=1)
            nz = np.linalg.norm(zz[:, 1:], axis=1)
            ds = np.sqrt((ss[:, 0] - ns) * (ss[:, 0] + ns))
            dz = np.sqrt((zz[:, 0] - nz) * (zz[:, 0] + nz))
            beta = np.sqrt(ds / dz)
            w = zz / dz[:, None]
            w[:, 1:] *= -1
            w += ss / ds[:, None]
            w /= np.sqrt(2 * (1 + np.sum(ss * zz, axis=1) / (ds * dz)))[:, None]
            # Renormalization controls cancellation close to a cone boundary.
            w[:, 0] = np.sqrt(1 + np.sum(w[:, 1:] ** 2, axis=1))
            v = w.copy()
            v[:, 0] = np.sqrt((w[:, 0] + 1) / 2)
            v[:, 1:] /= 2 * v[:, :1]
            jv = v.copy()
            jv[:, 1:] *= -1
            lam = beta[:, None] * q_apply(v, zz)
            self.soc.append((ids, beta, w, v, jv, lam))
            if d <= 48:
                H = 2 * w[:, :, None] * w[:, None, :]
                diag = np.ones(d)
                diag[0] = -1.
                H += np.diag(diag)[None, :, :]
                H *= beta[:, None, None] ** 2
                jw = w.copy()
                jw[:, 1:] *= -1
                Hi = 2 * jw[:, :, None] * jw[:, None, :]
                Hi += np.diag(diag)[None, :, :]
                Hi /= beta[:, None, None] ** 2
                self.addblock(ids, H, Hi)
            else:
                diag = np.ones_like(w)
                diag[:, 0] = -1
                hd = beta[:, None] ** 2 * diag
                self.add(ids.ravel(), ids.ravel(), hd.ravel(), None)
                self.ur.append(ids.ravel())
                self.uc.append(np.repeat(np.arange(aux, aux + ids.shape[0]), d))
                self.uv.append((SQRT2 * beta[:, None] * w).ravel())
                aux += ids.shape[0]
                # The inverse is used only in the reduced SDP formulation.
        self.naux = aux
        if K.ei.size:
            zz, ss = z[K.ei], s[K.ei]
            if K.exp_mode == "primal":
                g, H = exp_barrier(ss)
                self.eg = g
                self.eh = exp_barrier_inverse(ss) / mu
                self.ehi = mu * H
            else:
                transformed = zz @ EXP_T.T
                g, H = exp_barrier(transformed)
                self.eg = g @ EXP_T
                self.eh = mu * (EXP_T.T @ H @ EXP_T)
                self.ehi = (EXP_TI @ exp_barrier_inverse(transformed) @ EXP_TI.T) / mu
            if K.exp_mode == "bfgs":
                sz = np.sum(ss * zz, axis=1)
                hz = -mu * self.eg
                zh = np.sum(zz * hz, axis=1)
                self.eh -= hz[:, :, None] * hz[:, None, :] / zh[:, None, None]
                self.eh += ss[:, :, None] * ss[:, None, :] / sz[:, None, None]
                V = np.eye(3)[None, :, :] - (
                    zz[:, :, None] * ss[:, None, :] / sz[:, None, None])
                self.ehi = V @ self.ehi @ V.transpose(0, 2, 1)
                self.ehi += zz[:, :, None] * zz[:, None, :] / sz[:, None, None]
            self.addblock(K.ei, self.eh, self.ehi)
        for block in K.psd:
            S, Z = K.mat(s, block), K.mat(z, block)
            L = linalg.cholesky(S, lower=True, check_finite=False)
            ev, U = linalg.eigh(L.T @ Z @ L, check_finite=False, driver="evd")
            if ev[0] <= 0:
                raise np.linalg.LinAlgError("Nonpositive cone scaling")
            lam = np.sqrt(ev)
            R = (L @ U) / np.sqrt(lam)[None, :]
            Ri = linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True,
                                         check_finite=False)
            Ri = np.sqrt(lam)[:, None] * (U.T @ Ri)
            Wi = Ri.T @ Ri
            self.psd.append((block, R, Ri, lam, Wi))
            if K.small_psd:
                sl, d, ii, jj, weight = block
                W = R @ R.T
                factor = weight[:, None] / weight[None, :]
                factor[:, ii == jj] *= .5
                H = (W[ii[:, None], ii] * W[jj[:, None], jj] +
                     W[ii[:, None], jj] * W[jj[:, None], ii]) * factor
                Hi = (Wi[ii[:, None], ii] * Wi[jj[:, None], jj] +
                      Wi[ii[:, None], jj] * Wi[jj[:, None], ii]) * factor
                ids = np.arange(sl.start, sl.stop)[None, :]
                self.addblock(ids, H[None, :, :], Hi[None, :, :])

    def add(self, i, j, h, hi):
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(h)
        if hi is not None:
            self.irows.append(i)
            self.icols.append(j)
            self.ivals.append(hi)

    def addblock(self, ids, H, Hi):
        d = ids.shape[1]
        r = np.broadcast_to(ids[:, :, None], H.shape).ravel()
        c = np.broadcast_to(ids[:, None, :], H.shape).ravel()
        self.add(r, c, H.ravel(), Hi.ravel())

    def matrix(self, inverse=False):
        if inverse:
            rows, cols, vals = self.irows, self.icols, self.ivals
        else:
            rows, cols, vals = self.rows, self.cols, self.vals
        if not rows:
            return sparse.csc_matrix((self.K.m, self.K.m))
        return sparse.csc_matrix((np.concatenate(vals),
                                 (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.K.m, self.K.m))

    def auxiliaries(self):
        if not self.naux:
            return None
        return sparse.csc_matrix((np.concatenate(self.uv),
                                  (np.concatenate(self.ur), np.concatenate(self.uc))),
                                  shape=(self.K.m, self.naux))

    def apply_inverse(self, v):
        out = np.zeros_like(v)
        ni = self.K.ni
        out[ni] = v[ni] * self.z[ni] / self.s[ni]
        for ids, beta, w, _, _, _ in self.soc:
            jw = w.copy()
            jw[:, 1:] *= -1
            out[ids] = q_apply(jw, v[ids]) / beta[:, None] ** 2
        if self.K.ei.size:
            out[self.K.ei] = np.einsum("kij,kj->ki", self.ehi, v[self.K.ei])
        for block, R, Ri, lam, Wi in self.psd:
            out[block[0]] = self.K.vec(Wi @ self.K.mat(v, block) @ Wi, block)
        return out

    def rhs(self, sigma, ds=None, dz=None):
        K, s, z, mu = self.K, self.s, self.z, self.mu
        rc = np.zeros(K.m)
        ni = K.ni
        rc[ni] = -s[ni] + sigma * mu / z[ni]
        if ds is not None:
            rc[ni] -= ds[ni] * dz[ni] / z[ni]
        for ids, beta, w, v, jv, lam in self.soc:
            term = np.zeros_like(lam)
            term[:, 0] = sigma * mu
            if ds is not None:
                sh = q_apply(jv, ds[ids]) / beta[:, None]
                zh = q_apply(v, dz[ids]) * beta[:, None]
                term -= jordan(sh, zh)
            term = inv_jordan(lam, term)
            rc[ids] = -s[ids] + beta[:, None] * q_apply(v, term)
        if K.ei.size:
            if K.exp_mode == "primal":
                rc[K.ei] = sigma*s[K.ei] - np.einsum("kij,kj->ki", self.eh, z[K.ei])
            else:
                rc[K.ei] = -s[K.ei] - sigma * mu * self.eg
        for block, R, Ri, lam, Wi in self.psd:
            term = np.diag(-lam + sigma * mu / lam)
            if ds is not None:
                sh = Ri @ K.mat(ds, block) @ Ri.T
                zh = R.T @ K.mat(dz, block) @ R
                prod = sh @ zh
                term -= (prod + prod.T) / (lam[:, None] + lam[None, :])
            rc[block[0]] = K.vec(R @ term @ R.T, block)
        return rc

    def step(self, ds, dz):
        K = self.K
        alpha = 1.
        ni = K.ni
        for v, dv in ((self.s[ni], ds[ni]), (self.z[ni], dz[ni])):
            bad = dv < 0
            if np.any(bad):
                alpha = min(alpha, float(np.min(-v[bad] / dv[bad])))
        for ids, beta, w, v, jv, lam in self.soc:
            for u, du in ((self.s[ids], ds[ids]), (self.z[ids], dz[ids])):
                a = du[:, 0] ** 2 - np.sum(du[:, 1:] ** 2, axis=1)
                b = 2 * (u[:, 0] * du[:, 0] - np.sum(u[:, 1:] * du[:, 1:], axis=1))
                nr = np.linalg.norm(u[:, 1:], axis=1)
                c = (u[:, 0] - nr) * (u[:, 0] + nr)
                det = b * b - 4 * a * c
                den = -b + np.sqrt(np.maximum(det, 0))
                bad = (den > 0) & (det >= 0)
                if np.any(bad):
                    roots = 2 * c[bad] / den[bad]
                    alpha = min(alpha, float(np.min(roots)))
                bad = du[:, 0] < 0
                if np.any(bad):
                    alpha = min(alpha, float(np.min(-u[bad, 0] / du[bad, 0])))
        for block, R, Ri, lam, Wi in self.psd:
            invlam = 1 / np.sqrt(lam)
            for sh in (Ri @ K.mat(ds, block) @ Ri.T, R.T @ K.mat(dz, block) @ R):
                sh *= invlam[:, None] * invlam[None, :]
                ev = linalg.eigh(sh, subset_by_index=[0, 0], eigvals_only=True,
                                 check_finite=False, driver="evr")[0]
                if ev < 0:
                    alpha = min(alpha, -1 / ev)
        if K.ei.size:
            alpha = max(0., alpha)
            lo, hi = 0., alpha
            for v, dv, dual in ((self.s[K.ei], ds[K.ei], False),
                                (self.z[K.ei], dz[K.ei], True)):
                vv, dd = (v @ EXP_T.T, dv @ EXP_T.T) if dual else (v, dv)
                if not exp_inside(vv + alpha * dd):
                    lo, hi = 0., alpha
                    for _ in range(35):
                        mid = (lo + hi) * 0.5
                        if exp_inside(vv + mid * dd):
                            lo = mid
                        else:
                            hi = mid
                    alpha = lo
        return max(0., min(1., alpha))
