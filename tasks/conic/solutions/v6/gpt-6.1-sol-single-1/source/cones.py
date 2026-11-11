"""Product-cone algebra and Nesterov-Todd predictor-corrector scaling."""

import math
import numpy as np
from scipy import sparse, linalg
from common import norm, smat, svec


def jordan(a, b):
    out = a * b
    out[:, 0] = np.sum(a * b, axis=1)
    out[:, 1:] = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:]
    return out


def jsolve(a, b):
    dot = np.sum(a[:, 1:] * b[:, 1:], axis=1)
    det = a[:, 0] ** 2 - np.sum(a[:, 1:] ** 2, axis=1)
    t = (a[:, 0] * b[:, 0] - dot) / det
    return np.column_stack((t, (b[:, 1:] - a[:, 1:] * t[:, None]) / a[:, :1]))


def exp_barrier(s):
    x, y, z = s.T
    ell = np.log(z / y)
    t = y * ell - x
    a = np.column_stack((-np.ones(len(s)), ell - 1, y / z))
    g = -a / t[:, None]
    g[:, 1] -= 1 / y
    g[:, 2] -= 1 / z
    H = a[:, :, None] * a[:, None, :] / t[:, None, None] ** 2
    H[:, 1, 1] += 1 / (t * y) + 1 / y ** 2
    H[:, 1, 2] -= 1 / (t * z)
    H[:, 2, 1] -= 1 / (t * z)
    H[:, 2, 2] += y / (t * z ** 2) + 1 / z ** 2
    return g, H


def exp_inverse(s):
    x, y, z = s.T
    t = y * np.log(z / y) - x
    a = np.column_stack((np.log(z / y) - 1, y / z))
    r, u = (t + y) / (t + 2 * y), y / (t + 2 * y)
    G = np.empty((len(s), 2, 2))
    G[:, 0, 0], G[:, 1, 1] = y * y * r, z * z * r
    G[:, 0, 1] = G[:, 1, 0] = y * z * u
    v = np.einsum("bij,bj->bi", G, a)
    out = np.empty((len(s), 3, 3))
    out[:, 1:, 1:] = G
    out[:, 0, 1:] = out[:, 1:, 0] = v
    out[:, 0, 0] = t * t + np.sum(a * v, axis=1)
    return out


def exp_sqrt(s, mu):
    x, y, z = s.T
    ell = np.log(z / y)
    t = y * ell - x
    R = np.zeros((len(s), 3, 3))
    R[:, 0, 0] = 1 / t
    R[:, 0, 1] = -(ell - 1) / t
    R[:, 0, 2] = -y / (z * t)
    R[:, 1, 1] = np.sqrt((t + y) / t) / y
    R[:, 1, 2] = -y / (z * np.sqrt(t * (t + y)))
    R[:, 2, 2] = np.sqrt((t + 2 * y) / (t + y)) / z
    R *= np.sqrt(mu)
    Rs = np.sqrt(mu) * np.column_stack((-np.ones(len(s)), np.sqrt(t / (t + y)),
                                       np.sqrt((t + 2 * y) / (t + y))))
    return R, Rs


class ProductCone:
    def __init__(self, p):
        self.p = p
        nn, soc, exp, psd = [], {}, [], []
        for kind, d, i, j in p.blocks:
            if kind == "nonnegative":
                nn.extend(range(i, j))
            elif kind == "soc":
                soc.setdefault(d, []).append(i)
            elif kind == "exponential":
                exp.append(i)
            elif kind == "psd":
                r, c = np.triu_indices(d)
                order = np.lexsort((r, c))
                r, c = r[order], c[order]
                Ab = p.A[i:j].tocsc()
                active = np.unique(Ab.indices)
                psd.append({"d": d, "i": i, "j": j, "r": r, "c": c,
                            "active": active, "Aa": Ab[active].tocsr()})
        self.nn = np.array(nn, dtype=int)
        self.soc = {d: np.asarray(starts)[:, None] + np.arange(d) for d, starts in soc.items()}
        self.exp = np.asarray(exp, dtype=int)[:, None] + np.arange(3)
        self.psd = psd
        self.degree = len(nn) + sum(len(v) for v in self.soc.values()) + 3 * len(exp) + sum(d["d"] for d in psd)
        rows = [self.nn]
        cols = [self.nn]
        for idx in list(self.soc.values()) + ([self.exp] if len(exp) else []):
            d = idx.shape[1]
            rows.append(np.repeat(idx, d, axis=1).ravel())
            cols.append(np.tile(idx, (1, d)).ravel())
        self.hrows = np.concatenate(rows)
        self.hcols = np.concatenate(cols)
        self.dense = bool(psd)

    def initial(self):
        s, z = np.zeros(self.p.m), np.zeros(self.p.m)
        s[self.nn] = z[self.nn] = 1
        for idx in self.soc.values():
            s[idx[:, 0]] = z[idx[:, 0]] = 1
        if len(self.exp):
            s[self.exp] = [-1.0, 1.0, 1.0]
            z[self.exp] = [-1.0, 0.0, 1.0]
        for b in self.psd:
            v = svec(np.eye(b["d"]))
            s[b["i"]:b["j"]] = z[b["i"]:b["j"]] = v
        return s, z

    def scaling(self, s, z, mu):
        values = [z[self.nn] / s[self.nn]]
        inverses = [s[self.nn] / z[self.nn]]
        scale_values = [np.sqrt(z[self.nn] / s[self.nn])]
        scale_inverses = [np.sqrt(s[self.nn] / z[self.nn])]
        self.sdata = {}
        for d, idx in self.soc.items():
            a, b = s[idx], z[idx]
            na = np.linalg.norm(a[:, 1:], axis=1)
            nb = np.linalg.norm(b[:, 1:], axis=1)
            aa = np.sqrt(np.maximum((a[:, 0] - na) * (a[:, 0] + na), 1e-300))
            bb = np.sqrt(np.maximum((b[:, 0] - nb) * (b[:, 0] + nb), 1e-300))
            beta = np.sqrt(aa / bb)
            jb = b.copy()
            jb[:, 1:] *= -1
            w = (a / aa[:, None] + jb / bb[:, None])
            w /= np.sqrt(2 * (1 + np.sum(a * b, axis=1) / (aa * bb)))[:, None]
            w[:, 0] = np.sqrt(1 + np.sum(w[:, 1:] ** 2, axis=1))
            v = w.copy()
            v[:, 0] = np.sqrt((w[:, 0] + 1) / 2)
            v[:, 1:] /= 2 * v[:, :1]
            J = np.diag(np.r_[1.0, -np.ones(d - 1)])
            W = beta[:, None, None] * (2 * v[:, :, None] * v[:, None, :] - J)
            v[:, 1:] *= -1
            Wi = (2 * v[:, :, None] * v[:, None, :] - J) / beta[:, None, None]
            w[:, 1:] *= -1
            H = (2 * w[:, :, None] * w[:, None, :] - J) / beta[:, None, None] ** 2
            lam = np.einsum("bij,bj->bi", W, b)
            self.sdata[d] = (W, Wi, lam)
            values.append(H.ravel())
            inverses.append(np.matmul(W, W).ravel())
            scale_values.append(Wi.ravel())
            scale_inverses.append(W.ravel())
        if len(self.exp):
            self.eg, self.eH = exp_barrier(s[self.exp])
            values.append((mu * self.eH).ravel())
            inverses.append((exp_inverse(s[self.exp]) / mu).ravel())
            self.eR, self.eRs = exp_sqrt(s[self.exp], mu)
            scale_values.append(self.eR.ravel())
            scale_inverses.append(np.linalg.inv(self.eR).transpose(0, 2, 1).ravel())
        data = np.concatenate(values)
        self.H = sparse.coo_matrix((data, (self.hrows, self.hcols)),
                                  shape=(self.p.m, self.p.m)).tocsc()
        self.Hinv = sparse.coo_matrix((np.concatenate(inverses), (self.hrows, self.hcols)),
                                     shape=(self.p.m, self.p.m)).tocsc()
        eq = np.flatnonzero(self.p.eq)
        rows, cols = np.r_[self.hrows, eq], np.r_[self.hcols, eq]
        self.T = sparse.coo_matrix((np.r_[np.concatenate(scale_values), np.ones(len(eq))], (rows, cols)),
                                  shape=(self.p.m, self.p.m)).tocsc()
        self.TinvT = sparse.coo_matrix((np.r_[np.concatenate(scale_inverses), np.ones(len(eq))], (rows, cols)),
                                      shape=(self.p.m, self.p.m)).tocsc()
        for b in self.psd:
            S, Z = smat(s[b["i"]:b["j"]], b["d"]), smat(z[b["i"]:b["j"]], b["d"])
            L = linalg.cholesky(S, lower=True, check_finite=False)
            Lz = linalg.cholesky(Z, lower=True, check_finite=False)
            V, singular, _ = linalg.svd(L.T @ Lz, check_finite=False, lapack_driver="gesvd")
            singular = np.maximum(singular, 1e-300)
            W = (L @ V) / np.sqrt(singular)[None, :]
            Li = linalg.solve_triangular(L, np.eye(b["d"]), lower=True, check_finite=False)
            Wi = (np.sqrt(singular)[:, None] * V.T) @ Li
            b.update(W=W, Wi=Wi, lam=singular, M=Wi.T @ Wi)

    def apply(self, v):
        out = self.H @ v
        for b in self.psd:
            out[b["i"]:b["j"]] = svec(b["M"] @ smat(v[b["i"]:b["j"]], b["d"]) @ b["M"])
        return out

    def schur(self):
        out = self.p.P + self.p.A.T @ self.H @ self.p.A
        if self.psd:
            out = out.toarray()
        for b in self.psd:
            ids, Aa, M = b["active"], b["Aa"], b["M"]
            r, c = b["r"][ids], b["c"][ids]
            f = np.where(r == c, 1 / math.sqrt(2), 1.0)
            k = len(ids)
            T = np.empty((k, self.p.n))
            for lo in range(0, k, 256):
                hi = min(lo + 256, k)
                H = (M[r[lo:hi, None], r[None, :]] * M[c[lo:hi, None], c[None, :]]
                     + M[r[lo:hi, None], c[None, :]] * M[c[lo:hi, None], r[None, :]])
                H *= f[lo:hi, None] * f[None, :]
                T[lo:hi] = (Aa.T @ H.T).T
            out += Aa.T @ T
        return out

    def rhs(self, s, z, mu, sigma, ds=None, dz=None):
        h = -z.copy()
        h[self.p.eq] = 0
        nn = self.nn
        h[nn] += sigma * mu / s[nn]
        if ds is not None:
            h[nn] -= ds[nn] * dz[nn] / s[nn]
        for d, idx in self.soc.items():
            W, Wi, lam = self.sdata[d]
            e = np.zeros_like(lam)
            e[:, 0] = sigma * mu
            if ds is not None:
                a = np.einsum("bij,bj->bi", Wi, ds[idx])
                b = np.einsum("bij,bj->bi", W, dz[idx])
                e -= jordan(a, b)
            h[idx] += np.einsum("bij,bj->bi", Wi, jsolve(lam, e))
        if len(self.exp):
            h[self.exp] -= sigma * mu * self.eg
        for b in self.psd:
            lam, W, Wi = b["lam"], b["W"], b["Wi"]
            B = np.diag(sigma * mu / lam)
            if ds is not None:
                a = Wi @ smat(ds[b["i"]:b["j"]], b["d"]) @ Wi.T
                zz = W.T @ smat(dz[b["i"]:b["j"]], b["d"]) @ W
                prod = a @ zz
                B -= (prod + prod.T) / (lam[:, None] + lam[None, :])
            h[b["i"]:b["j"]] += svec(Wi.T @ B @ Wi)
        return h

    def rhs_scaled(self, s, z, mu, sigma, ds=None, dz=None):
        out = np.zeros(self.p.m)
        nn = self.nn
        lam = np.sqrt(s[nn] * z[nn])
        out[nn] = -lam + sigma * mu / lam
        if ds is not None:
            out[nn] -= ds[nn] * dz[nn] / lam
        for d, idx in self.soc.items():
            W, Wi, lam = self.sdata[d]
            e = np.zeros_like(lam)
            e[:, 0] = sigma * mu
            if ds is not None:
                a = np.einsum("bij,bj->bi", Wi, ds[idx])
                b = np.einsum("bij,bj->bi", W, dz[idx])
                e -= jordan(a, b)
            out[idx] = -lam + jsolve(lam, e)
        if len(self.exp):
            invtz = np.linalg.solve(self.eR.transpose(0, 2, 1), z[self.exp][..., None]).squeeze(-1)
            out[self.exp] = -invtz + sigma * self.eRs
        return out

    def interior(self, v, dual=False):
        if len(self.nn) and np.min(v[self.nn]) <= 0:
            return False
        for idx in self.soc.values():
            t = v[idx]
            if np.any(t[:, 0] <= np.linalg.norm(t[:, 1:], axis=1)):
                return False
        if len(self.exp):
            a = v[self.exp]
            if dual:
                a = np.column_stack((-a[:, 1], -a[:, 0], math.e * a[:, 2]))
            x, y, z = a.T
            if np.any(y <= 0) or np.any(z <= 0):
                return False
            if np.any(y * (np.log(z) - np.log(y)) <= x):
                return False
        for b in self.psd:
            try:
                linalg.cholesky(smat(v[b["i"]:b["j"]], b["d"]), check_finite=False)
            except linalg.LinAlgError:
                return False
        return True

    def max_step(self, v, dv, dual=False):
        alpha = 1.0
        ids = self.nn
        if len(ids):
            neg = dv[ids] < 0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-v[ids][neg] / dv[ids][neg])))
        for idx in self.soc.values():
            a, b = v[idx], dv[idx]
            aa = b[:, 0] ** 2 - np.sum(b[:, 1:] ** 2, axis=1)
            bb = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
            cc = (a[:, 0] - np.linalg.norm(a[:, 1:], axis=1)) * (a[:, 0] + np.linalg.norm(a[:, 1:], axis=1))
            disc = bb ** 2 - aa * cc
            valid = disc > 0
            root = np.sqrt(np.maximum(disc, 0))
            den = -bb + root
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(den > 0, cc / den, (-bb - root) / aa)
            valid &= r > 0
            if np.any(valid):
                alpha = min(alpha, float(np.min(r[valid])))
            neg = b[:, 0] < 0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-a[neg, 0] / b[neg, 0])))
        for b in self.psd:
            S, dS = smat(v[b["i"]:b["j"]], b["d"]), smat(dv[b["i"]:b["j"]], b["d"])
            L = linalg.cholesky(S, lower=True, check_finite=False)
            T = linalg.solve_triangular(L, dS, lower=True, check_finite=False)
            T = linalg.solve_triangular(L, T.T, lower=True, check_finite=False).T
            low = linalg.eigh(T, eigvals_only=True, subset_by_index=(0, 0), check_finite=False)[0]
            if low < 0:
                alpha = min(alpha, -1 / low)
        if len(self.exp):
            # Exponential boundary roots have no useful closed-form expression.
            lo, hi = 0.0, alpha
            for _ in range(30):
                t = (lo + hi) / 2
                a = (v + t * dv)[self.exp]
                if dual:
                    a = np.column_stack((-a[:, 1], -a[:, 0], math.e * a[:, 2]))
                x, y, z = a.T
                ok = (y > 0) & (z > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ok &= y * (np.log(z) - np.log(y)) > x
                if np.all(ok):
                    lo = t
                else:
                    hi = t
            alpha = lo
        return max(0.0, min(1.0, alpha))
