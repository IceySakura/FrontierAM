"""Cone geometry and Nesterov--Todd scalings in isometric coordinates."""
import ctypes
import math
import os

import numpy as np
from scipy import linalg, sparse


def mv(a, x):
    return np.einsum("...ij,...j->...i", a, x)


def jordan(a, b):
    r = a[:, :1] * b + b[:, :1] * a
    r[:, 0] = np.sum(a * b, axis=1)
    return r


def linv(a, r):
    det = a[:, 0] ** 2 - np.sum(a[:, 1:] ** 2, axis=1)
    t = (a[:, 0] * r[:, 0] - np.sum(a[:, 1:] * r[:, 1:], axis=1)) / det
    v = (r[:, 1:] - a[:, 1:] * t[:, None]) / a[:, :1]
    return np.column_stack((t, v))


def qrep(w, normalized=False):
    out = 2.0 * w[:, :, None] * w[:, None, :]
    det = np.ones(len(w), dtype=w.dtype) if normalized else (
        w[:, 0] ** 2 - np.sum(w[:, 1:] ** 2, axis=1))
    for i in range(w.shape[1]):
        out[:, i, i] += (-1.0 if i == 0 else 1.0) * det
    return out


def exp_barrier(z):
    # F*(z) = F(-z_1,-z_0,e*z_2) up to an irrelevant constant.
    v = -z[:, 0]
    w = math.e * z[:, 2]
    ell = np.log(w / v)
    a = v * ell + z[:, 1]
    grad_a = np.column_stack((-np.ones_like(v), ell - 1.0, v / w))
    g = -grad_a / a[:, None]
    g[:, 1] -= 1.0 / v
    g[:, 2] -= 1.0 / w
    h = grad_a[:, :, None] * grad_a[:, None, :] / a[:, None, None] ** 2
    h[:, 1, 1] += 1.0 / (a * v) + 1.0 / v ** 2
    h[:, 1, 2] -= 1.0 / (a * w)
    h[:, 2, 1] -= 1.0 / (a * w)
    h[:, 2, 2] += v / (a * w ** 2) + 1.0 / w ** 2
    perm = [1, 0, 2]
    signs = np.array([-1.0, -1.0, math.e])
    return g[:, perm] * signs, h[:, perm, :][:, :, perm] * signs[None, :, None] * signs[None, None, :]


class PSD:
    def __init__(self, start, order):
        self.start = start
        self.d = order
        self.width = order * (order + 1) // 2
        self.robust=False
        self.sl = slice(start, start + self.width)
        self.c, self.r = np.tril_indices(order)
        self.factor = np.where(self.r == self.c, 1.0, math.sqrt(2.0))

    def mat(self, x):
        a = np.zeros((self.d, self.d))
        v = x / self.factor
        a[self.r, self.c] = v
        a[self.c, self.r] = v
        return a

    def vec(self, x):
        return x[self.r, self.c] * self.factor

    def update(self, s, z):
        self.s = self.mat(s)
        self.z = self.mat(z)
        L = self.cholesky(self.s)
        e, u = linalg.eigh(L.T @ self.z @ L, check_finite=False)
        floor=max(1e-300,1e-12*float(np.max(e))) if self.robust else 1e-300
        e = np.maximum(e, floor)
        self.V = (L @ u) / e ** 0.25
        Li = linalg.solve_triangular(L, np.eye(self.d), lower=True, check_finite=False)
        self.Vinv = (u.T @ Li) * e[:, None] ** 0.25
        self.Dinv = self.Vinv.T @ self.Vinv
        self.ev = np.sqrt(e)

    def hinv(self, v):
        return self.vec(self.Dinv @ self.mat(v) @ self.Dinv)

    def corrector(self, sigma_mu, ds=None, dz=None):
        r = sigma_mu * np.eye(self.d)
        if ds is not None:
            ds = self.Vinv @ self.mat(ds) @ self.Vinv.T
            dz = self.V.T @ self.mat(dz) @ self.V
            r -= (ds @ dz + dz @ ds) * 0.5
        y = 2.0 * r / (self.ev[:, None] + self.ev)
        return self.vec(-self.s + self.V @ y @ self.V.T)

    def step(self, x, dx):
        x = self.mat(x)
        dx = self.mat(dx)
        L = self.cholesky(x)
        t = linalg.solve_triangular(L, dx, lower=True, check_finite=False)
        t = linalg.solve_triangular(L, t.T, lower=True, check_finite=False).T
        mineig = linalg.eigh(t, subset_by_index=[0, 0], eigvals_only=True,
                            check_finite=False)[0]
        return min(1.0, -1.0 / mineig) if mineig < 0 else 1.0

    def cholesky(self,x):
        try:
            return linalg.cholesky(x,lower=True,check_finite=False)
        except np.linalg.LinAlgError:
            if not self.robust:
                raise
            ev=linalg.eigh(x,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0]
            scale=max(1.,float(np.max(np.abs(x))))
            return linalg.cholesky(x+max(0.,-ev+5e-15*scale)*np.eye(self.d),
                                  lower=True,check_finite=False)


class Layout:
    def __init__(self, cones, m):
        self.m = m
        eq, nonneg, exp = [], [], []
        nnstarts=[]
        soc = {}
        self.psd = []
        self.degree = 0
        st = 0
        for c in cones:
            kind, d = c["type"], c["dim"]
            width = d * (d + 1) // 2 if kind == "psd" else d
            if kind == "zero":
                eq.extend(range(st, st + d))
            elif kind == "nonnegative":
                nnstarts.append(len(nonneg))
                nonneg.extend(range(st, st + d))
                self.degree += d
            elif kind == "soc":
                soc.setdefault(d, []).append(st)
                self.degree += 2
            elif kind == "exponential":
                exp.append(st)
                self.degree += 3
            elif kind == "psd":
                self.psd.append(PSD(st, d))
                self.degree += d
            st += width
        self.eq = np.array(eq, dtype=np.int32)
        self.nn = np.array(nonneg, dtype=np.int32)
        self.nnstarts=np.asarray(nnstarts,dtype=np.int32)
        self.ex = np.array(exp, dtype=np.int32)[:, None] + np.arange(3)
        self.soc = {d: np.array(v, dtype=np.int32)[:, None] + np.arange(d)
                    for d, v in soc.items()}
        self.oracles={}
        if self.soc and not os.environ.get("NO_SOC_NATIVE"):
            from soc import Oracle
            self.oracles={d:Oracle(ids) for d,ids in self.soc.items()}
        self.conic = np.ones(m, dtype=bool)
        self.conic[self.eq] = False
        self.hrows, self.hcols = [self.nn], [self.nn]
        for ids in list(self.soc.values()) + [self.ex]:
            d = ids.shape[1]
            if d>64:
                continue
            self.hrows.append(np.repeat(ids, d, axis=1).ravel())
            self.hcols.append(np.tile(ids, (1, d)).ravel())
        self.hrows = np.concatenate(self.hrows)
        self.hcols = np.concatenate(self.hcols)
        self.native = None
        if self.psd:
            self.native = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "native.so"))
            ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
            dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            self.native.psd_schur.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ip, ip, ip, dp, dp, dp]

    def initial(self, b=None):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.nn] = z[self.nn] = 1.0
        for ids in self.soc.values():
            s[ids[:, 0]] = z[ids[:, 0]] = math.sqrt(2.0)
        s[self.ex] = (-1.0, 1.0, 2.0)
        z[self.ex] = (-1.0, 0.0, 1.0)
        for block in self.psd:
            s[block.sl] = z[block.sl] = block.vec(np.eye(block.d))
        if b is not None:
            s[self.nn]=np.maximum(1.0,b[self.nn])
            z[self.nn]=1.0/s[self.nn]
            for ids in self.soc.values():
                t=b[ids].copy()
                vn=np.linalg.norm(t[:,1:],axis=1)
                t[:,0]=np.maximum(t[:,0],vn+math.sqrt(2.0))
                det=(t[:,0]-vn)*(t[:,0]+vn)
                s[ids]=t
                zz=2*t/det[:,None]
                zz[:,1:]*=-1
                z[ids]=zz
        return s, z

    def row_scale(self, scale):
        for ids in list(self.soc.values()) + [self.ex]:
            if len(ids):
                scale[ids] = np.min(scale[ids], axis=1)[:, None]
        for block in self.psd:
            scale[block.sl] = np.min(scale[block.sl])
        return scale

    def prepare_A(self, A):
        for block in self.psd:
            coo = A[block.sl].tocoo()
            block.ari = np.ascontiguousarray(block.r[coo.row], dtype=np.int32)
            block.arj = np.ascontiguousarray(block.c[coo.row], dtype=np.int32)
            block.acol = np.ascontiguousarray(coo.col, dtype=np.int32)
            block.aval = np.ascontiguousarray(coo.data, dtype=np.float64)

    def update(self, s, z, mu, inverse=True,expanded=False):
        self.s = s
        self.z = z
        self.mu = mu
        values = [z[self.nn] / s[self.nn]]
        hvalues = [s[self.nn] / z[self.nn]]
        br, bc, bv = [self.nn], [self.nn], [s[self.nn]/z[self.nn]]
        ur, uc, uv = [], [], []
        n_rank = 0
        self.soc_data = {}
        for d, ids in self.soc.items():
            if d in self.oracles:
                oracle=self.oracles[d]
                oracle.update(s,z,dense=not inverse and not expanded,inverse=inverse)
                if inverse:
                    values.append(oracle.hi)
                if not inverse and not expanded:
                    hvalues.append(oracle.h)
                br.append(ids.ravel())
                bc.append(ids.ravel())
                bv.append(oracle.diagonal)
                ur.append(ids.ravel())
                uc.append(np.repeat(np.arange(n_rank,n_rank+len(ids)),d))
                uv.append(oracle.u)
                n_rank+=len(ids)
                continue
            ss, zz = s[ids].astype(np.longdouble), z[ids].astype(np.longdouble)
            sn = np.linalg.norm(ss[:, 1:], axis=1)
            zn = np.linalg.norm(zz[:, 1:], axis=1)
            dets = (ss[:, 0] - sn) * (ss[:, 0] + sn)
            detz = (zz[:, 0] - zn) * (zz[:, 0] + zn)
            dets = np.maximum(dets, 1e-300)
            detz = np.maximum(detz, 1e-300)
            eta = (dets / detz) ** 0.25
            jz = zz.copy()
            jz[:, 1:] *= -1.0
            denom = np.sqrt(2.0 * (np.sum(ss * zz, axis=1) + np.sqrt(dets * detz)))
            w = (ss / eta[:, None] + jz * eta[:, None]) / denom[:, None]
            jw = w.copy()
            jw[:, 1:] *= -1.0
            h_inv = qrep(jw, True) / eta[:, None, None] ** 2
            values.append(h_inv.ravel())
            hvalues.append((qrep(w, True) * eta[:, None, None] ** 2).ravel())
            br.append(ids.ravel())
            bc.append(ids.ravel())
            diag = np.ones_like(w) * eta[:, None]**2
            diag[:, 0] *= -1
            bv.append(diag.ravel())
            ur.append(ids.ravel())
            uc.append(np.repeat(np.arange(n_rank, n_rank+len(ids)), d))
            uv.append((np.sqrt(np.longdouble(2.0))*eta[:, None]*w).ravel())
            n_rank += len(ids)
            sqw = w.copy()
            sqw[:, 0] = np.sqrt((w[:, 0] + 1.0) * 0.5)
            sqw[:, 1:] /= 2.0 * sqw[:, :1]
            W = qrep(sqw, True) * eta[:, None, None]
            jsqw = sqw.copy()
            jsqw[:, 1:] *= -1.0
            Winv = qrep(jsqw, True) / eta[:, None, None]
            lam = mv(W, zz)
            self.soc_data[d] = (W, Winv, lam)
        if len(self.ex):
            self.eg, eh = exp_barrier(z[self.ex])
            self.eh = mu * eh
            if inverse:
                values.append(np.linalg.inv(self.eh).ravel())
            hvalues.append(self.eh.ravel())
            br.append(np.repeat(self.ex, 3, axis=1).ravel())
            bc.append(np.tile(self.ex, (1, 3)).ravel())
            bv.append(self.eh.ravel())
        else:
            self.eg = None
        if inverse:
            self.Hi = sparse.csc_matrix((np.asarray(np.concatenate(values), dtype=np.float64),
                                         (self.hrows, self.hcols)),
                                        shape=(self.m, self.m))
        if not inverse and not expanded:
            self.H = sparse.csc_matrix((np.asarray(np.concatenate(hvalues), dtype=np.float64),
                                        (self.hrows, self.hcols)),
                                       shape=(self.m, self.m))
        self.Hb = sparse.csc_matrix((np.asarray(np.concatenate(bv), dtype=np.float64),
                                     (np.concatenate(br), np.concatenate(bc))),
                                    shape=(self.m, self.m))
        if n_rank:
            self.U = sparse.csc_matrix((np.asarray(np.concatenate(uv), dtype=np.float64),
                                        (np.concatenate(ur),np.concatenate(uc))),
                                       shape=(self.m,n_rank))
        else:
            self.U = None
        for block in self.psd:
            block.update(s[block.sl], z[block.sl])

    def apply_hinv(self, v):
        r = self.Hi @ v
        for block in self.psd:
            r[block.sl] = block.hinv(v[block.sl])
        return r

    def schur(self, P, A):
        h = P + A.T @ (self.Hi @ A)
        if not self.psd:
            return h.tocsc()
        out = np.ascontiguousarray(h.toarray())
        for block in self.psd:
            self.native.psd_schur(A.shape[1], block.d, len(block.aval),
                                  block.ari, block.arj, block.acol, block.aval,
                                  np.ascontiguousarray(block.Dinv), out)
        return (out + out.T) * 0.5

    def corrector(self, sigma_mu, ds=None, dz=None):
        c = -self.s.copy()
        corr = sigma_mu
        if ds is not None:
            corr = sigma_mu - ds[self.nn] * dz[self.nn]
        c[self.nn] += corr / self.z[self.nn]
        for d, ids in self.soc.items():
            if d in self.oracles:
                if ds is not None:
                    c[ids]=self.oracles[d].correct(self.s,ds,dz,sigma_mu)
                elif sigma_mu:
                    zeros=np.zeros_like(self.s)
                    c[ids]=self.oracles[d].correct(self.s,zeros,zeros,sigma_mu)
                continue
            W, Winv, lam = self.soc_data[d]
            r = np.zeros_like(lam)
            r[:, 0] = 2.0 * sigma_mu
            if ds is not None:
                r -= jordan(mv(Winv, ds[ids]), mv(W, dz[ids]))
            c[ids] += mv(W, linv(lam, r))
        if len(self.ex):
            c[self.ex] -= sigma_mu * self.eg
        for block in self.psd:
            c[block.sl] = block.corrector(sigma_mu,
                None if ds is None else ds[block.sl],
                None if dz is None else dz[block.sl])
        c[self.eq] = 0.0
        return c

    def interior(self, x, dual=False):
        if len(self.nn) and np.min(x[self.nn]) <= 0.0:
            return False
        for ids in self.soc.values():
            t = x[ids]
            if np.any(t[:, 0] <= np.linalg.norm(t[:, 1:], axis=1)):
                return False
        if len(self.ex):
            t = x[self.ex]
            if dual:
                t = t[:, [1, 0, 2]] * [-1.0, -1.0, math.e]
            a, b, c = t.T
            if np.any(b <= 0) or np.any(c <= 0):
                return False
            if np.any(a >= b * (np.log(c) - np.log(b))):
                return False
        return True

    def neighborhood(self, s, z, mu):
        if len(self.nn) and np.min(s[self.nn] * z[self.nn]) < 1e-3 * mu:
            return False
        for ids in self.soc.values():
            d=ids.shape[1]
            if d in self.oracles:
                if not self.oracles[d].neighborhood(s,z,mu):
                    return False
                continue
            ss, zz = s[ids].astype(np.longdouble), z[ids].astype(np.longdouble)
            sn = np.sqrt(np.sum(ss[:, 1:]**2, axis=1))
            zn = np.sqrt(np.sum(zz[:, 1:]**2, axis=1))
            ds = (ss[:, 0]-sn)*(ss[:, 0]+sn)
            dz = (zz[:, 0]-zn)*(zz[:, 0]+zn)
            if np.any(ds <= 0) or np.any(dz <= 0):
                return False
            gamma = np.sum(ss*zz, axis=1)
            prod = ds*dz
            small = prod / (gamma + np.sqrt(np.maximum(gamma**2-prod, 0.0)))
            if np.min(small) < 2e-2 * mu:
                return False
        return True

    def step(self, x, dx, dual=False):
        alpha = 1.0
        if len(self.nn):
            dd = dx[self.nn]
            mask = dd < 0
            if np.any(mask):
                alpha = min(alpha, np.min(-x[self.nn][mask] / dd[mask]))
        for ids in self.soc.values():
            d=ids.shape[1]
            if d in self.oracles:
                alpha=min(alpha,self.oracles[d].step(x,dx))
                continue
            u, v = x[ids], dx[ids]
            aa = v[:, 0] ** 2 - np.sum(v[:, 1:] ** 2, axis=1)
            bb = 2 * (u[:, 0] * v[:, 0] - np.sum(u[:, 1:] * v[:, 1:], axis=1))
            cc = u[:, 0] ** 2 - np.sum(u[:, 1:] ** 2, axis=1)
            disc = np.maximum(bb * bb - 4 * aa * cc, 0.0)
            den = -bb + np.sqrt(disc)
            roots = np.full(len(u), np.inf)
            mask = den > 0
            roots[mask] = 2 * cc[mask] / den[mask]
            candidates = (aa < 0) | ((bb < 0) & (disc > 0))
            if np.any(candidates):
                alpha = min(alpha, np.min(roots[candidates]))
            mask = v[:, 0] < 0
            if np.any(mask):
                alpha = min(alpha, np.min(-u[mask, 0] / v[mask, 0]))
        if len(self.ex):
            t, dt = x[self.ex], dx[self.ex]
            if dual:
                t = t[:, [1, 0, 2]] * [-1.0, -1.0, math.e]
                dt = dt[:, [1, 0, 2]] * [-1.0, -1.0, math.e]
            for j in (1, 2):
                mask = dt[:, j] < 0
                if np.any(mask):
                    alpha = min(alpha, np.min(-t[mask, j] / dt[mask, j]))
            alpha = max(0.0, float(alpha))
            for k in range(60):
                u = t + (alpha * (1.0 - 1e-12)) * dt
                if np.all(u[:, 1:] > 0):
                    val = u[:, 1] * (np.log(u[:, 2]) - np.log(u[:, 1])) - u[:, 0]
                    if np.all(val > 0):
                        break
                alpha *= 0.8
        for block in self.psd:
            alpha = min(alpha, block.step(x[block.sl], dx[block.sl]))
        return max(0.0, float(alpha))

    def violation(self, x, dual=False):
        if not np.isfinite(x).all():
            return np.inf
        worst = 0.0
        if len(self.eq) and not dual:
            worst = float(np.max(np.abs(x[self.eq]), initial=0.0))
        if len(self.nn):
            t = x[self.nn]
            scales=np.maximum(1.0,np.maximum.reduceat(np.abs(t),self.nnstarts))
            defects=-np.minimum.reduceat(t,self.nnstarts)/scales
            worst = max(worst,float(np.max(defects,initial=0.0)))
        for ids in self.soc.values():
            t = x[ids]
            t=t/np.maximum(1.0,np.max(np.abs(t),axis=1))[:,None]
            val = np.linalg.norm(t[:, 1:], axis=1) - t[:, 0]
            worst = max(worst, float(np.max(val, initial=0.0)))
        if len(self.ex):
            t = x[self.ex]
            t = t / np.maximum(1.0, np.max(np.abs(t), axis=1))[:, None]
            if dual:
                t = t[:, [1, 0, 2]] * [-1.0, -1.0, math.e]
            a, b, c = t.T
            boundary = np.maximum.reduce([np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)])
            val = boundary.copy()
            mask = (b > 0) & (c > 0)
            val[mask] = np.minimum(boundary[mask], np.maximum(0.0,
                a[mask] - b[mask] * (np.log(c[mask]) - np.log(b[mask]))))
            worst = max(worst, float(np.max(val, initial=0.0)))
        for block in self.psd:
            v = x[block.sl]
            mineig = linalg.eigh(block.mat(v / max(1.0, np.max(np.abs(v)))),
                subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
            worst = max(worst, -float(mineig))
        return worst
