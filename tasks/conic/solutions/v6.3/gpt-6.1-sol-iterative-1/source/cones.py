"""Cone algebra, Nesterov-Todd scaling, and exponential barrier oracles."""

import math

import numpy as np
from scipy import sparse
from numeric import linalg


SQRT2 = math.sqrt(2.0)


def soc_pow(x, power):
    t = x[:, 0]
    v = x[:, 1:]
    r = np.linalg.norm(v, axis=1)
    lo = np.maximum(t - r, 1e-300)
    hi = np.maximum(t + r, 1e-300)
    a, b = hi**power, lo**power
    out = np.empty_like(x)
    out[:, 0] = (a + b) * 0.5
    fac = np.divide((a - b) * 0.5, r, out=np.zeros_like(r), where=r > 1e-100)
    out[:, 1:] = v * fac[:, None]
    return out


def soc_q(x, determinant=None):
    k, d = x.shape
    t, v = x[:, 0], x[:, 1:]
    vv = np.sum(v * v, axis=1)
    out = np.zeros((k, d, d))
    out[:, 0, 0] = t * t + vv
    out[:, 0, 1:] = out[:, 1:, 0] = 2 * t[:, None] * v
    out[:, 1:, 1:] = 2 * v[:, :, None] * v[:, None, :]
    ix = np.arange(1, d)
    out[:, ix, ix] += ((t * t - vv) if determinant is None else determinant)[:, None]
    return out


def soc_qapply(w, x, determinant):
    out = 2 * w * np.sum(w * x, axis=1)[:, None]
    out[:, 0] -= determinant * x[:, 0]
    out[:, 1:] += determinant[:, None] * x[:, 1:]
    return out


def soc_prod(x, y):
    out = np.empty_like(x)
    out[:, 0] = np.sum(x * y, axis=1)
    out[:, 1:] = x[:, :1] * y[:, 1:] + y[:, :1] * x[:, 1:]
    return out


def soc_linv(x, y):
    t, v = x[:, 0], x[:, 1:]
    det = t * t - np.sum(v * v, axis=1)
    out = np.empty_like(y)
    out[:, 0] = (t * y[:, 0] - np.sum(v * y[:, 1:], axis=1)) / det
    out[:, 1:] = (y[:, 1:] - v * out[:, :1]) / t[:, None]
    return out


def exp_oracle(x):
    a, b, c = x.T
    lr = np.log(c) - np.log(b)
    t = b * lr - a
    v = np.column_stack((-np.ones_like(a), lr - 1, b / c))
    g = -v / t[:, None]
    g[:, 1] -= 1 / b
    g[:, 2] -= 1 / c
    h = v[:, :, None] * v[:, None, :] / (t * t)[:, None, None]
    h[:, 1, 1] += 1 / (b * t) + 1 / (b * b)
    h[:, 1, 2] -= 1 / (c * t)
    h[:, 2, 1] -= 1 / (c * t)
    h[:, 2, 2] += b / (c * c * t) + 1 / (c * c)
    return g, h


def exp_inside(x, dual=False):
    if dual:
        x = np.column_stack((-x[:, 1], -x[:, 0], math.e * x[:, 2]))
    a, b, c = x.T
    if np.any(b <= 0) or np.any(c <= 0):
        return False
    return bool(np.all(b * (np.log(c) - np.log(b)) - a > 0))


def exp_shadow(z):
    a, b, c = z.T
    t = -1 / a
    value = 1 - b / a - (np.log(-a) - np.log(c))
    value = np.maximum(value, 1e-300)
    xi = np.maximum(value - np.log1p(value), value * .5)
    for _ in range(8):
        xi -= (xi + np.log1p(xi) - value) / (1 + 1 / (1 + xi))
        xi = np.maximum(xi, 1e-300)
    y = t / xi
    w = (1 + 1 / xi) / c
    x = y * (np.log(w) - np.log(y)) - t
    return np.column_stack((x, y, w))


class ConeSystem:
    def __init__(self, cones):
        eq, nn, soc, exp, psd = [], [], {}, [], []
        self.nn_blocks = []
        pos = 0
        for c in cones:
            kind, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if kind == "psd" else d
            rows = np.arange(pos, pos + w)
            pos += w
            if kind == "zero":
                eq.extend(rows)
            elif kind == "nonnegative":
                self.nn_blocks.append(slice(len(nn), len(nn) + d))
                nn.extend(rows)
            elif kind == "soc":
                soc.setdefault(d, []).append(rows)
            elif kind == "exponential":
                exp.append(rows)
            else:
                psd.append((d, rows))
        parts = [np.asarray(eq, dtype=int), np.asarray(nn, dtype=int)]
        self.neq, self.nnonneg = len(eq), len(nn)
        self.soc = []
        off = len(nn)
        for d, rows in sorted(soc.items()):
            k = len(rows)
            self.soc.append((slice(off, off + k * d), k, d))
            off += k * d
            parts.append(np.concatenate(rows))
        self.exps = slice(off, off + 3 * len(exp))
        self.nexp = len(exp)
        off += 3 * len(exp)
        if exp:
            parts.append(np.concatenate(exp))
        self.nsmall = off
        self.naux = 2 * sum(k for _, k, _ in self.soc) + 5 * len(exp)
        self.aux_sign = np.r_[np.tile([1., -1.], sum(k for _, k, _ in self.soc)),
                              np.ones(5 * len(exp))]
        self.psd = []
        for d, rows in psd:
            w = len(rows)
            col = np.repeat(np.arange(d), np.arange(1, d + 1))
            row = np.concatenate([np.arange(i + 1) for i in range(d)])
            factor = np.where(row == col, 1.0, SQRT2)
            self.psd.append({"d": d, "sl": slice(off, off + w), "row": row,
                             "col": col, "factor": factor})
            off += w
            parts.append(rows)
        self.size = off
        self.perm = np.concatenate(parts) if parts else np.empty(0, dtype=int)
        self.nu = len(nn) + sum(2 * k for _, k, _ in self.soc) + 3 * len(exp) + sum(d for d, _ in psd)
        # Fixed block sparsity for the small-cone scaling operator.
        rr, cc = [np.arange(len(nn))], [np.arange(len(nn))]
        for sl, k, d in self.soc:
            if d > 32:
                rr.append(np.arange(sl.start, sl.stop))
                cc.append(np.arange(sl.start, sl.stop))
                continue
            base = np.arange(sl.start, sl.stop, d)
            rr.append((base[:, None, None] + np.arange(d)[None, :, None] +
                       np.zeros((1, 1, d), dtype=int)).ravel())
            cc.append((base[:, None, None] + np.arange(d)[None, None, :] +
                       np.zeros((1, d, 1), dtype=int)).ravel())
        if exp:
            base = np.arange(self.exps.start, self.exps.stop, 3)
            rr.append((base[:, None, None] + np.arange(3)[None, :, None] +
                       np.zeros((1, 1, 3), dtype=int)).ravel())
            cc.append((base[:, None, None] + np.arange(3)[None, None, :] +
                       np.zeros((1, 3, 1), dtype=int)).ravel())
        self.rr = np.concatenate(rr)
        self.cc = np.concatenate(cc)
        urows, ucols, naux = [], [], 0
        for sl, k, d in self.soc:
            rows = np.arange(sl.start, sl.stop).reshape(k, d)
            cols = np.arange(naux, naux + 2 * k, 2)
            urows.append(np.broadcast_to(rows[:, :, None], (k, d, 2)).ravel())
            ucols.append(np.broadcast_to(cols[:, None, None] + np.arange(2)[None, None, :],
                                         (k, d, 2)).ravel())
            naux += 2 * k
        if self.nexp:
            base = np.arange(self.exps.start, self.exps.stop, 3)
            aux = np.arange(naux, naux + 5 * self.nexp, 5)
            urows.append((base[:, None, None] + np.arange(3)[None, :, None] +
                          np.zeros((1, 1, 5), dtype=int)).ravel())
            ucols.append((aux[:, None, None] + np.arange(5)[None, None, :] +
                          np.zeros((1, 3, 1), dtype=int)).ravel())
        self.urows = np.concatenate(urows) if urows else np.empty(0, dtype=int)
        self.ucols = np.concatenate(ucols) if ucols else np.empty(0, dtype=int)
        self.large_rows, self.large_cols = [], []
        nlarge = 0
        for sl, k, d in self.soc:
            if d > 32:
                self.large_rows.append(np.arange(sl.start, sl.stop))
                self.large_cols.append(np.repeat(np.arange(nlarge, nlarge + k), d))
                nlarge += k
        self.nlarge = nlarge
        self.dorder = np.lexsort((self.rr, self.cc))
        self.dind = self.rr[self.dorder]
        self.dptr = np.r_[0, np.cumsum(np.bincount(self.cc, minlength=self.nsmall))]
        self.D = sparse.csc_matrix((np.zeros(len(self.rr)), self.dind, self.dptr),
                                   shape=(self.nsmall, self.nsmall))
        self.Di = self.D.copy()
        self.uorder = np.lexsort((self.urows, self.ucols))
        self.uind = self.urows[self.uorder]
        self.uptr = np.r_[0, np.cumsum(np.bincount(self.ucols, minlength=self.naux))]
        self.U = sparse.csc_matrix((np.zeros(len(self.urows)), self.uind, self.uptr),
                                   shape=(self.size, self.naux))

    @staticmethod
    def smat(x, block):
        d, r, c, f = block["d"], block["row"], block["col"], block["factor"]
        out = np.zeros((d, d))
        out[r, c] = x / f
        out[c, r] = out[r, c]
        return out

    @staticmethod
    def svec(x, block):
        return x[block["row"], block["col"]] * block["factor"]

    def initial(self):
        s = np.zeros(self.size)
        z = np.zeros(self.size)
        s[:self.nnonneg] = z[:self.nnonneg] = 1
        for sl, k, d in self.soc:
            s[sl].reshape(k, d)[:, 0] = SQRT2
            z[sl] = s[sl]
        if self.nexp:
            s[self.exps].reshape(-1, 3)[:] = (-1, 1, 1)
            g, _ = exp_oracle(s[self.exps].reshape(-1, 3))
            z[self.exps] = -g.ravel()
        for block in self.psd:
            sl = block["sl"]
            s[sl] = self.svec(np.eye(block["d"]), block)
            z[sl] = s[sl]
        return s, z

    def warm_pair(self, raw_s, raw_z, delta=.01):
        s, z = self.initial()
        k = self.nnonneg
        if k:
            sp, zp = np.maximum(raw_s[:k], 0), np.maximum(raw_z[:k], 0)
            s[:k], z[:k] = sp + delta / (1 + zp), zp + delta / (1 + sp)
        for sl, k, d in self.soc:
            value = raw_s[sl].reshape(k, d)
            r = np.linalg.norm(value[:, 1:], axis=1)
            eigen = np.column_stack((value[:, 0] + r, value[:, 0] - r))
            positive, negative = np.maximum(eigen, 0), np.maximum(-eigen, 0)
            se = positive + 2 * delta / (1 + negative)
            ze = negative + 2 * delta / (1 + positive)
            se = np.maximum(se, 1e-10 * np.maximum(1, se.max(axis=1))[:, None])
            ze = np.maximum(ze, 1e-10 * np.maximum(1, ze.max(axis=1))[:, None])
            for out, vals in ((s, se), (z, ze)):
                u = out[sl].reshape(k, d)
                u[:, 0] = np.mean(vals, axis=1)
                fac = np.divide(vals[:, 0] - vals[:, 1], 2 * r,
                                out=np.zeros(k), where=r > 1e-100)
                u[:, 1:] = value[:, 1:] * fac[:, None]
        if self.nexp:
            value = raw_s[self.exps].reshape(-1, 3)
            alpha = np.full(len(value), delta)
            direction = np.array([-1., 1., 1.])
            for _ in range(100):
                trial = value + alpha[:, None] * direction
                a, b, c = trial.T
                positive = (b > 0) & (c > 0)
                inside = np.zeros(len(value), dtype=bool)
                inside[positive] = (b[positive] * (np.log(c[positive]) - np.log(b[positive])) -
                                    a[positive]) > 1e-10
                if np.all(inside):
                    break
                alpha[~inside] *= 2
            s[self.exps] = trial.ravel()
            z[self.exps] = (-delta * exp_oracle(trial)[0]).ravel()
        return s, z

    def scale(self, s, z, mu):
        self._s, self._z = s, z
        self.ds = z[:self.nnonneg] / s[:self.nnonneg]
        ds, di = [self.ds], [1 / self.ds]
        uvalues = []
        large_values = []
        self.soc_state = []
        for sl, k, d in self.soc:
            ss, zz = s[sl].reshape(k, d), z[sl].reshape(k, d)
            sr, zr = np.linalg.norm(ss[:, 1:], axis=1), np.linalg.norm(zz[:, 1:], axis=1)
            sd = (ss[:, 0] - sr) * (ss[:, 0] + sr)
            zd = (zz[:, 0] - zr) * (zz[:, 0] + zr)
            gamma = np.sqrt(zd / sd)
            js = ss.copy()
            js[:, 1:] *= -1
            w = (zz + gamma[:, None] * js) / np.sqrt(2 * (np.sum(ss * zz, axis=1) + np.sqrt(sd * zd)))[:, None]
            wh = w.copy()
            wh[:, 0] += np.sqrt(gamma)
            wh /= np.sqrt(2 * (w[:, 0] + np.sqrt(gamma)))[:, None]
            whi = wh / np.sqrt(gamma)[:, None]
            whi[:, 1:] *= -1
            wi = w / gamma[:, None]
            wi[:, 1:] *= -1
            if d <= 32:
                D = soc_q(w, gamma)
                Dinv = np.zeros_like(D)
                ix = np.arange(d)
                Dinv[:, ix, ix] = 1 / gamma[:, None]
                ds.append(D.ravel())
                di.append(Dinv.ravel())
            else:
                diag = np.broadcast_to(gamma[:, None], (k, d)).copy()
                diag[:, 0] *= -1
                ds.append(diag.ravel())
                di.append(np.broadcast_to(1 / gamma[:, None], (k, d)).ravel())
                # D = diag(-gamma*J) + 2*w*w', without a dense d-by-d block.
                large_values.append(SQRT2 * w.ravel())
            # Q(w^-1) = g*I + u*u' - v*v', with ||v||^2 < g.
            # The negative auxiliary row joins a negative-definite block.
            g = 1 / gamma
            r = np.linalg.norm(wi[:, 1:], axis=1)
            p = wi[:, 0] + r
            unit = np.divide(wi[:, 1:], r[:, None], out=np.zeros((k, d - 1)),
                             where=r[:, None] > 1e-100)
            ratio = np.clip(1 - (g / p) / p, 0., 1.)
            positive = p * np.sqrt(.5 * ratio)
            negative = np.sqrt(.5 * g * ratio)
            U = np.empty((k, d, 2))
            U[:, 0, 0], U[:, 0, 1] = positive, negative
            U[:, 1:, 0] = positive[:, None] * unit
            U[:, 1:, 1] = -negative[:, None] * unit
            uvalues.append(U.ravel())
            lam = soc_qapply(wh, ss, np.sqrt(gamma))
            invs = js / sd[:, None]
            self.soc_state.append((wh, whi, np.sqrt(gamma), lam, invs, w, gamma))
        if self.nexp:
            ss, zz = s[self.exps].reshape(-1, 3), z[self.exps].reshape(-1, 3)
            self.eg, H = exp_oracle(ss)
            # Positive-definite secant scaling, exact on the homogeneous ray.
            H *= (np.sum(ss * zz, axis=1) / 3)[:, None, None]
            hs = np.einsum("ijk,ik->ij", H, ss)
            den = np.sum(hs * ss, axis=1)
            sz = np.sum(ss * zz, axis=1)
            H -= hs[:, :, None] * hs[:, None, :] / den[:, None, None]
            H += zz[:, :, None] * zz[:, None, :] / sz[:, None, None]
            localmu = sz / 3
            # An analytic factor of H(s)^-1 avoids inversion of a matrix whose
            # condition number grows quadratically as the gap closes.
            a, b, c = ss.T
            lr = np.log(c) - np.log(b)
            t = b * lr - a
            xi = t / b
            V = np.zeros((len(ss), 3, 5))
            V[:, 0, 0] = t / np.sqrt(localmu)
            V[:, :, 1] = np.column_stack((b * lr, b, c)) / np.sqrt(2 * localmu)[:, None]
            V[:, :, 2] = (np.column_stack((b * (lr - 2), b, -c)) *
                            np.sqrt(xi / (2 * localmu * (xi + 2)))[:, None])
            zv = np.einsum("ij,ijk->ik", zz, V[:, :, :3])
            V[:, :, :3] -= ss[:, :, None] * zv[:, None, :] / sz[:, None, None]
            V[:, :, 3] = ss / np.sqrt(sz)[:, None]
            shadow = exp_shadow(zz)
            p = shadow - ss / localmu[:, None]
            y = -self.eg - zz / localmu[:, None]
            hp = np.einsum("ijk,ik->ij", H, p)
            php = np.sum(p * hp, axis=1)
            py = np.sum(p * y, axis=1)
            use = (php > 1e-20) & (py > 1e-20) & np.isfinite(py)
            H[use] -= hp[use, :, None] * hp[use, None, :] / php[use, None, None]
            H[use] += y[use, :, None] * y[use, None, :] / py[use, None, None]
            yv = np.einsum("ij,ijk->ik", y[use], V[use, :, :4])
            V[use, :, :4] -= p[use, :, None] * yv[:, None, :] / py[use, None, None]
            V[use, :, 4] = p[use] / np.sqrt(py[use])[:, None]
            uvalues.append(V.ravel())
            self.eH = H
            ds.append(H.ravel())
            di.append(np.zeros(H.size))
        self.D.data[:] = np.concatenate(ds)[self.dorder]
        self.Di.data[:] = np.concatenate(di)[self.dorder]
        if uvalues:
            self.U.data[:] = np.concatenate(uvalues)[self.uorder]
        self.large_V = sparse.csc_matrix((np.concatenate(large_values) if large_values else np.empty(0),
                                         (np.concatenate(self.large_rows) if self.large_rows else np.empty(0, dtype=int),
                                          np.concatenate(self.large_cols) if self.large_cols else np.empty(0, dtype=int))),
                                        shape=(self.nsmall, self.nlarge))
        self.psd_state = []
        for block in self.psd:
            S, Z = (self.smat(v[block["sl"]], block) for v in (s, z))
            L = linalg.cholesky(S, lower=True, check_finite=False)
            vals, U = linalg.eigh(L.T @ Z @ L, check_finite=False)
            vals = np.maximum(vals, 1e-300)
            Li = linalg.solve_triangular(L, np.eye(len(L)), lower=True, check_finite=False)
            R = vals[:, None]**.25 * (U.T @ Li)
            Ri = (L @ U) * vals[None, :]**-.25
            W = R.T @ R
            Sinv = Li.T @ Li
            self.psd_state.append((W, R, Ri, np.sqrt(vals), Sinv))

    def apply(self, v):
        out = np.empty_like(v)
        out[:self.nsmall] = self.D @ v[:self.nsmall]
        for (sl, k, d), state in zip(self.soc, self.soc_state):
            if d > 32:
                out[sl] = soc_qapply(state[5], v[sl].reshape(k, d), state[6]).ravel()
        for block, state in zip(self.psd, self.psd_state):
            W = state[0]
            out[block["sl"]] = self.svec(W @ self.smat(v[block["sl"]], block) @ W, block)
        return out

    def inverse_energy(self, v):
        k = self.nnonneg
        value = float(np.sum((self._s[:k] / self._z[:k]) * v[:k]**2))
        for (sl, count, d), state in zip(self.soc, self.soc_state):
            scaled = soc_qapply(state[1], v[sl].reshape(count, d), 1 / state[2])
            value += float(np.sum(scaled * scaled))
        if self.nexp:
            offset = 2 * sum(count for _, count, _ in self.soc)
            scaled = (self.U.T @ v)[offset:]
            value += float(scaled @ scaled)
        return value

    def rhs(self, target, da=None, za=None):
        s, z = self._s, self._z
        out = -z.copy()
        corr = da is not None
        out[:self.nnonneg] += (target - (da[:self.nnonneg] * za[:self.nnonneg] if corr else 0)) / s[:self.nnonneg]
        for (sl, k, d), (wh, whi, det, lam, invs, _, _) in zip(self.soc, self.soc_state):
            res = 2 * target * invs
            if corr:
                dx = soc_qapply(wh, da[sl].reshape(k, d), det)
                dz = soc_qapply(whi, za[sl].reshape(k, d), 1 / det)
                cp = soc_linv(lam, soc_prod(dx, dz))
                res -= soc_qapply(wh, cp, det)
            out[sl] += res.ravel()
        if self.nexp:
            out[self.exps] -= target * self.eg.ravel()
        for block, (_, R, Ri, lam, Sinv) in zip(self.psd, self.psd_state):
            res = target * Sinv
            if corr:
                ds = R @ self.smat(da[block["sl"]], block) @ R.T
                dz = Ri.T @ self.smat(za[block["sl"]], block) @ Ri
                cp = (ds @ dz + dz @ ds) / (lam[:, None] + lam[None, :])
                res -= R.T @ cp @ R
            out[block["sl"]] += self.svec(res, block)
        return out

    def max_step(self, x, dx, dual=False):
        a = 1.0
        k = self.nnonneg
        neg = dx[:k] < 0
        if np.any(neg):
            a = min(a, float(np.min(-x[:k][neg] / dx[:k][neg])))
        for sl, n, d in self.soc:
            xx, dd = x[sl].reshape(n, d), dx[sl].reshape(n, d)
            trial = xx + a * dd
            crossing = trial[:, 0] <= np.linalg.norm(trial[:, 1:], axis=1)
            if np.any(crossing):
                xx, dd = xx[crossing], dd[crossing]
                scale = np.maximum(np.max(np.abs(xx), axis=1), np.max(np.abs(dd), axis=1))
                scale = np.maximum(scale, 1e-300)
                xx, dd = xx / scale[:, None], dd / scale[:, None]
                t, v, dt, dv = xx[:, 0], xx[:, 1:], dd[:, 0], dd[:, 1:]
                r = np.linalg.norm(v, axis=1)
                C = (t - r) * (t + r)
                B = t * dt - np.sum(v * dv, axis=1)
                AA = dt * dt - np.sum(dv * dv, axis=1)
                disc = np.maximum(0., B * B - AA * C)
                den = -B + np.sqrt(disc)
                roots = np.divide(C, den, out=np.full_like(C, np.inf), where=den > 0)
                bound = np.divide(-t, dt, out=np.full_like(t, np.inf), where=dt < 0)
                a = min(a, float(np.min(np.minimum(roots, bound))))
        if self.nexp:
            xx, dd = x[self.exps].reshape(-1, 3), dx[self.exps].reshape(-1, 3)
            if not exp_inside(xx + a * dd, dual):
                lo, hi = 0.0, a
                for _ in range(36):
                    mid = (lo + hi) / 2
                    if exp_inside(xx + mid * dd, dual):
                        lo = mid
                    else:
                        hi = mid
                a = lo
        for block in self.psd:
            X, dX = (self.smat(v[block["sl"]], block) for v in (x, dx))
            L = linalg.cholesky(X, lower=True, check_finite=False)
            V = linalg.solve_triangular(L, dX, lower=True, check_finite=False)
            V = linalg.solve_triangular(L, V.T, lower=True, check_finite=False)
            eig = linalg.eigh(V, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
            if eig < 0:
                a = min(a, -.999999999 / eig)
        return max(0., a)

    def defect(self, v, dual=False):
        if not np.any(v):
            return 0.
        ans = 0.0
        if self.nnonneg:
            u = v[:self.nnonneg]
            if u.min() < 0:
                for sl in self.nn_blocks:
                    block = u[sl]
                    ans = max(ans, -float(block.min()) / max(1., float(np.max(np.abs(block)))))
        for sl, k, d in self.soc:
            u = v[sl].reshape(k, d)
            val = (np.linalg.norm(u[:, 1:], axis=1) - u[:, 0]) / np.maximum(1, np.max(np.abs(u), axis=1))
            ans = max(ans, float(np.max(val)))
        if self.nexp:
            u = v[self.exps].reshape(-1, 3)
            u = u / np.maximum(1., np.max(np.abs(u), axis=1))[:, None]
            if dual:
                u = np.column_stack((-u[:, 1], -u[:, 0], math.e * u[:, 2]))
            a, b, c = u.T
            viol = np.maximum.reduce((np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)))
            ok = (b > 0) & (c > 0)
            viol[ok] = np.minimum(viol[ok], np.maximum(a[ok] - b[ok] * (np.log(c[ok]) - np.log(b[ok])), 0))
            ans = max(ans, float(np.max(viol)))
        for block in self.psd:
            x = v[block["sl"]]
            diagonal = block["row"] == block["col"]
            if not np.any(x[~diagonal]):
                ans = max(ans, -float(np.min(x[diagonal])) / max(1., float(np.max(np.abs(x)))))
                continue
            X = self.smat(x / max(1, float(np.max(np.abs(x)))), block)
            eig = linalg.eigh(X, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
            ans = max(ans, -float(eig))
        return max(0., ans)
