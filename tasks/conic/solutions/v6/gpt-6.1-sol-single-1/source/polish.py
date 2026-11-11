"""Active spectral-face Newton refinement of the conic KKT natural map."""

import sys
import time
import numpy as np
from scipy import sparse, linalg
from common import norm, smat, svec, assess


def spectral_rows(info, V, selected, n):
    r, c, ids, Aa = info
    sr, sc = r[selected], c[selected]
    u, v = r[ids], c[ids]
    fi = np.where(u == v, 1 / np.sqrt(2), 1.0)
    fo = np.where(sr == sc, 1 / np.sqrt(2), 1.0)
    T = np.empty((len(selected), n))
    for lo in range(0, len(selected), 192):
        hi = min(lo + 192, len(selected))
        a, b = sr[lo:hi, None], sc[lo:hi, None]
        Q = V[u[None, :], a] * V[v[None, :], b] + V[v[None, :], a] * V[u[None, :], b]
        Q *= fo[lo:hi, None] * fi[None, :]
        T[lo:hi] = (Aa.T @ Q.T).T
    return T


class NaturalMap:
    def __init__(self, p, x, z):
        self.p = p
        self.nn, self.soc, self.psd = [], {}, []
        s = p.b - p.A @ x
        self.rho = np.ones(p.m)
        self.weight = np.ones(p.m)
        for kind, d, i, j in p.blocks:
            if kind == "nonnegative":
                self.nn.extend(range(i, j))
            elif kind == "soc":
                self.soc.setdefault(d, []).append(i)
            elif kind == "psd":
                r, c = np.triu_indices(d)
                order = np.lexsort((r, c))
                r, c = r[order], c[order]
                Ab = p.A[i:j].tocsc()
                ids = np.unique(Ab.indices)
                self.psd.append((d, i, j, (r, c, ids, Ab[ids].tocsr())))
            if kind in ("soc", "psd"):
                self.rho[i:j] = np.clip(max(norm(z[i:j]), 1e-4) / max(norm(s[i:j]), 1e-4), 1e-8, 1e8)
            self.weight[i:j] = 1 / max(1, norm(z[i:j]), norm(self.rho[i:j] * s[i:j]))
        self.nn = np.asarray(self.nn, int)
        self.soc = {d: np.asarray(v)[:, None] + np.arange(d) for d, v in self.soc.items()}
        self.scalar = np.r_[np.flatnonzero(p.eq), self.nn]
        self.rdscale = 1 / max(1, norm(p.q), norm(p.P @ x))

    def evaluate(self, x, z, derivative=False):
        p = self.p
        s = p.b - p.A @ x
        y = z - self.rho * s
        r2 = np.zeros(p.m)
        r2[p.eq] = self.rho[p.eq] * s[p.eq]
        r2[self.nn] = z[self.nn] - np.maximum(y[self.nn], 0)
        if derivative:
            f = np.zeros(p.m)
            f[p.eq] = 1
            f[self.nn] = y[self.nn] > 0
            rows, cols, data = [self.scalar], [self.scalar], [np.ones(len(self.scalar))]
        for d, idx in self.soc.items():
            a = y[idx]
            t, v = a[:, 0], a[:, 1:]
            r = np.linalg.norm(v, axis=1)
            inside, zero = r <= t, r <= -t
            middle = ~(inside | zero)
            u = v / np.maximum(r[:, None], 1e-300)
            zz = np.zeros_like(a)
            zz[inside] = a[inside]
            zz[middle, 0] = (t[middle] + r[middle]) / 2
            zz[middle, 1:] = zz[middle, :1] * u[middle]
            r2[idx] = z[idx] - zz
            if derivative:
                J = np.zeros((len(idx), d, d))
                J[inside] = np.eye(d)
                J[middle, 0, 0] = 0.5
                J[middle, 0, 1:] = J[middle, 1:, 0] = u[middle] / 2
                tr = t[middle] / r[middle]
                J[middle, 1:, 1:] = ((1 + tr[:, None, None]) * np.eye(d - 1)
                    - tr[:, None, None] * u[middle, :, None] * u[middle, None, :]) / 2
                eigen, Q = np.linalg.eigh(J)
                f[idx] = np.clip(eigen, 0, 1)
                R = Q.transpose(0, 2, 1)
                rows.append(np.repeat(idx, d, axis=1).ravel())
                cols.append(np.tile(idx, (1, d)).ravel())
                data.append(R.ravel())
        pd = []
        for d, i, j, info in self.psd:
            Y = smat(y[i:j], d)
            lam, V = linalg.eigh(Y, check_finite=False)
            prj = (V * np.maximum(lam, 0)) @ V.T
            rr = smat(z[i:j], d) - prj
            r2[i:j] = svec(rr)
            if derivative:
                r, c = info[:2]
                positive = lam > 1e-12 * max(1, norm(lam))
                one = np.flatnonzero(positive[r] & positive[c])
                mix = np.flatnonzero(positive[r] != positive[c])
                hi = np.maximum(lam[r[mix]], lam[c[mix]])
                lo = np.minimum(lam[r[mix]], lam[c[mix]])
                w = np.clip(hi / np.maximum(-lo, 1e-12 * max(1, norm(lam))), 0, 1e10)
                tr2 = svec(V.T @ rr @ V)
                T = spectral_rows(info, V, np.r_[one, mix], p.n)
                pd.append((d, i, j, V, one, mix, w, tr2, T))
        rd = p.P @ x + p.q + p.A.T @ z
        merit = np.sqrt(np.sum((rd * self.rdscale) ** 2) + np.sum((r2 * self.weight) ** 2))
        if not derivative:
            return merit
        R = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(p.m, p.m)).tocsc()
        one = np.flatnonzero(f > 1 - 1e-8)
        mix = np.flatnonzero((f > 1e-8) & (f <= 1 - 1e-8))
        B = (R @ p.A).tocsr()
        rr = R @ r2
        w = f[mix] / (1 - f[mix])
        Bm, C = B[mix], B[one]
        M = p.P + Bm.T @ Bm.multiply((self.rho[mix] * w)[:, None])
        rhs = -rd + p.A.T @ r2 + Bm.T @ (w * rr[mix])
        crhs = rr[one] / self.rho[one]
        if self.psd:
            M, C = M.toarray(), C.toarray()
        counts = [len(one)]
        for d, i, j, V, po, pm, pw, tr2, T in pd:
            To, Tm = T[:len(po)], T[len(po):]
            M += self.rho[i] * (Tm.T @ (pw[:, None] * Tm))
            rhs += Tm.T @ (pw * tr2[pm])
            C = np.vstack((C, To))
            crhs = np.r_[crhs, tr2[po] / self.rho[i]]
            counts.append(len(po))
        return merit, M, C, rhs, crhs, (r2, R, one, mix, w, Bm, rr, pd, counts)

    def dual_step(self, dx, eta, state):
        r2, R, one, mix, w, Bm, rr, pd, counts = state
        extra = np.zeros(self.p.m)
        extra[one] = eta[:len(one)]
        extra[mix] = w * (self.rho[mix] * (Bm @ dx) - rr[mix])
        dz = -r2 + R.T @ extra
        off = len(one)
        for d, i, j, V, po, pm, pw, tr2, T in pd:
            mat = np.zeros(j - i)
            mat[po] = eta[off:off + len(po)]
            mat[pm] = pw * (self.rho[i] * (T[len(po):] @ dx) - tr2[pm])
            dz[i:j] += svec(V @ smat(mat, d) @ V.T)
            off += len(po)
        return dz


def semismooth_polish(p, x, z, recover, original, budget=35.0, verbose=False):
    if any(k == "exponential" for k, _, _, _ in p.blocks):
        return None
    from ipm import LinearSystem
    start = time.monotonic()
    mapping = NaturalMap(p, x, z)
    x, z = x.copy(), z.copy()
    best, besterr = None, float("inf")
    damping, failures = 1.0, 0
    for it in range(22):
        if time.monotonic() - start > budget:
            break
        candidate = recover({"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z})
        ok, err = assess(original, candidate["x"], candidate["s"], candidate["z"], check_all=True)
        if err < besterr:
            best, besterr = candidate, err
        if verbose:
            print("spectral", it, "err", err, file=sys.stderr)
        if ok:
            return candidate
        try:
            merit, M, C, rhs, crhs, state = mapping.evaluate(x, z, True)
            full_count = C.shape[0]
            selected = np.arange(full_count)
            if isinstance(C, np.ndarray) and len(C):
                Q, R, piv = linalg.qr(C.T, mode="economic", pivoting=True, check_finite=False)
                diag = abs(R.diagonal())
                rank = int(np.count_nonzero(diag > max(1e-12, norm(diag) * 1e-10)))
                selected = piv[:rank]
                C, crhs = C[selected], crhs[selected]
            diagonal = M.diagonal()
            reg = damping * min(1e-6, max(1e-13, merit * 1e-5))
            dd = reg * np.maximum(max(1e-12, norm(diagonal)) * 1e-8, abs(diagonal))
            if isinstance(M, np.ndarray):
                M = M + np.diag(dd)
                Cs = sparse.csc_matrix(C)
            else:
                M = (M + sparse.diags(dd)).tocsc()
                Cs = C.tocsc()
            system = LinearSystem(M, Cs, reg=1e-12)
            v = system.solve(np.r_[rhs, crhs])
            dx = v[:p.n]
            eta = np.zeros(full_count)
            eta[selected] = v[p.n:]
            dz = mapping.dual_step(dx, eta, state)
            if not np.isfinite(dx).all() or not np.isfinite(dz).all():
                break
            alpha, accepted = 1.0, False
            for _ in range(25):
                xn, zn = x + alpha * dx, z + alpha * dz
                mn = mapping.evaluate(xn, zn)
                if mn <= merit * (1 - 1e-3 * alpha):
                    x, z = xn, zn
                    accepted = True
                    damping = max(1, damping / 3)
                    failures = 0
                    break
                alpha *= 0.5
            if not accepted:
                damping *= 100
                failures += 1
                if failures >= 3:
                    break
        except (RuntimeError, ValueError, linalg.LinAlgError):
            break
    return best
