"""Primal-dual path following for products of small symmetric/exp cones."""

import math
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu

from ipm import equilibrate, infnorm, maxstep


def jordan(a, b):
    out = a * b[:, :1] + b * a[:, :1]
    out[:, 0] = np.sum(a * b, axis=1)
    return out


def socdet(v):
    r = np.linalg.norm(v[:, 1:], axis=1)
    return np.maximum((v[:, 0] - r) * (v[:, 0] + r), 1e-300)


def socinv(v):
    out = -v.copy()
    out[:, 0] = v[:, 0]
    return out / socdet(v)[:, None]


def socpower(v, power):
    r = np.linalg.norm(v[:, 1:], axis=1)
    lp = np.maximum(v[:, 0] + r, 1e-300) ** power
    lm = np.maximum(v[:, 0] - r, 1e-300) ** power
    out = np.empty_like(v)
    out[:, 0] = 0.5 * (lp + lm)
    f = 0.5 * (lp - lm) / np.maximum(r, 1e-300)
    out[:, 1:] = v[:, 1:] * f[:, None]
    return out


def qapply(w, v, determinant=None):
    a, u = w[:, 0], w[:, 1:]
    b, x = v[:, 0], v[:, 1:]
    uu = np.sum(u * u, axis=1)
    ux = np.sum(u * x, axis=1)
    out = np.empty_like(v)
    out[:, 0] = (a * a + uu) * b + 2 * a * ux
    det = a * a - uu if determinant is None else determinant
    out[:, 1:] = (det[:, None] * x +
                 2 * (a * b + ux)[:, None] * u)
    return out


def qmatrix(w, determinant=None):
    a, u = w[:, 0], w[:, 1:]
    d = w.shape[1]
    H = 2 * w[:, :, None] * w[:, None, :]
    v2 = np.sum(u * u, axis=1)
    H[:, 0, 0] = a * a + v2
    det = a * a - v2 if determinant is None else determinant
    H[:, np.arange(1, d), np.arange(1, d)] += det[:, None]
    return H


def jordan_solve(a, v, determinant=None):
    out = np.empty_like(v)
    det = socdet(a) if determinant is None else determinant
    out[:, 0] = (a[:, 0] * v[:, 0] - np.sum(a[:, 1:] * v[:, 1:], axis=1)) / det
    out[:, 1:] = (v[:, 1:] - out[:, :1] * a[:, 1:]) / a[:, :1]
    return out


def jordan_matrix(v, groups, size):
    rr, cc, vv = [], [], []
    for d, ix in groups.items():
        a = v[ix]
        rr.extend((ix.ravel(), np.repeat(ix[:, 0], d - 1), ix[:, 1:].ravel()))
        cc.extend((ix.ravel(), ix[:, 1:].ravel(), np.repeat(ix[:, 0], d - 1)))
        vv.extend((np.repeat(a[:, 0], d), a[:, 1:].ravel(), a[:, 1:].ravel()))
    return sparse.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                              shape=(size, size))


def exp_barrier(v):
    u, y, z = v.T
    L = np.log(z) - np.log(y)
    t = y * L - u
    a = np.column_stack((-np.ones(len(v)), L - 1, y / z))
    g = -a / t[:, None]
    g[:, 1] -= 1 / y
    g[:, 2] -= 1 / z
    H = a[:, :, None] * a[:, None, :] / t[:, None, None] ** 2
    H[:, 1, 1] += 1 / (y * t) + 1 / y ** 2
    H[:, 1, 2] -= 1 / (z * t)
    H[:, 2, 1] -= 1 / (z * t)
    H[:, 2, 2] += y / (z ** 2 * t) + 1 / z ** 2
    return g, H


def exp_interior(v):
    u, y, z = v.T
    if np.any(y <= 0) or np.any(z <= 0):
        return False
    return bool(np.all(y * (np.log(z) - np.log(y)) - u > 0))


class Product:
    def __init__(self, p):
        self.nz = np.flatnonzero(p.kind != 0)
        where = np.full(p.m, -1, dtype=np.int64)
        where[self.nz] = np.arange(len(self.nz))
        self.nn = where[p.ineq]
        groups = {}
        exp = []
        for t, d, lo, hi in p.nonlinear:
            if t == "soc":
                groups.setdefault(d, []).append(np.arange(where[lo], where[lo] + d))
            elif t == "exponential":
                exp.append(np.arange(where[lo], where[lo] + 3))
        self.soc = {d: np.array(v, dtype=np.int64) for d, v in groups.items()}
        self.exp = np.array(exp, dtype=np.int64).reshape(-1, 3)
        self.nu = len(self.nn) + sum(len(v) for v in self.soc.values()) + 3 * len(self.exp)
        rr, cc = [self.nn], [self.nn]
        for ix in list(self.soc.values()) + ([self.exp] if len(self.exp) else []):
            d = ix.shape[1]
            rr.append(np.repeat(ix, d, axis=1).ravel())
            cc.append(np.tile(ix, (1, d)).ravel())
        self.rr, self.cc = np.concatenate(rr), np.concatenate(cc)
        self.cache = None

    def initial(self, b):
        s, z = np.ones(len(self.nz)), np.ones(len(self.nz))
        s[self.nn] = np.maximum(np.abs(b[self.nn]), 1.0)
        z[self.nn] = 1 / s[self.nn]
        for ix in self.soc.values():
            v = b[ix].copy()
            v[:, 0] = np.maximum(np.abs(v[:, 0]), np.linalg.norm(v[:, 1:], axis=1)) + 1
            s[ix], z[ix] = v, socinv(v)
        if len(self.exp):
            v = b[self.exp].copy()
            v[:, 1:] = np.maximum(np.abs(v[:, 1:]), 1)
            v[:, 0] = np.minimum(v[:, 0], v[:, 1] * np.log(v[:, 2] / v[:, 1]) - 1)
            s[self.exp] = v
            z[self.exp] = -exp_barrier(v)[0]
        return s, z

    def scaling(self, s, z, mu):
        data = [z[self.nn] / s[self.nn]]
        idata = [s[self.nn] / z[self.nn]]
        rdata = [np.sqrt(z[self.nn] / s[self.nn])]
        cache = {}
        for d, ix in self.soc.items():
            a, b = s[ix], z[ix]
            ba, bb = np.sqrt(socdet(a)), np.sqrt(socdet(b))
            an, bn = a / ba[:, None], b / bb[:, None]
            gamma = np.sqrt(np.maximum(0.5 * (1 + np.sum(an * bn, axis=1)), 1e-300))
            aj = -an.copy()
            aj[:, 0] = an[:, 0]
            ratio = np.sqrt(bb / ba)
            w = ratio[:, None] * (bn + aj) / (2 * gamma[:, None])
            H = qmatrix(w, ratio ** 2)
            data.append(H.ravel())
            iw = -w.copy()
            iw[:, 0] = w[:, 0]
            iw /= ratio[:, None] ** 2
            idata.append(qmatrix(iw, 1 / ratio ** 2).ravel())
            wsqrt = w.copy()
            wsqrt[:, 0] += ratio
            wsqrt /= np.sqrt(2 * (w[:, 0] + ratio))[:, None]
            winvsqrt = -wsqrt.copy()
            winvsqrt[:, 0] = wsqrt[:, 0]
            winvsqrt /= ratio[:, None]
            rdata.append(qmatrix(wsqrt, ratio).ravel())
            lam = qapply(wsqrt, a, ratio)
            lam[:, 0] = np.sqrt(np.maximum((np.sum(a * b, axis=1) + ba * bb) / 2, 1e-300))
            lv = np.linalg.norm(lam[:, 1:], axis=1)
            target = np.sqrt(np.maximum(lam[:, 0] ** 2 - ba * bb, 0))
            lam[:, 1:] *= (target / np.maximum(lv, 1e-300))[:, None]
            cache[d] = (wsqrt, winvsqrt, lam, socinv(a), ratio, ba * bb)
        if len(self.exp):
            a, b = s[self.exp], z[self.exp]
            g, H = exp_barrier(a)
            hs = -mu * g
            sz = np.sum(a * b, axis=1)
            H *= mu
            H += b[:, :, None] * b[:, None, :] / np.maximum(sz, 1e-300)[:, None, None]
            H -= hs[:, :, None] * hs[:, None, :] / max(3 * mu, 1e-300)
            data.append(H.ravel())
            idata.append(np.zeros_like(H).ravel())
            rdata.append(np.zeros_like(H).ravel())
            cache["exp"] = g
        self.cache = cache
        self.Hinv = sparse.csc_matrix((np.concatenate(idata), (self.rr, self.cc)),
                                      shape=(len(s), len(s)))
        self.Hroot = sparse.csc_matrix((np.concatenate(rdata), (self.rr, self.cc)),
                                       shape=(len(s), len(s)))
        return sparse.csc_matrix((np.concatenate(data), (self.rr, self.cc)),
                                 shape=(len(s), len(s)))

    def rhs(self, s, z, mu, sigma, da=None):
        out = -z.copy()
        out[self.nn] += sigma * mu / s[self.nn]
        if da is not None:
            out[self.nn] -= da[0][self.nn] * da[1][self.nn] / s[self.nn]
        for d, ix in self.soc.items():
            ws, wi, lam, invs, ratio, detlam = self.cache[d]
            out[ix] += sigma * mu * invs
            if da is not None:
                ds = qapply(ws, da[0][ix], ratio)
                dz = qapply(wi, da[1][ix], 1 / ratio)
                out[ix] -= qapply(ws, jordan_solve(lam, jordan(ds, dz), detlam), ratio)
        if len(self.exp):
            out[self.exp] -= sigma * mu * self.cache["exp"]
        return out

    def scaled_rhs(self, s, z, mu, sigma, da=None):
        out = np.zeros_like(s)
        for d, ix in self.soc.items():
            ws, wi, lam, invs, ratio, detlam = self.cache[d]
            invlam = -lam.copy()
            invlam[:, 0] = lam[:, 0]
            invlam /= detlam[:, None]
            out[ix] = -lam + sigma * mu * invlam
            if da is not None:
                out[ix] -= jordan_solve(lam, jordan(da[0][ix], da[1][ix]), detlam)
        return out

    def step(self, v, dv, dual=False):
        alpha = maxstep(v[self.nn], dv[self.nn])
        for ix in self.soc.values():
            a, da = v[ix], dv[ix]
            aa = da[:, 0] ** 2 - np.sum(da[:, 1:] ** 2, axis=1)
            bb = 2 * (a[:, 0] * da[:, 0] - np.sum(a[:, 1:] * da[:, 1:], axis=1))
            cc = socdet(a)
            disc = bb ** 2 - 4 * aa * cc
            denom = -bb + np.sqrt(np.maximum(disc, 0))
            valid = (disc > 0) & (denom > 0)
            if np.any(valid):
                alpha = min(alpha, float(np.min(2 * cc[valid] / denom[valid])))
            alpha = min(alpha, maxstep(a[:, 0], da[:, 0]))
        if len(self.exp):
            a, da = v[self.exp], dv[self.exp]
            if dual:
                a, da = a[:, [1, 0, 2]].copy(), da[:, [1, 0, 2]].copy()
                a[:, :2] *= -1
                da[:, :2] *= -1
                a[:, 2] *= math.e
                da[:, 2] *= math.e
            alpha = min(alpha, maxstep(a[:, 1:].ravel(), da[:, 1:].ravel()))
            test = alpha
            for _ in range(100):
                if exp_interior(a + test * da):
                    break
                test *= 0.5
            else:
                return 0.0
            alpha = min(alpha, test)
        return min(1.0, max(0.0, alpha))


def solve_conic(p, maxiter=100, timelimit=100):
    from solver import accepted, residuals, log
    start = time.monotonic()
    prod = Product(p)
    P, A, q, b, D, E, cost = equilibrate(p.P, p.A, p.q, p.b, blocks=p.blocks)
    eq, nz = p.eq, prod.nz
    B, F = A[nz].tocsc(), A[eq].tocsc()
    Bt, Ft = B.T.tocsc(), F.T.tocsc()
    beq, biq = b[eq], b[nz]
    n, k = p.n, len(eq)
    si = np.flatnonzero(p.kind[nz] == 2)
    ni = prod.nn
    N, S = B[ni].tocsc(), B[si].tocsc()
    Nt = N.T.tocsc()
    local = np.full(len(nz), -1, dtype=int)
    local[si] = np.arange(len(si))
    soc_local = {d: local[ix] for d, ix in prod.soc.items()}
    x, y = np.zeros(n), np.zeros(k)
    s, z = prod.initial(biq)
    delta = 1e-11
    best = np.inf
    for it in range(maxiter):
        rd = P @ x + q + Ft @ y + Bt @ z
        rp, ri = F @ x - beq, B @ x + s - biq
        mu = float(s @ z) / max(prod.nu, 1)
        if it >= 2:
            xo = D * x
            zo, so = np.zeros(p.m), np.zeros(p.m)
            zo[eq], zo[nz] = E[eq] * y / cost, E[nz] * z / cost
            so[nz] = s / E[nz]
            r = {"status": "solved", "x": xo, "s": so, "z": zo}
            vals = residuals(p, r)
            err = max(vals)
            if it % 5 == 0:
                log("cone-ipm", it, vals, "mu", mu, "t", time.monotonic() - start)
            if err < 8e-9:
                return r
            best = min(best, err)
            if infnorm(zo) > 100:
                cert = {"status": "primal_infeasible", "z": zo}
                if accepted(p, cert, tol=5e-9):
                    return cert
            if infnorm(xo) > 100:
                cert = {"status": "dual_infeasible", "x": xo, "s": -p.A @ xo}
                if accepted(p, cert, tol=5e-9):
                    return cert
            if it > 15 and err > best * 1000:
                break
        if not np.isfinite(mu) or mu <= 0 or time.monotonic() - start > timelimit:
            break
        try:
            if prod.soc and not len(prod.exp):
                Ls = jordan_matrix(s[si], soc_local, len(si))
                Lz = jordan_matrix(z[si], soc_local, len(si))
                w = z[ni] / s[ni]
                G0 = (P + Nt @ N.multiply(w[:, None])).tocsc()
                K0 = sparse.bmat([
                    [G0, Ft, S.T],
                    [F, sparse.csc_matrix((k, k)), None],
                    [-Lz @ S, None, Ls],
                ], format="csc")
                row_sc = 1 / np.sqrt(np.maximum(
                    np.asarray(abs(K0).max(axis=1).toarray()).ravel(), 1e-12))
                col_sc = 1 / np.sqrt(np.maximum(
                    np.asarray(abs(K0).max(axis=0).toarray()).ravel(), 1e-12))
                scaled = sparse.diags(row_sc) @ K0 @ sparse.diags(col_sc)
                scaled += sparse.diags(np.r_[np.full(n, 1e-12),
                                              np.full(k, -1e-12), np.full(len(si), 1e-12)])
                lu = splu(scaled.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=0.1)

                def direction(rcent, physical_affine=None, scaled_affine=None, sigma=0):
                    rn = -z[ni] + sigma * mu / s[ni]
                    if physical_affine is not None:
                        rn -= physical_affine[2][ni] * physical_affine[3][ni] / s[ni]
                    rc = np.zeros(len(si))
                    for d, ix in soc_local.items():
                        oldix = si[ix]
                        vrc = -jordan(s[oldix], z[oldix])
                        vrc[:, 0] += sigma * mu
                        if physical_affine is not None:
                            vrc -= jordan(physical_affine[2][oldix], physical_affine[3][oldix])
                        rc[ix] = vrc
                    rhs = np.r_[-rd - Nt @ (rn + w * ri[ni]), -rp,
                                rc + Lz @ ri[si]]
                    rscaled = row_sc * rhs
                    u = col_sc * lu.solve(rscaled)
                    for _ in range(5):
                        rr = row_sc * (rhs - K0 @ u)
                        if infnorm(rr / (1 + np.abs(rscaled))) < 1e-13:
                            break
                        du = col_sc * lu.solve(rr)
                        if infnorm(du) > 1e5 * max(infnorm(u), 1):
                            break
                        u += du
                    dx, dy, zs = u[:n], u[n:n + k], u[n + k:]
                    ds = -ri - B @ dx
                    dz = np.zeros_like(z)
                    dz[ni], dz[si] = rn - w * ds[ni], zs
                    return dx, dy, ds, dz, None, None
            else:
                H = prod.scaling(s, z, mu)
                G = (P + Bt @ H @ B).tocsc()
                reg = delta * np.maximum(np.abs(G.diagonal()), 1)
                if k <= 20:
                    glu = splu(G + sparse.diags(reg), permc_spec="MMD_AT_PLUS_A",
                                diag_pivot_thresh=0.01)
                    GF = glu.solve(Ft.toarray())
                    Schur = F @ GF + delta * np.eye(k)

                    def solve(rhs):
                        u = glu.solve(rhs[:n])
                        dy = linalg.solve(Schur, F @ u - rhs[n:],
                                          assume_a="sym", check_finite=False)
                        return np.r_[u - GF @ dy, dy]
                else:
                    K = sparse.bmat([
                        [G + sparse.diags(reg), Ft],
                        [F, -delta * sparse.eye(k, format="csc")],
                    ], format="csc")
                    lu = splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.01)
                    solve = lu.solve

                def direction(rcent):
                    v = rcent + H @ ri
                    rhs = np.r_[-rd - Bt @ v, -rp]
                    u = solve(rhs)
                    for _ in range(3):
                        rlin = rhs - np.r_[G @ u[:n] + Ft @ u[n:], F @ u[:n]]
                        if infnorm(rlin) < 5e-12 * max(infnorm(rhs), 1):
                            break
                        du = solve(rlin)
                        if infnorm(du) > 1e5 * max(infnorm(u), 1):
                            break
                        u += du
                    dx, dy = u[:n], u[n:]
                    ds = -ri - B @ dx
                    dz = rcent - H @ ds
                    return dx, dy, ds, dz

            scaled_mode = bool(prod.soc and not len(prod.exp))
            da = direction(None) if scaled_mode else direction(prod.rhs(s, z, mu, 0))
            ap, ad = prod.step(s, da[2]), prod.step(z, da[3], dual=True)
            mua = float((s + ap * da[2]) @ (z + ad * da[3])) / max(prod.nu, 1)
            sigma = min(1.0, max(0.0, mua / mu) ** 3)
            if len(prod.exp):
                sigma = max(sigma, 0.08)
            if scaled_mode:
                dc = direction(None, da, (da[4], da[5]), sigma)
                dx, dy, ds, dz = dc[:4]
            else:
                dx, dy, ds, dz = direction(prod.rhs(s, z, mu, sigma, (da[2], da[3])))
            ap, ad = 0.99 * prod.step(s, ds), 0.99 * prod.step(z, dz, dual=True)
            if min(ap, ad) < 1e-12:
                break
            x += ap * dx
            s += ap * ds
            y += ad * dy
            z += ad * dz
            if not all(np.isfinite(v).all() for v in (x, y, s, z)):
                break
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            break
    return None
