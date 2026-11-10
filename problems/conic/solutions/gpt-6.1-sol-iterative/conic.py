"""Homogeneous primal-dual interior-point iterations for product cones."""

import os
import sys
import time

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla

from solver import norm, smat, svec, quality
from factor import factorize


def jordan(a, b):
    c = a[:, :1] * b + b[:, :1] * a
    c[:, 0] = np.einsum("ij,ij->i", a, b)
    return c


def soc_power(a, p):
    r = np.linalg.norm(a[:, 1:], axis=1)
    hi = np.maximum(a[:, 0] + r, 1e-300) ** p
    lo = np.maximum(a[:, 0] - r, 1e-300) ** p
    out = np.empty_like(a)
    out[:, 0] = (hi + lo) / 2.0
    out[:, 1:] = a[:, 1:] * ((hi - lo) / (2.0 * np.maximum(r, 1e-300)))[:, None]
    return out


def soc_q(a):
    d = a.shape[1]
    det = (a[:, 0] - np.linalg.norm(a[:, 1:], axis=1)) * (
        a[:, 0] + np.linalg.norm(a[:, 1:], axis=1))
    J = np.eye(d)
    J[0, 0] = -1.0
    return 2.0 * a[:, :, None] * a[:, None, :] + det[:, None, None] * J


def soc_lsolve(a, b):
    det = (a[:, 0] - np.linalg.norm(a[:, 1:], axis=1)) * (
        a[:, 0] + np.linalg.norm(a[:, 1:], axis=1))
    c = np.empty_like(a)
    c[:, 0] = (a[:, 0] * b[:, 0] - np.einsum(
        "ij,ij->i", a[:, 1:], b[:, 1:])) / det
    c[:, 1:] = (b[:, 1:] - a[:, 1:] * c[:, :1]) / a[:, :1]
    return c


def mv(a, x):
    return np.einsum("ijk,ik->ij", a, x)


def exp_barrier(s):
    x, y, z = s.T
    log = np.log(z / y)
    g = y * log - x
    v = np.column_stack((-np.ones(x.size), log - 1.0, y / z))
    grad = -v / g[:, None]
    grad[:, 1] -= 1.0 / y
    grad[:, 2] -= 1.0 / z
    H = v[:, :, None] * v[:, None, :] / g[:, None, None] ** 2
    H[:, 1, 1] += 1.0 / (g * y) + 1.0 / y ** 2
    H[:, 1, 2] -= 1.0 / (g * z)
    H[:, 2, 1] -= 1.0 / (g * z)
    H[:, 2, 2] += y / (g * z ** 2) + 1.0 / z ** 2
    return grad, H


class ConeSystem:
    def __init__(self, blocks, A):
        self.blocks = blocks
        self.m, self.n = A.shape
        self.eq = np.array([j for k, _, a, b in blocks if k == "zero"
                            for j in range(a, b)], dtype=int)
        self.nn = np.array([j for k, _, a, b in blocks if k == "nonnegative"
                            for j in range(a, b)], dtype=int)
        groups = {}
        self.psds = []
        exp = []
        self.nu = self.nn.size
        for k, d, a, b in blocks:
            if k == "soc":
                groups.setdefault(d, []).append(np.arange(a, b))
                self.nu += 2
            elif k == "exponential":
                exp.append(np.arange(a, b))
                self.nu += 3
            elif k == "psd":
                V = A[a:b].tocsc()
                rows = np.unique(V.indices)
                VV = V[rows].tocsc()
                ii, jj = np.tril_indices(d)
                self.psds.append({
                    "d": d, "a": a, "b": b, "V": VV, "rows": rows,
                    "ii": ii[rows], "jj": jj[rows],
                    "wt": np.where(ii[rows] == jj[rows], 1.0 / np.sqrt(2.0), 1.0),
                })
                self.nu += d
        self.socs = [(d, np.array(v)) for d, v in groups.items()]
        self.exp = np.array(exp, dtype=int).reshape(-1, 3)
        rr, cc = [self.nn], [self.nn]
        for d, ids in self.socs:
            rr.append(np.broadcast_to(ids[:, :, None], (ids.shape[0], d, d)).ravel())
            cc.append(np.broadcast_to(ids[:, None, :], (ids.shape[0], d, d)).ravel())
        if self.exp.size:
            ids = self.exp
            rr.append(np.broadcast_to(ids[:, :, None], (ids.shape[0], 3, 3)).ravel())
            cc.append(np.broadcast_to(ids[:, None, :], (ids.shape[0], 3, 3)).ravel())
        self.rr, self.cc = np.concatenate(rr), np.concatenate(cc)
        self.A = A

    def initial(self):
        s = np.zeros(self.m)
        z = np.zeros(self.m)
        s[self.nn] = z[self.nn] = 1.0
        for _, ids in self.socs:
            s[ids[:, 0]] = z[ids[:, 0]] = np.sqrt(2.0)
        if self.exp.size:
            s[self.exp] = np.array([-1.0, 1.0, 1.0])
            z[self.exp] = np.array([-1.0, 0.0, 2.0])
        for p in self.psds:
            s[p["a"]:p["b"]] = z[p["a"]:p["b"]] = svec(np.eye(p["d"]))
        return s, z

    def update(self, s, z, mu):
        self.s, self.z, self.mu = s, z, mu
        data = [z[self.nn] / s[self.nn]]
        inverse_data = [s[self.nn] / z[self.nn]]
        self.soc_state = []
        arrow_diag = np.zeros(self.m)
        arrow_diag[self.nn] = -s[self.nn] / z[self.nn]
        gr, gc, gv, auxiliary = [], [], [], 0
        for d, ids in self.socs:
            ss, zz = s[ids], z[ids]
            ds = ((ss[:, 0] - np.linalg.norm(ss[:, 1:], axis=1)) *
                  (ss[:, 0] + np.linalg.norm(ss[:, 1:], axis=1)))
            dz = ((zz[:, 0] - np.linalg.norm(zz[:, 1:], axis=1)) *
                  (zz[:, 0] + np.linalg.norm(zz[:, 1:], axis=1)))
            if min(ds.min(), dz.min()) <= 0:
                raise ArithmeticError("SOC interior lost")
            sn, zn = ss / np.sqrt(ds)[:, None], zz / np.sqrt(dz)[:, None]
            gamma = np.sqrt((1.0 + np.einsum("ij,ij->i", sn, zn)) / 2.0)
            zn[:, 1:] *= -1
            w = (sn + zn) / (2.0 * gamma[:, None]) * (ds / dz)[:, None] ** 0.25
            eta2 = np.sqrt(ds / dz)
            arrow_diag[ids] = -eta2[:, None]
            arrow_diag[ids[:, 0]] = eta2
            gr.append(ids.ravel())
            gc.append(np.repeat(np.arange(ids.shape[0]) + auxiliary, d))
            gv.append((np.sqrt(2.0) * w).ravel())
            auxiliary += ids.shape[0]
            rootw = soc_power(w, 0.5)
            W, Wi = soc_q(rootw), soc_q(soc_power(rootw, -1.0))
            H = np.einsum("ijk,ikl->ijl", Wi, Wi)
            lam = mv(W, zz)
            data.append(H.ravel())
            inverse_data.append(np.einsum("ijk,ikl->ijl", W, W).ravel())
            self.soc_state.append((ids, W, Wi, lam))
        if self.exp.size:
            self.exp_g, self.exp_H = exp_barrier(s[self.exp])
            data.append((mu * self.exp_H).ravel())
            inverse_data.append((np.linalg.inv(self.exp_H) / mu).ravel())
        self.H = sparse.coo_matrix(
            (np.concatenate(data), (self.rr, self.cc)),
            shape=(self.m, self.m)).tocsc()
        self.Hinv = sparse.coo_matrix(
            (np.concatenate(inverse_data), (self.rr, self.cc)),
            shape=(self.m, self.m)).tocsc()
        self.arrow_diag = arrow_diag
        self.auxiliary = auxiliary
        if auxiliary:
            self.arrow_G = sparse.coo_matrix(
                (np.concatenate(gv), (np.concatenate(gr), np.concatenate(gc))),
                shape=(self.m, auxiliary)).tocsc()
        for p in self.psds:
            a, b, d = p["a"], p["b"], p["d"]
            S, Z = smat(s[a:b], d), smat(z[a:b], d)
            L = linalg.cholesky(S, lower=True, check_finite=False)
            eig, U = linalg.eigh(L.T @ Z @ L, check_finite=False)
            if eig[0] <= 0:
                raise ArithmeticError("PSD interior lost")
            R = (L @ U) * eig[None, :] ** -0.25
            Ri = linalg.inv(R, check_finite=False)
            p["R"], p["Ri"], p["lam"] = R, Ri, np.sqrt(eig)
            p["Dinv"] = Ri.T @ Ri

    def apply(self, v):
        out = self.H @ v
        for p in self.psds:
            a, b, d, D = p["a"], p["b"], p["d"], p["Dinv"]
            out[a:b] = svec(D @ smat(v[a:b], d) @ D)
        return out

    def normal(self, P):
        out = P + self.A.T @ (self.H @ self.A)
        if not self.psds:
            return out.tocsc()
        out = out.toarray()
        for p in self.psds:
            R, i, j, w, V = p["Dinv"], p["ii"], p["jj"], p["wt"], p["V"]
            H = (R[i[:, None], i[None, :]] * R[j[:, None], j[None, :]]
                 + R[i[:, None], j[None, :]] * R[j[:, None], i[None, :]])
            H *= w[:, None] * w[None, :]
            HV = (V.T @ H.T).T
            out += V.T @ HV
        return (out + out.T) * 0.5

    def rhs(self, target, affine=None):
        s, z = self.s, self.z
        out = np.zeros(self.m)
        inverse = np.zeros(self.m)
        out[self.nn] = -z[self.nn] + target / s[self.nn]
        inverse[self.nn] = -s[self.nn] + target / z[self.nn]
        if affine is not None:
            ds, dz = affine
            out[self.nn] -= ds[self.nn] * dz[self.nn] / s[self.nn]
            inverse[self.nn] -= ds[self.nn] * dz[self.nn] / z[self.nn]
        for ids, W, Wi, lam in self.soc_state:
            rc = -jordan(lam, lam)
            rc[:, 0] += 2.0 * target
            if affine is not None:
                rc -= jordan(mv(Wi, ds[ids]), mv(W, dz[ids]))
            rr = soc_lsolve(lam, rc)
            out[ids] = mv(Wi, rr)
            inverse[ids] = mv(W, rr)
        if self.exp.size:
            out[self.exp] = -z[self.exp] - target * self.exp_g
            if affine is not None:
                # A directional Hessian derivative is the barrier's
                # second-order correction, evaluated without differencing f.
                dd = ds[self.exp]
                h = 1e-4
                for _ in range(15):
                    t = s[self.exp] + h * dd
                    if self.exp_interior(t):
                        break
                    h *= 0.5
                if h > 1e-8:
                    _, H1 = exp_barrier(t)
                    corr = 0.5 * self.mu / h * mv(H1 - self.exp_H, dd)
                    out[self.exp] -= corr
        for p in self.psds:
            a, b, d, Ri = p["a"], p["b"], p["d"], p["Ri"]
            lam = p["lam"]
            rc = np.diag(target - lam * lam)
            if affine is not None:
                R = p["R"]
                dss = Ri @ smat(ds[a:b], d) @ Ri.T
                dzz = R.T @ smat(dz[a:b], d) @ R
                rc -= (dss @ dzz + dzz @ dss) * 0.5
            rc *= 2.0 / (lam[:, None] + lam[None, :])
            out[a:b] = svec(Ri.T @ rc @ Ri)
        self.inverse_rhs = inverse
        return out

    @staticmethod
    def exp_interior(s):
        x, y, z = s.T
        if np.any(y <= 0) or np.any(z <= 0):
            return False
        return np.all(y * (np.log(z) - np.log(y)) > x)

    def step(self, v, dv, dual=False):
        alpha = 1.0
        bad = dv[self.nn] < 0
        if np.any(bad):
            alpha = min(alpha, float(np.min(-v[self.nn][bad] / dv[self.nn][bad])))
        exact = os.environ.get("SOLVER_EXACT_STEP") == "1"
        for _, ids in self.socs if exact else []:
            ss, dd = v[ids], dv[ids]
            scale = np.maximum(np.max(np.abs(ss), axis=1),
                               np.max(np.abs(dd), axis=1))
            scale = np.maximum(scale, 1e-300)
            ss, dd = ss / scale[:, None], dd / scale[:, None]
            sn, dn = np.linalg.norm(ss[:, 1:], axis=1), np.linalg.norm(dd[:, 1:], axis=1)
            c = (ss[:, 0] - sn) * (ss[:, 0] + sn)
            a = (dd[:, 0] - dn) * (dd[:, 0] + dn)
            b = ss[:, 0] * dd[:, 0] - np.einsum("ij,ij->i", ss[:, 1:], dd[:, 1:])
            disc = b * b - a * c
            has_root = (a < 0) | ((b < 0) & (disc >= 0))
            denominator = -b + np.sqrt(np.maximum(disc, 0.0))
            root = np.full(c.size, np.inf)
            np.divide(c, denominator, out=root, where=has_root & (denominator > 0))
            alpha = min(alpha, float(np.min(root)))
            negative_t = dd[:, 0] < 0
            if np.any(negative_t):
                alpha = min(alpha, float(np.min(-ss[negative_t, 0] / dd[negative_t, 0])))
        for p in self.psds if exact else []:
            a, b, d = p["a"], p["b"], p["d"]
            L = linalg.cholesky(smat(v[a:b], d), lower=True, check_finite=False)
            U = linalg.solve_triangular(L, smat(dv[a:b], d), lower=True, check_finite=False)
            U = linalg.solve_triangular(L, U.T, lower=True, check_finite=False).T
            eig = linalg.eigh((U + U.T) * 0.5, subset_by_index=[0, 0],
                              eigvals_only=True, check_finite=False)[0]
            if eig < 0:
                alpha = min(alpha, -1.0 / eig)
        if not self.exp.size and (exact or (not self.socs and not self.psds)):
            return max(0.0, alpha)
        for _ in range(60):
            t = v + alpha * dv
            okay = True
            for _, ids in self.socs:
                vv = t[ids]
                if np.any(vv[:, 0] <= np.linalg.norm(vv[:, 1:], axis=1)):
                    okay = False
                    break
            if okay and self.exp.size:
                vv = t[self.exp]
                if dual:
                    vv = np.column_stack((-vv[:, 1], -vv[:, 0], np.e * vv[:, 2]))
                okay = self.exp_interior(vv)
            if okay:
                for p in self.psds:
                    try:
                        linalg.cholesky(smat(t[p["a"]:p["b"]], p["d"]),
                                        lower=True, check_finite=False)
                    except linalg.LinAlgError:
                        okay = False
                        break
            if okay:
                break
            alpha *= 0.8
        return alpha


def equilibrate(p, iterations=4, cost_mode=None):
    A, P = p.A.copy(), p.P.copy()
    D, E = np.ones(p.n), np.ones(p.m)
    for _ in range(iterations):
        ca = np.asarray(abs(A).max(axis=0).toarray()).ravel()
        cp = np.asarray(abs(P).max(axis=0).toarray()).ravel()
        cn = np.maximum(ca, cp)
        cn[cn == 0] = 1.0
        d = 1.0 / np.sqrt(np.maximum(cn, 1e-4))
        d = np.clip(d, 1e-2, 1e2)
        A = A.multiply(d).tocsc()
        P = P.multiply(d).multiply(d[:, None]).tocsc()
        D *= d
        ra = np.asarray(abs(A).max(axis=1).toarray()).ravel()
        for k, _, a, b in p.blocks:
            if k not in ("zero", "nonnegative"):
                ra[a:b] = max(np.max(ra[a:b]), 1e-4)
        ra[ra == 0] = 1.0
        e = 1.0 / np.sqrt(np.maximum(ra, 1e-4))
        e = np.clip(e, 1e-2, 1e2)
        A = A.multiply(e[:, None]).tocsc()
        E *= e
    cscale = max(1.0, norm(p.q * D), norm(P.data))
    default = "full" if any(k in ("soc", "psd") for k, _, _, _ in p.blocks) else "sqrt"
    if (cost_mode or os.environ.get("SOLVER_COST", default)) == "sqrt":
        cscale = np.sqrt(cscale)
    cost = 1.0 / cscale
    return A, P * cost, p.q * D * cost, p.b * E, D, E, cost


def solve_conic(p, maxiter=100, legacy=False):
    A, P, q, b, D, E, cost = equilibrate(p, cost_mode="full" if legacy else None)
    cs = ConeSystem(p.blocks, A)
    eq = cs.eq
    Ae = A[eq, :].tocsc()
    be = b[eq]
    n, ne = p.n, eq.size
    x = np.zeros(n)
    s, z = cs.initial()
    tau, kap = 1.0, 1.0
    started = time.monotonic()
    debug = os.environ.get("SOLVER_DEBUG")
    best = {"status": "unresolved"}
    best_error = np.inf
    cached_factor = None
    last_polish = -5

    def recover():
        return {"status": "solved", "x": D * (x / tau),
                "s": (s / tau) / E, "z": E * (z / tau) / cost}

    for it in range(maxiter):
        if time.monotonic() - started > 245:
            break
        if (not legacy and not cs.socs and not cs.psds and tau > 0 and np.isfinite(tau)
                and (tau < 0.5 or tau > 2.0)):
            homogeneous_scale = 1.0 / tau
            if np.isfinite(homogeneous_scale) and homogeneous_scale < 1e100:
                x *= homogeneous_scale
                s *= homogeneous_scale
                z *= homogeneous_scale
                kap *= homogeneous_scale
                tau = 1.0
        px = P @ x
        rd, rp = px + q * tau + A.T @ z, A @ x + s - b * tau
        xpx = x @ px
        rg = q @ x + b @ z + kap + xpx / tau
        mu = (s @ z + tau * kap) / (cs.nu + 1.0)
        if debug:
            print("ipm", it, "mu", mu, "tau", tau, "kap", kap,
                  "res", max(norm(rd), norm(rp), abs(rg)), file=sys.stderr)
        if tau > 1e-12:
            r = recover()
            if quality(p, r):
                return r
            candidate_error = max(norm(rd), norm(rp), abs(rg)) / tau
            if candidate_error < best_error:
                best_error = candidate_error
                best = r
            if (p.n < 500 and it >= 7 and it - last_polish >= 3
                    and not cs.socs and not cs.psds and not cs.exp.size):
                xx, ss, zz = r["x"], r["s"], r["z"]
                ax0, px0, az0 = p.A @ xx, p.P @ xx, p.A.T @ zz
                rr_p = norm(ax0 + ss - p.b) / (1 + max(norm(ax0), norm(ss), norm(p.b)))
                rr_d = norm(px0 + p.q + az0) / (1 + max(norm(px0), norm(p.q), norm(az0)))
                pp0 = 0.5 * xx @ px0 + p.q @ xx
                dd0 = -0.5 * xx @ px0 - p.b @ zz
                rr_g = abs(pp0 - dd0) / max(1.0, abs(pp0), abs(dd0))
                if max(rr_p, rr_d) < 1e-6 and rr_g < 1e-5:
                    last_polish = it
                    if cs.socs:
                        from refine import polish_soc
                        polished = polish_soc(p, xx, zz)
                    else:
                        from solver import polish_qp
                        polished = polish_qp(p, xx, zz)
                    if polished is not None and quality(p, polished):
                        return polished
        if it > 3:
            r = {"status": "primal_infeasible", "z": E * z / cost}
            if quality(p, r):
                return r
            r = {"status": "dual_infeasible", "x": D * x, "s": s / E}
            if quality(p, r):
                return r
        if not (mu > 0 and tau > 0 and kap > 0):
            break
        try:
            cs.update(s, z, mu)
            default_system = "normal"
            system = os.environ.get("SOLVER_SYSTEM", default_system)
            arrow = bool(cs.socs) and (system == "arrow" or
                                      (system == "hybrid" and mu < 1e-5))
            augmented = (not cs.psds and not cs.exp.size and
                         (arrow or system == "aug" or (system == "auto" and mu < 1e-5)))
            M = None if augmented else cs.normal(P)
            if augmented:
                na = cs.auxiliary if arrow else 0
                if arrow:
                    K0 = sparse.bmat([
                        [P, A.T, sparse.csc_matrix((n, na))],
                        [A, sparse.diags(cs.arrow_diag, format="csc"), cs.arrow_G],
                        [sparse.csc_matrix((na, n)), cs.arrow_G.T, sparse.eye(na, format="csc")]],
                        format="csc")
                else:
                    K0 = sparse.bmat([[P, A.T], [A, -cs.Hinv]], format="csc")
                mx = np.asarray(abs(K0).max(axis=0).toarray()).ravel()
                kk_scale = np.clip(1.0 / np.sqrt(np.maximum(mx, 1e-8)), 1e-6, 1e6)
                KK = K0.multiply(kk_scale[:, None]).multiply(kk_scale).tocsc()
                reg = np.r_[np.full(n, 1e-12), np.full(p.m, -1e-12), np.full(na, 1e-12)]
                K = KK + sparse.diags(reg, format="csc")
                lu = sla.splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1,
                              options={"SymmetricMode": True})

                def lin(rhs):
                    rr = rhs * kk_scale
                    vv = lu.solve(rr)
                    for _ in range(5):
                        residual = rr - KK @ vv
                        if norm(residual) < max(1e-15, 1e-12 * norm(rr)):
                            break
                        vv += lu.solve(residual)
                    return vv * kk_scale
            elif sparse.issparse(M) and M.nnz == n and np.all(M.indices == np.arange(n)):
                md = M.diagonal()
                inv = 1.0 / (md + 1e-13)
                if ne:
                    S = (Ae.multiply(inv) @ Ae.T).tocsc()
                    lu = sla.splu(S + sparse.eye(ne, format="csc") * 1e-13,
                                  permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                  options={"SymmetricMode": True})

                def base(rhs):
                    rx, re = rhs[:n], rhs[n:]
                    y = lu.solve(Ae @ (inv * rx) - re) if ne else np.empty(0)
                    xx = inv * (rx - Ae.T @ y)
                    return np.r_[xx, y]

                def lin(rhs):
                    v = base(rhs)
                    for _ in range(2):
                        xx, y = v[:n], v[n:]
                        res = rhs - np.r_[md * xx + Ae.T @ y, Ae @ xx]
                        if norm(res) <= 1e-11 * max(1.0, norm(rhs)):
                            break
                        v += base(res)
                    return v
            elif sparse.issparse(M):
                reg = 1e-11
                K0 = sparse.bmat([[M, Ae.T], [Ae, sparse.csc_matrix((ne, ne))]],
                                 format="csc")
                diagonal = np.r_[np.full(n, reg), np.full(ne, -reg)]
                K = K0 + sparse.diags(diagonal, format="csc")
                if legacy:
                    lu = sla.splu(K, permc_spec="COLAMD")
                else:
                    lu, cached_factor = factorize(K, n, cached_factor)

                def lin(rhs):
                    v = lu.solve(rhs)
                    for _ in range(2):
                        res = rhs - K0 @ v
                        if norm(res) <= 1e-11 * max(1.0, norm(rhs)):
                            break
                        v += lu.solve(res)
                    return v
            else:
                K0 = np.block([[M, Ae.T.toarray()],
                               [Ae.toarray(), np.zeros((ne, ne))]])
                K = K0.copy()
                K.flat[::K.shape[0] + 1] += np.r_[np.full(n, 1e-12), np.full(ne, -1e-12)]
                lu = linalg.lu_factor(K, check_finite=False)

                def lin(rhs):
                    v = linalg.lu_solve(lu, rhs, check_finite=False)
                    for _ in range(2):
                        v += linalg.lu_solve(lu, rhs - K0 @ v, check_finite=False)
                    return v

            if augmented:
                vv = lin(np.r_[q, -b, np.zeros(na)])
                vx, vz = vv[:n], vv[n:n+p.m]
                az_tau = -vz
            else:
                hb = cs.apply(b)
                vv = lin(np.r_[q - A.T @ hb, -be])
                vx, vz = vv[:n], vv[n:]
                az_tau = -cs.apply(A @ vx + b)
                az_tau[eq] = -vz
            gx = q + 2 * px / tau
            scalar_d = -gx @ vx + b @ az_tau - xpx / tau ** 2 - kap / tau

            def direction(rc, rct, residual_scale=1.0, rg_correction=0.0):
                if augmented:
                    rhs = np.r_[-rd * residual_scale,
                                -rp * residual_scale - cs.inverse_rhs, np.zeros(na)]
                    u = lin(rhs)
                    ux, zz = u[:n], u[n:n+p.m]
                else:
                    h = rc + cs.apply(rp * residual_scale)
                    rhs = np.r_[-rd * residual_scale - A.T @ h,
                                -rp[eq] * residual_scale]
                    u = lin(rhs)
                    ux, uz = u[:n], u[n:]
                    zz = h + cs.apply(A @ ux)
                    zz[eq] = uz
                dg0 = gx @ ux + b @ zz
                dt = (-rg * residual_scale - rg_correction - dg0 - rct / tau) / scalar_d
                dk = (rct - kap * dt) / tau
                dx = ux - vx * dt
                dz = zz + az_tau * dt
                ds = -rp * residual_scale - A @ dx + b * dt
                ds[eq] = 0.0
                return dx, ds, dz, dt, dk

            aff = direction(cs.rhs(0.0), -tau * kap)
            dx, ds, dz, dt, dk = aff
            aa = min(cs.step(s, ds), cs.step(z, dz, True),
                     -tau / dt if dt < 0 else 1.0,
                     -kap / dk if dk < 0 else 1.0, 1.0)
            ma = ((s + aa * ds) @ (z + aa * dz) +
                  (tau + aa * dt) * (kap + aa * dk)) / (cs.nu + 1.0)
            floor = 0.0 if legacy or cs.socs or cs.psds else 0.01
            sigma = min(0.9, max(floor, max(0.0, ma / mu) ** 3))
            if cs.exp.size:
                sigma = max(0.15, sigma)
            rct = sigma * mu - tau * kap - dt * dk
            corr_q = ((dx - x * dt / tau) @ (P @ (dx - x * dt / tau))) / tau
            direct = direction(cs.rhs(sigma * mu, (ds, dz)), rct,
                               1.0 - sigma, corr_q)
            dx, ds, dz, dt, dk = direct
            alpha = min(cs.step(s, ds), cs.step(z, dz, True),
                        -tau / dt if dt < 0 else 1.0,
                        -kap / dk if dk < 0 else 1.0, 1.0)
            alpha *= 0.99
            if alpha < 1e-10 or not np.isfinite(alpha):
                break
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kap += alpha * dk
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError):
            if debug:
                import traceback
                traceback.print_exc()
            break
    if quality(p, best, tol=1e-8):
        return best
    if cs.socs and p.n <= 5000 and best.get("status") == "solved":
        from refine import polish_soc
        try:
            polished = polish_soc(p, best["x"], best["z"])
            if polished is not None and quality(p, polished):
                return polished
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
    if cs.psds:
        from certificates import refine_primal
        refined = refine_primal(p, E * z / cost)
        if refined is not None:
            return refined
    return {"status": "unresolved"}


def primal_barrier(p, maxiter=160):
    """Damped Newton continuation for the nonsymmetric logarithmic barrier."""
    A, P, q, b, D, E, cost = equilibrate(p)
    cs = ConeSystem(p.blocks, A)
    eq, nn, ee = cs.eq, cs.nn, cs.exp
    Ae = A[eq, :].tocsc()
    n, ne = p.n, eq.size
    x = np.zeros(n)
    s, z = cs.initial()
    z[eq] = 0.0
    mu = 1.0
    started = time.monotonic()
    debug = os.environ.get("SOLVER_DEBUG")

    def derivatives(s):
        g = np.zeros(p.m)
        g[nn] = -1.0 / s[nn]
        H = [1.0 / s[nn] ** 2]
        if ee.size:
            ge, He = exp_barrier(s[ee])
            g[ee] = ge
            H.append(He.ravel())
        return g, sparse.coo_matrix((np.concatenate(H), (cs.rr, cs.cc)),
                                    shape=(p.m, p.m)).tocsc()

    def residual(x, s, z, g, mu):
        rd, rp = P @ x + q + A.T @ z, A @ x + s - b
        rc = z + mu * g
        rc[eq] = 0
        value = (np.linalg.norm(rd) ** 2 + np.linalg.norm(rp) ** 2
                 + np.linalg.norm(rc) ** 2)
        return rd, rp, rc, value

    for it in range(maxiter):
        if time.monotonic() - started > 45:
            break
        r = {"status": "solved", "x": D * x, "s": s / E, "z": E * z / cost}
        if quality(p, r):
            return r
        try:
            g, H0 = derivatives(s)
            H = mu * H0
            rd, rp, rc, val = residual(x, s, z, g, mu)
            central = max(norm(rd), norm(rp), norm(rc))
            if debug:
                print("barrier", it, "mu", mu, "res", central, file=sys.stderr)
            if central < max(1e-10, min(0.05, 0.1 * np.sqrt(mu))):
                mu *= 0.15
                if mu < 1e-15:
                    break
                rc = z + mu * g
                rc[eq] = 0.0
                H = mu * H0
                val = residual(x, s, z, g, mu)[3]
            M = P + A.T @ (H @ A)
            K0 = sparse.bmat([[M, Ae.T], [Ae, sparse.csc_matrix((ne, ne))]],
                             format="csc")
            reg = np.r_[np.full(n, 1e-12), np.full(ne, -1e-12)]
            lu = sla.splu(K0 + sparse.diags(reg, format="csc"),
                          permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                          options={"SymmetricMode": True})
            hh = -rc + H @ rp
            rhs = np.r_[-rd - A.T @ hh, -rp[eq]]
            u = lu.solve(rhs)
            for _ in range(2):
                u += lu.solve(rhs - K0 @ u)
            dx = u[:n]
            ds = -rp - A @ dx
            ds[eq] = 0.0
            dz = hh + H @ (A @ dx)
            dz[eq] = u[n:]
            alpha = min(1.0, 0.99 * cs.step(s, ds))
            for _ in range(45):
                sx = s + alpha * ds
                gx, _ = derivatives(sx)
                newval = residual(x + alpha * dx, sx, z + alpha * dz, gx, mu)[3]
                if np.isfinite(newval) and newval < (1.0 - 0.001 * alpha) * val:
                    break
                alpha *= 0.5
            if alpha < 1e-10:
                break
            x += alpha * dx
            s = sx
            z += alpha * dz
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError):
            if debug:
                import traceback
                traceback.print_exc()
            break
    return {"status": "unresolved"}
