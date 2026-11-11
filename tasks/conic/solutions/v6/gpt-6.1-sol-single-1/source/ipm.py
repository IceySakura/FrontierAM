"""Homogeneous primal-dual interior-point method for product cones."""

import os
import sys
import time
import warnings
import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as sla
from common import norm, assess, equilibrate
from cones import ProductCone
from presolve import Presolve, AffineReduction


class LinearSystem:
    def __init__(self, M, C, reg=1e-10):
        self.n, self.k = M.shape[0], C.shape[0]
        self.dense = isinstance(M, np.ndarray)
        diagonal = np.diag(M) if self.dense else M.diagonal()
        primal_scale = 1 / np.sqrt(np.maximum(np.abs(diagonal), 1e-12))
        if self.k:
            row_norm = np.asarray(abs(C.multiply(primal_scale)).max(axis=1).toarray()).ravel()
            dual_scale = 1 / np.maximum(row_norm, 1e-8)
        else:
            dual_scale = np.zeros(0)
        self.scale = np.clip(np.r_[primal_scale, dual_scale], 1e-14, 1e14)
        if self.dense:
            K = np.block([[M, C.T.toarray()], [C.toarray(), np.zeros((self.k, self.k))]])
            self.K = K * self.scale[:, None] * self.scale[None, :]
            D = np.r_[np.full(self.n, reg), np.full(self.k, -reg)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fac = linalg.lu_factor(self.K + np.diag(D), check_finite=False)
            self.extended = self.K.astype(np.longdouble) if self.K.shape[0] <= 1400 else None
        else:
            K = sparse.bmat([[M, C.T], [C, sparse.csc_matrix((self.k, self.k))]], format="csc")
            self.K = K.multiply(self.scale).multiply(self.scale[:, None]).tocsc()
            D = np.r_[np.full(self.n, reg), np.full(self.k, -reg)]
            self.fac = sla.splu(self.K + sparse.diags(D), permc_spec="COLAMD")

    def solve(self, rhs):
        rhs = rhs * self.scale
        if self.dense:
            sol = linalg.lu_solve(self.fac, rhs, check_finite=False)
        else:
            sol = self.fac.solve(rhs)
        old = float("inf")
        extended = self.dense and self.extended is not None
        for _ in range(9 if extended else 5):
            res = (rhs.astype(np.longdouble) - self.extended @ sol.astype(np.longdouble)).astype(float) if extended else rhs - self.K @ sol
            err = norm(res) / (1 + norm(rhs))
            if err < (2e-17 if extended else 2e-13) or err > old * 0.98:
                break
            old = err
            if self.dense:
                sol += linalg.lu_solve(self.fac, res, check_finite=False)
            else:
                sol += self.fac.solve(res)
        return sol * self.scale


class AugmentedSystem:
    def __init__(self, p, Hinv, T=None):
        if T is None:
            self.K0 = sparse.bmat([[p.P, p.A.T], [p.A, -Hinv]], format="csc")
        else:
            B = (T @ p.A).tocsc()
            self.K0 = sparse.bmat([[p.P, B.T], [B, -sparse.diags((~p.eq).astype(float))]], format="csc")
        K = self.K0
        self.scale = np.ones(K.shape[0])
        for _ in range(4):
            a = np.asarray(abs(K).max(axis=1).toarray()).ravel()
            d = 1 / np.sqrt(np.maximum(a, 1e-12))
            d = np.clip(d, 1e-3, 1e3)
            K = K.multiply(d).multiply(d[:, None]).tocsc()
            self.scale *= d
        self.K = K
        self.refinement_steps = 5
        reg = np.r_[np.full(p.n, 1e-12), np.full(p.m, -1e-12)]
        fixed = (all(kind in ("zero", "nonnegative") for kind, _, _, _ in p.blocks)
                 and np.count_nonzero(p.eq) <= 8
                 and np.max(p.A[p.eq].getnnz(axis=1), initial=0) > 1000)
        if fixed:
            self.fac = sla.splu(K + sparse.diags(reg), permc_spec="MMD_AT_PLUS_A",
                               diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        else:
            self.fac = sla.splu(K + sparse.diags(reg), permc_spec="COLAMD")

    def solve(self, rhs):
        rhs = rhs * self.scale
        sol = self.fac.solve(rhs)
        old = float("inf")
        for _ in range(self.refinement_steps):
            r = rhs - self.K @ sol
            err = norm(r) / (1 + norm(rhs))
            if err < 5e-14 or err > 0.95 * old:
                break
            old = err
            sol += self.fac.solve(r)
        return sol * self.scale


class BandedSystem:
    def __init__(self, M, C):
        self.n, self.k = M.shape[0], C.shape[0]
        self.C = C
        T = sparse.tril(M, format="coo")
        width = int(np.max(T.row - T.col, initial=0))
        self.banded = width <= 32
        diagonal = np.maximum(M.diagonal(), 1e-20)
        if self.banded:
            band = np.zeros((width + 1, self.n))
            band[T.row - T.col, T.col] = T.data
            try:
                self.factor = linalg.cholesky_banded(band, lower=True, check_finite=False)
            except linalg.LinAlgError:
                band[0] += 1e-12 * max(1, norm(band[0]))
                self.factor = linalg.cholesky_banded(band, lower=True, check_finite=False)
        else:
            self.pscale = 1 / np.sqrt(diagonal)
            MM = M.multiply(self.pscale).multiply(self.pscale[:, None]).tocsc()
            self.factor = sla.splu(MM + sparse.eye(self.n, format="csc") * 1e-13,
                                  permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
        self.small = self.k <= 8
        if self.k == 0:
            return
        if self.small:
            self.V = self.primal(C.T.toarray())
            S = C @ self.V
            self.Sfactor = linalg.lu_factor(S, check_finite=False)
        else:
            S = (C.multiply(1 / diagonal) @ C.T).tocsc()
            self.Spre = sla.splu(S + sparse.diags(np.maximum(S.diagonal(), 1e-20) * 1e-10),
                                 permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
            self.operator = sla.LinearOperator((self.k, self.k),
                matvec=lambda v: C @ self.primal(C.T @ v))
            self.preconditioner = sla.LinearOperator((self.k, self.k), matvec=self.Spre.solve)

    def primal(self, rhs):
        if self.banded:
            return linalg.cho_solve_banded((self.factor, True), rhs, check_finite=False)
        if rhs.ndim == 2:
            return self.pscale[:, None] * self.factor.solve(self.pscale[:, None] * rhs)
        return self.pscale * self.factor.solve(self.pscale * rhs)

    def solve(self, rhs):
        v = self.primal(rhs[:self.n])
        if self.k == 0:
            return v
        r = self.C @ v - rhs[self.n:]
        if self.small:
            y = linalg.lu_solve(self.Sfactor, r, check_finite=False)
            x = v - self.V @ y
        else:
            y, info = sla.cg(self.operator, r, M=self.preconditioner,
                             rtol=1e-11, atol=1e-14, maxiter=1200)
            x = self.primal(rhs[:self.n] - self.C.T @ y)
        return np.r_[x, y]


def core(p, recover, original, maxiter=120, time_limit=270.0, verbose=False):
    start = time.monotonic()
    cone = ProductCone(p)
    Pcoo = p.P.tocoo()
    nn_degrees = np.asarray(p.A.getnnz(axis=1))[~p.eq]
    banded = (p.n >= 500 and bool(p.P.nnz) and not cone.psd and not len(cone.soc)
              and not len(cone.exp) and np.max(nn_degrees, initial=0) <= 1
              and np.max(abs(Pcoo.row - Pcoo.col), initial=0) <= 32
              and np.min(p.P.diagonal(), initial=1.0) > 0)
    C = p.A[p.eq].tocsc()
    eq = np.flatnonzero(p.eq)
    n, k = p.n, len(eq)
    x = np.zeros(n)
    s, z = cone.initial()
    tau, kap = 1.0, 1.0
    homogeneous = True
    best, besterr = {"status": "unresolved"}, float("inf")
    projector = None
    refined = False
    dual_repair, last_repair_error = None, float("inf")
    previous_error, slow_iterations = float("inf"), 0
    for it in range(maxiter):
        if time.monotonic() - start > time_limit:
            break
        if tau > 1e-14:
            candidate = recover({"status": "solved", "x": x / tau, "s": s / tau, "z": z / tau})
            ok, err = assess(original, candidate["x"], candidate["s"], candidate["z"],
                             check_all=besterr < 1e-3)
            if err < besterr:
                best, besterr = candidate, err
            if ok:
                return candidate
            slow_iterations = slow_iterations + 1 if besterr > previous_error * 0.8 else 0
            previous_error = besterr
            if (not len(cone.exp) and besterr < 1e-4 and it >= 15
                    and slow_iterations >= 3 and besterr < last_repair_error * 0.7):
                try:
                    from dual_repair import DualRepair
                    if dual_repair is None:
                        dual_repair = DualRepair(original, best["z"])
                    last_repair_error = besterr
                    repaired = dual_repair.repair(best, budget=min(5, max(0.1, time_limit - (time.monotonic() - start))))
                    ook, eek = assess(original, repaired["x"], repaired["s"], repaired["z"], check_all=True)
                    if eek < besterr:
                        best, besterr = repaired, eek
                    if ook:
                        return repaired
                except (ValueError, RuntimeError, MemoryError):
                    last_repair_error = 0
            if (not refined and p.n <= 6000 and not len(cone.exp) and besterr < 1e-4
                    and slow_iterations >= 6 and it >= 22
                    and time_limit - (time.monotonic() - start) > 2):
                from polish import semismooth_polish
                refined = True
                rr = semismooth_polish(p, x / tau, z / tau, recover, original,
                                      budget=min(45, time_limit - (time.monotonic() - start)), verbose=verbose)
                if rr is not None:
                    oo, ee = assess(original, rr["x"], rr["s"], rr["z"], check_all=True)
                    if ee < besterr:
                        best, besterr = rr, ee
                    if oo:
                        return rr
            if cone.psd and it > 8 and err < 1e-3:
                try:
                    if projector is None:
                        B = sparse.vstack((original.A.T, sparse.csr_matrix(original.b[None, :])), format="csc")
                        scale = np.asarray(abs(B).max(axis=1).toarray()).ravel()
                        scale = 1 / np.maximum(scale, 1e-8)
                        B = B.multiply(scale[:, None]).tocsc()
                        G = (B @ B.T).tocsc()
                        factor = sla.splu(G + sparse.eye(G.shape[0], format="csc") * 1e-14)
                        projector = (B, factor, scale)
                    B, factor, scale = projector
                    xx, zz = candidate["x"], candidate["z"].copy()
                    px0 = original.P @ xx
                    target = np.r_[-px0 - original.q, -xx @ px0 - original.q @ xx] * scale
                    for _ in range(3):
                        zz += B.T @ factor.solve(target - B @ zz)
                    okp, ep = assess(original, xx, candidate["s"], zz, check_all=True)
                    if ep < besterr:
                        best, besterr = dict(candidate, z=zz), ep
                    if okp:
                        return dict(candidate, z=zz)
                except (RuntimeError, ValueError):
                    pass
            ready = homogeneous and it > 3 and kap / tau < 1e-4 and err < 1e-3
            if homogeneous and it > 8 and tau > 1e-4 and kap / tau < 0.01:
                trp = p.A @ x + s - p.b * tau
                trd = p.P @ x + p.q * tau + p.A.T @ z
                ready |= (norm(trp) < 1e-5 * tau * (1 + norm(p.b))
                          and norm(trd) < 1e-5 * tau * (1 + norm(p.q)))
            if ready:
                x, s, z = x / tau, s / tau, z / tau
                tau, kap = 1.0, 0.0
                homogeneous = False
        if it >= 4:
            candidate = recover({"status": "primal_infeasible", "z": z})
            ok, erc = assess(original, z=candidate["z"], status="primal_infeasible")
            if ok:
                return candidate
            candidate = recover({"status": "dual_infeasible", "x": x, "s": s})
            ok, erd = assess(original, x=candidate["x"], s=candidate["s"], status="dual_infeasible")
            if ok:
                return candidate
        px = p.P @ x
        rd = px + p.q * tau + p.A.T @ z
        rp = p.A @ x + s - p.b * tau
        xpx = float(x @ px)
        rg = float(p.q @ x + p.b @ z + kap + xpx / tau)
        mu = (float(s @ z) + tau * kap) / (cone.degree + int(homogeneous))
        if verbose:
            print("ipm", it, "err", "%.3g" % besterr, "mu", "%.3g" % mu,
                  "tau", "%.3g" % tau, "kap", "%.3g" % kap, file=sys.stderr)
        if not np.isfinite(mu) or mu <= 0:
            break
        try:
            cone.scaling(s, z, mu)
            augmented = not cone.psd and not banded
            if augmented:
                system = AugmentedSystem(p, cone.Hinv, cone.T)
                Tb = cone.T @ p.b
                Trp = cone.T @ rp
                u = np.r_[-p.q, Tb]
                v = np.r_[p.q + 2 * px / tau, Tb]
                gamma = -xpx / tau ** 2 - kap / tau
            else:
                M = cone.schur()
                system = BandedSystem(M, C) if banded else LinearSystem(M, C, reg=1e-12)
                Hb = cone.apply(p.b)
                Hrp = cone.apply(rp)
                atHb = p.A.T @ Hb
                u = np.r_[-p.q + atHb, p.b[eq]]
                v = np.r_[p.q + 2 * px / tau + atHb, p.b[eq]]
                gamma = -float(p.b @ Hb) - xpx / tau ** 2 - kap / tau
            if homogeneous:
                l = system.solve(u)
                denom = gamma + v @ l
                if not np.isfinite(denom) or abs(denom) < 1e-300:
                    break

            def direction(h, rt, residual_scale=1.0):
                if augmented:
                    rhs = np.r_[-residual_scale * rd,
                                -residual_scale * Trp - h]
                    kk = system.solve(rhs)
                    dt = (-residual_scale * rg - rt / tau - v @ kk) / denom if homogeneous else 0.0
                    vv = kk + dt * l if homogeneous else kk
                    dx, dz = vv[:n], cone.T.T @ vv[n:]
                    ds = -residual_scale * rp - p.A @ dx + p.b * dt
                    ds[eq] = 0
                    dk = (rt - kap * dt) / tau if homogeneous else 0.0
                    return dx, ds, dz, float(dt), float(dk)
                hp = h + residual_scale * Hrp
                rhs = np.r_[-residual_scale * rd - p.A.T @ hp,
                            -residual_scale * rp[eq]]
                kk = system.solve(rhs)
                dt = (-residual_scale * rg - p.b @ hp - rt / tau - v @ kk) / denom if homogeneous else 0.0
                vv = kk + dt * l if homogeneous else kk
                dx = vv[:n]
                ds = -residual_scale * rp - p.A @ dx + p.b * dt
                ds[eq] = 0
                dz = h - cone.apply(ds)
                dz[eq] = vv[n:]
                dk = (rt - kap * dt) / tau if homogeneous else 0.0
                return dx, ds, dz, float(dt), float(dk)

            ha = cone.rhs_scaled(s, z, mu, 0.0) if augmented else cone.rhs(s, z, mu, 0.0)
            da = direction(ha, -tau * kap)
            dx, ds, dz, dt, dk = da
            ap = min(cone.max_step(s, ds), -tau / dt if dt < 0 else 1.0)
            ad = min(cone.max_step(z, dz, True), -kap / dk if dk < 0 else 1.0)
            aa = min(ap, ad)
            mua = ((s + aa * ds) @ (z + aa * dz) + (tau + aa * dt) * (kap + aa * dk)) / (cone.degree + int(homogeneous))
            sigma = min(0.95, max(0.0, float(mua / mu)) ** 3)
            if len(cone.exp):
                sigma = max(0.15, sigma)
            h = cone.rhs_scaled(s, z, mu, sigma, ds, dz) if augmented else cone.rhs(s, z, mu, sigma, ds, dz)
            if len(cone.exp) and not augmented:
                # A conservative centering term stabilizes nonsymmetric blocks.
                h[cone.exp] = -z[cone.exp] - sigma * mu * cone.eg
            d = direction(h, -tau * kap + sigma * mu - dt * dk, 1 - sigma)
            dx, ds, dz, dt, dk = d
            ap = min(cone.max_step(s, ds), -tau / dt if dt < 0 else 1.0)
            ad = min(cone.max_step(z, dz, True), -kap / dk if dk < 0 else 1.0)
            alpha = 0.99 * min(ap, ad)
            if len(cone.exp):
                alpha *= 0.9
            if alpha < 1e-12 or not np.isfinite(alpha):
                break
            while alpha > 1e-14:
                ss, zz = s + alpha * ds, z + alpha * dz
                if cone.interior(ss) and cone.interior(zz, True):
                    break
                alpha *= 0.5
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kap += alpha * dk
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError) as e:
            if verbose:
                print("ipm numerical stop:", e, file=sys.stderr)
            break
    return best


def solve_conic(p, time_limit=270.0):
    started = time.monotonic()
    pre = Presolve(p)
    if pre.early is not None:
        return pre.early
    reduced = pre.problem
    affine = AffineReduction(reduced, rounds=4)
    reduced = affine.problem
    if all(k == "zero" for k, _, _, _ in reduced.blocks):
        from poly import solve_poly
        r = solve_poly(reduced, time_limit=max(0.1, time_limit - (time.monotonic() - started)))
        return pre.restore(affine.restore(r))
    if any(k == "exponential" for k, _, _, _ in reduced.blocks):
        from gp import try_gp
        r = try_gp(reduced, lambda r: pre.restore(affine.restore(r)), p,
                   budget=min(45, max(1, time_limit * 0.4)))
        if r is not None and assess(p, r["x"], r["s"], r["z"])[0]:
            return r
    sp, D, E, c = equilibrate(reduced, 5)

    def recover(r):
        r = dict(r)
        if "x" in r:
            r["x"] = r["x"] * D
            r["s"] = r["s"] / E
        if "z" in r:
            r["z"] = r["z"] * E / c
        return pre.restore(affine.restore(r))

    result = core(sp, recover, p, time_limit=max(0.1, time_limit - (time.monotonic() - started)),
                  maxiter=180 if any(k == "exponential" for k, _, _, _ in p.blocks) else 120,
                  verbose=bool(os.environ.get("SOLVER_VERBOSE")))
    valid = result["status"] != "unresolved" and assess(
        p, status=result["status"], **{k: result[k] for k in ("x", "s", "z") if k in result})[0]
    remaining = time_limit - (time.monotonic() - started)
    if not valid and remaining > 2 and not p.P.nnz and p.n <= 1800 and any(
            k == "psd" and d <= 128 for k, d, _, _ in p.blocks):
        from outer_psd import solve_psd
        trial = solve_psd(p, budget=remaining)
        if trial["status"] != "unresolved" and assess(
                p, status=trial["status"], **{k: trial[k] for k in ("x", "s", "z") if k in trial})[0]:
            return trial
    return result
