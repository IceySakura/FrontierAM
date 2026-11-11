"""Nesterov--Todd primal-dual SDP predictor-corrector.

The Schur complement uses only the svec rows touched by A. Its sparse
congruence formula avoids allocating a full order-d(d+1)/2 cone Hessian.
"""

import os
import sys
import time
import warnings

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm, residuals, smat, svec
from scaling import equilibrate, recover


class SemidefiniteBlock:
    def __init__(self, dim, sl, A):
        self.d, self.sl = dim, sl
        self.A = A[sl].tocsr()
        self.used = np.flatnonzero(np.diff(self.A.indptr))
        self.B = self.A[self.used]
        ri, ci = np.tril_indices(dim)
        self.ri, self.ci = ri[self.used], ci[self.used]
        self.coeff = np.where(self.ri == self.ci, np.sqrt(0.5), 1.0)
        self.eye = np.eye(dim)

    def initialize(self, s, z):
        s[self.sl] = svec(self.eye)
        z[self.sl] = svec(self.eye)

    def update(self, s, z):
        S, Z = smat(s[self.sl], self.d), smat(z[self.sl], self.d)
        self.Z = Z
        L = linalg.cholesky(S, lower=True, check_finite=False)
        eig, U = linalg.eigh(L.T @ Z @ L, check_finite=False)
        if eig[0] <= 0:
            raise np.linalg.LinAlgError("NT scaling left the positive definite cone")
        self.lam = np.sqrt(eig)
        self.R = (L @ U) / np.sqrt(self.lam)[None, :]
        Linv = linalg.solve_triangular(L, self.eye, lower=True, check_finite=False)
        self.Rinv = np.sqrt(self.lam)[:, None] * (U.T @ Linv)
        self.M = self.Rinv.T @ self.Rinv

    def hessian(self):
        M, ri, ci = self.M, self.ri, self.ci
        H = M[np.ix_(ri, ri)] * M[np.ix_(ci, ci)]
        H += M[np.ix_(ri, ci)] * M[np.ix_(ci, ri)]
        H *= self.coeff[:, None]
        H *= self.coeff[None, :]
        BA = self.B.T @ H
        return self.B.T @ BA.T

    def inverse(self, v):
        V = smat(v[self.sl], self.d)
        return svec(self.M @ V @ self.M)

    def correction(self, ds, dz, target):
        DS = self.Rinv @ smat(ds[self.sl], self.d) @ self.Rinv.T
        DZ = self.R.T @ smat(dz[self.sl], self.d) @ self.R
        cross = DS @ DZ
        rhs = target * self.eye - 0.5 * (cross + cross.T)
        rhs *= 2.0 / (self.lam[:, None] + self.lam[None, :])
        return svec(self.Rinv.T @ rhs @ self.Rinv - self.Z)

    def step(self, v, dv):
        V, DV = smat(v[self.sl], self.d), smat(dv[self.sl], self.d)
        L = linalg.cholesky(V, lower=True, check_finite=False)
        temp = linalg.solve_triangular(L, DV, lower=True, check_finite=False)
        temp = linalg.solve_triangular(L, temp.T, lower=True, check_finite=False).T
        least = linalg.eigh(temp, eigvals_only=True, subset_by_index=[0, 0],
                           check_finite=False)[0]
        return min(1.0, -1.0 / least) if least < 0 else 1.0


def initialize(p, sdps, inn, ieq):
    x, s, z = np.zeros(p.n), np.zeros(p.m), np.zeros(p.m)
    identity = np.zeros(p.m)
    identity[inn] = 1.0
    for block in sdps:
        identity[block.sl] = svec(block.eye)
    try:
        Gram = (p.A.T @ p.A).tocsc()
        fac = sla.splu(Gram + sparse.eye(p.n, format="csc") * 1e-13)
        direction = -fac.solve(p.A.T @ identity)
        dslack = -p.A @ direction
        min_direction = float(np.min(dslack[inn], initial=np.inf))
        for block in sdps:
            ev = linalg.eigh(smat(dslack[block.sl], block.d), eigvals_only=True,
                             subset_by_index=[0, 0], check_finite=False)[0]
            min_direction = min(min_direction, ev)
        if min_direction <= 1e-7:
            direction = -p.A.T @ identity
            direction /= max(1.0, norm(direction))
            dslack = -p.A @ direction
        amount = 1.0
        for trial in range(12):
            trial_s = p.b + amount * dslack
            okay = bool(np.min(trial_s[inn], initial=1.0) > 1e-5)
            if okay:
                for block in sdps:
                    ev = linalg.eigh(smat(trial_s[block.sl], block.d), eigvals_only=True,
                                     subset_by_index=[0, 0], check_finite=False)[0]
                    if ev <= 1e-5:
                        okay = False
                        break
            if okay:
                x = amount * direction
                s = trial_s
                break
            amount *= 4
        if not np.any(s):
            s[inn] = np.maximum(1.0, p.b[inn])
            for block in sdps:
                s[block.sl] = svec(block.eye)
        z = -p.A @ fac.solve(p.q + p.P @ x)
        if inn.size:
            z[inn] = np.maximum(z[inn], 1e-3 * max(1e-6, norm(z[inn])))
        for block in sdps:
            Z = smat(z[block.sl], block.d)
            ev = linalg.eigh(Z, eigvals_only=True, subset_by_index=[0, 0], check_finite=False)[0]
            margin = 1e-3 * max(1e-6, norm(Z))
            if ev < margin:
                Z.flat[::block.d+1] += margin - ev
            z[block.sl] = svec(Z)
    except (RuntimeError, np.linalg.LinAlgError):
        s[inn], z[inn] = np.maximum(1.0, p.b[inn]), 1.0
        for block in sdps:
            block.initialize(s, z)
    s[ieq] = 0
    return x, s, z


def solve(original, deadline):
    p, D, E, cost = equilibrate(original, iterations=4)
    n, m = p.n, p.m
    use_spectral = n <= 1300 and np.any(np.diff(p.A.tocsr().indptr) > 1)
    if n > 12000:
        return {"status": "unresolved"}
    eq, nonneg = np.zeros(m, bool), np.zeros(m, bool)
    sdps = []
    degree = 0
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
        elif kind == "nonnegative":
            nonneg[sl] = True
            degree += dim
        elif kind == "psd":
            block = SemidefiniteBlock(dim, sl, p.A)
            if block.used.size > 18000:
                return {"status": "unresolved"}
            sdps.append(block)
            degree += dim
        else:
            return {"status": "unresolved"}
    ieq, inn = np.flatnonzero(eq), np.flatnonzero(nonneg)
    used_max = max((block.used.size for block in sdps), default=0)
    memory_estimate = 8 * (5 * (n + ieq.size)**2 + 4 * used_max**2 + 2 * n * used_max)
    if memory_estimate > 2800000000:
        return {"status": "unresolved"}
    B = p.A[ieq].toarray()
    G = p.A[inn].tocsc()
    AT = p.A.T
    x, s, z = initialize(p, sdps, inn, ieq)
    best_error = float("inf")
    best_result = None
    previous = None
    previous_mu = None
    polish_attempts = 0
    for iteration in range(100):
        if time.monotonic() > deadline:
            break
        rd = p.P @ x + p.q + AT @ z
        rp = p.A @ x + s - p.b
        rp[np.abs(rp) < 2e-14 * (1 + np.abs(p.b) + np.abs(s))] = 0.0
        mu = float(s @ z) / max(1, degree)
        scaled = {"status": "solved", "x": x, "s": s, "z": z}
        result = recover(scaled, D, E, cost)
        errors = residuals(original, result)
        error = max(errors.values())
        if os.environ.get("CONIC_DEBUG"):
            print("SDP", iteration, "mu", mu, "error", error,
                  "res", errors, file=sys.stderr)
        if error <= 8e-9:
            return result
        if error < best_error:
            best_result = {key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in result.items()}
        if use_spectral and error < (1e-5 if polish_attempts == 0 else 1e-7) and polish_attempts < 2:
            from spectral_polish import polish
            polished = polish(original, result, min(deadline, time.monotonic() + 25))
            polish_attempts += 1
            if polished is not None:
                return polished
        if previous is not None and previous_mu > 1.05 * mu and error < 2e-4:
            extrapolation = min(1.0, mu / (previous_mu - mu))
            for weight in (extrapolation, 0.5 * extrapolation, 1.5 * extrapolation):
                candidate = {"status": "solved"}
                for key in ("x", "s", "z"):
                    candidate[key] = result[key] + weight * (result[key] - previous[key])
                if acceptable(original, candidate):
                    return candidate
        if previous_mu is None or mu < 0.75 * previous_mu:
            previous = {key: val.copy() if isinstance(val, np.ndarray) else val
                        for key, val in result.items()}
            previous_mu = mu
        best_error = min(best_error, error)
        if iteration > 5:
            cr = recover({"status": "primal_infeasible", "z": z}, D, E, cost)
            if acceptable(original, cr):
                return cr
            cr = recover({"status": "dual_infeasible", "x": x, "s": s}, D, E, cost)
            if acceptable(original, cr):
                return cr
        try:
            H = p.P.toarray()
            if inn.size:
                H += (G.T @ sparse.diags(z[inn] / s[inn]) @ G).toarray()
            for block in sdps:
                block.update(s, z)
                H += block.hessian()
            H = 0.5 * (H + H.T)
            if ieq.size:
                K = np.block([[H, B.T], [B, np.zeros((ieq.size, ieq.size))]])
            else:
                K = H
            diag = np.maximum(1.0, np.abs(np.diag(K)))
            regularization = np.minimum(1e-7, 1e-12 * diag)
            if ieq.size:
                regularization[n:] *= -1
            regK = K.copy()
            regK.flat[::regK.shape[0] + 1] += regularization
            with warnings.catch_warnings():
                warnings.simplefilter("error", linalg.LinAlgWarning)
                factor = linalg.lu_factor(regK, check_finite=False)
        except (np.linalg.LinAlgError, linalg.LinAlgWarning, ValueError):
            break

        def apply_inverse(v):
            out = np.zeros(m)
            out[inn] = z[inn] / s[inn] * v[inn]
            for block in sdps:
                out[block.sl] = block.inverse(v)
            return out

        def direction(dual_rhs):
            w = apply_inverse(rp) + dual_rhs
            rhs = np.r_[-rd - AT @ w, -rp[ieq]]
            step = linalg.lu_solve(factor, rhs, check_finite=False)
            for refinement in range(6):
                residual = rhs - K @ step
                if norm(residual) <= 2e-12 * max(1.0, norm(rhs)):
                    break
                update = linalg.lu_solve(factor, residual, check_finite=False)
                trial = step + update
                if norm(rhs - K @ trial) > norm(residual) * 1.1:
                    break
                step = trial
            dx = step[:n]
            ds = -rp - p.A @ dx
            ds[ieq] = 0.0
            dz = dual_rhs + apply_inverse(-ds)
            dz[ieq] = step[n:]
            # Refine the full Newton equations, not just the assembled Schur matrix.
            for refinement in range(3):
                dual_res = -rd - p.P @ dx - AT @ dz
                eq_res = -rp[ieq] - B @ dx
                if max(norm(dual_res), norm(eq_res)) < 1e-11 * max(1.0, norm(rd)):
                    break
                rhs2 = np.r_[dual_res, eq_res]
                delta = linalg.lu_solve(factor, rhs2, check_finite=False)
                ddx = delta[:n]
                dds = -p.A @ ddx
                dds[ieq] = 0
                ddz = apply_inverse(-dds)
                ddz[ieq] = delta[n:]
                if norm(-rd - p.P @ (dx + ddx) - AT @ (dz + ddz)) > 1.1 * norm(dual_res):
                    break
                dx += ddx
                ds += dds
                dz += ddz
            return dx, ds, dz

        def step_size(v, dv):
            negative = (dv[inn] < 0)
            alpha = min(1.0, float(np.min(-v[inn][negative] / dv[inn][negative],
                                          initial=1.0)))
            for block in sdps:
                alpha = min(alpha, block.step(v, dv))
            return alpha

        try:
            aff_rhs = -z.copy()
            aff_rhs[ieq] = 0.0
            dx, ds, dz = direction(aff_rhs)
            ap, ad = step_size(s, ds), step_size(z, dz)
            mu_aff = float((s + ap * ds) @ (z + ad * dz)) / max(1, degree)
            sigma = float(np.clip((mu_aff / max(mu, 1e-300)) ** 3, 0.0, 1.0))
            cone_rhs = np.zeros(m)
            cone_rhs[inn] = -z[inn] + (sigma * mu - ds[inn] * dz[inn]) / s[inn]
            for block in sdps:
                cone_rhs[block.sl] = block.correction(ds, dz, sigma * mu)
            dx, ds, dz = direction(cone_rhs)
            ap, ad = min(1.0, 0.99 * step_size(s, ds)), min(1.0, 0.99 * step_size(z, dz))
            if not np.isfinite(np.r_[dx, ds, dz]).all() or min(ap, ad) < 1e-14:
                break
            x += ap * dx
            s += ap * ds
            z += ad * dz
        except (np.linalg.LinAlgError, ValueError):
            break
    if best_result is not None and best_error < 1e-3 and time.monotonic() < deadline and use_spectral:
        from spectral_polish import polish
        result = polish(original, best_result, deadline)
        if result is not None:
            return result
    return {"status": "unresolved"}
