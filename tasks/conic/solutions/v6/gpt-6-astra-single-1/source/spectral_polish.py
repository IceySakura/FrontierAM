"""Regularized semismooth Newton refinement of symmetric-cone KKT equations."""

import time
import os
import sys

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm, smat, svec, residuals
from scaling import equilibrate, recover


def polish(original, result, deadline):
    if original.n > 1300:
        return None
    allocation = sum(original.n * c["dim"]**2 for c in original.cones if c["type"] == "psd")
    if allocation > 40000000:
        return None
    p, D, E, cost = equilibrate(original, iterations=4)
    x, z = result["x"] / D, result["z"] * cost / E
    sdps, scalars = [], []
    for kind, dim, sl in blocks(p.cones):
        if kind == "psd":
            rows, cols = np.tril_indices(dim)
            A = p.A[sl].tocoo()
            stack = np.zeros((p.n, dim, dim))
            values = A.data / np.where(rows[A.row] == cols[A.row], 1.0, np.sqrt(2.0))
            stack[A.col, rows[A.row], cols[A.row]] = values
            stack[A.col, cols[A.row], rows[A.row]] = values
            sdps.append((dim, sl, rows, cols, stack))
        elif kind in ("zero", "nonnegative"):
            scalars.append((kind, sl, p.A[sl].toarray()))
        else:
            return None
    P = p.P.toarray()
    A, AT = p.A, p.A.T
    try:
        gram = (AT @ A).tocsc()
        affine_factor = sla.splu(gram + sparse.eye(p.n, format="csc") * 1e-14)
    except RuntimeError:
        affine_factor = None
    s = p.b - A @ x
    rho = float(np.clip(norm(s) / max(1e-12, norm(z)), 1e-4, 1e6))
    merit_scale = max(1.0, norm(s), rho * norm(z))

    def affine_candidate(xx, zz):
        if affine_factor is None:
            return None
        defect = -P @ xx - p.q - AT @ zz
        correction = affine_factor.solve(defect)
        for unused in range(3):
            correction += affine_factor.solve(defect - gram @ correction)
        projected = zz + A @ correction
        candidate = recover({"status": "solved", "x": xx, "s": p.b - A @ xx,
                             "z": projected}, D, E, cost)
        return candidate if acceptable(original, candidate) else None

    def state(xx, zz, derivatives=False, regularization=1e-8):
        ss = p.b - A @ xx
        rd = P @ xx + p.q + AT @ zz
        rr = np.zeros(p.m)
        systems = []
        for kind, sl, B in scalars:
            vv = ss[sl] - rho * zz[sl]
            if kind == "zero":
                rr[sl] = ss[sl]
                derivative = np.zeros(vv.size)
            else:
                rr[sl] = ss[sl] - np.maximum(vv, 0)
                derivative = (vv > 0).astype(float)
            if derivatives:
                t = derivative + regularization * (1 - derivative)
                weights = (1 - derivative) / (rho * t)
                target = rr[sl] / (rho * t)
                systems.append((sl, B, weights, target, None, derivative))
        for dim, sl, ri, ci, stack in sdps:
            S, Z = smat(ss[sl], dim), smat(zz[sl], dim)
            ev, U = linalg.eigh(S - rho * Z, check_finite=False)
            positive = np.maximum(ev, 0)
            R = S - (U * positive) @ U.T
            rr[sl] = svec(R)
            if derivatives:
                plus = ev > 0
                deriv = np.zeros((dim, dim))
                deriv[plus[:, None] & plus[None, :]] = 1
                cross = plus[:, None] != plus[None, :]
                numerator = positive[:, None] - positive[None, :]
                denominator = ev[:, None] - ev[None, :]
                deriv[cross] = numerator[cross] / denominator[cross]
                d = deriv[ri, ci]
                t = d + regularization * (1 - d)
                weights = (1 - d) / (rho * t)
                target = svec(U.T @ R @ U) / (rho * t)
                transformed = np.matmul(np.matmul(U.T, stack), U)
                B = (transformed[:, ri, ci] * np.where(ri == ci, 1.0, np.sqrt(2.0))).T
                systems.append((sl, B, weights, target, U, d))
        merit = np.sqrt(rho**2 * float(rd @ rd) + float(rr @ rr)) / merit_scale
        return ss, rd, rr, merit, systems

    for iteration in range(25):
        if time.monotonic() > deadline:
            return None
        result = recover({"status": "solved", "x": x, "s": p.b - A @ x, "z": z}, D, E, cost)
        if os.environ.get("CONIC_DEBUG"):
            print("SSN", iteration, "rho", rho, "res", residuals(original, result), file=sys.stderr)
        if acceptable(original, result):
            return result
        projected = affine_candidate(x, z)
        if projected is not None:
            return projected
        eta = 1e-7 if iteration < 4 else 1e-9 if iteration < 10 else 1e-11
        s, rd, rr, merit, systems = state(x, z, True, eta)
        H = P.copy()
        rhs = -rd.copy()
        for sl, B, weights, target, U, derivative in systems:
            H += B.T @ (weights[:, None] * B)
            rhs += B.T @ target
        H = 0.5 * (H + H.T)
        reg = 1e-14 * max(1e-6, np.max(np.diag(H), initial=0))
        K = H.copy()
        K.flat[::p.n+1] += reg
        try:
            factor = linalg.lu_factor(K, check_finite=False)
            dx = linalg.lu_solve(factor, rhs, check_finite=False)
            for refinement in range(4):
                residual = rhs - H @ dx
                if norm(residual) < 1e-12 * norm(rhs) + 1e-16:
                    break
                trial = dx + linalg.lu_solve(factor, residual, check_finite=False)
                if norm(rhs - H @ trial) > 1.1 * norm(residual):
                    break
                dx = trial
        except (ValueError, np.linalg.LinAlgError):
            return None
        dz = np.zeros(p.m)
        for sl, B, weights, target, U, derivative in systems:
            value = weights * (B @ dx) - target
            dz[sl] = value if U is None else svec(U @ smat(value, U.shape[0]) @ U.T)
        if not np.isfinite(np.r_[dx, dz]).all():
            return None
        accepted = False
        for method in range(2):
            if method:
                def projection_derivative(vector):
                    out = np.zeros(p.m)
                    for sl, B, weights, target, U, derivative in systems:
                        if U is None:
                            out[sl] = derivative * vector[sl]
                        else:
                            V = smat(vector[sl], U.shape[0])
                            diagonal = derivative * svec(U.T @ V @ U)
                            out[sl] = svec(U @ smat(diagonal, U.shape[0]) @ U.T)
                    return out

                def forward(v):
                    xx, zz = v[:p.n], v[p.n:]
                    Ax = A @ xx
                    return np.r_[rho * (P @ xx) + AT @ zz,
                                 -Ax + projection_derivative(Ax + zz)]

                def adjoint(v):
                    xx, zz = v[:p.n], v[p.n:]
                    projected = projection_derivative(zz)
                    return np.r_[rho * (P @ xx) - AT @ (zz - projected), A @ xx + projected]

                operator = sla.LinearOperator((p.n+p.m, p.n+p.m),
                                               matvec=forward, rmatvec=adjoint, dtype=float)
                direction = sla.lsmr(operator, -np.r_[rho * rd, rr], damp=1e-7,
                                     atol=1e-10, btol=1e-10, maxiter=min(700, 2*p.n))[0]
                dx, dz = direction[:p.n], direction[p.n:] / rho
            alpha = 1.0
            for backtrack in range(20):
                trial_x, trial_z = x + alpha * dx, z + alpha * dz
                trial_s, unused, unused2, trial_merit, unused3 = state(trial_x, trial_z)
                candidate = recover({"status": "solved", "x": trial_x, "s": trial_s,
                                     "z": trial_z}, D, E, cost)
                if acceptable(original, candidate):
                    return candidate
                if backtrack == 0:
                    projected = affine_candidate(trial_x, trial_z)
                    if projected is not None:
                        return projected
                if trial_merit < merit * (1 - 1e-3 * alpha) or trial_merit < 1e-13:
                    x, z = trial_x, trial_z
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break
        if not accepted:
            if os.environ.get("CONIC_DEBUG"):
                print("SSN line search", merit, trial_merit, "step", norm(dx), norm(dz), file=sys.stderr)
            return None
    return None
