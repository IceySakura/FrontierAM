"""Sparse infeasible-start Mehrotra predictor-corrector for convex QPs."""

import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from linear import masks, polish
from quality import acceptable, norm, residuals
from scaling import equilibrate, recover


def maximum_step(v, dv):
    negative = dv < 0
    return min(1.0, float(np.min(-v[negative] / dv[negative], initial=1.0)))


def solve(original, deadline, warm=None):
    p, D, E, cost = equilibrate(original)
    eq = masks(p)
    ieq, iineq = np.flatnonzero(eq), np.flatnonzero(~eq)
    B = p.A[ieq].tocsc()
    G = p.A[iineq].tocsc()
    GT = G.T.tocsc()
    n, ne, k = p.n, ieq.size, iineq.size
    x = np.zeros(n)
    y = np.zeros(ne)
    z = np.ones(k)
    s = np.maximum(1.0, p.b[iineq])
    if warm is not None and "x" in warm:
        x = warm["x"] / D
        s = np.maximum(1.0, p.b[iineq] - G @ x)
    best = None
    best_error = float("inf")
    regularization = 1e-10
    stagnant = 0
    substantial_curvature = bool(np.count_nonzero(p.P.diagonal()) > n // 2)
    for iteration in range(120):
        if time.monotonic() > deadline:
            break
        rd = p.P @ x + p.q + B.T @ y + GT @ z
        rp = B @ x - p.b[ieq]
        ri = G @ x + s - p.b[iineq]
        mu = float(s @ z) / max(1, k)
        allz = np.empty(p.m)
        allz[ieq], allz[iineq] = y, z
        allslack = p.b - p.A @ x
        scaled = {"status": "solved", "x": x, "s": allslack, "z": allz}
        result = recover(scaled, D, E, cost)
        errors = residuals(original, result)
        error = max(errors.values())
        stagnant = 0 if error < 0.95 * best_error else stagnant + 1
        if error < best_error:
            best, best_error = result, error
        if error <= 8e-9:
            return result
        if stagnant > 10 and (mu < 1e-10 or error > 0.5):
            break
        if os.environ.get("CONIC_DEBUG"):
            print("QPIPM", iteration, "mu", mu, "error", error,
                  "rp", norm(rp), "rd", norm(rd), file=sys.stderr)
        if iteration > 8 and (norm(x) > 1e8 or norm(allz) > 1e8):
            cert = recover({"status": "primal_infeasible", "z": allz}, D, E, cost)
            if acceptable(original, cert):
                return cert
            cert = recover({"status": "dual_infeasible", "x": x, "s": -p.A @ x},
                           D, E, cost)
            if acceptable(original, cert):
                return cert
        w = z / s
        H = p.P + GT @ sparse.diags(w, format="csc") @ G
        K = sparse.bmat([[H, B.T], [B, sparse.csc_matrix((ne, ne))]], format="csc")
        reg = np.r_[regularization * (1.0 + np.abs(p.P.diagonal())),
                    np.full(ne, -regularization)]
        try:
            large = bool(n > 2000 and substantial_curvature)
            factor = sla.splu(K + sparse.diags(reg, format="csc"),
                              diag_pivot_thresh=0.0 if large else 0.1,
                              permc_spec="MMD_AT_PLUS_A" if large else "COLAMD",
                              options={"SymmetricMode": large})
        except RuntimeError:
            regularization *= 10
            if regularization > 1e-3:
                break
            continue

        def direction(target, correction):
            rcz = (target - s * z - correction + z * ri) / s
            rhs = np.r_[-rd - GT @ rcz, -rp]
            step = factor.solve(rhs)
            for refinement in range(5):
                residual = rhs - K @ step
                if norm(residual) <= 1e-11 * max(1.0, norm(rhs)):
                    break
                update = factor.solve(residual)
                if not np.isfinite(update).all():
                    break
                trial = step + update
                if norm(rhs - K @ trial) > norm(residual) * 1.1:
                    break
                step = trial
            dx, dy = step[:n], step[n:]
            ds = -ri - G @ dx
            dz = (target - s * z - correction - z * ds) / s
            return dx, dy, ds, dz

        dx, dy, ds, dz = direction(0.0, 0.0)
        ap, ad = maximum_step(s, ds), maximum_step(z, dz)
        mu_aff = float((s + ap * ds) @ (z + ad * dz)) / max(1, k)
        sigma = np.clip((mu_aff / max(mu, 1e-300)) ** 3, 0.0, 1.0)
        correction = ds * dz
        dx, dy, ds, dz = direction(sigma * mu, correction)
        ap, ad = min(1.0, 0.995 * maximum_step(s, ds)), min(1.0, 0.995 * maximum_step(z, dz))
        if not np.isfinite(np.r_[dx, dy, ds, dz]).all() or min(ap, ad) < 1e-15:
            break
        x += ap * dx
        s += ap * ds
        y += ad * dy
        z += ad * dz
        if not k:
            continue
        if np.any(s <= 0) or np.any(z <= 0):
            break
    if best is not None and time.monotonic() < deadline:
        refined = polish(original, best, masks(original), max_iter=8)
        if acceptable(original, refined):
            return refined
    return {"status": "unresolved"}
