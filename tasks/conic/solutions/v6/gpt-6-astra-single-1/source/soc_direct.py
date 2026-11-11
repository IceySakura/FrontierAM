"""Direct infeasible-start primal-dual path for second-order cone QPs."""

import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from cone_ipm import Product
from quality import acceptable, norm, residuals
from scaling import equilibrate, recover


def solve(original, deadline, warm=None, return_best=False):
    p, D, E, cost = equilibrate(original)
    cone = Product(p.cones, p.m)
    if cone.exp.size:
        return {"status": "unresolved"}
    A, AT, n = p.A, p.A.T, p.n
    s, z = cone.initial()
    x = np.zeros(n)
    reg = np.r_[np.full(n, 1e-11), np.full(p.m, -1e-11)]

    def factorize(H):
        K = sparse.bmat([[p.P, AT], [A, -H]], format="csc")
        fac = sla.splu(K + sparse.diags(reg, format="csc"),
                      permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options={"SymmetricMode": True})
        return K, fac

    if warm is not None:
        x, s, z = warm["x"] / D, warm["s"] * E, warm["z"] * cost / E
    else:
        try:
            H = cone.update(s, z, 1.0)
            K, fac = factorize(H)
            rhs = np.r_[-p.q, p.b]
            initial = fac.solve(rhs)
            for unused in range(3):
                initial += fac.solve(rhs - K @ initial)
            x, z = initial[:n], initial[n:]
            s = p.b - A @ x
            z = cone.initial()[1]
        except RuntimeError:
            x = np.zeros(n)
            s, z = cone.initial()
    s[cone.eq] = 0
    for v, default_margin in ((s, 1.0), (z, 0.1)):
        cone_norm = norm(v[cone.nn])
        for group in cone.soc:
            cone_norm = max(cone_norm, norm(v[group["idx"]]))
        margin = default_margin if warm is None else 1e-6 * max(1e-4, cone_norm)
        v[cone.nn] = np.maximum(v[cone.nn], margin)
        for group in cone.soc:
            idx = group["idx"]
            length = np.linalg.norm(v[idx[:, 1:]], axis=1)
            v[idx[:, 0]] = np.maximum(v[idx[:, 0]], length) + margin
    best, best_error, stagnant, attempts = None, float("inf"), 0, 0
    best_raw = float("inf")
    for iteration in range(100):
        if time.monotonic() > deadline:
            break
        cone.repair_interior(s)
        cone.repair_interior(z)
        rd = p.P @ x + p.q + AT @ z
        rp = A @ x + s - p.b
        mu = float(s @ z) / max(1, cone.degree)
        result = recover({"status": "solved", "x": x, "s": s, "z": z}, D, E, cost)
        errors = residuals(original, result)
        error = max(errors.values())
        raw = norm(rd) / max(1.0, norm(p.q)) + norm(rp) / max(1.0, norm(p.b)) + mu
        stagnant = 0 if raw < .95 * best_raw else stagnant + 1
        best_raw = min(raw, best_raw)
        if error < best_error:
            best, best_error = result, error
        if os.environ.get("CONIC_DEBUG"):
            print("SOC direct", iteration, "mu", mu, "res", errors, file=sys.stderr)
        if error <= 8e-9:
            return result
        thresholds = [1e-3, 1e-5, 1e-7]
        if attempts < len(thresholds) and error < thresholds[attempts]:
            from polish_soc import polish
            polished = polish(original, result, deadline)
            attempts += 1
            if polished is not None:
                return polished
        if stagnant > 12:
            break
        if iteration > 5:
            cert = recover({"status": "primal_infeasible", "z": z}, D, E, cost)
            if acceptable(original, cert):
                return cert
            cert = recover({"status": "dual_infeasible", "x": x, "s": s}, D, E, cost)
            if acceptable(original, cert):
                return cert
        try:
            H = cone.update(s, z, mu)
            K, fac = factorize(H)
        except (RuntimeError, ArithmeticError, np.linalg.LinAlgError):
            break

        def direction(rhs_cone):
            rhs = np.r_[-rd, -rp - rhs_cone]
            delta = fac.solve(rhs)
            for refinement in range(6):
                defect = rhs - K @ delta
                if norm(defect) < 1e-14 * norm(rhs) + 1e-18:
                    break
                trial = delta + fac.solve(defect)
                if norm(rhs - K @ trial) > 1.1 * norm(defect):
                    break
                delta = trial
            dx, dz = delta[:n], delta[n:]
            ds = rhs_cone - cone.multiply(dz)
            ds[cone.eq] = 0
            return dx, ds, dz

        try:
            dx, ds, dz = direction(-s)
            ap, ad = cone.step(s, ds), cone.step(z, dz, True)
            muaff = float((s + ap * ds) @ (z + ad * dz)) / max(1, cone.degree)
            sigma = float(np.clip((muaff / max(mu, 1e-300))**3, 0.0, 1.0))
            rhs = cone.corrector(s, z, ds, dz, sigma * mu)
            dx, ds, dz = direction(rhs)
            ap, ad = .99 * cone.step(s, ds), .99 * cone.step(z, dz, True)
            if p.P.nnz:
                ap = ad = min(ap, ad)
            if min(ap, ad) < 1e-15 or not np.isfinite(np.r_[dx, ds, dz]).all():
                break
            x += ap * dx
            s += ap * ds
            z += ad * dz
        except (RuntimeError, ArithmeticError, np.linalg.LinAlgError):
            break
    return best if return_best and best is not None else {"status": "unresolved"}
