"""Fixed-factor operator splitting and active-set refinement for sparse QPs."""

import os

import numpy as np
from scipy import sparse

from ipm import equilibrate
from linear import SparseFactors, positions
from polish import candidates
from validation import valid
from boxkkt import make as make_box_system


def norm(x):
    return float(np.max(np.abs(x), initial=0.))


def prepare(P0, q0, A0, b0, cs):
    A, b = A0[cs.perm, :].tocsc(), b0[cs.perm]
    P, q, A, b, dx, er, cost = equilibrate(P0, q0, A, b, cs)
    return {"P": P, "q": q, "A": A, "b": b, "dx": dx, "er": er, "cost": cost}


def recover(candidate, prepared, cs):
    out = {"status": candidate["status"]}
    if "x" in candidate:
        out["x"] = prepared["dx"] * candidate["x"]
    if "s" in candidate:
        s = np.empty(len(candidate["s"]))
        s[cs.perm] = candidate["s"] / prepared["er"]
        out["s"] = s
    if "z" in candidate:
        z = np.empty(len(candidate["z"]))
        z[cs.perm] = prepared["er"] * candidate["z"] / prepared["cost"]
        out["z"] = z
    return out


def polish_result(P0, q0, A0, b0, cs, candidate, prepared, ordering=None):
    if candidate.get("status") != "solved":
        return None
    P, q, A, b = (prepared[k] for k in ("P", "q", "A", "b"))
    x = candidate["x"] / prepared["dx"]
    z = prepared["cost"] * candidate["z"][cs.perm] / prepared["er"]
    eq = np.arange(len(b)) < cs.neq
    try:
        for polished in candidates(P, q, A, b, eq, x, z, 1e-5, 3, ordering):
            result = recover(polished, prepared, cs)
            if valid(P0, q0, A0, b0, None, result, cs, tol=7e-9):
                return result
    except (np.linalg.LinAlgError, RuntimeError, ValueError, MemoryError):
        pass
    return None


def solve(P0, q0, A0, b0, cs, budget=.12):
    prepared = prepare(P0, q0, A0, b0, cs)
    P, q, A, b = (prepared[k] for k in ("P", "q", "A", "b"))
    n, m, ne = len(q), len(b), cs.neq
    eq = np.arange(m) < ne
    if cs.nnonneg == 0:
        try:
            for polished in candidates(P, q, A, b, eq, np.zeros(n), np.zeros(m), max_steps=1):
                result = recover(polished, prepared, cs)
                if valid(P0, q0, A0, b0, None, result, cs, tol=7e-9):
                    return result, prepared
        except (np.linalg.LinAlgError, RuntimeError, ValueError, MemoryError):
            pass
    AT = A.T.tocsc()
    base_rho = .1
    rho = np.full(m, base_rho)
    rho[:ne] *= 1000
    sigma, alpha = 1e-6, 1.6
    box = make_box_system(P, A, ne, sigma)
    if box is not None:
        try:
            guess = box.equality_guess(q, b)
            if guess is not None and all(np.isfinite(v).all() for v in guess):
                gx, gy = guess
                gs = b - A @ gx
                gs[:ne] = 0.
                gs[ne:] = np.maximum(gs[ne:], 0.)
                candidate = recover({"status": "solved", "x": gx, "s": gs, "z": gy}, prepared, cs)
                if valid(P0, q0, A0, b0, None, candidate, cs, tol=8.5e-9):
                    return candidate, prepared
                certificate = recover({"status": "primal_infeasible", "z": gy}, prepared, cs)
                if valid(P0, q0, A0, b0, None, certificate, cs, tol=5e-9):
                    return certificate, prepared
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError):
            pass
        lu = box.refactor(rho)
        ordering = box.ordering
    else:
        D = sparse.diags(1 / rho) if m else sparse.csc_matrix((0, 0))
        K = sparse.bmat([[P + sigma * sparse.eye(n), AT], [A, -D]], format="csc")
        K.sort_indices()
        diagonal = positions(K, n + np.arange(m), n + np.arange(m))
        factory = SparseFactors(native=False)
        lu = factory(K)
        ordering = factory.perm
    inherit_polish = box is None or os.getenv("ADMM_POLISH_ORDER", "1") != "0"
    work_per_iteration = max(1, lu.nnz + 8 * (n + m))
    work_iterations = max(5, int(3e8 * budget / work_per_iteration))
    x, y = np.zeros(n), np.zeros(m)
    v = np.minimum(np.zeros(m), b)
    v[:ne] = b[:ne]
    old_x, old_y = x.copy(), y.copy()
    check_iterations = {10, 30, 80, 160, 350, 700, 1200}
    max_iter = min(int(os.getenv("ADMM_ITERATIONS", "1500")), work_iterations)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        sol = lu.solve(np.r_[sigma * x - q, v - y / rho])
        xt, nu = sol[:n], sol[n:]
        vt = v + (nu - y) / rho
        relaxed = alpha * vt + (1 - alpha) * v
        x = alpha * xt + (1 - alpha) * x
        vn = np.minimum(relaxed + y / rho, b)
        vn[:ne] = b[:ne]
        y += rho * (relaxed - vn)
        y[ne:] = np.maximum(y[ne:], 0.)
        v = vn
        if iteration % 10:
            continue
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            break
        Ax, Px, ATy = A @ x, P @ x, AT @ y
        rp = norm(Ax - v) / (1 + max(norm(Ax), norm(v), norm(b)))
        rd = norm(Px + q + ATy) / (1 + max(norm(Px), norm(q), norm(ATy)))
        s = b - v
        approximate = {"status": "solved", "x": x, "s": s, "z": y}
        original = recover(approximate, prepared, cs)
        if max(rp, rd) < 1e-7 and valid(P0, q0, A0, b0, None, original, cs, tol=7e-9):
            return original, prepared
        if iteration in check_iterations and max(rp, rd) < .2:
            try:
                for polished in candidates(P, q, A, b, eq, x.copy(), y.copy(),
                                           max(1e-5, min(.02, 5 * rp)), 3,
                                           ordering if inherit_polish else None):
                    result = recover(polished, prepared, cs)
                    if valid(P0, q0, A0, b0, None, result, cs, tol=7e-9):
                        return result, prepared
            except (np.linalg.LinAlgError, RuntimeError, ValueError, MemoryError):
                pass
        if iteration % 50 == 0:
            dy, dx = y - old_y, x - old_x
            cert = recover({"status": "primal_infeasible", "z": dy}, prepared, cs)
            if valid(P0, q0, A0, b0, None, cert, cs, tol=5e-9):
                return cert, prepared
            cert = recover({"status": "dual_infeasible", "x": dx, "s": -A @ dx}, prepared, cs)
            if valid(P0, q0, A0, b0, None, cert, cs, tol=5e-9):
                return cert, prepared
            old_x, old_y = x.copy(), y.copy()
            ratio = np.sqrt(max(rp, 1e-16) / max(rd, 1e-16))
            if (ratio > 5 or ratio < .2) and iteration < max_iter // 2:
                base_rho = float(np.clip(base_rho * ratio, 1e-6, 1e6))
                rho[:] = base_rho
                rho[:ne] *= 1000
                if box is not None:
                    lu = box.refactor(rho)
                    ordering = box.ordering
                else:
                    K.data[diagonal] = -1 / rho
                    lu = factory(K)
                    ordering = factory.perm
    if np.isfinite(x).all() and np.isfinite(y).all():
        if iteration < 10:
            try:
                for polished in candidates(P, q, A, b, eq, x.copy(), y.copy(), .01, 1,
                                           ordering if inherit_polish else None):
                    result = recover(polished, prepared, cs)
                    if valid(P0, q0, A0, b0, None, result, cs, tol=8.5e-9):
                        return result, prepared
            except (np.linalg.LinAlgError, RuntimeError, ValueError, MemoryError):
                pass
        ss = np.maximum(b[ne:] - v[ne:], 0.)
        zz = np.maximum(y[ne:], 0.)
        prepared["initial"] = (x, y[:ne], ss + .01 / (1 + zz), zz + .01 / (1 + ss))
    prepared["ordering"] = ordering
    prepared["polish_ordering"] = ordering if inherit_polish else None
    return None, prepared
