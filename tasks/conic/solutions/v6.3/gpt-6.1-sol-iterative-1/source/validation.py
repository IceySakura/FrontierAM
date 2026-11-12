"""Checks in original coordinates before a candidate is written."""

import numpy as np

from cones import ConeSystem


def norm(v):
    return float(np.max(np.abs(v), initial=0.))


def scaled_matrix(matrix, cs, name):
    cached = getattr(cs, name, None)
    if cached is not None and cached[0] is matrix:
        return cached[1]
    result = matrix.copy()
    result.data /= norm(matrix.data) or 1.
    setattr(cs, name, (matrix, result))
    return result


def valid(P, q, A, b, cones, result, cs=None, tol=1e-8):
    status = result.get("status", "unresolved")
    if status == "unresolved":
        return False
    if cs is None:
        cs = ConeSystem(cones)
    for name in ("x", "s", "z"):
        if name in result and not np.isfinite(result[name]).all():
            return False
    ne = cs.neq
    if status == "solved":
        x, s, z = (result[k] for k in ("x", "s", "z"))
        ax, px, az = A @ x, P @ x, A.T @ z
        if norm(ax + s - b) > tol * (1 + max(norm(ax), norm(s), norm(b))):
            return False
        if norm(px + q + az) > tol * (1 + max(norm(px), norm(q), norm(az))):
            return False
        p, d = float(.5 * x @ px + q @ x), float(-.5 * x @ px - b @ z)
        if not np.isfinite([p, d]).all() or abs(p - d) > tol * max(1., abs(p), abs(d)):
            return False
        sp, zp = s[cs.perm], z[cs.perm]
        return norm(sp[:ne]) <= tol and cs.defect(sp[ne:]) <= tol and cs.defect(zp[ne:], True) <= tol
    if status == "primal_infeasible":
        zn, bn = norm(result["z"]), norm(b)
        if not zn or not bn:
            return False
        z = result["z"] / zn
        margin = -float((b / bn) @ z)
        return (margin > max(tol, 1e-8) and norm(scaled_matrix(A, cs, "_certificate_A").T @ z) <= tol * margin and
                cs.defect(z[cs.perm][ne:], True) <= tol * margin)
    if status == "dual_infeasible":
        xn, qn = norm(result["x"]), norm(q)
        if not xn or not qn:
            return False
        x, s = result["x"] / xn, result["s"] / xn
        margin = -float((q / qn) @ x)
        scale = norm(A.data) or 1.
        return (margin > max(tol, 1e-8) and norm(scaled_matrix(P, cs, "_certificate_P") @ x) <= tol * margin and
                norm(scaled_matrix(A, cs, "_certificate_A") @ x + s / scale) <= tol * margin and
                norm(s[cs.perm][:ne]) <= tol * scale * margin and
                cs.defect(s[cs.perm][ne:] / scale) <= tol * margin)
    return False
