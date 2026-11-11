"""Stopping tests in original, unscaled input coordinates."""

import math

import numpy as np
from scipy import linalg

from problem import blocks

_cone_cache = {}
_triangle_cache = {}


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def smat(v, d):
    if d not in _triangle_cache:
        _triangle_cache[d] = np.tril_indices(d)
    rows, cols = _triangle_cache[d]
    # Lower-row order equals the transpose of upper-column order.
    mat = np.zeros((d, d))
    mat[rows, cols] = v / np.where(rows == cols, 1.0, math.sqrt(2.0))
    mat[cols, rows] = mat[rows, cols]
    return mat


def svec(mat):
    d = mat.shape[0]
    if d not in _triangle_cache:
        _triangle_cache[d] = np.tril_indices(d)
    rows, cols = _triangle_cache[d]
    return mat[rows, cols] * np.where(rows == cols, 1.0, math.sqrt(2.0))


def exp_defect(a, b, c):
    boundary = max(abs(b), a, -c, 0.0)
    if b > 0.0 and c > 0.0:
        return min(boundary, max(0.0, a - b * (math.log(c) - math.log(b))))
    return boundary


def cone_error(v, cones, dual=False):
    error = 0.0
    key = id(cones)
    if key not in _cone_cache:
        scalar, soc, exponential, psd = [], {}, [], []
        for kind, dim, sl in blocks(cones):
            if kind in ("zero", "nonnegative"):
                scalar.append((kind, sl))
            elif kind == "soc":
                soc.setdefault(dim, []).append(np.arange(sl.start, sl.stop))
            elif kind == "exponential":
                exponential.append(np.arange(sl.start, sl.stop))
            else:
                psd.append((dim, sl))
        groups = [np.asarray(indices, dtype=np.int32) for indices in soc.values()]
        exp = np.asarray(exponential, dtype=np.int32).reshape(-1, 3)
        _cone_cache[key] = (cones, scalar, groups, exp, psd)
    unused, scalar, soc, exponential, psd = _cone_cache[key]
    for kind, sl in scalar:
        u = v[sl] / max(1.0, norm(v[sl]))
        if kind == "zero":
            err = 0.0 if dual else norm(u)
        else:
            err = max(0.0, -float(u.min()))
        error = max(error, err)
    for indices in soc:
        u = v[indices]
        u = u / np.maximum(1.0, np.max(np.abs(u), axis=1))[:, None]
        error = max(error, float(np.max(np.linalg.norm(u[:, 1:], axis=1) - u[:, 0])))
    if exponential.size:
        u = v[exponential]
        u = u / np.maximum(1.0, np.max(np.abs(u), axis=1))[:, None]
        a, b, c = (-u[:, 1], -u[:, 0], math.e * u[:, 2]) if dual else u.T
        boundary = np.maximum.reduce((np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)))
        interior = (b > 0) & (c > 0)
        boundary[interior] = np.minimum(boundary[interior],
            np.maximum(0, a[interior] - b[interior] * (np.log(c[interior]) - np.log(b[interior]))))
        error = max(error, float(np.max(boundary)))
    for dim, sl in psd:
        u = v[sl] / max(1.0, norm(v[sl]))
        eig = linalg.eigh(smat(u, dim), eigvals_only=True,
                         subset_by_index=[0, 0], check_finite=False)
        error = max(error, -float(eig[0]))
    return error


def residuals(p, r, cone=True):
    status = r["status"]
    if status == "unresolved":
        return {"unresolved": float("inf")}
    if not all(np.isfinite(v).all() for k, v in r.items() if k in ("x", "s", "z")):
        return {"nonfinite": float("inf")}
    if status == "solved":
        x, s, z = r["x"], r["s"], r["z"]
        if not np.isfinite(s @ z):
            return {"nonfinite_complementarity": float("inf")}
        Ax, Px, Atz = p.A @ x, p.P @ x, p.A.T @ z
        p0 = 0.5 * (x @ Px) + p.q @ x
        d0 = -0.5 * (x @ Px) - p.b @ z
        out = {
            "primal": norm(Ax + s - p.b) / (1 + max(norm(Ax), norm(s), norm(p.b))),
            "dual": norm(Px + p.q + Atz) / (1 + max(norm(Px), norm(p.q), norm(Atz))),
            "gap": abs(p0 - d0) / max(1.0, abs(p0), abs(d0)),
        }
        if cone:
            out["primal_cone"] = cone_error(s, p.cones)
            out["dual_cone"] = cone_error(z, p.cones, True)
        return out
    a = norm(p.A.data) or 1.0
    if status == "primal_infeasible":
        z = r["z"]
        if not norm(z) or not norm(p.b):
            return {"margin": float("inf")}
        z = z / norm(z)
        margin = -(p.b / norm(p.b)) @ z
        if margin <= 1e-8:
            return {"margin": float("inf")}
        As = p.A.copy()
        As.data /= a
        return {"linear": norm(As.T @ z) / margin,
                "cone": cone_error(z, p.cones, True) / margin}
    x, s = r["x"], r["s"]
    if not norm(x) or not norm(p.q):
        return {"margin": float("inf")}
    s = s / norm(x)
    x = x / norm(x)
    margin = -(p.q / norm(p.q)) @ x
    if margin <= 1e-8:
        return {"margin": float("inf")}
    As, Ps = p.A.copy(), p.P.copy()
    As.data /= a
    Ps.data /= norm(p.P.data) or 1.0
    return {"linear": norm(As @ x + s / a) / margin,
            "quadratic": norm(Ps @ x) / margin,
            "cone": cone_error(s / a, p.cones) / margin}


def acceptable(p, result, tol=8e-9):
    try:
        values = residuals(p, result)
        return bool(all(np.isfinite(v) and v <= tol for v in values.values()))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return False
