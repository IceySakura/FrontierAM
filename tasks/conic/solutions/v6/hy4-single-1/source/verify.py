"""Independent verification mirroring the task's float64 acceptance checks."""
from __future__ import annotations

import math

import numpy as np

from cones import ConeInfo, smat


def _norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def residual_report(P, q, A, b, ci: ConeInfo, x, s, z, tol=1e-8):
    """Return (passed, residuals dict, max ratio)."""
    try:
        x = np.asarray(x, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
            return False, {}, 1e9
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            Ax, Px, Atz = A @ x, P @ x, A.T @ z
            if not (np.isfinite(Ax).all() and np.isfinite(Px).all() and np.isfinite(Atz).all()):
                return False, {}, 1e9
            rp = _norm(Ax + s - b) / (1.0 + max(_norm(Ax), _norm(s), _norm(b)))
            rd = _norm(Px + q + Atz) / (1.0 + max(_norm(Px), _norm(q), _norm(Atz)))
            xPx, qx, bz = float(x @ Px), float(q @ x), float(b @ z)
            p0, d0 = 0.5 * xPx + qx, -0.5 * xPx - bz
            gap = abs(p0 - d0)
            scale = max(abs(p0), abs(d0))
            rel = gap / scale if scale else 0.0
            cp, _ = ci.defects(s, dual=False)
            cd, _ = ci.defects(z, dual=True)
            vals = [rp, rd, cp, cd, min(gap, rel)]
            r = dict(primal=rp, dual=rd, primal_cone=cp, dual_cone=cd,
                     gap_absolute=gap, gap_relative=rel,
                     complementarity=abs(float(s @ z)))
            ok = (rp <= tol and rd <= tol and cp <= tol and cd <= tol and
                  (gap <= tol or rel <= tol))
            return bool(ok), r, max(rp, rd, cp, cd, min(gap, rel)) / tol
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False, {}, 1e9


def check_primal_infeasible(P, q, A, b, ci: ConeInfo, z, tol=1e-8):
    try:
        z = np.asarray(z, dtype=np.float64)
        zn, bn = _norm(z), _norm(b)
        if zn <= 0 or bn <= 0 or not np.isfinite(z).all():
            return False, 1e9
        zh = z / zn
        beta = -float((b / bn) @ zh)
        if not (beta > tol):
            return False, 1e9
        a = _norm(A.data) if A.nnz else 1.0
        if a == 0:
            a = 1.0
        lin = _norm((A / a).T @ zh) / beta
        cd, _ = ci.defects(zh, dual=True)
        cone = cd / beta
        return (lin <= tol and cone <= tol), max(lin, cone) / tol
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False, 1e9


def check_dual_infeasible(P, q, A, b, ci: ConeInfo, x, s, tol=1e-8):
    try:
        x = np.asarray(x, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        xn, qn = _norm(x), _norm(q)
        if xn <= 0 or qn <= 0 or not (np.isfinite(x).all() and np.isfinite(s).all()):
            return False, 1e9
        xh, sh = x / xn, s / xn
        alpha = -float((q / qn) @ xh)
        if not (alpha > tol):
            return False, 1e9
        a = _norm(A.data) if A.nnz else 1.0
        h = _norm(P.data) if P.nnz else 1.0
        if a == 0:
            a = 1.0
        if h == 0:
            h = 1.0
        quad = _norm((P / h) @ xh) / alpha
        lin = _norm((A / a) @ xh + sh / a) / alpha
        cp, _ = ci.defects(sh / a, dual=False)
        cone = cp / alpha
        return (quad <= tol and lin <= tol and cone <= tol), max(quad, lin, cone) / tol
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False, 1e9
