"""Faithful re-implementation of the public independent checker.

Used internally to decide which (verified) status to report.  All tolerances
and normalisations mirror samples/io/check.py exactly.
"""
import math
import numpy as np

TOL = 1e-8


def _norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def _exp_violation(v):
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    boundary = max(abs(b), max(a, 0.0), max(-c, 0.0))
    if b > 0.0 and c > 0.0:
        interior = max(0.0, a - b * (math.log(c) - math.log(b)))
        return min(interior, boundary)
    return boundary


def _smat(u, d, rows, cols, scales):
    mat = np.zeros((d, d))
    vals = u * scales
    mat[rows, cols] = vals
    mat[cols, rows] = vals
    return mat


def psd_index(d):
    rows, cols, scales = [], [], []
    for c in range(d):
        for r in range(c + 1):
            rows.append(r)
            cols.append(c)
            scales.append(1.0 if r == c else 1.0 / math.sqrt(2.0))
    return (np.array(rows), np.array(cols), np.array(scales))


def _exp_violation_vec(U):
    """Vectorized exp cone defect over rows of U (k,3)."""
    A, B, C = U[:, 0], U[:, 1], U[:, 2]
    boundary = np.maximum(np.abs(B), np.maximum(np.maximum(A, 0.0), np.maximum(-C, 0.0)))
    interior = np.full(len(U), np.inf)
    ok = (B > 0) & (C > 0)
    if np.any(ok):
        with np.errstate(divide="ignore", invalid="ignore"):
            val = A[ok] - B[ok] * (np.log(C[ok]) - np.log(B[ok]))
        interior[ok] = np.maximum(val, 0.0)
    return np.minimum(boundary, interior)


def cone_defects(v, layout, dual=False):
    worst = 0.0
    groups = {}
    for kind, dim, start, width in layout:
        groups.setdefault((kind, dim, width), []).append(start)
    for (kind, dim, width), starts in groups.items():
        idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(width)[None, :]
        B = v[idx]
        scale = np.maximum(1.0, np.max(np.abs(B), axis=1))
        U = B / scale[:, None]
        if kind == "zero":
            viol = 0.0 if dual else float(np.max(np.abs(U)))
        elif kind == "nonnegative":
            viol = max(0.0, -float(np.min(U)))
        elif kind == "soc":
            viol = max(0.0, float(np.max(np.linalg.norm(U[:, 1:], axis=1) - U[:, 0])))
        elif kind == "exponential":
            T = U.copy()
            if dual:
                T = np.column_stack((-U[:, 1], -U[:, 0], math.e * U[:, 2]))
            viol = float(np.max(_exp_violation_vec(T)))
        else:
            rows, cols, scales = psd_index(dim)
            viol = 0.0
            for row in U:
                mat = _smat(row, dim, rows, cols, scales)
                try:
                    ev = np.linalg.eigvalsh(mat)[0]
                except np.linalg.LinAlgError:
                    ev = -1e300
                viol = max(viol, max(0.0, -float(ev)))
        worst = max(worst, viol)
    return worst


def check_solved(P, q, A, b, layout, x, s, z, tol=TOL):
    """Return (passed, diagnostics dict)."""
    try:
        Ax = A.matvec(x)
        Px = P.matvec(x)
        Atz = A.rmatvec(z)
        rp = _norm(Ax + s - b) / (1.0 + max(_norm(Ax), _norm(s), _norm(b)))
        rd = _norm(Px + q + Atz) / (1.0 + max(_norm(Px), _norm(q), _norm(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        gap = abs(p0 - d0)
        scale = max(abs(p0), abs(d0))
        gaprel = gap / scale if scale else 0.0
        cp = cone_defects(s, layout, False)
        cd = cone_defects(z, layout, True)
        ok = (rp <= tol and rd <= tol and cp <= tol and cd <= tol
              and (gap <= tol or gaprel <= tol)
              and np.isfinite([rp, rd, p0, d0, cp, cd]).all())
        return bool(ok), dict(rp=rp, rd=rd, gap=gap, gaprel=gaprel, cp=cp, cd=cd)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
        return False, {}


def check_primal_infeasible(P, q, A, b, layout, z, tol=TOL):
    try:
        a = A.maxabs or 1.0
        zn = _norm(z)
        bn = _norm(b)
        if zn <= 0 or bn <= 0:
            return False, {}
        zh = z / zn
        beta = -float((b / bn) @ zh)
        if not (beta > tol):
            return False, {}
        lin = _norm(A.rmatvec(zh)) / a / beta
        cd = cone_defects(zh, layout, True) / beta
        return bool(lin <= tol and cd <= tol), dict(beta=beta, lin=lin, cd=cd)
    except (ValueError, FloatingPointError, OverflowError):
        return False, {}


def check_dual_infeasible(P, q, A, b, layout, x, s, tol=TOL):
    try:
        a = A.maxabs or 1.0
        h = P.maxabs or 1.0
        xn = _norm(x)
        qn = _norm(q)
        if xn <= 0 or qn <= 0:
            return False, {}
        xh = x / xn
        sh = s / xn
        alpha = -float((q / qn) @ xh)
        if not (alpha > tol):
            return False, {}
        quad = _norm(P.matvec(xh)) / h / alpha
        lin = _norm(A.matvec(xh) + sh) / a / alpha
        cp = cone_defects(sh / a, layout, False) / alpha
        return bool(quad <= tol and lin <= tol and cp <= tol), dict(alpha=alpha, quad=quad, lin=lin, cp=cp)
    except (ValueError, FloatingPointError, OverflowError):
        return False, {}
