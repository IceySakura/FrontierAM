"""Residuals in the original input coordinates."""
import numpy as np


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def assess(P, q, A, b, K, result, tol=1e-8):
    status, x, s, z = result
    if status == "unresolved":
        return float("inf"), {}
    if any(v is not None and not np.isfinite(v).all() for v in (x, s, z)):
        return float("inf"), {}
    if status == "solved":
        Ax, Px, Atz = A @ x, P @ x, A.T @ z
        if not all(np.isfinite(v).all() for v in (Ax, Px, Atz)):
            return float("inf"), {}
        rp = norm(Ax + s - b) / (1 + max(norm(Ax), norm(s), norm(b)))
        rd = norm(Px + q + Atz) / (1 + max(norm(Px), norm(q), norm(Atz)))
        p = 0.5 * float(x @ Px) + float(q @ x)
        d = -0.5 * float(x @ Px) - float(b @ z)
        if not np.isfinite(p) or not np.isfinite(d):
            return float("inf"), {}
        gap = abs(p - d) / max(1.0, abs(p), abs(d))
        ck = K.violation(s)
        cd = K.violation(z, dual=True)
        res = dict(primal=rp, dual=rd, gap=gap, primal_cone=ck, dual_cone=cd)
    else:
        a, h = norm(A.data) or 1.0, norm(P.data) or 1.0
        if status == "primal_infeasible":
            nz, nb = norm(z), norm(b)
            if not nz or not nb:
                return float("inf"), {}
            z = z / nz
            margin = -float((b / nb) @ z)
            if not np.isfinite(margin) or margin <= tol:
                return float("inf"), {}
            res = dict(linear=norm(A.T @ z) / a / margin,
                       dual_cone=K.violation(z, dual=True) / margin)
        else:
            nx, nq = norm(x), norm(q)
            if not nx or not nq:
                return float("inf"), {}
            x, s = x / nx, s / nx
            margin = -float((q / nq) @ x)
            if not np.isfinite(margin) or margin <= tol:
                return float("inf"), {}
            res = dict(linear=norm(A @ x + s) / a / margin,
                       quadratic=norm(P @ x) / h / margin,
                       primal_cone=K.violation(s / a) / margin)
    if not all(np.isfinite(v) for v in res.values()):
        return float("inf"), res
    return max(res.values(), default=0.0) / tol, res
