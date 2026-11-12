"""Active-set KKT polishing for polyhedral quadratic programs."""

import numpy as np
from scipy import sparse
from numeric import linalg, splu

from linear import SparseFactors


def norm(v):
    return float(np.max(np.abs(v), initial=0.))


def correction(P, B, rhs, factory=None):
    n, k = P.shape[0], B.shape[0]
    K = sparse.bmat([[P, B.T], [B, None]], format="csc")
    scale = np.ones(n + k)
    for _ in range(2):
        magnitude = np.asarray(abs(K).max(axis=0).toarray()).ravel()
        d = 1 / np.sqrt(np.maximum(magnitude, 1e-12))
        scale *= d
        K = (sparse.diags(d) @ K @ sparse.diags(d)).tocsc()
    target = scale * rhs
    if n + k <= 600:
        sol = linalg.lstsq(K.toarray(), target, cond=1e-14,
                           lapack_driver="gelsy", check_finite=False)[0]
    else:
        reg = np.r_[np.full(n, 1e-12), np.full(k, -1e-12)]
        regularized = K + sparse.diags(reg, format="csc")
        lu = (factory(regularized) if factory is not None else
              splu(regularized, permc_spec="MMD_AT_PLUS_A",
                   diag_pivot_thresh=0., options={"SymmetricMode": True}))
        sol = lu.solve(target)
        previous = np.inf
        for _ in range(5):
            residual = target - K @ sol
            error = norm(residual)
            if error < 1e-13 * max(1., norm(target)) or error >= previous:
                break
            previous = error
            sol += lu.solve(residual)
    return scale * sol


def candidates(P, q, A, b, eq, x, z, tolerance=1e-5, max_steps=5, ordering=None):
    rownorm = np.asarray(abs(A).max(axis=1).toarray()).ravel()
    row_scale = np.maximum(np.maximum(rownorm, np.abs(b)), 1e-12)
    slack = b - A @ x
    strength = np.abs(z) * rownorm
    dual_scale = max(norm(q), norm(P @ x), 1e-8)
    active = eq | ((slack <= tolerance * row_scale) &
                   ((strength > 1e-8 * dual_scale) | (np.abs(slack) <= 1e-10 * row_scale)))
    previous = None
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        B = A[idx, :].tocsc()
        zz = np.zeros_like(z)
        zz[idx] = z[idx]
        rhs = np.r_[-P @ x - q - B.T @ zz[idx], b[idx] - B @ x]
        factory = None
        if ordering is not None and len(ordering) == len(x) + len(b):
            mapping = np.full(len(ordering), -1, dtype=np.int64)
            mapping[:len(x)] = np.arange(len(x))
            mapping[len(x) + idx] = len(x) + np.arange(len(idx))
            p = mapping[ordering]
            factory = SparseFactors(native=False)
            factory.perm = p[p >= 0]
        try:
            step = correction(P, B, rhs, factory)
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return
        xx = x + step[:len(x)]
        zz[idx] += step[len(x):]
        ss = b - A @ xx
        candidate_s = ss.copy()
        candidate_s[eq] = 0.
        candidate_s[~eq] = np.maximum(candidate_s[~eq], 0.)
        tiny = (~eq) & (zz < 0) & (-zz * rownorm <= 1e-10 * dual_scale)
        zz[tiny] = 0.
        yield {"status": "solved", "x": xx, "s": candidate_s, "z": zz}
        negative = active & ~eq & (zz * rownorm < -1e-9 * dual_scale)
        violated = ~active & (ss < -1e-9 * row_scale)
        if not np.any(negative) and not np.any(violated):
            return
        active[negative] = False
        active[violated] = True
        key = np.packbits(active).tobytes()
        if key == previous:
            return
        previous = key
        x, z = xx, zz
