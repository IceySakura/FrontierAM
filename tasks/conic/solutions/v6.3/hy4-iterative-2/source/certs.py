"""Infeasibility certificate construction.

A certificate is produced by solving an auxiliary convex problem and then
checking the exact independent conditions before reporting it.

  primal_infeasible: find z in K* with A'z = 0 and b'z < 0.
  dual_infeasible:   find x with Px = 0, s = -Ax in K, and q'x < 0.

Both search problems are made bounded by a small quadratic regularizer
(P = eps*I), which removes the need for box constraints in the conic case.
"""
import math
import time

import numpy as np

from nputil import CSC


def dense_csc(M):
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows, cols = np.nonzero(M)
    # np.nonzero gives row-major order; need column-major (CSC)
    order = np.lexsort((rows, cols))
    rows, cols = rows[order], cols[order]
    data = M[rows, cols]
    indptr = np.searchsorted(cols, np.arange(M.shape[1] + 1))
    return CSC(data, rows.astype(np.int64), indptr.astype(np.int64), M.shape)


def _primal_cert_linear(P, q, A, b, layout, deadline, check_fn):
    """Farkas certificate for zero/nonnegative cones, via HiGHS."""
    import highs
    n, m = len(q), len(b)
    lo = np.full(m, -1.0)
    hi = np.ones(m)
    for kind, dim, start, width in layout:
        if kind == "nonnegative":
            lo[start:start + width] = 0.0
        elif kind != "zero":
            return None
    try:
        hs = highs.HighsSolver(m, n)
        At = A.transpose_csc()
        cap = 0.5 if n + m < 30000 else (2.0 if n + m < 300000 else 6.0)
        hs.set_model(At, np.zeros(n), np.zeros(n), b, None,
                     highs._time_options(deadline, cap), col_lower=lo, col_upper=hi)
        if time.monotonic() > deadline:
            return None
        status, sol, obj = hs.solve()
        if "kOptimal" not in str(status):
            return None
        z = np.array(sol.col_value, dtype=np.float64).ravel()
        if float(b @ z) >= 0:
            return None
        ok, _ = check_fn(z)
        return z if ok else None
    except Exception:
        return None


def _dual_cert_linear(P, q, A, b, layout, deadline, check_fn):
    """Dual infeasibility direction for zero/nonnegative cones, via HiGHS."""
    import highs
    n, m = len(q), len(b)
    lower = np.empty(m)
    upper = np.empty(m)
    for kind, dim, start, width in layout:
        if kind == "zero":
            lower[start:start + width] = 0.0
            upper[start:start + width] = 0.0
        elif kind == "nonnegative":
            lower[start:start + width] = -np.inf
            upper[start:start + width] = 0.0
        else:
            return None
    if P.nnz:
        AA = CSC.vstack([A, P])
        lower = np.concatenate((lower, np.zeros(n)))
        upper = np.concatenate((upper, np.zeros(n)))
    else:
        AA = A
    try:
        hs = highs.HighsSolver(n, len(lower))
        cap = 0.5 if n + m < 30000 else (2.0 if n + m < 300000 else 6.0)
        hs.set_model(AA, lower, upper, q, None, highs._time_options(deadline, cap),
                     col_lower=np.full(n, -1.0), col_upper=np.ones(n))
        if time.monotonic() > deadline:
            return None
        status, sol, obj = hs.solve()
        if "kOptimal" not in str(status):
            return None
        x = np.array(sol.col_value, dtype=np.float64).ravel()
        if float(q @ x) >= 0:
            return None
        s = -A.matvec(x)
        ok, _ = check_fn(x, s)
        return (x, s) if ok else None
    except Exception:
        return None


T_EXP = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., math.e]])


def _primal_cert_conic(P, q, A, b, layout, deadline, check_fn, solve_fn):
    """Farkas certificate for general cones, via an auxiliary conic solve."""
    n, m = len(q), len(b)
    Ad = A.dense()
    rows, rhs, cones = [], [], []
    # A' z = 0, encoded as zero cone of size n
    rows.append(Ad.T.copy())
    rhs.append(np.zeros(n))
    cones.append({"type": "zero", "dim": n})
    for kind, dim, start, width in layout:
        if kind == "zero":
            continue
        block = np.zeros((width, m))
        cols = np.arange(start, start + width)
        if kind == "exponential":
            block[:, cols] = -T_EXP
        else:
            block[:, cols] = -np.eye(width)
        rows.append(block)
        rhs.append(np.zeros(width))
        cones.append({"type": kind, "dim": dim})
    AA = np.vstack(rows)
    bb = np.concatenate(rhs)
    eps = 1e-6 * max(1.0, float(np.max(np.abs(b))) if m else 1.0)
    Paux = dense_csc(eps * np.eye(m))
    Aaux = dense_csc(AA)
    laux = []
    pos = 0
    for c in cones:
        w = c["dim"] * (c["dim"] + 1) // 2 if c["type"] == "psd" else c["dim"]
        laux.append((c["type"], c["dim"], pos, w))
        pos += w
    try:
        res = solve_fn(Paux, b.copy(), Aaux, bb, laux, deadline)
        if res.get("status") != "solved":
            return None
        z = res["x"]
        if float(b @ z) >= 0:
            return None
        ok, _ = check_fn(z)
        return z if ok else None
    except Exception:
        return None


def _dual_cert_conic(P, q, A, b, layout, deadline, check_fn, solve_fn):
    """Dual infeasibility direction for general cones."""
    n, m = len(q), len(b)
    Ad, Pd = A.dense(), P.dense()
    AA = Ad.copy()
    bb = np.zeros(m)
    cones = [{"type": k, "dim": d} for k, d, _, _ in layout]
    if P.nnz and np.any(Pd):
        AA = np.vstack((AA, Pd))
        bb = np.concatenate((bb, np.zeros(n)))
        cones.append({"type": "zero", "dim": n})
    eps = 1e-6 * max(1.0, float(np.max(np.abs(q))) if n else 1.0)
    Paux = dense_csc(eps * np.eye(n))
    Aaux = dense_csc(AA)
    laux = []
    pos = 0
    for c in cones:
        w = c["dim"] * (c["dim"] + 1) // 2 if c["type"] == "psd" else c["dim"]
        laux.append((c["type"], c["dim"], pos, w))
        pos += w
    try:
        res = solve_fn(Paux, q.copy(), Aaux, bb, laux, deadline)
        if res.get("status") != "solved":
            return None
        x = res["x"]
        s = -A.matvec(x)
        if float(q @ x) >= 0:
            return None
        ok, _ = check_fn(x, s)
        return (x, s) if ok else None
    except Exception:
        return None


def find_primal(P, q, A, b, layout, deadline, check_fn, conic_solve=None):
    linear_only = all(k in ("zero", "nonnegative") for k, _, _, _ in layout)
    if linear_only:
        z = _primal_cert_linear(P, q, A, b, layout, deadline, check_fn)
        if z is not None:
            return z
    if (conic_solve is not None and len(q) <= 2500 and len(b) <= 6000
            and A.nnz + P.nnz + len(q) + len(b) < 60000):
        return _primal_cert_conic(P, q, A, b, layout, deadline, check_fn, conic_solve)
    return None


def find_dual(P, q, A, b, layout, deadline, check_fn, conic_solve=None):
    linear_only = all(k in ("zero", "nonnegative") for k, _, _, _ in layout)
    if linear_only:
        r = _dual_cert_linear(P, q, A, b, layout, deadline, check_fn)
        if r is not None:
            return r
    if (conic_solve is not None and len(q) <= 2500 and len(b) <= 6000
            and A.nnz + P.nnz + len(q) + len(b) < 60000):
        return _dual_cert_conic(P, q, A, b, layout, deadline, check_fn, conic_solve)
    return None
