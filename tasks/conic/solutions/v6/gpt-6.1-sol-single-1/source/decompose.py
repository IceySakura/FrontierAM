"""Within-instance connected-component decomposition of large LPs and QPs."""

import hashlib
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from common import Problem, assess


def extract(A, rows, cols):
    T = A[rows].tocoo()
    return sparse.coo_matrix((T.data, (T.row, np.searchsorted(cols, T.col))),
                             shape=(len(rows), len(cols))).tocsc()


def solve_decomposed(p, budget=265.0):
    from poly import solve_poly, zero_lower_bound
    start = time.monotonic()
    if p.P.nnz and p.n < 600000:
        zero = zero_lower_bound(p, min(15.0, budget * 0.1))
        if zero is not None:
            return zero
    ac, pc = p.A.getnnz(axis=0), p.P.getnnz(axis=0)
    free_descent = (ac == 0) & (pc == 0) & (p.q != 0)
    if np.any(free_descent):
        j = np.flatnonzero(free_descent)[0]
        x = np.zeros(p.n)
        x[j] = -np.sign(p.q[j])
        return {"status": "dual_infeasible", "x": x, "s": np.zeros(p.m)}
    active = np.flatnonzero((ac != 0) | (pc != 0) | (p.q != 0))
    A, P = p.A[:, active].tocsr(), p.P[active][:, active].tocsr()
    n = len(active)
    G = sparse.bmat([[P.astype(bool), A.T.astype(bool)], [A.astype(bool), None]],
                    format="csr", dtype=bool)
    nc, labels = connected_components(G, directed=False)
    del G
    lv, lr = labels[:n], labels[n:]
    cv, cr = np.bincount(lv, minlength=nc), np.bincount(lr, minlength=nc)
    opv, opr = np.argsort(lv, kind="stable"), np.argsort(lr, kind="stable")
    ov, orow = np.r_[0, np.cumsum(cv)], np.r_[0, np.cumsum(cr)]
    pnz = np.bincount(lv, weights=P.getnnz(axis=0), minlength=nc)
    x, s, z = np.zeros(p.n), p.b.copy(), np.zeros(p.m)
    empty = np.flatnonzero(cv == 0)
    for c in empty:
        rows = opr[orow[c]:orow[c + 1]]
        if not len(rows):
            continue
        bad = p.eq[rows] & (np.abs(p.b[rows]) > 1e-12) | (~p.eq[rows] & (p.b[rows] < 0))
        if np.any(bad):
            r = rows[np.flatnonzero(bad)[0]]
            z[r] = -np.sign(p.b[r]) if p.eq[r] else 1
            return {"status": "primal_infeasible", "z": z}
    tiny = np.flatnonzero((cv == 2) & (cr == 2) & (pnz == 0))
    done = np.zeros(nc, dtype=bool)
    done[empty] = True
    done[tiny] = True
    # Two-row, two-variable LP vertices can be solved in one vectorized pass.
    if len(tiny):
        v0, v1 = opv[ov[tiny]], opv[ov[tiny] + 1]
        r0, r1 = opr[orow[tiny]], opr[orow[tiny] + 1]
        a = np.asarray(A[r0, v0]).ravel()
        b = np.asarray(A[r0, v1]).ravel()
        c = np.asarray(A[r1, v0]).ravel()
        d = np.asarray(A[r1, v1]).ravel()
        det = a * d - b * c
        valid = np.abs(det) > 1e-14 * np.maximum(1, np.maximum(abs(a * d), abs(b * c)))
        safe = np.where(valid, det, 1)
        xx0 = (p.b[r0] * d - b * p.b[r1]) / safe
        xx1 = (a * p.b[r1] - p.b[r0] * c) / safe
        q0, q1 = p.q[active[v0]], p.q[active[v1]]
        zz0, zz1 = (-q0 * d + c * q1) / safe, (-a * q1 + b * q0) / safe
        valid &= (p.eq[r0] | (zz0 >= 0)) & (p.eq[r1] | (zz1 >= 0))
        done[tiny[~valid]] = False
        x[active[v0[valid]]], x[active[v1[valid]]] = xx0[valid], xx1[valid]
        z[r0[valid]], z[r1[valid]] = zz0[valid], zz1[valid]
        s[r0[valid]], s[r1[valid]] = 0, 0
    cache = {}
    pending = np.flatnonzero(~done)
    pending = pending[np.argsort(-(cv[pending] + cr[pending]), kind="stable")]
    for c in pending:
        remaining = budget - (time.monotonic() - start)
        if remaining <= 1:
            return {"status": "unresolved"}
        cols, rows = opv[ov[c]:ov[c + 1]], opr[orow[c]:orow[c + 1]]
        q, b = p.q[active[cols]], p.b[rows]
        As, Ps = extract(A, rows, cols), extract(P, cols, cols)
        eq = p.eq[rows]
        cones = []
        ends = np.r_[0, np.flatnonzero(eq[1:] != eq[:-1]) + 1, len(rows)]
        for i, j in zip(ends[:-1], ends[1:]):
            if j > i:
                cones.append({"type": "zero" if eq[i] else "nonnegative", "dim": int(j - i)})
        sub = Problem(Ps, q, As, b, cones)
        key = hashlib.blake2b(digest_size=24)
        for a in (q, b, eq, As.data, As.indices, As.indptr, Ps.data, Ps.indices, Ps.indptr):
            key.update(a.tobytes())
        key = key.digest()
        if key in cache:
            r = cache[key]
        else:
            r = solve_poly(sub, time_limit=min(remaining - 0.5, 45.0))
            cache[key] = r
        status = r["status"]
        if status == "primal_infeasible":
            zz = np.zeros(p.m)
            zz[rows] = r["z"]
            out = {"status": status, "z": zz}
            if assess(p, z=zz, status=status)[0]:
                return out
            return {"status": "unresolved"}
        if status == "dual_infeasible":
            xx = np.zeros(p.n)
            xx[active[cols]] = r["x"]
            ss = -p.A @ xx
            if assess(p, x=xx, s=ss, status=status)[0]:
                return {"status": status, "x": xx, "s": ss}
            return {"status": "unresolved"}
        if status != "solved":
            return {"status": "unresolved"}
        x[active[cols]], z[rows] = r["x"], r["z"]
        s[rows] = r["s"]
    return {"status": "solved", "x": x, "s": s, "z": z}
