"""Exact sparse substitution of variables defined by one equality."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse


def prune(p):
    if any(k == "soc" for k, _, _, _ in p.blocks):
        return p, None
    lengths = np.diff(p.A.tocsr().indptr)
    keep_rows = np.ones(p.m, dtype=bool)
    for k, _, a, b in p.blocks:
        if k in ("zero", "nonnegative"):
            feasible = (p.b[a:b] == 0) if k == "zero" else (p.b[a:b] >= 0)
            keep_rows[a:b] &= (lengths[a:b] != 0) | ~feasible
    cols = ((np.diff(p.A.indptr) != 0) | (np.diff(p.P.indptr) != 0) | (p.q != 0))
    if keep_rows.all() and cols.all():
        return p, None
    rr, cc = np.flatnonzero(keep_rows), np.flatnonzero(cols)
    blocks, start = [], 0
    for k, d, a, b in p.blocks:
        width = np.count_nonzero(keep_rows[a:b])
        if width:
            blocks.append((k, width if k in ("zero", "nonnegative") else d,
                           start, start + width))
            start += width
    child = SimpleNamespace(A=p.A[rr][:, cc].tocsc(), P=p.P[cc][:, cc].tocsc(),
                            q=p.q[cc], b=p.b[rr], equal=p.equal[rr], blocks=blocks,
                            n=cc.size, m=rr.size,
                            core_offset=getattr(p, "core_offset", 0.0))
    return child, ("prune", p, cc, rr)


def eliminate(p):
    eqrows = np.flatnonzero(p.equal)
    if not eqrows.size or not p.n:
        return None
    Ae = p.A[eqrows].tocsc()
    single = np.flatnonzero(np.diff(Ae.indptr) == 1)
    if not single.size:
        return None
    er = Ae.indices[Ae.indptr[single]]
    row_norm = np.asarray(abs(Ae).max(axis=1).toarray()).ravel()
    strong = np.abs(Ae.data[Ae.indptr[single]]) >= 0.01 * row_norm[er]
    single, er = single[strong], er[strong]
    if not single.size:
        return None
    score = (np.diff(p.A.indptr)[single] +
             10 * np.diff(p.P.indptr)[single])
    order = np.lexsort((score, er))
    _, first = np.unique(er[order], return_index=True)
    chosen = order[first]
    piv = single[chosen]
    rows = eqrows[er[chosen]]
    coeff = Ae.data[Ae.indptr[piv]]
    free = np.ones(p.n, dtype=bool)
    free[piv] = False
    free = np.flatnonzero(free)
    keep = np.ones(p.m, dtype=bool)
    keep[rows] = False
    keep = np.flatnonzero(keep)
    T = p.A[rows, :][:, free].multiply((-1.0 / coeff)[:, None]).tocsc()
    constant = p.b[rows] / coeff
    Ac = p.A[keep, :][:, piv].tocsc()
    A = (p.A[keep, :][:, free] + Ac @ T).tocsc()
    if A.nnz > max(4 * p.A.nnz, p.A.nnz + 1000000):
        return None
    b = p.b[keep] - Ac @ constant
    Pf, Pc = p.P[free, :][:, free], p.P[piv, :][:, piv]
    cross = p.P[free, :][:, piv]
    CT = cross @ T
    P = (Pf + CT + CT.T + T.T @ (Pc @ T)).tocsc()
    P = ((P + P.T) * 0.5).tocsc()
    q = p.q[free] + cross @ constant + T.T @ (p.q[piv] + Pc @ constant)
    A.eliminate_zeros()
    P.eliminate_zeros()
    equal = p.equal[keep]
    blocks = []
    start = 0
    removed = np.zeros(p.m, dtype=bool)
    removed[rows] = True
    for k, d, a, bb in p.blocks:
        width = bb - a - np.count_nonzero(removed[a:bb])
        if width:
            blocks.append((k, width if k == "zero" else d, start, start + width))
            start += width
    child = SimpleNamespace(A=A, P=P, q=np.asarray(q), b=np.asarray(b),
                            n=free.size, m=keep.size, equal=equal, blocks=blocks,
                            core_offset=float(getattr(p, "core_offset", 0.0)
                                + p.q[piv] @ constant + 0.5 * constant @ (Pc @ constant)))
    record = (p, piv, rows, coeff, free, keep, T, constant)
    return child, record


def presolve(p, rounds=6):
    records = []
    p, record = prune(p)
    if record is not None:
        records.append(record)
    if p.P.nnz > 3 * p.n:
        return p, records
    if not p.P.nnz and all(k in ("zero", "nonnegative") for k, _, _, _ in p.blocks):
        return p, records
    if not p.P.nnz and p.n > 2000 and p.m < 1.6 * p.n and all(
            k in ("zero", "nonnegative") for k, _, _, _ in p.blocks):
        row_nnz = np.diff(p.A.tocsr().indptr)
        if np.all(row_nnz[~p.equal] <= 1):
            return p, records
    for _ in range(rounds):
        r = eliminate(p)
        if r is None:
            break
        child, record = r
        records.append(record)
        p = child
        p, record = prune(p)
        if record is not None:
            records.append(record)
        if not p.n:
            break
    return p, records


def restore(result, records):
    status = result["status"]
    if status == "unresolved":
        return result
    for record in reversed(records):
        if isinstance(record[0], str):
            _, p, free, keep = record
            if status in ("solved", "dual_infeasible"):
                xx = np.zeros(p.n)
                xx[free] = result["x"]
                ss = np.zeros(p.m)
                if status == "solved":
                    ss[:] = p.b
                ss[keep] = result["s"]
                result["x"], result["s"] = xx, ss
            if status in ("solved", "primal_infeasible"):
                zz = np.zeros(p.m)
                zz[keep] = result["z"]
                result["z"] = zz
            continue
        p, piv, rows, coeff, free, keep, T, constant = record
        if status in ("solved", "dual_infeasible"):
            xx = np.zeros(p.n)
            xx[free] = result["x"]
            xx[piv] = T @ result["x"]
            if status == "solved":
                xx[piv] += constant
            ss = np.zeros(p.m)
            ss[keep] = result["s"]
            result["x"], result["s"] = xx, ss
        if status in ("solved", "primal_infeasible"):
            zz = np.zeros(p.m)
            zz[keep] = result["z"]
            g = p.A.T @ zz
            if status == "solved":
                g += p.P @ xx + p.q
            zz[rows] = -g[piv] / coeff
            result["z"] = zz
    return result
