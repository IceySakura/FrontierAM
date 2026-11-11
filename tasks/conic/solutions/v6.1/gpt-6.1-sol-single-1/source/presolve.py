"""Exact affine elimination of structurally singleton equality columns."""

import numpy as np
from scipy import sparse


class Stage:
    def __init__(self, p, T, offset, keep, pivrows, pivcols, coeff):
        self.p, self.T, self.offset, self.keep = p, T, offset, keep
        self.pivrows, self.pivcols, self.coeff = pivrows, pivcols, coeff

    def recover(self, r):
        p = self.p
        if r["status"] in ("solved", "dual_infeasible"):
            x = self.T @ r["x"]
            if r["status"] == "solved":
                x += self.offset
            r["x"] = x
            s = np.zeros(p.m)
            s[self.keep] = r["s"]
            r["s"] = s
        if r["status"] in ("solved", "primal_infeasible"):
            z = np.zeros(p.m)
            z[self.keep] = r["z"]
            grad = p.A.T @ z
            if r["status"] == "solved":
                grad += p.P @ r["x"] + p.q
            z[self.pivrows] = -grad[self.pivcols] / self.coeff
            r["z"] = z
        return r


def reduce_once(p):
    from solver import Problem, log
    if len(p.eq) == 0 or p.n > 250000:
        return p, None
    F = p.A[p.eq].tocsc()
    degrees = np.diff(F.indptr)
    cand = np.flatnonzero(degrees == 1)
    if not len(cand):
        return p, None
    rows = F.indices[F.indptr[cand]]
    rowdeg = np.diff(F.tocsr().indptr)
    ok = rowdeg[rows] <= 40
    cand, rows = cand[ok], rows[ok]
    if not len(cand):
        return p, None
    score = np.diff(p.A.indptr)[cand] + 10 * np.diff(p.P.indptr)[cand]
    order = np.lexsort((cand, score, rows))
    cand, rows = cand[order], rows[order]
    first = np.r_[True, rows[1:] != rows[:-1]]
    pivcols, erows = cand[first], rows[first]
    pivrows = p.eq[erows]
    coeff = F.data[F.indptr[pivcols]]
    ix = np.ones(p.n, dtype=bool)
    ix[pivcols] = False
    free = np.flatnonzero(ix)
    if not len(free):
        return p, None
    C = p.A[pivrows][:, free].tocoo()
    # Restrict fill-in from eliminated quadratic variables.
    if p.P.nnz and np.sum(rowdeg[erows].astype(float) ** 2) > 2e7:
        return p, None
    T = sparse.csc_matrix(
        (np.r_[np.ones(len(free)), -C.data / coeff[C.row]],
         (np.r_[free, pivcols[C.row]], np.r_[np.arange(len(free)), C.col])),
        shape=(p.n, len(free)),
    )
    offset = np.zeros(p.n)
    offset[pivcols] = p.b[pivrows] / coeff
    rm = np.zeros(p.m, dtype=bool)
    rm[pivrows] = True
    keep = np.flatnonzero(~rm)
    out = object.__new__(Problem)
    out.P = (T.T @ p.P @ T).tocsc()
    out.P.sum_duplicates()
    out.P.eliminate_zeros()
    out.q = np.asarray(T.T @ (p.q + p.P @ offset)).ravel()
    out.A = (p.A[keep] @ T).tocsc()
    out.A.sum_duplicates()
    out.A.eliminate_zeros()
    out.Ar = out.A.tocsr()
    out.b = p.b[keep] - p.A[keep] @ offset
    out.meta = p.meta
    out.n, out.m = len(free), len(keep)
    out.kind = p.kind[keep].copy()
    out.blocks = []
    pos = 0
    for t, d, lo, hi in p.blocks:
        width = int(np.sum(~rm[lo:hi])) if t == "zero" else hi - lo
        if width:
            nd = width if t == "zero" else d
            out.blocks.append((t, nd, pos, pos + width))
            pos += width
    out.eq = np.flatnonzero(out.kind == 0)
    out.ineq = np.flatnonzero(out.kind == 1)
    out.linear = np.flatnonzero(out.kind < 2)
    out.nonlinear = [v for v in out.blocks if v[0] not in ("zero", "nonnegative")]
    log("presolve", (p.n, p.m, p.P.nnz, p.A.nnz), "->",
        (out.n, out.m, out.P.nnz, out.A.nnz))
    return out, Stage(p, T, offset, keep, pivrows, pivcols, coeff)


def reduce_problem(p, rounds=5):
    from solver import Problem, log
    stages = []
    inactive = ((np.diff(p.A.indptr) == 0) & (np.diff(p.P.indptr) == 0) & (p.q == 0))
    if np.any(inactive):
        free = np.flatnonzero(~inactive)
        T = sparse.csc_matrix((np.ones(len(free)), (free, np.arange(len(free)))),
                              shape=(p.n, len(free)))
        out = object.__new__(Problem)
        out.__dict__.update(p.__dict__)
        out.n = len(free)
        out.P = p.P[free][:, free].tocsc()
        out.A = p.A[:, free].tocsc()
        out.Ar = out.A.tocsr()
        out.q = p.q[free]
        stages.append(Stage(p, T, np.zeros(p.n), np.arange(p.m),
                            np.array([], dtype=int), np.array([], dtype=int), np.array([])))
        log("remove empty columns", p.n, "->", out.n)
        p = out
    for _ in range(rounds):
        p, st = reduce_once(p)
        if st is None:
            break
        stages.append(st)
    return p, stages
