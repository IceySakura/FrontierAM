"""Reversible sparse affine reductions for the lightweight conic paths."""

import numpy as np

from native import affine_product


class Stage:
    def __init__(self, p, tr, tc, tv, offset, keep, prow, pcol, coeff, bnew):
        self.p, self.tr, self.tc, self.tv = p, tr, tc, tv
        self.offset, self.keep = offset, keep
        self.prow, self.pcol, self.coeff = prow, pcol, coeff
        self.bnew = bnew

    def recover(self, r):
        p = self.p
        out = {"status": r["status"]}
        if r["status"] in ("solved", "dual_infeasible"):
            x = np.bincount(self.tr, weights=self.tv * r["x"][self.tc], minlength=p.n)
            if r["status"] == "solved":
                x += self.offset
            out["x"] = x
            s = np.zeros(p.m)
            if r["status"] == "solved":
                dropped = np.ones(p.m, dtype=bool)
                dropped[self.keep] = False
                s[dropped & p.nn] = self.bnew[dropped & p.nn]
            s[self.keep] = r["s"]
            out["s"] = s
        if r["status"] in ("solved", "primal_infeasible"):
            z = np.zeros(p.m)
            z[self.keep] = r["z"]
            g = p.mul("A", z, True)
            if r["status"] == "solved":
                g += p.q
            z[self.prow] = -g[self.pcol] / self.coeff
            out["z"] = z
        return out


def reduce_once(p):
    from fast import Raw
    from minisparse import CSC
    if len(p.P[0]) or not np.any(p.eq):
        return p, None
    ad, ar, ap, ac = p.A
    eqe = p.eq[ar]
    deg = np.bincount(ac[eqe], minlength=p.n)
    cand = np.flatnonzero(eqe & (deg[ac] == 1))
    rowdeg = np.bincount(ar[eqe], minlength=p.m)
    cand = cand[rowdeg[ar[cand]] <= 40]
    if not len(cand):
        return p, None
    order = np.lexsort((ac[cand], np.diff(ap)[ac[cand]], ar[cand]))
    cand = cand[order]
    cand = cand[np.r_[True, ar[cand][1:] != ar[cand][:-1]]]
    prows, pcols, coeff = ar[cand], ac[cand], ad[cand]
    free_mask = np.ones(p.n, dtype=bool)
    free_mask[pcols] = False
    free = np.flatnonzero(free_mask)
    if not len(free):
        return p, None
    newcols = np.full(p.n, -1, np.int32)
    newcols[free] = np.arange(len(free))
    rowpiv = np.full(p.m, -1, np.int32)
    rowpiv[prows] = pcols
    rowcoef = np.ones(p.m)
    rowcoef[prows] = coeff
    use = (rowpiv[ar] >= 0) & free_mask[ac]
    tr = np.r_[free, rowpiv[ar[use]]].astype(np.int32)
    tc = np.r_[np.arange(len(free), dtype=np.int32), newcols[ac[use]]]
    tv = np.r_[np.ones(len(free)), -ad[use] / rowcoef[ar[use]]]
    order = np.argsort(tr, kind="stable")
    tr, tc, tv = tr[order], tc[order], tv[order]
    tp = np.r_[0, np.cumsum(np.bincount(tr, minlength=p.n))].astype(np.int64)
    capacity = int(np.diff(ap).astype(np.int64) @ np.diff(tp))
    if capacity > 12000000:
        return p, None
    offset = np.zeros(p.n)
    offset[pcols] = p.b[prows] / coeff
    bnew = p.b - p.mul("A", offset)
    keep_mask = np.ones(p.m, np.uint8)
    keep_mask[prows] = 0
    rr, cc, aa = affine_product(p.A, (tp, tc, tv), keep_mask, capacity)
    full = CSC.coo(p.m, len(free), rr, cc, aa)
    rowcount = np.bincount(full.rows, minlength=p.m)
    redundant = (rowcount == 0) & ((p.nn & (bnew >= 0)) | (p.eq & (bnew == 0)))
    keep_mask[redundant] = 0
    keep = np.flatnonzero(keep_mask)
    A = full.select_rows(keep)
    out = object.__new__(Raw)
    out.meta, out.n, out.m = p.meta, len(free), len(keep)
    out.q = np.bincount(tc, weights=tv * p.q[tr], minlength=len(free))
    out.b = bnew[keep]
    out.A = A.data, A.rows, A.ptr, A.cols
    out.P = np.array([]), np.array([], np.int32), np.zeros(out.n + 1, np.int32), np.array([], np.int32)
    out.eq, out.nn = p.eq[keep].copy(), p.nn[keep].copy()
    out.blocks = []
    pos = 0
    for t, d, lo, hi in p.blocks:
        width = int(np.sum(keep_mask[lo:hi])) if t in ("zero", "nonnegative") else hi - lo
        if width:
            out.blocks.append((t, width if t in ("zero", "nonnegative") else d, pos, pos + width))
            pos += width
    out.nonlinear = p.nonlinear
    return out, Stage(p, tr, tc, tv, offset, keep, prows, pcols, coeff, bnew)


def reduce(p, rounds=5):
    stages = []
    for _ in range(rounds):
        p, stage = reduce_once(p)
        if stage is None:
            break
        stages.append(stage)
    return p, stages


def recover(result, stages):
    for stage in reversed(stages):
        result = stage.recover(result)
    return result
