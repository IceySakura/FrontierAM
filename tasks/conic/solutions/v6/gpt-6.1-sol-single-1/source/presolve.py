"""Eliminate fixed variables and empty scalar constraints reversibly."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from common import Problem, norm, cone_defect


class Presolve:
    def __init__(self, p):
        self.original = p
        self.fixed = np.zeros(p.n)
        free = np.ones(p.n, dtype=bool)
        keep = np.ones(p.m, dtype=bool)
        b = p.b.copy()
        fr, fc, fa, layers = [], [], [], []
        A = p.A.tocsr()
        scalar = np.zeros(p.m, dtype=bool)
        nonnegative = np.zeros(p.m, dtype=bool)
        for kind, _, i, j in p.blocks:
            if kind in ("zero", "nonnegative"):
                scalar[i:j] = True
            if kind == "nonnegative":
                nonnegative[i:j] = True
        self.early = None
        for _ in range(40):
            cols = np.flatnonzero(free)
            C = A[:, cols].tocsr()
            counts = np.diff(C.indptr)
            empty = keep & scalar & (counts == 0)
            bad = empty & ((p.eq & (np.abs(b) > 1e-9)) | (~p.eq & (b < -1e-10)))
            if np.any(bad):
                r = np.flatnonzero(bad)[0]
                z = np.zeros(p.m)
                z[r] = -np.sign(b[r]) if p.eq[r] else 1.0
                self.fix_rows, self.fix_cols = np.array(fr, int), np.array(fc, int)
                self.fix_alts, self.layers = np.array(fa, int), layers
                self.free, self.keep = cols, np.flatnonzero(keep)
                self.constants = b.copy()
                self.early = self.restore({"status": "primal_infeasible", "z": z},
                                          already_full=True)
                return
            keep[empty] = False
            cumulative = np.r_[0, np.cumsum(counts)]
            for kind, dim, lo, hi in p.blocks:
                if kind in ("zero", "nonnegative") or not keep[lo] or cumulative[hi] != cumulative[lo]:
                    continue
                block = b[lo:hi]
                if cone_defect(block, [(kind, dim, 0, hi - lo)]) <= 1e-11:
                    keep[lo:hi] = False
                elif kind == "soc":
                    z = np.zeros(p.m)
                    r = np.linalg.norm(block[1:])
                    z[lo] = 1
                    if r:
                        z[lo + 1:hi] = -block[1:] / r
                    self.fix_rows, self.fix_cols = np.array(fr, int), np.array(fc, int)
                    self.fix_alts, self.layers = np.array(fa, int), layers
                    self.free, self.keep = cols, np.flatnonzero(keep)
                    self.constants = b.copy()
                    self.early = self.restore({"status": "primal_infeasible", "z": z}, already_full=True)
                    return
            ids = np.flatnonzero(keep & p.eq & (counts == 1))
            loc = C.indices[C.indptr[ids]]
            _, first = np.unique(loc, return_index=True)
            ids = ids[first]
            loc = loc[first]
            alts = np.full(len(ids), -1, dtype=int)
            boundrows = np.flatnonzero(keep & nonnegative & (counts == 1))
            if len(boundrows):
                bc = C.indices[C.indptr[boundrows]]
                bv = C.data[C.indptr[boundrows]]
                val = b[boundrows] / bv
                ub = np.full(len(cols), np.inf)
                lb = np.full(len(cols), -np.inf)
                np.minimum.at(ub, bc[bv > 0], val[bv > 0])
                np.maximum.at(lb, bc[bv < 0], val[bv < 0])
                both = np.isfinite(ub) & np.isfinite(lb)
                same = both & (np.abs(ub - lb) <= 1e-12 * np.maximum(1, np.maximum(abs(ub), abs(lb))))
                same[loc] = False
                newloc = np.flatnonzero(same)
                if len(newloc):
                    ur, lr = np.full(len(cols), p.m, int), np.full(len(cols), p.m, int)
                    isup = (bv > 0) & (val == ub[bc])
                    islo = (bv < 0) & (val == lb[bc])
                    np.minimum.at(ur, bc[isup], boundrows[isup])
                    np.minimum.at(lr, bc[islo], boundrows[islo])
                    ids = np.r_[ids, ur[newloc]]
                    loc = np.r_[loc, newloc]
                    alts = np.r_[alts, lr[newloc]]
            if not len(ids):
                break
            origcols = cols[loc]
            values = b[ids] / C.data[C.indptr[ids]]
            self.fixed[origcols] = values
            b -= p.A[:, origcols] @ values
            keep[ids] = False
            free[origcols] = False
            fr.extend(ids.tolist())
            fc.extend(origcols.tolist())
            fa.extend(alts.tolist())
            layers.append(len(fr))
        self.fix_rows, self.fix_cols = np.array(fr, int), np.array(fc, int)
        self.fix_alts, self.layers = np.array(fa, int), layers
        self.free, self.keep = np.flatnonzero(free), np.flatnonzero(keep)
        self.constants = b.copy()
        cones = []
        for kind, d, i, j in p.blocks:
            w = int(np.count_nonzero(keep[i:j]))
            if w:
                cones.append({"type": kind, "dim": d if kind == "psd" else w})
        self.problem = Problem(p.P[self.free][:, self.free].tocsc(),
                               (p.q + p.P @ self.fixed)[self.free],
                               p.A[self.keep][:, self.free].tocsc(),
                               b[self.keep], cones)

    def restore(self, r, already_full=False):
        status = r["status"]
        p = self.original
        if status == "unresolved":
            return r
        out = {"status": status}
        if "x" in r:
            x = self.fixed.copy() if status == "solved" else np.zeros(p.n)
            x[self.free] += r["x"]
            out["x"] = x
            out["s"] = p.b - p.A @ x if status == "solved" else -p.A @ x
            if status == "solved" and "s" in r:
                out["s"][self.keep] = r["s"]
                removed = np.ones(p.m, bool)
                removed[self.keep] = False
                out["s"][removed] = self.constants[removed]
                out["s"][p.eq] = 0
        if "z" in r:
            if already_full:
                z = r["z"].copy()
            else:
                z = np.zeros(p.m)
                z[self.keep] = r["z"]
            if len(self.fix_rows):
                residual = p.A.T @ z
                if status == "solved":
                    residual += p.P @ out["x"] + p.q
                if np.all(self.fix_alts < 0):
                    B = p.A[self.fix_rows][:, self.fix_cols].T.tocsr()
                    v = spsolve_triangular(B, -residual[self.fix_cols], lower=False)
                    z[self.fix_rows] += v
                else:
                    ends = [0] + self.layers
                    for lo, hi in reversed(list(zip(ends[:-1], ends[1:]))):
                        rows = self.fix_rows[lo:hi].copy()
                        cols = self.fix_cols[lo:hi]
                        coef = np.asarray(p.A[rows, cols]).ravel()
                        val = -residual[cols] / coef
                        flip = (self.fix_alts[lo:hi] >= 0) & (val < 0)
                        rows[flip] = self.fix_alts[lo:hi][flip]
                        coef = np.asarray(p.A[rows, cols]).ravel()
                        val = -residual[cols] / coef
                        z[rows] += val
                        residual += p.A[rows].T @ val
            out["z"] = z
        return out


class AffineReduction:
    """Substitute equality-local columns in independent sparse batches."""

    def __init__(self, p, rounds=5):
        self.steps = []
        for _ in range(rounds):
            if p.n == 0 or not np.any(p.eq) or p.P.nnz:
                break
            eqids = np.flatnonzero(p.eq)
            C = p.A[eqids].tocsc()
            count = np.diff(C.indptr)
            degree = p.A.getnnz(axis=0)
            candidates = np.flatnonzero((count == 1) & (degree <= 10))
            if not len(candidates):
                break
            relrow = C.indices[C.indptr[candidates]]
            vals = C.data[C.indptr[candidates]]
            rowdeg = C.getnnz(axis=1)
            good = (abs(vals) > 1e-10) & (rowdeg[relrow] <= 40)
            candidates, relrow, vals = candidates[good], relrow[good], vals[good]
            cost = degree[candidates] + 20 * (p.q[candidates] != 0) + 0.01 / np.maximum(abs(vals), 1e-8)
            order = np.lexsort((candidates, cost, relrow))
            candidates, relrow = candidates[order], relrow[order]
            _, first = np.unique(relrow, return_index=True)
            piv, rows = candidates[first], eqids[relrow[first]]
            if len(piv) < 2:
                break
            coef = np.asarray(p.A[rows, piv]).ravel()
            free = np.flatnonzero(~np.isin(np.arange(p.n), piv))
            keep = np.ones(p.m, bool)
            keep[rows] = False
            E = p.A[rows][:, free].tocoo()
            data = np.r_[np.ones(len(free)), -E.data / coef[E.row]]
            rr, cc = np.r_[free, piv[E.row]], np.r_[np.arange(len(free)), E.col]
            T = sparse.coo_matrix((data, (rr, cc)), shape=(p.n, len(free))).tocsc()
            f = np.zeros(p.n)
            f[piv] = p.b[rows] / coef
            Anew = (p.A[keep] @ T).tocsc()
            Anew.sum_duplicates()
            Anew.eliminate_zeros()
            if Anew.nnz > max(3000000, p.A.nnz * 4):
                break
            cones = []
            for kind, d, i, j in p.blocks:
                w = int(np.count_nonzero(keep[i:j]))
                if w:
                    cones.append({"type": kind, "dim": d if kind == "psd" else w})
            self.steps.append((p, T, f, np.flatnonzero(keep), piv, rows, coef))
            p = Problem(sparse.csc_matrix((len(free), len(free))), np.asarray(T.T @ p.q).ravel(),
                        Anew, (p.b - p.A @ f)[keep], cones)
        self.problem = p

    def restore(self, r):
        if r["status"] == "unresolved":
            return r
        for p, T, f, keep, piv, rows, coef in reversed(self.steps):
            status = r["status"]
            out = {"status": status}
            if "x" in r:
                x = T @ r["x"]
                if status == "solved":
                    x += f
                out["x"] = x
                out["s"] = p.b - p.A @ x if status == "solved" else -p.A @ x
                if status == "solved" and "s" in r:
                    out["s"][keep] = r["s"]
                    out["s"][p.eq] = 0
            if "z" in r:
                z = np.zeros(p.m)
                z[keep] = r["z"]
                rd = p.A.T @ z
                if status == "solved":
                    rd += p.q + p.P @ out["x"]
                z[rows] = -rd[piv] / coef
                out["z"] = z
            r = out
        return r
