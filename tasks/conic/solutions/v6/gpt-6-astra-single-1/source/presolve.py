"""Exact elimination of unpenalized singleton equality columns."""

import numpy as np
from scipy import sparse

from problem import Problem, blocks


class Elimination:
    def __init__(self, original, free, pivot, erows, kept, B, C, diagonal):
        self.original = original
        self.free, self.pivot = free, pivot
        self.erows, self.kept = erows, kept
        self.B, self.C, self.diagonal = B, C, diagonal

    def recover(self, result):
        r = dict(result)
        if "x" in r:
            x = np.zeros(self.original.n)
            x[self.free] = r["x"]
            rhs = -self.B @ r["x"]
            if r["status"] == "solved":
                rhs += self.original.b[self.erows]
            x[self.pivot] = rhs / self.diagonal
            r["x"] = x
        if "s" in r:
            s = np.zeros(self.original.m)
            s[self.kept] = r["s"]
            r["s"] = s
        if "z" in r:
            z = np.zeros(self.original.m)
            z[self.kept] = r["z"]
            z[self.erows] = -(self.C.T @ r["z"]) / self.diagonal
            r["z"] = z
        return r


def eliminate(p):
    eq = np.zeros(p.m, bool)
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
    erows = np.flatnonzero(eq)
    if erows.size == 0 or p.n == 0:
        return p, None
    E = p.A[erows].tocsc()
    counts = np.diff(E.indptr)
    eligible = (counts == 1) & (p.q == 0) & (np.diff(p.P.indptr) == 0)
    cand = np.flatnonzero(eligible)
    if cand.size < 4:
        return p, None
    rows = E.indices[E.indptr[cand]]
    total = np.diff(p.A.indptr)[cand]
    order = np.lexsort((cand, total, rows))
    cand, rows = cand[order], rows[order]
    first = np.r_[True, rows[1:] != rows[:-1]]
    pivot, eliminated = cand[first], rows[first]
    if pivot.size < 4:
        return p, None
    eqcounts = np.diff(E.tocsr().indptr)
    degree = np.diff(p.A.indptr)[pivot] - 1
    estimated = np.sum(eqcounts[eliminated] * degree, dtype=np.int64)
    if estimated > max(2000000, 5 * p.A.nnz):
        return p, None
    diagonal = E.data[E.indptr[pivot]]
    eliminated = erows[eliminated]
    free_mask = np.ones(p.n, bool)
    free_mask[pivot] = False
    free = np.flatnonzero(free_mask)
    keep_mask = np.ones(p.m, bool)
    keep_mask[eliminated] = False
    kept = np.flatnonzero(keep_mask)
    B = p.A[eliminated][:, free].tocsc()
    C = p.A[kept][:, pivot].tocsc()
    Cd = C @ sparse.diags(1 / diagonal, format="csc")
    A = (p.A[kept][:, free] - Cd @ B).tocsc()
    A.eliminate_zeros()
    b = p.b[kept] - Cd @ p.b[eliminated]
    cones = []
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            size = int(np.sum(keep_mask[sl]))
            if size:
                cones.append({"type": kind, "dim": size})
        else:
            cones.append({"type": kind, "dim": dim})
    reduced = Problem(p.P[free][:, free].tocsc(), p.q[free], A, b, cones)
    return reduced, Elimination(p, free, pivot, eliminated, kept, B, C, diagonal)


class FixedVariables:
    def __init__(self, original, free, fixed, values, eq_rows, lower_rows, upper_rows):
        self.original, self.free, self.fixed = original, free, fixed
        self.values = values
        self.eq_rows, self.lower_rows, self.upper_rows = eq_rows, lower_rows, upper_rows
        self.rows = original.A.tocsr()

    def recover(self, result):
        r = dict(result)
        p = self.original
        if "x" in r:
            x = np.zeros(p.n)
            x[self.free] = r["x"]
            if r["status"] == "solved":
                x[self.fixed] = self.values
            r["x"] = x
        if "z" in r:
            z = r["z"].copy()
            selected = np.r_[self.eq_rows, self.lower_rows, self.upper_rows]
            z[selected[selected >= 0]] = 0
            gradient = p.A.T @ z
            if r["status"] == "solved":
                gradient += p.P @ r["x"] + p.q
            for k, j in enumerate(self.fixed):
                eq = self.eq_rows[k]
                if eq >= 0:
                    z[eq] = -gradient[j] / self.rows.data[self.rows.indptr[eq]]
                elif gradient[j] >= 0:
                    row = self.lower_rows[k]
                    z[row] = -gradient[j] / self.rows.data[self.rows.indptr[row]]
                else:
                    row = self.upper_rows[k]
                    z[row] = -gradient[j] / self.rows.data[self.rows.indptr[row]]
            r["z"] = z
        return r


def fixed_variables(p):
    if p.n > 200000:
        return p, None
    rows = p.A.tocsr()
    singleton = np.diff(rows.indptr) == 1
    eq = np.zeros(p.m, bool)
    nn = np.zeros(p.m, bool)
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
        elif kind == "nonnegative":
            nn[sl] = True
    fixed = np.zeros(p.n, bool)
    values = np.zeros(p.n)
    eqrow = np.full(p.n, -1, np.int32)
    lower_row = np.full(p.n, -1, np.int32)
    upper_row = np.full(p.n, -1, np.int32)
    lower, upper = np.full(p.n, -np.inf), np.full(p.n, np.inf)
    for row in np.flatnonzero(singleton & (eq | nn)):
        pos = rows.indptr[row]
        col, val = rows.indices[pos], rows.data[pos]
        bound = p.b[row] / val
        if eq[row] and not fixed[col]:
            fixed[col], values[col], eqrow[col] = True, bound, row
        elif nn[row]:
            if val > 0 and bound < upper[col]:
                upper[col], upper_row[col] = bound, row
            elif val < 0 and bound > lower[col]:
                lower[col], lower_row[col] = bound, row
    tight = ~fixed & np.isfinite(lower) & (lower == upper)
    fixed[tight], values[tight] = True, lower[tight]
    index = np.flatnonzero(fixed)
    if not index.size:
        return p, None
    free = np.flatnonzero(~fixed)
    val = values[index]
    q = p.q[free] + p.P[free][:, index] @ val
    b = p.b - p.A[:, index] @ val
    reduced = Problem(p.P[free][:, free].tocsc(), q, p.A[:, free].tocsc(), b, p.cones)
    transform = FixedVariables(p, free, index, val, eqrow[index], lower_row[index], upper_row[index])
    return reduced, transform


class ConeReduction:
    def __init__(self, original, kept, zero_soc):
        self.original, self.kept, self.zero_soc = original, kept, zero_soc

    def recover(self, result):
        r = dict(result)
        if "s" in r:
            s = self.original.b.copy() if r["status"] == "solved" else np.zeros(self.original.m)
            s[self.kept] = r["s"]
            r["s"] = s
        if "z" in r:
            z = np.zeros(self.original.m)
            z[self.kept] = r["z"]
            for sl in self.zero_soc:
                z[sl.start] = np.linalg.norm(z[sl.start+1:sl.stop])
            r["z"] = z
        return r


def simplify_cones(p):
    from quality import cone_error
    empty = np.diff(p.A.tocsr().indptr) == 0
    keep = np.ones(p.m, bool)
    cones, converted = [], []
    for kind, dim, sl in blocks(p.cones):
        if kind == "soc" and empty[sl.start] and p.b[sl.start] == 0:
            converted.append(sl)
            kind = "zero"
        if kind in ("zero", "nonnegative"):
            redundant = empty[sl] & ((p.b[sl] == 0) if kind == "zero" else (p.b[sl] >= 0))
            keep[sl] &= ~redundant
            count = int(np.sum(keep[sl]))
            if count:
                cones.append({"type": kind, "dim": count})
        elif np.all(empty[sl]) and cone_error(p.b[sl], [{"type": kind, "dim": dim}]) <= 1e-13:
            keep[sl] = False
        else:
            cones.append({"type": kind, "dim": dim})
    if np.all(keep) and not converted:
        return p, None
    kept = np.flatnonzero(keep)
    reduced = Problem(p.P, p.q, p.A[kept].tocsc(), p.b[kept], cones)
    return reduced, ConeReduction(p, kept, converted)
