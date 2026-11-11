"""Sparse input, reversible equilibration, and original-coordinate checks."""

import json
import math
import numpy as np
from scipy import sparse, linalg


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def load(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata_json"]))
        mats = []
        for k in ("P", "A"):
            a = sparse.csc_matrix(
                (f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]),
                shape=tuple(f[k + "_shape"]),
            )
            a.sum_duplicates()
            a.eliminate_zeros()
            a.sort_indices()
            mats.append(a)
        return Problem(mats[0], f["q"], mats[1], f["b"], meta["cones"])


class ConeBlocks(list):
    def finish(self):
        groups, exp = {}, []
        self.scalar, self.psd_blocks = [], []
        for kind, d, i, j in self:
            if kind == "soc":
                groups.setdefault(d, []).append(i)
            elif kind == "exponential":
                exp.append(i)
            elif kind == "psd":
                self.psd_blocks.append((kind, d, i, j))
            else:
                self.scalar.append((kind, d, i, j))
        self.soc_groups = {d: np.asarray(starts)[:, None] + np.arange(d) for d, starts in groups.items()}
        self.exp_indices = np.asarray(exp, dtype=int)[:, None] + np.arange(3)


class Problem:
    def __init__(self, P, q, A, b, cones):
        self.P, self.q, self.A, self.b = P, q, A, b
        self.cones = cones
        self.n, self.m = len(q), len(b)
        self.blocks = ConeBlocks()
        self.eq = np.zeros(self.m, dtype=bool)
        p = 0
        for c in cones:
            kind, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if kind == "psd" else d
            self.blocks.append((kind, d, p, p + w))
            if kind == "zero":
                self.eq[p:p + w] = True
            p += w
        self.blocks.finish()


def smat(v, d):
    i, j = np.triu_indices(d)
    order = np.lexsort((i, j))
    i, j = i[order], j[order]
    a = np.zeros((d, d))
    a[i, j] = v / np.where(i == j, 1.0, math.sqrt(2.0))
    a[j, i] = a[i, j]
    return a


def svec(a):
    d = len(a)
    i, j = np.triu_indices(d)
    order = np.lexsort((i, j))
    i, j = i[order], j[order]
    return a[i, j] * np.where(i == j, 1.0, math.sqrt(2.0))


def exp_defect(v):
    a, b, c = np.moveaxis(np.asarray(v), -1, 0)
    boundary = np.maximum.reduce((np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)))
    ok = (b > 0) & (c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        interior = np.maximum(0, a - b * (np.log(c) - np.log(b)))
    return np.where(ok, np.minimum(boundary, interior), boundary)


def cone_defect(v, blocks, dual=False):
    worst = 0.0
    if isinstance(blocks, ConeBlocks):
        for idx in blocks.soc_groups.values():
            a = v[idx]
            a = a / np.maximum(1, np.max(abs(a), axis=1))[:, None]
            worst = max(worst, float(np.max(np.linalg.norm(a[:, 1:], axis=1) - a[:, 0], initial=0)))
        if len(blocks.exp_indices):
            a = v[blocks.exp_indices]
            a = a / np.maximum(1, np.max(abs(a), axis=1))[:, None]
            if dual:
                a = np.column_stack((-a[:, 1], -a[:, 0], math.e * a[:, 2]))
            worst = max(worst, float(np.max(exp_defect(a), initial=0)))
        iterable = blocks.scalar + blocks.psd_blocks
    else:
        iterable = blocks
    for kind, d, i, j in iterable:
        u = v[i:j] / max(1.0, norm(v[i:j]))
        if kind == "zero":
            e = 0 if dual else norm(u)
        elif kind == "nonnegative":
            e = max(0.0, -float(np.min(u)))
        elif kind == "soc":
            e = max(0.0, float(np.linalg.norm(u[1:])) - u[0])
        elif kind == "exponential":
            e = float(exp_defect(np.array([-u[1], -u[0], math.e * u[2]]) if dual else u))
        else:
            e = max(0.0, -float(linalg.eigh(
                smat(u, d), eigvals_only=True, subset_by_index=(0, 0),
                check_finite=False, driver="evr")[0]))
        worst = max(worst, e)
    return worst


def assess(p, x=None, s=None, z=None, tol=9e-9, status="solved", check_all=False):
    try:
        if status == "unresolved":
            return False, float("inf")
        if status == "solved":
            if not all(np.isfinite(v).all() for v in (x, s, z)):
                return False, float("inf")
            ax, px, atz = p.A @ x, p.P @ x, p.A.T @ z
            rp = norm(ax + s - p.b) / (1 + max(norm(ax), norm(s), norm(p.b)))
            rd = norm(px + p.q + atz) / (1 + max(norm(px), norm(p.q), norm(atz)))
            pp = 0.5 * x @ px + p.q @ x
            dd = -0.5 * x @ px - p.b @ z
            gap = abs(pp - dd) / max(1.0, abs(pp), abs(dd))
            err = max(rp, rd, gap)
            if err > tol and not check_all:
                return False, err
            err = max(err, cone_defect(s, p.blocks), cone_defect(z, p.blocks, True))
            return err <= tol, err
        if status == "primal_infeasible":
            if norm(z) == 0 or norm(p.b) == 0 or not np.isfinite(z).all():
                return False, float("inf")
            z = z / norm(z)
            beta = -(p.b / norm(p.b)) @ z
            if beta <= 1e-8:
                return False, float("inf")
            err = max(norm(p.A.T @ z) / (norm(p.A.data) or 1.0),
                      cone_defect(z, p.blocks, True)) / beta
        else:
            if norm(x) == 0 or norm(p.q) == 0 or not np.isfinite(x).all():
                return False, float("inf")
            t = norm(x)
            x, s = x / t, s / t
            alpha = -(p.q / norm(p.q)) @ x
            if alpha <= 1e-8:
                return False, float("inf")
            a = norm(p.A.data) or 1.0
            err = max(norm(p.P @ x) / (norm(p.P.data) or 1.0),
                      norm(p.A @ x + s) / a, cone_defect(s / a, p.blocks)) / alpha
        return err <= tol, err
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return False, float("inf")


def equilibrate(p, rounds=5):
    A, P = p.A.copy(), p.P.copy()
    D, E = np.ones(p.n), np.ones(p.m)
    for _ in range(rounds):
        ac = np.asarray(abs(A).max(axis=0).toarray()).ravel()
        pc = np.asarray(abs(P).max(axis=0).toarray()).ravel()
        colnorm = np.maximum(ac, pc)
        colnorm[colnorm == 0] = 1.0
        d = 1.0 / np.sqrt(np.maximum(colnorm, 1e-8))
        d = np.clip(d, 1e-3, 1e3)
        A = A.multiply(d).tocsc()
        P = P.multiply(d).multiply(d[:, None]).tocsc()
        D *= d
        ar = np.asarray(abs(A).max(axis=1).toarray()).ravel()
        for kind, dim, i, j in p.blocks:
            if kind not in ("zero", "nonnegative"):
                ar[i:j] = float(np.max(ar[i:j])) or 1.0
        ar[ar == 0] = 1.0
        e = 1.0 / np.sqrt(np.maximum(ar, 1e-8))
        e = np.clip(e, 1e-3, 1e3)
        A = A.multiply(e[:, None]).tocsc()
        E *= e
    q, b = p.q * D, p.b * E
    c = 1.0 / max(1.0, norm(q), norm(P.data))
    return Problem(P * c, q * c, A, b, p.cones), D, E, c
