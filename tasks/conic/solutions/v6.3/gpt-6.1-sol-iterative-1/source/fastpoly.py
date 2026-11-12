"""Low-startup-cost polyhedral path using raw CSC arrays and bundled HiGHS."""

import importlib.util
import os
import sys

import numpy as np

from simple import free_direction, solve as separable_solve


def norm(x):
    return float(np.max(np.abs(x), initial=0.))


class CSC:
    def __init__(self, data, indices, indptr, shape):
        self.data, self.indices, self.indptr = np.asarray(data, dtype=np.float64), indices, indptr
        self.shape, self.nnz = tuple(shape), len(data)
        self._cols = None

    @property
    def cols(self):
        if self._cols is None:
            self._cols = np.repeat(np.arange(self.shape[1]), np.diff(self.indptr))
        return self._cols

    def __matmul__(self, x):
        return np.bincount(self.indices, weights=self.data * x[self.cols], minlength=self.shape[0])

    def transpose_dot(self, x):
        return np.bincount(self.cols, weights=self.data * x[self.indices], minlength=self.shape[1])

    def diagonal(self):
        keep = self.indices == self.cols
        return np.bincount(self.indices[keep], weights=self.data[keep], minlength=self.shape[1])

    def canonical(self):
        return not np.any((self.indices[1:] <= self.indices[:-1]) & (self.cols[1:] == self.cols[:-1]))


def trivial(raw):
    q, b, cones = raw["q"], raw["b"], raw["cones"]
    unused = (np.diff(raw["A_indptr"]) == 0) & (np.diff(raw["P_indptr"]) == 0) & (q != 0)
    if np.any(unused):
        margin = float(np.sum(np.abs(q[unused]) / norm(q)))
        if margin > 1.1e-8:
            x = np.zeros(len(q))
            x[unused] = -np.sign(q[unused])
            return {"status": "dual_infeasible", "x": x, "s": np.zeros(len(b))}
    indices, weights, off = [], [], 0
    for c in cones:
        kind, d = c["type"], c["dim"]
        width = d * (d + 1) // 2 if kind == "psd" else d
        if kind == "zero":
            selected = off + np.flatnonzero(b[off:off + width] != 0)
            if len(selected):
                indices.append(selected)
                weights.append(-np.sign(b[selected]))
        else:
            if kind == "nonnegative":
                selected = off + np.flatnonzero(b[off:off + width] < 0)
            elif kind == "soc":
                selected = np.array([off], dtype=int) if b[off] < 0 else np.empty(0, dtype=int)
            elif kind == "exponential":
                rows = np.array([off + 1, off + 2])
                selected = rows[b[rows] < 0]
            else:
                j = np.arange(d)
                rows = off + j * (j + 3) // 2
                selected = rows[b[rows] < 0]
            if len(selected):
                indices.append(selected)
                weights.append(np.ones(len(selected)))
        off += width
    if indices:
        rows, values = np.concatenate(indices), np.concatenate(weights)
        counts = np.bincount(raw["A_indices"], minlength=len(b))
        keep = counts[rows] == 0
        rows, values = rows[keep], values[keep]
        if len(rows) and -float((b[rows] / norm(b)) @ values) > 1.1e-8:
            z = np.zeros(len(b))
            z[rows] = values
            return {"status": "primal_infeasible", "z": z}
    return None


def highs_module():
    name = "scipy.optimize._highspy._core"
    if name in sys.modules:
        return sys.modules[name]
    base = os.path.dirname(importlib.util.find_spec("scipy").origin)
    directory = os.path.join(base, "optimize", "_highspy")
    path = next(os.path.join(directory, f) for f in os.listdir(directory)
                if f.startswith("_core.") and f.endswith(".so"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def valid(P, q, A, b, eq, result):
    status = result["status"]
    if any(not np.isfinite(result[k]).all() for k in ("x", "s", "z") if k in result):
        return False
    if status == "solved":
        x, s, z = (result[k] for k in ("x", "s", "z"))
        ax, px, az = A @ x, P @ x, A.transpose_dot(z)
        if norm(ax + s - b) > 1e-8 * (1 + max(norm(ax), norm(s), norm(b))):
            return False
        if norm(px + q + az) > 1e-8 * (1 + max(norm(px), norm(q), norm(az))):
            return False
        p, d = float(.5 * x @ px + q @ x), float(-.5 * x @ px - b @ z)
        return (np.isfinite([p, d]).all() and abs(p - d) <= 1e-8 * max(1., abs(p), abs(d)) and
                norm(s[eq]) <= 1e-8 and np.min(s[~eq], initial=0.) >= -1e-9 and
                np.min(z[~eq], initial=0.) >= -1e-9)
    if status == "primal_infeasible":
        zn, bn = norm(result["z"]), norm(b)
        if not zn or not bn:
            return False
        z = result["z"] / zn
        margin = -float((b / bn) @ z)
        return (margin > 1e-8 and norm(A.transpose_dot(z)) <= 1e-8 * (norm(A.data) or 1.) * margin and
                np.min(z[~eq], initial=0.) >= -1e-8 * margin)
    if status == "dual_infeasible":
        xn, qn = norm(result["x"]), norm(q)
        if not xn or not qn:
            return False
        x, s = result["x"] / xn, result["s"] / xn
        margin = -float((q / qn) @ x)
        scale = norm(A.data) or 1.
        return (margin > 1e-8 and norm(P @ x) <= 1e-8 * (norm(P.data) or 1.) * margin and
                norm(A @ x + s) <= 1e-8 * scale * margin and norm(s[eq]) <= 1e-8 * scale * margin and
                np.min(s[~eq], initial=0.) >= -1e-8 * scale * margin)
    return False


def polish(P, q, A, b, eq, x, z):
    slack = b - A @ x
    active = eq | ((np.abs(slack) < 1e-6 * (1 + np.abs(b))) & (z > 1e-9))
    idx = np.flatnonzero(active)
    n, k = len(x), len(idx)
    if n + k > 700:
        return x, z
    B = np.zeros((k, n))
    mapping = np.full(len(b), -1, dtype=int)
    mapping[idx] = np.arange(k)
    use = active[A.indices]
    B[mapping[A.indices[use]], A.cols[use]] = A.data[use]
    PP = np.zeros((n, n))
    PP[P.indices, P.cols] = P.data
    K = np.zeros((n + k, n + k))
    K[:n, :n], K[:n, n:], K[n:, :n] = PP, B.T, B
    old = np.r_[x, z[idx]]
    rhs = np.r_[-q, b[idx]] - K @ old
    correction = np.linalg.lstsq(K, rhs, rcond=1e-13)[0]
    new = old + correction
    zz = np.zeros(len(b))
    zz[idx] = new[n:]
    return (new[:n], zz) if np.min(zz[~eq], initial=0.) >= -1e-9 else (x, z)


def try_solve(raw):
    cones, q, b = raw["cones"], raw["q"], raw["b"]
    if any(c["type"] not in ("zero", "nonnegative") for c in cones) or os.getenv("FORCE_IPM"):
        return None
    P = CSC(*(raw["P_" + k] for k in ("data", "indices", "indptr", "shape")))
    A = CSC(*(raw["A_" + k] for k in ("data", "indices", "indptr", "shape")))
    eq = np.zeros(len(b), dtype=bool)
    off = 0
    for c in cones:
        if c["type"] == "zero":
            eq[off:off + c["dim"]] = True
        off += c["dim"]
    if not np.any(q) and not np.any(b[eq]) and np.min(b[~eq], initial=0.) >= 0:
        return {"status": "solved", "x": np.zeros(len(q)), "s": b.copy(), "z": np.zeros(len(b))}
    direction = free_direction(P, q, A, b)
    if direction is not None and valid(P, q, A, b, eq, direction):
        return direction
    if P.nnz <= len(q) and not np.any(P.indices != P.cols):
        diag = P.diagonal()
        if not np.any((diag <= 0) & (q != 0)):
            xx = np.zeros(len(q))
            use = diag > 0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                xx[use] = -q[use] / diag[use]
            if np.isfinite(xx).all():
                ss = b - A @ xx
                ss[eq] = 0.
                ss[~eq] = np.maximum(ss[~eq], 0.)
                result = {"status": "solved", "x": xx, "s": ss, "z": np.zeros(len(b))}
                if valid(P, q, A, b, eq, result):
                    return result
    if P.nnz and len(q) >= 200:
        # A large diagonal/box problem can still take the exact path.
        if P.nnz > len(q) or np.any(np.bincount(A.indices, minlength=len(b)) > 1):
            return None
    if not P.canonical() or not A.canonical():
        return None
    result = separable_solve(P, q, A, b, cones)
    if result is not None and valid(P, q, A, b, eq, result):
        return result
    if P.nnz and len(q) >= 200:
        return None
    h = highs_module()
    n, m = len(q), len(b)
    model = h.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_, lp.col_upper_ = np.full(n, -h.kHighsInf), np.full(n, h.kHighsInf)
    lp.row_lower_, lp.row_upper_ = np.where(eq, b, -h.kHighsInf), b
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    if P.nnz:
        take = P.indices >= P.cols
        ptr = np.r_[0, np.cumsum(np.bincount(P.cols[take], minlength=n))]
        model.hessian_.dim_ = n
        model.hessian_.format_ = h.HessianFormat.kTriangular
        model.hessian_.start_, model.hessian_.index_, model.hessian_.value_ = ptr, P.indices[take], P.data[take]
    hi = h._Highs()
    for key, val in (("output_flag", False), ("threads", 1), ("parallel", "off"),
                     ("primal_feasibility_tolerance", 1e-10), ("dual_feasibility_tolerance", 1e-10),
                     ("ipm_optimality_tolerance", 1e-12), ("small_matrix_value", 1e-12)):
        hi.setOptionValue(key, val)
    if not P.nnz:
        hi.setOptionValue("solver", os.getenv("HIGHS_SOLVER", "ipm" if n >= 300 else "choose"))
        hi.setOptionValue("run_crossover", os.getenv("HIGHS_CROSSOVER", "off"))
    if hi.passModel(model) == h.HighsStatus.kError:
        return None
    hi.run()
    if hi.getModelStatus() != h.HighsModelStatus.kOptimal:
        return None
    sol = hi.getSolution()
    x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
    if P.nnz:
        x, z = polish(P, q, A, b, eq, x, z)
    s = b - A @ x
    s[eq] = 0
    s[~eq] = np.maximum(s[~eq], 0)
    result = {"status": "solved", "x": x, "s": s, "z": z}
    return result if valid(P, q, A, b, eq, result) else None
