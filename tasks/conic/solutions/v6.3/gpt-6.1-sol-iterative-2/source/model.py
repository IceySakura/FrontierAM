import json

import numpy as np


class Problem:
    def __init__(self, P, q, A, b, cones):
        self.P, self.q, self.A, self.b = P, q, A, b
        self.cones = cones
        self.n, self.m = q.size, b.size
        self.cone_plan = None
        self.certificate_A = None
        self.certificate_P = None
        self.linear_slices = []
        start = 0
        for cone in cones:
            kind, dim = cone["type"], cone["dim"]
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            if kind in ("zero", "nonnegative"):
                self.linear_slices.append((kind, slice(start, start + width)))
            start += width


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["metadata"] = json.loads(str(arrays["metadata_json"]))
    return arrays


def from_arrays(data):
    from scipy import sparse
    matrices = []
    for prefix in ("P", "A"):
        matrix = sparse.csc_matrix(
            (data[prefix + "_data"], data[prefix + "_indices"],
             data[prefix + "_indptr"]),
            shape=tuple(data[prefix + "_shape"]),
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        matrices.append(matrix)
    return Problem(matrices[0], data["q"], matrices[1], data["b"],
                   data["metadata"]["cones"])


def load(path):
    return from_arrays(load_arrays(path))


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def repair_slack(problem, x, value=None):
    s = problem.b - problem.A @ x if value is None else np.asarray(value, dtype=float).copy()
    for kind, block in problem.linear_slices:
        if kind == "zero":
            s[block] = 0.0
        else:
            np.maximum(s[block], 0.0, out=s[block])
    return s


def svec(matrix, rows, cols):
    result = matrix[rows, cols].copy()
    result[rows != cols] *= np.sqrt(2.0)
    return result


def smat(vector, dim, rows=None, cols=None):
    if rows is None:
        cols, rows = np.tril_indices(dim)
    values = vector.copy()
    values[rows != cols] *= 1.0 / np.sqrt(2.0)
    result = np.zeros((dim, dim))
    result[rows, cols] = values
    result[cols, rows] = values
    return result


class ConePlan:
    def __init__(self, cones):
        zero, linear, lengths, exponential, psd2 = [], [], [], [], []
        soc, psd = {}, []
        start = 0
        for cone in cones:
            kind, dim = cone["type"], cone["dim"]
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            ids = np.arange(start, start + width)
            if kind == "zero":
                zero.append(ids)
            elif kind == "nonnegative" or (kind == "psd" and dim == 1):
                linear.append(ids)
                lengths.append(width)
            elif kind == "soc":
                soc.setdefault(dim, []).append(ids)
            elif kind == "exponential":
                exponential.append(ids)
            elif dim == 2:
                psd2.append(ids)
            else:
                diagonal = np.arange(dim)
                diagonal = diagonal * (diagonal + 3) // 2
                psd.append(dict(block=slice(start, start + width), dim=dim,
                                diagonal=diagonal, coordinates=None))
            start += width
        self.zero = np.concatenate(zero) if zero else np.empty(0, dtype=np.int64)
        self.linear = np.concatenate(linear) if linear else np.empty(0, dtype=np.int64)
        self.linear_starts = np.r_[0, np.cumsum(lengths)][:-1].astype(np.int64)
        self.soc = {d: np.asarray(ids) for d, ids in soc.items()}
        self.exponential = np.asarray(exponential, dtype=np.int64).reshape((-1, 3))
        self.psd2 = np.asarray(psd2, dtype=np.int64).reshape((-1, 3))
        self.psd = psd

    def error(self, vector, dual=False):
        worst = 0.0
        if not dual and self.zero.size:
            value = norm(vector[self.zero])
            worst = value / max(1.0, value)
        if self.linear.size:
            v = vector[self.linear]
            if np.min(v) < 0.0:
                scale = np.maximum(1.0, np.maximum.reduceat(np.abs(v), self.linear_starts))
                negative = np.maximum(0.0, -np.minimum.reduceat(v, self.linear_starts))
                worst = max(worst, float(np.max(negative / scale)))
        for ids in self.soc.values():
            v = vector[ids]
            u = v / np.maximum(1.0, np.max(np.abs(v), axis=1))[:, None]
            value = np.linalg.norm(u[:, 1:], axis=1) - u[:, 0]
            worst = max(worst, float(np.max(value, initial=0.0)))
        if self.exponential.size:
            v = vector[self.exponential]
            u = v / np.maximum(1.0, np.max(np.abs(v), axis=1))[:, None]
            a, b, c = (-u[:, 1], -u[:, 0], np.e * u[:, 2]) if dual else u.T
            value = np.maximum.reduce((np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)))
            mask = (b > 0) & (c > 0)
            value[mask] = np.minimum(
                value[mask], np.maximum(0, a[mask] - b[mask] * (np.log(c[mask]) - np.log(b[mask]))))
            worst = max(worst, float(np.max(value, initial=0.0)))
        if self.psd2.size:
            v = vector[self.psd2]
            u = v / np.maximum(1.0, np.max(np.abs(v), axis=1))[:, None]
            a, b, c = u.T
            value = 0.5 * (np.sqrt((a - c) ** 2 + 2 * b ** 2) - a - c)
            worst = max(worst, float(np.max(value, initial=0.0)))
        for block in self.psd:
            v = vector[block["block"]]
            magnitude = norm(v)
            if not magnitude:
                continue
            u = v / max(1.0, magnitude)
            diagonal = u[block["diagonal"]]
            if np.count_nonzero(u) == np.count_nonzero(diagonal):
                value = -float(np.min(diagonal))
            else:
                from scipy.linalg import eigh
                if block["coordinates"] is None:
                    cols, rows = np.tril_indices(block["dim"])
                    block["coordinates"] = rows, cols
                rows, cols = block["coordinates"]
                value = -float(eigh(smat(u, block["dim"], rows, cols),
                                    subset_by_index=[0, 0], eigvals_only=True,
                                    check_finite=False)[0])
            worst = max(worst, value)
        return max(0.0, float(worst)) if np.isfinite(worst) else np.inf


def cone_error(vector, cones, dual=False, plan=None):
    return (plan or ConePlan(cones)).error(vector, dual)


def problem_cone_error(problem, vector, dual=False):
    if problem.cone_plan is None:
        problem.cone_plan = ConePlan(problem.cones)
    return problem.cone_plan.error(vector, dual)


def assess(problem, result, tol=8e-9, full=True):
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    status = result["status"]
    if status == "unresolved":
        return False, {}
    for key in ("x", "s", "z"):
        if key in result and not np.isfinite(result[key]).all():
            return False, {}
    if status == "solved":
        x, s, z = result["x"], result["s"], result["z"]
        Ax, Px, Az = A @ x, P @ x, A.T @ z
        rp = norm(Ax + s - b) / (1 + max(norm(Ax), norm(s), norm(b)))
        rd = norm(Px + q + Az) / (1 + max(norm(Px), norm(q), norm(Az)))
        xPx, qx, bz = float(x @ Px), float(q @ x), float(b @ z)
        primal, dual = 0.5 * xPx + qx, -0.5 * xPx - bz
        gap = abs(primal - dual)
        relgap = gap / max(1e-300, abs(primal), abs(dual))
        report = dict(primal=rp, dual=rd, gap=gap, relative_gap=relgap,
                      primal_objective=primal, dual_objective=dual)
        cheap_ok = max(rp, rd) <= tol and min(gap, relgap) <= tol
        if not full and not cheap_ok:
            return False, report
        cs = problem_cone_error(problem, s)
        cz = problem_cone_error(problem, z, dual=True)
        report.update(primal_cone=cs, dual_cone=cz)
        return bool(cheap_ok and max(cs, cz) <= tol), report
    a, h = norm(A.data) or 1.0, norm(P.data) or 1.0
    if status == "primal_infeasible":
        z = result["z"]
        zn, bn = norm(z), norm(b)
        if zn == 0 or bn == 0:
            return False, {}
        z = z / zn
        margin = -float((b / bn) @ z)
        if margin <= 1e-8:
            return False, dict(margin=margin)
        linear = norm(A.T @ z) / a / margin
        if linear <= tol or not np.isfinite(linear):
            if problem.certificate_A is None:
                problem.certificate_A = A.copy()
                problem.certificate_A.data /= a
            linear = norm(problem.certificate_A.T @ z) / margin
        cone = problem_cone_error(problem, z, dual=True) / margin
        return bool(max(linear, cone) <= tol), dict(
            margin=margin, linear=linear, cone=cone)
    x, s = result["x"], result["s"]
    xn, qn = norm(x), norm(q)
    if xn == 0 or qn == 0:
        return False, {}
    x, s = x / xn, s / xn
    margin = -float((q / qn) @ x)
    if margin <= 1e-8:
        return False, dict(margin=margin)
    quadratic = norm(P @ x) / h / margin
    if quadratic <= tol or not np.isfinite(quadratic):
        if problem.certificate_P is None:
            problem.certificate_P = P.copy()
            problem.certificate_P.data /= h
        quadratic = norm(problem.certificate_P @ x) / margin
    if problem.certificate_A is None:
        problem.certificate_A = A.copy()
        problem.certificate_A.data /= a
    linear = norm(problem.certificate_A @ x + s / a) / margin
    cone = problem_cone_error(problem, s / a) / margin
    return bool(max(linear, quadratic, cone) <= tol), dict(
        margin=margin, linear=linear, quadratic=quadratic, cone=cone)


def equilibrate(problem, passes=6):
    P, A = problem.P.copy(), problem.A.copy()
    q, b = problem.q.copy(), problem.b.copy()
    D, E = np.ones(problem.n), np.ones(problem.m)
    acol = np.repeat(np.arange(problem.n), np.diff(A.indptr))
    pcol = np.repeat(np.arange(problem.n), np.diff(P.indptr))
    for unused in range(passes):
        cn = np.zeros(problem.n)
        np.maximum.at(cn, acol, np.abs(A.data))
        np.maximum.at(cn, pcol, np.abs(P.data))
        d = 1.0 / np.sqrt(np.maximum(cn, 1e-8))
        d[cn == 0] = 1.0
        d = np.clip(d, 1e-3, 1e3)
        A.data *= d[acol]
        P.data *= d[P.indices] * d[pcol]
        q *= d
        D *= d
        rn = np.zeros(problem.m)
        np.maximum.at(rn, A.indices, np.abs(A.data))
        start = 0
        for cone in problem.cones:
            kind, dim = cone["type"], cone["dim"]
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            if kind not in ("zero", "nonnegative"):
                rn[start:start + width] = rn[start:start + width].max(initial=0)
            start += width
        e = 1.0 / np.sqrt(np.maximum(rn, 1e-8))
        e[rn == 0] = 1.0
        e = np.clip(e, 1e-3, 1e3)
        A.data *= e[A.indices]
        b *= e
        E *= e
        if norm(d - 1.0) < 0.03 and norm(e - 1.0) < 0.03:
            break
    cost = max(norm(q), norm(P.data))
    c = 1.0 / cost if cost else 1.0
    c = float(np.clip(c, 1e-8, 1e8))
    P.data *= c
    q *= c
    return Problem(P, q, A, b, problem.cones), D, E, c
