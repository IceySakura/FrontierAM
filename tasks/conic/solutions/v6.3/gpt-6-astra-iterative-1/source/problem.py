import json
import numpy as np
from scipy import sparse


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


class Problem:
    def __init__(self, P, q, A, b, cones):
        self.P, self.q, self.A, self.b, self.cones = P, q, A, b, cones
        self.n, self.m = len(q), len(b)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            matrices = []
            for key in ("P", "A"):
                mat = sparse.csc_matrix(
                    (data[key + "_data"], data[key + "_indices"],
                     data[key + "_indptr"]), shape=tuple(data[key + "_shape"]))
                mat.sum_duplicates()
                mat.eliminate_zeros()
                mat.sort_indices()
                matrices.append(mat)
            return cls(matrices[0], data["q"], matrices[1], data["b"],
                       meta["cones"])


def write_result(path, result):
    metadata = {"schema_version": 1, "status": result["status"]}
    arrays = {k: np.asarray(result[k], dtype=np.float64) for k in ("x", "s", "z")
              if k in result}
    arrays["metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
    with open(path, "wb") as stream:
        np.savez(stream, **arrays)


def exp_defect(v):
    a, b, c = v.T
    boundary = np.maximum.reduce((np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)))
    good = (b > 0) & (c > 0)
    inside = np.full_like(a, np.inf)
    inside[good] = np.maximum(
        a[good] - b[good] * (np.log(c[good]) - np.log(b[good])), 0)
    return float(np.max(np.minimum(boundary, inside), initial=0))


def cone_defect(v, cones, dual=False):
    start, worst = 0, 0.0
    exp = []
    for cone in cones:
        kind, d = cone["type"], cone["dim"]
        width = d * (d + 1) // 2 if kind == "psd" else d
        u = v[start:start + width]
        start += width
        u = u / max(1.0, norm(u))
        if kind == "zero":
            error = 0 if dual else norm(u)
        elif kind == "nonnegative":
            error = max(0.0, -float(np.min(u)))
        elif kind == "soc":
            error = max(0.0, float(np.linalg.norm(u[1:])) - u[0])
        elif kind == "exponential":
            exp.append((-u[1], -u[0], np.e * u[2]) if dual else u)
            error = 0
        else:
            from scipy.linalg import eigvalsh
            c, r = np.tril_indices(d)
            mat = np.zeros((d, d))
            vals = u.copy()
            vals[r != c] /= np.sqrt(2.0)
            mat[r, c] = vals
            mat[c, r] = vals
            error = max(0.0, -eigvalsh(mat, subset_by_index=[0, 0],
                                     check_finite=False)[0])
        worst = max(worst, error)
    if exp:
        worst = max(worst, exp_defect(np.asarray(exp)))
    return worst


def assess(problem, result, tolerance=8e-9, cones=True):
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    status = result["status"]
    errors = []
    try:
        if status == "solved":
            x, s, z = result["x"], result["s"], result["z"]
            if not all(np.isfinite(v).all() for v in (x, s, z)):
                return False, [np.inf]
            Ax, Px, Atz = A @ x, P @ x, A.T @ z
            errors.append(norm(Ax + s - b) / (1 + max(norm(Ax), norm(s), norm(b))))
            errors.append(norm(Px + q + Atz) / (1 + max(norm(Px), norm(q), norm(Atz))))
            p = .5 * x.dot(Px) + q.dot(x)
            d = -.5 * x.dot(Px) - b.dot(z)
            errors.append(abs(p - d) / max(1, abs(p), abs(d)))
            if cones and (not np.isfinite(errors).all() or max(errors) > tolerance):
                return False, errors
            if cones:
                errors += [cone_defect(s, problem.cones),
                           cone_defect(z, problem.cones, True)]
        elif status == "primal_infeasible":
            if norm(result["z"]) == 0 or norm(b) == 0:
                return False, [np.inf]
            z = result["z"] / norm(result["z"])
            beta = -(b / norm(b)).dot(z)
            if beta <= 1e-8:
                return False, [np.inf]
            errors = [norm(A.T @ z) / (norm(A.data) or 1.0) / beta]
            if not np.isfinite(errors).all() or errors[0] > tolerance:
                return False, errors
            errors.append(cone_defect(z, problem.cones, True) / beta)
        elif status == "dual_infeasible":
            scale = norm(result["x"])
            if scale == 0 or norm(q) == 0:
                return False, [np.inf]
            x, s = result["x"] / scale, result["s"] / scale
            alpha = -(q / norm(q)).dot(x)
            if alpha <= 1e-8:
                return False, [np.inf]
            a = norm(A.data) or 1.0
            errors = [norm(P @ x) / (norm(P.data) or 1.0) / alpha,
                      norm(A @ x + s) / a / alpha]
            if not np.isfinite(errors).all() or max(errors) > tolerance:
                return False, errors
            errors.append(cone_defect(s / a, problem.cones) / alpha)
        else:
            return False, [np.inf]
        return bool(np.isfinite(errors).all() and max(errors, default=0) <= tolerance), errors
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return False, [np.inf]


def equilibrate(problem, iterations=5):
    P, A = problem.P.copy(), problem.A.copy()
    q, b = problem.q.copy(), problem.b.copy()
    D, E = np.ones(problem.n), np.ones(problem.m)
    grouped = {}
    start = 0
    for cone in problem.cones:
        kind, d = cone["type"], cone["dim"]
        width = d * (d + 1) // 2 if kind == "psd" else d
        if kind not in ("zero", "nonnegative"):
            grouped.setdefault(width, []).append(start)
        start += width
    groups = []
    for width, starts in grouped.items():
        if len(starts) == 1:
            groups.append(slice(starts[0], starts[0]+width))
        else:
            groups.append(np.asarray(starts, dtype=np.int32)[:,None]
                          + np.arange(width, dtype=np.int32))
    for _ in range(iterations if problem.n and (problem.m or P.nnz) else 0):
        cm = (np.asarray(abs(A).max(axis=0).toarray()).ravel()
              if problem.m else np.zeros(problem.n))
        if P.nnz:
            cm = np.maximum(cm, np.asarray(abs(P).max(axis=0).toarray()).ravel())
        rm = (np.asarray(abs(A).max(axis=1).toarray()).ravel()
              if problem.m else np.empty(0))
        for ids in groups:
            if isinstance(ids, slice):
                rm[ids] = np.max(rm[ids], initial=0)
            else:
                rm[ids] = np.max(rm[ids], axis=1)[:,None]
        dc = 1 / np.sqrt(np.clip(cm, 1e-4, 1e4))
        er = 1 / np.sqrt(np.clip(rm, 1e-4, 1e4))
        dc[cm == 0] = 1
        er[rm == 0] = 1
        D *= dc
        E *= er
        A.data *= er[A.indices] * np.repeat(dc, np.diff(A.indptr))
        P.data *= dc[P.indices] * np.repeat(dc, np.diff(P.indptr))
    q *= D
    b *= E
    cost = 1 / max(1e-4, norm(q), norm(P.data))
    cost = min(cost, 1e4)
    q *= cost
    P.data *= cost
    return Problem(P, q, A, b, problem.cones), D, E, cost


def constant_certificate(problem):
    """Separate an infeasible constant cone block from its cone."""
    from scipy.linalg import eigh
    start = 0
    for cone in problem.cones:
        kind, d = cone["type"], cone["dim"]
        width = d*(d+1)//2 if kind == "psd" else d
        block = problem.b[start:start+width]
        u = block/max(1., norm(block))
        z = np.zeros(width)
        if kind == "zero":
            z = -u
        elif kind == "nonnegative":
            i = int(np.argmin(u))
            if u[i] < 0:
                z[i] = 1.
        elif kind == "soc":
            length = np.linalg.norm(u[1:])
            if u[0] < length:
                z[0] = length if length else 1.
                z[1:] = -u[1:]
        elif kind == "psd":
            from cones import PSD
            psd = PSD(0, d)
            eig, vectors = eigh(psd.mat(u), subset_by_index=[0, 0], check_finite=False)
            if eig[0] < 0:
                z = psd.vec(np.outer(vectors[:, 0], vectors[:, 0]))
        else:
            a, v, w = u
            if v < 0:
                z[1] = 1.
            elif w < 0:
                z[2] = 1.
            elif v > 0 and w > 0:
                lr = np.log(w)-np.log(v)
                if a > v*lr:
                    z = np.array([-w, w*(lr-1), v])
            elif v == 0 and a > 0:
                t = min(1., a/(2*max(w, 1e-300)))
                z = np.array([-1., -np.log(t)-1., t])
            elif v > 0 and w == 0:
                t = np.clip(a/v-1., -700., 700.)
                z = np.array([-1., t, np.exp(-t-1.)])
        if norm(z):
            full = np.zeros(problem.m)
            full[start:start+width] = z
            result = {"status": "primal_infeasible", "z": full}
            if assess(problem, result)[0]:
                return result
        start += width
    return {"status": "unresolved"}
