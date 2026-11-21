import json
import math
from functools import lru_cache
import numpy as np
from scipy import sparse, linalg


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        mats = []
        for k in ("P", "A"):
            M = sparse.csc_matrix(
                (data[k + "_data"], data[k + "_indices"], data[k + "_indptr"]),
                shape=tuple(data[k + "_shape"]),
            )
            M.sum_duplicates()
            M.eliminate_zeros()
            M.sort_indices()
            mats.append(M)
        return mats[0], data["q"], mats[1], data["b"], meta["cones"]


def save(path, result):
    arrays = {k: np.asarray(v, dtype=float) for k, v in result.items()
              if k in ("x", "s", "z")}
    arrays["metadata_json"] = np.array(json.dumps(
        {"schema_version": 1, "status": result["status"]}, allow_nan=False))
    with open(path, "wb") as stream:
        np.savez(stream, **arrays)


def blocks(cones):
    start = 0
    for c in cones:
        d = c["dim"]
        end = start + (d * (d + 1) // 2 if c["type"] == "psd" else d)
        yield c["type"], d, start, end
        start = end


@lru_cache(maxsize=8)
def tri_indices(d):
    c, r = np.tril_indices(d)
    return r, c, r != c


def smat(v, d):
    r, c, off = tri_indices(d)
    M = np.zeros((d, d))
    a = v.copy()
    a[off] /= np.sqrt(2.0)
    M[r, c] = a
    M[c, r] = a
    return M


def svec(M):
    r, c, off = tri_indices(M.shape[0])
    a = M[r, c]
    a[off] *= np.sqrt(2.0)
    return a


def exp_def(v):
    a, b, c = v
    boundary = max(abs(b), a, -c, 0.0)
    if b > 0.0 and c > 0.0:
        return min(boundary, max(0.0, a - b * (math.log(c) - math.log(b))))
    return boundary


@lru_cache(maxsize=8)
def cone_groups(layout):
    groups, psd, start = {}, [], 0
    for kind, d in layout:
        width = d * (d + 1) // 2 if kind == "psd" else d
        if kind == "psd":
            psd.append((d, start, start + width))
        else:
            groups.setdefault((kind, d), []).append(np.arange(start, start + width))
        start += width
    return [(kind, np.asarray(ids)) for (kind, d), ids in groups.items()], psd


def cone_def(v, cones, dual=False):
    if len(cones) > 16:
        groups, psd = cone_groups(tuple((c["type"], c["dim"]) for c in cones))
        err = 0.0
        for kind, ids in groups:
            if kind == "zero" and dual:
                continue
            u = v[ids]
            u = u / np.maximum(1., np.max(np.abs(u), axis=1))[:, None]
            if kind == "zero":
                e = norm(u)
            elif kind == "nonnegative":
                e = max(0., -np.min(u))
            elif kind == "soc":
                e = max(0., np.max(np.linalg.norm(u[:, 1:], axis=1) - u[:, 0]))
            else:
                if dual:
                    a, b, c = -u[:, 1], -u[:, 0], math.e * u[:, 2]
                else:
                    a, b, c = u.T
                defects = np.maximum.reduce([np.abs(b), np.maximum(a, 0), np.maximum(-c, 0)])
                interior = (b > 0) & (c > 0)
                defects[interior] = np.minimum(
                    defects[interior],
                    np.maximum(0., a[interior] - b[interior] *
                               (np.log(c[interior]) - np.log(b[interior]))))
                e = np.max(defects)
            err = max(err, e)
        for d, i, j in psd:
            u = v[i:j] / max(1., norm(v[i:j]))
            e = max(0., -linalg.eigh(smat(u, d), subset_by_index=[0, 0],
                                    eigvals_only=True, check_finite=False)[0])
            err = max(err, e)
        return float(err)
    err = 0.0
    for kind, d, i, j in blocks(cones):
        u = v[i:j] / max(1.0, norm(v[i:j]))
        if kind == "zero":
            e = 0.0 if dual else norm(u)
        elif kind == "nonnegative":
            e = max(0.0, -np.min(u))
        elif kind == "soc":
            e = max(0.0, np.linalg.norm(u[1:]) - u[0])
        elif kind == "exponential":
            e = exp_def(np.array([-u[1], -u[0], math.e * u[2]]) if dual else u)
        else:
            e = max(0.0, -linalg.eigh(smat(u, d), subset_by_index=[0, 0],
                                     eigvals_only=True, check_finite=False)[0])
        err = max(err, e)
    return float(err)


def assess(P, q, A, b, cones, result, tol=6e-9):
    status = result["status"]
    for key in ("x", "s", "z"):
        if key in result and not np.isfinite(result[key]).all():
            return False, float("inf")
    if status == "solved":
        x, s, z = result["x"], result["s"], result["z"]
        ax, px, az = A @ x, P @ x, A.T @ z
        rp = norm(ax + s - b) / (1 + max(norm(ax), norm(s), norm(b)))
        rd = norm(px + q + az) / (1 + max(norm(px), norm(q), norm(az)))
        p = 0.5 * (x @ px) + q @ x
        d = -0.5 * (x @ px) - b @ z
        gap = abs(p - d) / max(1.0, abs(p), abs(d))
        err = max(rp, rd, gap)
        if err <= tol * 10:
            err = max(err, cone_def(s, cones), cone_def(z, cones, True))
        return err <= tol, err
    if status == "primal_infeasible":
        z = result["z"]
        if not norm(z) or not norm(b):
            return False, float("inf")
        z = z / norm(z)
        margin = -(b / norm(b)) @ z
        if margin <= 1e-8:
            return False, float("inf")
        error = max(norm(A.T @ z) / (norm(A.data) or 1),
                    cone_def(z, cones, True)) / margin
        return error <= tol, error
    if status == "dual_infeasible":
        x, s = result["x"], result["s"]
        scale = norm(x)
        if not scale or not norm(q):
            return False, float("inf")
        x, s = x / scale, s / scale
        margin = -(q / norm(q)) @ x
        if margin <= 1e-8:
            return False, float("inf")
        a, h = norm(A.data) or 1, norm(P.data) or 1
        error = max(norm(P @ x) / h, norm(A @ x + s) / a,
                    cone_def(s / a, cones)) / margin
        return error <= tol, error
    return False, float("inf")
