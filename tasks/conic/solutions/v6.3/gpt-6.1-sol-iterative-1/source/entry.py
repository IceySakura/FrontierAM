#!/usr/bin/env python3
"""Lightweight input and separable presolve, with the general solver fallback."""
import argparse
import json
import math
import os
import sys
import numpy as np


def norm(a):
    return float(np.max(np.abs(a), initial=0.0))


def cone_defect(v, blocks, dual=False):
    off = 0
    worst = 0.0
    for block in blocks:
        kind, d = block["type"], block["dim"]
        width = d * (d + 1) // 2 if kind == "psd" else d
        u = v[off:off + width]
        u = u / max(1.0, norm(u))
        off += width
        if kind == "zero":
            error = 0.0 if dual else norm(u)
        elif kind == "nonnegative":
            error = max(0.0, -float(np.min(u)))
        elif kind == "soc":
            error = max(0.0, float(np.linalg.norm(u[1:])) - u[0])
        elif kind == "exponential":
            x, y, z = (-u[1], -u[0], math.e * u[2]) if dual else u
            error = max(abs(y), max(x, 0.0), max(-z, 0.0))
            if y > 0 and z > 0:
                error = min(error, max(0.0, x - y * (math.log(z) - math.log(y))))
        else:
            if d > 100:
                return np.inf
            c, r = np.tril_indices(d)
            M = np.zeros((d, d))
            M[r, c] = u / np.where(r == c, 1.0, math.sqrt(2.0))
            M[c, r] = M[r, c]
            error = max(0.0, -float(np.linalg.eigvalsh(M)[0]))
        worst = max(worst, error)
    return worst


def fast_solve(d, blocks):
    q, b = d["q"], d["b"]
    n, m = len(q), len(b)
    pv, pi, pp = (d["P_" + k] for k in ("data", "indices", "indptr"))
    av, ai, ap = (d["A_" + k] for k in ("data", "indices", "indptr"))
    pcounts = np.diff(pp)
    pdiag = (np.all(pcounts <= 1) and np.array_equal(pi, np.flatnonzero(pcounts)))
    if not norm(q) and cone_defect(b, blocks) <= 6e-9:
        return {"status": "solved", "x": np.zeros(n), "s": b, "z": np.zeros(m)}
    if not pdiag:
        return None
    p = np.zeros(n)
    p[np.flatnonzero(pcounts)] = pv
    if np.any(p < 0):
        return None
    poly = all(c["type"] in ("zero", "nonnegative") for c in blocks)
    if not poly:
        return None
    rowcounts = np.bincount(ai, minlength=m)
    if np.any(rowcounts > 1) or np.any(av == 0):
        return None
    eq = np.zeros(m, dtype=bool)
    off = 0
    for c in blocks:
        if c["type"] == "zero":
            eq[off:off + c["dim"]] = True
        off += c["dim"]
    rows, vals = ai, av
    cols = np.repeat(np.arange(n), np.diff(ap))

    def matvec(x):
        return np.bincount(rows, weights=vals * x[cols], minlength=m)

    def transvec(z):
        return np.bincount(cols, weights=vals * z[rows], minlength=n)

    def verify(out):
        if any(not np.isfinite(v).all() for k, v in out.items() if k != "status"):
            return False
        tol = 7e-9
        if out["status"] == "solved":
            x, s, z = out["x"], out["s"], out["z"]
            Ax, Px, Atz = matvec(x), p * x, transvec(z)
            if norm(Ax + s - b) > tol * (1 + max(norm(Ax), norm(s), norm(b))):
                return False
            if norm(Px + q + Atz) > tol * (1 + max(norm(Px), norm(q), norm(Atz))):
                return False
            xp = float(x @ Px)
            primal, dual = .5 * xp + float(q @ x), -.5 * xp - float(b @ z)
            if abs(primal - dual) > tol * max(1.0, abs(primal), abs(dual)):
                return False
            return max(cone_defect(s, blocks), cone_defect(z, blocks, True)) <= tol
        if out["status"] == "primal_infeasible":
            z = out["z"] / norm(out["z"])
            if norm(b) == 0:
                return False
            margin = -float((b / norm(b)) @ z)
            return (margin > 1e-8 and
                    max(norm(transvec(z)) / (norm(av) or 1),
                        cone_defect(z, blocks, True)) <= tol * margin)
        x, s = out["x"], out["s"]
        xs = norm(x)
        if not xs or not norm(q):
            return False
        x, s = x / xs, s / xs
        margin = -float((q / norm(q)) @ x)
        a = norm(av) or 1
        return (margin > 1e-8 and
                max(norm(p * x) / (norm(pv) or 1), norm(matvec(x) + s) / a,
                    cone_defect(s / a, blocks)) <= tol * margin)

    empty = np.flatnonzero((rowcounts == 0) & ((eq & (b != 0)) | (~eq & (b < 0))))
    if empty.size:
        z = np.zeros(m)
        j = empty[0]
        z[j] = -np.sign(b[j]) if eq[j] else 1.0
        out = {"status": "primal_infeasible", "z": z}
        return out if verify(out) else None
    bounds = b[rows] / vals
    low = (vals < 0) | eq[rows]
    high = (vals > 0) | eq[rows]
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(lb, cols[low], bounds[low])
    np.minimum.at(ub, cols[high], bounds[high])
    lr, ur = np.full(n, m, dtype=np.int64), np.full(n, m, dtype=np.int64)
    low &= bounds == lb[cols]
    high &= bounds == ub[cols]
    np.minimum.at(lr, cols[low], rows[low])
    np.minimum.at(ur, cols[high], rows[high])
    coefficient = np.ones(m)
    coefficient[rows] = vals
    inconsistent = np.flatnonzero(lb > ub)
    if inconsistent.size:
        j = inconsistent[0]
        z = np.zeros(m)
        z[lr[j]], z[ur[j]] = -1 / coefficient[lr[j]], 1 / coefficient[ur[j]]
        out = {"status": "primal_infeasible", "z": z}
        return out if verify(out) else None
    free = (p == 0) & (((q > 0) & ~np.isfinite(lb)) | ((q < 0) & ~np.isfinite(ub)))
    if np.any(free):
        x = np.zeros(n)
        x[free] = -np.sign(q[free])
        out = {"status": "dual_infeasible", "x": x, "s": -matvec(x)}
        return out if verify(out) else None
    x = np.zeros(n)
    nz = p > 0
    x[nz] = -q[nz] / p[nz]
    low = (p == 0) & (q > 0)
    high = (p == 0) & (q < 0)
    x[low], x[high] = lb[low], ub[high]
    np.clip(x, lb, ub, out=x)
    grad = p * x + q
    z = np.zeros(m)
    low, high = (grad > 0) & (lr < m), (grad < 0) & (ur < m)
    z[lr[low]] = -grad[low] / coefficient[lr[low]]
    z[ur[high]] = -grad[high] / coefficient[ur[high]]
    s = np.maximum(0, b - matvec(x))
    s[eq] = 0
    out = {"status": "solved", "x": x, "s": s, "z": z}
    return out if verify(out) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = None
    try:
        with np.load(args.input, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        meta = json.loads(str(data["metadata_json"]))
        blocks = meta["cones"]
        result = fast_solve(data, blocks)
        if result is None:
            import solver
            csc_matrix = solver.sp.csc_matrix
            matrices = []
            for k in ("P", "A"):
                M = csc_matrix((data[k + "_data"], data[k + "_indices"],
                                data[k + "_indptr"]), shape=tuple(data[k + "_shape"]))
                M.sum_duplicates()
                M.eliminate_zeros()
                M.sort_indices()
                matrices.append(M)
            result = solver.solve_model(matrices[0], data["q"], matrices[1], data["b"], blocks)
    except Exception:
        if os.environ.get("SOLVER_VERBOSE"):
            import traceback
            traceback.print_exc()
    if result is None:
        result = {"status": "unresolved"}
    arrays = {k: v for k, v in result.items() if k in ("x", "s", "z")}
    arrays["metadata_json"] = np.array(json.dumps(
        {"schema_version": 1, "status": result["status"]}, allow_nan=False))
    with open(args.output, "wb") as f:
        np.savez(f, **arrays)


if __name__ == "__main__":
    main()
