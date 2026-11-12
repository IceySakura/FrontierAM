#!/usr/bin/env python3
"""Portable NPZ entry point for high-accuracy sparse conic optimization."""

import json
import os
import sys

import numpy as np


def read(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        result = {key: data[key] for key in data.files if key != "metadata_json"}
        result["cones"] = meta["cones"]
        return result


def load(source):
    # Array-API wildcard imports otherwise load NumPy's compiler/test helpers.
    exports = np.__all__
    try:
        np.__all__ = [name for name in exports if name not in ("f2py", "testing")]
        from scipy import sparse
    finally:
        np.__all__ = exports
    raw = source if isinstance(source, dict) else read(source)
    mats = []
    for prefix in ("P", "A"):
        M = sparse.csc_matrix((raw[prefix + "_data"], raw[prefix + "_indices"],
                               raw[prefix + "_indptr"]), shape=tuple(raw[prefix + "_shape"]),
                              dtype=np.float64)
        M.sum_duplicates()
        M.eliminate_zeros()
        M.sort_indices()
        mats.append(M)
    return mats[0], raw["q"], mats[1], raw["b"], raw["cones"]


def solve_problem(P, q, A, b, cones, allow_reduce=True, allow_affine=True):
    from simple import free_direction, solve as simple_solve
    from validation import valid
    if not np.any(q):
        zero = {"status": "solved", "x": np.zeros(len(q)), "s": b.copy(), "z": np.zeros(len(b))}
        if valid(P, q, A, b, cones, zero):
            return zero
    result = free_direction(P, q, A, b)
    if result is None:
        result = simple_solve(P, q, A, b, cones)
    if result is not None and valid(P, q, A, b, cones, result):
        return result
    if allow_affine:
        from affine import reduce as affine_reduce
        reduction = affine_reduce(P, q, A, b, cones)
        if reduction is not None:
            rr = solve_problem(reduction.P, reduction.q, reduction.A, reduction.b,
                               reduction.cones, allow_reduce, False)
            result = reduction.recover(rr)
            if valid(P, q, A, b, cones, result):
                return result
    if allow_reduce:
        from reductions import reduce
        reduction = reduce(A, b, cones)
        if isinstance(reduction, dict):
            if valid(P, q, A, b, cones, reduction):
                return reduction
        elif reduction is not None:
            T = reduction.T
            rr = solve_problem(P, q, (T @ A).tocsc(), T @ b, reduction.cones, False, False)
            result = reduction.recover(rr)
            if valid(P, q, A, b, cones, result):
                return result
    result = {"status": "unresolved"}
    polyhedral = all(c["type"] in ("zero", "nonnegative") for c in cones)
    prefer_ipm = bool(P.nnz and len(q) >= 200)
    if polyhedral and not prefer_ipm and not os.getenv("FORCE_IPM"):
        from polyhedral import solve
        result = solve(P, q, A, b, cones)
        if valid(P, q, A, b, cones, result):
            return result
    from cones import ConeSystem
    from ipm import solve
    cs = ConeSystem(cones)
    prepared = None
    if polyhedral and prefer_ipm and os.getenv("QP_METHOD", "admm") == "admm":
        from admm import solve as admm_solve
        try:
            result, prepared = admm_solve(P, q, A, b, cs, float(os.getenv("ADMM_BUDGET", ".12")))
            if result is not None:
                return result
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            prepared = None
    result = solve(P, q, A, b, cs, prepared=prepared)
    if valid(P, q, A, b, cones, result, cs):
        return result
    if polyhedral and prepared is not None:
        from admm import polish_result
        polished = polish_result(P, q, A, b, cs, result, prepared,
                                 prepared.get("polish_ordering", prepared.get("ordering")))
        if polished is not None:
            return polished
        if prepared.get("initial") is not None:
            result = solve(P, q, A, b, cs, settings={"use_initial": False}, prepared=prepared)
            if valid(P, q, A, b, cones, result, cs):
                return result
    if polyhedral and result.get("_normal"):
        cs = ConeSystem(cones)
        result = solve(P, q, A, b, cs, settings={"form": "augmented"})
        if valid(P, q, A, b, cones, result, cs):
            return result
    if not polyhedral:
        cs = ConeSystem(cones)
        result = solve(P, q, A, b, cs, settings={"order": "COLAMD", "pivot": 1., "final_sigma": .5,
                                                "initialize": False})
        if valid(P, q, A, b, cones, result, cs):
            return result
    if polyhedral and prefer_ipm and not os.getenv("FORCE_IPM"):
        from polyhedral import solve as highs_solve
        result = highs_solve(P, q, A, b, cones)
        if valid(P, q, A, b, cones, result, cs):
            return result
    return {"status": "unresolved"}


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    input_path, output_path = args["--input"], args["--output"]
    result = {"status": "unresolved"}
    try:
        raw = read(input_path)
        from fastpoly import try_solve, trivial
        try:
            result = trivial(raw)
            if result is None:
                result = try_solve(raw)
        except Exception:
            result = None
        if result is None:
            P, q, A, b, cones = load(raw)
            result = solve_problem(P, q, A, b, cones)
    except Exception as error:
        if os.getenv("SOLVER_VERBOSE"):
            import traceback
            traceback.print_exc(file=sys.stderr)
        result = {"status": "unresolved"}
    meta = {"schema_version": 1, "status": result["status"]}
    arrays = {k: np.asarray(result[k], dtype=np.float64) for k in ("x", "s", "z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(meta, allow_nan=False))
    with open(output_path, "wb") as stream:
        np.savez(stream, **arrays)


if __name__ == "__main__":
    main()
