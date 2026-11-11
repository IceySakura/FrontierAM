#!/usr/bin/env python3
"""Offline sparse convex conic optimization."""
import argparse
import json
import os
import sys
import time

START = time.monotonic()

import numpy as np
from scipy import sparse


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def load(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata_json"]))
        mats = []
        for k in ("P", "A"):
            M = sparse.csc_matrix(
                (f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]),
                shape=tuple(f[k + "_shape"]),
            )
            M.sum_duplicates()
            M.eliminate_zeros()
            M.sort_indices()
            mats.append(M)
        return mats[0], f["q"], mats[1], f["b"], meta["cones"]


def save(path, result):
    status, x, s, z = result
    arr = {"metadata_json": np.array(json.dumps({"schema_version": 1, "status": status}))}
    for k, v in (("x", x), ("s", s), ("z", z)):
        if v is not None:
            arr[k] = np.asarray(v, dtype=np.float64).reshape(-1)
    with open(path, "wb") as f:
        np.savez(f, **arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = ("unresolved", None, None, None)
    try:
        P, q, A, b, cones = load(args.input)
        from cones import ConeProduct
        from accuracy import assess
        K = ConeProduct(cones)
        from presolve import trivial
        direct = trivial(P, q, A, b, K)
        if direct is not None:
            save(args.output, direct)
            return
        linear = all(c["type"] in ("zero", "nonnegative") for c in cones)
        if linear:
            from presolve import solve_separable
            direct = solve_separable(P, q, A, b, K)
            if direct is not None:
                save(args.output, direct)
                return
            from polyhedral import solve_polyhedral
            result = solve_polyhedral(P, q, A, b, K)
        if not linear or result[0] == "unresolved":
            from interior import solve_conic
            if K.ei.size and P.nnz and "SOLVER_EXP_SCALING" not in os.environ:
                K.exp_mode = "primal"
            reduced = None
            if not linear and "SOLVER_NO_PRESOLVE" not in os.environ:
                from presolve import reduce_equalities
                reduced = reduce_equalities(P, q, A, b, K)
            if reduced is not None:
                data, restore = reduced
                result = restore(solve_conic(*data, start=START))
            else:
                result = solve_conic(P, q, A, b, K, start=START)
            score = assess(P, q, A, b, K, result)[0]
            if K.ei.size and score > 1.:
                K.exp_mode = "primal" if K.exp_mode != "primal" else "bfgs"
                alternative = solve_conic(P, q, A, b, K, start=START)
                if assess(P, q, A, b, K, alternative)[0] < score:
                    result = alternative
            elif reduced is not None and score > 1.:
                alternative = solve_conic(P, q, A, b, K, start=START)
                if assess(P, q, A, b, K, alternative)[0] < score:
                    result = alternative
        if os.environ.get("SOLVER_DEBUG"):
            print("check", assess(P, q, A, b, K, result), file=sys.stderr)
            print("total", time.monotonic() - START, file=sys.stderr)
    except Exception as e:
        if os.environ.get("SOLVER_DEBUG"):
            import traceback
            traceback.print_exc()
        else:
            print(type(e).__name__ + ": " + str(e), file=sys.stderr)
    save(args.output, result)


if __name__ == "__main__":
    main()
