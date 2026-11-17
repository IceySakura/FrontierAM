#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

from model import assess, from_arrays


def solve_model(problem):
    from presolve import equalities, small_cones, unconstrained
    result = unconstrained(problem)
    if result is None:
        result = equalities(problem)
    if result is None:
        working, transform = small_cones(problem)
        polyhedral = all(c["type"] in ("zero", "nonnegative") for c in working.cones)
        if polyhedral and (not working.P.nnz or working.n <= 50):
            from polyhedral import solve
            result = solve(working)
        if result is None:
            from interior import solve
            result = solve(working)
        if result is not None and transform is not None:
            for name in ("s", "z"):
                if name in result:
                    result[name] = transform.T @ result[name]
    return result if result is not None and assess(problem, result, tol=1e-8)[0] else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = None
    problem = None
    try:
        with np.load(args.input, allow_pickle=False) as archive:
            data = {name: archive[name] for name in ("q", "b", "metadata_json")}
            data["metadata"] = json.loads(str(data["metadata_json"]))
            from presolve import zero_objective
            result = zero_objective(data)
            if result is None:
                for name in archive.files:
                    if name not in data:
                        data[name] = archive[name]
        if result is None:
            from separable import solve as separable_solve
            result = separable_solve(data)
        if result is None:
            from bootstrap import prepare
            prepare()
            problem = from_arrays(data)
            del data
            reduction = None
            if os.environ.get("SOLVER_REDUCE", "0") != "0":
                from presolve import AffineReduction
                try:
                    reduction = AffineReduction(problem)
                except (MemoryError, ValueError, RuntimeError, OverflowError):
                    reduction = None
            if reduction is not None and reduction.problem is not None:
                if os.environ.get("SOLVER_LOG"):
                    print("reduction", problem.n, problem.m, "to",
                          reduction.problem.n, reduction.problem.m, flush=True)
                reduced = solve_model(reduction.problem)
                if reduced is not None:
                    recovered = reduction.restore(reduced)
                    if assess(problem, recovered, tol=1e-8)[0]:
                        result = recovered
            if result is None:
                result = solve_model(problem)
        if result is not None and problem is not None and not assess(problem, result, tol=1e-8)[0]:
            result = None
    except Exception:
        if os.environ.get("SOLVER_LOG"):
            import traceback
            traceback.print_exc()
    if result is None:
        result = {"status": "unresolved"}
    metadata = json.dumps({"schema_version": 1, "status": result["status"]},
                          allow_nan=False)
    arrays = {k: np.asarray(result[k], dtype=np.float64) for k in ("x", "s", "z")
              if k in result}
    arrays["metadata_json"] = np.array(metadata)
    with open(args.output, "wb") as stream:
        np.savez(stream, **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
