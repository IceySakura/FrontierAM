#!/usr/bin/env python3
"""Offline, single-threaded convex conic solver."""

import argparse
import os
import sys
import time

import numpy as np

from problem import load, save


def optimize(original, deadline):
    from basic import solve as immediate
    from quality import acceptable
    result = immediate(original)
    if result is not None:
        return result
    kinds = {c["type"] for c in original.cones}
    p, elimination = original, None
    transforms = []
    if "exponential" in kinds and "psd" not in kinds:
        from presolve import eliminate
        p, elimination = eliminate(p)
    if kinds <= {"zero", "nonnegative"}:
        from presolve import fixed_variables
        p, elimination = fixed_variables(p)
        from linear import solve
        result = immediate(p) if elimination is not None else None
        if result is None:
            result = solve(p, deadline)
        if result["status"] == "unresolved" and elimination is not None and time.monotonic() < deadline:
            result = solve(original, deadline)
            elimination = None
    elif kinds <= {"zero", "nonnegative", "psd"}:
        from sdp import solve
        result = solve(p, deadline)
    elif kinds <= {"zero", "nonnegative", "soc"}:
        from presolve import fixed_variables, simplify_cones
        for unused in range(6):
            p, transform = fixed_variables(p)
            changed = transform is not None
            if transform is not None:
                transforms.append(transform)
            p, transform = simplify_cones(p)
            changed |= transform is not None
            if transform is not None:
                transforms.append(transform)
            if not changed:
                break
        from cone_ipm import solve as ipm
        from soc_direct import solve as direct
        budget = min(90.0, 0.65 * max(0.0, deadline - time.monotonic()))
        result = immediate(p)
        if result is None:
            if p.n > 5000:
                from conic import solve as oa
                result = oa(p, deadline, try_interior=True)
            else:
                try:
                    result = ipm(p, time.monotonic() + budget, return_best=True)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    result = {"status": "unresolved"}
        if not acceptable(p, result) and p.n <= 5000:
            from conic import solve as oa
            warm = result if result["status"] == "solved" else None
            result = oa(p, deadline, warm=warm)
    elif kinds <= {"zero", "nonnegative", "exponential"}:
        from gp import solve as gp_solve
        result = gp_solve(p, min(deadline, time.monotonic() + 120))
        if result is None:
            from conic import solve
            result = solve(p, deadline)
    else:
        from conic import solve
        result = solve(p, deadline)
    if elimination is not None:
        result = elimination.recover(result)
    for transform in reversed(transforms):
        result = transform.recover(result)
    if acceptable(original, result, tol=1e-8):
        return result
    return {"status": "unresolved"}


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"status": "unresolved"}
    try:
        p = load(args.input)
        deadline = start + float(os.environ.get("CONIC_TIME_LIMIT", "285"))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = optimize(p, deadline)
    except Exception as error:
        if os.environ.get("CONIC_DEBUG"):
            import traceback
            traceback.print_exc()
        else:
            print(type(error).__name__ + ": " + str(error)[:300], file=sys.stderr)
    save(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
