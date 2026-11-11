#!/usr/bin/env python3
"""Offline conic solver entry point."""

import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
from common import load, assess, norm, smat, svec


def solve(p):
    if p.n == 0:
        x, z = np.zeros(0), np.zeros(p.m)
        if assess(p, x, p.b, z)[0]:
            return {"status": "solved", "x": x, "s": p.b, "z": z}
        for kind, d, i, j in p.blocks:
            zz = np.zeros(p.m)
            b = p.b[i:j]
            if kind == "zero":
                zz[i:j] = -b
            elif kind == "nonnegative":
                zz[i + int(np.argmin(b))] = 1
            elif kind == "soc":
                zz[i] = 1
                r = np.linalg.norm(b[1:])
                if r:
                    zz[i + 1:j] = -b[1:] / r
            elif kind == "psd":
                from scipy import linalg
                ev, V = linalg.eigh(smat(b, d), subset_by_index=(0, 0), check_finite=False)
                zz[i:j] = svec(np.outer(V[:, 0], V[:, 0]))
            else:
                import math
                a, v, w = b
                if v < 0:
                    zz[i + 1] = 1
                elif w < 0:
                    zz[i + 2] = 1
                else:
                    r = math.log(w / v) if v > 0 and w > 0 else (
                        max(0, math.log(w / a) + 2) if a > 0 and w > 0 else 0)
                    r = float(np.clip(r, -700, 700))
                    zz[i:j] = [-1, r - 1, math.exp(-r)]
            if assess(p, z=zz, status="primal_infeasible")[0]:
                return {"status": "primal_infeasible", "z": zz}
        return {"status": "unresolved"}
    if p.m == 0:
        from scipy.sparse import linalg as sla
        if not p.P.nnz:
            if norm(p.q):
                return {"status": "dual_infeasible", "x": -p.q, "s": np.zeros(0)}
            return {"status": "solved", "x": np.zeros(p.n), "s": np.zeros(0), "z": np.zeros(0)}
        x = sla.lsmr(p.P, -p.q, atol=1e-13, btol=1e-13, conlim=1e14, maxiter=5000)[0]
        if assess(p, x, np.zeros(0), np.zeros(0))[0]:
            return {"status": "solved", "x": x, "s": np.zeros(0), "z": np.zeros(0)}
        direction = -(p.P @ x + p.q)
        if assess(p, x=direction, s=np.zeros(0), status="dual_infeasible")[0]:
            return {"status": "dual_infeasible", "x": direction, "s": np.zeros(0)}
        return {"status": "unresolved"}
    if all(k in ("zero", "nonnegative") for k, _, _, _ in p.blocks):
        from poly import solve_poly
        if p.n > 200000 or p.m > 500000:
            from decompose import solve_decomposed
            return solve_decomposed(p)
        return solve_poly(p)
    if not p.P.nnz and all(k in ("zero", "nonnegative", "soc") for k, _, _, _ in p.blocks):
        from ipm import solve_conic
        started = time.monotonic()
        r = solve_conic(p, time_limit=255)
        if r["status"] != "unresolved" and assess(
                p, status=r["status"], **{k: r[k] for k in ("x", "s", "z") if k in r})[0]:
            return r
        from outer import solve_soc
        remaining = 280 - (time.monotonic() - started)
        return solve_soc(p, budget=remaining) if remaining > 2 else r
    from ipm import solve_conic
    return solve_conic(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        p = load(args.input)
        result = solve(p)
        if result["status"] != "unresolved":
            ok, err = assess(p, status=result["status"],
                             **{k: result[k] for k in ("x", "s", "z") if k in result})
            if not ok:
                print("Final residual:", err, file=sys.stderr)
                result = {"status": "unresolved"}
    except Exception:
        traceback.print_exc(file=sys.stderr)
        result = {"status": "unresolved"}
    meta = {"schema_version": 1, "status": result["status"]}
    arrays = {k: np.asarray(result[k], dtype=np.float64).reshape(-1)
              for k in ("x", "s", "z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(meta, allow_nan=False))
    with open(args.output, "wb") as f:
        np.savez(f, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
