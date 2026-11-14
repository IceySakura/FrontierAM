#!/usr/bin/env python3
"""Small driver for testing lpqp.solve_lpqp on NPZ inputs."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

IO_DIR = Path("/workspace/samples/io")
if str(IO_DIR) not in sys.path:
    sys.path.insert(0, str(IO_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    import lpqp
    t_import = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as npz:
        meta = json.loads(str(npz["metadata_json"]))
        arrays = {key: npz[key] for key in npz.files}
    t_read = time.perf_counter()
    result = lpqp.solve_lpqp(
        arrays["P_data"], arrays["P_indices"], arrays["P_indptr"], arrays["P_shape"],
        arrays["q"],
        arrays["A_data"], arrays["A_indices"], arrays["A_indptr"], arrays["A_shape"],
        arrays["b"], meta["cones"], time_budget=args.time_budget)
    t_solve = time.perf_counter()

    args.result.parent.mkdir(parents=True, exist_ok=True)
    out = {"metadata_json": np.array(json.dumps({"schema_version": 1,
                                                 "status": result["status"]}))}
    for key in ("x", "s", "z"):
        out[key] = np.asarray(result[key], dtype=np.float64)
    np.savez(args.result, **out)
    t_write = time.perf_counter()

    from model import load_problem
    from check import check_solution
    problem = load_problem(args.input)
    report = check_solution(problem, result, tol=1e-8)
    summary = {
        "status": result["status"],
        "checker_passed": bool(report["passed"]),
        "checker_reason": report["reason"],
        "residuals": report.get("residuals", {}),
        "import_s": round(t_import - t0, 6),
        "read_s": round(t_read - t_import, 6),
        "solve_s": round(t_solve - t_read, 6),
        "write_s": round(t_write - t_solve, 6),
        "total_s": round(t_write - t0, 6),
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
