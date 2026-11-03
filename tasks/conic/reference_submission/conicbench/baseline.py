"""Pinned Clarabel 0.11.1 candidate producer; acceptance belongs to check_solution.

Run in its own process: python -m conicbench.baseline --input INPUT --output OUTPUT.
Thread environment variables are set before importing numeric libraries. Importing
this module into a process that already initialized BLAS is not the benchmark path.

Upstream ClarabelBenchmarks@3679912c6bbd3f64c5c962f9d1c09d524561c412,
src/benchmarks/solver_config.jl explicitly sets max_threads=1 and
direct_solve_method=auto for normal Clarabel/Rust runs. It does not override
presolve/equilibration/chordal defaults; a separate nonchordal variant disables
chordal decomposition. Here the 0.11.1 defaults are pinned and recorded, with
those three features explicitly enabled, complete_dual enabled for original-row
dual output, and five full-accuracy tolerances set by --tol. Reduced tolerances
and tol_ktratio retain the recorded 0.11.1 defaults. The upstream OMP=16 setting
is not reused. This is a local baseline, not the unidentified paper solver build.

NPZ contract: metadata_json is scalar JSON with schema_version=1, a *claimed*
status, optional objective_primal/objective_dual (including the input offset),
and diagnostics. Solved candidates have x/s/z; primal-infeasible candidates z;
dual-infeasible candidates x/s. Unresolved candidates have no vector payload.
No solver status, including AlmostSolved, is an independent acceptance decision.
"""

import os

THREAD_ENVIRONMENT = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")}
os.environ.update(THREAD_ENVIRONMENT)

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import clarabel
import numpy as np
import scipy
from scipy import sparse

from .model import load_problem


EXPECTED_VERSION = "0.11.1"


def _json_safe(value):
    """Keep diagnostic nonfinite values explicit, never emit nonstandard JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def run_baseline(input_path, output_path, *, time_limit=300.0, tol=1e-8):
    """Write a candidate NPZ and return its JSON metadata, without accepting it."""
    if clarabel.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"Expected Clarabel {EXPECTED_VERSION}, found {clarabel.__version__}")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    if not math.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must lie strictly between 0 and 1")
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the input problem")
    with input_path.open("rb") as stream:
        input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    problem = load_problem(input_path)
    settings = clarabel.DefaultSettings()
    for name, value in {
        "verbose": False, "max_threads": 1, "direct_solve_method": "auto",
        "presolve_enable": True, "equilibrate_enable": True,
        "chordal_decomposition_enable": True,
        "chordal_decomposition_compact": True,
        "chordal_decomposition_complete_dual": True,
        "chordal_decomposition_merge_method": "clique_graph",
        "time_limit": float(time_limit), "tol_gap_abs": float(tol),
        "tol_gap_rel": float(tol), "tol_feas": float(tol),
        "tol_infeas_abs": float(tol), "tol_infeas_rel": float(tol),
    }.items():
        setattr(settings, name, value)
    constructors = {"zero": clarabel.ZeroConeT, "nonnegative": clarabel.NonnegativeConeT,
                    "soc": clarabel.SecondOrderConeT, "psd": clarabel.PSDTriangleConeT}
    cones = [clarabel.ExponentialConeT() if cone["type"] == "exponential"
             else constructors[cone["type"]](cone["dim"]) for cone in problem.cones]
    started = perf_counter()
    solver = clarabel.DefaultSolver(sparse.triu(problem.P, format="csc"), problem.q,
                                   problem.A, problem.b, cones, settings)
    setup_seconds = perf_counter() - started
    started = perf_counter()
    solution = solver.solve()
    solve_wall_seconds = perf_counter() - started
    info = solver.get_info()
    raw_status = str(solution.status)
    status = {"Solved": "solved", "AlmostSolved": "solved",
              "PrimalInfeasible": "primal_infeasible",
              "AlmostPrimalInfeasible": "primal_infeasible",
              "DualInfeasible": "dual_infeasible",
              "AlmostDualInfeasible": "dual_infeasible"}.get(raw_status, "unresolved")
    actual_settings = solver.get_settings()
    metadata = {
        "schema_version": 1, "status": status,
        "diagnostics": {
            "candidate_only": True, "solver": "Clarabel", "raw_status": raw_status,
            "input_sha256": input_sha256, "input_name": problem.name,
            "settings": {name: getattr(actual_settings, name) for name in dir(actual_settings)
                         if not name.startswith("_") and not callable(getattr(actual_settings, name))},
            "versions": {"clarabel": clarabel.__version__, "python": platform.python_version(),
                         "numpy": np.__version__, "scipy": scipy.__version__},
            "platform": platform.platform(), "machine": platform.machine(),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
            "linear_solver": {name: getattr(info.linsolver, name)
                              for name in ("name", "threads", "direct", "nnzA", "nnzL")},
            "solve_time": float(solution.solve_time), "iterations": int(solution.iterations),
            "setup_wall_seconds": setup_seconds, "solve_wall_seconds": solve_wall_seconds,
            "timing_note": "solve_time is solver-reported; external end-to-end time is separate",
            "configuration_note": "Local pinned 0.11.1 defaults plus explicit settings; not a verified paper build",
            "coordinate_system": "original input P/q/A/b and cone ordering; solver transforms reversed",
        },
    }
    vector_names = {"solved": ("x", "s", "z"), "primal_infeasible": ("z",),
                    "dual_infeasible": ("x", "s"), "unresolved": ()}[status]
    arrays = {name: np.asarray(getattr(solution, name), dtype=np.float64) for name in vector_names}
    if status == "solved":
        for name, value in (("objective_primal", solution.obj_val),
                            ("objective_dual", solution.obj_val_dual)):
            value = float(value) + problem.objective_offset
            if math.isfinite(value):
                metadata[name] = value
    metadata = _json_safe(metadata)
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=output_path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()
    run_baseline(args.input, args.output, time_limit=args.time_limit, tol=args.tol)


if __name__ == "__main__":
    main()
