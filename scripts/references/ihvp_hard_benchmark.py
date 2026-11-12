#!/usr/bin/env python3
"""Serial local IHVP diagnostics. All records stay under ignored var/; no scores.

Example (maintainer-provided data, not a formal calibration):
  python scripts/references/ihvp_hard_benchmark.py --manifest var/.../manifest.json \
      --out var/outputs/ihvp-hard/public --methods cold_cg lbfgs_pcg galerkin16 galerkin48 warm_cg
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "tasks/ihvp-sequence"
VERSION = TASK / "versions/v3"
METHODS = ("cold_cg", "lbfgs_pcg", "galerkin16", "galerkin48", "warm_cg", "projected16")
sys.path.insert(0, str(VERSION / "grader"))
from runtime.api import run_case, clean
from runtime.backend import Case


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configurations(args):
    candidate = TASK / "references/candidates/galerkin_cg"
    paths = {"cold_cg": TASK / "versions/v1/grader/calibration_references/cold_cg",
             "lbfgs_pcg": TASK / "references/r1"}
    output = {}
    for method in args.methods:
        options = {}
        if method in ("galerkin16", "galerkin48", "warm_cg", "projected16"):
            source = candidate
            options = {"history_size": 48 if method == "galerkin48" else
                       (0 if method == "warm_cg" else 16),
                       "warm_start": True, "projected": method == "projected16",
                       "wall_budget": args.wall_budget, "maxiter_factor": 50,
                       "residual_refresh": 128, "basis_rtol": 1e-10,
                       "coarse_rtol": 1e-12}
        else:
            source = paths[method]
        if not (source / "solver.py").is_file():
            raise FileNotFoundError("missing reference entry point")
        files = {str(path.relative_to(REPO)): digest(path)
                 for path in sorted(source.rglob("*.py")) if "__pycache__" not in path.parts}
        output[method] = {"source": str(source.relative_to(REPO)), "options": options,
                          "source_sha256": files,
                          "source_options_fingerprint": fingerprint({"files": files, "options": options})}
    return output


def runtime_identity():
    spec = importlib.util.spec_from_file_location("ihvp_v3_grader_identity",
                                                 VERSION / "grader/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.identity()


def aggregate(rows):
    if any(row.get("status") == "infrastructure_failure" for row in rows):
        status = "infrastructure_failure"
    elif all(row.get("valid") is True for row in rows) and rows:
        status = "ok"
    else:
        status = "submission_failure"
    all_losses = [row.get("mean_energy_loss") for row in rows]
    finite_losses = [float(loss) for loss in all_losses
                     if loss is not None and np.isfinite(loss)]
    walls = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
    peaks = [float(row["peak_rss_mib"]) for row in rows if row.get("peak_rss_mib") is not None]
    return {"status": status, "sequences": len(rows),
            "valid_sequences": sum(row.get("valid") is True for row in rows),
            "resource_compliant_sequences": sum(row.get("resource_compliant") is True for row in rows),
            "mean_energy_loss": float(np.mean(finite_losses))
                if status == "ok" and len(finite_losses) == len(rows) else None,
            "mean_energy_loss_finite_diagnostic": float(np.mean(finite_losses)) if finite_losses else None,
            "mean_wall_attempt_seconds": float(np.mean(walls)) if walls else None,
            "max_wall_attempt_seconds": max(walls) if walls else None,
            "mean_peak_rss_mib": float(np.mean(peaks)) if peaks else None,
            "max_peak_rss_mib": max(peaks) if peaks else None,
            "score": None, "native_score": None}


def run(args):
    out = args.out.resolve()
    if not out.is_relative_to(REPO / "var"):
        raise ValueError("--out must be under ignored repository var/")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    full_entries = manifest["cases"]
    if not full_entries:
        raise ValueError("empty data manifest")
    entries = full_entries[:args.case_limit] if args.case_limit is not None else full_entries
    identifiers = [str(entry["id"]) for entry in entries]
    if len(set(identifiers)) != len(identifiers) or any(
            re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier) is None for identifier in identifiers):
        raise ValueError("unsafe or duplicated case identifiers")
    methods = configurations(args)
    runtime, runtime_fingerprint = runtime_identity()
    controls = {"schema": "ihvp-hard-local-diagnostics-v1", "version": "v3",
                "manifest_sha256": digest(manifest_path),
                "cases": [{"id": entry["id"], "sha256": entry["sha256"]} for entry in entries],
                "full_case_count": len(full_entries), "selected_case_count": len(entries),
                "methods": methods, "method_order": list(args.methods),
                "wall_seconds": args.wall_seconds, "memory_mib": 256.0,
                "runtime_fingerprint": runtime_fingerprint,
                "harness_sha256": digest(__file__), "score": None, "native_score": None}
    run_fingerprint = fingerprint(controls)
    header_path = out / "run_manifest.json"
    if header_path.exists():
        previous = json.loads(header_path.read_text())
        if not args.resume:
            raise FileExistsError("output already exists; pass --resume with matching fingerprints")
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError("resume fingerprint mismatch: data, source, options, runtime or limits changed")
    else:
        write_json(header_path, {**controls, "run_fingerprint": run_fingerprint,
                                 "runtime_identity": runtime,
                                 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "scope": "local diagnostic only; never a formal task score"})

    # Workers run immutable local source snapshots. A concurrent source edit
    # cannot silently change a later method while retaining an earlier hash.
    staged_sources = {}
    for method, configuration in methods.items():
        original = REPO / configuration["source"]
        staged = out / "source_snapshots" / method
        staged.mkdir(parents=True, exist_ok=True)
        for relative, expected in configuration["source_sha256"].items():
            source = REPO / relative
            destination = staged / source.relative_to(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            if digest(destination) != expected:
                raise ValueError("staged source snapshot digest mismatch")
        staged_sources[method] = staged

    results = {method: [] for method in args.methods}
    for index, entry in enumerate(entries, 1):
        # Stop rather than mixing results across a source edit made during a run.
        current_methods = configurations(args)
        _, current_runtime_fingerprint = runtime_identity()
        if current_methods != methods or current_runtime_fingerprint != runtime_fingerprint:
            raise RuntimeError("source/runtime changed during benchmark")
        teacher = (manifest_path.parent / entry["path"]).resolve()
        if not teacher.is_relative_to(manifest_path.parent) or digest(teacher) != entry["sha256"]:
            raise ValueError("teacher path or SHA does not match manifest")
        case = None
        teacher_error = None
        begin = time.monotonic()
        try:
            case = Case(teacher)
            for key in ("dimension", "steps"):
                if key in entry and int(entry[key]) != int(getattr(case, key)):
                    raise ValueError("case dimensions differ from manifest")
        except Exception as error:
            teacher_error = f"{type(error).__name__}: {error}"
        teacher_load_seconds = time.monotonic() - begin
        for method in args.methods:
            directory = out / method / f"case_{index:03d}"
            record_path = directory / "result.json"
            if record_path.exists():
                row = json.loads(record_path.read_text())
                if (row.get("run_fingerprint") != run_fingerprint or
                        row.get("teacher_sha256") != entry["sha256"] or
                        row.get("method") != method or row.get("case_id") != entry["id"] or
                        row.get("source_options_fingerprint") !=
                        methods[method]["source_options_fingerprint"]):
                    raise ValueError("case resume fingerprint mismatch")
                prediction = row.get("prediction_file")
                if prediction and digest(directory / prediction) != row.get("prediction_sha256"):
                    raise ValueError("case prediction resume digest mismatch")
                results[method].append(row)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            if teacher_error is not None:
                row = {"valid": False, "resource_compliant": False,
                       "status": "infrastructure_failure", "mean_energy_loss": None,
                       "termination_reason": "teacher_setup_failure", "error": teacher_error}
            else:
                try:
                    row = run_case(staged_sources[method], case,
                                   wall_seconds=args.wall_seconds, memory_mib=256.0,
                                   seed=1, options=methods[method]["options"],
                                   log_path=directory / "worker.log")
                    predictions = row.pop("prediction_array")
                    temporary = directory / "predictions.npy.tmp"
                    with temporary.open("wb") as stream:
                        np.save(stream, predictions, allow_pickle=False)
                    prediction_path = directory / "predictions.npy"
                    temporary.replace(prediction_path)
                    row.update(prediction_file="predictions.npy",
                               prediction_sha256=digest(prediction_path),
                               prediction_shape=list(predictions.shape))
                    row["status"] = ("ok" if row["valid"] else
                                     ("infrastructure_failure" if row.get("infrastructure_failure")
                                      else "submission_failure"))
                except Exception as error:
                    row = {"valid": False, "resource_compliant": False,
                           "status": "infrastructure_failure", "mean_energy_loss": None,
                           "termination_reason": "trusted_runner_exception",
                           "error": f"{type(error).__name__}: {error}"}
            row.update(method=method, case_index=index, case_id=entry["id"],
                       teacher_sha256=entry["sha256"], teacher_load_seconds=teacher_load_seconds,
                       run_fingerprint=run_fingerprint, runtime_fingerprint=runtime_fingerprint,
                       source_options_fingerprint=methods[method]["source_options_fingerprint"],
                       version="v3", score=None, native_score=None,
                       scope="local diagnostic only; never a formal task score")
            write_json(record_path, row)
            results[method].append(row)
            loss = row.get("mean_energy_loss")
            label = f"{loss:.9g}" if loss is not None else "none"
            print(f"case={index:03d} method={method} status={row['status']} loss={label}", flush=True)
        write_json(out / "aggregate.json", {
            "run_fingerprint": run_fingerprint, "version": "v3",
            "completed_cases": index, "selected_cases": len(entries),
            "full_manifest_cases": len(full_entries),
            "complete_selected_manifest": index == len(entries),
            "complete_full_manifest": index == len(full_entries),
            "methods": {method: aggregate(rows) for method, rows in results.items()},
            "scope": "local diagnostic only; never a formal task score", "score": None})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS[:5]))
    parser.add_argument("--wall-seconds", type=float, default=30.0)
    parser.add_argument("--wall-budget", type=float, default=27.5)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 30:
        parser.error("--wall-seconds must be in [1,30]")
    if args.wall_budget <= 0 or (args.case_limit is not None and args.case_limit <= 0):
        parser.error("wall budget and case limit must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    try:
        return run(args)
    except Exception as error:
        out = args.out.resolve()
        if out.is_relative_to(REPO / "var"):
            write_json(out / "abort.json", {"status": "infrastructure_failure",
                       "error": f"{type(error).__name__}: {error}",
                       "score": None, "native_score": None})
        print("benchmark aborted; inspect local abort.json (no formal score)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
