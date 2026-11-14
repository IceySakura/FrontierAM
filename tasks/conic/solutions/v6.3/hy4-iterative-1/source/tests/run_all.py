#!/usr/bin/env python3
"""Run a solver over every generated case and check each output.

Examples:
  python tests/run_all.py
  python tests/run_all.py --only 'socp,sdp' --timeout 30
  python tests/run_all.py './run.sh' --only primal_infeasible --timeout 10
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import fnmatch
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
TESTS = Path(__file__).resolve().parent
SUBMISSION = TESTS.parent
ROOT = SUBMISSION.parent
IO_DIR = ROOT / "samples" / "io"
if not (IO_DIR / "check_result.py").exists():
    IO_DIR = Path("/workspace/samples/io")
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(TESTS / "_pycache"))


def patterns_from(values):
    return [piece.strip() for value in values for piece in value.split(",")
            if piece.strip()]


def selected(case, patterns):
    if not patterns:
        return True
    haystacks = [case.get("id", ""), case.get("family", ""),
                 case.get("expected_status", "")]
    haystacks.extend(case.get("tags", []))
    haystacks.extend(case.get("cone_types", []))
    for pattern in patterns:
        lowered = pattern.lower()
        for value in haystacks:
            text = str(value)
            if lowered in text.lower() or fnmatch.fnmatchcase(text, pattern):
                return True
    return False


def solver_missing(command, cwd):
    if not command:
        return True
    first = command[0]
    if os.path.isabs(first) or "/" in first:
        path = Path(first) if os.path.isabs(first) else cwd / first
        return not path.exists()
    return shutil.which(first) is None


def residual_summary(residuals):
    if not isinstance(residuals, dict) or not residuals:
        return "-"
    abbreviations = {
        "primal": "rp", "dual": "rd", "gap_absolute": "gap",
        "gap_relative": "grel", "complementarity_absolute": "comp",
        "primal_cone": "pK", "dual_cone": "dK",
        "certificate_linear": "lin", "certificate_quadratic": "quad",
        "certificate_margin_normalized": "margin",
        "objective_primal_reported_error": "op",
        "objective_dual_reported_error": "od",
    }
    values = []
    for key, value in residuals.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append((key, number))
    if not values:
        return "-"
    worst = max((abs(number) for key, number in values
                 if key != "certificate_margin_normalized"), default=0.0)
    parts = [f"max={worst:.1e}"]
    for key, number in values:
        label = abbreviations.get(key, key[:5])
        parts.append(f"{label}={number:.1e}")
        if len(parts) >= 8:
            break
    return " ".join(parts)


def load_checker_report(report_path, stdout):
    for text in (report_path.read_text() if report_path.exists() else "", stdout):
        if not text.strip():
            continue
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "outcome" in data:
                return data
        except (ValueError, OSError):
            pass
    return {"outcome": "checker_error", "check": {"status": None,
                                                  "residuals": {}}}


def run_case(case, command, cwd, checker, output_dir, timeout, env,
             missing_solver):
    case_id = case["id"]
    input_path = (TESTS / case["input"]).resolve()
    out_path = (output_dir / f"{case_id}.solver.npz").resolve()
    report_path = (output_dir / f"{case_id}.check.json").resolve()
    for path in (out_path, report_path):
        if path.exists():
            path.unlink()
    if not input_path.exists():
        return {"id": case_id, "outcome_status": "input_missing/-",
                "residuals": "-", "time": 0.0, "passing": False,
                "solver_error": True, "checker_error": True,
                "timeout": False, "status_mismatch": False}
    if missing_solver:
        return {"id": case_id, "outcome_status": "solver_missing/-",
                "residuals": "no solver output", "time": 0.0,
                "passing": False, "solver_error": True,
                "checker_error": True, "timeout": False,
                "status_mismatch": False}

    solver_command = list(command) + ["--input", str(input_path),
                                      "--output", str(out_path)]
    started = time.perf_counter()
    solver_rc, solver_error, timed_out, solver_note = None, False, False, ""
    try:
        process = subprocess.Popen(solver_command, cwd=str(cwd), env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        try:
            _, _ = process.communicate(timeout=timeout)
            solver_rc = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
    except (FileNotFoundError, PermissionError, OSError) as error:
        solver_error = True
        solver_note = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started

    outcome, status, residuals, checker_error = "no_output", None, {}, True
    if out_path.exists():
        check_command = [sys.executable, str(checker), "--input", str(input_path),
                         "--result", str(out_path), "--output", str(report_path)]
        try:
            check = subprocess.run(check_command, cwd=str(TESTS), env=env,
                                   capture_output=True, text=True,
                                   timeout=max(30.0, timeout))
            report = load_checker_report(report_path, check.stdout)
            outcome = str(report.get("outcome"))
            details = report.get("check", {}) or {}
            status = details.get("status")
            residuals = details.get("residuals", {}) or {}
            checker_error = outcome in {"checker_error", "evaluation_error"}
            if report.get("error"):
                solver_note = (solver_note + " " + str(report["error"])).strip()
        except subprocess.TimeoutExpired:
            outcome, timed_out, checker_error = "checker_timeout", True, True
        except OSError as error:
            outcome, checker_error = "checker_error", True
            solver_note = (solver_note + " " + f"{type(error).__name__}: {error}").strip()

    status_mismatch = status is not None and status != case["expected_status"]
    outcome_status = f"{outcome}/{status if status is not None else '-'}"
    if timed_out:
        outcome_status = f"timeout/{status if status is not None else '-'}"
    elif solver_error:
        outcome_status = "solver_error/-"
    if solver_rc not in (None, 0):
        outcome_status += f"!rc{solver_rc}"
    if status_mismatch:
        outcome_status += "!status"
    if solver_note:
        outcome_status += "!"
    verified = outcome == "verified"
    passing = (verified and not solver_error and not timed_out and
               not checker_error and not status_mismatch and solver_rc == 0)
    return {"id": case_id, "outcome_status": outcome_status,
            "residuals": residual_summary(residuals), "time": elapsed,
            "passing": passing, "solver_error": bool(solver_error or solver_rc not in (None, 0)),
            "checker_error": bool(checker_error or not verified),
            "timeout": bool(timed_out), "status_mismatch": bool(status_mismatch),
            "note": solver_note}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", nargs="?", default=None,
                        help="solver command string (overrides --solver)")
    parser.add_argument("--solver", dest="solver_option", default="./run.sh",
                        help="solver command string (default: ./run.sh)")
    parser.add_argument("--cwd", type=Path, default=SUBMISSION,
                        help="working directory for the solver (default: submission root)")
    parser.add_argument("--only", action="append", default=[],
                        help="comma-separated id/family/tag/status filter; repeatable")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-case solver timeout in seconds")
    parser.add_argument("--checker", type=Path,
                        default=IO_DIR / "check_result.py",
                        help="path to the official check_result.py")
    parser.add_argument("--manifest", type=Path,
                        default=TESTS / "cases_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=TESTS / "_runs",
                        help="solver outputs and checker reports (below tests/)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(TESTS.resolve()):
        parser.error("--output-dir must be inside the tests/ directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cwd = args.cwd.resolve()
    solver_text = args.solver if args.solver is not None else args.solver_option
    try:
        command = shlex.split(solver_text)
    except ValueError as error:
        parser.error(f"invalid solver command: {error}")
    if not command:
        parser.error("solver command is empty")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONPYCACHEPREFIX", str(TESTS / "_pycache"))

    if not args.manifest.exists():
        print(f"manifest missing: {args.manifest}")
        return 1
    try:
        manifest = json.loads(args.manifest.read_text())
    except (ValueError, OSError) as error:
        print(f"cannot read manifest: {type(error).__name__}: {error}")
        return 1
    patterns = patterns_from(args.only)
    cases = [case for case in manifest.get("cases", [])
             if selected(case, patterns)]
    if not cases:
        print("no cases selected")
        return 0

    missing = solver_missing(command, cwd)
    checker = args.checker.resolve()
    if missing:
        print(f"solver command not found (cwd={cwd}): {solver_text}")
    if not checker.exists():
        print(f"checker not found: {checker}")

    header = (f"{'case':<34} {'family':<18} {'n':>5} {'m':>5} "
              f"{'time':>7} {'outcome/status':<30} residuals")
    print(header)
    print("-" * len(header))
    results = []
    for case in cases:
        result = run_case(case, command, cwd, checker, output_dir,
                          max(0.001, float(args.timeout)), env, missing)
        results.append(result)
        print(f"{case['id']:<34} {case['family']:<18} "
              f"{int(case['n']):>5} {int(case['m']):>5} "
              f"{result['time']:>7.2f} {result['outcome_status']:<30} "
              f"{result['residuals']}")
        if result.get("note"):
            print(f"    note: {result['note'][:160]}")

    passing = sum(1 for result in results if result["passing"])
    solver_failures = sum(1 for result in results if result["solver_error"])
    checker_failures = sum(1 for result in results if result["checker_error"])
    timeouts = sum(1 for result in results if result["timeout"])
    mismatches = sum(1 for result in results if result["status_mismatch"])
    max_time = max((result["time"] for result in results), default=0.0)
    print(f"\n{len(results)} cases: {passing} passing, "
          f"{solver_failures} solver failures, {checker_failures} checker failures, "
          f"{timeouts} timeouts, {mismatches} status mismatches; "
          f"max solver time {max_time:.2f}s")
    return 0 if passing == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
