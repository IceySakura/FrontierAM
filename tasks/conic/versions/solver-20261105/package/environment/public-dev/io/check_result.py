#!/usr/bin/env python3
"""Check a candidate-v1 result locally; timing/resource evaluation is separate."""

import argparse
import json
from pathlib import Path

import numpy as np

from model import load_problem
from check import check_solution


def check_files(input_path, result_path):
    try:
        problem = load_problem(input_path)
    except Exception as error:
        return {"outcome": "evaluation_error", "error": f"Input: {type(error).__name__}: {error}"}
    try:
        with np.load(result_path, allow_pickle=False) as data:
            raw = data["metadata_json"]
            if raw.ndim != 0 or raw.dtype.kind != "U":
                raise ValueError("metadata_json must be a scalar Unicode array")
            metadata = json.loads(str(raw))
            if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
                    or metadata["schema_version"] != 1):
                raise ValueError("Result metadata must have schema_version=1")
            json.dumps(metadata, allow_nan=False)
            result = {k: metadata[k] for k in ("status", "objective_primal", "objective_dual")
                      if k in metadata}
            result.update({k: data[k] for k in ("x", "s", "z") if k in data})
    except Exception as error:
        return {"outcome": "rejected", "error": f"Result: {type(error).__name__}: {error}"}
    report = check_solution(problem, result, tol=1e-8)
    outcome = ("verified" if report["passed"] else "evaluation_error"
               if report["reason"] == "evaluation_error" else "unresolved"
               if result.get("status") == "unresolved" else "rejected")
    return {"outcome": outcome, "check": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() in (args.input.resolve(), args.result.resolve()):
        parser.error("The report must not overwrite an input or candidate result")
    report = check_files(args.input, args.result)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["outcome"] == "verified" else 2 if report["outcome"] == "evaluation_error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
