#!/usr/bin/env python3
"""Fresh-process JSON Lines checks using independent analytic functions."""
import argparse
import json
import os
import selectors
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments import make_case


def run_one(folder, dimension, budget, seed, function, exact, limit=30):
    started = time.monotonic()
    proc = subprocess.Popen(
        [str(Path(folder) / "run.sh")], cwd=folder, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    init = {"type": "init", "dimension": dimension, "budget": budget,
            "seed": seed, "bounds": [[0, 1]] * dimension}
    proc.stdin.write(json.dumps(init) + "\n")
    proc.stdin.flush()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    used = 0
    estimate = None
    try:
        while estimate is None:
            remaining = started + limit - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("process exceeded wall-clock limit")
            line = proc.stdout.readline()
            if not line:
                raise ValueError("process exited without a result")
            message = json.loads(line)
            if message["type"] == "query":
                points = np.asarray(message["points"], dtype=float)
                if (points.ndim != 2 or points.shape[1] != dimension
                        or len(points) > 512 or len(points) == 0
                        or not np.all(np.isfinite(points))
                        or np.any(points < 0) or np.any(points > 1)):
                    raise ValueError("invalid query")
                used += len(points)
                if used > budget:
                    raise ValueError("budget exceeded")
                values = np.asarray(function(points), dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError("non-finite test function")
                proc.stdin.write(json.dumps({"type": "values", "values": values.tolist()}) + "\n")
                proc.stdin.flush()
            elif message["type"] == "result":
                estimate = float(message["estimate"])
                if not np.isfinite(estimate):
                    raise ValueError("non-finite result")
            else:
                raise ValueError("unexpected message")
        proc.stdin.close()
        proc.wait(timeout=max(0.01, started + limit - time.monotonic()))
        if proc.returncode != 0:
            raise ValueError(f"nonzero exit {proc.returncode}")
        if proc.stdout.read().strip():
            raise ValueError("output after result")
    except Exception:
        proc.kill()
        proc.wait()
        raise
    finally:
        selector.close()
    elapsed = time.monotonic() - started
    relative = abs(estimate - exact) / max(abs(exact), 1e-300)
    return used, elapsed, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 5, 8])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2048])
    parser.add_argument("--families", nargs="+", default=[
        "wave3", "wavehigh1", "hingeparallel12", "corner_relu", "gauss1",
        "powridge3"])
    parser.add_argument("--seed", type=int, default=8917)
    args = parser.parse_args()
    worst = 0.0
    cases = 0
    for family in args.families:
        for dimension in args.dimensions:
            rng = np.random.default_rng(args.seed + 137 * dimension)
            function, exact = make_case(family, rng, dimension)
            for budget in args.budgets:
                used, elapsed, error = run_one(
                    args.submission, dimension, budget, args.seed,
                    function, exact)
                cases += 1
                worst = max(worst, elapsed)
                print(f"{family} d={dimension} budget={budget} points={used} "
                      f"time={elapsed:.3f} relative_error={error:.4g}", flush=True)
    print(f"PASS: {cases} fresh processes; maximum time {worst:.3f}s", flush=True)


if __name__ == "__main__":
    main()
