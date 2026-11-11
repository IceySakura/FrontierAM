#!/usr/bin/env python3
"""Small generated mathematical regression tests; no development data needed."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import math
import numpy as np
from scipy import sparse
from common import Problem, assess
from solver import solve


def problem(P, q, A, b, cones):
    return Problem(sparse.csc_matrix(P, shape=(len(q), len(q))), np.asarray(q, float),
                   sparse.csc_matrix(A, shape=(len(b), len(q))), np.asarray(b, float), cones)


def main():
    cases = [
        (problem([[2.0]], [-4.0], [[-1.0]], [0.0],
                 [{"type": "nonnegative", "dim": 1}]), "solved", 2.0),
        (problem([[0.0]], [1.0], [[-1.0], [0.0]], [0.0, 1.0],
                 [{"type": "soc", "dim": 2}]), "solved", 1.0),
        (problem([[0.0]], [1.0], [[0.0], [0.0], [-1.0]], [0.0, 1.0, 0.0],
                 [{"type": "exponential", "dim": 3}]), "solved", 1.0),
        (problem([[0.0]], [1.0], [[-1.0], [0.0], [-1.0]], [0.0, math.sqrt(2), 0.0],
                 [{"type": "psd", "dim": 2}]), "solved", 1.0),
        (problem([[0.0]], [0.0], [[-1.0], [1.0]], [-1.0, 0.0],
                 [{"type": "nonnegative", "dim": 2}]), "primal_infeasible", None),
        (problem([[0.0]], [-1.0], [[-1.0]], [0.0],
                 [{"type": "nonnegative", "dim": 1}]), "dual_infeasible", None),
        (problem(np.empty((0, 0)), [], np.empty((2, 0)), [0.0, 1.0],
                 [{"type": "nonnegative", "dim": 2}]), "solved", None),
        (problem(np.empty((0, 0)), [], np.empty((2, 0)), [-1.0, 1.0],
                 [{"type": "nonnegative", "dim": 2}]), "primal_infeasible", None),
        (problem([[2.0]], [-4.0], np.empty((0, 1)), [], []), "solved", 2.0),
        (problem([[0.0]], [-1.0], np.empty((0, 1)), [], []), "dual_infeasible", None),
        (problem([[1.0]], [0.0], [[1.0], [1.0]], [1.0, 2.0],
                 [{"type": "zero", "dim": 2}]), "primal_infeasible", None),
    ]
    for index, (p, status, optimum) in enumerate(cases):
        r = solve(p)
        assert r["status"] == status, (index, r["status"], status)
        ok, error = assess(p, status=r["status"],
                          **{k: r[k] for k in ("x", "s", "z") if k in r})
        assert ok, (index, error)
        if optimum is not None:
            assert abs(r["x"][0] - optimum) <= 1e-6
    print(f"{len(cases)} generated regression cases passed")


if __name__ == "__main__":
    main()
