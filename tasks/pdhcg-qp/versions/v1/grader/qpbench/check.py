"""Independent KKT validation for the inequality QP interface."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import Problem


def objective(problem: Problem, x: np.ndarray) -> float:
    qx = problem.Q.matvec(x)
    return float(0.5 * np.dot(x, qx) + np.dot(problem.c, x) + problem.c0)


def check_solution(problem: Problem, result: dict[str, Any], *, tol: float = 1e-7) -> dict[str, Any]:
    """Return JSON-compatible diagnostics; never trust candidate self-reports."""
    report: dict[str, Any] = {
        "passed": False, "reason": "invalid_output", "primal_violation": None,
        "dual_violation": None, "stationarity": None, "complementarity": None,
        "objective": None,
    }
    try:
        metadata = result.get("metadata", {})
        if metadata.get("status") != "solved":
            report["reason"] = "status_not_solved"
            return report
        x = np.asarray(result["x"], dtype=np.float64)
        y = np.asarray(result["y"], dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size != problem.n or y.size != problem.m:
            report["reason"] = "wrong_shape"
            return report
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            report["reason"] = "nonfinite_output"
            return report
        ax = problem.A.matvec(x)
        qx = problem.Q.matvec(x)
        aty = problem.A.t_matvec(y)
        scale_p = 1.0 + max(float(np.max(np.abs(ax))) if ax.size else 0.0,
                            float(np.max(np.abs(problem.b))) if problem.b.size else 0.0)
        scale_d = 1.0 + max(float(np.max(np.abs(qx))) if qx.size else 0.0,
                            float(np.max(np.abs(problem.c))) if problem.c.size else 0.0,
                            float(np.max(np.abs(aty))) if aty.size else 0.0)
        slack = problem.b - ax
        primal = max(0.0, float(np.max(-slack))) / scale_p if slack.size else 0.0
        dual = max(0.0, -float(np.min(y))) / (1.0 + float(np.max(np.abs(y)))) if y.size else 0.0
        stationarity = float(np.max(np.abs(qx + problem.c + aty))) / scale_d
        # Complementarity is y_i * (b_i - A_i x), not only the product on
        # violated rows.  A feasible row with a positive multiplier is also
        # invalid and must be rejected.
        complementarity = float(np.max(np.abs(y * slack))) / (
            1.0 + (float(np.max(np.abs(problem.b))) if problem.b.size else 0.0))
        value = objective(problem, x)
        fields = (primal, dual, stationarity, complementarity, value)
        if not all(math.isfinite(v) for v in fields):
            report["reason"] = "nonfinite_diagnostic"
            return report
        report.update(primal_violation=primal, dual_violation=dual,
                      stationarity=stationarity, complementarity=complementarity,
                      objective=value)
        report["passed"] = max(primal, dual, stationarity, complementarity) <= tol
        report["reason"] = "passed" if report["passed"] else "kkt_failure"
        return report
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        report["reason"] = f"invalid_output:{type(exc).__name__}"
        return report
