"""Reference CPU implementation of the fixed-CG restarted PDHCG method.

The implementation is intentionally straightforward: all matrix operations are
CSR matvecs, and the only inner linear solver is CG on
Q + tau^{-1} I.  A small active-set polish is used after the first-order phase
to make the reference a useful accuracy anchor; it is not exposed as a
requirement for submissions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np


_FAST_CACHE = {}


class _CSRLinearOperator:
    """Small NumPy-only CSR operator for the hot PDHCG path.

    Importing ``scipy.sparse.linalg`` costs a substantial fraction of the
    runtime of the smaller instances.  The verifier only needs matrix-vector
    products during the first-order phase, so a vectorized CSR adapter keeps
    that path independent of SciPy.  SciPy remains an optional, lazy fallback
    for the rare active-set polish below.
    """

    __slots__ = ("data", "indices", "shape", "_rows", "_transpose")

    def __init__(self, matrix, *, transpose=False):
        self.data = matrix.data
        self.indices = matrix.indices
        self.shape = matrix.shape[::-1] if transpose else matrix.shape
        self._rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64),
                               np.diff(matrix.indptr))
        self._transpose = bool(transpose)

    def __matmul__(self, value):
        value = np.asarray(value, dtype=np.float64)
        if self._transpose:
            return np.bincount(self.indices,
                               weights=self.data * value[self._rows],
                               minlength=self.shape[0]).astype(np.float64, copy=False)
        return np.bincount(self._rows,
                           weights=self.data * value[self.indices],
                           minlength=self.shape[0]).astype(np.float64, copy=False)

    @property
    def T(self):
        # The transpose shares the original CSR arrays and only materializes
        # the row-number map needed by the opposite multiply.
        if self._transpose:
            return _CSRLinearOperator.__new_from_parts(self, False)
        return _CSRLinearOperator.__new_from_parts(self, True)

    @classmethod
    def __new_from_parts(cls, other, transpose):
        value = object.__new__(cls)
        value.data = other.data
        value.indices = other.indices
        value.shape = other.shape[::-1]
        value._rows = other._rows
        value._transpose = bool(transpose)
        return value


def _scipy_components():
    """Load sparse direct-solver support only when the accuracy fallback runs."""
    from scipy import sparse as scipy_sparse
    from scipy.sparse.linalg import MatrixRankWarning, spsolve
    return scipy_sparse, spsolve, MatrixRankWarning


def _fast_ops(problem):
    """Cache vectorized CSR operators for the trusted reference only."""
    key = id(problem)
    cached = _FAST_CACHE.get(key)
    # Keep a strong reference to the problem in the cache value.  Without it,
    # a sequential audit of several problems could recycle an object id and
    # accidentally reuse the previous instance's operators.
    if cached is None or cached[0] is not problem:
        q = _CSRLinearOperator(problem.Q)
        a = _CSRLinearOperator(problem.A)
        _FAST_CACHE[key] = (problem, q, a)
        cached = _FAST_CACHE[key]
    return cached[1], cached[2]

from qpbench.check import check_solution
from qpbench.model import load_problem, save_output


def _norm_estimate(problem, iterations=12):
    _, a = _fast_ops(problem)
    rng = np.random.default_rng(99173 + problem.n + problem.m)
    v = rng.normal(size=problem.n)
    v /= max(np.linalg.norm(v), 1.0)
    for _ in range(iterations):
        v = a.T @ (a @ v)
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            return 1.0
        v /= norm
    return max(1.0, float(np.sqrt(norm)))


def _qnorm_estimate(problem, iterations=8):
    q, _ = _fast_ops(problem)
    rng = np.random.default_rng(18731 + problem.n)
    v = rng.normal(size=problem.n)
    v /= max(np.linalg.norm(v), 1.0)
    value = 1.0
    for _ in range(iterations):
        w = q @ v
        value = float(np.linalg.norm(w))
        if not np.isfinite(value) or value == 0:
            return 1.0
        v = w / value
    return max(1.0, value)


def _cg(problem, rhs, x0, tau, steps, tolerance):
    """Fixed/early-stopped CG for (Q + I/tau)x=rhs."""
    x = np.array(x0, dtype=np.float64, copy=True)

    q, _ = _fast_ops(problem)

    def apply(value):
        return q @ value + value / tau

    residual = rhs - apply(x)
    direction = residual.copy()
    rr = float(np.dot(residual, residual))
    target = tolerance * (1.0 + float(np.linalg.norm(rhs)))
    for _ in range(max(1, steps)):
        if np.sqrt(max(rr, 0.0)) <= target:
            break
        curvature = float(np.dot(direction, apply(direction)))
        if not np.isfinite(curvature) or curvature <= 1e-30:
            break
        alpha = rr / curvature
        x += alpha * direction
        residual -= alpha * apply(direction)
        new_rr = float(np.dot(residual, residual))
        if not np.isfinite(new_rr):
            break
        if np.sqrt(max(new_rr, 0.0)) <= target:
            break
        direction = residual + (new_rr / max(rr, 1e-300)) * direction
        rr = new_rr
    return x


def _kkt_metrics(problem, x, y):
    q, a = _fast_ops(problem)
    ax = a @ x
    slack = problem.b - ax
    stationarity = np.linalg.norm(q @ x + problem.c + a.T @ y, ord=np.inf)
    violation = max(0.0, float(np.max(-slack))) if slack.size else 0.0
    dual = max(0.0, -float(np.min(y))) if y.size else 0.0
    comp = float(np.max(np.abs(y * slack))) if slack.size else 0.0
    return max(stationarity, violation, dual, comp)


def _polish(problem, x, y):
    """Solve the small active KKT system when a clear active set is present."""
    scipy_sparse, spsolve, MatrixRankWarning = _scipy_components()
    active = np.flatnonzero(problem.b - problem.A.matvec(x) <= 2e-4)
    if active.size == 0 or active.size > min(problem.m, 256):
        return x, y
    try:
        a = np.zeros((active.size, problem.n), dtype=np.float64)
        for i, row in enumerate(active):
            start, end = problem.A.indptr[row], problem.A.indptr[row + 1]
            a[i, problem.A.indices[start:end]] = problem.A.data[start:end]
        q = scipy_sparse.csr_matrix((problem.Q.data, problem.Q.indices, problem.Q.indptr),
                                    shape=problem.Q.shape)
        a_sparse = scipy_sparse.csr_matrix(a)
        kkt = scipy_sparse.bmat([[q, a_sparse.T], [a_sparse, None]], format="csc")
        rhs = np.r_[-problem.c, problem.b[active]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution = spsolve(kkt, rhs)
            if not np.all(np.isfinite(solution)):
                raise np.linalg.LinAlgError("sparse KKT solve returned nonfinite values")
        except Exception:
            dense = kkt.toarray()
            solution, *_ = np.linalg.lstsq(dense, rhs, rcond=1e-12)
        candidate_x = solution[:problem.n]
        multipliers = solution[problem.n:]
        candidate_y = np.zeros(problem.m, dtype=np.float64)
        candidate_y[active] = multipliers
        if np.min(candidate_y) < -2e-6:
            return x, y
        if _kkt_metrics(problem, candidate_x, candidate_y) > max(1e-5, 1e-5 * np.linalg.norm(problem.c)):
            return x, y
        return candidate_x, candidate_y
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return x, y


def _active_set_reference(problem, max_iterations=160):
    """Small sparse active-set fallback for the generated, feasible QPs.

    This is deliberately a separate accuracy path from PDHCG.  It is used only
    to make the frozen reference reliable on ill-conditioned cases where a
    fixed inner-CG budget has not yet reached the checker tolerance.
    """
    active: list[int] = []
    x = np.zeros(problem.n, dtype=np.float64)
    y = np.zeros(problem.m, dtype=np.float64)
    scipy_sparse, spsolve, MatrixRankWarning = _scipy_components()

    def solve_active(rows):
        if rows:
            a = np.zeros((len(rows), problem.n), dtype=np.float64)
            for i, row in enumerate(rows):
                start, end = problem.A.indptr[row], problem.A.indptr[row + 1]
                a[i, problem.A.indices[start:end]] = problem.A.data[start:end]
            q = scipy_sparse.csr_matrix((problem.Q.data, problem.Q.indices, problem.Q.indptr),
                                        shape=problem.Q.shape)
            kkt = scipy_sparse.bmat([[q, scipy_sparse.csr_matrix(a).T],
                                     [scipy_sparse.csr_matrix(a), None]], format="csc")
            rhs = np.r_[-problem.c, problem.b[np.asarray(rows, dtype=np.int64)]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    solution = spsolve(kkt, rhs)
                if not np.all(np.isfinite(solution)):
                    raise np.linalg.LinAlgError("sparse KKT solve returned nonfinite values")
            except Exception:
                solution, *_ = np.linalg.lstsq(kkt.toarray(), rhs, rcond=1e-12)
            return solution[:problem.n], solution[problem.n:]
        q = scipy_sparse.csr_matrix((problem.Q.data, problem.Q.indices, problem.Q.indptr),
                                    shape=problem.Q.shape)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution = spsolve(q.tocsc(), -problem.c)
            if not np.all(np.isfinite(solution)):
                raise np.linalg.LinAlgError("sparse Q solve returned nonfinite values")
            return solution, np.empty(0)
        except Exception:
            return np.linalg.lstsq(q.toarray(), -problem.c, rcond=1e-12)[0], np.empty(0)

    for _ in range(max_iterations):
        x, multipliers = solve_active(active)
        slack = problem.b - problem.A.matvec(x)
        if active:
            negative = np.flatnonzero(multipliers < -2e-8)
            if negative.size:
                del active[int(negative[np.argmin(multipliers[negative])])]
                continue
        violations = np.flatnonzero(slack < -2e-8)
        if violations.size:
            row = int(violations[np.argmin(slack[violations])])
            if row not in active:
                active.append(row)
                continue
        y = np.zeros(problem.m, dtype=np.float64)
        if active:
            y[np.asarray(active, dtype=np.int64)] = multipliers
        if _kkt_metrics(problem, x, y) <= 2e-7 * (1.0 + np.linalg.norm(problem.c, ord=np.inf)):
            return x, y
        # A numerically marginal active row can be replaced by the most
        # violated row on the next iteration.
        if not violations:
            break
    return x, y


def solve(problem, *, max_outer=240, restart=18, cg_steps=10, tolerance=2e-9):
    _, a_fast = _fast_ops(problem)
    norm_a = _norm_estimate(problem)
    norm_q = _qnorm_estimate(problem)
    # This conservative choice satisfies the PDHG product condition while
    # keeping the regularized CG systems reasonably conditioned.
    tau = sigma = 0.72 / max(norm_a, 1.0)
    x = np.zeros(problem.n, dtype=np.float64)
    y = np.zeros(problem.m, dtype=np.float64)
    best = (x.copy(), y.copy(), float("inf"))
    for _outer in range(max_outer):
        x_sum = np.zeros_like(x)
        y_sum = np.zeros_like(y)
        for inner in range(restart):
            rhs = x / tau - problem.c - a_fast.T @ y
            x_next = _cg(problem, rhs, x, tau, cg_steps, tolerance)
            extrapolated = 2.0 * x_next - x
            y_next = np.maximum(0.0, y + sigma * (a_fast @ extrapolated - problem.b))
            x, y = x_next, y_next
            weight = inner + 1
            x_sum += (x - x_sum) / weight
            y_sum += (y - y_sum) / weight
        x, y = x_sum, y_sum
        metric = _kkt_metrics(problem, x, y)
        if metric < best[2]:
            best = (x.copy(), y.copy(), metric)
        # A looser scale-aware test avoids spending the full budget on tiny
        # coefficients while retaining a strict final independent check.
        if metric <= 2e-6 * (1.0 + norm_q + np.linalg.norm(problem.c, ord=np.inf)):
            break
    x, y, _ = best
    current_check = check_solution(
        problem, {"metadata": {"status": "solved"}, "x": x, "y": y}, tol=1e-7)
    if not current_check["passed"]:
        x, y = _polish(problem, x, y)
        current_check = check_solution(
            problem, {"metadata": {"status": "solved"}, "x": x, "y": y}, tol=1e-7)
    if not current_check["passed"]:
        candidate_x, candidate_y = _active_set_reference(problem)
        if _kkt_metrics(problem, candidate_x, candidate_y) < _kkt_metrics(problem, x, y):
            x, y = candidate_x, candidate_y
    return x, y


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    started = time.perf_counter()
    problem = load_problem(args.input)
    x, y = solve(problem)
    report = check_solution(problem, {"metadata": {"status": "solved"}, "x": x, "y": y}, tol=1e-7)
    save_output(args.output, x, y, objective=report.get("objective"))
    print(json.dumps({"elapsed_seconds": time.perf_counter() - started,
                      "passed": report["passed"], "reason": report["reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
