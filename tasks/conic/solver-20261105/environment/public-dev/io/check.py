"""Independent checks in the supplied problem's original coordinates.

The benchmark input is trusted to define a convex QP (P is PSD); this module
checks its dimensions, symmetry and finite coefficients, not P's convexity.
All acceptance tolerances and certificate scaling are explicit below.
"""

import math
from time import perf_counter

import numpy as np
from scipy import linalg, sparse


STATUSES = {"solved", "primal_infeasible", "dual_infeasible", "unresolved"}


class CheckerResourceLimit(Exception):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def _finite(value):
    _require(np.isfinite(value).all(), "nonfinite arithmetic result")
    return value


def _ratio(numerator, denominator):
    return float(_finite(numerator / denominator))


def _vector(value, size, name):
    v = np.asarray(value)
    _require(v.ndim == 1 and v.size == size, f"{name} must be a vector of length {size}")
    _require(v.dtype.kind in "fiu", f"{name} must contain real numbers")
    v = v.astype(np.float64, copy=False)
    _require(np.isfinite(v).all(), f"{name} contains nonfinite values")
    return v


def _scalar(value, name):
    v = np.asarray(value)
    _require(v.ndim == 0 and v.dtype.kind in "fiu", f"{name} must be a real scalar")
    number = float(v)
    _require(math.isfinite(number), f"{name} is nonfinite")
    return number


def _scaled_matrix(matrix, scale):
    result = matrix.copy()
    result.data /= scale  # Avoid forming 1/scale, which can overflow for tiny scales.
    return result


def _problem_data(problem):
    q = _vector(problem.q, len(problem.q), "q")
    b = _vector(problem.b, len(problem.b), "b")
    matrices = []
    for name, shape in (("P", (q.size, q.size)), ("A", (b.size, q.size))):
        matrix = getattr(problem, name)
        _require(sparse.issparse(matrix) and matrix.dtype.kind in "fiu", f"{name} must be a real sparse matrix")
        matrix = matrix.tocsc(copy=True).astype(np.float64)
        matrix.check_format(full_check=True)
        _require(matrix.shape == shape, f"{name} dimension mismatch")
        _require(np.isfinite(matrix.data).all(), f"{name} contains nonfinite values")
        matrix.sum_duplicates()
        _require(np.isfinite(matrix.data).all(), f"{name} overflows when summing duplicates")
        matrices.append(matrix)
    P, A = matrices
    scaled = P / max(1.0, _norm(P.data))
    _require(_norm((scaled - scaled.T).data) <= 1e-12, "P is not symmetric")
    total = 0
    cones = []
    for cone in problem.cones:
        kind, dim = cone["type"], cone["dim"]
        _require(kind in {"zero", "nonnegative", "soc", "exponential", "psd"}, "unsupported cone")
        _require(type(dim) is int and dim > 0, "invalid cone dimension")
        _require(kind != "exponential" or dim == 3, "exponential cone dimension must be 3")
        _require(kind != "soc" or dim >= 2, "SOC dimension must be at least 2")
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        cones.append((kind, dim, width))
        total += width
    _require(total == b.size, "cone dimensions do not partition b")
    offset = _scalar(problem.objective_offset, "objective_offset")
    return P, q, A, b, cones, offset


def _exp_violation(v):
    """Homogeneous inequality defect; closure is checked as a separate branch."""
    x, y, z = map(float, v)
    boundary = max(abs(y), max(x, 0.0), max(-z, 0.0))
    if y > 0.0 and z > 0.0:
        interior = max(0.0, x - y * (math.log(z) - math.log(y)))
        return min(interior, boundary)
    return boundary


def _cones(v, cones, dual, cost):
    worst, blocks, start = 0.0, [], 0
    for kind, dim, width in cones:
        block = v[start:start + width]
        start += width
        u = block / max(1.0, _norm(block))
        if kind == "zero":
            violation = 0.0 if dual else _norm(u)  # The dual of {0} is free.
        elif kind == "nonnegative":
            violation = max(0.0, -float(np.min(u)))
        elif kind == "soc":
            violation = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
        elif kind == "exponential":
            # Scale first: e*w cannot overflow, even for enormous finite w.
            transformed = (-u[1], -u[0], math.e * u[2]) if dual else u
            violation = _exp_violation(transformed)
        else:
            limit = cost["psd_order_limit"]
            if limit is not None and dim > limit:
                raise CheckerResourceLimit(f"PSD order {dim} exceeds checker limit {limit}")
            matrix = np.zeros((dim, dim))
            position = 0
            for column in range(dim):
                values = u[position:position + column + 1].copy()
                values[:column] /= math.sqrt(2.0)
                matrix[:column + 1, column] = values
                matrix[column, :column + 1] = values
                position += column + 1
            cost["psd_eigenvalue_calls"] += 1
            cost["psd_max_order_checked"] = max(cost["psd_max_order_checked"], dim)
            cost["psd_max_matrix_bytes"] = max(cost["psd_max_matrix_bytes"], matrix.nbytes)
            started = perf_counter()
            try:
                eigenvalues = linalg.eigh(matrix, subset_by_index=[0, 0],
                    eigvals_only=True, driver="evr", check_finite=False)
                if not np.isfinite(eigenvalues).all():
                    raise np.linalg.LinAlgError("nonfinite minimum eigenvalue")
            finally:
                cost["psd_eigenvalue_seconds"] += perf_counter() - started
            violation = max(0.0, -float(eigenvalues[0]))
        _finite(violation)
        blocks.append({"type": kind, "dim": dim, "residual": violation})
        worst = max(worst, violation)
    return worst, blocks


def check_solution(problem, result, tol=1e-8, *, max_psd_order=None):
    """Check claimed solutions/certificates, never a solver's own success flag.

    Solved: linear defects / (1 + max term norm) <= tol, each normalized
    cone defect <= tol, and absolute OR relative objective gap <= tol.
    The objective offset is omitted from the relative-gap denominator.

    Certificates: normalize the supplied direction to infinity norm 1.
    Scale A and P by their largest absolute entry (zero matrices use 1),
    and b/q by their infinity norms. The negative normalized inner product
    must exceed tol; scaled zero-equation and cone defects divided by this
    margin must be <= tol. Dual-certificate slack is scaled along with A.
    This avoids relaxing tests merely by enlarging b/q or shrinking A.
    """
    report = {"passed": False, "reason": "not_checked", "status": None,
              "residuals": {}, "cone_blocks": {},
              "checker_cost": {"psd_order_limit": None,
                               "psd_eigenvalue_calls": 0, "psd_eigenvalue_seconds": 0.0,
                               "psd_max_order_checked": 0, "psd_max_matrix_bytes": 0}}
    try:
        tol = _scalar(tol, "tol")
        _require(0.0 < tol < 1.0, "tol must lie strictly between 0 and 1")
        report["tolerance"] = tol
        _require(max_psd_order is None or (type(max_psd_order) is int and max_psd_order > 0), "invalid PSD order limit")
        report["checker_cost"]["psd_order_limit"] = max_psd_order
        P, q, A, b, cones, offset = _problem_data(problem)
        _require(isinstance(result, dict), "result must be a dictionary")
        status = result.get("status")
        _require(isinstance(status, str) and status in STATUSES, "invalid result status")
        report["status"] = status
        vectors = {}
        for name, size in (("x", q.size), ("s", b.size), ("z", b.size)):
            if result.get(name) is not None:
                vectors[name] = _vector(result[name], size, name)
        objectives = {name: _scalar(result[name], name)
                      for name in ("objective_primal", "objective_dual") if result.get(name) is not None}
        if status == "unresolved":
            report["reason"] = "unresolved"
            return report
        required = {"solved": {"x", "s", "z"}, "primal_infeasible": {"z"},
                    "dual_infeasible": {"x", "s"}}[status]
        _require(required <= vectors.keys(), f"missing required vectors: {sorted(required - vectors.keys())}")
        if status != "solved":
            _require(not objectives, "finite optimal objectives are not defined for an infeasibility certificate")
        residuals = report["residuals"]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if status == "solved":
                x, s, z = (vectors[key] for key in ("x", "s", "z"))
                Ax, Px, Atz = (_finite(v) for v in (A @ x, P @ x, A.T @ z))
                residuals["primal"] = _norm(_finite(Ax + s - b)) / (1.0 + max(_norm(Ax), _norm(s), _norm(b)))
                residuals["dual"] = _norm(_finite(Px + q + Atz)) / (1.0 + max(_norm(Px), _norm(q), _norm(Atz)))
                xPx, qx, bz = (float(_finite(v)) for v in (x @ Px, q @ x, b @ z))
                primal0, dual0 = 0.5 * xPx + qx, -0.5 * xPx - bz
                _finite(np.array([primal0, dual0]))
                gap = abs(primal0 - dual0)
                scale = max(abs(primal0), abs(dual0))
                residuals["gap_absolute"] = float(_finite(gap))
                residuals["gap_relative"] = gap / scale if scale else 0.0
                report["objectives"] = {"primal": float(_finite(primal0 + offset)),
                                        "dual": float(_finite(dual0 + offset)),
                                        "primal_without_offset": primal0, "dual_without_offset": dual0}
                residuals["complementarity_absolute"] = abs(float(_finite(s @ z)))
                for key, vector, dual in (("primal_cone", s, False), ("dual_cone", z, True)):
                    residuals[key], report["cone_blocks"][key] = _cones(vector, cones, dual, report["checker_cost"])
                objectives_ok = True
                for name, reported in objectives.items():
                    side = name.removeprefix("objective_")
                    core = primal0 if side == "primal" else dual0
                    error = abs(reported - report["objectives"][side]) / (1.0 + abs(core))
                    residuals[name + "_reported_error"] = float(_finite(error))
                    objectives_ok &= error <= tol
                report["passed"] = bool(all(residuals[key] <= tol for key in
                    ("primal", "dual", "primal_cone", "dual_cone")) and
                    (gap <= tol or residuals["gap_relative"] <= tol) and objectives_ok)
            else:
                A_scale, P_scale = _norm(A.data) or 1.0, _norm(P.data) or 1.0
                if status == "primal_infeasible":
                    z = vectors["z"]
                    direction_scale, data_scale = _norm(z), _norm(b)
                    _require(direction_scale > 0.0 and data_scale > 0.0, "certificate has no negative margin")
                    z = z / direction_scale
                    margin = -float(_finite((b / data_scale) @ z))
                    residuals["certificate_margin_normalized"] = margin
                    _require(margin > tol, "certificate negative margin is too small or has wrong sign")
                    residuals["certificate_linear"] = _ratio(_norm(_finite(_scaled_matrix(A, A_scale).T @ z)), margin)
                    cone_error, report["cone_blocks"]["dual_cone"] = _cones(z, cones, True, report["checker_cost"])
                    residuals["dual_cone"] = _ratio(cone_error, margin)
                    report["passed"] = bool(residuals["certificate_linear"] <= tol and residuals["dual_cone"] <= tol)
                else:
                    x, s = vectors["x"], vectors["s"]
                    direction_scale, data_scale = _norm(x), _norm(q)
                    _require(direction_scale > 0.0 and data_scale > 0.0, "certificate has no negative margin")
                    x, s = x / direction_scale, _finite(s / direction_scale)
                    margin = -float(_finite((q / data_scale) @ x))
                    residuals["certificate_margin_normalized"] = margin
                    _require(margin > tol, "certificate negative margin is too small or has wrong sign")
                    residuals["certificate_quadratic"] = _ratio(_norm(_finite(_scaled_matrix(P, P_scale) @ x)), margin)
                    scaled_slack = _finite(s / A_scale)
                    residuals["certificate_linear"] = _ratio(_norm(_finite(_scaled_matrix(A, A_scale) @ x + scaled_slack)), margin)
                    cone_error, report["cone_blocks"]["primal_cone"] = _cones(scaled_slack, cones, False, report["checker_cost"])
                    residuals["primal_cone"] = _ratio(cone_error, margin)
                    report["passed"] = bool(all(residuals[key] <= tol for key in
                        ("certificate_quadratic", "certificate_linear", "primal_cone")))
            report["reason"] = status + "_verified" if report["passed"] else "residual_tolerance_exceeded"
    except CheckerResourceLimit as error:
        report["reason"] = "evaluation_error"
        report["error_type"] = "checker_resource_limit"
        report["detail"] = str(error)
    except np.linalg.LinAlgError as error:
        report["reason"] = "evaluation_error"
        report["error_type"] = "eigensolver_failure"
        report["detail"] = str(error)
    except MemoryError:
        report["reason"] = "evaluation_error"
        report["error_type"] = "checker_memory_limit"
    except (FloatingPointError, OverflowError) as error:
        report["reason"] = "checker_numerical_failure"
        report["detail"] = str(error)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
        report["reason"] = "invalid_input_or_result"
        report["detail"] = str(error)
    return report
