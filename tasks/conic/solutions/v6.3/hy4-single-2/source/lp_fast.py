"""High-accuracy LP specialization for the conic benchmark.

The public problem has the form

    minimize q.T x + c0     subject to A x + s = b, s in K,

with P = 0, and K a product of zero cones (equalities) and nonnegative
orthants (inequalities).  Independent verification uses the original
coordinates, so all vectors returned here are checked with an inline copy
of the verifier's residuals before being exposed.

Only numpy/scipy are used.  The module never assumes that the development
sample checker is available at runtime.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import lsqr

# The checker uses 1e-8.  A modest margin protects against differences in
# sparse summation order after its CSC canonicalization.
_TOL = 1e-8
_VERIFY_TOL = 0.8e-8
_MARGIN_MIN = 1.05e-8


def _inf_norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def _all_finite(*values):
    return all(bool(np.isfinite(v).all()) for v in values)


def _cone_defect(v, blocks, dual):
    """Verifier's cone defect, specialized to zero and nonnegative cones."""
    worst = 0.0
    for kind, start, width in blocks:
        block = v[start:start + width]
        u = block / max(1.0, _inf_norm(block))
        if kind == "zero":
            violation = 0.0 if dual else _inf_norm(u)
        else:
            violation = max(0.0, -float(np.min(u))) if u.size else 0.0
        if not np.isfinite(violation):
            return np.inf
        worst = max(worst, violation)
    return worst


def _check_solved(A, q, b, blocks, x, s, z, tol=_VERIFY_TOL):
    """Inline equivalent of check.py's solved branch for P = 0."""
    if not _all_finite(x, s, z):
        return False
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            Ax = A @ x
            Atz = A.T @ z
            if not _all_finite(Ax, Atz):
                return False
            rp = _inf_norm(Ax + s - b) / (1.0 + max(
                _inf_norm(Ax), _inf_norm(s), _inf_norm(b)))
            rd = _inf_norm(q + Atz) / (1.0 + max(
                _inf_norm(q), _inf_norm(Atz)))
            cone_s = _cone_defect(s, blocks, False)
            cone_z = _cone_defect(z, blocks, True)
            p0 = float(q @ x)
            d0 = -float(b @ z)
            gap = abs(p0 - d0)
            scale = max(abs(p0), abs(d0))
            gap_ok = gap <= tol or (gap / scale <= tol if scale > 0.0 else True)
            residuals = (rp, rd, cone_s, cone_z, gap)
            return (np.isfinite(residuals).all() and rp <= tol and rd <= tol
                    and cone_s <= tol and cone_z <= tol and gap_ok)
    except (FloatingPointError, ValueError, TypeError, OverflowError):
        return False


def _check_primal_cert(A_scaled, b, blocks, z, tol=_VERIFY_TOL):
    """Inline equivalent of check.py's primal-infeasibility branch."""
    if not _all_finite(z):
        return False
    b_scale, z_scale = _inf_norm(b), _inf_norm(z)
    if b_scale <= 0.0 or z_scale <= 0.0:
        return False
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            zhat = z / z_scale
            beta = -float((b / b_scale) @ zhat)
            if not np.isfinite(beta) or beta <= _MARGIN_MIN:
                return False
            linear = _inf_norm(A_scaled.T @ zhat) / beta
            cone = _cone_defect(zhat, blocks, True) / beta
            return (np.isfinite(linear) and np.isfinite(cone)
                    and linear <= tol and cone <= tol)
    except (FloatingPointError, ValueError, TypeError, OverflowError):
        return False


def _check_dual_cert(A_scaled, q, blocks, a, x, s, tol=_VERIFY_TOL):
    """Inline equivalent of check.py's dual-infeasibility branch (P = 0)."""
    if not _all_finite(x, s):
        return False
    q_scale, x_scale = _inf_norm(q), _inf_norm(x)
    if q_scale <= 0.0 or x_scale <= 0.0:
        return False
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            xhat = x / x_scale
            shat = s / x_scale
            alpha = -float((q / q_scale) @ xhat)
            if not np.isfinite(alpha) or alpha <= _MARGIN_MIN:
                return False
            scaled_slack = shat / a
            linear = _inf_norm(A_scaled @ xhat + scaled_slack) / alpha
            cone = _cone_defect(scaled_slack, blocks, False) / alpha
            # The quadratic certificate term is exactly zero because P = 0.
            return (np.isfinite(linear) and np.isfinite(cone)
                    and linear <= tol and cone <= tol)
    except (FloatingPointError, ValueError, TypeError, OverflowError):
        return False


def _row_blocks(cones, m):
    """Return (blocks, zero_rows, nonnegative_rows)."""
    blocks = []
    zeros, nonnegs = [], []
    start = 0
    for cone in cones:
        if not isinstance(cone, dict):
            return None, None, None
        kind, dim = cone.get("type"), cone.get("dim")
        if kind not in ("zero", "nonnegative"):
            return None, None, None
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim <= 0:
            return None, None, None
        dim = int(dim)
        if start + dim > m:
            return None, None, None
        blocks.append((kind, start, dim))
        rows = np.arange(start, start + dim, dtype=np.int64)
        if kind == "zero":
            zeros.append(rows)
        else:
            nonnegs.append(rows)
        start += dim
    if start != m:
        return None, None, None
    zero_rows = np.concatenate(zeros) if zeros else np.empty(0, dtype=np.int64)
    nonneg_rows = np.concatenate(nonnegs) if nonnegs else np.empty(0, dtype=np.int64)
    return blocks, zero_rows, nonneg_rows


def _scaled_A(A, a):
    result = A.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result.data /= a  # Same operation as the checker's _scaled_matrix.
    return result


def _row_slice(A, rows):
    if rows.size == 0:
        return None
    return A.tocsr()[rows].tocsc()


def _row_scaling(A, b, q):
    """Normalize each constraint row and the objective for HiGHS.

    Multiplying row i by r_i > 0 leaves the feasible set unchanged.  Solving
    min (q/q_scale).T x s.t. (R A) x <= R b gives duals that map back as
    z = R * q_scale * z_scaled.  No column scaling is used, so the stationarity
    residual in the checker's original coordinates is not amplified.
    """
    m = A.shape[0]
    if m == 0:
        q_scale = max(1.0, _inf_norm(q))
        return A.copy(), b.copy(), q / q_scale, np.empty(0), q_scale
    A_abs = A.tocsr(copy=True)
    A_abs.data = np.abs(A_abs.data)
    row_scale = np.abs(b).astype(np.float64, copy=True)
    for i in range(m):
        lo, hi = A_abs.indptr[i], A_abs.indptr[i + 1]
        if hi > lo:
            row_scale[i] = max(row_scale[i],
                               float(np.max(A_abs.data[lo:hi])))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        safe = np.clip(row_scale, 1e-100, 1e100)
        r = np.ones(m, dtype=np.float64)
        nz = safe > 0.0
        r[nz] = 1.0 / safe[nz]
        A_s = (sparse.diags(r, format="csc") @ A).tocsc()
        b_s = r * b
        q_scale = max(1.0, _inf_norm(q))
        q_s = q / q_scale
    if not _all_finite(A_s.data, b_s, q_s, r):
        return None
    return A_s, b_s, q_s, r, q_scale


def _run_linprog(c, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 bounds=(None, None), method="highs", options=None,
                 deadline=None):
    kwargs = {"c": np.asarray(c, dtype=np.float64), "bounds": bounds,
              "method": method, "options": dict(options or {})}
    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        kwargs["A_ub"] = A_ub
        kwargs["b_ub"] = np.asarray(b_ub, dtype=np.float64)
    if A_eq is not None and b_eq is not None and np.size(b_eq) > 0:
        kwargs["A_eq"] = A_eq
        kwargs["b_eq"] = np.asarray(b_eq, dtype=np.float64)
    if deadline is not None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.02:
            return None
        # HiGHS interprets time_limit in seconds; leave a little time for
        # verification and for a possible subsequent attempt.
        kwargs["options"]["time_limit"] = min(
            max(0.01, 0.85 * remaining),
            float(kwargs["options"].get("time_limit", np.inf)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return linprog(**kwargs)


def _primary_options(tol=1e-9, presolve=True, method="highs"):
    opts = {"presolve": presolve, "disp": False,
            "primal_feasibility_tolerance": tol,
            "dual_feasibility_tolerance": tol}
    if method != "highs-ds":
        # hiGHS requires this only for IPM / automatic method selection.
        opts["ipm_optimality_tolerance"] = max(1e-12, min(1e-10, tol))
    return opts


def _primal_polish_candidates(A, b, blocks, x, zero_rows, nonneg_rows,
                              row_factor):
    """Small corrections that make active slacks nonnegative to roundoff.

    HiGHS applies a feasibility tolerance to its own scaled rows.  When those
    rows are very large in the checker's coordinates, an active slack can be
    slightly negative.  One or two minimum-norm least-squares corrections of
    the active equalities/inequalities remove this without re-solving the LP.
    """
    candidates = [x]
    if row_factor is None or b.size == 0 or x.size == 0:
        return candidates
    n = x.size
    A_csr = A.tocsr()
    current = x.copy()
    max_active = max(50, 4 * n)
    try:
        for _ in range(2):
            with np.errstate(over="ignore", invalid="ignore"):
                s = b - A @ current
                normalized = row_factor * s
            if not _all_finite(s, normalized):
                break
            selected_zero = np.empty(0, dtype=np.int64)
            if zero_rows.size:
                residual = np.abs(normalized[zero_rows])
                selected_zero = zero_rows[residual > 1e-12]
                if selected_zero.size > max_active:
                    keep = np.argpartition(residual[residual > 1e-12],
                                           -max_active)[-max_active:]
                    selected_zero = selected_zero[keep]
            selected_nonneg = np.empty(0, dtype=np.int64)
            if nonneg_rows.size:
                slack = normalized[nonneg_rows]
                active = slack < 1e-12
                selected_nonneg = nonneg_rows[active]
                if selected_nonneg.size > max_active:
                    keep = np.argpartition(slack[active], max_active - 1)[:max_active]
                    selected_nonneg = selected_nonneg[keep]
            rows = np.concatenate((selected_zero, selected_nonneg))
            if rows.size == 0:
                break
            is_nonneg = np.zeros(rows.size, dtype=bool)
            is_nonneg[selected_zero.size:] = True
            target = -normalized[rows].copy()
            # In normalized row units, a small positive original-coordinate
            # slack is r_i * 1e-9; cap it so tiny rows cannot demand a large
            # correction in x.
            with np.errstate(over="ignore", invalid="ignore"):
                desired = np.minimum(
                    1e-10, 1e-14 + 1e-9 * row_factor[rows[is_nonneg]])
            target[is_nonneg] += desired
            M = (sparse.diags(row_factor[rows], format="csc")
                 @ A_csr[rows].tocsc()).tocsc()
            if n <= 200 and rows.size <= 2000:
                dx = np.linalg.lstsq(M.toarray(), target, rcond=None)[0]
            else:
                dx = lsqr(M, target, atol=1e-12, btol=1e-12,
                          conlim=1e12, iter_lim=min(200, max(20, n)))[0]
            dx = np.asarray(dx, dtype=np.float64).reshape(-1)
            if not _all_finite(dx):
                break
            scored = []
            for step in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 1.5, -0.5):
                candidate = current + step * dx
                if not _all_finite(candidate):
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    slack = b - A @ candidate
                if not _all_finite(slack):
                    continue
                defect = _cone_defect(slack, blocks, False)
                # Prefer small corrections when cone defects are comparable.
                penalty = 1e-14 * abs(step) * _inf_norm(dx) / (
                    1.0 + _inf_norm(current))
                scored.append((defect + penalty, abs(step), candidate))
            if not scored:
                break
            scored.sort(key=lambda item: (item[0], item[1]))
            for _, _, candidate in scored[:3]:
                candidates.append(candidate)
            current = scored[0][2]
    except Exception:
        pass
    return candidates


def _build_solved(A, q, b, blocks, zero_rows, nonneg_rows, x,
                  marginal_n, marginal_z, dual_scale=None, row_factor=None):
    """Try a few numerically harmless repair variants of the dual vector.

    If the LP was solved after multiplying row i by r_i and dividing the
    objective by q_scale, ``dual_scale`` is r * q_scale: HiGHS marginals are
    mapped back with z_original = -dual_scale * marginals.
    """
    if x is None or not _all_finite(x):
        return None
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if dual_scale is None:
        dual_scale = np.ones(b.size, dtype=np.float64)
    z_base = np.zeros(b.size, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if nonneg_rows.size:
            z_base[nonneg_rows] = -dual_scale[nonneg_rows] * np.asarray(
                marginal_n, dtype=np.float64)
        if zero_rows.size:
            z_base[zero_rows] = -dual_scale[zero_rows] * np.asarray(
                marginal_z, dtype=np.float64)
    if not _all_finite(z_base):
        return None

    candidates = [z_base]
    # HiGHS can return -0 or a tiny negative inequality multiplier.
    clipped = z_base.copy()
    if nonneg_rows.size:
        clipped[nonneg_rows] = np.maximum(clipped[nonneg_rows], 0.0)
    candidates.append(clipped)
    # Treat numerically zero multipliers as exact zeros: this can improve
    # complementarity without materially changing stationarity.
    magnitude = _inf_norm(clipped)
    if magnitude > 0.0:
        sparse_z = clipped.copy()
        sparse_z[np.abs(sparse_z) < 1e-13 * magnitude] = 0.0
        candidates.append(sparse_z)
        if nonneg_rows.size:
            sparse_z2 = sparse_z.copy()
            sparse_z2[nonneg_rows] = np.maximum(sparse_z2[nonneg_rows], 0.0)
            candidates.append(sparse_z2)
    candidates.append(-z_base)
    clipped_neg = -clipped
    if nonneg_rows.size:
        clipped_neg[nonneg_rows] = np.maximum(clipped_neg[nonneg_rows], 0.0)
    candidates.append(clipped_neg)

    x_candidates = _primal_polish_candidates(A, b, blocks, x, zero_rows,
                                             nonneg_rows, row_factor)
    for x_try in x_candidates:
        with np.errstate(over="ignore", invalid="ignore"):
            s = b - A @ x_try  # Recomputed in the checker's original coordinates.
        if not _all_finite(s):
            continue
        s_variants = [s]
        # With very large row scales, cancellation in b - A @ x can leave a
        # zero-cone residual or a tiny negative orthant slack even though the
        # point is feasible to HiGHS's tolerance.  Projecting those roundoff
        # values onto K is accepted only when the verifier's relative primal
        # residual passes with the same safety margin.
        if _cone_defect(s, blocks, False) > _VERIFY_TOL:
            projected = s.copy()
            if nonneg_rows.size:
                projected[nonneg_rows] = np.maximum(projected[nonneg_rows], 0.0)
            if zero_rows.size:
                projected[zero_rows] = 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                Ax = A @ x_try
                denom = 1.0 + max(_inf_norm(Ax), _inf_norm(projected),
                                  _inf_norm(b))
                rp_projected = _inf_norm(Ax + projected - b) / denom
            if (_all_finite(projected) and np.isfinite(rp_projected)
                    and rp_projected <= _VERIFY_TOL):
                s_variants.append(projected)
        for s_try in s_variants:
            for z in candidates:
                if not _all_finite(z):
                    continue
                if _check_solved(A, q, b, blocks, x_try, s_try, z):
                    return {"status": "solved", "x": x_try, "s": s_try, "z": z}
    return None


def _certificate_row_factors(A, kappa):
    """Positive row multipliers for better-conditioned auxiliary LPs.

    Row i of the checker matrix is multiplied by r_i in (0, 1].  kappa bounds
    the ratio between the smallest and largest multiplier, avoiding a complete
    loss of the large rows' contribution to the certificate objective.
    """
    m = A.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.float64)
    A_abs = A.tocsr(copy=True)
    A_abs.data = np.abs(A_abs.data)
    norms = np.zeros(m, dtype=np.float64)
    for i in range(m):
        lo, hi = A_abs.indptr[i], A_abs.indptr[i + 1]
        if hi > lo:
            norms[i] = float(np.max(A_abs.data[lo:hi]))
    global_max = _inf_norm(norms)
    if global_max <= 0.0:
        return np.ones(m, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        floor = max(global_max * kappa, np.finfo(np.float64).tiny)
        r = np.ones(m, dtype=np.float64)
        nonzero = norms > 0.0
        r[nonzero] = 1.0 / np.maximum(norms[nonzero], floor)
        largest = _inf_norm(r)
        if largest > 0.0:
            r /= largest
    if not _all_finite(r):
        return np.ones(m, dtype=np.float64)
    return r


def _certificate_configs(A_scaled):
    """(row multipliers, scaled matrix) pairs, unscaled first."""
    configs = [(np.ones(A_scaled.shape[0]), A_scaled)]
    for kappa in (1e-3, 1e-8, 1e-12):
        r = _certificate_row_factors(A_scaled, kappa)
        M = (sparse.diags(r, format="csc") @ A_scaled).tocsc()
        if _all_finite(M.data):
            configs.append((r, M))
    return configs


def _primal_certificate(A_scaled, b, blocks, nonneg_rows, deadline):
    """Farkas LP: min (b/|b|).T z s.t. (A/a).T z = 0, z_n >= 0, |z| <= 1.

    For a row-scaled configuration, solve for w with z = R w.  Positive R
    leaves the dual cone unchanged and can make the nullspace equations much
    better conditioned; the final certificate is always checked in the
    checker's original coordinates.
    """
    b_scale = _inf_norm(b)
    m, n = A_scaled.shape
    if b_scale <= 0.0 or m == 0:
        return None
    b_hat = b / b_scale
    lower = -np.ones(m, dtype=np.float64)
    lower[nonneg_rows] = 0.0
    bounds = list(zip(lower.tolist(), [1.0] * m))
    methods = (
        ("highs", _primary_options(1e-10)),
        ("highs-ds", _primary_options(1e-10, method="highs-ds")),
    )
    for config_index, (r, M) in enumerate(_certificate_configs(A_scaled)):
        c = r * b_hat
        A_eq = M.T.tocsc()  # M.T w = (A/a).T (R w) = 0
        b_eq = np.zeros(n, dtype=np.float64)
        attempts = methods
        if config_index == 0:
            attempts = methods + (("highs", {"presolve": True, "disp": False}),)
        for method, options in attempts:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            res = _run_linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                               method=method, options=options,
                               deadline=deadline)
            if res is None or res.status != 0 or res.x is None:
                continue
            w = np.asarray(res.x, dtype=np.float64).reshape(-1)
            if nonneg_rows.size:
                w[nonneg_rows] = np.maximum(w[nonneg_rows], 0.0)
            with np.errstate(over="ignore", invalid="ignore"):
                z = r * w
            scale = _inf_norm(z)
            if scale <= 0.0 or not np.isfinite(scale) or not _all_finite(z):
                continue
            z /= scale
            if nonneg_rows.size:
                z[nonneg_rows] = np.maximum(z[nonneg_rows], 0.0)
            if _check_primal_cert(A_scaled, b, blocks, z):
                return {"status": "primal_infeasible", "z": z}
    return None


def _dual_certificate(A_scaled, q, blocks, a, nonneg_rows, zero_rows, deadline):
    """Direction LP: min (q/|q|).T x s.t. (A/a)_n x <= 0, (A/a)_z x = 0.

    Positive row scaling of the auxiliary constraints does not change the
    recession cone.  The returned slack is always built from the unscaled
    checker matrix A/a.
    """
    q_scale = _inf_norm(q)
    if q_scale <= 0.0 or q.size == 0:
        return None
    c = q / q_scale
    b_ub = np.zeros(nonneg_rows.size, dtype=np.float64)
    b_eq = np.zeros(zero_rows.size, dtype=np.float64)
    methods = (
        ("highs", _primary_options(1e-10)),
        ("highs-ds", _primary_options(1e-10, method="highs-ds")),
    )
    for config_index, (_, M) in enumerate(_certificate_configs(A_scaled)):
        A_ub = _row_slice(M, nonneg_rows)
        A_eq = _row_slice(M, zero_rows)
        attempts = methods
        if config_index == 0:
            attempts = methods + (("highs", {"presolve": True, "disp": False}),)
        for method, options in attempts:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            res = _run_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                               bounds=(-1.0, 1.0), method=method,
                               options=options, deadline=deadline)
            if res is None or res.status != 0 or res.x is None:
                continue
            x = np.asarray(res.x, dtype=np.float64).reshape(-1)
            scale = _inf_norm(x)
            if scale <= 0.0 or not np.isfinite(scale):
                continue
            x /= scale
            # Build s/a directly from the checker's A/a, clipping only tiny
            # numerical violations to make cone membership exact.
            with np.errstate(over="ignore", invalid="ignore"):
                t = -(A_scaled @ x)
                if nonneg_rows.size:
                    t[nonneg_rows] = np.maximum(t[nonneg_rows], 0.0)
                if zero_rows.size:
                    t[zero_rows] = 0.0
                s = t * a
            if not _all_finite(t, s):
                continue
            if _check_dual_cert(A_scaled, q, blocks, a, x, s):
                return {"status": "dual_infeasible", "x": x, "s": s}
    return None


def _primal_certificate_polish(A_scaled, b, blocks, nonneg_rows, deadline):
    """Last resort: directly minimize the nullspace residual for a fixed beta.

    This is the same Farkas cone with an extra variable t, constraints
    |(A/a).T z| <= t and -(b/|b|).T z >= beta_target.  It can recover a
    certificate when the exact-equality LP leaves a residual that is small
    relative to a comfortably positive beta.
    """
    b_scale = _inf_norm(b)
    m, n = A_scaled.shape
    if b_scale <= 0.0 or m == 0:
        return None
    b_hat = b / b_scale
    MT = A_scaled.T.tocsr()
    ones = sparse.csr_matrix(-np.ones((n, 1)))
    A_ub = sparse.hstack([
        sparse.vstack([MT, -MT,
                       sparse.csr_matrix(b_hat.reshape(1, -1))], format="csr"),
        sparse.vstack([ones, ones, sparse.csr_matrix((1, 1))], format="csr"),
    ], format="csc")
    lower = -np.ones(m + 1)
    lower[nonneg_rows] = 0.0
    lower[-1] = 0.0
    upper = np.ones(m + 1)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    objective = np.zeros(m + 1)
    objective[-1] = 1.0
    methods = (
        ("highs", _primary_options(1e-10)),
        ("highs-ds", _primary_options(1e-10, method="highs-ds")),
    )
    for beta_target in (2e-8, 1e-6, 1e-4, 1e-2, 0.1):
        b_ub = np.zeros(2 * n + 1)
        b_ub[-1] = -beta_target
        infeasible = 0
        for method, options in methods:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            res = _run_linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                               method=method, options=options,
                               deadline=deadline)
            if res is None:
                continue
            if res.status == 2:
                infeasible += 1
                continue
            if res.status != 0 or res.x is None:
                continue
            z = np.asarray(res.x[:m], dtype=np.float64).reshape(-1).copy()
            if nonneg_rows.size:
                z[nonneg_rows] = np.maximum(z[nonneg_rows], 0.0)
            scale = _inf_norm(z)
            if scale <= 0.0 or not np.isfinite(scale):
                continue
            z /= scale
            if _check_primal_cert(A_scaled, b, blocks, z):
                return {"status": "primal_infeasible", "z": z}
        if infeasible == len(methods):
            # A larger beta target cannot become feasible.
            break
    return None


def _dual_certificate_polish(A_scaled, q, blocks, a, nonneg_rows,
                             zero_rows, deadline):
    """Last resort: directly minimize the recession-cone residual.

    An extra variable t bounds every (A/a) row violation in the checker's
    coordinates, while q/|q|.T x is constrained to be sufficiently negative.
    """
    q_scale = _inf_norm(q)
    if q_scale <= 0.0 or q.size == 0:
        return None
    n = q.size
    q_hat = q / q_scale
    A_n = _row_slice(A_scaled, nonneg_rows)
    A_z = _row_slice(A_scaled, zero_rows)
    row_blocks = []
    if A_n is not None:
        row_blocks.append(sparse.hstack(
            [A_n, sparse.csr_matrix(-np.ones((nonneg_rows.size, 1)))],
            format="csc"))
    if A_z is not None:
        row_blocks.append(sparse.hstack(
            [A_z, sparse.csr_matrix(-np.ones((zero_rows.size, 1)))],
            format="csc"))
        row_blocks.append(sparse.hstack(
            [-A_z, sparse.csr_matrix(-np.ones((zero_rows.size, 1)))],
            format="csc"))
    row_blocks.append(sparse.hstack(
        [sparse.csr_matrix(q_hat.reshape(1, -1)), sparse.csr_matrix((1, 1))],
        format="csc"))
    A_ub = sparse.vstack(row_blocks, format="csc")
    n_rows = nonneg_rows.size + 2 * zero_rows.size + 1
    lower = -np.ones(n + 1)
    lower[-1] = 0.0
    upper = np.ones(n + 1)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    objective = np.zeros(n + 1)
    objective[-1] = 1.0
    methods = (
        ("highs", _primary_options(1e-10)),
        ("highs-ds", _primary_options(1e-10, method="highs-ds")),
    )
    for alpha_target in (1e-4, 1e-2, 0.1, 1.0):
        b_ub = np.zeros(n_rows)
        b_ub[-1] = -alpha_target
        infeasible = 0
        for method, options in methods:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            res = _run_linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                               method=method, options=options,
                               deadline=deadline)
            if res is None:
                continue
            if res.status == 2:
                infeasible += 1
                continue
            if res.status != 0 or res.x is None:
                continue
            x = np.asarray(res.x[:n], dtype=np.float64).reshape(-1).copy()
            scale = _inf_norm(x)
            if scale <= 0.0 or not np.isfinite(scale):
                continue
            x /= scale
            with np.errstate(over="ignore", invalid="ignore"):
                t = -(A_scaled @ x)
                if nonneg_rows.size:
                    t[nonneg_rows] = np.maximum(t[nonneg_rows], 0.0)
                if zero_rows.size:
                    t[zero_rows] = 0.0
                s = t * a
            if not _all_finite(t, s):
                continue
            if _check_dual_cert(A_scaled, q, blocks, a, x, s):
                return {"status": "dual_infeasible", "x": x, "s": s}
        if infeasible == len(methods):
            break
    return None


def _solve_zero_variables(A, q, b, blocks, A_scaled, a, nonneg_rows):
    """Handle n = 0, where scipy.optimize.linprog rejects an empty c."""
    x = np.empty(0, dtype=np.float64)
    s = b.copy()
    z = np.zeros(b.size, dtype=np.float64)
    if _check_solved(A, q, b, blocks, x, s, z):
        return {"status": "solved", "x": x, "s": s, "z": z}
    # Farkas certificate: with no columns, (A/a).T z is the empty vector.
    b_scale = _inf_norm(b)
    if b_scale > 0.0:
        z = np.zeros(b.size, dtype=np.float64)
        for kind, start, width in blocks:
            block = b[start:start + width]
            if kind == "zero":
                z[start:start + width] = -np.sign(block)
            else:
                z[start:start + width] = (block < 0.0).astype(np.float64)
        if _check_primal_cert(A_scaled, b, blocks, z):
            return {"status": "primal_infeasible", "z": z}
    return None


def solve_lp(P, q, A, b, cones, objective_offset=0.0, time_left=None):
    """Solve an LP cone program, or return None if no result can be verified.

    Parameters are the original problem data.  ``objective_offset`` does not
    influence the vectors accepted by the checker and is accepted for API
    compatibility.  ``time_left``, when finite, caps the internal HiGHS work.
    """
    try:
        # Only the zero-Hessian, zero/nonnegative-cone specialization.
        if not sparse.issparse(P) or not sparse.issparse(A):
            return None
        if P.nnz and np.any(P.data != 0.0):
            return None
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        A = A.tocsc(copy=True).astype(np.float64)
        if A.shape != (b.size, q.size):
            return None
        A.sum_duplicates()
        A.eliminate_zeros()
        A.sort_indices()
        if not _all_finite(q, b, A.data, P.data):
            return None
        blocks, zero_rows, nonneg_rows = _row_blocks(cones, b.size)
        if blocks is None:
            return None
        a = _inf_norm(A.data) or 1.0
        A_scaled = _scaled_A(A, a)

        if q.size == 0:
            return _solve_zero_variables(A, q, b, blocks, A_scaled, a,
                                         nonneg_rows)

        start = time.perf_counter()
        deadline = None
        if time_left is not None:
            remaining = float(time_left)
            if np.isfinite(remaining):
                if remaining <= 0.0:
                    return None
                # Preserve time for the caller's other solver paths.
                deadline = start + 0.70 * remaining
        A_n = _row_slice(A, nonneg_rows)
        A_z = _row_slice(A, zero_rows)
        b_n = b[nonneg_rows]
        b_z = b[zero_rows]

        # Row scaling is important when coefficients differ by many orders of
        # magnitude: HiGHS's absolute slack tolerance is then applied to the
        # normalized rows rather than to the checker's original coordinates.
        scaled_data = _row_scaling(A, b, q)
        scaled_blocks = None
        row_factor = None
        if scaled_data is not None:
            A_s, b_s, q_s, r, q_obj_scale = scaled_data
            if _all_finite(r):
                row_factor = r
            with np.errstate(over="ignore", invalid="ignore"):
                dual_scale = r * q_obj_scale
            if _all_finite(dual_scale):
                scaled_blocks = (q_s, _row_slice(A_s, nonneg_rows), b_s[nonneg_rows],
                                 _row_slice(A_s, zero_rows), b_s[zero_rows],
                                 dual_scale)

        # Several solver settings, ordered from most accurate to most robust.
        # The first successful, independently verified result wins.
        methods = (
            ("highs", _primary_options(1e-9)),
            ("highs-ds", _primary_options(1e-10, method="highs-ds")),
            ("highs-ipm", _primary_options(1e-10, method="highs-ipm")),
            ("highs", _primary_options(1e-7, presolve=False)),
        )
        primary_attempts = []
        for method, options in methods:
            if scaled_blocks is not None:
                primary_attempts.append((method, options, True))
            primary_attempts.append((method, options, False))
        saw_unbounded = False
        saw_infeasible = False
        for method, options, use_scaled in primary_attempts:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if use_scaled:
                c_use, A_ub_use, b_ub_use, A_eq_use, b_eq_use, ds = scaled_blocks
            else:
                c_use, A_ub_use, b_ub_use, A_eq_use, b_eq_use = (
                    q, A_n, b_n, A_z, b_z)
                ds = None
            res = _run_linprog(c_use, A_ub=A_ub_use, b_ub=b_ub_use,
                               A_eq=A_eq_use, b_eq=b_eq_use,
                               bounds=(None, None), method=method,
                               options=options, deadline=deadline)
            if res is None:
                continue
            if res.status == 0:
                try:
                    marginal_n = np.asarray(getattr(getattr(res, "ineqlin", None),
                                    "marginals", np.empty(0)), dtype=np.float64).reshape(-1)
                    marginal_z = np.asarray(getattr(getattr(res, "eqlin", None),
                                    "marginals", np.empty(0)), dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if (marginal_n.size != nonneg_rows.size
                        or marginal_z.size != zero_rows.size
                        or not _all_finite(marginal_n, marginal_z)):
                    continue
                result = _build_solved(A, q, b, blocks, zero_rows,
                                       nonneg_rows, res.x, marginal_n,
                                       marginal_z, ds, row_factor)
                if result is not None:
                    return result
            elif res.status == 2:
                saw_infeasible = True
            elif res.status == 3:
                saw_unbounded = True

        # A status-2 problem gets a Farkas certificate; status-3 gets a
        # certified decreasing direction.  On numerical failure (status 4),
        # try both, always subject to the exact certificate checks.
        if saw_infeasible and not saw_unbounded:
            cert_attempts = ["primal", "dual"]
        elif saw_unbounded and not saw_infeasible:
            cert_attempts = ["dual", "primal"]
        else:
            # Both flags, numerical failure, or iteration/time limit: try the
            # Farkas certificate first and then the decreasing direction.
            cert_attempts = ["primal", "dual"]
        for kind in cert_attempts:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if kind == "primal":
                result = _primal_certificate(A_scaled, b, blocks,
                                             nonneg_rows, deadline)
                if result is None:
                    result = _primal_certificate_polish(
                        A_scaled, b, blocks, nonneg_rows, deadline)
            else:
                result = _dual_certificate(A_scaled, q, blocks, a,
                                           nonneg_rows, zero_rows, deadline)
                if result is None:
                    result = _dual_certificate_polish(
                        A_scaled, q, blocks, a, nonneg_rows, zero_rows,
                        deadline)
            if result is not None:
                return result
        return None
    except Exception:
        return None
