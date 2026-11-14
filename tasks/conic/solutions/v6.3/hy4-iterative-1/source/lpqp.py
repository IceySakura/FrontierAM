"""HiGHS-based LP/QP solver for problems whose cones are only zero/nonnegative.

Only numpy is imported at module import time.  The SciPy HiGHS extension is
located and loaded on demand by :func:`load_highs`; SciPy itself is imported
only as a fallback (and lazily for large sparse linear algebra).

The public entry point is :func:`solve_lpqp`, which accepts raw CSC arrays and
returns a dictionary with a status and finite, correctly sized x/s/z vectors.
"""

import numpy as np

_CORE = None
_NOW = None
_TOL = 1e-8
_SAFE_TOL = 0.95e-8
_INF = float("inf")


def load_highs():
    """Load and cache the bundled HiGHS extension under the module name _core."""
    global _CORE
    if _CORE is not None:
        return _CORE
    import importlib.util
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here]
    for entry in sys.path:
        if entry:
            try:
                roots.append(os.path.abspath(entry))
            except Exception:
                pass
    seen = set()
    for root in roots:
        dirs = (root, os.path.join(root, "scipy", "optimize", "_highspy"))
        for directory in dirs:
            try:
                names = sorted(os.listdir(directory))
            except (OSError, TypeError, ValueError):
                continue
            for name in names:
                if not name.startswith("_core"):
                    continue
                if not (name.endswith(".so") or name.endswith(".pyd")):
                    continue
                path = os.path.join(directory, name)
                if path in seen:
                    continue
                seen.add(path)
                previous = sys.modules.get("_core")
                try:
                    spec = importlib.util.spec_from_file_location("_core", path)
                    if spec is None or spec.loader is None:
                        raise RuntimeError("cannot create extension spec")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["_core"] = module
                    spec.loader.exec_module(module)
                    if hasattr(module, "HighsLp") and hasattr(module, "_Highs"):
                        _CORE = module
                        return module
                    raise RuntimeError("not a HiGHS _core module")
                except Exception:
                    if previous is None:
                        sys.modules.pop("_core", None)
                    else:
                        sys.modules["_core"] = previous
                    continue
    try:
        from scipy.optimize._highspy import _core
    except Exception as exc:
        raise RuntimeError("HiGHS extension is unavailable") from exc
    _CORE = _core
    return _core


def _now():
    global _NOW
    if _NOW is None:
        import time
        _NOW = time.perf_counter
    return _NOW()


def _norm(v):
    return float(np.max(np.abs(v), initial=0.0))


class _CSC:
    __slots__ = ("data", "indices", "indptr", "nrow", "ncol", "nnz", "_cols")

    def __init__(self, data, indices, indptr, nrow, ncol):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int32)
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.nnz = int(self.data.size)
        self._cols = None

    def cols(self):
        if self._cols is None:
            counts = np.diff(self.indptr.astype(np.int64))
            self._cols = np.repeat(np.arange(self.ncol, dtype=np.int64), counts)
        return self._cols

    def matvec(self, x):
        if self.nnz == 0 or self.nrow == 0:
            return np.zeros(self.nrow, dtype=np.float64)
        cols = self.cols()
        return np.bincount(self.indices, weights=self.data * x[cols],
                           minlength=self.nrow).astype(np.float64, copy=False)

    def rmatvec(self, y):
        if self.nnz == 0 or self.ncol == 0:
            return np.zeros(self.ncol, dtype=np.float64)
        cols = self.cols()
        return np.bincount(cols, weights=self.data * y[self.indices],
                           minlength=self.ncol).astype(np.float64, copy=False)


class _Data:
    __slots__ = ("P", "A", "q", "b", "zero", "nonneg", "blocks",
                 "a", "h", "bn", "qn", "n", "m", "has_p", "p_pd")


def _canonical_csc(data, indices, indptr, shape):
    nrow, ncol = int(shape[0]), int(shape[1])
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices).reshape(-1)
    indptr = np.asarray(indptr).reshape(-1)
    if nrow < 0 or ncol < 0:
        raise ValueError("negative matrix dimension")
    if indptr.size != ncol + 1:
        raise ValueError("bad CSC indptr length")
    indptr = indptr.astype(np.int64, copy=False)
    if indptr[0] != 0 or indptr[-1] != data.size:
        raise ValueError("bad CSC indptr endpoints")
    if indices.size != data.size:
        raise ValueError("bad CSC index length")
    if bool(np.any(indptr[1:] < indptr[:-1])):
        raise ValueError("unsorted CSC indptr")
    if not np.isfinite(data).all():
        raise ValueError("nonfinite matrix coefficient")
    if data.size == 0:
        return _CSC(np.empty(0, np.float64), np.empty(0, np.int32),
                    np.zeros(ncol + 1, np.int32), nrow, ncol)
    idx = indices.astype(np.int64, copy=False)
    if int(idx.min(initial=0)) < 0 or int(idx.max(initial=0)) >= nrow:
        raise ValueError("CSC row index out of range")
    need_sort = False
    cols = None
    if data.size > 1:
        cols = np.repeat(np.arange(ncol, dtype=np.int64), np.diff(indptr))
        same_column = cols[1:] == cols[:-1]
        need_sort = bool(np.any((idx[1:] <= idx[:-1]) & same_column))
    if need_sort:
        if nrow == 0:
            raise ValueError("nonzero entries in zero-row matrix")
        if cols is None:
            cols = np.repeat(np.arange(ncol, dtype=np.int64), np.diff(indptr))
        keys = cols * nrow + idx
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_values = data[order]
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        values = np.add.reduceat(sorted_values, starts)
        new_indptr = np.searchsorted(unique_keys,
                                     np.arange(ncol + 1, dtype=np.int64) * nrow)
        new_indices = (unique_keys % nrow).astype(np.int32)
        return _CSC(values, new_indices, new_indptr, nrow, ncol)
    return _CSC(data, idx, indptr, nrow, ncol)


def _parse_cones(cones, m):
    zero = np.zeros(m, dtype=bool)
    nonneg = np.zeros(m, dtype=bool)
    blocks = []
    start = 0
    for cone in cones:
        kind = cone["type"]
        dim = int(cone["dim"])
        if dim <= 0 or start + dim > m:
            raise ValueError("invalid cone partition")
        if kind == "zero":
            zero[start:start + dim] = True
        elif kind == "nonnegative":
            nonneg[start:start + dim] = True
        else:
            raise ValueError("unsupported cone type")
        blocks.append((kind, start, start + dim))
        start += dim
    if start != m:
        raise ValueError("cone dimensions do not partition rows")
    return zero, nonneg, blocks


def _lower_hessian(P):
    """Column-wise lower triangle CSC of the full symmetric P."""
    n = P.nrow
    if n == 0 or P.nnz == 0:
        return (np.empty(0, np.float64), np.empty(0, np.int32),
                np.zeros(n + 1, np.int32))
    rows = P.indices.astype(np.int64)
    cols = P.cols()
    values = P.data
    lower = rows >= cols
    upper = ~lower
    low_rows = rows[lower]
    low_cols = cols[lower]
    low_values = values[lower]
    low_keys = low_cols * n + low_rows
    # P is column-wise with sorted row indices, hence low_keys is sorted.
    low_indptr = np.searchsorted(low_keys,
                                 np.arange(n + 1, dtype=np.int64) * n)
    if upper.any():
        up_rows = rows[upper]
        up_cols = cols[upper]
        # Mirror of an upper entry (r,c), r<c, is (row=c, col=r).
        mirror_keys = up_rows * n + up_cols
        pos = np.searchsorted(low_keys, mirror_keys)
        clipped = np.minimum(pos, low_keys.size - 1) if low_keys.size else pos
        missing = (pos >= low_keys.size)
        if low_keys.size:
            missing |= low_keys[clipped] != mirror_keys
        if missing.any():
            extra_keys = mirror_keys[missing]
            extra_rows = up_cols[missing]
            extra_values = values[upper][missing]
            all_keys = np.concatenate((low_keys, extra_keys))
            all_rows = np.concatenate((low_rows, extra_rows))
            all_values = np.concatenate((low_values, extra_values))
            order = np.argsort(all_keys, kind="stable")
            sorted_keys = all_keys[order]
            indptr = np.searchsorted(sorted_keys,
                                     np.arange(n + 1, dtype=np.int64) * n)
            return (np.ascontiguousarray(all_values[order], dtype=np.float64),
                    np.ascontiguousarray(all_rows[order], dtype=np.int32),
                    np.ascontiguousarray(indptr, dtype=np.int32))
    return (np.ascontiguousarray(low_values, dtype=np.float64),
            np.ascontiguousarray(low_rows, dtype=np.int32),
            np.ascontiguousarray(low_indptr, dtype=np.int32))


def _build_scaling(data, objective_mode=0):
    """Return a positively scaled copy of the data, or None if unnecessary.

    Row scale r, column scale C and objective scale o transform the model as
    A_s = (A C)/r, b_s = b/r, q_s = (q C)/o, P_s = (C P C)/o.  The original
    primal/dual pair is recovered with x = C y, z = (o/r) z_s and
    s = b - A x.  Direct division (rather than multiplying by a reciprocal)
    keeps extreme but finite coefficients from overflowing.
    """
    A, P = data.A, data.P
    if A.nnz > 50_000_000 or P.nnz > 50_000_000:
        return None
    a_raw = _norm(A.data)
    h_raw = _norm(P.data)
    bn, qn = data.bn, data.qn
    a_eff = a_raw if a_raw > 0.0 else 1.0
    limit = 1e200
    cap = 1e100
    try:
        if qn > 0.0:
            ratio = limit / qn
            if np.isfinite(ratio):
                cap = min(cap, ratio)
        if h_raw > 0.0:
            ratio = limit / h_raw
            if np.isfinite(ratio):
                cap = min(cap, ratio ** 0.5)
        if not np.isfinite(cap) or cap < 1.0:
            return None
        if bn > 0.0 and a_eff > 0.0:
            ratio = bn / a_eff
            C = min(cap, max(1.0, ratio)) if np.isfinite(ratio) else cap
        else:
            C = 1.0
        r = max(a_eff * C, bn)
        C2 = C * C
        o = max(qn * C, h_raw * C2)
        if not np.isfinite(r) or r <= 0.0 or not np.isfinite(C2) or C2 <= 0.0:
            return None
        if not np.isfinite(o) or o <= 0.0:
            o = 1.0
        zfac = o / r
        if not np.isfinite(zfac) or zfac <= 0.0 or zfac > 1e200:
            return None
        if (0.5 <= C <= 2.0 and 0.5 <= r <= 2.0 and
                0.5 <= o <= 2.0 and 0.5 <= zfac <= 2.0):
            return None
    except (OverflowError, FloatingPointError, ValueError, ZeroDivisionError):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        A_s_data = (A.data / r) * C
        P_s_data = (P.data / o) * C2
        q_s = (data.q / o) * C
        b_s = data.b / r
    if not (np.isfinite(A_s_data).all() and np.isfinite(P_s_data).all() and
            np.isfinite(q_s).all() and np.isfinite(b_s).all()):
        return None
    if (A.nnz and not np.any(A_s_data != 0.0)) or \
            (P.nnz and not np.any(P_s_data != 0.0)) or \
            (qn > 0.0 and not np.any(q_s != 0.0)) or \
            (bn > 0.0 and not np.any(b_s != 0.0)):
        return None
    scaled = _Data()
    scaled.A = _CSC(A_s_data, A.indices, A.indptr, A.nrow, A.ncol)
    scaled.P = _CSC(P_s_data, P.indices, P.indptr, P.nrow, P.ncol)
    scaled.q = np.ascontiguousarray(q_s, dtype=np.float64)
    scaled.b = np.ascontiguousarray(b_s, dtype=np.float64)
    scaled.zero, scaled.nonneg, scaled.blocks = data.zero, data.nonneg, data.blocks
    scaled.n, scaled.m = data.n, data.m
    scaled.a = _norm(A_s_data) or 1.0
    scaled.h = _norm(P_s_data) or 1.0
    scaled.bn = _norm(b_s)
    scaled.qn = _norm(q_s)
    scaled.has_p = bool(np.any(P_s_data != 0.0))
    scaled.p_pd = data.p_pd
    if scaled.has_p != data.has_p:
        return None
    return scaled, C, r, o


def _build_model_for_data(core, data):
    row_lower = np.where(data.zero, data.b, -_INF).astype(np.float64)
    row_upper = data.b.copy()
    col_lower = np.full(data.n, -_INF, dtype=np.float64)
    col_upper = np.full(data.n, _INF, dtype=np.float64)
    lp = _make_lp(core, data.q, col_lower, col_upper, row_lower, row_upper,
                  data.A.data, data.A.indices, data.A.indptr, data.n, data.m)
    hess_arrays = _lower_hessian(data.P) if data.has_p else None
    return lp, _make_model(core, lp, hess_arrays, data.n)


def _make_lp(core, cost, col_lower, col_upper, row_lower, row_upper,
             data, indices, indptr, ncol, nrow):
    lp = core.HighsLp()
    lp.num_col_ = int(ncol)
    lp.num_row_ = int(nrow)
    lp.col_cost_ = np.ascontiguousarray(cost, dtype=np.float64)
    lp.col_lower_ = np.ascontiguousarray(col_lower, dtype=np.float64)
    lp.col_upper_ = np.ascontiguousarray(col_upper, dtype=np.float64)
    lp.row_lower_ = np.ascontiguousarray(row_lower, dtype=np.float64)
    lp.row_upper_ = np.ascontiguousarray(row_upper, dtype=np.float64)
    lp.sense_ = core.ObjSense.kMinimize
    mat = lp.a_matrix_
    mat.format_ = core.MatrixFormat.kColwise
    mat.num_col_ = int(ncol)
    mat.num_row_ = int(nrow)
    mat.start_ = np.ascontiguousarray(indptr, dtype=np.int32)
    mat.index_ = np.ascontiguousarray(indices, dtype=np.int32)
    mat.value_ = np.ascontiguousarray(data, dtype=np.float64)
    return lp


def _make_model(core, lp, hess_arrays, n):
    if hess_arrays is None:
        return lp
    data, indices, indptr = hess_arrays
    hess = core.HighsHessian()
    hess.dim_ = int(n)
    hess.format_ = core.HessianFormat.kTriangular
    hess.start_ = np.ascontiguousarray(indptr, dtype=np.int32)
    hess.index_ = np.ascontiguousarray(indices, dtype=np.int32)
    hess.value_ = np.ascontiguousarray(data, dtype=np.float64)
    model = core.HighsModel()
    model.lp_ = lp
    model.hessian_ = hess
    return model


def _set_options(core, highs, presolve, time_limit, solver="choose"):
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("threads", 1)
    highs.setOptionValue("presolve", "on" if presolve else "off")
    highs.setOptionValue("solver", solver)
    highs.setOptionValue("primal_feasibility_tolerance", 1e-10)
    highs.setOptionValue("dual_feasibility_tolerance", 1e-10)
    highs.setOptionValue("ipm_optimality_tolerance", 1e-12)
    if time_limit is not None:
        tl = float(time_limit)
        if np.isfinite(tl) and tl > 0:
            highs.setOptionValue("time_limit", max(0.01, tl))


def _run_model(core, model, presolve, time_limit, solver="choose"):
    highs = core._Highs()
    _set_options(core, highs, presolve, time_limit, solver)
    status = highs.passModel(model)
    if status != core.HighsStatus.kOk:
        return highs, core.HighsModelStatus.kModelError
    run_status = highs.run()
    model_status = highs.getModelStatus()
    if (run_status != core.HighsStatus.kOk and
            model_status == core.HighsModelStatus.kNotset):
        model_status = core.HighsModelStatus.kUnknown
    try:
        highs.modelStatusToString(model_status)
    except Exception:
        pass
    return highs, model_status


def _col_values(highs, ncol):
    values = np.asarray(highs.getSolution().col_value, dtype=np.float64).reshape(-1)
    if values.size != ncol or not np.isfinite(values).all():
        raise ValueError("invalid HiGHS column values")
    return values.copy()


def _row_duals(highs, nrow):
    values = np.asarray(highs.getSolution().row_dual, dtype=np.float64).reshape(-1)
    if values.size != nrow or not np.isfinite(values).all():
        raise ValueError("invalid HiGHS row duals")
    return values.copy()


def _raw_solution(highs, data):
    x = _col_values(highs, data.n)
    z = -_row_duals(highs, data.m)
    s = data.b - data.A.matvec(x)
    return x, s, z


def _cone_defect(v, blocks, dual):
    worst = 0.0
    for kind, start, end in blocks:
        block = v[start:end]
        if block.size == 0:
            continue
        scale = max(1.0, _norm(block))
        u = block / scale
        if kind == "zero":
            violation = 0.0 if dual else _norm(u)
        else:
            violation = max(0.0, -float(np.min(u)))
        worst = max(worst, violation)
    return worst


def _validate_solution(data, x, s, z):
    try:
        if not (np.isfinite(x).all() and np.isfinite(s).all() and
                np.isfinite(z).all()):
            return False
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            A = data.A
            P = data.P
            Ax = A.matvec(x)
            Px = P.matvec(x)
            Atz = A.rmatvec(z)
            rp_num = _norm(Ax + s - data.b)
            rp_den = 1.0 + max(_norm(Ax), _norm(s), _norm(data.b))
            rd_num = _norm(Px + data.q + Atz)
            rd_den = 1.0 + max(_norm(Px), _norm(data.q), _norm(Atz))
            rp = rp_num / rp_den
            rd = rd_num / rd_den
            cp = _cone_defect(s, data.blocks, False)
            cz = _cone_defect(z, data.blocks, True)
            xPx = float(x @ Px)
            p0 = 0.5 * xPx + float(data.q @ x)
            d0 = -0.5 * xPx - float(data.b @ z)
            gap = abs(p0 - d0)
            scale = max(abs(p0), abs(d0))
            gap_ok = gap <= _TOL or (scale > 0.0 and gap / scale <= _TOL)
            return bool(np.isfinite(rp) and np.isfinite(rd) and
                        np.isfinite(cp) and np.isfinite(cz) and
                        np.isfinite(gap) and
                        rp <= _SAFE_TOL and rd <= _SAFE_TOL and
                        cp <= _SAFE_TOL and cz <= _SAFE_TOL and gap_ok)
    except (FloatingPointError, ValueError, TypeError, IndexError):
        return False


def _validate_primal_cert(data, z):
    try:
        if data.bn <= 0.0 or not np.isfinite(z).all():
            return False
        zn = _norm(z)
        if zn <= 0.0:
            return False
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            zhat = z / zn
            margin = -float((data.b / data.bn) @ zhat)
            if not np.isfinite(margin) or margin <= 1.0001 * _TOL:
                return False
            linear = _norm(data.A.rmatvec(zhat) / data.a) / margin
            cone = _cone_defect(zhat, data.blocks, True) / margin
            return bool(np.isfinite(linear) and np.isfinite(cone) and
                        linear <= _SAFE_TOL and cone <= _SAFE_TOL)
    except (FloatingPointError, ValueError, TypeError, IndexError):
        return False


def _validate_dual_cert(data, x, s):
    try:
        if data.qn <= 0.0 or not (np.isfinite(x).all() and np.isfinite(s).all()):
            return False
        xn = _norm(x)
        if xn <= 0.0:
            return False
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            xhat = x / xn
            shat = s / xn
            margin = -float((data.q / data.qn) @ xhat)
            if not np.isfinite(margin) or margin <= 1.0001 * _TOL:
                return False
            quadratic = _norm(data.P.matvec(xhat) / data.h) / margin
            linear = _norm(data.A.matvec(xhat) / data.a + shat / data.a) / margin
            cone = _cone_defect(shat / data.a, data.blocks, False) / margin
            return bool(np.isfinite(quadratic) and np.isfinite(linear) and
                        np.isfinite(cone) and quadratic <= _SAFE_TOL and
                        linear <= _SAFE_TOL and cone <= _SAFE_TOL)
    except (FloatingPointError, ValueError, TypeError, IndexError):
        return False


def _clip_small(v, mask, scale):
    out = v.copy()
    if out.size and mask.any():
        limit = 1e-9 * scale
        idx = mask & (out < 0.0) & (out > -limit)
        out[idx] = 0.0
    return out


def _prepare_solution(data, x, s, z):
    """Enforce exact zero-cone slack and clip tiny cone violations."""
    s = s.copy()
    s[data.zero] = 0.0
    s = _clip_small(s, data.nonneg,
                    max(1.0, _norm(s), _norm(data.b)))
    z = _clip_small(z, data.nonneg,
                    max(1.0, _norm(z), _norm(data.q)))
    return x, s, z


def _dense_matrix(C):
    if C.nnz == 0 or C.nrow == 0 or C.ncol == 0:
        return np.zeros((C.nrow, C.ncol), dtype=np.float64)
    rows = C.indices.astype(np.int64)
    cols = C.cols()
    keys = rows * C.ncol + cols
    flat = np.bincount(keys, weights=C.data, minlength=C.nrow * C.ncol)
    return flat.reshape(C.nrow, C.ncol)


def _p_positive_definite(P, deadline):
    """Cheap conservative test for a positive-definite Hessian."""
    n = P.nrow
    if not P.nnz or n == 0 or n > 2000 or n * n > 4_000_000:
        return False
    if _norm(P.data) > 1e150:
        return False
    if deadline - _now() < 0.05:
        return False
    try:
        dense = _dense_matrix(P)
        dense = 0.5 * (dense + dense.T)
        diagonal = np.diag(dense).copy()
        if np.all(diagonal > 0.0):
            off_diagonal = dense.copy()
            np.fill_diagonal(off_diagonal, 0.0)
            if not np.any(off_diagonal):
                return True
        np.linalg.cholesky(dense)
        return True
    except (np.linalg.LinAlgError, ValueError, MemoryError, FloatingPointError):
        return False


def _dense_active(A, active_rows, k):
    n = A.ncol
    if k == 0 or A.nnz == 0:
        return np.zeros((k, n), dtype=np.float64)
    pos = np.full(A.nrow, -1, dtype=np.int64)
    pos[active_rows] = np.arange(k, dtype=np.int64)
    mapped = pos[A.indices]
    mask = mapped >= 0
    if not mask.any():
        return np.zeros((k, n), dtype=np.float64)
    rows = mapped[mask]
    cols = A.cols()[mask]
    keys = rows * n + cols
    flat = np.bincount(keys, weights=A.data[mask], minlength=k * n)
    return flat.reshape(k, n)


def _dense_kkt_solve(K, rhs, n, N):
    best = None
    best_res = _INF
    try:
        sol = np.linalg.solve(K, rhs)
        if np.isfinite(sol).all():
            res = rhs - K @ sol
            rnorm = _norm(res)
            if rnorm < 1e-9 * (1.0 + _norm(rhs)):
                for _ in range(2):
                    if rnorm <= 1e-14 * (1.0 + _norm(rhs)):
                        break
                    sol = sol + np.linalg.solve(K, res)
                    res = rhs - K @ sol
                    rnorm = _norm(res)
                return sol
            if rnorm < best_res:
                best, best_res = sol.copy(), rnorm
    except np.linalg.LinAlgError:
        pass
    scale = max(1.0, _norm(K))
    diag = np.arange(N, dtype=np.int64)
    for factor in (1e-12, 1e-10, 1e-8):
        eps = factor * scale
        Kreg = K.copy()
        Kreg[diag[:n], diag[:n]] += eps
        Kreg[diag[n:], diag[n:]] -= eps
        try:
            sol = np.linalg.solve(Kreg, rhs)
            for _ in range(4):
                res = rhs - K @ sol
                rnorm = _norm(res)
                if rnorm <= 1e-13 * (1.0 + _norm(rhs)):
                    break
                sol = sol + np.linalg.solve(Kreg, res)
            if not np.isfinite(sol).all():
                continue
            res = rhs - K @ sol
            rnorm = _norm(res)
            if rnorm < best_res:
                best, best_res = sol.copy(), rnorm
            if rnorm <= 1e-9 * (1.0 + _norm(rhs)):
                return sol
        except np.linalg.LinAlgError:
            continue
    return best


def _sparse_kkt_solve(A, P, q, b, active_rows, deadline):
    if _now() >= deadline - 0.25:
        return None
    try:
        import warnings
        from scipy import sparse
        from scipy.sparse.linalg import spsolve, lsqr
        n = A.ncol
        k = active_rows.size
        P_sp = sparse.csc_matrix((P.data, P.indices, P.indptr), shape=(P.nrow, P.ncol))
        A_sp = sparse.csc_matrix((A.data, A.indices, A.indptr), shape=(A.nrow, A.ncol))
        A_act = A_sp[active_rows, :].tocsc()
        K = sparse.bmat([[P_sp, A_act.T], [A_act, None]], format="csc")
        rhs = np.concatenate((-q, b[active_rows]))
        sol = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                candidate = np.asarray(spsolve(K, rhs), dtype=np.float64).reshape(-1)
                if candidate.size == n + k and np.isfinite(candidate).all():
                    sol = candidate
            except Exception:
                sol = None
            if sol is None or _norm(rhs - K @ sol) > 1e-9 * (1.0 + _norm(rhs)):
                try:
                    result = lsqr(K, rhs, atol=1e-13, btol=1e-13,
                                  iter_lim=min(5000, 20 * (n + k)))
                    candidate = np.asarray(result[0], dtype=np.float64).reshape(-1)
                    if candidate.size == n + k and np.isfinite(candidate).all():
                        sol = candidate
                except Exception:
                    return None
        return sol
    except Exception:
        return None


def _solve_kkt(A, P, q, b, active_rows, deadline, dense_limit):
    n = A.ncol
    k = active_rows.size
    N = n + k
    if N == 0:
        return None
    use_dense = (N <= dense_limit and n * n <= 6_000_000 and
                 k * n <= 6_000_000 and n <= 2500)
    if use_dense:
        Pd = _dense_matrix(P)
        Ad = _dense_active(A, active_rows, k)
        K = np.zeros((N, N), dtype=np.float64)
        K[:n, :n] = Pd
        K[:n, n:] = Ad.T
        K[n:, :n] = Ad
        rhs = np.concatenate((-q, b[active_rows]))
        return _dense_kkt_solve(K, rhs, n, N)
    if N > 200000 or _now() >= deadline - 0.3:
        return None
    return _sparse_kkt_solve(A, P, q, b, active_rows, deadline)


def _polished_candidates(data, x0, z0, s0, deadline, dense_limit):
    A, P, q, b = data.A, data.P, data.q, data.b
    n, m = data.n, data.m
    if n > 200000 or m > 200000:
        return
    zscale = max(1.0, _norm(z0))
    sscale = np.maximum(1.0, np.abs(b))
    dual_active = z0 >= 1e-7 * zscale
    for threshold in (1e-9, 1e-7, 1e-6, 1e-5, 1e-4):
        if _now() >= deadline - 0.015:
            return
        active = data.zero | (data.nonneg &
                              ((s0 <= threshold * sscale) | dual_active))
        rows = np.flatnonzero(active)
        k = rows.size
        if k == 0 and not data.has_p:
            # Unconstrained LP with zero objective: any feasible x works;
            # keep the HiGHS point rather than solving a singular KKT system.
            continue
        if k * n > 6_000_000 and n + k > dense_limit:
            continue
        sol = _solve_kkt(A, P, q, b, rows, deadline, dense_limit)
        if sol is None or sol.size != n + k or not np.isfinite(sol).all():
            continue
        x = sol[:n]
        z = np.zeros(m, dtype=np.float64)
        z[rows] = sol[n:]
        if np.any(z[data.nonneg] < -1e-9 * max(1.0, _norm(z))):
            continue
        s = b - A.matvec(x)
        if np.any(s[data.nonneg & ~active] < -1e-9 * max(1.0, _norm(s))):
            continue
        x, s, z = _prepare_solution(data, x, s, z)
        yield x, s, z


def _transpose_csc(A, divisor):
    m, n = A.nrow, A.ncol
    if A.nnz == 0 or m == 0 or n == 0:
        return (np.empty(0, np.float64), np.empty(0, np.int32),
                np.zeros(m + 1, np.int32))
    rows = A.indices.astype(np.int64)
    cols = A.cols()
    keys = rows * n + cols  # column of A^T is the original row.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    values = A.data[order] / divisor
    indptr = np.searchsorted(sorted_keys,
                             np.arange(m + 1, dtype=np.int64) * n)
    indices = (sorted_keys % n).astype(np.int32)
    return (np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(indices, dtype=np.int32),
            np.ascontiguousarray(indptr, dtype=np.int32))


def _dense_At(A, divisor, max_entries):
    m, n = A.nrow, A.ncol
    if m * n == 0 or m * n > max_entries:
        return None
    At = np.zeros((n, m), dtype=np.float64)
    if A.nnz:
        rows = A.indices.astype(np.int64)
        cols = A.cols()
        keys = cols * m + rows
        flat = np.bincount(keys, weights=A.data / divisor, minlength=n * m)
        At = flat.reshape(n, m)
    return At


def _project_primal_cert(data, z, deadline):
    rem = deadline - _now()
    if rem < 0.02:
        return None
    m, n = data.A.nrow, data.A.ncol
    cap = 1_000_000 if rem > 0.5 else 250_000
    if m * n > cap or m > 2000 or n > 2000:
        return None
    At = _dense_At(data.A, data.a, cap)
    if At is None:
        return None
    z = z.copy()
    for _ in range(4):
        if _now() >= deadline - 0.01:
            break
        residual = At @ z
        if _norm(residual) <= 1e-14:
            break
        try:
            delta = np.linalg.lstsq(At, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        z = z + delta
        z[data.nonneg & (z < 0.0)] = 0.0
        zn = _norm(z)
        if zn <= 0.0 or not np.isfinite(z).all():
            return None
        z = z / zn
        if _validate_primal_cert(data, z):
            return z
    return z if _validate_primal_cert(data, z) else None


def _stack_constraints(A, P, divisor_a, divisor_p, has_p):
    """CSC of [A/a; P/h] for the recession-direction LP (all A rows included)."""
    m, n = A.nrow, A.ncol
    if has_p and P.nnz:
        counts = np.bincount(P.indices, minlength=n)
        p_rows = np.flatnonzero(counts > 0)
        p_map = np.full(n, -1, dtype=np.int64)
        p_map[p_rows] = np.arange(p_rows.size, dtype=np.int64)
        rP = int(p_rows.size)
    else:
        p_map = np.full(n, -1, dtype=np.int64)
        rP = 0
    nrows = m + rP
    if nrows == 0 or n == 0:
        return (np.empty(0, np.float64), np.empty(0, np.int32),
                np.zeros(n + 1, np.int32), 0)
    row_parts = []
    col_parts = []
    val_parts = []
    if A.nnz:
        row_parts.append(A.indices.astype(np.int64))
        col_parts.append(A.cols())
        val_parts.append(A.data / divisor_a)
    if P.nnz and rP:
        rows = P.indices.astype(np.int64)
        row_parts.append(m + p_map[rows])
        col_parts.append(P.cols())
        val_parts.append(P.data / divisor_p)
    if not row_parts:
        return (np.empty(0, np.float64), np.empty(0, np.int32),
                np.zeros(n + 1, np.int32), 0)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    vals = np.concatenate(val_parts)
    keys = cols * nrows + rows
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    indptr = np.searchsorted(sorted_keys,
                             np.arange(n + 1, dtype=np.int64) * nrows)
    return (np.ascontiguousarray(vals[order], dtype=np.float64),
            np.ascontiguousarray(rows[order], dtype=np.int32),
            np.ascontiguousarray(indptr, dtype=np.int32), nrows)


def _dense_equality(data, active_nonneg, max_entries):
    A, P = data.A, data.P
    m, n = A.nrow, A.ncol
    selected = data.zero | active_nonneg
    a_map = np.full(m, -1, dtype=np.int64)
    a_map[selected] = np.arange(int(selected.sum()), dtype=np.int64)
    rA = int(selected.sum())
    if data.has_p and P.nnz:
        counts = np.bincount(P.indices, minlength=n)
        p_rows = np.flatnonzero(counts > 0)
        p_map = np.full(n, -1, dtype=np.int64)
        p_map[p_rows] = np.arange(p_rows.size, dtype=np.int64)
        rP = int(p_rows.size)
    else:
        p_rows = np.empty(0, np.int64)
        p_map = np.full(n, -1, dtype=np.int64)
        rP = 0
    r = rA + rP
    if r == 0 or r * n > max_entries or r > 2500 or n > 2500:
        return None
    E = np.zeros((r, n), dtype=np.float64)
    if A.nnz and rA:
        rows = A.indices.astype(np.int64)
        mapped = a_map[rows]
        mask = mapped >= 0
        if mask.any():
            keys = mapped[mask] * n + A.cols()[mask]
            flat = np.bincount(keys, weights=A.data[mask] / data.a,
                               minlength=rA * n)
            E[:rA] = flat.reshape(rA, n)
    if P.nnz and rP:
        rows = P.indices.astype(np.int64)
        keys = p_map[rows] * n + P.cols()
        flat = np.bincount(keys, weights=P.data / data.h, minlength=rP * n)
        E[rA:] = flat.reshape(rP, n)
    return E


def _project_dual_cert(data, x, deadline):
    rem = deadline - _now()
    if rem < 0.02:
        return None
    A = data.A
    Ax = A.matvec(x)
    s0 = -Ax
    active_nonneg = data.nonneg & (s0 <= 1e-6 * np.maximum(1.0, np.abs(Ax)))
    cap = 1_000_000 if rem > 0.5 else 250_000
    E = _dense_equality(data, active_nonneg, cap)
    if E is None:
        return None
    x = x.copy()
    for _ in range(4):
        if _now() >= deadline - 0.01:
            break
        residual = E @ x
        if _norm(residual) <= 1e-14:
            break
        try:
            delta = np.linalg.lstsq(E, -residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x = x + delta
        xn = _norm(x)
        if xn <= 0.0 or not np.isfinite(x).all():
            return None
        x = x / xn
        s = -A.matvec(x)
        s[data.zero] = 0.0
        s[data.nonneg & (s < 0.0)] = 0.0
        if _validate_dual_cert(data, x, s):
            return x, s
    s = -A.matvec(x)
    s[data.zero] = 0.0
    s[data.nonneg & (s < 0.0)] = 0.0
    return (x, s) if _validate_dual_cert(data, x, s) else None


def _aux_too_big(data, deadline, include_p):
    """Bail out only when an auxiliary LP would clearly blow the time budget."""
    if data.n > 2_000_000 or data.m > 2_000_000:
        return True
    nnz = data.A.nnz + (data.P.nnz if include_p else 0)
    rem = deadline - _now()
    if nnz > 50_000_000:
        return True
    if (data.n > 200000 or data.m > 200000) and nnz > 200000 and rem < 2.0:
        return True
    if nnz > 5_000_000 and rem < 5.0:
        return True
    return False


def _try_primal_certificate(core, data, deadline):
    if data.bn <= 0.0 or _aux_too_big(data, deadline, False):
        return None
    if _now() >= deadline - 0.02:
        return None
    A, b = data.A, data.b
    trans = _transpose_csc(A, data.a)
    cost = b / data.bn
    col_lower = np.where(data.nonneg, 0.0, -1.0).astype(np.float64)
    col_upper = np.ones(data.m, dtype=np.float64)
    row_lower = np.zeros(data.n, dtype=np.float64)
    row_upper = np.zeros(data.n, dtype=np.float64)
    lp = _make_lp(core, cost, col_lower, col_upper, row_lower, row_upper,
                  trans[0], trans[1], trans[2], data.m, data.n)
    solver = "simplex" if max(data.m, data.n) <= 5000 else "choose"
    for presolve in (True, False):
        rem = deadline - _now()
        if rem < 0.02:
            break
        tl = max(0.01, min(0.55 * rem, 20.0))
        highs, status = _run_model(core, lp, presolve, tl, solver)
        if status not in (core.HighsModelStatus.kOptimal,
                          core.HighsModelStatus.kTimeLimit,
                          core.HighsModelStatus.kIterationLimit):
            continue
        try:
            z = _col_values(highs, data.m)
        except ValueError:
            continue
        if float(cost @ z) >= -1e-12:
            continue
        z[data.nonneg & (z < 0.0)] = 0.0
        zn = _norm(z)
        if zn <= 0.0:
            continue
        z = z / zn
        if _validate_primal_cert(data, z):
            return z
        polished = _project_primal_cert(data, z, deadline)
        if polished is not None:
            return polished
    return None


def _try_dual_certificate(core, data, deadline):
    if data.qn <= 0.0 or _aux_too_big(data, deadline, True):
        return None
    if _now() >= deadline - 0.02:
        return None
    A, P, q = data.A, data.P, data.q
    cost = q / data.qn
    # The LP uses all A rows; the projection polish treats only active
    # inequalities as equalities.
    stacked = _stack_constraints(A, P, data.a, data.h, data.has_p)
    c_data, c_indices, c_indptr, nrows = stacked
    row_lower = np.concatenate((np.where(data.zero, 0.0, -_INF),
                                np.zeros(nrows - data.m, dtype=np.float64)))
    row_upper = np.zeros(nrows, dtype=np.float64)
    col_lower = -np.ones(data.n, dtype=np.float64)
    col_upper = np.ones(data.n, dtype=np.float64)
    lp = _make_lp(core, cost, col_lower, col_upper, row_lower, row_upper,
                  c_data, c_indices, c_indptr, data.n, nrows)
    solver = "simplex" if max(data.n, nrows) <= 5000 else "choose"
    for presolve in (True, False):
        rem = deadline - _now()
        if rem < 0.02:
            break
        tl = max(0.01, min(0.55 * rem, 20.0))
        highs, status = _run_model(core, lp, presolve, tl, solver)
        if status not in (core.HighsModelStatus.kOptimal,
                          core.HighsModelStatus.kTimeLimit,
                          core.HighsModelStatus.kIterationLimit):
            continue
        try:
            x = _col_values(highs, data.n)
        except ValueError:
            continue
        if float(cost @ x) >= -1e-12:
            continue
        xn = _norm(x)
        if xn <= 0.0:
            continue
        x = x / xn
        s = -A.matvec(x)
        s[data.zero] = 0.0
        s[data.nonneg & (s < 0.0)] = 0.0
        if _validate_dual_cert(data, x, s):
            return x, s
        polished = _project_dual_cert(data, x, deadline)
        if polished is not None:
            return polished
    return None


def _try_certificates(core, data, deadline, prefer_primal):
    if prefer_primal:
        order = (_try_primal_certificate, _try_dual_certificate)
    else:
        order = (_try_dual_certificate, _try_primal_certificate)
    for attempt in order:
        if _now() >= deadline - 0.02:
            break
        if attempt is _try_primal_certificate:
            cert = _try_primal_certificate(core, data, deadline)
            if cert is not None:
                return "primal_infeasible", None, None, cert
        else:
            if data.p_pd:
                # A positive-definite Hessian makes a recession direction
                # impossible; avoid wasting time on that auxiliary LP.
                continue
            cert = _try_dual_certificate(core, data, deadline)
            if cert is not None:
                return "dual_infeasible", cert[0], cert[1], None
    return None


def _clean(v, size):
    if v is None:
        return np.zeros(size, dtype=np.float64)
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size != size:
        return np.zeros(size, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _result(status, x, s, z, n, m):
    return {"status": status, "x": _clean(x, n), "s": _clean(s, m),
            "z": _clean(z, m)}


def _unresolved(n, m):
    return _result("unresolved", None, None, None, n, m)


def _map_solution(y, s_model, z_model, original, scale_info):
    if scale_info is None:
        x, s, z = y, s_model.copy(), z_model.copy()
    else:
        C, r, o = scale_info
        with np.errstate(over="ignore", invalid="ignore"):
            x = C * y
            z = (o / r) * z_model
            s = original.b - original.A.matvec(x)
        if not (np.isfinite(x).all() and np.isfinite(s).all() and
                np.isfinite(z).all()):
            return None
    x, s, z = _prepare_solution(original, x, s, z)
    if not (np.isfinite(x).all() and np.isfinite(s).all() and
            np.isfinite(z).all()):
        return None
    return x, s, z


def _solve_model_variant(core, model, model_data, original, scale_info,
                         deadline, main_deadline, dense_limit, budget,
                         solution_states, certificate_states):
    last_status = core.HighsModelStatus.kNotset
    for attempt_index, presolve in enumerate((True, False)):
        remaining = deadline - _now()
        main_remaining = main_deadline - _now()
        if remaining < 0.02 or main_remaining < 0.02:
            break
        main_tl = max(0.01, min(
            0.80 * budget if attempt_index == 0 else 0.85 * main_remaining,
            0.95 * main_remaining, 0.95 * remaining))
        highs, status = _run_model(core, model, presolve, main_tl)
        last_status = status
        if status in solution_states:
            try:
                y, s_model, z_model = _raw_solution(highs, model_data)
                mapped = _map_solution(y, s_model, z_model, original, scale_info)
                if mapped is not None:
                    x, s, z = mapped
                    if _validate_solution(original, x, s, z):
                        return _result("solved", x, s, z, original.n, original.m), status
                    x2, sc, zc = _prepare_solution(original, x, s, z)
                    if _validate_solution(original, x2, sc, zc):
                        return _result("solved", x2, sc, zc, original.n, original.m), status
                    if not (original.n > 200000 or original.m > 200000):
                        for py, ps, pz in _polished_candidates(
                                model_data, y, z_model, s_model, deadline,
                                dense_limit):
                            candidate = _map_solution(py, ps, pz, original,
                                                      scale_info)
                            if candidate is None:
                                continue
                            px, ps_orig, pz_orig = candidate
                            if _validate_solution(original, px, ps_orig, pz_orig):
                                return _result("solved", px, ps_orig, pz_orig,
                                               original.n, original.m), status
            except Exception:
                pass
        if status in certificate_states:
            break
        if status == core.HighsModelStatus.kOptimal:
            break
    return None, last_status


def solve_lpqp(P_data, P_indices, P_indptr, P_shape, q,
               A_data, A_indices, A_indptr, A_shape, b, cones,
               time_budget=1.0):
    start = _now()
    n_guess = 0
    m_guess = 0
    try:
        n_guess = int(np.asarray(q).size)
    except Exception:
        pass
    try:
        m_guess = int(np.asarray(b).size)
    except Exception:
        pass
    try:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        n, m = q.size, b.size
        n_guess, m_guess = n, m
        if not (np.isfinite(q).all() and np.isfinite(b).all()):
            return _unresolved(n, m)
        P_shape = tuple(int(v) for v in P_shape)
        A_shape = tuple(int(v) for v in A_shape)
        if P_shape != (n, n) or A_shape != (m, n):
            return _unresolved(n, m)
        P = _canonical_csc(P_data, P_indices, P_indptr, P_shape)
        A = _canonical_csc(A_data, A_indices, A_indptr, A_shape)
        zero, nonneg, blocks = _parse_cones(cones, m)
        core = load_highs()
        try:
            budget = float(time_budget)
        except Exception:
            budget = 1.0
        if not np.isfinite(budget) or budget <= 0.0:
            budget = 1.0
        deadline = start + budget

        data = _Data()
        data.P, data.A, data.q, data.b = P, A, q, b
        data.zero, data.nonneg, data.blocks = zero, nonneg, blocks
        data.n, data.m = n, m
        data.a = _norm(A.data) or 1.0
        data.h = _norm(P.data) or 1.0
        data.bn = _norm(b)
        data.qn = _norm(q)
        data.has_p = bool(np.any(P.data != 0.0))
        data.p_pd = _p_positive_definite(P, deadline) if data.has_p else False

        lp_u, model_u = _build_model_for_data(core, data)
        lp_probe = lp_u
        variants = [(model_u, data, None)]
        try:
            scaling = _build_scaling(data)
            if scaling is not None:
                data_s, C, r, o = scaling
                lp_s, model_s = _build_model_for_data(core, data_s)
                lp_probe = lp_s
                variants = [(model_s, data_s, (C, r, o)),
                            (model_u, data, None)]
        except Exception:
            lp_probe = lp_u
            variants = [(model_u, data, None)]

        dense_limit = 1200 if budget >= 2.0 else 800
        solution_states = (core.HighsModelStatus.kOptimal,
                           core.HighsModelStatus.kTimeLimit,
                           core.HighsModelStatus.kIterationLimit,
                           core.HighsModelStatus.kSolutionLimit,
                           core.HighsModelStatus.kObjectiveBound,
                           core.HighsModelStatus.kObjectiveTarget,
                           core.HighsModelStatus.kUnknown)
        certificate_states = (core.HighsModelStatus.kInfeasible,
                              core.HighsModelStatus.kUnbounded,
                              core.HighsModelStatus.kUnboundedOrInfeasible)
        last_status = core.HighsModelStatus.kNotset
        prefer_primal_cert = True

        # A short LP-relaxation probe is useful for singular-Hessian QPs:
        # HiGHS's QP "choose" solver can consume the whole budget on an
        # unbounded QP, whereas the LP relaxation detects unboundedness
        # immediately.  Certificate LPs include P x = 0 and are validated
        # independently, so an unsuccessful probe cannot misclassify.
        if (data.has_p and not data.p_pd and n <= 200000 and m <= 200000 and
                budget >= 0.15):
            probe_tl = min(0.10 * budget, 0.20)
            if deadline - _now() > probe_tl + 0.02:
                _, probe_status = _run_model(core, lp_probe, True,
                                             max(0.01, probe_tl), "simplex")
                if probe_status in certificate_states:
                    prefer_primal_cert = (probe_status !=
                                          core.HighsModelStatus.kUnbounded)
                    early_deadline = min(deadline,
                                         _now() + min(0.25 * budget, 2.0))
                    early = _try_certificates(core, data, early_deadline,
                                              prefer_primal_cert)
                    if early is not None:
                        status, x, s, z = early
                        return _result(status, x, s, z, n, m)

        if data.p_pd:
            reserve = max(0.05, min(0.35, 0.15 * budget))
            main_deadline = min(start + 0.92 * budget, deadline - reserve)
        else:
            main_deadline = start + 0.78 * budget
        for model_v, data_v, scale_info in variants:
            if _now() >= main_deadline - 0.02:
                break
            solved, status = _solve_model_variant(
                core, model_v, data_v, data, scale_info, deadline,
                main_deadline, dense_limit, budget, solution_states,
                certificate_states)
            last_status = status
            if solved is not None:
                return solved
            if status in certificate_states:
                prefer_primal_cert = status != core.HighsModelStatus.kUnbounded
                break

        if last_status == core.HighsModelStatus.kUnbounded:
            prefer_primal_cert = False
        elif last_status == core.HighsModelStatus.kInfeasible:
            prefer_primal_cert = True
        final = _try_certificates(core, data, deadline, prefer_primal_cert)
        if final is not None:
            status, x, s, z = final
            return _result(status, x, s, z, n, m)
        return _unresolved(n, m)
    except Exception:
        return _unresolved(n_guess, m_guess)
