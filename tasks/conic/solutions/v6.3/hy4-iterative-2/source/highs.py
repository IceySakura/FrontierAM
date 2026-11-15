"""Thin wrapper around the HiGHS shared library bundled inside SciPy.

The extension module is loaded directly by path (no scipy package import),
which keeps interpreter startup well below the 1 s per-case budget.
"""
import glob
import importlib.util
import os

import numpy as np

_CORE = None


def _load_core():
    global _CORE
    if _CORE is not None:
        return _CORE
    # Direct extension load avoids importing scipy.optimize (~0.35 s).
    paths = (glob.glob("/usr/local/lib/python3*/site-packages/scipy/optimize/_highspy/_core*.so")
             + glob.glob("/usr/lib/python3/dist-packages/scipy/optimize/_highspy/_core*.so"))
    for p in paths:
        try:
            spec = importlib.util.spec_from_file_location("_core", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _CORE = mod
            return mod
        except Exception:
            continue
    try:
        import scipy.optimize._highspy._core as c  # noqa: F401
        _CORE = c
        return c
    except Exception:
        return None


class HighsSolver:
    def __init__(self, n, m):
        c = _load_core()
        if c is None:
            raise RuntimeError("HiGHS unavailable")
        self.c = c
        self.h = c._Highs()
        self.h.setOptionValue("output_flag", False)
        self.inf = self.h.getInfinity()
        self.n, self.m = n, m
        self.lp = None
        self.hessian = None

    def set_model(self, A, b_lower, b_upper, q, P=None, options=None,
                  col_lower=None, col_upper=None):
        """A: dense or CSC-like (m,n); P: dense (n,n) or None."""
        c, h = self.c, self.h
        n, m = self.n, self.m
        lp = c.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = np.asarray(q, dtype=np.float64).ravel()
        cl = (np.full(n, -self.inf) if col_lower is None
              else np.asarray(col_lower, dtype=np.float64).ravel().copy())
        cu = (np.full(n, self.inf) if col_upper is None
              else np.asarray(col_upper, dtype=np.float64).ravel().copy())
        cl[np.isneginf(cl)] = -self.inf
        cu[np.isposinf(cu)] = self.inf
        lp.col_lower_ = cl
        lp.col_upper_ = cu
        low = np.asarray(b_lower, dtype=np.float64).ravel().copy()
        up = np.asarray(b_upper, dtype=np.float64).ravel().copy()
        low[np.isneginf(low)] = -self.inf
        up[np.isposinf(up)] = self.inf
        lp.row_lower_ = low
        lp.row_upper_ = up
        am = c.HighsSparseMatrix()
        am.num_col_ = n
        am.num_row_ = m
        am.format_ = c.MatrixFormat.kColwise
        if hasattr(A, "indptr"):
            Ac = A.canonical() if hasattr(A, "canonical") else A
            am.start_ = np.asarray(Ac.indptr, dtype=np.int32)
            am.index_ = np.asarray(Ac.indices, dtype=np.int32)
            am.value_ = np.asarray(Ac.data, dtype=np.float64)
        else:
            Ad = np.asarray(A, dtype=np.float64)
            inds, ptr, vals = [], [0], []
            for j in range(n):
                rows = np.flatnonzero(Ad[:, j])
                inds.extend(rows.tolist())
                vals.extend(Ad[rows, j].tolist())
                ptr.append(len(inds))
            am.start_ = np.asarray(ptr, dtype=np.int32)
            am.index_ = np.asarray(inds, dtype=np.int32)
            am.value_ = np.asarray(vals, dtype=np.float64)
        lp.a_matrix_ = am
        h.passModel(lp)
        if P is not None:
            # HiGHS expects a triangular part of the symmetric Hessian (CSC).
            if hasattr(P, "tril"):
                Hc = P.tril()
                if not Hc.nnz:
                    Hc = None
                else:
                    indptr, indices, values = Hc.indptr, Hc.indices, Hc.data
            else:
                Pm = np.asarray(P, dtype=np.float64)
                H = np.tril(Pm)
                inds, ptr, vals = [], [0], []
                for j in range(n):
                    rows = np.flatnonzero(H[:, j])
                    inds.extend(rows.tolist())
                    vals.extend(H[rows, j].tolist())
                    ptr.append(len(inds))
                indptr = np.asarray(ptr, dtype=np.int32)
                indices = np.asarray(inds, dtype=np.int32)
                values = np.asarray(vals, dtype=np.float64)
                Hc = values if len(values) else None
            if Hc is not None:
                hh = c.HighsHessian()
                hh.dim_ = n
                hh.format_ = c.HessianFormat.kTriangular
                hh.start_ = np.asarray(indptr, dtype=np.int32)
                hh.index_ = np.asarray(indices, dtype=np.int32)
                hh.value_ = np.asarray(values, dtype=np.float64)
                h.passHessian(hh)
        for key, val in (options or {}).items():
            try:
                h.setOptionValue(key, val)
            except Exception:
                pass
        self.lp = lp

    def solve(self):
        self.h.run()
        status = self.h.getModelStatus()
        sol = self.h.getSolution()
        return status, sol, self.h.getObjectiveValue()


DEFAULT_OPTIONS = {
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
    "ipm_optimality_tolerance": 1e-10,
}


def _time_options(deadline, cap):
    import time as _time
    tl = min(cap, max(0.05, deadline - _time.monotonic()))
    return dict(DEFAULT_OPTIONS, time_limit=float(tl))


def solve_linear_cones(P, q, A, b, layout, deadline, verify_solved):
    """Solve LP/QP over zero+nonnegative cones with HiGHS.

    Returns a result dict with status and vectors.
    """
    n, m = len(q), len(b)
    lo = np.empty(m)
    hi = np.empty(m)
    for kind, dim, start, width in layout:
        if kind == "zero":
            lo[start:start + width] = b[start:start + width]
            hi[start:start + width] = b[start:start + width]
        elif kind == "nonnegative":
            lo[start:start + width] = -np.inf
            hi[start:start + width] = b[start:start + width]
        else:
            return None
    try:
        import time
        hs = HighsSolver(n, m)
        if P.nnz and n > 300:
            cap = 1.4
        else:
            cap = 0.6 if n + m < 30000 else (2.5 if n + m < 300000 else 5.0)
        hs.set_model(A, lo, hi, q, P if P.nnz else None,
                     _time_options(deadline, cap))
        if time.monotonic() > deadline:
            return None
        status, sol, obj = hs.solve()
        name = str(status)
        if "kOptimal" not in name:
            return {"status": "infeasible" if "kInfeasible" in name
                    else "unbounded" if "kUnbounded" in name else "failed",
                    "raw_status": name}
        x = np.array(sol.col_value, dtype=np.float64).ravel()
        s = b - A.matvec(x)
        z = -np.array(sol.row_dual, dtype=np.float64).ravel()
        ok, diag = verify_solved(x, s, z)
        if not ok:
            x2, s2, z2 = polish_kkt(P, q, A, b, layout, x, s, z)
            ok2, diag2 = verify_solved(x2, s2, z2)
            if ok2:
                return {"status": "solved", "x": x2, "s": s2, "z": z2}
        else:
            return {"status": "solved", "x": x, "s": s, "z": z}
        return {"status": "failed", "x": x, "s": s, "z": z}
    except Exception:
        return None


def polish_kkt(P, q, A, b, layout, x, s, z, iters=6):
    """Recover accurate duals / primal point from an active-set KKT solve."""
    n, m = len(q), len(b)
    if m * n > 2_000_000 or n > 2500:
        return _polish_sparse(P, q, A, b, layout, x, s, z, iters)
    Ad, Pd = A.dense(), P.dense()
    scale = max(1.0, float(np.max(np.abs(b))) if m else 1.0,
                float(np.max(np.abs(q))) if n else 1.0)
    zero_rows = np.concatenate([np.arange(st, st + w) for k, d, st, w in layout
                                if k == "zero"]).astype(int) if any(
                                    k == "zero" for k, _, _, _ in layout) else np.empty(0, int)
    nn_rows = np.concatenate([np.arange(st, st + w) for k, d, st, w in layout
                              if k == "nonnegative"]).astype(int) if any(
                                  k == "nonnegative" for k, _, _, _ in layout) else np.empty(0, int)
    tol = 1e-7 * scale
    active = set(zero_rows.tolist()) | set(nn_rows[np.abs(s[nn_rows]) <= tol].tolist())
    for _ in range(iters):
        R = np.array(sorted(active), dtype=int)
        if R.size == 0:
            break
        AR = Ad[R, :]
        K = np.zeros((n + R.size, n + R.size))
        K[:n, :n] = Pd
        K[:n, n:] = AR.T
        K[n:, :n] = AR
        rhs = np.concatenate((-q, b[R]))
        try:
            from ipm import _lin_solve
            y = _lin_solve(K, rhs)
        except Exception:
            try:
                y = np.linalg.lstsq(K, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
        xn, zR = y[:n], y[n:]
        sn = b - Ad @ xn
        bad = [int(r) for r, zi in zip(R, zR) if r in nn_rows and zi < -1e-10 * scale]
        violate = [int(r) for r in nn_rows if sn[r] < -1e-9 * scale and r not in active]
        if not bad and not violate:
            zn = np.zeros(m)
            zn[R] = zR
            return xn, b - A.matvec(xn), zn
        for r in bad:
            active.discard(r)
        for r in violate[:20]:
            active.add(r)
    return x, s, z


def _polish_sparse(P, q, A, b, layout, x, s, z, iters=6):
    """Active-set KKT polish using sparse linear algebra for large instances."""
    if n + m > 60000 or A.nnz > 3_000_000:
        return x, s, z
    try:
        from scipy import sparse as sp
        from scipy.sparse import linalg as spla
        import numpy as np
    except Exception:
        return x, s, z
    n, m = len(q), len(b)
    scale = max(1.0, float(np.max(np.abs(b))) if m else 1.0,
                float(np.max(np.abs(q))) if n else 1.0)
    zero_rows = np.concatenate([np.arange(st, st + w) for k, d, st, w in layout
                                if k == "zero"]).astype(int) if any(
                                    k == "zero" for k, _, _, _ in layout) else np.empty(0, int)
    nn_rows = np.concatenate([np.arange(st, st + w) for k, d, st, w in layout
                              if k == "nonnegative"]).astype(int) if any(
                                  k == "nonnegative" for k, _, _, _ in layout) else np.empty(0, int)
    nnmask = np.zeros(m, dtype=bool)
    nnmask[nn_rows] = True
    Ps = sp.csc_matrix((P.data, P.indices, P.indptr), shape=P.shape)
    As = sp.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)
    active = set(zero_rows.tolist()) | set(nn_rows[np.abs(s[nn_rows]) <= 1e-7 * scale].tolist())
    for _ in range(iters):
        R = np.array(sorted(active), dtype=int)
        if R.size == 0:
            break
        AR = As[R, :]
        K = sp.bmat([[Ps, AR.T], [AR, None]], format="csc")
        rhs = np.concatenate((-q, b[R]))
        try:
            lu = spla.splu(K)
            y = lu.solve(rhs)
            y += lu.solve(rhs - K @ y)
        except Exception:
            return x, s, z
        xn, zR = y[:n], y[n:]
        sn = b - As @ xn
        bad = [int(r) for r, zi in zip(R, zR) if nnmask[r] and zi < -1e-10 * scale]
        violate = [int(r) for r in nn_rows if sn[r] < -1e-9 * scale and r not in active]
        if not bad and not violate:
            zn = np.zeros(m)
            zn[R] = zR
            return xn, b - A.matvec(xn), zn
        for r in bad:
            active.discard(r)
        for r in violate[:20]:
            active.add(r)
    return x, s, z
