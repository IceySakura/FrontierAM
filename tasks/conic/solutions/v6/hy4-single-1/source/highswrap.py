"""Thin wrapper around the HiGHS LP/QP bindings bundled with SciPy."""
from __future__ import annotations

import numpy as np
from scipy import sparse
def _load_highs_core():
    """Load the HiGHS pybind extension without importing scipy.optimize."""
    try:
        import glob
        import importlib.util
        root = os.path.dirname(os.path.dirname(sparse.__file__))
        paths = glob.glob(os.path.join(root, "optimize", "_highspy", "_core*.so"))
        if paths:
            spec = importlib.util.spec_from_file_location("_core", paths[0])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules.setdefault("_highs_core_direct", mod)
            if hasattr(mod, "_Highs"):
                return mod
    except Exception:
        pass
    from scipy.optimize._highspy import _core
    return _core

import os
import sys
_core = _load_highs_core()

INF = _core.kHighsInf


class HighsLP:
    """LP/QP with variables in R^n and rows  lo <= M v <= hi.

    ``M`` is CSC (nrow x nvar).  Extra rows can be appended with
    :meth:`add_rows`, preserving a warm-started simplex basis.
    """

    def __init__(self, cost, col_lo, col_hi, M, row_lo, row_hi,
                 hessian=None, solver="simplex", time_limit=None,
                 small_matrix_value=None, dual_tol=1e-10):
        self.h = _core._Highs()
        h = self.h
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("primal_feasibility_tolerance", 1e-10)
        h.setOptionValue("dual_feasibility_tolerance", dual_tol)
        h.setOptionValue("solver", solver)
        if small_matrix_value is not None:
            h.setOptionValue("small_matrix_value", small_matrix_value)
            h.setOptionValue("large_matrix_value", 1e30)
        if time_limit is not None:
            h.setOptionValue("time_limit", max(0.05, float(time_limit)))
        M = M.tocsc()
        lp = _core.HighsLp()
        lp.num_col_ = int(M.shape[1])
        lp.num_row_ = int(M.shape[0])
        lp.col_cost_ = np.ascontiguousarray(cost, dtype=np.float64)
        lp.col_lower_ = np.ascontiguousarray(col_lo, dtype=np.float64)
        lp.col_upper_ = np.ascontiguousarray(col_hi, dtype=np.float64)
        lp.row_lower_ = np.ascontiguousarray(row_lo, dtype=np.float64)
        lp.row_upper_ = np.ascontiguousarray(row_hi, dtype=np.float64)
        mat = _core.HighsSparseMatrix()
        mat.num_col_ = int(M.shape[1])
        mat.num_row_ = int(M.shape[0])
        mat.format_ = _core.MatrixFormat.kColwise
        mat.start_ = np.ascontiguousarray(M.indptr, dtype=np.int32)
        mat.index_ = np.ascontiguousarray(M.indices, dtype=np.int32)
        mat.value_ = np.ascontiguousarray(M.data, dtype=np.float64)
        lp.a_matrix_ = mat
        if hessian is not None and hessian.nnz:
            H = hessian.tocsc()
            hess = _core.HighsHessian()
            hess.dim_ = int(H.shape[0])
            hess.format_ = _core.HessianFormat.kTriangular
            hess.start_ = np.ascontiguousarray(H.indptr, dtype=np.int32)
            hess.index_ = np.ascontiguousarray(H.indices, dtype=np.int32)
            hess.value_ = np.ascontiguousarray(H.data, dtype=np.float64)
            model = _core.HighsModel()
            model.lp_ = lp
            model.hessian_ = hess
            h.passModel(model)
        else:
            h.passModel(lp)
        self.nvar, self.nrow = M.shape[1], M.shape[0]
        self.hessian = hessian
        self.solver_kind = solver
        self._retried = False

    def add_rows(self, rows_csr, lo, hi):
        rows_csr = rows_csr.tocsr()
        k = rows_csr.shape[0]
        if k == 0:
            return
        st = self.h.addRows(
            int(k),
            np.ascontiguousarray(lo, dtype=np.float64),
            np.ascontiguousarray(hi, dtype=np.float64),
            int(rows_csr.nnz),
            np.ascontiguousarray(rows_csr.indptr, dtype=np.int32),
            np.ascontiguousarray(rows_csr.indices, dtype=np.int32),
            np.ascontiguousarray(rows_csr.data, dtype=np.float64),
        )
        self.nrow += k
        return st

    def solve(self, time_limit=None):
        if time_limit is not None:
            self.h.setOptionValue("time_limit", max(0.05, float(time_limit)))
        self.h.run()
        status = self.h.modelStatusToString(self.h.getModelStatus())
        if status in ("Not Set", "Error", "Unknown") and self.solver_kind == "simplex":
            # HiGHS 1.8 dual simplex can stall at the last 1e-10 dual
            # infeasibility after rows are appended.  Relaxed tolerances and
            # the primal simplex repair it without affecting basis accuracy.
            if not self._retried:
                self._retried = True
                self.h.setOptionValue("dual_feasibility_tolerance", 1e-8)
                self.h.setOptionValue("simplex_strategy", 4)
            self.h.run()
            status = self.h.modelStatusToString(self.h.getModelStatus())
        sol = self.h.getSolution()
        info = self.h.getInfo()
        return dict(status=status,
                    value=np.asarray(sol.col_value, dtype=np.float64),
                    row_value=np.asarray(sol.row_value, dtype=np.float64),
                    col_dual=np.asarray(sol.col_dual, dtype=np.float64),
                    row_dual=np.asarray(sol.row_dual, dtype=np.float64),
                    value_valid=bool(sol.value_valid),
                    dual_valid=bool(sol.dual_valid),
                    objective=float(info.objective_function_value),
                    basis=self.h.getBasis())

    def objective(self):
        return float(self.h.getObjectiveValue())
