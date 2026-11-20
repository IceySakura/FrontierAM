import numpy as np
from scipy import sparse

from model import Scaled, maxabs


def highs_model(A, c, lo, hi, lb=None, ub=None, P=None, mode='choose', limit=1.0,
                tight=True, warm=None):
    from scipy.optimize._highspy import _core as h

    m, n = A.shape
    high = h._Highs()
    opts = h.HighsOptions()
    opts.output_flag = opts.log_to_console = False
    opts.threads = 1
    opts.parallel = 'off'
    opts.time_limit = max(0.002, limit)
    opts.solver = mode
    opts.primal_feasibility_tolerance = 1.e-9 if tight else 1.e-8
    opts.dual_feasibility_tolerance = 1.e-9 if tight else 1.e-8
    opts.ipm_optimality_tolerance = 1.e-11
    opts.small_matrix_value = 1.e-12
    opts.large_matrix_value = 1.e19
    opts.infinite_bound = 1.e30
    opts.infinite_cost = 1.e30
    high.passOptions(opts)
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)
    noint = np.zeros(n, dtype=np.int32)
    status = high.passModel(n, m, A.nnz, 1, 1, 0., c, lb, ub, lo, hi,
                            A.indptr, A.indices, A.data, noint)
    if status == h.HighsStatus.kError:
        return 'error', None
    if P is not None and P.nnz:
        H = sparse.tril(P, format='csc')
        status = high.passHessian(n, H.nnz, 1, H.indptr, H.indices, H.data)
        if status == h.HighsStatus.kError:
            return 'error', None
        high.setOptionValue('qp_regularization_value', 1.e-10)
    if warm is not None:
        high.setSolution(n, np.arange(n, dtype=np.int32), warm)
    high.run()
    status = high.getModelStatus()
    if status in (h.HighsModelStatus.kOptimal, h.HighsModelStatus.kTimeLimit,
                  h.HighsModelStatus.kUnknown, h.HighsModelStatus.kIterationLimit):
        solution = high.getSolution()
        if solution.value_valid:
            return ('optimal' if status == h.HighsModelStatus.kOptimal else 'partial',
                    (np.asarray(solution.col_value), -np.asarray(solution.row_dual)))
    if status == h.HighsModelStatus.kInfeasible:
        return 'infeasible', None
    if status in (h.HighsModelStatus.kUnbounded, h.HighsModelStatus.kUnboundedOrInfeasible):
        return 'unbounded', None
    if status == h.HighsModelStatus.kModelEmpty:
        return 'empty', None
    return 'other', None


def certificate(p, which, limit, scaled=None):
    if which == 'infeasible':
        if p.bn == 0.:
            return None
        A = p.A.T.tocsc()
        b = np.zeros(p.n)
        lb = np.zeros(p.m)
        lb[p.K.eq] = -1.
        status, sol = highs_model(A, p.b/p.bn, b, b, lb, np.ones(p.m), limit=limit)
        return p.primal_cert(sol[0]) if sol is not None else None
    if p.qn == 0.:
        return None
    if p.P.nnz:
        A = sparse.vstack([p.A, p.P], format='csc')
    else:
        A = p.A
    b = np.zeros(A.shape[0])
    lo = np.full(A.shape[0], -np.inf)
    lo[p.K.eq] = 0.
    lo[p.m:] = 0.
    status, sol = highs_model(A, p.q/p.qn, lo, b, -np.ones(p.n), np.ones(p.n),
                              limit=limit)
    return p.dual_cert(sol[0]) if sol is not None else None


def solve_linear(p, limit, allow_cert=True, scale_first=False, mode=None):
    def attempt(w, seconds, solver):
        lo = np.full(p.m, -np.inf)
        lo[p.K.eq] = w.b[p.K.eq]
        status, sol = highs_model(w.A, w.q, lo, w.b, P=w.P, mode=solver, limit=seconds)
        answer = None
        if sol is not None:
            x, z = sol
            if w is not p:
                x = w.D*x
                z = (w.E/w.c)*z
            answer = p.optimal(x, z)
            if answer is None and status == 'optimal' and p.P.nnz and p.n+p.m < 23000:
                from poly import polish_polyhedral
                answer = polish_polyhedral(p,x,z,start+limit)
        return status, answer

    from time import perf_counter
    start = perf_counter()
    big = p.n + p.m > 100000
    if mode is None:
        mode = 'ipm' if big and not p.P.nnz else 'choose'
    w = Scaled(p, passes=7, scale_rhs=True) if scale_first else p
    status, ans = attempt(w, limit * (0.8 if big else 0.85), mode)
    if ans is not None:
        return ans
    remaining = limit - (perf_counter() - start)
    if allow_cert and status in ('infeasible', 'unbounded') and remaining > 0.015:
        ans = certificate(p, status, remaining * .85)
        if ans is not None:
            return ans
    remaining = limit - (perf_counter() - start)
    if remaining > 0.04 and not big:
        alt = p if w is not p else Scaled(p, passes=10, scale_rhs=True)
        status, ans = attempt(alt, remaining * .8, 'choose')
        if ans is not None:
            return ans
    return None
