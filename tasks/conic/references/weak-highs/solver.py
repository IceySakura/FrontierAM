"""Weak conic baseline: the HiGHS LP/QP solver bundled with SciPy, applied as is.

Problems whose cones are all zero or nonnegative are passed to HiGHS (rows L <= Ax <= U with free columns; zero rows
fixed at b, nonnegative rows Ax <= b); anything with a second-order, exponential or PSD cone is reported unresolved.
Optimal: x, s = b - Ax and z = -(row duals). Infeasible or unbounded models use HiGHS's rays when it has one, otherwise
unresolved. It stands for what calling the bundled library gives, the 0-point anchor of the board's conic mapping.
"""

import json
import sys

import numpy as np


def write(path, status, **vectors):
    meta = np.array(json.dumps({'schema_version': 1, 'status': status}))
    np.savez(path, metadata_json=meta, **{k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in vectors.items()})


def csc(data, prefix):
    from scipy import sparse
    shape = tuple(int(v) for v in data[prefix + '_shape'])
    matrix = sparse.csc_matrix((data[prefix + '_data'], data[prefix + '_indices'], data[prefix + '_indptr']), shape=shape)
    matrix.sum_duplicates()
    return matrix


def main():
    args = sys.argv[1:]
    source, target = args[args.index('--input') + 1], args[args.index('--output') + 1]
    with np.load(source, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}
    meta = json.loads(str(data['metadata_json']))
    if any(cone['type'] not in ('zero', 'nonnegative') for cone in meta['cones']):
        return write(target, 'unresolved')
    from scipy.optimize._highspy import _core as hs
    q, b = data['q'].astype(np.float64), data['b'].astype(np.float64)
    P, A = csc(data, 'P'), csc(data, 'A')
    n, m = q.size, b.size
    lower = np.full(m, -np.inf)
    row = 0
    for cone in meta['cones']:
        if cone['type'] == 'zero':
            lower[row:row + cone['dim']] = b[row:row + cone['dim']]
        row += cone['dim']
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_, lp.col_upper_ = np.full(n, -np.inf), np.full(n, np.inf)
    lp.row_lower_, lp.row_upper_ = lower, b.copy()
    lp.offset_ = float(meta.get('objective_offset') or 0.0)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    model = hs.HighsModel()
    model.lp_ = lp
    if P.nnz:
        from scipy import sparse
        lower_p = sparse.tril(P, format='csc')
        hessian = hs.HighsHessian()
        hessian.dim_, hessian.format_ = n, hs.HessianFormat.kTriangular
        hessian.start_, hessian.index_, hessian.value_ = lower_p.indptr, lower_p.indices, lower_p.data
        model.hessian_ = hessian
    highs = hs._Highs()
    highs.setOptionValue('output_flag', False)
    for option in ('primal_feasibility_tolerance', 'dual_feasibility_tolerance'):
        highs.setOptionValue(option, 1e-9)
    highs.passModel(model)
    highs.run()
    status = highs.getModelStatus()
    if status == hs.HighsModelStatus.kOptimal:
        solution = highs.getSolution()
        x = np.asarray(solution.col_value, dtype=np.float64)
        return write(target, 'solved', x=x, s=b - A @ x, z=-np.asarray(solution.row_dual, dtype=np.float64))
    if status == hs.HighsModelStatus.kInfeasible and hasattr(highs, 'getDualRay'):
        has_ray, ray = highs.getDualRay()
        if has_ray:
            return write(target, 'primal_infeasible', z=-np.asarray(ray, dtype=np.float64))
    if status == hs.HighsModelStatus.kUnbounded and hasattr(highs, 'getPrimalRay'):
        has_ray, ray = highs.getPrimalRay()
        if has_ray:
            x = np.asarray(ray, dtype=np.float64)
            return write(target, 'dual_infeasible', x=x, s=-(A @ x))
    return write(target, 'unresolved')


if __name__ == '__main__':
    main()
