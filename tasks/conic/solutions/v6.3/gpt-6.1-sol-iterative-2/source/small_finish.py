import os

import numpy as np
from scipy import linalg, sparse

from model import assess, repair_slack, smat


def solve(original, problem, D, E, cost, initial):
    if problem.n > 80 or problem.m > 4000:
        return None
    if any(c["type"] == "psd" and c["dim"] > 12 for c in problem.cones):
        return None
    from scipy.optimize import minimize
    A, b, P, q = problem.A.toarray(), problem.b, problem.P, problem.q
    zero, blocks, start = [], [], 0
    for cone in problem.cones:
        kind, dim = cone["type"], cone["dim"]
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        ids = np.arange(start, start + width)
        if kind == "zero":
            zero.extend(ids)
        else:
            coordinates = None
            if kind == "psd":
                cols, rows = np.tril_indices(dim)
                coordinates = rows, cols
            blocks.append((kind, dim, ids, coordinates))
        start += width
    zero = np.asarray(zero, dtype=np.int64)
    if zero.size:
        unused, R, piv = linalg.qr(A[zero].T, mode="economic", pivoting=True,
                                  check_finite=False)
        diagonal = np.abs(np.diag(R))
        rank = int(np.sum(diagonal > 1e-11 * max(1.0, np.max(diagonal, initial=0.0))))
        zero = zero[piv[:rank]]
    cache = [None, None]

    def evaluate(x):
        if cache[0] is not None and np.array_equal(cache[0], x):
            return cache[1]
        s = b - A @ x
        values, jacobian, gr, gc, gv = [], [], [], [], []
        row = 0
        for kind, dim, ids, coordinates in blocks:
            sb = s[ids]
            if kind == "nonnegative":
                values.append(sb)
                jacobian.append(-A[ids])
                gr.extend(row + np.arange(dim))
                gc.extend(ids)
                gv.extend(np.ones(dim))
                row += dim
                continue
            elif kind == "soc":
                radius = float(np.linalg.norm(sb[1:]))
                v = np.array([sb[0] - radius])
                G = np.r_[1.0, -sb[1:] / max(radius, 1e-15)][None, :]
            elif kind == "exponential":
                u, v0, w0 = sb
                v1, w1 = max(v0, 1e-12), max(w0, 1e-12)
                L = np.log(w1) - np.log(v1)
                v = np.array([v1 * L - u, v0, w0])
                G = np.array([[-1.0, L - 1.0, v1 / w1],
                              [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            else:
                rows, cols = coordinates
                v, U = linalg.eigh(smat(sb, dim, rows, cols), check_finite=False)
                G = (U[rows, :] * U[cols, :]).T
                G[:, rows != cols] *= np.sqrt(2.0)
            values.append(v)
            jacobian.append(-G @ A[ids])
            rr, cc = np.nonzero(G)
            gr.extend(row + rr)
            gc.extend(ids[cc])
            gv.extend(G[rr, cc])
            row += v.size
        value = np.concatenate(values) if values else np.empty(0)
        jac = np.vstack(jacobian) if jacobian else np.empty((0, problem.n))
        G = sparse.csc_matrix((gv, (gr, gc)), shape=(row, problem.m))
        result = (s[zero], -A[zero], value, jac, G)
        cache[:] = [x.copy(), result]
        return result

    constraints = []
    if zero.size:
        constraints.append(dict(type="eq", fun=lambda x: evaluate(x)[0],
                                jac=lambda x: evaluate(x)[1]))
    if blocks:
        constraints.append(dict(type="ineq", fun=lambda x: evaluate(x)[2],
                                jac=lambda x: evaluate(x)[3]))
    result = minimize(lambda x: 0.5 * float(x @ (P @ x)) + float(q @ x),
                      initial, jac=lambda x: P @ x + q, method="SLSQP",
                      constraints=constraints,
                      options=dict(ftol=2e-13, maxiter=150, disp=False))
    multipliers = getattr(result, "multipliers", None)
    if multipliers is None or not np.isfinite(result.x).all():
        return None
    evaluated = evaluate(result.x)
    z = np.zeros(problem.m)
    z[zero] = multipliers[:zero.size]
    z += evaluated[4].T @ np.maximum(multipliers[zero.size:zero.size + evaluated[2].size], 0.0)
    x = D * result.x
    candidate = dict(status="solved", x=x, s=repair_slack(original, x), z=E * z / cost)
    good, report = assess(original, candidate)
    if os.environ.get("SOLVER_LOG"):
        print("small_finish", result.message, good, report, flush=True)
    return candidate if good else None
