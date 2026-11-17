import numpy as np

from model import Problem, assess, cone_error, equilibrate, norm


def zero_objective(data):
    q, b = data["q"], data["b"]
    if np.any(q):
        return None
    cones = data["metadata"]["cones"]
    s = b.copy()
    start = 0
    dense_psd = False
    for cone in cones:
        kind, dim = cone["type"], cone["dim"]
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        block = s[start:start + width]
        if kind == "zero":
            block[:] = 0.0
        elif kind == "nonnegative":
            np.maximum(block, 0.0, out=block)
        elif kind == "psd":
            diagonal = np.arange(dim)
            diagonal = diagonal * (diagonal + 3) // 2
            if np.count_nonzero(block) == np.count_nonzero(block[diagonal]):
                block[diagonal] = np.maximum(block[diagonal], 0.0)
            elif dim > 2:
                if np.min(block[diagonal]) < -7e-9 * max(1.0, norm(block)):
                    return None
                dense_psd = True
        start += width
    if norm(s - b) > 7e-9 * (1 + max(norm(s), norm(b))):
        return None
    if dense_psd:
        from bootstrap import prepare
        prepare()
    if cone_error(s, cones) > 7e-9:
        return None
    return dict(status="solved", x=np.zeros(q.size), s=s, z=np.zeros(b.size))


def unconstrained(problem):
    if problem.m or not problem.n:
        return None
    from scipy import linalg, sparse
    from scipy.sparse.linalg import splu
    scale = norm(problem.P.data)
    if not scale:
        return None
    P = problem.P / scale
    delta = 1e-9
    try:
        if problem.n <= 2000 and P.nnz > 0.08 * problem.n ** 2:
            H = P.toarray()
            H[np.diag_indices(problem.n)] += delta
            factor = linalg.cho_factor(H, lower=True, check_finite=False)
            solve = lambda r: linalg.cho_solve(factor, r, check_finite=False)
        else:
            factor = splu(P + sparse.eye(problem.n, format="csc") * delta)
            solve = factor.solve
        rhs = -problem.q / scale
        x = solve(rhs)
        for unused in range(6):
            candidate = dict(status="solved", x=x, s=np.empty(0), z=np.empty(0))
            if assess(problem, candidate)[0]:
                return candidate
            direction = dict(status="dual_infeasible", x=x, s=np.empty(0))
            if assess(problem, direction)[0]:
                return direction
            x += solve(rhs - P @ x)
    except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError):
        pass
    return None


def equalities(original):
    if not original.m or any(c["type"] != "zero" for c in original.cones):
        return None
    from scipy import sparse
    from scipy.sparse.linalg import splu
    problem, D, E, cost = equilibrate(original, passes=4)
    n, m = problem.n, problem.m
    K = sparse.bmat([[problem.P, problem.A.T], [problem.A, None]], format="csc")
    rhs = np.r_[-problem.q, problem.b]
    reg = np.r_[np.full(n, 1e-11), np.full(m, -1e-11)]
    try:
        factor = splu(K + sparse.diags(reg, format="csc"))
        v = factor.solve(rhs)
        for unused in range(8):
            x = D * v[:n]
            z = E * v[n:] / cost
            if not original.P.nnz and not np.any(original.q):
                z = np.zeros(m)
            result = dict(status="solved", x=x, s=np.zeros(m), z=z)
            if assess(original, result)[0]:
                return result
            certificate = dict(status="primal_infeasible", z=E * v[n:] / cost)
            if assess(original, certificate)[0]:
                return certificate
            certificate = dict(status="dual_infeasible", x=x, s=np.zeros(m))
            if assess(original, certificate)[0]:
                return certificate
            v += factor.solve(rhs - K @ v)
    except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError):
        pass
    return None


def small_cones(problem):
    from scipy import sparse
    changed, blocks, cones = False, [], []
    start = 0
    for cone in problem.cones:
        kind, dim = cone["type"], cone["dim"]
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        if kind == "psd" and dim == 1:
            cones.append(dict(type="nonnegative", dim=1))
            changed = True
        elif kind == "soc" and dim == 2:
            blocks.append((start, np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.0)))
            cones.append(dict(type="nonnegative", dim=2))
            changed = True
        elif kind == "psd" and dim == 2:
            a = 1.0 / np.sqrt(2.0)
            blocks.append((start, np.array([[a, 0., a], [a, 0., -a], [0., 1., 0.]])))
            cones.append(dict(type="soc", dim=3))
            changed = True
        else:
            cones.append(cone)
        start += width
    if not changed:
        return problem, None
    merged = []
    for cone in cones:
        kind = cone["type"]
        if merged and kind in ("zero", "nonnegative") and merged[-1]["type"] == kind:
            merged[-1]["dim"] += cone["dim"]
        else:
            merged.append(dict(cone))
    cones = merged
    if not blocks:
        return Problem(problem.P, problem.q, problem.A, problem.b, cones), None
    identity = np.ones(problem.m, dtype=bool)
    rr, cc, vv = [], [], []
    for start, U in blocks:
        d = U.shape[0]
        identity[start:start + d] = False
        rows, cols = np.nonzero(U)
        rr.append(rows + start)
        cc.append(cols + start)
        vv.append(U[rows, cols])
    ids = np.flatnonzero(identity)
    rr.append(ids)
    cc.append(ids)
    vv.append(np.ones(ids.size))
    T = sparse.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                          shape=(problem.m, problem.m))
    A = (T @ problem.A).tocsc()
    A.sort_indices()
    return Problem(problem.P, problem.q, A, T @ problem.b, cones), T


class AffineReduction:
    def __init__(self, original):
        self.original = original
        self.problem = None
        n, m = original.n, original.m
        if n < 40 or n > 200000 or m > 600000:
            return
        zero = np.zeros(m, dtype=bool)
        for kind, block in original.linear_slices:
            if kind == "zero":
                zero[block] = True
        zero_ids = np.flatnonzero(zero)
        if not zero_ids.size:
            return
        eligible = (original.q == 0) & (np.diff(original.P.indptr) == 0)
        eligible[original.P.indices] = False
        if not np.any(eligible):
            return
        A = original.A
        Ae = A[zero_ids, :].tocsc()
        columns = np.flatnonzero(eligible & (np.diff(Ae.indptr) == 1)
                                 & (np.diff(A.indptr) <= 10))
        if not columns.size:
            return
        rows = zero_ids[Ae.indices[Ae.indptr[columns]]]
        pivots = Ae.data[Ae.indptr[columns]]
        csr = A.tocsr()
        row_count = np.diff(csr.indptr)
        row_max = np.zeros(m)
        np.maximum.at(row_max, A.indices, np.abs(A.data))
        keep = (row_count[rows] <= 12) & (np.abs(pivots) >= 1e-3 * row_max[rows])
        columns, rows, pivots = columns[keep], rows[keep], pivots[keep]
        if not columns.size:
            return
        order = np.lexsort((-columns, np.diff(A.indptr)[columns], rows))
        columns, rows, pivots = columns[order], rows[order], pivots[order]
        unused, first = np.unique(rows, return_index=True)
        columns, rows, pivots = columns[first], rows[first], pivots[first]
        beta = original.b[rows] / pivots
        keep = np.isfinite(beta)
        columns, rows, pivots, beta = columns[keep], rows[keep], pivots[keep], beta[keep]
        if columns.size < max(4, int(0.10 * n)):
            return
        free_mask = np.ones(n, dtype=bool)
        free_mask[columns] = False
        free = np.flatnonzero(free_mask)
        mapping = np.full(n, -1, dtype=np.int64)
        mapping[free] = np.arange(free.size)
        equations = A[rows, :].tocoo()
        other = equations.col != columns[equations.row]
        tc = mapping[equations.col[other]]
        if np.any(tc < 0):
            return
        from scipy import sparse
        T = sparse.csc_matrix(
            (-equations.data[other] / pivots[equations.row[other]],
             (equations.row[other], tc)), shape=(columns.size, free.size))
        eliminated_A = A[:, columns]
        reduced_A = (A[:, free] + eliminated_A @ T).tocsc()
        reduced_A.sum_duplicates()
        reduced_A.eliminate_zeros()
        reduced_A.sort_indices()
        reduced_b = original.b - eliminated_A @ beta
        if not np.isfinite(reduced_A.data).all() or not np.isfinite(reduced_b).all():
            return
        counts = np.asarray(reduced_A.getnnz(axis=1)).ravel()
        active = np.ones(m, dtype=bool)
        active[rows] = False
        cones, start = [], 0
        threshold = 1e-12 * (1 + norm(original.b))
        for cone in original.cones:
            kind, dim = cone["type"], cone["dim"]
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            ids = np.arange(start, start + width)
            if kind == "zero":
                active[ids[(counts[ids] == 0) & (np.abs(reduced_b[ids]) <= threshold)]] = False
                size = int(np.sum(active[ids]))
                if size:
                    cones.append(dict(type=kind, dim=size))
            elif kind == "nonnegative":
                active[ids[(counts[ids] == 0) & (reduced_b[ids] >= 0)]] = False
                size = int(np.sum(active[ids]))
                if size:
                    cones.append(dict(type=kind, dim=size))
            else:
                cones.append(dict(cone))
            start += width
        self.free, self.eliminated = free, columns
        self.rows, self.pivots, self.beta, self.T = rows, pivots, beta, T
        self.active = np.flatnonzero(active)
        self.reduced_b = reduced_b
        self.problem = Problem(original.P[free, :][:, free].tocsc(),
                               original.q[free].copy(),
                               reduced_A[self.active, :].tocsc(),
                               reduced_b[self.active].copy(), cones)

    def restore(self, result):
        original = self.original
        restored = dict(status=result["status"])
        if "x" in result:
            x = np.zeros(original.n)
            x[self.free] = result["x"]
            x[self.eliminated] = self.T @ result["x"]
            if result["status"] == "solved":
                x[self.eliminated] += self.beta
            restored["x"] = x
        if "s" in result:
            s = np.zeros(original.m)
            if result["status"] == "solved":
                s[:] = self.reduced_b
                for kind, block in original.linear_slices:
                    if kind == "zero":
                        s[block] = 0.0
                    else:
                        s[block] = np.maximum(s[block], 0.0)
            s[self.active] = result["s"]
            restored["s"] = s
        if "z" in result:
            z = np.zeros(original.m)
            z[self.active] = result["z"]
            gradient = original.A.T @ z
            z[self.rows] = -gradient[self.eliminated] / self.pivots
            restored["z"] = z
        return restored
