import numpy as np
from scipy import sparse
from problem import Problem, assess, norm
from box_solver import solve as solve_box


def solve(problem):
    types = {c["type"] for c in problem.cones}
    if types == {"exponential"}:
        result = exponential_bounds(problem)
    elif len(problem.cones) == 1 and types == {"soc"}:
        result = soc_least_squares(problem)
    elif problem.n == 1 and "psd" in types and types <= {"psd", "zero", "nonnegative"}:
        result = scalar_psd(problem)
    else:
        return None
    if result is not None:
        passed, errors = assess(problem, result)
        if passed:
            return result
        if result["status"] == "solved" and max(errors) < 1e-5:
            from refine import project_solution
            result = project_solution(problem, result)
            if assess(problem, result)[0]:
                return result
    return None


def soc_least_squares(p):
    if p.P.nnz or np.count_nonzero(p.q) > 1:
        return None
    head = p.A.getrow(0)
    if head.nnz != 1:
        return None
    column, a = int(head.indices[0]), float(head.data[0])
    if p.A.indptr[column+1]-p.A.indptr[column] != 1:
        return None
    if p.q.any() and p.q[column] == 0:
        return None
    cost = p.q[column]
    if cost/a > 0:
        x = np.zeros(p.n)
        x[column] = -np.sign(cost)
        return {"status":"dual_infeasible","x":x,"s":-p.A@x}
    B = p.A[1:,:].tocsc()
    x = np.zeros(p.n)
    if cost != 0 and B.nnz:
        from scipy.sparse.linalg import lsmr
        magnitudes = np.asarray(abs(B).max(axis=0).toarray()).ravel()
        scale = np.ones(p.n)
        good = magnitudes > 0
        scale[good] = 1/magnitudes[good]
        if not np.isfinite(scale).all():
            return None
        scaled = B.copy()
        scaled.data *= np.repeat(scale, np.diff(scaled.indptr))
        x = scale*lsmr(scaled, p.b[1:], atol=1e-13, btol=1e-13,
                       conlim=1e12, maxiter=min(150,4*min(scaled.shape)))[0]
    residual = p.b[1:]-B@x
    z = np.zeros(p.m)
    z[0] = -cost/a
    s = np.zeros(p.m)
    if norm(residual) <= 2e-9*(1+norm(p.b[1:])):
        x[column] = p.b[0]/a
    else:
        length = np.linalg.norm(residual)
        if not np.isfinite(length):
            return None
        x[column] = (p.b[0]-length)/a
        s[0], s[1:] = length, residual
        z[1:] = -z[0]*residual/length
    return {"status":"solved","x":x,"s":s,"z":z}


def exponential_bounds(p):
    if np.any(p.A.indices % 3 != 2):
        return None
    b = p.b.reshape(-1, 3)
    u, v = b[:, 0], b[:, 1]
    if np.any(v <= 0):
        return None
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        ratio = u/v
        logw = np.log(v)+ratio
        threshold = np.maximum(np.exp(logw), np.nextafter(0., 1.))
    if not np.isfinite(threshold).all():
        return None
    reduced = Problem(p.P, p.q, p.A[2::3, :].tocsc(), b[:, 2]-threshold,
                      [{"type": "nonnegative", "dim": len(b)}])
    if not np.isfinite(reduced.b).all():
        return None
    result = solve_box(reduced)
    if result is None:
        return None
    out = {"status": result["status"]}
    if "x" in result:
        out["x"] = result["x"]
        s = np.zeros_like(b)
        if result["status"] == "solved":
            s[:, :2] = b[:, :2]
            s[:, 2] = threshold+result["s"]
        else:
            s[:, 2] = result["s"]
        out["s"] = s.ravel()
    if "z" in result:
        z = np.zeros_like(b)
        wdual = result["z"]
        active = wdual > 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            alpha = np.exp(np.log(wdual[active])+np.log(threshold[active])-np.log(v[active]))
            z[active, 0] = -alpha
            z[active, 1] = np.where(alpha > 0, alpha*(ratio[active]-1), 0.)
        z[:, 2] = wdual
        if not np.isfinite(z).all():
            return None
        out["z"] = z.ravel()
    return out


def scalar_psd(p):
    from scipy import linalg
    from cones import PSD
    coeff, rhs, cones, records = [], [], [], []
    a = p.A.toarray().ravel()
    start = 0
    for block in p.cones:
        kind, d = block["type"], block["dim"]
        width = d*(d+1)//2 if kind == "psd" else d
        sl = slice(start, start+width)
        first = len(rhs)
        if kind != "psd":
            coeff.extend(a[sl])
            rhs.extend(p.b[sl])
            cones.append(block)
            records.append((sl, slice(first, first+width), None))
        else:
            psd = PSD(start, d)
            B, M = psd.mat(p.b[sl]), psd.mat(a[sl])
            if norm(a[sl]) == 0:
                value, vec = linalg.eigh(B, subset_by_index=[0,0], check_finite=False)
                sign = 0.
            else:
                try:
                    linalg.cholesky(M, check_finite=False)
                    D, sign = M, 1.
                except linalg.LinAlgError:
                    try:
                        linalg.cholesky(-M, check_finite=False)
                        D, sign = -M, -1.
                    except linalg.LinAlgError:
                        return None
                value, vec = linalg.eigh(B, D, subset_by_index=[0,0], check_finite=False)
            coeff.append(sign)
            rhs.append(value[0])
            cones.append({"type": "nonnegative", "dim": 1})
            basis = psd.vec(np.outer(vec[:, 0], vec[:, 0]))
            records.append((sl, slice(first, first+1), (psd, basis)))
        start += width
    reduced = Problem(p.P, p.q, sparse.csc_matrix(np.asarray(coeff)[:,None]),
                      np.asarray(rhs), cones)
    result = solve_box(reduced)
    if result is None:
        return None
    out = {"status": result["status"]}
    if "x" in result:
        out["x"] = result["x"]
        s = (p.b-p.A@out["x"] if result["status"] == "solved"
             else -p.A@out["x"])
        for sl, rsl, data in records:
            if data is None:
                s[sl] = result["s"][rsl]
            elif result["status"] == "solved":
                psd, basis = data
                minimum = linalg.eigvalsh(psd.mat(s[sl]), subset_by_index=[0,0],
                                          check_finite=False)[0]
                if minimum < 0:
                    s[psd.start+psd.diag] -= minimum
        out["s"] = s
    if "z" in result:
        z = np.zeros(p.m)
        for sl, rsl, data in records:
            z[sl] = result["z"][rsl] if data is None else result["z"][rsl][0]*data[1]
        out["z"] = z
    return out
