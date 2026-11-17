import numpy as np

from model import norm


def defect(vector, cones, dual=False):
    worst, start = 0.0, 0
    for cone in cones:
        kind, dim = cone["type"], cone["dim"]
        block = vector[start:start + dim]
        start += dim
        scale = max(1.0, norm(block))
        if kind == "zero":
            value = 0.0 if dual else norm(block) / scale
        else:
            value = max(0.0, -float(block.min())) / scale
        worst = max(worst, value)
    return worst


def solve(data):
    cones = data["metadata"]["cones"]
    if any(cone["type"] not in ("zero", "nonnegative") for cone in cones):
        return None
    q, b = data["q"], data["b"]
    n, m = q.size, b.size
    if np.count_nonzero(data["A_data"]) > m:
        return None
    pc = np.repeat(np.arange(n), np.diff(data["P_indptr"]))
    pd, pi = data["P_data"], data["P_indices"]
    pnz = pd != 0.0
    if np.any(pc[pnz] != pi[pnz]):
        return None
    diagonal = np.bincount(pc, weights=pd, minlength=n)
    if np.any(diagonal < 0):
        return None
    ac = np.repeat(np.arange(n), np.diff(data["A_indptr"]))
    ar, ad = data["A_indices"], data["A_data"]
    anz = ad != 0.0
    ac, ar, ad = ac[anz], ar[anz], ad[anz]
    counts = np.bincount(ar, minlength=m)
    if np.any(counts > 1):
        return None
    equal = np.zeros(m, dtype=bool)
    start = 0
    for cone in cones:
        if cone["type"] == "zero":
            equal[start:start + cone["dim"]] = True
        start += cone["dim"]

    def ax(x):
        return np.bincount(ar, weights=ad * x[ac], minlength=m)

    def at(z, scale=1.0):
        return np.bincount(ac, weights=(ad / scale) * z[ar], minlength=n)

    def verified(result):
        tol = 8e-9
        for key in ("x", "s", "z"):
            if key in result and not np.isfinite(result[key]).all():
                return False
        status = result["status"]
        if status == "solved":
            x, s, z = result["x"], result["s"], result["z"]
            Ax, Px, Atz = ax(x), diagonal * x, at(z)
            rp = norm(Ax + s - b) / (1 + max(norm(Ax), norm(s), norm(b)))
            rd = norm(Px + q + Atz) / (1 + max(norm(Px), norm(q), norm(Atz)))
            xPx, qx, bz = float(x @ Px), float(q @ x), float(b @ z)
            p0, d0 = 0.5 * xPx + qx, -0.5 * xPx - bz
            gap = abs(p0 - d0)
            objective_scale = max(abs(p0), abs(d0))
            finite = np.isfinite([p0, d0, float(s @ z)]).all()
            return bool(finite and max(rp, rd) <= tol
                        and (gap <= tol or gap <= tol * objective_scale)
                        and defect(s, cones) <= tol
                        and defect(z, cones, dual=True) <= tol)
        a, h = norm(ad) or 1.0, norm(diagonal) or 1.0
        if status == "primal_infeasible":
            z = result["z"]
            zn, bn = norm(z), norm(b)
            if not zn or not bn:
                return False
            z = z / zn
            margin = -float((b / bn) @ z)
            return bool(margin > 1e-8 and norm(at(z, a)) <= tol * margin
                        and defect(z, cones, dual=True) <= tol * margin)
        x, s = result["x"], result["s"]
        xn, qn = norm(x), norm(q)
        if not xn or not qn:
            return False
        x, s = x / xn, s / xn
        margin = -float((q / qn) @ x)
        return bool(margin > 1e-8
                    and norm((diagonal / h) * x) <= tol * margin
                    and norm(ax(x) / a + s / a) <= tol * margin
                    and defect(s / a, cones) <= tol * margin)

    bad = (counts == 0) & ((equal & (b != 0)) | (~equal & (b < 0)))
    if np.any(bad):
        z = np.zeros(m)
        z[bad & equal] = -np.sign(b[bad & equal])
        z[bad & ~equal] = 1.0
        result = dict(status="primal_infeasible", z=z)
        if verified(result):
            return result
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    values = b[ar] / ad
    lower = equal[ar] | (ad < 0.0)
    upper = equal[ar] | (ad > 0.0)
    np.maximum.at(lo, ac[lower], values[lower])
    np.minimum.at(hi, ac[upper], values[upper])
    lower_row, upper_row = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    responsible = lower & (values == lo[ac])
    lower_row[ac[responsible]] = ar[responsible]
    responsible = upper & (values == hi[ac])
    upper_row[ac[responsible]] = ar[responsible]
    row_value = np.zeros(m)
    row_value[ar] = ad
    conflicts = lo > hi
    if np.any(conflicts):
        j = np.argmax(np.where(conflicts, lo - hi, -np.inf))
        lr, ur = lower_row[j], upper_row[j]
        al, au = row_value[lr], row_value[ur]
        scale = max(abs(al), abs(au))
        z = np.zeros(m)
        z[lr] += -np.sign(al) * abs(au) / scale
        z[ur] += np.sign(au) * abs(al) / scale
        result = dict(status="primal_infeasible", z=z)
        if verified(result):
            return result
    unbounded = (diagonal == 0.0) & (
        ((q > 0.0) & np.isneginf(lo)) | ((q < 0.0) & np.isposinf(hi)))
    if np.any(unbounded):
        x = np.zeros(n)
        x[unbounded] = -np.sign(q[unbounded])
        result = dict(status="dual_infeasible", x=x, s=-ax(x))
        if verified(result):
            return result
    x = np.zeros(n)
    positive = diagonal > 0.0
    x[positive] = -q[positive] / diagonal[positive]
    linear = ~positive
    x[linear & (q > 0.0)] = lo[linear & (q > 0.0)]
    x[linear & (q < 0.0)] = hi[linear & (q < 0.0)]
    x = np.minimum(np.maximum(x, lo), hi)
    x[conflicts] = 0.5 * lo[conflicts] + 0.5 * hi[conflicts]
    gradient = diagonal * x + q
    z = np.zeros(m)
    use_lower = (x == lo) & (gradient > 0.0) & (lower_row >= 0)
    use_upper = (x == hi) & (gradient < 0.0) & (upper_row >= 0)
    chosen = lower_row[use_lower]
    z[chosen] = -gradient[use_lower] / row_value[chosen]
    chosen = upper_row[use_upper]
    z[chosen] = -gradient[use_upper] / row_value[chosen]
    slack = b - ax(x)
    slack[equal] = 0.0
    slack[~equal] = np.maximum(slack[~equal], 0.0)
    result = dict(status="solved", x=x, s=slack, z=z)
    return result if verified(result) else None
