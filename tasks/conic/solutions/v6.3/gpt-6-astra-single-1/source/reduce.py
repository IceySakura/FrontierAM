import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from common import blocks, cone_def


def reduce_problem(P, q, A, b, cones):
    """Exact affine eliminations and orthogonal cone isomorphisms."""
    n0, m0 = len(q), len(b)
    P0, q0, A0, b0 = P, q, A, b
    eq = np.zeros(m0, dtype=bool)
    for k, d, i, j in blocks(cones):
        if k == "zero":
            eq[i:j] = True
    keep = np.arange(n0)
    fixed = np.zeros(n0)
    eliminations = []
    ar = A.tocsr()
    for _ in range(12):
        counts = np.diff(ar.indptr)
        rows = np.flatnonzero(eq & (counts == 1))
        if not len(rows):
            break
        positions = ar.indptr[rows]
        variables = ar.indices[positions]
        variables, unique = np.unique(variables, return_index=True)
        rows, positions = rows[unique], positions[unique]
        coefficients = ar.data[positions]
        values = b[rows] / coefficients
        old_variables = keep[variables]
        fixed[old_variables] = values
        eliminations.append((rows, old_variables, coefficients))
        b = b - ar[:, variables] @ values
        b[rows] = 0.0
        eq[rows] = False
        retain = np.ones(len(keep), dtype=bool)
        retain[variables] = False
        keep = keep[retain]
        ar = ar[:, retain].tocsr()
    if eliminations:
        A = ar.tocsc()
        q = (q0 + P0 @ fixed)[keep]
        P = P0[keep][:, keep].tocsc()
    counts = np.diff(ar.indptr)
    base_slack = np.zeros(m0)
    src, dst, val = [], [], []
    shifts, lifts = [], []
    newcones = []
    row = 0
    changed = bool(eliminations)

    def select(indices, kind, dim):
        nonlocal row
        indices = np.asarray(indices, dtype=int)
        src.append(indices)
        dst.append(np.arange(row, row + len(indices)))
        val.append(np.ones(len(indices)))
        row += len(indices)
        if kind in ("zero", "nonnegative") and newcones and newcones[-1]["type"] == kind:
            newcones[-1]["dim"] += dim
        else:
            newcones.append({"type": kind, "dim": int(dim)})

    def transform2(indices, psd):
        nonlocal row, changed
        a = np.asarray(indices)
        if psd:
            src.append(a[[0, 2, 0, 2, 1]])
            dst.append(row + np.array([0, 0, 1, 1, 2]))
            val.append(np.array([1, 1, 1, -1, np.sqrt(2.0)]) / np.sqrt(2.0))
            newcones.append({"type": "soc", "dim": 3})
            row += 3
        else:
            src.append(a[[0, 1, 0, 1]])
            dst.append(row + np.array([0, 0, 1, 1]))
            val.append(np.array([1, 1, 1, -1]) / np.sqrt(2.0))
            newcones.append({"type": "nonnegative", "dim": 2})
            row += 2
        changed = True

    for k, d, i, j in blocks(cones):
        if k in ("zero", "nonnegative"):
            indices = np.arange(i, j)
            discard = (counts[i:j] == 0) & ((b[i:j] == 0) if k == "zero" else (b[i:j] >= 0))
            if np.any(discard):
                changed = True
                base_slack[indices[discard]] = b[indices[discard]]
                indices = indices[~discard]
            if len(indices):
                select(indices, k, len(indices))
        elif not np.any(counts[i:j]) and cone_def(b[i:j], [{"type": k, "dim": d}]) == 0:
            changed = True
            base_slack[i:j] = b[i:j]
        elif k == "soc" and not np.any(counts[i+1:j]):
            length = np.linalg.norm(b[i+1:j])
            if np.isfinite(length):
                target = row
                select([i], "nonnegative", 1)
                shifts.append((target, -length))
                base_slack[i:j] = b[i:j]
                base_slack[i] = length
                if length:
                    lifts.append((target, np.arange(i+1, j), -b[i+1:j] / length))
                changed = True
            else:
                select(np.arange(i, j), k, d)
        elif k == "exponential" and counts[i] == 0 and counts[i+1] == 0 and b[i+1] > 0:
            ratio = b[i] / b[i+1]
            if -700 < ratio < 700:
                exponential = np.exp(ratio)
                bound = b[i+1] * exponential
                target = row
                select([i+2], "nonnegative", 1)
                shifts.append((target, -bound))
                base_slack[i:j] = [b[i], b[i+1], bound]
                lifts.append((target, np.array([i, i+1]),
                              np.array([-exponential, exponential * (ratio - 1)])))
                changed = True
            else:
                select(np.arange(i, j), k, d)
        elif k == "exponential" and counts[i+1] == 0 and counts[i+2] == 0 and b[i+1] > 0 and b[i+2] > 0:
            ratio = np.log(b[i+2]) - np.log(b[i+1])
            bound = b[i+1] * ratio
            target = row
            select([i], "nonnegative", 1)
            val[-1][0] = -1.0
            shifts.append((target, bound))
            base_slack[i:j] = [bound, b[i+1], b[i+2]]
            lifts.append((target, np.array([i+1, i+2]),
                          np.array([ratio - 1, b[i+1] / b[i+2]])))
            changed = True
        elif k == "soc" and d == 2:
            transform2(np.arange(i, j), False)
        elif k == "psd":
            if d == 1:
                select([i], "nonnegative", 1)
                changed = True
            elif d == 2:
                transform2(np.arange(i, j), True)
            else:
                used = np.flatnonzero((counts[i:j] != 0) | (b[i:j] != 0))
                col = ((np.sqrt(8.0 * used + 1) - 1) / 2).astype(np.int64)
                rr = used - col * (col + 1) // 2
                graph = sparse.coo_matrix((np.ones(len(used)), (rr, col)), shape=(d, d))
                nc, labels = connected_components(graph, directed=False)
                if nc == 1:
                    select(np.arange(i, j), k, d)
                else:
                    changed = True
                    order = np.argsort(labels, kind="stable")
                    borders = np.r_[0, np.cumsum(np.bincount(labels))]
                    for h in range(nc):
                        vertices = np.sort(order[borders[h]:borders[h+1]])
                        dd = len(vertices)
                        cc, rr = np.tril_indices(dd)
                        r, c = vertices[rr], vertices[cc]
                        ids = i + c * (c + 1) // 2 + r
                        if dd == 1:
                            select(ids, "nonnegative", 1)
                        elif dd == 2:
                            transform2(ids, True)
                        else:
                            select(ids, "psd", dd)
        else:
            select(np.arange(i, j), k, d)
    if changed:
        C = sparse.coo_matrix((np.concatenate(val) if val else np.empty(0),
                               (np.concatenate(dst) if dst else np.empty(0, dtype=int),
                                np.concatenate(src) if src else np.empty(0, dtype=int))),
                              shape=(row, m0)).tocsc()
        A, b = (C @ A).tocsc(), C @ b
        for index, amount in shifts:
            b[index] += amount
    else:
        C = None
        newcones = cones

    def restore(result, base_changed=changed):
        status = result["status"]
        if status == "unresolved" or not base_changed:
            return result
        out = {"status": status}
        if "x" in result:
            x = fixed.copy() if status == "solved" else np.zeros(n0)
            x[keep] = result["x"]
            out["x"] = x
        if "s" in result:
            out["s"] = C.T @ result["s"]
            if status == "solved":
                out["s"] += base_slack
        if "z" in result:
            z = C.T @ result["z"]
            for index, targets, coefficients in lifts:
                z[targets] += coefficients * result["z"][index]
            if eliminations:
                residual = A0.T @ z
                if status == "solved":
                    residual += P0 @ out["x"] + q0
                for rows, variables, coefficients in reversed(eliminations):
                    values = -residual[variables] / coefficients
                    z[rows] += values
                    residual += A0[rows].T @ values
            out["z"] = z
        return out

    problem = (P, q, A, b, newcones)
    recoveries = []
    for _ in range(3):
        problem, recovery, did_change = eliminate_equalities(*problem)
        if not did_change:
            break
        recoveries.append(recovery)
        changed = True

    def restore_all(result):
        for recovery in reversed(recoveries):
            result = recovery(result)
        return restore(result)

    return problem, restore_all, changed


def eliminate_equalities(P, q, A, b, cones):
    n, m = len(q), len(b)
    original = (P, q, A, b, cones)
    if n > 100000 or A.nnz > 1000000 or not n:
        return original, None, False
    eq = []
    for k, d, i, j in blocks(cones):
        if k == "zero":
            eq.extend(range(i, j))
    eq = np.asarray(eq, dtype=int)
    if not len(eq):
        return original, None, False
    E = A[eq].tocsr()
    eqcounts = np.bincount(E.indices, minlength=n)
    rowcounts = np.diff(E.indptr)
    erows = np.repeat(np.arange(len(eq)), rowcounts)
    candidate = (eqcounts[E.indices] == 1) & (rowcounts[erows] <= 4)
    positions = np.flatnonzero(candidate)
    if not len(positions):
        return original, None, False
    # A globally sparse pivot avoids introducing unnecessary fill.
    costs = np.diff(A.indptr)[E.indices[positions]]
    permutation = np.lexsort((costs, erows[positions]))
    positions = positions[permutation]
    _, unique = np.unique(erows[positions], return_index=True)
    positions = positions[unique]
    rowids, pivots = eq[erows[positions]], E.indices[positions]
    coeff = E.data[positions]
    pivot_rows = A[rowids].tocsr()
    rowmax = np.asarray(abs(pivot_rows).max(axis=1).toarray()).ravel()
    good = abs(coeff) >= 1e-3 * rowmax
    rowids, pivots, coeff = rowids[good], pivots[good], coeff[good]
    if not len(pivots):
        return original, None, False
    keep = np.ones(n, dtype=bool)
    keep[pivots] = False
    retained = np.flatnonzero(keep)
    R = -(sparse.diags(1 / coeff) @ A[rowids][:, keep]).tocoo()
    tr = np.r_[retained, pivots[R.row]]
    tc = np.r_[np.arange(len(retained)), R.col]
    tv = np.r_[np.ones(len(retained)), R.data]
    T = sparse.coo_matrix((tv, (tr, tc)), shape=(n, len(retained))).tocsc()
    offset = np.zeros(n)
    offset[pivots] = b[rowids] / coeff
    rowkeep = np.ones(m, dtype=bool)
    rowkeep[rowids] = False
    rr = np.flatnonzero(rowkeep)
    output_cones = []
    for k, d, i, j in blocks(cones):
        dim = int(np.count_nonzero(rowkeep[i:j])) if k == "zero" else d
        if dim:
            output_cones.append({"type": k, "dim": dim})
    Pnew = (T.T @ P @ T).tocsc()
    qnew = np.asarray(T.T @ (q + P @ offset))
    Anew = (A[rr] @ T).tocsc()
    bnew = b[rr] - A[rr] @ offset
    Anew.eliminate_zeros()
    Pnew.eliminate_zeros()

    def recover(result):
        status = result["status"]
        if status == "unresolved":
            return result
        out = {"status": status}
        if "x" in result:
            out["x"] = T @ result["x"]
            if status == "solved":
                out["x"] += offset
        if "s" in result:
            out["s"] = np.zeros(m)
            out["s"][rr] = result["s"]
        if "z" in result:
            z = np.zeros(m)
            z[rr] = result["z"]
            residual = A.T @ z
            if status == "solved":
                residual += P @ out["x"] + q
            z[rowids] = -residual[pivots] / coeff
            out["z"] = z
        return out

    return (Pnew, qnew, Anew, bnew, output_cones), recover, True
