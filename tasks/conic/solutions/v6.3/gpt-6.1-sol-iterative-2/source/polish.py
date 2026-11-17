import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from model import assess, norm, repair_slack


def smooth(original, problem, D, E, cost, x, z, max_iter=8):
    if any(c["type"] == "psd" for c in problem.cones):
        return None
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    n, m = problem.n, problem.m
    s = b - A @ x
    descriptors = []
    multipliers = []
    nonnegative = []
    fill = 0
    nn_rows = []
    start = 0
    for cone in problem.cones:
        kind, dim = cone["type"], cone["dim"]
        ids = np.arange(start, start + dim)
        if kind == "zero":
            for i in ids:
                descriptors.append(("linear", np.array([i]), None))
                multipliers.append(z[i])
                nonnegative.append(False)
        elif kind == "nonnegative":
            nn_rows.append(ids)
            selected = ids[((z[ids] > 1e-8) & (s[ids] < 0.01 * z[ids]))
                           | (s[ids] < -1e-9)]
            for i in selected:
                descriptors.append(("linear", np.array([i]), None))
                multipliers.append(max(z[i], 0.0))
                nonnegative.append(True)
        elif kind == "soc":
            radius = np.linalg.norm(s[ids[1:]])
            if radius > 1e-10 and (
                (z[ids[0]] > 1e-8 and s[ids[0]] - radius < 0.01 * z[ids[0]])
                or s[ids[0]] < radius - 1e-9
            ):
                block = A[ids, :].tocsc()
                variables = np.flatnonzero(np.diff(block.indptr))
                fill += variables.size ** 2
                if fill > 2000000:
                    return None
                local = block[:, variables].toarray()
                descriptors.append(("soc", ids, (variables, local)))
                multipliers.append(max(z[ids[0]], 0.0))
                nonnegative.append(True)
        elif kind == "exponential":
            if s[ids[1]] > 1e-10 and s[ids[2]] > 1e-10 and z[ids[2]] > 1e-8:
                block = A[ids, :].tocsc()
                variables = np.flatnonzero(np.diff(block.indptr))
                fill += variables.size ** 2
                if fill > 2000000:
                    return None
                local = block[:, variables].toarray()
                descriptors.append(("exponential", ids, (variables, local)))
                multipliers.append(max(z[ids[2]], 0.0))
                nonnegative.append(True)
        start += dim
    if not descriptors:
        return None
    lam = np.array(multipliers)
    constrained = np.array(nonnegative)
    nn_rows = np.concatenate(nn_rows) if nn_rows else np.empty(0, dtype=np.int64)
    k = lam.size
    pcoo = P.tocoo()
    for iteration in range(max_iter):
        s = b - A @ x
        gr, gc, gv = [], [], []
        hr, hc, hv = [pcoo.row], [pcoo.col], [pcoo.data]
        h = np.zeros(k)
        for index, (kind, ids, local_data) in enumerate(descriptors):
            if kind == "linear":
                g = np.ones(1)
                h[index] = -s[ids[0]]
            elif kind == "soc":
                v = s[ids[1:]]
                radius = float(np.linalg.norm(v))
                if radius < 1e-12:
                    return None
                e = v / radius
                g = np.r_[1.0, -e]
                h[index] = radius - s[ids[0]]
                variables, local = local_data
                Hess = lam[index] * (np.eye(v.size) - np.outer(e, e)) / radius
                normal = local[1:, :].T @ Hess @ local[1:, :]
                rr, cc = np.meshgrid(variables, variables, indexing="ij")
                hr.append(rr.ravel())
                hc.append(cc.ravel())
                hv.append(normal.ravel())
            else:
                u, v, w = s[ids]
                if v <= 0 or w <= 0:
                    return None
                L = u / v
                if abs(L) > 500:
                    return None
                exponential = np.exp(L)
                g = np.array([-exponential, exponential * (L - 1.0), 1.0])
                h[index] = v * exponential - w
                variables, local = local_data
                direction = local[0, :] - L * local[1, :]
                normal = (lam[index] * exponential / v
                          * np.outer(direction, direction))
                rr, cc = np.meshgrid(variables, variables, indexing="ij")
                hr.append(rr.ravel())
                hc.append(cc.ravel())
                hv.append(normal.ravel())
            gr.extend([index] * ids.size)
            gc.extend(ids)
            gv.extend(g)
        G = sparse.csc_matrix((gv, (gr, gc)), shape=(k, m))
        B = (G @ A).tocsc()
        zn = np.asarray(G.T @ lam).reshape(-1)
        xo = D * x
        zo = E * zn / cost
        candidate = dict(status="solved", x=xo, s=repair_slack(original, xo), z=zo)
        good, report = assess(original, candidate)
        if os.environ.get("SOLVER_LOG"):
            print("polish", iteration, k, "min_lambda",
                  np.min(lam[constrained], initial=0.0), report, flush=True)
        if np.all(lam[constrained] >= -1e-10) and good:
            return candidate
        present = np.zeros(m, dtype=bool)
        for kind, ids, unused in descriptors:
            if kind == "linear":
                present[ids] = True
        violated = nn_rows[(s[nn_rows] < -1e-9) & ~present[nn_rows]]
        if violated.size:
            descriptors.extend(("linear", np.array([i]), None) for i in violated)
            lam = np.r_[lam, np.zeros(violated.size)]
            constrained = np.r_[constrained, np.ones(violated.size, dtype=bool)]
            k = lam.size
            continue
        H = sparse.csc_matrix(
            (np.concatenate(hv), (np.concatenate(hr), np.concatenate(hc))),
            shape=(n, n))
        K = sparse.bmat([[H, B.T], [B, None]], format="csc")
        rhs = np.r_[-P @ x - q - B.T @ lam, -h]
        try:
            regularization = np.r_[np.full(n, 1e-12), np.full(k, -1e-12)]
            lu = splu(K + sparse.diags(regularization, format="csc"),
                       diag_pivot_thresh=0.5)
            delta = lu.solve(rhs)
            for unused in range(3):
                residual = rhs - K @ delta
                if norm(residual) < 1e-12 * (1 + norm(rhs)):
                    break
                delta += lu.solve(residual)
            if not np.isfinite(delta).all():
                return None
            x += delta[:n]
            lam += delta[n:]
            negative = constrained & (lam < -1e-7)
            if np.any(negative):
                keep = ~negative
                descriptors = [descriptor for descriptor, take in zip(descriptors, keep)
                               if take]
                lam = lam[keep]
                constrained = constrained[keep]
                k = lam.size
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return None
    return None
