import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from common import blocks, norm, assess


def refine(P, q, A, b, cones, result):
    if result.get("status") != "solved" or any(c["type"] == "psd" for c in cones):
        return result
    n, m = len(q), len(b)
    if n > 5000 or m > 50000:
        return result
    x, z = result["x"].copy(), result["z"].copy()
    s = b - A @ x
    ar = A.tocsr()
    linrows, nonlinear = [], []
    zscale = max(1.0, norm(z))
    for kind, d, i, j in blocks(cones):
        if kind == "zero":
            linrows.extend(range(i, j))
        elif kind == "nonnegative":
            ids = np.arange(i, j)
            active = (z[i:j] > 1e-9 * zscale) & (s[i:j] < 1e-3 * (1 + np.abs(b[i:j])))
            linrows.extend(ids[active])
        elif kind == "soc":
            v = s[i:j]
            if z[i] > 1e-9 * zscale and v[0] - np.linalg.norm(v[1:]) < 1e-3 * (1 + abs(v[0])):
                nonlinear.append((kind, i, j, ar[i:j], z[i]))
        else:
            v = s[i:j]
            if -z[i] > 1e-9 * zscale and v[1] > 1e-12 and v[2] > 1e-12:
                nonlinear.append((kind, i, j, ar[i:j], -z[i]))
    linrows = np.asarray(linrows, dtype=int)
    Al = ar[linrows].tocsc()
    multipliers = np.r_[z[linrows], [entry[4] for entry in nonlinear]]
    best = result
    _, besterr = assess(P, q, A, b, cones, best)
    for iteration in range(8):
        s = b - A @ x
        H = P.copy()
        gradients, values = [], []
        valid = True
        for k, (kind, i, j, B, old) in enumerate(nonlinear):
            v = s[i:j]
            lam = multipliers[len(linrows) + k]
            if kind == "soc":
                length = np.linalg.norm(v[1:])
                if length <= 1e-14:
                    valid = False
                    break
                u = v[1:] / length
                normal = np.r_[-1.0, u]
                value = length - v[0]
                Bt = B[1:]
                h = np.asarray(Bt.T @ u)
                nz = np.flatnonzero(h)
                hr, hc = np.meshgrid(nz, nz, indexing="ij")
                outer = sparse.coo_matrix(((h[nz, None] * h[None, nz]).ravel(),
                                           (hr.ravel(), hc.ravel())), shape=(n, n))
                H += lam / length * ((Bt.T @ Bt) - outer)
            else:
                u, vv, w = v
                if vv <= 0 or w <= 0:
                    valid = False
                    break
                logratio = np.log(w) - np.log(vv)
                normal = np.array([1., 1. - logratio, -vv / w])
                value = u - vv * logratio
                h = np.asarray(B.T @ np.array([0., 1 / np.sqrt(vv), -np.sqrt(vv) / w]))
                nz = np.flatnonzero(h)
                hr, hc = np.meshgrid(nz, nz, indexing="ij")
                H += sparse.coo_matrix(((lam * h[nz, None] * h[None, nz]).ravel(),
                                        (hr.ravel(), hc.ravel())), shape=(n, n))
            gradients.append(sparse.csr_matrix(-normal.reshape(1, -1)) @ B)
            values.append(value)
        if not valid:
            break
        G = sparse.vstack([Al] + gradients, format="csc")
        residual_d = P @ x + q + G.T @ multipliers
        residual_p = np.r_[Al @ x - b[linrows], values]
        K = sparse.bmat([[H, G.T], [G, None]], format="csc")
        rhs = -np.r_[residual_d, residual_p]
        reg = np.r_[np.full(n, 1e-12), np.full(G.shape[0], -1e-12)]
        try:
            lu = splu(K + sparse.diags(reg))
            step = lu.solve(rhs)
            for _ in range(2):
                step += lu.solve(rhs - K @ step)
        except (RuntimeError, ValueError):
            break
        if not np.isfinite(step).all():
            break
        x += step[:n]
        multipliers += step[n:]
        s = b - A @ x
        z = np.zeros(m)
        z[linrows] = multipliers[:len(linrows)]
        for k, (kind, i, j, B, old) in enumerate(nonlinear):
            v = s[i:j]
            if kind == "soc":
                length = np.linalg.norm(v[1:])
                normal = np.r_[-1.0, v[1:] / max(length, 1e-300)]
            elif v[1] > 0 and v[2] > 0:
                normal = np.array([1., 1 - np.log(v[2] / v[1]), -v[1] / v[2]])
            else:
                valid = False
                break
            z[i:j] = -multipliers[len(linrows) + k] * normal
        if not valid:
            break
        for kind, d, i, j in blocks(cones):
            if kind == "zero":
                s[i:j] = 0
        candidate = {"status": "solved", "x": x.copy(), "s": s, "z": z}
        ok, err = assess(P, q, A, b, cones, candidate)
        if ok:
            return candidate
        if err < besterr:
            best, besterr = candidate, err
        if err > 10 * max(besterr, 1e-12):
            break
    return best
