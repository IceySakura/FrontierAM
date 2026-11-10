"""Barrier-free KKT refinement on an identified optimal face."""
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla


def kkt_solve(P, G, rx, rg):
    n, m = P.shape[0], G.shape[0]
    K = sparse.bmat([[P, G.T], [G, sparse.csc_matrix((m, m))]], format="csc")
    mx = np.asarray(abs(K).max(axis=0).toarray()).ravel()
    scale = 1.0 / np.sqrt(np.maximum(mx, 1e-8))
    scale = np.clip(scale, 1e-6, 1e6)
    KK = K.multiply(scale[:, None]).multiply(scale).tocsc()
    rhs = np.r_[rx, rg] * scale
    if n <= 600 and n + m <= 1800:
        delta = linalg.lstsq(KK.toarray(), rhs, cond=1e-12, lapack_driver="gelsy",
                             check_finite=False)[0]
        return delta * scale
    try:
        lu = sla.splu(KK, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1)
    except RuntimeError:
        reg = np.r_[np.full(n, 1e-12), np.full(m, -1e-12)]
        lu = sla.splu(KK + sparse.diags(reg, format="csc"),
                      permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1)
    v = lu.solve(rhs)
    for _ in range(5):
        residual = rhs - KK @ v
        if np.max(np.abs(residual), initial=0.0) < 1e-12 * max(
                1.0, np.max(np.abs(rhs), initial=0.0)):
            break
        v += lu.solve(residual)
    return v * scale


def polish_soc(p, x, z):
    from solver import norm, quality
    x, z = x.copy(), z.copy()
    slack = p.b - p.A @ x
    eq, nn, apex, normals = [], [], [], []
    zscale = max(1.0, norm(z))
    for kind, d, a, b in p.blocks:
        ids = np.arange(a, b)
        if kind == "zero":
            eq.extend(ids)
        elif kind == "nonnegative":
            active = ((slack[ids] < 1e-5 * (1 + norm(p.b)))
                      & (z[ids] > 1e-8 * zscale))
            nn.extend(ids[active])
        elif kind == "soc":
            s = slack[ids]
            rr = np.linalg.norm(s[1:])
            if z[a] < 1e-8 * zscale or s[0] - rr > 1e-4 * max(1.0, norm(s)):
                continue
            if rr < 1e-7 * max(1.0, norm(s)):
                apex.extend(ids)
            else:
                normals.append((ids, float(z[a])))
        else:
            return None
    if not normals and not apex:
        return None
    linear = np.array(eq + nn + apex, dtype=int)
    nonnegative = np.array([False] * len(eq) + [True] * len(nn) + [False] * len(apex))
    Glin = p.A[linear].tocsc()
    zl = z[linear].copy()
    if normals:
        groups = {}
        for ids, eta in normals:
            groups.setdefault(ids.size, []).append((ids, eta))
        states = [(np.array([t[0] for t in vs]), np.array([t[1] for t in vs]))
                  for _, vs in groups.items()]
    else:
        states = []
    for _ in range(5):
        sl = p.b - p.A @ x
        H = p.P.copy()
        Gs, gs, multipliers = [Glin], [p.A[linear] @ x - p.b[linear]], [zl]
        zfull = np.zeros(p.m)
        zfull[linear] = zl
        for ids, eta in states:
            vv = sl[ids[:, 1:]]
            radius = np.linalg.norm(vv, axis=1)
            if np.any(radius < 1e-12):
                return None
            u = vv / radius[:, None]
            count, width = u.shape
            V = p.A[ids[:, 1:].ravel()].tocsc()
            R = sparse.csr_matrix((u.ravel(), (
                np.repeat(np.arange(count), width), np.arange(count * width))),
                shape=(count, count * width))
            RV = R @ V
            J = p.A[ids[:, 0]] - RV
            weight = eta / radius
            H = H + V.T @ V.multiply(np.repeat(weight, width)[:, None])
            H = H - RV.T @ RV.multiply(weight[:, None])
            Gs.append(J)
            gs.append(radius - sl[ids[:, 0]])
            multipliers.append(eta)
            zfull[ids[:, 0]] = eta
            zfull[ids[:, 1:]] = -eta[:, None] * u
        G = sparse.vstack(Gs, format="csc")
        lam = np.concatenate(multipliers)
        rd = p.P @ x + p.q + p.A.T @ zfull
        try:
            delta = kkt_solve(H, G, -rd, -np.concatenate(gs))
        except (RuntimeError, ValueError):
            return None
        if not np.isfinite(delta).all():
            return None
        lam += delta[p.n:]
        if (np.any(lam[:linear.size][nonnegative] < -1e-7 * zscale)
                or np.any(lam[linear.size:] < -1e-7 * zscale)):
            return None
        x += delta[:p.n]
        zl = lam[:linear.size]
        zl[nonnegative] = np.maximum(zl[nonnegative], 0)
        pos = linear.size
        for ids, eta in states:
            eta[:] = np.maximum(lam[pos:pos + ids.shape[0]], 0)
            pos += ids.shape[0]
        sl = p.b - p.A @ x
        zfull.fill(0)
        zfull[linear] = zl
        for ids, eta in states:
            v = sl[ids[:, 1:]]
            radius = np.linalg.norm(v, axis=1)
            if np.any(radius < 1e-12):
                return None
            zfull[ids[:, 0]] = eta
            zfull[ids[:, 1:]] = -eta[:, None] * v / radius[:, None]
        r = {"status": "solved", "x": x.copy(), "s": sl, "z": zfull.copy()}
        if quality(p, r):
            return r
    return None
