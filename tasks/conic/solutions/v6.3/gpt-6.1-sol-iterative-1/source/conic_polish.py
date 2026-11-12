"""Local Newton refinement on identified smooth cone boundaries."""

import numpy as np
from scipy import sparse

from linear import SparseFactors
from polish import correction, norm


def candidates(P, q, A, b, cs, x, z, tolerance=.01, max_steps=4):
    if cs.psd:
        return
    n, m, ne = len(x), len(b), cs.neq
    if not n:
        return
    slack = b - A @ x
    rownorm = np.asarray(abs(A).max(axis=1).toarray()).ravel()
    scale = np.maximum(np.maximum(rownorm, np.abs(b)), 1e-12)
    dual_scale = max(norm(q), norm(P @ x), 1e-8)
    nn = ne + np.arange(cs.nnonneg)
    active_nn = nn[(slack[nn] <= tolerance * scale[nn]) &
                   (z[nn] * rownorm[nn] > 1e-7 * dual_scale)]
    fixed_parts = [np.arange(ne), active_nn]
    descriptors, lam_parts = [], []
    ncurv = nsmooth = 0
    fill_estimate = 0.
    counts = np.bincount(A.indices, minlength=m)
    for sl, k, d in cs.soc:
        rows = (ne + np.arange(sl.start, sl.stop)).reshape(k, d)
        s, zz = slack[rows], z[rows]
        r = np.linalg.norm(s[:, 1:], axis=1)
        blockscale = np.maximum(1., np.max(np.abs(s), axis=1))
        strength = zz[:, 0] * np.maximum(np.max(rownorm[rows], axis=1), 1e-12)
        active = (s[:, 0] - r <= tolerance * blockscale) & (strength > 1e-7 * dual_scale)
        apex = active & (r < 1e-6 * blockscale) & (np.abs(s[:, 0]) < 1e-5 * blockscale)
        if np.any(apex):
            fixed_parts.append(rows[apex].ravel())
        take = active & ~apex
        if np.any(take):
            rr = rows[take]
            ncurv += len(rr) * (d - 1)**2
            fill_estimate += float(np.sum(np.sum(counts[rr[:, 1:]], axis=1).astype(float)**2))
            descriptors.append({"kind": "soc", "rows": rr, "start": nsmooth, "d": d})
            lam_parts.append(zz[take, 0])
            nsmooth += len(rr)
    domain_rows = np.empty((0, 3), dtype=int)
    if cs.nexp:
        domain_rows = (ne + np.arange(cs.exps.start, cs.exps.stop)).reshape(-1, 3)
        s, zz = slack[domain_rows], z[domain_rows]
        blockscale = np.maximum(1., np.max(np.abs(s), axis=1))
        if np.any(s[:, 1] <= 1e-14 * blockscale) or np.any(s[:, 2] <= 1e-14 * blockscale):
            return
        f = s[:, 1] * (np.log(s[:, 2]) - np.log(s[:, 1])) - s[:, 0]
        strength = -zz[:, 0] * np.maximum(np.max(rownorm[domain_rows], axis=1), 1e-12)
        active = (f <= tolerance * blockscale) & (strength > 1e-7 * dual_scale)
        if np.any(active):
            rr = domain_rows[active]
            ncurv += 9 * len(rr)
            fill_estimate += float(np.sum(np.sum(counts[rr[:, 1:]], axis=1).astype(float)**2))
            descriptors.append({"kind": "exponential", "rows": rr, "start": nsmooth, "d": 3})
            lam_parts.append(-zz[active, 0])
            nsmooth += len(rr)
    if ncurv > 2_000_000 or min(float(n)**2, fill_estimate) > 10_000_000:
        return
    fixed_rows = np.concatenate(fixed_parts)
    fixed = A[fixed_rows, :].tocsc()
    nf = len(fixed_rows)
    multipliers = np.r_[z[fixed_rows], np.concatenate(lam_parts) if lam_parts else np.empty(0)]
    bounded = np.r_[ne + np.arange(len(active_nn)), nf + np.arange(nsmooth)]
    zero_rows = np.r_[np.arange(ne), np.concatenate(fixed_parts[2:]) if len(fixed_parts) > 2 else np.empty(0, dtype=int)]
    factory = SparseFactors(native=False)

    def geometry(value, lambdas, curvature=True):
        gr, gc, gv, dr, dc, dv, values = [], [], [], [], [], [], []
        for item in descriptors:
            rows, start, d = item["rows"], item["start"], item["d"]
            ss = value[rows]
            ll = lambdas[start:start + len(rows)]
            if item["kind"] == "soc":
                radius = np.linalg.norm(ss[:, 1:], axis=1)
                if np.any(radius <= 1e-14):
                    return None
                unit = ss[:, 1:] / radius[:, None]
                grad = np.c_[np.ones(len(rows)), -unit]
                f = ss[:, 0] - radius
                if curvature:
                    rr = rows[:, 1:]
                    D = (np.eye(d - 1)[None, :, :] - unit[:, :, None] * unit[:, None, :])
                    D *= (ll / radius)[:, None, None]
            else:
                a, y, w = ss.T
                if np.any(y <= 0) or np.any(w <= 0):
                    return None
                lr, ratio = np.log(w) - np.log(y), y / w
                grad = np.c_[-np.ones(len(rows)), lr - 1, ratio]
                f = y * lr - a
                if curvature:
                    rr = rows
                    v = np.c_[np.zeros(len(rows)), np.ones(len(rows)), -ratio]
                    D = (ll / y)[:, None, None] * v[:, :, None] * v[:, None, :]
            gr.append(np.repeat(np.arange(start, start + len(rows)), d))
            gc.append(rows.ravel())
            gv.append(grad.ravel())
            values.append(f)
            if curvature:
                width = rr.shape[1]
                dr.append(np.broadcast_to(rr[:, :, None], (len(rr), width, width)).ravel())
                dc.append(np.broadcast_to(rr[:, None, :], (len(rr), width, width)).ravel())
                dv.append(D.ravel())
        G = sparse.csc_matrix((np.concatenate(gv) if gv else np.empty(0),
                               (np.concatenate(gr) if gr else np.empty(0, dtype=int),
                                np.concatenate(gc) if gc else np.empty(0, dtype=int))),
                              shape=(nsmooth, m))
        if curvature:
            D = sparse.csc_matrix((np.concatenate(dv) if dv else np.empty(0),
                                   (np.concatenate(dr) if dr else np.empty(0, dtype=int),
                                    np.concatenate(dc) if dc else np.empty(0, dtype=int))), shape=(m, m))
            D.eliminate_zeros()
        else:
            D = None
        if not np.isfinite(G.data).all() or (D is not None and not np.isfinite(D.data).all()):
            return None
        return G, D, np.concatenate(values) if values else np.empty(0)

    for _ in range(max_steps):
        raw_s = b - A @ x
        data = geometry(raw_s, multipliers[nf:])
        if data is None:
            return
        G, D, f = data
        dual = np.zeros(m)
        dual[fixed_rows] = multipliers[:nf]
        dual += G.T @ multipliers[nf:]
        jacobian = sparse.vstack((fixed, G @ A), format="csc")
        H = (P + A.T @ D @ A).tocsc()
        if not np.isfinite(H.data).all():
            return
        rhs = np.r_[-P @ x - q - A.T @ dual, b[fixed_rows] - fixed @ x, f]
        try:
            step = correction(H, jacobian, rhs, factory)
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return
        dx, dm = step[:n], step[n:]
        alpha = 1.
        if len(bounded):
            neg = dm[bounded] < 0
            if np.any(neg):
                alpha = min(alpha, .99 * float(np.min(-multipliers[bounded][neg] / dm[bounded][neg])))
        if len(domain_rows):
            ds = (-A @ dx)[domain_rows[:, 1:]]
            old = raw_s[domain_rows[:, 1:]]
            neg = ds < 0
            if np.any(neg):
                alpha = min(alpha, .99 * float(np.min(-old[neg] / ds[neg])))
        if not np.isfinite(alpha) or alpha < 1e-5:
            return
        x = x + alpha * dx
        multipliers += alpha * dm
        raw_s = b - A @ x
        data = geometry(raw_s, multipliers[nf:], False)
        if data is None:
            return
        dual = np.zeros(m)
        dual[fixed_rows] = multipliers[:nf]
        dual += data[0].T @ multipliers[nf:]
        slack = raw_s.copy()
        slack[zero_rows] = 0.
        slack[nn] = np.maximum(slack[nn], 0.)
        yield {"status": "solved", "x": x.copy(), "s": slack, "z": dual}
