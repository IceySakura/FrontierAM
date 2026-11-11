"""Primal-dual Newton reduction for monotone exponential epigraph sections."""
import time
import numpy as np


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def solve(P, q, A, b, cones, la, check):
    started = time.monotonic()
    exp = cones.exp
    k, n = len(exp), len(q)
    if (k < 50 or len(cones.eq) or cones.soc or cones.psd or
            not len(cones.pos) or len(cones.pos) > 25000):
        return None
    ar = A.tocsr()
    counts = np.diff(ar.indptr)
    ur, vr, wr = exp.T
    if np.any(counts[vr]) or np.any(b[vr] <= 0) or np.any(counts[wr] != 1):
        return None
    wvars = ar.indices[ar.indptr[wr]]
    if len(np.unique(wvars)) != k or np.any(np.diff(P.indptr)[wvars]):
        return None
    a = ar.data[ar.indptr[wr]]
    if np.any(a == 0):
        return None
    h = -1.0 / a
    mask = np.ones(n, dtype=bool)
    mask[wvars] = False
    free = np.flatnonzero(mask)
    nf, ml = len(free), len(cones.pos)
    if not 0 < nf <= 200 or nf * k > 5000000 or nf * ml > 5000000:
        return None
    U = A[ur].tocsc()
    if U[:, wvars].nnz:
        return None
    Lin = A[cones.pos].tocsc()
    W = Lin[:, wvars].tocsc()
    W.data *= np.repeat(h, np.diff(W.indptr))
    qw = q[wvars] * h
    if np.any(qw < 0):
        return None
    v, bw = b[vr], b[wr]
    C = -U[:, free].toarray() / v[:, None]
    base = b[ur] / v
    if np.max(np.abs(base)) > 600:
        return None
    AL = Lin[:, free].toarray()
    BL = b[cones.pos] + W @ bw
    linear_rows = cones.pos
    if np.any(W.data < 0):
        has_positive = np.zeros(ml, dtype=bool)
        has_positive[W.indices[W.data > 0]] = True
        has_negative = np.zeros(ml, dtype=bool)
        has_negative[W.indices[W.data < 0]] = True
        redundant = has_negative & ~has_positive & (BL >= 0) & ~np.any(AL != 0, axis=1)
        keep = np.flatnonzero(~redundant)
        W, AL, BL = W[keep].tocsc(), AL[keep], BL[keep]
        linear_rows = linear_rows[keep]
        ml = len(keep)
    if not ml or ml > 500 or np.any(W.data < 0):
        return None
    Q = P[free][:, free].toarray()
    qf = q[free].copy()
    d = 1.0 / np.maximum(1.0, np.maximum(np.max(np.abs(C), axis=0),
                                        np.max(np.abs(AL), axis=0)))
    C *= d
    AL *= d
    Q *= d[:, None] * d[None, :]
    qf *= d
    cost = 1.0 / max(1.0, norm(Q), norm(qf), norm(qw))
    Q *= cost
    qf *= cost
    qw *= cost
    w0 = v * np.exp(base)
    row_scale = 1.0 / np.maximum(1.0, np.maximum(np.abs(BL), np.abs(W @ w0)))
    AL *= row_scale[:, None]
    W.data *= row_scale[W.indices]
    if hasattr(W, "_dense_cache"):
        del W._dense_cache
    BL *= row_scale
    y = np.zeros(nf)
    f0 = W @ w0
    s = np.maximum(1.0, BL - f0)
    z = np.ones(ml)
    wall_limit = .65 if nf * k >= 1000000 else .3
    last_out = None

    def evaluate(yy, zz):
        t = base + C @ yy
        if np.max(np.abs(t)) > 650 or not np.isfinite(t).all():
            return None
        w = v * np.exp(t)
        F = AL @ yy + W @ w
        J = AL + W @ (w[:, None] * C)
        grad0 = Q @ yy + qf + C.T @ (qw * w)
        rd = grad0 + J.T @ zz
        return t, w, F, J, rd

    def candidate(yy, zz, ev):
        t, w = ev[:2]
        xx = np.zeros(n)
        xx[free] = d * yy
        xx[wvars] = h * (w - bw)
        ss = b - A @ xx
        dual = np.zeros(len(b))
        dual[linear_rows] = row_scale * zz / cost
        lam = (qw + W.T @ zz) / cost
        et = w / v
        dual[exp] = np.stack((-lam * et, lam * et * (t - 1.0), lam), axis=1)
        return {"status": "solved", "x": xx, "s": ss, "z": dual}

    def step_bound(vv, dv):
        bad = dv < 0
        return min(1.0, float(np.min(-vv[bad] / dv[bad]))) if np.any(bad) else 1.0

    try:
        for iteration in range(80):
            if time.monotonic() - started > wall_limit:
                break
            ev = evaluate(y, z)
            if ev is None:
                break
            t, w, F, J, rd = ev
            rp = F + s - BL
            mu = float(s @ z) / ml
            merit = max(norm(rd), norm(rp), mu)
            if max(norm(rd), norm(rp), mu) < 1e-4:
                last_out = candidate(y, z, ev)
                if check(last_out):
                    return last_out
            lam = qw + W.T @ z
            H = Q + C.T @ ((lam * w)[:, None] * C)
            invh = z / s
            G = H + (J.T * invh) @ J
            delta = 1e-13 * max(1.0, norm(G))
            Gfactor = G.copy()
            Gfactor[np.diag_indices(nf)] += delta
            factor = la.cho_factor(Gfactor, lower=True, check_finite=False)

            def direction(rc):
                rhs = -rd - J.T @ (invh * (rp + rc))
                dy = la.cho_solve(factor, rhs, check_finite=False)
                for _ in range(3):
                    r = rhs - G @ dy
                    if norm(r) <= 1e-13 * max(1e-8, norm(rhs)):
                        break
                    dy += la.cho_solve(factor, r, check_finite=False)
                ds = -rp - J @ dy
                dz = invh * (J @ dy + rp + rc)
                return dy, ds, dz

            ay, ass, az = direction(-s)
            aa = min(step_bound(s, ass), step_bound(z, az))
            ma = float((s + aa * ass) @ (z + aa * az)) / ml
            sigma = min(.9, max(0.0, ma / mu) ** 3)
            if merit < 1e-3:
                trial_ev = evaluate(y + ay, z + az)
                if trial_ev is not None:
                    trial = candidate(y + ay, z + az, trial_ev)
                    if check(trial):
                        return trial
            dy, ds, dz = direction(-s + sigma * mu / z - ass * az / z)
            accepted = False
            for attempt in range(2):
                alpha = .99 * min(step_bound(s, ds), step_bound(z, dz))
                for _ in range(24):
                    yn, sn, zn = y + alpha * dy, s + alpha * ds, z + alpha * dz
                    en = evaluate(yn, zn)
                    if en is not None:
                        pn = en[2] + sn - BL
                        mn = max(norm(en[4]), norm(pn), float(sn @ zn) / ml)
                        if mn < (1 - .01 * alpha) * merit or mn < 1e-11:
                            y, s, z = yn, sn, zn
                            accepted = True
                            break
                    alpha *= .5
                if accepted:
                    break
                dy, ds, dz = direction(-s + .2 * mu / z)
            if not accepted:
                break
    except (ValueError, RuntimeError, la.LinAlgError, FloatingPointError, OverflowError):
        return None
    return None
