"""Eliminate separable exponential epigraphs into log-sum-exp constraints."""
import os
import sys
import time

import numpy as np
from scipy import linalg, sparse

from solver import norm, quality


def solve_geometric(p, maxiter=100, start=None, phase=False):
    if p.P.nnz or np.any(p.equal):
        return {"status": "unresolved"}
    if any(k not in ("nonnegative", "exponential") for k, _, _, _ in p.blocks):
        return {"status": "unresolved"}
    ex = np.array([np.arange(a, b) for k, _, a, b in p.blocks if k == "exponential"])
    nn = np.array([i for k, _, a, b in p.blocks if k == "nonnegative"
                   for i in range(a, b)], dtype=int)
    if not ex.size or not nn.size:
        return {"status": "unresolved"}
    if p.A[ex[:, 1]].nnz or np.any(p.b[ex[:, 1]] <= 0):
        return {"status": "unresolved"}
    Aw = p.A[ex[:, 2]].tocsr()
    if np.any(np.diff(Aw.indptr) != 1) or np.any(Aw.data >= 0):
        return {"status": "unresolved"}
    wc, gamma = Aw.indices, -Aw.data
    if np.unique(wc).size != wc.size or norm(p.q[wc]) > 0:
        return {"status": "unresolved"}
    free = np.ones(p.n, dtype=bool)
    free[wc] = False
    free = np.flatnonzero(free)
    n, m, nt = free.size, nn.size, ex.shape[0]
    if n > 1500 or m > 20000 or not n:
        return {"status": "unresolved"}
    if p.A[ex[:, 0]][:, wc].nnz:
        return {"status": "unresolved"}
    W = p.A[nn][:, wc].multiply(1.0 / gamma).tocsc()
    W.eliminate_zeros()
    if np.any(np.diff(W.indptr) != 1) or np.any(W.data <= 0):
        return {"status": "unresolved"}
    bound = p.b[nn] + W @ p.b[ex[:, 2]]
    C = p.A[nn][:, free].toarray()
    group, coefficient = W.indices, W.data
    counts = np.bincount(group, minlength=m)
    raw = np.any(C != 0, axis=1) | (bound <= 0) | (counts == 0)
    logrows = ~raw
    y = p.b[ex[:, 1]]
    B = p.A[ex[:, 0]][:, free].multiply((-1.0 / y)[:, None]).tocsr()
    offset = p.b[ex[:, 0]] / y + np.log(y) + np.log(coefficient)
    rawscale = np.maximum(1.0, np.maximum(np.abs(bound), np.sum(np.abs(C), axis=1)))
    q = p.q[free]
    qscale = max(1.0, norm(q))
    q = q / qscale
    d = np.zeros(n) if start is None else np.asarray(start).copy()
    started = time.monotonic()
    debug = os.environ.get("SOLVER_DEBUG")
    termcols = np.arange(nt)

    def values(v):
        term = offset + B @ v
        mx = np.full(m, -np.inf)
        np.maximum.at(mx, group, term)
        ww = np.exp(term - mx[group])
        sums = np.bincount(group, weights=ww, minlength=m)
        logs = mx.copy()
        nonempty = sums > 0
        logs[nonempty] += np.log(sums[nonempty])
        ww /= sums[group]
        T = sparse.csr_matrix((ww, (group, termcols)), shape=(m, nt))
        Jlog = (T @ B).toarray()
        g = np.empty(m)
        J = Jlog.copy()
        g[logrows] = logs[logrows] - np.log(bound[logrows])
        rowfactor = np.ones(m)
        if np.any(raw):
            F = np.exp(np.minimum(logs[raw], 700))
            g[raw] = (F + C[raw] @ v - bound[raw]) / rawscale[raw]
            J[raw] = (F[:, None] * Jlog[raw] + C[raw]) / rawscale[raw, None]
            rowfactor[raw] = F / rawscale[raw]
        return g, J, ww, logs, term, rowfactor, Jlog

    def recover(v, lam, weights, logs, term, rowfactor):
        logf = term - np.log(coefficient)
        if np.max(np.abs(logf)) > 690 or np.max(np.abs(logs[counts > 0])) > 690:
            return {"status": "unresolved"}
        f = np.exp(logf)
        xx = np.zeros(p.n)
        xx[free] = v
        xx[wc] = (f - p.b[ex[:, 2]]) / gamma
        zz = np.zeros(p.m)
        lam = lam * qscale
        zz[nn[logrows]] = lam[logrows] * np.exp(-logs[logrows])
        zz[nn[raw]] = lam[raw] / rawscale[raw]
        mass = lam[group] * rowfactor[group] * weights
        ratio = p.b[ex[:, 0]] / y + B @ v
        zz[ex[:, 0]] = -mass / y
        zz[ex[:, 1]] = mass / y * (ratio - 1.0)
        zz[ex[:, 2]] = mass / f
        return {"status": "solved", "x": xx, "s": p.b - p.A @ xx, "z": zz}

    def hessian(lam, weights, rowfactor, Jlog):
        weighted = B.multiply((lam[group] * rowfactor[group] * weights)[:, None])
        return ((B.T @ weighted).toarray()
                - Jlog[logrows].T @ (lam[logrows, None] * Jlog[logrows]))

    def polish(v, lm):
        v, lm = v.copy(), lm.copy()
        gg, jj, ww, ll, tt, rf, jl = values(v)
        active = (gg > -1e-4) & (lm > 1e-6 * max(1.0, norm(lm)))
        lm[~active] = 0.0
        for _ in range(5):
            ids = np.flatnonzero(active)
            Ja = jj[ids]
            HH = hessian(lm, ww, rf, jl)
            KK = np.block([[HH, Ja.T], [Ja, np.zeros((ids.size, ids.size))]])
            rhs = np.r_[-q - jj.T @ lm, -gg[ids]]
            delta = linalg.lstsq(KK, rhs, cond=1e-12, lapack_driver="gelsy",
                                 check_finite=False)[0]
            dz = lm.copy()
            dz[ids] += delta[n:]
            bad = dz < -1e-9
            if np.any(bad):
                active[bad] = False
                lm[bad] = 0
                continue
            v += delta[:n]
            lm = np.maximum(dz, 0.0)
            gg, jj, ww, ll, tt, rf, jl = values(v)
            r = recover(v, lm, ww, ll, tt, rf)
            if quality(p, r):
                return r
            extra = gg > 1e-8
            active[extra] = True
        return {"status": "unresolved"}

    g, J, weights, logs, term, rowfactor, Jlog = values(d)
    sl = np.maximum(1.0, -g)
    lam = np.ones(m)
    for it in range(maxiter):
        if time.monotonic() - started > 25:
            break
        rd, rp = q + J.T @ lam, g + sl
        mu = float(sl @ lam) / m
        if debug:
            print("gp", n, m, it, "mu", mu, "rd", norm(rd),
                  "rp", norm(rp), "g", np.max(g), file=sys.stderr)
        if it > 2 and (max(norm(rd), norm(rp), mu) < 1e-4 or norm(lam) > 1e5):
            r = recover(d, lam, weights, logs, term, rowfactor)
            if quality(p, r):
                return r
            if mu < 1e-8 and norm(rd) < 1e-5 and norm(rp) < 1e-4:
                polished = polish(d, lam)
                if polished["status"] != "unresolved":
                    return polished
            if r["status"] == "solved" and norm(lam) > 1e5:
                cert = {"status": "primal_infeasible", "z": r["z"]}
                if quality(p, cert):
                    return cert
        try:
            H = hessian(lam, weights, rowfactor, Jlog)
            M = H + J.T @ ((lam / sl)[:, None] * J)
            M = (M + M.T) * 0.5
            scale = 1.0 / np.sqrt(np.maximum(M.diagonal(), 1e-30))
            scaled = M * scale[:, None] * scale[None, :]
            chol = linalg.cho_factor(scaled + np.eye(n) * 1e-14, check_finite=False)

            def linear(rhs):
                dx = scale * linalg.cho_solve(chol, rhs * scale, check_finite=False)
                for _ in range(2):
                    res = rhs - M @ dx
                    if norm(res) < 1e-11 * max(1.0, norm(rhs)):
                        break
                    dx += scale * linalg.cho_solve(chol, res * scale, check_finite=False)
                return dx

            def direction(target, correction=0.0):
                hh = (-sl * lam + target - correction + lam * rp) / sl
                dx = linear(-rd - J.T @ hh)
                jdx = J @ dx
                ds = -rp - jdx
                dl = hh + (lam / sl) * jdx
                return dx, ds, dl

            def step(ds, dl):
                alpha = 1.0
                bad = ds < 0
                if np.any(bad):
                    alpha = min(alpha, float(np.min(-sl[bad] / ds[bad])))
                bad = dl < 0
                if np.any(bad):
                    alpha = min(alpha, float(np.min(-lam[bad] / dl[bad])))
                return alpha

            da, sa, la = direction(0.0)
            aa = step(sa, la)
            mua = float((sl + aa * sa) @ (lam + aa * la)) / m
            sigma = np.clip((mua / mu) ** 3, 0.01, 0.8)
            target = sigma * mu
            dx, ds, dl = direction(target, sa * la)
            alpha = 0.995 * step(ds, dl)
            merit = (np.linalg.norm(rd) ** 2 + np.linalg.norm(rp) ** 2
                     + np.linalg.norm(sl * lam - target) ** 2)
            accepted = False
            for attempt in range(2):
                if attempt:
                    target = 0.5 * mu
                    dx, ds, dl = direction(target)
                    alpha = 0.99 * step(ds, dl)
                    merit = (np.linalg.norm(rd) ** 2 + np.linalg.norm(rp) ** 2
                             + np.linalg.norm(sl * lam - target) ** 2)
                for _ in range(30):
                    gd, Jd, wd, ld, td, rfd, jld = values(d + alpha * dx)
                    sd, zd = sl + alpha * ds, lam + alpha * dl
                    rdd, rpd = q + Jd.T @ zd, gd + sd
                    value = (np.linalg.norm(rdd) ** 2 + np.linalg.norm(rpd) ** 2
                             + np.linalg.norm(sd * zd - target) ** 2)
                    if np.isfinite(value) and value < (1 - 0.001 * alpha) * merit:
                        accepted = True
                        break
                    alpha *= 0.5
                if accepted:
                    break
            if not accepted or alpha < 1e-10:
                break
            d += alpha * dx
            sl, lam = sd, zd
            g, J, weights, logs, term, rowfactor, Jlog = gd, Jd, wd, ld, td, rfd, jld
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError):
            if debug:
                import traceback
                traceback.print_exc()
            break
    if not phase:
        try:
            from scipy.optimize import minimize
            initial = np.r_[d, max(0.0, np.max(g)) + 1.0]
            gradient = np.r_[np.zeros(n), 1.0]

            def phase_constraints(v):
                return v[-1] - values(v[:-1])[0]

            def phase_jacobian(v):
                return np.column_stack((-values(v[:-1])[1], np.ones(m)))

            phase_result = minimize(
                lambda v: v[-1], initial, jac=lambda v: gradient,
                constraints={"type": "ineq", "fun": phase_constraints,
                             "jac": phase_jacobian},
                bounds=[(None, None)] * n + [(-1.0, None)],
                method="SLSQP", options={"ftol": 1e-12, "maxiter": 800, "disp": False})
            v = phase_result.x
            gg, jj, ww, ll, tt, rf, jl = values(v[:-1])
            if np.max(gg) < -1e-5:
                return solve_geometric(p, maxiter=140, start=v[:-1], phase=True)
            lm = np.maximum(np.asarray(phase_result.multipliers[-m:]), 0)
            active = ((gg - v[-1] > -1e-5) & (lm > 1e-8))
            lm[~active] = 0
            for _ in range(8):
                rr = recover(v[:-1], lm, ww, ll, tt, rf)
                if rr["status"] == "solved":
                    certificate = {"status": "primal_infeasible", "z": rr["z"]}
                    if quality(p, certificate):
                        return certificate
                ids = np.flatnonzero(active)
                Jp = np.column_stack((jj[ids], -np.ones(ids.size)))
                Hp = np.zeros((n + 1, n + 1))
                Hp[:n, :n] = hessian(lm, ww, rf, jl)
                K = np.block([[Hp, Jp.T], [Jp, np.zeros((ids.size, ids.size))]])
                rhs = np.r_[-gradient - Jp.T @ lm[ids], -(gg[ids] - v[-1])]
                delta = linalg.lstsq(K, rhs, cond=1e-13, check_finite=False,
                                     lapack_driver="gelsy")[0]
                v += delta[:n + 1]
                lm[ids] += delta[n + 1:]
                bad = lm < 0
                lm[bad] = 0
                active[bad] = False
                gg, jj, ww, ll, tt, rf, jl = values(v[:-1])
        except (ValueError, ArithmeticError, RuntimeError, AttributeError, np.linalg.LinAlgError):
            if debug:
                import traceback
                traceback.print_exc()
    return {"status": "unresolved"}
