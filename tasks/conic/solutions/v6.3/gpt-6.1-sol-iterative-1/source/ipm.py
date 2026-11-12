"""Infeasible-start homogeneous primal-dual conic optimization."""

import os
import sys
import time

import numpy as np
from scipy import sparse

from linear import SparseFactors, AugmentedMatrix


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def equilibrate(P, q, A, b, cones):
    n, m = len(q), len(b)
    dx, er = np.ones(n), np.ones(m)
    P, A, q, b = P.copy(), A.copy(), q.copy(), b.copy()
    for _ in range(5):
        ac = np.asarray(abs(A).max(axis=0).toarray()).ravel() if m else np.zeros(n)
        pc = np.asarray(abs(P).max(axis=0).toarray()).ravel() if n else np.zeros(n)
        col = 1 / np.sqrt(np.maximum(np.maximum(ac, pc), 1e-4))
        col = np.clip(col, .01, 100)
        row = np.asarray(abs(A).max(axis=1).toarray()).ravel() if n else np.zeros(m)
        # Only scalar scaling on irreducible nonpolyhedral cone blocks.
        off = cones.neq + cones.nnonneg
        for sl, k, d in cones.soc:
            rr = slice(cones.neq + sl.start, cones.neq + sl.stop)
            r = row[rr].reshape(k, d)
            r[:] = np.max(r, axis=1)[:, None]
        if cones.nexp:
            rr = slice(cones.neq + cones.exps.start, cones.neq + cones.exps.stop)
            r = row[rr].reshape(-1, 3)
            r[:] = np.max(r, axis=1)[:, None]
        for block in cones.psd:
            rr = slice(cones.neq + block["sl"].start, cones.neq + block["sl"].stop)
            row[rr] = np.max(row[rr])
        row = np.clip(1 / np.sqrt(np.maximum(row, 1e-4)), .01, 100)
        dx *= col
        er *= row
        A = (sparse.diags(row) @ A @ sparse.diags(col)).tocsc()
        P = (sparse.diags(col) @ P @ sparse.diags(col)).tocsc()
        q *= col
        b *= row
    cost = 1 / max(norm(q), norm(P.data), 1e-4)
    cost = np.clip(cost, 1e-8, 1e8)
    return P * cost, q * cost, A, b, dx, er, cost


def solve(P0, q0, A0, b0, cs, max_iter=120, settings=None, prepared=None):
    settings = settings or {}
    verbose = bool(os.getenv("SOLVER_VERBOSE"))
    n, m, ne = len(q0), len(b0), cs.neq
    if prepared is None:
        Ap = A0[cs.perm, :].tocsc()
        bp = b0[cs.perm]
        P, q, A, b, dxscale, rowscale, cost = equilibrate(P0, q0, Ap, bp, cs)
    else:
        P, q, A, b, dxscale, rowscale, cost = (prepared[k] for k in ("P", "q", "A", "b", "dx", "er", "cost"))
    E, C = A[:ne, :].tocsc(), A[ne:, :].tocsc()
    be, bc = b[:ne], b[ne:]
    CT, ET = C.T.tocsc(), E.T.tocsc()
    x = np.zeros(n)
    y = np.zeros(ne)
    s, z = cs.initial()
    tau, kap = 1., 1.
    if prepared is not None and prepared.get("initial") is not None and settings.get("use_initial", True):
        x, y, s, z = (v.copy() for v in prepared["initial"])
        kap = max(1e-4, float(s @ z) / max(1, cs.nu))
    phase2 = False
    best, besterr = None, np.inf
    qnorm, bnorm = norm(q0), norm(b0)
    ascale, pscale = norm(A0.data) or 1., norm(P0.data) or 1.
    ip = sparse.eye(n, format="csc")
    ie = sparse.eye(ne, format="csc")
    form = settings.get("form", os.getenv("KKT_FORM"))
    normal = bool(cs.psd) or form == "normal"
    order = settings.get("order", os.getenv("KKT_ORDER", os.getenv("SPARSE_ORDER", "MMD_AT_PLUS_A")))
    pivot = float(settings.get("pivot", os.getenv("KKT_PIVOT", "0.0")))
    factor = SparseFactors(order, pivot, native=True)
    if prepared is not None and not normal and prepared.get("ordering") is not None:
        factor.perm = prepared["ordering"]
    matrix_cache = None
    use_matrix_cache = os.getenv("KKT_MATRIX_REUSE", "1") != "0"
    initialize = not cs.psd and settings.get("initialize", os.getenv("IPM_INIT", "0") != "0")

    def linear_accuracy(residual, rhs):
        if not norm(residual[:n]) < 2e-13 * max(1., norm(rhs[:n])):
            return False
        if normal:
            return norm(residual[n:]) < 2e-13 * max(1., norm(rhs[n:]))
        end = n + ne + cs.size
        return (norm(residual[n:end]) < 2e-13 * max(1., norm(rhs[n:end])) and
                norm(residual[end:]) < 2e-13 * max(1., norm(rhs[end:])))
    # Sparse SDP models often touch far fewer packed entries than their matrix
    # order suggests. Use the Hessian on that support instead of dense tensors.
    for block in cs.psd:
        Ai = C[block["sl"], :].tocsc()
        cols = np.flatnonzero(np.diff(Ai.indptr))
        rows = np.unique(Ai.indices)
        d = block["d"]
        block["Ai"], block["cols"] = Ai[:, cols], cols
        if len(rows)**2 <= min(16_000_000, 4 * max(1, len(cols)) * d * d):
            block["support"] = rows
            block["Bsupport"] = Ai[rows, :][:, cols].tocsc()
        else:
            block["support"] = None
            if len(cols) * d * d <= 8_000_000:
                B = Ai[:, cols].toarray()
                mats = np.zeros((len(cols), d, d))
                mats[:, block["row"], block["col"]] = B.T / block["factor"]
                mats[:, block["col"], block["row"]] = B.T / block["factor"]
                block["mats"] = mats
            else:
                block["mats"] = None
    reg = float(settings.get("regularization", os.getenv("KKT_REG", "1e-10")))
    poly_qp = (bool(P0.nnz) and not cs.soc and not cs.psd and not cs.nexp and
               os.getenv("IPM_QP_POLISH", "0") != "0")
    polish_attempts = 0
    conic_attempts = 0
    last_improvement = 0
    solve_tolerance = float(os.getenv("SOLVE_TOL", "9.5e-9"))

    def recovered_polish(iterator):
        from validation import valid
        try:
            for polished in iterator:
                px = dxscale * polished["x"]
                pz, ps = np.empty(m), np.empty(m)
                pz[cs.perm] = rowscale * polished["z"] / cost
                ps[cs.perm] = polished["s"] / rowscale
                candidate = {"status": "solved", "x": px, "s": ps, "z": pz}
                if valid(P0, q0, A0, b0, None, candidate, cs, tol=8.5e-9):
                    return candidate
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError, MemoryError):
            pass
        return None

    for iteration in range(max_iter):
        if not all(np.isfinite(v).all() for v in (x, s, z, y)) or tau <= 0 or kap <= 0:
            break
        Px, Cx, Ex = P @ x, C @ x, E @ x
        rd = Px + q * tau + CT @ z + ET @ y
        rp, re = Cx + s - bc * tau, Ex - be * tau
        xPx = float(x @ Px)
        rg = float(q @ x + bc @ z + be @ y + xPx / tau + kap)
        mu = float((s @ z + (tau * kap if not phase2 else 0.)) / (cs.nu + (0 if phase2 else 1)))
        if mu <= 0 or not np.isfinite(mu):
            break
        origx = dxscale * (x / tau)
        origz = np.empty(m)
        origz[cs.perm] = rowscale * np.concatenate((y, z)) / (cost * tau)
        origs = np.empty(m)
        origs[cs.perm] = np.concatenate((np.zeros(ne), s)) / (rowscale * tau)
        opx, atz = P0 @ origx, A0.T @ origz
        dr = norm(opx + q0 + atz) / (1 + max(norm(opx), qnorm, norm(atz)))
        p0 = float(.5 * origx @ opx + q0 @ origx)
        d0 = float(-.5 * origx @ opx - b0 @ origz)
        gap = abs(p0 - d0) / max(1., abs(p0), abs(d0))
        # Linear primal residual is exact after slack recovery; cone feasibility
        # is checked before returning any candidate.
        ax = A0 @ origx
        pr = norm(ax + origs - b0) / (1 + max(norm(ax), norm(origs), bnorm))
        err = max(dr, gap, pr)
        if err < besterr:
            if err < .99 * besterr:
                last_improvement = iteration
            besterr = err
            best = {"status": "solved", "x": origx.copy(), "s": origs.copy(), "z": origz.copy()}
        if verbose:
            print(f"{iteration:3d} mu {mu:.3e} tau {tau:.3e} kap {kap:.3e} rp {pr:.2e} rd {dr:.2e} gap {gap:.2e}", file=sys.stderr)
        if poly_qp and ((polish_attempts == 0 and err < 1e-5) or
                        (polish_attempts == 1 and err < 1e-7)):
            from polish import candidates
            eq = np.arange(m) < ne
            dual = np.concatenate((y, z)) / tau
            tolerance = 1e-4 if polish_attempts == 0 else 1e-6
            polish_attempts += 1
            candidate = recovered_polish(candidates(P, q, A, b, eq, x / tau, dual, tolerance))
            if candidate is not None:
                return candidate
        if (not cs.psd and (cs.soc or cs.nexp) and os.getenv("CONIC_POLISH", "1") != "0" and
                ((conic_attempts == 0 and err < 1e-3 and pr < 1e-4) or
                 (conic_attempts == 1 and err < 1e-6))):
            from conic_polish import candidates
            dual = np.concatenate((y, z)) / tau
            tolerance = .01 if conic_attempts == 0 else 1e-4
            conic_attempts += 1
            candidate = recovered_polish(candidates(P, q, A, b, cs, x / tau, dual, tolerance))
            if candidate is not None:
                return candidate
        if max(dr, gap, pr) < solve_tolerance:
            ps = origs[cs.perm]
            zs = origz[cs.perm]
            if (norm(ps[:ne]) <= 1e-8 and cs.defect(ps[ne:]) <= 8e-9 and
                    cs.defect(zs[ne:], True) <= 8e-9):
                return {"status": "solved", "x": origx, "s": origs, "z": origz}
        if not phase2 and cs.nu > 0 and max(dr, gap, pr) < 1e-3:
            x /= tau
            y /= tau
            s /= tau
            z /= tau
            kap = 1e-100
            tau = 1.
            phase2 = True
            continue
        if iteration > 25 and iteration - last_improvement > 12:
            break
        # Homogeneous infeasibility rays, checked with the prescribed scaling.
        if not phase2 and iteration >= 3:
            from validation import valid
            certz = np.empty(m)
            certz[cs.perm] = rowscale * np.concatenate((y, z)) / cost
            zn = norm(certz)
            if zn and bnorm:
                zz = certz / zn
                margin = -float((b0 / bnorm) @ zz)
                estimate = norm(atz) * tau / zn
                if margin > 1e-8 and (not np.isfinite(estimate) or
                        estimate < 1e-7 * ascale * margin or iteration % 20 == 0):
                    certificate = {"status": "primal_infeasible", "z": certz}
                    if valid(P0, q0, A0, b0, None, certificate, cs, tol=5e-9):
                        return certificate
            certx = dxscale * x
            xn = norm(certx)
            if xn and qnorm:
                xx = certx / xn
                margin = -float((q0 / qnorm) @ xx)
                estimate = norm(opx) * tau / xn
                if margin > 1e-8 and (not np.isfinite(estimate) or
                        estimate < 1e-7 * pscale * margin or iteration % 20 == 0):
                    ss = -A0 @ xx
                    certificate = {"status": "dual_infeasible", "x": xx, "s": ss}
                    if valid(P0, q0, A0, b0, None, certificate, cs, tol=5e-9):
                        return certificate
        try:
            cs.scale(s, z, mu)
            if normal:
                Cs = C[:cs.nsmall, :]
                H = (P + Cs.T @ cs.D @ Cs).tocsc()
                if cs.nlarge:
                    HV = Cs.T @ cs.large_V
                    H = H + HV @ HV.T
                for block, state in zip(cs.psd, cs.psd_state):
                    W, cols, rows = state[0], block["cols"], block["support"]
                    if rows is not None:
                        r, c = block["row"][rows], block["col"][rows]
                        f = block["factor"][rows]
                        Dh = W[r[:, None], r[None, :]] * W[c[:, None], c[None, :]]
                        Dh += (W[r[:, None], c[None, :]] * W[c[:, None], r[None, :]] *
                               (r != c)[None, :])
                        Dh *= f[:, None] / f[None, :]
                        B = block["Bsupport"]
                        hh = np.asarray(B.T @ (B.T @ Dh).T)
                    else:
                        d = block["d"]
                        hh = np.empty((len(cols), len(cols)))
                        chunk = max(1, 8_000_000 // (d * d))
                        for first in range(0, len(cols), chunk):
                            last = min(len(cols), first + chunk)
                            if block["mats"] is not None:
                                mats = block["mats"][first:last]
                            else:
                                B = block["Ai"][:, first:last].toarray()
                                mats = np.zeros((last - first, d, d))
                                mats[:, block["row"], block["col"]] = B.T / block["factor"]
                                mats[:, block["col"], block["row"]] = B.T / block["factor"]
                            transformed = W @ mats @ W
                            B = (transformed[:, block["row"], block["col"]] * block["factor"]).T
                            hh[:, first:last] = block["Ai"].T @ B
                    hh = .5 * (hh + hh.T)
                    if len(cols) == n:
                        H = H + sparse.csc_matrix(hh)
                    else:
                        rr = np.repeat(cols, len(cols))
                        cc = np.tile(cols, len(cols))
                        H = H + sparse.csc_matrix((hh.ravel(), (rr, cc)), shape=(n, n))
                K = sparse.bmat([[H + reg * ip, ET], [E, -reg * ie]], format="csc")
                lu = factor(K)
                regdiag = np.concatenate((np.full(n, reg), np.full(ne, -reg)))
                def linear_solve(rhs):
                    sol = lu.solve(rhs)
                    for _ in range(3):
                        residual = rhs - K @ sol + regdiag * sol
                        if linear_accuracy(residual, rhs):
                            break
                        sol += lu.solve(residual)
                    return sol
                Db = cs.apply(bc)
                aa = np.concatenate((q - CT @ Db, -be))
                v = linear_solve(aa) if not phase2 else np.zeros(n + ne)
                c = np.concatenate((q + 2 * Px / tau + CT @ Db, be))
                w = C @ v[:n] + bc
                h = v[:n] + x / tau
                denominator = -float(w @ cs.apply(w) + h @ (P @ h)) - kap / tau

                def direction(rc, rt, rg_extra=0.):
                    rr = cs.apply(rp) + rc
                    rhs = np.concatenate((-rd - CT @ rr, -re))
                    u = linear_solve(rhs)
                    dt = ((-rg - rg_extra - bc @ rr - rt / tau - c @ u) / denominator) if not phase2 else 0.
                    sol = u - v * dt
                    dx, dy = sol[:n], sol[n:]
                    ds = -rp - C @ dx + bc * dt
                    dz = rc - cs.apply(ds)
                    dk = (rt - kap * dt) / tau if not phase2 else 0.
                    return dx, dy, ds, dz, float(dt), float(dk)
            else:
                if use_matrix_cache:
                    try:
                        if matrix_cache is None:
                            matrix_cache = AugmentedMatrix(P, E, C, cs, reg)
                        K = matrix_cache.update(cs)
                    except (OSError, RuntimeError):
                        use_matrix_cache = False
                if not use_matrix_cache:
                    Z = sparse.csc_matrix((ne, cs.size))
                    xn = sparse.csc_matrix((n, cs.naux))
                    en = sparse.csc_matrix((ne, cs.naux))
                    aux = (sparse.diags(cs.aux_sign * (1 + reg)) if cs.naux else
                           sparse.csc_matrix((0, 0)))
                    K = sparse.bmat([[P + reg * ip, ET, CT, xn],
                                     [E, -reg * ie, Z, en],
                                     [C, Z.T, -cs.Di - reg * sparse.eye(cs.size, format="csc"), cs.U],
                                     [xn.T, en.T, cs.U.T, aux]], format="csc")
                lu = factor(K)
                regdiag = np.concatenate((np.full(n, reg), np.full(ne + cs.size, -reg), cs.aux_sign * reg))
                def linear_solve(rhs):
                    sol = lu.solve(rhs)
                    for _ in range(3):
                        residual = rhs - K @ sol + regdiag * sol
                        if linear_accuracy(residual, rhs):
                            break
                        sol += lu.solve(residual)
                    return sol
                aa = np.concatenate((-q, be, bc, np.zeros(cs.naux)))
                v = linear_solve(aa) if not phase2 else np.zeros(len(aa))
                c = np.concatenate((q + 2 * Px / tau, be, bc, np.zeros(cs.naux)))
                dd = -xPx / (tau * tau) - kap / tau
                h = v[:n] - x / tau
                cone_energy = cs.inverse_energy(v[n + ne:n + ne + cs.size])
                denominator = -max(0., float(h @ (P @ h))) - cone_energy - kap / tau

                def direction(rc, rt, rg_extra=0.):
                    rhs = np.concatenate((-rd - CT @ rc, -re, -rp, np.zeros(cs.naux)))
                    u = linear_solve(rhs)
                    dt = ((-rg - rg_extra - bc @ rc - rt / tau - c @ u) / denominator) if not phase2 else 0.
                    sol = u + v * dt
                    dx, dy, dz = sol[:n], sol[n:n + ne], sol[n + ne:n + ne + cs.size] + rc
                    ds = -rp - C @ dx + bc * dt
                    dk = (rt - kap * dt) / tau if not phase2 else 0.
                    return dx, dy, ds, dz, float(dt), float(dk)
            if iteration == 0 and initialize:
                if normal:
                    guess = linear_solve(np.r_[-q + CT @ cs.apply(bc), be])
                    wx, wy = guess[:n], guess[n:]
                    wz = cs.apply(C @ wx - bc)
                else:
                    guess = linear_solve(np.r_[-q, be, bc, np.zeros(cs.naux)])
                    wx, wy, wz = guess[:n], guess[n:n + ne], guess[n + ne:n + ne + cs.size]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    ws, wz = cs.warm_pair(bc - C @ wx, wz)
                    wm = float(ws @ wz) / max(cs.nu, 1)
                if np.isfinite(wm) and wm >= 0 and all(np.isfinite(v).all() for v in (wx, wy, ws, wz)):
                    x, y, s, z = wx, wy, ws, wz
                    tau, kap = 1., max(1e-4, wm)
                    continue
            aff = direction(cs.rhs(0.), -tau * kap)
            dxa, dya, dsa, dza, dta, dka = aff
            ap = min(cs.max_step(s, dsa), -tau / dta if dta < 0 else 1.)
            ad = min(cs.max_step(z, dza, True), -kap / dka if dka < 0 else 1.)
            am = min(ap, ad)
            mua = ((s + am * dsa) @ (z + am * dza) + ((tau + am * dta) * (kap + am * dka) if not phase2 else 0.)) / (cs.nu + (0 if phase2 else 1))
            sigma = min(.9, max(0., float(mua / mu))**3)
            if cs.nexp:
                sigmamin = (settings.get("final_sigma", os.getenv("EXP_SIGMA_FINAL", ".1")) if phase2
                            else os.getenv("EXP_SIGMA", ".5"))
                sigma = max(float(sigmamin), sigma)
            rc = cs.rhs(sigma * mu, dsa, dza)
            perspective = dxa - x * (dta / tau)
            extra = float(perspective @ (P @ perspective)) / tau
            dx, dy, ds, dz, dt, dk = direction(rc, -tau * kap + sigma * mu - dta * dka, extra)
            step = min(cs.max_step(s, ds), cs.max_step(z, dz, True),
                       -tau / dt if dt < 0 else 1., -kap / dk if dk < 0 else 1.)
            step = min(1., step * .99)
            if verbose:
                print(f"    aff {am:.3e} sigma {sigma:.3e} step {step:.3e}", file=sys.stderr)
            if not np.isfinite(step) or step < 1e-12:
                break
            x += step * dx
            y += step * dy
            s += step * ds
            z += step * dz
            tau += step * dt
            kap += step * dk
            magnitude = max(tau, kap)
            if magnitude > 1e40:
                x /= magnitude
                y /= magnitude
                s /= magnitude
                z /= magnitude
                tau = max(1e-300, tau / magnitude)
                kap = max(1e-300, kap / magnitude)
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError) as error:
            if verbose:
                print("IPM error:", repr(error), file=sys.stderr)
            break
    if best is not None:
        best["_normal"] = normal
        return best
    return {"status": "unresolved"}
