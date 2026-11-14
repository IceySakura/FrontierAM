"""Primal-dual interior point solver for the general conic QP

    min  .5 x'Px + q'x     s.t.   A x + s = b,   s in K

The Newton system is reduced to symmetric positive definite normal
equations using the *primal* barrier Hessian H(s):

    (P + mu A' H A) dx + Az' dz_zero = rhs ,     Az dx = -rp_zero .

Only the barrier gradient and Hessian-vector product of each cone are
needed, so zero / nonnegative / SOC / PSD / exponential cones all fit in
one framework (see cones.py).  Equality (zero cone) rows carry free dual
multipliers and are handled by a Schur complement on the normal matrix.
"""
import time
import os
import numpy as np

DEBUG = bool(os.environ.get("CONIC_DEBUG"))


def _dbg(*args):
    if DEBUG:
        import sys
        print(*args, file=sys.stderr, flush=True)


def _csc_dense(data, indices, indptr, shape):
    out = np.zeros(shape, dtype=np.float64)
    n = shape[1]
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
    if cols.size:
        np.add.at(out, (indices.astype(np.int64), cols), data)
    return out


def _ninf(v):
    return float(np.max(np.abs(v), initial=0.0))


def _make_scaling(Ad, Pd, q, b, cd, enabled=True):
    """Ruiz equilibration, with row factors constant inside each cone block."""
    n, m = Ad.shape[1], Ad.shape[0]
    if not enabled:
        return np.ones(n), np.ones(m)
    dx = np.ones(n)
    dr = np.ones(m)
    for _ in range(4):
        M = (dr[:, None] * Ad) * dx[None, :]
        cn = np.max(np.abs(M), axis=0) if m else np.ones(n)
        rn = np.max(np.abs(M), axis=1) if n else np.ones(m)
        cn = np.maximum(cn, 1e-12)
        rn = np.maximum(rn, 1e-12)
        # a cone block must be scaled with ONE factor to preserve the cone
        for blk in cd.blocks:
            lo, hi = blk["start"], blk["stop"]
            rn[lo:hi] = np.max(rn[lo:hi])
        dx /= np.sqrt(cn)
        dr /= np.sqrt(rn)
    # Keep factors in a numerically sane range.
    dx = np.clip(dx, 1e-8, 1e8)
    dr = np.clip(dr, 1e-8, 1e8)
    return dx, dr


def _checker_residuals(Ad, Pd, q, b, cd, x, s, z, tol):
    """Reproduce the independent checker in the ORIGINAL coordinates."""
    Ax = Ad @ x
    Px = Pd @ x
    Atz = Ad.T @ z
    rp = _ninf(Ax + s - b) / (1.0 + max(_ninf(Ax), _ninf(s), _ninf(b)))
    rd = _ninf(Px + q + Atz) / (1.0 + max(_ninf(Px), _ninf(q), _ninf(Atz)))
    with np.errstate(over="ignore", invalid="ignore"):
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        comp = float(s @ z)
    if not (np.isfinite(p0) and np.isfinite(d0) and np.isfinite(comp)):
        return False, (rp, rd, float("inf"), float("inf"), 0.0, 0.0)
    gap = abs(p0 - d0)
    scale = max(abs(p0), abs(d0))
    rel = gap / scale if scale > 0 else 0.0
    cp = cd.defect_primal(s)
    cz = cd.defect_dual(z)
    ok = (np.isfinite(rp) and np.isfinite(rd) and
          rp <= tol and rd <= tol and cp <= tol and cz <= tol and
          (gap <= tol or rel <= tol))
    return ok, (rp, rd, gap, rel, cp, cz)


def _attempt(Ad, Pd, q, b, cones, cd, dx, dr, deadline, mu_floor, max_iter,
             eta=1.0, use_lstsq=True, x_start=None, schedule="centered"):
    """One IPM run in scaled coordinates; returns (x,s,z,status) or None.

    The objective is divided by `eta`; the dual of the original problem is
    recovered as z_orig = eta * dr * z_scaled.
    """
    n, m = Ad.shape[1], Ad.shape[0]
    Ahat = (dr[:, None] * Ad) * dx[None, :]
    Phat = ((dx[:, None] * Pd) * dx[None, :]) / eta
    qhat = (dx * q) / eta
    bhat = dr * b

    cidx = cd.cone_idx
    zidx = cd.zero_idx
    mc, mz = cidx.size, zidx.size
    Ac = Ahat[cidx, :]
    Az = Ahat[zidx, :]
    bc = bhat[cidx]
    bz = bhat[zidx]
    nu = max(1.0, float(cd.nu))

    # ---- infeasible start, strictly interior cone point -------------------
    s = np.asarray(cd.s0(), dtype=np.float64).copy()
    s[zidx] = 0.0
    x = np.zeros(n)
    z = np.zeros(m)
    mu = 1.0
    g0 = np.asarray(cd.grad(s), dtype=np.float64)
    z[cidx] = -mu * g0[cidx]
    if x_start is not None:
        x = np.asarray(x_start, dtype=np.float64).copy()
    if mz and use_lstsq and x_start is None:
        try:
            x = np.linalg.lstsq(Az, bz, rcond=None)[0]
        except Exception:
            x = np.zeros(n)
    if mz and use_lstsq:
        try:
            z[zidx] = np.linalg.lstsq(Az.T, -(qhat + Ac.T @ z[cidx]), rcond=None)[0]
        except Exception:
            pass
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        x = np.zeros(n)
        z = np.zeros(m)
        z[cidx] = -mu * g0[cidx]

    # workspace reused for Hessian applications
    Y = np.zeros((m, n))
    Y[cidx, :] = Ac
    rp_full = np.zeros(m)
    ds_full = np.zeros(m)
    dz_full = np.zeros(m)
    eye_mz = np.eye(mz) if mz else None

    best = None
    best_score = float("inf")
    t0 = time.perf_counter()
    last_mu = mu

    for it in range(max_iter):
        if time.perf_counter() > deadline:
            break
        # ---------------- residuals ---------------------------------------
        Ax = Ahat @ x
        rp = Ax + s - bhat
        Atz = Ahat.T @ z
        Px = Phat @ x
        rd = Px + qhat + Atz
        g = np.asarray(cd.grad(s), dtype=np.float64)
        rc = z[cidx] + mu * g[cidx]
        rp_rel = _ninf(rp) / (1.0 + max(_ninf(Ax), _ninf(s), _ninf(bhat)))
        rd_rel = _ninf(rd) / (1.0 + max(_ninf(Px), _ninf(qhat), _ninf(Atz)))
        gap_s = float(s[cidx] @ z[cidx])
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(qhat @ x)
        d0 = -0.5 * xPx - float(bhat @ z)
        gap_abs = abs(p0 - d0)
        gap_rel = gap_abs / max(1e-300, max(abs(p0), abs(d0)))
        score = max(rp_rel, rd_rel, min(gap_abs / 1e-8, gap_rel / 1e-8))
        if score < best_score:
            best_score = score
            best = (x.copy(), s.copy(), z.copy())
        if DEBUG and it % 12 == 0:
            _dbg("it %3d t=%.3f mu=%.2e rp=%.2e rd=%.2e rc=%.2e gap=%.2e rel=%.2e score=%.2e" %
                 (it, time.perf_counter() - t0, mu, rp_rel, rd_rel, _ninf(rc), gap_abs, gap_rel, score))

        # ---------------- convergence + validation in original coords ------
        # Build s = b - A x in the original coordinates: that makes the
        # primal residual exactly zero, so a scaled-space residual that is
        # merely small (not tiny) can already be a fully valid answer.
        if (rp_rel < 1e-5 and rd_rel < 1e-5 and
                (gap_abs < 1e-5 or gap_rel < 1e-5)):
            xo = dx * x
            zo = (eta * dr) * z
            so = b - Ad @ xo
            # The fast homotopy may only short-circuit on a clearly valid
            # point; borderline iterates are left for the centered homotopy.
            vtol = 5e-10 if schedule == "aggressive" else 3e-9
            ok, res = _checker_residuals(Ad, Pd, q, b, cd, xo, so, zo, vtol)
            if DEBUG:
                _dbg("  validate it", it, ok, res)
            if ok:
                return xo, so, zo

        # ---------------- Newton system ------------------------------------
        try:
            HY = np.asarray(cd.hess_apply(s, Y), dtype=np.float64)
            HAc = HY[cidx, :]
            M = Phat + mu * (Ac.T @ HAc)
            M = 0.5 * (M + M.T)
            reg = 1e-14 * max(1.0, _ninf(M))
            M.flat[::n + 1] += reg
            rp_full[:] = 0.0
            rp_full[cidx] = rp[cidx]
            Hrp = np.asarray(cd.hess_apply(s, rp_full), dtype=np.float64)[cidx]
            rhs_x = -rd + Ac.T @ rc - mu * (Ac.T @ Hrp)

            if mz:
                # Solve the quasi-definite KKT saddle system directly:
                # M may be singular (cone rows need not span R^n), in which
                # case a Schur complement based on M^{-1} is unusable.
                K = np.zeros((n + mz, n + mz))
                K[:n, :n] = M
                K[:n, n:] = Az.T
                K[n:, :n] = Az
                K[n:, n:].flat[::mz + 1] = -1e-16 * max(1.0, _ninf(M))
                rhs = np.concatenate((rhs_x, -rp[zidx]))
                try:
                    sol = np.linalg.solve(K, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
                dxv = sol[:n]
                dz_z = sol[n:]
            else:
                dxv = np.linalg.solve(M, rhs_x)
                dz_z = np.zeros(0)

            ds_c = -rp[cidx] - Ac @ dxv
            ds_full[:] = 0.0
            ds_full[cidx] = ds_c
            Hds = np.asarray(cd.hess_apply(s, ds_full), dtype=np.float64)[cidx]
            dz_c = -rc - mu * Hds
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            _dbg("newton failure", repr(exc))
            break
        if not (np.isfinite(dxv).all() and np.isfinite(ds_c).all() and
                np.isfinite(dz_c).all() and np.isfinite(dz_z).all()):
            _dbg("nonfinite direction")
            break

        dz_full[:] = 0.0
        dz_full[cidx] = dz_c
        dz_full[zidx] = dz_z

        # ---------------- fraction to boundary ------------------------------
        try:
            ap = float(cd.max_step_primal(s, ds_full))
            ad = 1.0
        except Exception:
            break
        # z is a free Newton variable: r_c -> 0 drives it to -mu*grad F(s),
        # which lies in int K*.  Restricting the step by the dual-cone
        # boundary makes the method stall on PSD/SOC blocks, so use only the
        # primal fraction-to-boundary.
        if DEBUG and it < 6:
            _dbg("      dirnorm dx=%.3e ds=%.3e dz=%.3e |M|=%.3e" %
                 (_ninf(dxv), _ninf(ds_c), _ninf(dz_c), _ninf(M)))
        alpha = min(1.0, ap)
        if not np.isfinite(alpha) or alpha <= 1e-12:
            _dbg("alpha too small", alpha, ap, ad)
            break
        phi0 = float(rp @ rp + rd @ rd + rc @ rc)
        if DEBUG and it % 3 == 0:
            _dbg("      alpha=%.4f ap=%.4f ad=%.4f phi0=%.3e" % (alpha, ap, ad, phi0))
        accepted = False
        for _ in range(28):
            a = alpha
            xn = x + a * dxv
            sn = s + a * ds_full
            sn[zidx] = 0.0
            zn = z + a * dz_full
            try:
                gn = np.asarray(cd.grad(sn), dtype=np.float64)
            except Exception:
                alpha *= 0.5
                continue
            rpn = Ahat @ xn + sn - bhat
            rdn = Phat @ xn + qhat + Ahat.T @ zn
            rcn = zn[cidx] + mu * gn[cidx]
            phin = float(rpn @ rpn + rdn @ rdn + rcn @ rcn)
            if np.isfinite(phin) and phin <= (1.0 - 1e-4 * a) * phi0 + 1e-300:
                x, s, z = xn, sn, zn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            _dbg("no armijo at it", it, "final alpha", alpha)
            # tiny damped step can still help when residuals are at roundoff
            if alpha > 1e-6:
                x = x + alpha * dxv
                s = s + alpha * ds_full
                s[zidx] = 0.0
                z = z + alpha * dz_full
            else:
                break

        gap_new = max(0.0, float(s[cidx] @ z[cidx]))
        if schedule == "aggressive":
            # Gap-driven homotopy: fast on most conic instances.
            target = 0.25 * gap_new / nu
            if max(rp_rel, rd_rel) > 1e-8:
                target = max(target, 0.55 * mu)
            mu = max(mu_floor, min(0.9 * mu, target))
        else:
            # Only shrink mu once the iterate is centred for the current mu.
            rc_now = np.asarray(cd.grad(s), dtype=np.float64)
            rc_now = _ninf(z[cidx] + mu * rc_now[cidx])
            zc_scale = 1.0 + _ninf(z[cidx])
            centered = (rc_now / zc_scale < 0.02 and
                        max(rp_rel, rd_rel) < 1e-5)
            if centered:
                target = 0.25 * gap_new / nu
                mu = max(mu_floor, min(0.9 * mu, target))
            elif rc_now / zc_scale > 0.8 and mu < 10.0:
                mu = min(10.0, mu * 1.15)
        if abs(mu - last_mu) < 1e-18 * mu and it > 20:
            pass
        last_mu = mu
        if _ninf(x) > 1e13 or _ninf(s) > 1e13 or _ninf(z) > 1e13:
            break
        if time.perf_counter() - t0 > 60.0:
            break

    # Last-chance validation of the best iterate seen.
    if best is not None:
        x, s, z = best
        xo = dx * x
        zo = (eta * dr) * z
        so = b - Ad @ xo
        last_tol = 5e-10 if schedule == "aggressive" else 1e-8
        ok, _ = _checker_residuals(Ad, Pd, q, b, cd, xo, so, zo, last_tol)
        if ok:
            return xo, so, zo
    return None


def solve_conic(P_data, P_indices, P_indptr, P_shape, q,
                A_data, A_indices, A_indptr, A_shape, b,
                cones, deadline=None):
    if deadline is None:
        deadline = time.perf_counter() + 10.0
    n = int(q.size)
    m = int(b.size)
    try:
        from cones import ConeData
        cd = ConeData(cones, m)
    except Exception as exc:
        return {"status": "unresolved"}

    # Dense algebra is used for the (n x n) normal matrix.  Larger conic
    # problems are left unresolved rather than risking a timeout/OOM.
    if n > 2200 or m > 30000 or float(n) * float(m) > 2.4e7:
        return {"status": "unresolved"}

    Ad = _csc_dense(A_data, A_indices, A_indptr, (m, n))
    Pd = _csc_dense(P_data, P_indices, P_indptr, (n, n))
    Pd = 0.5 * (Pd + Pd.T)

    dx, dr = _make_scaling(Ad, Pd, q, b, cd, enabled=True)
    Pd_s = (dx[:, None] * Pd) * dx[None, :]
    eta = max(1.0, _ninf(dx * q), _ninf(Pd_s))
    eta = min(eta, 1e10)
    # Two homotopies behave quite differently on hard cones: the gap-driven
    # one converges in a few dozen iterations on most models, while the
    # centrality-gated one rescues a few degenerate PSD instances.  Try both.
    big_psd = any(c["type"] == "psd" and c["dim"] > 12 for c in cones)
    out = None
    if big_psd:
        # Large PSD blocks are where the centered homotopy is strongest.
        attempts = [("centered", min(deadline, time.perf_counter() + 1.8)),
                    ("aggressive", deadline)]
    else:
        attempts = [("aggressive", min(deadline, time.perf_counter() + 1.1)),
                    ("centered", deadline)]
    for sch, dl in attempts:
        if time.perf_counter() >= deadline - 0.12:
            break
        out = _attempt(Ad, Pd, q, b, cones, cd, dx, dr, dl,
                       mu_floor=1e-14, max_iter=140, eta=eta,
                       schedule=sch)
        if out is not None:
            break
    if out is None and time.perf_counter() < deadline - 0.12:
        eta2 = max(1.0, _ninf(q), _ninf(Pd))
        out = _attempt(Ad, Pd, q, b, cones, cd,
                       np.ones(n), np.ones(m), deadline,
                       mu_floor=1e-13, max_iter=110, eta=eta2,
                       schedule="aggressive")
    if out is None:
        return {"status": "unresolved"}
    x, s, z = out
    return {"status": "solved", "x": x, "s": s, "z": z}
