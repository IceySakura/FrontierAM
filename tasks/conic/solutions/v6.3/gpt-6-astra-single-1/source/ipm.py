import os
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from common import blocks, norm, assess
from cones import ConeSystem


def equilibrate(P, q, A, b, cones):
    n, m = A.shape[1], A.shape[0]
    D, E = np.ones(n), np.ones(m)
    A, P, b, q = A.copy(), P.copy(), b.copy(), q.copy()
    for _ in range(6):
        an = np.asarray(abs(A).max(axis=0).toarray()).ravel() if m else np.zeros(n)
        pn = np.asarray(abs(P).max(axis=0).toarray()).ravel() if P.nnz else np.zeros(n)
        dn = np.maximum(an, pn)
        dn[dn == 0] = 1.0
        dn = 1 / np.sqrt(np.clip(dn, 1e-4, 1e4))
        rn = np.asarray(abs(A).max(axis=1).toarray()).ravel() if n else np.ones(m)
        for k, d, i, j in blocks(cones):
            if k not in ("zero", "nonnegative"):
                rn[i:j] = np.max(rn[i:j]) or 1.0
        rn[rn == 0] = 1.0
        en = 1 / np.sqrt(np.clip(rn, 1e-4, 1e4))
        D *= dn
        E *= en
        A = (sparse.diags(en) @ A @ sparse.diags(dn)).tocsc()
        P = (sparse.diags(dn) @ P @ sparse.diags(dn)).tocsc()
        q *= dn
        b *= en
    c = 1 / max(norm(q), norm(P.data), 1e-4)
    q *= c
    P *= c
    return P, q, A, b, D, E, c


def solve(P0, q0, A0, b0, cones0, validator=None):
    debug = bool(os.environ.get("SOLVER_DEBUG"))
    P, q, A, b, D, E, c = equilibrate(P0, q0, A0, b0, cones0)
    n, m = len(q), len(b)
    ie, ic, cones = [], [], []
    for k, d, i, j in blocks(cones0):
        if k == "zero":
            ie.extend(range(i, j))
        else:
            ic.extend(range(i, j))
            cones.append({"type": k, "dim": d})
    ie, ic = np.asarray(ie, dtype=int), np.asarray(ic, dtype=int)
    Ae, Ac = A[ie].tocsc(), A[ic].tocsc()
    be, bc = b[ie], b[ic]
    ne = len(ie)
    eq_restore = None
    if ne and n <= 500 and any(cn["type"] in ("psd", "exponential") for cn in cones):
        Qe, Re, piv = linalg.qr(Ae.toarray().T, mode="full", pivoting=True, check_finite=False)
        diag = np.abs(np.diag(Re))
        rank = int(np.count_nonzero(diag > max(norm(diag), 1e-30) * 1e-12))
        fixed = Qe[:, :rank] @ linalg.solve_triangular(
            Re[:rank, :rank].T, be[piv[:rank]], lower=True, check_finite=False)
        if norm(Ae @ fixed - be) < 1e-10 * (1 + norm(be)) and rank < n:
            N = Qe[:, rank:]
            eq_restore = (P, q, Ac, fixed, N, Qe[:, :rank], Re[:rank, :rank], piv, rank)
            q = N.T @ (q + P @ fixed)
            P = sparse.csc_matrix(N.T @ (P @ N))
            bc = bc - Ac @ fixed
            Ac = sparse.csc_matrix(Ac @ N)
            n = N.shape[1]
            ne = 0
            Ae, be = sparse.csc_matrix((0, n)), np.empty(0)
    cs = ConeSystem(cones, Ac)
    augmented = not cs.psd and not len(cs.exp)
    dense_qr = bool((cs.psd or (len(cs.exp) and n <= 60)) and ne == 0 and 0 < n <= 300
                    and n * len(ic) <= 3000000)
    if dense_qr and P.nnz:
        pe, pv = linalg.eigh(P.toarray(), check_finite=False)
        Proot = np.sqrt(np.maximum(pe, 0))[:, None] * pv.T
    else:
        Proot = np.empty((0, n))
    s, z = cs.initial()
    x, y = np.zeros(n), np.zeros(ne)
    tau, kap = 1.0, 1.0
    homogeneous = True
    best = {"status": "unresolved"}
    besterr = float("inf")
    stalled = 0
    repairs = 0
    started = time.perf_counter()
    validate = validator if validator is not None else (
        lambda result: assess(P0, q0, A0, b0, cones0, result))

    def recover():
        zz, ss = np.zeros(m), np.zeros(m)
        if eq_restore is not None:
            Pe, qe, Ace, fixed, N, Qe, Re, piv, rank = eq_restore
            xf = N @ x + fixed * tau
            residual = -(Pe @ xf + qe * tau + Ace.T @ z)
            yy = np.zeros(len(ie))
            yy[piv[:rank]] = linalg.solve_triangular(Re, Qe.T @ residual, check_finite=False)
        else:
            xf, yy = x, y
        zz[ic], zz[ie], ss[ic] = z, yy, s
        xx = D * xf
        zz *= E / c
        ss /= E
        if tau > 1e-100:
            result = {"status": "solved", "x": xx / tau,
                      "s": ss / tau, "z": zz / tau}
            ok, err = validate(result)
            if ok:
                return result, err
        else:
            result, err = {"status": "unresolved"}, float("inf")
        if kap > tau:
            rz = {"status": "primal_infeasible", "z": zz}
            if validate(rz)[0]:
                return rz, 0.0
            rx = {"status": "dual_infeasible", "x": xx, "s": ss}
            if validate(rx)[0]:
                return rx, 0.0
        return result, err

    for it in range(150):
        result, err = recover()
        if err >= besterr * 0.999:
            stalled += 1
        else:
            stalled = 0
        if err < besterr:
            best, besterr = result, err
        if err <= 6e-9:
            return result
        if it > 35 and stalled >= 6 and besterr < 1e-4:
            break
        if time.perf_counter() - started > 275:
            break
        if not (np.isfinite(x).all() and np.isfinite(s).all() and
                np.isfinite(z).all() and tau > 0 and np.isfinite(tau + kap)):
            break
        if homogeneous and kap < 1e-4 * tau and err < 1e-3:
            x /= tau
            y /= tau
            s /= tau
            z /= tau
            tau, kap = 1.0, 0.0
            homogeneous = False
        px = P @ x
        rd = px + q * tau + Ac.T @ z + Ae.T @ y
        rp = Ac @ x + s - bc * tau
        re = Ae @ x - be * tau
        xpx = x @ px
        rg = q @ x + bc @ z + be @ y + xpx / tau + kap
        mu = (s @ z + tau * kap) / (cs.nu + int(homogeneous))
        if debug:
            print(it, "err", "%.3g" % err, "mu", "%.3g" % mu,
                  "tau", "%.3g" % tau, "kap", "%.3g" % kap,
                  "rp", "%.2g" % norm(rp), "rd", "%.2g" % norm(rd), flush=True)
        try:
            cs.scale(s, z)
            if dense_qr:
                B = cs.scaled_A(Ac)
                bscaled = cs.scale_primal(bc)
                Bb = np.vstack([B, Proot, 1e-6 * np.eye(n)])
                Qr, Rr, pivr = linalg.qr(Bb, mode="economic", pivoting=True, check_finite=False)
                QB = Qr[:len(ic)]
                system_size = n + len(ic)

                def matvec(v):
                    dx, zz = v[:n], v[n:]
                    return np.concatenate([P @ dx + B.T @ zz, B @ dx - zz])

                def linear(rhs):
                    t = linalg.solve_triangular(Rr.T, rhs[:n][pivr], lower=True, check_finite=False)
                    t += QB.T @ rhs[n:]
                    xp = linalg.solve_triangular(Rr, t, check_finite=False)
                    xx = np.empty(n)
                    xx[pivr] = xp
                    zz = QB @ t - rhs[n:]
                    return np.concatenate([xx, zz])
            elif augmented:
                B = cs.T @ Ac
                Bfull = sparse.vstack([Ae, B], format="csc")
                bscaled = np.concatenate([be, cs.scale_primal(bc)])
                Hf = sparse.diags(np.concatenate([np.zeros(ne), np.ones(len(ic))]))
                K = sparse.bmat([[P, Bfull.T], [Bfull, -Hf]], format="csc")
            else:
                Hbc = cs.hmul(bc)
                AtHb = Ac.T @ Hbc
                G = P + cs.gram(Ac)
                K = sparse.bmat([[G, Ae.T], [Ae, None]], format="csc") if ne else G
            if not dense_qr:
                system_size = K.shape[0]
                lowrank = cs.lowrank_columns(Ac, ne) if augmented and cs.large_soc else None
                if lowrank is not None:
                    Kbase = K
                    count = lowrank.shape[1]
                    indices = np.arange(count)
                    J = sparse.coo_matrix((-np.ones(count), (indices, indices ^ 1)),
                                          shape=(count, count)).tocsc()
                    K = sparse.bmat([[Kbase, lowrank], [lowrank.T, J]], format="csc")
                reg = np.concatenate([np.full(n, 1e-12), np.full(K.shape[0] - n, -1e-12)])
                Kr = K + sparse.diags(reg)
                if lowrank is None:
                    matvec = lambda v: K @ v
                else:
                    def matvec(v):
                        t = np.asarray(lowrank.T @ v).reshape(-1, 2)[:, ::-1].ravel()
                        return Kbase @ v + lowrank @ t
                if K.shape[0] < 250 and K.nnz > K.shape[0]**2 * 0.25:
                    kd = Kr.toarray()
                    lu = linalg.lu_factor(kd, check_finite=False)
                    linear = lambda rhs: linalg.lu_solve(lu, rhs, check_finite=False)
                else:
                    lu = splu(Kr, permc_spec="COLAMD")
                    linear = lu.solve
                if lowrank is not None:
                    raw_linear = linear
                    linear = lambda rhs: raw_linear(np.r_[rhs, np.zeros(count)])[:system_size]

            def lin(rhs):
                out = linear(rhs)
                for _ in range(3):
                    residual = rhs - matvec(out)
                    if (norm(residual[:n]) <= 2e-12 * max(1.0, norm(rhs[:n])) and
                            norm(residual[n:]) <= 2e-12 * max(1.0, norm(rhs[n:]))):
                        break
                    out += linear(residual)
                return out

            if augmented or dense_qr:
                dtvec = -lin(np.concatenate([q, -bscaled])) if homogeneous else np.zeros(system_size)
                row = np.concatenate([q + 2 * px / tau, bscaled])
                gamma = -xpx / tau**2 - kap / tau
                denom = gamma + row @ dtvec

                def direction(rhscomp, rhstau, factor=1.0, rgextra=0.0):
                    rhs = np.concatenate([-factor * rd, -factor * re,
                                          -factor * cs.scale_primal(rp) - cs.scale_dual(rhscomp)])
                    base = lin(rhs)
                    dt = (-factor * rg - rgextra - rhstau - row @ base) / denom if homogeneous else 0.0
                    sol = base + dt * dtvec
                    dx, dy = sol[:n], sol[n:n+ne]
                    dz = cs.unscale_dual(sol[n+ne:])
                    ds = -factor * rp - Ac @ dx + bc * dt
                    dk = rhstau - kap / tau * dt if homogeneous else 0.0
                    return dx, dy, ds, dz, dt, dk
            else:
                column = np.concatenate([q - AtHb, -be])
                dtvec = -lin(column) if homogeneous else np.zeros(system_size)
                row = np.concatenate([q + 2 * px / tau + AtHb, be])
                gamma = -(bc @ Hbc) - xpx / tau**2 - kap / tau
                denom = gamma + row @ dtvec

                def direction(rhscomp, rhstau, factor=1.0, rgextra=0.0):
                    v = rhscomp + cs.hmul(factor * rp)
                    rhs = np.concatenate([-factor * rd - Ac.T @ v, -factor * re])
                    base = lin(rhs)
                    dt = (-factor * rg - rgextra - bc @ v - rhstau - row @ base) / denom if homogeneous else 0.0
                    sol = base + dt * dtvec
                    dx, dy = sol[:n], sol[n:]
                    ds = -factor * rp - Ac @ dx + bc * dt
                    dz = rhscomp - cs.hmul(ds)
                    dk = rhstau - kap / tau * dt if homogeneous else 0.0
                    return dx, dy, ds, dz, dt, dk

            aff = direction(cs.complement_rhs(), -kap)
            dx, dy, ds, dz, dt, dk = aff
            if err < 5e-4 and tau + dt > 0:
                saved = (x, y, s, z, tau, kap)
                x, y, s, z = x + dx, y + dy, s + ds, z + dz
                tau, kap = tau + dt, kap + dk
                trial, trial_error = recover()
                x, y, s, z, tau, kap = saved
                if trial_error <= 6e-9:
                    return trial
            alpha = cs.step(s, ds, z, dz)
            if dt < 0:
                alpha = min(alpha, -tau / dt)
            if dk < 0:
                alpha = min(alpha, -kap / dk)
            mua = ((s + alpha * ds) @ (z + alpha * dz) +
                   (tau + alpha * dt) * (kap + alpha * dk)) / (cs.nu + int(homogeneous))
            sigma = min(1.0, max(0.0, mua / mu))**3
            sigma = min(sigma, 0.95)
            rcomp = cs.complement_rhs(sigma * mu, (ds, dz))
            rtau = -kap + (sigma * mu - dt * dk) / tau
            vv = dx - x / tau * dt
            rgextra = (vv @ (P @ vv)) / tau
            combined = direction(rcomp, rtau, 1.0 - sigma, rgextra)
            dx, dy, ds, dz, dt, dk = combined
            alpha = cs.step(s, ds, z, dz)
            if dt < 0:
                alpha = min(alpha, -tau / dt)
            if dk < 0:
                alpha = min(alpha, -kap / dk)
            alpha *= 0.99
            if alpha < 1e-12:
                break
            x += alpha * dx
            y += alpha * dy
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kap += alpha * dk
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            if debug:
                import traceback
                traceback.print_exc()
            if cs.psd and repairs < 3:
                from common import smat, svec
                repaired = False
                for ps in cs.psd:
                    for vector in (s, z):
                        matrix = smat(vector[ps["i"]:ps["j"]], ps["d"])
                        eigen = linalg.eigh(matrix, subset_by_index=[0, 0],
                                           eigvals_only=True, check_finite=False)[0]
                        if eigen <= 1e-15 * norm(matrix):
                            shift = max(0., -eigen) + 1e-14 * max(norm(matrix), 1e-100)
                            matrix.flat[::ps["d"]+1] += shift
                            vector[ps["i"]:ps["j"]] = svec(matrix)
                            repaired = True
                if repaired:
                    repairs += 1
                    continue
            break
    return best
