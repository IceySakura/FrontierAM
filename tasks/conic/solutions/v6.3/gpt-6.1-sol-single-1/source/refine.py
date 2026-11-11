"""Semismooth Newton refinement of the cone-projection KKT equations."""
import os
import sys
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import lsmr
from accuracy import assess, norm


def face_system(P, A, s, z, K):
    """Eliminate the nonsingular part of the projection derivative."""
    n, m = A.shape[1], A.shape[0]
    w = s - z
    projection = np.zeros(m)
    h = np.zeros(m)
    lr, lc, lv = [], [], []
    tr, tc, tv = [], [], []
    active_rhs = []
    active_count = 0
    psd = []

    def add_active(rows, weights, rhs):
        nonlocal active_count
        rows = np.asarray(rows, dtype=int)
        tr.append(np.full(rows.size, active_count))
        tc.append(rows)
        tv.append(np.asarray(weights))
        active_rhs.append(float(rhs))
        active_count += 1

    ids = np.r_[K.zi, K.ni[w[K.ni] <= 0]]
    if ids.size:
        tr.append(np.arange(ids.size))
        tc.append(ids)
        tv.append(np.ones(ids.size))
        active_rhs.extend(s[ids])
        active_count = ids.size
    ids = K.ni[w[K.ni] > 0]
    projection[ids] = w[ids]
    h[ids] = z[ids]
    for dim, ids in K.soc.items():
        for rows in ids:
            ww = w[rows]
            r = np.linalg.norm(ww[1:])
            if ww[0] >= r:
                projection[rows] = ww
                h[rows] = z[rows]
            elif ww[0] <= -r:
                for j in rows:
                    add_active([j], [1.], s[j])
            else:
                u = ww[1:] / r
                p = .5 * (ww[0] + r) * np.r_[1., u]
                projection[rows] = p
                r3 = s[rows] - p
                dot = float(u @ r3[1:])
                h[rows] = .5 * (r3[0] + dot) * np.r_[1., u]
                alpha = .5 * (1 + ww[0] / r)
                h[rows[1:]] += (r3[1:] - dot * u) / max(alpha, 1e-14)
                vminus = np.r_[1., -u] / np.sqrt(2.)
                add_active(rows, vminus, vminus @ r3)
                ratio = (1 - alpha) / max(alpha, 1e-14)
                L = np.zeros((dim, dim))
                L[1:, 1:] = ratio * (np.eye(dim-1) - np.outer(u, u))
                rr, cc = np.meshgrid(rows, rows, indexing="ij")
                lr.append(rr.ravel())
                lc.append(cc.ravel())
                lv.append(L.ravel())
    if K.ei.size:
        ss, zz = s[K.ei], z[K.ei]
        szscale = np.maximum(1., np.max(np.abs(ss), axis=1))
        closure = (ss[:, 1] < 1e-7*szscale) & (ss[:, 0] < -1e-6*szscale)
        rows = K.ei[closure]
        if len(rows):
            projection[rows] = ss[closure]
            projection[rows[:, 1]] = 0.
            h[rows] = zz[closure]
            h[rows[:, 1]] = 0.
            tr.append(np.arange(active_count, active_count+len(rows)))
            tc.append(rows[:, 1])
            tv.append(np.ones(len(rows)))
            active_rhs.extend(ss[closure, 1])
            active_count += len(rows)
        rows, ss, zz, szscale = K.ei[~closure], ss[~closure], zz[~closure], szscale[~closure]
        if len(rows):
            v, wv = np.maximum(ss[:, 1], 1e-14), np.maximum(ss[:, 2], 1e-14)
            ell = np.log(wv)-np.log(v)
            psi = v*ell-ss[:, 0]
            p = np.stack((-np.ones(len(rows)), ell-1., v/wv), axis=1)
            pnorm = np.sum(p*p, axis=1)
            lam = np.maximum(0., np.sum(zz*p, axis=1)/pnorm)
            interior = (np.max(np.abs(zz), axis=1) < 1e-6) & (psi > 1e-4*szscale)
            h[rows[interior]] = zz[interior]
            projection[rows[interior]] = ss[interior]-zz[interior]
            active = ~interior
            rows, ss, zz = rows[active], ss[active], zz[active]
            p, lam, psi, pnorm = p[active], lam[active], psi[active], pnorm[active]
            v, wv = v[active], wv[active]
            if len(rows):
                h[rows] = zz-lam[:, None]*p
                projection[rows] = ss-(psi/pnorm)[:, None]*p-h[rows]
                tr.append(np.repeat(np.arange(active_count, active_count+len(rows)), 3))
                tc.append(rows.ravel())
                tv.append(p.ravel())
                active_rhs.extend(psi)
                active_count += len(rows)
                L = np.zeros((len(rows), 3, 3))
                L[:, 1, 1] = lam/v
                L[:, 1, 2] = L[:, 2, 1] = -lam/wv
                L[:, 2, 2] = lam*v/wv**2
                lr.append(np.broadcast_to(rows[:, :, None], L.shape).ravel())
                lc.append(np.broadcast_to(rows[:, None, :], L.shape).ravel())
                lv.append(L.ravel())
    C = P.copy()
    dense_add = None
    for block in K.psd:
        sl, d, ii, jj, weight = block
        S, Z = K.mat(s, block), K.mat(z, block)
        eig, U = linalg.eigh(S - Z, check_finite=False, driver="evd")
        pos = eig > 0
        neg = ~pos
        positive = np.maximum(eig, 0)
        den = eig[:, None] - eig[None, :]
        num = positive[:, None] - positive[None, :]
        omega = np.zeros_like(den)
        different = np.abs(den) > 1e-14 * max(1., norm(eig))
        np.divide(num, den, out=omega, where=different)
        omega[pos[:, None] & pos[None, :]] = 1.
        omega[neg[:, None] & neg[None, :]] = 0.
        R3 = U.T @ S @ U - np.diag(positive)
        mask = omega > 0
        Hh = np.zeros_like(omega)
        np.divide(R3, omega, out=Hh, where=mask)
        Lop = np.zeros_like(omega)
        np.divide(1-omega, omega, out=Lop, where=mask)
        Lop = np.maximum(Lop, 0)
        projection[sl] = K.vec((U * positive) @ U.T, block)
        h[sl] = K.vec(U @ Hh @ U.T, block)
        AB = A[sl, :].tocsc()
        cols = np.flatnonzero(np.diff(AB.indptr))
        B = AB[:, cols].toarray().T
        mats = np.zeros((cols.size, d, d))
        mats[:, ii, jj] = B / weight
        mats[:, jj, ii] = mats[:, ii, jj]
        rotated = U.T @ mats @ U
        flat = (rotated * np.sqrt(Lop)).reshape(cols.size, d*d)
        gram = flat @ flat.T
        if dense_add is None:
            dense_add = np.zeros((n, n))
        dense_add[np.ix_(cols, cols)] += gram
        ni = np.flatnonzero(neg)
        ai, aj = np.triu_indices(ni.size)
        ai, aj = ni[ai], ni[aj]
        aw = np.where(ai == aj, 1., np.sqrt(2.))
        Erows = rotated[:, ai, aj].T * aw[:, None]
        rhs = R3[ai, aj] * aw
        local_start = active_count
        for j in range(ai.size):
            add_active([], [], rhs[j])
        psd.append((block, U, Lop, ni, ai, aj, aw, local_start, cols, Erows))
    if lr:
        Lsp = sparse.csc_matrix((np.concatenate(lv), (np.concatenate(lr), np.concatenate(lc))),
                               shape=(m, m))
        C += A.T @ Lsp @ A
    else:
        Lsp = sparse.csc_matrix((m, m))
    if dense_add is not None:
        C += sparse.csc_matrix(dense_add)
    if tr:
        T = sparse.csc_matrix((np.concatenate(tv), (np.concatenate(tr), np.concatenate(tc))),
                              shape=(active_count, m))
    else:
        T = sparse.csc_matrix((active_count, m))
    E = (T @ A).tocsc()
    for block, U, Lop, ni, ai, aj, aw, st, cols, Erows in psd:
        rows = np.arange(st, st + ai.size)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        E += sparse.csc_matrix((Erows.ravel(), (rr.ravel(), cc.ravel())),
                               shape=(active_count, n))
    return C, E, np.asarray(active_rhs), projection, h, Lsp, T, psd


def refine(P, q, A, b, K, initial, original=None, maxiter=8):
    if q.size > 10000 or (K.psd and q.size > 3000):
        if original is None:
            return initial
        _, _, _, _, D, E, c = original
        return ("solved", D * initial[1], initial[2] / E, E * initial[3] / c)
    if any(d > 300 for d in K.soc):
        if original is None:
            return initial
        _, _, _, _, D, E, c = original
        return ("solved", D * initial[1], initial[2] / E, E * initial[3] / c)
    status, x, s, z = initial
    if status != "solved":
        return initial
    x, z = x.copy(), z.copy()
    if original is None:
        P0, q0, A0, b0, D, E0, c = P, q, A, b, np.ones(q.size), np.ones(b.size), 1.
    else:
        P0, q0, A0, b0, D, E0, c = original

    def restore(xx, ss, zz):
        return "solved", D * xx, ss / E0, E0 * zz / c

    best = restore(x, s, z)
    score = assess(P0, q0, A0, b0, K, best)[0]
    rd = P @ x + q + A.T @ z
    gap = float(x @ (P @ x) + q @ x + b @ z)
    if norm(rd) < 1e-3*(1+norm(q)):
        for objective in (False, True):
            if objective:
                bs = max(1., norm(b))
                B = sparse.vstack([A.T, sparse.csr_matrix((b/bs)[None, :])], format="csc")
                rhs = np.r_[-rd, -gap/bs]
            else:
                B, rhs = A.T.tocsc(), -rd
            try:
                if min(B.shape) <= 250 and B.shape[0]*B.shape[1] <= 2000000:
                    dz = linalg.lstsq(B.toarray(), rhs, cond=1e-12,
                                      lapack_driver="gelsd", check_finite=False)[0]
                else:
                    dz = lsmr(B, rhs, atol=1e-14, btol=1e-14,
                              maxiter=min(250, 2*min(B.shape)+1))[0]
                for slack in (s, b-A@x):
                    trial = restore(x, slack, z+dz)
                    sc = assess(P0, q0, A0, b0, K, trial)[0]
                    if sc < score:
                        best, score = trial, sc
                    if score <= .5:
                        return best
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass
    for it in range(maxiter):
        s = b - A @ x
        try:
            C, E, arhs, projection, h, Lsp, T, psd = face_system(P, A, s, z, K)
            r1 = P @ x + q + A.T @ z
            r3 = s - projection
            merit = norm(r1) / (1 + norm(q)) + norm(r3) / (1 + norm(s))
            M = sparse.bmat([[C, E.T], [E, None]], format="csc")
            rhs = np.r_[-r1 + A.T @ h, arhs]
            if ((K.psd and M.shape[0] < 300) or
                    (not P.nnz and M.shape[0] < 350)):
                default_rcond = (max(1e-11, 1e-7 * .1**(max(0, it-3)//2))
                                 if K.psd else 1e-12)
                rcond = float(os.environ.get("SOLVER_FACE_RCOND", str(default_rcond)))
                v = linalg.lstsq(M.toarray(), rhs, cond=rcond,
                                 lapack_driver="gelsd", check_finite=False)[0]
            else:
                delta = 1e-12
                reg = np.r_[np.full(q.size, delta), np.full(E.shape[0], -delta)]
                lu = splu(M + sparse.diags(reg, format="csc"), permc_spec="COLAMD")
                v = lu.solve(rhs)
                for _ in range(5):
                    rr = rhs - M @ v
                    if norm(rr) < 1e-13 * (1 + norm(rhs)):
                        break
                    v += lu.solve(rr)
            dx, y = v[:q.size], v[q.size:]
            adx = A @ dx
            dz = Lsp @ adx - h + T.T @ y
            for block, U, Lop, ni, ai, aj, aw, st, cols, Erows in psd:
                W = Lop * (U.T @ K.mat(adx, block) @ U)
                vals = y[st:st+ai.size] / aw
                W[ai, aj] += vals
                W[aj, ai] = W[ai, aj]
                dz[block[0]] += K.vec(U @ W @ U.T, block)
            accepted = False
            for back in range(12):
                alpha = .5 ** back
                xx, zz = x + alpha * dx, z + alpha * dz
                ss = b - A @ xx
                trial = restore(xx, ss, zz)
                sc, res = assess(P0, q0, A0, b0, K, trial)
                if sc < score:
                    best, score = trial, sc
                if os.environ.get("SOLVER_DEBUG") and back == 0:
                    print("refine", it, sc, res, file=sys.stderr)
                if score <= .5:
                    return best
                # Far from the root, cone defects give a useful merit test.
                newmerit = norm(P @ xx + q + A.T @ zz) / (1 + norm(q))
                newmerit += K.violation(ss) + K.violation(zz, dual=True)
                if sc < 1. or newmerit <= merit * (1 - .01 * alpha):
                    x, z = xx, zz
                    accepted = True
                    break
            if not accepted:
                break
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            if os.environ.get("SOLVER_DEBUG"):
                import traceback
                traceback.print_exc()
            break
    return best
