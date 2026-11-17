import os
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from cones import ConeSystem
from model import assess, equilibrate, norm, repair_slack


class KKT:
    def __init__(self, problem, cones):
        n, other = problem.n, cones.other
        self.n, self.other, self.cones = n, other, cones
        self.A = problem.A if other.size == problem.m else problem.A[other, :].tocsc()
        cursor = n + other.size
        for dim, block in cones.soc.items():
            if dim > 24:
                block["aux"] = np.arange(cursor, cursor + block["ids"].shape[0])
                cursor += block["ids"].shape[0]
                block["aux_minus"] = np.arange(cursor, cursor + block["ids"].shape[0])
                cursor += block["ids"].shape[0]
        self.size = cursor
        self.backend = None
        self.backend_attempted = False
        self.dense_allowed = (not cones.psd and not cones.soc and not cones.exp.size
                              and 64 <= n <= 2000 and cones.zero.size <= 128
                              and problem.P.nnz > 0.08 * n * n)
        self.problem = problem
        self.fast = not cones.psd
        if self.fast:
            self._initialize_fast(problem, cones, cursor)
            return
        rows, cols, vals = [], [], []

        def add(rr, cc, vv):
            start = sum(a.size for a in vals)
            rows.append(np.asarray(rr).ravel())
            cols.append(np.asarray(cc).ravel())
            vals.append(np.asarray(vv).ravel())
            return slice(start, start + vals[-1].size)

        P = problem.P.tocoo()
        add(P.row, P.col, P.data)
        A = self.A.tocoo()
        add(A.col, n + A.row, A.data)
        add(n + A.row, A.col, A.data)
        self.diag_slice = add(np.arange(cursor), np.arange(cursor), np.zeros(cursor))
        if cones.linear.size:
            ids = n + cones.inverse_index[cones.linear]
            self.linear_slice = add(ids, ids, np.zeros(ids.size))
        for dim, block in cones.soc.items():
            ids = n + cones.inverse_index[block["ids"]]
            if "aux" in block:
                block["diag_slice"] = add(ids, ids, np.zeros(ids.size))
                aux = np.repeat(block["aux"], dim)
                block["aux1"] = add(ids.ravel(), aux, np.zeros(ids.size))
                block["aux2"] = add(aux, ids.ravel(), np.zeros(ids.size))
                add(block["aux"], block["aux"], np.ones(block["aux"].size))
                minus = np.repeat(block["aux_minus"], dim)
                block["minus1"] = add(ids.ravel(), minus, np.zeros(ids.size))
                block["minus2"] = add(minus, ids.ravel(), np.zeros(ids.size))
                add(block["aux_minus"], block["aux_minus"], -np.ones(block["aux_minus"].size))
            else:
                rr = np.broadcast_to(ids[:, :, None], (ids.shape[0], dim, dim))
                cc = np.broadcast_to(ids[:, None, :], (ids.shape[0], dim, dim))
                block["slice"] = add(rr, cc, np.zeros(rr.size))
        if cones.exp.size:
            ids = n + cones.inverse_index[cones.exp]
            rr = np.broadcast_to(ids[:, :, None], (ids.shape[0], 3, 3))
            cc = np.broadcast_to(ids[:, None, :], (ids.shape[0], 3, 3))
            self.exp_slice = add(rr, cc, np.zeros(rr.size))
        for block in cones.psd:
            variables = block["variables"]
            rr, cc = np.meshgrid(variables, variables, indexing="ij")
            block["kkt_slice"] = add(rr, cc, np.zeros(rr.size))
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        self.values = np.concatenate(vals).astype(float)
        keys = cc.astype(np.int64) * cursor + rr
        unique, self.mapping = np.unique(keys, return_inverse=True)
        self.nnz = unique.size
        indices = (unique % cursor).astype(np.int32)
        column = (unique // cursor).astype(np.int64)
        indptr = np.r_[0, np.cumsum(np.bincount(column, minlength=cursor))].astype(np.int32)
        self.matrix = sparse.csc_matrix(
            (np.zeros(unique.size), indices, indptr), shape=(cursor, cursor))
        self.regularization = np.full(cursor, -1e-9)
        self.regularization[:n] = 1e-9
        for block in cones.soc.values():
            if "aux" in block:
                self.regularization[block["aux"]] = 1e-9
        self.permutation = None
        self.regularized = self.matrix.copy()
        columns = np.repeat(np.arange(cursor), np.diff(self.matrix.indptr))
        self.reg_positions = np.flatnonzero(self.matrix.indices == columns)

    def _initialize_fast(self, problem, cones, cursor):
        n, r = self.n, cursor - self.n
        if not cones.soc and not cones.exp.size:
            self.values = np.zeros(r)
            self.diag_slice = slice(0, r)
            self.linear_slice = cones.inverse_index[cones.linear]
            self.mapping = None
            B = sparse.csc_matrix((np.zeros(r), np.arange(r, dtype=np.int32),
                                   np.arange(r + 1, dtype=np.int32)), shape=(r, r))
        else:
            rows, cols, vals = [], [], []
            position = 0

            def add(rr, cc, vv):
                nonlocal position
                rr, cc, vv = np.asarray(rr).ravel(), np.asarray(cc).ravel(), np.asarray(vv).ravel()
                rows.append(rr)
                cols.append(cc)
                vals.append(vv)
                result = slice(position, position + vv.size)
                position += vv.size
                return result

            self.diag_slice = add(np.arange(r), np.arange(r), np.zeros(r))
            if cones.linear.size:
                ids = cones.inverse_index[cones.linear]
                self.linear_slice = add(ids, ids, np.zeros(ids.size))
            for dim, block in cones.soc.items():
                ids = cones.inverse_index[block["ids"]]
                if "aux" in block:
                    aux = block["aux"] - n
                    block["diag_slice"] = add(ids, ids, np.zeros(ids.size))
                    repeated = np.repeat(aux, dim)
                    block["aux1"] = add(ids.ravel(), repeated, np.zeros(ids.size))
                    block["aux2"] = add(repeated, ids.ravel(), np.zeros(ids.size))
                    add(aux, aux, np.ones(aux.size))
                    minus = block["aux_minus"] - n
                    repeated_minus = np.repeat(minus, dim)
                    block["minus1"] = add(ids.ravel(), repeated_minus, np.zeros(ids.size))
                    block["minus2"] = add(repeated_minus, ids.ravel(), np.zeros(ids.size))
                    add(minus, minus, -np.ones(minus.size))
                else:
                    rr = np.broadcast_to(ids[:, :, None], (ids.shape[0], dim, dim))
                    cc = np.broadcast_to(ids[:, None, :], (ids.shape[0], dim, dim))
                    block["slice"] = add(rr, cc, np.zeros(rr.size))
            if cones.exp.size:
                ids = cones.inverse_index[cones.exp]
                rr = np.broadcast_to(ids[:, :, None], (ids.shape[0], 3, 3))
                cc = np.broadcast_to(ids[:, None, :], (ids.shape[0], 3, 3))
                self.exp_slice = add(rr, cc, np.zeros(rr.size))
            rr, cc = np.concatenate(rows), np.concatenate(cols)
            self.values = np.concatenate(vals).astype(float)
            unique, self.mapping = np.unique(cc.astype(np.int64) * r + rr, return_inverse=True)
            indices = (unique % r).astype(np.int32)
            column = unique // r
            indptr = np.r_[0, np.cumsum(np.bincount(column, minlength=r))].astype(np.int32)
            B = sparse.csc_matrix((np.zeros(unique.size), indices, indptr), shape=(r, r))
        P = problem.P.copy()
        P.setdiag(P.diagonal())
        P.sort_indices()
        extra = r - self.other.size
        if extra:
            augmented_A = sparse.vstack(
                (self.A, sparse.csc_matrix((extra, n))), format="csc")
        else:
            augmented_A = self.A
        AT = augmented_A.T.tocsc()
        self.matrix = sparse.bmat([[P, AT], [augmented_A, B]], format="csc")
        bcol = np.repeat(np.arange(r), np.diff(B.indptr))
        self.b_positions = (self.matrix.indptr[n + bcol] + np.diff(AT.indptr)[bcol]
                            + np.arange(B.nnz) - B.indptr[bcol])
        pcol = np.repeat(np.arange(n), np.diff(P.indptr))
        pdiag = np.flatnonzero(P.indices == pcol)
        p_positions = self.matrix.indptr[:n] + pdiag - P.indptr[:-1]
        bdiag = np.flatnonzero(B.indices == bcol)
        self.reg_positions = np.r_[p_positions, self.b_positions[bdiag]]
        self.regularization = np.full(cursor, -1e-9)
        self.regularization[:n] = 1e-9
        for block in cones.soc.values():
            if "aux" in block:
                self.regularization[block["aux"]] = 1e-9
        self.regularized = self.matrix.copy()
        self.nnz = B.nnz
        self.permutation = None

    def update(self):
        cones, values = self.cones, self.values
        values[self.diag_slice] = 0.0
        if cones.linear.size:
            values[self.linear_slice] = -cones.linear_inverse
        for dim, block in cones.soc.items():
            w, det = block["w"], block["det"]
            if "aux" in block:
                values[block["diag_slice"]] = np.broadcast_to(-det[:, None], w.shape).ravel()
                values[block["aux1"]] = block["uplus"].ravel()
                values[block["aux2"]] = block["uplus"].ravel()
                values[block["minus1"]] = block["uminus"].ravel()
                values[block["minus2"]] = block["uminus"].ravel()
            else:
                Hinv = 2.0 * w[:, :, None] * w[:, None, :]
                ar = np.arange(dim)
                Hinv[:, ar, ar] -= det[:, None] * np.r_[1.0, -np.ones(dim - 1)]
                values[block["slice"]] = -Hinv.ravel()
        if cones.exp.size:
            values[self.exp_slice] = -cones.exp_inverse.ravel()
        for block in cones.psd:
            values[block["kkt_slice"]] = block["normal"].ravel()
        if self.fast:
            updated = (values if self.mapping is None else
                       np.bincount(self.mapping, weights=values, minlength=self.nnz))
            self.matrix.data[self.b_positions] = updated
        else:
            self.matrix.data[:] = np.bincount(
                self.mapping, weights=values, minlength=self.nnz)
        self.regularized.data[:] = self.matrix.data
        self.regularized.data[self.reg_positions] += self.regularization
        mode = os.environ.get("SOLVER_FACTOR", "auto")
        if self.backend is not None and mode != "superlu":
            try:
                updated = self.backend.update(cones) if self.dense_allowed else self.backend.update(self.regularized)
                if updated is not False:
                    return
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
            self.backend = None
        if not self.backend_attempted and mode != "superlu" and self.dense_allowed:
            self.backend_attempted = True
            try:
                from factor import DenseFactor
                backend = DenseFactor(self.problem, cones)
                backend.update(cones)
                self.backend = backend
                return
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                self.dense_allowed = False
        large_soc = any("aux" in block for block in cones.soc.values())
        if not self.backend_attempted and mode != "superlu" and self.fast and large_soc:
            self.backend_attempted = True
            try:
                from factor import SparseFactor
                from scipy.sparse.csgraph import reverse_cuthill_mckee
                order = reverse_cuthill_mckee(self.regularized, symmetric_mode=True)
                self.backend = SparseFactor(self.regularized, np.argsort(order),
                                            np.sign(self.regularization))
                return
            except (OSError, RuntimeError, ValueError, MemoryError):
                self.backend = None
        order = os.environ.get("SOLVER_ORDER", "MMD_AT_PLUS_A" if self.fast else "COLAMD")
        pivot = float(os.environ.get("SOLVER_PIVOT", "0.0" if self.fast else "0.1"))
        try:
            self.lu = splu(self.regularized, permc_spec=order, diag_pivot_thresh=pivot)
        except RuntimeError:
            self.lu = splu(self.regularized, permc_spec="COLAMD", diag_pivot_thresh=0.1)
        eligible = (self.fast and self.size >= 300 and not self.dense_allowed
                    and not large_soc)
        if not self.backend_attempted and mode != "superlu" and eligible:
            self.backend_attempted = True
            try:
                from factor import SparseFactor
                self.backend = SparseFactor(self.regularized, self.lu.perm_c,
                                            np.sign(self.regularization))
                self.lu = None
            except (OSError, RuntimeError, ValueError, MemoryError):
                self.backend = None

    def solve(self, rhs):
        linear_solve = self.backend.solve if self.backend is not None else self.lu.solve
        sol = linear_solve(rhs)
        for unused in range(6):
            residual = rhs - self.matrix @ sol
            if norm(residual) < 2e-12 * (1 + norm(rhs)):
                break
            correction = linear_solve(residual)
            if not np.isfinite(correction).all():
                break
            sol += correction
        if self.backend is not None and norm(rhs - self.matrix @ sol) > 1e-7 * (1 + norm(rhs)):
            self.backend = None
            self.lu = splu(self.regularized, permc_spec="COLAMD", diag_pivot_thresh=0.1)
            return self.solve(rhs)
        return sol


def solve(original, max_iter=100):
    started = time.monotonic()
    log = os.environ.get("SOLVER_LOG")
    problem, D, E, cost = equilibrate(original, passes=4 if original.n > 10000 else 6)
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    n, m = problem.n, problem.m
    cones = ConeSystem(problem)
    kkt = KKT(problem, cones)
    other = cones.other
    x = np.zeros(n)
    s, z = cones.initial()
    if cones.soc:
        s, z = s.astype(np.longdouble), z.astype(np.longdouble)
    tau, kappa = 1.0, 1.0
    best, best_error = None, np.inf
    polish_threshold = 1e-5
    qnorm, q_original_norm = norm(q), norm(original.q)
    for iteration in range(max_iter):
        Px, Ax, Atz = P @ x, A @ x, A.T @ z
        xPx = float(x @ Px)
        rd = Px + q * tau + Atz
        rp = Ax + s - b * tau
        rg = float(q @ x + b @ z + xPx / tau + kappa)
        mu = float((s @ z + tau * kappa) / (cones.nu + 1))
        cost_scale = cost * D * tau
        estimate_rd = norm(rd / cost_scale) / (
            1.0 + max(norm(Px / cost_scale), q_original_norm, norm(Atz / cost_scale)))
        estimate_p = (0.5 * xPx / tau ** 2 + float(q @ x) / tau) / cost
        estimate_d = (-0.5 * xPx / tau ** 2 - float(b @ z) / tau) / cost
        estimate_gap = min(abs(estimate_p - estimate_d),
                           abs(estimate_p - estimate_d) / max(1e-300, abs(estimate_p), abs(estimate_d)))
        estimate = max(estimate_rd, estimate_gap)
        if tau > 1e-14 and iteration >= 3 and (estimate < 5e-4 or log):
            xo = np.asarray(D * (x / tau), dtype=float)
            zo = np.asarray(E * (z / tau) / cost, dtype=float)
            so = repair_slack(original, xo, np.asarray(s / tau / E, dtype=float))
            candidate = dict(status="solved", x=xo, s=so, z=zo)
            good, report = assess(original, candidate, full=False)
            if (not good and report.get("dual", np.inf) < 8e-9
                    and min(report.get("gap", np.inf), report.get("relative_gap", np.inf)) < 8e-9):
                repaired = dict(status="solved", x=xo, s=repair_slack(original, xo), z=zo)
                if assess(original, repaired, full=False)[0]:
                    return repaired
            error = max(report.get("dual", np.inf), report.get("primal", np.inf),
                        min(report.get("gap", np.inf),
                            report.get("relative_gap", np.inf)))
            if error < best_error:
                best, best_error = candidate, error
            if log:
                print(iteration, "mu", mu, "tau", tau, "kappa", kappa,
                      "error", error, report, flush=True)
                if iteration >= 12:
                    from cones import soc_det
                    for dim, block in cones.soc.items():
                        for name, vector in (("s", s), ("z", z)):
                            values = vector[block["ids"]]
                            det = soc_det(values)
                            worst = np.argmin(det)
                            print("soc", dim, name, "det", det[worst],
                                  "value", values[worst], flush=True)
            if good:
                return candidate
            polish_size = (n <= 20000 and m <= 150000
                           and max(cones.soc.keys(), default=0) <= 32)
            if error < polish_threshold and polish_size and not cones.psd:
                from polish import smooth
                polished = smooth(original, problem, D, E, cost,
                                  (x / tau).copy(),
                                  np.asarray(z / tau, dtype=float).copy())
                polish_threshold = error * 0.03
                if polished is not None:
                    return polished
        if iteration >= 3 and (tau < 0.05 * kappa or iteration % 10 == 0):
            cert = dict(status="primal_infeasible",
                        z=np.asarray(E * z / cost, dtype=float))
            if assess(original, cert, full=False)[0]:
                return cert
            if q_original_norm:
                cert = dict(status="dual_infeasible", x=D * x,
                            s=-(original.A @ (D * x)))
                if assess(original, cert, full=False)[0]:
                    return cert
        if not np.isfinite(mu) or mu <= 0.0 or tau <= 0.0 or kappa <= 0.0:
            break
        try:
            cones.scaling(s, z)
            kkt.update()
            Hbp = cones.psd_apply(b) if cones.psd else None
            HbA = A.T @ Hbp if cones.psd else np.zeros(n)
            g = np.zeros(kkt.size)
            g[:n] = q - HbA
            g[n:n + other.size] = -b[other]
            w = np.zeros(kkt.size)
            w[:n] = q + 2.0 * Px / tau + HbA
            w[n:n + other.size] = b[other]
            scalar = -xPx / tau ** 2 - kappa / tau
            if cones.psd:
                scalar -= float(b @ Hbp)
            vg = kkt.solve(g)
            denominator = scalar - float(w @ vg)
            if not np.isfinite(denominator) or abs(denominator) < 1e-30:
                break

            def direction(cs, ct, rg_extra=0.0):
                rhs = np.zeros(kkt.size)
                psd_const = None
                rhs[:n] = -rd
                if cones.psd:
                    psd_const = cs.copy()
                    psd_const[cones.nonpsd] = 0.0
                    psd_const += cones.psd_apply(rp)
                    rhs[:n] -= A.T @ psd_const
                Hcs = cones.inverse_apply(cs)
                rhs[n:n + other.size] = -rp[other] - Hcs[other]
                vr = kkt.solve(rhs)
                scalar_rhs = -rg - ct / tau - rg_extra
                if cones.psd:
                    scalar_rhs -= float(b @ psd_const)
                dt = (scalar_rhs - float(w @ vr)) / denominator
                dk = (ct - kappa * dt) / tau
                v = vr - vg * dt
                dx = v[:n]
                dz = np.zeros(m)
                dz[other] = v[n:n + other.size]
                Adx = A @ dx
                ds = -rp - Adx + b * dt
                ds[cones.zero] = 0.0
                if cones.psd:
                    dz += psd_const + cones.psd_apply(Adx - b * dt)
                return dx, ds, dz, dt, dk

            affine_cs = cones.complementarity_rhs(s, z)
            affine = direction(affine_cs, -tau * kappa)
            dxa, dsa, dza, dta, dka = affine
            aa = cones.step(s, z, dsa, dza, tau, kappa, dta, dka)
            mu_affine = float(
                ((s + aa * dsa) @ (z + aa * dza)
                 + (tau + aa * dta) * (kappa + aa * dka)) / (cones.nu + 1))
            sigma = float(np.clip(mu_affine / mu, 0.0, 1.0) ** 3)
            if cones.exp.size:
                sigma = max(sigma, 0.05)
            cs = cones.complementarity_rhs(s, z, sigma * mu, (dsa, dza))
            ct = -tau * kappa + sigma * mu - dta * dka
            quad_correction = 0.0
            if P.nnz:
                v = dxa - (dta / tau) * x
                quad_correction = float(v @ (P @ v)) / tau
            dx, ds, dz, dt, dk = direction(cs, ct, quad_correction)
            alpha = 0.99 * cones.step(s, z, ds, dz, tau, kappa, dt, dk)
            if alpha < 0.2 * aa:
                centered = cones.complementarity_rhs(s, z, max(0.1, sigma) * mu)
                alt = direction(centered, -tau * kappa + max(0.1, sigma) * mu)
                ac = 0.99 * cones.step(s, z, alt[1], alt[2], tau, kappa,
                                       alt[3], alt[4])
                if ac > alpha:
                    dx, ds, dz, dt, dk = alt
                    alpha = ac
                if alpha < 0.2 * aa:
                    dx, ds, dz, dt, dk = affine
                    alpha = 0.95 * aa
            if log:
                print("step", alpha, "sigma", sigma, "affine", aa,
                      "time", time.monotonic() - started, flush=True)
            if alpha < 1e-12:
                break
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kappa += alpha * dk
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError):
            if log:
                import traceback
                traceback.print_exc()
            break
        if time.monotonic() - started > 250.0:
            break
    if best is not None and assess(original, best, tol=1e-8)[0]:
        return best
    if best is not None and n <= 80 and m <= 4000:
        try:
            from small_finish import solve as finish
            result = finish(original, problem, D, E, cost, best["x"] / D)
            if result is not None:
                return result
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
    return None
