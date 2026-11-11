"""Homogeneous primal-dual interior-point method for small cone blocks."""

import math
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm, residuals
from scaling import equilibrate, recover


def batch_mv(A, v):
    return np.einsum("bij,bj->bi", A, v)


def exp_barrier(v, derivative=None):
    """Gradient, Hessian, and optional third directional derivative."""
    a, b, c = v.T
    ell = np.log(c) - np.log(b)
    t = b * ell - a
    dt = np.column_stack((-np.ones(a.size), ell - 1, b / c))
    g = -dt / t[:, None]
    g[:, 1] -= 1 / b
    g[:, 2] -= 1 / c
    Ht = np.zeros((a.size, 3, 3))
    Ht[:, 1, 1] = -1 / b
    Ht[:, 1, 2] = Ht[:, 2, 1] = 1 / c
    Ht[:, 2, 2] = -b / c**2
    H = dt[:, :, None] * dt[:, None, :] / t[:, None, None]**2
    H -= Ht / t[:, None, None]
    H[:, 1, 1] += 1 / b**2
    H[:, 2, 2] += 1 / c**2
    third = None
    if derivative is not None:
        da, db, dc = derivative.T
        td = np.sum(dt * derivative, axis=1)
        Htd = batch_mv(Ht, derivative)
        tdd = np.sum(Htd * derivative, axis=1)
        Tdd = np.zeros_like(v)
        Tdd[:, 1] = db**2 / b**2 - dc**2 / c**2
        Tdd[:, 2] = -2 * db * dc / c**2 + 2 * b * dc**2 / c**3
        third = -Tdd / t[:, None]
        third += (2 * Htd * td[:, None] + dt * tdd[:, None]) / t[:, None]**2
        third -= 2 * dt * td[:, None]**2 / t[:, None]**3
        third[:, 1] -= 2 * db**2 / b**3
        third[:, 2] -= 2 * dc**2 / c**3
    return g, H, third


def exp_transform(v):
    out = np.empty_like(v)
    out[:, 0], out[:, 1], out[:, 2] = -v[:, 1], -v[:, 0], math.e * v[:, 2]
    return out


class Product:
    def __init__(self, cones, m):
        eq, nn, soc, exp = [], [], {}, []
        self.degree = 0
        for kind, dim, sl in blocks(cones):
            idx = np.arange(sl.start, sl.stop)
            if kind == "zero":
                eq.extend(idx)
            elif kind == "nonnegative":
                nn.extend(idx)
                self.degree += dim
            elif kind == "soc":
                soc.setdefault(dim, []).append(idx)
                self.degree += 1
            elif kind == "exponential":
                exp.append(idx)
                self.degree += 3
            else:
                raise ValueError("Unsupported cone in sparse homogeneous backend")
        self.eq = np.asarray(eq, dtype=np.int32)
        self.nn = np.asarray(nn, dtype=np.int32)
        self.soc = []
        for d, indices in soc.items():
            if d > 1000:
                raise ValueError("SOC block too large for explicit scaling")
            idx = np.asarray(indices, dtype=np.int32)
            J = -np.ones(d)
            J[0] = 1
            self.soc.append({"idx": idx, "J": J, "d": d})
        self.exp = np.asarray(exp, dtype=np.int32).reshape(-1, 3)
        self.m = m
        rows, cols = [self.nn], [self.nn]
        for group in self.soc:
            idx, d = group["idx"], group["d"]
            rows.append(np.repeat(idx, d, axis=1).ravel())
            cols.append(np.tile(idx, (1, d)).ravel())
        if self.exp.size:
            rows.append(np.repeat(self.exp, 3, axis=1).ravel())
            cols.append(np.tile(self.exp, (1, 3)).ravel())
        self.rows, self.cols = np.concatenate(rows), np.concatenate(cols)
        self.L = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., math.e]])

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.nn], z[self.nn] = 1, 1
        for group in self.soc:
            s[group["idx"][:, 0]] = 1
            z[group["idx"][:, 0]] = 1
        s[self.exp] = np.array([-1., 1., 2.])
        z[self.exp] = np.array([-1., 0., 1.])
        return s, z

    def update(self, s, z, mu):
        self.mu = mu
        self.ratio = z[self.nn] / s[self.nn]
        values = [1.0 / self.ratio]
        for group in self.soc:
            idx, J, d = group["idx"], group["J"], group["d"]
            ss, zz = s[idx], z[idx]
            sn, zn = np.linalg.norm(ss[:, 1:], axis=1), np.linalg.norm(zz[:, 1:], axis=1)
            ds = (ss[:, 0] - sn) * (ss[:, 0] + sn)
            dz = (zz[:, 0] - zn) * (zz[:, 0] + zn)
            if np.min(ds) <= 0 or np.min(dz) <= 0:
                raise ArithmeticError("SOC scaling is not interior")
            detw = np.sqrt(ds / dz)
            rootdet = np.sqrt(ds * dz)
            w = (ss + detw[:, None] * zz * J) / np.sqrt(
                2 * (np.sum(ss * zz, axis=1) + rootdet))[:, None]
            winv = w * J / detw[:, None]
            Hi = 2 * winv[:, :, None] * winv[:, None, :]
            Hi[:, np.arange(d), np.arange(d)] -= J / detw[:, None]
            H = 2 * w[:, :, None] * w[:, None, :]
            H[:, np.arange(d), np.arange(d)] -= J * detw[:, None]
            wn = np.linalg.norm(w[:, 1:], axis=1)
            a = np.sqrt(np.maximum(w[:, 0] + wn, 1e-300))
            b = np.sqrt(np.maximum(detw, 1e-300)) / a
            wr = np.empty_like(w)
            wr[:, 0] = 0.5 * (a + b)
            wr[:, 1:] = w[:, 1:] / (a + b)[:, None]
            R = 2 * wr[:, :, None] * wr[:, None, :]
            R[:, np.arange(d), np.arange(d)] -= J * np.sqrt(detw)[:, None]
            wi = wr * J / np.sqrt(detw)[:, None]
            Ri = 2 * wi[:, :, None] * wi[:, None, :]
            Ri[:, np.arange(d), np.arange(d)] -= J / np.sqrt(detw)[:, None]
            lam = batch_mv(R, zz)
            group.update(Hi=Hi, H=H, R=R, Ri=Ri, lam=lam, detlam=rootdet)
            values.append(H.ravel())
        if self.exp.size:
            transformed = exp_transform(z[self.exp])
            grad, H, unused = exp_barrier(transformed)
            self.eg = grad @ self.L
            self.eH = np.matmul(np.matmul(self.L.T, H), self.L)
            self.eHi = np.linalg.inv(mu * self.eH)
            self.ez = transformed
            values.append((mu * self.eH).ravel())
        return sparse.csc_matrix((np.concatenate(values), (self.rows, self.cols)),
                                 shape=(self.m, self.m))

    def inverse(self, v):
        out = np.zeros(self.m)
        out[self.nn] = self.ratio * v[self.nn]
        for group in self.soc:
            out[group["idx"]] = batch_mv(group["Hi"], v[group["idx"]])
        if self.exp.size:
            out[self.exp] = batch_mv(self.eHi, v[self.exp])
        return out

    def multiply(self, v):
        out = np.zeros(self.m)
        out[self.nn] = v[self.nn] / self.ratio
        for group in self.soc:
            out[group["idx"]] = batch_mv(group["H"], v[group["idx"]])
        if self.exp.size:
            out[self.exp] = self.mu * batch_mv(self.eH, v[self.exp])
        return out

    def corrector(self, s, z, ds, dz, target):
        rhs = -s.copy()
        rhs[self.nn] += (target - ds[self.nn] * dz[self.nn]) / z[self.nn]
        for group in self.soc:
            idx, lam = group["idx"], group["lam"]
            a, b = batch_mv(group["Ri"], ds[idx]), batch_mv(group["R"], dz[idx])
            r = -(a[:, :1] * b + b[:, :1] * a)
            r[:, 0] = target - np.sum(a * b, axis=1)
            t = (lam[:, 0] * r[:, 0] - np.sum(lam[:, 1:] * r[:, 1:], axis=1)) / group["detlam"]
            v = np.empty_like(r)
            v[:, 0] = t
            v[:, 1:] = (r[:, 1:] - lam[:, 1:] * t[:, None]) / lam[:, :1]
            rhs[idx] += batch_mv(group["R"], v)
        if self.exp.size:
            dx = exp_transform(dz[self.exp])
            unused, unused2, third = exp_barrier(self.ez, dx)
            correction = self.mu * batch_mv(self.eH, dz[self.exp])
            correction -= 0.5 * self.mu * (third @ self.L)
            rhs[self.exp] += -target * self.eg + correction
        return rhs

    def step(self, v, dv, dual=False):
        negative = dv[self.nn] < 0
        alpha = min(1.0, float(np.min(-v[self.nn][negative] / dv[self.nn][negative], initial=1.0)))
        for group in self.soc:
            vv, dd = v[group["idx"]], dv[group["idx"]]
            length = np.linalg.norm(vv[:, 1:], axis=1)
            c = (vv[:, 0] - length) * (vv[:, 0] + length)
            a = dd[:, 0]**2 - np.sum(dd[:, 1:]**2, axis=1)
            b = 2 * (vv[:, 0] * dd[:, 0] - np.sum(vv[:, 1:] * dd[:, 1:], axis=1))
            den = np.sqrt(np.maximum(0, b * b - 4 * a * c)) - b
            valid = den > 0
            if np.any(valid):
                alpha = min(alpha, float(np.min(2 * c[valid] / den[valid])))
            valid = dd[:, 0] < 0
            if np.any(valid):
                alpha = min(alpha, float(np.min(-vv[valid, 0] / dd[valid, 0])))
        if self.exp.size:
            vv, dd = v[self.exp], dv[self.exp]
            if dual:
                vv, dd = exp_transform(vv), exp_transform(dd)
            for col in (1, 2):
                negative = dd[:, col] < 0
                if np.any(negative):
                    alpha = min(alpha, 0.999 * float(np.min(-vv[negative, col] / dd[negative, col])))
            for search in range(100):
                trial = vv + alpha * dd
                a, b, c = trial.T
                if np.all((b > 0) & (c > 0)):
                    t = b * (np.log(c) - np.log(b)) - a
                    if np.all(t > 0):
                        break
                alpha *= 0.8
        return max(0.0, min(1.0, alpha))

    def repair_interior(self, v):
        v[self.nn] = np.maximum(v[self.nn], 1e-100)
        for group in self.soc:
            idx = group["idx"]
            length = np.linalg.norm(v[idx[:, 1:]], axis=1)
            v[idx[:, 0]] = np.maximum(v[idx[:, 0]], np.maximum(
                length * (1 + 4 * np.finfo(float).eps), 1e-100))


def solve(original, deadline, return_best=False, target=8e-9):
    p, D, E, cost = equilibrate(original)
    product = Product(p.cones, p.m)
    n, ieq = p.n, product.eq
    A, AT = p.A, p.A.T
    B = A[ieq].tocsc()
    x = np.zeros(n)
    s, z = product.initial()
    tau, kappa = 1.0, 1.0
    degree = product.degree + 1
    prior = None
    prior_mu = None
    best = None
    best_error = float("inf")
    polish_attempts = 0
    stagnant = 0
    for iteration in range(150):
        if time.monotonic() > deadline:
            break
        product.repair_interior(s)
        product.repair_interior(z)
        Px = p.P @ x
        rd = Px + p.q * tau + AT @ z
        rp = A @ x + s - p.b * tau
        xPx = float(x @ Px)
        rg = float(p.q @ x + p.b @ z + xPx / tau + kappa)
        mu = float(s @ z + tau * kappa) / degree
        result = recover({"status": "solved", "x": x / tau,
                          "s": s / tau, "z": z / tau}, D, E, cost)
        errors = residuals(original, result)
        error = max(errors.values())
        stagnant = 0 if error < 0.95 * best_error else stagnant + 1
        if error < best_error:
            best_error = error
            best = {key: val.copy() if isinstance(val, np.ndarray) else val
                    for key, val in result.items()}
        if os.environ.get("CONIC_DEBUG"):
            print("HSD", iteration, "mu", mu, "tau", tau, "k/t", kappa / tau,
                  "res", errors, file=sys.stderr)
        if error <= target:
            return result
        if error < (1e-4 if polish_attempts == 0 else 1e-6) and not product.exp.size and polish_attempts < 2:
            from polish_soc import polish
            polished = polish(original, result, deadline)
            polish_attempts += 1
            if polished is not None:
                return polished
        if stagnant >= (8 if best_error < 1e-4 else 25):
            break
        if prior is not None and prior_mu > mu * 1.1 and error < 1e-4:
            weight = min(0.5, mu / (prior_mu - mu))
            trial = {"status": "solved"}
            for key in ("x", "s", "z"):
                trial[key] = result[key] + weight * (result[key] - prior[key])
            if acceptable(original, trial):
                return trial
        if prior_mu is None or mu < 0.75 * prior_mu:
            prior = {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in result.items()}
            prior_mu = mu
        if iteration > 4:
            cert = recover({"status": "primal_infeasible", "z": z}, D, E, cost)
            if acceptable(original, cert):
                return cert
            cert = recover({"status": "dual_infeasible", "x": x, "s": s}, D, E, cost)
            if acceptable(original, cert):
                return cert
        try:
            H = product.update(s, z, mu)
            K = sparse.bmat([[p.P, AT], [A, -H]], format="csc")
            reg = np.r_[np.full(n, 1e-10), np.full(p.m, -1e-10)]
            large = n > 1000
            factor = sla.splu(K + sparse.diags(reg, format="csc"),
                              diag_pivot_thresh=0.0 if large else 0.01,
                              permc_spec="MMD_AT_PLUS_A" if large else "COLAMD",
                              options={"SymmetricMode": large})
            col = np.r_[p.q, -p.b]
            row = np.r_[p.q + 2 * Px / tau, p.b]
            diagonal = -xPx / tau**2 - kappa / tau
        except (RuntimeError, ArithmeticError, np.linalg.LinAlgError):
            break

        def linear_solve(rhs):
            value = factor.solve(rhs)
            for refinement in range(8):
                res = rhs - K @ value
                if norm(res) < 1e-14 * norm(rhs) + 1e-18:
                    break
                update = factor.solve(res)
                trial = value + update
                if norm(rhs - K @ trial) > 1.1 * norm(res):
                    break
                value = trial
            return value

        v = linear_solve(col)
        denominator = diagonal - row @ v
        if not np.isfinite(denominator) or abs(denominator) < 1e-20:
            break

        def direction(cone_rhs, scalar_rhs, reduction, quadratic_correction=0.0):
            rhs = np.r_[-reduction * rd, -reduction * rp - cone_rhs]
            u = linear_solve(rhs)
            rhs_g = -reduction * rg - kappa / tau * scalar_rhs
            rhs_g -= quadratic_correction
            dtau = float((rhs_g - row @ u) / denominator)
            sol = u - v * dtau
            for refinement in range(3):
                r1 = rhs - K @ sol - col * dtau
                r2 = rhs_g - row @ sol - diagonal * dtau
                if max(norm(r1), abs(r2)) < 1e-14 * max(norm(rhs), abs(rhs_g)) + 1e-18:
                    break
                du = linear_solve(r1)
                ddt = float((r2 - row @ du) / denominator)
                sol += du - v * ddt
                dtau += ddt
            dx = sol[:n]
            dz = sol[n:]
            ds = cone_rhs - product.multiply(dz)
            ds[ieq] = 0
            dkappa = kappa / tau * (scalar_rhs - dtau)
            return dx, ds, dz, dtau, dkappa

        def steps(ds, dz, dtau, dkappa):
            ap, ad = product.step(s, ds), product.step(z, dz, True)
            if dtau < 0:
                ap = min(ap, -tau / dtau)
            if dkappa < 0:
                ad = min(ad, -kappa / dkappa)
            return ap, ad

        try:
            dx, ds, dz, dtau, dkappa = direction(-s, -tau, 1.0)
            ap, ad = steps(ds, dz, dtau, dkappa)
            muaff = ((s + ap * ds) @ (z + ad * dz)
                     + (tau + ap * dtau) * (kappa + ad * dkappa)) / degree
            sigma = float(np.clip((muaff / max(mu, 1e-300))**3, 0.0, 1.0))
            cone_rhs = product.corrector(s, z, ds, dz, sigma * mu)
            scalar_rhs = -tau + (sigma * mu - dtau * dkappa) / kappa
            qcor = 0.0
            if p.P.nnz:
                d = dx - x / tau * dtau
                qcor = float(d @ (p.P @ d)) / tau
            dx, ds, dz, dtau, dkappa = direction(cone_rhs, scalar_rhs, 1 - sigma, qcor)
            ap, ad = steps(ds, dz, dtau, dkappa)
            alpha = min(1.0, 0.99 * min(ap, ad))
            if alpha < 1e-14 or not np.isfinite(np.r_[dx, ds, dz, dtau, dkappa]).all():
                break
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dtau
            kappa += alpha * dkappa
        except (ArithmeticError, np.linalg.LinAlgError):
            break
    if return_best and best is not None:
        return best
    return {"status": "unresolved"}
