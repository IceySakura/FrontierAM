#!/usr/bin/env python3
"""Homogeneous self-dual embedding interior-point solver for
    min 1/2 x'Px + q'x  s.t.  Ax + s = b,  s in K
with K a product of zero, nonnegative, second-order, exponential and PSD cones."""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

from cones import Cones, EXP_CENTRAL
import kkt as kktmod


T0 = time.time()


TIME_CAP = float(os.environ.get("SOLVER_TIME_CAP", "290"))
PIV_TOL = float(os.environ.get("PIV_TOL", "1e-9"))
PIV_MAXDIM = int(os.environ.get("PIV_MAXDIM", "0"))
RETRY_PIV_MAXDIM = int(os.environ.get("RETRY_PIV_MAXDIM", "20000"))
CENT_GAMMA = float(os.environ.get("CENT_GAMMA", "1e-4"))
DS_PRIMAL = os.environ.get("DS_PRIMAL", "0") == "1"


def load_problem(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        q = np.asarray(d["q"], dtype=np.float64).reshape(-1)
        b = np.asarray(d["b"], dtype=np.float64).reshape(-1)
        mats = {}
        for p in ("P", "A"):
            shape = tuple(int(v) for v in np.asarray(d[p + "_shape"]).reshape(-1))
            M = sp.csc_matrix((np.asarray(d[p + "_data"], dtype=np.float64),
                               np.asarray(d[p + "_indices"]), np.asarray(d[p + "_indptr"])), shape=shape)
            M.sum_duplicates()
            M.eliminate_zeros()
            M.sort_indices()
            mats[p] = M
    off = meta.get("objective_offset", 0.0)
    off = 0.0 if off is None else float(off)
    return mats["P"], q, mats["A"], b, meta["cones"], off


def write_result(path, status, x=None, s=None, z=None):
    arrays = {"metadata_json": np.array(json.dumps({"schema_version": 1, "status": status}))}
    if x is not None:
        arrays["x"] = np.asarray(x, dtype=np.float64)
    if s is not None:
        arrays["s"] = np.asarray(s, dtype=np.float64)
    if z is not None:
        arrays["z"] = np.asarray(z, dtype=np.float64)
    tmp = str(path) + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- scaling
def col_absmax(M):
    M = M.tocsc()
    out = np.zeros(M.shape[1])
    if M.nnz:
        nz = np.diff(M.indptr) > 0
        out[nz] = np.maximum.reduceat(np.abs(M.data), M.indptr[:-1][nz])
    return out


def equilibrate(P, A, q, b, cones, iters=25):
    n, m = P.shape[0], A.shape[0]
    D = np.ones(n)
    E = np.ones(m)
    Ps, As = P.tocsc(copy=True), A.tocsc(copy=True)
    AsT = As.T.tocsc()
    for _ in range(iters):
        cn = np.maximum(col_absmax(Ps), col_absmax(As)) if m else col_absmax(Ps)
        rn = col_absmax(AsT) if m else np.zeros(0)
        cn = np.where(cn < 1e-4, 1.0, np.minimum(cn, 1e4))
        rn = np.where(rn < 1e-4, 1.0, np.minimum(rn, 1e4))
        rn = cones.uniformize(rn)
        dD = 1.0 / np.sqrt(cn)
        dE = 1.0 / np.sqrt(rn)
        if np.max(np.abs(cn - 1)) < 1e-3 and (m == 0 or np.max(np.abs(rn - 1)) < 1e-3):
            break
        D *= dD
        E *= dE
        Ps = sp.diags(dD) @ Ps @ sp.diags(dD)
        As = sp.diags(dE) @ As @ sp.diags(dD)
        AsT = As.T.tocsc()
    Ps = Ps.tocsc()
    As = As.tocsc()
    qs = D * q
    bs = E * b
    pn = col_absmax(Ps)
    cscale = max(np.mean(pn) if n else 0.0, np.max(np.abs(qs), initial=0.0))
    cscale = 1.0 if cscale < 1e-4 else min(cscale, 1e4)
    c = 1.0 / cscale
    return Ps * c, qs * c, As, bs, D, E, c


# ----------------------------------------------------------------------------- IPM
class Solver:
    def __init__(self, P, q, A, b, cones_list, offset):
        self.P0 = (0.5 * (P + P.T)).tocsc()
        self.q0, self.A0, self.b0 = q, A.tocsc(), b
        self.off = offset
        self.n, self.m = P.shape[0], A.shape[0]
        self.cones = Cones(cones_list, self.m)
        self.cl = cones_list
        self.data_norms()

    def data_norms(self):
        self.nq = np.max(np.abs(self.q0), initial=0.0)
        self.nb = np.max(np.abs(self.b0), initial=0.0)
        self.Ascale = np.max(np.abs(self.A0.data), initial=0.0) or 1.0
        self.Pscale = np.max(np.abs(self.P0.data), initial=0.0) or 1.0
        self.A0T = self.A0.T.tocsr()

    # --- residual metrics in original coordinates (mirror of the official check)
    def metrics_opt(self, x, s, z):
        Ax = self.A0 @ x
        Px = self.P0 @ x
        Atz = self.A0T @ z
        nrm = lambda v: float(np.max(np.abs(v), initial=0.0))
        rp = nrm(Ax + s - self.b0) / (1 + max(nrm(Ax), nrm(s), self.nb))
        rd = nrm(Px + self.q0 + Atz) / (1 + max(nrm(Px), self.nq, nrm(Atz)))
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(self.q0 @ x)
        d0 = -0.5 * xPx - float(self.b0 @ z)
        gap = abs(p0 - d0)
        sc = max(abs(p0), abs(d0))
        rg = gap / sc if sc > 0 else 0.0
        return rp, rd, min(gap, rg)

    def cert_primal(self, z):
        nz = np.max(np.abs(z), initial=0.0)
        if nz <= 0 or self.nb <= 0 or not np.all(np.isfinite(z)):
            return np.inf
        zh = z / nz
        beta = -float((self.b0 / self.nb) @ zh)
        if beta <= 1e-8:
            return np.inf
        lin = np.max(np.abs(self.A0T @ zh), initial=0.0) / self.Ascale / beta
        return lin

    def cert_dual(self, x, s):
        nx = np.max(np.abs(x), initial=0.0)
        if nx <= 0 or self.nq <= 0 or not np.all(np.isfinite(x)):
            return np.inf
        xh, sh = x / nx, s / nx
        alpha = -float((self.q0 / self.nq) @ xh)
        if alpha <= 1e-8:
            return np.inf
        quad = np.max(np.abs(self.P0 @ xh), initial=0.0) / self.Pscale / alpha
        lin = np.max(np.abs((self.A0 @ xh) + sh), initial=0.0) / self.Ascale / alpha
        return max(quad, lin)

    def run(self, dpt0=0.0, reg0=None, piv_maxdim=None):
        piv_maxdim = PIV_MAXDIM if piv_maxdim is None else piv_maxdim
        n, m = self.n, self.m
        cones = self.cones
        P, q, A, b, D, E, c = equilibrate(self.P0, self.q0, self.A0, self.b0, cones) if False else \
            equilibrate(self.P0, self.A0, self.q0, self.b0, cones)
        self.D, self.E, self.c = D, E, c
        AT = A.T.tocsr()
        dense_mode = False
        big_psd = any(len(r) > 60 for d, r in cones.psd)
        if big_psd and n <= 8000:
            dense_mode = True
        K = kktmod.KKTDense(P, A, cones) if dense_mode else kktmod.KKTSparse(P, A, cones)
        if not dense_mode:
            K.dpt = dpt0
        nu = cones.nu
        zr = cones.zero_rows

        # ---------------- initial point
        if cones.has_exp or dense_mode and False:
            x = np.zeros(n)
            s = cones.identity()
            z = cones.identity()
            if cones.has_exp:
                R = cones.exp_rows
                z[R] = EXP_CENTRAL[None, :]
                s[R] = EXP_CENTRAL[None, :]
        else:
            # H = I for nonzero cones
            st = {}
            cones.st = {"nn_H": np.ones(len(cones.nn_rows)), "nn_lam": np.ones(len(cones.nn_rows)),
                        "soc": [dict(w=None, eta=None, lam=None, H=np.broadcast_to(np.eye(d), (R.shape[0], d, d)).copy())
                                for d, R in cones.soc_groups],
                        "psd": [dict(R=np.eye(d), Rinv=np.eye(d), lam=np.ones(d)) for d, r in cones.psd]}
            Hp = cones.H_blocks()
            if not dense_mode:
                Hp += [np.eye(len(r)).ravel() for d, r in cones.psd]
            K.factor(Hp, 1e-8, 1e-8)
            x, zz = K.solve(-q, b, refine=2)
            s = -zz
            s, z = cones.shift_init(s.copy(), zz.copy())
        s[zr] = 0.0
        tau, kappa = 1.0, 1.0

        best = None
        stall = 0
        short = 0
        prev_mu = np.inf
        status = "unresolved"
        result = None
        maxit = 200
        dp_reg = float(os.environ.get("DPREG", "1e-9")) if reg0 is None else reg0
        mu = (s @ z + tau * kappa) / (nu + 1)
        self.iters = 0
        for it in range(maxit):
            self.iters = it
            if time.time() - T0 > TIME_CAP:
                break
            Px = P @ x
            rx = Px + AT @ z + q * tau
            rz = A @ x + s - b * tau
            xPx = float(x @ Px)
            rtau = kappa + float(q @ x) + float(b @ z) + xPx / tau
            mu = (float(s @ z) + tau * kappa) / (nu + 1)

            # ---------------- termination checks in original coordinates
            xo = D * x / tau
            so = s / E / tau
            zo = E * z / (c * tau)
            rp, rd, rg = self.metrics_opt(xo, so, zo)
            if os.environ.get("SOLVER_VERBOSE"):
                print(f"it {it:3d} mu {mu:.2e} tau {tau:.2e} kap {kappa:.2e} rp {rp:.2e} rd {rd:.2e} gap {rg:.2e}",
                      file=sys.stderr)
            if rp <= 5e-9 and rd <= 5e-9 and rg <= 5e-9:
                return "solved", (xo, so, zo)
            if max(rp, rd, rg) <= 1e-8 and (best is None or max(rp, rd, rg) < best[0]):
                best = (max(rp, rd, rg), (xo.copy(), so.copy(), zo.copy()))
            # infeasibility
            bz = float(b @ z)
            qx = float(q @ x)
            if bz < 0:
                zc = E * z
                if self.cert_primal(zc) <= 5e-9:
                    return "primal_infeasible", (None, None, zc)
            if qx < 0:
                xc, sc_ = D * x, s / E
                if self.cert_dual(xc, sc_) <= 5e-9:
                    return "dual_infeasible", (xc, sc_, None)

            # ---------------- scaling + factorisation
            try:
                cones.update_scaling(s, z, mu)
            except Exception:
                break
            Hp = cones.H_blocks()
            if not dense_mode:
                for i in range(len(cones.psd)):
                    Hp.append(cones.psd_H_dense(i).ravel())
            ok = False
            for attempt in range(4):
                try:
                    K.factor(Hp, dp_reg, dp_reg)
                    ok = True
                    break
                except Exception:
                    dp_reg *= 100
            if not ok:
                break

            # constant solve (raise the regularisation if the solve is inaccurate)
            x1, z1 = K.solve(-q, b)
            if dense_mode and not K.use_qr and K.last_res > 1e-6 and K.qr_applicable():
                K.use_qr = True
                try:
                    K.factor(Hp, dp_reg, dp_reg)
                except Exception:
                    break
                x1, z1 = K.solve(-q, b)
            for _rg in range(3):
                if K.last_res <= 1e-6 or dp_reg >= 1e-5:
                    break
                dp_reg = min(dp_reg * 100.0, 1e-5)
                try:
                    K.factor(Hp, dp_reg, dp_reg)
                except Exception:
                    break
                x1, z1 = K.solve(-q, b)
            if not dense_mode and K.last_res > PIV_TOL and K.dpt == 0.0 and not K.no_pivot and n + m <= piv_maxdim and (K.base_time or 0) < 0.05:
                K.dpt = 0.1
                try:
                    K.factor(Hp, dp_reg, dp_reg)
                except Exception:
                    break
                x1, z1 = K.solve(-q, b)
            if os.environ.get("SOLVER_VERBOSE"):
                print(f"   kkt res {getattr(K, 'last_res', 0):.1e} dpt {getattr(K, 'dpt', 0)} ft {getattr(K, 'last_ft', 0):.2f} fill {getattr(K, 'last_fill', 0)}", file=sys.stderr)
            xi = x / tau
            Hz1 = cones.apply_H(z1)
            dx1 = x1 - xi
            denom = float(dx1 @ (P @ dx1)) + float(z1 @ Hz1) + kappa / tau
            two_Pxi_q = 2.0 * (P @ xi) + q

            def newton(dx_, dz_, dtau_, ds_, dkap_):
                x2, z2 = K.solve(-dx_, -dz_ + ds_)
                dtau = (dtau_ - dkap_ / tau + float(two_Pxi_q @ x2) + float(b @ z2)) / denom
                Dx = x2 + dtau * x1
                Dz = z2 + dtau * z1
                if dense_mode or DS_PRIMAL:
                    Ds = -dz_ - A @ Dx + b * dtau
                else:
                    Ds = -ds_ - cones.apply_H(Dz)
                Ds[zr] = 0.0
                Dk = -(dkap_ + kappa * dtau) / tau
                return Dx, Dz, Ds, dtau, Dk

            def steplen(Ds, Dz, Dt, Dk):
                a = cones.step_length(s, Ds, z, Dz, 1.0)
                if Dt < 0:
                    a = min(a, -tau / Dt)
                if Dk < 0:
                    a = min(a, -kappa / Dk)
                return a

            # predictor
            ds_aff = s.copy()
            ds_aff[zr] = 0.0
            Dxa, Dza, Dsa, Dta, Dka = newton(rx, rz, rtau, ds_aff, tau * kappa)
            aa = steplen(Dsa, Dza, Dta, Dka)
            sigma = (1.0 - min(aa, 1.0)) ** 3
            # corrector
            ds_c = cones.ds_combined(s, z, Dsa, Dza, sigma, mu)
            ds_c[zr] = 0.0
            dk_c = tau * kappa + Dka * Dta - sigma * mu
            Dx, Dz, Ds, Dt, Dk = newton((1 - sigma) * rx, (1 - sigma) * rz, (1 - sigma) * rtau, ds_c, dk_c)
            a = steplen(Ds, Dz, Dt, Dk)
            a = min(1.0, 0.99 * a)
            if not np.isfinite(a) or a <= 0:
                break
            if CENT_GAMMA > 0 and (len(cones.nn_rows) or cones.soc_groups):
                cm_cur = min(cones.centrality_min(s, z), tau * kappa)
                for _bt in range(40):
                    s_n, z_n = s + a * Ds, z + a * Dz
                    t_n, k_n = tau + a * Dt, kappa + a * Dk
                    mu_n = (float(s_n @ z_n) + t_n * k_n) / (nu + 1)
                    cm = min(cones.centrality_min(s_n, z_n), t_n * k_n)
                    if cm >= min(CENT_GAMMA * mu_n, 0.5 * cm_cur):
                        break
                    a *= 0.8
            x = x + a * Dx
            s = s + a * Ds
            z = z + a * Dz
            tau = tau + a * Dt
            kappa = kappa + a * Dk
            s[zr] = 0.0
            # normalise the homogeneous scale to avoid over/underflow
            sc_ = max(tau, kappa)
            if sc_ > 1e10 or sc_ < 1e-10:
                pass
            if a < 1e-7:
                stall += 1
            else:
                stall = 0
            short = short + 1 if a < 0.2 else 0
            if short >= 3 and not dense_mode and K.dpt == 0.0 and not K.no_pivot and n + m <= piv_maxdim and (K.base_time or 0) < 0.05:
                K.dpt = 0.1
                short = 0
            if stall >= 5:
                break
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.isfinite(tau)):
                break
        if best is not None:
            return "solved", best[1]
        return "unresolved", None


def main():
    import warnings
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        P, q, A, b, cones, off = load_problem(args.input)
        S = Solver(P, q, A, b, cones, off)
        status, vecs = S.run()
        retries = [dict(piv_maxdim=RETRY_PIV_MAXDIM),
                   dict(dpt0=0.1 if S.n + S.m <= RETRY_PIV_MAXDIM else 0.0, reg0=1e-7)]
        for kw in retries:
            if status != "unresolved" or time.time() - T0 >= 0.3 * TIME_CAP or os.environ.get("SOLVER_NO_RETRY"):
                break
            try:
                st2, v2 = S.run(**kw)
                if st2 != "unresolved":
                    status, vecs = st2, v2
            except Exception:
                pass
        if status == "solved":
            x, s, z = vecs
            write_result(args.output, "solved", x, s, z)
        elif status == "primal_infeasible":
            write_result(args.output, "primal_infeasible", z=vecs[2])
        elif status == "dual_infeasible":
            write_result(args.output, "dual_infeasible", x=vecs[0], s=vecs[1])
        else:
            write_result(args.output, "unresolved")
        if os.environ.get("SOLVER_VERBOSE"):
            print(f"status {status} iters {S.iters} time {time.time()-T0:.2f}", file=sys.stderr)
    except Exception as e:  # never fail without writing an output
        if os.environ.get("SOLVER_VERBOSE"):
            import traceback
            traceback.print_exc()
        try:
            write_result(args.output, "unresolved")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
