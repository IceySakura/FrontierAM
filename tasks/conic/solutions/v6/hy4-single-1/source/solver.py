#!/usr/bin/env python3
"""Offline conic QP solver.

Method (summary)
----------------
* Linear / quadratic objectives over zero + nonnegative cones are solved with
  the HiGHS LP/QP engine bundled with SciPy.  A basis-active-set Newton polish
  removes the ~1e-7 dual error of the HiGHS QP active-set solver.
* General cones are handled by an outer polyhedral approximation: each cone is
  replaced by supporting hyperplanes whose normals are *exact* dual-cone
  vectors.  The resulting LP/QP relaxation is resolved with warm starts, cuts
  are added where the primal iterate leaves the cone, and the simplex basis
  yields an exact dual certificate as a conic combination of the cut normals.
  A tiny primal-slack shift (well below the acceptance tolerance) is used to
  move degenerate basic solutions strictly inside the flat parts of the
  exponential/PSD cones.
* Infeasibility/unboundedness certificates are computed by auxiliary LPs.

All computations are single threaded; no files other than the output are
written, and no state is carried between invocations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import sparse

from cones import ConeInfo, SQ2, exp_dual_cut, smat, svec
from highswrap import HighsLP, INF
from verify import (residual_report, check_primal_infeasible,
                    check_dual_infeasible)

START = time.monotonic()
HARD_DEADLINE = START + 280.0
TOL = 1e-8
SAFE = 0.8e-8           # internal safety margin on residuals


def remaining():
    return HARD_DEADLINE - time.monotonic()


def log(*a):
    if os.environ.get("SOLVER_VERBOSE"):
        print("[solver]", *a, file=sys.stderr, flush=True)


def load_problem(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        q = np.asarray(d["q"], dtype=np.float64).reshape(-1)
        b = np.asarray(d["b"], dtype=np.float64).reshape(-1)
        P = sparse.csc_matrix((np.asarray(d["P_data"], dtype=np.float64),
                               np.asarray(d["P_indices"], dtype=np.int32),
                               np.asarray(d["P_indptr"], dtype=np.int32)),
                              shape=tuple(int(v) for v in d["P_shape"]))
        A = sparse.csc_matrix((np.asarray(d["A_data"], dtype=np.float64),
                               np.asarray(d["A_indices"], dtype=np.int32),
                               np.asarray(d["A_indptr"], dtype=np.int32)),
                              shape=tuple(int(v) for v in d["A_shape"]))
    for M in (P, A):
        M.sum_duplicates()
        M.eliminate_zeros()
        M.sort_indices()
    # Symmetrise P: the input stores the full matrix but may differ by 1e-12.
    if P.nnz:
        P = 0.5 * (P + P.T)
        P = sparse.csc_matrix(P)
        P.eliminate_zeros()
    cones = meta["cones"]
    ci = ConeInfo.build(cones)
    if ci.m != b.size or A.shape != (b.size, q.size) or P.shape != (q.size, q.size):
        raise ValueError("inconsistent problem dimensions")
    return dict(P=P, q=q, A=A, b=b, cones=cones, ci=ci,
                offset=float(meta.get("objective_offset", 0.0)))


def write_output(path, status, x=None, s=None, z=None, obj=None):
    meta = {"schema_version": 1, "status": status}
    arrays = {"metadata_json": np.array(json.dumps(meta, ensure_ascii=False))}
    if status == "solved":
        for name, v in (("x", x), ("s", s), ("z", z)):
            arrays[name] = np.asarray(v, dtype=np.float64).reshape(-1)
    elif status == "primal_infeasible":
        arrays["z"] = np.asarray(z, dtype=np.float64).reshape(-1)
    elif status == "dual_infeasible":
        arrays["x"] = np.asarray(x, dtype=np.float64).reshape(-1)
        arrays["s"] = np.asarray(s, dtype=np.float64).reshape(-1)
    np.savez_compressed(path, **arrays)


class Solver:
    def __init__(self, prob):
        self.__dict__.update(prob)
        self.n = self.q.size
        self.m = self.b.size
        self.Ar = self.A.tocsr()
        self.has_general = any(k not in ("zero", "nonnegative")
                               for k, _, _, _ in self.ci.blocks)
        self.zero_rows = []
        self.nonneg_rows = []
        for kind, dim, st, w in self.ci.blocks:
            if kind == "zero":
                self.zero_rows.extend(range(st, st + w))
            elif kind == "nonnegative":
                self.nonneg_rows.extend(range(st, st + w))
        self.zero_rows = np.asarray(self.zero_rows, dtype=np.int32)
        self.nonneg_rows = np.asarray(self.nonneg_rows, dtype=np.int32)

    # ------------------------------------------------------------------ verify
    def verify(self, x, s, z, safe=SAFE):
        ok, r, ratio = residual_report(self.P, self.q, self.A, self.b,
                                       self.ci, x, s, z, tol=safe)
        return ok, r, ratio

    # ------------------------------------------------------------- LP/QP path
    def solve_polyhedral(self):
        """All cones are zero / nonnegative: direct LP or QP with HiGHS."""
        n, m = self.n, self.m
        base = np.concatenate((self.zero_rows, self.nonneg_rows))
        # Reorder so that row j of the model is exactly row base[j] of A.
        M = self.Ar[base].tocsc()
        lo = np.full(m, -INF)
        hi = np.full(m, INF)
        zr = len(self.zero_rows)
        lo[:zr] = self.b[self.zero_rows]
        hi[:zr] = self.b[self.zero_rows]
        hi[zr:] = self.b[self.nonneg_rows]
        is_qp = self.P.nnz > 0
        # HiGHS 1.8's QP active-set solver forms dense reduced-Hessian factor
        #isations; on million-variable QPs that exceeds the memory budget and
        # would kill the process.  Bail out cleanly with unresolved instead.
        if is_qp and n > 150000:
            return None
        solver_kind = "choose" if is_qp else "simplex"
        # HiGHS 1.8's convex-QP active-set solver can cycle indefinitely on
        # degenerate problems; cap its time and use our own KKT active-set
        # method below in that case.
        if is_qp:
            tl = 3.0 if n <= 20000 else 200.0
        else:
            tl = 3.0 if m <= 50000 else 200.0
        tl = min(remaining(), tl)
        try:
            lp = HighsLP(self.q, np.full(n, -INF), np.full(n, INF), M, lo, hi,
                         hessian=self.P if is_qp else None,
                         solver=solver_kind, time_limit=tl)
        except Exception as e:
            log("highs model failed", e)
            return None
        res = lp.solve(time_limit=tl)
        status = res["status"]
        log("polyhedral status", status, "objective", res["objective"])
        terminal = ("Optimal", "Infeasible", "Unbounded", "UnboundedOrInfeasible")
        if status not in terminal and not is_qp and remaining() > 5:
            # Dual simplex can stall on a few degenerate infeasible LPs; the
            # interior-point solver detects the status quickly.
            try:
                lp2 = HighsLP(self.q, np.full(n, -INF), np.full(n, INF), M, lo, hi,
                              solver="ipm", time_limit=min(remaining(), 30.0))
                res2 = lp2.solve(time_limit=min(remaining(), 30.0))
                log("ipm retry", res2["status"])
                if res2["status"] in terminal:
                    res, lp, status = res2, lp2, res2["status"]
            except Exception as e:
                log("ipm retry failed", e)
        if status in ("Infeasible", "Unbounded", "UnboundedOrInfeasible"):
            if status == "Infeasible":
                return self.certificate_primal()
            if status == "Unbounded":
                return self.certificate_dual()
            c = self.certificate_primal()
            return c if c else self.certificate_dual()
        if status != "Optimal" or not res["value_valid"]:
            return self.fallback_qp() if is_qp else None
        x = res["value"]
        pi = res["row_dual"]
        z = np.zeros(m)
        z[base] = -pi
        s = self.b - self.A @ x
        ok, r, ratio = self.verify(x, s, z)
        best = ("solved", x, s, z)
        if ok:
            log("polyhedral direct verified", r)
            return best
        log("polyhedral direct failed", r)
        # QP (and occasionally LP) polish from the simplex / QP active set.
        pol = self.polish(x, z, res["basis"], base)
        if pol is not None:
            xp, sp, zp = pol
            ok2, r2, _ = self.verify(xp, sp, zp)
            log("polish", ok2, r2)
            if ok2:
                return ("solved", xp, sp, zp)
        # Last resort: keep the direct solution if it passes the official tol.
        ok3, _, _ = self.verify(x, s, z, safe=TOL)
        if ok3:
            return best
        if is_qp:
            return self.fallback_qp()
        return None

    def fallback_qp(self, max_iter=40, max_size=6000):
        """Primal active-set QP warm-started from the LP relaxation.

        Used when HiGHS's convex-QP active-set solver stalls.  The LP simplex
        solution supplies a good initial active set; each iteration solves the
        equality-constrained KKT system, then repairs the active set by
        dropping negative multipliers and adding violated rows.
        """
        if self.n > max_size or remaining() < 2:
            return None
        try:
            n, m = self.n, self.m
            base = np.concatenate((self.zero_rows, self.nonneg_rows))
            M = self.Ar[base].tocsc()
            zr = len(self.zero_rows)
            lo = np.full(m, -INF)
            hi = np.full(m, INF)
            lo[:zr] = self.b[self.zero_rows]
            hi[:zr] = self.b[self.zero_rows]
            hi[zr:] = self.b[self.nonneg_rows]
            lp = HighsLP(self.q, np.full(n, -INF), np.full(n, INF), M, lo, hi,
                         solver="simplex", time_limit=min(remaining(), 20.0))
            r = lp.solve(time_limit=min(remaining(), 20.0))
            if not r["value_valid"]:
                return None
            x0 = r["value"]
            s0 = self.b - self.A @ x0
            zero = set(int(i) for i in self.zero_rows)
            nonneg = self.nonneg_rows
            scale = 1.0 + max(float(np.max(np.abs(x0), initial=0.0)),
                              float(np.max(np.abs(s0), initial=0.0)),
                              float(np.max(np.abs(self.b), initial=0.0)),
                              float(np.max(np.abs(self.q), initial=0.0)))
            act = zero | set(int(i) for i in nonneg[s0[nonneg] < 1e-7 * scale])
            t0 = time.monotonic()
            for it in range(max_iter):
                if remaining() < 1.0 or time.monotonic() - t0 > 18.0:
                    break
                rows = np.asarray(sorted(act), dtype=np.int64)
                Aa = self.Ar[rows].tocsc()
                na = rows.size
                rhs = np.concatenate((-self.q, self.b[rows]))
                u = None
                for reg in (0.0, 1e-16 * scale, 1e-14 * scale, 1e-12 * scale):
                    K = sparse.bmat([[self.P + reg * sparse.eye(n), Aa.T],
                                     [Aa, -reg * sparse.eye(na)]], format="csc")
                    try:
                        lu = sparse.linalg.splu(K, permc_spec="MMD_AT_PLUS_A",
                                                diag_pivot_thresh=0.1,
                                                options=dict(Equil=False))
                        ut = lu.solve(rhs)
                        for _ in range(2):
                            ut += lu.solve(rhs - K @ ut)
                        if np.max(np.abs(K @ ut - rhs)) > 1e-8 * (1 + np.max(np.abs(rhs))):
                            continue
                        u = ut
                        break
                    except Exception:
                        continue
                if u is None:
                    return None
                x, y = u[:n], u[n:]
                s = self.b - self.A @ x
                z = np.zeros(m)
                z[rows] = y
                ok, _, _ = self.verify(x, s, z, safe=TOL)
                if ok:
                    log("fallback_qp verified at iter", it)
                    return ("solved", x, s, z)
                ytol = 1e-9 * max(1.0, float(np.max(np.abs(y), initial=0.0)))
                rem = set(int(rows[i]) for i in range(na)
                          if rows[i] not in zero and y[i] < -ytol)
                add = set(int(i) for i in nonneg
                          if s[i] < -1e-9 * scale and i not in act)
                if not rem and not add:
                    break
                act = (act - rem) | add
        except Exception as e:
            log("fallback_qp failed", type(e).__name__, str(e)[:120])
        return None

    def polish(self, x, z, basis, base, max_size=60000):
        """Active-set KKT Newton polish for the HiGHS LP/QP solution.

        The HiGHS QP active-set solver stops with ~1e-7 reduced-cost error,
        which is too large for problems with big |Px| or |x|.  Re-solving the
        (unregularised) KKT system on the binding rows gives machine accuracy.
        """
        try:
            n = self.n
            if n > max_size:
                return None
            zero = set(int(i) for i in self.zero_rows)
            nonneg = self.nonneg_rows
            s0 = self.b - self.A @ x
            scale = 1.0 + max(float(np.max(np.abs(x), initial=0.0)),
                              float(np.max(np.abs(s0), initial=0.0)),
                              float(np.max(np.abs(self.b), initial=0.0)),
                              float(np.max(np.abs(self.q), initial=0.0)),
                              float(np.max(np.abs(self.P.data), initial=0.0)))
            act = set(zero) | set(int(i) for i in nonneg[s0[nonneg] < 1e-7 * scale])
            t_polish = time.monotonic()
            for it in range(10):
                if time.monotonic() - t_polish > 10.0 or remaining() < 1.5:
                    return None
                rows = np.asarray(sorted(act), dtype=np.int64)
                if len(rows) + n > max_size:
                    return None
                Aa = self.Ar[rows].tocsc()
                na = Aa.shape[0]
                solved = False
                rhs = np.concatenate((-self.q, self.b[rows]))
                for reg in (0.0, 1e-16 * scale, 1e-14 * scale,
                            1e-12 * scale):
                    K = sparse.bmat([[self.P + reg * sparse.eye(n), Aa.T],
                                     [Aa, -reg * sparse.eye(na)]], format="csc")
                    try:
                        lu = sparse.linalg.splu(K, permc_spec="MMD_AT_PLUS_A",
                                                diag_pivot_thresh=0.1,
                                                options=dict(Equil=False))
                        u = lu.solve(rhs)
                        for _ in range(2):
                            u += lu.solve(rhs - K @ u)
                        rr = float(np.max(np.abs(K @ u - rhs))) / (
                            1.0 + float(np.max(np.abs(rhs))))
                        if rr > 1e-8:
                            continue
                    except Exception:
                        continue
                    solved = True
                    xn, yn = u[:n], u[n:]
                    zn = np.zeros(self.m)
                    zn[rows] = yn
                    sn = self.b - self.A @ xn
                    ok, _, ratio = self.verify(xn, sn, zn, safe=TOL)
                    if ok:
                        return xn, sn, zn
                    break
                if not solved:
                    return None
                # Repair the active set: drop negative multipliers, add violated rows.
                ytol = 1e-9 * max(1.0, float(np.max(np.abs(u[n:]), initial=0.0)))
                rem = set(int(rows[i]) for i in range(len(rows))
                          if rows[i] not in zero and u[n + i] < -ytol)
                add = set(int(i) for i in nonneg
                          if sn[i] < -1e-9 * scale and i not in act)
                if not rem and not add:
                    break
                act = (act - rem) | add
            return None
        except Exception as e:
            log("polish failed", type(e).__name__, str(e)[:100])
            return None

    # ----------------------------------------------------------- certificates
    def certificate_primal(self, cut_C=None):
        """Farkas certificate z in K* with A'z=0 and -b'z>0."""
        if remaining() < 3:
            return None
        n, m = self.n, self.m
        bscale = float(np.max(np.abs(self.b)))
        if bscale <= 0:
            return None
        cost = self.b / bscale
        try:
            if cut_C is None:
                # nonnegative dual cone: z_i >= 0 on nonneg rows
                lo = np.full(m, -1.0)
                lo[self.nonneg_rows] = 0.0
                lp = HighsLP(cost, lo, np.ones(m), self.A.T.tocsr().tocsc(),
                             np.zeros(n), np.zeros(n), solver="simplex",
                             time_limit=max(0.2, remaining() - 1))
                res = lp.solve()
                z = res["value"] if res["value_valid"] else None
            elif cut_C.shape[0] > 5000 or m * cut_C.shape[0] > 4_000_000:
                return None
            else:
                # z = C'*lam + z_zero/nonneg coordinates; lam >= 0.
                k = cut_C.shape[0]
                nv = k + m
                c = np.zeros(nv)
                clo = np.zeros(nv)
                chi = np.ones(nv)
                # rows: A' (C' lam + z) = 0
                M = sparse.hstack([self.A.T @ cut_C.T, self.A.T], format="csc")
                # set box for the direct coordinates
                clo[k:] = -1.0
                clo[k + self.nonneg_rows] = 0.0
                chi[k:] = 1.0
                cost2 = np.concatenate((cut_C @ cost, cost))
                lp = HighsLP(cost2, clo, chi, M, np.zeros(n), np.zeros(n),
                             solver="simplex", time_limit=max(0.2, remaining() - 1))
                res = lp.solve()
                if not res["value_valid"]:
                    return None
                v = res["value"]
                z = cut_C.T @ v[:k] + v[k:]
            if z is None:
                return None
            ok, ratio = check_primal_infeasible(self.P, self.q, self.A, self.b,
                                                self.ci, z, tol=TOL)
            if ok:
                log("primal certificate", ratio)
                return ("primal_infeasible", None, None, z)
        except Exception as e:
            log("primal certificate failed", type(e).__name__, str(e)[:120])
        return None

    def certificate_dual(self, cut_C=None):
        """Recession direction x, s in K with P x = 0 and q'x < 0."""
        if remaining() < 3:
            return None
        n, m = self.n, self.m
        qscale = float(np.max(np.abs(self.q)))
        if qscale <= 0:
            return None
        cost = self.q / qscale
        rows = []
        lo, hi = [], []
        if len(self.zero_rows):
            rows.append(self.Ar[self.zero_rows])
            lo.extend(np.zeros(len(self.zero_rows)))
            hi.extend(np.zeros(len(self.zero_rows)))
        if len(self.nonneg_rows):
            rows.append(self.Ar[self.nonneg_rows])
            lo.extend([-INF] * len(self.nonneg_rows))
            hi.extend(np.zeros(len(self.nonneg_rows)))
        if self.P.nnz:
            rows.append(sparse.csc_matrix(self.P))
            lo.extend(np.zeros(n))
            hi.extend(np.zeros(n))
        if cut_C is not None and cut_C.shape[0]:
            rows.append((-cut_C @ self.A).tocsr())
            lo.extend([-INF] * cut_C.shape[0])
            hi.extend(np.zeros(cut_C.shape[0]))
        try:
            M = sparse.vstack([r.tocsc() for r in rows], format="csc") if rows else sparse.csc_matrix((0, n))
            lp = HighsLP(cost, np.full(n, -1.0), np.ones(n), M,
                         np.asarray(lo), np.asarray(hi), solver="simplex",
                         time_limit=max(0.2, remaining() - 1))
            res = lp.solve()
            if not res["value_valid"]:
                return None
            x = res["value"]
            s = -self.A @ x
            ok, ratio = check_dual_infeasible(self.P, self.q, self.A, self.b,
                                              self.ci, x, s, tol=TOL)
            if ok:
                log("dual certificate", ratio)
                return ("dual_infeasible", x, s, None)
        except Exception as e:
            log("dual certificate failed", type(e).__name__, str(e)[:120])
        return None

    # --------------------------------------------------------- conic path
    def solve_conic(self, budget=None):
        return ConicCuttingPlane(self).run(budget=budget)


class ConicCuttingPlane:
    def __init__(self, sv: Solver):
        self.sv = sv
        self.A = sv.Ar
        self.b = sv.b
        self.q = sv.q
        self.ci = sv.ci
        self.m, self.n = sv.m, sv.n
        self.base = np.concatenate((sv.zero_rows, sv.nonneg_rows)).astype(np.int32)
        self.nz = len(sv.zero_rows)
        self.C = None                  # sparse (n_cuts x m), normals in K*
        self.best = None
        self.best_ratio = 1e9
        self.unbounded_repairs = 0

    # --------------------------------------------------------------- cuts
    def make_C(self, entries):
        """entries: list of (block_index, c_vector) -> sparse (k x m)."""
        if not entries:
            return sparse.csr_matrix((0, self.m), dtype=np.float64)
        rows, cols, vals = [], [], []
        for i, (bi, c) in enumerate(entries):
            c = np.asarray(c, dtype=np.float64)
            nr = np.max(np.abs(c))
            if nr == 0 or not np.isfinite(nr):
                continue
            c = c / nr
            _, _, st, w = self.ci.blocks[bi]
            nz = np.nonzero(c)[0]
            rows.extend([i] * len(nz))
            cols.extend((st + nz).tolist())
            vals.extend(c[nz].tolist())
        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(len(entries), self.m), dtype=np.float64)

    def initial_C(self):
        entries = []
        for bi, (kind, dim, st, w) in enumerate(self.ci.blocks):
            if kind == "soc":
                c = np.zeros(dim)
                c[0] = 1.0
                entries.append((bi, c))
            elif kind == "exponential":
                for c in (np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                          np.array([-1., -1., 1.]),
                          np.array([-1., -26., math.exp(25.)]),
                          np.array([-1., 9., math.exp(-10.)])):
                    entries.append((bi, c))
            elif kind == "psd":
                k = 0
                for j in range(dim):
                    c = np.zeros(w)
                    c[k + j] = 1.0
                    entries.append((bi, c))
                    k += j + 1
        return self.make_C(entries)

    def new_cuts(self, s, defects, threshold=3e-10, max_new=60000):
        entries = []
        counts = {}
        for key, g in self.ci.groups.items():
            kind, dim, w = g["kind"], g["dim"], g["width"]
            if kind in ("zero", "nonnegative"):
                continue
            d = defects[g["bis"]]
            sel = np.nonzero(d > threshold)[0]
            if sel.size == 0:
                continue
            blk = s[g["idx"]]
            scale = np.maximum(1.0, np.max(np.abs(blk), axis=1))
            u = blk / scale[:, None]
            bis = g["bis"]
            if kind == "soc":
                nv = np.linalg.norm(u[sel, 1:], axis=1)
                for k, i in enumerate(sel):
                    c = np.empty(dim)
                    c[0] = 1.0
                    if nv[k] > 0:
                        c[1:] = -u[i, 1:] / nv[k]
                    else:
                        c[1:] = 0.0
                    entries.append((int(bis[i]), c))
            elif kind == "exponential":
                for i in sel:
                    for c in exp_dual_cut(u[i]):
                        entries.append((int(bis[i]), c))
            elif kind == "psd":
                for i in sel:
                    M = smat(u[i], dim)
                    ev, V = np.linalg.eigh(M)
                    for k in range(len(ev)):
                        if ev[k] < -1e-13:
                            v = V[:, k]
                            entries.append((int(bis[i]), svec(np.outer(v, v), dim)))
            counts[kind] = counts.get(kind, 0) + len(sel)
            if len(entries) >= max_new:
                break
        return self.make_C(entries[:max_new]), counts

    # ------------------------------------------------------------- model
    def build_model(self, C):
        base_M = self.A[self.base]
        cut_M = (C @ self.A).tocsr() if C.shape[0] else sparse.csr_matrix((0, self.n))
        M = sparse.vstack([base_M, cut_M], format="csc")
        nb = len(self.base)
        lo = np.full(nb + C.shape[0], -INF)
        hi = np.full(nb + C.shape[0], INF)
        lo[:self.nz] = self.b[self.sv.zero_rows]
        hi[:self.nz] = self.b[self.sv.zero_rows]
        hi[self.nz:nb] = self.b[self.sv.nonneg_rows]
        if C.shape[0]:
            cut_rhs = np.asarray(C @ self.b).reshape(-1)
            hi[nb:] = cut_rhs
        return M, lo, hi

    def perturb(self, s, scale):
        """Tiny interior shift: fixes flat exp/PSD/SOC boundary degeneracies."""
        delta = 2e-9 * scale
        sp = s.copy()
        for kind, dim, st, w in self.ci.blocks:
            if kind == "exponential":
                sp[st + 2] += delta
            elif kind == "soc":
                sp[st] += delta
            elif kind == "psd":
                k = 0
                for j in range(dim):
                    sp[st + k + j] += delta
                    k += j + 1
        return sp

    def z_from_pi(self, pi, C):
        nb = len(self.base)
        lam = np.maximum(0.0, -pi[nb:]) if len(pi) > nb else np.zeros(0)
        z = np.asarray(C.T @ lam).reshape(-1) if C.shape[0] else np.zeros(self.m)
        z[self.base] = -pi[:nb]
        return z

    def run(self, budget=None):
        sv = self.sv
        if budget is None:
            budget = 230.0
        local_deadline = min(HARD_DEADLINE - 2.0, time.monotonic() + budget)
        C = self.initial_C()
        M, lo, hi = self.build_model(C)
        solver_kind = "simplex" if sv.P.nnz == 0 else "choose"
        try:
            lp = HighsLP(sv.q, np.full(self.n, -INF), np.full(self.n, INF),
                         M, lo, hi, hessian=sv.P if sv.P.nnz else None,
                         solver=solver_kind, time_limit=max(0.2, remaining() - 2),
                         small_matrix_value=1e-12 if sv.has_general else None,
                         dual_tol=1e-9)
        except Exception as e:
            log("conic model failed", e)
            return None
        self.C = C
        max_iter = 80
        for it in range(max_iter):
            if remaining() < 2.0 or time.monotonic() > local_deadline:
                break
            res = lp.solve(time_limit=max(0.2, min(remaining() - 1.5,
                                                  local_deadline - time.monotonic() + 1.0)))
            status = res["status"]
            log("cut iter", it, status, "cuts", C.shape[0], "t", round(remaining(), 1))
            if status in ("Infeasible", "Unbounded", "UnboundedOrInfeasible"):
                if status == "Infeasible":
                    c = sv.certificate_primal(cut_C=C)
                    return c if c else self.best_result()
                if status == "Unbounded" and self.unbounded_repairs < 2:
                    self.unbounded_repairs += 1
                    C2 = self.repair_unbounded(C)
                    if C2 is not None and C2.shape[0] > C.shape[0]:
                        new_rows = (C2[C.shape[0]:] @ self.A).tocsr()
                        lo2 = np.full(C2.shape[0] - C.shape[0], -INF)
                        hi2 = np.asarray(C2[C.shape[0]:] @ self.b).reshape(-1)
                        lp.add_rows(new_rows, lo2, hi2)
                        C = C2
                        self.C = C
                        continue
                if status == "Unbounded":
                    c = sv.certificate_dual(cut_C=C)
                    return c if c else self.best_result()
                c = sv.certificate_primal(cut_C=C)
                return c if c else self.best_result()
            if not res["value_valid"]:
                break
            x = res["value"]
            pi = res["row_dual"]
            Ax = sv.A @ x
            s0 = sv.b - Ax
            scale = 1.0 + max(float(np.max(np.abs(Ax), initial=0.0)),
                              float(np.max(np.abs(s0), initial=0.0)),
                              float(np.max(np.abs(sv.b), initial=0.0)))
            s = self.perturb(s0, scale)
            z = self.z_from_pi(pi, C)
            ok, r, ratio = sv.verify(x, s, z)
            if ratio < self.best_ratio:
                self.best_ratio = ratio
                self.best = ("solved", x, s, z)
            if ok:
                log("conic verified at iter", it, r)
                return ("solved", x, s, z)
            # generate cuts
            worst, defects = sv.ci.defects(s, dual=False)
            Cnew, counts = self.new_cuts(s, defects)
            if Cnew.shape[0] == 0:
                log("no new cuts", worst, r)
                # Maybe the duals are inaccurate; try a polish-free retry.
                if worst <= 1e-7:
                    ok2, r2, _ = sv.verify(x, s, z, safe=TOL)
                    if ok2:
                        return ("solved", x, s, z)
                break
            new_rows = (Cnew @ self.A).tocsr()
            lo2 = np.full(Cnew.shape[0], -INF)
            hi2 = np.asarray(Cnew @ self.b).reshape(-1)
            lp.add_rows(new_rows, lo2, hi2)
            C = sparse.vstack([C, Cnew], format="csr")
            self.C = C
            if C.shape[0] > 1500000:
                break
        return self.best_result()

    def repair_unbounded(self, C):
        """Add a small fixed grid of cuts to bound coarse relaxations."""
        entries = []
        for bi, (kind, dim, st, w) in enumerate(self.ci.blocks):
            if kind == "exponential":
                for t in (-20., -15., -5., 5., 15., 25.):
                    entries.append((bi, np.array([-1., t - 1., math.exp(-t)])))
            elif kind == "soc":
                for j in range(1, dim):
                    for sg in (1., -1.):
                        c = np.zeros(dim)
                        c[0] = 1.0
                        c[j] = sg
                        entries.append((bi, c))
            elif kind == "psd":
                for i in range(dim):
                    for j in range(i + 1, dim):
                        for sg in (1., -1.):
                            v = np.zeros(dim)
                            v[i] = 1.0
                            v[j] = sg
                            entries.append((bi, svec(np.outer(v, v), dim)))
        C2 = self.make_C(entries)
        return sparse.vstack([C, C2], format="csr") if C2.shape[0] else None

    def best_result(self):
        # Only return a verified candidate.
        if self.best is not None:
            status, x, s, z = self.best
            ok, _, _ = self.sv.verify(x, s, z, safe=TOL)
            if ok:
                return self.best
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        prob = load_problem(args.input)
        sv = Solver(prob)
        result = None
        try:
            if not sv.has_general:
                result = sv.solve_polyhedral()
            else:
                kinds = set(k for k, _, _, _ in sv.ci.blocks)
                psd_only = "psd" in kinds and not ({"soc", "exponential"} & kinds)
                budget = 18.0 if psd_only else min(remaining() - 3.0, 230.0)
                result = sv.solve_conic(budget=budget)
        except Exception as e:
            log("solver path failed", type(e).__name__, str(e)[:200])
        if result is None:
            write_output(args.output, "unresolved")
            return 0
        status, x, s, z = result
        # Sanitise: never emit missing, wrong-sized or non-finite vectors.
        need = {"solved": (x, s, z), "primal_infeasible": (z,),
                "dual_infeasible": (x, s)}.get(status, ())
        for v in need:
            if v is None:
                write_output(args.output, "unresolved")
                return 0
        for v in (x, s, z):
            if v is not None:
                va = np.asarray(v, dtype=np.float64).reshape(-1)
                want = sv.n if v is x else sv.m
                if va.size != want or not np.isfinite(va).all():
                    write_output(args.output, "unresolved")
                    return 0
        write_output(args.output, status, x=x, s=s, z=z)
        return 0
    except Exception as e:
        log("fatal", type(e).__name__, str(e)[:300])
        try:
            write_output(args.output, "unresolved")
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
