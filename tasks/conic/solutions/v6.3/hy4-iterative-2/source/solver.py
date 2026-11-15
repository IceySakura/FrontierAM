#!/usr/bin/env python3
"""Conic solver entry point.  See README.md for the method.

Pipeline (each candidate result is checked against the independent acceptance
conditions before it is written):

  1. zero/nonnegative cones only: HiGHS LP/QP (simplex/IPM, exact basic
     solutions), with an active-set KKT polish to recover accurate duals.
  2. otherwise (SOC / PSD / exponential): primal-dual interior point with
     Nesterov-Todd scaling on symmetric cones and a barrier-Hessian scaling
     on exponential cones; a primal barrier path-following method is a fallback.
  3. infeasible / unbounded instances: auxiliary conic or LP problems produce
     Farkas / dual-ray certificates, which are verified before being reported.
"""
import time
_T0 = time.monotonic()
import argparse
import os
import threading
import sys

import numpy as np
np.seterr(all="ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nputil import load_problem, save_result, cone_layout
from verify import (check_solved, check_primal_infeasible,
                    check_dual_infeasible)
import ipm
import ipm_sparse
import highs
import certs


def _verified_solved(P, q, A, b, layout, res):
    if not res or res.get("status") != "solved":
        return None
    x, s, z = res.get("x"), res.get("s"), res.get("z")
    if x is None or s is None or z is None:
        return None
    ok, _ = check_solved(P, q, A, b, layout, x, s, z)
    return (x, s, z) if ok else None


def _try_solve_variants(P, q, A, b, layout, deadline, has_exp, sparse=False,
                        sink=None):
    """Return a verified (x,s,z) or None."""
    variants = []
    if sparse:
        lin = all(k in ("zero", "nonnegative") for k, _, _, _ in layout)
        if lin:
            variants.append(("diag", dict(maxit=100)))
        else:
            variants.append(("sparse", dict(maxit=150, exp_factor=0.5 if has_exp else 0.0)))
    else:
        if has_exp:
            variants.append(("pd", dict(exp_factor=0.5, maxit=200)))
            variants.append(("pd", dict(exp_nt=True, maxit=200)))
            variants.append(("pd", dict(exp_factor=0.0, maxit=200)))
        else:
            variants.append(("pd", dict(exp_factor=0.0, maxit=200)))
        variants.append(("barrier", dict(maxit=300)))
    for idx, (kind, kw) in enumerate(variants):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            sub_deadline = (deadline if idx == 0
                            else min(deadline, time.monotonic() + min(remaining, 0.30)))
            if kind == "pd":
                res = ipm.solve(P, q, A, b, layout, sub_deadline, **kw)
            elif kind == "sparse":
                res = ipm_sparse.solve(P, q, A, b, layout, sub_deadline, **kw)
            elif kind == "diag":
                res = ipm_sparse.solve_diag_sparse(P, q, A, b, layout, sub_deadline, **kw)
            else:
                res = ipm.solve_barrier(P, q, A, b, layout, sub_deadline, **kw)
        except Exception:
            continue
        if sink is not None and res:
            xx, ss, zz = res.get("x"), res.get("s"), res.get("z")
            if xx is not None and ss is not None and zz is not None:
                sink.append((xx, ss, zz))
        got = _verified_solved(P, q, A, b, layout, res)
        if got is not None:
            return got
        best = res.get("best") if res else None
        if best is not None:
            ok, _ = check_solved(P, q, A, b, layout, best[0], best[1], best[2])
            if ok:
                return best
    return None



def _certs_from_sink(P, q, A, b, layout, sink):
    """Extract verified infeasibility certificates from diverging iterates."""
    # Fixed priority keeps the reported status stable across the three runs.
    for xx, ss, zz in sink[:8]:
        try:
            okp, _ = check_primal_infeasible(P, q, A, b, layout, zz)
            if okp:
                return ("primal_infeasible", None, None, zz)
        except Exception:
            pass
    for xx, ss, zz in sink[:8]:
        try:
            okd, _ = check_dual_infeasible(P, q, A, b, layout, xx, ss)
            if okd:
                return ("dual_infeasible", xx, ss, None)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    # Leave a valid result on disk immediately: if this process is killed by a
    # short per-instance time limit, the evaluator still sees a valid NPZ.
    save_result(out, "unresolved")
    try:
        P, q, A, b, cones, offset = load_problem(args.input)
        layout = cone_layout(cones, len(b))
    except Exception:
        save_result(out, "unresolved")
        return 0
    n, m = len(q), len(b)
    size = A.nnz + P.nnz + n + m
    zero_rows = sum(w for k, d, st, w in layout if k == "zero")
    dense_ok = (m * n <= 4_000_000 and (n + zero_rows) <= 2600)
    has_exp = any(c["type"] == "exponential" for c in cones)
    linear_only = all(c["type"] in ("zero", "nonnegative") for c in cones)
    # A conservative early deadline for the ubiquitous 1 s cases, with extra
    # room for larger instances whose budgets are proportionally longer.
    if size < 30000:
        early, late, hard = 0.70, 1.20, 1.55
    elif size < 300000:
        early, late, hard = 1.10, 2.40, 2.90
    elif size < 3_000_000:
        early, late, hard = 2.20, 6.00, 7.00
    else:
        early, late, hard = 3.00, 9.00, 10.50
    qp_large = linear_only and P.nnz > 0 and n > 300
    if qp_large:
        early, late, hard = 1.20, 1.90, 2.30
    if not linear_only:
        # Non-polyhedral instances where the reference struggles carry an 18 s
        # budget; medium-size conic IPMs can use a few seconds productively.
        extra = 3.60 if size < 300000 else 8.00
        late = max(late, extra)
        hard = max(hard, late + 0.60)
    early = _T0 + early
    late = _T0 + late
    # Hard cap: a daemon watchdog thread never blocks normal exit but exits the
    # process if a single case monopolises the evaluation.  The output file
    # already contains a valid result (possibly unresolved).
    try:
        wd = threading.Timer(max(0.05, hard - (time.monotonic() - _T0)),
                             lambda: os._exit(0))
        wd.daemon = True
        wd.start()
    except Exception:
        pass

    def checks(x, s, z):
        return check_solved(P, q, A, b, layout, x, s, z)

    hint = None
    # ---------------- phase 1: HiGHS for LP / QP ----------------
    if linear_only:
        try:
            r = highs.solve_linear_cones(P, q, A, b, layout,
                                         late if qp_large else early, checks)
        except Exception:
            r = None
        if r is not None:
            if r["status"] == "solved":
                got = _verified_solved(P, q, A, b, layout, r)
                if got is not None:
                    save_result(out, "solved", *got)
                    return 0
            elif r["status"] in ("infeasible", "unbounded"):
                hint = r["status"]
    # ---------------- phase 2: primal-dual / barrier IPM ----------------
    got = None
    sink = []
    if dense_ok and hint is None and time.monotonic() < late:
        got = _try_solve_variants(P, q, A, b, layout, late, has_exp, sink=sink)
    if got is not None:
        save_result(out, "solved", *got)
        return 0
    c = _certs_from_sink(P, q, A, b, layout, sink)
    if c is not None:
        st, xx, ss, zz = c
        save_result(out, st, x=xx, s=ss, z=zz)
        return 0
    # sparse interior point for large instances (and as a fallback for QPs)
    if hint is None and time.monotonic() < late and (not dense_ok or got is None):
        if P.nnz and linear_only and n > 250:
            pass  # HiGHS already tried; sparse IPM below is the alternative
        got2 = _try_solve_variants(P, q, A, b, layout, late, has_exp,
                                   sparse=True, sink=sink)
        if got2 is not None:
            save_result(out, "solved", *got2)
            return 0
        c = _certs_from_sink(P, q, A, b, layout, sink)
        if c is not None:
            st, xx, ss, zz = c
            save_result(out, st, x=xx, s=ss, z=zz)
            return 0
    if got is not None:
        save_result(out, "solved", *got)
        return 0
    # ---------------- phase 3: infeasibility certificates ----------------
    conic_solve = None
    if dense_ok:
        def conic_solve(Pa, qa, Aa, ba, la, dl):
            r = _try_solve_variants(Pa, qa, Aa, ba, la, dl, False)
            if r is None:
                return {"status": "unresolved"}
            return {"status": "solved", "x": r[0], "s": r[1], "z": r[2]}
    cp = lambda z: check_primal_infeasible(P, q, A, b, layout, z)
    cd = lambda x, s: check_dual_infeasible(P, q, A, b, layout, x, s)
    dl = late
    order = (("primal", "dual") if hint == "infeasible" else
             ("dual", "primal") if hint == "unbounded" else
             ("primal", "dual"))
    for which in order:
        if time.monotonic() > dl:
            break
        try:
            if which == "primal":
                z = certs.find_primal(P, q, A, b, layout, dl, cp, conic_solve)
                if z is not None:
                    save_result(out, "primal_infeasible", z=z)
                    return 0
            else:
                r = certs.find_dual(P, q, A, b, layout, dl, cd, conic_solve)
                if r is not None:
                    save_result(out, "dual_infeasible", x=r[0], s=r[1])
                    return 0
        except Exception:
            pass
    save_result(out, "unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
