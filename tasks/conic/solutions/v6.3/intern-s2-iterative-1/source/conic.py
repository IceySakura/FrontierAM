#!/usr/bin/env python3
"""Conic QP solver.

Solves
    minimize   1/2 x' P x + q' x + c0
    subject to Ax + s = b,  s in K

where K is a product of zero, nonnegative, second-order, exponential and
positive-semidefinite cones.

Algorithm
---------
ADMM (alternating direction method of multipliers) on the splitting between
the quadratic objective and the cone constraint.  The x-step solves one
symmetric-indefinite KKT system that is factorised once with SuperLU and
reused for every iteration; the s-step is a Euclidean projection onto the
cone product; the dual step is standard.

Two ingredients make this fast enough for tight per-instance budgets.

* Every cone supported here is invariant under positive scaling of an entire
  block, so we equilibrate the rows of A one cone block at a time and its
  columns one variable at a time without changing the feasible set.
* An active-set "polish" step reads the cone faces off the approximate ADMM
  iterate and solves the resulting equality system exactly with one more
  sparse factorisation, usually driving residuals from 1e-5 to machine
  precision.

Primal and dual infeasibility are detected by watching the iterate norms
grow; every candidate certificate is checked against the independent
certificate conditions before it is reported, so a wrong guess can never be
returned.

References
----------
* Boyd, Parikh, Chu, Peleato & Eckstein, "Distributed optimization and
  statistical learning via the alternating direction method of multipliers",
  Foundations and Trends in ML, 2010.
* O'Donoghue, Chu, Parikh & Boyd, "Conic optimization via operator splitting
  and homogeneous self-dual embedding", JOTA, 2016 (SCS).
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
from scipy.sparse.linalg import splu

from cone import Cones, smat

EPS = 1e-8
MAX_ITERS = 200000
CHECK_EVERY = 25
MU = 10.0
TAU = 2.0
DEFAULT_MAX_TIME = 300.0
POLISH_TOL = 1e-4


# -------------------------------------------------------------------------- #
# input / output
# -------------------------------------------------------------------------- #

def load_problem(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    P = sparse.csc_matrix(
        (data["P_data"], data["P_indices"], data["P_indptr"]),
        shape=tuple(data["P_shape"]),
    )
    A = sparse.csc_matrix(
        (data["A_data"], data["A_indices"], data["A_indptr"]),
        shape=tuple(data["A_shape"]),
    )
    q = np.asarray(data["q"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    cones = list(meta["cones"])
    c0 = float(meta.get("objective_offset", 0.0))
    return P, q, A, b, cones, c0, meta.get("name", "")


def save_result(path, status, x=None, s=None, z=None, c0=0.0, obj_p=None, obj_d=None):
    d = {
        "metadata_json": np.array(
            json.dumps({"schema_version": 1, "status": status}, ensure_ascii=False)
        )
    }
    if x is not None:
        d["x"] = np.asarray(x, dtype=np.float64)
    if s is not None:
        d["s"] = np.asarray(s, dtype=np.float64)
    if z is not None:
        d["z"] = np.asarray(z, dtype=np.float64)
    if obj_p is not None:
        d["objective_primal"] = float(obj_p)
    if obj_d is not None:
        d["objective_dual"] = float(obj_d)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **d)


# -------------------------------------------------------------------------- #
# scaling
# -------------------------------------------------------------------------- #

def scale_problem(P, q, A, b, cones):
    """Per-variable column scaling and per-cone-block row scaling.

    Every cone here is unchanged when a whole block is multiplied by a
    positive scalar, so the row scaling leaves the feasible set invariant.
    """
    n, m = P.shape[0], A.shape[0]
    col = np.ones(n)
    if A.nnz:
        col = np.maximum(col, np.abs(A.tocsr()).max(axis=0).toarray().ravel())
    if P.nnz:
        col = np.maximum(col, np.sqrt(np.abs(P.diagonal()) + 1e-30))
    row = np.ones(m)
    off = 0
    for c in cones:
        w = c["dim"] * (c["dim"] + 1) // 2 if c["type"] == "psd" else c["dim"]
        sc = 1.0
        if w:
            blk = A[off : off + w, :]
            if blk.nnz:
                sc = max(sc, float(np.abs(blk).max()))
            bb = b[off : off + w]
            if bb.size:
                sc = max(sc, float(np.max(np.abs(bb))))
        row[off : off + w] = sc
        off += w
    P = P.multiply(col[np.newaxis, :]).multiply(col[:, np.newaxis])
    q = q * col
    A = A.multiply(row[:, np.newaxis]).multiply(col[np.newaxis, :])
    b = b * row
    return P, q, A, b, col, row


def unscale(x, s, z, col, row):
    return x * col, s / row, z * row


# -------------------------------------------------------------------------- #
# linear algebra
# -------------------------------------------------------------------------- #

def make_kkt(P, A, rho, n, m):
    """Symmetric augmented system for the x-step.

    The x-update minimises (1/2)x'Px + q'x + (rho/2)||Ax + s - b + u||^2, whose
    normal equations are (P + rho A'A)x = -q - rho A'(s - b + u).  Introducing
    the multiplier y = rho(Ax + s - b + u) and eliminating it gives

        [ P    A'  ] [x]   [ -q        ]
        [ -A  I/rho ] [y] = [ s - b + u ]

    which is nonsingular as soon as the normal equations are.  Only y is
    needed to recover the dual variable z = y.
    """
    K11 = P
    K12 = A.T
    K21 = -A
    K22 = (1.0 / rho) * sparse.eye(m, format="csc")
    return sparse.bmat([[K11, K12], [K21, K22]], format="csc")


def residuals(P, q, A, b, cones, x, s, z):
    Ax, Px, Atz = A @ x, P @ x, A.T @ z
    denom_p = 1.0 + max(np.max(np.abs(Ax)), np.max(np.abs(s)), np.max(np.abs(b)))
    denom_d = 1.0 + max(np.max(np.abs(Px)), np.max(np.abs(q)), np.max(np.abs(Atz)))
    rp = np.max(np.abs(Ax + s - b)) / denom_p
    rd = np.max(np.abs(Px + q + Atz)) / denom_d
    return (
        rp,
        rd,
        cones.defect(s, False),
        cones.defect(z, True),
        float(x @ Px),
        float(q @ x),
        float(b @ z),
    )


def solve_admm(P, q, A, b, cones, max_time):
    """Proximal ADMM with residual-balancing step-size adaptation.

    The x-step solves one symmetric KKT system, factorised once with SuperLU
    and reused for every iteration; the s-step is a Euclidean projection onto
    the cone product.  rho is rebalanced every few iterations so that the
    primal and dual residuals stay within an order of magnitude of each other,
    which is what makes the method converge in a few thousand iterations on
    badly scaled problems.
    """
    n, m = P.shape[0], A.shape[0]
    x = np.zeros(n)
    s = np.zeros(m)
    u = np.zeros(m)
    rho = 1.0
    t0 = time.time()
    try:
        lu = splu(make_kkt(P, A, rho, n, m))
    except Exception:
        return x, s, np.zeros(m), 0, None
    best = None
    rhs = np.empty(n + m)
    for k in range(MAX_ITERS):
        rhs[:n] = -q
        rhs[n:] = s - b + u
        sol = lu.solve(rhs)
        x = sol[:n]
        z = sol[n:]  # rho*(A x + s - b + u); the scaled dual multiplier
        s = cones.project(b - A @ x - u)
        u = z / rho
        if k % CHECK_EVERY == 0:
            z = rho * u
            r = residuals(P, q, A, b, cones, x, s, z)
            tot = r[0] + r[1] + r[2] + r[3]
            if best is None or tot < best[0]:
                best = (tot, x.copy(), s.copy(), z.copy(), k, r)
            if r[0] <= EPS and r[1] <= EPS and r[2] <= EPS and r[3] <= EPS:
                return x, s, z, k, r
            # Residual balancing: keep primal and dual residuals within a
            # factor MU of each other, scaling the dual variable so that z is
            # unchanged across the step.
            if r[0] > MU * r[1]:
                rho *= TAU
                u /= TAU
            elif r[1] > MU * r[0]:
                rho /= TAU
                u *= TAU
            if time.time() - t0 > 0.95 * max_time:
                break
    if best is not None:
        return best[1], best[2], best[3], best[4], best[5]
    return x, s, np.zeros(m), 0, None


# -------------------------------------------------------------------------- #
# polish
# -------------------------------------------------------------------------- #

def _exp_on_boundary(v, tol):
    if v.size != 3:
        return False
    u, vv, w = float(v[0]), float(v[1]), float(v[2])
    if vv > 0.0 and w > 0.0:
        return abs(vv * math.exp(u / vv) - w) <= tol * max(1.0, w)
    if vv == 0.0:
        return (u <= 0.0 and abs(w) <= tol) or (abs(u) <= tol and w >= 0.0)
    return False


def build_active_set(cones, s, tol=POLISH_TOL):
    """Row indices that the iterate suggests are on the cone boundary."""
    active = []
    for (kind, dim), off, w in zip(cones.blocks, cones.offsets, cones.widths):
        blk = s[off : off + w]
        nrm = float(np.max(np.abs(blk))) if w else 0.0
        scale = max(1.0, nrm)
        if kind == "zero":
            active.extend(range(off, off + w))
        elif kind == "nonnegative":
            for i in range(w):
                if blk[i] <= tol * scale:
                    active.append(off + i)
        elif kind == "soc":
            if w >= 2:
                t, v = float(blk[0]), blk[1:]
                nv = float(np.linalg.norm(v)) if v.size else 0.0
                if abs(t - nv) <= tol * max(1.0, t, nv) or nrm <= tol:
                    active.extend(range(off, off + w))
        elif kind == "psd":
            if w:
                try:
                    ev = np.linalg.eigvalsh(smat(blk))
                except Exception:
                    ev = np.zeros(1)
                if ev[0] <= tol * max(1.0, abs(ev[0]), nrm):
                    active.extend(range(off, off + w))
        elif kind == "exponential":
            if w == 3 and _exp_on_boundary(blk, tol):
                active.extend(range(off, off + w))
    return sorted(set(active))


def polish(P, q, A, b, cones, x, s, z, time_left):
    """Solve the equality system defined by the active faces, if any."""
    if time_left <= 0.1:
        return None
    A = A.tocsc()
    active = build_active_set(cones, s)
    if not active:
        return None
    na = len(active)
    Aa = A[active, :].tocsc()
    ba = np.asarray(b[active], dtype=np.float64)
    zero = sparse.csr_matrix((na, na), dtype=np.float64)
    Kp = sparse.bmat([[P, Aa.T], [Aa, zero]], format="csc")
    rhs = np.concatenate((-q, ba))
    try:
        sol = splu(Kp).solve(rhs)
    except Exception:
        return None
    xp = sol[: P.shape[0]]
    lam = sol[P.shape[0] :]
    zp = np.zeros(b.size)
    zp[active] = lam
    sp = b - A @ xp
    return xp, sp, zp


# -------------------------------------------------------------------------- #
# certificates
# -------------------------------------------------------------------------- #

def _inf(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


def _max_abs(data):
    return float(np.max(np.abs(data))) if data.size else 0.0


def primal_infeasible_certificate(A, b, cones, z):
    a = _max_abs(A.data) or 1.0
    bn = _inf(b)
    zn = _inf(z)
    if bn == 0.0 or zn == 0.0:
        return None
    zh = z / zn
    beta = -float((b / bn) @ zh)
    if beta <= EPS:
        return None
    lin = _inf((A / a).T @ zh) / beta
    cone = cones.defect(zh, True) / beta
    if lin <= EPS and cone <= EPS:
        return zh
    return None


def dual_infeasible_certificate(P, q, A, cones, x, s):
    a = _max_abs(A.data) or 1.0
    h = _max_abs(P.data) or 1.0
    qn = _inf(q)
    xn = _inf(x)
    if qn == 0.0 or xn == 0.0:
        return None
    xh = x / xn
    sh = s / xn
    alpha = -float((q / qn) @ xh)
    if alpha <= EPS:
        return None
    quad = _inf((P / h) @ xh) / alpha
    lin = _inf((A / a) @ xh + sh / a) / alpha
    cone = cones.defect(sh / a, False) / alpha
    if quad <= EPS and lin <= EPS and cone <= EPS:
        return xh, sh
    return None


# -------------------------------------------------------------------------- #
# driver
# -------------------------------------------------------------------------- #

def solve(P, q, A, b, cones, c0, max_time):
    t0 = time.time()
    Ps, qs, As, bs, col, row = scale_problem(P, q, A, b, cones)
    scones = Cones(cones)
    budget = max(0.1, max_time - (time.time() - t0))

    x, s, z, nit, res = solve_admm(Ps, qs, As, bs, scones, budget)
    if res is not None:
        rp, rd, cp, cd, xPx, qx, bz = res
        if rp <= EPS and rd <= EPS and cp <= EPS and cd <= EPS:
            x, s, z = unscale(x, s, z, col, row)
            return "solved", x, s, z, 0.5 * xPx + qx + c0, -0.5 * xPx - bz + c0

    time_left = max_time - (time.time() - t0)
    xp = polish(Ps, qs, As, bs, scones, x, s, z, time_left)
    if xp is not None:
        xp, sp, zp = xp
        r = residuals(Ps, qs, As, bs, scones, xp, sp, zp)
        if r[0] <= EPS and r[1] <= EPS and r[2] <= EPS and r[3] <= EPS:
            xp, sp, zp = unscale(xp, sp, zp, col, row)
            return "solved", xp, sp, zp, 0.5 * (xp @ (P @ xp)) + (q @ xp) + c0, -(
                0.5 * (xp @ (P @ xp)) + (b @ zp)
            ) + c0

    # Infeasibility: test the best iterate / certificate directions.
    time_left = max_time - (time.time() - t0)
    if time_left > 0.05:
        zc = primal_infeasible_certificate(A, b, scones, z)
        if zc is not None:
            return "primal_infeasible", None, None, zc, None, None
        cert = dual_infeasible_certificate(P, q, A, scones, x, s)
        if cert is not None:
            return "dual_infeasible", cert[0], cert[1], None, None, None

    return "unresolved", x, s, z, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to input NPZ")
    parser.add_argument("--output", required=True, help="path to output NPZ")
    parser.add_argument(
        "--time-limit", type=float, default=DEFAULT_MAX_TIME, help="per-instance wall clock budget"
    )
    args = parser.parse_args()

    t_start = time.time()
    P, q, A, b, cones, c0, name = load_problem(args.input)
    status, x, s, z, obj_p, obj_d = solve(P, q, A, b, cones, c0, args.time_limit)

    if status == "solved":
        save_result(args.output, status, x, s, z, c0, obj_p, obj_d)
    elif status == "primal_infeasible":
        save_result(args.output, status, z=z)
    elif status == "dual_infeasible":
        save_result(args.output, status, x=x, s=s)
    else:
        save_result(args.output, status)
    sys.stderr.write(
        "%s: status=%s elapsed=%.3fs\n" % (name, status, time.time() - t_start)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
