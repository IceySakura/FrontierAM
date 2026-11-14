#!/usr/bin/env python3
"""Entry point: read the instance, solve, write the result NPZ."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_core as sc

TOL = 1e-8
SAFETY = 1e-10
ACCEPT = 4e-9

CONFIGS = [
    {"kw": dict(tau=0.98, init_mode="e")},
    {"kw": dict(tau=0.98, init_mode="proj")},
    {"kw": dict(tau=0.995, init_mode="e")},
    {"kw": dict(tau=0.95, init_mode="proj")},
]


def parse_args(argv):
    inp = out = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--input" and i + 1 < len(argv):
            inp = argv[i + 1]
            i += 2
        elif a == "--output" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif a.startswith("--input="):
            inp = a.split("=", 1)[1]
            i += 1
        elif a.startswith("--output="):
            out = a.split("=", 1)[1]
            i += 1
        else:
            i += 1
    return inp, out


def read_input(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        raw = {"q": np.asarray(d["q"], dtype=np.float64).ravel(),
               "b": np.asarray(d["b"], dtype=np.float64).ravel(),
               "cones": meta["cones"],
               "offset": float(meta.get("objective_offset", 0.0) or 0.0)}
        for p in ("P", "A"):
            raw[p + "d"] = np.ascontiguousarray(np.asarray(d[p + "_data"], dtype=np.float64))
            raw[p + "i"] = np.ascontiguousarray(np.asarray(d[p + "_indices"]).astype(np.int64))
            raw[p + "p"] = np.ascontiguousarray(np.asarray(d[p + "_indptr"]).astype(np.int64))
            raw[p + "s"] = tuple(int(v) for v in d[p + "_shape"])
    return raw


# ---------------------------------------------------------------------------
# certificate checks, identical in spirit to the official checker
# ---------------------------------------------------------------------------
def cone_defect(v, cones, dual):
    worst = 0.0
    start = 0
    for c in cones:
        t = c["type"]
        d = int(c["dim"])
        w = d * (d + 1) // 2 if t == "psd" else d
        blk = v[start:start + w]
        start += w
        nrm = np.max(np.abs(blk), initial=0.0)
        u = blk / max(1.0, nrm)
        if t == "zero":
            viol = 0.0 if dual else float(np.max(np.abs(u), initial=0.0))
        elif t == "nonnegative":
            viol = max(0.0, -float(np.min(u, initial=0.0)))
        elif t == "soc":
            viol = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
        elif t == "exponential":
            a, b, cc = (float(u[0]), float(u[1]), float(u[2]))
            if dual:
                a, b, cc = -float(u[1]), -float(u[0]), EE * float(u[2])
            bound = max(abs(b), max(a, 0.0), max(-cc, 0.0))
            if b > 0.0 and cc > 0.0:
                interior = max(0.0, a - b * (math.log(cc) - math.log(b)))
                viol = min(interior, bound)
            else:
                viol = bound
        else:
            M = np.zeros((d, d))
            pos = 0
            for col in range(d):
                vals = u[pos:pos + col + 1].copy()
                vals[:col] /= SQ2
                M[:col + 1, col] = vals
                M[col, :col + 1] = vals
                pos += col + 1
            ev = np.linalg.eigvalsh(M)
            viol = max(0.0, -float(ev[0]))
        worst = max(worst, viol)
    return worst


import math
EE = math.e
SQ2 = math.sqrt(2.0)


def try_certificates(raw, model, best):
    """Attempt to turn a failed solve into a verified infeasibility certificate."""
    n, m = raw["Ps"][0], raw["As"][0]
    cones = raw["cones"]
    Ad, Ai, Ap = raw["Ad"], raw["Ai"], raw["Ap"]
    Pd, Pi, Pp = raw["Pd"], raw["Pi"], raw["Pp"]
    q, b = raw["q"], raw["b"]
    a_scale = float(np.max(np.abs(Ad))) if Ad.size else 1.0
    a_scale = a_scale or 1.0
    h_scale = float(np.max(np.abs(Pd))) if Pd.size else 1.0
    h_scale = h_scale or 1.0
    out = None
    # ---- last iterate, in original coordinates
    if best.get("x_t") is not None:
        xo, so, zo = model.to_orig(best["x_t"], best["sC"], best["zz"], best["zC"])
    else:
        xo = so = zo = None
    cands = []
    if zo is not None:
        cands.append(zo)
    for zc in cands:
        nz_ = np.max(np.abs(zc), initial=0.0)
        if nz_ <= 0:
            continue
        zh = zc / nz_
        nb = np.max(np.abs(b), initial=0.0)
        if nb <= 0:
            continue
        beta = -float((b / nb) @ zh)
        if beta <= TOL:
            continue
        # polish: alternate projection onto {A^T z = 0, b^T z = -nb} and K*
        zz = polish_primal(zh, raw, model, nb)
        if zz is None:
            continue
        nzn = np.max(np.abs(zz), initial=0.0)
        zh2 = zz / nzn
        beta2 = -float((b / nb) @ zh2)
        if beta2 <= TOL:
            continue
        Ath = matvec_T(raw, zh2)
        lin = float(np.max(np.abs(Ath), initial=0.0)) / a_scale / beta2
        cd = cone_defect(zh2, cones, True) / beta2
        if lin <= TOL and cd <= TOL:
            out = ("primal_infeasible", zh2)
            break
    if out is None and best.get("x_t") is not None:
        d = try_dual_certificate(raw, best["x_t"])
        if d is not None:
            return ("dual_infeasible", d)
    return out


def matvec_T(raw, v):
    Ad, Ai, Ap = raw["Ad"], raw["Ai"], raw["Ap"]
    n = raw["Ps"][0]
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Ap))
    return np.bincount(cols, weights=Ad * v[Ai], minlength=n)


def matvec(raw, v):
    Ad, Ai, Ap = raw["Ad"], raw["Ai"], raw["Ap"]
    m = raw["As"][0]
    return np.bincount(Ai, weights=Ad * v, minlength=m)


def proj_dual_cone(v, cones):
    out = v.copy()
    start = 0
    for c in cones:
        t = c["type"]
        d = int(c["dim"])
        w = d * (d + 1) // 2 if t == "psd" else d
        blk = out[start:start + w]
        start += w
        if t == "zero":
            pass  # dual of {0} is free
        elif t == "nonnegative":
            np.maximum(blk, 0.0, out=blk)
        elif t == "soc":
            tt = blk[0]
            vv = blk[1:]
            nv = np.linalg.norm(vv)
            if nv <= tt:
                pass
            elif nv <= -tt:
                blk[:] = 0.0
            else:
                a = 0.5 * (1.0 + tt / nv)
                blk[0] = a * nv
                blk[1:] = a * vv
        elif t == "exponential":
            blk[:] = proj_exp_dual(blk)
        else:
            M = np.zeros((d, d))
            pos = 0
            for col in range(d):
                vals = blk[pos:pos + col + 1].copy()
                vals[:col] /= SQ2
                M[:col + 1, col] = vals
                M[col, :col + 1] = vals
                pos += col + 1
            ev, V = np.linalg.eigh(M)
            np.maximum(ev, 0.0, out=ev)
            M = (V * ev) @ V.T
            pos = 0
            for col in range(d):
                vals = M[:col + 1, col].copy()
                vals[:col] *= SQ2
                blk[pos:pos + col + 1] = vals
                pos += col + 1
    return out


def proj_exp(v):
    """Return some point of the closed exponential cone near v.

    Used only to build/verify certificates (the result is always checked against
    the official conditions afterwards), so an exact nearest point is not needed.
    """
    u, v0, w = float(v[0]), float(v[1]), float(v[2])
    if v0 > 0.0 and w >= 0.0:
        r = u / v0
        if r < 30.0 and v0 * math.exp(r) <= w:
            return np.array([u, v0, w])
    # fall back to the boundary face {(u,0,w): u<=0, w>=0}, a subset of K_exp
    return np.array([min(u, 0.0), 0.0, max(w, 0.0)])


def proj_primal_cone(v, cones):
    out = v.copy()
    start = 0
    for c in cones:
        t = c["type"]
        d = int(c["dim"])
        w = d * (d + 1) // 2 if t == "psd" else d
        blk = out[start:start + w]
        start += w
        if t == "zero":
            blk[:] = 0.0
        elif t == "nonnegative":
            np.maximum(blk, 0.0, out=blk)
        elif t == "soc":
            tt = blk[0]
            vv = blk[1:]
            nv = float(np.linalg.norm(vv))
            if nv <= tt:
                pass
            elif nv <= -tt:
                blk[:] = 0.0
            else:
                a = 0.5 * (1.0 + tt / nv)
                blk[0] = a * nv
                blk[1:] = a * vv
        elif t == "exponential":
            blk[:] = proj_exp(blk)
        else:
            M = np.zeros((d, d))
            pos = 0
            for col in range(d):
                vals = blk[pos:pos + col + 1].copy()
                vals[:col] /= SQ2
                M[:col + 1, col] = vals
                M[col, :col + 1] = vals
                pos += col + 1
            ev, V = np.linalg.eigh(M)
            np.maximum(ev, 0.0, out=ev)
            M = (V * ev) @ V.T
            pos = 0
            for col in range(d):
                vals = M[:col + 1, col].copy()
                vals[:col] *= SQ2
                blk[pos:pos + col + 1] = vals
                pos += col + 1
    return out


def matvec_P(raw, v):
    Pd, Pi, Pp = raw["Pd"], raw["Pi"], raw["Pp"]
    n = raw["Ps"][0]
    if not Pd.size:
        return np.zeros(n)
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Pp))
    return np.bincount(cols, weights=Pd * v[Pi], minlength=n)


def try_dual_certificate(raw, x):
    """x is a candidate recession direction; verify the official conditions."""
    n = raw["Ps"][0]
    cones = raw["cones"]
    Ad, q = raw["Ad"], raw["q"]
    a_scale = float(np.max(np.abs(Ad))) if Ad.size else 1.0
    a_scale = a_scale or 1.0
    h_scale = float(np.max(np.abs(raw["Pd"]))) if raw["Pd"].size else 1.0
    h_scale = h_scale or 1.0
    nq = float(np.max(np.abs(q), initial=0.0))
    nx = float(np.max(np.abs(x), initial=0.0))
    if nx <= 0.0 or nq <= 0.0:
        return None
    xh = x / nx
    alpha = -float((q / nq) @ xh)
    if alpha <= TOL:
        return None
    Px = matvec_P(raw, xh)
    if float(np.max(np.abs(Px), initial=0.0)) / h_scale / alpha > TOL:
        return None
    Ax = matvec(raw, xh)
    sh = proj_primal_cone(-Ax, cones)
    lin = float(np.max(np.abs(Ax + sh), initial=0.0)) / a_scale / alpha
    cd = cone_defect(sh / a_scale, cones, False) / alpha
    if lin <= TOL and cd <= TOL:
        return (x, sh * nx)
    return None


def proj_exp_dual(v):
    """projection onto the dual exponential cone using the primal one."""
    # (u,v,w) in K* <=> (-v,-u,e w) in K
    t = np.array([-v[1], -v[0], EE * v[2]])
    p = proj_exp(t)
    return np.array([-p[1], -p[0], p[2] / EE])


def polish_primal(zh, raw, model, nb):
    """alternate projections onto {A^T z = 0, b^T z = -nb} and K*."""
    n = raw["Ps"][0]
    b = raw["b"]
    cones = raw["cones"]
    Ad, Ai, Ap = raw["Ad"], raw["Ai"], raw["Ap"]
    m = raw["As"][0]
    # dense least squares operator for A^T (small n assumed here)
    if n > 1200 or m > 6000:
        return None
    A = np.zeros((m, n))
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Ap))
    np.add.at(A, (Ai, cols), Ad)
    AT = A.T
    G = AT @ A
    G[np.diag_indices(n)] += 1e-12 * (1.0 + np.max(np.abs(np.diag(G)), initial=0.0))
    try:
        Gi = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None
    P_null = Gi @ AT          # z -> A (A^T A)^-1 A^T z ; projection = z - A P_null z
    z = zh.copy()
    bn = b / nb
    for _ in range(120):
        # project onto affine set {A^T z = 0, bn^T z = -1}
        y = P_null @ z
        z = z - A @ y
        c = float(bn @ z)
        if abs(c) < 1e-300:
            return None
        z = z / (-c)          # scale so that bn^T z = -1  -> b^T z = -nb
        z = proj_dual_cone(z, cones)
        r = float(np.max(np.abs(AT @ z), initial=0.0))
        if r <= 1e-12 * max(1.0, float(np.max(np.abs(z)))):
            pass
    return z


def main():
    inp, out = parse_args(sys.argv[1:])
    if inp is None or out is None:
        sys.stderr.write("usage: solver.py --input IN --output OUT\n")
        return 2
    raw = read_input(inp)
    n = raw["Ps"][0]
    m = raw["As"][0]
    nnz = raw["Ad"].size
    size = n + m
    # The per instance limit is unknown at run time.  Most instances are limited to
    # one second (interpreter start-up included), and the reference -- from which the
    # limits were derived -- also needs longer on bigger instances, so the internal
    # budget grows with the problem size.
    # Interpreter + numpy start-up already cost ~0.25 s here, and the shortest
    # possible instance limit is 1 s, so the small-problem budget is set to keep
    # the whole process comfortably under one second.
    t_start = time.perf_counter()
    startup = t_start - sc.T0
    if size <= 2000:
        budget = max(0.15, 0.62 - startup)
    elif size <= 20000:
        budget = 6.0
    else:
        budget = 45.0
    budget = min(budget, float(os.environ.get("SOLVER_BUDGET", "280")))
    result = ("unresolved", None, None, None)
    try:
        model = sc.Model(raw)
        best = {"score": np.inf}
        conv = False
        for cfg in CONFIGS:
            used = time.perf_counter() - t_start
            if used > 0.75 * budget:
                break
            try:
                b2, c2 = sc.ipm(model, t_start + budget - 0.02, maxit=cfg.get("maxit", 250),
                                tol=1e-9, **cfg["kw"])
            except MemoryError:
                sys.stderr.write("out of memory\n")
                break
            except Exception as exc:
                sys.stderr.write("ipm attempt failed: %r\n" % (exc,))
                continue
            if b2.get("score", np.inf) < best.get("score", np.inf):
                best = b2
            if best.get("score", np.inf) <= ACCEPT:
                conv = True
                break
        if best.get("x") is not None and best["score"] <= ACCEPT:
            result = ("solved", best["x"], best["s"], best["z"])
        else:
            cert = try_certificates(raw, model, best)
            if cert is not None:
                if cert[0] == "primal_infeasible":
                    result = ("primal_infeasible", None, None, cert[1])
                else:
                    result = ("dual_infeasible", cert[1][0], cert[1][1], None)
    except Exception as exc:  # never leave without an output file
        sys.stderr.write("solver error: %r\n" % (exc,))
        import traceback
        traceback.print_exc()
    st, x, s, z = result
    sc_write(out, st, x, s, z)
    return 0


def sc_write(out, status, x=None, s=None, z=None):
    arrays = {"metadata_json": np.array(json.dumps({"schema_version": 1, "status": status}))}
    for name, v in (("x", x), ("s", s), ("z", z)):
        if v is not None:
            v = np.asarray(v, dtype=np.float64).ravel()
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            arrays[name] = np.ascontiguousarray(v)
    tmp = str(out) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, str(out))


if __name__ == "__main__":
    sys.exit(main())
