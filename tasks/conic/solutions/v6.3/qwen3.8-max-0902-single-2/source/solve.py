#!/usr/bin/env python3
"""Thin NPZ I/O wrapper around the compiled C conic solver core."""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONE_TYPE = {"zero": 0, "nonnegative": 1, "soc": 2, "exponential": 3, "psd": 4}
STATUS = {1: "solved", 2: "primal_infeasible", 3: "dual_infeasible", 0: "unresolved"}


def _load_lib():
    for name in ("libcore.so", "core.so", "libcore.dylib"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return ctypes.CDLL(p)
    raise OSError("core library not found")


def _i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)



def _defects(v, blocks, dual):
    worst = 0.0
    for off, dim, t in blocks:
        blk = v[off:off + dim]
        nrm = float(np.max(np.abs(blk))) if dim else 0.0
        u = blk / max(1.0, nrm)
        if t == "zero":
            d = 0.0 if dual else float(np.max(np.abs(u))) if dim else 0.0
        else:
            d = float(max(0.0, -np.min(u))) if dim else 0.0
        if not np.isfinite(d):
            return np.inf
        worst = max(worst, d)
    return worst


def try_lp(meta, q, b, Acsr, n, m, blocks):
    """Exact HiGHS path for LPs with only zero/nonnegative cones."""
    if m == 0 and n == 0:
        return None
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    eqr, ubr = [], []
    for off, dim, t in blocks:
        (eqr if t == "zero" else ubr).extend(range(off, off + dim))
    eqr = np.array(eqr, dtype=int) if eqr else None
    ubr = np.array(ubr, dtype=int) if ubr else None
    try:
        res = linprog(q, A_ub=Acsr[ubr] if ubr is not None else None,
                      b_ub=b[ubr] if ubr is not None else None,
                      A_eq=Acsr[eqr] if eqr is not None else None,
                      b_eq=b[eqr] if eqr is not None else None,
                      bounds=[(None, None)] * n, method="highs")
    except Exception:
        return None
    if res.status != 0 or res.x is None:
        return None
    x = np.ascontiguousarray(np.asarray(res.x, dtype=np.float64).ravel())
    me = np.zeros(m)
    mu = np.zeros(m)
    try:
        if eqr is not None:
            me[eqr] = np.asarray(res.eqlin.marginals, dtype=np.float64)
        if ubr is not None:
            mu[ubr] = np.asarray(res.ineqlin.marginals, dtype=np.float64)
    except Exception:
        return None
    s = b - Acsr @ x
    Ax = Acsr @ x
    nAx = float(np.max(np.abs(Ax))) if m else 0.0
    ns = float(np.max(np.abs(s))) if m else 0.0
    nb = float(np.max(np.abs(b))) if m else 0.0
    nq = float(np.max(np.abs(q))) if n else 0.0
    ep = 1.0 + max(nAx, ns, nb)
    rp = float(np.max(np.abs(Ax + s - b))) if m else 0.0
    tol = 1e-8
    if not (rp / ep <= 0.5 * tol):
        return None
    if _defects(s, blocks, False) > 0.5 * tol:
        return None
    for se in (-1.0, 1.0):
        for su in (-1.0, 1.0):
            z = se * me + su * mu
            z = np.ascontiguousarray(z)
            if not np.isfinite(z).all():
                continue
            if _defects(z, blocks, True) > 0.5 * tol:
                continue
            Atz = Acsr.T @ z
            nAtz = float(np.max(np.abs(Atz))) if n else 0.0
            ed = 1.0 + max(nq, nAtz)
            rd = float(np.max(np.abs(q + Atz))) if n else 0.0
            if not (rd / ed <= 0.5 * tol):
                continue
            p0 = float(q @ x)
            d0 = float(-(b @ z))
            gp = abs(p0 - d0)
            mg = max(abs(p0), abs(d0))
            if mg == 0.0:
                gp = 0.0
            if not (gp <= tol or gp / mg <= tol):
                continue
            return x, s, z
    return None


def main(argv):
    inp = out = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--input" and i + 1 < len(argv):
            inp = argv[i + 1]; i += 2
        elif a == "--output" and i + 1 < len(argv):
            out = argv[i + 1]; i += 2
        elif a.startswith("--input="):
            inp = a.split("=", 1)[1]; i += 1
        elif a.startswith("--output="):
            out = a.split("=", 1)[1]; i += 1
        else:
            i += 1
    if inp is None or out is None:
        return 2

    lib = _load_lib()
    lib.run_solve.restype = ctypes.c_int
    lib.run_solve.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # dims
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # Pv
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # Pp
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # Pi
        ctypes.c_long,                                       # Pnnz
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # Av
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # Ap
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # Ai
        ctypes.c_long,                                       # Annz
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # q
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # b
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # btype
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # bstart
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # bdim
        ctypes.c_long,                                       # nblk
        ctypes.c_double,                                     # c0
        ctypes.c_double,                                     # budget
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # xo
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # so
        np.ctypeslib.ndpointer(np.float64, flags="C"),       # zo
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # status
        np.ctypeslib.ndpointer(ctypes.c_long, flags="C"),    # iters
    ]

    with np.load(inp, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        q = np.ascontiguousarray(d["q"], dtype=np.float64).ravel()
        b = np.ascontiguousarray(d["b"], dtype=np.float64).ravel()
        Pd = np.ascontiguousarray(d["P_data"], dtype=np.float64).ravel()
        Pi = np.ascontiguousarray(d["P_indices"], dtype=np.int64).ravel()
        Pp = np.ascontiguousarray(d["P_indptr"], dtype=np.int64).ravel()
        Ad = np.ascontiguousarray(d["A_data"], dtype=np.float64).ravel()
        Ai = np.ascontiguousarray(d["A_indices"], dtype=np.int64).ravel()
        Ap = np.ascontiguousarray(d["A_indptr"], dtype=np.int64).ravel()
    n = int(q.size)
    m = int(b.size)
    if Pp.size != n + 1:
        Pp = np.arange(n + 1, dtype=np.int64)
        Pi = np.zeros(0, dtype=np.int64)
        Pd = np.zeros(0)
    if Ap.size != n + 1:
        Ap = np.arange(n + 1, dtype=np.int64)
        Ai = np.zeros(0, dtype=np.int64)
        Ad = np.zeros(0)

    cones = meta.get("cones", []) or []
    bt, bs, bd = [], [], []
    off = 0
    for c in cones:
        t = c["type"]
        dim = int(c["dim"])
        rows = dim * (dim + 1) // 2 if t == "psd" else dim
        bt.append(CONE_TYPE[t])
        bs.append(off)
        bd.append(dim)
        off += rows
    if off != m:
        raise ValueError("cone blocks do not cover rows")
    if not bt:
        bt, bs, bd = [0], [0], [0]
    bt = np.ascontiguousarray(bt, dtype=np.int64)
    bs = np.ascontiguousarray(bs, dtype=np.int64)
    bd = np.ascontiguousarray(bd, dtype=np.int64)

    blocks = list(zip([int(v) for v in bs], [int(v) for v in bd],
                      [c["type"] for c in (cones if cones else [{"type": "zero", "dim": 0}])]))
    lpres = None
    if Pd.size == 0 and all(t in ("zero", "nonnegative") for t in [c["type"] for c in cones]):
        try:
            from scipy.sparse import csc_matrix
            Acsr = csc_matrix((Ad, Ai, Ap), shape=(m, n)).tocsr()
            lpres = try_lp(meta, q, b, Acsr, n, m, blocks)
        except Exception as _e:
            sys.stderr.write('lp path: %r\n' % (_e,))
            lpres = None
    if lpres is not None:
        xo, so, zo = (np.ascontiguousarray(v, dtype=np.float64) for v in lpres)
        np.savez(out, metadata_json=np.array(json.dumps({"schema_version": 1, "status": "solved"})),
                 x=xo[:n], s=so[:m], z=zo[:m])
        return 0

    budget = float(os.environ.get("SOLVER_BUDGET", "250"))
    dims = np.array([n, m], dtype=np.int64)
    xo = np.zeros(max(n, 1))
    so = np.zeros(max(m, 1))
    zo = np.zeros(max(m, 1))
    st = np.zeros(1, dtype=np.int64)
    it = np.zeros(1, dtype=np.int64)
    c0 = float(meta.get("objective_offset", 0.0) or 0.0)

    try:
        lib.run_solve(dims, Pd, Pp, Pi, Pd.size, Ad, Ap, Ai, Ad.size, q, b,
                      bt, bs, bd, bt.size, c0, budget, xo, so, zo, st, it)
    except Exception:
        st[0] = 0
    status = STATUS.get(int(st[0]), "unresolved")
    if status != "solved":
        # scrub non finite values
        for arr in (xo, so, zo):
            if not np.isfinite(arr).all():
                arr[:] = 0.0

    res = {"metadata_json": np.array(json.dumps({"schema_version": 1, "status": status}))}
    if status == "solved":
        res["x"] = xo[:n]
        res["s"] = so[:m]
        res["z"] = zo[:m]
    elif status == "primal_infeasible":
        res["z"] = zo[:m]
    elif status == "dual_infeasible":
        res["x"] = xo[:n]
        res["s"] = so[:m]

    tmp = out
    try:
        np.savez(tmp, **res)
    except Exception:
        np.savez(tmp, metadata_json=np.array(
            json.dumps({"schema_version": 1, "status": "unresolved"})))
    return 0


if __name__ == "__main__":
    try:
        rc = main(sys.argv[1:])
    except Exception as exc:  # never crash without writing output
        sys.stderr.write("error: %r\n" % (exc,))
        rc = 0
        try:
            out = None
            a = sys.argv[1:]
            for i in range(len(a) - 1):
                if a[i] == "--output":
                    out = a[i + 1]
            if out:
                np.savez(out, metadata_json=np.array(
                    json.dumps({"schema_version": 1, "status": "unresolved"})))
        except Exception:
            pass
    sys.exit(rc)
