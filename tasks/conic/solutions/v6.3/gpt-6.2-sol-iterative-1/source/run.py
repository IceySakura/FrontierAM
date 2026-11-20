"""Offline driver for the sparse conic solver."""

import sys
import time

START = time.monotonic()

import argparse
import ctypes
import glob
import importlib.util
import json
import os
import sysconfig

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = bool(os.environ.get("CONIC_TRACE"))
KINDS = {"zero": 0, "nonnegative": 1, "soc": 2, "psd": 3, "exponential": 4}
STATUSES = ("unresolved", "solved", "primal_infeasible", "dual_infeasible")
LIB = ctypes.CDLL(os.path.join(HERE, "libconic.so"))
PINT = ctypes.c_void_p
CINT = ctypes.c_int
COMMON = [CINT, CINT] + [PINT] * 8 + [CINT, PINT, PINT]
LIB.conic_solve.argtypes = COMMON + [ctypes.c_double, CINT] + [PINT] * 3
LIB.conic_solve.restype = CINT
LIB.conic_check.argtypes = COMMON + [CINT] + [PINT] * 3
LIB.conic_check.restype = CINT
LIB.conic_polish.argtypes = COMMON + [ctypes.c_double] + [PINT] * 3
LIB.conic_polish.restype = CINT


def csc(data, prefix, shape):
    p = np.ascontiguousarray(data[prefix + "_indptr"], dtype=np.int32)
    i = np.ascontiguousarray(data[prefix + "_indices"], dtype=np.int32)
    v = np.ascontiguousarray(data[prefix + "_data"], dtype=np.float64)
    canonical = True
    if i.size > 1:
        bad = i[:-1] >= i[1:]
        if bad.any():
            ends = p[1:-1]
            ends = ends[(ends > 0) & (ends < i.size)]
            bad[ends - 1] = False
            canonical = not bad.any()
    if not canonical or (v.size and (v == 0).any()):
        from scipy.sparse import csc_matrix

        a = csc_matrix((v, i, p), shape=shape)
        a.sum_duplicates()
        a.eliminate_zeros()
        p, i, v = a.indptr, a.indices, a.data
    return p, i, v


class Model:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.q = np.ascontiguousarray(d["q"], dtype=np.float64)
            self.b = np.ascontiguousarray(d["b"], dtype=np.float64)
            self.n, self.m = self.q.size, self.b.size
            self.p = csc(d, "P", (self.n, self.n))
            self.a = csc(d, "A", (self.m, self.n))
            cones = json.loads(str(d["metadata_json"]))["cones"]
        self.kinds = np.array([KINDS[c["type"]] for c in cones], dtype=np.int32)
        self.dims = np.array([c["dim"] for c in cones], dtype=np.int32)
        self.linear = not np.any(self.kinds > 1)
        self.results = (np.empty(self.n), np.empty(self.m), np.empty(self.m))
        self.base = (self.n, self.m, *(v.ctypes.data for v in (*self.p, *self.a, self.q, self.b)),
                     len(cones), self.kinds.ctypes.data, self.dims.ctypes.data)
        self.out = tuple(v.ctypes.data for v in self.results)

    def native(self, budget, mode=0):
        return LIB.conic_solve(*self.base, float(budget), mode, *self.out)

    def check(self, status):
        return LIB.conic_check(*self.base, status, *self.out)

    def polish(self, budget):
        return LIB.conic_polish(*self.base, float(budget), *self.out)


def highs_module():
    if hasattr(highs_module, "cached"):
        return highs_module.cached
    # Loading only the bundled extension avoids importing scipy.optimize in a timed process.
    directory = os.path.join(sysconfig.get_path("platlib"), "scipy", "optimize", "_highspy")
    path, = glob.glob(os.path.join(directory, "_core.*.so"))
    spec = importlib.util.spec_from_file_location("_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    highs_module.cached = module
    return module


def highs_new(budget, engine, tolerance=2e-9):
    mod = highs_module()
    h = mod._Highs()
    for k, v in (("output_flag", False), ("threads", 1), ("parallel", "off"),
                 ("time_limit", float(max(.04, budget))), ("primal_feasibility_tolerance", tolerance),
                 ("dual_feasibility_tolerance", tolerance), ("ipm_optimality_tolerance", 1e-11),
                 ("small_matrix_value", 1e-12), ("large_matrix_value", 1e20),
                 ("infinite_bound", 1e25), ("infinite_cost", 1e25)):
        h.setOptionValue(k, v)
    if engine:
        h.setOptionValue("solver", engine)
    if engine == "ipm":
        h.setOptionValue("run_crossover", "off")
    return h


def highs_model(h, cost, lowers, uppers, row_lower, row_upper, matrix, rowwise=False):
    p, i, v = matrix
    return h.passModel(cost.size, row_upper.size, v.size, 2 if rowwise else 1, 1, 0.,
                       cost, lowers, uppers, row_lower, row_upper, p, i, v, np.zeros(cost.size, np.int32))


def highs_hessian(h, model):
    p, i, v = model.p
    col = np.repeat(np.arange(model.n, dtype=np.int32), np.diff(p))
    take = i >= col
    ptr = np.empty(i.size + 1, dtype=np.int32)
    ptr[0] = 0
    np.cumsum(take, out=ptr[1:])
    return h.passHessian(model.n, int(ptr[-1]), 1, ptr[p], i[take], v[take])


class SimpleBounds:
    def __init__(self, model, eq):
        n, m = model.n, model.m
        p, i, v = model.a
        self.l = np.full(n, -np.inf)
        self.u = np.full(n, np.inf)
        self.upper_rows = self.lower_rows = self.keep = None
        self.matrix = model.a
        if m < 100 or i.size < 100:
            return
        counts = np.bincount(i, minlength=m)
        single = counts == 1
        if np.count_nonzero(single) < max(24, m // 18):
            return
        take = single[i] & (np.abs(v) >= 1e-12)
        pos = np.flatnonzero(take)
        row = i[pos]
        col = np.searchsorted(p, pos, side="right") - 1
        coeff = v[pos]
        val = model.b[row] / coeff
        if np.max(np.abs(val), initial=0.) >= 1e24:
            return
        l, u = self.l.copy(), self.u.copy()
        is_eq = eq[row]
        lo, hi = (coeff < 0) | is_eq, (coeff > 0) | is_eq
        np.maximum.at(l, col[lo], val[lo])
        np.minimum.at(u, col[hi], val[hi])
        if np.any(l > u):
            return
        low_match = lo & (val == l[col])
        high_match = hi & (val == u[col])
        self.lower_rows = np.full(n, -1, dtype=np.int32)
        self.upper_rows = np.full(n, -1, dtype=np.int32)
        self.lower_rows[col[low_match]] = pos[low_match]
        self.upper_rows[col[high_match]] = pos[high_match]
        keep = np.ones(m, dtype=bool)
        keep[row] = False
        c = np.empty(i.size + 1, dtype=np.int32)
        c[0] = 0
        np.cumsum(~take, out=c[1:])
        rowmap = np.cumsum(keep, dtype=np.int32) - 1
        self.matrix = (c[p], rowmap[i[~take]], v[~take])
        self.l, self.u, self.keep = l, u, keep

    def lift(self, model, values, z):
        z.fill(0.)
        if self.keep is None:
            z[:] = -np.asarray(values.row_dual)
            return
        i, v = model.a[1:]
        z[self.keep] = -np.asarray(values.row_dual)
        g = np.asarray(values.col_dual)
        ids = np.where(g > 0, self.lower_rows, self.upper_rows)
        take = ids >= 0
        ids = ids[take]
        z[i[ids]] = -g[take] / v[ids]


def lp_ray(model, primal, eq, budget):
    x, s, z = model.results
    n, m = model.n, model.m
    p, i, v = model.a
    amax = np.max(np.abs(v), initial=0.) or 1.
    h = highs_new(budget, "simplex", 1e-10)
    if primal:
        bmax = np.max(np.abs(model.b), initial=0.)
        if bmax == 0:
            return 0
        zero = np.zeros(n)
        if v.size and p.size < 150000 and v.size < 1200000:
            starts = p[:-1][p[:-1] < p[-1]]
            column_scale = np.ones(n)
            nzcol = p[:-1] < p[1:]
            # Reductions with repeated starts are only read on the last index
            # for a given start, which is the nonempty column.
            values = np.maximum.reduceat(np.abs(v), starts)
            column_scale[nzcol] = values[nzcol[p[:-1] < p[-1]]]
            vs = v / np.repeat(column_scale, np.diff(p))
        else:
            vs = v / amax
        if int(highs_model(h, model.b / bmax, -eq.astype(np.float64), np.ones(m),
                           zero, zero, (p, i, vs), True)) < 0:
            return 0
        h.run()
        values = h.getSolution()
        if values.value_valid and len(values.col_value) == m:
            z[:] = values.col_value
            return model.check(2)
    else:
        qmax = np.max(np.abs(model.q), initial=0.)
        if qmax == 0:
            return 0
        l = np.where(eq, 0., -np.inf)
        u = np.zeros(m)
        if model.p[2].size:
            from scipy.sparse import csc_matrix, vstack

            pmat, imat, vmat = model.p
            pmax = np.max(np.abs(vmat)) or 1.
            a = csc_matrix((v / amax, i, p), shape=(m, n))
            pp = csc_matrix((vmat / pmax, imat, pmat), shape=(n, n))
            aa = vstack([a, pp], format="csc")
            p, i, v = aa.indptr, aa.indices, aa.data
            l, u = np.pad(l, (0, n)), np.pad(u, (0, n))
        else:
            v = v / amax
        if int(highs_model(h, model.q / qmax, np.full(n, -1.), np.ones(n), l, u, (p, i, v))) < 0:
            return 0
        h.run()
        values = h.getSolution()
        if values.value_valid and len(values.col_value) == n:
            x[:] = values.col_value
            s.fill(0.)
            return model.check(3)
    return 0


def highs_solve(model, budget, engine=None, rays=True, tolerance=2e-9):
    start = time.monotonic()
    qp = bool(model.p[2].size)
    n, m = model.n, model.m
    h = highs_new(budget, engine, tolerance)
    eq = np.repeat(model.kinds == 0, model.dims)
    bounds = SimpleBounds(model, eq)
    b = model.b if bounds.keep is None else model.b[bounds.keep]
    el = eq if bounds.keep is None else eq[bounds.keep]
    lows = np.where(el, b, -np.inf)
    if int(highs_model(h, model.q, bounds.l, bounds.u, lows, b, bounds.matrix)) < 0:
        return 0
    if qp:
        if int(highs_hessian(h, model)) < 0:
            return 0
    h.run()
    status = int(h.getModelStatus())
    if TRACE:
        print(f"HiGHS {engine}: {status}, {time.monotonic()-start:.3f}s", file=sys.stderr)
    if status in (6, 7, 13, 14):
        values = h.getSolution()
        if values.value_valid and values.dual_valid:
            x, s, z = model.results
            x[:] = values.col_value
            bounds.lift(model, values, z)
            if model.check(17):
                return 1
            if n+m < 210000 and (status in (6, 7) or time.monotonic()-start < .45):
                if model.polish(.10):
                    return 1
    if status in (8, 9, 10) and rays:
        remaining = max(.09, budget - (time.monotonic() - start))
        result = lp_ray(model, status in (8, 9), eq, remaining)
        if result or status != 9:
            return result
        return lp_ray(model, False, eq, remaining)
    return 0


def solve(model):
    native_mode = os.environ.get("CONIC_MODE")
    if native_mode is not None:
        if int(native_mode) == 29:
            from pivoted import enable
            global PIVOTED
            PIVOTED = enable(LIB)
        return model.native(30., int(native_mode))
    qp = model.p[2].size > 0
    linear = model.linear
    size = model.n + model.m
    a_nnz, p_nnz = model.a[2].size, model.p[2].size
    soc, psd, exp = (bool(np.any(model.kinds == k)) for k in (2, 3, 4))
    if linear and a_nnz <= model.m and p_nnz <= model.n:
        status = model.native(.15, -1)
        if status:
            return status
    if linear and qp:
        if a_nnz <= model.n + model.m and p_nnz <= model.n:
            status = model.native(.16, 22)
            if status:
                return status
        elif a_nnz <= model.m:
            status = model.native(.19, 23)
            if status:
                return status
    # Short first pass, followed by recovery variants for higher-budget problems.
    if linear and not qp and model.n and model.m:
        solver = "ipm" if (a_nnz > 20000 and size > 5000) else "simplex"
        status = highs_solve(model, .59, solver)
        if status:
            return status
    if linear and qp and not np.any(model.kinds == 1):
        status = model.native(.29, 14)
        if status:
            return status
    first = 2 if soc and not psd and not exp else 0
    status = model.native(.57 if size < 24000 else 8., first)
    if status:
        return status
    elapsed = time.monotonic() - START
    if linear and model.n and model.m:
        if qp:
            status = model.native(max(.15,.66-elapsed), 13)
            if not status:
                status = highs_solve(model, .45 if elapsed < .52 else 4.)
        else:
            status = highs_solve(model, .65 if elapsed < .3 else 8., "ipm" if solver == "simplex" else "simplex",
                                 tolerance=1e-10)
        if status:
            return status
    variants = (1, 17, 2, 3) if exp else (2, 3, 4) if psd else (15, 18, 3, 16) if soc else (2, 3)
    for k, mode in enumerate(variants):
        status = model.native(2.2 if soc or exp else 4., mode)
        if status:
            return status
        if k == 0 and (soc or exp) and size < 70000 and time.monotonic()-START < 3.:
            status = model.native(max(.12,.69-(time.monotonic()-START)), 24)
            if status:
                return status
    if not linear and size < 58000:
        from outer import optimal_outer

        status = optimal_outer(model, max(.3, 15. - (time.monotonic()-START)), highs_new, highs_model,
                              highs_hessian)
        if status:
            return status
    if linear and qp:
        return model.native(4., 21)
    if linear and not qp and model.n and model.m and time.monotonic() - START < 12:
        return highs_solve(model, 5., "simplex", tolerance=1e-10)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--outer", action="store_true")
    ap.add_argument("--backend", default="")
    ap.add_argument("--budget", type=float, default=16.)
    args = ap.parse_args()
    arrays = {}
    status = 0
    try:
        model = Model(args.input)
        limit = max(.05, args.budget - (time.monotonic() - START))
        if args.outer or args.backend == "outer":
            from outer import optimal_outer

            status = optimal_outer(model, min(8.5, limit), highs_new, highs_model, highs_hessian)
        elif args.backend == "pivoted":
            from pivoted import enable
            PIVOTED = enable(LIB)
            status = model.native(min(3.5,limit*.7),29)
            if not status and not model.linear:
                from outer import optimal_outer
                status = optimal_outer(model, max(.09,min(6.,args.budget-(time.monotonic()-START))),
                                      highs_new,highs_model,highs_hessian)
        elif args.backend == "quadratic":
            status = highs_solve(model, min(4., limit), tolerance=1e-10)
            if not status and limit-(time.monotonic()-START)>.12:
                status = model.native(min(5., max(.12, args.budget-(time.monotonic()-START))),21)
        else:
            status = solve(model)
        x, s, z = model.results
        if status in (1, 3):
            arrays.update(x=x, s=s)
        if status in (1, 2):
            arrays.update(z=z)
    except Exception as error:
        if TRACE:
            raise
        print(f"conic: {type(error).__name__}: {error}", file=sys.stderr)
        status = 0
    arrays["metadata_json"] = np.array(json.dumps({"schema_version": 1, "status": STATUSES[status]}))
    with open(args.output, "wb") as f:
        np.savez(f, **arrays)


if __name__ == "__main__":
    main()
