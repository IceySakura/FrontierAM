#!/usr/bin/env python3
"""Single-process, single-threaded sparse convex optimization."""
import argparse
import json
import os
import sys
import time

import numpy as np
_numpy_exports=np.__all__
try:
    # SciPy's compatibility namespace otherwise eagerly imports NumPy's
    # compiler and test suites, neither of which is needed for solving.
    np.__all__=[k for k in _numpy_exports if k not in ("f2py","testing")]
    from scipy import sparse
finally:
    np.__all__=_numpy_exports


def highs_core():
    # Import the extension directly, avoiding scipy.optimize's unrelated imports.
    if "_core" not in sys.modules:
        import importlib.util
        import scipy
        path = os.path.join(os.path.dirname(scipy.__file__), "optimize", "_highspy",
                            "_core.cpython-312-x86_64-linux-gnu.so")
        spec = importlib.util.spec_from_file_location("_core", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_core"] = module
        sys.modules["scipy.optimize._highspy._core"]=module
        spec.loader.exec_module(module)
    return sys.modules["_core"]


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


def load(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata_json"]))
        matrices = []
        for p in ("P", "A"):
            a = sparse.csc_matrix((f[p + "_data"], f[p + "_indices"],
                                   f[p + "_indptr"]), shape=tuple(f[p + "_shape"]))
            a.sum_duplicates()
            a.eliminate_zeros()
            a.sort_indices()
            matrices.append(a)
        return matrices[0], f["q"], matrices[1], f["b"], meta["cones"]


def poly_candidate(P,q,A,b,cones,x,s,z,tol=8e-9):
    if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
        return False
    ax=A@x
    px=P@x
    az=A.T@z
    if not (np.isfinite(ax).all() and np.isfinite(px).all() and np.isfinite(az).all()):
        return False
    if norm(ax+s-b)>tol*(1+max(norm(ax),norm(s),norm(b))):
        return False
    if norm(px+q+az)>tol*(1+max(norm(px),norm(q),norm(az))):
        return False
    primal=0.5*(x@px)+q@x
    dual=-0.5*(x@px)-b@z
    gap=abs(primal-dual)
    if not (gap<=tol or gap<=tol*max(abs(primal),abs(dual))):
        return False
    st=0
    for c in cones:
        sl=s[st:st+c["dim"]]
        zz=z[st:st+c["dim"]]
        if c["type"]=="zero":
            if norm(sl)>tol*max(1.,norm(sl)):
                return False
        else:
            if (np.min(sl,initial=0.)< -tol*max(1.,norm(sl)) or
                np.min(zz,initial=0.)< -tol*max(1.,norm(zz))):
                return False
        st+=c["dim"]
    return True


def highs_model(P, q, A, lo, hi, col_lo=None, col_hi=None, method="choose",
                time_limit=240.0,feas_tol=1e-9,crossover=False):
    hc = highs_core()
    n, m = len(q), len(hi)
    model = hc.HighsModel()
    lp = model.lp_
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -hc.kHighsInf) if col_lo is None else col_lo
    lp.col_upper_ = np.full(n, hc.kHighsInf) if col_hi is None else col_hi
    lp.row_lower_ = lo
    lp.row_upper_ = hi
    sm = lp.a_matrix_
    sm.num_col_ = n
    sm.num_row_ = m
    sm.format_ = hc.MatrixFormat.kColwise
    sm.start_ = A.indptr
    sm.index_ = A.indices
    sm.value_ = A.data
    if P is not None and P.nnz:
        pp = sparse.tril(P, format="csc")
        he = model.hessian_
        he.dim_ = n
        he.format_ = hc.HessianFormat.kTriangular
        he.start_ = pp.indptr
        he.index_ = pp.indices
        he.value_ = pp.data
    h = hc._Highs()
    for k, v in (("output_flag", False), ("threads", 1), ("parallel", "off"),
                 ("random_seed",0),
                 ("primal_feasibility_tolerance", feas_tol),
                 ("dual_feasibility_tolerance", feas_tol),
                 ("ipm_optimality_tolerance", 1e-11),
                 ("time_limit", time_limit), ("solver", method)):
        h.setOptionValue(k, v)
    if method=="ipm":
        h.setOptionValue("run_crossover", "on" if crossover else "off")
    h.passModel(model)
    h.run()
    return h


def polyhedral(P, q, A, b, cones):
    hc = highs_core()
    lo = np.full(len(b), -hc.kHighsInf)
    eq = np.zeros(len(b), dtype=bool)
    i = 0
    for c in cones:
        if c["type"] == "zero":
            eq[i:i + c["dim"]] = True
        i += c["dim"]
    lo[eq] = b[eq]
    method=os.environ.get("LP_METHOD")
    if method is None:
        method="ipm" if not P.nnz and len(q)>=10000 and A.nnz>5*len(q) else "choose"
    objective_scale=(min(1e4/max(norm(P.data),1e-100),
        1e12/max(norm(q),1.)) if P.nnz else 1.)
    h = highs_model(P*objective_scale,q*objective_scale,A,lo,b,method=method)
    status = h.getModelStatus()
    sol=h.getSolution()
    if os.environ.get("SOLVER_DEBUG"):
        print("HiGHS",h.modelStatusToString(status),sol.value_valid,sol.dual_valid,
              file=sys.stderr,flush=True)
    if len(sol.col_value)==len(q) and len(sol.row_dual)==len(b):
        x = np.asarray(sol.col_value)
        z = -np.asarray(sol.row_dual)/objective_scale
        s=b-A@x
        if poly_candidate(P,q,A,b,cones,x,s,z):
            return {"status": "solved", "x": x, "s": s, "z": z}
        projected=s.copy()
        projected[eq]=0.
        projected[~eq]=np.maximum(projected[~eq],0.)
        zp=z.copy()
        zp[~eq]=np.maximum(zp[~eq],0.)
        if poly_candidate(P,q,A,b,cones,x,projected,zp):
            return {"status":"solved","x":x,"s":projected,"z":zp}
    if status in (hc.HighsModelStatus.kInfeasible,
                  hc.HighsModelStatus.kUnboundedOrInfeasible):
        # Farkas alternative: A' z=0, b'z=-1, z in the dual orthant.
        ascale = norm(A.data) or 1.0
        bscale = norm(b)
        if bscale:
            aa = sparse.vstack([A.T / ascale, sparse.csr_matrix(b / bscale)],
                               format="csc")
            rhs = np.r_[np.zeros(len(q)), -1.0]
            cl = np.zeros(len(b))
            cl[eq] = -hc.kHighsInf
            hh = highs_model(None, np.zeros(len(b)), aa, rhs, rhs, cl)
            if hh.getModelStatus() == hc.HighsModelStatus.kOptimal:
                return {"status": "primal_infeasible",
                        "z": np.asarray(hh.getSolution().col_value)}
    if status in (hc.HighsModelStatus.kUnbounded,
                  hc.HighsModelStatus.kUnboundedOrInfeasible):
        ascale = norm(A.data) or 1.0
        pscale = norm(P.data) or 1.0
        qscale = norm(q)
        if qscale:
            aa = sparse.vstack([A / ascale, P / pscale,
                                sparse.csr_matrix(q / qscale)], format="csc")
            hi = np.r_[np.zeros(len(b) + len(q)), -1.0]
            ll = hi.copy()
            ll[:len(b)][~eq] = -hc.kHighsInf
            hh = highs_model(None, np.zeros(len(q)), aa, ll, hi)
            if hh.getModelStatus() == hc.HighsModelStatus.kOptimal:
                x = np.asarray(hh.getSolution().col_value)
                return {"status": "dual_infeasible", "x": x, "s": -A @ x}
    return {"status": "unresolved"}


def solve(P, q, A, b, cones):
    if not len(q) or not len(b) or not np.any(q):
        from edge import simple_case
        result=simple_case(P,q,A,b,cones)
        if result is not None:
            return result
    if any(c["type"] == "exponential" for c in cones):
        from barrier import solve_barrier
        from presolve import Substitution
        from gp import solve_gp
        reduced = Substitution(P,q,A,b,cones).eliminate_equalities()
        result=solve_gp(*reduced.data())
        if result is None:
            result=solve_barrier(*reduced.data())
        return reduced.restore(result)
    if all(c["type"] in ("zero", "nonnegative") for c in cones):
        if len(q)>100000:
            from presolve import zero_cost_feasible
            quick=zero_cost_feasible(P,q,A,b,cones)
            if quick is not None:
                return quick
            if P.nnz:
                from lasso import solve_lasso
                quick=solve_lasso(P,q,A,b,cones)
                if quick is not None:
                    return quick
        if P.nnz == 0 and not os.environ.get("FORCE_IPM"):
            return polyhedral(P, q, A, b, cones)
        backend=os.environ.get("QP_BACKEND","auto")
        if (backend=="highs" or (backend=="auto" and
            500<=len(q)<=7000 and P.nnz>5*len(q))):
            result=polyhedral(P,q,A,b,cones)
            if result["status"]!="unresolved":
                return result
        if P.nnz and len(q)>=2000 and os.environ.get("QP_BACKEND")=="fw":
            from qp_lp import solve_qp_lp
            result=solve_qp_lp(P,q,A,b,cones)
            if result is not None:
                return result
        if P.nnz and os.environ.get("QP_BACKEND")=="admm":
            from admm import solve_admm
            result=solve_admm(P,q,A,b,cones)
            if result is not None:
                return result
    from ipm import conic
    if (os.environ.get("SOC_BACKEND")=="oa" and not P.nnz and
        any(c["type"]=="soc" for c in cones) and
        all(c["type"] in ("zero","nonnegative","soc") for c in cones)):
        from soc_lp import outer_approx
        result,warm=outer_approx(P,q,A,b,cones)
        if result is not None:
            return result
        return conic(P,q,A,b,cones,warm=warm)
    reduce_qp=(P.nnz and os.environ.get("QP_PRESOLVE") and
        all(c["type"] in ("zero","nonnegative") for c in cones))
    if (any(c["type"] == "soc" for c in cones) or reduce_qp) and not os.environ.get("NO_PRESOLVE"):
        from presolve import Substitution
        from cones import Layout
        from ipm import accepted,certificate,polish
        reduced=Substitution(P,q,A,b,cones).remove_constants()
        if os.environ.get("EQ_PRESOLVE"):
            reduced.eliminate_equalities(rounds=5,max_degree=int(os.environ["EQ_PRESOLVE"]))
        elif reduce_qp:
            reduced.eliminate_equalities(rounds=10,max_degree=int(os.environ["QP_PRESOLVE"]))
        if not reduced.steps:
            return conic(P,q,A,b,cones)
        original_layout=Layout(cones,len(b))
        def verify(candidate):
            full=reduced.restore(dict(candidate))
            if full["status"]=="solved":
                if accepted(P,q,A,b,original_layout,full["x"],full["s"],full["z"],tol=7e-9):
                    return full
                return polish(P,q,A,b,original_layout,full["x"],full["s"],full["z"])
            cert=certificate(P,q,A,b,original_layout,full.get("x",np.zeros(len(q))),
                full.get("s",np.zeros(len(b))),full.get("z",np.zeros(len(b))),tol=7e-9)
            return cert if cert and cert["status"]==full["status"] else None
        return conic(*reduced.data(),verifier=verify)
    result=conic(P,q,A,b,cones)
    if (result["status"]=="unresolved" and P.nnz and len(q)<=2000 and
        all(c["type"] in ("zero","nonnegative") for c in cones)):
        from admm import solve_admm
        alternative=solve_admm(P,q,A,b,cones)
        if alternative is not None:
            result=alternative
    if result["status"]=="unresolved" and any(c["type"]=="psd" for c in cones):
        result=conic(P,q,A,b,cones,max_iter=150,robust=True,unscaled=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        result = solve(*load(args.input))
    except Exception as error:
        if os.environ.get("SOLVER_DEBUG"):
            import traceback
            traceback.print_exc()
        result = {"status": "unresolved"}
    status = result.pop("status")
    result["metadata_json"] = np.array(json.dumps(
        {"schema_version": 1, "status": status}, allow_nan=False))
    with open(args.output, "wb") as f:
        np.savez(f, **result)


if __name__ == "__main__":
    main()
