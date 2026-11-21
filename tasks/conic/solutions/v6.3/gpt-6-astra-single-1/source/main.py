import argparse
import os
import traceback
import numpy as np
from common import load, save, assess, cone_def, blocks, smat, svec


def constant_problem(P, q, A, b, cones):
    result = {"status": "solved", "x": np.empty(0), "s": b.copy(), "z": np.zeros(len(b))}
    if assess(P, q, A, b, cones, result, tol=1e-8)[0]:
        return result
    z = np.zeros(len(b))
    for k, d, i, j in blocks(cones):
        v = b[i:j]
        if k == "zero":
            z[i:j] = -v
        elif k == "nonnegative":
            z[i:j] = np.maximum(-v, 0)
        elif k == "soc":
            w = -v
            r = np.linalg.norm(w[1:])
            if r <= w[0]:
                z[i:j] = w
            elif r > -w[0]:
                z[i:j] = .5 * (r + w[0]) * np.r_[1., w[1:] / r]
        elif k == "psd":
            a, U = np.linalg.eigh(smat(v, d))
            z[i:j] = svec((U * np.maximum(-a, 0)) @ U.T)
        elif cone_def(v, [{"type": k, "dim": d}]) > 1e-8:
            u, vv, w = v
            if vv < 0:
                z[i:j] = [0, 1, 0]
            elif w < 0:
                z[i:j] = [0, 0, 1]
            else:
                t = u / vv if vv > 0 else np.log(max(1., 2 * abs(w) / max(u, 1e-300)))
                t = np.clip(t, -700, 700)
                z[i:j] = [-1, t - 1, np.exp(-t)]
    result = {"status": "primal_infeasible", "z": z}
    if assess(P, q, A, b, cones, result, tol=1e-8)[0]:
        return result
    return {"status": "unresolved"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"status": "unresolved"}
    try:
        P, q, A, b, cones = load(args.input)
        original = (P, q, A, b, cones)
        if not np.any(q):
            zero = {"status": "solved", "x": np.zeros(len(q)),
                    "s": b.copy(), "z": np.zeros(len(b))}
            if assess(*original, zero, tol=1e-8)[0]:
                save(args.output, zero)
                return
        from separable import solve as separable_solve
        direct = separable_solve(*original)
        if direct is not None and assess(*original, direct, tol=1e-8)[0]:
            save(args.output, direct)
            return
        polyhedral = all(c["type"] in ("zero", "nonnegative") for c in cones)
        changed = False
        if not polyhedral or (P.nnz and len(q) > 1000):
            from reduce import reduce_problem
            reduced, restore, changed = reduce_problem(*original)
        else:
            reduced = original
            restore = lambda result: result
        Pr, qr, Ar, br, cr = reduced
        direct = separable_solve(*reduced)
        if not len(qr):
            result = constant_problem(*reduced)
        elif direct is not None and assess(*reduced, direct, tol=1e-8)[0]:
            result = direct
        elif all(c["type"] in ("zero", "nonnegative") for c in cr) and not (Pr.nnz and len(qr) > 1000):
            from highs_solver import solve
            result = solve(*reduced)
        else:
            from ipm import solve
            validator = (lambda candidate: assess(*original, restore(candidate))) if changed else None
            result = solve(*reduced, validator=validator)
        result = restore(result)
        verified, error = assess(*original, result, tol=1e-8)
        if not verified and error < 1e-3 and result["status"] == "solved":
            from polish import refine
            result = refine(*original, result)
        if changed and not assess(*original, result, tol=1e-8)[0]:
            from ipm import solve
            other = solve(*original)
            if assess(*original, other, tol=1e-8)[0]:
                result = other
        if not assess(*original, result, tol=1e-8)[0]:
            result = {"status": "unresolved"}
    except Exception:
        if os.environ.get("SOLVER_DEBUG"):
            traceback.print_exc()
        result = {"status": "unresolved"}
    save(args.output, result)


if __name__ == "__main__":
    main()
