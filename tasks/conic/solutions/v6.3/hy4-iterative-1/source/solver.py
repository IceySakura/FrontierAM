#!/usr/bin/env python3
"""Driver for the offline conic QP solver.

Reads an NPZ instance, dispatches to the HiGHS LP/QP path or to the
primal-dual interior-point conic path, validates the answer locally and
writes the result NPZ.  A watchdog thread writes a valid `unresolved`
file before a short (1 s) case deadline so that a killed process still
leaves a well formed output.
"""
import sys
import os
import json
import time
import threading

T_START = time.perf_counter()


def _elapsed():
    return time.perf_counter() - T_START


def _csc_dense(data, indices, indptr, shape, dtype):
    """Build a dense matrix from CSC, summing duplicate entries."""
    import numpy as np
    out = np.zeros(shape, dtype=dtype)
    n = shape[1]
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
    if cols.size:
        np.add.at(out, (indices.astype(np.int64), cols), data)
    return out


def _write_result(path, status, x=None, s=None, z=None, objectives=None):
    import numpy as np
    md = {"schema_version": 1, "status": status}
    if objectives is not None:
        md.update(objectives)
    arrays = {"metadata_json": np.array(json.dumps(md, allow_nan=False))}
    for key, value in (("x", x), ("s", s), ("z", z)):
        if value is not None:
            arrays[key] = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    # Write atomically: a process killed by a short deadline must leave a
    # complete, valid NPZ (the provisional `unresolved` file), never a
    # half-written ZIP.
    tmp = str(path) + ".tmp"
    try:
        with open(tmp, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            with open(path, "wb") as stream:
                np.savez(stream, **arrays)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _finite_vec(v, size):
    import numpy as np
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size != size or not np.isfinite(v).all():
        return None
    return v


def main():
    # Keep startup lean: only numpy is imported eagerly.
    import numpy as np
    argv = sys.argv[1:]
    input_path = output_path = None
    i = 0
    while i < len(argv):
        if argv[i] == "--input" and i + 1 < len(argv):
            input_path = argv[i + 1]
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        else:
            i += 1
    if input_path is None or output_path is None:
        return 2

    with np.load(input_path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        q = np.asarray(d["q"], dtype=np.float64).reshape(-1)
        b = np.asarray(d["b"], dtype=np.float64).reshape(-1)
        P_data = np.asarray(d["P_data"], dtype=np.float64)
        P_indices = np.asarray(d["P_indices"])
        P_indptr = np.asarray(d["P_indptr"])
        P_shape = tuple(int(v) for v in d["P_shape"])
        A_data = np.asarray(d["A_data"], dtype=np.float64)
        A_indices = np.asarray(d["A_indices"])
        A_indptr = np.asarray(d["A_indptr"])
        A_shape = tuple(int(v) for v in d["A_shape"])
    cones = meta["cones"]
    n = q.size
    m = b.size

    holder = {"done": False, "provisional": False}
    lock = threading.Lock()

    def emit(result):
        with lock:
            if holder["done"]:
                return
            holder["done"] = True
            status = result.get("status", "unresolved")
            if status == "solved":
                x = _finite_vec(result.get("x"), n)
                s = _finite_vec(result.get("s"), m)
                z = _finite_vec(result.get("z"), m)
                if x is None or s is None or z is None:
                    status = "unresolved"
            elif status == "primal_infeasible":
                z = _finite_vec(result.get("z"), m)
                if z is None:
                    status = "unresolved"
                x = s = None
            elif status == "dual_infeasible":
                x = _finite_vec(result.get("x"), n)
                s = _finite_vec(result.get("s"), m)
                if x is None or s is None:
                    status = "unresolved"
                z = None
            else:
                status = "unresolved"
                x = s = z = None
            try:
                _write_result(output_path, status, x, s, z)
            except Exception:
                try:
                    _write_result(output_path, "unresolved")
                except Exception:
                    pass

    def watchdog():
        # Write a legal (but unresolved) answer early: on 1-second cases a
        # late kill would otherwise leave no output file at all.
        while True:
            time.sleep(0.05)
            with lock:
                if holder["done"] or holder["provisional"]:
                    return
                if _elapsed() > 0.72:
                    holder["provisional"] = True
                    try:
                        _write_result(output_path, "unresolved")
                    except Exception:
                        pass
                    return

    try:
        threading.Thread(target=watchdog, daemon=True).start()
    except Exception:
        pass

    size = n + m
    if size < 3000:
        hard_cap = 4.0
    elif size < 60000:
        hard_cap = 9.0
    else:
        hard_cap = 14.0
    # Bench-side safety: never spend more than this fraction of the budget.
    hard_cap = min(hard_cap, 25.0)

    simple = all(c["type"] in ("zero", "nonnegative") for c in cones)
    has_p = bool(P_data.size and np.any(P_data != 0))
    big_psd = any(c["type"] == "psd" and c["dim"] > 12 for c in cones)
    deadline = T_START + hard_cap
    result = {"status": "unresolved"}
    try:
        if simple:
            # cuts.py passes the LP/QP straight to HiGHS with no scaling,
            # certificate preflights or repeated polishing; for LPs its
            # simplex solution is already exact to 1e-8 and for QPs the
            # active-set KKT polish supplies the missing accuracy.
            import cuts
            r = cuts.solve_by_cuts(P_data, P_indices, P_indptr, P_shape, q,
                                   A_data, A_indices, A_indptr, A_shape, b,
                                   cones,
                                   deadline=min(deadline, time.perf_counter() +
                                                (0.8 if not has_p else 0.3 * hard_cap)),
                                   max_iter=14)
            if r.get("status") in ("solved", "primal_infeasible",
                                   "dual_infeasible"):
                result = r
            if result.get("status") == "unresolved" and has_p and n <= 1800 and m <= 20000:
                import conic
                r = conic.solve_conic(P_data, P_indices, P_indptr, P_shape, q,
                                      A_data, A_indices, A_indptr, A_shape, b,
                                      cones, deadline=min(deadline, time.perf_counter() +
                                                          0.3 * hard_cap))
                if r.get("status") == "solved":
                    result = r
            if (result.get("status") not in ("solved", "primal_infeasible",
                                             "dual_infeasible")
                    and not (has_p and n > 20000)):
                import lpqp
                result = lpqp.solve_lpqp(P_data, P_indices, P_indptr, P_shape, q,
                                         A_data, A_indices, A_indptr, A_shape, b,
                                         cones, time_budget=max(0.2, hard_cap - _elapsed()))
        else:
            if n <= 2000000 and m <= 2000000:
                import cuts
                if big_psd:
                    cut_budget, cut_iters = 0.45, 12
                else:
                    cut_budget = 0.9 if size < 5000 else (2.0 if size < 100000 else 3.5)
                    cut_iters = 60
                r = cuts.solve_by_cuts(P_data, P_indices, P_indptr, P_shape, q,
                                       A_data, A_indices, A_indptr, A_shape, b,
                                       cones,
                                       deadline=min(deadline, time.perf_counter() + cut_budget),
                                       max_iter=cut_iters)
                if r.get("status") in ("solved", "primal_infeasible",
                                       "dual_infeasible"):
                    result = r
            if result.get("status") not in ("solved", "primal_infeasible",
                                            "dual_infeasible"):
                import conic
                r = conic.solve_conic(P_data, P_indices, P_indptr, P_shape, q,
                                      A_data, A_indices, A_indptr, A_shape, b,
                                      cones, deadline=deadline)
                if r.get("status") == "solved":
                    result = r
    except Exception as exc:
        try:
            sys.stderr.write("solver error: %r\n" % (exc,))
        except Exception:
            pass
        result = {"status": "unresolved"}

    emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
