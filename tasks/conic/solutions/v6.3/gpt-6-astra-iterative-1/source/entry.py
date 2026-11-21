#!/usr/bin/env python3
import json
import sys
import numpy as np


def fast():
    args = sys.argv[1:]
    if len(args) != 4 or set(args[::2]) != {"--input", "--output"}:
        return False
    paths = dict(zip(args[::2], args[1::2]))
    result = None
    with np.load(paths["--input"], allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        cones = meta["cones"]
        if any(c["type"] not in ("zero","nonnegative") for c in cones):
            return False
        q = np.asarray(data["q"], dtype=np.float64)
        if q.size == 0:
            return False
        b = np.asarray(data["b"], dtype=np.float64)
        eq = np.zeros(len(b), dtype=bool)
        start = 0
        for cone in cones:
            end = start+cone["dim"]
            if cone["type"] == "zero":
                eq[start:end] = True
            start = end
        if not q.any():
            s = np.maximum(b, 0.)
            s[eq] = 0.
            scale = 1+np.max(np.abs(b), initial=0.)
            if np.max(np.abs(s-b), initial=0.) <= 8e-9*scale:
                result = {"status":"solved","x":np.zeros(len(q)),
                          "s":s,"z":np.zeros(len(b))}
        if result is None:
            if (len(q) > 10000 or len(b) > 200000
                    or data.zip.getinfo("P_data.npy").file_size > 256
                    or len(data["P_data"]) != 0):
                return False
            ap = np.asarray(data["A_indptr"], dtype=np.int32)
            ai = np.asarray(data["A_indices"], dtype=np.int32)
            av = np.asarray(data["A_data"], dtype=np.float64)
            from fast_lp import solve
            result = solve(q, b, ap, ai, av, eq, cones)
    if result is None:
        return False
    arrays = {k: result[k] for k in ("x","s","z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(
        {"schema_version":1,"status":result["status"]}))
    with open(paths["--output"], "wb") as stream:
        np.savez(stream, **arrays)
    return True


def main():
    try:
        if fast():
            return
    except Exception:
        pass
    from bootstrap import load_solver
    general = load_solver()
    general()


if __name__ == "__main__":
    main()
