"""HumanPortfolio-v1 (the author's frozen selector) through the query protocol: pcubature for d = 2, 5 and QMCPy
DigitalNetB2 for d = 3, 8, with the parameters of private_hidden/run_portfolio_baseline.py; every value comes from a
query, one point per pcubature callback."""
import ctypes
import json
import os
import sys

import numpy as np

CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
                      ctypes.c_uint, ctypes.POINTER(ctypes.c_double))


def ask(points):
    print(json.dumps({"type": "query", "points": [list(map(float, p)) for p in points]}), flush=True)
    return json.loads(sys.stdin.readline())["values"]


def pcubature(d, budget):
    fn = ctypes.CDLL(os.environ["FRONTIS_PCUBATURE_LIB"]).pcubature
    fn.argtypes = [ctypes.c_uint, CB, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    fn.restype = ctypes.c_int
    count = [0]

    @CB
    def callback(ndim, xptr, _data, _fdim, yptr):
        if count[0] >= budget:
            return 1
        yptr[0] = ask([np.ctypeslib.as_array(xptr, shape=(ndim,)).copy()])[0]
        count[0] += 1
        return 0

    lo = (ctypes.c_double * d)(*([0.0] * d)); hi = (ctypes.c_double * d)(*([1.0] * d))
    value = (ctypes.c_double * 1)(); error = (ctypes.c_double * 1)()
    fn(1, callback, None, d, lo, hi, budget, 0.0, 0.0, 0, value, error)
    return float(value[0])


def digital_net(d, budget, seed):
    from qmcpy.discrete_distribution import DigitalNetB2
    points = np.asarray(DigitalNetB2(d, seed=seed)(budget), dtype=float)
    values = []
    for start in range(0, len(points), 512):
        values += ask(points[start:start + 512])
    return float(np.mean(values))


def main():
    init = json.loads(sys.stdin.readline())
    d, budget, seed = int(init["dimension"]), int(init["budget"]), int(init["seed"])
    estimate = pcubature(d, budget) if d in (2, 5) else digital_net(d, budget, seed)
    print(json.dumps({"type": "result", "estimate": estimate}), flush=True)


if __name__ == "__main__":
    main()
