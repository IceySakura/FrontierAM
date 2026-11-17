"""Continuum (x -> 0) limit estimation from 4-5 exact observations.

See README.md for the method. Always writes a finite estimate.
"""
import json
import math
import os
import sys


def read_input(path):
    with open(path) as fh:
        obj = json.load(fh)
    x = [float(v) for v in obj["x"]]
    y = [float(v) for v in obj["y"]]
    return x, y


def write_output(path, value):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps({"estimate": float(value)}, allow_nan=False))
    os.replace(tmp, path)


def finest_value(x, y):
    i = min(range(len(x)), key=lambda k: x[k])
    return y[i]


def plausible(z, x, y):
    if z is None or not math.isfinite(z):
        return False
    span = max(y) - min(y)
    return abs(z - finest_value(x, y)) <= 1000.0 * span + 1e-9 * max(1.0, abs(z)) + 1e-12


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    x, y = read_input(inp)
    z = None
    try:
        import signal

        def _timeout(signum, frame):
            raise TimeoutError("estimator time budget exceeded")

        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(40)
    except Exception:
        pass
    try:
        import numpy as np
        np.seterr(all="ignore")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import est
        z = est.robust_estimate(x, y)
    except Exception:
        z = None
    try:
        signal.alarm(0)
    except Exception:
        pass
    if not plausible(z, x, y):
        z = finest_value(x, y)
    if not math.isfinite(z):
        z = 0.0
    write_output(out, z)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # last resort: never fail without an output
        out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
        try:
            x, y = read_input(sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json")
            v = finest_value(x, y)
            if not math.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        write_output(out, v)
