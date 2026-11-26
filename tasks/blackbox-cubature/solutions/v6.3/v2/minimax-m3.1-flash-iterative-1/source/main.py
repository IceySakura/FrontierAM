#!/usr/bin/env python3
"""JSON Lines driver for the cubature submission.

The evaluator sends an ``init`` message, answers ``query`` messages with
``values`` and finally expects a ``result`` message.  Every function value
consumes budget, so all requests are batched (at most 512 points per
message) and never exceed the announced budget.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from cubature import integrate  # noqa: E402

MAX_CHUNK = 512
TIME_LIMIT = 24.0


class _Timeout(Exception):
    pass


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read():
    line = sys.stdin.readline()
    if not line:
        raise EOFError("evaluator closed the connection")
    return json.loads(line)


def main() -> int:
    t0 = time.time()
    try:
        msg = _read()
    except Exception:
        return 0
    if not isinstance(msg, dict) or msg.get("type") != "init":
        _send({"type": "result", "estimate": 0.0})
        return 0

    try:
        dim = int(msg["dimension"])
        budget = int(msg.get("budget", 2048))
        seed = int(msg.get("seed", 0))
    except Exception:
        _send({"type": "result", "estimate": 0.0})
        return 0
    if dim <= 0 or budget <= 0:
        _send({"type": "result", "estimate": 0.0})
        return 0

    bounds = msg.get("bounds") or [[0.0, 1.0]] * dim
    try:
        lo = np.array([float(b[0]) for b in bounds], float)
        hi = np.array([float(b[1]) for b in bounds], float)
        if len(lo) != dim or len(hi) != dim:
            lo = np.zeros(dim)
            hi = np.ones(dim)
    except Exception:
        lo = np.zeros(dim)
        hi = np.ones(dim)
    span = hi - lo

    seen = {"n": 0}
    values_seen = []

    def f_batch(X):
        X = np.atleast_2d(np.asarray(X, float))
        if len(X) == 0:
            return np.zeros(0)
        if time.time() - t0 > TIME_LIMIT:
            raise _Timeout()
        out = []
        for i in range(0, len(X), MAX_CHUNK):
            chunk = X[i:i + MAX_CHUNK]
            pts = (lo[None, :] + chunk * span[None, :]).tolist()
            _send({"type": "query", "points": pts})
            resp = _read()
            vals = resp.get("values")
            if vals is None or len(vals) != len(chunk):
                raise ValueError("malformed values message")
            for v in vals:
                fv = float(v)
                if not math.isfinite(fv):
                    fv = 0.0
                out.append(fv)
            seen["n"] += len(chunk)
            values_seen.extend(out[-len(chunk):])
        return np.array(out, float)

    estimate = None
    try:
        estimate = integrate(f_batch, dim, budget, seed)
    except Exception:
        estimate = None
    if estimate is None or not math.isfinite(float(estimate)):
        if values_seen:
            estimate = float(np.mean(values_seen))
        else:
            estimate = 0.0
    # The algorithm integrates over the unit cube; map back to the bounds.
    volume = float(np.prod(span)) if np.all(np.isfinite(span)) else 1.0
    try:
        estimate = float(estimate) * volume
    except Exception:
        estimate = 0.0
    if not math.isfinite(estimate):
        estimate = 0.0
    _send({"type": "result", "estimate": float(estimate)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            _send({"type": "result", "estimate": 0.0})
        except Exception:
            pass
        sys.exit(0)
