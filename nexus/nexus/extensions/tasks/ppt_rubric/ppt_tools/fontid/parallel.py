#!/usr/bin/env python3
"并行渲染比对：把 219 个候选分批发给常驻进程池。"
import multiprocessing as mp
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_POOL = None
_NW = int(os.environ.get("FONTID_WORKERS", "48"))
def _work(task):
    sys.path.insert(0, HERE)
    import glyphsig as G
    import bench_v3 as V3
    qv, qar, items = task
    q = {"v": qv, "ar": qar, "mode": "stretch"}
    out = []
    for f, i, text, px, sw, sk in items:
        ink = G.render_text(f, i, text, px, stroke=sw, skew=sk)
        if ink is None:
            out.append(None)
            continue
        d = V3.dist(q, V3.n_stretch(ink, 0.0))
        out.append(None if d is None else float(d))
    return out
def pool(nw=None):
    global _POOL
    if _POOL is None:
        n = nw or _NW
        try:
            ctx = mp.get_context("fork")
            _POOL = ctx.Pool(n)
        except Exception:  # noqa: BLE001
            _POOL = False
    return _POOL or None
def distances(q, recs, text, px, nw=None, batch=None):
    items = [(f, i, text, px, sw, sk) for f, i, sw, sk in recs]
    p = pool(nw)
    if p is None or len(items) < 24:
        return _work((q["v"], q["ar"], items))
    n = nw or _NW
    b = batch or max(4, (len(items) + n - 1) // n)
    tasks = [(q["v"], q["ar"], items[i:i + b]) for i in range(0, len(items), b)]
    out = []
    for r in p.map(tasks and _work or _work, tasks):
        out.extend(r)
    return out
def shutdown():
    global _POOL
    if _POOL:
        _POOL.terminate()
        _POOL = None
