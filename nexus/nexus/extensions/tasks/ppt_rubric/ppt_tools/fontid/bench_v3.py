#!/usr/bin/env python3
"v3 标定：解决**累积字距漂移**。四种归一化/对齐策略对拍。"
import argparse
import hashlib
import json
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog  # noqa: E402
import glyphsig as G  # noqa: E402
import lo_probe  # noqa: E402
from bench_render import pick_faces  # noqa: E402
H0 = 40
W0 = 320
def n_fix(ink, blur):
    v, ar = G.normalise(ink, h0=H0, w0=W0 * 2, blur=blur)
    return {"v": v, "ar": ar, "mode": "fix"}
def n_stretch(ink, blur):
    h, w = ink.shape
    im = Image.fromarray((ink * 255).astype(np.uint8)).resize((W0, H0), Image.LANCZOS)
    v = np.asarray(im).astype(np.float32) / 255.0
    if blur:
        v = ndimage.gaussian_filter(v, blur)
    return {"v": v, "ar": w / float(h), "mode": "stretch"}
def _colprof(ink, n=256):
    c = ink.sum(axis=0).astype(np.float32)
    if c.sum() <= 0:
        return None
    x = np.linspace(0, len(c) - 1, n)
    return np.interp(x, np.arange(len(c)), c) / max(1e-6, c.max())
def n_dtw(ink, blur):
    d = n_stretch(ink, blur)
    d["cp"] = _colprof(ink)
    d["mode"] = "dtw"
    return d
def _split_glyphs(ink, min_gap=1):
    c = ink.sum(axis=0)
    on = c > (0.02 * max(1e-6, c.max()))
    idx = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            idx.append((i, j))
            i = j
        else:
            i += 1
    return idx
def n_glyph(ink, blur, nb=10, gw=28):
    segs = _split_glyphs(ink)
    if not segs:
        return {"v": None, "ar": 1.0, "mode": "glyph"}
    out = np.zeros((H0, gw * nb), np.float32)
    ars = []
    for k, (a, b) in enumerate(segs[:nb]):
        sub = ink[:, a:b]
        ys = np.nonzero(sub.sum(axis=1) > 0)[0]
        if len(ys) == 0:
            continue
        sub = sub[ys.min():ys.max() + 1]
        ars.append(sub.shape[1] / float(sub.shape[0]))
        im = Image.fromarray((sub * 255).astype(np.uint8)).resize((gw, H0), Image.LANCZOS)
        v = np.asarray(im).astype(np.float32) / 255.0
        if blur:
            v = ndimage.gaussian_filter(v, blur)
        out[:, k * gw:(k + 1) * gw] = v
    return {"v": out, "ar": float(np.mean(ars)) if ars else 1.0,
            "mode": "glyph", "nseg": len(segs)}
NORM = {"fix": n_fix, "stretch": n_stretch, "dtw": n_dtw, "glyph": n_glyph}
def _dtw_dist(a, b, band=24):
    n, m = len(a), len(b)
    INF = 1e9
    prev = np.full(m + 1, INF, np.float32); prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF, np.float32)
        j0 = max(1, i - band); j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            c = abs(a[i - 1] - b[j - 1])
            cur[j] = c + min(prev[j], prev[j - 1], cur[j - 1])
        prev = cur
    return float(prev[m]) / max(n, m)
def dist(d1, d2, w_ar=0.35):
    if d1.get("v") is None or d2.get("v") is None:
        return None
    mode = d1["mode"]
    if mode == "fix":
        return G.distance(d1["v"], d1["ar"], d2["v"], d2["ar"], w_ar=w_ar)
    base = float(np.abs(d1["v"] - d2["v"]).mean())
    if mode == "dtw" and d1.get("cp") is not None and d2.get("cp") is not None:
        base = 0.5 * base + 0.5 * _dtw_dist(d1["cp"][::4], d2["cp"][::4])
    return base + w_ar * abs(np.log(max(1e-6, d1["ar"]) / max(1e-6, d2["ar"])))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--script", default="lat")
    ap.add_argument("--work", default=os.path.join(HERE, "_bench"))
    ap.add_argument("--modes", default="fix,stretch,dtw,glyph")
    ap.add_argument("--blurs", default="0,0.6,1.2")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    faces, txt = pick_faces(a.n, a.script)
    names = [f["families"][0] for f in faces]
    _s, png = lo_probe.probe(names, txt, a.work, f"b3_{a.script}")
    im = np.asarray(Image.open(png).convert("L"))
    step = im.shape[0] / float(len(names))
    lo, cls = {}, {}
    for i, nm in enumerate(names):
        ink = G.ink_mask(im[int(round(i * step)):int(round((i + 1) * step))])
        if ink is None:
            continue
        lo[nm] = ink
        cls[nm] = hashlib.md5(np.ascontiguousarray(ink > 0.25).tobytes()
                              + str(ink.shape).encode()).hexdigest()
    pil = {}
    for f in faces:
        nm = f["families"][0]
        if nm in lo:
            ink = G.render_text(f["file"], f["index"], txt)
            if ink is not None:
                pil[nm] = ink
    names = [n for n in names if n in lo and n in pil]
    nclass = len({cls[n] for n in names})
    print(f"{len(names)} names, {nclass} distinct LO renderings, "
          f"random top1 = {1.0/nclass:.2%}", flush=True)
    res = {}
    for mode in a.modes.split(","):
        for blur in [float(x) for x in a.blurs.split(",")]:
            fn = NORM[mode]
            L = {n: fn(lo[n], blur) for n in names}
            P = {n: fn(pil[n], blur) for n in names}
            self_d, cross_d = [], []
            r1 = r5 = 0
            for n in names:
                cand = []
                for m in names:
                    d = dist(L[n], P[m])
                    if d is None:
                        continue
                    cand.append((d, m))
                    if m == n:
                        self_d.append(d)
                    elif cls[m] != cls[n]:
                        cross_d.append(d)
                cand.sort()
                if cand and cls[cand[0][1]] == cls[n]:
                    r1 += 1
                if any(cls[m] == cls[n] for _d, m in cand[:5]):
                    r5 += 1
            s = float(np.median(self_d)); c = float(np.median(cross_d))
            k = f"{mode}@{blur}"
            res[k] = {"sep": c / s if s else None, "self_med": s, "cross_med": c,
                      "self_p90": float(np.percentile(self_d, 90)),
                      "cross_p10": float(np.percentile(cross_d, 10)),
                      "top1": r1 / len(names), "top5": r5 / len(names)}
            r = res[k]
            print(f"  {k:14s} sep={r['sep']:.2f}  self p90={r['self_p90']:.4f} "
                  f"cross p10={r['cross_p10']:.4f}  top1={r['top1']:.1%} "
                  f"top5={r['top5']:.1%}", flush=True)
    best = max(res.items(), key=lambda kv: (kv[1]["top1"], kv[1]["sep"]))
    print(f"\nBEST: {best[0]}  top1={best[1]['top1']:.1%} top5={best[1]['top5']:.1%} "
          f"sep={best[1]['sep']:.2f}   (random {1.0/nclass:.2%})", flush=True)
    json.dump({"n_names": len(names), "n_class": nclass, "res": res,
               "best": best[0]}, open(os.path.join(HERE, f"bench_v3_{a.script}.json"), "w"),
              indent=1)
