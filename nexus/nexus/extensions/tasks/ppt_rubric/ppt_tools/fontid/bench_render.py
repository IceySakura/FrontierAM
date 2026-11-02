#!/usr/bin/env python3
"标定实验：**PIL/FreeType 渲染能不能代替 LibreOffice 渲染做比对？**"
import argparse
import collections
import json
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog  # noqa: E402
import glyphsig as G  # noqa: E402
import lo_probe  # noqa: E402
from PIL import Image  # noqa: E402
def pick_faces(n, script):
    txt = lo_probe.PROBE_CJK if script == "cjk" else lo_probe.PROBE_LAT
    out, seen = [], set()
    for f in catalog.load():
        if f["format"] == "Type 1":
            continue
        fam = f["families"][0]
        if fam.lower() in seen:
            continue
        if not G.covers(f["file"], f["index"], txt):
            continue
        seen.add(fam.lower())
        out.append(f)
        if len(out) >= n:
            break
    return out, txt
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--script", default="lat")
    ap.add_argument("--work", default=os.path.join(HERE, "_bench"))
    ap.add_argument("--blurs", default="0,0.6,1.2,2.0,3.0,4.5")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    faces, txt = pick_faces(a.n, a.script)
    print(f"{len(faces)} faces cover the probe text ({a.script})", flush=True)
    names = [f["families"][0] for f in faces]
    sigs, png = lo_probe.probe(names, txt, a.work, "bench_" + a.script)
    im = np.asarray(Image.open(png).convert("L"))
    H = im.shape[0]
    step = H / float(len(names))
    lo_ink = {}
    for i, n in enumerate(names):
        sub = im[int(round(i * step)):int(round((i + 1) * step))]
        lo_ink[n] = G.ink_mask(sub)
    names = [n for n in names if lo_ink.get(n) is not None]
    faces = [f for f in faces if f["families"][0] in names]
    print(f"{len(names)} rendered non-blank by LO", flush=True)
    import hashlib
    cls = {}
    for n in names:
        cls[n] = hashlib.md5(np.ascontiguousarray(
            (lo_ink[n] > 0.25)).tobytes() + str(lo_ink[n].shape).encode()).hexdigest()
    ndistinct = len(set(cls.values()))
    print(f"{ndistinct} distinct LO renderings among {len(names)} names", flush=True)
    pil_ink = {}
    for f in faces:
        pil_ink[f["families"][0]] = G.render_text(f["file"], f["index"], txt)
    names = [n for n in names if pil_ink.get(n) is not None]
    print(f"{len(names)} also rendered by PIL", flush=True)
    res = {}
    for blur in [float(x) for x in a.blurs.split(",")]:
        lo_s = {n: G.normalise(lo_ink[n], blur=blur) for n in names}
        pil_s = {n: G.normalise(pil_ink[n], blur=blur) for n in names}
        self_d, cross_d, same_pil = [], [], []
        rank1 = 0; rank5 = 0; n_eval = 0
        for n in names:
            v1, a1 = lo_s[n]
            cand = []
            for m in names:
                v2, a2 = pil_s[m]
                d = G.distance(v1, a1, v2, a2)
                if d is None:
                    continue
                cand.append((d, m))
                if m == n:
                    self_d.append(d)
                elif cls[m] != cls[n]:
                    cross_d.append(d)
            cand.sort()
            n_eval += 1
            if cand and cls[cand[0][1]] == cls[n]:
                rank1 += 1
            if any(cls[m] == cls[n] for _d, m in cand[:5]):
                rank5 += 1
        for n in names[:25]:
            v1, a1 = pil_s[n]
            for m in names[:25]:
                if m != n and cls[m] != cls[n]:
                    d = G.distance(v1, a1, *pil_s[m])
                    if d is not None:
                        same_pil.append(d)
        s_med = float(np.median(self_d)) if self_d else float("nan")
        c_med = float(np.median(cross_d)) if cross_d else float("nan")
        p_med = float(np.median(same_pil)) if same_pil else float("nan")
        res[blur] = {"self_med": s_med, "cross_med": c_med, "pil_cross_med": p_med,
                     "separation": c_med / s_med if s_med else None,
                     "top1": rank1 / max(1, n_eval), "top5": rank5 / max(1, n_eval),
                     "n_eval": n_eval, "self_p90": float(np.percentile(self_d, 90)),
                     "cross_p10": float(np.percentile(cross_d, 10))}
        r = res[blur]
        print(f"blur={blur:4.1f}  self={s_med:.4f}(p90 {r['self_p90']:.4f})  "
              f"cross={c_med:.4f}(p10 {r['cross_p10']:.4f})  sep={r['separation']:.2f}  "
              f"top1={r['top1']:.1%} top5={r['top5']:.1%}", flush=True)
    json.dump({"script": a.script, "n_names": len(names), "n_classes": ndistinct,
               "results": {str(k): v for k, v in res.items()}},
              open(os.path.join(HERE, f"bench_render_{a.script}.json"), "w"), indent=1)
