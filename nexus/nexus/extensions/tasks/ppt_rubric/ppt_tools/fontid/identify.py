#!/usr/bin/env python3
"字体识别核心：给定**一块墨迹**与**那块墨迹上的文字**，返回候选字体。"
import json
import os
import re
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glyphsig as G  # noqa: E402
import bench_v3 as V3  # noqa: E402
_IDX = None
_CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯豈-﫿]")
def load_index(path=None):
    global _IDX
    if _IDX is None:
        p = path or os.path.join(HERE, "font_index2.json")
        if not os.path.exists(p):
            p = os.path.join(HERE, "font_index.json")
        _IDX = json.load(open(p))
        for sc, d in _IDX.items():
            by = {}
            for r in d["records"]:
                r.setdefault("bold", False)
                r.setdefault("italic", False)
                r.setdefault("stroke", 0)
                r.setdefault("skew", 0.0)
                by.setdefault(r["class"], []).append(r)
            d["by_class"] = by
    return _IDX
def gold_class(script, name, bold=False, italic=False):
    d = load_index().get(script) or {}
    g = d.get("gold_class") or {}
    return g.get(f"{name}|{int(bool(bold))}{int(bool(italic))}")
def script_of(text):
    return "cjk" if _CJK_RE.search(text or "") else "lat"
def candidates(script, text):
    idx = load_index()
    d = idx.get(script) or idx["lat"]
    out = []
    for cid, recs in d["by_class"].items():
        r = min(recs, key=lambda x: x["cal_dist"])
        if not G.covers(r["file"], r["index"], text):
            continue
        al = sorted({(x["family"] + (" Bold" if x.get("bold") else "")
                      + (" Italic" if x.get("italic") else "")) for x in recs})
        out.append((cid, r, al))
    return out
def identify(ink, text, topn=5, script=None, blur=0.0, parallel=True, px=None):
    t0 = time.time()
    if ink is None or ink.size == 0 or not (text or "").strip():
        return [], {"ok": False, "reason": "empty ink or text"}
    sc = script or script_of(text)
    q = V3.n_stretch(ink, blur)
    cands = candidates(sc, text)
    if not cands:
        return [], {"ok": False, "reason": f"no candidate covers the text ({sc})"}
    px = px or G.RENDER_PX
    if parallel:
        import parallel as PAR
        ds = PAR.distances(q, [(r["file"], r["index"], r.get("stroke", 0),
                                r.get("skew", 0.0)) for _c, r, _a in cands],
                           text, px)
    else:
        ds = []
        for _cid, r, _al in cands:
            rink = G.render_text(r["file"], r["index"], text, px,
                                 stroke=r.get("stroke", 0), skew=r.get("skew", 0.0))
            ds.append(None if rink is None else V3.dist(q, V3.n_stretch(rink, 0.0)))
    scored = [(d, cid, r, al) for d, (cid, r, al) in zip(ds, cands) if d is not None]
    if not scored:
        return [], {"ok": False, "reason": "no candidate rendered"}
    scored.sort(key=lambda t: t[0])
    ds2 = np.array([s[0] for s in scored], np.float64)
    d1 = ds2[0]
    d2 = ds2[1] if len(ds2) > 1 else d1 * 2
    margin = float((d2 - d1) / max(1e-6, d2))
    out = []
    for d, cid, r, aliases in scored[:topn]:
        out.append({"family": r["family"], "aliases": sorted(set(aliases)),
                    "bold": bool(r.get("bold")), "italic": bool(r.get("italic")),
                    "dist": round(float(d), 4),
                    "rel": round(float(d / max(1e-6, d1)), 3),
                    "class": cid, "file": os.path.basename(r["file"]),
                    "styles": r["styles"]})
    meta = {"ok": True, "script": sc, "n_candidates": len(scored),
            "margin": round(margin, 4), "best_dist": round(float(d1), 4),
            "ms": round((time.time() - t0) * 1000, 1)}
    return out, meta
def identify_region(arr, box, text, **kw):
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    H, W = arr.shape[:2]
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return [], {"ok": False, "reason": "region too small"}
    ink = G.ink_mask(arr[y1:y2, x1:x2])
    return identify(ink, text, **kw)
def warmup():
    idx = load_index()
    n = 0
    for sc in ("lat", "cjk"):
        for _cid, recs in idx[sc]["by_class"].items():
            r = min(recs, key=lambda x: x["cal_dist"])
            G.cmap_of(r["file"], r["index"])
            n += 1
    G.save_cmap_cache()
    return n
