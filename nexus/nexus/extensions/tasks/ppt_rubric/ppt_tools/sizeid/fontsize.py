#!/usr/bin/env python3
"从截图的**墨迹行框**反推字号（pt）。核心算法，无状态。"
import os
import re
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fontid"))
import glyphsig as G  # noqa: E402
CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯豈-﫿]")
_ASCII_RE = re.compile(r"[0-9A-Za-z]")
SMIN, SMAX, STEP = 4.0, 600.0, 0.5
_GRID = np.arange(SMIN, SMAX + 1e-9, STEP)
INK_COV = 28.0 / 255.0
COARSE_PX = 48
PANEL = {
    "cjk": ["Noto Sans CJK SC", "Noto Serif CJK SC", "Arial Unicode MS",
            "Microsoft YaHei"],
    "lat": ["Arial", "Times New Roman", "Verdana", "Georgia", "DejaVu Sans"],
}
_FACES = None
_BY_FAMILY = None
_W_REGULAR, _W_BOLD = 80.0, 200.0
def _load_faces():
    global _FACES, _BY_FAMILY
    if _FACES is not None:
        return
    import json
    p = os.path.join(os.path.dirname(HERE), "fontid", "installed_faces.json")
    _FACES = json.load(open(p, encoding="utf-8"))
    _BY_FAMILY = {}
    for f in _FACES:
        for fam in f.get("families") or []:
            _BY_FAMILY.setdefault(fam.strip().lower(), []).append(f)
def _face_cost(face, want_w, want_slant, query):
    w = face.get("weight")
    w = _W_REGULAR if w is None else float(w)
    sl = float(face.get("slant") or 0.0)
    wd = face.get("width")
    wd = 100.0 if wd is None else float(wd)
    cost = abs(w - want_w) / 50.0
    cost += 4.0 if (sl > 0) != (want_slant > 0) else 0.0
    cost += abs(wd - 100.0) / 25.0
    fams = [x.strip().lower() for x in (face.get("families") or [])]
    if query not in fams:
        cost += 8.0
    st = " ".join(face.get("styles") or []).lower()
    if want_w == _W_REGULAR and ("regular" in st or "book" in st):
        cost -= 0.01
    return cost
def resolve(family, bold=False, italic=False):
    _load_faces()
    q = (family or "").strip().lower()
    cands = _BY_FAMILY.get(q)
    if not cands:
        return None
    want_w = _W_BOLD if bold else _W_REGULAR
    want_sl = 100.0 if italic else 0.0
    best = min(cands, key=lambda f: _face_cost(f, want_w, want_sl, q))
    bw = best.get("weight")
    bw = _W_REGULAR if bw is None else float(bw)
    bsl = float(best.get("slant") or 0.0)
    stroke = 1 if (bold and bw < 150) else 0
    skew = 0.25 if (italic and bsl <= 0) else 0.0
    return best["file"], int(best.get("index") or 0), stroke, skew
def script_of(text):
    return "cjk" if CJK_RE.search(text or "") else "lat"
def kind_of(text):
    c = bool(CJK_RE.search(text or ""))
    a = bool(_ASCII_RE.search(text or ""))
    return "mixed" if (c and a) else ("cjk" if c else "lat")
_HW_CACHE = {}
def _render_raw(path, index, text, px, stroke=0, skew=0.0):
    from PIL import Image, ImageDraw
    ft = G.load_face(path, index, px)
    if ft is None:
        return None
    try:
        bb = ft.getbbox(text)
    except Exception:  # noqa: BLE001
        return None
    if not (bb and bb[2] > bb[0] and bb[3] > bb[1]):
        return None
    pad = max(4, px // 4) + 2 * int(stroke)
    w = int(bb[2] - bb[0]) + 2 * pad
    h = int(bb[3] - bb[1]) + 2 * pad
    ox, oy = pad - int(bb[0]), pad - int(bb[1])
    if skew:
        w += int(abs(skew) * h) + 2
    if w <= 0 or h <= 0 or w > 80000:
        return None
    im = Image.new("L", (w, h), 255)
    try:
        if stroke:
            ImageDraw.Draw(im).text((ox, oy), text, font=ft, fill=0,
                                    stroke_width=int(stroke), stroke_fill=0)
        else:
            ImageDraw.Draw(im).text((ox, oy), text, font=ft, fill=0)
    except Exception:  # noqa: BLE001
        return None
    if skew:
        im = im.transform((w, h), Image.AFFINE, (1, skew, -skew * h, 0, 1, 0),
                          resample=Image.BILINEAR, fillcolor=255)
    return (255 - np.asarray(im, dtype=np.uint8)).astype(np.float32) / 255.0
def ink_hw(path, index, text, px, stroke=0, skew=0.0, cov=None):
    cov = INK_COV if cov is None else cov
    k = (path, index, text, int(px), stroke, round(skew, 3), round(cov, 4))
    v = _HW_CACHE.get(k)
    if v is None:
        a = _render_raw(path, index, text, int(px), stroke, skew)
        if a is None:
            v = (0, 0)
        else:
            m = a > cov
            if not m.any():
                v = (0, 0)
            else:
                ys, xs = np.nonzero(m)
                v = (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        _HW_CACHE[k] = v
    return v
def _solve_face(h_px, w_px, text, ratio, path, index, stroke, skew, cov):
    memo = {}
    def f(i):
        v = memo.get(i)
        if v is None:
            px = max(4, int(round(float(_GRID[i]) * ratio)))
            v = ink_hw(path, index, text, px, stroke, skew, cov)
            memo[i] = v
        return v
    n = len(_GRID)
    H0, W0 = ink_hw(path, index, text, COARSE_PX, stroke, skew, cov)
    if H0 <= 0:
        return None
    s0 = h_px / H0 * COARSE_PX / ratio
    i0 = int(round((min(max(s0, SMIN), SMAX) - SMIN) / STEP))
    i0 = max(0, min(n - 1, i0))
    def bracket(target, d):
        lo = hi = i0
        step = max(2, int(round(i0 * 0.10)) or 2)
        for _ in range(24):
            lo = max(0, lo - step)
            hi = min(n - 1, hi + step)
            if f(lo)[d] <= target <= f(hi)[d]:
                return lo, hi
            if lo == 0 and hi == n - 1:
                break
            step *= 2
        return lo, hi
    def first_ge(target, d, lo, hi):
        while lo < hi:
            mid = (lo + hi) // 2
            if f(mid)[d] >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo
    th = int(round(h_px))
    lo, hi = bracket(th, 0)
    if f(lo)[0] >= th:
        hm = hlo = hhi = float(_GRID[lo])
    elif f(hi)[0] <= th:
        hm = hlo = hhi = float(_GRID[hi])
    else:
        a = first_ge(th, 0, lo, hi)
        if f(a)[0] == th:
            b = first_ge(th + 1, 0, a, hi)
            hlo, hhi = float(_GRID[a]), float(_GRID[max(a, b - 1)])
            hm = (hlo + hhi) / 2.0
        else:
            v0, v1 = f(a - 1)[0], f(a)[0]
            t = (th - v0) / float(max(1, v1 - v0))
            hm = hlo = hhi = float(_GRID[a - 1]) + t * STEP
    wm = None
    if w_px and w_px > 0:
        tw = int(round(w_px))
        lo, hi = bracket(tw, 1)
        if f(lo)[1] >= tw:
            wm = float(_GRID[lo])
        elif f(hi)[1] <= tw:
            wm = float(_GRID[hi])
        else:
            a = first_ge(tw, 1, lo, hi)
            if f(a)[1] == tw:
                b = first_ge(tw + 1, 1, a, hi)
                wm = (float(_GRID[a]) + float(_GRID[max(a, b - 1)])) / 2.0
            else:
                v0, v1 = f(a - 1)[1], f(a)[1]
                t = (tw - v0) / float(max(1, v1 - v0))
                wm = float(_GRID[a - 1]) + t * STEP
    return hm, hlo, hhi, wm
def _solve(target, which, text, ratio, path, index, stroke, skew, cov):
    r = _solve_face(target if which == 0 else 0,
                    target if which == 1 else 0,
                    text, ratio, path, index, stroke, skew, cov)
    if r is None:
        return None, None, None
    if which == 0:
        return r[0], r[1], r[2]
    return r[3], r[3], r[3]
def _faces_for(text, font, bold, italic):
    out = []
    if font:
        r = resolve(font, bold, italic)
        if r and G.covers(r[0], r[1], text):
            out.append((font, ) + r)
    if out:
        return out, True
    for fam in PANEL[script_of(text)]:
        r = resolve(fam, bold, italic)
        if r and G.covers(r[0], r[1], text):
            out.append((fam, ) + r)
    if not out:
        _load_faces()
        for f in _FACES:
            if G.covers(f["file"], int(f.get("index") or 0), text):
                out.append(((f.get("families") or ["?"])[0], f["file"],
                            int(f.get("index") or 0), 0, 0.0))
                if len(out) >= 4:
                    break
    return out, False
_CONF_HIGH = 0.035
_CONF_MED = 0.09
_WIDTH_ONLY_TOL = 0.25
def estimate(h_px, w_px, text, ratio, font=None, bold=False, italic=False,
             cov=None, faces=None, exact_font=None):
    text = (text or "").strip()
    if not text or h_px <= 0 or ratio <= 0:
        return None
    if faces is None:
        faces, ef = _faces_for(text, font, bold, italic)
    else:
        ef = True
    if exact_font is not None:
        ef = bool(exact_font)
    exact_font = ef
    if not faces:
        return None
    est, est_h, est_w, used, plat = [], [], [], [], []
    kind = kind_of(text)
    width_only = (kind == "cjk") and (not exact_font)
    for label, path, idx, stroke, skew in faces:
        got = _solve_face(h_px, w_px, text, ratio, path, idx, stroke, skew, cov)
        if got is None:
            continue
        mh, lo_s, hi_s, mw = got
        est_h.append(mh)
        plat.append(hi_s - lo_s)
        used.append(label)
        s = mh
        if mw is not None:
            est_w.append(mw)
            if width_only:
                s = mw if abs(mw - mh) <= _WIDTH_ONLY_TOL * max(1e-6, mh) \
                    else min(max(mw, lo_s), hi_s)
            else:
                s = min(max(mw, lo_s), hi_s)
        est.append(s)
    if not est:
        return None
    pt = float(np.median(est))
    ph_ = float(np.median(est_h))
    pw_ = float(np.median(est_w)) if est_w else None
    spread = (max(est) - min(est)) / max(1e-6, pt) if len(est) > 1 else 0.0
    quant = float(np.median(plat)) / max(1e-6, pt) if plat else 0.0
    disagree = abs(ph_ - pw_) / max(1e-6, ph_) if pw_ else 0.0
    clamped = bool(pw_ is not None and not width_only
                   and abs(pt - pw_) > 0.05 * max(1e-6, pt))
    unc = max(spread, quant / 2.0, 0.005)
    if clamped:
        unc = max(unc, min(disagree, 0.30) * 0.5)
    if width_only:
        unc = max(unc, spread, 0.02, min(disagree, 0.30) * 0.5)
    if not exact_font:
        unc = max(unc, spread, 0.02)
    if pw_ is not None:
        _lo = min(pt * (1.0 - unc), ph_, pw_)
        _hi = max(pt * (1.0 + unc), ph_, pw_)
        unc = max(unc, max(pt - _lo, _hi - pt) / max(1e-6, pt))
    _at_edge = (pt <= SMIN + 1e-9) or (pt >= SMAX - 1e-9)
    if _at_edge:
        unc = max(unc, _CONF_MED + 1e-6)
    conf = "high" if unc <= _CONF_HIGH else ("medium" if unc <= _CONF_MED else "low")
    why = []
    if _at_edge:
        why.append(f"the measurement hit the edge of the search range "
                   f"({SMIN:g}-{SMAX:g} pt), so the true size is probably "
                   f"OUTSIDE that range and this number is only a boundary, "
                   f"not a measurement. Check that the box really holds one "
                   f"line of text: a box a few pixels tall usually means a "
                   f"rule, a border or an anti-aliasing edge was measured "
                   f"instead of glyphs, and a box in POINT coordinates passed "
                   f"where PIXELS were expected lands here too")
    if not exact_font:
        if width_only:
            why.append(f"font not given, but this line is all CJK -- every CJK "
                       f"typeface advances exactly one em per character, so the "
                       f"line's width pins the size down regardless "
                       f"(spread across {len(used)} reference typefaces "
                       f"{spread:.1%})")
        else:
            why.append(f"font not given -- glyph heights differ between "
                       f"typefaces, and across {len(used)} reference typefaces "
                       f"this line spans {spread:.1%}. This is usually the "
                       f"largest source of error for text containing Latin "
                       f"letters or digits: call identify_font and pass its "
                       f"answer in as `font`")
    if clamped:
        why.append(f"the line's width implies {pw_:.1f}pt but its height implies "
                   f"{ph_:.1f}pt, so the answer was held to what the height "
                   f"allows. A gap this size usually means the text you gave is "
                   f"not exactly what is on that line, or the line mixes Latin "
                   f"and CJK (the renderer inserts extra spacing at the "
                   f"boundary)")
        why.append(f"this report's range was widened to cover BOTH paths "
                   f"({min(ph_, pw_):.1f}-{max(ph_, pw_):.1f}pt) because the "
                   f"tool cannot tell which of the two inputs is at fault: the "
                   f"same size and direction of gap is produced both by a box "
                   f"whose top/bottom clipped the ink AND by `text` missing a "
                   f"character. To resolve it, re-check that the box's TOP and "
                   f"BOTTOM really touch the tallest and lowest ink of the line, "
                   f"and that `text` is character for character what is on it")
    if quant > 0.07:
        why.append(f"the line is only {h_px:.0f}px tall, so pixel rounding alone "
                   f"leaves a +-{quant/2:.0%} window")
    return {"pt": pt, "pt_h": ph_, "pt_w": pw_,
            "lo": pt * (1 - unc), "hi": pt * (1 + unc),
            "unc": unc, "conf": conf, "faces": used, "exact_font": exact_font,
            "why": why, "script": script_of(text), "kind": kind,
            "n_faces": len(used), "width_only": width_only, "clamped": clamped}
RATIO_CONST = {}
def _load_const():
    global RATIO_CONST
    if RATIO_CONST:
        return RATIO_CONST
    import json
    p = os.path.join(HERE, "ratio_const.json")
    if os.path.exists(p):
        RATIO_CONST = json.load(open(p))
    return RATIO_CONST
def estimate_no_text(h_px, ratio, script="cjk"):
    c = _load_const().get(script)
    if not c:
        return None
    h_pt = h_px / ratio
    return {"pt": h_pt / c["k50"], "lo": h_pt / c["k90"], "hi": h_pt / c["k10"],
            "unc": (c["k90"] - c["k10"]) / 2.0 / max(1e-6, c["k50"]),
            "conf": "low", "script": script, "pt_h": h_pt / c["k50"], "pt_w": None,
            "faces": [], "exact_font": False,
            "why": [f"no text was given; for {script} text the ink-height/size ratio "
                    f"spans {c['k10']:.2f}-{c['k90']:.2f} between the 10th and 90th "
                    f"percentile, so this is a range, not a measurement"]}
