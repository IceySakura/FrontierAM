#!/usr/bin/env python3
"字形签名与比对：**把一段文字的墨迹变成一个可比的数值向量**。"
import hashlib
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
H0 = 40
W0 = 480
BLUR = 1.2
RENDER_PX = 40
def ink_mask(arr, thresh=None):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 8:
        return None
    hist, edges = np.histogram(a, bins=32, range=(lo, hi))
    bg = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    dev = np.abs(a - bg)
    ink = dev / max(1e-6, dev.max())
    m = ink > 0.25
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
def normalise(ink, h0=H0, w0=W0, blur=BLUR):
    if ink is None or ink.size == 0:
        return None, None
    h, w = ink.shape
    ar = w / float(h)
    nw = max(1, min(w0, int(round(ar * h0))))
    im = Image.fromarray((ink * 255).astype(np.uint8)).resize((nw, h0), Image.LANCZOS)
    v = np.asarray(im).astype(np.float32) / 255.0
    if blur:
        v = ndimage.gaussian_filter(v, blur)
    out = np.zeros((h0, w0), np.float32)
    out[:, :nw] = v
    return out, ar
def signature(arr, **kw):
    ink = ink_mask(arr)
    return normalise(ink, **kw)
_FT_CACHE = {}
def load_face(path, index=0, px=RENDER_PX):
    k = (path, index, px)
    f = _FT_CACHE.get(k)
    if f is None:
        try:
            f = ImageFont.truetype(path, px, index=index)
        except Exception:  # noqa: BLE001
            f = False
        _FT_CACHE[k] = f
    return f or None
def render_text(path, index, text, px=RENDER_PX, stroke=0, skew=0.0):
    ft = load_face(path, index, px)
    if ft is None:
        return None
    try:
        bb = ft.getbbox(text)
    except Exception:  # noqa: BLE001
        bb = None
    pad = max(2, px // 8) + 2 * int(stroke)
    if bb and bb[2] > bb[0] and bb[3] > bb[1]:
        w = int(bb[2] - bb[0]) + 2 * pad
        h = int(bb[3] - bb[1]) + 2 * pad
        ox, oy = pad - int(bb[0]), pad - int(bb[1])
    else:
        w = 2 * pad + int(px * 1.3 * max(1, len(text)))
        h = px * 3
        ox, oy = pad, px
    if skew:
        w += int(abs(skew) * h) + 2
    if w <= 0 or h <= 0 or w > 60000:
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
    a = 255 - np.asarray(im, dtype=np.uint8)
    m = a > 63
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return sub.astype(np.float32) / 255.0
_CMAP_CACHE = {}
_CMAP_DISK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmap_cache.pkl")
_CMAP_LOADED = False
_CMAP_DIRTY = False
def _load_disk_cache():
    global _CMAP_LOADED
    if _CMAP_LOADED:
        return
    _CMAP_LOADED = True
    try:
        import pickle
        with open(_CMAP_DISK, "rb") as f:
            for k, spans in pickle.load(f).items():
                s = set()
                for a, b in spans:
                    s.update(range(a, b + 1))
                _CMAP_CACHE[k] = s
    except Exception:  # noqa: BLE001
        pass
def save_cmap_cache():
    if not _CMAP_DIRTY:
        return
    try:
        import pickle
        out = {}
        for k, s in _CMAP_CACHE.items():
            spans, start, prev = [], None, None
            for c in sorted(s):
                if start is None:
                    start = prev = c
                elif c == prev + 1:
                    prev = c
                else:
                    spans.append((start, prev)); start = prev = c
            if start is not None:
                spans.append((start, prev))
            out[k] = spans
        tmp = _CMAP_DISK + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=4)
        os.replace(tmp, _CMAP_DISK)
    except Exception:  # noqa: BLE001
        pass
def cmap_of(path, index=0):
    global _CMAP_DIRTY
    _load_disk_cache()
    k = (path, index)
    c = _CMAP_CACHE.get(k)
    if c is None:
        try:
            from fontTools.ttLib import TTFont
            kw = {"fontNumber": index} if path.lower().endswith((".ttc", ".otc")) else {}
            tt = TTFont(path, lazy=True, **kw)
            c = set(tt.getBestCmap().keys())
            tt.close()
        except Exception:  # noqa: BLE001
            c = set()
        _CMAP_CACHE[k] = c
        _CMAP_DIRTY = True
    return c
def covers(path, index, text):
    need = {ord(ch) for ch in text if not ch.isspace()}
    if not need:
        return False
    if load_face(path, index) is None:
        return False
    cm = cmap_of(path, index)
    return bool(cm) and need.issubset(cm)
def distance(v1, ar1, v2, ar2, w_ar=0.35):
    if v1 is None or v2 is None:
        return None
    n1 = int(round(min(W0, ar1 * H0)))
    n2 = int(round(min(W0, ar2 * H0)))
    n = max(1, min(n1, n2))
    a = v1[:, :n]; b = v2[:, :n]
    shape = float(np.abs(a - b).mean())
    extra = 0.0
    if n1 != n2:
        wide, nw = (v1, n1) if n1 > n2 else (v2, n2)
        extra = float(wide[:, n:nw].mean()) * (nw - n) / max(1, nw)
    ar = abs(np.log(max(1e-6, ar1) / max(1e-6, ar2)))
    return shape + extra + w_ar * ar
