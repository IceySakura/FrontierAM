#!/usr/bin/env python3
"测量 API：模型能**测**目标图，但拿不到**像素数据本身**。"
import collections
import io
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "bx"))
from text_blocks import list_text_blocks  # noqa: E402  ⛔ 原样复用，已标定
_MAX_POINTS = 64
_MAX_SAMPLES = 2000
_MAX_REGIONS = 16
_PATCH_MAX = 15
_MIN_SIDE = 2
_RATIO_SKEW_TOL = 0.02
class StateContractError(TypeError):
    "state 不满足契约时抛出。"
def make_state(image, page_width_pt, page_height_pt, ratio=None,
               strict=True, warn=None):
    if warn is None:
        def warn(msg):
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    img = _coerce_image(image)
    try:
        pw = float(page_width_pt)
        ph = float(page_height_pt)
    except (TypeError, ValueError):
        raise StateContractError(
            f"page_width_pt / page_height_pt must be numbers in POINTS, got "
            f"{page_width_pt!r} / {page_height_pt!r}.") from None
    if pw <= 0 or ph <= 0:
        raise StateContractError(
            f"page size must be positive, got {pw} x {ph} pt.")
    px_w, px_h = img.size
    r_w = px_w / pw
    r_h = px_h / ph
    skew = abs(r_w - r_h) / max(r_w, r_h)
    if skew > _RATIO_SKEW_TOL:
        msg = (f"screenshot aspect ratio does not match the slide: "
               f"{px_w}x{px_h}px over {pw}x{ph}pt gives {r_w:.4f} px/pt "
               f"horizontally but {r_h:.4f} vertically ({skew:.1%} apart). "
               f"The screenshot was probably scaled non-uniformly or the page "
               f"size is wrong; every point coordinate will be off.")
        if strict and ratio is None:
            raise StateContractError(msg + " Pass ratio=... explicitly to override.")
        warn(msg)
    if ratio is None:
        ratio = r_w
    else:
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            raise StateContractError(
                f"ratio must be a number (px per pt), got {ratio!r}.") from None
        if ratio <= 0:
            raise StateContractError(f"ratio must be positive, got {ratio}.")
    return {"reference": img,
            "ratio": ratio,
            "page_size": {"width": pw, "height": ph}}
def _coerce_image(image):
    from PIL import Image as _PILImage
    if isinstance(image, _PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            return _PILImage.open(io.BytesIO(bytes(image))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise StateContractError(
                f"could not decode the {len(bytes(image))} bytes given as "
                f"`image`: {e}") from None
    if isinstance(image, (str, os.PathLike)):
        p = os.fspath(image)
        if not os.path.exists(p):
            raise StateContractError(f"image file does not exist: {p!r}")
        try:
            return _PILImage.open(p).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise StateContractError(f"could not open image {p!r}: {e}") from None
    raise StateContractError(
        f"`image` must be a path (str / os.PathLike), a PIL.Image, or raw "
        f"bytes -- got {type(image).__name__}.")
def _check_state(state):
    if not isinstance(state, dict):
        return (f"state must be a dict, got {type(state).__name__}. "
                f"Build it with make_state(image, page_w_pt, page_h_pt).")
    if "reference" not in state or state.get("reference") is None:
        return ("state has no 'reference' image. "
                "Build the state with make_state(image, page_w_pt, page_h_pt).")
    ref = state["reference"]
    if state.get("_ref_arr") is None and not _is_pil_image(ref):
        return (f"reference must be a PIL.Image (got {type(ref).__name__}). "
                f"Use make_state(path, w, h) to build the state -- it accepts a "
                f"path, a PIL.Image or raw bytes and opens the image for you.")
    ps = state.get("page_size")
    if ps is not None and not isinstance(ps, dict):
        return (f"state['page_size'] must be a dict like "
                f"{{'width': 960.0, 'height': 540.0}} in points, got "
                f"{type(ps).__name__}.")
    return None
def _is_pil_image(obj):
    try:
        from PIL import Image as _PILImage
    except Exception:  # noqa: BLE001  PIL 缺失时退化成 duck typing
        return hasattr(obj, "convert") and hasattr(obj, "size")
    return isinstance(obj, _PILImage.Image)
def _contract_error(msg):
    return ([{"type": "text", "text": "Error: " + msg}],
            {"success": False, "contract_error": True})
def _img(state):
    ref = state.get("reference")
    if ref is None:
        return None
    arr = state.get("_ref_arr")
    if arr is None:
        arr = np.asarray(ref.convert("RGB"))
        state["_ref_arr"] = arr
    return arr
def _hex(rgb):
    return "#%02X%02X%02X" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
def _clip_box(box, W, H):
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x2 - x1 < _MIN_SIDE or y2 - y1 < _MIN_SIDE:
        return None
    return x1, y1, x2, y2
def _modal_colour(px):
    if px.size == 0:
        return None, 0.0
    q = px.astype(np.int32) // 16
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[np.argmax(counts)]
    frac = float(counts.max()) / len(keys)
    med = np.median(px[keys == k], axis=0)
    return _hex(med), frac
def _top_colours(px, k=4):
    if px.size == 0:
        return []
    q = px.astype(np.int32) // 16
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)[:k]
    out = []
    for i in order:
        sel = keys == vals[i]
        med = np.median(px[sel], axis=0)
        out.append((_hex(med), float(counts[i]) / len(keys)))
    return out
def t_page_info(state, args):
    err = _check_state(state)
    if err:
        return _contract_error(err)
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    H, W = arr.shape[:2]
    ratio = state.get("ratio") or 1.0
    ps = state.get("page_size") or {}
    corners = []
    for (y, x) in ((0, 0), (0, W - 12), (H - 12, 0), (H - 12, W - 12)):
        y, x = max(0, y), max(0, x)
        corners.append(arr[y:y + 12, x:x + 12].reshape(-1, 3))
    bg, bgfrac = _modal_colour(np.concatenate(corners, axis=0))
    whole, _ = _modal_colour(arr.reshape(-1, 3))
    txt = (f"page_info:\n"
           f"  screenshot   {W} x {H} px\n"
           f"  slide canvas {ps.get('width')} x {ps.get('height')} pt\n"
           f"  scale        1 pt = {ratio:.4f} px   (pt = px / {ratio:.4f})\n"
           f"  background   {bg}  (corner agreement {bgfrac:.0%})\n"
           f"  most common colour on the whole slide  {whole}")
    return ([{"type": "text", "text": txt}],
            {"success": True, "size_px": [W, H], "background": bg})
SPEC_PAGE_INFO = {
    "type": "function",
    "function": {
        "name": "page_info",
        "description": (
            "Canvas facts measured from the screenshot: pixel size, slide size in points, "
            "the px-per-pt scale, and the background colour. Takes no arguments."),
        "parameters": {"type": "object", "properties": {}},
    },
}
def t_text_blocks(state, args):
    err = _check_state(state)
    if err:
        return _contract_error(err)
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    ratio = state.get("ratio") or 1.0
    try:
        r = list_text_blocks(arr, ratio=ratio)
    except Exception:  # noqa: BLE001
        import traceback
        return ([{"type": "text",
                  "text": "Error: the text scan failed.\n" + traceback.format_exc()[-400:]}],
                {"success": False})
    if not r.get("ok"):
        return ([{"type": "text", "text": f"No text blocks found: {r.get('reason')}"}],
                {"success": False})
    rows, values, n = [], [], 0
    for b in r["blocks"]:
        for i, ln in enumerate(zip(b["lines_pt"], b["lines_px"])):
            lp, lx = ln
            n += 1
            rows.append(f"  {n:3d}. x={lp[0]:.1f} y={lp[1]:.1f} "
                        f"w={lp[2]-lp[0]:.1f} h={lp[3]-lp[1]:.1f} pt   colour {b['colour']}"
                        + f"\n        box in pixels [{int(round(lx[0]))}, "
                          f"{int(round(lx[1]))}, {int(round(lx[2]))}, "
                          f"{int(round(lx[3]))}]  <- pass this to other tools"
                        + (f"   (part {i+1}/{len(b['lines_pt'])} of one block)"
                           if len(b["lines_pt"]) > 1 else ""))
            values.append({"x": round(lp[0], 1), "y": round(lp[1], 1),
                           "w": round(lp[2] - lp[0], 1), "h": round(lp[3] - lp[1], 1),
                           "colour": b["colour"],
                           "box_px": [int(round(v)) for v in lx]})
    txt = (f"text_blocks: {n} text line(s).\n"
           f"Each line gives TWO forms of the same box:\n"
           f"  - x/y/w/h in POINTS -- the unit you write into python-pptx "
           f"(Inches/Pt), origin top-left;\n"
           f"  - '[x1, y1, x2, y2] in pixels' -- the unit EVERY other tool "
           f"takes (`estimate_font_size`, `identify_font`, `region_stats`, "
           f"`find_edges`, `sample_colour`). Copy the pixel box when calling "
           f"them; do not pass the point values, and do not convert by hand.\n"
           + "\n".join(rows)
           + "\n\nThese are glyph boxes measured from the pixels. Some rows may be "
             "lettering inside a photo or decorative artwork rather than slide text -- "
             "use the screenshot to decide which to keep.")
    state.setdefault("tool_values", []).extend(values)
    return [{"type": "text", "text": txt}], {"success": True, "n_lines": n, "values": values}
SPEC_TEXT_BLOCKS = {
    "type": "function",
    "function": {
        "name": "text_blocks",
        "description": (
            "List every text line on the slide, already measured: bounding box plus the "
            "text colour. Takes no arguments; it scans the whole screenshot.\n\n"
            "Each line is reported twice: as x/y/w/h in POINTS (what you write into "
            "python-pptx) and as [x1, y1, x2, y2] in PIXELS (what every other tool "
            "takes as input). Pass the pixel box straight to `estimate_font_size`, "
            "`identify_font`, `region_stats`, `find_edges` or `sample_colour` -- never "
            "the point values, and never a box you converted yourself.\n\n"
            "You do not need to pick a region or measure text yourself. Boxes are computed "
            "deterministically from the pixels and are accurate to about 2 points.\n\n"
            "The list may include rows that are not slide text (lettering baked into a photo, "
            "decorative artwork, chart labels). Use the screenshot to decide which are real. "
            "It reports the glyph box, tight around the letters, not an invisible text frame."),
        "parameters": {"type": "object", "properties": {}},
    },
}
def t_sample_colour(state, args):
    err = _check_state(state)
    if err:
        return _contract_error(err)
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    H, W = arr.shape[:2]
    pts = args.get("points") or []
    if not isinstance(pts, list) or not pts:
        return ([{"type": "text",
                  "text": "Error: `points` must be a non-empty list of [x, y] pixel pairs."}],
                {"success": False})
    patch = int(args.get("patch") or 3)
    patch = max(1, min(patch, _PATCH_MAX))
    used = state.get("_n_samples", 0)
    if used >= _MAX_SAMPLES:
        return ([{"type": "text",
                  "text": f"Error: sampling budget exhausted ({_MAX_SAMPLES} points). "
                          f"Use text_blocks / region_stats, which summarise instead of "
                          f"sampling point by point."}],
                {"success": False, "budget_exhausted": True})
    _n_req = len(pts)
    _dropped_cap = max(0, _n_req - _MAX_POINTS)
    if len(pts) > _MAX_POINTS:
        pts = pts[:_MAX_POINTS]
    room = _MAX_SAMPLES - used
    _dropped_budget = max(0, len(pts) - room)
    if len(pts) > room:
        pts = pts[:room]
    rows, values, bad = [], [], 0
    for p in pts:
        try:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
        except (TypeError, ValueError, IndexError):
            bad += 1
            continue
        if not (0 <= x < W and 0 <= y < H):
            rows.append(f"  ({p[0]}, {p[1]}) -> outside the {W}x{H} image")
            bad += 1
            continue
        h = patch // 2
        y1, y2 = max(0, y - h), min(H, y + h + 1)
        x1, x2 = max(0, x - h), min(W, x + h + 1)
        px = arr[y1:y2, x1:x2].reshape(-1, 3)
        exact = arr[y, x]
        mod, frac = _modal_colour(px)
        uniform = frac >= 0.999 and patch > 1
        rows.append(f"  ({x:4d}, {y:4d})  {_hex(exact)}"
                    + (f"   [{patch}x{patch} patch: {mod}, {frac:.0%} of it]"
                       if patch > 1 else "")
                    + ("   uniform" if uniform else ""))
        values.append({"x": x, "y": y, "hex": _hex(exact),
                       "rgb": [int(v) for v in exact],
                       "patch_modal": mod, "patch_frac": round(frac, 3)})
    state["_n_samples"] = used + len(values)
    left = _MAX_SAMPLES - state["_n_samples"]
    txt = (f"sample_colour: {len(values)} point(s) sampled"
           + (f", {bad} skipped" if bad else "") + f". Budget left {left}.\n"
           + "\n".join(rows))
    if _dropped_cap or _dropped_budget:
        txt += (f"\n\n[You asked for {_n_req} points but "
                f"{_dropped_cap + _dropped_budget} were NOT sampled: "
                + (f"{_dropped_cap} over the {_MAX_POINTS}-per-call limit"
                   if _dropped_cap else "")
                + ("; " if _dropped_cap and _dropped_budget else "")
                + (f"{_dropped_budget} over the remaining "
                   f"{_MAX_SAMPLES}-point budget" if _dropped_budget else "")
                + ". The list above is INCOMPLETE -- the missing points were "
                  "dropped silently in earlier versions, so re-ask for them in "
                  "another call rather than assuming they matched.]")
    state.setdefault("tool_values", []).extend(values)
    return ([{"type": "text", "text": txt}],
            {"success": bool(values), "n": len(values), "values": values,
             "n_requested": _n_req,
             "n_dropped_over_limit": _dropped_cap + _dropped_budget})
SPEC_SAMPLE_COLOUR = {
    "type": "function",
    "function": {
        "name": "sample_colour",
        "description": (
            "Read the exact colour at one or more points of the slide, in pixel coordinates "
            "(origin top-left). Returns hex and RGB numbers.\n\n"
            "Give up to 64 points per call. By default it also reports the modal colour of a "
            "3x3 patch around each point, which tells you whether you landed on a flat fill or "
            "on an edge or a glyph. Set `patch` to 1 for the single pixel, or larger (max 15) "
            "for a small area.\n\n"
            "Use this for fills, backgrounds, accent bars and rules. For text colour prefer "
            "`text_blocks`, which already reports it per line."),
        "parameters": {
            "type": "object",
            "properties": {
                "points": {"type": "array",
                           "items": {"type": "array", "items": {"type": "number"}},
                           "description": "List of [x, y] pixel coordinates."},
                "patch": {"type": "integer",
                          "description": "Patch side in px (1 = single pixel, default 3)."},
            },
            "required": ["points"],
        },
    },
}
def t_region_stats(state, args):
    err = _check_state(state)
    if err:
        return _contract_error(err)
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    H, W = arr.shape[:2]
    regs = args.get("regions") or ([args["bbox"]] if args.get("bbox") else [])
    if not regs:
        return ([{"type": "text",
                  "text": "Error: give `regions` as a list of [x1, y1, x2, y2] boxes."}],
                {"success": False})
    if len(regs) > _MAX_REGIONS:
        regs = regs[:_MAX_REGIONS]
    _n_req_regs = len(args.get("regions") or
                      ([args["bbox"]] if args.get("bbox") else []))
    _dropped_regs = max(0, _n_req_regs - _MAX_REGIONS)
    out, values = [], []
    for box in regs:
        bb = _clip_box(box, W, H)
        if bb is None:
            out.append(f"  {box} -> degenerate after clipping to {W}x{H}")
            continue
        x1, y1, x2, y2 = bb
        sub = arr[y1:y2, x1:x2]
        px = sub.reshape(-1, 3)
        mod, frac = _modal_colour(px)
        tops = _top_colours(px, 4)
        base = np.array([int(mod[1:3], 16), int(mod[3:5], 16), int(mod[5:7], 16)])
        d = np.abs(sub.astype(np.int16) - base).max(axis=2)
        ink = d > 28
        ratio_ink = float(ink.mean())
        if ink.any():
            ys, xs = np.where(ink)
            cb = [int(x1 + xs.min()), int(y1 + ys.min()),
                  int(x1 + xs.max()) + 1, int(y1 + ys.max()) + 1]
        else:
            cb = None
        r = state.get("ratio") or 1.0
        out.append(
            f"  [{x1},{y1},{x2},{y2}] {x2-x1}x{y2-y1}px "
            f"({(x2-x1)/r:.1f}x{(y2-y1)/r:.1f}pt)\n"
            f"      dominant {mod} ({frac:.0%})   ink vs dominant {ratio_ink:.1%}\n"
            f"      top colours " + ", ".join(f"{h} {f:.0%}" for h, f in tops)
            + (f"\n      non-dominant content spans [{cb[0]},{cb[1]},{cb[2]},{cb[3]}] px "
               f"= [{cb[0]/r:.1f},{cb[1]/r:.1f},{cb[2]/r:.1f},{cb[3]/r:.1f}] pt"
               if cb else "\n      the region is a single flat colour"))
        values.append({"bbox_px": [x1, y1, x2, y2], "dominant": mod,
                       "dominant_frac": round(frac, 3),
                       "ink_frac": round(ratio_ink, 4),
                       "top_colours": [h for h, _ in tops], "content_bbox_px": cb})
    state.setdefault("tool_values", []).extend(values)
    _txt = "region_stats:\n" + "\n".join(out)
    if _dropped_regs:
        _txt += (f"\n\n[You asked for {_n_req_regs} regions but only the first "
                 f"{_MAX_REGIONS} were measured; {_dropped_regs} were NOT. "
                 f"The list above is INCOMPLETE -- re-ask for the rest.]")
    if not values:
        return ([{"type": "text",
                  "text": _txt + "\n\nNothing could be measured: every box you "
                          "gave was empty or fell outside the image after "
                          "clipping. This is NOT a statement that those areas "
                          "are blank -- re-check the coordinates (they must be "
                          "PIXELS, [x1, y1, x2, y2], origin top-left) and ask "
                          "again."}],
                {"success": False, "n": 0, "values": [],
                 "n_requested": _n_req_regs, "all_degenerate": True})
    return ([{"type": "text", "text": _txt}],
            {"success": True, "n": len(values), "values": values,
             "n_requested": _n_req_regs,
             "n_dropped_over_limit": _dropped_regs})
SPEC_REGION_STATS = {
    "type": "function",
    "function": {
        "name": "region_stats",
        "description": (
            "Summarise one or more rectangular regions of the slide as numbers: the dominant "
            "colour and how much of the region it covers, the top colours by area, how much of "
            "the region differs from the dominant colour, and the bounding box of that "
            "non-dominant content in both px and pt.\n\n"
            "This is how you measure a shape: put a generous box around it and read back where "
            "its content actually starts and ends, plus its fill colour. Up to 16 regions per "
            "call; boxes are [x1, y1, x2, y2] in pixels, origin top-left."),
        "parameters": {
            "type": "object",
            "properties": {
                "regions": {"type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "List of [x1, y1, x2, y2] pixel boxes."},
            },
            "required": ["regions"],
        },
    },
}
def t_find_edges(state, args):
    err = _check_state(state)
    if err:
        return _contract_error(err)
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    H, W = arr.shape[:2]
    box = args.get("bbox")
    bb = _clip_box(box, W, H) if box else (0, 0, W, H)
    if bb is None:
        return ([{"type": "text", "text": f"Error: bbox {box} is degenerate."}],
                {"success": False})
    x1, y1, x2, y2 = bb
    sub = arr[y1:y2, x1:x2].astype(np.int16)
    r = state.get("ratio") or 1.0
    dy = np.abs(np.diff(sub, axis=0)).max(axis=2)
    dx = np.abs(np.diff(sub, axis=1)).max(axis=2)
    min_frac = float(args.get("min_span") or 0.6)
    min_frac = max(0.1, min(min_frac, 1.0))
    hrow = (dy > 28).mean(axis=1)
    vcol = (dx > 28).mean(axis=0)
    def runs(sig):
        idx = np.where(sig >= min_frac)[0]
        if idx.size == 0:
            return []
        groups, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > 2:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        return groups
    hs, vs = runs(hrow), runs(vcol)
    lines = []
    _EDGE_LIST_CAP = 40
    for a, b in hs[:_EDGE_LIST_CAP]:
        yy = y1 + (a + b) / 2 + 0.5
        lines.append(f"  horizontal at y={yy:.0f}px = {yy/r:.1f}pt "
                     f"(spans {hrow[a:b+1].max():.0%} of the width)")
    for a, b in vs[:_EDGE_LIST_CAP]:
        xx = x1 + (a + b) / 2 + 0.5
        lines.append(f"  vertical   at x={xx:.0f}px = {xx/r:.1f}pt "
                     f"(spans {vcol[a:b+1].max():.0%} of the height)")
    _omitted = max(0, len(hs) - _EDGE_LIST_CAP) + max(0, len(vs) - _EDGE_LIST_CAP)
    _near = []
    if not hs and hrow.size:
        _near.append(("horizontal", float(hrow.max())))
    if not vs and vcol.size:
        _near.append(("vertical", float(vcol.max())))
    txt = (f"find_edges in [{x1},{y1},{x2},{y2}]: "
           f"{len(hs)} horizontal, {len(vs)} vertical edge(s) spanning at least "
           f"{min_frac:.0%} of the region.\n"
           + ("\n".join(lines) if lines else "  (none at this threshold)"))
    if _near:
        _best = max(_near, key=lambda t: t[1])
        if _best[1] >= 0.5 * min_frac:
            txt += (f"\n\n[Not necessarily empty: the strongest {_best[0]} "
                    f"candidate here spans {_best[1]:.0%} of the region, just "
                    f"under the {min_frac:.0%} you asked for. Re-run with "
                    f"min_span={max(0.1, _best[1] - 0.02):.2f} to see it. "
                    f"⇒ Do NOT conclude there is no rule in this region.]")
        else:
            txt += (f"\n\n[The strongest candidate spans only {_best[1]:.0%} of "
                    f"the region, well under your {min_frac:.0%} threshold, so "
                    f"this region most likely really has no straight rule.]")
    if _omitted:
        txt += (f"\n\n[Only the first {_EDGE_LIST_CAP} edges of each direction are "
                f"listed above; {_omitted} more were found but NOT shown. This list "
                f"is INCOMPLETE -- do not treat it as every edge in the region. "
                f"A count this high usually means `min_span` is low enough to be "
                f"picking up glyph and texture edges rather than rules: raise "
                f"`min_span`, or ask for a smaller region, to get a list you can "
                f"use in full.]")
    return ([{"type": "text", "text": txt}],
            {"success": True, "n_h": len(hs), "n_v": len(vs),
             "n_omitted": _omitted, "list_truncated": bool(_omitted)})
SPEC_FIND_EDGES = {
    "type": "function",
    "function": {
        "name": "find_edges",
        "description": (
            "Find long straight horizontal and vertical edges inside a region, reported as "
            "y / x positions in both pixels and points. Use it to locate divider rules, table "
            "gridlines, underlines and the boundaries of large colour blocks, instead of "
            "estimating them by eye.\n\n"
            "`bbox` is [x1, y1, x2, y2] in pixels (default: the whole slide). `min_span` is how "
            "much of the region an edge must cross to count, 0..1, default 0.6 -- lower it to "
            "catch short rules."),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {"type": "array", "items": {"type": "number"},
                         "description": "[x1, y1, x2, y2] in pixels."},
                "min_span": {"type": "number",
                             "description": "Minimum fraction of the region an edge spans."},
            },
        },
    },
}
MEASURE_SPECS = [SPEC_PAGE_INFO, SPEC_TEXT_BLOCKS, SPEC_SAMPLE_COLOUR,
                 SPEC_REGION_STATS, SPEC_FIND_EDGES]
MEASURE_DISPATCH = {
    "page_info": t_page_info,
    "text_blocks": t_text_blocks,
    "sample_colour": t_sample_colour,
    "region_stats": t_region_stats,
    "find_edges": t_find_edges,
}
