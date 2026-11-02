#!/usr/bin/env python3
"`estimate_font_size` —— 与 `mx/measure_api.py` / `fontid/font_tool.py` 同构的测量工具。"
import math
import os
import re
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fontsize as FS  # noqa: E402
SIZE_MODE = "dual"
_MAX_CALLS = 40
_MAX_LINES = 24
_MIN_SIDE = 4
_MAX_TEXT = 120
_HINT = {
    "high": ("trustworthy -- measurements at this confidence land within 1 pt "
             "about 3 times in 4, better than judging the size by eye"),
    "medium": ("a hint, not a measurement -- at this confidence it is only about "
               "as good as your own reading of the screenshot. If your own "
               "judgement disagrees, keep yours"),
    "low": ("do NOT use this number -- at this confidence it is within 1 pt "
            "only about a third of the time, well below what you get by "
            "judging the size from the image. Use your own reading instead, "
            "or re-ask with a tighter box around a single line and the exact "
            "characters on it"),
}
_PRECISION_P75 = {"high": 0.38, "medium": 0.75, "low": 2.75}
_GRID_CANDIDATES_MAX = 4
def _grid_candidates(pt, conf):
    w = _PRECISION_P75.get(conf)
    if w is None or conf == "low":
        return []
    lo, hi = pt - w, pt + w
    cands = []
    k = math.floor(lo * 2) / 2.0
    while k <= hi + 1e-9:
        if lo - 1e-9 <= k <= hi + 1e-9 and k > 0:
            cands.append(k)
        k += 0.5
    if len(cands) > _GRID_CANDIDATES_MAX:
        return []
    return sorted(cands, key=lambda v: (abs(v - pt), v))
def _fmt_pt(v):
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    if abs(v * 2 - round(v * 2)) < 1e-9:
        return f"{v:.1f}"
    return f"{v:.2f}"
def _img(state):
    ref = state.get("reference")
    if ref is None:
        return None
    arr = state.get("_ref_arr")
    if arr is None:
        arr = np.asarray(ref.convert("RGB"))
        state["_ref_arr"] = arr
    return arr
def _ink_box(arr, box):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bx"))
    from text_blocks import _glyph_mask  # noqa: E402  ⛔ 原样复用，已标定
    x1, y1, x2, y2 = box
    sub = np.ascontiguousarray(arr[y1:y2, x1:x2])
    if sub.size == 0:
        return None
    m = _glyph_mask(sub)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return (int(x1 + xs.min()), int(y1 + ys.min()),
            int(x1 + xs.max()) + 1, int(y1 + ys.max()) + 1)
def t_estimate_font_size(state, args):
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    used = state.get("_n_fontsize", 0)
    if used >= _MAX_CALLS:
        return ([{"type": "text",
                  "text": f"Error: estimate_font_size budget spent "
                          f"({_MAX_CALLS} calls)."}],
                {"success": False})
    lines = args.get("lines")
    if not lines and args.get("box") is not None:
        lines = [{k: args.get(k) for k in
                  ("box", "text", "font", "bold", "italic")}]
    if not isinstance(lines, list) or not lines:
        return ([{"type": "text",
                  "text": "Error: give `lines` as a list of "
                          "{box:[x1,y1,x2,y2], text:\"...\"} objects, one per "
                          "line of text. Box is in screenshot pixels."}],
                {"success": False})
    _n_req_lines = len(lines)
    _dropped_lines = max(0, _n_req_lines - _MAX_LINES)
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES]
    H, W = arr.shape[:2]
    ratio = state.get("ratio") or 1.0
    if not ratio or ratio <= 0:
        ps = state.get("page_size") or {}
        ratio = (W / float(ps["width"])) if ps.get("width") else 1.3333
    rows, values, bad = [], [], []
    _clipped_lines = []
    _hint_lines, _why_lines = {}, {}
    for i, ln in enumerate(lines, 1):
        box = ln.get("box")
        text = (ln.get("text") or "").strip()
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            bad.append(f"  {i}. `box` must be [x1, y1, x2, y2] in pixels")
            continue
        if not text:
            bad.append(f"  {i}. `text` is required -- pass the characters you "
                       f"can read on that line")
            continue
        if len(text) > _MAX_TEXT:
            text = text[:_MAX_TEXT]
            _clipped_lines.append(i)
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        except (TypeError, ValueError):
            bad.append(f"  {i}. box values must be numbers")
            continue
        _ox1, _oy1, _ox2, _oy2 = x1, y1, x2, y2
        x1, x2 = sorted((max(0, min(x1, W)), max(0, min(x2, W))))
        y1, y2 = sorted((max(0, min(y1, H)), max(0, min(y2, H))))
        if x2 - x1 < _MIN_SIDE or y2 - y1 < _MIN_SIDE:
            _ob = (_ox2 > W or _oy2 > H or _ox1 < 0 or _oy1 < 0)
            if _ob:
                bad.append(
                    f"  {i}. that box lies partly or wholly OUTSIDE the "
                    f"{W}x{H} px screenshot (you gave "
                    f"[{_ox1},{_oy1},{_ox2},{_oy2}]), so after clipping it "
                    f"collapsed to {x2-x1}x{y2-y1} px. The box is not too "
                    f"small -- its coordinates are out of range. Boxes are in "
                    f"SCREENSHOT PIXELS as [x1, y1, x2, y2] with the origin at "
                    f"the top-left; check you did not pass point coordinates "
                    f"or swap x and y.")
            else:
                bad.append(f"  {i}. that box is too small to hold text")
            continue
        tight = _ink_box(arr, (x1, y1, x2, y2))
        if tight is None:
            bad.append(f"  {i}. no text-like pixels inside [{x1},{y1},{x2},{y2}]")
            continue
        tx1, ty1, tx2, ty2 = tight
        e = FS.estimate(ty2 - ty1, tx2 - tx1, text, ratio,
                        font=ln.get("font"), bold=bool(ln.get("bold")),
                        italic=bool(ln.get("italic")))
        if e is None:
            bad.append(f"  {i}. could not model that text (no installed font "
                       f"covers every character)")
            continue
        pt = e["pt"]
        if SIZE_MODE == "dual":
            cands = _grid_candidates(pt, e["conf"])
            pt_s = f"{pt:.2f} pt"
            if cands:
                near = " or ".join(f"{_fmt_pt(c)}" for c in cands)
                pt_s += (f"   [tidy value{'s' if len(cands) > 1 else ''} within "
                         f"measurement precision: {near}]")
            rng_s = f"{e['lo']:.1f}-{e['hi']:.1f} pt"
            pt_val = round(pt, 2)
        elif SIZE_MODE == "round":
            pt_s = f"{int(round(pt)):d} pt"
            rng_s = f"{int(round(e['lo']))}-{int(round(e['hi']))} pt"
            pt_val = float(round(pt))
        else:
            pt_s = f"{pt:5.1f} pt"
            rng_s = f"{e['lo']:.1f}-{e['hi']:.1f} pt"
            pt_val = round(pt, 1)
        rows.append(
            f"  {i}. {pt_s}   ({rng_s}, confidence {e['conf']})"
            f"   [{ty2-ty1}x{tx2-tx1} px ink box"
            + ("" if e["exact_font"] else ", font assumed")
            + "]")
        _hint_lines.setdefault(e["conf"], []).append(i)
        for w in (e["why"] or []):
            _tpl = re.sub(r"\d+(?:\.\d+)?", "#", w)
            _why_lines.setdefault(_tpl, {"text": w, "lines": [], "nums": []})
            _why_lines[_tpl]["lines"].append(i)
            _why_lines[_tpl]["nums"].append(
                (i, ", ".join(re.findall(r"\d+(?:\.\d+)?", w))))
        values.append({"line": i, "pt": pt_val,
                       "pt_measured": round(pt, 3),
                       "grid_candidates": (_grid_candidates(pt, e["conf"])
                                           if SIZE_MODE == "dual" else []),
                       "lo": round(e["lo"], 1), "hi": round(e["hi"], 1),
                       "conf": e["conf"], "script": e["script"],
                       "ink_box_px": [tx1, ty1, tx2, ty2],
                       "pt_from_height": round(e["pt_h"], 2),
                       "pt_from_width": (None if e["pt_w"] is None
                                         else round(e["pt_w"], 2))})
    state["_n_fontsize"] = used + 1
    if not values:
        return ([{"type": "text",
                  "text": "estimate_font_size: nothing could be measured.\n"
                          + "\n".join(bad)}],
                {"success": False})
    n_trust = sum(1 for v in values if v["conf"] == "high")
    if SIZE_MODE == "dual":
        _trust_line = (f"\n{n_trust} of {len(values)} line(s) came back at high "
                       "confidence." if n_trust else "")
        _whole = (
            "\nThe number before the brackets is the raw measurement -- write it "
            "into Pt(...) unless you have a reason not to. Where a tidy value "
            "(a whole point, or a half point such as 10.5) falls inside the "
            "measurement precision, it is listed in square brackets: decks are "
            "often laid out on tidy sizes, so a listed value is usually the "
            "better choice. But this tool cannot tell a true 10.5 from an 11 "
            "measured slightly low, so the decision is yours.\n"
            "Do NOT force sizes to whole numbers. Fractional font sizes are "
            "normal and deliberate in real decks: 10.5 and 7.5 are standard "
            "Chinese type sizes (wu-hao and liu-hao), text pasted from Word or "
            "a page with autofit enabled yields values like 12.1, and a deck "
            "rescaled from 16:9 to 4:3 puts every size on a fraction such as "
            "18.67. If the measurement is clearly not tidy, that is a finding, "
            "not noise -- keep it.")
    elif SIZE_MODE == "round":
        _trust_line = (f"\n{n_trust} of {len(values)} line(s) came back at high "
                       "confidence -- write those numbers into Pt(...) as given."
                       if n_trust else "")
        _whole = ("\nSizes are reported as whole points because that is what "
                  "real decks use: in the reference corpus 99.4% of declared "
                  "font sizes are whole numbers (n=1,496). A fractional size "
                  "such as Pt(17.5) is almost always a mistake, so keep your "
                  "own sizes whole too, including ones you did not measure.")
    else:
        _trust_line = (f"\n{n_trust} of {len(values)} line(s) came back at high "
                       "confidence -- use those numbers as written, and do not "
                       "round them to a 'nice' value unless the range brackets one."
                       if n_trust else "")
        _whole = ""
    def _ref(ns):
        ns = [str(n) for n in ns]
        if len(ns) == 1:
            return f"line {ns[0]}"
        return "lines " + (", ".join(ns[:-1]) + " and " + ns[-1])
    _merged = []
    for conf in ("low", "medium", "high"):
        ns = _hint_lines.get(conf)
        if ns:
            _merged.append(f"\n{_ref(ns).capitalize()} came back at "
                           f"{conf} confidence -- {_HINT[conf]}")
    for tpl, g in _why_lines.items():
        _merged.append(f"\n{_ref(g['lines']).capitalize()}: {g['text']}")
        _rest = [f"{i}:{nums}" for i, nums in g["nums"][1:] if nums]
        if _rest:
            _merged.append(f"\n   (per-line figures -- "
                           + "; ".join(_rest) + ")")
    _lat_lines = [i for i, ln in enumerate(lines, 1)
                  if FS.script_of((ln.get("text") or "")) == "lat"]
    _gave_font = any((ln.get("font") or "").strip() for ln in lines)
    if _lat_lines and not _gave_font:
        _merged.append(
            f"\n{_ref([str(i) for i in _lat_lines]).capitalize()} "
            f"{'is' if len(_lat_lines) == 1 else 'are'} Latin-script. Sizes on "
            f"Latin text are less certain than on CJK: CJK characters all "
            f"advance exactly one em, so the line width alone pins the size "
            f"down, while Latin letter widths vary by typeface. Measured on "
            f"Latin pages: 75.5% within 1pt, and the `high` band is not "
            f"reachable at all without a font. Call identify_font on one of "
            f"these lines and pass its answer back as `font` -- that lifts it "
            f"to 84.3% overall and 97.9% on the lines that then qualify as "
            f"high confidence. Large text is the weak spot (54.7% at 20pt+).")
    txt = (f"estimate_font_size: {len(values)} line(s) measured. Sizes are in "
           f"points, the same unit you write into Pt(...).\n"
           + "\n".join(rows)
           + "".join(_merged)
           + ("\n\nSkipped:\n" + "\n".join(bad) if bad else "")
           + (f"\n\n[You asked for {_n_req_lines} lines but only the first "
              f"{_MAX_LINES} were measured; {_dropped_lines} were NOT. This "
              f"answer is INCOMPLETE -- ask for the rest in another call.]"
              if _dropped_lines else "")
           + (f"\n\n[{_ref(_clipped_lines).capitalize()}: the text you gave was "
              f"longer than {_MAX_TEXT} characters and was cut to the first "
              f"{_MAX_TEXT}, then compared against the ink of the WHOLE box -- "
              f"a mismatched comparison, so trust {'those' if len(_clipped_lines) > 1 else 'that'} "
              f"row less than its confidence suggests. Ask again with a box "
              f"around a single shorter line.]"
              if _clipped_lines else "")
           + "\n\nThese come from rendering the exact characters you gave at "
             "every candidate size and matching the ink, so they already account "
             "for capitals, descenders, digits and CJK having different heights."
           + _trust_line
           + (f"\n{len(values) - n_trust} line(s) came back at medium or low "
              "confidence. At medium the estimate is only about as good as your "
              "own reading of the screenshot; at low it is worse. Judge those "
              "sizes from the image rather than taking the number."
              if len(values) - n_trust else "")
           + _whole)
    state.setdefault("tool_values", []).extend(values)
    return ([{"type": "text", "text": txt}],
            {"success": True, "n": len(values), "values": values,
             "n_requested_lines": _n_req_lines,
             "n_dropped_over_limit": _dropped_lines,
             "text_clipped_lines": _clipped_lines})
SPEC_ESTIMATE_FONT_SIZE = {
    "type": "function",
    "function": {
        "name": "estimate_font_size",
        "description": (
            "Measure the point size of a line of text on the screenshot. Pass "
            "the box of ONE line plus the characters you can read there; boxes "
            "from `text_blocks` work directly and are the recommended input. "
            "Several lines per call.\n\n"
            "Read the confidence -- it decides whether to use the answer. At "
            "`high` (about a third of lines, almost all CJK here) it lands "
            "within 1 pt about 3 times in 4 and still beats judging the size "
            "by eye: use the number as given. At "
            "`medium` it is only about as good as your own "
            "reading, so treat it as a hint and keep your own judgement if they "
            "disagree. At `low` it is right only about a third of the time, well "
            "worse "
            "than reading the size off the image: ignore the number. Latin text "
            "is the weak case; passing a known `font` sharpens it.\n\n"
            "One line per entry: a box covering a whole paragraph reports a size "
            "several times too large. The characters matter, because digits, "
            "descenders and CJK have different ink heights at the same size.\n\n"
            "The answer is the raw measurement, not a rounded one. Where a tidy "
            "value (whole point, or a half point like 10.5) lies within the "
            "tool's measured precision it is offered in brackets as a candidate, "
            "because decks are often laid out on tidy sizes. Do not force sizes "
            "to whole numbers on your own: fractional sizes are normal and "
            "deliberate (10.5 and 7.5 are standard Chinese type sizes, Word "
            "paste and autofit produce values like 12.1, and rescaling a deck "
            "from 16:9 to 4:3 makes every size fractional, e.g. 18.67)."),
        "parameters": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "array",
                    "description": "One entry per line of text to measure.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box": {"type": "array",
                                    "items": {"type": "number"},
                                    "description": ("[x1, y1, x2, y2] in "
                                                    "screenshot pixels, around a "
                                                    "single line of text.")},
                            "text": {"type": "string",
                                     "description": ("The characters on that "
                                                     "line, as they appear.")},
                            "font": {"type": "string",
                                     "description": ("Typeface name, if known "
                                                     "(e.g. from identify_font). "
                                                     "Optional but makes the "
                                                     "answer markedly sharper "
                                                     "for Latin text.")},
                            "bold": {"type": "boolean"},
                            "italic": {"type": "boolean"},
                        },
                        "required": ["box", "text"],
                    },
                },
            },
            "required": ["lines"],
        },
    },
}
SIZE_SPECS = [SPEC_ESTIMATE_FONT_SIZE]
SIZE_DISPATCH = {"estimate_font_size": t_estimate_font_size}
