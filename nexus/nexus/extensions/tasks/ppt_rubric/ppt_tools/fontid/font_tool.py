#!/usr/bin/env python3
"`identify_font` —— 与 `mx/measure_api.py` 同构的测量工具。"
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import identify as ID  # noqa: E402
_CONF_HIGH = 0.075
_CONF_MED = 0.030
WITHHOLD_BELOW_HIGH = True
_MAX_CALLS = 24
_MIN_SIDE = 6
_MAX_TEXT = 60
def _img(state):
    ref = state.get("reference")
    if ref is None:
        return None
    arr = state.get("_ref_arr")
    if arr is None:
        arr = np.asarray(ref.convert("RGB"))
        state["_ref_arr"] = arr
    return arr
def t_identify_font(state, args):
    arr = _img(state)
    if arr is None:
        return [{"type": "text", "text": "Error: no reference image."}], {"success": False}
    used = state.get("_n_fontid", 0)
    if used >= _MAX_CALLS:
        return ([{"type": "text",
                  "text": f"Error: identify_font budget spent ({_MAX_CALLS} calls)."}],
                {"success": False})
    box = args.get("box")
    text = (args.get("text") or "").strip()
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return ([{"type": "text",
                  "text": "Error: `box` must be [x1, y1, x2, y2] in screenshot pixels."}],
                {"success": False})
    if not text:
        return ([{"type": "text",
                  "text": "Error: `text` is required -- pass the words you can read "
                          "inside that box, exactly as they appear (same capitalisation)."}],
                {"success": False})
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT]
        _text_clipped = True
    else:
        _text_clipped = False
    H, W = arr.shape[:2]
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return [{"type": "text", "text": "Error: box values must be numbers."}], {"success": False}
    _ox1, _oy1, _ox2, _oy2 = x1, y1, x2, y2
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x2 - x1 < _MIN_SIDE or y2 - y1 < _MIN_SIDE:
        if _ox2 > W or _oy2 > H or _ox1 < 0 or _oy1 < 0:
            return ([{"type": "text",
                      "text": f"Error: that box lies partly or wholly OUTSIDE "
                              f"the {W}x{H} px screenshot (you gave "
                              f"[{_ox1},{_oy1},{_ox2},{_oy2}]), so after "
                              f"clipping it collapsed to {x2-x1}x{y2-y1} px. "
                              f"The box is not too small -- its coordinates are "
                              f"out of range. Boxes are in SCREENSHOT PIXELS as "
                              f"[x1, y1, x2, y2], origin top-left; check you did "
                              f"not pass point coordinates or swap x and y."}],
                    {"success": False, "off_canvas": True})
        return ([{"type": "text", "text": "Error: that box is too small to hold text."}],
                {"success": False})
    topn = int(args.get("top_n") or 3)
    topn = max(1, min(5, topn))
    res, meta = ID.identify_region(arr, [x1, y1, x2, y2], text, topn=topn)
    state["_n_fontid"] = used + 1
    if not res:
        return ([{"type": "text",
                  "text": "identify_font: could not read glyphs there. "
                          f"({meta.get('reason', 'no result')})  "
                          "Try a tighter box around a single line of text."}],
                {"success": False})
    lines = []
    for i, r in enumerate(res):
        alias = [a for a in r["aliases"] if a.lower() != r["family"].lower()][:3]
        tag = []
        if r.get("bold"):
            tag.append("bold")
        if r.get("italic"):
            tag.append("italic")
        lines.append(
            f"  {i+1}. {r['family']}" + (" (" + ", ".join(tag) + ")" if tag else "")
            + f"   score {r['rel']:.3f}"
            + (f"   renders identically to: {', '.join(alias)}" if alias else ""))
    conf = ("high" if meta["margin"] >= _CONF_HIGH else
            "medium" if meta["margin"] >= _CONF_MED else "low")
    _clip_note = ""
    if _text_clipped:
        _clip_note = (
            f"\n\n[You passed {len(args.get('text') or '')} characters but only "
            f"the first {_MAX_TEXT} were used, and they were compared against "
            f"the ink of the WHOLE box -- so this answer is based on a "
            f"mismatched comparison and is less reliable than its confidence "
            f"suggests. Ask again with a box around a shorter span of text "
            f"whose characters you can give in full.]")
    _lat = (meta.get("script") == "lat")
    if WITHHOLD_BELOW_HIGH and conf != "high" and not _lat:
        txt = (f"identify_font: confidence {conf} -- this line does not pin down "
               "a typeface, so no name is reported.\n"
               "Leave the font unset for this text: it then inherits a face that "
               "is actually installed on the renderer, which is measured right "
               "29.9% of the time here, whereas acting on an answer at this "
               "confidence is right only 13-16% of the time. Do not name a font "
               "from general knowledge either -- that picks a face the renderer "
               "does not have, and it silently falls back.\n"
               "If another line on this slide uses the same style but is longer, "
               "asking about that one may reach high confidence. Do not run "
               "`fc-list` or look for font files -- every typeface the renderer "
               "has was already compared.")
        return ([{"type": "text", "text": txt + _clip_note}],
                {"success": True, "top": [], "withheld": [r["family"] for r in res],
                 "conf": conf, "margin": meta["margin"], "script": meta["script"],
                 "ms": meta["ms"], "text_clipped": _text_clipped})
    _is_lat = (meta.get("script") == "lat")
    if _is_lat:
        hint = {"high": ("on Latin text the top answer is right about "
                         "19 times in 20 -- strong enough to act on"),
                "medium": ("on Latin text the top answer is right about "
                           "4 times in 5 -- usually worth using"),
                "low": ("on Latin text the top answer is still right about "
                        "2 times in 3 -- better than guessing, but verify "
                        "if you can")}[conf]
    else:
        hint = {"high": ("the top answer is right about half the time -- the best "
                         "signal available here, but not a certainty"),
                "medium": "the top answer is right well under half the time",
                "low": "the top answer is right well under half the time"}[conf]
    txt = (f"identify_font: closest-rendering typefaces for the text you gave, ranked.\n"
           + "\n".join(lines)
           + f"\n\nconfidence {conf} -- {hint}. "
             f"(gap to runner-up {meta['margin']:.1%}; compared against "
             f"{meta['n_candidates']} distinct typefaces installed on the renderer.)\n"
           "score 1.000 is the best match; the others are relative to it -- "
           "values very close to 1.000 mean those fonts are nearly "
           "indistinguishable at this size.\n"
           "These are render-equivalence classes, not the true font name. Fonts "
           "listed as rendering identically are fully interchangeable: the "
           "renderer draws the same pixels for any of them, so pick whichever "
           "name you prefer and do not spend further calls narrowing it down.")
    if conf == "low" and not _is_lat:
        txt += ("\nDo not rely on this answer, and do not fall back on a font "
                "name from general knowledge -- leave the font unset instead, "
                "so the text inherits a face that is actually installed. "
                "On CJK text an unset font already lands on the most common "
                "reference face, so leaving it unset is the safer play. "
                "If another line uses this same style but is longer, asking "
                "about that one may give a usable answer. Looking for font "
                "files on disk will not help.")
    elif conf == "low" and _is_lat:
        txt += ("\nThis is the weakest band, but on Latin text it is still "
                "right about two thirds of the time -- considerably better "
                "than leaving the font unset, which on a Latin page renders "
                "as Calibri regardless of what the original used. Prefer the "
                "top answer here over a font name from general knowledge. "
                "If another line uses this same style but is longer, asking "
                "about that one will give a sharper answer.")
    elif conf == "medium":
        txt += ("\nIf another line on this slide uses the same style but is longer, "
                "asking about that one instead will give a sharper answer.")
    return ([{"type": "text", "text": txt + _clip_note}],
            {"success": True, "top": [r["family"] for r in res],
             "margin": meta["margin"], "script": meta["script"],
             "ms": meta["ms"], "text_clipped": _text_clipped})
SPEC_IDENTIFY_FONT = {
    "type": "function",
    "function": {
        "name": "identify_font",
        "description": (
            "Report which installed typefaces render a line of text on the "
            "screenshot most closely. Pass a box around ONE line plus the words "
            "you can read there.\n\n"
            "It returns a RENDER-EQUIVALENCE CLASS, not the true font name -- a "
            "hard information limit, not a shortcoming. On this renderer 999 "
            "font names collapse to 72 distinct pixel results on CJK text, so "
            "most font names are indistinguishable from an image: Arial, "
            "Calibri, Verdana, Tahoma and Microsoft YaHei all render "
            "identically, as do SimSun and SimHei. Any name in the class gives "
            "the same pixels, so pick one and move on.\n\n"
            "Only `high` confidence is worth acting on, and even there the top "
            "answer is right only about half the time -- it is the best signal "
            "available, not a certainty. Below that no name is reported at all, "
            "because acting on one was measured worse than simply leaving the "
            "font unset: when no "
            "name comes back, leave the font unset so the text inherits an "
            "installed face -- do NOT fall back on a name from general "
            "knowledge, which was measured wrong 79.3% of the time. Do not "
            "spend rounds pinning down an exact name and do not run `fc-list` "
            "or look for font files -- every font the renderer has was already "
            "compared, and every font this corpus uses is installed. Best used "
            "to confirm serif-versus-sans and weight."),
        "parameters": {
            "type": "object",
            "properties": {
                "box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": ("[x1, y1, x2, y2] in screenshot pixels, around a "
                                    "single line of text. Tighter is better; a box "
                                    "holding several lines or a logo will not work."),
                },
                "text": {
                    "type": "string",
                    "description": ("The words inside that box, exactly as they appear "
                                    "including capitalisation. This is what gets "
                                    "rendered in each candidate font for comparison."),
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many candidates to return (1-5, default 3).",
                },
            },
            "required": ["box", "text"],
        },
    },
}
FONT_SPECS = [SPEC_IDENTIFY_FONT]
FONT_DISPATCH = {"identify_font": t_identify_font}
def warmup():
    return ID.warmup()
