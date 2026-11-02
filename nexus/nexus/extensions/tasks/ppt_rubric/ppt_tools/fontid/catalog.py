#!/usr/bin/env python3
"安装字体目录：把本机 fontconfig 能看见的每个**字体面(face)**列成一行。"
import json
import os
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "installed_faces.json")
_SEP = "\x1f"
_FMT = _SEP.join(["%{file}", "%{index}", "%{family}", "%{style}",
                  "%{weight}", "%{width}", "%{slant}", "%{fontformat}",
                  "%{charset}"]) + "\n"
def _fc_list():
    fmt = _SEP.join(["%{file}", "%{index}", "%{family}", "%{style}",
                     "%{weight}", "%{width}", "%{slant}", "%{fontformat}"]) + "\n"
    r = subprocess.run(["fc-list", "--format", fmt], capture_output=True, text=True)
    rows = []
    for ln in r.stdout.splitlines():
        p = ln.split(_SEP)
        if len(p) < 8:
            continue
        rows.append(p)
    return rows
def build():
    faces = {}
    for p in _fc_list():
        path, idx, fam, style, wgt, wid, slant, fmt = p[:8]
        try:
            idx = int(idx)
        except ValueError:
            idx = 0
        key = f"{path}#{idx}"
        fams = [x.strip() for x in fam.split(",") if x.strip()]
        styles = [x.strip() for x in style.split(",") if x.strip()]
        if key in faces:
            for f in fams:
                if f not in faces[key]["families"]:
                    faces[key]["families"].append(f)
            for s in styles:
                if s not in faces[key]["styles"]:
                    faces[key]["styles"].append(s)
            continue
        faces[key] = {
            "key": key, "file": path, "index": idx,
            "families": fams, "styles": styles,
            "weight": _num(wgt), "width": _num(wid), "slant": _num(slant),
            "format": fmt,
        }
    out = sorted(faces.values(), key=lambda d: (d["file"], d["index"]))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    return out
def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
def load():
    if not os.path.exists(OUT):
        return build()
    return json.load(open(OUT))
def name_index(faces=None):
    faces = faces if faces is not None else load()
    idx = {}
    for f in faces:
        for fam in f["families"]:
            idx.setdefault(fam.strip().lower(), []).append(f)
            for st in f["styles"]:
                idx.setdefault(f"{fam} {st}".strip().lower(), []).append(f)
    return idx
