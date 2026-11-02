#!/usr/bin/env python3
"实测 LibreOffice 的字体替换：**声明的字体名 != 像素里的字形**。"
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_LAT = "Hamburgefonstiv 0123 RQgaey"
PROBE_CJK = "汉字测试永水国"
_EMU_PT = 12700
_ROW_H = 46.0
_BOX_W = 640.0
_PAD = 8.0
_FS = 24
def _pptx(pairs, text, path, page_w=720.0):
    from pptx import Presentation
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN
    prs = Presentation()
    prs.slide_width = Pt(page_w)
    prs.slide_height = Pt(_PAD * 2 + _ROW_H * len(pairs))
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for i, item in enumerate(pairs):
        if len(item) >= 4:
            _lab, fam, bold, ital = item[0], item[1], item[2], item[3]
        else:
            _lab, fam, bold, ital = item[0], item[1], False, False
        tb = slide.shapes.add_textbox(Pt(_PAD), Pt(_PAD + i * _ROW_H),
                                      Pt(_BOX_W), Pt(_ROW_H - 4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = text
        r.font.size = Pt(_FS)
        r.font.name = fam
        r.font.bold = bool(bold)
        r.font.italic = bool(ital)
        rPr = r._r.get_or_add_rPr()
        from pptx.oxml.ns import qn
        for tag in ("a:ea", "a:cs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = rPr.makeelement(qn(tag), {})
                rPr.append(el)
            el.set("typeface", fam)
    prs.save(path)
    return len(pairs)
def render_pptx(path, outdir, dpi=150):
    env = dict(os.environ)
    env["HOME"] = outdir
    subprocess.run(["soffice", "--headless", "--norestore",
                    "--convert-to", f"png:impress_png_Export:"
                    '{"PixelWidth":{"type":"long","value":%d},'
                    '"PixelHeight":{"type":"long","value":%d}}' % (0, 0),
                    "--outdir", outdir, path],
                   capture_output=True, env=env, timeout=600)
    png = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".png")
    return png if os.path.exists(png) else None
def render_pdf_png(path, outdir, scale=2.0):
    outdir = os.path.abspath(outdir)
    path = os.path.abspath(path)
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = outdir
    subprocess.run(["soffice", "--headless", "--norestore", "--convert-to", "pdf",
                    "--outdir", outdir, path], capture_output=True, env=env, timeout=900)
    pdf = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".pdf")
    if not os.path.exists(pdf):
        return None
    png = pdf[:-4] + ".png"
    subprocess.run(["pdftoppm", "-r", str(int(72 * scale)), "-png", "-singlefile",
                    pdf, png[:-4]], capture_output=True, timeout=600)
    return png if os.path.exists(png) else None
def row_signatures(png, n, scale=2.0):
    im = Image.open(png).convert("L")
    a = np.asarray(im)
    H = a.shape[0]
    rows = []
    step = H / float(n)
    for i in range(n):
        y0 = int(round(i * step))
        y1 = int(round((i + 1) * step))
        sub = a[y0:y1]
        ink = sub < 200
        if not ink.any():
            rows.append({"md5": None, "bbox": None, "px": 0})
            continue
        ys, xs = np.nonzero(ink)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        crop = sub[bb[1]:bb[3], bb[0]:bb[2]]
        rows.append({"md5": hashlib.md5(crop.tobytes()).hexdigest(),
                     "bbox": bb, "px": int(ink.sum()),
                     "shape": list(crop.shape)})
    return rows
def probe(names, text, workdir, tag):
    os.makedirs(workdir, exist_ok=True)
    norm = []
    for n in names:
        if isinstance(n, (tuple, list)):
            norm.append((n[0], bool(n[1]), bool(n[2])))
        else:
            norm.append((n, False, False))
    pairs = [(f"{n}|{int(b)}{int(i)}", n, b, i) for n, b, i in norm]
    px = os.path.join(workdir, f"probe_{tag}.pptx")
    _pptx(pairs, text, px)
    png = render_pdf_png(px, workdir)
    if png is None:
        raise RuntimeError("render failed for " + px)
    sigs = row_signatures(png, len(pairs))
    keys = [k for k, _n, _b, _i in pairs]
    return {k: s for k, s in zip(keys, sigs)}, png
