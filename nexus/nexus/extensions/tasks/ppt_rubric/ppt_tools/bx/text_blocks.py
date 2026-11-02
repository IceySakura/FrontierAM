#!/usr/bin/env python3
'`list_text_blocks` -- a deterministic "where is every text block" tool.'
import numpy as np
from scipy import ndimage
_RESID_K = 3
_INK_T = 28
_BG_QUANT = 8
_BG_MIN_COVER = 0.50
_THICK_K = 17
_LINE_L = 41
_MIN_GLYPH_H = 4
_MAX_GLYPH_H = 160
_MIN_BLOCK_PX = 24
_SPECKLE_PX = 6
_SPECKLE_EXT = 3
_MIN_SEG_W = 8
_MIN_SEG_H = 8
_REGION_K = 51
_LINE_GAP_X = 1.45
_BLOCK_GAP_Y = 0.55
_BLOCK_H_RATIO = 1.35
_BLOCK_OVERLAP = 0.80
_BLOCK_EDGE_TOL = 0.6
def _sep_open(m, k, axis=None):
    a = m.astype(np.uint8)
    axes = (0, 1) if axis is None else (axis,)
    for ax in axes:
        a = ndimage.minimum_filter1d(a, k, axis=ax, mode='constant')
    for ax in axes:
        a = ndimage.maximum_filter1d(a, k, axis=ax, mode='constant')
    return a.astype(bool)
def _sep_dilate(m, kx, ky):
    a = m.astype(np.uint8)
    if kx > 1:
        a = ndimage.maximum_filter1d(a, kx, axis=1, mode='constant')
    if ky > 1:
        a = ndimage.maximum_filter1d(a, ky, axis=0, mode='constant')
    return a.astype(bool)
def _residual(img):
    c = img.astype(np.int16)
    k = _RESID_K
    best = np.zeros(c.shape[:2], dtype=np.int16)
    for ch in range(c.shape[2]):
        a = c[:, :, ch]
        d = ndimage.maximum_filter1d(a, k, axis=0, mode='nearest')
        d = ndimage.maximum_filter1d(d, k, axis=1, mode='nearest')
        cl = ndimage.minimum_filter1d(d, k, axis=0, mode='nearest')
        cl = ndimage.minimum_filter1d(cl, k, axis=1, mode='nearest')
        e = ndimage.minimum_filter1d(a, k, axis=0, mode='nearest')
        e = ndimage.minimum_filter1d(e, k, axis=1, mode='nearest')
        op = ndimage.maximum_filter1d(e, k, axis=0, mode='nearest')
        op = ndimage.maximum_filter1d(op, k, axis=1, mode='nearest')
        np.maximum(best, cl - a, out=best)
        np.maximum(best, a - op, out=best)
    return best
def _page_background(img):
    q = img.astype(np.int32) // _BG_QUANT
    keys = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    vals, counts = np.unique(keys, return_counts=True)
    i = int(np.argmax(counts))
    sel = keys == int(vals[i])
    col = np.median(img[sel].astype(np.int16), axis=0).astype(np.int16)
    return col, float(counts[i]) / keys.size
def _bg_difference(img):
    bg, cover = _page_background(img)
    if cover < _BG_MIN_COVER:
        return None
    return np.abs(img.astype(np.int16) - bg[None, None, :]).max(axis=2)
def _drop_non_glyph(ink):
    if not ink.any():
        return ink
    thick = _sep_open(ink, _THICK_K)
    rules = (_sep_open(ink, _LINE_L, axis=1)
             | _sep_open(ink, _LINE_L, axis=0))
    return ink & ~thick & ~rules
def _glyph_mask(img):
    ink = _drop_non_glyph(_residual(img) > _INK_T)
    bgd = _bg_difference(img)
    if bgd is not None:
        ink = ink | _drop_non_glyph(bgd > _INK_T)
    if not ink.any():
        return ink
    lab, n = ndimage.label(ink)
    if n == 0:
        return ink
    sizes = ndimage.sum(ink, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sizes[i - 1] < _SPECKLE_PX:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if max(h, w) < _SPECKLE_EXT:
            continue
        keep[i] = True
    return keep[lab]
def _split_tall_band(prof, max_h):
    out = [(0, len(prof))]
    for _ in range(64):
        nxt = []
        changed = False
        for a, b in out:
            if b - a <= max_h:
                nxt.append((a, b))
                continue
            inner = prof[a + 1:b - 1]
            if inner.size == 0:
                nxt.append((a, b))
                continue
            depth = inner.astype(np.float64)
            centre = np.abs(np.arange(inner.size) - inner.size / 2.0)
            cut = a + 1 + int(np.lexsort((centre, depth))[0])
            if cut <= a or cut >= b:
                nxt.append((a, b))
                continue
            nxt.append((a, cut))
            nxt.append((cut, b))
            changed = True
        out = nxt
        if not changed:
            break
    return [(a, b) for a, b in out if b > a]
def _segments(mask):
    out = []
    if not mask.any():
        return out
    regions = _sep_dilate(mask, _REGION_K, _REGION_K)
    lab, _ = ndimage.label(regions)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = mask[sl] & (lab[sl] == i)
        if not sub.any():
            continue
        y0, x0 = sl[0].start, sl[1].start
        for a0, b0 in _runs(sub.any(axis=1)):
            if (b0 - a0) > _MAX_GLYPH_H:
                prof = sub[a0:b0].sum(axis=1)
                pieces = [(a0 + s, a0 + e) for s, e in _split_tall_band(prof, _MAX_GLYPH_H)]
            else:
                pieces = [(a0, b0)]
            for a, b in pieces:
                band = sub[a:b]
                gh = b - a
                if gh < _MIN_GLYPH_H or gh > _MAX_GLYPH_H:
                    continue
                kx = max(3, int(round(gh * _LINE_GAP_X)))
                d = ndimage.maximum_filter1d(band.astype(np.uint8), kx,
                                             axis=1, mode='constant')
                for c, e in _runs(d.any(axis=0)):
                    seg = band[:, c:e]
                    if not seg.any():
                        continue
                    ys, xs = np.where(seg)
                    x1 = x0 + c + int(xs.min())
                    y1 = y0 + a + int(ys.min())
                    x2 = x0 + c + int(xs.max()) + 1
                    y2 = y0 + a + int(ys.max()) + 1
                    if (x2 - x1) < _MIN_SEG_W or (y2 - y1) < _MIN_SEG_H:
                        continue
                    out.append((y1, x1, y2, x2))
    return out
def _runs(rows):
    out, start = [], None
    for i, v in enumerate(rows):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(rows)))
    return out
def _glyph_height(mask):
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return max(mask.shape[0], 3)
    lens = []
    for x in cols[:: max(1, len(cols) // 64)]:
        for a, b in _runs(mask[:, x]):
            lens.append(b - a)
    if not lens:
        return max(mask.shape[0], 3)
    return max(float(np.median(lens)), 3.0)
def _group_lines(lines, mask):
    if not lines:
        return []
    idx = sorted(range(len(lines)), key=lambda i: (lines[i][0], lines[i][1]))
    parent = list(range(len(lines)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    for oi, i in enumerate(idx):
        y1i, x1i, y2i, x2i = lines[i]
        hi = y2i - y1i
        for j in idx[oi + 1:]:
            y1j, x1j, y2j, x2j = lines[j]
            hj = y2j - y1j
            if y1j - y2i > _BLOCK_GAP_Y * max(hi, hj):
                break
            ov = min(x2i, x2j) - max(x1i, x1j)
            if ov <= 0:
                continue
            if ov < _BLOCK_OVERLAP * min(x2i - x1i, x2j - x1j):
                continue
            same_left = abs(x1i - x1j) <= _BLOCK_EDGE_TOL * max(hi, hj)
            same_width = (abs((x2i - x1i) - (x2j - x1j))
                          <= _BLOCK_EDGE_TOL * max(hi, hj))
            if not (same_left or same_width):
                continue
            if max(hi, hj) / max(min(hi, hj), 1) > _BLOCK_H_RATIO:
                continue
            union(i, j)
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v, key=lambda i: (lines[i][0], lines[i][1])) for v in groups.values()]
def _dominant_colour(img, mask, box):
    y1, x1, y2, x2 = box
    sub = img[y1:y2, x1:x2]
    m = mask[y1:y2, x1:x2]
    if not m.any():
        return None
    px = sub[m].astype(np.int32)
    q = px // 16
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = int(vals[np.argmax(counts)])
    sel = keys == k
    med = np.median(px[sel], axis=0).astype(int)
    return '#%02X%02X%02X' % (int(med[0]), int(med[1]), int(med[2]))
def list_text_blocks(img, region=None, ratio=1.0, max_blocks=120):
    H, W = img.shape[:2]
    if region is not None:
        x1 = max(0, min(int(region[0]), W - 1))
        y1 = max(0, min(int(region[1]), H - 1))
        x2 = max(x1 + 1, min(int(region[2]), W))
        y2 = max(y1 + 1, min(int(region[3]), H))
    else:
        x1, y1, x2, y2 = 0, 0, W, H
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    if crop.size == 0:
        return {'ok': False, 'reason': 'empty region', 'n_blocks': 0, 'blocks': []}
    gmask = _glyph_mask(crop)
    if not gmask.any():
        return {'ok': False, 'reason': 'no text-like pixels found (only fills, '
                                       'icons, rules or photo noise)',
                'n_blocks': 0, 'blocks': []}
    lines = _segments(gmask)
    if not lines:
        return {'ok': False, 'reason': 'no text lines of plausible glyph size found',
                'n_blocks': 0, 'blocks': []}
    groups = _group_lines(lines, gmask)
    blocks = []
    for g in groups:
        ls = [lines[i] for i in g]
        by1 = min(l[0] for l in ls)
        bx1 = min(l[1] for l in ls)
        by2 = max(l[2] for l in ls)
        bx2 = max(l[3] for l in ls)
        n_px = int(gmask[by1:by2, bx1:bx2].sum())
        if n_px < _MIN_BLOCK_PX:
            continue
        col = _dominant_colour(crop, gmask, (by1, bx1, by2, bx2))
        blocks.append({
            'bbox_px': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
            'lines_px': [[l[1] + x1, l[0] + y1, l[3] + x1, l[2] + y1] for l in ls],
            'n_lines': len(ls),
            'colour': col,
            'n_ink_px': n_px,
        })
    if not blocks:
        return {'ok': False, 'reason': 'no block carried enough ink to be text',
                'n_blocks': 0, 'blocks': []}
    blocks.sort(key=lambda b: (b['bbox_px'][1], b['bbox_px'][0]))
    if len(blocks) > max_blocks:
        blocks = sorted(blocks, key=lambda b: -b['n_ink_px'])[:max_blocks]
        blocks.sort(key=lambda b: (b['bbox_px'][1], b['bbox_px'][0]))
    def to_pt(b):
        return [round(v / ratio, 2) for v in b]
    for i, b in enumerate(blocks, 1):
        b['id'] = i
        b['bbox_pt'] = to_pt(b['bbox_px'])
        b['lines_pt'] = [to_pt(l) for l in b['lines_px']]
    return {'ok': True, 'n_blocks': len(blocks), 'blocks': blocks}
