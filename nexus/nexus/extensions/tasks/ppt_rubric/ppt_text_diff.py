"""Compare the visible text of two rendered decks: a deterministic score plus a diff report.

Two decks that look the same can be built very differently -- one text box per
line or one per paragraph, python-pptx or pptxgenjs, grouped or flat. Matching
box-to-box punishes those differences, which are not defects. So matching here
ignores boxes entirely:

1. Every visible rendered line is normalised (NFKC, case, whitespace,
   quote/dash variants) and split into characters, each carrying an estimated
   on-canvas position, font size and colour.
2. The two character streams are aligned with greedy string tiling: repeatedly
   take the longest common substring that is still unmatched. Unlike a plain
   diff this tolerates *moved* blocks (a column laid out in another order), and
   because it works on characters, where a line wraps or which box holds it
   does not matter.
3. Matched character pairs give text recall / precision, and -- through their
   positions -- how far the text moved, how its size changed and whether its
   colour is right. Characters are weighted by font size, so a wrong title
   costs more than a wrong footnote.

The same alignment produces a short, human-readable report of the differences
(missing / partial / extra / moved / size / colour / hidden text). The judge
reads that report instead of two programs: finding differences is the
program's job, deciding how much they matter is the judge's.
"""

from __future__ import annotations

import math
import unicodedata

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .ppt_structure import TextLine


# Below this many visible reference characters there is nothing to score.
MIN_GOLD_CHARS = 8
# Greedy string tiling: shortest run accepted as a match. 1 would let unrelated
# text match character by character; single-character lines get a second pass.
MIN_TILE = 2
# Hard caps so a pathological deck cannot make alignment slow.
MAX_CHARS = 4000
MAX_TILES = 600

# Score shaping (all on canvas-normalised units).
# Displacement, as a fraction of the canvas diagonal, costing 1/e. 0.05 made a
# 2% shift of all text cost more (0.91) than deleting 10% of it (0.97) in the
# Reward CI; content has to dominate placement.
POS_SCALE = 0.10
SIZE_SCALE = 0.30  # |ln(size ratio)| costing 1/e (ratio 1.35x)
# Content score = F-beta(recall, precision) x layout^LAYOUT_EXPONENT. Content
# was f1^0.6 before; that let 30% of the text go missing for a 0.84.
CONTENT_BETA = 2.0
LAYOUT_EXPONENT = 0.5
COLOR_SCALE = 25.0  # CIE76 delta-E costing 1/e

# Report thresholds.
MISSING_BELOW = 0.3
PARTIAL_BELOW = 0.95
MOVED_ABOVE = 0.04
SIZE_RATIO_OUTSIDE = (0.8, 1.25)
COLOR_DE_ABOVE = 30.0
MAX_REPORT_ITEMS = 25
QUOTE_CHARS = 40

_PUNCT_MAP = str.maketrans({
    '“': '"',
    '”': '"',
    '„': '"',
    '「': '"',
    '」': '"',
    '『': '"',
    '』': '"',
    '‘': "'",
    '’': "'",
    '‚': "'",
    '–': '-',
    '—': '-',
    '−': '-',
    '‐': '-',
    '－': '-',
    '~': '-',
    '～': '-',
    '…': '.',
    '·': '.',
    '•': '.',
    '●': '.',
    '▪': '.',
    '■': '.',
    '◆': '.',
    '►': '.',
    '、': ',',
    '，': ',',
    '。': '.',
    '：': ':',
    '；': ';',
    '！': '!',
    '？': '?',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '《': '<',
    '》': '>',
})


def normalise(text: str) -> str:
    """Canonical form for comparison: width, case, punctuation variants, no whitespace."""
    text = unicodedata.normalize('NFKC', text or '').translate(_PUNCT_MAP).lower()
    return ''.join(ch for ch in text if not ch.isspace() and unicodedata.category(ch)[0] != 'C')


@dataclass
class _Char:
    ch: str
    line: int
    x: float
    y: float
    size: float
    color: tuple[int, int, int]


def _reading_order(lines: list[TextLine]) -> list[int]:
    """Indices of ``lines`` top-to-bottom, left-to-right (row tolerance 1.5% of height)."""
    return sorted(range(len(lines)), key=lambda i: (round(lines[i].y0 / 0.015), lines[i].x0))


def _explode(lines: list[TextLine], aspect: float) -> list[_Char]:
    chars: list[_Char] = []
    for idx in _reading_order(lines):
        ln = lines[idx]
        norm = normalise(ln.text)
        n = len(norm)
        for k, ch in enumerate(norm):
            # Characters are spread evenly over the line's ink box; x is scaled by
            # the aspect ratio so distances are isotropic on the real canvas.
            x = (ln.x0 + (k + 0.5) / n * (ln.x1 - ln.x0)) * aspect
            chars.append(_Char(ch, idx, x, ln.cy, max(ln.size, 0.1), ln.color))
    return chars[:MAX_CHARS]


def _greedy_tiles(a: str, b: str, min_len: int) -> list[tuple[int, int, int]]:
    """Non-overlapping common substrings, longest first (greedy string tiling)."""
    am, bm = list(a), list(b)
    tiles: list[tuple[int, int, int]] = []
    for _ in range(MAX_TILES):
        sm = SequenceMatcher(None, ''.join(am), ''.join(bm), autojunk=False)
        i, j, k = sm.find_longest_match(0, len(am), 0, len(bm))
        if k < min_len:
            break
        tiles.append((i, j, k))
        for t in range(k):
            am[i + t] = '\ue000'  # sentinels: never equal to each other or to text
            bm[j + t] = '\ue001'
    return tiles


def _lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(v)) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def delta_e(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    l1, l2 = _lab(c1), _lab(c2)
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(l1, l2, strict=True)))


@dataclass
class TextDiff:
    """Result of comparing reference text with output text."""

    score: float | None  # None = not applicable (reference has no visible text)
    detail: dict[str, Any] = field(default_factory=dict)
    report: str = ''


def _region(x: float, y: float) -> str:
    col = 'left' if x < 1 / 3 else ('right' if x > 2 / 3 else 'center')
    row = 'top' if y < 1 / 3 else ('bottom' if y > 2 / 3 else 'middle')
    return 'center' if (row, col) == ('middle', 'center') else f'{row}-{col}'


def _quote(text: str) -> str:
    text = ' '.join((text or '').split())
    return '"' + (text if len(text) <= QUOTE_CHARS else text[: QUOTE_CHARS - 1] + '…') + '"'


def _hex(rgb: tuple[int, int, int]) -> str:
    return '#{:02X}{:02X}{:02X}'.format(*rgb)


def compare_text(
    gold_lines: list[TextLine],
    cand_lines: list[TextLine],
    *,
    aspect: float = 16.0 / 9.0,
    gold_summary: dict[str, Any] | None = None,
    cand_summary: dict[str, Any] | None = None,
) -> TextDiff:
    """Score how faithfully ``cand_lines`` reproduce ``gold_lines``, and describe the gaps.

    Only visible lines take part; hidden output lines are listed in the report
    so the judge (and the logs) can see text that was written but not shown.
    """
    gold_vis = [ln for ln in gold_lines if ln.visible]
    cand_vis = [ln for ln in cand_lines if ln.visible]
    cand_hidden = [ln for ln in cand_lines if not ln.visible]

    g = _explode(gold_vis, aspect)
    c = _explode(cand_vis, aspect)
    detail: dict[str, Any] = {
        'gold_chars': len(g),
        'cand_chars': len(c),
        'hidden_chars': sum(len(normalise(ln.text)) for ln in cand_hidden),
        'truncated': len(g) >= MAX_CHARS or len(c) >= MAX_CHARS,
    }
    if len(g) < MIN_GOLD_CHARS:
        detail['reason'] = 'reference_has_no_visible_text'
        return TextDiff(None, detail, _report_header(detail, None) + _elements_block(gold_summary, cand_summary))

    a = ''.join(ch.ch for ch in g)
    b = ''.join(ch.ch for ch in c)
    pairs: list[tuple[int, int]] = []
    for i, j, k in _greedy_tiles(a, b, MIN_TILE):
        pairs.extend((i + t, j + t) for t in range(k))
    pairs.extend(_single_char_pass(g, c, gold_vis, cand_vis, pairs))

    # Font-size weights: a character's importance grows with its size, clipped
    # so one huge title cannot swamp the slide.
    g_med = _median([ch.size for ch in g]) or 1.0
    c_med = _median([ch.size for ch in c]) or g_med
    wg = [min(3.0, max(0.5, ch.size / g_med)) for ch in g]
    wc = [min(3.0, max(0.5, ch.size / c_med)) for ch in c]

    diag = math.hypot(aspect, 1.0)
    pos_s = size_s = col_s = 0.0
    wsum = 0.0
    g_hit = [0] * len(g)
    c_hit = [0] * len(c)
    for i, j in pairs:
        gi, cj = g[i], c[j]
        g_hit[i] = c_hit[j] = 1
        w = wg[i]
        d = math.hypot(gi.x - cj.x, gi.y - cj.y) / diag
        pos_s += w * math.exp(-d / POS_SCALE)
        size_s += w * math.exp(-abs(math.log(cj.size / gi.size)) / SIZE_SCALE)
        col_s += w * math.exp(-delta_e(gi.color, cj.color) / COLOR_SCALE)
        wsum += w

    recall = sum(w for w, h in zip(wg, g_hit, strict=True) if h) / sum(wg)
    precision = sum(w for w, h in zip(wc, c_hit, strict=True) if h) / sum(wc) if c else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    # F2: recall counts four times precision. In reconstruction the failure that
    # matters is missing content; stray extra text is a lesser defect.
    b2 = CONTENT_BETA**2
    fb = (1 + b2) * recall * precision / (b2 * precision + recall) if recall + precision > 0 else 0.0
    if wsum > 0:
        pos_s, size_s, col_s = pos_s / wsum, size_s / wsum, col_s / wsum
    layout = (pos_s**0.6) * (size_s**0.25) * (col_s**0.15) if wsum > 0 else 0.0
    # Content is the base; how the matched text is laid out scales it down.
    score = fb * (layout**LAYOUT_EXPONENT) if fb > 0 else 0.0

    detail.update({
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
        'position': round(pos_s, 4),
        'size': round(size_s, 4),
        'color': round(col_s, 4),
        'matched_chars': len(pairs),
    })
    items = _report_items(g, c, pairs, gold_vis, cand_vis, cand_hidden, diag)
    detail['n_missing_lines'] = sum(1 for _, kind, _ in items if kind == 'MISSING')
    detail['n_extra_lines'] = sum(1 for _, kind, _ in items if kind == 'EXTRA')
    report = _report_header(detail, score) + _elements_block(gold_summary, cand_summary)
    if items:
        items.sort(key=lambda t: -t[0])
        shown = items[:MAX_REPORT_ITEMS]
        report += '\nDifferences (most important first):\n' + '\n'.join(f'- {kind} {text}' for _, kind, text in shown)
        if len(items) > len(shown):
            report += f'\n- ... {len(items) - len(shown)} smaller differences omitted'
    else:
        report += '\nNo text differences found.'
    return TextDiff(score, detail, report)


def _single_char_pass(
    g: list[_Char], c: list[_Char], gold_vis: list[TextLine], cand_vis: list[TextLine], pairs: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Match one-character lines ("壹", "A") that tiling cannot, nearest same character first."""
    used_g = {i for i, _ in pairs}
    used_c = {j for _, j in pairs}
    extra: list[tuple[int, int]] = []
    for i, gi in enumerate(g):
        if i in used_g or len(normalise(gold_vis[gi.line].text)) != 1:
            continue
        best, best_d = -1, float('inf')
        for j, cj in enumerate(c):
            if j in used_c or cj.ch != gi.ch:
                continue
            d = math.hypot(gi.x - cj.x, gi.y - cj.y)
            if d < best_d:
                best, best_d = j, d
        if best >= 0:
            used_c.add(best)
            extra.append((i, best))
    return extra


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _report_items(
    g: list[_Char],
    c: list[_Char],
    pairs: list[tuple[int, int]],
    gold_vis: list[TextLine],
    cand_vis: list[TextLine],
    cand_hidden: list[TextLine],
    diag: float,
) -> list[tuple[float, str, str]]:
    """(importance, kind, description) for every reportable difference."""
    items: list[tuple[float, str, str]] = []
    g_total: dict[int, int] = {}
    for ch in g:
        g_total[ch.line] = g_total.get(ch.line, 0) + 1
    c_total: dict[int, int] = {}
    for ch in c:
        c_total[ch.line] = c_total.get(ch.line, 0) + 1

    by_gline: dict[int, list[tuple[int, int]]] = {}
    c_line_hits: dict[int, int] = {}
    for i, j in pairs:
        by_gline.setdefault(g[i].line, []).append((i, j))
        c_line_hits[c[j].line] = c_line_hits.get(c[j].line, 0) + 1

    for li, total in g_total.items():
        ln = gold_vis[li]
        weight = total * max(ln.size, 1.0)
        matched = by_gline.get(li, [])
        cov = len(matched) / total
        where = _region(ln.cx, ln.cy)
        size_tag = f'{ln.size:.0f}pt'
        if cov < MISSING_BELOW:
            items.append((weight * 1.0, 'MISSING', f'[{where}, {size_tag}] {_quote(ln.text)}'))
            continue
        # The output line holding most of this reference line's characters.
        counts: dict[int, int] = {}
        for _, j in matched:
            counts[c[j].line] = counts.get(c[j].line, 0) + 1
        out_line = cand_vis[max(counts, key=lambda k: counts[k])]
        if cov < PARTIAL_BELOW:
            items.append((
                weight * (1 - cov),
                'PARTIAL',
                f'[{where}] reference {_quote(ln.text)} | output {_quote(out_line.text)}',
            ))

        dx = sum(c[j].x - g[i].x for i, j in matched) / len(matched)
        dy = sum(c[j].y - g[i].y for i, j in matched) / len(matched)
        shift = math.hypot(dx, dy) / diag
        if shift > MOVED_ABOVE:
            # x was stretched by the aspect ratio in _explode; undo it to report
            # the shift as a fraction of the canvas width.
            aspect = math.sqrt(max(diag * diag - 1.0, 1e-9))
            horiz = f'{"right" if dx > 0 else "left"} {abs(dx) / aspect:.0%}' if abs(dx) / aspect >= 0.01 else ''
            vert = f'{"down" if dy > 0 else "up"} {abs(dy):.0%}' if abs(dy) >= 0.01 else ''
            items.append((
                weight * min(1.0, shift * 5),
                'MOVED',
                f'{_quote(ln.text)} shifted {", ".join(p for p in (horiz, vert) if p)}',
            ))

        ratio = _median([c[j].size for _, j in matched]) / max(ln.size, 0.1)
        if not SIZE_RATIO_OUTSIDE[0] <= ratio <= SIZE_RATIO_OUTSIDE[1]:
            items.append((
                weight * min(1.0, abs(math.log(ratio))),
                'SIZE',
                f'{_quote(ln.text)} reference {ln.size:.0f}pt -> output {out_line.size:.0f}pt',
            ))
        de = delta_e(ln.color, out_line.color)
        if de > COLOR_DE_ABOVE:
            items.append((
                weight * min(1.0, de / 100.0) * 0.5,
                'COLOR',
                f'{_quote(ln.text)} reference {_hex(ln.color)} -> output {_hex(out_line.color)}',
            ))

    for lj, total in c_total.items():
        if c_line_hits.get(lj, 0) / total < MISSING_BELOW:
            ln = cand_vis[lj]
            items.append((
                total * max(ln.size, 1.0) * 0.6,
                'EXTRA',
                f'[{_region(ln.cx, ln.cy)}] output has text not in the reference: {_quote(ln.text)}',
            ))

    for ln in cand_hidden:
        n = len(normalise(ln.text))
        if n:
            items.append((n * 10.0, 'HIDDEN', f'output text not visible ({ln.hidden_reason}): {_quote(ln.text)}'))
    return items


def _report_header(detail: dict[str, Any], score: float | None) -> str:
    if score is None:
        return (
            'Text comparison: not applicable (the reference slide has no visible text). '
            f'Output shows {detail.get("cand_chars", 0)} visible characters.'
        )
    return (
        f'Text comparison (program-measured): recall {detail["recall"]:.0%}, precision {detail["precision"]:.0%} '
        f'(font-size weighted); position {detail["position"]:.2f}, size {detail["size"]:.2f}, colour {detail["color"]:.2f} '
        f'(1.00 = identical). Reference {detail["gold_chars"]} visible chars, output {detail["cand_chars"]}'
        + (
            f', plus {detail["hidden_chars"]} chars that are written but not visible'
            if detail.get('hidden_chars')
            else ''
        )
        + '.'
    )


def _elements_block(gold: dict[str, Any] | None, cand: dict[str, Any] | None) -> str:
    if not gold or not cand:
        return ''
    gk, ck = gold.get('kinds') or {}, cand.get('kinds') or {}
    parts = []
    for kind, label in (('image', 'pictures'), ('table_cell', 'table cells'), ('chart', 'charts')):
        if gk.get(kind) or ck.get(kind):
            parts.append(f'{label} {gk.get(kind, 0)} vs {ck.get(kind, 0)}')
    parts.append(
        f'picture coverage {gold.get("image_coverage", 0):.0%} vs {cand.get("image_coverage", 0):.0%} of the canvas'
    )
    return '\nElements (reference vs output): ' + '; '.join(parts) + '.'


__all__ = ['TextDiff', 'compare_text', 'delta_e', 'normalise']
