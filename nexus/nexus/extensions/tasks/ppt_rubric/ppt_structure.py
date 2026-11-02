from __future__ import annotations

import json
import math

from dataclasses import dataclass, field
from typing import Any


EMU_PER_PT = 12700.0
REFERENCE_CANVAS_HEIGHT_PT = 540.0


Affine = tuple[float, float, float, float, float, float]
IDENTITY: Affine = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _compose(outer: Affine, inner: Affine) -> Affine:
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + b1 * d2,
        a1 * b2 + b1 * e2,
        a1 * c2 + b1 * f2 + c1,
        d1 * a2 + e1 * d2,
        d1 * b2 + e1 * e2,
        d1 * c2 + e1 * f2 + f1,
    )


def _apply(m: Affine, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + b * y + c, d * x + e * y + f


def _rotation_about(cx: float, cy: float, degrees: float, flip_h: bool = False, flip_v: bool = False) -> Affine:
    sx = -1.0 if flip_h else 1.0
    sy = -1.0 if flip_v else 1.0
    t = math.radians(degrees or 0.0)
    cos_t, sin_t = math.cos(t), math.sin(t)
    a, b = cos_t * sx, -sin_t * sy
    d, e = sin_t * sx, cos_t * sy
    return (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy)


def _group_transform(group: Any) -> Affine:
    from pptx.oxml.ns import qn  # noqa: PLC0415

    try:
        xfrm = group._element.grpSpPr.find(qn('a:xfrm'))
    except Exception:  # noqa: BLE001
        xfrm = None
    if xfrm is None:
        return IDENTITY

    def _xy(tag: str, kx: str, ky: str) -> tuple[float, float] | None:
        node = xfrm.find(qn(tag))
        if node is None:
            return None
        try:
            return float(node.get(kx, 0)), float(node.get(ky, 0))
        except (TypeError, ValueError):
            return None

    off = _xy('a:off', 'x', 'y') or (0.0, 0.0)
    ext = _xy('a:ext', 'cx', 'cy') or (0.0, 0.0)
    ch_off = _xy('a:chOff', 'x', 'y') or off
    ch_ext = _xy('a:chExt', 'cx', 'cy') or ext

    sx = ext[0] / ch_ext[0] if ch_ext[0] else 1.0
    sy = ext[1] / ch_ext[1] if ch_ext[1] else 1.0
    scale: Affine = (sx, 0.0, off[0] - ch_off[0] * sx, 0.0, sy, off[1] - ch_off[1] * sy)

    rot = float(xfrm.get('rot', 0) or 0) / 60000.0
    flip_h = xfrm.get('flipH') in ('1', 'true')
    flip_v = xfrm.get('flipV') in ('1', 'true')
    if not rot and not flip_h and not flip_v:
        return scale
    cx, cy = off[0] + ext[0] / 2.0, off[1] + ext[1] / 2.0
    return _compose(_rotation_about(cx, cy, rot, flip_h, flip_v), scale)


def _box_to_canvas(
    m: Affine, left: float, top: float, width: float, height: float, rotation: float
) -> tuple[float, float, float, float]:
    cx, cy = left + width / 2.0, top + height / 2.0
    own = _rotation_about(cx, cy, rotation) if rotation else IDENTITY
    full = _compose(m, own)
    pts = [
        _apply(full, x, y)
        for x, y in ((left, top), (left + width, top), (left, top + height), (left + width, top + height))
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass
class DeckElement:
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ''
    font_pt: float | None = None
    color: tuple[int, int, int] | None = None
    bold: bool = False
    fill: tuple[int, int, int] | None = None
    z: int = 0
    depth: int = 0
    rotation: float = 0.0
    stroke_pt: float | None = None
    dash: str = ''
    stroke_color: tuple[int, int, int] | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass
class DeckStructure:
    elements: list[DeckElement] = field(default_factory=list)
    slide_aspect: float | None = None
    n_skipped: int = 0
    max_depth: int = 0
    error: str | None = None

    def texts(self) -> list[str]:
        return [e.text for e in self.elements if e.text.strip()]

    def text_chars(self) -> int:
        return sum(len(''.join(e.text.split())) for e in self.elements)


def _rgb(color_format: Any) -> tuple[int, int, int] | None:
    try:
        from pptx.enum.dml import MSO_COLOR_TYPE  # noqa: PLC0415

        if color_format is None or color_format.type != MSO_COLOR_TYPE.RGB:
            return None
        rgb = color_format.rgb
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    except Exception:  # noqa: BLE001 - theme / inherited / absent colour
        return None


def _fill_rgb(shape: Any) -> tuple[int, int, int] | None:
    try:
        fill = shape.fill
        if fill.type is None:
            return None
        return _rgb(fill.fore_color)
    except Exception:  # noqa: BLE001 - no fill / unsupported fill
        return None


def _solid_srgb(element: Any) -> tuple[int, int, int] | None:
    if element is None:
        return None
    from pptx.oxml.ns import qn  # noqa: PLC0415

    solid = element.find(qn('a:solidFill'))
    srgb = solid.find(qn('a:srgbClr')) if solid is not None else None
    value = srgb.get('val', '') if srgb is not None else ''
    if len(value) != 6:
        return None
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except ValueError:
        return None


def _centipoints(element: Any) -> float | None:
    size = element.get('sz') if element is not None else None
    if not size:
        return None
    try:
        return int(size) / 100.0
    except ValueError:
        return None


def _has_text_body(element: Any) -> bool:
    if element is None:
        return False
    from pptx.oxml.ns import qn  # noqa: PLC0415

    return element.find(qn('p:txBody')) is not None or element.find(qn('a:txBody')) is not None


def _run_properties(run: Any) -> Any:
    from pptx.oxml.ns import qn  # noqa: PLC0415

    return run._r.find(qn('a:rPr'))


def _shape_line_element(shape: Any) -> Any:
    element = getattr(shape, '_element', None)
    if element is None:
        return None
    from pptx.oxml.ns import qn  # noqa: PLC0415

    sp_pr = element.find(qn('p:spPr'))
    return sp_pr.find(qn('a:ln')) if sp_pr is not None else None


def _line_props(shape: Any) -> tuple[float | None, str, tuple[int, int, int] | None]:
    try:
        line = shape.line
    except Exception:  # noqa: BLE001 - tables and graphic frames carry no line
        return None, '', None
    width = None
    try:
        if line.width:
            width = round(float(line.width.pt), 2)
    except Exception:  # noqa: BLE001
        width = None
    dash = ''
    try:
        if line.dash_style is not None:
            dash = str(line.dash_style).split('.')[-1].split(' ')[0].lower()
    except Exception:  # noqa: BLE001 - unsupported or inherited dash style
        dash = ''
    return width, dash, _solid_srgb(_shape_line_element(shape))


def _has_picture_fill(shape: Any) -> bool:
    try:
        return bool(shape._element.xpath('./p:spPr/a:blipFill'))
    except Exception:  # noqa: BLE001
        return False


def _text_props(text_frame: Any, size_scale: float) -> tuple[str, float | None, tuple[int, int, int] | None, bool]:
    lines: list[str] = []
    sizes: list[float] = []
    color = None
    bold = False
    from pptx.oxml.ns import qn  # noqa: PLC0415

    for para in text_frame.paragraphs:
        paragraph_properties = para._p.find(qn('a:pPr'))
        default_pt = _centipoints(
            paragraph_properties.find(qn('a:defRPr')) if paragraph_properties is not None else None
        )
        parts = []
        for run in para.runs:
            parts.append(run.text or '')
            run_properties = _run_properties(run)
            size_pt = _centipoints(run_properties) or default_pt
            if size_pt is not None:
                sizes.append(size_pt * size_scale)
            if color is None:
                color = _solid_srgb(run_properties)
            if run_properties is not None and run_properties.get('b') in ('1', 'true'):
                bold = True
        lines.append(''.join(parts) if para.runs else (para.text or ''))
    text = '\n'.join(lines).strip()
    return text, (max(sizes) if sizes else None), color, bold


def extract_elements(pptx_path: str) -> DeckStructure:
    try:
        from pptx import Presentation  # noqa: PLC0415
        from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: PLC0415
    except ImportError as exc:
        return DeckStructure(error=f'python-pptx unavailable: {exc}')

    try:
        prs = Presentation(pptx_path)
    except Exception as exc:  # noqa: BLE001
        return DeckStructure(error=f'cannot open pptx: {type(exc).__name__}: {exc}'[:200])
    if len(prs.slides) == 0:
        return DeckStructure(error='pptx has no slides')

    slide = prs.slides[0]
    sw = float(prs.slide_width or 0) or 13.333 * 914400.0
    sh = float(prs.slide_height or 0) or 7.5 * 914400.0
    size_scale = REFERENCE_CANVAS_HEIGHT_PT / (sh / EMU_PER_PT)
    out = DeckStructure(slide_aspect=sw / sh if sh else None)
    z_counter = [0]

    def norm(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return box[0] / sw, box[1] / sh, box[2] / sw, box[3] / sh

    def emit(kind: str, box: tuple[float, float, float, float], depth: int, rotation: float, **kw: Any) -> None:
        x0, y0, x1, y1 = norm(box)
        out.elements.append(
            DeckElement(kind=kind, x0=x0, y0=y0, x1=x1, y1=y1, z=z_counter[0], depth=depth, rotation=rotation, **kw)
        )
        z_counter[0] += 1

    def visit(shapes: Any, m: Affine, depth: int) -> None:
        out.max_depth = max(out.max_depth, depth)
        for shape in shapes:
            try:
                st = shape.shape_type
            except Exception:  # noqa: BLE001
                st = None
            try:
                if st == MSO_SHAPE_TYPE.GROUP:
                    visit(shape.shapes, _compose(m, _group_transform(shape)), depth + 1)
                    continue
                if shape.left is None or shape.top is None:
                    out.n_skipped += 1
                    continue
                left, top = float(shape.left), float(shape.top)
                width, height = float(shape.width or 0), float(shape.height or 0)
                try:
                    rotation = float(shape.rotation or 0.0)
                except Exception:  # noqa: BLE001 - graphic frames carry no rotation
                    rotation = 0.0
                box = _box_to_canvas(m, left, top, width, height, rotation)

                if getattr(shape, 'has_table', False) and shape.has_table:
                    _emit_table(shape, m, left, top, depth, emit, size_scale)
                    continue
                if getattr(shape, 'has_chart', False) and shape.has_chart:
                    emit('chart', box, depth, rotation)
                    continue
                is_picture = st == MSO_SHAPE_TYPE.PICTURE or _has_picture_fill(shape)
                if not is_picture and st == MSO_SHAPE_TYPE.PLACEHOLDER:
                    is_picture = hasattr(shape, 'image') and _safe_has_image(shape)
                if is_picture:
                    emit('image', box, depth, rotation)
                if getattr(shape, 'has_text_frame', False) and _has_text_body(getattr(shape, '_element', None)):
                    text, font_pt, color, bold = _text_props(shape.text_frame, size_scale)
                    if text:
                        emit(
                            'text',
                            box,
                            depth,
                            rotation,
                            text=text,
                            font_pt=font_pt,
                            color=color,
                            bold=bold,
                            fill=_fill_rgb(shape),
                        )
                        continue
                if is_picture:
                    continue
                if st in (MSO_SHAPE_TYPE.LINE,) or shape.__class__.__name__ == 'Connector':
                    stroke_pt, dash, stroke_color = _line_props(shape)
                    emit('line', box, depth, rotation, stroke_pt=stroke_pt, dash=dash, stroke_color=stroke_color)
                    continue
                emit('shape', box, depth, rotation, fill=_fill_rgb(shape))
            except Exception:  # noqa: BLE001 - one bad shape must not drop the slide
                out.n_skipped += 1
                continue

    visit(slide.shapes, IDENTITY, 0)
    return out


def _safe_has_image(shape: Any) -> bool:
    try:
        return shape.image is not None
    except Exception:  # noqa: BLE001
        return False


def _emit_table(shape: Any, m: Affine, left: float, top: float, depth: int, emit: Any, size_scale: float) -> None:
    table = shape.table
    col_x = [left]
    for col in table.columns:
        col_x.append(col_x[-1] + float(col.width or 0))
    row_y = [top]
    for row in table.rows:
        row_y.append(row_y[-1] + float(row.height or 0))
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row.cells):
            if cell.is_spanned:
                continue
            c1 = min(len(col_x) - 1, c + (cell.span_width if cell.is_merge_origin else 1))
            r1 = min(len(row_y) - 1, r + (cell.span_height if cell.is_merge_origin else 1))
            box = _box_to_canvas(m, col_x[c], row_y[r], col_x[c1] - col_x[c], row_y[r1] - row_y[r], 0.0)
            text, font_pt, color, bold = (
                _text_props(cell.text_frame, size_scale) if _has_text_body(cell._tc) else ('', None, None, False)
            )
            emit('table_cell', box, depth, 0.0, text=text, font_pt=font_pt, color=color, bold=bold)


def coverage_of(elements: list[DeckElement], kind: str, grid: tuple[int, int] = (64, 36)) -> float:
    gx, gy = grid
    hit = 0
    boxes = [(e.x0, e.y0, e.x1, e.y1) for e in elements if e.kind == kind]
    if not boxes:
        return 0.0
    for j in range(gy):
        cy = (j + 0.5) / gy
        for i in range(gx):
            cx = (i + 0.5) / gx
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes):
                hit += 1
    return hit / float(gx * gy)


def summarise(structure: DeckStructure) -> dict[str, Any]:
    els = structure.elements
    kinds: dict[str, int] = {}
    for e in els:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {
        'n_elements': len(els),
        'kinds': kinds,
        'image_coverage': round(coverage_of(els, 'image'), 4),
        'max_group_depth': structure.max_depth,
        'n_skipped': structure.n_skipped,
        'text_chars': structure.text_chars(),
    }


@dataclass
class TextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    color: tuple[int, int, int] = (0, 0, 0)
    visible: bool = True
    hidden_reason: str = ''
    invisible_text: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _int_rgb(value: int) -> tuple[int, int, int]:
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


RENDER_HEIGHT_PX = 720
INK_DIFF = 24
MIN_INK_FRACTION = 0.01
MIN_INK_PIXELS = 6
PDF_INVISIBLE_TEXT_FLAG = 16


@dataclass
class PageRenders:
    full: Any
    notext: Any


def render_pair(pdf_path: str, *, height_px: int = RENDER_HEIGHT_PX) -> tuple[PageRenders | None, dict[str, Any]]:
    try:
        import fitz  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        return None, {'render_pair_error': f'unavailable: {exc}'}

    def _render(remove_text: bool) -> Any:
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            if remove_text:
                page.add_redact_annot(page.rect, fill=False)
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,  # type: ignore
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,  # type: ignore
                    text=fitz.PDF_REDACT_TEXT_REMOVE,  # type: ignore
                )
            zoom = float(height_px) / float(page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        finally:
            doc.close()

    try:
        full = _render(False)
        notext = _render(True)
    except Exception as exc:  # noqa: BLE001 - visibility degrades, it must not fail scoring
        return None, {'render_pair_error': f'{type(exc).__name__}: {exc}'[:200]}
    if full.shape != notext.shape:
        return None, {'render_pair_error': f'shape mismatch {full.shape} vs {notext.shape}'}
    return PageRenders(full=full, notext=notext), {}


def extract_text_lines(
    pdf_path: str,
    *,
    check_visibility: bool = True,
    renders: PageRenders | None = None,
) -> tuple[list[TextLine] | None, dict[str, Any]]:
    try:
        import fitz  # noqa: PLC0415 - PyMuPDF
    except ImportError as exc:
        return None, {'lines_error': f'pymupdf unavailable: {exc}'}
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return None, {'lines_error': f'cannot open pdf: {type(exc).__name__}'}
    try:
        if doc.page_count == 0:
            return None, {'lines_error': 'pdf has no pages'}
        page = doc[0]
        pw, ph = float(page.rect.width), float(page.rect.height)
        if pw <= 0 or ph <= 0:
            return None, {'lines_error': 'pdf page has zero size'}
        size_scale = REFERENCE_CANVAS_HEIGHT_PT / ph

        lines: list[TextLine] = []
        n_invisible_text = 0
        for block in page.get_text('dict').get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                spans = [s for s in line.get('spans', []) if (s.get('text') or '').strip()]
                if not spans:
                    continue
                text = ''.join(s.get('text', '') for s in spans).strip()
                main = max(spans, key=lambda s: len((s.get('text') or '').strip()))
                bx = line.get('bbox') or main.get('bbox') or (0, 0, 0, 0)
                invisible_text = any((s.get('flags') or 0) & PDF_INVISIBLE_TEXT_FLAG for s in spans)
                n_invisible_text += int(invisible_text)
                lines.append(
                    TextLine(
                        text=text,
                        x0=float(bx[0]) / pw,
                        y0=float(bx[1]) / ph,
                        x1=float(bx[2]) / pw,
                        y1=float(bx[3]) / ph,
                        size=max(float(s.get('size', 0.0)) for s in spans) * size_scale,
                        color=_int_rgb(int(main.get('color', 0))),
                        invisible_text=invisible_text,
                    )
                )

        diag: dict[str, Any] = {'n_lines': len(lines), 'n_invisible_text': n_invisible_text}
    finally:
        doc.close()

    if check_visibility and lines:
        if renders is None:
            renders, rdiag = render_pair(pdf_path)
            diag.update(rdiag)
        _mark_visibility(lines, renders, diag)
    return lines, diag


def _mark_visibility(lines: list[TextLine], renders: PageRenders | None, diag: dict[str, Any]) -> None:
    import numpy as np  # noqa: PLC0415

    ink = None
    if renders is not None:
        ink = np.abs(renders.full.astype(np.int16) - renders.notext.astype(np.int16)).max(axis=2) > INK_DIFF
    else:
        diag['visibility_degraded'] = 'no_renders'

    n_hidden = 0
    for ln in lines:
        reason = ''
        if not (0.0 <= ln.cx <= 1.0 and 0.0 <= ln.cy <= 1.0):
            reason = 'off_canvas'
        elif ln.invisible_text:
            pass
        elif ink is not None:
            h, w = ink.shape
            x0, x1 = max(0, int(ln.x0 * w)), min(w, int(math.ceil(ln.x1 * w)))
            y0, y1 = max(0, int(ln.y0 * h)), min(h, int(math.ceil(ln.y1 * h)))
            if x1 > x0 and y1 > y0:
                box = ink[y0:y1, x0:x1]
                hits = int(box.sum())
                min_ink_pixels = MIN_INK_PIXELS * (h / RENDER_HEIGHT_PX) ** 2
                if hits < min_ink_pixels or hits < MIN_INK_FRACTION * box.size:
                    reason = 'no_ink'
        if reason:
            ln.visible = False
            ln.hidden_reason = reason
            n_hidden += 1
    diag['n_hidden_lines'] = n_hidden


BAKED_BLUR_PX = 3.0
BAKED_MIN_CORR = 0.5
BAKED_ENERGY_FRACTION = 0.3
MIN_GLYPH_PIXELS_PER_CHAR = 6


def baked_text_ratio(gold_lines: list[TextLine], ref: PageRenders, cand: PageRenders) -> float | None:
    import numpy as np  # noqa: PLC0415

    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None

    h, w = ref.full.shape[:2]
    blur_px = BAKED_BLUR_PX * (h / RENDER_HEIGHT_PX)
    min_glyph_pixels = MIN_GLYPH_PIXELS_PER_CHAR * (h / RENDER_HEIGHT_PX) ** 2
    cand_notext = cand.notext
    if cand_notext.shape != ref.full.shape:
        cand_notext = cv2.resize(cand_notext, (w, h), interpolation=cv2.INTER_AREA)

    def gray(a: Any) -> Any:
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)

    g_full, g_bg, g_out = gray(ref.full), gray(ref.notext), gray(cand_notext)
    glyph_sig = cv2.GaussianBlur(g_full - g_bg, (0, 0), blur_px)
    added_sig = cv2.GaussianBlur(g_out - g_bg, (0, 0), blur_px)
    glyph_mask = np.abs(g_full - g_bg) > INK_DIFF

    baked = total = 0.0
    for ln in gold_lines:
        if not ln.visible:
            continue
        chars = ln.text.strip()
        if not chars:
            continue
        lx0, lx1 = ln.x0 * w, ln.x1 * w
        y0, y1 = max(0, int(ln.y0 * h)), min(h, int(math.ceil(ln.y1 * h)))
        cw = (lx1 - lx0) / len(chars)
        for k, ch in enumerate(chars):
            if not _is_word_char(ch):
                continue
            x0 = max(0, int(lx0 + k * cw))
            x1 = min(w, int(math.ceil(lx0 + (k + 1) * cw)))
            if x1 - x0 < 2 or int(glyph_mask[y0:y1, x0:x1].sum()) < min_glyph_pixels:
                continue
            total += 1.0
            a = glyph_sig[y0:y1, x0:x1].ravel()
            b = added_sig[y0:y1, x0:x1].ravel()
            if a.std() < 1e-3 or b.std() < 1e-3:
                continue
            da, db = a - a.mean(), b - b.mean()
            slope = float(np.dot(da, db) / max(1e-6, float(np.dot(da, da))))
            if slope >= BAKED_ENERGY_FRACTION and float(np.corrcoef(a, b)[0, 1]) >= BAKED_MIN_CORR:
                baked += 1.0
    return baked / total if total else None


def _is_word_char(ch: str) -> bool:
    import unicodedata  # noqa: PLC0415

    return unicodedata.category(ch)[0] in ('L', 'N')


def lines_from_elements(structure: DeckStructure) -> list[TextLine]:
    lines: list[TextLine] = []
    for e in structure.elements:
        paras = [p.strip() for p in e.text.split('\n') if p.strip()]
        if not paras:
            continue
        step = (e.y1 - e.y0) / len(paras)
        for k, para in enumerate(paras):
            lines.append(
                TextLine(
                    text=para,
                    x0=e.x0,
                    y0=e.y0 + k * step,
                    x1=e.x1,
                    y1=e.y0 + (k + 1) * step,
                    size=e.font_pt or 18.0,
                    color=e.color or (0, 0, 0),
                )
            )
    return lines


def render_structure_listing(structure: DeckStructure, *, visible_chars: int | None = None) -> str:
    lines = []
    if visible_chars is not None and visible_chars >= 0:
        lines.append(
            f'# rendered visible characters: {visible_chars}; the elements below are read from the '
            'file, so anything hidden or covered by another element is still listed'
        )
    for e in sorted(structure.elements, key=lambda el: el.z):
        box = f'({e.x0:.3f}, {e.y0:.3f}) {e.x1 - e.x0:.3f}x{e.y1 - e.y0:.3f}'
        attrs = []
        if e.font_pt is not None:
            attrs.append(f'{e.font_pt:.0f}pt')
        if e.color is not None:
            attrs.append(f'#{e.color[0]:02X}{e.color[1]:02X}{e.color[2]:02X}')
        if e.fill is not None:
            attrs.append(f'fill#{e.fill[0]:02X}{e.fill[1]:02X}{e.fill[2]:02X}')
        if e.stroke_pt is not None:
            attrs.append(f'stroke{e.stroke_pt:g}pt')
        if e.dash:
            attrs.append(f'dash:{e.dash}')
        if e.stroke_color is not None:
            attrs.append(f'line#{e.stroke_color[0]:02X}{e.stroke_color[1]:02X}{e.stroke_color[2]:02X}')
        if e.rotation:
            attrs.append(f'rot{e.rotation:.0f}')
        if e.depth:
            attrs.append(f'depth{e.depth}')
        text = f' {json.dumps(e.text, ensure_ascii=False)}' if e.text else ''
        lines.append(f'[{e.z}] {e.kind} {box} ' + ' '.join(attrs) + text)
    return '\n'.join(lines)


__all__ = [
    'DeckElement',
    'DeckStructure',
    'PageRenders',
    'TextLine',
    'baked_text_ratio',
    'coverage_of',
    'extract_elements',
    'extract_text_lines',
    'render_structure_listing',
    'lines_from_elements',
    'render_pair',
    'summarise',
]
