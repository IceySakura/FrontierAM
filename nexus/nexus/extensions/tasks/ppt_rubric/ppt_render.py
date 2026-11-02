from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import threading

from pathlib import Path
from typing import Any

from .ppt_metrics import SlideElement, SlideGeometry
from .ppt_pdf_layout import _run_soffice_convert, measure_text_boxes


logger = logging.getLogger(__name__)


EMU_PER_INCH = 914400.0


DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720


_SOFFICE_SEMAPHORE: threading.Semaphore | None = None
_SOFFICE_SEMAPHORE_LOCK = threading.Lock()


def _soffice_semaphore(limit: int) -> threading.Semaphore:
    global _SOFFICE_SEMAPHORE
    with _SOFFICE_SEMAPHORE_LOCK:
        if _SOFFICE_SEMAPHORE is None:
            _SOFFICE_SEMAPHORE = threading.Semaphore(limit)
    return _SOFFICE_SEMAPHORE


def _rgb_of(color_fmt: Any) -> tuple[int, int, int] | None:

    try:
        if color_fmt is None:
            return None

        rgb = getattr(color_fmt, 'rgb', None)
        if rgb is None:
            return None
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:  # noqa: BLE001 - colour is optional metadata
        return None


def _shape_fill_rgb(shape: Any) -> tuple[int, int, int] | None:
    try:
        fill = getattr(shape, 'fill', None)
        if fill is None:
            return None
        if getattr(fill, 'type', None) is None:
            return None
        return _rgb_of(fill.fore_color)
    except Exception:  # noqa: BLE001 - no-fill / theme fill / unsupported
        return None


DEFAULT_STROKE_EMU = 9525.0


def _stroke_width_px(shape: Any, scale: float) -> float:
    try:
        width = shape.line.width
    except Exception:  # noqa: BLE001 - tables and graphic frames carry no line
        return 0.0
    emu = float(width) if width else DEFAULT_STROKE_EMU
    return max(0.0, emu * scale)


def geometry_from_pptx(
    pptx_path: str,
    *,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
    pdf_layout: Any | None = None,
    local_bg_pdf_path: str | None = None,
) -> tuple[SlideGeometry | None, dict]:

    try:
        from pptx import Presentation
        from pptx.util import Emu  # noqa: F401  (import validates the package)
    except ImportError as exc:
        return None, {'geometry_error': f'python-pptx unavailable: {exc}'}

    try:
        prs = Presentation(pptx_path)
    except Exception as exc:  # noqa: BLE001
        return None, {'geometry_error': f'cannot open pptx: {type(exc).__name__}: {exc}'}

    if len(prs.slides) == 0:
        return None, {'geometry_error': 'pptx has no slides'}

    slide = prs.slides[0]

    slide_w_emu = float(prs.slide_width or 0) or (13.333 * EMU_PER_INCH)
    slide_h_emu = float(prs.slide_height or 0) or (7.5 * EMU_PER_INCH)
    sx = render_width / slide_w_emu
    sy = render_height / slide_h_emu

    measured: list[dict] = []
    text_shape_idx: list[int] = []
    pdf_to_px_x = pdf_to_px_y = 1.0

    from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: PLC0415

    from .ppt_structure import (  # noqa: PLC0415
        IDENTITY,
        _box_to_canvas,
        _centipoints,
        _compose,
        _group_transform,
        _has_text_body,
        _run_properties,
        _solid_srgb,
    )

    flat_shapes: list[tuple[Any, Any, int]] = []
    group_counter = [0]

    def _collect(shapes: Any, transform: Any, group_id: int) -> None:
        for shape in shapes:
            try:
                is_group = shape.shape_type == MSO_SHAPE_TYPE.GROUP
            except Exception:  # noqa: BLE001 - graphic frames carry no shape_type
                is_group = False
            if is_group:
                group_counter[0] += 1
                _collect(shape.shapes, _compose(transform, _group_transform(shape)), group_counter[0])
                continue
            flat_shapes.append((shape, transform, group_id))

    _collect(slide.shapes, IDENTITY, 0)

    def _transformed_box_emu(shape: Any, transform: Any) -> tuple[float, float, float, float]:
        try:
            rotation = float(shape.rotation or 0.0)
        except Exception:  # noqa: BLE001 - graphic frames carry no rotation
            rotation = 0.0
        return _box_to_canvas(
            transform, float(shape.left), float(shape.top), float(shape.width or 0), float(shape.height or 0), rotation
        )

    if pdf_layout is not None and getattr(pdf_layout, 'page_width', 0) > 0:
        pdf_sx = pdf_layout.page_width / slide_w_emu
        pdf_sy = pdf_layout.page_height / slide_h_emu
        pdf_to_px_x = render_width / pdf_layout.page_width
        pdf_to_px_y = render_height / pdf_layout.page_height

        boxes_pt: list[tuple[float, float, float, float]] = []
        for i, (shp, transform, _group_id) in enumerate(flat_shapes):
            try:
                if shp.left is None or shp.top is None:
                    continue
                if not (
                    getattr(shp, 'has_text_frame', False)
                    and _has_text_body(getattr(shp, '_element', None))
                    and shp.text_frame.text.strip()
                ):
                    continue
                ex0, ey0, ex1, ey1 = _transformed_box_emu(shp, transform)
                boxes_pt.append((ex0 * pdf_sx, ey0 * pdf_sy, ex1 * pdf_sx, ey1 * pdf_sy))
                text_shape_idx.append(i)
            except Exception:  # noqa: BLE001 - a bad shape must not kill measurement
                continue
        if boxes_pt:
            measured = measure_text_boxes(pdf_layout, boxes_pt)

    measured_by_shape = dict(zip(text_shape_idx, measured, strict=False)) if measured else {}
    pdf_bg = getattr(pdf_layout, 'background_rgb', None) if pdf_layout is not None else None

    elements: list[SlideElement] = []
    n_skipped = 0
    skip_reasons: list[str] = []
    n_measured = 0

    for shape_idx, (shape, transform, group_id) in enumerate(flat_shapes):
        try:
            if shape.left is None or shape.top is None:
                n_skipped += 1
                continue
            ex0, ey0, ex1, ey1 = _transformed_box_emu(shape, transform)
            x = ex0 * sx
            y = ey0 * sy
            w = (ex1 - ex0) * sx
            h = (ey1 - ey0) * sy

            stroke_px = _stroke_width_px(shape, sy)
            if stroke_px > 0 and (w > 0) != (h > 0):
                if h <= 0:
                    y -= stroke_px / 2.0
                    h = stroke_px
                if w <= 0:
                    x -= stroke_px / 2.0
                    w = stroke_px

            kind = 'shape'
            text = ''
            font_px = 0.0
            fg = None
            content_h: float | None = None

            content_w_measured: float | None = None

            if shape.shape_type is not None and 'PICTURE' in str(shape.shape_type):
                kind = 'image'
            elif shape.shape_type == MSO_SHAPE_TYPE.LINE or shape.__class__.__name__ == 'Connector':
                kind = 'line'
            elif getattr(shape, 'has_table', False):
                kind = 'table'
            elif (
                getattr(shape, 'has_text_frame', False)
                and _has_text_body(getattr(shape, '_element', None))
                and shape.text_frame.text.strip()
            ):
                kind = 'text'
                text = shape.text_frame.text
                sizes = []
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        run_properties = _run_properties(run)
                        size_pt = _centipoints(run_properties)
                        if size_pt is not None:
                            sizes.append(size_pt * (render_height / 540.0))
                        if fg is None:
                            fg = _solid_srgb(run_properties)
                font_px = max(sizes) if sizes else 0.0
                content_h = _estimate_text_content_height(text=text, box_width_px=w, font_px=font_px or 18.0)

                m = measured_by_shape.get(shape_idx)
                if m and m.get('n_spans'):
                    if m.get('content_y0') is not None and m.get('content_y1') is not None:
                        content_h = (m['content_y1'] - m['content_y0']) * pdf_to_px_y
                    if m.get('content_x0') is not None and m.get('content_x1') is not None:
                        content_w_measured = (m['content_x1'] - m['content_x0']) * pdf_to_px_x
                    if m.get('max_size'):
                        font_px = float(m['max_size']) * pdf_to_px_y
                    if m.get('colors'):
                        fg = m['colors'][0]
                    n_measured += 1

            bg = _shape_fill_rgb(shape)
            bg_filled = bg is not None
            if bg is None and kind == 'text' and pdf_bg is not None:
                bg = pdf_bg

            elements.append(
                SlideElement(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    kind=kind,
                    text=text,
                    text_len=len(text.strip()),
                    font_size_px=font_px,
                    content_width=content_w_measured if content_w_measured is not None else w,
                    content_height=content_h,
                    fg_rgb=fg,
                    bg_rgb=bg or ((255, 255, 255) if kind == 'text' else None),
                    bg_filled=bg_filled,
                    group_id=group_id,
                    is_decorative=(kind == 'shape' and not text and (w * h) >= 0.55 * render_width * render_height),
                    shape_id=int(getattr(shape, 'shape_id', 0) or 0),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad shape must not kill the slide
            n_skipped += 1
            if skip_reasons.count(type(exc).__name__) == 0:
                logger.warning(
                    '[ppt_rubric] skipped a shape while reading geometry: %s: %s',
                    type(exc).__name__,
                    exc,
                )
            skip_reasons.append(type(exc).__name__)
            continue

    if not elements:
        return None, {
            'geometry_error': 'no readable shapes on slide',
            'n_skipped': n_skipped,
            'skip_reasons': sorted(set(skip_reasons)),
        }

    geo = SlideGeometry(
        elements=elements,
        view_width=float(render_width),
        view_height=float(render_height),
        pdf_layout=pdf_layout,
        slide_aspect=slide_w_emu / slide_h_emu if slide_h_emu > 0 else None,
    )
    diag: dict[str, Any] = {
        'n_elements': len(elements),
        'n_slides': len(prs.slides),
        'slide_aspect': round(slide_w_emu / slide_h_emu, 4),
        'n_text_measured': n_measured,
        'pdf_measured': bool(measured_by_shape),
    }
    if n_skipped:
        diag['n_shapes_skipped'] = n_skipped
        diag['skip_reasons'] = sorted(set(skip_reasons))
    if local_bg_pdf_path:
        diag.update(
            refine_local_backgrounds(geo, local_bg_pdf_path, render_width=render_width, render_height=render_height)
        )
    return geo, diag


DECORATION_INK_DELTA = 8
DECORATION_INK_FRACTION = 0.001
DECORATION_PROBE_EDGE = 320


def _probe_render(pdf_path: str) -> Any:
    import fitz  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        scale = DECORATION_PROBE_EDGE / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    finally:
        doc.close()


def count_inkless_decorations(
    pptx_path: str, geo: SlideGeometry, pdf_path: str, *, timeout: float = 120.0
) -> tuple[int | None, dict[str, Any]]:
    ids = {e.shape_id for e in geo.elements if e.is_decorative and e.shape_id}
    if not ids:
        return 0, {}

    import numpy as np  # noqa: PLC0415

    from .ppt_pdf_layout import pptx_to_pdf  # noqa: PLC0415

    scratch = tempfile.mkdtemp(prefix='ppt-decor-')
    try:
        from pptx import Presentation  # noqa: PLC0415
        from pptx.oxml.ns import qn  # noqa: PLC0415

        prs = Presentation(pptx_path)
        tree = prs.slides[0].shapes._spTree
        removed = 0
        for c_nv_pr in list(tree.iter(qn('p:cNvPr'))):
            if int(c_nv_pr.get('id', '0') or 0) not in ids:
                continue
            shape_element = c_nv_pr.getparent().getparent()
            parent = shape_element.getparent()
            if parent is not None:
                parent.remove(shape_element)
                removed += 1
        stripped = os.path.join(scratch, 'stripped.pptx')
        prs.save(stripped)
        stripped_pdf, pdf_diag = pptx_to_pdf(stripped, scratch, timeout=timeout)
        if not stripped_pdf:
            return None, {'decorative_ink_error': str(pdf_diag.get('pdf_error', 'stripped render failed'))[:120]}
        before = _probe_render(pdf_path).astype(np.int16)
        after = _probe_render(stripped_pdf).astype(np.int16)
        if before.shape != after.shape:
            return None, {'decorative_ink_error': 'probe renders differ in size'}
        changed = float((np.abs(before - after).max(axis=2) > DECORATION_INK_DELTA).mean())
        diag = {'decorative_removed': removed, 'decorative_changed_fraction': round(changed, 5)}
        return (removed if changed <= DECORATION_INK_FRACTION else 0), diag
    except Exception as exc:  # noqa: BLE001 - an observation probe must never fail scoring
        return None, {'decorative_ink_error': f'{type(exc).__name__}: {exc}'[:120]}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def refine_local_backgrounds(
    geo: SlideGeometry,
    pdf_path: str,
    *,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
) -> dict[str, Any]:
    from dataclasses import replace  # noqa: PLC0415

    try:
        import fitz  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        return {'local_bg_error': f'unavailable: {exc}'}
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        page.add_redact_annot(page.rect, fill=False)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,  # type: ignore
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,  # type: ignore
            text=fitz.PDF_REDACT_TEXT_REMOVE,  # type: ignore
        )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(render_width / page.rect.width, render_height / page.rect.height), alpha=False
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    except Exception as exc:  # noqa: BLE001 - backgrounds degrade, they must not fail scoring
        return {'local_bg_error': f'{type(exc).__name__}'}
    finally:
        if doc is not None:
            doc.close()

    height, width = arr.shape[:2]
    refined = 0
    replaced: list[SlideElement] = []
    for e in geo.elements:
        if e.kind != 'text' or e.bg_filled:
            replaced.append(e)
            continue
        x0 = max(0, int(e.x / geo.view_width * width))
        y0 = max(0, int(e.y / geo.view_height * height))
        x1 = min(width, int(math.ceil((e.x + e.width) / geo.view_width * width)))
        y1 = min(height, int(math.ceil((e.y + e.height) / geo.view_height * height)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            replaced.append(e)
            continue
        pixels = arr[y0:y1, x0:x1].reshape(-1, 3)
        quant = (pixels // 16) * 16
        values, counts = np.unique(quant, axis=0, return_counts=True)
        bg = tuple(int(v) for v in values[int(np.argmax(counts))])
        replaced.append(replace(e, bg_rgb=bg))
        refined += 1
    geo.elements = replaced
    return {'local_bg_refined': refined}


def _estimate_text_content_height(*, text: str, box_width_px: float, font_px: float) -> float:

    if box_width_px <= 0 or font_px <= 0:
        return 0.0
    chars_per_line = max(1.0, box_width_px / (font_px * 0.6))
    lines = 0.0
    for para in text.split('\n'):
        n = max(1, len(para.strip()))
        lines += max(1.0, n / chars_per_line)
    return lines * font_px * 1.25


def rasterise_pptx(
    pptx_path: str,
    workdir: str,
    *,
    timeout: float = 60.0,
    concurrency: int = 4,
) -> tuple[bytes | None, dict]:
    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    if not soffice:
        return None, {'raster_error': 'soffice_not_found'}

    sem = _soffice_semaphore(concurrency)
    acquired = sem.acquire(timeout=timeout)
    if not acquired:
        return None, {'raster_error': 'soffice_concurrency_timeout'}

    try:
        render_directory = tempfile.mkdtemp(prefix='ppt-png-', dir=workdir)
        with tempfile.TemporaryDirectory(prefix='lo-profile-') as profile_directory:
            proc, error = _run_soffice_convert(
                [
                    soffice,
                    f'-env:UserInstallation={Path(profile_directory).as_uri()}',
                    '--headless',
                    '--norestore',
                    '--convert-to',
                    'png',
                    '--outdir',
                    render_directory,
                    os.path.abspath(pptx_path),
                ],
                timeout=timeout,
            )
        if proc is None:
            return None, {'raster_error': error}
        if proc.returncode != 0:
            tail = (proc.stderr or '').strip().splitlines()[-2:]
            return None, {
                'raster_error': 'soffice_failed',
                'raster_returncode': proc.returncode,
                'raster_stderr': ' | '.join(tail)[:300],
            }

        png_files = [p for p in Path(render_directory).glob('*.png') if p.stat().st_size > 0]
        if not png_files:
            tail = (proc.stderr or '').strip().splitlines()[-2:]
            return None, {'raster_error': 'soffice_no_output', 'raster_stderr': ' | '.join(tail)[:300]}

        with open(png_files[0], 'rb') as f:
            return f.read(), {}
    except Exception as exc:  # noqa: BLE001
        return None, {'raster_error': f'soffice_exception: {type(exc).__name__}'}
    finally:
        sem.release()
        shutil.rmtree(render_directory, ignore_errors=True)


__all__ = [
    'EMU_PER_INCH',
    'DEFAULT_RENDER_WIDTH',
    'DEFAULT_RENDER_HEIGHT',
    'geometry_from_pptx',
    'rasterise_pptx',
]
