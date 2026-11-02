from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


EMU_PER_POINT = 12700.0


@dataclass
class TextSpan:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    color_rgb: tuple[int, int, int]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass
class PdfTextLayout:
    spans: list[TextSpan] = field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0

    background_rgb: tuple[int, int, int] | None = None


def _int_to_rgb(value: int) -> tuple[int, int, int]:

    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


def _run_soffice_convert(
    command: list[str],
    *,
    timeout: float,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.communicate()
                return None, 'soffice_pdf_timeout'
    except OSError as exc:
        return None, f'soffice_pdf_exception: {type(exc).__name__}'

    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr), None


def pptx_to_pdf(
    pptx_path: str,
    workdir: str,
    *,
    timeout: float = 120.0,
) -> tuple[str | None, dict[str, Any]]:

    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    if not soffice:
        return None, {'pdf_error': 'soffice_not_found'}

    render_directory = tempfile.mkdtemp(prefix='ppt-pdf-', dir=workdir)
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix='lo-profile-') as profile_directory:
            proc, error = _run_soffice_convert(
                [
                    soffice,
                    f'-env:UserInstallation={Path(profile_directory).as_uri()}',
                    '--headless',
                    '--norestore',
                    '--convert-to',
                    'pdf',
                    '--outdir',
                    render_directory,
                    os.path.abspath(pptx_path),
                ],
                timeout=timeout,
            )
        if proc is None:
            return None, {'pdf_error': error}
        if proc.returncode != 0:
            tail = (proc.stderr or '').strip().splitlines()[-2:]
            return None, {
                'pdf_error': 'soffice_pdf_failed',
                'pdf_returncode': proc.returncode,
                'pdf_stderr': ' | '.join(tail)[:300],
            }
        base = os.path.splitext(os.path.basename(pptx_path))[0]
        pdf_path = os.path.join(render_directory, f'{base}.pdf')
        if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) == 0:
            tail = (proc.stderr or '').strip().splitlines()[-2:]
            return None, {'pdf_error': 'soffice_pdf_no_output', 'pdf_stderr': ' | '.join(tail)[:300]}
        completed = True
        return pdf_path, {}
    except Exception as exc:  # noqa: BLE001
        return None, {'pdf_error': f'soffice_pdf_exception: {type(exc).__name__}'}
    finally:
        if not completed:
            shutil.rmtree(render_directory)


def extract_text_layout(pdf_path: str) -> tuple[PdfTextLayout | None, dict[str, Any]]:

    try:
        import pymupdf  # noqa: PLC0415 - lazy: reward workers may lack it
    except ImportError as exc:
        return None, {'pdf_error': f'pymupdf_unavailable: {exc}'}

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return None, {'pdf_error': f'pdf_open_failed: {type(exc).__name__}'}

    try:
        if doc.page_count == 0:
            return None, {'pdf_error': 'pdf_has_no_pages'}
        page = doc[0]

        spans: list[TextSpan] = []
        raw = page.get_text('dict')
        for block in raw.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '')
                    if not text.strip():
                        continue
                    bbox = span.get('bbox') or (0, 0, 0, 0)
                    spans.append(
                        TextSpan(
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            text=text,
                            size=float(span.get('size', 0.0)),
                            color_rgb=_int_to_rgb(int(span.get('color', 0))),
                        )
                    )

        layout = PdfTextLayout(
            spans=spans,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            background_rgb=_sample_background(page),
        )
        return layout, {'pdf_spans': len(spans)}
    finally:
        doc.close()


def _sample_background(page: Any) -> tuple[int, int, int] | None:
    try:
        import pymupdf  # noqa: PLC0415

        pix = page.get_pixmap(matrix=pymupdf.Matrix(0.12, 0.12), alpha=False)
        if pix.width == 0 or pix.height == 0:
            return None
        samples = pix.samples
        counts: dict[tuple[int, int, int], int] = {}
        stride = pix.stride
        n = pix.n
        for y in range(pix.height):
            row = y * stride
            for x in range(pix.width):
                off = row + x * n
                key = (samples[off], samples[off + 1], samples[off + 2])
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]
    except Exception:  # noqa: BLE001 - background is an optimisation, not required
        return None


def _overlap_ratio(
    span: TextSpan,
    box: tuple[float, float, float, float],
) -> float:

    bx0, by0, bx1, by1 = box
    ix0, iy0 = max(span.x0, bx0), max(span.y0, by0)
    ix1, iy1 = min(span.x1, bx1), min(span.y1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = span.width * span.height
    return inter / area if area > 0 else 0.0


def measure_text_boxes(
    layout: PdfTextLayout,
    boxes_pt: list[tuple[float, float, float, float]],
    *,
    assign_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = [
        {
            'content_x0': None,
            'content_y0': None,
            'content_x1': None,
            'content_y1': None,
            'n_spans': 0,
            'min_size': None,
            'max_size': None,
            'colors': [],
        }
        for _ in boxes_pt
    ]
    if not boxes_pt or not layout.spans:
        return results

    for span in layout.spans:
        best_idx = -1
        best_score = 0.0
        cx = (span.x0 + span.x1) / 2.0
        for i, box in enumerate(boxes_pt):
            score = _overlap_ratio(span, box)
            if score == 0.0:
                bx0, _by0, bx1, _by1 = box
                if bx0 <= cx <= bx1:
                    score = 1e-3
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score < 1e-4:
            continue

        r = results[best_idx]
        r['n_spans'] += 1
        r['content_x0'] = span.x0 if r['content_x0'] is None else min(r['content_x0'], span.x0)
        r['content_y0'] = span.y0 if r['content_y0'] is None else min(r['content_y0'], span.y0)
        r['content_x1'] = span.x1 if r['content_x1'] is None else max(r['content_x1'], span.x1)
        r['content_y1'] = span.y1 if r['content_y1'] is None else max(r['content_y1'], span.y1)
        if span.size > 0:
            r['min_size'] = span.size if r['min_size'] is None else min(r['min_size'], span.size)
            r['max_size'] = span.size if r['max_size'] is None else max(r['max_size'], span.size)
        r['colors'].append(span.color_rgb)

    return results


JUDGE_MAX_LONG_EDGE = 2048


def normalise_source_size(source_size: Any) -> tuple[float, float] | None:
    if not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
        return None
    try:
        width, height = float(source_size[0]), float(source_size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def judge_long_edge_px(source_size: Any) -> int | None:
    normalised = normalise_source_size(source_size)
    if normalised is None:
        return None
    return min(int(max(normalised)), JUDGE_MAX_LONG_EDGE)


def fit_image_long_edge(
    png_or_path: Any,
    target_long_edge: int,
) -> tuple[bytes | None, dict[str, Any]]:

    try:
        from PIL import Image
    except ImportError as exc:
        return None, {'fit_error': f'pillow unavailable: {exc}'}

    import io

    try:
        if isinstance(png_or_path, (bytes, bytearray)):
            src = Image.open(io.BytesIO(bytes(png_or_path)))
        else:
            src = Image.open(str(png_or_path))
    except Exception as exc:  # noqa: BLE001
        return None, {'fit_error': f'cannot open image: {type(exc).__name__}: {exc}'}

    try:
        want = min(int(target_long_edge), JUDGE_MAX_LONG_EDGE)
        w, h = src.size
        if w <= 0 or h <= 0:
            return None, {'fit_error': 'image_has_zero_size'}
        long_edge = max(w, h)
        scale = want / long_edge
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))

        img = src.convert('RGB') if src.mode not in ('RGB', 'L') else src
        if new_size != (w, h):
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        png = buf.getvalue()
        return png, {
            'fit_from': [w, h],
            'fit_to': list(new_size),
            'fit_long_edge': max(new_size),
            'fit_png_bytes': len(png),
        }
    except Exception as exc:  # noqa: BLE001
        return None, {'fit_error': f'{type(exc).__name__}: {exc}'}
    finally:
        src.close()


def render_pdf_page_png(
    pdf_path: str,
    *,
    dpi: int = 96,
    target_long_edge: int | None = None,
) -> tuple[bytes | None, dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        return None, {'raster_error': f'pymupdf unavailable: {exc}'}

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return None, {'raster_error': f'cannot open pdf: {type(exc).__name__}: {exc}'}

    try:
        if doc.page_count == 0:
            return None, {'raster_error': 'pdf_has_no_pages'}
        page = doc[0]

        diag: dict[str, Any] = {}
        if target_long_edge is not None and target_long_edge > 0:
            want = min(int(target_long_edge), JUDGE_MAX_LONG_EDGE)
            if want != int(target_long_edge):
                diag['raster_long_edge_capped'] = [int(target_long_edge), want]
            rect = page.rect
            page_long = max(rect.width, rect.height)
            if page_long <= 0:
                return None, {'raster_error': 'pdf_page_has_zero_size'}
            zoom = want / page_long
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            diag['raster_target_long_edge'] = want
        else:
            rect = page.rect
            page_long = max(rect.width, rect.height)
            est_long_edge = int(page_long * dpi / 72.0)
            if est_long_edge > JUDGE_MAX_LONG_EDGE and page_long > 0:
                zoom = JUDGE_MAX_LONG_EDGE / page_long
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                diag['raster_dpi_capped'] = [dpi, int(dpi * zoom)]
            else:
                pix = page.get_pixmap(dpi=dpi)
                diag['raster_dpi'] = dpi

        png = pix.tobytes('png')
        long_edge = max(pix.width, pix.height)
        diag.update({
            'raster_png_bytes': len(png),
            'raster_size': [pix.width, pix.height],
            'raster_long_edge': long_edge,
        })
        if long_edge > JUDGE_MAX_LONG_EDGE:
            diag['raster_error'] = f'long_edge {long_edge} exceeds {JUDGE_MAX_LONG_EDGE}'
            return None, diag
        return png, diag
    except Exception as exc:  # noqa: BLE001
        diag['raster_error'] = f'{type(exc).__name__}: {exc}'
        return None, diag
    finally:
        with contextlib.suppress(Exception):
            doc.close()


__all__ = [
    'TextSpan',
    'PdfTextLayout',
    'pptx_to_pdf',
    'extract_text_layout',
    'measure_text_boxes',
    'render_pdf_page_png',
    'fit_image_long_edge',
    'JUDGE_MAX_LONG_EDGE',
    'EMU_PER_POINT',
]
