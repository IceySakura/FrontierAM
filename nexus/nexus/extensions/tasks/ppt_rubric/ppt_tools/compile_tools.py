"Slide compilation tool: model-authored code -> pptx -> PNG."

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .code_tool import WORKDIR_KEY, _IMAGE_EXTS
from .image_tools import (
    IMAGE_KEY,
    RENDER_KEY,
    TOOL_RECORDS_KEY,
    _image_note,
    _img_return_suffix,
)


logger = logging.getLogger(__name__)


def _persist_deck(pptx_path: str, context: ToolContext, deck_filename: str) -> str | None:
    work = context.tool_state.get(WORKDIR_KEY)
    if not isinstance(work, str) or not os.path.isdir(work):
        return None
    target = os.path.join(work, deck_filename)
    try:
        shutil.copyfile(pptx_path, target)
    except OSError:
        logger.warning('could not persist the built deck to %s', target, exc_info=True)
        return None
    return target


DEFAULT_PAGE_W_PT = 960.0
DEFAULT_PAGE_H_PT = 540.0
RASTER_DPI = 96

CODE_TIMEOUT_S = 120
SOFFICE_TIMEOUT_S = 180
RASTER_TIMEOUT_S = 120

_MAX_PAGES_RETURNED = 4


class RenderInput(BaseModel):
    code: str = Field(description='The source that produces the slide.')
    language: str = Field(
        default='auto',
        description=(
            "Usually leave as 'auto' and it is detected from the code. "
            "Set explicitly to 'python-pptx', 'pptxgenjs' or 'slideir' to override."
        ),
    )


def _detect_language(code: str) -> str:
    stripped = _strip_code_fence(code).lstrip()
    if stripped.startswith('{'):
        return 'slideir'
    js_markers = (
        'require(',
        'import pptxgenjs',
        "from 'pptxgenjs'",
        'from "pptxgenjs"',
        'new pptxgen',
        'const ',
        'let ',
    )
    py_markers = ('from pptx', 'import pptx', 'Presentation(', 'Inches(', 'Pt(')
    js_hits = sum(1 for m in js_markers if m in stripped)
    py_hits = sum(1 for m in py_markers if m in stripped)
    if js_hits > py_hits:
        return 'pptxgenjs'
    return 'python-pptx'


def _strip_code_fence(code: str) -> str:
    match = re.match(r'^\s*```[^\n]*\n(.*?)```\s*$', code, re.DOTALL)
    return match.group(1) if match else code.strip()


async def _run(cmd: list[str], cwd: str, timeout: int, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(env or {})},
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await proc.wait()
        return 124, '', f"timed out after {timeout}s (killpg'd process group)"
    return proc.returncode or 0, out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace')


def _product_is_complete(pptx_path: str) -> tuple[bool, str]:
    try:
        from pptx import Presentation

        return (len(Presentation(pptx_path).slides) > 0), ''
    except Exception as exc:  # noqa: BLE001 - any failure to open IS the answer
        return False, f'{type(exc).__name__}: {exc}'


async def _build_pptx(code: str, language: str, work: str, assets_dir: str | None = None) -> tuple[str | None, str]:
    if language == 'python-pptx':
        script = os.path.join(work, 'build.py')
        with open(script, 'w', encoding='utf-8') as handle:
            handle.write(code)
        rc, out, err = await _run([sys.executable, script], work, CODE_TIMEOUT_S)

    elif language == 'pptxgenjs':
        script = os.path.join(work, 'build.ts')
        with open(script, 'w', encoding='utf-8') as handle:
            handle.write(code)
        node_path = os.environ.get('NODE_PATH', '/usr/local/lib/node_modules')
        rc, out, err = await _run(['npx', '--yes', 'tsx', script], work, CODE_TIMEOUT_S, env={'NODE_PATH': node_path})

    elif language == 'slideir':
        try:
            ir = json.loads(_strip_code_fence(code))
        except json.JSONDecodeError as exc:
            return None, f'SlideIR is not valid JSON: {exc}'
        ir_path, ir_note = _build_from_slideir(ir, work, assets_dir=assets_dir)
        if ir_path is None:
            return None, ir_note
        complete, why = _product_is_complete(ir_path)
        if not complete:
            return None, f'the converter wrote a .pptx that will not open: {why}\n{ir_note}'
        return ir_path, ir_note

    else:
        return None, f"unknown language '{language}'. Use 'python-pptx', 'pptxgenjs' or 'slideir'."

    log = (out or '') + (('\n' + err) if err else '')
    candidates = sorted(f for f in os.listdir(work) if f.endswith('.pptx'))

    if rc != 0:
        residual = f' (a .pptx was left on disk by the crashed script: {candidates[0]})' if candidates else ''
        return None, f'the build script exited {rc}{residual}.\n{log[-2000:]}'

    if not candidates:
        return None, f'no .pptx was produced (exit {rc}).\n{log[-2000:]}'

    pptx_path = os.path.join(work, candidates[0])
    complete, why = _product_is_complete(pptx_path)
    if not complete:
        return None, f'a .pptx was written but it will not open: {why}\n{log[-2000:]}'

    return pptx_path, log[-2000:]


def _load_ir_converter():
    import importlib.util

    default_dir = '/apdcephfs_gy8/share_305933007/fitzfanyang/visual2code/render/ppt_rich'
    harness_dir = os.environ.get('SLIDEIR_HARNESS', default_dir)
    module_path = os.path.join(harness_dir, 'ir_harness.py')
    if not os.path.exists(module_path):
        return None, f'ir_harness.py not found at {module_path}. Set SLIDEIR_HARNESS.'
    try:
        spec = importlib.util.spec_from_file_location('_slideir_harness', module_path)
        if spec is None or spec.loader is None:
            return None, f'could not load a module spec from {module_path}'
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure to the model
        return None, f'failed to import ir_harness: {type(exc).__name__}: {exc}'
    converter = getattr(module, 'ir_to_pptx', None)
    if converter is None:
        return None, 'ir_harness.py has no ir_to_pptx function'
    return converter, ''


def _build_from_slideir(ir: dict[str, Any], work: str, assets_dir: str | None = None) -> tuple[str | None, str]:
    converter, why = _load_ir_converter()
    if converter is None:
        return None, f'{why}\nAnswer in python-pptx or pptxgenjs instead, or configure the harness.'

    page_size = ir.get('pageSize') or {'width': DEFAULT_PAGE_W_PT, 'height': DEFAULT_PAGE_H_PT}
    out_path = os.path.join(work, 'slide.pptx')
    try:
        unsupported = converter(ir, out_path, assets_dir=assets_dir, page_size=page_size)
    except Exception as exc:  # noqa: BLE001 - surface converter errors to the model
        return None, f'SlideIR conversion failed: {type(exc).__name__}: {exc}'

    if not os.path.exists(out_path):
        return None, 'the converter reported success but wrote no .pptx'

    note = ''
    if unsupported:
        try:
            reasons: dict[str, int] = {}
            for item in unsupported:
                key = item.get('why', 'unknown') if isinstance(item, dict) else str(item)
                reasons[key] = reasons.get(key, 0) + 1
            summary = ', '.join(f'{k} x{v}' for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
            note = f'{len(unsupported)} element(s) could not be rendered by the converter: {summary}'
        except (TypeError, AttributeError):
            note = f'converter reported unsupported elements: {unsupported}'
    return out_path, note


def _png_is_complete(png_path: str) -> tuple[bool, str]:
    try:
        with Image.open(png_path) as probe:
            probe.load()
            if probe.size[0] <= 0 or probe.size[1] <= 0:
                return False, f'degenerate size {probe.size}'
        return True, ''
    except Exception as exc:  # noqa: BLE001 - any failure to decode IS the answer
        return False, f'{type(exc).__name__}: {exc}'


def _pdf_declared_pages(pdf_path: str) -> int | None:
    try:
        proc = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith('Pages:'):
            try:
                return int(line.split(':', 1)[1].strip())
            except ValueError:
                return None
    return None


async def _pptx_to_pngs(pptx_path: str, work: str) -> tuple[list[str], str, int]:
    profile = os.path.join(work, f'lo_profile_{uuid.uuid4().hex[:8]}')
    expected_pdf = os.path.splitext(os.path.basename(pptx_path))[0] + '.pdf'
    rc, out, err = await _run(
        [
            'soffice',
            '--headless',
            '--norestore',
            f'-env:UserInstallation=file://{profile}',
            '--convert-to',
            'pdf',
            '--outdir',
            work,
            pptx_path,
        ],
        work,
        SOFFICE_TIMEOUT_S,
    )
    log = f'[soffice rc={rc}] {out.strip()} {err.strip()}'.strip()

    if 'source file could not be loaded' in err:
        return [], f'{log}\nLibreOffice could not load the .pptx (it is not a readable deck).', 0

    pdf_path = os.path.join(work, expected_pdf)
    if not os.path.exists(pdf_path):
        strays = sorted(f for f in os.listdir(work) if f.endswith('.pdf'))
        note = f' (unrelated .pdf files present: {strays}; they are NOT used)' if strays else ''
        return [], f'{log}\nLibreOffice produced no PDF at {expected_pdf}{note}.', 0

    if 'using filter' in out and 'impress' not in out.lower():
        used = out.strip().rsplit('filter :', 1)[-1].strip() or '(unknown)'
        return (
            [],
            f'{log}\nLibreOffice did not convert this as a presentation: it used '
            f'{used}, which means the file was not readable as a deck.',
            0,
        )

    declared_pages = _pdf_declared_pages(pdf_path)

    raster_dir = os.path.join(work, 'pages')
    os.makedirs(raster_dir, exist_ok=True)
    rc2, out2, err2 = await _run(
        ['pdftoppm', '-png', '-r', str(RASTER_DPI), pdf_path, os.path.join(raster_dir, 'pg')],
        work,
        RASTER_TIMEOUT_S,
    )
    log += f'\n[pdftoppm rc={rc2}] {err2.strip()}'

    def page_index(name: str) -> int:
        match = re.search(r'-(\d+)\.png$', name)
        return int(match.group(1)) if match else 0

    pages = sorted((f for f in os.listdir(raster_dir) if f.endswith('.png')), key=page_index)

    if rc2 != 0:
        return (
            [],
            f'{log}\nRasterising the PDF failed (exit {rc2}) after writing '
            f'{len(pages)} page(s); the page set is incomplete.',
            0,
        )

    if not pages:
        return [], f'{log}\nNo PNG pages were produced.', 0

    png_paths = [os.path.join(raster_dir, f) for f in pages]

    for path in png_paths:
        complete, why = _png_is_complete(path)
        if not complete:
            return (
                [],
                f'{log}\nA rendered page is not a readable image ({os.path.basename(path)}: {why}).',
                0,
            )

    if declared_pages is not None and len(pages) < declared_pages:
        return (
            [],
            f"{log}\nOnly {len(pages)} of the PDF's {declared_pages} page(s) were rasterised; the render is truncated.",
            0,
        )

    return png_paths, log, len(pages)


def _stage_model_inputs(context: ToolContext, work: str) -> dict[str, Any]:
    staged: list[str] = []
    failed: list[str] = []
    report: dict[str, Any] = {}

    code_dir = context.tool_state.get(WORKDIR_KEY)
    if isinstance(code_dir, str) and os.path.isdir(code_dir):
        records = context.tool_state.get(TOOL_RECORDS_KEY)
        wrote: set[str] = set()
        have_records = False
        if isinstance(records, (list, tuple)):
            for rec in records:
                if not isinstance(rec, dict) or rec.get('tool') != 'run_python':
                    continue
                have_records = True
                for name in rec.get('saved') or ():
                    if isinstance(name, str):
                        wrote.add(os.path.basename(name))
        if not have_records:
            report['staging_fallback'] = 'no run_python records; staged by listing'

        for name in sorted(os.listdir(code_dir)):
            if not name.lower().endswith(_IMAGE_EXTS):
                continue
            if name in ('target.png', 'render.png'):
                continue
            if have_records and name not in wrote:
                continue
            try:
                shutil.copyfile(os.path.join(code_dir, name), os.path.join(work, name))
                staged.append(name)
            except OSError:
                failed.append(name)

    for key, filename in ((IMAGE_KEY, 'target.png'), (RENDER_KEY, 'render.png')):
        image = context.tool_state.get(key)
        if image is None or not hasattr(image, 'save'):
            continue
        try:
            image.save(os.path.join(work, filename))
            staged.append(filename)
        except (OSError, ValueError):
            failed.append(filename)
            logger.warning('could not stage %s into the build dir', filename)

    report['staged'] = staged
    if failed:
        report['staging_failed'] = failed
    return report


@tool_registry.register(
    'visual/ppt_render_slide',
    meta={
        'name': 'render_slide',
        'description': (
            'Compile your slide code and render it. The reply is text: the page count, '
            'the pixel size of each page, and any compiler note.\n\n'
            "Set `language` to 'python-pptx' (a script that saves slide.pptx), 'pptxgenjs' "
            "(TypeScript run under tsx), or 'slideir' (declarative JSON).\n\n"
            'Writing files to disk is expected and allowed: save your deck as slide.pptx.\n'
            'To judge the result, use compare_render (mean difference and the worst tiles, '
            "as numbers), or inspect_region / overlay_grid with source='render' to "
            'measure your own render.' + _img_return_suffix('render')
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def render_slide_tool(params: RenderInput, context: ToolContext) -> ToolResponse:
    code = _strip_code_fence(params.code)
    if not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: empty code.'}], meta={'success': False, 'stage': 'input'}
        )

    language = params.language if params.language and params.language != 'auto' else _detect_language(code)

    cfg = context.tool_configs.get('visual_agent_ppt12', {}) if hasattr(context, 'tool_configs') else {}
    keep = bool(cfg.get('keep_workdir', False))
    assets_dir = context.tool_state.get('assets_dir') or cfg.get('assets_dir')
    work = tempfile.mkdtemp(prefix='pptx_render_')

    staging = _stage_model_inputs(context, work)

    _existing = context.tool_state.get(TOOL_RECORDS_KEY)
    records = list(_existing) if isinstance(_existing, (list, tuple)) else []
    try:
        pptx_path, build_log = await _build_pptx(code, language, work, assets_dir=assets_dir)
        if pptx_path is None:
            records.append({'tool': 'render_slide', 'success': False, 'stage': 'build'})
            return ToolResponse(
                output=[{'type': 'text', 'text': f'Build failed.\n{build_log}'}],
                meta={'success': False, 'stage': 'build', **staging},
                context_updates={TOOL_RECORDS_KEY: records},
            )

        deck_saved = _persist_deck(pptx_path, context, str(cfg.get('deck_filename') or 'slide.pptx'))
        if deck_saved:
            records.append({'tool': 'render_slide', 'event': 'deck_persisted', 'path': deck_saved})

        pngs, render_log, n_pages = await _pptx_to_pngs(pptx_path, work)
        if not pngs:
            records.append({'tool': 'render_slide', 'success': False, 'stage': 'render'})
            return ToolResponse(
                output=[{'type': 'text', 'text': f'The .pptx was produced but rendering failed.\n{render_log}'}],
                meta={'success': False, 'stage': 'render'},
                context_updates={TOOL_RECORDS_KEY: records},
            )

        images = [Image.open(p).convert('RGB') for p in pngs[:_MAX_PAGES_RETURNED]]
        for img in images:
            img.load()

        parts: list[dict[str, Any]] = [
            {
                'type': 'text',
                'text': (
                    f'Rendered {n_pages} page(s) at {RASTER_DPI}dpi '
                    f'({images[0].size[0]}x{images[0].size[1]}px).'
                    + (f' Showing the first {len(images)}.' if n_pages > len(images) else '')
                ),
            }
        ]
        if build_log:
            parts.append({'type': 'text', 'text': f'Note from the compiler: {build_log}'})
        extra_updates: dict[str, Any] = {}
        for idx, img in enumerate(images, start=1):
            note = _image_note(context, img, f'render_slide_page{idx}')
            extra_updates.update(note['updates'])
            parts.append({'type': 'text', 'text': f'Page {idx}: rendered {img.size[0]}x{img.size[1]}px.'})
            parts.append(note['part'])

        records.append({
            'tool': 'render_slide',
            'success': True,
            'n_pages': n_pages,
            'language': language,
            'build_note': build_log or '',
            'deck_saved': deck_saved,
            'code': code,
        })
        return ToolResponse(
            output=[*parts],
            meta={
                'success': True,
                'n_pages': n_pages,
                'language': language,
                'size': list(images[0].size),
                'build_note': build_log or '',
                'deck_saved': deck_saved,
                'workdir': work if keep else None,
            },
            context_updates={
                TOOL_RECORDS_KEY: records,
                RENDER_KEY: images[0],
                **extra_updates,
            },
        )
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
