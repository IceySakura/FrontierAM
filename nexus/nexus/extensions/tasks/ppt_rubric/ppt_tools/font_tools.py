"Font tools: the model reads the TYPEFACE and the POINT SIZE off the target slide."

import hashlib
import logging
import os
import sys

from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .image_tools import IMAGE_KEY, _get_image, _record
from .measure_tools import _build_state


logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

FONT_ROOT_ENV = 'PPTX_AGENT_FONT_ROOT'

_MIN_RESOLVE_FRACTION = 0.95

# Guard: md5s of the vendored algorithm snapshot as shipped in this MR (after
# the comment/blank-line slimming). check_vendor_sync() reports any file whose
# bytes drift from what this MR shipped, catching silent edits.
VENDOR_MD5 = {
    'fontid/font_tool.py': '6c0dcee220677d20441ccec8083b20da',
    'fontid/identify.py': '1486229af3482a7c9e83a6627a824835',
    'fontid/glyphsig.py': 'b1092b01d8b1fca57ee908e88f2dc30a',
    'fontid/catalog.py': '89ea418e40fabc0ee2dc89fe456f7352',
    'fontid/parallel.py': '63c6780021f8807bd55e416daea27133',
    'fontid/bench_v3.py': '78636868ec10c8e8079b50e1293abc91',
    'fontid/bench_render.py': 'dfba59c4b51235a0ee1212e7c26fa524',
    'fontid/lo_probe.py': '160b16e3be88627764d70834fd651032',
    'fontid/installed_faces.json': '8cf719acb943bca92df8746e00f1bb88',
    'fontid/font_index2.json': 'a170961901a7293ad19f83fb6e1a6bb9',
    'fontid/equiv_classes.json': '6e7271e3a060625512d2b0029f781042',
    'sizeid/fontsize_api.py': '03cab522d8e2ddb44b0d0b3c4589e6b9',
    'sizeid/fontsize.py': '1782cde3d115fcddb8a0903a65be547f',
    'sizeid/ratio_const.json': 'de3fd7488c2ec756eeac0bc5c123e882',
}


def check_vendor_sync() -> dict[str, str]:
    """Report vendored files whose bytes no longer match the recorded md5.

    Empty dict means in sync. JSON manifests are included on purpose:
    font_index2.json decides which faces are candidates, so swapping it
    changes every answer while leaving all code identical.
    """
    findings: dict[str, str] = {}
    for rel, want in VENDOR_MD5.items():
        local = os.path.join(_HERE, rel)
        if not os.path.exists(local):
            findings[rel] = 'vendored copy missing'
            continue
        got = hashlib.md5(Path(local).read_bytes()).hexdigest()
        if got != want:
            findings[rel] = f'vendored copy edited: md5 {got} != recorded {want}'
    return findings


FONTID_COUNT_KEY = 'visual_agent.ppt12.n_fontid'
FONTSIZE_COUNT_KEY = 'visual_agent.ppt12.n_fontsize'
MEASURED_VALUES_KEY = 'visual_agent.ppt12.measured_values'

_READY: tuple[bool, dict[str, Any]] | None = None
_MODULES: dict[str, Any] = {}


def font_root() -> str | None:
    root = (os.environ.get(FONT_ROOT_ENV) or '').strip()
    return root or None


def _remap(path: str, root: str | None) -> str:
    if not root:
        return path
    return os.path.join(root, path.lstrip(os.sep))


def _import_vendored() -> dict[str, Any]:
    if _MODULES:
        return _MODULES
    # The vendored trees import each other by flat module name (import font_tool,
    # import glyphsig), which only resolves with their directories on sys.path.
    # Keep that window as short as possible: add the paths, import everything
    # once, then remove them again -- the modules stay resident in sys.modules,
    # so the rest of the process (host-side schema resolution included) never
    # sees ppt_tools/fontid/sizeid/bx on its import search path.
    added = [
        p
        for p in (_HERE, os.path.join(_HERE, 'fontid'), os.path.join(_HERE, 'sizeid'), os.path.join(_HERE, 'bx'))
        if p not in sys.path
    ]
    for p in added:
        sys.path.insert(0, p)
    try:
        import font_tool  # noqa: E402  ⛔ vendored, algorithms unedited
        import fontsize_api  # noqa: E402  ⛔ vendored, algorithms unedited

        # The vendored identify() defaults to parallel=True, which lazily forks
        # a 48-worker resident pool (FONTID_WORKERS) from inside this
        # multi-threaded asyncio process -- a deadlock hazard via inherited
        # locks, and the pool is never shut down. The vendored files are
        # md5-frozen, so defuse the pool here instead: force the in-process
        # serial path by making pool() look already-failed. pool() returns
        # `_POOL or None`, so False means None and distances() runs serially.
        import identify as _identify  # noqa: E402  ⛔ vendored, algorithms unedited
        import parallel as _parallel  # noqa: E402  ⛔ vendored, algorithms unedited

        _parallel._POOL = False

        # The import-time cleanup above is not enough: the vendored code also
        # mutates sys.path at RUN time, on every call. parallel._work() does
        # sys.path.insert(0, fontid_dir) each invocation (and now runs on the
        # serial path the pool defusal above routes through -- identify.py
        # does `import parallel as PAR` then calls PAR.distances, which reaches
        # _work serially), and fontsize_api._ink_box() inserts bx/ on every
        # estimate_font_size call (the `from text_blocks import _glyph_mask`
        # at its top needs it). Both re-pollute sys.path -- and this module
        # runs inside a host that resolves OTHER tools' imports concurrently,
        # so a stale vendored path at the front could shadow host modules.
        # Wrap the two mutating entry points so sys.path is exactly restored
        # around every call.
        _orig_work = _parallel._work
        _orig_ink_box = fontsize_api._ink_box

        def _patched_work(task):  # noqa: ANN202 -- vendored signature passthrough
            saved = list(sys.path)
            try:
                return _orig_work(task)
            finally:
                sys.path[:] = saved

        def _patched_ink_box(*a, **kw):  # noqa: ANN202 -- vendored signature passthrough
            saved = list(sys.path)
            try:
                return _orig_ink_box(*a, **kw)
            finally:
                sys.path[:] = saved

        _parallel._work = _patched_work
        fontsize_api._ink_box = _patched_ink_box
    finally:
        # remove() drops only the FIRST match, and several vendored modules
        # insert the same directory at import time (font_tool, identify and
        # fontsize all add fontid/), so clean each path out entirely or the
        # duplicates survive the cleanup.
        for p in added:
            while p in sys.path:
                sys.path.remove(p)

    _MODULES.update({'FT': font_tool, 'FZ': fontsize_api, 'ID': _identify, 'PAR': _parallel})
    return _MODULES


def _load_and_remap(root: str | None) -> dict[str, Any]:
    modules = _import_vendored()
    identify = modules['FT'].ID
    fontsize = modules['FZ'].FS

    n_total = 0
    n_ok = 0
    missing: list[str] = []
    unrebased: list[str] = []

    def _fix(rec: dict[str, Any]) -> None:
        nonlocal n_total, n_ok
        f = rec.get('file')
        if not f:
            return
        n_total += 1
        p = _remap(f, root)
        rec['file'] = p
        if root and not p.startswith(root) and len(unrebased) < 5:
            unrebased.append(p)
        if os.path.exists(p):
            n_ok += 1
        elif len(missing) < 5:
            missing.append(p)

    idx = identify.load_index()
    for _script, d in idx.items():
        for rec in d.get('records') or []:
            _fix(rec)
        for _cid, recs in (d.get('by_class') or {}).items():
            for rec in recs:
                f = rec.get('file') or ''
                if root and f and not f.startswith(root) and len(unrebased) < 5:
                    unrebased.append(f)

    fontsize._load_faces()
    for rec in fontsize._FACES or []:
        _fix(rec)

    frac = (n_ok / n_total) if n_total else 0.0
    return {
        'root': root,
        'n_refs': n_total,
        'n_resolved': n_ok,
        'fraction': round(frac, 4),
        'missing_examples': missing,
        'unrebased_examples': unrebased,
    }


def _assert_meta_safe(detail: dict[str, Any]) -> None:
    ok = (str, int, float, bool, type(None))
    for k, v in detail.items():
        if isinstance(v, ok):
            continue
        if isinstance(v, (list, tuple)) and all(isinstance(x, ok) for x in v):
            continue
        raise TypeError(
            f'font detail[{k!r}] is {type(v).__name__}, which is not a JSON '
            'primitive. It would reach ToolResponse.meta and hang the tracer.'
        )


def ensure_fonts_ready() -> tuple[bool, dict[str, Any]]:
    global _READY
    if _READY is not None:
        return _READY
    root = font_root()
    try:
        detail = _load_and_remap(root)
        _assert_meta_safe(detail)
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        logger.exception('font tree resolution failed')
        _READY = (False, {'error': f'{type(exc).__name__}: {exc}', 'root': root})
        return _READY
    ready = detail['n_refs'] > 0 and detail['fraction'] >= _MIN_RESOLVE_FRACTION
    _READY = (ready, detail)
    return _READY


def _reset_for_test() -> None:
    global _READY
    _READY = None
    _MODULES.clear()
    for m in ('identify', 'fontsize', 'font_tool', 'fontsize_api'):
        sys.modules.pop(m, None)


def _not_ready(context: ToolContext, tool: str, detail: dict[str, Any]) -> ToolResponse:
    if 'error' in detail:
        why = f'the font index could not be loaded ({detail["error"]})'
    else:
        why = f'only {detail.get("n_resolved", 0)} of {detail.get("n_refs", 0)} font files are reachable'
    text = (
        f'Error: {tool} is unavailable in this environment -- {why}. No estimate is '
        'reported: this tool works by rendering candidate typefaces and comparing '
        'them to the slide, so without the font files any number it returned would '
        'come from the wrong face and be confidently wrong. Do not call it again on '
        'this slide. Judge the size and typeface from the image instead, and prefer '
        'leaving the font unset so the text inherits an installed face.'
    )
    updates = _record(context, {'tool': tool, 'success': False, 'error': 'fonts unavailable'})
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={'success': False, 'error': 'fonts unavailable', 'font_detail': detail},
        context_updates=updates,
    )


def _no_image(context: ToolContext, tool: str) -> ToolResponse:
    updates = _record(context, {'tool': tool, 'success': False, 'error': 'no reference image'})
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Error: no reference image available.'}],
        meta={'success': False, 'error': 'no reference image'},
        context_updates=updates,
    )


def _finish(
    context: ToolContext,
    tool: str,
    state: dict[str, Any],
    parts: list[dict[str, Any]],
    meta: dict[str, Any],
    count_key: str,
    state_count_field: str,
    **extra: Any,
) -> ToolResponse:
    updates = _record(
        context,
        {'tool': tool, 'success': bool(meta.get('success')), **extra},
    )
    updates[count_key] = int(state.get(state_count_field) or 0)
    new_values = state.get('tool_values') or []
    if new_values:
        prior = context.tool_state.get(MEASURED_VALUES_KEY)
        prior_list = list(prior) if isinstance(prior, (list, tuple)) else []
        updates[MEASURED_VALUES_KEY] = prior_list + [{'tool': tool, **v} for v in new_values if isinstance(v, dict)]
    return ToolResponse(output=[*parts], meta=meta, context_updates=updates)


def _state_for(context: ToolContext, image: Image.Image, count_key: str, field: str) -> dict[str, Any]:
    state = _build_state(context, image)
    used = context.tool_state.get(count_key)
    state[field] = int(used) if isinstance(used, int) else 0
    return state


class IdentifyFontInput(BaseModel):
    box: list[float] = Field(description='[x1, y1, x2, y2] in screenshot pixels, around a SINGLE line of text.')
    text: str = Field(description='The words inside that box, exactly as they appear, including capitalisation.')
    top_n: int | None = Field(default=None, description='How many candidates to return (1-5, default 3).')


@tool_registry.register(
    'visual/ppt_identify_font',
    meta={
        'name': 'identify_font',
        'description': (
            'Report which installed typefaces render a line of text on the '
            'screenshot most closely. Pass a box around ONE line plus the words '
            'you can read there.\n\n'
            'It returns a RENDER-EQUIVALENCE CLASS, not the true font name -- a '
            'hard information limit, not a shortcoming. On this renderer 999 '
            'font names collapse to 72 distinct pixel results on CJK text, so '
            'most font names are indistinguishable from an image: Arial, '
            'Calibri, Verdana, Tahoma and Microsoft YaHei all render '
            'identically, as do SimSun and SimHei. Any name in the class gives '
            'the same pixels, so pick one and move on.\n\n'
            'Only `high` confidence is worth acting on, and even there the top '
            'answer is right only about half the time -- it is the best signal '
            'available, not a certainty. Below that no name is reported at all, '
            'because acting on one was measured worse than simply leaving the '
            'font unset: when no '
            'name comes back, leave the font unset so the text inherits an '
            'installed face -- do NOT fall back on a name from general '
            'knowledge, which was measured wrong 79.3% of the time. Do not '
            'spend rounds pinning down an exact name and do not run `fc-list` '
            'or look for font files -- every font the renderer has was already '
            'compared, and every font this corpus uses is installed. Best used '
            'to confirm serif-versus-sans and weight.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def identify_font_tool(params: IdentifyFontInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'identify_font')
    ready, detail = ensure_fonts_ready()
    if not ready:
        return _not_ready(context, 'identify_font', detail)
    state = _state_for(context, image, FONTID_COUNT_KEY, '_n_fontid')
    args: dict[str, Any] = {'box': params.box, 'text': params.text}
    if params.top_n:
        args['top_n'] = params.top_n
    parts, meta = _MODULES['FT'].t_identify_font(state, args)
    return _finish(
        context,
        'identify_font',
        state,
        parts,
        meta,
        FONTID_COUNT_KEY,
        '_n_fontid',
        conf=meta.get('conf'),
        margin=meta.get('margin'),
    )


class FontSizeLine(BaseModel):
    box: list[float] = Field(description='[x1, y1, x2, y2] in screenshot pixels, around a single line of text.')
    text: str = Field(description='The characters on that line, as they appear.')
    font: str | None = Field(
        default=None,
        description=(
            'Typeface name, if known (e.g. from identify_font). Optional but makes the '
            'answer markedly sharper for Latin text.'
        ),
    )
    bold: bool | None = None
    italic: bool | None = None


class EstimateFontSizeInput(BaseModel):
    lines: list[FontSizeLine] = Field(description='One entry per line of text to measure.')


@tool_registry.register(
    'visual/ppt_estimate_font_size',
    meta={
        'name': 'estimate_font_size',
        'description': (
            'Measure the point size of a line of text on the screenshot. Pass '
            'the box of ONE line plus the characters you can read there; boxes '
            'from `text_blocks` work directly and are the recommended input. '
            'Several lines per call.\n\n'
            'Read the confidence -- it decides whether to use the answer. At '
            '`high` (about a third of lines, almost all CJK here) it lands '
            'within 1 pt about 3 times in 4 and still beats judging the size '
            'by eye: use the number as given. At '
            '`medium` it is only about as good as your own '
            'reading, so treat it as a hint and keep your own judgement if they '
            'disagree. At `low` it is right only about a third of the time, well '
            'worse '
            'than reading the size off the image: ignore the number. Latin text '
            'is the weak case; passing a known `font` sharpens it.\n\n'
            'One line per entry: a box covering a whole paragraph reports a size '
            'several times too large. The characters matter, because digits, '
            'descenders and CJK have different ink heights at the same size.\n\n'
            'The answer is the raw measurement, not a rounded one. Where a tidy '
            'value (whole point, or a half point like 10.5) lies within the '
            "tool's measured precision it is offered in brackets as a candidate, "
            'because decks are often laid out on tidy sizes. Do not force sizes '
            'to whole numbers on your own: fractional sizes are normal and '
            'deliberate (10.5 and 7.5 are standard Chinese type sizes, Word '
            'paste and autofit produce values like 12.1, and rescaling a deck '
            'from 16:9 to 4:3 makes every size fractional, e.g. 18.67).'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def estimate_font_size_tool(params: EstimateFontSizeInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'estimate_font_size')
    ready, detail = ensure_fonts_ready()
    if not ready:
        return _not_ready(context, 'estimate_font_size', detail)
    state = _state_for(context, image, FONTSIZE_COUNT_KEY, '_n_fontsize')
    lines = [{k: v for k, v in ln.model_dump().items() if v is not None} for ln in params.lines]
    parts, meta = _MODULES['FZ'].t_estimate_font_size(state, {'lines': lines})
    return _finish(
        context,
        'estimate_font_size',
        state,
        parts,
        meta,
        FONTSIZE_COUNT_KEY,
        '_n_fontsize',
        n=meta.get('n'),
    )


FONT_TOOLS = [
    'visual/ppt_identify_font',
    'visual/ppt_estimate_font_size',
]

__all__ = [
    'VENDOR_MD5',
    'check_vendor_sync',
    'FONTID_COUNT_KEY',
    'FONTSIZE_COUNT_KEY',
    'FONT_ROOT_ENV',
    'FONT_TOOLS',
    'ensure_fonts_ready',
    'font_root',
]
