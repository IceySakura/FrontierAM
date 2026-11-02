"Measurement tools: the model reads NUMBERS off the target slide."

import hashlib
import logging
import os
import sys

from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .compile_tools import RASTER_DPI
from .image_tools import IMAGE_KEY, _get_image, _record


logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ._mx import measure_api  # noqa: E402  ⛔ vendored, algorithms unedited


# The vendored measure_api prepends its own bx/ directory to sys.path for the
# flat `import text_blocks`; once that module is resident in sys.modules the
# path entry serves no purpose, so drop everything this adapter added. The
# host process (tool-schema resolution) then never sees ppt_tools/bx on its
# import search path.
for _p in list(sys.path):
    if _p in (_HERE, os.path.join(_HERE, 'bx'), os.path.join(_HERE, '_mx')):
        sys.path.remove(_p)


# Guard: md5s of the vendored algorithm snapshot as shipped in this MR (after
# the comment/blank-line slimming); see font_tools.check_vendor_sync.
VENDOR_MD5 = {
    '_mx/measure_api.py': '29b1d03dadc2f81d728a777e1f65c7fb',
    'bx/text_blocks.py': '6b3a5bc725fa44dda6fea5f55582fe16',
}


def check_vendor_sync() -> dict[str, str]:
    """Report vendored files whose bytes no longer match the recorded md5."""
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


_PX_PER_PT = RASTER_DPI / 72.0

PAGE_PT_KEY = 'visual_agent.ppt12.page_pt'
SAMPLE_COUNT_KEY = 'visual_agent.ppt12.n_colour_samples'
MEASURED_VALUES_KEY = 'visual_agent.ppt12.measured_values'


def _page_pt(context: ToolContext, image: Image.Image) -> tuple[dict[str, float], float]:
    w_px, h_px = image.size
    override = context.tool_state.get(PAGE_PT_KEY)
    if isinstance(override, dict) and override.get('height'):
        h_pt = float(override['height'])
        ratio = h_px / h_pt if h_pt else _PX_PER_PT
        w_pt = float(override.get('width') or (w_px / ratio if ratio else 0.0))
        return {'width': round(w_pt, 2), 'height': round(h_pt, 2)}, ratio
    ratio = _PX_PER_PT
    return {'width': round(w_px / ratio, 2), 'height': round(h_px / ratio, 2)}, ratio


def _build_state(context: ToolContext, image: Image.Image) -> dict[str, Any]:
    page_pt, ratio = _page_pt(context, image)
    used = context.tool_state.get(SAMPLE_COUNT_KEY)
    return {
        'reference': image,
        'ratio': ratio,
        'page_size': page_pt,
        '_n_samples': int(used) if isinstance(used, int) else 0,
    }


def _finish(
    context: ToolContext,
    tool: str,
    state: dict[str, Any],
    parts: list[dict[str, Any]],
    meta: dict[str, Any],
    **extra: Any,
) -> ToolResponse:
    updates = _record(
        context,
        {'tool': tool, 'success': bool(meta.get('success')), **extra},
    )
    updates[SAMPLE_COUNT_KEY] = int(state.get('_n_samples') or 0)
    new_values = state.get('tool_values') or []
    if new_values:
        prior = context.tool_state.get(MEASURED_VALUES_KEY)
        prior_list = list(prior) if isinstance(prior, (list, tuple)) else []
        updates[MEASURED_VALUES_KEY] = prior_list + [{'tool': tool, **v} for v in new_values if isinstance(v, dict)]
    return ToolResponse(output=[*parts], meta=meta, context_updates=updates)


def _no_image(context: ToolContext, tool: str) -> ToolResponse:
    updates = _record(context, {'tool': tool, 'success': False, 'error': 'no reference image'})
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Error: no reference image available.'}],
        meta={'success': False, 'error': 'no reference image'},
        context_updates=updates,
    )


_MEASURE_FIRST = (
    '\n\nMeasure, do not estimate. Accuracy of the final deck matters more than the '
    'tokens a measurement costs, so calling this is the expected behaviour rather '
    'than an optimisation. Every coordinate and colour you write into code should be '
    'traceable to a value some measurement tool returned; estimate by eye only the '
    'quantities no tool can report.'
)


class PageInfoInput(BaseModel):
    """No arguments: the tool reports facts about the whole canvas."""


@tool_registry.register(
    'visual/ppt_page_info',
    meta={
        'name': 'page_info',
        'description': (
            'Canvas facts measured from the screenshot: pixel size, slide size in points, '
            'the px-per-pt scale, and the background colour. Takes no arguments.\n\n'
            'Call this FIRST, before writing any geometry. It gives you the pt-per-pixel '
            'conversion every other measurement depends on, and the background colour. '
            'Guessing the canvas size is a measured failure mode: models got it wrong on '
            '31 of 40 slides, and a wrong canvas puts every element at the wrong place.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def page_info_tool(params: PageInfoInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'page_info')
    state = _build_state(context, image)
    parts, meta = measure_api.t_page_info(state, {})
    return _finish(context, 'page_info', state, parts, meta)


class TextBlocksInput(BaseModel):
    """No arguments: the whole screenshot is scanned."""


@tool_registry.register(
    'visual/ppt_text_blocks',
    meta={
        'name': 'text_blocks',
        'description': (
            'List every text line on the slide, already measured: bounding box in points '
            'plus the text colour. Takes no arguments; it scans the whole screenshot.\n\n'
            'You do not need to pick a region or measure text yourself. Boxes are computed '
            'deterministically from the pixels and are accurate to about 2 points.\n\n'
            'Use this for every slide that contains text, instead of reading positions off '
            'the image by eye. The list may include rows that are not slide text (lettering '
            'baked into a photo, decorative artwork, chart labels); use the screenshot to '
            'decide which are real. It reports the glyph box, tight around the letters, not '
            'an invisible text frame -- so a text frame you declare may need to be slightly '
            'larger than the box reported here.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def text_blocks_tool(params: TextBlocksInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'text_blocks')
    state = _build_state(context, image)
    parts, meta = measure_api.t_text_blocks(state, {})
    return _finish(context, 'text_blocks', state, parts, meta, n_lines=meta.get('n_lines'))


class SampleColourInput(BaseModel):
    points: list[list[float]] = Field(description='List of [x, y] pixel coordinates, origin top-left.')
    patch: int = Field(
        default=3,
        description='Patch side in px (1 = single pixel, default 3, max 15).',
    )


@tool_registry.register(
    'visual/ppt_sample_colour',
    meta={
        'name': 'sample_colour',
        'description': (
            'Read the exact colour at one or more points of the slide, in pixel coordinates '
            '(origin top-left). Returns hex and RGB numbers.\n\n'
            'Give up to 64 points per call. By default it also reports the modal colour of a '
            '3x3 patch around each point, which tells you whether you landed on a flat fill or '
            'on an edge or a glyph. Set `patch` to 1 for the single pixel, or larger (max 15) '
            'for a small area.\n\n'
            'Use this for fills, backgrounds, accent bars and rules: sample the colour rather '
            'than naming the one it looks like. For text colour prefer `text_blocks`, which '
            'already reports it per line.' + _MEASURE_FIRST
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def sample_colour_tool(params: SampleColourInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'sample_colour')
    state = _build_state(context, image)
    parts, meta = measure_api.t_sample_colour(state, {'points': params.points, 'patch': params.patch})
    return _finish(context, 'sample_colour', state, parts, meta, n=meta.get('n'))


class RegionStatsInput(BaseModel):
    regions: list[list[float]] = Field(
        description='List of [x1, y1, x2, y2] pixel boxes, origin top-left. Up to 16 per call.'
    )


@tool_registry.register(
    'visual/ppt_region_stats',
    meta={
        'name': 'region_stats',
        'description': (
            'Summarise one or more rectangular regions of the slide as numbers: the dominant '
            'colour and how much of the region it covers, the top colours by area, how much of '
            'the region differs from the dominant colour, and the bounding box of that '
            'non-dominant content in both px and pt.\n\n'
            'This is how you measure a shape: put a generous box around it and read back where '
            'its content actually starts and ends, plus its fill colour. Up to 16 regions per '
            'call; boxes are [x1, y1, x2, y2] in pixels, origin top-left.' + _MEASURE_FIRST
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def region_stats_tool(params: RegionStatsInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'region_stats')
    state = _build_state(context, image)
    parts, meta = measure_api.t_region_stats(state, {'regions': params.regions})
    return _finish(context, 'region_stats', state, parts, meta, n=meta.get('n'))


class FindEdgesInput(BaseModel):
    bbox: list[float] | None = Field(
        default=None,
        description='[x1, y1, x2, y2] in pixels. Omit to scan the whole slide.',
    )
    min_span: float = Field(
        default=0.6,
        description='Minimum fraction of the region an edge must span, 0..1. Lower it to catch short rules.',
    )


@tool_registry.register(
    'visual/ppt_find_edges',
    meta={
        'name': 'find_edges',
        'description': (
            'Find long straight horizontal and vertical edges inside a region, reported as '
            'y / x positions in both pixels and points. Use it to locate divider rules, table '
            'gridlines, underlines and the boundaries of large colour blocks, instead of '
            'estimating them by eye.\n\n'
            '`bbox` is [x1, y1, x2, y2] in pixels (default: the whole slide). `min_span` is how '
            'much of the region an edge must cross to count, 0..1, default 0.6 -- lower it to '
            'catch short rules.' + _MEASURE_FIRST
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def find_edges_tool(params: FindEdgesInput, context: ToolContext) -> ToolResponse:
    image = _get_image(context, IMAGE_KEY)
    if image is None:
        return _no_image(context, 'find_edges')
    state = _build_state(context, image)
    args: dict[str, Any] = {'min_span': params.min_span}
    if params.bbox:
        args['bbox'] = params.bbox
    parts, meta = measure_api.t_find_edges(state, args)
    return _finish(context, 'find_edges', state, parts, meta, n_h=meta.get('n_h'), n_v=meta.get('n_v'))


MEASURE_TOOLS = [
    'visual/ppt_page_info',
    'visual/ppt_text_blocks',
    'visual/ppt_sample_colour',
    'visual/ppt_region_stats',
    'visual/ppt_find_edges',
]

__all__ = [
    'VENDOR_MD5',
    'check_vendor_sync',
    'MEASURE_TOOLS',
    'MEASURED_VALUES_KEY',
    'PAGE_PT_KEY',
    'SAMPLE_COUNT_KEY',
]
