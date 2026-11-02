"Image inspection tools for precise element localization in slide images."

import base64
import io
import logging
import os
import re

from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


logger = logging.getLogger(__name__)

IMAGE_KEY = 'image'
RENDER_KEY = 'rendered_image'
TOOL_RECORDS_KEY = 'pptx_tool_records'

_MIN_CROP_PX = 8
_MAX_UPSCALE_PX = 1_600


def _image_to_base64_url(image: Image.Image, fmt: str = 'PNG') -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    mime = 'image/png' if fmt.upper() == 'PNG' else 'image/jpeg'
    return f'data:{mime};base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}'


_TOOL_IMAGE_RETURN = os.environ.get('PPTX_AGENT_TOOL_IMAGE_RETURN', '1').strip().lower() not in (
    '0',
    'false',
    'no',
    'off',
    '',
)

_DUMP_DIR_KEY = 'visual_agent.ppt12:image_dump_dir'
_DUMP_SEQ_KEY = 'visual_agent.ppt12:image_dump_seq'


def _tool_image_return_enabled() -> bool:
    return _TOOL_IMAGE_RETURN


_IMG_RETURN_SUFFIX = {
    'inspect': (
        '\n\nReturns the crop as an image, plus the measured numbers read off it '
        '(dominant colour and its share, ink fraction, content bbox in px and pt, '
        'glyph blocks).'
    ),
    'grid': (
        '\n\nReturns the gridded slide as an image, plus the page background colour '
        'and the content colour and coverage of each non-empty cell.'
    ),
    'compare': (
        '\n\nReturns three images: the target, your latest render, and the difference '
        'map, plus the measured difference numbers.'
    ),
    'render': (
        '\n\nReturns the rendered page(s) as image(s), plus the page count, the pixel '
        'size of each page and any compiler note.'
    ),
    'code': ('\n\nImages you save with a relative path are returned to you as images, up to 4 per call.'),
}
_TEXT_ONLY_SUFFIX = {
    'inspect': (
        '\n\nReports as TEXT, never as a picture: the crop is measured, not shown. Read the numbers in the reply.'
    ),
    'grid': (
        '\n\nReports as TEXT, never as a picture: the grid is measured, not shown. Read the numbers in the reply.'
    ),
    'compare': (
        '\n\nReports as TEXT, never as a picture: the three images are measured, not '
        'shown. Read the numbers in the reply.'
    ),
    'render': (
        '\n\nReports as TEXT, never as a picture: the render is saved, not shown. Read the numbers in the reply.'
    ),
    'code': (
        '\n\nImages you save are NOT returned to you as pictures -- print the numbers you need from them instead.'
    ),
}


def _img_return_suffix(which: str) -> str:
    return (_IMG_RETURN_SUFFIX if _TOOL_IMAGE_RETURN else _TEXT_ONLY_SUFFIX)[which]


def _image_note(context: ToolContext, image: Image.Image, stem: str) -> dict[str, Any]:
    import os
    import tempfile

    dump_dir = context.tool_state.get(_DUMP_DIR_KEY)
    if not isinstance(dump_dir, str) or not os.path.isdir(dump_dir):
        dump_dir = tempfile.mkdtemp(prefix='pptx_toolimg_')
    seq = context.tool_state.get(_DUMP_SEQ_KEY)
    seq = seq + 1 if isinstance(seq, int) else 1
    safe_stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', stem).strip('_') or 'image'
    path = os.path.join(dump_dir, f'{seq:02d}_{safe_stem}.png')
    image.save(path)
    return {
        'part': (
            {'type': 'image_url', 'image_url': {'url': _image_to_base64_url(image)}}
            if _tool_image_return_enabled()
            else {
                'type': 'text',
                'text': f'[image saved for inspection ({seq:02d}_{safe_stem}.png); measurements are in the text above]',
            }
        ),
        'updates': {
            _DUMP_DIR_KEY: dump_dir,
            _DUMP_SEQ_KEY: seq,
            f'visual_agent.ppt12:image_dump_path_{seq:02d}': path,
        },
    }


def _record(context: ToolContext, entry: dict[str, Any]) -> dict[str, Any]:
    existing = context.tool_state.get(TOOL_RECORDS_KEY)
    records = list(existing) if isinstance(existing, (list, tuple)) else []
    records.append(entry)
    return {TOOL_RECORDS_KEY: records}


def _get_image(context: ToolContext, key: str) -> Image.Image | None:
    image = context.tool_state.get(key)
    return image if isinstance(image, Image.Image) else None


def _err(context: ToolContext, tool: str, msg: str, **extra: Any) -> ToolResponse:
    updates = _record(context, {'tool': tool, 'success': False, 'error': msg, **extra})
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Error: {msg}'}],
        meta={'success': False, 'error': msg},
        context_updates=updates,
    )


def _clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    return x1, y1, x2, y2


_EDGE_T = 28

_GRID_MAX_CELLS = 64

_GRID_CELL_CONTENT_T = 0.02

_DIFF_TILE_PX = 64
_DIFF_TOP_TILES = 6

_TEXT_MAX_BLOCKS = 6


def _rgb_hex(rgb: Any) -> str:
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'


def _pt_box(box: list[float], ratio: float) -> str:
    return '[{:.1f},{:.1f},{:.1f},{:.1f}]pt'.format(*(v / ratio for v in box))


def _modal_hex(block: Any) -> tuple[str, float]:
    import numpy as np

    flat = np.asarray(block).reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    return _rgb_hex(uniq[cnt.argmax()]), float(cnt.max()) / float(len(flat))  # ty: ignore -- np.unique(axis=0) returns under-typed arrays


def _measure_crop(
    context: ToolContext, image: Image.Image, box: tuple[int, int, int, int]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    import numpy as np

    from . import measure_tools

    updates: dict[str, Any] = {}
    state = measure_tools._build_state(context, image)
    ratio = float(state.get('ratio') or 1.0)

    parts, meta = measure_tools.measure_api.t_region_stats(state, {'regions': [list(box)]})
    out: list[str] = []
    values: dict[str, Any] = {}
    if meta.get('success'):
        body = parts[0].get('text', '') if parts else ''
        body = body.split('\n', 1)[1] if body.startswith('region_stats:') else body
        out.append(body.rstrip())
        pv = (meta.get('values') or [{}])[0]
        values = {
            'dominant': pv.get('dominant'),
            'dominant_frac': pv.get('dominant_frac'),
            'ink_frac': pv.get('ink_frac'),
        }
    else:
        why = (parts[0].get('text', '') if parts else '').replace('\n', ' ')
        out.append(f'  (colours could not be measured for this box: {why[:200]})')

    new_values = state.get('tool_values') or []
    if new_values:
        prior = context.tool_state.get(measure_tools.MEASURED_VALUES_KEY)
        prior_list = list(prior) if isinstance(prior, (list, tuple)) else []
        updates[measure_tools.MEASURED_VALUES_KEY] = prior_list + [
            {'tool': 'inspect_region', **v} for v in new_values if isinstance(v, dict)
        ]

    x1, y1, x2, y2 = box
    arr = np.asarray(image.convert('RGB'))[y1:y2, x1:x2]
    luma = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
    dx = np.abs(np.diff(luma, axis=1)) >= _EDGE_T
    dy = np.abs(np.diff(luma, axis=0)) >= _EDGE_T
    ex = np.zeros(luma.shape, dtype=bool)
    ex[:, :-1] |= dx
    ex[:, 1:] |= dx
    ey = np.zeros(luma.shape, dtype=bool)
    ey[:-1, :] |= dy
    ey[1:, :] |= dy
    edge_density = float((ex | ey).mean())
    mean_luma = float(luma.mean())
    out.append(
        f'  mean luminance {mean_luma:.1f}/255; edge density {edge_density:.1%} of pixels '
        f'(a pixel is an edge when some 4-neighbour differs by >= {_EDGE_T}/255)'
    )
    values['mean_luminance'] = round(mean_luma, 1)
    values['edge_density'] = round(edge_density, 4)

    blocks: list[dict[str, Any]] = []
    try:
        full = np.asarray(image.convert('RGB'))
        res = measure_tools.measure_api.list_text_blocks(full, region=list(box), ratio=ratio)
        blocks = list(res.get('blocks') or []) if res.get('ok') else []
        scan_reason = res.get('reason') if not res.get('ok') else None
    except Exception as exc:  # noqa: BLE001
        res, scan_reason = {}, f'the text scan raised {type(exc).__name__}: {exc}'
        logger.warning('inspect_region text scan failed: %s', exc)
    n_lines = sum(len(b.get('lines_px') or []) for b in blocks)
    values['n_text_blocks'] = len(blocks)
    values['n_text_lines'] = n_lines
    if not blocks:
        reason = scan_reason
        out.append(
            '  text: no glyph-shaped marks found'
            + (f' ({reason})' if reason else '')
            + ' -- the scan counts text strokes only, so a fill, a rule or a photo is not "text" here'
        )
    else:
        out.append(f'  text: {len(blocks)} block(s) / {n_lines} line(s) of glyph strokes')
        for i, b in enumerate(blocks[:_TEXT_MAX_BLOCKS], 1):
            bb = [int(round(v)) for v in (b.get('bbox_px') or [])]
            nl = len(b.get('lines_px') or [])
            if len(bb) == 4:
                out.append(f'      block {i}: px {bb} = {_pt_box(bb, ratio)}, colour {b.get("colour")}, {nl} line(s)')
        if len(blocks) > _TEXT_MAX_BLOCKS:
            out.append(
                f'      ({len(blocks) - _TEXT_MAX_BLOCKS} more block(s) not listed; the counts above are complete)'
            )
        out.append('      these are glyph boxes tight around the letters, not the text frame around them')
    return '\n'.join(out), values, updates


def _grid_reading(context: ToolContext, image: Image.Image, step: int) -> str:
    import numpy as np

    from . import measure_tools

    ratio = float(measure_tools._build_state(context, image).get('ratio') or 1.0)
    w, h = image.size
    arr = np.asarray(image.convert('RGB'))
    bg_hex, _ = _modal_hex(arr[::2, ::2])
    bg = np.array([int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)], dtype=np.int16)

    n_cells = 0
    marked: list[tuple[int, int]] = []
    by_row: dict[int, list[str]] = {}
    shown = 0
    for cy in range(0, h, step):
        for cx in range(0, w, step):
            n_cells += 1
            sub = arr[cy : min(cy + step, h), cx : min(cx + step, w)]
            mask = np.abs(sub.astype(np.int16) - bg).max(axis=2) > _EDGE_T
            frac = float(mask.mean())
            if frac < _GRID_CELL_CONTENT_T:
                continue
            cell_hex = _modal_hex(sub[mask])[0] if mask.any() else _modal_hex(sub)[0]
            marked.append((cx, cy))
            if shown < _GRID_MAX_CELLS:
                shown += 1
                by_row.setdefault(cy, []).append(f'({cx},{cy})={cell_hex} {frac:.0%}')

    out = [f'page background {bg_hex} (modal colour of the whole slide)']
    out.append(
        f'{len(marked)} of {n_cells} cells differ from that background; listing {shown} '
        f'(cap {_GRID_MAX_CELLS} cells per call). For each cell: colour = the modal colour of the '
        f'pixels that differ from the page background, % = the share of the cell they cover.'
    )
    if len(marked) > shown:
        out.append(
            f'  ⚠ {len(marked) - shown} further non-empty cell(s) are NOT listed -- raise `step` '
            f'for a coarser grid that fits, or call region_stats on the area you actually care about'
        )
    for cy in sorted(by_row):
        out.append(f'  y={cy}: ' + '  '.join(by_row[cy]))
    out.append(
        '  each (x,y) is the cell whose top-left corner carries that label -- the grid line '
        'labelled x is the LEFT edge of the cell starting at x, so a mark you see between '
        'the labels x=300 and x=400, y=200 and y=300 lies in cell (300,200)'
    )
    if marked:
        xs = [c[0] for c in marked]
        ys = [c[1] for c in marked]
        box = [min(xs), min(ys), min(max(xs) + step, w), min(max(ys) + step, h)]
        out.append(
            f'  content spans roughly px {box} = {_pt_box(box, ratio)} (coarse: it snaps to '
            f'{step}px cells; measure the exact edge with region_stats or find_edges)'
        )
    else:
        out.append(
            f'  no cell differs from the page background: this slide is a flat fill as far as a {step}px grid can see'
        )
    return '\n'.join(out)


def _diff_profile(
    context: ToolContext, ref_rgb: Image.Image, cand: Image.Image, origin: tuple[int, int] = (0, 0)
) -> tuple[str, dict[str, Any]]:
    import numpy as np

    from . import measure_tools

    state = measure_tools._build_state(context, ref_rgb)
    ratio = float(state.get('ratio') or 1.0)
    a = np.asarray(ref_rgb, dtype=np.int16)
    b = np.asarray(cand, dtype=np.int16)
    delta = np.abs(a - b)
    per = delta.max(axis=2)
    h, w = per.shape
    ox, oy = origin

    true_mean = float(per.mean()) / 255.0
    frac_any = float((per > 8).mean())
    chans = [float(delta[:, :, i].mean()) for i in range(3)]

    tiles: list[tuple[float, int, int, int, int]] = []
    t = _DIFF_TILE_PX
    for ty in range(0, h, t):
        for tx in range(0, w, t):
            blk = per[ty : ty + t, tx : tx + t]
            if blk.size:
                tiles.append((float(blk.mean()) / 255.0, tx, ty, min(tx + t, w), min(ty + t, h)))
    tiles.sort(key=lambda z: -z[0])
    worst = [z for z in tiles[:_DIFF_TOP_TILES] if z[0] > 0]

    out = [
        f'  pixels that disagree at all (>8/255 on the worst channel): {frac_any:.1%}',
        f'  mean per-channel |delta| (0..255): R {chans[0]:.1f}  G {chans[1]:.1f}  B {chans[2]:.1f} '
        f'-- the largest one names the channel your colours are wrong in',
    ]
    if not worst:
        out.append(f'  no {t}px tile disagrees: within this region your render matches the target')
    else:
        out.append(
            f'  worst {t}px tiles, mean normalised difference, coordinates in pixels'
            + (' of the full slide' if (ox or oy) else '')
            + ':'
        )
        for i, (val, tx, ty, tx2, ty2) in enumerate(worst, 1):
            box = [tx + ox, ty + oy, tx2 + ox, ty2 + oy]
            out.append(f'      {i}. px {box} = {_pt_box(box, ratio)}  {val:.3f}')
        out.append('  fix these tiles first; a tile at 0.000 needs no edit')

    values = {
        'mean_diff': round(true_mean, 5),
        'frac_pixels_differing': round(frac_any, 4),
        'channel_mean_abs_delta': [round(c, 2) for c in chans],
        'worst_tiles': [
            {'bbox_px': [tx + ox, ty + oy, tx2 + ox, ty2 + oy], 'mean_diff': round(v, 4)}
            for v, tx, ty, tx2, ty2 in worst
        ],
    }
    return '\n'.join(out), values


class CropInput(BaseModel):
    bbox: list[float] = Field(
        description='Crop region as [x1, y1, x2, y2] in pixels of the reference image, origin top-left.'
    )
    source: str = Field(
        default='reference',
        description="Which image to crop: 'reference' (the target slide) or 'render' (your own last render).",
    )
    upscale: bool = Field(
        default=True,
        description='Upscale small crops so fine detail (small text, thin rules) stays legible.',
    )
    label: str = Field(default='', description='Optional note on what you are inspecting, for the trace.')


@tool_registry.register(
    'visual/ppt_inspect_region',
    meta={
        'name': 'inspect_region',
        'description': (
            'Crop and magnify a region of a slide image to read fine detail precisely.\n\n'
            'Use this to determine exact element positions, text content, font sizes and colours '
            'that are ambiguous in the full-page view.\n\n'
            'Coordinates are pixels of the image shown to you, origin at top-left, [x1, y1, x2, y2].\n'
            "Set source='render' to inspect your own latest render instead of the target."
            + _img_return_suffix('inspect')
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def inspect_region_tool(params: CropInput, context: ToolContext) -> ToolResponse:
    key = RENDER_KEY if params.source == 'render' else IMAGE_KEY
    image = _get_image(context, key)
    if image is None:
        which = 'render' if key == RENDER_KEY else 'reference'
        hint = ' Call render_slide first.' if key == RENDER_KEY else ''
        return _err(context, 'inspect_region', f'no {which} image available.{hint}', bbox=params.bbox)

    box = _clamp_bbox(params.bbox, *image.size)
    if box is None:
        return _err(
            context,
            'inspect_region',
            f'bbox {[int(v) for v in params.bbox]} is degenerate after clamping to '
            f'{image.size[0]}x{image.size[1]} (min side {_MIN_CROP_PX}px). Provide a larger region.',
            bbox=params.bbox,
        )

    cropped = image.crop(box)
    w, h = cropped.size

    if params.upscale and max(w, h) < _MAX_UPSCALE_PX:
        factor = min(_MAX_UPSCALE_PX / max(w, h), 4.0)
        if factor > 1.05:
            cropped = cropped.resize((int(w * factor), int(h * factor)), Image.Resampling.LANCZOS)

    updates = _record(
        context,
        {'tool': 'inspect_region', 'success': True, 'bbox': list(box), 'source': params.source, 'label': params.label},
    )
    measured, mvalues, mupdates = _measure_crop(context, image, box)
    updates.update(mupdates)
    updates[TOOL_RECORDS_KEY][-1].update(mvalues)
    note = (
        f'Region {list(box)} of {params.source} image '
        f'({image.size[0]}x{image.size[1]}), saved at {cropped.size[0]}x{cropped.size[1]}.\n'
        f'Measured in that region:\n{measured}'
    )
    note_img = _image_note(context, cropped, f'inspect_region_{list(box)}')
    updates.update(note_img['updates'])
    return ToolResponse(
        output=[
            {'type': 'text', 'text': note},
            note_img['part'],
        ],
        meta={'success': True, 'bbox': list(box), 'crop_size': list(cropped.size), **mvalues},
        context_updates=updates,
    )


class GridInput(BaseModel):
    step: int = Field(default=100, description='Grid spacing in pixels. Use 50 for dense layouts.')
    source: str = Field(default='reference', description="'reference' or 'render'.")


@tool_registry.register(
    'visual/ppt_overlay_grid',
    meta={
        'name': 'overlay_grid',
        'description': (
            'Overlay a labelled coordinate grid on the slide image to read off element positions.\n\n'
            'Use this before estimating x/y/width/height for elements: it converts "roughly one third '
            'across" into a number you can put in code. Labels show pixel coordinates.' + _img_return_suffix('grid')
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def overlay_grid_tool(params: GridInput, context: ToolContext) -> ToolResponse:
    from PIL import ImageDraw

    key = RENDER_KEY if params.source == 'render' else IMAGE_KEY
    image = _get_image(context, key)
    if image is None:
        return _err(context, 'overlay_grid', f'no {params.source} image available.')

    step = max(20, min(int(params.step), 400))
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    colour = (255, 0, 255)
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=colour, width=1)
        draw.text((x + 2, 2), str(x), fill=colour)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=colour, width=1)
        draw.text((2, y + 2), str(y), fill=colour)

    updates = _record(context, {'tool': 'overlay_grid', 'success': True, 'step': step, 'source': params.source})
    reading = _grid_reading(context, image, step)
    grid_img = _image_note(context, canvas, f'overlay_grid_{step}px')
    updates.update(grid_img['updates'])
    return ToolResponse(
        output=[
            {
                'type': 'text',
                'text': (
                    f'{params.source} image {w}x{h} with {step}px grid '
                    f'({-(-w // step)}x{-(-h // step)} cells). Labels are pixel coords.\n'
                    f'{reading}'
                ),
            },
            grid_img['part'],
        ],
        meta={'success': True, 'step': step, 'size': [w, h]},
        context_updates=updates,
    )


class CompareInput(BaseModel):
    bbox: list[float] | None = Field(
        default=None, description='Optional [x1,y1,x2,y2] to compare just one region instead of the whole slide.'
    )


@tool_registry.register(
    'visual/ppt_compare_render',
    meta={
        'name': 'compare_render',
        'description': (
            'Show your latest render beside the target slide, plus a per-pixel difference map, '
            'so you can see exactly which elements are wrong.\n\n'
            'Bright areas in the difference map are where your render disagrees with the target. '
            'Call render_slide first.' + _img_return_suffix('compare')
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def compare_render_tool(params: CompareInput, context: ToolContext) -> ToolResponse:
    from PIL import ImageChops

    ref = _get_image(context, IMAGE_KEY)
    rendered = _get_image(context, RENDER_KEY)
    if ref is None:
        return _err(context, 'compare_render', 'no reference image available.')
    if rendered is None:
        return _err(context, 'compare_render', 'no render available yet. Call render_slide first.')

    ref_rgb = ref.convert('RGB')
    cand = rendered.convert('RGB')
    if cand.size != ref_rgb.size:
        cand = cand.resize(ref_rgb.size, Image.Resampling.LANCZOS)

    if params.bbox:
        box = _clamp_bbox(params.bbox, *ref_rgb.size)
        if box is None:
            return _err(context, 'compare_render', f'bbox {params.bbox} is degenerate.', bbox=params.bbox)
        ref_rgb, cand = ref_rgb.crop(box), cand.crop(box)

    true_diff = ImageChops.difference(ref_rgb, cand)
    diff = true_diff.point(lambda v: min(255, v * 3))

    bbox_note = f' region {list(box)}' if params.bbox else ''
    amp_mean = sum(diff.convert('L').tobytes()) / (diff.size[0] * diff.size[1]) / 255.0
    origin = (int(box[0]), int(box[1])) if params.bbox else (0, 0)
    profile, dvalues = _diff_profile(context, ref_rgb, cand, origin=origin)

    updates = _record(context, {'tool': 'compare_render', 'success': True, 'mean_diff': round(amp_mean, 5)})
    updates[TOOL_RECORDS_KEY][-1].update(dvalues)
    updates[TOOL_RECORDS_KEY][-1]['mean_diff_amplified3x'] = round(amp_mean, 5)
    ref_img = _image_note(context, ref_rgb, 'compare_render_target')
    cand_img = _image_note(context, cand, 'compare_render_candidate')
    diff_img = _image_note(context, diff, 'compare_render_diffmap')
    updates.update(ref_img['updates'])
    updates.update(cand_img['updates'])
    updates.update(diff_img['updates'])
    return ToolResponse(
        output=[
            {'type': 'text', 'text': f'Target slide{bbox_note}:'},
            ref_img['part'],
            {'type': 'text', 'text': 'Your current render:'},
            cand_img['part'],
            {
                'type': 'text',
                'text': (
                    f'Difference map saved (brighter = larger disagreement, amplified 3x; '
                    f'amplified mean {amp_mean:.4f}).\n'
                    f'Measured difference, target vs your render:\n'
                    f'  mean normalised difference {dvalues["mean_diff"]:.4f}\n'
                    f'{profile}'
                ),
            },
            diff_img['part'],
        ],
        meta={'success': True, 'mean_diff': dvalues['mean_diff'], 'mean_diff_amplified3x': amp_mean},
        context_updates=updates,
    )
