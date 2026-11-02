"""DeepEyes visual toolbox tools.

Provides the image_zoom_in_tool used by DeepEyes agents for visual reasoning.
"""

import base64
import io
import logging

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from PIL import Image

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


logger = logging.getLogger(__name__)

DEEPEYES_TOOL_RECORDS_KEY = '__deepeyes_tool_records__'


def _build_tool_context_updates(context: ToolContext, record: dict[str, Any]) -> dict[str, Any]:
    records = list(context.tool_state.get(DEEPEYES_TOOL_RECORDS_KEY, []))
    records.append(record)
    return {DEEPEYES_TOOL_RECORDS_KEY: records}


class ImageZoomInInput(BaseModel):
    """Input for image_zoom_in_tool."""

    bbox_2d: list[float] = Field(
        description='The bounding box of the region to zoom in, as [x1, y1, x2, y2], '
        'where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.',
        min_length=4,
        max_length=4,
    )
    label: str | None = Field(
        default=None,
        description='The name or label of the object in the specified bounding box (optional).',
    )


def _image_to_base64(image: 'Image.Image', fmt: str = 'JPEG') -> str:
    """Convert a PIL image to a base64-encoded string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _crop_image(image: 'Image.Image', bbox: list[float]) -> 'Image.Image | None':
    """Crop an image to the given bounding box [x1, y1, x2, y2].

    Returns None if the bounding box is degenerate (zero area after clamping).
    """
    x1, y1, x2, y2 = bbox
    width, height = image.size

    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        logger.warning(f'Degenerate bounding box {bbox} after clamping to {width}x{height}')
        return None

    return image.crop((x1, y1, x2, y2))


@tool_registry.register(
    'deepeyes/image_zoom_in_tool',
    meta={
        'name': 'image_zoom_in_tool',
        'description': (
            'Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) '
            'and an optional object label.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def image_zoom_in_tool(params: ImageZoomInInput, context: ToolContext) -> ToolResponse:
    """Zoom in on a specific region of an image.

    Reads the source image from tool_state['image'] (a PIL Image) and crops it
    to the given bounding box. The image stored in tool_state should already be
    resized to match the coordinate space the model uses (e.g. via smart_resize
    in the caller), so bbox coordinates map directly to image pixels.
    """
    from PIL import Image

    image: Any = context.tool_state.get('image')
    if image is None:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: no image available in tool state.'}],
            meta={'error': 'no_image'},
            context_updates=_build_tool_context_updates(
                context,
                {
                    'tool_name': 'image_zoom_in_tool',
                    'success': False,
                    'error': 'no_image',
                    'bbox_2d': params.bbox_2d,
                    'label': params.label,
                },
            ),
        )

    if not isinstance(image, Image.Image):
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: image in tool state is not a PIL Image.'}],
            meta={'error': 'invalid_image_type'},
            context_updates=_build_tool_context_updates(
                context,
                {
                    'tool_name': 'image_zoom_in_tool',
                    'success': False,
                    'error': 'invalid_image_type',
                    'bbox_2d': params.bbox_2d,
                    'label': params.label,
                },
            ),
        )

    cropped = _crop_image(image, params.bbox_2d)
    bbox_int = [int(v) for v in params.bbox_2d]
    if cropped is None:
        msg = (
            f'Error: the bounding box {bbox_int} is degenerate '
            f'(zero area after clamping to image size {image.size[0]}x{image.size[1]}). '
            f'Please provide a valid bounding box.'
        )
        logger.warning(f'image_zoom_in_tool: degenerate bbox={bbox_int}, image_size={image.size}')
        return ToolResponse(
            output=[{'type': 'text', 'text': msg}],
            meta={'error': 'degenerate_bbox', 'bbox_2d': params.bbox_2d},
            context_updates=_build_tool_context_updates(
                context,
                {
                    'tool_name': 'image_zoom_in_tool',
                    'success': False,
                    'error': 'degenerate_bbox',
                    'bbox_2d': params.bbox_2d,
                    'label': params.label,
                },
            ),
        )

    crop_w, crop_h = cropped.size
    if min(crop_w, crop_h) == 0 or max(crop_w, crop_h) / min(crop_w, crop_h) > 200:
        ratio = float('inf') if min(crop_w, crop_h) == 0 else max(crop_w, crop_h) / min(crop_w, crop_h)
        msg = (
            f'Error: the bounding box {bbox_int} produces an image with '
            f'extreme aspect ratio ({ratio:.1f}:1, size {crop_w}x{crop_h}). '
            f'Please provide a more reasonable bounding box.'
        )
        logger.warning(
            f'image_zoom_in_tool: rejected bbox={bbox_int}, cropped_size={cropped.size}, aspect_ratio={ratio:.1f}'
        )
        return ToolResponse(
            output=[{'type': 'text', 'text': msg}],
            meta={'error': 'extreme_aspect_ratio', 'bbox_2d': params.bbox_2d, 'cropped_size': list(cropped.size)},
            context_updates=_build_tool_context_updates(
                context,
                {
                    'tool_name': 'image_zoom_in_tool',
                    'success': False,
                    'error': 'extreme_aspect_ratio',
                    'bbox_2d': params.bbox_2d,
                    'label': params.label,
                    'cropped_size': list(cropped.size),
                },
            ),
        )

    img_b64 = _image_to_base64(cropped)

    label_info = f' with label "{params.label}"' if params.label else ''
    desc = f'Zoomed in on the image to the region {bbox_int}{label_info}.'

    logger.debug(f'image_zoom_in_tool: bbox={bbox_int}, cropped_size={cropped.size}{label_info}')

    return ToolResponse(
        output=[
            {'type': 'text', 'text': desc},
            {
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'},
            },
        ],
        meta={
            'bbox_2d': params.bbox_2d,
            'label': params.label,
            'cropped_size': list(cropped.size),
        },
        context_updates=_build_tool_context_updates(
            context,
            {
                'tool_name': 'image_zoom_in_tool',
                'success': True,
                'error': None,
                'bbox_2d': params.bbox_2d,
                'label': params.label,
                'cropped_size': list(cropped.size),
            },
        ),
    )
