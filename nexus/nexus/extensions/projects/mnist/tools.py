"""MNIST scrambled-digit multiplication tools.

Two LOCAL tools that together make the task solvable only through tool use:

``decode``
    Undo the fixed 2x2 block permutation applied to some of the images, so the
    digit becomes readable. The scrambled images are unreadable to the model as
    they stand, which is what forces the call.
``calculate``
    Multiply the recognised digits. The product is deliberately delegated to a
    tool so the task exercises tool calling rather than arithmetic.

The image entities live in ``context.tool_state['image_list']``, written by
``mnist_task``; a tool parameter can only carry the index. Model output is JSON
and the pydantic-to-OpenAI schema only admits JSON primitives, so a PIL image
cannot be declared as a parameter at all.

``decode`` is adapted verbatim from the verl reference implementation
(``$BUNDLE/runtime/verl/mnist_tool.py``) so the permutation matches the one the
dataset was generated with.
"""

from __future__ import annotations

import base64
import io
import logging

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_LIST_STATE_KEY = 'image_list'

# The generator cut each scrambled image into CUT_PIECE x CUT_PIECE tiles and
# wrote them back in SCRAMBLE_ORDER. Both constants must match the dataset.
CUT_PIECE = 2
SCRAMBLE_ORDER = (2, 0, 3, 1)


class DecodeInput(BaseModel):
    """Input for the decode tool."""

    image_idx: int = Field(description='Zero-based index of the image to decode, in the order they were shown.')


class CalculateInput(BaseModel):
    """Input for the calculate tool."""

    numbers: list[int] = Field(description='The digits read from the images, in any order.')


class ReadImageInput(BaseModel):
    """Address an original attachment independently of compacted message history."""

    image_idx: int = Field(description='Stable zero-based index of the original task image to reread.')


def _tile_bounds(index: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Return the (left, upper, right, lower) box of tile ``index``.

    Both sides must divide by CUT_PIECE, so every tile has the same size and the
    permutation is a true bijection over them; ``decode`` enforces that.
    """
    piece_width = width // CUT_PIECE
    piece_height = height // CUT_PIECE
    row, column = divmod(index, CUT_PIECE)
    left = column * piece_width
    upper = row * piece_height
    return left, upper, left + piece_width, upper + piece_height


def decode(scrambled_image: Image.Image) -> Image.Image:
    """Undo the fixed tile permutation that hid the digit.

    Returns a new image with the tiles put back in reading order; the input is
    left untouched.

    Raises:
        ValueError: If either side is not divisible by CUT_PIECE.
    """
    from PIL import Image as PILImage

    width, height = scrambled_image.size
    # NOTE(arlenxu): a side that does not divide by CUT_PIECE has no exact inverse.
    # The tiles then differ in size, so the permutation would have to move a tile
    # into a slot holding a different number of pixels; resizing it would invent
    # pixels and pasting it as-is would leave part of the slot unwritten. Both
    # corrupt the digit the model has to read, so refuse instead. The dataset
    # generator emits 28x28, which divides evenly.
    if width % CUT_PIECE or height % CUT_PIECE:
        raise ValueError(
            f'decode needs both sides divisible by CUT_PIECE={CUT_PIECE}, got {width}x{height}; '
            'the tile permutation has no exact inverse otherwise'
        )
    tiles = [scrambled_image.crop(_tile_bounds(index, width, height)) for index in range(CUT_PIECE * CUT_PIECE)]

    unscramble_order = [0] * (CUT_PIECE * CUT_PIECE)
    for destination, source in enumerate(SCRAMBLE_ORDER):
        unscramble_order[source] = destination

    decoded = PILImage.new('RGB', (width, height))
    for index in range(CUT_PIECE * CUT_PIECE):
        left, upper, _right, _lower = _tile_bounds(index, width, height)
        decoded.paste(tiles[unscramble_order[index]], (left, upper))
    return decoded


def calculate(numbers: list[int]) -> int:
    """Return the product of ``numbers`` (the empty product is 1)."""
    product = 1
    for number in numbers:
        product *= number
    return product


def image_to_base64_url(image: Image.Image, fmt: str = 'PNG') -> str:
    """Encode a PIL image as a data-URL string.

    PNG rather than JPEG: these are 28x28 line drawings where JPEG ringing would
    blur the very strokes the model has to read.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/{fmt.lower()};base64,{b64}'


@tool_registry.register(
    'mnist/decode',
    meta={
        'name': 'decode',
        'description': (
            'Decode an image whose digit is unreadable because the picture was cut into four '
            'tiles and shuffled. Returns the restored image.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def decode_tool(params: DecodeInput, context: ToolContext) -> ToolResponse:
    """Decode the image at ``params.image_idx`` and return it as a new image part."""
    images: Any = context.tool_state.get(IMAGE_LIST_STATE_KEY)
    if not images:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: no images available for this task.'}],
            meta={'error': 'no_images'},
        )

    # NOTE(arlenxu): report an out-of-range index instead of clamping it. Clamping
    # would decode a different image than the model asked for and still look like
    # a success, which is exactly the kind of silent mismatch this task exists to
    # catch.
    if not 0 <= params.image_idx < len(images):
        message = f'Error: image_idx {params.image_idx} is out of range; this task has {len(images)} images (0-based).'
        logger.warning(f'mnist/decode: {message}')
        return ToolResponse(
            output=[{'type': 'text', 'text': message}],
            meta={'error': 'index_out_of_range', 'image_idx': params.image_idx},
        )

    decoded = decode(images[params.image_idx])
    logger.debug(f'mnist/decode: image_idx={params.image_idx}, size={decoded.size}')
    return ToolResponse(
        output=[
            {'type': 'text', 'text': f'Image {params.image_idx} decoded.'},
            {'type': 'image_url', 'image_url': {'url': image_to_base64_url(decoded)}},
        ],
        meta={'image_idx': params.image_idx},
    )


@tool_registry.register(
    'mnist/read_image',
    meta={
        'name': 'read_image',
        'description': (
            'Reread an original task image after context compaction using its stable zero-based image_idx. '
            'Returns unchanged pixels, without decoding shuffled tiles or revealing a digit label. '
            'Use decode only if the original image is scrambled.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def read_image_tool(params: ReadImageInput, context: ToolContext) -> ToolResponse:
    """Recover original pixels from task state, which survives history compaction."""
    images = context.tool_state.get(IMAGE_LIST_STATE_KEY)
    if not images:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: no original images available for this task.'}],
            meta={'error': 'no_images'},
        )
    if not 0 <= params.image_idx < len(images):
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'Error: image_idx must be in [0, {len(images) - 1}], got {params.image_idx}.',
                }
            ],
            meta={'error': 'index_out_of_range', 'image_idx': params.image_idx},
        )
    return ToolResponse(
        output=[
            {'type': 'text', 'text': f'Original image {params.image_idx} (unchanged).'},
            {'type': 'image_url', 'image_url': {'url': image_to_base64_url(images[params.image_idx])}},
        ],
        meta={'image_idx': params.image_idx},
    )


@tool_registry.register(
    'mnist/calculate',
    meta={
        'name': 'calculate',
        'description': 'Apply the new math function to a set of numbers and return its result.',
        'tool_type': ToolType.LOCAL,
    },
)
async def calculate_tool(params: CalculateInput, context: ToolContext) -> ToolResponse:
    """Return the product of ``params.numbers``."""
    result = calculate(params.numbers)
    logger.debug(f'mnist/calculate: numbers={params.numbers}, result={result}')
    return ToolResponse(
        output=[{'type': 'text', 'text': str(result)}],
        meta={'result': result},
    )
