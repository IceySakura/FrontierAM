"""Define the MNIST v2 fixed-grid contract and immutable per-episode tool state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from nexus.extensions.projects.mnist.tools import image_to_base64_url
from nexus.utils.multimodal import IMAGE_PLACEHOLDER, place_images_in_question


STATE_KEY: str = 'mnist_v2_state'
SELECTION_ORDER: Literal['spiral_down_right_up_left_v1'] = 'spiral_down_right_up_left_v1'
TOOL_NAMES: list[str] = ['mnist_v2/dissect', 'mnist_v2/calculator']
ImageLongSide = Annotated[StrictInt, Field(ge=28, le=4096)]


class CompositeImageConfig(BaseModel):
    """Describe one external 280x280 image and its required spiral prefix length."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    image_index: Annotated[StrictInt, Field(ge=0)]
    image_path: str
    rows: Literal[10] = 10
    columns: Literal[10] = 10
    tile_width: Literal[28] = 28
    tile_height: Literal[28] = 28
    selection_count: Annotated[StrictInt, Field(ge=1, le=100)]

    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        """Accept absolute local or mounted Ceph paths, not remote URLs."""
        if not Path(value).is_absolute():
            raise ValueError('image_path must be an absolute path readable by the task worker')
        return value


class CompositePhase(str, Enum):
    """Require a model observation between dependent tool calls."""

    DISSECT = 'dissect'
    DIGITS_PENDING = 'digits_pending'
    SUM = 'image'
    IMAGE_PENDING = 'image_pending'
    FINAL_PENDING = 'final_pending'
    FINAL = 'final'
    COMPLETE = 'complete'
    INVALID = 'invalid'


@dataclass(frozen=True)
class CompositeTaskState:
    """Retain the displayed image and publish state changes through context_updates.

    Args:
        composites: Ordered grid specifications; no labels or expected results.
        input_image_long_side: Display size of each incoming composite.
        tool_image_long_side: Display size of each selected digit.
        composite_images: All processed grids supplied together in the initial message.
        image: Processed pixels of the current grid, identical to the image sent to the model.
        image_index: Current grid index.
        phase: Next allowed operation or pending observation boundary.
        image_results: Values actually computed from the model's digit lists.
        image_digits: First accepted digit list per image, retained only for terminal scoring.
        final_result: Result returned by the final calculator call.
    """

    composites: tuple[CompositeImageConfig, ...]
    input_image_long_side: int
    tool_image_long_side: int
    composite_images: tuple[Image.Image, ...]
    image: Image.Image | None
    image_index: int = 0
    phase: CompositePhase = CompositePhase.DISSECT
    image_results: tuple[int, ...] = ()
    image_digits: tuple[tuple[int, ...], ...] = ()
    final_result: int | None = None


def build_spiral_indices() -> list[int]:
    """Visit the fixed grid down/right/up/left, proceeding inward without duplicates."""
    top = left = 0
    bottom = right = 9
    indices = []
    while top <= bottom and left <= right:
        indices.extend(row * 10 + left for row in range(top, bottom + 1))
        left += 1
        indices.extend(bottom * 10 + column for column in range(left, right + 1))
        bottom -= 1
        if left <= right:
            indices.extend(row * 10 + right for row in range(bottom, top - 1, -1))
            right -= 1
        if top <= bottom:
            indices.extend(top * 10 + column for column in range(right, left - 1, -1))
            top += 1
    return indices


def load_composite_image(config: CompositeImageConfig, long_side: int) -> tuple[Image.Image, dict[str, Any]]:
    """Prepare one resized image for both the model observation and tool state."""
    with Image.open(config.image_path) as source:
        if source.size != (280, 280):
            raise ValueError(f'Expected composite size 280x280, got {source.size}: {config.image_path}')
        image = source.convert('RGB')
    display = image.resize((long_side, long_side), Image.Resampling.NEAREST)
    return display, {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_to_base64_url(display)}},
        ],
    }


def build_initial_observation(
    user_prompt: str, composites: list[CompositeImageConfig], long_side: int
) -> tuple[tuple[Image.Image, ...], list[dict[str, Any]]]:
    """Load all grids and place them at the shared multimodal prompt markers."""
    if user_prompt.count(IMAGE_PLACEHOLDER) != len(composites):
        raise ValueError('Expected exactly one <image> placeholder per composite')
    images = []
    image_parts = []
    for composite in composites:
        image, message = load_composite_image(composite, long_side)
        images.append(image)
        image_parts.extend(message['content'])
    return tuple(images), place_images_in_question(user_prompt, image_parts)


def build_digit_parts(
    image: Image.Image, count: int, long_side: int, *, image_index: int | None = None
) -> list[dict[str, Any]]:
    """Crop the displayed grid in spiral order and label the returned digit order.

    Args:
        image: Processed composite pixels already sent to the model.
        count: Number of selected digits to return.
        long_side: Pixel side length for each returned digit image.
        image_index: Optional composite index included in the display labels.

    Grid boundaries use integer floor division so adjacent cells share an edge,
    with no gaps or overlaps when the displayed size is not divisible by ten.
    """
    parts = []
    label_prefix = f'Image {image_index}, digit' if image_index is not None else 'Digit'
    for position, index in enumerate(build_spiral_indices()[:count], start=1):
        row, column = divmod(index, 10)
        digit = image.crop((
            column * image.width // 10,
            row * image.height // 10,
            (column + 1) * image.width // 10,
            (row + 1) * image.height // 10,
        ))
        digit = digit.resize((long_side, long_side), Image.Resampling.NEAREST)
        parts.extend([
            {'type': 'text', 'text': f'{label_prefix} {position} of {count}:'},
            {'type': 'image_url', 'image_url': {'url': image_to_base64_url(digit)}},
        ])
    return parts
