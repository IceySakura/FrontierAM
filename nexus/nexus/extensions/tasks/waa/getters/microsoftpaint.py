"""Microsoft Paint getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import logging
import os

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.microsoftpaint')


def _find_red_circle(image_data: bytes) -> bool:
    """Detect a red circle with white interior in image data."""
    from io import BytesIO

    import numpy as np

    from PIL import Image

    if not image_data:
        return False
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')
    image_np = np.array(image)

    # Define the red color range for circumference
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Define the white color range for interior
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    # Create masks for red and white colors
    red_mask = ((image_np >= lower_red) & (image_np <= upper_red)).all(axis=-1)
    white_mask = ((image_np >= lower_white) & (image_np <= upper_white)).all(axis=-1)

    # Find the coordinates of the red pixels
    red_pixels = np.where(red_mask)

    if red_pixels[0].size == 0:
        return False

    # Calculate the centroid of the red pixels
    centroid_x = np.mean(red_pixels[1])
    centroid_y = np.mean(red_pixels[0])

    # Calculate the radius as the average distance from the centroid to the red pixels
    distances = np.sqrt((red_pixels[1] - centroid_x) ** 2 + (red_pixels[0] - centroid_y) ** 2)
    radius = int(np.mean(distances))

    # Define the bounding box of the detected circle
    top_left = (int(centroid_x - radius), int(centroid_y - radius))
    bottom_right = (int(centroid_x + radius), int(centroid_y + radius))

    # Extract the region inside the detected circle
    circle_interior = white_mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]  # type: ignore[index]

    # Check if the interior is predominantly white
    white_pixel_count = np.sum(circle_interior)
    total_pixel_count = circle_interior.size
    if total_pixel_count == 0:
        return False
    white_ratio = white_pixel_count / total_pixel_count

    # Define a threshold for considering it a valid white interior
    white_threshold = 0.8
    return white_ratio > white_threshold


@getter_registry.register()
async def get_is_red_circle_present_on_canvas(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a red circle is present in the specified screenshot file."""
    if not config['filepath']:
        return False
    screenshot_data = await runtime.get_file(config['filepath'])
    return await asyncio.to_thread(_find_red_circle, screenshot_data)


@getter_registry.register()
async def get_image_dimension_matches_input(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if an image file has the expected width and height."""
    from nexus.extensions.tasks.waa.getters.file import get_vm_file

    if not config['filepath']:
        return False
    if not config['width']:
        return False
    if not config['height']:
        return False

    filepath = config['filepath']
    expected_width = float(config['width'])
    expected_height = float(config['height'])

    # Download the file from VM
    file_local_path = await get_vm_file(runtime, {'path': filepath, 'dest': os.path.split(filepath)[-1]})

    def _check_dimensions(path: str) -> bool:
        from PIL import Image

        with Image.open(path) as img:
            return img.width == expected_width and img.height == expected_height

    return await asyncio.to_thread(_check_dimensions, file_local_path)
