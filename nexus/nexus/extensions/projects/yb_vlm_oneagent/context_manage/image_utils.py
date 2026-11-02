"""Image loading helpers for yb_vlm_oneagent context management."""

import base64

from pathlib import Path
from typing import Any

import requests

from PIL import Image, ImageFile


# Some datasets contain JPEGs with malformed tail markers. Pillow can still
# decode them if truncated image loading is enabled.
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]


def load_image_bytes(image_source: Any, timeout: int = 30) -> bytes:
    """Load raw image bytes from a local path, URL, or dataset patch dict."""
    if isinstance(image_source, dict):
        if 'bytearray' in image_source and isinstance(image_source['bytearray'], dict):
            patch_info = image_source['bytearray']
            encoding = image_source.get('image_encoding', 'bytearray')
        elif 'patch' in image_source:
            patch_info = image_source
            encoding = image_source.get('image_encoding') or image_source.get('image_encoing')
        else:
            raise ValueError(f'Unsupported image dict format, keys: {list(image_source.keys())}')

        patch_file = str(patch_info['patch'])
        start_bytes = int(patch_info['start_num'])
        file_size = int(patch_info['size'])

        if encoding is None:
            if any(key in patch_file for key in ('OBELISC', 'mp/url_base64', 'WIT/WIT-image')):
                encoding = 'base64'
            else:
                encoding = 'bytearray'

        with Path(patch_file).open('rb') as f:
            f.seek(start_bytes)
            raw = f.read(file_size)

        return base64.b64decode(raw.decode()) if encoding == 'base64' else raw

    if isinstance(image_source, str):
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source, timeout=timeout)
            response.raise_for_status()
            return response.content
        return Path(image_source).read_bytes()

    raise ValueError(f'Unsupported image_source type: {type(image_source)}')


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert a PIL image to RGB, compositing transparent pixels on white."""
    if image.mode == 'RGB':
        return image

    if image.mode in ('RGBA', 'LA', 'PA', 'P'):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background

    return image.convert('RGB')
