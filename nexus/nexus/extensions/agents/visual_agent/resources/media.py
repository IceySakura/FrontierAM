"""Bounded local reads for visual media and workspace uploads."""

import base64
import struct

from pathlib import Path


MAX_MEDIA_BYTES = 64 * 1024 * 1024


def read_media_file(path: Path) -> bytes:
    try:
        if path.stat().st_size > MAX_MEDIA_BYTES:
            raise ValueError('Local media exceeds the 64 MiB limit')
        with path.open('rb') as stream:
            payload = stream.read(MAX_MEDIA_BYTES + 1)
    except OSError as exc:
        raise ValueError('Media source must be a readable local file; wrap raw base64 in a data: URI') from exc
    if len(payload) > MAX_MEDIA_BYTES:
        raise ValueError('Local media exceeds the 64 MiB limit')
    return payload


IMAGE_HEADER_BYTES = 64 * 1024


def image_header(source: str) -> bytes:
    """Read at most 64 KiB; unusually large JPEG metadata may hide dimensions."""
    if source.startswith('data:'):
        comma = source.find(',')
        if comma < 0 or not source[:comma].endswith(';base64'):
            raise ValueError('Image requires a base64 data URI')
        # Decode complete base64 groups without copying the full payload.
        return base64.b64decode(source[comma + 1 : comma + 1 + (IMAGE_HEADER_BYTES // 3) * 4], validate=True)
    with Path(source).expanduser().open('rb') as stream:
        return stream.read(IMAGE_HEADER_BYTES)


def image_mime(raw: bytes) -> str | None:
    """Prefer byte signatures over filenames for supported image formats."""
    for signature, mime in (
        (b'\x89PNG\r\n\x1a\n', 'image/png'),
        (b'\xff\xd8', 'image/jpeg'),
        (b'GIF87a', 'image/gif'),
        (b'GIF89a', 'image/gif'),
        (b'BM', 'image/bmp'),
        (b'II\x2a\x00', 'image/tiff'),
        (b'MM\x00\x2a', 'image/tiff'),
    ):
        if raw.startswith(signature):
            return mime
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        return 'image/webp'
    return None


def pixel_size(raw: bytes) -> tuple[int, int] | None:
    """Read raw image dimensions without decoding pixels."""
    if raw[:6] in (b'GIF87a', b'GIF89a') and len(raw) >= 10:
        width, height = struct.unpack('<HH', raw[6:10])
        return (width, height) if width and height else None
    if raw[:2] == b'BM' and len(raw) >= 26:
        dib_size = int.from_bytes(raw[14:18], 'little')
        if dib_size == 12:
            width, height = struct.unpack('<HH', raw[18:22])
        elif dib_size >= 40:
            width, height = struct.unpack('<ii', raw[18:26])
        else:
            return None
        return (width, abs(height)) if width > 0 and height else None
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        kind = raw[12:16]
        if kind == b'VP8X' and len(raw) >= 30:
            return 1 + int.from_bytes(raw[24:27], 'little'), 1 + int.from_bytes(raw[27:30], 'little')
        if kind == b'VP8L' and len(raw) >= 25 and raw[20] == 0x2F:
            bits = int.from_bytes(raw[21:25], 'little')
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if kind == b'VP8 ' and len(raw) >= 30 and raw[23:26] == b'\x9d\x01\x2a':
            width, height = struct.unpack('<HH', raw[26:30])
            width, height = width & 0x3FFF, height & 0x3FFF
            return (width, height) if width and height else None
        return None
    if raw[:8] == b'\x89PNG\r\n\x1a\n' and raw[12:16] == b'IHDR':
        try:
            width, height = struct.unpack('>II', raw[16:24])
        except struct.error:
            return None
        return (width, height) if width and height else None
    if raw[:2] == b'\xff\xd8':
        i = 2
        while i < len(raw):
            if raw[i] != 0xFF:
                return None
            while i < len(raw) and raw[i] == 0xFF:
                i += 1
            if i == len(raw):
                return None
            marker = raw[i]
            i += 1
            if marker in (0, 0xD9, 0xDA):
                return None  # Stop before entropy-coded data or end of image.
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue  # Standalone markers have no length field.
            if i + 2 > len(raw):
                return None
            (length,) = struct.unpack('>H', raw[i : i + 2])
            if length < 2:
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if length < 7 or i + 7 > len(raw):
                    return None
                height, width = struct.unpack('>HH', raw[i + 3 : i + 7])
                return (width, height) if width and height else None
            i += length
    return None
