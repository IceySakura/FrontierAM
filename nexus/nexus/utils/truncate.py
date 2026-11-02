"""Utilities for truncating text and nested data structures."""

from __future__ import annotations

from typing import Any


# Default truncation marker
DEFAULT_MARKER = '\n... [truncated] ...\n'

# Default max size in bytes (256KB)
DEFAULT_MAX_BYTES = 256 * 1024


def truncate_string(
    text: str,
    max_size: int = DEFAULT_MAX_BYTES,
    marker: str = DEFAULT_MARKER,
    *,
    use_bytes: bool = True,
) -> str:
    """Truncate a string from the middle, preserving head and tail.

    Args:
        text: The string to truncate.
        max_size: Maximum size (in bytes if use_bytes=True, else characters).
        marker: The marker to insert in the middle when truncating.
        use_bytes: If True, measure size in UTF-8 bytes; if False, in characters.

    Returns:
        The original string if under limit, or truncated string with marker.
    """
    if use_bytes:
        encoded = text.encode('utf-8')
        if len(encoded) <= max_size:
            return text

        marker_size = len(marker.encode('utf-8'))
        available = max_size - marker_size
        if available <= 0:
            return marker[:max_size] if len(marker.encode('utf-8')) > max_size else marker

        head_size = available // 2
        tail_size = available - head_size

        # Decode safely to avoid breaking multi-byte characters
        head = encoded[:head_size].decode('utf-8', errors='ignore')
        tail = encoded[-tail_size:].decode('utf-8', errors='ignore')
        return head + marker + tail
    if len(text) <= max_size:
        return text

    available = max_size - len(marker)
    if available <= 0:
        return marker[:max_size]

    head_size = available // 2
    tail_size = available - head_size
    return text[:head_size] + marker + (text[-tail_size:] if tail_size > 0 else '')


def truncate_recursive(
    data: Any,
    max_size: int = DEFAULT_MAX_BYTES,
    marker: str = DEFAULT_MARKER,
    *,
    use_bytes: bool = True,
) -> Any:
    """Recursively truncate string values in nested data structures.

    Preserves the original data structure (dict, list, etc.) and only truncates
    string values that exceed the size limit.

    Args:
        data: The data to process (can be any type).
        max_size: Maximum size for string values.
        marker: The marker to insert when truncating.
        use_bytes: If True, measure size in UTF-8 bytes; if False, in characters.

    Returns:
        Data with the same structure but with long strings truncated.
    """
    if data is None:
        return None

    if isinstance(data, str):
        return truncate_string(data, max_size, marker, use_bytes=use_bytes)

    if isinstance(data, dict):
        return {k: truncate_recursive(v, max_size, marker, use_bytes=use_bytes) for k, v in data.items()}

    if isinstance(data, list):
        return [truncate_recursive(item, max_size, marker, use_bytes=use_bytes) for item in data]

    if isinstance(data, tuple):
        return tuple(truncate_recursive(item, max_size, marker, use_bytes=use_bytes) for item in data)

    # For other types (int, float, bool, etc.), return as-is
    return data
