"""Small formatting helpers for dataset diagnostics."""

from __future__ import annotations


def format_bytes(size: int) -> str:
    """Format a byte count with decimal units for logs."""

    if size < 0:
        return f'{size} B'
    value = float(size)
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    for unit in units:
        if value < 1000 or unit == units[-1]:
            if unit == 'B':
                return f'{size} B'
            return f'{value:.1f} {unit}'
        value /= 1000
    return f'{size} B'


def format_bytes_detail(size: int) -> str:
    """Format a byte count with a human value and exact bytes."""

    human = format_bytes(size)
    if human == f'{size} B':
        return human
    return f'{human} ({size} bytes)'
