"""Strict normalization for equivalent worksheet-qualified pivot ranges."""

import re

from openpyxl.utils import coordinate_to_tuple


def _pivot_source_coordinates(ref: str, sheet: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalize equivalent A1 ranges without discarding a conflicting sheet."""
    if '!' in ref:
        prefix, ref = ref.rsplit('!', 1)
        if prefix.startswith("'") and prefix.endswith("'"):
            prefix = prefix[1:-1].replace("''", "'")
        if prefix != sheet or '[' in prefix or ']' in prefix:
            raise ValueError('Pivot source range refers to a different or external worksheet')
    if not re.fullmatch(r'\$?[A-Z]{1,3}\$?[1-9][0-9]*:\$?[A-Z]{1,3}\$?[1-9][0-9]*', ref):
        raise ValueError('Pivot source must be a bounded A1 cell range')
    start, end = (coordinate_to_tuple(part.replace('$', '')) for part in ref.split(':'))
    if not (1 <= start[0] <= end[0] <= 1048576 and 1 <= start[1] <= end[1] <= 16384):
        raise ValueError('Pivot source range is reversed or outside worksheet bounds')
    return start, end
