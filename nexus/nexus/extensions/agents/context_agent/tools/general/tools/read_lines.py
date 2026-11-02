"""``read_lines`` — read a line range from a file on disk by path.

0-based, inclusive start / exclusive end; ``end=-1`` reads to EOF. Output is
capped at 80k chars / 500 lines, which is the main reason this exists
alongside ``bash``: ``cat`` on a 10 MB extracted document would dump the whole
thing into the conversation, while this degrades to a truncation notice.

Because this package does no long-line rewriting, ``L0`` is physical line 1 of
the file, so a range found with shell tools can be read back directly (minding
that shell line numbers start at 1).
"""

from __future__ import annotations

from ..fileio import FileReadError
from ..state import GeneralState


MAX_READ_CHARS = 80_000
MAX_READ_LINES = 500


def read_lines(state: GeneralState, path: str, start: int, end: int) -> str:
    if not path or not str(path).strip():
        return "(the 'path' argument is required — pass the file path to read)"

    resolved = state.resolve(path)
    try:
        corpus = state.read_lines(resolved)
    except FileReadError as exc:
        return f'[Error] {exc}'

    label = state.display(resolved)
    if not corpus:
        return f'(`{label}` is empty)'

    total = len(corpus)
    if end == -1:
        end = total
    start = max(0, min(start, total))
    end = max(start, min(end, total))

    if start == end:
        return f'(empty range in `{label}`; it has {total} lines)'

    if end - start > MAX_READ_LINES:
        end = start + MAX_READ_LINES

    lines: list[str] = []
    char_count = 0
    actual_end = start
    for i in range(start, end):
        formatted = f'L{i}: {corpus[i]}'
        char_count += len(formatted) + 1
        if char_count > MAX_READ_CHARS and lines:
            break
        lines.append(formatted)
        actual_end = i + 1

    warning = ''
    if actual_end < end:
        warning = f'\n\n(truncated at L{actual_end - 1} due to size limit; read smaller ranges for remaining content)'

    header = f'{label} — lines {start}-{actual_end - 1} (total: {total} lines)'
    return header + '\n\n' + '\n'.join(lines) + warning
