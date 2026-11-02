"""Text shapes the HITL recipes build repeatedly."""

from typing import Any


def text_parts(text: str) -> list[Any]:
    """``text`` as the single content part a ``ToolResponse`` carries."""
    return [{'type': 'text', 'text': text}]


def section(title: str, body: str) -> str:
    """``body`` under ``title``, or just the body when there is no title."""
    body = body.strip()
    return f'{title}\n\n{body}' if title else body


def tagged(tag: str, body: str) -> str:
    """``body`` wrapped in ``<tag>`` on its own lines, or '' when empty."""
    stripped = body.strip()
    if not stripped:
        return ''
    return f'<{tag}>\n{stripped}\n</{tag}>'


def demote_headings(text: str, *, levels: int = 1) -> str:
    """``text`` with every ATX heading pushed down ``levels``.

    For quoting a document that was written to stand alone inside one that is not:
    its own sections open at ``#``, which would otherwise outrank the sections of
    whatever quotes it. A ``#`` inside a fenced block is content, so fences are
    tracked and their contents left alone.
    """
    lines: list[str] = []
    fenced = False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced and line.startswith('#'):
            line = '#' * levels + line
        lines.append(line)
    return '\n'.join(lines)


def truncate(text: str, limit: int) -> str:
    """``text`` shortened to ``limit`` characters, keeping both ends.

    The omission marker is added on top of ``limit``.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    dropped = len(text) - limit
    return f'{text[:head]}\n... [{dropped} characters omitted] ...\n{text[-tail:]}'


__all__ = ['demote_headings', 'section', 'tagged', 'text_parts', 'truncate']
