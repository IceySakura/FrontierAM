"""Long-lived bashlex parser; execute by path to avoid importing the training stack.

Protocol: one JSON request per stdin line (``{"text": ..., "expansionlimit": ...}``) and one JSON
response per stdout line: ``{"commands": [[word, ...], ...]}``, or ``{"commands": null}`` when
bashlex rejects the input. The process exits on stdin EOF, or right after a response flagged
``"exiting": true`` once its peak RSS exceeds the limit given as ``argv[1]`` in MB, so a single
memory-hungry parse cannot pin that memory for the rest of the run.
"""

from __future__ import annotations

import json
import sys

from typing import Any


try:
    import resource
except ImportError:  # Windows has no resource module; RSS recycling is skipped there.
    resource = None  # type: ignore[assignment]


def _bashlex_command_words(node: Any) -> list[str]:
    """Return the literal words of a bashlex ``command`` AST node.

    Only ``word`` parts are collected (assignments / redirects skipped).
    Leading ``VAR=value`` assignment-only words are dropped so that
    ``FOO=1 python script.py`` classifies as ``python``, not as the env var.
    Returns ``[]`` for pure-assignment commands (``x=$(date +%s)``).
    """
    words: list[str] = []
    for p in getattr(node, 'parts', []) or []:
        if getattr(p, 'kind', '') != 'word':
            continue
        w = getattr(p, 'word', None)
        if w is None:
            continue
        s = str(w)
        # Detect ``VAR=value`` assignment-only word: starts with an identifier
        # followed by '='. We do not import re; do a manual character scan.
        if not words and _is_leading_assignment(s):
            continue
        words.append(s)
    return words


def _is_leading_assignment(token: str) -> bool:
    """Return True if *token* matches ``[A-Za-z_][A-Za-z0-9_]*=...`` (no regex)."""
    if not token:
        return False
    c0 = token[0]
    if not (c0.isalpha() or c0 == '_'):
        return False
    for i in range(1, len(token)):
        ch = token[i]
        if ch == '=':
            return True
        if not (ch.isalnum() or ch == '_'):
            return False
    return False


def _bashlex_collect_command_nodes(node: Any, acc: list[Any]) -> None:
    """Recursively collect ``kind == 'command'`` nodes from a bashlex AST."""
    if node is None:
        return
    kind = getattr(node, 'kind', '')
    if kind == 'command':
        acc.append(node)
    for attr in ('parts', 'list', 'commands', 'command'):
        v = getattr(node, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for c in v:
                if hasattr(c, 'kind'):
                    _bashlex_collect_command_nodes(c, acc)
        elif hasattr(v, 'kind'):
            _bashlex_collect_command_nodes(v, acc)


def _parse_command_words(bashlex: Any, request: dict[str, Any]) -> list[list[str]]:
    trees = bashlex.parse(request['text'], expansionlimit=request['expansionlimit'])
    cmd_nodes: list[Any] = []
    for tree in trees or []:
        _bashlex_collect_command_nodes(tree, cmd_nodes)
    commands = []
    for node in cmd_nodes:
        words = _bashlex_command_words(node)
        if words:
            commands.append(words)
    return commands


def _get_peak_rss_mb() -> float | None:
    if resource is None:
        return None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is reported in bytes on macOS and in KiB on Linux.
    return peak_rss / (1024 * 1024) if sys.platform == 'darwin' else peak_rss / 1024


def main() -> int:
    max_rss_mb = float(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        import bashlex
    except Exception:
        return 2
    for line in sys.stdin:
        try:
            commands: list[list[str]] | None = _parse_command_words(bashlex, json.loads(line))
        except Exception:
            commands = None
        peak_rss_mb = _get_peak_rss_mb()
        exiting = max_rss_mb is not None and peak_rss_mb is not None and peak_rss_mb > max_rss_mb
        sys.stdout.write(json.dumps({'commands': commands, 'exiting': exiting}) + '\n')
        sys.stdout.flush()
        if exiting:
            return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
