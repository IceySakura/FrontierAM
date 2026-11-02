"""``grep`` — search a file (or everything downloaded) for a substring/regex.

Returns matching line numbers, a cluster summary grouping nearby matches, and
a context block around every match. A large ``after`` turns this into an exact
marker locator: ``grep(pattern=MARKER, after=25)`` returns the marker line plus
what follows, which BM25 file_search cannot do because random strings and UUIDs
carry no semantics.

Omitting ``path`` searches every text file the agent has: the task's own local
files plus anything it downloaded or wrote. Both halves matter — a local-path
task never puts anything in the workspace, so a workspace-only sweep would
silently miss its entire corpus.

Line numbers are 0-based to match ``read_lines``; shell ``grep -n`` is 1-based,
so the two differ by one.

**Regex intent is recovered rather than punished.** ``use_regex`` defaults to
false, so a pattern is escaped and matched literally — which silently returns
nothing when the model wrote a regex and forgot the flag. Measured over one
5,690-task run that was the single largest source of wasted rounds: 1,029 grep
calls passed a regex-shaped pattern with the flag unset and 889 of them (86%)
came back empty, indistinguishable from "the document does not contain this".
The most common shapes were case classes (``[Ss]upplier``), anchors (``^#``)
and alternations (``melt|broke|tensile``) — all reasonable things to write.

So a literal search that finds nothing is retried as a regex before giving up,
and the answer says which interpretation produced the hits. The retry can only
ever turn an empty result into a non-empty one, so it costs nothing when the
model really did mean a literal string.

**Wide results are paginated, not silently dropped.** Two ideas are taken from
``claude-code``'s ripgrep-backed Grep: an ``output_mode`` (``content`` /
``files_with_matches`` / ``count``) and ``head_limit`` + ``offset``. The same
run showed why both are needed: 47.7% of successful greps hit the block cap and
only 9.4% of the matched blocks were ever rendered, with no way to reach the
rest; and cross-file searches spanned a median of 189 files, where printing
content is the wrong answer entirely and "which files contain this" is the
right one. Every truncated result now ends with the exact call that fetches the
next page.
"""

from __future__ import annotations

import re

from pathlib import Path

from ..fileio import FileReadError
from ..state import GeneralState


MAX_PREVIEW_CHARS = 2000  # per-line cap
MAX_MATCH_BLOCKS = 10  # default context blocks per page (content mode)
MAX_FILE_ROWS = 100  # default file rows per page (compact modes)
MAX_TOTAL_CHARS = 12_000  # max total chars of rendered context blocks

_OUTPUT_MODES = frozenset({'content', 'files_with_matches', 'count'})

# Characters that mean something other than themselves in a regex. A bare "."
# is included on purpose: alone it is weak evidence of intent (``doc_1.md``,
# ``Fig. 4``), but the regex retry only runs after a literal search already
# returned nothing, so the worst case is a superset of zero.
_REGEX_META = frozenset(r'[]|^$*+?{}()\.')


def _looks_like_regex(pattern: str) -> bool:
    return any(ch in _REGEX_META for ch in pattern)


def _cap_line(line: str, limit: int = MAX_PREVIEW_CHARS) -> str:
    if len(line) <= limit:
        return line
    return line[:limit] + ' ...'


def _cluster_indices(indices: list[int], gap: int = 5) -> list[list[int]]:
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= gap:
            clusters[-1].append(indices[i])
        else:
            clusters.append([indices[i]])
    return clusters


def _cluster_summary(match_indices: list[int]) -> list[str]:
    clusters = _cluster_indices(match_indices, gap=5)
    cluster_strs: list[str] = []
    for cl in clusters:
        if len(cl) == 1:
            cluster_strs.append(f'L{cl[0]} (1)')
        else:
            cluster_strs.append(f'L{cl[0]}-L{cl[-1]} ({len(cl)})')
    return [
        f'Found {len(match_indices)} matches at lines: {match_indices}',
        'Clusters: ' + ', '.join(cluster_strs),
    ]


def _render_block(corpus: list[str], idx: int, before: int, after: int) -> str:
    """Render one match's context window.

    No "more lines follow, use read_lines" footer: it was emitted after every
    block, and since a match is almost never at EOF it fired essentially
    always. Over one 5,690-task run that single repeated sentence accounted for
    29% of all grep output — 10.5M characters, ~2.6M tokens — while saying
    nothing the cluster summary (which lists every matching line number) and
    the truncation notice do not already say.
    """
    start = max(0, idx - before)
    end = min(len(corpus), idx + after + 1)
    return '\n'.join(f'L{j}: {_cap_line(corpus[j])}' + ('  <<< match' if j == idx else '') for j in range(start, end))


def _resolve_targets(state: GeneralState, path: str | list | None) -> tuple[list[Path], str]:
    """Map ``path`` (None / str / list[str]) to target files plus a note."""
    if path is None or path == '' or path == []:
        found = state.known_files()
        if not found:
            return [], state.no_targets_hint()
        return found, ''

    raw_list = [path] if isinstance(path, str) else [str(p) for p in path]
    targets: list[Path] = []
    missing: list[str] = []
    for raw in raw_list:
        resolved = state.resolve(raw)
        if resolved.is_file():
            targets.append(resolved)
        else:
            missing.append(raw)
    note = ''
    if missing:
        note = f'Note: no such file(s): {missing}.\n'
    return targets, note


def grep(
    state: GeneralState,
    pattern: str,
    *,
    path: str | list | None = None,
    use_regex: bool = False,
    context_lines: int = 1,
    before: int | None = None,
    after: int | None = None,
    output_mode: str = 'content',
    head_limit: int | None = None,
    offset: int = 0,
) -> str:
    if not pattern or not pattern.strip():
        return '(empty pattern — please provide a non-empty search term)'

    eff_before = max(0, int(before if before is not None else context_lines))
    eff_after = max(0, int(after if after is not None else context_lines))
    output_mode = str(output_mode or 'content')
    if output_mode not in _OUTPUT_MODES:
        return f'(unknown output_mode {output_mode!r}; expected one of {", ".join(sorted(_OUTPUT_MODES))})'
    default_limit = MAX_MATCH_BLOCKS if output_mode == 'content' else MAX_FILE_ROWS
    eff_limit = default_limit if head_limit is None else max(1, int(head_limit))
    eff_offset = max(0, int(offset))

    targets, note = _resolve_targets(state, path)
    if not targets:
        return note or '(no files to search)'

    # Read every target once; both the primary pattern and the fallback are
    # then matched against the same in-memory lines.
    loaded: list[tuple[str, list[str]]] = []
    for target in targets:
        label = state.display(target)
        try:
            loaded.append((label, state.read_lines(target)))
        except FileReadError as exc:
            note += f'[skipped {label}: {exc}]\n'
    if not loaded:
        return note or '(no readable files to search)'

    def run(expr: str) -> tuple[list[tuple[str, list[str], list[int]]], int] | None:
        """Match *expr* over the loaded files, or None if it will not compile."""
        try:
            compiled = re.compile(expr, re.IGNORECASE)
        except re.error:
            return None
        out: list[tuple[str, list[str], list[int]]] = []
        total = 0
        for label, corpus in loaded:
            idxs = [i for i, line in enumerate(corpus) if compiled.search(line)]
            out.append((label, corpus, idxs))
            total += len(idxs)
        return out, total

    primary = run(pattern if use_regex else re.escape(pattern))
    if primary is None:
        # An invalid regex used to be a dead end. Fall back to a literal search
        # so a malformed pattern still returns whatever it matches verbatim.
        literal = run(re.escape(pattern))
        if literal is None or literal[1] == 0:
            return note + (
                f'(invalid regex pattern: "{pattern}" — it does not compile, and '
                'searching for it as literal text found nothing either)'
            )
        per_file, total_matches = literal
        note += f'Note: "{pattern}" is not a valid regex; searched for it as literal text instead.\n'
    else:
        per_file, total_matches = primary
        if total_matches == 0 and not use_regex and _looks_like_regex(pattern):
            # The flag was probably just forgotten. Recover instead of
            # reporting an empty result that looks like a genuine absence.
            as_regex = run(pattern)
            if as_regex is not None and as_regex[1] > 0:
                per_file, total_matches = as_regex
                note += (
                    f'Note: no LITERAL match for "{pattern}", but reading it as a '
                    f'regular expression found {total_matches}. Showing those. '
                    f'Pass use_regex=true to ask for this directly.\n'
                )

    if total_matches == 0:
        where = per_file[0][0] if len(per_file) == 1 else f'{len(per_file)} file(s)'
        extra = ''
        if not use_regex and _looks_like_regex(pattern):
            extra = ' (also tried interpreting it as a regular expression)'
        return note + (f'(no matches found for pattern: "{pattern}" in {where}{extra})')

    hit_files = [(label, corpus, idxs) for label, corpus, idxs in per_file if idxs]

    # ---- compact modes: which files, and how many hits in each -------------
    if output_mode in ('files_with_matches', 'count'):
        rows = sorted(hit_files, key=lambda x: -len(x[2]))
        shown = rows[eff_offset : eff_offset + eff_limit]
        lines = [
            f'grep "{pattern}": {total_matches} match(es) in {len(hit_files)} of {len(per_file)} file(s) searched.'
        ]
        for label, _corpus, idxs in shown:
            if output_mode == 'count':
                lines.append(f'  {len(idxs):6d}  {label}')
            else:
                preview = ', '.join(f'L{i}' for i in idxs[:8])
                more = f', +{len(idxs) - 8} more' if len(idxs) > 8 else ''
                lines.append(f'  {label}  ({len(idxs)}): {preview}{more}')
        if eff_offset + len(shown) < len(rows):
            lines.append(
                f'[showing {eff_offset + 1}-{eff_offset + len(shown)} of {len(rows)} '
                f'matching file(s); pass offset={eff_offset + len(shown)} for more]'
            )
        lines.append('Next: grep(pattern=..., path="<one of the files above>") to see the matching lines in context.')
        return note + '\n'.join(lines)

    # ---- content mode ------------------------------------------------------
    sections: list[str] = []
    blocks_used = 0
    chars_used = 0
    skipped = eff_offset
    rendered_total = 0
    truncated = False
    for label, corpus, idxs in hit_files:
        body_parts = _cluster_summary(idxs)
        rendered_here = 0
        for idx in idxs:
            if skipped:
                skipped -= 1
                continue
            if blocks_used >= eff_limit or chars_used >= MAX_TOTAL_CHARS:
                truncated = True
                break
            block = _render_block(corpus, idx, eff_before, eff_after)
            body_parts.append('')
            body_parts.append(block)
            blocks_used += 1
            rendered_here += 1
            rendered_total += 1
            chars_used += len(block)
        if rendered_here < len(idxs):
            body_parts.append(f'[showing {rendered_here} of {len(idxs)} match block(s) in {label}]')
        sections.append(f'=== {label} ===\n' + '\n'.join(body_parts))
        if truncated:
            break

    tail: list[str] = []
    if truncated or rendered_total + eff_offset < total_matches:
        # Truncation used to be a dead end — the remaining blocks were simply
        # unreachable through grep. Say exactly how to get the next page.
        tail.append(
            f'[{rendered_total} of {total_matches} match block(s) shown. '
            f'For the next page: grep(pattern=..., offset='
            f'{eff_offset + rendered_total}). For an overview instead: '
            f'output_mode="files_with_matches".]'
        )
    if len(per_file) > 1:
        head = (
            f'grep "{pattern}" across {len(per_file)} file(s); '
            f'{len(hit_files)} had matches (before={eff_before}, after={eff_after}).\n'
        )
        if len(per_file) > 8:
            head += (
                f'Tip: with {len(per_file)} files, output_mode="files_with_matches" '
                f'shows where the hits are without printing content.\n'
            )
        body = head + '\n\n'.join(sections)
    else:
        body = sections[0] if sections else ''
    return note + body + ('\n' + '\n'.join(tail) if tail else '')
