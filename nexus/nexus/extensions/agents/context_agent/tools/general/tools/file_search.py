"""``file_search`` — msearch / mclick BM25 retrieval over files on disk.

Each target file is chunked independently (line numbers stay file-local) and
indexed in a single pure-stdlib BM25 index. Candidate chunks carry a pointer
``{fidx}:c{cid}`` where ``fidx`` indexes the file list of the index that
produced it, so ``2:c5`` is the 6th chunk of the 3rd file. ``mclick`` expands a
pointer to its neighbouring chunks within the same file.

The index is rebuilt whenever the target set changes or any target's
mtime/size changes. That matters more here than for a design with a fixed
document set: the agent writes its own extracted ``.txt`` mid-rollout, and a
stale index would silently search the previous version.

No network, no external dependencies — BM25 plus quoted-substring boosting on
the standard library.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass
from pathlib import Path

from ..fileio import FileReadError
from ..state import GeneralState


_TOKEN_RE = re.compile(r'\w+', re.UNICODE)
_QUOTED_RE = re.compile(r'"([^"]+)"')

_SNIPPET_MAX_CHARS = 1_500  # per chunk in msearch results
_BM25_K1 = 1.5
_BM25_B = 0.75
_MCLICK_MAX_POINTERS = 3
_MCLICK_NEIGHBORS = 1  # chunks of context on each side
_POINTER_RE = re.compile(r'\s*(\d+):c(\d+)\s*')


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _IndexedFile:
    file_idx: int
    label: str
    lines: list[str]


@dataclass
class _Chunk:
    file_idx: int
    cid: int  # 0-based chunk index WITHIN its file
    l_start: int  # inclusive 0-based line (file-local)
    l_end: int  # inclusive 0-based line (file-local)
    text: str
    tokens: list[str]

    @property
    def pointer(self) -> str:
        return f'{self.file_idx}:c{self.cid}'


class _BM25Index:
    """Minimal Okapi BM25 over fixed-size line chunks across all target files."""

    def __init__(self, files: list[_IndexedFile], chunk_lines: int):
        self.files = files
        self.by_idx: dict[int, _IndexedFile] = {f.file_idx: f for f in files}
        self.chunks: list[_Chunk] = []
        self.by_file: dict[int, list[_Chunk]] = {}
        step = max(1, chunk_lines)
        for f in files:
            file_chunks: list[_Chunk] = []
            n = len(f.lines)
            for cid, start in enumerate(range(0, n, step)):
                end = min(start + step, n)
                text = '\n'.join(f.lines[start:end])
                ch = _Chunk(f.file_idx, cid, start, end - 1, text, _tokenize(text))
                self.chunks.append(ch)
                file_chunks.append(ch)
            self.by_file[f.file_idx] = file_chunks

        self._df: dict[str, int] = {}
        for ch in self.chunks:
            for term in set(ch.tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self.chunks)
        self._avgdl = sum(len(c.tokens) for c in self.chunks) / self._n if self._n else 0.0
        self._idf: dict[str, float] = {}
        for term, df in self._df.items():
            self._idf[term] = math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def _score(self, query_terms: list[str], ch: _Chunk) -> float:
        if not ch.tokens:
            return 0.0
        freq: dict[str, int] = {}
        for t in ch.tokens:
            freq[t] = freq.get(t, 0) + 1
        dl = len(ch.tokens)
        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (self._avgdl or 1))
            score += idf * (tf * (_BM25_K1 + 1)) / (denom or 1)
        return score

    def search(self, query: str, top_k: int) -> list[tuple[_Chunk, float]]:
        terms = _tokenize(query)
        phrases = [p.lower() for p in _QUOTED_RE.findall(query)]
        scored: list[tuple[_Chunk, float]] = []
        for ch in self.chunks:
            s = self._score(terms, ch)
            if phrases:
                low = ch.text.lower()
                for ph in phrases:
                    if ph and ph in low:
                        s += 5.0  # substring boost
            if s > 0:
                scored.append((ch, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def by_pointer(self, pointer: str) -> _Chunk | None:
        m = _POINTER_RE.fullmatch(pointer)
        if not m:
            return None
        fidx, cid = int(m.group(1)), int(m.group(2))
        file_chunks = self.by_file.get(fidx)
        if file_chunks and 0 <= cid < len(file_chunks):
            return file_chunks[cid]
        return None


def _resolve_targets(state: GeneralState, files: str | list | None) -> tuple[list[Path], str]:
    if files is None or files == '' or files == []:
        found = state.known_files()
        if not found:
            return [], state.no_targets_hint()
        return found, ''

    raw_list = [files] if isinstance(files, str) else [str(p) for p in files]
    targets: list[Path] = []
    missing: list[str] = []
    for raw in raw_list:
        resolved = state.resolve(raw)
        if resolved.is_file():
            targets.append(resolved)
        else:
            missing.append(raw)
    note = f'Note: no such file(s): {missing}.\n' if missing else ''
    return targets, note


def _get_index(state: GeneralState, targets: list[Path]) -> tuple[_BM25Index | None, str]:
    """Build (or reuse) the BM25 index over *targets*.

    Keyed on every target's mtime/size so an extract the agent rewrote is
    re-indexed rather than silently searched at its old content.
    """
    key_parts: list[tuple] = []
    for path in targets:
        try:
            stat = path.stat()
        except OSError:
            continue
        key_parts.append((str(path), stat.st_mtime_ns, stat.st_size))
    key = (tuple(key_parts), state.knobs.file_search_chunk_lines)

    if state._bm25 is not None and state._bm25_key == key:
        return state._bm25, ''

    note = ''
    indexed: list[_IndexedFile] = []
    for path in targets:
        try:
            lines = state.read_lines(path)
        except FileReadError as exc:
            note += f'[skipped {state.display(path)}: {exc}]\n'
            continue
        # Stable per-rollout index, NOT the position in this build: a pointer
        # the model was handed earlier must keep pointing at the same file even
        # after the searchable set grows.
        fidx = state.file_index_for(path)
        indexed.append(_IndexedFile(file_idx=fidx, label=state.display(path), lines=lines))
        try:
            stat = path.stat()
            content_key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            content_key = None
        previous = state._indexed_content.get(fidx)
        if previous is not None and previous != content_key:
            # The file changed since it was last chunked, so old cNN offsets no
            # longer mean what they meant. Drop only that file's pointers.
            prefix = f'{fidx}:'
            state.msearch_pointers.difference_update({p for p in state.msearch_pointers if p.startswith(prefix)})
        state._indexed_content[fidx] = content_key

    if not indexed:
        return None, note or '(no readable files to search)'

    index = _BM25Index(indexed, state.knobs.file_search_chunk_lines)
    state._bm25 = index
    state._bm25_key = key
    return index, note


def _render_snippet(f: _IndexedFile, l_start: int, l_end: int) -> tuple[str, int]:
    out: list[str] = []
    used = 0
    last = l_start
    for i in range(l_start, l_end + 1):
        formatted = f'L{i}: {f.lines[i]}'
        if out and used + len(formatted) + 1 > _SNIPPET_MAX_CHARS:
            out.append(f'...(snippet truncated at L{last}; use file_search_mclick or read_lines to see L{i}-L{l_end})')
            break
        out.append(formatted)
        used += len(formatted) + 1
        last = i
    return '\n'.join(out), last


def msearch(
    state: GeneralState,
    queries: list[str],
    top_k: int | None = None,
    files: str | list | None = None,
) -> str:
    """Run one or more BM25 queries; return ranked CANDIDATE chunk hits."""
    if not queries:
        return '(no queries provided — pass a non-empty list of query strings)'

    targets, target_note = _resolve_targets(state, files)
    if not targets:
        return target_note or '(no files to search)'

    index, index_note = _get_index(state, targets)
    if index is None:
        return target_note + index_note
    note = target_note + index_note

    k = top_k or state.knobs.file_search_top_k

    best: dict[str, tuple[_Chunk, float]] = {}
    per_query: list[str] = []
    for q in queries:
        hits = index.search(str(q), k)
        per_query.append(f'  "{q}" -> {len(hits)} hit(s)')
        for ch, score in hits:
            prev = best.get(ch.pointer)
            if prev is None or score > prev[1]:
                best[ch.pointer] = (ch, score)

    file_summary = ', '.join(f'{f.label} ({len(f.lines)} lines)' for f in index.files)
    if not best:
        return (
            note + f'Searched {len(index.files)} file(s): {file_summary}\n'
            'file_search.msearch found NO matching candidates for:\n'
            + '\n'.join(per_query)
            + '\n\nTry broader / different keywords (add a short recall query), '
            'or use read_lines / grep.'
        )

    ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
    for ch, _ in ranked:
        state.msearch_pointers.add(ch.pointer)

    parts: list[str] = []
    if note:
        parts.append(note.rstrip('\n'))
    parts.append(f'Searched {len(index.files)} file(s): {file_summary}')
    parts.append(
        f'file_search.msearch: {len(queries)} quer(y/ies), {len(ranked)} unique '
        f'CANDIDATE chunk(s). These are search candidates, not the final answer, '
        f'and snippets may be partial — expand a pointer with file_search_mclick '
        f'or read exact lines with read_lines.'
    )
    parts.append('\n'.join(per_query))
    parts.append('')

    for rank, (ch, score) in enumerate(ranked, 1):
        f = index.by_idx[ch.file_idx]
        snippet, last_shown = _render_snippet(f, ch.l_start, ch.l_end)
        if last_shown < ch.l_end:
            range_label = f'lines [L{ch.l_start}-L{last_shown} shown of L{ch.l_start}-L{ch.l_end}]'
        else:
            range_label = f'lines [L{ch.l_start}-L{ch.l_end}]'
        parts.append(
            f'[{rank}] Source: {f.label}   Mclick Target: "{ch.pointer}"   {range_label}   (relevance {score:.2f})'
        )
        parts.append(snippet)
        parts.append('---')
    parts.append(
        'Next: file_search_mclick(pointers=["fidx:cNN", ...]) to expand a pointer '
        'for fuller surrounding context, or read_lines(path, start, end) for exact '
        'lines.'
    )
    return '\n'.join(parts)


def mclick(state: GeneralState, pointers: list[str]) -> str:
    """Expand previously-returned pointers to their surrounding context."""
    if not pointers:
        return '(no pointers provided — pass pointers returned by file_search_msearch)'
    index = state._bm25
    if index is None:
        return (
            '(no search index yet — run file_search_msearch first; its '
            '"Mclick Target" pointers are what this tool expands)'
        )

    pointers = pointers[:_MCLICK_MAX_POINTERS]
    parts: list[str] = []
    for ptr in pointers:
        ch = index.by_pointer(str(ptr))
        if ch is None:
            parts.append(
                f'=== {ptr}: ERROR — invalid pointer (no such chunk). mclick '
                f'pointers use the format fidx:cNN and come from a prior '
                f'file_search_msearch result. ==='
            )
            continue
        prefix = ''
        if ch.pointer not in state.msearch_pointers:
            prefix = (
                f'Note: {ch.pointer} was not returned by a prior '
                f'file_search_msearch — run msearch first to discover the most '
                f'relevant pointers.\n'
            )
        file_chunks = index.by_file.get(ch.file_idx, [])
        lo = max(0, ch.cid - _MCLICK_NEIGHBORS)
        hi = min(len(file_chunks) - 1, ch.cid + _MCLICK_NEIGHBORS)
        l_start = file_chunks[lo].l_start
        l_end = file_chunks[hi].l_end
        f = index.by_idx[ch.file_idx]
        body_lines = [f'L{i}: {f.lines[i]}' for i in range(l_start, l_end + 1)]
        parts.append(f'{prefix}=== {ch.pointer} expanded with context: [L{l_start}-L{l_end}] in {f.label} ===')
        parts.append('\n'.join(body_lines))
        parts.append('')
    return '\n'.join(parts)
