"""Shared mutable state for one HyGeneralToolSystem rollout.

Contrast with a content-bearing design, which owns a list of pre-loaded
``Document`` objects and gates every tool on membership in it. There is no such
list here. Files are addressed by path and
read from disk on demand, so this class holds only what genuinely has to
outlive a single tool call: the workspace, the tuning knobs, the read cache,
the BM25 index cache, and the persistent Python REPL handle.

``declared_files`` records the URIs the task handed the agent. It is metadata
for the trace and the opening user turn only — the tools never consult it, and
the agent is free to read files it discovered by other means.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fileio import (
    MAX_FILE_BYTES,
    FileCache,
    FileReadError,
    iter_text_files,
    resolve_path,
    sniff_kind,
)
from .workspace import Workspace, sanitize_filename


logger = logging.getLogger(__name__)

# Nexus repo root. Relative local paths in a task's ``context_files`` resolve
# against this when the caller does not pass an explicit ``files_base``.
# nexus/extensions/agents/context_agent/tools/general/state.py -> parents[6] is the
# repo root.
REPO_ROOT = Path(__file__).resolve().parents[6]


def format_size(num_bytes: int) -> str:
    """Return a compact human-readable byte size, e.g. ``12.3 KB``."""
    size = float(max(0, int(num_bytes)))
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            if unit == 'B':
                return f'{int(size)} B'
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GB'


@dataclass
class Knobs:
    """Tunable knobs forwarded from the CLI."""

    file_search_top_k: int = 5
    file_search_chunk_lines: int = 50
    python_timeout: int = 120
    curl_timeout: int = 120
    curl_max_bytes: int = 200 * 1024 * 1024


@dataclass
class DeclaredFile:
    """One entry of the task's ``context_files``.

    ``uri`` is the string the agent is **shown** in its manifest, and the one it
    will copy back into a tool call. Two forms are valid, and both mean the same
    thing to all seven tools:

    * an absolute path — the form used when the file is read where it lies. A
      dataset-relative path could not work here: ``bash`` and ``python`` are
      real subprocesses whose cwd is the rollout log directory, they never pass
      through :meth:`GeneralState.resolve`, and a repo-relative path handed to
      either resolves under the rollout folder and fails.
    * ``./workspace/<name>`` — the form used once the file has been staged into
      the rollout's own workspace (see :func:`stage_declared_files`). This one
      *is* relative, and works precisely because it is relative to the cwd those
      subprocesses actually run in.

    ``local_path`` is where the bytes really are, which is what the tools use
    after :meth:`GeneralState.resolve` maps the advertised uri through. It is
    ``None`` for URLs and ``task-files://`` references until staging materializes
    them into the Runtime workspace.

    ``runtime_preloaded`` means ``local_path`` already names a file inside the
    configured Runtime. The host must not try to read or upload that path.
    Runtime bridges use this for files prepared before the agent starts while
    still advertising the standard ``./workspace/...`` path to the model.
    """

    name: str
    uri: str
    local_path: Path | None = None
    runtime_preloaded: bool = False

    @property
    def is_url(self) -> bool:
        return self.uri.lower().startswith(('http://', 'https://'))

    @property
    def is_task_file(self) -> bool:
        return self.uri.lower().startswith('task-files://')

    @property
    def task_file_path(self) -> str | None:
        if not self.is_task_file:
            return None
        return self.uri[len('task-files://') :].lstrip('/')


def allocate_staged_name(declared: DeclaredFile, used: set[str]) -> str:
    """Return a safe, unique workspace filename for a declared local file."""
    source = declared.local_path
    fallback = source.name if source is not None else 'file'
    name = sanitize_filename(declared.name or fallback, fallback=fallback) or fallback
    if name in used:
        stem, suffix = Path(name).stem, Path(name).suffix
        n = 2
        while f'{stem}_{n}{suffix}' in used:
            n += 1
        name = f'{stem}_{n}{suffix}'
    used.add(name)
    return name


def stage_declared_files(
    declared_files: list[DeclaredFile], workspace: Workspace
) -> tuple[list[DeclaredFile], list[dict[str, str]]]:
    """Copy the task's local files into the workspace and re-point them there.

    Returns ``(declared_files, staged)`` where the files now advertise
    ``./workspace/<name>`` and ``staged`` records what was copied, for the trace.

    Worth doing even though the agent could read the originals in place:

    * The rollout directory becomes self-contained. A trace stays readable, and
      re-runnable, after the dataset has moved or changed underneath it.
    * The agent holds ``bash``. Reading in place means one stray ``>`` or ``rm``
      lands in the shared dataset, corrupting it for every later run — and the
      corruption would be attributed to whichever run noticed it. Copies make
      the blast radius one attempt directory.
    * Converting in place is what a task usually wants next anyway ("extract
      this PDF, then read the .txt"), and the workspace is where the prompt
      already tells it to write.

    URLs are left alone — there is nothing local to copy, and ``curl`` puts its
    downloads in this same directory. A source that does not exist is also left
    alone, so the error the agent gets still names the path the dataset asked
    for rather than a copy that was never made.
    """
    dest = Path(workspace.docs_dir)
    dest.mkdir(parents=True, exist_ok=True)

    out: list[DeclaredFile] = []
    staged: list[dict[str, str]] = []
    used: set[str] = set()
    for declared in declared_files:
        source = declared.local_path
        if declared.is_url or source is None or not source.is_file():
            out.append(declared)
            continue

        # The source's own basename is known to be a legal filename, since it is
        # sitting on disk; the declared name is dataset-authored and may not be.
        # Runtime staging uses this same allocator, so both backends advertise
        # identical names for the same inputs.
        name = allocate_staged_name(declared, used)

        target = dest / name
        try:
            shutil.copy2(source, target)
        except OSError as exc:  # noqa: BLE001
            logger.warning('could not stage %s into the workspace: %s', source, exc)
            out.append(declared)
            continue

        out.append(
            DeclaredFile(
                name=declared.name or name,
                uri=workspace.path_for(name),
                local_path=target,
            )
        )
        staged.append({'name': name, 'source': str(source)})
    return out, staged


@dataclass
class GeneralState:
    """Mutable per-rollout state shared by all tools."""

    workspace: Workspace
    knobs: Knobs = field(default_factory=Knobs)
    declared_files: list[DeclaredFile] = field(default_factory=list)

    files: FileCache = field(default_factory=FileCache, init=False, repr=False)
    _bm25: Any = field(default=None, init=False, repr=False)
    _bm25_key: Any = field(default=None, init=False, repr=False)
    # path -> the ``fidx`` half of an ``fidx:cNN`` pointer. Assigned once and
    # never reused, so a pointer keeps meaning the same file for the whole
    # rollout. Positional numbering would silently repoint an earlier pointer
    # at a different file as soon as the searchable set changed.
    _file_indices: dict = field(default_factory=dict, init=False, repr=False)
    # fidx -> (mtime_ns, size) at index time, so pointers into a file the agent
    # has since rewritten can be invalidated while the rest stay valid.
    _indexed_content: dict = field(default_factory=dict, init=False, repr=False)
    _python_repl: Any = field(default=None, init=False, repr=False)
    _file_reader: Any = field(default=None, init=False, repr=False)
    # Pointers a prior file_search_msearch returned — used to enforce the
    # "msearch before mclick" discipline (soft).
    msearch_pointers: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        # Exact-string lookup from what the manifest advertised to where the
        # file really is. This is what lets the agent copy a path out of the
        # manifest and have it work, without the tools having to guess among
        # several roots — a guess that would silently pick the wrong file when
        # the same relative path exists under more than one of them.
        self._declared_by_uri: dict[str, Path] = {}
        self._uri_by_path: dict[str, str] = {}
        for declared in self.declared_files:
            if declared.local_path is None:
                continue
            self._declared_by_uri[declared.uri] = declared.local_path
            self._declared_by_uri.setdefault(declared.uri.lstrip('./'), declared.local_path)
            self._uri_by_path.setdefault(str(declared.local_path), declared.uri)

    def resolve(self, raw: str) -> Path:
        """Resolve a model-supplied path to a real file.

        Order: a path the manifest advertised, then relative to the rollout
        working dir (so ``./workspace/<name>`` keeps working), then — only if
        that does not exist — relative to the repo root, which recovers a
        near-miss where the agent retyped a repo-relative path instead of
        copying it.
        """
        text = str(raw or '').strip()
        declared = self._declared_by_uri.get(text) or self._declared_by_uri.get(text.lstrip('./'))
        if declared is not None:
            return declared

        candidate = resolve_path(text, self.workspace.cwd)
        if candidate.exists():
            return candidate
        if not Path(text).expanduser().is_absolute():
            from_repo = resolve_path(text, REPO_ROOT)
            if from_repo.exists():
                return from_repo
        return candidate

    def read_lines(self, path: Path) -> list[str]:
        """Read a file's lines through the cache (raises ``FileReadError``)."""
        return self.files.read_lines(path)

    def file_index_for(self, path: Path) -> int:
        """Return this file's stable ``fidx``, assigning one on first sight."""
        key = str(path)
        idx = self._file_indices.get(key)
        if idx is None:
            idx = len(self._file_indices)
            self._file_indices[key] = idx
        return idx

    def known_files(self) -> list[Path]:
        """Every local text file the agent can search without naming one.

        This is the task's own files PLUS whatever it has fetched or written.
        Both halves are needed: a URL task starts with nothing local and fills
        the workspace, while a local-path task's files sit outside the rollout
        folder and its workspace stays empty forever. Scanning only the
        workspace — the literal translation of a content-bearing design's
        "search all loaded documents" — silently finds nothing on the entire
        local branch.

        Unreadable entries (URLs not yet downloaded, binaries, oversized files)
        are skipped: they are not searchable as text, and reporting them as
        targets would only produce noise.
        """
        out: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            key = str(path)
            if key in seen:
                return
            try:
                if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                    return
                if sniff_kind(path) is not None:
                    return
            except (OSError, FileReadError):
                return
            seen.add(key)
            out.append(path)

        for declared in self.declared_files:
            if declared.local_path is not None:
                add(declared.local_path)
        for path in iter_text_files(self.workspace.docs_dir):
            add(path)
        return out

    def no_targets_hint(self) -> str:
        """Explain why :meth:`known_files` came back empty, in the agent's terms."""
        pending_urls = sum(1 for f in self.declared_files if f.is_url)
        unreadable_local = sum(1 for f in self.declared_files if f.local_path is not None and f.local_path.is_file())
        if pending_urls:
            return (
                f"(nothing to search yet: {pending_urls} of the task's file(s) are "
                'URLs you have not downloaded. Fetch one with curl first, convert '
                'it to text if it is binary, then search.)'
            )
        if unreadable_local:
            return (
                f"(nothing to search yet: the task's {unreadable_local} local "
                'file(s) are not readable as text — convert them with python '
                'first, e.g. a PDF to .txt — then search the converted file.)'
            )
        return (
            '(nothing to search yet: no readable text file is available. Check '
            'the file list in the first message, fetch or convert what you need, '
            'or pass an explicit path.)'
        )

    def display(self, path: Path) -> str:
        """Render *path* the way the model refers to it.

        A declared file is echoed back in the exact form the manifest used, so
        tool output and manifest agree and the agent can copy either. Otherwise
        a path inside the workspace is shown as ``./workspace/<name>``.
        """
        declared = self._uri_by_path.get(str(path))
        if declared is not None:
            return declared
        try:
            return './' + str(path.relative_to(self.workspace.root))
        except ValueError:
            return str(path)

    def close(self) -> None:
        """Tear down any long-lived resources (the Python REPL)."""
        if self._file_reader is not None:
            self._file_reader.close()
            self._file_reader = None
        repl = self._python_repl
        if repl is not None:
            with contextlib.suppress(Exception):
                repl.close()
            self._python_repl = None


def parse_declared_files(raw: Any, base: str | Path | None = None) -> list[DeclaredFile]:
    """Normalise a task's ``context_files`` into :class:`DeclaredFile` items.

    Accepts the canonical ``{"name", "uri"}`` dicts and degrades gracefully for
    bare strings and the ``url`` / ``path`` spellings, so a hand-written or
    older row still runs instead of failing at rollout start.

    **Relative local paths are expanded to absolute here**, against *base*
    (default: the repo root). This has to happen at load time rather than in the
    tools, because the tools resolve relative paths against the ROLLOUT log
    directory — which is correct for ``./workspace/<name>`` but would turn a
    repo-relative data path into ``outputs/.../attempt_0/data/...``. Expanding
    up front also means the manifest shows the agent an absolute path, so the
    path it reads there still works when it passes it straight back to a tool.

    URLs are never touched, and paths that are already absolute are left exactly
    as written.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    root = Path(base) if base is not None else REPO_ROOT

    out: list[DeclaredFile] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            uri = item.strip()
            name = ''
        elif isinstance(item, dict):
            uri = str(item.get('uri') or item.get('url') or item.get('path') or '').strip()
            name = str(item.get('name') or '').strip()
        else:
            continue
        if not uri:
            continue
        if not name:
            name = uri.split('?')[0].rstrip('/').rsplit('/', 1)[-1] or f'file_{idx + 1}'

        local_path: Path | None = None
        runtime_preloaded = False
        runtime_path = str(item.get('runtime_path') or '').strip() if isinstance(item, dict) else ''
        if runtime_path:
            path = Path(runtime_path).expanduser()
            if not path.is_absolute():
                raise ValueError(f'context_agent_general: runtime_path must be absolute: {runtime_path!r}')
            local_path = Path(os.path.normpath(path))
            runtime_preloaded = True
        elif uri.lower().startswith('task-files://'):
            local_path = None
        elif not uri.lower().startswith(('http://', 'https://')):
            path = Path(uri).expanduser()
            if not path.is_absolute():
                path = root / path
            local_path = Path(os.path.normpath(path))
            # Absolute is what the agent is shown, because bash and python get
            # the string verbatim and run from the rollout directory.
            uri = str(local_path)

        out.append(
            DeclaredFile(
                name=name,
                uri=uri,
                local_path=local_path,
                runtime_preloaded=runtime_preloaded,
            )
        )
    return out
