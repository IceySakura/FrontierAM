"""Path-native file access — the foundation of HyGeneralToolSystem.

Every tool in this package addresses files by PATH and reads them from disk,
so there is no "loaded document" concept and nothing to register up front.
This module owns the three things all of them need: path resolution, binary
detection, and a small read cache.

Two deliberate divergences from the content-bearing design this grew out of:

* **No long-line preprocessing.** That design hard-wraps lines at 4000
  chars and pretty-prints JSON lines, which makes tool line numbers drift from
  the file's physical lines. Here ``bash`` is a first-class citizen, so a model
  that locates something with ``sed``/``awk`` must be able to act on it with
  ``read_lines``. Lines are therefore returned exactly as they appear on disk:
  ``L0`` is physical line 1. (``wc -l`` counts newline characters, so it still
  reports one fewer than the line count on a file with no trailing newline.)
* **Binary files are refused, not mangled.** A PDF read as text is noise that
  wastes a round and pollutes the context. :func:`read_lines_cached` raises
  :class:`FileReadError` with an actionable message telling the model to
  extract text first.
"""

from __future__ import annotations

import os

from collections import OrderedDict
from pathlib import Path


# A single file this large is not something to pull into memory line by line;
# the model should stream it with bash/python instead.
MAX_FILE_BYTES = 100 * 1024 * 1024

# Bytes sampled from the head of a file for the binary sniff.
_SNIFF_BYTES = 8192

# Magic prefixes worth naming explicitly, so the error message can tell the
# model WHICH extraction path to take rather than just "it's binary".
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b'%PDF-', 'PDF'),
    (b'PK\x03\x04', 'ZIP container'),
    (b'\xd0\xcf\x11\xe0', 'legacy Office file (doc/xls/ppt)'),
    (b'\x89PNG\r\n\x1a\n', 'PNG image'),
    (b'\xff\xd8\xff', 'JPEG image'),
    (b'GIF87a', 'GIF image'),
    (b'GIF89a', 'GIF image'),
    (b'\x1f\x8b', 'gzip archive'),
)

# Magics too short to be trusted on their own — "BM" and "II*\0" occur often
# enough in ordinary text that matching them before the text check would
# misclassify real documents. These are only consulted once a file has already
# failed the UTF-8 / NUL-byte test.
_WEAK_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b'BM', 'BMP image'),
    (b'II*\x00', 'TIFF image'),
    (b'MM\x00*', 'TIFF image'),
)

# A .docx/.xlsx/.pptx is a zip container too, so magic alone cannot tell an
# archive from an Office document — and the recovery advice differs (unzip vs
# openpyxl). The extension is what disambiguates them.
_OOXML_EXT: dict[str, str] = {
    '.docx': 'Word document (OOXML Office file)',
    '.xlsx': 'Excel workbook (OOXML Office file)',
    '.xlsm': 'Excel workbook (OOXML Office file)',
    '.pptx': 'PowerPoint deck (OOXML Office file)',
}

_PDF_HINT = (
    'Extract the text first with the `python` tool:\n'
    '    from pypdf import PdfReader\n'
    "    r = PdfReader('<path>')\n"
    "    open('<out>.txt','w').write('\\n'.join(p.extract_text() or '' "
    'for p in r.pages))\n'
    '(pypdf / pdfplumber / pdfminer are installed; there is no pdftotext '
    'binary.) Then run read_lines / grep / file_search on the .txt you wrote.'
)

_ARCHIVE_HINT = (
    'Unpack it first, then inspect what came out:\n'
    '    bash: cd <dir> && unzip -qq <file>.zip -d <file> && ls -R <file>\n'
    '(or `python` + zipfile). Each extracted file is its own document and may '
    'itself need converting — a .docx/.xlsx/.pptx is also a zip container, so '
    'read those with python-docx / openpyxl / python-pptx rather than unzipping '
    'them by hand.'
)

_OFFICE_HINT = (
    'Read it with the `python` tool — openpyxl for .xlsx, python-docx for '
    '.docx, python-pptx for .pptx — and write what you need to a .txt under '
    'the workspace before using the line-based tools.'
)

_IMAGE_HINT = (
    'This harness has no vision tool, so image content cannot be read at all. '
    'Do not guess at what the image shows. If the answer exists only in this '
    'image, say so in your final answer; otherwise look for the same '
    'information in a machine-readable file.'
)

_GENERIC_HINT = (
    'Convert it to text with the `python` or `bash` tool, write the result to '
    'a .txt under the workspace, then use the line-based tools on that file.'
)


def _hint_for(kind: str) -> str:
    """Return format-appropriate recovery advice for a binary file.

    Naming the right library matters: telling a model to run pypdf on a zip
    archive is advice it has to notice is wrong before it can act.
    """
    low = kind.lower()
    if 'pdf' in low:
        return _PDF_HINT
    if 'office' in low:
        return _OFFICE_HINT if 'legacy' not in low else _GENERIC_HINT
    if 'archive' in low or 'zip' in low or 'gzip' in low:
        return _ARCHIVE_HINT
    if 'image' in low:
        return _IMAGE_HINT
    return _GENERIC_HINT


class FileReadError(Exception):
    """A file could not be read as text. The message is shown to the model."""


def resolve_path(raw: str, base: str | Path) -> Path:
    """Resolve *raw* to a concrete path.

    Relative paths resolve against *base* (the rollout working directory), so
    ``./workspace/doc.txt`` means what the model expects after a download.
    Absolute paths are used verbatim, which is what lets a task point at a
    read-only corpus file living outside the rollout folder.
    """
    path = Path(str(raw or '').strip()).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path


def sniff_kind(path: Path) -> str | None:
    """Return a human name for the binary format, or ``None`` if it is text.

    Magic-number match wins; otherwise a NUL byte or an undecodable head is
    taken as binary. The tail of the sniff window is allowed to be a truncated
    multi-byte character, which would otherwise flag valid UTF-8 as binary.
    """
    try:
        with path.open('rb') as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError as exc:
        raise FileReadError(f'cannot read {path}: {exc}') from exc

    if not head:
        return None
    for magic, name in _MAGIC:
        if head.startswith(magic):
            if name == 'ZIP container':
                ext = path.suffix.lower()
                return _OOXML_EXT.get(ext, 'ZIP archive')
            return name
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'WEBP image'

    if b'\x00' not in head:
        for cut in range(0, 4):
            chunk = head[: len(head) - cut] if cut else head
            try:
                chunk.decode('utf-8')
                return None
            except UnicodeDecodeError:
                continue
    for magic, name in _WEAK_MAGIC:
        if head.startswith(magic):
            return name
    return 'binary data'


def describe(path: Path) -> str:
    """One-line ``name — size, lines`` description used by error messages."""
    try:
        size = path.stat().st_size
    except OSError:
        return f'{path} (stat failed)'
    return f'{path} ({size} bytes)'


class FileCache:
    """Per-rollout cache of ``path -> lines``, keyed by (mtime_ns, size).

    Purely a performance optimisation and invisible to the model: a stale
    entry can never be served because any write changes mtime or size, and
    ``grep`` over a 10 MB extract would otherwise re-read the file on every
    call. Bounded so a rollout that writes many intermediate files cannot grow
    without limit.
    """

    def __init__(self, max_entries: int = 16) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, int, int], list[str]] = OrderedDict()

    def read_lines(self, path: Path) -> list[str]:
        """Return the file's lines, raising :class:`FileReadError` on refusal."""
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise FileReadError(
                f'no such file: {path}\n'
                'Check the path. Files you downloaded live under ./workspace/; '
                'list them with bash (`ls -la ./workspace/`).'
            ) from exc
        except OSError as exc:
            raise FileReadError(f'cannot stat {path}: {exc}') from exc

        if stat.st_size > MAX_FILE_BYTES:
            raise FileReadError(
                f'{path} is {stat.st_size} bytes, over the '
                f'{MAX_FILE_BYTES}-byte limit for line-based tools. '
                'Process it with the `python` or `bash` tool instead '
                '(stream it, or split out the part you need).'
            )
        if path.is_dir():
            raise FileReadError(f'{path} is a directory, not a file. List its contents with bash (`ls -la`).')

        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached

        kind = sniff_kind(path)
        if kind is not None:
            raise FileReadError(f'{path} is a {kind} and cannot be read as text.\n\n{_hint_for(kind)}')

        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            raise FileReadError(f'cannot read {path}: {exc}') from exc

        lines = text.split('\n')
        # A trailing newline yields a final empty element that is not a real
        # line; dropping it keeps the count equal to `wc -l`.
        if lines and lines[-1] == '':
            lines.pop()

        self._entries[key] = lines
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return lines


def iter_text_files(root: Path) -> list[Path]:
    """Return readable text files under *root*, sorted, for whole-dir search.

    Binary files are skipped rather than reported: this backs the "omit the
    path and search everything you have" behaviour of grep / file_search, and
    a downloaded PDF sitting next to its extracted .txt should not derail it.
    """
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for name in sorted(filenames):
            if name.startswith('.'):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                if sniff_kind(path) is not None:
                    continue
            except (OSError, FileReadError):
                continue
            out.append(path)
    return out
