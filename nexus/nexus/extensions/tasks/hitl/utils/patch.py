"""Pure helpers for inspecting and compacting unified git patches."""

import re

from dataclasses import dataclass, field


DEFAULT_MAX_FILE_CHARS = 15_000
DEFAULT_MAX_TOTAL_CHARS = 40_000

_FILE_HEADER_RE = re.compile(r'^diff --git ', re.MULTILINE)
_FILE_PATH_RE = re.compile(r'^diff --git a/(.+?) b/(.+?)$', re.MULTILINE)


@dataclass(frozen=True)
class PatchFileStat:
    path: str
    added: int = 0
    removed: int = 0

    @property
    def changed(self) -> int:
        return self.added + self.removed


@dataclass(frozen=True)
class PatchFileDiff:
    path: str
    text: str
    stat: PatchFileStat


@dataclass(frozen=True)
class PatchScaleStats:
    """Whole-patch scale, with per-file line counts already bucketed."""

    file_count: int = 0
    added: int = 0
    removed: int = 0
    file_types: tuple[str, ...] = field(default_factory=tuple)
    file_change_buckets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> int:
        return self.added + self.removed

    @property
    def is_empty(self) -> bool:
        return self.file_count == 0 and self.changed == 0


def split_patch_by_file(patch: str) -> list[PatchFileDiff]:
    """Split a unified git diff into per-file chunks."""
    positions = [match.start() for match in _FILE_HEADER_RE.finditer(patch or '')]
    if not positions:
        return []

    chunks: list[PatchFileDiff] = []
    for index, start in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(patch)
        text = patch[start:end]
        path = _extract_file_path(text)
        added, removed = count_patch_lines(text)
        chunks.append(PatchFileDiff(path=path, text=text, stat=PatchFileStat(path, added, removed)))
    return chunks


def extract_patch_file_stats(patch: str) -> list[PatchFileStat]:
    """Return changed files and added/removed line counts from a unified diff."""
    return [chunk.stat for chunk in split_patch_by_file(patch)]


def extract_patch_files(patch: str) -> list[str]:
    """Return changed file paths from a unified diff."""
    return [stat.path for stat in extract_patch_file_stats(patch)]


def extract_patch_scale_stats(patch: str) -> PatchScaleStats:
    """Aggregate a unified diff into whole-patch scale stats."""
    files = extract_patch_file_stats(patch)
    if not files:
        return PatchScaleStats()

    return PatchScaleStats(
        file_count=len(files),
        added=sum(stat.added for stat in files),
        removed=sum(stat.removed for stat in files),
        file_types=tuple(sorted({_file_type(stat.path) for stat in files})),
        file_change_buckets=tuple(_line_count_bucket(stat.changed) for stat in files),
    )


def format_patch_scale_reference(patch: str) -> str:
    """Format a coarse reference-patch scale signal for private calibration.

    The file count is exact, but line counts are bucketed. This signal should
    help an orchestrator notice suspiciously tiny or broad worker changes
    without exposing the solution patch itself.
    """
    stats = extract_patch_file_stats(patch)
    if not stats:
        return ''

    total_changed = sum(stat.changed for stat in stats)
    file_types = sorted({_file_type(stat.path) for stat in stats})
    lines = [
        f'Expected changed files: {len(stats)}',
        f'Expected total changed-line bucket: {_line_count_bucket(total_changed)}',
        f'Expected file types: {", ".join(file_types) or "unknown"}',
        'Expected per-file changed-line buckets:',
    ]
    for stat in stats:
        lines.append(f'- {stat.path}: {_line_count_bucket(stat.changed)}')
    return '\n'.join(lines)


def format_current_patch_scale(current_patch: str) -> str:
    """Format the scale of the patch a worker has produced so far.

    Unlike ``format_patch_scale_reference``, which describes the solution patch and
    so must stay coarse, this describes work the worker already did and can be
    reported back to it verbatim.
    """
    current = extract_patch_scale_stats(current_patch)
    if current.is_empty:
        return '<current_patch_scale>\nCurrent worker patch: no file changes.\n</current_patch_scale>'

    return '\n'.join([
        '<current_patch_scale>',
        (
            f'Current worker patch scale: {_scale_label(current)}; '
            f'{_file_count_text(current.file_count)}; '
            f'per-file changed-line buckets: {_file_bucket_text(current)}; '
            f'file types: {", ".join(current.file_types) or "unknown"}.'
        ),
        '</current_patch_scale>',
    ])


def count_patch_lines(diff_text: str) -> tuple[int, int]:
    """Count added and removed content lines in a diff chunk."""
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    return added, removed


def normalize_patch(
    patch: str,
    *,
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Normalize a git patch for prompt/context injection."""
    if not patch or not patch.strip():
        return '<empty>'

    files = split_patch_by_file(patch)
    if not files:
        return _truncate_unstructured_patch(patch, max_total_chars)

    stat_lines: list[str] = []
    kept_diffs: list[str] = []
    total_added = 0
    total_removed = 0
    skipped_count = 0

    for file_diff in files:
        stat = file_diff.stat
        total_added += stat.added
        total_removed += stat.removed
        stat_lines.append(f' {stat.path} | +{stat.added} -{stat.removed}')

        if len(file_diff.text) > max_file_chars:
            skipped_count += 1
            kept_diffs.append(
                f'diff --git a/{stat.path} b/{stat.path}\n'
                f'[skipped: +{stat.added} -{stat.removed} lines, '
                f'{len(file_diff.text)} chars, too large for context]'
            )
        else:
            kept_diffs.append(file_diff.text.rstrip())

    stat_header = f'{len(files)} file{"s" if len(files) != 1 else ""} changed, +{total_added} -{total_removed}'
    if skipped_count:
        stat_header += f' ({skipped_count} large file{"s" if skipped_count != 1 else ""} summarized)'

    stat_block = stat_header + '\n' + '\n'.join(stat_lines)
    diff_body = '\n\n'.join(kept_diffs)
    result = stat_block + '\n\n' + diff_body

    if len(result) <= max_total_chars:
        return result

    budget = max_total_chars - len(stat_block) - 160
    if budget <= 0:
        return stat_block + '\n\n[diff too large for context; inspect git diff for details]'

    truncated_body = diff_body[:budget]
    last_boundary = truncated_body.rfind('\ndiff --git ')
    if last_boundary > budget // 2:
        truncated_body = truncated_body[:last_boundary]
    return stat_block + '\n\n' + truncated_body + '\n\n[truncated; inspect git diff for details]'


def _extract_file_path(file_diff: str) -> str:
    match = _FILE_PATH_RE.match(file_diff)
    if match:
        return match.group(2)
    for line in file_diff.splitlines():
        if line.startswith('+++ b/'):
            return line[len('+++ b/') :]
    return '(unknown)'


def _truncate_unstructured_patch(patch: str, max_total_chars: int) -> str:
    if len(patch) <= max_total_chars:
        return patch
    return patch[:max_total_chars] + '\n\n[truncated; inspect git diff for details]'


def _scale_label(stats: PatchScaleStats) -> str:
    if stats.changed <= 10 and stats.file_count <= 1:
        return 'tiny'
    if stats.changed <= 40 and stats.file_count <= 2:
        return 'small'
    if stats.changed <= 120 and stats.file_count <= 5:
        return 'medium'
    if stats.changed <= 400 and stats.file_count <= 10:
        return 'large'
    return 'very large'


def _file_count_text(count: int) -> str:
    return f'{count} file' if count == 1 else f'{count} files'


def _file_bucket_text(stats: PatchScaleStats) -> str:
    if not stats.file_change_buckets:
        return 'unknown'
    if len(stats.file_change_buckets) <= 8:
        return ', '.join(stats.file_change_buckets)

    counts: dict[str, int] = {}
    for bucket in stats.file_change_buckets:
        counts[bucket] = counts.get(bucket, 0) + 1
    return ', '.join(f'{count} file(s) at {bucket}' for bucket, count in counts.items())


def _line_count_bucket(count: int) -> str:
    if count <= 0:
        return '0'
    if count <= 10:
        return '1-10'
    if count <= 30:
        return '11-30'
    if count <= 80:
        return '31-80'
    if count <= 200:
        return '81-200'
    if count <= 500:
        return '201-500'
    if count <= 1000:
        return '501-1000'
    return '1000+'


def _file_type(path: str) -> str:
    lowered = path.lower()
    parts = lowered.split('/')
    name = parts[-1] if parts else lowered

    if name in {'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock', 'cargo.lock'}:
        return 'generated/lock'
    if name.endswith(('.md', '.rst', '.txt')) or 'docs' in parts or 'doc' in parts:
        return 'docs'
    if (
        'test' in parts
        or 'tests' in parts
        or '__tests__' in parts
        or name.startswith('test_')
        or name.endswith(('_test.go', '.test.ts', '.test.tsx', '.spec.ts', '.spec.tsx'))
    ):
        return 'tests'
    if name.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf')):
        return 'config'
    if name.endswith((
        '.py',
        '.js',
        '.jsx',
        '.ts',
        '.tsx',
        '.go',
        '.rs',
        '.java',
        '.kt',
        '.rb',
        '.php',
        '.c',
        '.cc',
        '.cpp',
        '.h',
        '.hpp',
        '.cs',
        '.swift',
        '.m',
        '.mm',
    )):
        return 'source'
    return 'other'
