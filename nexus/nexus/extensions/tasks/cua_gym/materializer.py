"""CUA-Gym task bundle materialization helpers."""

from __future__ import annotations

import os
import re
import shutil

from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(r'__[A-Z0-9_]+__')
TEXT_SUFFIXES = {
    '',
    '.css',
    '.csv',
    '.env',
    '.html',
    '.js',
    '.json',
    '.jsx',
    '.md',
    '.py',
    '.sh',
    '.svg',
    '.ts',
    '.tsx',
    '.txt',
    '.xml',
    '.yaml',
    '.yml',
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def is_probably_text_file(path: Path) -> bool:
    """Return whether a file can be safely decoded and rewritten as UTF-8 text."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    data = path.read_bytes()
    if b'\x00' in data:
        return False
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def detect_placeholders(bundle_dir: str | Path) -> list[str]:
    """Find placeholder tokens such as ``__CUA_GYM_INSTAGRAM_URL__`` in text files."""
    root = Path(bundle_dir)
    placeholders: set[str] = set()
    for path in root.rglob('*'):
        if not path.is_file() or not is_probably_text_file(path):
            continue
        placeholders.update(PLACEHOLDER_PATTERN.findall(path.read_text(encoding='utf-8')))
    return sorted(placeholders)


def load_env_replacements(path: str | Path | None) -> dict[str, str]:
    """Load placeholder replacements from a KEY=VALUE env file."""
    if not path:
        return {}
    env_path = Path(path)
    replacements: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        token = key.strip()
        if not token.startswith('__'):
            token = f'__{token}__'
        replacements[token] = os.path.expandvars(value.strip().strip('"').strip("'"))
    return replacements


def materialize_task_bundle(
    source_dir: str | Path,
    replacements: dict[str, str],
    *,
    output_dir: str | Path,
) -> Path:
    """Copy a CUA-Gym task bundle and replace text placeholders in the copy."""
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f'CUA-Gym bundle is not a directory: {source}')
    if source == output or _is_relative_to(source, output) or _is_relative_to(output, source):
        raise ValueError('CUA-Gym materialized output must be separate from the source bundle')
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)

    if not replacements:
        return output

    for path in output.rglob('*'):
        if not path.is_file() or not is_probably_text_file(path):
            continue
        text = path.read_text(encoding='utf-8')
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        path.write_text(text, encoding='utf-8')
    return output


__all__ = [
    'PLACEHOLDER_PATTERN',
    'detect_placeholders',
    'is_probably_text_file',
    'load_env_replacements',
    'materialize_task_bundle',
]
