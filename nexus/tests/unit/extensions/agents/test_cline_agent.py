from __future__ import annotations

import json

from pathlib import Path

import pytest

from nexus.extensions.agents.cline import installer
from nexus.extensions.agents.cline.runtime_setup import _is_silent_noop


def _materialize_package(prefix: Path, manifest: dict | None, files: tuple[str, ...]) -> Path:
    pkg_dir = prefix / 'node_modules' / 'cline'
    pkg_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (pkg_dir / 'package.json').write_text(json.dumps(manifest))
    for rel in files:
        target = pkg_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#!/usr/bin/env node\n')
    return pkg_dir


@pytest.mark.parametrize(
    ('manifest', 'files', 'expected'),
    [
        ({'name': 'cline', 'bin': {'cline': './bin/cline'}}, ('bin/cline',), 'bin/cline'),
        ({'name': 'cline', 'bin': {'cline': './dist/cli.mjs'}}, ('dist/cli.mjs',), 'dist/cli.mjs'),
        # A layout no release has used yet: the manifest must drive resolution so
        # that bumping cline does not require a new hardcoded candidate.
        ({'name': 'cline', 'bin': {'cline': './out/main.js'}}, ('out/main.js',), 'out/main.js'),
        ({'name': 'cline', 'bin': './bin/cline'}, ('bin/cline',), 'bin/cline'),
        ({'name': 'cline'}, ('bin/cline',), 'bin/cline'),
        (None, ('dist/cli.mjs',), 'dist/cli.mjs'),
        ({'name': 'cline', 'bin': {'cline': './gone.js'}}, ('bin/cline',), 'bin/cline'),
        ({'name': 'cline'}, (), None),
    ],
)
def test_resolve_cline_package_entry_follows_manifest(
    tmp_path: Path,
    manifest: dict | None,
    files: tuple[str, ...],
    expected: str | None,
) -> None:
    pkg_dir = _materialize_package(tmp_path, manifest, files)

    resolved = installer._resolve_cline_package_entry(pkg_dir)

    assert (None if resolved is None else str(resolved.relative_to(pkg_dir))) == expected


def test_resolve_cline_executable_prefers_package_entry_over_bin_link(tmp_path: Path) -> None:
    _materialize_package(tmp_path, {'name': 'cline', 'bin': {'cline': './bin/cline'}}, ('bin/cline',))
    bin_dir = tmp_path / 'node_modules' / '.bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'cline').write_text('')

    resolved = installer._resolve_cline_executable(str(tmp_path))

    assert resolved == tmp_path / 'node_modules' / 'cline' / 'bin' / 'cline'


def test_resolve_cline_executable_skips_dangling_bin_link(tmp_path: Path) -> None:
    _materialize_package(tmp_path, {'name': 'cline', 'bin': {'cline': './bin/cline'}}, ('bin/cline',))
    bin_dir = tmp_path / 'node_modules' / '.bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'cline').symlink_to('/nonexistent/stale/cline')

    resolved = installer._resolve_cline_executable(str(tmp_path))

    assert resolved is not None
    assert resolved.exists()


def test_resolve_cline_executable_falls_back_to_bin_link(tmp_path: Path) -> None:
    _materialize_package(tmp_path, {'name': 'cline'}, ())
    bin_dir = tmp_path / 'node_modules' / '.bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'cline').write_text('')

    resolved = installer._resolve_cline_executable(str(tmp_path))

    assert resolved == bin_dir / 'cline'


def test_resolve_cline_executable_returns_none_when_absent(tmp_path: Path) -> None:
    _materialize_package(tmp_path, {'name': 'cline'}, ())

    assert installer._resolve_cline_executable(str(tmp_path)) is None


@pytest.mark.parametrize(
    ('exit_code', 'stdout', 'expected'),
    [
        (0, '', True),
        (0, '   \n\t ', True),
        (0, 'I will start by exploring the repository.', False),
        # A non-zero exit already reports itself; only a claimed success is a no-op.
        (1, '', False),
        (1, 'partial output', False),
        (-1, '', False),
    ],
)
def test_is_silent_noop(exit_code: int, stdout: str, expected: bool) -> None:
    assert _is_silent_noop(exit_code, stdout) is expected
