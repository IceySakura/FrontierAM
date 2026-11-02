from __future__ import annotations

import tarfile

from pathlib import Path

import pytest
import zstandard

from nexus.distributed.submission.source_package import build_source_package


pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFINE_PATH = 'nexus/extensions/tasks/aidata/engine/confine'


def test_repository_root_builds_source_package_with_internal_symlink(tmp_path: Path) -> None:
    package = build_source_package(PROJECT_ROOT, tmp_path / 'source.tar.zst')

    assert package.file_count > 0
    assert package.size_bytes > 0
    with package.path.open('rb') as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw, closefd=False) as decompressed:
            with tarfile.open(fileobj=decompressed, mode='r|') as archive:
                confine = next(member for member in archive if member.name == CONFINE_PATH)
    assert confine.issym()
    assert confine.linkname == '../mock_servers/confine'
