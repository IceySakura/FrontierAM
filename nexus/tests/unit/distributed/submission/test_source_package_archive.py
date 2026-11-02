from __future__ import annotations

import io
import os
import stat
import tarfile

from pathlib import Path, PurePosixPath

import pytest
import zstandard

from nexus.distributed.submission.source_package import (
    PackageLimits,
    _validate_path_type,
    build_source_package,
    safe_extract_source_package,
)


def _members(path: Path) -> list[tarfile.TarInfo]:
    with path.open('rb') as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw, closefd=False) as decompressed:
            with tarfile.open(fileobj=decompressed, mode='r|') as archive:
                return list(archive)


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    with path.open('wb') as raw:
        with zstandard.ZstdCompressor(write_checksum=True).stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for info, content in members:
                    archive.addfile(info, io.BytesIO(content) if content is not None else None)


def test_source_package_is_deterministic_tar_zst_without_manifest(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'empty').mkdir()
    (source / 'module.py').write_text('VALUE = 1\n')
    script = source / 'run.sh'
    script.write_text('#!/bin/sh\necho ok\n')
    script.chmod(0o700)

    first = build_source_package(source, tmp_path / 'first.tar.zst')
    second = build_source_package(source, tmp_path / 'second.tar.zst')

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes
    assert first.path.read_bytes() == second.path.read_bytes()
    assert zstandard.get_frame_parameters(first.path.read_bytes()).has_checksum is True
    members = _members(first.path)
    assert [member.name for member in members] == ['empty', 'module.py', 'run.sh']
    assert [member.mode for member in members] == [0o755, 0o644, 0o755]

    target = tmp_path / 'target'
    safe_extract_source_package(first.path, target)

    assert (target / 'module.py').read_text() == 'VALUE = 1\n'
    assert stat.S_IMODE((target / 'module.py').stat().st_mode) == 0o644
    assert stat.S_IMODE((target / 'run.sh').stat().st_mode) == 0o755
    assert (target / 'empty').is_dir()


def test_source_package_preserves_internal_relative_symlinks(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    target_dir = source / 'data'
    target_dir.mkdir(parents=True)
    (target_dir / 'value.txt').write_text('linked\n')
    (source / 'value-link').symlink_to('data/value.txt')
    (source / 'dir-link').symlink_to('data', target_is_directory=True)

    package = build_source_package(source, tmp_path / 'source.tar.zst')
    target = tmp_path / 'target'
    safe_extract_source_package(package.path, target)

    assert os.readlink(target / 'value-link') == 'data/value.txt'
    assert os.readlink(target / 'dir-link') == 'data'
    assert (target / 'value-link').read_text() == 'linked\n'
    assert (target / 'dir-link' / 'value.txt').read_text() == 'linked\n'


@pytest.mark.parametrize('target', ['../outside.txt', '/tmp/outside.txt', 'missing.txt'])
def test_source_package_rejects_unsafe_or_dangling_symlink_targets(tmp_path: Path, target: str) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (tmp_path / 'outside.txt').write_text('outside\n')
    (source / 'link').symlink_to(target)

    with pytest.raises(ValueError, match='source symlink'):
        build_source_package(source, tmp_path / 'source.tar.zst')


def test_source_package_rejects_symlink_to_excluded_target(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.gitignore').write_text('secret.txt\n')
    (source / 'secret.txt').write_text('secret\n')
    (source / 'link').symlink_to('secret.txt')

    with pytest.raises(ValueError, match='excluded target'):
        build_source_package(source, tmp_path / 'source.tar.zst')


def test_source_package_uses_gitignore_directory_and_negation_rules(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    (source / 'secrets').mkdir(parents=True)
    (source / 'generated').mkdir()
    (source / '.gitignore').write_text('secrets/\ngenerated/*\n!generated/keep.txt\n')
    (source / 'secrets' / 'key.json').write_text('secret\n')
    (source / 'generated' / 'drop.txt').write_text('drop\n')
    (source / 'generated' / 'keep.txt').write_text('keep\n')

    package = build_source_package(source, tmp_path / 'source.tar.zst')

    names = [member.name for member in _members(package.path)]
    assert 'secrets' not in names
    assert 'secrets/key.json' not in names
    assert 'generated/drop.txt' not in names
    assert 'generated/keep.txt' in names


def test_gitignore_cannot_reinclude_mandatory_secret_excludes(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.gitignore').write_text('!.env\n!private.key\n')
    (source / '.env').write_text('TOKEN=secret\n')
    (source / 'private.key').write_text('secret\n')

    package = build_source_package(source, tmp_path / 'source.tar.zst')

    names = [member.name for member in _members(package.path)]
    assert '.env' not in names
    assert 'private.key' not in names


def test_explicit_credential_exclusion_covers_resolved_path(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    credential = source / 'credential.json'
    credential.write_text('{"key":"secret"}\n')
    outside_link = tmp_path / 'credential-link.json'
    outside_link.symlink_to(credential)

    package = build_source_package(
        source,
        tmp_path / 'source.tar.zst',
        excluded_paths=[outside_link],
    )

    assert 'credential.json' not in [member.name for member in _members(package.path)]


def test_source_package_rejects_hardlinked_source_files(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    original = source / 'original.txt'
    original.write_text('content\n')
    os.link(original, source / 'alias.txt')

    with pytest.raises(ValueError, match='hardlinked source entry'):
        build_source_package(source, tmp_path / 'source.tar.zst')


def test_source_package_failed_extract_keeps_existing_workspace(tmp_path: Path) -> None:
    malicious = tmp_path / 'malicious.tar.zst'
    safe = tarfile.TarInfo('safe.txt')
    safe.size = 4
    escape = tarfile.TarInfo('../escape.txt')
    escape.size = 6
    _write_archive(malicious, [(safe, b'safe'), (escape, b'escape')])
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'existing.txt').write_text('keep\n')

    with pytest.raises(ValueError, match='invalid source package path'):
        safe_extract_source_package(malicious, target)

    assert (target / 'existing.txt').read_text() == 'keep\n'
    assert not (target / 'safe.txt').exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_source_package_rejects_non_regular_tar_members(tmp_path: Path) -> None:
    archive_path = tmp_path / 'hardlink.tar.zst'
    hardlink = tarfile.TarInfo('linked')
    hardlink.type = tarfile.LNKTYPE
    hardlink.linkname = 'target'
    _write_archive(archive_path, [(hardlink, None)])

    with pytest.raises(ValueError, match='unsupported source package entry'):
        safe_extract_source_package(archive_path, tmp_path / 'target')


def test_source_path_type_validation_uses_descendant_index_without_scanning_seen_paths() -> None:
    class NonIterableSeen(dict[PurePosixPath, str]):
        def __iter__(self):
            raise AssertionError('path type validation must not scan all previously seen paths')

    seen = NonIterableSeen({PurePosixPath(f'peer-{index}.txt'): 'file' for index in range(100)})

    _validate_path_type(
        PurePosixPath('next.txt'),
        kind='file',
        seen=seen,
        paths_with_descendants=set(),
    )


def test_source_package_rejects_file_that_replaces_an_existing_path_prefix(tmp_path: Path) -> None:
    archive_path = tmp_path / 'path-conflict.tar.zst'
    child = tarfile.TarInfo('package/module.py')
    child.size = 4
    parent_file = tarfile.TarInfo('package')
    parent_file.size = 4
    _write_archive(archive_path, [(child, b'code'), (parent_file, b'file')])

    with pytest.raises(ValueError, match='source package contains a path type conflict'):
        safe_extract_source_package(archive_path, tmp_path / 'target')


def test_source_package_rejects_archive_symlink_that_escapes_target(tmp_path: Path) -> None:
    archive_path = tmp_path / 'symlink.tar.zst'
    symlink = tarfile.TarInfo('link')
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = '../outside'
    _write_archive(archive_path, [(symlink, None)])

    with pytest.raises(ValueError, match='source symlink target escapes source root'):
        safe_extract_source_package(archive_path, tmp_path / 'target')

    assert not (tmp_path / 'target').exists()


def test_source_package_enforces_archive_size_limit(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'module.py').write_text('VALUE = 1\n')
    package = build_source_package(source, tmp_path / 'source.tar.zst')

    with pytest.raises(ValueError, match='archive size limit'):
        safe_extract_source_package(
            package.path,
            tmp_path / 'target',
            limits=PackageLimits(max_archive_bytes=package.size_bytes - 1),
        )
