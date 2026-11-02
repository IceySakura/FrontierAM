from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.error

from pathlib import Path

import pytest

from nexus.extensions.agents.dsh import prebuilt


def _manifest(**overrides) -> dict[str, object]:
    data: dict[str, object] = {
        'manifest_version': 1,
        'dsh_version': '0.1.0-rc.6',
        'acp_version': '0.1.0-rc.6',
        'target': 'linux-x64-glibc-node22',
        'archive': 'nexus-dsh-runtime.tar.gz',
        'archive_sha256': 'a' * 64,
        'archive_size': 123,
        'package_lock_sha256': 'b' * 64,
        'minimum_libc_version': '2.34',
    }
    data.update(overrides)
    return data


def test_prebuilt_manifest_url_uses_exact_identity() -> None:
    target = prebuilt.RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)

    assert prebuilt.prebuilt_manifest_url(
        'https://artifacts.test/nexus-dsh/v1/',
        dsh_version='0.1.0-rc.6',
        acp_version='0.1.0-rc.6',
        target=target,
    ) == ('https://artifacts.test/nexus-dsh/v1/dsh-0.1.0-rc.6/acp-0.1.0-rc.6/linux-x64-glibc-node22/manifest.json')


def test_fetch_prebuilt_manifest_returns_none_only_for_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(request, timeout):
        del timeout
        raise urllib.error.HTTPError(request.full_url, 404, 'not found', {}, None)

    monkeypatch.setattr(prebuilt.urllib.request, 'urlopen', not_found)
    target = prebuilt.RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)

    assert (
        prebuilt.fetch_prebuilt_manifest(
            'https://artifacts.test/v1',
            dsh_version='0.1.0-rc.6',
            acp_version='0.1.0-rc.6',
            target=target,
        )
        is None
    )


def test_fetch_prebuilt_manifest_validates_identity_and_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(_manifest()).encode()

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(prebuilt.urllib.request, 'urlopen', lambda _request, timeout: Response(payload))
    compatible = prebuilt.RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)
    manifest = prebuilt.fetch_prebuilt_manifest(
        'https://artifacts.test/v1',
        dsh_version='0.1.0-rc.6',
        acp_version='0.1.0-rc.6',
        target=compatible,
    )

    assert manifest is not None
    assert manifest.archive_url.endswith('/nexus-dsh-runtime.tar.gz')

    incompatible = prebuilt.RuntimeTarget('linux', 'x64', 'glibc', '2.31', 22)
    with pytest.raises(prebuilt.PrebuiltCompatibilityError, match='glibc >= 2.34'):
        prebuilt.fetch_prebuilt_manifest(
            'https://artifacts.test/v1',
            dsh_version='0.1.0-rc.6',
            acp_version='0.1.0-rc.6',
            target=incompatible,
        )


def test_download_archive_resumes_after_interrupted_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b'prebuilt archive bytes'
    split = 9
    requests = []

    class Response:
        def __init__(self, chunks: list[bytes | BaseException], *, status: int, content_range: str = '') -> None:
            self.chunks = iter(chunks)
            self.status = status
            self.headers = {'Content-Range': content_range}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self) -> int:
            return self.status

        def read(self, _size: int) -> bytes:
            value = next(self.chunks, b'')
            if isinstance(value, BaseException):
                raise value
            return value

    responses = iter(
        [
            Response([payload[:split], TimeoutError('stalled')], status=200),
            Response(
                [payload[split:]],
                status=206,
                content_range=f'bytes {split}-{len(payload) - 1}/{len(payload)}',
            ),
        ]
    )

    def urlopen(request, timeout):
        del timeout
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(prebuilt.urllib.request, 'urlopen', urlopen)
    manifest = prebuilt.PrebuiltManifest(
        manifest_url='https://artifacts.test/manifest.json',
        dsh_version='0.1.0-rc.6',
        acp_version='0.1.0-rc.6',
        target='linux-x64-glibc-node22',
        archive='runtime.tar.gz',
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_size=len(payload),
        package_lock_sha256='b' * 64,
        minimum_libc_version='2.34',
    )
    archive_path = tmp_path / 'runtime.tar.gz'

    size, digest = prebuilt._download_archive(manifest, archive_path=archive_path, timeout=60)

    assert size == len(payload)
    assert digest == manifest.archive_sha256
    assert archive_path.read_bytes() == payload
    assert requests[0].get_header('Range') is None
    assert requests[1].get_header('Range') == f'bytes={split}-'


def test_safe_extract_tar_preserves_internal_symlink(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    (source / 'node_modules' / 'pkg').mkdir(parents=True)
    (source / 'node_modules' / '.bin').mkdir()
    (source / 'node_modules' / 'pkg' / 'bin.js').write_text('')
    (source / 'node_modules' / '.bin' / 'pkg').symlink_to('../pkg/bin.js')
    (source / 'package.json').write_text('{}\n')
    (source / 'package-lock.json').write_text('{}\n')
    archive_path = tmp_path / 'runtime.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in ('package.json', 'package-lock.json', 'node_modules'):
            archive.add(source / name, arcname=name, recursive=True)

    destination = tmp_path / 'destination'
    destination.mkdir()
    prebuilt._safe_extract_tar(archive_path, destination)

    link = destination / 'node_modules' / '.bin' / 'pkg'
    assert link.is_symlink()
    assert link.resolve() == destination / 'node_modules' / 'pkg' / 'bin.js'


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        member = tarfile.TarInfo('../escape')
        member.size = 1
        archive.addfile(member, io.BytesIO(b'x'))
    destination = tmp_path / 'destination'
    destination.mkdir()

    with pytest.raises(prebuilt.PrebuiltIntegrityError, match='unsafe path'):
        prebuilt._safe_extract_tar(archive_path, destination)

    assert not (tmp_path / 'escape').exists()
