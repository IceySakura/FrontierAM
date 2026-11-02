from __future__ import annotations

import io
import stat
import tarfile
import zipfile

from pathlib import Path

import pytest
import zstandard

from pydantic import ValidationError

from nexus.core.module_dependencies import ModuleSelector
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.dependency_build.wheelhouse import (
    NexusWheelMetadata,
    WheelhouseBuildSpec,
    WheelhouseMetadata,
    extract_wheelhouse_archive,
    inspect_nexus_wheel,
    resolve_wheelhouse_build_key,
    wheelhouse_build_key,
    write_wheelhouse_archive,
)
from nexus.distributed.repositories.artifact import artifact_object_key
from nexus.distributed.utils.identity import canonical_sha256


def _ref(kind: str, marker: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f'art-{kind}',
        kind=kind,
        sha256=marker * 64,
        size_bytes=100,
    )


def _build_spec(
    *,
    extras: list[str] | None = None,
    module_selectors: list[ModuleSelector] | None = None,
) -> WheelhouseBuildSpec:
    return WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=_ref('source_package', 'a'),
        nexus_wheel=_ref('nexus_wheel', 'b'),
        uv_lock=_ref('uv_lock', 'c'),
        extras=extras or ['runtime', 'distributed'],
        module_selectors=module_selectors or [],
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )


def _nexus_wheel(
    path: Path,
    *,
    metadata_dir: str = 'nexus_agent-1.5.0.dist-info',
    wheel_dir: str | None = None,
    metadata: bytes | None = None,
) -> None:
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(
            f'{metadata_dir}/METADATA',
            metadata or b'Name: nexus-agent\nVersion: 1.5.0\nProvides-Extra: runtime\nProvides-Extra: distributed\n',
        )
        archive.writestr(
            f'{wheel_dir or metadata_dir}/WHEEL',
            b'Wheel-Version: 1.0\nTag: py3-none-any\n',
        )
        archive.writestr('nexus/', b'')
        archive.writestr('nexus/__init__.py', b'')


def _archive_names(path: Path) -> list[str]:
    with path.open('rb') as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw, closefd=False) as decompressed:
            with tarfile.open(fileobj=decompressed, mode='r|') as archive:
                return [member.name for member in archive]


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with path.open('wb') as raw:
        with zstandard.ZstdCompressor(write_checksum=True).stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for info, content in members:
                    archive.addfile(info, io.BytesIO(content))


def test_wheelhouse_build_key_is_canonical_and_binds_all_inputs() -> None:
    first = _build_spec(extras=['runtime', 'distributed', 'runtime'])
    second = _build_spec(extras=['distributed', 'runtime'])

    assert first.extras == ['distributed', 'runtime']
    assert wheelhouse_build_key(first) == wheelhouse_build_key(second)

    changed = second.model_copy(update={'builder_image': f'registry.example/builder@sha256:{"f" * 64}'})
    assert wheelhouse_build_key(first) != wheelhouse_build_key(changed)

    with_modules = _build_spec(
        module_selectors=[ModuleSelector(kind='object', object_type='openai_clients', value='litellm')]
    )
    assert wheelhouse_build_key(first) != wheelhouse_build_key(with_modules)


def test_wheelhouse_build_key_uses_content_identity_instead_of_artifact_id() -> None:
    first = _build_spec()
    repeated_upload = first.model_copy(
        update={
            'source_package': first.source_package.model_copy(update={'artifact_id': 'art-source-reuploaded'}),
            'nexus_wheel': first.nexus_wheel.model_copy(update={'artifact_id': 'art-wheel-reuploaded'}),
            'uv_lock': first.uv_lock.model_copy(update={'artifact_id': 'art-lock-reuploaded'}),
        }
    )
    changed_content = first.model_copy(
        update={
            'source_package': first.source_package.model_copy(update={'sha256': 'f' * 64}),
        }
    )
    changed_limits = WheelhouseBuildSpec.model_validate(
        {
            **first.model_dump(mode='json'),
            'source_package_limits': {
                'max_files': 99_999,
                'max_single_file_bytes': 256 * 1024 * 1024,
                'max_total_bytes': 2 * 1024 * 1024 * 1024,
                'max_archive_bytes': 1024 * 1024 * 1024,
            },
        }
    )

    assert wheelhouse_build_key(first) == wheelhouse_build_key(repeated_upload)
    assert wheelhouse_build_key(first) != wheelhouse_build_key(changed_content)
    assert wheelhouse_build_key(first) != wheelhouse_build_key(changed_limits)


@pytest.mark.parametrize('version', [3, 4])
def test_wheelhouse_build_key_uses_joint_resolution_policy_version(version) -> None:
    spec = _build_spec()
    expected = canonical_sha256(
        {
            'version': version,
            'owner_principal_id': spec.owner_principal_id,
            'environment_id': spec.environment_id,
            'source_package': {
                'kind': spec.source_package.kind,
                'sha256': spec.source_package.sha256,
                'size_bytes': spec.source_package.size_bytes,
            },
            'nexus_wheel': {
                'kind': spec.nexus_wheel.kind,
                'sha256': spec.nexus_wheel.sha256,
                'size_bytes': spec.nexus_wheel.size_bytes,
            },
            'uv_lock': {
                'kind': spec.uv_lock.kind,
                'sha256': spec.uv_lock.sha256,
                'size_bytes': spec.uv_lock.size_bytes,
            },
            'source_package_limits': spec.source_package_limits.model_dump(mode='json'),
            'extras': spec.extras,
            'module_selectors': [],
            'workload_image': spec.workload_image,
            'builder_image': spec.builder_image,
            'python_abi': 'cp313',
            'platform': 'linux_x86_64',
            'os_family': 'tlinux4',
        }
    )

    assert wheelhouse_build_key(spec, version=version) == expected


@pytest.mark.parametrize('version', [3, 4])
def test_recorded_build_key_accepts_same_inputs_and_rejects_different_inputs(version) -> None:
    spec = _build_spec()
    key = wheelhouse_build_key(spec, version=version)
    assert resolve_wheelhouse_build_key(spec, key) == key
    changed = spec.model_copy(update={'builder_image': f'registry.example/builder@sha256:{"f" * 64}'})
    with pytest.raises(ValueError, match='does not match the build inputs'):
        resolve_wheelhouse_build_key(changed, key)


@pytest.mark.parametrize('key', ['', 'not-a-key', '0' * 64])
def test_recorded_build_key_rejects_invalid_or_unrelated_keys(key) -> None:
    with pytest.raises(ValueError, match='does not match the build inputs'):
        resolve_wheelhouse_build_key(_build_spec(), key)


def test_recorded_build_key_defaults_to_current_policy() -> None:
    assert resolve_wheelhouse_build_key(_build_spec()) == wheelhouse_build_key(_build_spec())


def test_wheelhouse_build_spec_accepts_hyphenated_extra() -> None:
    assert _build_spec(extras=['distributed-db-shell']).extras == ['distributed-db-shell']


def test_wheelhouse_build_spec_rejects_invalid_extra() -> None:
    with pytest.raises(ValidationError, match='invalid wheelhouse extra'):
        _build_spec(extras=['not an extra'])


def test_wheelhouse_build_spec_requires_exact_artifact_kinds() -> None:
    with pytest.raises(ValidationError, match='nexus_wheel'):
        _build_spec().model_copy(update={'nexus_wheel': _ref('source_package', 'f')}).model_validate(
            _build_spec().model_copy(update={'nexus_wheel': _ref('source_package', 'f')}).model_dump()
        )


def test_wheelhouse_metadata_v2_records_module_lock_provenance() -> None:
    metadata = WheelhouseMetadata(
        schema_version=2,
        build_key=wheelhouse_build_key(_build_spec()),
        nexus_wheel_path='wheels/nexus_agent-1.5.0-py3-none-any.whl',
        module_names=['litellm'],
        module_paths=['nexus.extensions.openai_clients.litellm'],
        module_host_requirements=['boto3', 'litellm>=1.91.2,<1.92.0'],
        module_lock_path='locks/module-host.lock',
        module_lock_sha256='f' * 64,
    )

    assert metadata.model_dump(mode='json') == {
        'schema_version': 2,
        'build_key': wheelhouse_build_key(_build_spec()),
        'nexus_wheel_path': 'wheels/nexus_agent-1.5.0-py3-none-any.whl',
        'module_names': ['litellm'],
        'module_paths': ['nexus.extensions.openai_clients.litellm'],
        'module_host_requirements': ['boto3', 'litellm>=1.91.2,<1.92.0'],
        'module_lock_path': 'locks/module-host.lock',
        'module_lock_sha256': 'f' * 64,
    }
    with pytest.raises(ValidationError, match='nexus_wheel_path'):
        WheelhouseMetadata(build_key='a' * 64, nexus_wheel_path='../nexus.whl')


def test_wheelhouse_archive_uses_schema_v2_for_selected_module_without_host_requirements(tmp_path: Path) -> None:
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    nexus_name = 'nexus_agent-1.5.0-py3-none-any.whl'
    (wheels / nexus_name).write_bytes(b'nexus-wheel')

    metadata = write_wheelhouse_archive(
        wheels,
        _build_spec(),
        tmp_path / 'wheelhouse.tar.zst',
        nexus_wheel_path=f'wheels/{nexus_name}',
        module_names=['identity'],
        module_paths=['nexus.extensions.tasks.identity'],
    )

    assert metadata.schema_version == 2
    assert metadata.module_names == ['identity']
    assert metadata.module_paths == ['nexus.extensions.tasks.identity']
    assert metadata.module_host_requirements == []
    assert metadata.module_lock_path is None


@pytest.mark.parametrize(
    ('kind', 'filename'),
    [
        ('source_package', 'source.tar.zst'),
        ('nexus_wheel', 'nexus-agent.whl'),
        ('uv_lock', 'uv.lock'),
        ('wheelhouse', 'wheelhouse.tar.zst'),
    ],
)
def test_dependency_artifact_kinds_have_unambiguous_object_names(kind: str, filename: str) -> None:
    key = artifact_object_key(
        owner_principal_id='user-1',
        artifact_id='artifact-1',
        sha256='a' * 64,
        kind=kind,
    )

    assert key.endswith(f'/artifact-1/{"a" * 64}/{filename}')


def test_nexus_wheel_is_inspected_once_into_reusable_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / 'nexus.whl'
    _nexus_wheel(wheel)

    assert inspect_nexus_wheel(wheel) == NexusWheelMetadata(
        version='1.5.0',
        tags=('py3-none-any',),
        extras=frozenset({'distributed', 'runtime'}),
        filename='nexus_agent-1.5.0-py3-none-any.whl',
    )


def test_nexus_wheel_rejects_mismatched_dist_info_directories(tmp_path: Path) -> None:
    wheel = tmp_path / 'nexus.whl'
    _nexus_wheel(wheel, wheel_dir='other-1.0.dist-info')

    with pytest.raises(ValueError, match='dist-info'):
        inspect_nexus_wheel(wheel)


def test_nexus_wheel_rejects_dist_info_directory_that_disagrees_with_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / 'nexus.whl'
    _nexus_wheel(wheel, metadata_dir='other-1.5.0.dist-info')

    with pytest.raises(ValueError, match='dist-info directory does not match'):
        inspect_nexus_wheel(wheel)


def test_nexus_wheel_rejects_oversized_metadata_without_unbounded_read(tmp_path: Path) -> None:
    wheel = tmp_path / 'nexus.whl'
    _nexus_wheel(wheel, metadata=b'x' * (4 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match='METADATA exceeds limit'):
        inspect_nexus_wheel(wheel)


def test_nexus_wheel_rejects_symlink_disguised_as_directory(tmp_path: Path) -> None:
    wheel = tmp_path / 'nexus.whl'
    _nexus_wheel(wheel)
    symlink = zipfile.ZipInfo('linked/')
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr(symlink, 'target')

    with pytest.raises(ValueError, match='non-directory member'):
        inspect_nexus_wheel(wheel)


def test_wheelhouse_archive_is_deterministic_and_contains_wheels_and_module_lock(tmp_path: Path) -> None:
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (wheels / 'alpha-1.0-py3-none-any.whl').write_bytes(b'alpha-wheel')
    nexus_name = 'nexus_agent-1.5.0-py3-none-any.whl'
    (wheels / nexus_name).write_bytes(b'nexus-wheel')
    first = tmp_path / 'first.tar.zst'
    second = tmp_path / 'second.tar.zst'
    nexus_path = f'wheels/{nexus_name}'
    module_lock = tmp_path / 'module-host.lock'
    module_lock.write_text('boto3==1.42.0 --hash=sha256:abc\n')

    metadata = write_wheelhouse_archive(
        wheels,
        _build_spec(),
        first,
        nexus_wheel_path=nexus_path,
        module_names=['litellm'],
        module_paths=['nexus.extensions.openai_clients.litellm'],
        module_host_requirements=['boto3'],
        module_lock=module_lock,
    )
    repeated = write_wheelhouse_archive(
        wheels,
        _build_spec(),
        second,
        nexus_wheel_path=nexus_path,
        module_names=['litellm'],
        module_paths=['nexus.extensions.openai_clients.litellm'],
        module_host_requirements=['boto3'],
        module_lock=module_lock,
    )

    assert (
        metadata
        == repeated
        == WheelhouseMetadata(
            schema_version=2,
            build_key=wheelhouse_build_key(_build_spec()),
            nexus_wheel_path=nexus_path,
            module_names=['litellm'],
            module_paths=['nexus.extensions.openai_clients.litellm'],
            module_host_requirements=['boto3'],
            module_lock_path='locks/module-host.lock',
            module_lock_sha256='6606aa1d42be47ca7c748525f7cb2aa121c90dde4287aa6dc7e0eb4967641c12',
        )
    )
    assert first.read_bytes() == second.read_bytes()
    assert _archive_names(first) == [
        'wheels/alpha-1.0-py3-none-any.whl',
        nexus_path,
        'locks/module-host.lock',
    ]
    extracted = tmp_path / 'extracted'
    extract_wheelhouse_archive(first, extracted, max_files=10, max_total_bytes=1024)
    assert (extracted / 'wheels' / 'alpha-1.0-py3-none-any.whl').read_bytes() == b'alpha-wheel'
    assert (extracted / 'locks' / 'module-host.lock').read_text() == 'boto3==1.42.0 --hash=sha256:abc\n'


def test_wheelhouse_extractor_rejects_non_wheel_members(tmp_path: Path) -> None:
    archive = tmp_path / 'wheelhouse.tar.zst'
    info = tarfile.TarInfo('manifest.json')
    info.size = 2
    _write_archive(archive, [(info, b'{}')])

    with pytest.raises(ValueError, match='wheelhouse member path is invalid'):
        extract_wheelhouse_archive(archive, tmp_path / 'target', max_files=10, max_total_bytes=1024)


def test_wheelhouse_extractor_rejects_symlink_target(tmp_path: Path) -> None:
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    nexus_name = 'nexus_agent-1.5.0-py3-none-any.whl'
    (wheels / nexus_name).write_bytes(b'nexus-wheel')
    archive = tmp_path / 'wheelhouse.tar.zst'
    write_wheelhouse_archive(
        wheels,
        _build_spec(),
        archive,
        nexus_wheel_path=f'wheels/{nexus_name}',
    )
    real_target = tmp_path / 'real-target'
    real_target.mkdir()
    target = tmp_path / 'target'
    target.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(ValueError, match='target must be a directory'):
        extract_wheelhouse_archive(archive, target, max_files=10, max_total_bytes=1024)

    assert not any(real_target.iterdir())
