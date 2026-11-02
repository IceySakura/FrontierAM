from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import textwrap
import zipfile

from pathlib import Path

import pytest

from nexus.core.module_dependencies import ModuleSelector
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.dependency_build import worker
from nexus.distributed.dependency_build.validation import read_target_lock
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseBuildSpec,
    extract_wheelhouse_archive,
    wheelhouse_build_key,
)
from nexus.distributed.dependency_build.worker import WheelhouseBuildError, build_wheelhouse
from nexus.distributed.submission.source_package import build_source_package


_SHARED = 'nexus-resolver-shared'
_MARKER = 'nexus-resolver-marker'
_CONSTRAINED = 'nexus-resolver-constrained'
_PATCH = 'nexus-resolver-patch'
_SDIST = 'nexus-resolver-sdist'
_PARENT = 'nexus-resolver-parent'
_CHILD = 'nexus-resolver-child'
_SOURCE_CONTROLLED = 'nexus-resolver-source-controlled'
_BASE_EQUAL = 'nexus-resolver-base-equal'
_BASE_NOT_EQUAL = 'nexus-resolver-base-not-equal'
_BASE_COMPOUND = 'nexus-resolver-base-compound'


def _wheel(
    directory: Path,
    name: str,
    version: str,
    *,
    requirements: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    module_requirement: str | None = None,
) -> Path:
    normalized = name.replace('-', '_').replace('.', '_')
    dist_info = f'{normalized}-{version}.dist-info'
    package = 'nexus' if name == 'nexus-agent' else normalized
    members = {
        f'{package}/__init__.py': f'__version__ = {version!r}\n'.encode(),
        f'{dist_info}/METADATA': (
            'Metadata-Version: 2.3\n'
            f'Name: {name}\n'
            f'Version: {version}\n'
            'Requires-Python: >=3.13\n'
            + ''.join(f'Provides-Extra: {extra}\n' for extra in extras)
            + ''.join(f'Requires-Dist: {requirement}\n' for requirement in requirements)
        ).encode(),
        f'{dist_info}/WHEEL': (
            b'Wheel-Version: 1.0\nGenerator: nexus-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
        ),
    }
    if module_requirement is not None:
        members['nexus/extensions/openai_clients/resolution_probe.py'] = (
            '# /// script\n'
            '# dependencies = []\n'
            '#\n'
            '# [project]\n'
            '# name = "resolution-probe"\n'
            '#\n'
            '# [optional-dependencies]\n'
            f'# host = [{module_requirement!r}]\n'
            '#\n'
            '# [tool.nexus]\n'
            '# openai_clients = { resolution_probe = '
            '"nexus.extensions.openai_clients.resolution_probe" }\n'
            '# ///\n'
        ).encode()
    record_rows = [[path, '', str(len(content))] for path, content in members.items()]
    record_rows.append([f'{dist_info}/RECORD', '', ''])
    record = io.StringIO()
    csv.writer(record, lineterminator='\n').writerows(record_rows)
    members[f'{dist_info}/RECORD'] = record.getvalue().encode()
    path = directory / f'{normalized}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return path


def _sdist(directory: Path, name: str, version: str) -> Path:
    normalized = name.replace('-', '_').replace('.', '_')
    root = f'{normalized}-{version}'
    metadata = f'Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nRequires-Python: >=3.13\n'
    backend = textwrap.dedent(
        f"""\
        from pathlib import Path
        import zipfile

        NAME = {name!r}
        VERSION = {version!r}
        NORMALIZED = {normalized!r}
        DIST_INFO = f'{{NORMALIZED}}-{{VERSION}}.dist-info'
        METADATA = {metadata!r}
        WHEEL = 'Wheel-Version: 1.0\\nGenerator: nexus-test\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n'

        def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
            target = Path(metadata_directory) / DIST_INFO
            target.mkdir()
            (target / 'METADATA').write_text(METADATA)
            return DIST_INFO

        def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
            filename = f'{{NORMALIZED}}-{{VERSION}}-py3-none-any.whl'
            with zipfile.ZipFile(Path(wheel_directory) / filename, 'w') as archive:
                archive.writestr(f'{{NORMALIZED}}/__init__.py', f'__version__ = {{VERSION!r}}\\n')
                archive.writestr(f'{{DIST_INFO}}/METADATA', METADATA)
                archive.writestr(f'{{DIST_INFO}}/WHEEL', WHEEL)
                archive.writestr(f'{{DIST_INFO}}/RECORD', '')
            return filename
        """
    ).encode()
    members = {
        'pyproject.toml': (
            '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
            f'[project]\nname = "{name}"\nversion = "{version}"\nrequires-python = ">=3.13"\n'
        ).encode(),
        'backend.py': backend,
        'PKG-INFO': metadata.encode(),
    }
    path = directory / f'{normalized}-{version}.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for relative, content in members.items():
            info = tarfile.TarInfo(f'{root}/{relative}')
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    return path


def _project(
    path: Path,
    *,
    shared_requirement: str,
    base_marker_requirements: tuple[str, ...] = (),
) -> None:
    path.mkdir()
    (path / 'nexus').mkdir()
    (path / 'nexus' / '__init__.py').write_text('')
    override_directory = path / 'source-overrides'
    override_directory.mkdir()
    override_wheel = _wheel(override_directory, _SOURCE_CONTROLLED, '9.9.9')
    dependencies = (
        shared_requirement,
        f'{_CONSTRAINED}>=1,<3',
        f'{_SDIST}==1.0.0',
        f'{_SOURCE_CONTROLLED}>=1',
        *base_marker_requirements,
    )
    (path / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "nexus-agent"\n'
        'version = "1.5.0"\n'
        'requires-python = ">=3.13"\n'
        f'dependencies = [{", ".join(repr(requirement) for requirement in dependencies)}]\n'
        '[project.optional-dependencies]\n'
        f'runtime = ["{_MARKER}==1.0.0; python_version >= \'3.13\'", '
        f'"{_PATCH}==1.0.0; python_full_version >= \'3.13.1\'"]\n'
        'distributed = []\n'
        'other = []\n'
        '[tool.uv]\n'
        f'constraint-dependencies = ["{_CONSTRAINED}<2"]\n'
        '[tool.uv.sources]\n'
        f'{_SOURCE_CONTROLLED} = {{ path = "source-overrides/{override_wheel.name}" }}\n'
    )


def _nexus_requirements(
    shared_requirement: str,
    *,
    base_marker_requirements: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        shared_requirement,
        f'{_CONSTRAINED}>=1,<3',
        f'{_SDIST}==1.0.0',
        f'{_SOURCE_CONTROLLED}>=1',
        *base_marker_requirements,
        f'{_MARKER}==1.0.0; python_version >= "3.13" and extra == "runtime"',
        f'{_PATCH}==1.0.0; python_full_version >= "3.13.1" and extra == "runtime"',
    )


def _ref(kind: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f'art-{kind}',
        kind=kind,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _spec(source: Path, nexus_wheel: Path, lock: Path, *, extras: list[str] | None = None) -> WheelhouseBuildSpec:
    return WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=_ref('source_package', source),
        nexus_wheel=_ref('nexus_wheel', nexus_wheel),
        uv_lock=_ref('uv_lock', lock),
        extras=['runtime', 'distributed'] if extras is None else extras,
        module_selectors=[
            ModuleSelector(kind='object', object_type='openai_clients', value='resolution_probe'),
        ],
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )


def _lock(project: Path, wheels: Path) -> Path:
    subprocess.run(
        [
            'uv',
            'lock',
            '--no-index',
            '--find-links',
            str(wheels),
            '--python',
            sys.executable,
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return project / 'uv.lock'


def _installed_versions(python: Path) -> dict[str, str]:
    script = (
        'import importlib.metadata, json; '
        'print(json.dumps({d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}, '
        'sort_keys=True))'
    )
    completed = subprocess.run(
        [str(python), '-c', script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _marker_project(path: Path, requirement: str) -> None:
    path.mkdir()
    (path / 'nexus').mkdir()
    (path / 'nexus' / '__init__.py').write_text('')
    (path / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "nexus-agent"\n'
        'version = "1.5.0"\n'
        'requires-python = ">=3.13"\n'
        f'dependencies = [{requirement!r}]\n'
        '[project.optional-dependencies]\n'
        'runtime = []\n'
        'other = []\n'
    )


@pytest.mark.integration
@pytest.mark.parametrize('locked_version', ['1.0.0', '2.0.0'])
def test_offline_verification_ignores_source_and_pythonpath_but_checks_installed_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locked_version: str
) -> None:
    workspace = tmp_path / 'workspace'
    wheels = workspace / 'wheels'
    source = workspace / 'source'
    ambient = tmp_path / 'ambient'
    wheels.mkdir(parents=True)
    for directory in (source, ambient):
        directory.mkdir()
        metadata = directory / 'uninstalled_probe-1.0.0.dist-info'
        metadata.mkdir()
        (metadata / 'METADATA').write_text('Name: uninstalled-probe\nVersion: 1.0.0\n')
        (directory / 'nexus').mkdir()
        (directory / 'nexus' / '__init__.py').write_text('raise RuntimeError("imported uninstalled source")\n')
    monkeypatch.setenv('PYTHONPATH', str(ambient))
    nexus_wheel = _wheel(wheels, 'nexus-agent', '1.5.0', requirements=(f'{_SHARED}==1.0.0',))
    _wheel(wheels, _SHARED, '1.0.0')

    def verify() -> None:
        worker._verify_offline_install(
            subprocess.run,
            workspace=workspace,
            source_dir=source,
            wheels_dir=wheels,
            nexus_wheel=nexus_wheel,
            extras=[],
            module_host_requirements=[],
            locked_versions={_SHARED: locked_version},
            expected_version='1.5.0',
        )

    if locked_version == '1.0.0':
        verify()
    else:
        with pytest.raises(WheelhouseBuildError) as exc_info:
            verify()
        assert exc_info.value.code == 'dependency_offline_validation_failed'
        assert 'installed distributions do not match target lock' in str(exc_info.value)


@pytest.mark.integration
def test_offline_verification_resolves_selected_extra_and_rejects_missing_transitive_wheel(tmp_path: Path) -> None:
    workspace = tmp_path / 'workspace'
    wheels = workspace / 'wheels'
    source = workspace / 'source'
    wheels.mkdir(parents=True)
    source.mkdir()
    nexus_wheel = _wheel(
        wheels,
        'nexus-agent',
        '1.5.0',
        requirements=(f'{_PARENT}==1.0.0; extra == "runtime"',),
        extras=('runtime',),
    )
    _wheel(wheels, _PARENT, '1.0.0', requirements=(f'{_CHILD}==1.0.0',))
    with pytest.raises(WheelhouseBuildError) as exc_info:
        worker._verify_offline_install(
            subprocess.run,
            workspace=workspace,
            source_dir=source,
            wheels_dir=wheels,
            nexus_wheel=nexus_wheel,
            extras=['runtime'],
            module_host_requirements=[],
            locked_versions={_PARENT: '1.0.0', _CHILD: '1.0.0'},
            expected_version='1.5.0',
        )

    assert exc_info.value.code == 'dependency_offline_install_failed'
    assert 'Nexus wheel and selected extras cannot install offline' in str(exc_info.value)
    assert _CHILD in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('leaf', 'marker', 'extras', 'succeeds', 'leaf_installed'),
    [
        (_BASE_EQUAL, 'extra == "runtime"', [], True, False),
        (_BASE_EQUAL, 'extra == "runtime"', ['runtime'], True, True),
        (_BASE_EQUAL, 'extra == "runtime"', ['runtime', 'other'], True, True),
        (_BASE_NOT_EQUAL, 'extra != "runtime"', [], False, False),
        (_BASE_NOT_EQUAL, 'extra != "runtime"', ['runtime'], False, False),
        (_BASE_NOT_EQUAL, 'extra != "runtime"', ['runtime', 'other'], False, False),
        (_BASE_COMPOUND, 'extra == "runtime" and extra == "other"', [], True, False),
        (_BASE_COMPOUND, 'extra == "runtime" and extra == "other"', ['runtime'], True, False),
        (_BASE_COMPOUND, 'extra == "runtime" and extra == "other"', ['runtime', 'other'], True, False),
    ],
    ids=[
        'equal-zero',
        'equal-one',
        'equal-multiple',
        'not-equal-zero',
        'not-equal-one',
        'not-equal-multiple',
        'compound-zero',
        'compound-one',
        'compound-multiple',
    ],
)
def test_nexus_wheel_root_extra_marker_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf: str,
    marker: str,
    extras: list[str],
    succeeds: bool,
    leaf_installed: bool,
) -> None:
    repository = tmp_path / 'repository'
    repository.mkdir()
    _wheel(repository, _SHARED, '1.0.0')
    _wheel(repository, leaf, '1.0.0')
    requirement = f'{leaf}==1.0.0; {marker}'
    project = tmp_path / 'project'
    _marker_project(project, requirement)
    lock = _lock(project, repository)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    nexus_wheel = _wheel(
        tmp_path,
        'nexus-agent',
        '1.5.0',
        requirements=(requirement,),
        extras=('runtime', 'other'),
        module_requirement=f'{_SHARED}==1.0.0',
    )
    monkeypatch.setenv('UV_OFFLINE', '1')
    monkeypatch.setenv('UV_FIND_LINKS', str(repository))
    monkeypatch.setenv('PIP_NO_INDEX', '1')
    monkeypatch.setenv('PIP_FIND_LINKS', str(repository))
    workspace = tmp_path / 'workspace'

    if succeeds:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock, extras=extras),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=workspace,
        )
        assert (leaf in read_target_lock(workspace / 'module-host.lock')) is leaf_installed
        installed = _installed_versions(workspace / 'verify-venv' / 'bin' / 'python')
        assert (installed.get(leaf) == '1.0.0') is leaf_installed
    else:
        with pytest.raises(WheelhouseBuildError) as exc_info:
            build_wheelhouse(
                spec=_spec(source, nexus_wheel, lock, extras=extras),
                source_package=source,
                nexus_wheel=nexus_wheel,
                uv_lock=lock,
                workspace=workspace,
            )
        assert exc_info.value.code == 'dependency_offline_install_failed'
        assert leaf not in read_target_lock(workspace / 'module-host.lock')
        assert not (workspace / 'wheelhouse.tar.zst').exists()


@pytest.mark.integration
@pytest.mark.parametrize('module_requirement', [f'{_SHARED}>=1', f'{_SHARED}==1.0.0'])
def test_joint_resolution_uses_wheel_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_requirement: str,
) -> None:
    repository = tmp_path / 'repository'
    repository.mkdir()
    _wheel(repository, _SHARED, '1.0.0')
    _wheel(repository, _SHARED, '2.0.0')
    project = tmp_path / 'project'
    _marker_project(project, f'{_SHARED}==1.0.0')
    lock = _lock(project, repository)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    nexus_wheel = _wheel(
        tmp_path,
        'nexus-agent',
        '1.5.0',
        requirements=(f'{_SHARED}==2.0.0',),
        extras=('runtime', 'other'),
        module_requirement=module_requirement,
    )
    monkeypatch.setenv('UV_OFFLINE', '1')
    monkeypatch.setenv('UV_FIND_LINKS', str(repository))
    monkeypatch.setenv('PIP_NO_INDEX', '1')
    monkeypatch.setenv('PIP_FIND_LINKS', str(repository))
    workspace = tmp_path / 'workspace'
    spec = _spec(source, nexus_wheel, lock, extras=['runtime'])

    if module_requirement == f'{_SHARED}==1.0.0':
        with pytest.raises(WheelhouseBuildError) as exc_info:
            build_wheelhouse(
                spec=spec,
                source_package=source,
                nexus_wheel=nexus_wheel,
                uv_lock=lock,
                workspace=workspace,
            )
        assert exc_info.value.code == 'dependency_module_lock_failed'
        assert not (workspace / 'wheelhouse.tar.zst').exists()
    else:
        archive, _metadata = build_wheelhouse(
            spec=spec,
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=workspace,
        )
        assert archive.is_file()
        assert str(read_target_lock(workspace / 'module-host.lock')[_SHARED].specifier) == '==2.0.0'
        assert _installed_versions(workspace / 'verify-venv' / 'bin' / 'python')[_SHARED] == '2.0.0'


@pytest.mark.integration
@pytest.mark.parametrize('key_version', [3, 4])
def test_joint_resolution_and_offline_install_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key_version
) -> None:
    repository = tmp_path / 'repository'
    repository.mkdir()
    _wheel(repository, _SHARED, '1.0.0')
    _wheel(repository, _MARKER, '1.0.0')
    _wheel(repository, _PATCH, '1.0.0')
    _wheel(repository, _CONSTRAINED, '1.0.0')
    _wheel(repository, _CONSTRAINED, '2.0.0')
    _sdist(repository, _SDIST, '1.0.0')
    _wheel(repository, _SOURCE_CONTROLLED, '1.0.0')
    _wheel(repository, _BASE_EQUAL, '1.0.0')
    _wheel(repository, _BASE_NOT_EQUAL, '1.0.0')
    _wheel(repository, _BASE_COMPOUND, '1.0.0')

    compatible_project = tmp_path / 'compatible-project'
    _project(compatible_project, shared_requirement=f'{_SHARED}>=1,<3')
    compatible_lock = _lock(compatible_project, repository)
    source = build_source_package(compatible_project, tmp_path / 'source.tar.zst').path
    _wheel(repository, _SHARED, '2.0.0')

    monkeypatch.setenv('UV_OFFLINE', '1')
    monkeypatch.setenv('UV_FIND_LINKS', str(repository))
    monkeypatch.setenv('PIP_NO_INDEX', '1')
    monkeypatch.setenv('PIP_FIND_LINKS', str(repository))

    scenarios: dict[str, dict[str, str]] = {}
    for name, module_requirement, expected_shared, extras in (
        ('compatible_upgrade_multiple_extras', f'{_SHARED}>=2,<3', '2.0.0', ['runtime', 'other']),
        ('stable_compatible_pin_one_extra', f'{_SHARED}>=1,<3', '1.0.0', ['runtime']),
        ('zero_extras', f'{_SHARED}>=1,<3', '1.0.0', []),
    ):
        nexus_wheel = _wheel(
            tmp_path,
            'nexus-agent',
            '1.5.0',
            requirements=_nexus_requirements(f'{_SHARED}>=1,<3'),
            extras=('runtime', 'distributed', 'other'),
            module_requirement=module_requirement,
        )
        workspace = tmp_path / name
        spec = _spec(source, nexus_wheel, compatible_lock, extras=extras)
        expected_build_key = wheelhouse_build_key(spec, version=key_version)
        archive, metadata = build_wheelhouse(
            spec=spec,
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=compatible_lock,
            workspace=workspace,
            expected_build_key=expected_build_key,
        )
        assert metadata.build_key == expected_build_key
        extracted = tmp_path / f'{name}-archive'
        extract_wheelhouse_archive(archive, extracted, max_files=20, max_total_bytes=10 * 1024 * 1024)
        locked = read_target_lock(extracted / 'locks' / 'module-host.lock')
        assert '.whl' not in (extracted / 'locks' / 'module-host.lock').read_text()
        assert next(iter(locked[_SHARED].specifier)).version == expected_shared
        assert next(iter(locked[_CONSTRAINED].specifier)).version == '1.0.0'
        assert next(iter(locked[_SDIST].specifier)).version == '1.0.0'
        assert next(iter(locked[_SOURCE_CONTROLLED].specifier)).version == '1.0.0'
        if 'runtime' in extras:
            assert next(iter(locked[_MARKER].specifier)).version == '1.0.0'
            assert next(iter(locked[_PATCH].specifier)).version == '1.0.0'
        else:
            assert _MARKER not in locked
            assert _PATCH not in locked
        assert 'source-overrides' not in (extracted / 'locks' / 'module-host.lock').read_text()
        assert metadata.module_lock_path == 'locks/module-host.lock'
        built_sdist_wheel = next((extracted / 'wheels').glob('nexus_resolver_sdist-*.whl'))
        built_sdist_hash = hashlib.sha256(built_sdist_wheel.read_bytes()).hexdigest()
        assert f'sha256:{built_sdist_hash}' not in (extracted / 'locks' / 'module-host.lock').read_text()
        installed = _installed_versions(workspace / 'verify-venv' / 'bin' / 'python')
        expected_installed = {
            _CONSTRAINED: '1.0.0',
            _SDIST: '1.0.0',
            _SOURCE_CONTROLLED: '1.0.0',
            _SHARED: expected_shared,
            'nexus-agent': '1.5.0',
        }
        if 'runtime' in extras:
            expected_installed.update({_MARKER: '1.0.0', _PATCH: '1.0.0'})
        assert installed == expected_installed
        scenarios[name] = {**installed, 'generated_sdist_wheel_sha256': built_sdist_hash}

    conflict_project = tmp_path / 'conflict-project'
    _project(conflict_project, shared_requirement=f'{_SHARED}==1.0.0')
    conflict_lock = _lock(conflict_project, repository)
    conflict_source = build_source_package(conflict_project, tmp_path / 'conflict-source.tar.zst').path
    conflict_wheel = _wheel(
        tmp_path,
        'nexus-agent',
        '1.5.0',
        requirements=_nexus_requirements(f'{_SHARED}==1.0.0'),
        extras=('runtime', 'distributed'),
        module_requirement=f'{_SHARED}>=2,<3',
    )
    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(conflict_source, conflict_wheel, conflict_lock),
            source_package=conflict_source,
            nexus_wheel=conflict_wheel,
            uv_lock=conflict_lock,
            workspace=tmp_path / 'conflict',
        )
    assert exc_info.value.code == 'dependency_module_lock_failed'
    scenarios['genuine_conflict'] = {'error_code': exc_info.value.code}

    evidence = Path(__file__).parent / 'outputs' / 'dependency_build_resolution' / 'evidence.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(scenarios, indent=2, sort_keys=True) + os.linesep)
