from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import zipfile

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.core.module_dependencies import ModuleSelector
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.dependency_build import worker
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseBuildSpec,
    WheelhouseMetadata,
    extract_wheelhouse_archive,
    wheelhouse_build_key,
)
from nexus.distributed.dependency_build.worker import WheelhouseBuildError, build_wheelhouse, run_builder
from nexus.distributed.submission.source_package import build_source_package
from nexus.distributed.utils.clock import utcnow


def test_builder_main_monitors_its_process_event_loop(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source.tar.zst'
    source.write_bytes(b'source')
    nexus_wheel = tmp_path / 'nexus.whl'
    nexus_wheel.write_bytes(b'wheel')
    lock = tmp_path / 'uv.lock'
    lock.write_bytes(b'lock')
    spec = _spec(source, nexus_wheel, lock)
    result = ArtifactRef(
        artifact_id='wheelhouse-a',
        kind='wheelhouse',
        sha256='a' * 64,
        size_bytes=123,
    )
    events: list[str] = []

    async def fake_run_builder(**kwargs):
        assert kwargs['spec'] == spec
        assert kwargs['expected_build_key'] == wheelhouse_build_key(spec, version=3)
        events.append('run')
        return result

    async def fake_monitor_event_loop(name, operation):
        events.append(f'monitor:{name}')
        return await operation

    monkeypatch.setattr(worker, '_parse_args', lambda: SimpleNamespace(workspace=str(tmp_path)))
    monkeypatch.setattr(worker, 'run_builder', fake_run_builder)
    monkeypatch.setattr(worker, 'monitor_event_loop', fake_monitor_event_loop, raising=False)
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_SPEC_JSON', spec.model_dump_json())
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_KEY', wheelhouse_build_key(spec, version=3))
    monkeypatch.setenv('NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL', 'http://application.internal')
    monkeypatch.setenv('NEXUS_DIST_ARTIFACT_CONTENT_TOKEN_FILE', str(tmp_path / 'ticket'))

    worker.main()

    assert events == ['monitor:dependency-builder', 'run']
    assert 'wheelhouse-a' in capsys.readouterr().out


def _ref(kind: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f'art-{kind}',
        kind=kind,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _nexus_wheel(
    path: Path,
    *,
    version: str = '1.5.0',
    extras: tuple[str, ...] = ('runtime', 'distributed'),
    requirements: tuple[str, ...] = (),
    with_litellm_module: bool = False,
) -> None:
    with zipfile.ZipFile(path, 'w') as archive:
        metadata = f'Name: nexus-agent\nVersion: {version}\n'
        metadata += ''.join(f'Provides-Extra: {extra}\n' for extra in extras)
        metadata += ''.join(f'Requires-Dist: {requirement}\n' for requirement in requirements)
        archive.writestr(f'nexus_agent-{version}.dist-info/METADATA', metadata)
        archive.writestr(f'nexus_agent-{version}.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: py3-none-any\n')
        if with_litellm_module:
            archive.writestr(
                'nexus/extensions/openai_clients/litellm.py',
                '# /// script\n'
                '# dependencies = []\n'
                '#\n'
                '# [project]\n'
                '# name = "litellm"\n'
                '#\n'
                '# [optional-dependencies]\n'
                '# host = ["boto3", "litellm>=1.91.2,<1.92.0"]\n'
                '#\n'
                '# [tool.nexus]\n'
                '# openai_clients = { litellm = "nexus.extensions.openai_clients.litellm" }\n'
                '# ///\n',
            )


def _project(
    tmp_path: Path,
    *,
    version: str = '1.5.0',
    extras: tuple[str, ...] = ('runtime', 'distributed'),
    dependencies: tuple[str, ...] = (),
    optional_dependencies: dict[str, tuple[str, ...]] | None = None,
    constraints: tuple[str, ...] = (),
) -> Path:
    project = tmp_path / 'project'
    project.mkdir()
    optional_dependencies = optional_dependencies or dict.fromkeys(extras, ())
    dependency_lines = ', '.join(repr(value) for value in dependencies)
    optional_lines = '\n'.join(
        f'{extra} = [{", ".join(repr(value) for value in values)}]' for extra, values in optional_dependencies.items()
    )
    constraint_lines = ', '.join(repr(value) for value in constraints)
    tool_uv = f'\n[tool.uv]\nconstraint-dependencies = [{constraint_lines}]\n' if constraints else ''
    (project / 'pyproject.toml').write_text(
        f'[project]\nname = "nexus-agent"\nversion = "{version}"\ndependencies = [{dependency_lines}]\n'
        f'[project.optional-dependencies]\n{optional_lines}\n{tool_uv}'
    )
    (project / 'nexus').mkdir()
    (project / 'nexus' / '__init__.py').write_text('')
    return project


def _spec(
    source: Path,
    nexus_wheel: Path,
    lock: Path,
    *,
    extras: list[str] | None = None,
    module_selectors: list[ModuleSelector] | None = None,
) -> WheelhouseBuildSpec:
    return WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=_ref('source_package', source),
        nexus_wheel=_ref('nexus_wheel', nexus_wheel),
        uv_lock=_ref('uv_lock', lock),
        extras=extras if extras is not None else ['runtime', 'distributed'],
        module_selectors=module_selectors or [],
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )


@pytest.mark.parametrize('dangling', [False, True], ids=['directory-target', 'dangling-target'])
def test_builder_rejects_symlink_workspace_without_touching_target(tmp_path, dangling):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)
    target = tmp_path / 'unrelated-target'
    if not dangling:
        target.mkdir()
        (target / 'sentinel').write_text('preserve me')
    workspace = tmp_path / 'workspace-link'
    workspace.symlink_to(target, target_is_directory=True)

    def unexpected_run(*args, **kwargs):
        pytest.fail('build commands must not start for a symlink workspace')

    with pytest.raises(WheelhouseBuildError, match='workspace must not be a symbolic link') as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=workspace,
            runner=unexpected_run,
        )

    assert exc_info.value.code == 'dependency_workspace_invalid'
    assert workspace.is_symlink()
    assert workspace.readlink() == target
    if dangling:
        assert not target.exists()
    else:
        assert {path.name for path in target.iterdir()} == {'sentinel'}
        assert (target / 'sentinel').read_text() == 'preserve me'


@pytest.mark.parametrize('key_version', [None, 3, 4])
def test_builder_exports_frozen_lock_and_packages_all_wheels(tmp_path, key_version):
    project = _project(tmp_path, dependencies=('dependency==2.0',))
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, requirements=('dependency==2.0',))
    spec = _spec(source, nexus_wheel, lock)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        assert check is True
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('dependency==2.0 --hash=sha256:dependency\n')
        elif command[:3] == [sys.executable, '-m', 'pip']:
            wheel_dir = Path(command[command.index('--wheel-dir') + 1])
            (wheel_dir / 'dependency-2.0-py3-none-any.whl').write_bytes(b'dependency')

    archive, metadata = build_wheelhouse(
        spec=spec,
        source_package=source,
        nexus_wheel=nexus_wheel,
        uv_lock=lock,
        workspace=tmp_path / 'workspace',
        runner=fake_run,
        expected_build_key=wheelhouse_build_key(spec, version=key_version) if key_version else None,
    )

    assert '--frozen' in commands[0]
    assert '--no-emit-project' in commands[0]
    assert commands[0].count('--extra') == 2
    assert commands[1][1:3] == ['-m', 'pip']
    assert '--no-build-isolation' in commands[1]
    assert '--only-binary=:all:' not in commands[1]
    assert commands[2][:2] == ['uv', 'venv']
    assert '--no-config' in commands[2]
    assert commands[3][:3] == ['uv', 'pip', 'install']
    assert '--no-deps' not in commands[3]
    assert '--require-hashes' not in commands[3]
    assert commands[3][commands[3].index('--find-links') + 1].endswith('/workspace/wheels')
    assert commands[3][-1].endswith('/workspace/wheels/nexus_agent-1.5.0-py3-none-any.whl[distributed,runtime]')
    assert commands[4][:3] == ['uv', 'pip', 'check']
    assert commands[5][0].endswith('/workspace/verify-venv/bin/python')
    assert commands[5][-2] == '1.5.0'
    assert '--no-config' not in commands[0]
    assert metadata == WheelhouseMetadata(
        build_key=wheelhouse_build_key(spec, version=key_version or 4),
        nexus_wheel_path='wheels/nexus_agent-1.5.0-py3-none-any.whl',
    )
    extracted = tmp_path / 'extracted'
    extract_wheelhouse_archive(archive, extracted, max_files=10, max_total_bytes=1024)
    assert {path.name for path in (extracted / 'wheels').iterdir()} == {
        'dependency-2.0-py3-none-any.whl',
        'nexus_agent-1.5.0-py3-none-any.whl',
    }


def test_builder_compiles_and_archives_selected_module_host_requirements(tmp_path, monkeypatch):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, with_litellm_module=True)
    spec = _spec(
        source,
        nexus_wheel,
        lock,
        module_selectors=[ModuleSelector(kind='object', object_type='openai_clients', value='litellm')],
    )
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        assert check is True
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('')
        elif command[:3] == ['uv', 'pip', 'compile']:
            Path(command[command.index('--output-file') + 1]).write_text(
                'boto3==1.42.0 --hash=sha256:boto3\nlitellm==1.91.9 --hash=sha256:litellm\n'
            )
        elif command[:3] == [sys.executable, '-m', 'pip']:
            wheel_dir = Path(command[command.index('--wheel-dir') + 1])
            (wheel_dir / 'boto3-1.42.0-py3-none-any.whl').write_bytes(b'boto3')
            (wheel_dir / 'litellm-1.91.9-py3-none-any.whl').write_bytes(b'litellm')

    monkeypatch.chdir(tmp_path)
    archive, metadata = build_wheelhouse(
        spec=spec,
        source_package=source,
        nexus_wheel=nexus_wheel,
        uv_lock=lock,
        workspace=Path('workspace'),
        runner=fake_run,
    )

    compile_command = next(command for command in commands if command[:3] == ['uv', 'pip', 'compile'])
    assert '--generate-hashes' in compile_command
    assert '--no-sources' in compile_command
    assert compile_command[compile_command.index('--python-version') + 1] == platform.python_version()
    assert compile_command[compile_command.index('--no-emit-package') + 1] == 'nexus-agent'
    assert '--extra' not in compile_command
    assert compile_command[-2].endswith('/nexus-root.in')
    assert compile_command[-1].endswith('/module-host.in')
    nexus_root = Path(compile_command[-2]).read_text()
    assert 'nexus-agent[distributed,runtime] @ file://' in nexus_root
    assert 'nexus_agent-1.5.0-py3-none-any.whl' in nexus_root
    assert Path(nexus_root.split('file://', 1)[1].strip()).is_file()
    wheel_command = next(command for command in commands if command[:3] == [sys.executable, '-m', 'pip'])
    assert wheel_command.count('--requirement') == 1
    assert wheel_command[wheel_command.index('--requirement') + 1].endswith('/module-host.lock')
    offline_installs = [command for command in commands if command[:3] == ['uv', 'pip', 'install']]
    assert len(offline_installs) == 2
    assert '--no-deps' not in offline_installs[0]
    assert '--require-hashes' not in offline_installs[0]
    assert offline_installs[0][-1].endswith('/workspace/wheels/nexus_agent-1.5.0-py3-none-any.whl[distributed,runtime]')
    assert offline_installs[1][-2:] == ['boto3', 'litellm>=1.91.2,<1.92.0']
    assert '--require-hashes' not in offline_installs[1]
    assert any(command[:3] == ['uv', 'pip', 'check'] for command in commands)
    offline_validation = next(command for command in commands if command[0].endswith('/verify-venv/bin/python'))
    assert offline_validation[-2] == '1.5.0'
    assert 'litellm>=1.91.2,<1.92.0' not in offline_validation
    assert metadata.module_names == ['litellm']
    assert metadata.module_paths == ['nexus.extensions.openai_clients.litellm']
    assert metadata.module_host_requirements == ['boto3', 'litellm>=1.91.2,<1.92.0']
    extracted = tmp_path / 'module-extracted'
    extract_wheelhouse_archive(archive, extracted, max_files=10, max_total_bytes=4096)
    assert (extracted / 'locks' / 'module-host.lock').read_text().startswith('boto3==1.42.0')


def test_builder_jointly_resolves_compatible_module_requirement_from_stale_base_pin(tmp_path):
    project = _project(
        tmp_path,
        dependencies=('litellm>=1.90,<1.92',),
        optional_dependencies={
            'runtime': ('marker-dep==2; python_version >= "3.13"',),
            'distributed': (),
        },
        constraints=('litellm<1.92',),
    )
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(
        nexus_wheel,
        requirements=(
            'litellm<1.92,>=1.90',
            'marker-dep==2; python_version >= "3.13" and extra == "runtime"',
        ),
        with_litellm_module=True,
    )
    spec = _spec(
        source,
        nexus_wheel,
        lock,
        module_selectors=[ModuleSelector(kind='object', object_type='openai_clients', value='litellm')],
    )

    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text(
                'litellm==1.90.0 --hash=sha256:old\nmarker-dep==2 --hash=sha256:marker\n'
            )
        elif command[:3] == ['uv', 'pip', 'compile']:
            output = Path(command[command.index('--output-file') + 1])
            assert output.read_text().startswith('litellm==1.90.0')
            output.write_text(
                'boto3==1.42.0 --hash=sha256:boto3\n'
                'litellm==1.91.9 --hash=sha256:new\n'
                'marker-dep==2 --hash=sha256:marker\n'
            )
        elif command[:3] == [sys.executable, '-m', 'pip']:
            wheel_dir = Path(command[command.index('--wheel-dir') + 1])
            (wheel_dir / 'boto3-1.42.0-py3-none-any.whl').write_bytes(b'boto3')
            (wheel_dir / 'litellm-1.91.9-py3-none-any.whl').write_bytes(b'litellm')
            (wheel_dir / 'marker_dep-2-py3-none-any.whl').write_bytes(b'marker')

    build_wheelhouse(
        spec=spec,
        source_package=source,
        nexus_wheel=nexus_wheel,
        uv_lock=lock,
        workspace=tmp_path / 'workspace',
        runner=fake_run,
    )

    compile_command = next(command for command in commands if command[:3] == ['uv', 'pip', 'compile'])
    assert compile_command[-2].endswith('/nexus-root.in')
    assert compile_command[-1].endswith('/module-host.in')
    assert compile_command.count('--extra') == 0
    constraints_path = Path(compile_command[compile_command.index('--constraint') + 1])
    assert constraints_path.read_text() == 'litellm<1.92\n'


def test_builder_reports_joint_resolution_conflict_from_real_project_pin(tmp_path):
    project = _project(tmp_path, dependencies=('litellm==1.90.0',))
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, requirements=('litellm==1.90.0',), with_litellm_module=True)
    spec = _spec(
        source,
        nexus_wheel,
        lock,
        module_selectors=[ModuleSelector(kind='object', object_type='openai_clients', value='litellm')],
    )
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('litellm==1.90.0 --hash=sha256:old\n')
        elif command[:3] == ['uv', 'pip', 'compile']:
            raise subprocess.CalledProcessError(
                1, command, stderr='No solution found for litellm==1.90.0 and litellm>=1.91.2'
            )

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=spec,
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=fake_run,
        )

    assert exc_info.value.code == 'dependency_module_lock_failed'
    assert 'No solution found' in str(exc_info.value)
    assert any(command[:3] == ['uv', 'pip', 'compile'] for command in commands)


def test_builder_rejects_source_and_wheel_version_mismatch_before_running_commands(tmp_path):
    project = _project(tmp_path, version='1.5.1')
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, version='1.5.0')
    commands = []

    with pytest.raises(WheelhouseBuildError, match='source project version does not match nexus-agent wheel'):
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=lambda *args, **kwargs: commands.append(args),
        )

    assert commands == []


def test_builder_rejects_unrelated_recorded_key_before_processing_inputs(tmp_path):
    source = tmp_path / 'source.tar.zst'
    nexus_wheel = tmp_path / 'nexus.whl'
    lock = tmp_path / 'uv.lock'
    for path in (source, nexus_wheel, lock):
        path.write_bytes(b'unprocessed input')
    workspace = tmp_path / 'workspace'

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=workspace,
            expected_build_key='0' * 64,
        )

    assert exc_info.value.code == 'dependency_build_key_mismatch'
    assert not workspace.exists()


def test_builder_enforces_source_package_limits_from_build_spec(tmp_path):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)
    spec = WheelhouseBuildSpec.model_validate({
        **_spec(source, nexus_wheel, lock).model_dump(mode='json'),
        'source_package_limits': {
            'max_files': 1,
            'max_single_file_bytes': 1024 * 1024,
            'max_total_bytes': 1024 * 1024,
            'max_archive_bytes': 1024 * 1024,
        },
    })
    commands = []

    with pytest.raises(WheelhouseBuildError, match='entry count limit'):
        build_wheelhouse(
            spec=spec,
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=lambda *args, **kwargs: commands.append(args),
        )

    assert commands == []


def test_builder_rejects_selected_extra_missing_from_source_project(tmp_path):
    project = _project(tmp_path, extras=('runtime',))
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)

    with pytest.raises(WheelhouseBuildError, match='source project does not define extras: distributed'):
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
        )


def test_builder_rejects_selected_extra_missing_from_nexus_wheel(tmp_path):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, extras=('runtime',))

    with pytest.raises(WheelhouseBuildError, match='nexus-agent wheel does not define extras: distributed'):
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
        )


def test_builder_reports_frozen_lock_mismatch(tmp_path):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)

    def fake_run(command, *, cwd, check, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr='lockfile needs to be updated')

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=fake_run,
        )

    assert exc_info.value.code == 'dependency_lock_export_failed'
    assert 'lockfile needs to be updated' in str(exc_info.value)


def test_builder_fails_clearly_when_locked_source_cannot_build_with_pinned_environment(tmp_path):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)

    def fake_run(command, *, cwd, check, **kwargs):
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('')
            return
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr='BackendUnavailable: Cannot import build backend',
        )

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=fake_run,
        )

    assert exc_info.value.code == 'dependency_wheel_build_failed'
    assert 'Cannot import build backend' in str(exc_info.value)


def test_builder_rejects_wheel_version_that_disagrees_with_target_lock_before_offline_install(tmp_path):
    project = _project(tmp_path, dependencies=('dependency==2.0',))
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel, requirements=('dependency==2.0',))
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('dependency==2.0 --hash=sha256:dependency\n')
        elif command[:3] == [sys.executable, '-m', 'pip']:
            wheel_dir = Path(command[command.index('--wheel-dir') + 1])
            (wheel_dir / 'dependency-1.0-py3-none-any.whl').write_bytes(b'wrong dependency')

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=fake_run,
        )

    assert exc_info.value.code == 'dependency_wheelhouse_inconsistent'
    assert 'does not satisfy target lock' in str(exc_info.value)
    assert not any(command[:2] == ['uv', 'venv'] for command in commands)


def test_builder_fails_before_publish_when_wheelhouse_cannot_install_offline(tmp_path):
    project = _project(tmp_path)
    source = build_source_package(project, tmp_path / 'source.tar.zst').path
    lock = tmp_path / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    nexus_wheel = tmp_path / 'nexus-agent.whl'
    _nexus_wheel(nexus_wheel)

    def fake_run(command, *, cwd, check, **kwargs):
        if command[:2] == ['uv', 'export']:
            Path(command[command.index('--output-file') + 1]).write_text('')
        elif command[:3] == [sys.executable, '-m', 'pip']:
            return
        elif command[:3] == ['uv', 'pip', 'install']:
            raise subprocess.CalledProcessError(1, command, stderr='missing dependency wheel')

    with pytest.raises(WheelhouseBuildError) as exc_info:
        build_wheelhouse(
            spec=_spec(source, nexus_wheel, lock),
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=lock,
            workspace=tmp_path / 'workspace',
            runner=fake_run,
        )

    assert exc_info.value.code == 'dependency_offline_install_failed'
    assert 'missing dependency wheel' in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_builder_reports_bounded_failure_before_job_exits(tmp_path, monkeypatch):
    source = tmp_path / 'source.tar.zst'
    source.write_bytes(b'source')
    nexus_wheel = tmp_path / 'nexus.whl'
    nexus_wheel.write_bytes(b'wheel')
    lock = tmp_path / 'uv.lock'
    lock.write_bytes(b'lock')
    spec = _spec(source, nexus_wheel, lock)
    reported = {}
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_LOG_DIR', str(tmp_path / 'collected'))
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_ID', 'bld-failure')
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_ATTEMPT', '1')

    class FakeTransport:
        def __init__(self, **kwargs):
            self.closed = False

        async def close(self):
            self.closed = True

    class FakeGateway:
        def __init__(self, transport):
            reported['transport'] = transport

        async def download(self, artifact_ref, destination, *, deadline):
            reported['download_deadline'] = deadline
            raise WheelhouseBuildError('dependency_offline_install_failed', 'x' * 3000)

        async def report_failure(self, *, error_code, error_message):
            reported['error_code'] = error_code
            reported['error_message'] = error_message

    monkeypatch.setattr(
        'nexus.distributed.dependency_build.worker.ApplicationGatewayArtifactTransport',
        FakeTransport,
    )
    monkeypatch.setattr(
        'nexus.distributed.dependency_build.worker.DependencyBuildArtifactGateway',
        FakeGateway,
    )
    monkeypatch.setenv(
        'NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT',
        (utcnow() + timedelta(minutes=1)).isoformat(),
    )

    with pytest.raises(WheelhouseBuildError, match='x{20}'):
        await run_builder(
            spec=spec,
            base_url='http://application.internal',
            token_file=tmp_path / 'ticket',
            workspace=tmp_path / 'workspace',
        )

    assert reported['error_code'] == 'dependency_offline_install_failed'
    assert len(reported['error_message']) == 2048
    assert reported['download_deadline'] is not None
    assert reported['transport'].closed is True
    assert 'bld-failure' in (tmp_path / 'collected' / 'build.jsonl').read_text()
    assert 'Build failed' in (tmp_path / 'collected' / 'build.jsonl').read_text()


def test_command_output_is_visible_before_process_exits(tmp_path, monkeypatch):
    import builtins

    marker = tmp_path / 'observed'
    output = []

    def observe(*args, **kwargs):
        text = ' '.join(map(str, args))
        output.append(text)
        if 'first-line' in text:
            marker.touch()

    monkeypatch.setattr(builtins, 'print', observe)
    script = (
        'import pathlib,time,sys; print("first-line", flush=True); '
        f'p=pathlib.Path({str(marker)!r}); deadline=time.monotonic()+5; '
        '\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)'
        '\nsys.exit(0 if p.exists() else 1)'
    )
    worker._run_checked(
        worker._run_streaming,
        [sys.executable, '-c', script],
        cwd=tmp_path,
        code='test_failed',
        description='test command',
    )
    assert marker.exists()


def test_command_output_and_failure_tail_are_bounded_and_redacted(tmp_path, capsys):
    script = (
        'import sys; '
        'print("x" * 3000000); '
        'print("https://user:secret@example.org/simple?token=private"); '
        'print("last failure", file=sys.stderr); sys.exit(7)'
    )
    with pytest.raises(WheelhouseBuildError) as error:
        worker._run_checked(
            worker._run_streaming,
            [sys.executable, '-c', script],
            cwd=tmp_path,
            code='test_failed',
            description='test command',
        )
    output = capsys.readouterr().out
    assert len(output.encode()) < 2200000
    assert 'omitted' in output
    assert len(str(error.value)) < 2100
    assert 'last failure' in str(error.value)
    assert 'secret' not in str(error.value)
    assert 'private' not in str(error.value)


def test_many_short_command_lines_have_a_total_output_limit(tmp_path, capsys):
    worker._run_streaming(
        [sys.executable, '-c', 'for i in range(4000): print("x" * 1000)'],
        cwd=tmp_path,
    )
    output = capsys.readouterr().out
    assert len(output.encode()) < 2100000
    assert output.count('further output available in build log file') == 1
