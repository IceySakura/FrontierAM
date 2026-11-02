from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
import zipfile

from datetime import timedelta
from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.config import DistributedConfig
from nexus.distributed.instance_manager import bootstrap
from nexus.distributed.instance_manager.bootstrap import _dynamic_manager_id
from nexus.distributed.utils.clock import utcnow


def test_bootstrap_main_monitors_preparation_event_loop(tmp_path, monkeypatch):
    prepared = bootstrap.PreparedEnvironment(
        workspace=tmp_path / 'workspace',
        venv=tmp_path / 'venv',
        environment_id='environment-a',
        pool_id='pool-a',
        workload='execution',
        manager_labels={},
    )
    events: list[str] = []

    async def fake_prepare(*args, **kwargs):
        del args, kwargs
        events.append('prepare')
        return prepared

    async def fake_monitor_event_loop(name, operation):
        events.append(f'monitor:{name}')
        return await operation

    def fake_exec(**kwargs):
        assert kwargs['prepared'] is prepared
        events.append('exec')

    monkeypatch.setattr(bootstrap, 'prepare_allocation_environment', fake_prepare)
    monkeypatch.setattr(bootstrap, 'monitor_event_loop', fake_monitor_event_loop, raising=False)
    monkeypatch.setattr(bootstrap, 'exec_instance_manager', fake_exec)
    monkeypatch.setenv('NEXUS_DIST_ALLOCATION_SPEC_JSON', '{}')
    monkeypatch.setattr(
        bootstrap.sys,
        'argv',
        [
            'bootstrap',
            '--allocation-id',
            'allocation-a',
            '--environment-id',
            'environment-a',
            '--pool-id',
            'pool-a',
            '--workload',
            'execution',
            '--allocation-spec-hash',
            'a' * 64,
            '--cache-root',
            str(tmp_path / 'cache'),
        ],
    )

    bootstrap.main()

    assert events == ['monitor:environment-bootstrap', 'prepare', 'exec']


def test_dynamic_manager_id_uses_allocation_identity_and_pod_ordinal():
    assert _dynamic_manager_id('allocation-abc', 'nexus-allocation-abc-17') == 'allocation-abc-17'

    with pytest.raises(ValueError, match='pod_ordinal_unavailable'):
        _dynamic_manager_id('allocation-abc', 'nexus-allocation-abc')


def test_exec_instance_manager_uses_only_allocation_identity_arguments(tmp_path, monkeypatch):
    captured = {}
    python = tmp_path / 'venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.touch()
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    prepared = bootstrap.PreparedEnvironment(
        workspace=workspace,
        venv=tmp_path / 'venv',
        python=python,
        environment_id='environment-a',
        pool_id='pool-a',
        workload='execution',
        manager_labels={'region': 'gz'},
    )
    monkeypatch.setenv('POD_NAME', 'allocation-a-3')
    monkeypatch.setattr(bootstrap.os, 'chdir', lambda path: captured.setdefault('cwd', path))

    def fake_execve(executable, argv, environment):
        captured.update(executable=executable, argv=argv, environment=environment)

    monkeypatch.setattr(bootstrap.os, 'execve', fake_execve)

    bootstrap.exec_instance_manager(
        prepared=prepared,
        port=8091,
        capacity=20,
        allocation_id='allocation-a',
        allocation_spec_hash='a' * 64,
    )

    argv = captured['argv']
    assert argv[argv.index('--environment-id') + 1] == 'environment-a'
    assert argv[argv.index('--pool-id') + 1] == 'pool-a'
    assert argv[argv.index('--workload') + 1] == 'execution'
    assert argv[argv.index('--instance-manager-id') + 1] == 'allocation-a-3'
    assert {'--environment-key', '--environment-name', '--workload-class'}.isdisjoint(argv)
    assert captured['environment']['NEXUS_DIST_MANAGER_ALLOCATION_ID'] == 'allocation-a'


def test_manager_venv_adds_required_distributed_runtime_extras():
    assert bootstrap._manager_venv_extras([]) == ['distributed', 'runtime']
    assert bootstrap._manager_venv_extras(['browser', 'runtime']) == [
        'browser',
        'distributed',
        'runtime',
    ]


def test_workspace_pythonpath_preserves_preinstalled_code_paths(tmp_path):
    workspace = tmp_path / 'workspace'

    assert bootstrap._workspace_pythonpath(workspace, None) == str(workspace)
    assert bootstrap._workspace_pythonpath(workspace, '/app:/opt/preinstalled') == (
        f'{workspace}:/app:/opt/preinstalled'
    )


async def test_application_gateway_artifact_reader_uses_only_scoped_content_ticket(tmp_path):
    from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport, ArtifactContentReader

    content = b'allocation-scoped artifact content'
    observed = {}

    async def download(request: web.Request) -> web.Response:
        observed.update(request.headers)
        return web.Response(body=content)

    app = web.Application()
    app.router.add_get('/v1/internal/artifacts/source-a/content', download)
    server = TestServer(app)
    await server.start_server()
    token_file = tmp_path / 'artifact-ticket'
    token_file.write_text('scoped-ticket')
    target = tmp_path / 'source.tar.zst'
    transport = ApplicationGatewayArtifactTransport(
        base_url=str(server.make_url('/')).rstrip('/'),
        content_ticket_file=token_file,
    )
    reader = ArtifactContentReader(transport)
    ref = ArtifactRef(
        artifact_id='source-a',
        kind='source_package',
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    try:
        await reader.download(ref, target)
    finally:
        await transport.close()
        await server.close()

    assert target.read_bytes() == content
    assert observed['X-Nexus-Artifact-Content-Ticket'] == 'scoped-ticket'
    assert observed['X-Nexus-Artifact-Kind'] == 'source_package'
    assert observed['X-Nexus-Sha256'] == ref.sha256
    assert 'Authorization' not in observed
    assert 'X-Nexus-Internal-Token' not in observed


async def test_prepare_environment_enforces_allocation_source_package_limits(tmp_path, monkeypatch):
    event_loop_thread = threading.get_ident()
    captured = {'deadlines': [], 'extract_threads': []}

    async def fake_download(_reader, path, _ref, *, deadline):
        captured['deadlines'].append(deadline)
        path.write_bytes(b'content')

    def fake_extract(_archive, _workspace, *, limits):
        captured['limits'] = limits
        captured['extract_threads'].append(threading.get_ident())
        raise ValueError('source-limit-test-stop')

    metadata = SimpleNamespace(build_key='e' * 64, nexus_wheel_path='wheels/nexus.whl')
    wheelhouse_ref = ArtifactRef(
        artifact_id='wheelhouse-a',
        kind='wheelhouse',
        sha256='d' * 64,
        size_bytes=7,
    )
    monkeypatch.setattr(bootstrap, '_ensure_gateway_download', fake_download)
    monkeypatch.setattr(bootstrap, '_ready_dependency_build', lambda *args, **kwargs: (wheelhouse_ref, metadata))

    def fake_extract_wheelhouse(_archive, target, **_kwargs):
        captured['extract_threads'].append(threading.get_ident())
        wheel = target / metadata.nexus_wheel_path
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(b'wheel')

    monkeypatch.setattr(bootstrap, 'extract_wheelhouse_archive', fake_extract_wheelhouse)
    monkeypatch.setattr(bootstrap, 'safe_extract_source_package', fake_extract)
    monkeypatch.setenv(
        'NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT',
        (utcnow() + timedelta(minutes=1)).isoformat(),
    )
    allocation_spec = {
        'code': {
            'mode': 'source_package',
            'source_package': {
                'artifact_id': 'source-a',
                'kind': 'source_package',
                'sha256': 'a' * 64,
                'size_bytes': 7,
            },
            'nexus_wheel': {
                'artifact_id': 'wheel-a',
                'kind': 'nexus_wheel',
                'sha256': 'b' * 64,
                'size_bytes': 7,
            },
            'uv_lock': {
                'artifact_id': 'lock-a',
                'kind': 'uv_lock',
                'sha256': 'c' * 64,
                'size_bytes': 7,
            },
            'extras': [],
        },
        'workspace_size_bytes': 8192,
        'source_package_limits': {
            'max_files': 12,
            'max_single_file_bytes': 1024,
            'max_total_bytes': 4096,
            'max_archive_bytes': 2048,
        },
    }

    with pytest.raises(ValueError, match='source-limit-test-stop'):
        await bootstrap.prepare_allocation_environment(
            'allocation-a',
            environment_id='environment-a',
            pool_id='execution',
            workload='execution',
            allocation_spec=allocation_spec,
            cache_root=tmp_path,
            config=DistributedConfig(),
            reader=object(),  # type: ignore[arg-type]
        )

    limits = captured['limits']
    assert limits.max_files == 12
    assert limits.max_single_file_bytes == 1024
    assert limits.max_total_bytes == 4096
    assert limits.max_archive_bytes == 2048
    assert len(captured['deadlines']) == 2
    assert all(value is not None for value in captured['deadlines'])
    assert len(captured['extract_threads']) == 2
    assert all(thread_id != event_loop_thread for thread_id in captured['extract_threads'])


async def test_cached_artifact_identity_check_runs_outside_event_loop(tmp_path, monkeypatch):
    event_loop_thread = threading.get_ident()
    identity_threads = []
    content = b'cached-content'
    path = tmp_path / 'artifact'
    path.write_bytes(content)
    ref = ArtifactRef(
        artifact_id='artifact-a',
        kind='source_package',
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )

    def fake_file_identity(candidate):
        identity_threads.append(threading.get_ident())
        return len(content), ref.sha256

    class Reader:
        async def download(self, *_args, **_kwargs):
            pytest.fail('matching cached artifact must not be downloaded')

    monkeypatch.setattr(bootstrap, 'file_identity', fake_file_identity)

    await bootstrap._ensure_gateway_download(Reader(), path, ref, deadline=None)  # type: ignore[arg-type]

    assert len(identity_threads) == 1
    assert identity_threads[0] != event_loop_thread


async def test_prepare_image_environment_uses_preinstalled_workspace_without_artifacts(tmp_path):
    workspace = tmp_path / 'image-workspace'
    workspace.mkdir()
    (workspace / 'tasks.py').write_text('VALUE = 7\n')

    class Reader:
        async def download(self, *_args, **_kwargs):
            pytest.fail('image mode must not download source or wheelhouse artifacts')

    prepared = await bootstrap.prepare_allocation_environment(
        'allocation-image',
        environment_id='environment-a',
        pool_id='execution',
        workload='execution',
        allocation_spec={'code': {'mode': 'image', 'image_version': 'user-build-1'}},
        cache_root=tmp_path / 'cache',
        config=DistributedConfig(),
        reader=Reader(),  # type: ignore[arg-type]
        image_workspace=workspace,
    )

    assert prepared.workspace == workspace
    assert prepared.venv == bootstrap.Path(bootstrap.sys.prefix)
    assert prepared.python == bootstrap.Path(bootstrap.sys.executable)


async def test_readonly_image_workspace_is_copied_to_writable_allocation_directory(tmp_path, monkeypatch):
    source = tmp_path / 'image'
    source.mkdir()
    (source / 'tasks.py').write_text('VALUE = 7\n')
    monkeypatch.setattr(bootstrap.os, 'access', lambda *_args: False)
    prepared = await bootstrap.prepare_allocation_environment(
        'allocation-image',
        environment_id='environment-a',
        pool_id='execution',
        workload='execution',
        allocation_spec={'code': {'mode': 'image', 'image_version': 'user-build-1'}, 'workspace_size_bytes': 1024},
        cache_root=tmp_path / 'cache',
        config=DistributedConfig(),
        image_workspace=source,
    )
    assert prepared.workspace == tmp_path / 'cache' / 'allocation-image' / 'workspace'
    assert (prepared.workspace / 'tasks.py').read_text() == 'VALUE = 7\n'
    (prepared.workspace / 'case-01').mkdir()
    assert not (source / 'case-01').exists()
    assert bootstrap._copy_image_workspace(source, tmp_path / 'cache' / 'allocation-image', 1024) == prepared.workspace


def test_image_workspace_copy_enforces_allocation_size_limit(tmp_path):
    source = tmp_path / 'image'
    source.mkdir()
    (source / 'large').write_bytes(b'x' * 1024)
    with pytest.raises(ValueError, match='image_workspace_exceeds_byte_limit'):
        bootstrap._copy_image_workspace(source, tmp_path / 'cache', 1023)


@pytest.mark.parametrize('dependency_check_fails', [False, True])
@pytest.mark.parametrize('legacy_cache', [False, True])
async def test_prepare_allocation_environment_uses_ready_wheelhouse_offline(
    tmp_path, monkeypatch, dependency_check_fails, legacy_cache
):
    from nexus.distributed.dependency_build.wheelhouse import WheelhouseBuildSpec, write_wheelhouse_archive
    from nexus.distributed.submission.source_package import build_source_package

    project = tmp_path / 'project'
    project.mkdir()
    (project / 'pyproject.toml').write_text('[project]\nname = "nexus-agent"\nversion = "1.5.0"\n')
    (project / 'task.py').write_text('VALUE = 7\n')
    source_path = build_source_package(project, tmp_path / 'source.tar.zst').path
    nexus_wheel = tmp_path / 'nexus_agent-1.5.0-py3-none-any.whl'
    with zipfile.ZipFile(nexus_wheel, 'w') as archive:
        archive.writestr('nexus_agent-1.5.0.dist-info/METADATA', 'Name: nexus-agent\nVersion: 1.5.0\n')
        archive.writestr('nexus_agent-1.5.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: py3-none-any\n')
    uv_lock = tmp_path / 'uv.lock'
    uv_lock.write_text('version = 1\n')

    def ref(path, kind):
        return ArtifactRef(
            artifact_id=f'art-{kind}',
            kind=kind,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )

    source_ref = ref(source_path, 'source_package')
    nexus_ref = ref(nexus_wheel, 'nexus_wheel')
    lock_ref = ref(uv_lock, 'uv_lock')
    build_spec = WheelhouseBuildSpec(
        owner_principal_id='principal-a',
        environment_id='environment-a',
        source_package=source_ref,
        nexus_wheel=nexus_ref,
        uv_lock=lock_ref,
        extras=['distributed', 'runtime'],
        workload_image='workload:test',
        builder_image='builder:test',
    )
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    (wheels / nexus_wheel.name).write_bytes(nexus_wheel.read_bytes())
    module_lock = tmp_path / 'module-host.lock'
    module_lock.write_text('boto3==1.42.0 --hash=sha256:boto3\n')
    wheelhouse_path = tmp_path / 'wheelhouse.tar.zst'
    metadata = write_wheelhouse_archive(
        wheels,
        build_spec,
        wheelhouse_path,
        nexus_wheel_path=f'wheels/{nexus_wheel.name}',
        module_names=['litellm'],
        module_host_requirements=['boto3', 'litellm>=1.91.2,<1.92.0'],
        module_lock=module_lock,
    )
    wheelhouse_ref = ref(wheelhouse_path, 'wheelhouse')

    class Reader:
        async def download(self, artifact_ref, path, *, deadline=None):
            del deadline
            source = {
                source_ref.artifact_id: source_path,
                wheelhouse_ref.artifact_id: wheelhouse_path,
            }[artifact_ref.artifact_id]
            path.write_bytes(source.read_bytes())

    captured = {}

    class FakeVenvManager:
        def __init__(self, *, root):
            self.root = root

        async def destroy(self, venv_id):
            captured.setdefault('calls', []).append(('destroy', venv_id))

        async def create(self, venv_id, spec, *, wheelhouse=None):
            captured.setdefault('calls', []).append(('create', venv_id))
            captured['venv_id'] = venv_id
            captured['spec'] = spec
            captured['wheelhouse'] = wheelhouse
            python = self.root / venv_id / 'bin' / 'python'
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text('')
            return SimpleNamespace(state='ready', error=None)

    marker = tmp_path / 'cache' / 'allocation-a' / '.prepared'
    if legacy_cache:
        old_python = marker.parent / 'manager-venvs' / 'manager' / 'bin' / 'python'
        old_python.parent.mkdir(parents=True)
        old_python.touch()
        marker.write_text(
            f'{source_ref.sha256}\n{wheelhouse_ref.sha256}\n{metadata.build_key}\ndistributed,runtime\n{{}}\n'
        )

    async def check_dependencies(python):
        assert python.is_file()
        assert not marker.exists()
        captured.setdefault('calls', []).append(('check', 'manager'))
        if dependency_check_fails:
            raise RuntimeError('manager_dependency_check_failed: incompatible fixture')

    monkeypatch.setattr(bootstrap, 'VenvManager', FakeVenvManager)
    monkeypatch.setattr(bootstrap, '_check_manager_dependencies', check_dependencies, raising=False)
    operation = bootstrap.prepare_allocation_environment(
        'allocation-a',
        environment_id='environment-a',
        pool_id='execution',
        workload='execution',
        allocation_spec={
            'code': {
                'mode': 'source_package',
                'source_package': source_ref.model_dump(mode='json'),
                'nexus_wheel': nexus_ref.model_dump(mode='json'),
                'uv_lock': lock_ref.model_dump(mode='json'),
                'extras': [],
            },
            'dependency_build': {
                'dependency_build_id': 'build-a',
                'build_key': metadata.build_key,
                'wheelhouse': wheelhouse_ref.model_dump(mode='json'),
                'metadata': metadata.model_dump(mode='json'),
            },
            'base_image': 'workload:test',
            'workspace_size_bytes': 1024 * 1024,
        },
        cache_root=tmp_path / 'cache',
        config=DistributedConfig(),
        reader=Reader(),  # type: ignore[arg-type]
    )

    if dependency_check_fails:
        with pytest.raises(RuntimeError, match='manager_dependency_check_failed'):
            await operation
        assert not marker.exists()
        return

    prepared = await operation
    assert marker.is_file()
    assert (prepared.workspace / 'task.py').read_text() == 'VALUE = 7\n'
    assert prepared.environment_id == 'environment-a'
    assert prepared.pool_id == 'execution'
    assert prepared.workload == 'execution'
    assert captured['venv_id'] == 'manager'
    assert captured['calls'] == [('destroy', 'manager'), ('create', 'manager'), ('check', 'manager')]
    assert captured['wheelhouse'].is_dir()
    assert captured['spec'].artifacts[0].extras == ['distributed', 'runtime']
    assert captured['spec'].extra_requirements == ['boto3', 'litellm>=1.91.2,<1.92.0']


@pytest.mark.parametrize('leaf_version', ['1.0', '2.0'])
async def test_manager_dependency_check_detects_inconsistent_installed_environment(tmp_path, leaf_version):
    venv = tmp_path / 'venv'
    subprocess.run(
        [bootstrap.VenvManager._resolve_uv(), 'venv', '--no-config', '--python', sys.executable, str(venv)],
        check=True,
        capture_output=True,
    )
    python = venv / 'bin' / 'python'
    site_packages = bootstrap.Path(
        subprocess.check_output(
            [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
            text=True,
        ).strip()
    )
    for name, version, requires in [
        ('probe-root', '1.0', 'Requires-Dist: probe-leaf>=2.0\n'),
        ('probe-leaf', leaf_version, ''),
    ]:
        metadata = site_packages / f'{name.replace("-", "_")}-{version}.dist-info'
        metadata.mkdir()
        (metadata / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requires}\n')
        (metadata / 'WHEEL').write_text('Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    if leaf_version == '1.0':
        with pytest.raises(RuntimeError, match=r'manager_dependency_check_failed[\s\S]*requires.*probe-leaf'):
            await bootstrap._check_manager_dependencies(python)
    else:
        await bootstrap._check_manager_dependencies(python)
