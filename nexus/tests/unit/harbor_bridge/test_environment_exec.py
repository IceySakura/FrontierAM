import base64
import io
import tarfile
import threading

from types import SimpleNamespace

import pytest

from nexus.extensions.harbor_bridge import environment as environment_module
from nexus.extensions.harbor_bridge.environment import GongfengEnvironment
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.models import RemoteInvokeError


class FakeRuntime:
    def __init__(self, *, upload_success=True, cleanup_success=True, command_failures=None):
        self.commands = []
        self.uploads = []
        self.upload_success = upload_success
        self.cleanup_success = cleanup_success
        self.command_failures = command_failures or {}

    async def upload_file(self, *, files, overwrite):
        self.uploads.extend(files)
        assert overwrite is True
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    path=file.path,
                    success=self.upload_success,
                    message=None if self.upload_success else 'simulated upload failure',
                )
                for file in files
            ]
        )

    async def run_command(self, *, command, timeout):
        self.commands.append((command, timeout))
        if command.startswith('rm -rf ') and not self.cleanup_success:
            return SimpleNamespace(stdout='', stderr='simulated cleanup failure', return_code=1)
        for marker, message in self.command_failures.items():
            if marker in command:
                return SimpleNamespace(stdout='', stderr=message, return_code=1)
        return SimpleNamespace(stdout='ok', stderr='', return_code=0)


class FakeDownloadRuntime(FakeRuntime):
    def __init__(self, *, download_content, download_error=None):
        super().__init__()
        self.download_content = download_content
        self.download_error = download_error
        self.downloads = []

    async def download_file(self, *, files):
        self.downloads.extend(files)
        if self.download_error:
            return SimpleNamespace(
                files=[],
                errors=[
                    SimpleNamespace(
                        path=files[0].path,
                        message=self.download_error,
                    )
                ],
            )
        return SimpleNamespace(
            files=[
                SimpleNamespace(
                    path=file.path,
                    content=self.download_content,
                    encoding=file.encoding,
                )
                for file in files
            ],
            errors=None,
        )


class RaisingRuntime(FakeRuntime):
    def __init__(self, error):
        super().__init__()
        self.error = error

    async def run_command(self, *, command, timeout):
        self.commands.append((command, timeout))
        raise self.error


class MissingExitRuntime(FakeRuntime):
    def __init__(self, status: CommandRunStatus, *, stderr: str = ''):
        super().__init__()
        self.status = status
        self.stderr = stderr

    async def run_command(self, *, command, timeout):
        self.commands.append((command, timeout))
        return SimpleNamespace(
            status=self.status,
            return_code=None,
            stdout='',
            stderr=self.stderr,
        )


def make_environment(runtime):
    environment = object.__new__(GongfengEnvironment)
    environment._runtime = runtime
    environment._default_timeout_sec = 30
    environment._resolve_user = lambda user: 'root'
    environment._merge_env = lambda env: env or {}
    return environment


def test_capabilities_only_claim_network_isolation_when_policy_is_active():
    environment = object.__new__(GongfengEnvironment)

    environment._network_isolation_enabled = False
    assert environment.capabilities.disable_internet is False

    environment._network_isolation_enabled = True
    assert environment.capabilities.disable_internet is True


def test_capabilities_describe_allowlist_and_gpu_runtime(tmp_path):
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    task_dir = tmp_path / 'task'
    environment_dir = task_dir / 'environment'
    environment_dir.mkdir(parents=True)
    trial_paths = TrialPaths(tmp_path / 'trial')
    trial_paths.mkdir()
    policy = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=['judge.woa.com'],
    )

    environment = GongfengEnvironment(
        runtime=FakeRuntime(),
        environment_dir=environment_dir,
        environment_name='test',
        session_id='test',
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image='test:v1',
            gpus=1,
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=['judge.woa.com'],
        ),
        network_isolation_enabled=True,
        harbor_network_policy=policy,
        gpus_enabled=True,
    )

    assert environment.capabilities.gpus is True
    assert environment.capabilities.network_allowlist is True
    assert environment.capabilities.network_allowlist_hostnames is True
    assert environment.capabilities.dynamic_network_policy is False


@pytest.mark.asyncio
async def test_exec_uploads_env_without_logging_secret_values():
    runtime = FakeRuntime()
    environment = make_environment(runtime)

    result = await environment.exec(
        command='python /tests/judge.py',
        env={'OPENAI_API_KEY': 'super-secret-value', 'SAFE_FLAG': '1'},
    )

    assert result.return_code == 0
    assert len(runtime.uploads) == 1
    assert 'super-secret-value' in runtime.uploads[0].content
    assert all('super-secret-value' not in command for command, _ in runtime.commands)
    assert runtime.commands[0][0].startswith('install -d -m 700 /tmp/.harbor_exec_env_')
    assert any('chmod 600 /tmp/.harbor_exec_env_' in command for command, _ in runtime.commands)
    assert runtime.commands[-1][0].startswith('rm -rf /tmp/.harbor_exec_env_')


@pytest.mark.asyncio
async def test_exec_fails_when_env_upload_reports_failure():
    runtime = FakeRuntime(upload_success=False)
    environment = make_environment(runtime)

    with pytest.raises(RuntimeError, match='Failed to upload verifier env file'):
        await environment.exec(
            command='python /tests/judge.py',
            env={'OPENAI_API_KEY': 'super-secret-value'},
        )
    assert not any('python /tests/judge.py' in command for command, _ in runtime.commands)
    assert runtime.commands[-1][0].startswith('rm -rf /tmp/.harbor_exec_env_')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [
        RuntimeUnhealthyError('sandbox stopped'),
        RemoteInvokeError('runtime RPC failed'),
        ConnectionError('runtime disconnected'),
        TimeoutError('runtime request timed out'),
    ],
)
async def test_exec_propagates_runtime_transport_errors(error):
    environment = make_environment(RaisingRuntime(error))

    with pytest.raises(type(error), match=str(error)):
        await environment.exec(command='python /tests/judge.py')


@pytest.mark.asyncio
async def test_exec_preserves_ordinary_nonzero_command_result():
    environment = make_environment(FakeRuntime(command_failures={'judge.py': 'tests failed'}))

    result = await environment.exec(command='python /tests/judge.py')

    assert result.return_code == 1
    assert result.stderr == 'tests failed'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status', 'expected_return_code'),
    [
        (CommandRunStatus.TimeLimitExceeded, 124),
        (CommandRunStatus.Error, 1),
    ],
)
async def test_exec_normalizes_missing_return_code(status, expected_return_code):
    environment = make_environment(MissingExitRuntime(status, stderr='runtime detail'))

    result = await environment.exec(command='python /tests/judge.py')

    assert result.return_code == expected_return_code
    assert result.stderr == f'runtime detail\nruntime command status={status.value}; no process exit code was returned'


@pytest.mark.asyncio
async def test_exec_rejects_finished_command_without_return_code():
    environment = make_environment(MissingExitRuntime(CommandRunStatus.Finished))

    with pytest.raises(RuntimeError, match='Finished command returned no exit code'):
        await environment.exec(command='python /tests/judge.py')


@pytest.mark.asyncio
async def test_exec_cleanup_failure_does_not_mask_transport_error():
    error = RuntimeUnhealthyError('primary runtime failure')
    environment = make_environment(RaisingRuntime(error))

    with pytest.raises(RuntimeUnhealthyError, match='primary runtime failure') as exc_info:
        await environment.exec(
            command='python /tests/judge.py',
            env={'SAFE_FLAG': '1'},
        )

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_exec_cleanup_failure_does_not_silently_succeed():
    runtime = FakeRuntime(cleanup_success=False)
    environment = make_environment(runtime)

    with pytest.raises(RuntimeError, match='Verifier env cleanup failed'):
        await environment.exec(
            command='python /tests/judge.py',
            env={'OPENAI_API_KEY': 'super-secret-value'},
        )


@pytest.mark.asyncio
async def test_upload_file_uses_native_api_and_checks_result(tmp_path):
    source = tmp_path / 'empty.txt'
    source.write_bytes(b'')
    runtime = FakeRuntime()
    environment = make_environment(runtime)

    await environment.upload_file(source, '/tests/empty.txt')

    assert runtime.uploads[0].path == '/tests/empty.txt'
    assert runtime.uploads[0].content == ''
    assert runtime.commands == []

    failing_environment = make_environment(FakeRuntime(upload_success=False))
    with pytest.raises(RuntimeError, match='Failed to upload file'):
        await failing_environment.upload_file(source, '/tests/empty.txt')


@pytest.mark.asyncio
async def test_upload_dir_uses_unique_archive_and_checks_extraction(tmp_path):
    source = tmp_path / 'tests'
    source.mkdir()
    (source / 'test.sh').write_text('#!/bin/sh\n')
    (source / 'empty-dir').mkdir()
    runtime = FakeRuntime()
    environment = make_environment(runtime)

    await environment.upload_dir(source, '/tests-a')
    await environment.upload_dir(source, '/tests-b')

    archive_paths = [upload.path for upload in runtime.uploads]
    assert all(path.startswith('/tmp/.harbor_upload_') for path in archive_paths)
    assert len(set(archive_paths)) == 2
    archive_bytes = base64.b64decode(runtime.uploads[0].content)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as archive:
        assert './empty-dir' in archive.getnames()
    assert any('tar xzf' in command and '/tests-a' in command for command, _ in runtime.commands)
    assert any('tar xzf' in command and '/tests-b' in command for command, _ in runtime.commands)

    failing_runtime = FakeRuntime(command_failures={'tar xzf': 'simulated extraction failure'})
    failing_environment = make_environment(failing_runtime)
    with pytest.raises(RuntimeError, match='Extract directory archive'):
        await failing_environment.upload_dir(source, '/tests')
    assert failing_runtime.commands[-1][0].startswith('rm -f /tmp/.harbor_upload_')


@pytest.mark.asyncio
async def test_download_file_uses_native_api_for_binary_content(tmp_path):
    raw = b'\x00\xffbinary'
    runtime = FakeDownloadRuntime(download_content=base64.b64encode(raw).decode('ascii'))
    environment = make_environment(runtime)
    target = tmp_path / 'result.bin'

    await environment.download_file('/logs/verifier/result.bin', target)

    assert target.read_bytes() == raw
    assert runtime.downloads[0].path == '/logs/verifier/result.bin'
    assert runtime.downloads[0].encoding == 'base64'
    assert runtime.commands == []


@pytest.mark.asyncio
async def test_download_file_rejects_runtime_error_and_invalid_base64(tmp_path):
    failing_environment = make_environment(
        FakeDownloadRuntime(download_content='', download_error='simulated download failure')
    )
    with pytest.raises(RuntimeError, match='simulated download failure'):
        await failing_environment.download_file('/remote/file', tmp_path / 'failed')

    invalid_environment = make_environment(FakeDownloadRuntime(download_content='not-base64!'))
    with pytest.raises(RuntimeError, match='invalid base64 content'):
        await invalid_environment.download_file('/remote/file', tmp_path / 'invalid')


@pytest.mark.asyncio
async def test_download_file_tail_materializes_only_remote_preview(tmp_path):
    runtime = FakeDownloadRuntime(download_content=base64.b64encode(b'last').decode('ascii'))
    environment = make_environment(runtime)
    target = tmp_path / 'test-stdout-tail.txt'

    await environment.download_file_tail(
        '/logs/verifier/test-stdout.txt',
        target,
        max_bytes=4,
    )

    assert target.read_bytes() == b'last'
    assert runtime.commands[0][0].startswith('tail -c 4 /logs/verifier/test-stdout.txt > ')
    assert runtime.downloads[0].path.startswith('/tmp/.harbor_tail_')
    assert runtime.commands[-1][0].startswith('rm -f /tmp/.harbor_tail_')


@pytest.mark.asyncio
async def test_download_dir_uses_native_api_and_extracts_archive(tmp_path):
    archive_buffer = io.BytesIO()
    payload = b'reward output'
    with tarfile.open(fileobj=archive_buffer, mode='w:gz') as archive:
        info = tarfile.TarInfo('reward.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    runtime = FakeDownloadRuntime(download_content=base64.b64encode(archive_buffer.getvalue()).decode('ascii'))
    environment = make_environment(runtime)
    target = tmp_path / 'verifier'

    await environment.download_dir('/logs/verifier', target)

    assert (target / 'reward.json').read_bytes() == payload
    assert runtime.downloads[0].path.startswith('/tmp/.harbor_download_')
    assert not any(command.startswith('base64 ') for command, _ in runtime.commands)


@pytest.mark.asyncio
async def test_directory_transfer_archive_work_runs_off_event_loop_thread(tmp_path, monkeypatch):
    event_loop_thread = threading.get_ident()
    worker_threads: dict[str, int] = {}

    def record_thread(name):
        function = getattr(environment_module, name)

        def wrapper(*args):
            worker_threads[name] = threading.get_ident()
            return function(*args)

        monkeypatch.setattr(environment_module, name, wrapper)

    for name in ('_build_directory_archive_item', '_decode_base64_file_content', '_extract_directory_archive'):
        record_thread(name)

    source = tmp_path / 'tests'
    (source / 'data').mkdir(parents=True)
    (source / 'data' / 'truth.npy').write_bytes(b'\x93NUMPY\x00\xff')
    upload_runtime = FakeRuntime()
    await make_environment(upload_runtime).upload_dir(source, '/tests')

    download_runtime = FakeDownloadRuntime(download_content=upload_runtime.uploads[0].content)
    target = tmp_path / 'downloaded'
    await make_environment(download_runtime).download_dir('/tests', target)

    assert (target / 'data' / 'truth.npy').read_bytes() == b'\x93NUMPY\x00\xff'
    assert set(worker_threads) == {
        '_build_directory_archive_item',
        '_decode_base64_file_content',
        '_extract_directory_archive',
    }
    assert event_loop_thread not in worker_threads.values()


@pytest.mark.asyncio
async def test_download_dir_rejects_empty_archive_and_cleans_up(tmp_path):
    runtime = FakeDownloadRuntime(download_content='')
    environment = make_environment(runtime)

    with pytest.raises(RuntimeError, match='directory archive.*is empty'):
        await environment.download_dir('/logs/verifier', tmp_path / 'download')

    assert runtime.commands[-1][0].startswith('rm -f /tmp/.harbor_download_')


@pytest.mark.asyncio
async def test_download_dir_preserves_archive_failure_and_cleans_up(tmp_path):
    runtime = FakeRuntime(command_failures={'tar czf': 'missing source directory'})
    environment = make_environment(runtime)

    with pytest.raises(RuntimeError, match='Archive remote directory'):
        await environment.download_dir('/missing', tmp_path / 'download')

    assert runtime.commands[-1][0].startswith('rm -f /tmp/.harbor_download_')
