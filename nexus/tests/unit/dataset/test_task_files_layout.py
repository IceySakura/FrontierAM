from pathlib import Path

import pytest

from nexus.core.context import ExecutionContext
from nexus.dataset.auth import TaiPatAuthenticator, WedataKey
from nexus.dataset.client import DatasetClient
from nexus.dataset.exceptions import DatasetConfigurationError
from nexus.dataset.item import DatasetItem, RemoteDatasetRef, TaskFilesSpec
from nexus.dataset.layout import create_dataset_template, load_local_dataset
from nexus.dataset.task_files import RuntimeTaskFiles, task_files_from_spec
from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.runtime.context import RuntimeControl, set_runtime_control


@pytest.mark.asyncio
async def test_create_terminal_template_uses_task_files(tmp_path):
    root = create_dataset_template(tmp_path / 'my_terminal')

    dataset = load_local_dataset(root)
    items = dataset.load_items()

    assert (root / 'dataset.yaml').exists()
    assert dataset.raw['authors'] == []
    assert dataset.raw['maintainers'] == []
    assert [item.id for item in items] == ['case-001', 'case-002', 'case-003']
    item = items[0]
    assert item.id == 'case-001'
    assert item.files is not None
    assert item.files.type == 'directory'
    assert item.body['task_config']['message'] == 'Solve the task described in task_files/instruction.md.'
    assert 'task_dir' not in item.body['task_config']
    assert item.metadata == {}

    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None
    assert await files.read_text('instruction.md') == (
        'Solve the task in this directory. Add tests or solution files as needed.\n'
    )


@pytest.mark.asyncio
async def test_local_task_files_download_glob_strips_literal_prefix(tmp_path):
    root = create_dataset_template(tmp_path / 'my_terminal', template='terminal', item_id='case-001')
    test_file = root / 'items' / 'case-001' / 'files' / 'tests' / 'test.sh'
    test_file.write_text('echo ok\n', encoding='utf-8')
    item = load_local_dataset(root).load_items()[0]
    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None

    target = tmp_path / 'out'
    await files.download('tests/**', target)

    assert (target / 'test.sh').read_text(encoding='utf-8') == 'echo ok\n'


@pytest.mark.asyncio
async def test_local_directory_task_files_list_recursive_pattern_includes_root_files(tmp_path):
    root = tmp_path / 'files'
    (root / 'nested').mkdir(parents=True)
    (root / 'marker.txt').write_text('root\n', encoding='utf-8')
    (root / 'nested' / 'child.txt').write_text('nested\n', encoding='utf-8')

    files = task_files_from_spec({'type': 'directory', 'path': str(root)})
    assert files is not None

    assert await files.list() == ['marker.txt', 'nested/child.txt']
    assert await files.list('**/*.txt') == ['marker.txt', 'nested/child.txt']


@pytest.mark.asyncio
async def test_local_zip_task_files_list_recursive_pattern_includes_root_files(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'marker.txt').write_text('root\n', encoding='utf-8')
    archive = tmp_path / 'files.zip'
    write_nexus_zip_directory_to_file(source, archive)

    files = task_files_from_spec({'type': 'zip', 'path': str(archive)})
    assert files is not None

    assert await files.list() == ['marker.txt']
    assert await files.list('**/*.txt') == ['marker.txt']


@pytest.mark.asyncio
async def test_runtime_task_files_passive_session_request_downloads_local_files(tmp_path):
    session = {
        'task_files': {'type': 'directory', 'path': '/host/files'},
        'run_id': 'run-1',
    }
    sio = _FakeTaskFilesSIO(
        session=session,
        files={'tests/test.sh': b'echo passive\n'},
    )
    runtime_files = RuntimeTaskFiles(sio=sio)

    target = tmp_path / 'runtime_tests'
    await runtime_files.download('tests/**', target)

    assert (target / 'test.sh').read_text(encoding='utf-8') == 'echo passive\n'
    assert [item['event'] for item in sio.emitted] == [
        'task_files:session_request',
        'task_files:local_file_request',
        'task_files:local_file_request',
    ]
    assert [item['data'].get('op') for item in sio.emitted if item['event'] == 'task_files:local_file_request'] == [
        'list',
        'download',
    ]


@pytest.mark.asyncio
async def test_runtime_task_files_read_bytes_materializes_local_file():
    session = {
        'task_files': {'type': 'directory', 'path': '/host/files'},
        'run_id': 'run-1',
    }
    sio = _FakeTaskFilesSIO(
        session=session,
        files={'tests/test.sh': b'echo read\n'},
    )
    runtime_files = RuntimeTaskFiles(sio=sio)

    assert await runtime_files.read_bytes('tests/test.sh') == b'echo read\n'
    assert [item['data'].get('op') for item in sio.emitted if item['event'] == 'task_files:local_file_request'] == [
        'materialize'
    ]


@pytest.mark.asyncio
async def test_host_commanded_runtime_download_reuses_runtime_task_files(tmp_path):
    root = create_dataset_template(tmp_path / 'my_terminal', template='terminal', item_id='case-001')
    test_file = root / 'items' / 'case-001' / 'files' / 'tests' / 'test.sh'
    test_file.write_text('echo active\n', encoding='utf-8')
    item = load_local_dataset(root).load_items()[0]
    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None
    runtime = _FakeTaskFilesRuntime({'tests/test.sh': b'echo active\n'})

    target = tmp_path / 'active_tests'
    await files.download('tests/**', target, runtime=runtime)

    assert runtime.invoked_target == 'nexus.dataset.task_files::runtime_download_task_files'
    assert (target / 'test.sh').read_text(encoding='utf-8') == 'echo active\n'


@pytest.mark.asyncio
async def test_host_local_file_request_uses_runtime_invoke_transfer(tmp_path):
    root = create_dataset_template(tmp_path / 'my_terminal', template='terminal', item_id='case-001')
    test_file = root / 'items' / 'case-001' / 'files' / 'tests' / 'test.sh'
    test_file.write_text('echo bridge\n', encoding='utf-8')
    item = load_local_dataset(root).load_items()[0]
    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None
    runtime = _FakeBridgeRuntime()
    ctx = ExecutionContext(task_files=files, run_id='run-1')
    ctx._attach_task_files_runtime(runtime)

    target = tmp_path / 'runtime' / 'test.sh'
    await runtime.handlers['task_files:local_file_request'](
        {
            'request_id': 'request-1',
            'op': 'download',
            'path': 'tests/test.sh',
            'target': str(target),
        }
    )

    assert runtime.invoked_target == 'nexus.dataset.task_files::runtime_receive_task_file'
    assert target.read_text(encoding='utf-8') == 'echo bridge\n'
    assert runtime.emitted == [
        {
            'event': 'task_files:local_file_response:request-1',
            'data': {'ok': True, 'path': str(target), 'size': len(b'echo bridge\n')},
        }
    ]


@pytest.mark.asyncio
async def test_remote_task_files_session_includes_auth_when_key_available(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', lambda: None)
    monkeypatch.setattr(
        task_files_module,
        'load_wedata_key',
        lambda: WedataKey(subject='alice', key='cmk', key_id='kid'),
    )
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })
    assert files is not None

    session = await files.create_session(run_id='run-1')

    assert session.auth == {
        'kind': 'wedata_key',
        'key': {'subject': 'alice', 'key': 'cmk', 'id': 'kid'},
    }


@pytest.mark.asyncio
async def test_remote_task_files_session_omits_auth_when_key_missing(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    def raise_missing_key():
        raise DatasetConfigurationError('missing key')

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', lambda: None)
    monkeypatch.setattr(task_files_module, 'load_wedata_key', raise_missing_key)
    monkeypatch.setattr(task_files_module, 'wedata_key_available', lambda: False)
    monkeypatch.setattr(task_files_module, 'default_taihu_pat', lambda: None)
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })
    assert files is not None

    session = await files.create_session(run_id='run-1')

    assert session.auth is None


@pytest.mark.asyncio
async def test_remote_task_files_session_uses_tai_pat_when_only_default_pat_file_available(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    def raise_missing_key():
        raise DatasetConfigurationError('missing key')

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', lambda: None)
    monkeypatch.setattr(task_files_module, 'load_wedata_key', raise_missing_key)
    monkeypatch.setattr(task_files_module, 'wedata_key_available', lambda: False)
    monkeypatch.setattr(task_files_module, 'default_taihu_pat', lambda: 'tai_pat_secret')
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })
    assert files is not None

    session = await files.create_session(run_id='run-1')

    assert session.auth == {'kind': 'tai_pat', 'pat': 'tai_pat_secret'}


@pytest.mark.asyncio
async def test_remote_task_files_session_prefers_wedata_over_default_pat_file(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', lambda: None)
    monkeypatch.setattr(
        task_files_module,
        'load_wedata_key',
        lambda: WedataKey(subject='alice', key='cmk', key_id='kid'),
    )
    monkeypatch.setattr(task_files_module, 'default_taihu_pat', lambda: 'tai_pat_secret')
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })
    assert files is not None

    session = await files.create_session(run_id='run-1')

    assert session.auth == {
        'kind': 'wedata_key',
        'key': {'subject': 'alice', 'key': 'cmk', 'id': 'kid'},
    }


@pytest.mark.asyncio
async def test_remote_task_files_session_prefers_tai_pat_env_over_wedata(monkeypatch):
    """TAI_PAT env must win over a (possibly stale) Wedata key, matching
    build_default_authenticator()'s precedence."""

    import nexus.dataset.task_files as task_files_module

    def fail_wedata():
        raise AssertionError('TAI_PAT env must take precedence over the Wedata key')

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', lambda: 'env_pat_secret')
    monkeypatch.setattr(task_files_module, 'load_wedata_key', fail_wedata)
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })
    assert files is not None

    session = await files.create_session(run_id='run-1')

    assert session.auth == {'kind': 'tai_pat', 'pat': 'env_pat_secret'}


@pytest.mark.asyncio
async def test_task_files_for_item_reuses_explicit_client_authenticator(monkeypatch):
    """An explicit DatasetClient authenticator must flow into task_files signing
    instead of re-reading process-global config (e.g. a leftover Wedata key)."""

    import nexus.dataset.task_files as task_files_module

    def fail_global_auth():
        raise AssertionError('explicit authenticator must not fall back to global auth config')

    monkeypatch.setattr(task_files_module, 'tai_pat_from_env', fail_global_auth)
    monkeypatch.setattr(task_files_module, 'load_wedata_key', fail_global_auth)

    item = DatasetItem(
        id='case-001',
        body={},
        files=TaskFilesSpec(type='zip', path='blobs/case-001.zip', size=123),
        remote_ref=RemoteDatasetRef(namespace='ns', dataset='ds', version='v1', split='test'),
    )
    client = DatasetClient(authenticator=TaiPatAuthenticator(pat='explicit-pat'))

    files = client.task_files_for_item(item, run_id='run-1')
    assert files is not None

    session = await files.create_session()

    assert session.auth == {'kind': 'tai_pat', 'pat': 'explicit-pat'}


@pytest.mark.asyncio
async def test_remote_task_files_sign_uses_tai_pat_authenticator(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    calls = []

    class FakeIoClient:
        def __init__(self, *, session, authenticator=None, allow_anonymous=False):
            calls.append({'authenticator': authenticator, 'allow_anonymous': allow_anonymous})

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            return {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    spec = {
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    }
    auth = {'kind': 'tai_pat', 'pat': 'tai_pat_secret'}

    signed = await task_files_module._sign_remote_task_files(
        task_files_module.TaskFilesSpec.model_validate(spec),
        auth=auth,
    )

    assert signed == {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}
    assert isinstance(calls[0]['authenticator'], TaiPatAuthenticator)
    assert calls[0]['authenticator'].pat == 'tai_pat_secret'
    assert calls[0]['allow_anonymous'] is False


@pytest.mark.asyncio
async def test_runtime_remote_task_files_sign_uses_tai_pat_session_auth(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    calls = []

    class FakeIoClient:
        def __init__(self, *, session, authenticator=None, allow_anonymous=False):
            calls.append({'authenticator': authenticator, 'allow_anonymous': allow_anonymous})

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            return {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    session = task_files_module.TaskFilesSession.model_validate({
        'run_id': 'run-1',
        'task_files': {
            'type': 'zip',
            'path': 'blobs/case-001.zip',
            'size': 123,
            'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
        },
        'auth': {'kind': 'tai_pat', 'pat': 'tai_pat_secret'},
    })
    delegate = task_files_module._RuntimeRemoteZipTaskFiles(session)

    signed_url = await delegate._get_signed_url()

    assert signed_url == 'https://cos/signed'
    assert isinstance(calls[0]['authenticator'], TaiPatAuthenticator)
    assert calls[0]['authenticator'].pat == 'tai_pat_secret'
    assert calls[0]['allow_anonymous'] is False


@pytest.mark.asyncio
async def test_remote_task_files_sign_uses_auth_when_present(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    calls = []

    class FakeIoClient:
        def __init__(self, *, session, authenticator=None, allow_anonymous=False):
            calls.append({'authenticator': authenticator, 'allow_anonymous': allow_anonymous})

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            return {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    spec = {
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    }
    auth = {
        'kind': 'wedata_key',
        'key': {'subject': 'alice', 'key': 'cmk', 'id': 'kid'},
    }

    signed = await task_files_module._sign_remote_task_files(
        task_files_module.TaskFilesSpec.model_validate(spec),
        auth=auth,
    )

    assert signed == {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}
    assert calls[0]['authenticator'] is not None
    assert calls[0]['allow_anonymous'] is False


@pytest.mark.asyncio
async def test_remote_task_files_sign_uses_anonymous_when_auth_missing(monkeypatch):
    import nexus.dataset.task_files as task_files_module

    calls = []

    class FakeIoClient:
        def __init__(self, *, session, authenticator=None, allow_anonymous=False):
            calls.append({'authenticator': authenticator, 'allow_anonymous': allow_anonymous})

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            return {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    spec = {
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    }

    signed = await task_files_module._sign_remote_task_files(
        task_files_module.TaskFilesSpec.model_validate(spec),
        auth=None,
    )

    assert signed == {'urls': {'blobs/case-001.zip': 'https://cos/signed'}}
    assert calls[0]['authenticator'] is None
    assert calls[0]['allow_anonymous'] is True


class _FakeTaskFilesSIO:
    def __init__(self, *, session: dict | None = None, files: dict[str, bytes] | None = None) -> None:
        self.session = session
        self.files = files or {}
        self.handlers = {}
        self.emitted = []

    def on(self, event, handler):
        self.handlers[event] = handler

    async def emit(self, event, data=None):
        self.emitted.append({'event': event, 'data': data})
        if event == 'task_files:session_request':
            request_id = data['request_id']
            await self.handlers[f'task_files:session_response:{request_id}']({'ok': True, 'session': self.session})
            return
        if event == 'task_files:local_file_request':
            request_id = data['request_id']
            if data['op'] == 'list':
                pattern = data['pattern']
                paths = sorted(path for path in self.files if path.startswith(pattern.split('*', 1)[0]))
                await self.handlers[f'task_files:local_file_response:{request_id}']({'ok': True, 'paths': paths})
                return
            if data['op'] in {'materialize', 'download'}:
                content = self.files[data['path']]
                target = Path(data['target'])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                await self.handlers[f'task_files:local_file_response:{request_id}'](
                    {
                        'ok': True,
                        'path': str(target),
                        'size': len(content),
                    }
                )
                return
        raise AssertionError(f'unhandled fake SIO event: {event}')


class _FakeTaskFilesRuntime:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.invoked_target = None

    async def invoke(self, target, *args, **_kwargs):
        from nexus.dataset.task_files import runtime_download_task_files
        from nexus.runtime.context import _runtime_control

        self.invoked_target = target
        assert target == 'nexus.dataset.task_files::runtime_download_task_files'
        ctrl = RuntimeControl(sio=_FakeTaskFilesSIO(files=self.files))
        token = set_runtime_control(ctrl)
        try:
            return await runtime_download_task_files(*args)
        finally:
            _runtime_control.reset(token)


class _FakeBridgeRuntime:
    def __init__(self) -> None:
        self.handlers = {}
        self.emitted = []
        self.invoked_target = None

    def on(self, event, handler):
        self.handlers[event] = handler

    async def emit(self, event, data=None):
        self.emitted.append({'event': event, 'data': data})

    async def invoke(self, target, *args, **_kwargs):
        from nexus.dataset.task_files import runtime_receive_task_file

        self.invoked_target = target
        assert target == 'nexus.dataset.task_files::runtime_receive_task_file'
        return await runtime_receive_task_file(*args)
