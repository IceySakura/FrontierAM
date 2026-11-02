"""Unit tests for :mod:`nexus.utils.artifact.client`."""

import base64
import hashlib
import inspect

import pytest

from nexus.utils.artifact.auth import ARTIFACT_SHARED_VALUE_NAME, ArtifactCredentials
from nexus.utils.artifact.client import (
    ArtifactClient,
    _drop_none,
    _guess_content_type,
    _iso,
    _needs_idempotency,
    _parse_result,
)
from nexus.utils.artifact.errors import ArtifactClientError, ArtifactUploadError
from nexus.utils.artifact.models import ArtifactInfo, DownloadUrl, UploadSession
from nexus.utils.cos import CosSourceChangedError, iter_file_segment


@pytest.fixture(autouse=True)
def _clear_artifact_identity_environment(monkeypatch, tmp_path):
    monkeypatch.delenv('ARTIFACT_TAIHU_PAT', raising=False)
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    monkeypatch.delenv('ARTIFACT_TAIHU_PAT_FILE', raising=False)
    monkeypatch.delenv('ARTIFACT_SERVICE_URL', raising=False)
    # Isolate the on-disk PAT fallback from the host's ~/.nexus/auth/taihu.
    from nexus.utils.artifact import auth as auth_module

    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', tmp_path / 'no-such-key')


# --- pure helpers -----------------------------------------------------------


def test_needs_idempotency():
    assert _needs_idempotency('POST', '/v1/artifact-uploads')
    assert _needs_idempotency('POST', '/v1/folders')
    assert _needs_idempotency('PUT', '/v1/resources/r1/grants/read/bob')
    assert not _needs_idempotency('POST', '/v1/artifact-uploads/u1/complete')
    assert not _needs_idempotency('DELETE', '/v1/folders/f1')
    assert not _needs_idempotency('GET', '/v1/resources')


def test_drop_none_and_iso_and_content_type():
    assert _drop_none({'a': 1, 'b': None, 'c': 0}) == {'a': 1, 'c': 0}
    assert _iso(None) is None
    assert _iso('2026-01-01') == '2026-01-01'
    from datetime import datetime

    assert _iso(datetime(2026, 1, 2, 3, 4, 5)) == '2026-01-02T03:04:05'
    assert _guess_content_type('a.json') == 'application/json'
    assert _guess_content_type('blob') == 'application/octet-stream'


# --- _parse_result ----------------------------------------------------------


class FakeResponse:
    def __init__(self, *, status=200, json_body=None, text_body='', raise_json=False):
        self.status = status
        self._json = json_body
        self._text = text_body
        self._raise = raise_json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def json(self, content_type=None):
        if self._raise:
            raise ValueError('no json')
        return self._json

    async def text(self):
        return self._text

    async def read(self):
        return b''


async def test_parse_result_204_returns_empty():
    assert await _parse_result(FakeResponse(status=204)) == {}


async def test_parse_result_success_dict():
    assert await _parse_result(FakeResponse(json_body={'folder_id': 'fld_1'})) == {'folder_id': 'fld_1'}


async def test_parse_result_error_envelope():
    response = FakeResponse(
        status=409,
        json_body={'error': {'code': 'IdempotencyConflict', 'message': 'boom', 'request_id': 'req-1'}},
    )
    with pytest.raises(ArtifactClientError) as excinfo:
        await _parse_result(response)
    assert excinfo.value.code == 'IdempotencyConflict'
    assert excinfo.value.request_id == 'req-1'
    assert excinfo.value.http_status == 409


async def test_parse_result_non_json_error():
    with pytest.raises(ArtifactClientError, match='HTTP 500'):
        await _parse_result(FakeResponse(status=500, text_body='oops', raise_json=True))


# --- transport (_request) ---------------------------------------------------


class RecordingSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def request(self, method, url, params=None, json=None, headers=None):
        self.requests.append({'method': method, 'url': url, 'params': params, 'json': json, 'headers': headers})
        return self.response


async def test_from_runtime_requests_host_credentials_and_preserves_options(monkeypatch):
    from nexus.runtime import context as runtime_context

    requests = []

    class FakeRuntimeControl:
        async def get_shared_value(self, name, expected_type, *, decode):
            requests.append((name, expected_type))
            return decode({'taihu_pat': 'Bearer tai_pat_host'})

    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: FakeRuntimeControl())

    client = await ArtifactClient.from_runtime(
        base_url='https://artifact.example/',
        timeout=12,
        transfer_timeout=34,
    )
    headers: dict[str, str] = {}
    await client._apply_auth_headers(headers)

    assert requests == [(ARTIFACT_SHARED_VALUE_NAME, ArtifactCredentials)]
    assert client.base_url == 'https://artifact.example'
    assert client.timeout == 12
    assert client.transfer_timeout == 34
    assert headers == {'Authorization': 'Bearer tai_pat_host'}


async def test_from_runtime_does_not_fall_back_to_sandbox_credentials(monkeypatch):
    from nexus.runtime import context as runtime_context

    class FakeRuntimeControl:
        async def get_shared_value(self, name, expected_type, *, decode):
            return decode({})

    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: FakeRuntimeControl())
    monkeypatch.setenv('ARTIFACT_TAIHU_PAT', 'tai_pat_sandbox')

    client = await ArtifactClient.from_runtime(base_url='http://localhost:8080')
    headers: dict[str, str] = {}
    await client._apply_auth_headers(headers)

    assert headers == {}


async def test_from_runtime_without_host_pat_preserves_explicit_dev_identity(monkeypatch):
    from nexus.runtime import context as runtime_context

    class FakeRuntimeControl:
        async def get_shared_value(self, name, expected_type, *, decode):
            return decode({})

    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: FakeRuntimeControl())

    client = await ArtifactClient.from_runtime(dev_employee='sandbox-user')
    headers: dict[str, str] = {}
    await client._apply_auth_headers(headers)

    assert headers == {'X-Dev-Employee': 'sandbox-user'}


async def test_from_runtime_requires_service_url_when_host_pat_is_present(monkeypatch):
    from nexus.runtime import context as runtime_context

    class FakeRuntimeControl:
        async def get_shared_value(self, name, expected_type, *, decode):
            return decode({'taihu_pat': 'tai_pat_host'})

    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: FakeRuntimeControl())

    with pytest.raises(ValueError, match='base_url'):
        await ArtifactClient.from_runtime()


async def test_request_sets_auth_content_type_and_idempotency():
    client = ArtifactClient(base_url='http://host:8080/', taihu_pat='tai_pat_xyz')
    client.session = RecordingSession(FakeResponse(json_body={'folder_id': 'fld_1'}))

    out = await client._request('POST', '/v1/folders', json_body={'name': 'x'})

    assert out == {'folder_id': 'fld_1'}
    req = client.session.requests[0]
    assert req['url'] == 'http://host:8080/v1/folders'
    assert req['headers']['Authorization'] == 'Bearer tai_pat_xyz'
    assert req['headers']['Content-Type'] == 'application/json'
    assert 'Idempotency-Key' in req['headers']  # auto-generated for POST /v1/folders


async def test_request_explicit_idempotency_and_get_without_key():
    client = ArtifactClient(base_url='http://host')
    client.session = RecordingSession(FakeResponse(json_body={'items': []}))

    await client._request('GET', '/v1/resources', params={'limit': 10})
    req = client.session.requests[0]
    assert 'Idempotency-Key' not in req['headers']
    assert 'Content-Type' not in req['headers']
    assert req['params'] == {'limit': 10}


class _SequencedResponseSession:
    """Return a scripted sequence of FakeResponses across successive requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def request(self, method, url, params=None, json=None, headers=None):
        self.requests.append({'method': method, 'url': url, 'params': params, 'json': json, 'headers': headers})
        return self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]


async def test_request_retries_transient_gateway_502_then_succeeds(monkeypatch):
    from nexus.utils.artifact import client as client_module

    monkeypatch.setattr(client_module, 'DEFAULT_REQUEST_RETRY_BACKOFF', 0.0)
    client = ArtifactClient(base_url='http://host', taihu_pat='tai_pat_xyz')
    # Two 502 "socket hang up" gateway errors, then a success.
    client.session = _SequencedResponseSession([
        FakeResponse(status=502, text_body='socket hang up', raise_json=True),
        FakeResponse(status=502, text_body='socket hang up', raise_json=True),
        FakeResponse(json_body={'folder_id': 'fld_1'}),
    ])

    out = await client._request('POST', '/v1/folders', json_body={'name': 'x'})

    assert out == {'folder_id': 'fld_1'}
    assert len(client.session.requests) == 3  # two retries, third succeeds
    # The Idempotency-Key must be stable across retries to stay idempotent.
    keys = {req['headers']['Idempotency-Key'] for req in client.session.requests}
    assert len(keys) == 1


async def test_request_does_not_retry_client_4xx(monkeypatch):
    from nexus.utils.artifact import client as client_module

    monkeypatch.setattr(client_module, 'DEFAULT_REQUEST_RETRY_BACKOFF', 0.0)
    client = ArtifactClient(base_url='http://host', taihu_pat='tai_pat_xyz')
    client.session = _SequencedResponseSession([
        FakeResponse(status=409, json_body={'error': {'code': 'IdempotencyConflict', 'message': 'boom'}}),
        FakeResponse(json_body={'folder_id': 'should-not-be-reached'}),
    ])

    with pytest.raises(ArtifactClientError) as excinfo:
        await client._request('POST', '/v1/folders', json_body={'name': 'x'})

    assert excinfo.value.http_status == 409
    assert len(client.session.requests) == 1  # not retried


async def test_request_exhausts_gateway_retries_and_raises(monkeypatch):
    from nexus.utils.artifact import client as client_module

    monkeypatch.setattr(client_module, 'DEFAULT_REQUEST_RETRY_BACKOFF', 0.0)
    attempts = client_module.DEFAULT_REQUEST_MAX_RETRIES + 1
    client = ArtifactClient(base_url='http://host', taihu_pat='tai_pat_xyz')
    client.session = _SequencedResponseSession(
        [FakeResponse(status=503, text_body='unavailable', raise_json=True)] * attempts
    )

    with pytest.raises(ArtifactClientError) as excinfo:
        await client._request('GET', '/v1/resources')

    assert excinfo.value.http_status == 503
    assert len(client.session.requests) == attempts


# --- high-level method delegation -------------------------------------------


def _client_with_recorder():
    client = ArtifactClient()
    calls = []

    async def fake_request(method, path, *, params=None, json_body=None, idempotency_key=None):
        calls.append((method, path, params, json_body, idempotency_key))
        return fake_request.result

    fake_request.result = {}
    client._request = fake_request
    return client, calls, fake_request


async def test_create_folder_builds_request():
    client, calls, fake = _client_with_recorder()
    fake.result = {'folder_id': 'fld_1', 'name': 'x'}
    folder = await client.create_folder('x', parent_folder_id='fld_p', idempotency_key='k1')
    assert folder.folder_id == 'fld_1'
    assert calls == [('POST', '/v1/folders', None, {'name': 'x', 'parent_folder_id': 'fld_p'}, 'k1')]


async def test_list_resources_drops_none_params():
    client, calls, fake = _client_with_recorder()
    fake.result = {'items': [{'resource_id': 'r1'}], 'next_cursor': 'c2'}
    page = await client.list_resources(parent_folder_id='fld_1', limit=25)
    assert page.next_cursor == 'c2'
    assert calls[0] == ('GET', '/v1/resources', {'parent_folder_id': 'fld_1', 'limit': 25}, None, None)


async def test_put_grant_builds_employee_read_path_and_drops_expiry():
    client, calls, fake = _client_with_recorder()
    await client.put_grant('res_1', 'alice', grantee_type='employee', permission='read', idempotency_key='k')
    assert calls == [('PUT', '/v1/resources/res_1/grants/read/alice', None, {}, 'k')]


async def test_put_grant_rejects_application_grantee():
    client, calls, fake = _client_with_recorder()
    with pytest.raises(ValueError, match='employee'):
        await client.put_grant('res_1', 'app_x', grantee_type='application')
    assert calls == []


async def test_delete_grant_builds_employee_read_path():
    client, calls, fake = _client_with_recorder()
    await client.delete_grant('res_1', 'alice', grantee_type='employee', permission='read')
    assert calls == [('DELETE', '/v1/resources/res_1/grants/read/alice', None, None, None)]


async def test_delete_grant_rejects_write_permission():
    client, calls, fake = _client_with_recorder()
    with pytest.raises(ValueError, match='read'):
        await client.delete_grant('res_1', 'alice', permission='write')
    assert calls == []


async def test_delete_grant_rejects_application_grantee():
    client, calls, fake = _client_with_recorder()
    with pytest.raises(ValueError, match='employee'):
        await client.delete_grant('res_1', 'app_x', grantee_type='application')
    assert calls == []


async def test_create_download_url_returns_model():
    client, calls, fake = _client_with_recorder()
    fake.result = {'method': 'GET', 'url': 'http://cos/get', 'expires_in': 900}
    download = await client.create_download_url('art_1')
    assert isinstance(download, DownloadUrl)
    assert download.url == 'http://cos/get'
    assert download.expires_in == 900
    assert calls == [('POST', '/v1/artifacts/art_1/download-url', None, None, None)]


async def test_list_grants_defaults_to_employee_grantee_type():
    client, calls, fake = _client_with_recorder()
    fake.result = {'items': [{'grantee_id': 'alice', 'permission': 'read'}]}
    grants = await client.list_grants('res_1')
    assert len(grants) == 1
    assert grants[0].grantee_type == 'employee'
    assert grants[0].grantee_id == 'alice'


# --- upload orchestration ---------------------------------------------------


def _patch_upload(client):
    captured = {}

    async def fake_create_upload(**kwargs):
        captured['create'] = kwargs
        return captured.get('session')

    async def fake_put(session_info, data):
        captured['put'] = (session_info.url, data)

    async def fake_put_file(session_info, path, *, source=None, file_size=None):
        captured['put_file'] = (session_info.url, path, source, file_size)

    async def fake_upload_parts(
        session_info,
        path,
        *,
        max_concurrency=5,
        source=None,
        file_size=None,
    ):
        captured['upload_parts'] = (
            list(session_info.part_urls),
            path,
            max_concurrency,
            source,
            file_size,
        )
        return [{'part_number': 1, 'etag': '"e1"'}]

    async def fake_upload_bytes_parts(session_info, data, *, max_concurrency=5):
        captured['upload_bytes_parts'] = (list(session_info.part_urls), data, max_concurrency)
        return [{'part_number': 1, 'etag': '"e1"'}]

    async def fake_complete(upload_id, parts=None):
        captured['complete'] = (upload_id, parts)
        return {}

    async def fake_get(artifact_id):
        captured['get'] = artifact_id
        return ArtifactInfo(artifact_id=artifact_id, name='n')

    client.create_upload = fake_create_upload
    client._put_to_cos = fake_put
    client._put_file_to_cos = fake_put_file
    client._upload_parts_to_cos = fake_upload_parts
    client._upload_bytes_parts_to_cos = fake_upload_bytes_parts
    client.complete_upload = fake_complete
    client.get_artifact = fake_get
    return captured


async def test_upload_bytes_full_flow():
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_1', upload_id='upl_1', status='pending', method='PUT', url='http://cos', headers={'h': 'v'}
    )

    info = await client.upload_bytes(b'hello', 'n.txt', parent_folder_id='fld', idempotency_key='k')

    assert info.artifact_id == 'art_1'
    expected_md5 = base64.b64encode(hashlib.md5(b'hello').digest()).decode('ascii')  # noqa: S324
    assert captured['create']['content_md5'] == expected_md5
    assert captured['create']['size_bytes'] == 5
    assert captured['create']['parent_folder_id'] == 'fld'
    assert captured['create']['idempotency_key'] == 'k'
    assert captured['put'] == ('http://cos', b'hello')
    assert captured['complete'] == ('upl_1', None)
    assert captured['get'] == 'art_1'


async def test_upload_bytes_dispatches_to_server_selected_multipart():
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/part1', 'http://cos/part2'],
        part_size=3,
        part_count=2,
    )

    info = await client.upload_bytes(b'abcdef', 'memory.bin')

    assert info.artifact_id == 'art_mp'
    assert 'put' not in captured
    assert captured['upload_bytes_parts'] == (
        ['http://cos/part1', 'http://cos/part2'],
        b'abcdef',
        5,
    )
    assert captured['complete'] == ('upl_mp', [{'part_number': 1, 'etag': '"e1"'}])


async def test_upload_bytes_ready_short_circuits():
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_1', upload_id=None, status='ready', method=None, url=None, headers={}
    )

    info = await client.upload_bytes(b'x', 'n.txt')

    assert info.artifact_id == 'art_1'
    assert 'put' not in captured  # no COS transfer for an already-ready (idempotent replay) artifact
    assert 'complete' not in captured
    assert captured['get'] == 'art_1'


async def test_upload_file_reads_and_completes(tmp_path):
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_2', upload_id='upl_2', status='pending', method='PUT', url='http://cos', headers={}
    )
    local = tmp_path / 'model.bin'
    local.write_bytes(b'abc')

    info = await client.upload_file(local, parent_folder_id='fld')

    assert info.artifact_id == 'art_2'
    assert captured['create']['filename'] == 'model.bin'
    assert captured['create']['size_bytes'] == 3
    assert captured['put_file'][0] == 'http://cos'
    assert captured['complete'] == ('upl_2', None)


async def test_upload_bytes_rejects_server_multipart_part_count_mismatch():
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/part1'],
        part_size=3,
        part_count=1,
    )
    del client._upload_bytes_parts_to_cos

    with pytest.raises(ArtifactUploadError, match='part count'):
        await client.upload_bytes(b'abcdef', 'memory.bin')


async def test_upload_file_rejects_source_replaced_after_hashing(tmp_path, monkeypatch):
    from nexus.utils.artifact import client as client_module

    client = ArtifactClient()
    captured = _patch_upload(client)
    local = tmp_path / 'changing.bin'
    local.write_bytes(b'original')
    original_md5 = client_module._open_file_md5_base64

    async def hash_then_replace(source):
        digest = await original_md5(source)
        replacement = tmp_path / 'replacement.bin'
        replacement.write_bytes(b'replaced')
        replacement.replace(local)
        return digest

    monkeypatch.setattr(client_module, '_open_file_md5_base64', hash_then_replace)

    with pytest.raises(ArtifactClientError, match='changed'):
        await client.upload_file(local)

    assert 'create' not in captured


async def test_upload_file_no_follow_rejects_symlink_leaf(tmp_path):
    """``follow_symlinks=False`` must refuse a symlinked path before any I/O.

    This is the client-side half of the publish symlink hardening: even if a
    symlink is swapped in at the leaf after a bulk caller scanned the tree, the
    O_NOFOLLOW open fails so the read is never redirected to the target.
    """
    secret = tmp_path / 'secret.txt'
    secret.write_bytes(b'TOP-SECRET-HOST-FILE')
    link = tmp_path / 'result'
    link.symlink_to(secret)

    client = ArtifactClient()
    # Fails at the O_NOFOLLOW open, before any session/network is touched.
    with pytest.raises(ArtifactUploadError, match='not a readable regular file'):
        await client.upload_file(link, follow_symlinks=False)


async def test_upload_file_no_follow_allows_regular_file(tmp_path, monkeypatch):
    """``follow_symlinks=False`` still uploads an ordinary (non-symlink) file."""
    client = ArtifactClient()
    captured = _patch_upload(client)
    # A ready session lets upload_file return right after create_upload without
    # needing the COS transfer legs mocked further.
    captured['session'] = UploadSession(
        artifact_id='art_ok', upload_id=None, status='ready', method=None, url=None, headers={}
    )
    local = tmp_path / 'plain.bin'
    local.write_bytes(b'payload')

    await client.upload_file(local, follow_symlinks=False)

    # The regular file was accepted and an upload session was created.
    assert 'create' in captured


async def test_iter_file_segment_rejects_premature_eof(tmp_path):
    # Part streaming now reuses nexus.utils.cos.iter_file_segment, which raises a
    # CosSourceChangedError if the file is shorter than the requested segment.
    local = tmp_path / 'truncated.bin'
    local.write_bytes(b'abc')

    with pytest.raises(CosSourceChangedError):
        async for _ in iter_file_segment(local, 0, 5):
            pass


async def test_upload_file_dispatches_to_multipart_when_signed(tmp_path):
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/part1', 'http://cos/part2'],
        part_size=1024 * 1024,
        part_count=2,
    )
    local = tmp_path / 'big.bin'
    local.write_bytes(b'abcdef')

    info = await client.upload_file(local, multipart=True, max_concurrency=8)

    assert info.artifact_id == 'art_mp'
    assert captured['create']['multipart'] is True
    # Single-PUT path must not be taken when the server signed a multipart session.
    assert 'put_file' not in captured
    part_urls, upload_path, concurrency, source, file_size = captured['upload_parts']
    assert part_urls == ['http://cos/part1', 'http://cos/part2']
    assert upload_path == local
    assert concurrency == 8
    assert source.closed is True
    assert file_size == 6
    # The collected part ETags are forwarded to the server-side completion.
    assert captured['complete'] == ('upl_mp', [{'part_number': 1, 'etag': '"e1"'}])


async def test_upload_file_without_multipart_hint_omits_body_field(tmp_path):
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_s', upload_id='upl_s', status='pending', method='PUT', url='http://cos', headers={}
    )
    local = tmp_path / 'small.bin'
    local.write_bytes(b'abc')

    await client.upload_file(local)

    # multipart=None must be dropped from the request body, not sent as null.
    assert captured['create']['multipart'] is None


# --- multipart part upload --------------------------------------------------


class _FakePutResponse:
    def __init__(self, etag='"etag"', status=200, data=None, record=None):
        self._etag = etag
        self.status = status
        self.headers = {'ETag': etag} if etag is not None else {}
        self._data = data
        self._record = record

    async def __aenter__(self):
        if hasattr(self._data, '__aiter__'):
            chunks = [chunk async for chunk in self._data]
            self._record['body'] = b''.join(chunks)
        return self

    async def __aexit__(self, *exc):
        return None

    async def text(self):
        return 'error'


class _RecordingPartSession:
    def __init__(self, etags):
        self._etags = etags
        self.puts = []

    def put(self, url, data=None, headers=None, timeout=None):
        record = {'url': url, 'headers': headers, 'body': None}
        self.puts.append(record)
        index = len(self.puts) - 1
        return _FakePutResponse(etag=self._etags[index], data=data, record=record)


async def test_upload_parts_streams_each_part_and_collects_etags(tmp_path):
    client = ArtifactClient()
    client.session = _RecordingPartSession(['"e1"', '"e2"', '"e3"'])
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/p1', 'http://cos/p2', 'http://cos/p3'],
        part_size=4,
        part_count=3,
    )
    local = tmp_path / 'f.bin'
    local.write_bytes(b'0123456789')  # 10 bytes -> parts of 4, 4, 2

    parts = await client._upload_parts_to_cos(session_info, local, max_concurrency=2)

    assert parts == [
        {'part_number': 1, 'etag': '"e1"'},
        {'part_number': 2, 'etag': '"e2"'},
        {'part_number': 3, 'etag': '"e3"'},
    ]
    assert len(client.session.puts) == 3
    assert client.session.puts[0]['headers']['Content-Length'] == '4'
    assert client.session.puts[2]['headers']['Content-Length'] == '2'
    assert [put['body'] for put in client.session.puts] == [b'0123', b'4567', b'89']


async def test_upload_bytes_parts_streams_expected_bodies_and_collects_etags():
    client = ArtifactClient()
    client.session = _RecordingPartSession(['"e1"', '"e2"', '"e3"'])
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/p1', 'http://cos/p2', 'http://cos/p3'],
        part_size=4,
        part_count=3,
    )

    parts = await client._upload_bytes_parts_to_cos(session_info, b'0123456789', max_concurrency=2)

    assert parts == [
        {'part_number': 1, 'etag': '"e1"'},
        {'part_number': 2, 'etag': '"e2"'},
        {'part_number': 3, 'etag': '"e3"'},
    ]
    assert [put['body'] for put in client.session.puts] == [b'0123', b'4567', b'89']


async def test_upload_parts_missing_etag_raises(tmp_path):
    from nexus.utils.artifact.errors import ArtifactUploadError

    client = ArtifactClient()
    client.session = _RecordingPartSession([None])
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/p1'],
        part_size=4,
        part_count=1,
    )
    local = tmp_path / 'f.bin'
    local.write_bytes(b'abc')

    with pytest.raises(ArtifactUploadError, match='returned no ETag'):
        await client._upload_parts_to_cos(session_info, local)


class _SequencedPutResponse:
    def __init__(self, etag='"etag"', status=200, data=None, record=None):
        self.status = status
        self.headers = {'ETag': etag} if etag is not None else {}
        self._data = data
        self._record = record

    async def __aenter__(self):
        if hasattr(self._data, '__aiter__'):
            chunks = [chunk async for chunk in self._data]
            if self._record is not None:
                self._record['body'] = b''.join(chunks)
        return self

    async def __aexit__(self, *exc):
        return None

    async def text(self):
        return 'transient error'


class _RetryingPartSession:
    """Fake session returning a scripted response sequence per part URL.

    ``responses`` maps a part URL to an iterable of ``(status, etag)`` tuples,
    yielding one entry per ``put`` call so a part can fail then succeed on retry.
    """

    def __init__(self, responses):
        self._responses = {url: list(seq) for url, seq in responses.items()}
        self.put_counts = dict.fromkeys(responses, 0)

    def put(self, url, data=None, headers=None, timeout=None):
        self.put_counts[url] += 1
        status, etag = self._responses[url].pop(0)
        return _SequencedPutResponse(status=status, etag=etag, data=data)


async def test_upload_parts_retries_transient_status_then_succeeds(monkeypatch):
    from nexus.utils.artifact import client as client_module

    # Avoid real backoff sleeps between attempts to keep the test fast.
    monkeypatch.setattr(client_module, 'DEFAULT_MULTIPART_RETRY_BACKOFF', 0.0)
    client = ArtifactClient()
    # First PUT returns HTTP 500 (transient); the retry succeeds with an ETag.
    client.session = _RetryingPartSession({'http://cos/p1': [(500, None), (200, '"e1"')]})
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/p1'],
        part_size=4,
        part_count=1,
    )

    parts = await client._upload_bytes_parts_to_cos(session_info, b'abc', max_concurrency=1)

    assert parts == [{'part_number': 1, 'etag': '"e1"'}]
    # The transient 500 must have triggered exactly one retry (two PUTs total).
    assert client.session.put_counts['http://cos/p1'] == 2


async def test_upload_parts_exhausts_retries_and_raises_upload_error(monkeypatch):
    from nexus.utils.artifact import client as client_module

    # Every attempt returns HTTP 500 so the retry budget is exhausted.
    attempts = client_module.DEFAULT_MULTIPART_MAX_RETRIES + 1
    # Avoid real backoff sleeps between attempts to keep the test fast.
    monkeypatch.setattr(client_module, 'DEFAULT_MULTIPART_RETRY_BACKOFF', 0.0)
    client = ArtifactClient()
    client.session = _RetryingPartSession({'http://cos/p1': [(500, None)] * attempts})
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['http://cos/p1'],
        part_size=4,
        part_count=1,
    )

    with pytest.raises(ArtifactUploadError, match='part 1 failed'):
        await client._upload_bytes_parts_to_cos(session_info, b'abc', max_concurrency=1)

    # All attempts (initial + retries) were exhausted before giving up.
    assert client.session.put_counts['http://cos/p1'] == attempts


class _NetworkErrorPutResponse:
    def __init__(self, data=None):
        self._data = data

    async def __aenter__(self):
        # Consume the streamed body (mirrors the real aiohttp flow) then fail
        # with a transport-level error, the branch that embeds the retry key
        # into the raised message.
        if hasattr(self._data, '__aiter__'):
            _ = [chunk async for chunk in self._data]
        import aiohttp

        raise aiohttp.ClientError('connection reset')

    async def __aexit__(self, *exc):
        return None


class _NetworkErrorPartSession:
    """Fake session whose every PUT fails with an aiohttp transport error."""

    def __init__(self):
        self.put_count = 0

    def put(self, url, data=None, headers=None, timeout=None):
        self.put_count += 1
        return _NetworkErrorPutResponse(data=data)


async def test_upload_part_network_failure_does_not_leak_presigned_url(monkeypatch):
    """Retry-exhausted transport errors must not embed the presigned URL.

    The COS ``with_retries`` helper stringifies the retry key into the raised
    message on transport failures. Passing the full presigned URL there would
    leak the ``q-signature`` / temporary token into CLI or task logs, so the
    client must pass a credential-free label (``part N``) instead.
    """
    from nexus.utils.artifact import client as client_module

    monkeypatch.setattr(client_module, 'DEFAULT_MULTIPART_RETRY_BACKOFF', 0.0)
    client = ArtifactClient()
    client.session = _NetworkErrorPartSession()
    secret_url = (
        'http://cos/p1?q-sign-algorithm=sha1&q-signature=DEADBEEFSECRETSIG'
        '&x-cos-security-token=TEMP_TOKEN_SHOULD_NOT_LEAK'
    )
    session_info = UploadSession(
        artifact_id='art_mp',
        upload_id='upl_mp',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=[secret_url],
        part_size=4,
        part_count=1,
    )

    with pytest.raises(ArtifactUploadError) as excinfo:
        await client._upload_bytes_parts_to_cos(session_info, b'abc', max_concurrency=1)

    message = str(excinfo.value)
    # The safe, credential-free label is used to identify the failing PUT.
    assert 'part 1' in message
    # None of the presigned URL's credential material may appear anywhere in
    # the exception chain (message + the wrapped cause).
    chain = message + repr(excinfo.value.__cause__)
    assert 'q-signature' not in chain
    assert 'DEADBEEFSECRETSIG' not in chain
    assert 'TEMP_TOKEN_SHOULD_NOT_LEAK' not in chain
    assert 'x-cos-security-token' not in chain


# --- UploadSession multipart parsing ----------------------------------------


def test_upload_session_is_multipart_true_when_parts_signed():
    session = UploadSession.from_dict({
        'artifact_id': 'art_mp',
        'upload_id': 'upl_mp',
        'status': 'pending',
        'upload': {
            'part_urls': ['http://cos/p1', 'http://cos/p2'],
            'part_size': 8 * 1024 * 1024,
            'part_count': 2,
        },
    })
    assert session.is_multipart is True
    assert session.part_urls == ['http://cos/p1', 'http://cos/p2']
    assert session.part_size == 8 * 1024 * 1024
    assert session.part_count == 2


def test_upload_session_is_multipart_false_for_single_put():
    session = UploadSession.from_dict({
        'artifact_id': 'art_s',
        'upload_id': 'upl_s',
        'status': 'pending',
        'upload': {'method': 'PUT', 'url': 'http://cos', 'headers': {'h': 'v'}},
    })
    assert session.is_multipart is False
    assert session.part_urls == []
    assert session.url == 'http://cos'


# --- download streaming -----------------------------------------------------


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _n):
        for chunk in self._chunks:
            yield chunk


class _StreamResponse:
    def __init__(self, chunks, status=200):
        self.status = status
        self._chunks = chunks
        self.content = _Content(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def text(self):
        return ''

    async def read(self):
        return b''.join(self._chunks)


class _StreamingSession:
    def __init__(self, chunks):
        self._chunks = chunks

    def get(self, url, timeout=None):
        self.last_url = url
        return _StreamResponse(self._chunks)


async def test_download_to_file_streams_chunks(tmp_path):
    client = ArtifactClient()

    async def fake_url(artifact_id):
        return 'http://cos/get'

    client.get_download_url = fake_url
    client.session = _StreamingSession([b'ab', b'cd'])

    target = tmp_path / 'out' / 'f.bin'
    saved = await client.download_to_file('art_1', target)

    assert saved == target
    assert target.read_bytes() == b'abcd'


async def test_download_bytes_returns_content(tmp_path):
    client = ArtifactClient()

    async def fake_url(artifact_id):
        return 'http://cos/get'

    client.get_download_url = fake_url
    client.session = _StreamingSession([b'xy', b'z'])

    assert await client.download_bytes('art_1') == b'xyz'


def test_require_session_outside_context():
    client = ArtifactClient()
    with pytest.raises(RuntimeError, match='async context manager'):
        client._require_session()


def test_client_signature_excludes_legacy_tauth_parameters():
    parameters = inspect.signature(ArtifactClient).parameters
    legacy_parameters = {'token', 'authenticator', 'cmk_file', 'user', 'proxy_user', 'region', 'target'}

    assert legacy_parameters.isdisjoint(parameters)


async def test_request_without_credentials_sends_no_identity_headers(monkeypatch):
    monkeypatch.delenv('ARTIFACT_TAIHU_PAT', raising=False)
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    client = ArtifactClient(base_url='http://localhost:8080')
    client.session = RecordingSession(FakeResponse(json_body={'items': []}))

    await client._request('GET', '/v1/resources')

    headers = client.session.requests[0]['headers']
    assert 'Authorization' not in headers
    assert 'X-Dev-Employee' not in headers


async def test_taihu_pat_from_env(monkeypatch):
    monkeypatch.setenv('ARTIFACT_TAIHU_PAT', 'Bearer tai_pat_env')
    client = ArtifactClient()
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert client._taihu_pat == 'tai_pat_env'
    assert headers == {'Authorization': 'Bearer tai_pat_env'}


async def test_request_sends_bearer_pat_header():
    client = ArtifactClient(base_url='https://artifact-nexus-agent.mcp.woa.com', taihu_pat='tai_pat_xyz')
    client.session = RecordingSession(FakeResponse(json_body={'items': []}))

    await client._request('GET', '/v1/resources')

    req = client.session.requests[0]
    assert req['url'] == 'https://artifact-nexus-agent.mcp.woa.com/v1/resources'
    assert req['headers']['Authorization'] == 'Bearer tai_pat_xyz'
    assert 'X-Dev-Employee' not in req['headers']


async def test_request_sends_dev_employee_header_without_authorization():
    client = ArtifactClient(base_url='http://localhost:8080', dev_employee='junlinyin')
    client.session = RecordingSession(FakeResponse(json_body={'items': []}))

    await client._request('GET', '/v1/resources')

    req = client.session.requests[0]
    assert req['headers']['X-Dev-Employee'] == 'junlinyin'
    assert 'Authorization' not in req['headers']


async def test_dev_employee_takes_precedence_over_pat():
    client = ArtifactClient(dev_employee='alice', taihu_pat='tai_pat_xyz')
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert headers == {'X-Dev-Employee': 'alice'}


async def test_explicit_pat_takes_precedence_over_environment_dev_employee(monkeypatch):
    monkeypatch.setenv('ARTIFACT_DEV_EMPLOYEE', 'env-alice')
    client = ArtifactClient(taihu_pat='tai_pat_explicit')
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert headers == {'Authorization': 'Bearer tai_pat_explicit'}


async def test_explicit_dev_employee_takes_precedence_over_environment_pat(monkeypatch):
    monkeypatch.setenv('ARTIFACT_TAIHU_PAT', 'tai_pat_env')
    client = ArtifactClient(dev_employee='explicit-alice')
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert headers == {'X-Dev-Employee': 'explicit-alice'}


async def test_taihu_pat_loaded_from_local_key_file(monkeypatch, tmp_path):
    from nexus.utils.artifact import auth as auth_module

    key_file = tmp_path / 'key'
    key_file.write_text('Bearer tai_pat_file', encoding='utf-8')
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)

    client = ArtifactClient(base_url='http://localhost:8080')
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert client._taihu_pat == 'tai_pat_file'
    assert headers == {'Authorization': 'Bearer tai_pat_file'}


async def test_explicit_dev_employee_takes_precedence_over_file_pat(monkeypatch, tmp_path):
    from nexus.utils.artifact import auth as auth_module

    key_file = tmp_path / 'key'
    key_file.write_text('tai_pat_file', encoding='utf-8')
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)

    client = ArtifactClient(dev_employee='explicit-alice')
    headers: dict[str, str] = {}

    await client._apply_auth_headers(headers)

    assert headers == {'X-Dev-Employee': 'explicit-alice'}


# --- path resolution --------------------------------------------------------
#
# resolve_path walks the folder tree by calling list_resources one level at a
# time. These tests drive it with an in-memory tree keyed by parent id, and also
# exercise NFC name normalization and cursor paging.


import unicodedata  # noqa: E402

from nexus.utils.artifact.errors import ArtifactPathError  # noqa: E402
from nexus.utils.artifact.models import Page  # noqa: E402


def _client_with_tree(tree, *, pages=None, name_aware=True):
    """Return a client whose list_resources reads from an in-memory tree.

    ``tree`` maps parent_folder_id (None for top level) -> list of item dicts.
    ``pages`` optionally maps parent id -> list of (items, next_cursor) tuples to
    simulate multi-page listings; when set for a parent it overrides ``tree``.

    ``name_aware`` (default True) emulates a server that honors the ``?name=``
    point-lookup filter -- it returns only the matching child (<=1 item). Set it
    to False to emulate an older server that ignores ``name`` and returns an
    ordinary listing, which exercises the client's streaming fallback.
    """

    client = ArtifactClient()
    calls = []

    async def fake_list(*, parent_folder_id=None, view=None, limit=50, cursor=None, name=None):
        calls.append((parent_folder_id, cursor))
        if name is not None and name_aware:
            target = unicodedata.normalize('NFC', name)
            matches = [
                item
                for item in tree.get(parent_folder_id, [])
                if unicodedata.normalize('NFC', str(item.get('name') or '')) == target
            ]
            return Page(items=matches[:1], next_cursor=None)
        if pages and parent_folder_id in pages:
            steps = pages[parent_folder_id]
            index = 0 if cursor is None else int(cursor)
            items, next_cursor = steps[index]
            return Page(items=list(items), next_cursor=next_cursor)
        return Page(items=list(tree.get(parent_folder_id, [])), next_cursor=None)

    client.list_resources = fake_list
    return client, calls


async def test_resolve_view_and_owner_self_reference_is_owned_view():
    # ``artifact:///~/...`` must resolve to the caller's own namespace
    # (view=None, no owner filter), never view='shared' with owner '~'.
    from nexus.utils.artifact.uri import parse_artifact_uri

    client = ArtifactClient()
    # Guard against a real whoami() round-trip: the``~`` short-circuit must
    # not need it. Any network attempt would raise instead of returning here.

    async def _fail_whoami():
        raise AssertionError('whoami() must not be called for the ~ self-reference')

    client.whoami = _fail_whoami

    parsed = parse_artifact_uri('artifact:///~/models/v1.bin')
    view, owner_filter = await client._resolve_view_and_owner(parsed)

    assert view is None
    assert owner_filter is None


async def test_resolve_path_self_reference_finds_own_resource():
    # Regression: artifact:///~/<path> previously fell into view='shared',
    # owner='~' and could not find the caller's own resources.
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models', 'owner_id': 'me'}],
        'fld_1': [{'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin', 'owner_id': 'me'}],
    }
    views = []

    client = ArtifactClient()
    client._whoami_cache = 'me'

    async def fake_list(*, parent_folder_id=None, view=None, limit=50, cursor=None, name=None):
        views.append(view)
        items = tree.get(parent_folder_id, [])
        if name is not None:
            target = unicodedata.normalize('NFC', name)
            items = [i for i in items if unicodedata.normalize('NFC', str(i.get('name') or '')) == target]
        return Page(items=list(items), next_cursor=None)

    client.list_resources = fake_list

    assert await client.resolve_resource_id('artifact:///~/models/model.bin', expect='artifact') == 'art_1'
    # The top-level listing must use the owned view, not the shared view.
    assert views[0] is None
    assert 'shared' not in views


async def test_resolve_path_walks_folder_tree():
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}],
        'fld_1': [
            {'resource_type': 'folder', 'resource_id': 'fld_2', 'name': 'v1'},
            {'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin'},
        ],
        'fld_2': [{'resource_type': 'artifact', 'resource_id': 'art_2', 'name': 'w.bin'}],
    }
    client, _ = _client_with_tree(tree)

    assert await client.resolve_resource_id('/models') == 'fld_1'
    assert await client.resolve_resource_id('models/model.bin') == 'art_1'  # leading slash optional
    assert await client.resolve_resource_id('/models/v1/w.bin', expect='artifact') == 'art_2'


async def test_resolve_path_uses_server_side_name_point_lookup():
    # Against a name-aware server each segment resolves with a single request
    # that carries ``name=`` -- the huge-directory O(N) scan is avoided.
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}],
        'fld_1': [{'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin'}],
    }
    names_seen: list[tuple[str | None, str | None, int]] = []
    client, _ = _client_with_tree(tree)
    inner = client.list_resources

    async def recording(*, parent_folder_id=None, view=None, limit=50, cursor=None, name=None):
        names_seen.append((parent_folder_id, name, limit))
        return await inner(parent_folder_id=parent_folder_id, view=view, limit=limit, cursor=cursor, name=name)

    client.list_resources = recording

    assert await client.resolve_resource_id('/models/model.bin', expect='artifact') == 'art_1'
    # One request per segment, each a limit=1 point lookup keyed by name.
    assert names_seen == [(None, 'models', 1), ('fld_1', 'model.bin', 1)]


async def test_resolve_path_name_miss_fails_fast_without_scan():
    # A name-aware server returns an empty page for a missing child; the client
    # must treat that as not-found immediately, not fall back to a full scan.
    tree = {None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}]}
    client, calls = _client_with_tree(tree)

    with pytest.raises(ArtifactPathError, match='not found'):
        await client.resolve_resource_id('/absent')
    # Exactly one point-lookup request; no streaming/paging fallback.
    assert calls == [(None, None)]


async def test_resolve_path_falls_back_when_server_ignores_name():
    # An old server ignores ``name`` and returns the whole directory; the client
    # must still resolve correctly by scanning, even when the target is not the
    # first item returned by the point-lookup probe.
    tree = {
        None: [
            {'resource_type': 'folder', 'resource_id': 'fld_a', 'name': 'a'},
            {'resource_type': 'folder', 'resource_id': 'fld_b', 'name': 'b'},
        ],
    }
    client, _ = _client_with_tree(tree, name_aware=False)

    assert await client.resolve_resource_id('/b', expect='folder') == 'fld_b'


async def test_resolve_path_shared_view_skips_name_lookup():
    # A cross-owner shared top-level lookup cannot be keyed server-side, so the
    # client must not send ``name`` and instead scan with client-side owner
    # filtering (two grantors share the same file name).
    names_seen: list[str | None] = []
    tree = {
        None: [
            {'resource_type': 'artifact', 'resource_id': 'art_a', 'name': 'x.sh', 'owner_id': 'alice'},
            {'resource_type': 'artifact', 'resource_id': 'art_b', 'name': 'x.sh', 'owner_id': 'bob'},
        ],
    }
    client, _ = _client_with_tree(tree)
    client._whoami_cache = 'me'
    inner = client.list_resources

    async def recording(*, parent_folder_id=None, view=None, limit=50, cursor=None, name=None):
        names_seen.append(name)
        return await inner(parent_folder_id=parent_folder_id, view=view, limit=limit, cursor=cursor, name=name)

    client.list_resources = recording

    assert await client.resolve_resource_id('artifact:///bob/x.sh', expect='artifact') == 'art_b'
    assert names_seen == [None]  # name was never sent for the shared lookup


async def test_resolve_path_type_enforcement():
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}],
        'fld_1': [{'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin'}],
    }
    client, _ = _client_with_tree(tree)

    with pytest.raises(ArtifactPathError):
        await client.resolve_resource_id('/models', expect='artifact')
    with pytest.raises(ArtifactPathError):
        await client.resolve_resource_id('/models/model.bin', expect='folder')


async def test_resolve_path_missing_segment_and_non_folder_intermediate():
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}],
        'fld_1': [{'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin'}],
    }
    client, _ = _client_with_tree(tree)

    with pytest.raises(ArtifactPathError, match='not found'):
        await client.resolve_resource_id('/models/nope.bin')
    # model.bin is an artifact, so it cannot be an intermediate path segment.
    with pytest.raises(ArtifactPathError, match='not a folder'):
        await client.resolve_resource_id('/models/model.bin/deeper')


async def test_resolve_path_empty_is_error():
    client, _ = _client_with_tree({None: []})
    with pytest.raises(ArtifactPathError):
        await client.resolve_resource_id('/')


async def test_resolve_path_normalizes_nfc_names():
    # Name stored in NFC; query provided in NFD must still match.
    nfc = unicodedata.normalize('NFC', 'café')
    nfd = unicodedata.normalize('NFD', 'café')
    assert nfc != nfd
    tree = {None: [{'resource_type': 'folder', 'resource_id': 'fld_c', 'name': nfc}]}
    client, _ = _client_with_tree(tree)

    assert await client.resolve_resource_id(nfd, expect='folder') == 'fld_c'


async def test_resolve_path_pages_through_cursors():
    # Emulate an OLD server that ignores ``?name=``: the client first probes for
    # a point lookup (one call), the server returns an ordinary listing whose
    # first item does not match, so the client falls back to the streaming scan
    # and pages through the cursors. The target sits on the second page.
    pages = {
        None: [
            ([{'resource_type': 'folder', 'resource_id': 'fld_a', 'name': 'a'}], '1'),
            ([{'resource_type': 'folder', 'resource_id': 'fld_b', 'name': 'b'}], None),
        ]
    }
    client, calls = _client_with_tree({}, pages=pages, name_aware=False)

    assert await client.resolve_resource_id('/b', expect='folder') == 'fld_b'
    # One O(1) probe (ignored by the old server), then the paged fallback scan.
    assert calls == [(None, None), (None, None), (None, '1')]


async def test_resolve_parent_folder_id_and_leaf_name():
    tree = {
        None: [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models'}],
        'fld_1': [{'resource_type': 'folder', 'resource_id': 'fld_2', 'name': 'v1'}],
    }
    client, _ = _client_with_tree(tree)

    # Single-segment path has no parent (top level).
    assert await client.resolve_parent_folder_id('/newtop') is None
    assert client.leaf_name('/newtop') == 'newtop'
    # Nested path: parent resolves to the existing folder id.
    assert await client.resolve_parent_folder_id('/models/v1/new.bin') == 'fld_2'
    assert client.leaf_name('/models/v1/new.bin') == 'new.bin'


async def test_resolve_parent_folder_id_rejects_single_segment_foreign_owner():
    # Regression (P1): artifact:///<other>/new must not silently resolve to the
    # caller's own root; it must be rejected rather than creating /new for self.
    client, _ = _client_with_tree({None: []})
    client._whoami_cache = 'me'

    with pytest.raises(ArtifactPathError, match='another employee'):
        await client.resolve_parent_folder_id('artifact:///alice/new')


async def test_resolve_parent_folder_id_allows_single_segment_self_owner_and_self_ref():
    # An explicit self-owner and the ``~`` self-reference both mean the caller's
    # own root, so the single-segment shortcut returns None (no parent).
    client, _ = _client_with_tree({None: []})
    client._whoami_cache = 'me'

    assert await client.resolve_parent_folder_id('artifact:///me/new') is None
    assert await client.resolve_parent_folder_id('artifact:///~/new') is None


# ---------------------------------------------------------------------------
# download_folder_to_dir: recursive mirror
# ---------------------------------------------------------------------------


async def test_download_folder_to_dir_mirrors_nested_tree(tmp_path):
    """Recursive download reproduces the artifact tree under ``dest_dir``."""
    tree = {
        'root': [
            {'resource_type': 'folder', 'resource_id': 'fld_sub', 'name': 'sub'},
            {'resource_type': 'artifact', 'resource_id': 'art_1', 'artifact_id': 'art_1', 'name': 'top.txt'},
        ],
        'fld_sub': [
            {'resource_type': 'artifact', 'resource_id': 'art_2', 'artifact_id': 'art_2', 'name': 'nested.txt'},
        ],
    }
    client, _ = _client_with_tree(tree)

    downloaded: list[tuple[str, str]] = []

    async def fake_download(artifact_id, local_path):
        # Simulate a successful transfer by writing a small file with the
        # id as its body -- lets the assertion below verify the mapping.
        from pathlib import Path

        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact_id.encode())
        downloaded.append((artifact_id, str(path)))
        return path

    client.download_to_file = fake_download

    dest = tmp_path / 'mirror'
    successes, failures = await client.download_folder_to_dir('root', dest, concurrency=2)

    assert failures == []
    assert successes == 2
    # Both leaves land under dest with the correct relative structure.
    assert (dest / 'top.txt').read_bytes() == b'art_1'
    assert (dest / 'sub' / 'nested.txt').read_bytes() == b'art_2'
    # And nothing else -- no stray directories from folders that had only
    # nested artifacts (``sub`` is materialised implicitly by its child).
    assert sorted(p.name for p in dest.iterdir()) == ['sub', 'top.txt']


async def test_download_folder_to_dir_collects_failures_without_aborting(tmp_path):
    """A single failed download should not stop the others; it's surfaced in the summary."""
    tree = {
        'root': [
            {'resource_type': 'artifact', 'resource_id': 'art_ok', 'artifact_id': 'art_ok', 'name': 'ok.txt'},
            {'resource_type': 'artifact', 'resource_id': 'art_bad', 'artifact_id': 'art_bad', 'name': 'bad.txt'},
        ],
    }
    client, _ = _client_with_tree(tree)

    async def fake_download(artifact_id, local_path):
        if artifact_id == 'art_bad':
            raise ArtifactUploadError('simulated 500')
        from pathlib import Path

        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'ok')
        return path

    client.download_to_file = fake_download

    dest = tmp_path / 'partial'
    successes, failures = await client.download_folder_to_dir('root', dest, concurrency=4)

    # One survived, one was reported.
    assert successes == 1
    assert len(failures) == 1
    assert failures[0].artifact_id == 'art_bad'
    assert failures[0].relative_path == 'bad.txt'
    assert 'simulated 500' in failures[0].error
    # And the surviving one actually landed on disk.
    assert (dest / 'ok.txt').read_bytes() == b'ok'


async def test_download_folder_to_dir_creates_empty_dest_for_empty_source(tmp_path):
    """An empty folder still creates ``dest_dir`` so the caller's -o path exists."""
    client, _ = _client_with_tree({'root': []})

    dest = tmp_path / 'empty'
    successes, failures = await client.download_folder_to_dir('root', dest)

    assert successes == 0
    assert failures == []
    assert dest.exists()
    assert dest.is_dir()
    assert list(dest.iterdir()) == []


# ---------------------------------------------------------------------------
# download_folder_to_dir: symlink / path-escape hardening (P1)
# ---------------------------------------------------------------------------


def test_resolve_within_rejects_parent_escape():
    from pathlib import Path

    from nexus.utils.artifact.client import _resolve_within

    base = Path('/tmp/base')
    # Normal nested path stays inside base.
    assert _resolve_within(base, 'sub/file.txt') == base.resolve() / 'sub' / 'file.txt'
    # ``..`` segments that would climb out of base are rejected.
    with pytest.raises(ArtifactPathError, match='parent-directory'):
        _resolve_within(base, '../evil.txt')
    with pytest.raises(ArtifactPathError, match='parent-directory'):
        _resolve_within(base, 'sub/../../evil.txt')


def test_mkdir_within_rejects_symlinked_intermediate(tmp_path):
    from nexus.utils.artifact.client import _mkdir_within

    outside = tmp_path / 'outside'
    outside.mkdir()
    dest = tmp_path / 'dest'
    dest.mkdir()
    # Pre-plant dest/sub as a symlink pointing outside the destination.
    (dest / 'sub').symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactUploadError, match='symlinked directory'):
        _mkdir_within(dest, dest / 'sub' / 'deeper')
    # Nothing was created under the escape target.
    assert list(outside.iterdir()) == []


async def test_download_folder_to_dir_refuses_symlinked_subdir(tmp_path):
    """A symlinked intermediate directory must fail that leaf, not escape ``dest``."""
    tree = {
        'root': [
            {'resource_type': 'folder', 'resource_id': 'fld_sub', 'name': 'sub'},
            {'resource_type': 'artifact', 'resource_id': 'art_top', 'artifact_id': 'art_top', 'name': 'top.txt'},
        ],
        'fld_sub': [
            {'resource_type': 'artifact', 'resource_id': 'art_in', 'artifact_id': 'art_in', 'name': 'nested.txt'},
        ],
    }
    client, _ = _client_with_tree(tree)

    outside = tmp_path / 'outside'
    outside.mkdir()
    dest = tmp_path / 'dest'
    dest.mkdir()
    # dest/sub is a symlink to an external directory; writing sub/nested.txt
    # must be refused rather than following the link into ``outside``.
    (dest / 'sub').symlink_to(outside, target_is_directory=True)

    written: list[str] = []

    async def fake_download(artifact_id, local_path):
        from pathlib import Path

        path = Path(local_path)
        path.write_bytes(b'x')
        written.append(str(path))
        return path

    client.download_to_file = fake_download

    successes, failures = await client.download_folder_to_dir('root', dest, concurrency=1)

    # top.txt (no symlink in its path) succeeds; nested.txt is refused.
    assert successes == 1
    assert len(failures) == 1
    assert failures[0].relative_path == 'sub/nested.txt'
    assert 'symlink' in failures[0].error.lower()
    # The external directory was never written through.
    assert list(outside.iterdir()) == []
    # download_to_file was never invoked for the refused leaf.
    assert all('nested.txt' not in w for w in written)


def test_open_new_no_follow_rejects_symlink_target(tmp_path):
    import errno
    import os

    from nexus.utils.artifact.client import _open_new_no_follow

    if not getattr(os, 'O_NOFOLLOW', 0):
        pytest.skip('O_NOFOLLOW not available on this platform')

    outside = tmp_path / 'secret.txt'
    outside.write_bytes(b'original')
    link = tmp_path / 'link.part'
    link.symlink_to(outside)

    # Opening a symlinked path with O_NOFOLLOW must fail instead of writing
    # through to the linked file. The errno (ELOOP/EEXIST) is asserted below;
    # the message text is platform-specific so no ``match`` is used here.
    with pytest.raises(OSError) as excinfo:  # noqa: PT011
        _open_new_no_follow(link)
    assert excinfo.value.errno in (errno.ELOOP, errno.EEXIST)
    assert outside.read_bytes() == b'original'


@pytest.mark.parametrize('mutate', [False, True])
async def test_upload_reuses_received_hashes_only_while_file_is_unchanged(tmp_path, monkeypatch, mutate):
    from nexus.distributed.artifacts.file_io import write_chunks
    from nexus.utils.artifact import client as client_module

    local = tmp_path / 'received.bin'
    payload = b'content already verified at receipt' * 1000

    async def chunks():
        for offset in range(0, len(payload), 123):
            yield payload[offset : offset + 123]

    identity = await write_chunks(local, chunks(), compute_md5=True)
    assert identity.sha256 == hashlib.sha256(payload).hexdigest()
    assert identity.content_md5 == base64.b64encode(hashlib.md5(payload).digest()).decode('ascii')  # noqa: S324

    async def redundant_scan(source):
        pytest.fail('verified upload must not scan the file for MD5 again')

    monkeypatch.setattr(client_module, '_open_file_md5_base64', redundant_scan)
    client = ArtifactClient()
    captured = _patch_upload(client)
    captured['session'] = UploadSession(
        artifact_id='art_verified',
        upload_id='upl_verified',
        status='pending',
        method='PUT',
        url='http://cos',
        headers={},
    )
    if mutate:
        # An inode change is observable even on filesystems with coarse timestamps.
        replacement = tmp_path / 'replacement.bin'
        replacement.write_bytes(b'x' * len(payload))
        replacement.replace(local)
        with pytest.raises(ArtifactUploadError, match='changed'):
            await client.upload_file(local, _identity=identity)
        assert 'create' not in captured
    else:
        result = await client.upload_file(local, _identity=identity)
        assert result.artifact_id == 'art_verified'
        assert captured['create']['content_md5'] == identity.content_md5
        assert captured['create']['size_bytes'] == len(payload)
        assert captured['complete'] == ('upl_verified', None)
