import pytest

from nexus.dataset.auth import TaiPatAuthenticator
from nexus.dataset.exceptions import DatasetClientError, DatasetConfigurationError, UcAssetNotFoundError
from nexus.dataset.io import DatasetIoClient
from nexus.dataset.uc import DEFAULT_TAIJI_UC_BASE_URL, UcMetadataClient


class FakeResponse:
    def __init__(self, *, status=200, payload=None, text='', json_error=False):
        self.status = status
        self._payload = payload if payload is not None else {'code': 0, 'data': {}}
        self._text = text
        self._json_error = json_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def json(self, content_type=None):
        if self._json_error:
            raise ValueError('not JSON')
        return self._payload

    async def text(self):
        return self._text

    async def read(self):
        return self._text.encode()


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def request(self, method, url, *, params=None, json=None, headers=None):
        self.calls.append({'method': method, 'url': url, 'params': params, 'json': json, 'headers': headers})
        return self.response

    def post(self, url, *, json=None, headers=None):
        self.calls.append({'method': 'POST', 'url': url, 'json': json, 'headers': headers})
        return self.response

    def get(self, url):
        self.calls.append({'method': 'GET', 'url': url})
        return self.response

    def put(self, url, *, data=None, headers=None):
        self.calls.append({'method': 'PUT', 'url': url, 'data': data, 'headers': headers})
        return self.response


class FailingAuthenticator:
    async def get_headers(self, session):
        raise DatasetConfigurationError('missing test credentials')


@pytest.mark.asyncio
async def test_io_client_shapes_upload_commit_publish_requests():
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {'ok': True}}))
    client = DatasetIoClient(server_url='http://io', auth_token='auth.token', session=session)

    await client.sign_uploads('ns', 'ds', 'dev-1', ['_meta/version.json'], expires=3600)
    await client.commit('ns', 'ds', 'dev-1')
    await client.publish('ns', 'ds', source_snapshot='dev-1', release_version='v1')

    assert session.calls[0] == {
        'method': 'POST',
        'url': 'http://io/api/datasets/ns/ds/uploads:sign',
        'params': None,
        'json': {'target_version': 'dev-1', 'files': ['_meta/version.json'], 'expires': 3600},
        'headers': {'Content-Type': 'application/json', 'secure-authentication': 'auth.token'},
    }
    assert session.calls[1]['url'] == 'http://io/api/datasets/ns/ds:commit'
    assert session.calls[1]['json'] == {'snapshot': 'dev-1'}
    assert session.calls[2]['url'] == 'http://io/api/datasets/ns/ds:publish'
    assert session.calls[2]['json'] == {'source_snapshot': 'dev-1', 'release_version': 'v1'}


@pytest.mark.asyncio
async def test_io_client_shapes_multipart_signing_request():
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {'upload_id': 'upload-id'}}))
    client = DatasetIoClient(server_url='http://io', auth_token='auth.token', session=session)

    result = await client.sign_multipart_uploads(
        'ns',
        'ds',
        'dev-1',
        'test/large.parquet',
        5 * 1024**3,
        part_size=100 * 1024**2,
        expires=3600,
    )

    assert result == {'upload_id': 'upload-id'}
    assert session.calls == [
        {
            'method': 'POST',
            'url': 'http://io/api/datasets/ns/ds/uploads:sign-multipart',
            'params': None,
            'json': {
                'target_version': 'dev-1',
                'key': 'test/large.parquet',
                'total_size': 5 * 1024**3,
                'part_size': 100 * 1024**2,
                'expires': 3600,
            },
            'headers': {'Content-Type': 'application/json', 'secure-authentication': 'auth.token'},
        }
    ]


@pytest.mark.asyncio
async def test_io_client_legacy_signed_url_upload_remains_available():
    session = FakeSession(FakeResponse(status=204))
    client = DatasetIoClient(server_url='http://io', auth_token='auth.token', session=session)

    await client.upload_file('https://cos/upload', b'legacy', content_type='application/json')

    assert session.calls == [
        {
            'method': 'PUT',
            'url': 'https://cos/upload',
            'data': b'legacy',
            'headers': {'Expect': '', 'Content-Type': 'application/json'},
        }
    ]


@pytest.mark.asyncio
async def test_io_client_reads_static_auth_token_from_environment(monkeypatch):
    monkeypatch.setenv('TAIJI_DATASET_AUTH_TOKEN', 'environment.token')
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {'latest_release': 'v1'}}))
    client = DatasetIoClient(server_url='http://io', session=session)

    await client.get_dataset('ns', 'ds')

    assert session.calls[0]['headers'] == {
        'Content-Type': 'application/json',
        'secure-authentication': 'environment.token',
    }


@pytest.mark.asyncio
async def test_io_client_anonymous_read_sends_empty_auth_header():
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {'ok': True}}))
    client = DatasetIoClient(
        server_url='http://io',
        authenticator=FailingAuthenticator(),
        session=session,
        allow_anonymous=True,
    )

    await client.get_dataset('ns', 'ds')
    await client.sign_files('ns', 'ds', 'v1', ['blobs/case.zip'], split='test')

    assert session.calls[0]['headers'] == {'Content-Type': 'application/json', 'secure-authentication': ''}
    assert session.calls[1]['url'] == 'http://io/api/datasets/ns/ds/versions/v1/files:sign'
    assert session.calls[1]['headers'] == {'Content-Type': 'application/json', 'secure-authentication': ''}


@pytest.mark.asyncio
async def test_io_client_anonymous_mode_does_not_apply_to_uploads():
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {'ok': True}}))
    client = DatasetIoClient(
        server_url='http://io',
        authenticator=FailingAuthenticator(),
        session=session,
        allow_anonymous=True,
    )

    with pytest.raises(DatasetConfigurationError):
        await client.sign_uploads('ns', 'ds', 'dev-1', ['_meta/version.json'])
    with pytest.raises(DatasetConfigurationError):
        await client.sign_multipart_uploads('ns', 'ds', 'dev-1', 'large.bin', 5 * 1024**3)

    assert session.calls == []


@pytest.mark.asyncio
async def test_uc_client_shapes_register_and_list_requests():
    session = FakeSession(FakeResponse(payload={'Response': {'Data': {'FullName': 'taiji.nexus.ns/ds'}}}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    await client.register_nexus_asset('ns', 'ds', creator='alice', admin_list=['alice'], visibility='PRIVATE')
    await client.list_assets(asset_name='ns/', page_size=5)

    assert session.calls[0]['url'] == 'http://uc/api/1.0/unity-catalog/taiji/CreateNexusAssetMeta'
    assert session.calls[0]['json'] == {
        'Namespace': 'ns',
        'DatasetName': 'ds',
        'Creator': 'alice',
        'AdminList': ['alice'],
        'Visibility': 'PRIVATE',
        'AssetDesc': '',
        'Status': 'PENDING',
    }
    assert session.calls[0]['headers']['secure-authentication'] == 'auth.token'
    assert session.calls[1]['url'] == 'http://uc/api/1.0/unity-catalog/taiji/ListAssetMetasBySchema'
    assert session.calls[1]['json'] == {'SchemaName': 'nexus', 'AssetName': 'ns/', 'PageSize': 5}


@pytest.mark.asyncio
async def test_uc_client_raises_structured_not_found_for_missing_asset():
    session = FakeSession(
        FakeResponse(
            payload={
                'Response': {
                    'Error': {
                        'Code': 'ResourceNotFound.MetaResourceNotFound',
                        'Message': '资产元数据不存在: taiji.nexus.nexus-test/example_bench_01',
                    }
                }
            }
        )
    )
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(UcAssetNotFoundError) as exc:
        await client.get_asset('nexus-test', 'example_bench_01')

    assert exc.value.code == 'ResourceNotFound.MetaResourceNotFound'
    assert exc.value.http_status == 200


@pytest.mark.asyncio
async def test_uc_client_creates_namespace_permission_with_flat_json_protocol():
    payload = {
        'id': 1,
        'schema': 'nexus',
        'namespace': 'research',
        'admin_list': ['alice', 'bob'],
        'creator': 'alice',
    }
    session = FakeSession(FakeResponse(payload=payload))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    result = await client.create_namespace_permission('research', ['alice', 'bob'])

    assert result == payload
    assert session.calls == [
        {
            'method': 'POST',
            'url': 'http://uc/api/1.0/unity-catalog/asset-namespace-permissions',
            'params': None,
            'json': {'schema': 'nexus', 'namespace': 'research', 'admin_list': ['alice', 'bob']},
            'headers': {'Content-Type': 'application/json', 'secure-authentication': 'auth.token'},
        }
    ]


@pytest.mark.asyncio
async def test_uc_client_gets_namespace_permission_from_paginated_response():
    permission = {
        'id': 1,
        'schema': 'nexus',
        'namespace': 'research',
        'admin_list': ['alice'],
        'creator': 'alice',
    }
    session = FakeSession(FakeResponse(payload={'data': [permission], 'total_elements': 1}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    result = await client.get_namespace_permission('research')

    assert result == permission
    assert session.calls[0]['method'] == 'GET'
    assert session.calls[0]['params'] == {'schema': 'nexus', 'namespace': 'research'}
    assert session.calls[0]['json'] is None


@pytest.mark.asyncio
async def test_uc_client_returns_none_for_missing_namespace_permission():
    session = FakeSession(FakeResponse(payload={'data': [], 'total_elements': 0}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    assert await client.get_namespace_permission('missing') is None


@pytest.mark.asyncio
async def test_uc_client_updates_namespace_permission_with_full_admin_list():
    payload = {
        'id': 1,
        'schema': 'nexus',
        'namespace': 'research',
        'admin_list': ['alice', 'carol'],
        'creator': 'alice',
    }
    session = FakeSession(FakeResponse(payload=payload))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    result = await client.update_namespace_permission('research', ['alice', 'carol'])

    assert result == payload
    assert session.calls[0]['method'] == 'PATCH'
    assert session.calls[0]['json'] == {
        'schema': 'nexus',
        'namespace': 'research',
        'admin_list': ['alice', 'carol'],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['create_namespace_permission', 'update_namespace_permission'])
async def test_uc_client_rejects_empty_namespace_permission_inputs(method):
    session = FakeSession(FakeResponse(payload={}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(ValueError, match='namespace must not be empty'):
        await getattr(client, method)('', ['alice'])
    with pytest.raises(ValueError, match='admin_list must not be empty'):
        await getattr(client, method)('research', ['', ''])

    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status', 'message'),
    [
        (400, 'adminList must not be empty'),
        (401, 'only the namespace creator may update it'),
        (404, 'namespace permission does not exist'),
        (409, 'namespace permission already exists'),
    ],
)
async def test_uc_namespace_permission_errors_preserve_status_and_message(status, message):
    session = FakeSession(FakeResponse(status=status, payload={'message': message}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(DatasetClientError, match=message) as exc:
        await client.update_namespace_permission('research', ['alice'])

    assert exc.value.http_status == status


@pytest.mark.asyncio
async def test_uc_namespace_permission_non_json_error_preserves_response_text():
    session = FakeSession(FakeResponse(status=500, text='Internal Server Error', json_error=True))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(DatasetClientError, match='Internal Server Error') as exc:
        await client.create_namespace_permission('research', ['alice'])

    assert exc.value.http_status == 500


@pytest.mark.asyncio
async def test_uc_register_namespace_not_found_is_not_asset_not_found():
    message = 'namespace 未注册: nexus.research，请先创建该 namespace 权限记录'
    session = FakeSession(
        FakeResponse(
            payload={
                'Response': {
                    'Error': {
                        'Code': 'ResourceNotFound.MetaResourceNotFound',
                        'Message': message,
                    }
                }
            }
        )
    )
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(DatasetClientError, match='namespace 未注册') as exc:
        await client.register_nexus_asset('research', 'dataset', creator='alice', admin_list=['alice'])

    assert not isinstance(exc.value, UcAssetNotFoundError)
    assert exc.value.code == 'ResourceNotFound.MetaResourceNotFound'
    assert exc.value.http_status == 200


@pytest.mark.asyncio
async def test_uc_http_404_error_envelope_preserves_server_message():
    session = FakeSession(
        FakeResponse(
            status=404,
            payload={
                'Response': {
                    'Error': {
                        'Code': 'ResourceNotFound.MetaResourceNotFound',
                        'Message': 'namespace permission is missing',
                    }
                }
            },
        )
    )
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='auth.token', session=session)

    with pytest.raises(DatasetClientError, match='namespace permission is missing') as exc:
        await client.register_nexus_asset('research', 'dataset', creator='alice', admin_list=['alice'])

    assert not isinstance(exc.value, UcAssetNotFoundError)
    assert exc.value.http_status == 404


def test_uc_client_defaults_to_taihu_gateway_with_pat_authenticator():
    client = UcMetadataClient(authenticator=TaiPatAuthenticator(pat='test-token'))
    assert client.uc_base_url == DEFAULT_TAIJI_UC_BASE_URL


def test_uc_client_explicit_url_wins_over_taihu_gateway():
    client = UcMetadataClient(uc_base_url='http://custom-uc', authenticator=TaiPatAuthenticator(pat='test-token'))
    assert client.uc_base_url == 'http://custom-uc'


def test_uc_client_defaults_to_legacy_url_without_pat():
    client = UcMetadataClient(auth_token='auth.token')
    assert client.uc_base_url == 'http://taiji-nexus-dataset-uc-dev.woa.com:8080'


def test_io_client_defaults_to_taihu_gateway_with_pat_authenticator():
    client = DatasetIoClient(authenticator=TaiPatAuthenticator(pat='test-token'))
    assert client.server_url == 'https://taiji-nexus-dataset-io-dev.mcp.woa.com'


def test_io_client_explicit_url_wins_over_taihu_gateway():
    client = DatasetIoClient(server_url='http://custom-io', authenticator=TaiPatAuthenticator(pat='test-token'))
    assert client.server_url == 'http://custom-io'


def test_io_client_env_url_wins_over_taihu_gateway(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'env-pat-token')
    monkeypatch.setenv('TAIJI_DATASET_SERVER_URL', 'http://env-io')
    client = DatasetIoClient()
    assert client.server_url == 'http://env-io'


def test_uc_client_env_url_wins_over_taihu_gateway(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'env-pat-token')
    monkeypatch.setenv('UC_BASE_URL', 'http://env-uc')
    client = UcMetadataClient()
    assert client.uc_base_url == 'http://env-uc'


# ---------------------------------------------------------------------------
# tai_pat_from_file tests
# ---------------------------------------------------------------------------


def test_tai_pat_from_file_reads_token(tmp_path):
    from nexus.dataset.auth import tai_pat_from_file

    token_file = tmp_path / 'token'
    token_file.write_text('my-secret-token\n', encoding='utf-8')
    token_file.chmod(0o600)

    assert tai_pat_from_file(token_file) == 'my-secret-token'


def test_tai_pat_from_file_rejects_group_readable(tmp_path):
    from nexus.dataset.auth import tai_pat_from_file

    token_file = tmp_path / 'token'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o640)

    with pytest.raises(DatasetConfigurationError, match='must not be readable or writable by group/other'):
        tai_pat_from_file(token_file)


def test_tai_pat_from_file_rejects_world_readable(tmp_path):
    from nexus.dataset.auth import tai_pat_from_file

    token_file = tmp_path / 'token'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o644)

    with pytest.raises(DatasetConfigurationError, match='must not be readable or writable by group/other'):
        tai_pat_from_file(token_file)


def test_tai_pat_from_file_rejects_empty_file(tmp_path):
    from nexus.dataset.auth import tai_pat_from_file

    token_file = tmp_path / 'token'
    token_file.write_text('\n', encoding='utf-8')
    token_file.chmod(0o600)

    with pytest.raises(DatasetConfigurationError, match='is empty'):
        tai_pat_from_file(token_file)


# ---------------------------------------------------------------------------
# TaiPatAuthenticator with pat_file= parameter
# ---------------------------------------------------------------------------


def test_tai_pat_authenticator_reads_from_file(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('file-token\n', encoding='utf-8')
    token_file.chmod(0o600)

    auth = TaiPatAuthenticator(pat_file=token_file)
    assert auth.pat == 'file-token'


def test_tai_pat_authenticator_pat_and_pat_file_are_mutually_exclusive(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o600)

    with pytest.raises(DatasetConfigurationError, match='mutually exclusive'):
        TaiPatAuthenticator(pat='inline-tok', pat_file=token_file)


def test_tai_pat_authenticator_prefers_pat_over_env(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'env-token')
    auth = TaiPatAuthenticator(pat='explicit-token')
    assert auth.pat == 'explicit-token'


# ---------------------------------------------------------------------------
# build_default_authenticator fallback in _auth_headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_io_client_uses_tai_pat_from_env_when_no_authenticator_passed(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'env-pat-token')
    session = FakeSession(FakeResponse(payload={'code': 0, 'data': {}}))
    client = DatasetIoClient(server_url='http://io', session=session)

    await client.get_dataset('ns', 'ds')

    assert session.calls[0]['headers'].get('Authorization') == 'Bearer env-pat-token'


@pytest.mark.asyncio
async def test_uc_client_uses_tai_pat_from_env_when_no_authenticator_passed(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'env-pat-token')
    session = FakeSession(FakeResponse(payload={'Response': {'Data': {}}}))
    client = UcMetadataClient(uc_base_url='http://uc', session=session)

    await client.list_assets()

    assert session.calls[0]['headers'].get('Authorization') == 'Bearer env-pat-token'


# ---------------------------------------------------------------------------
# _read_taihu_token and _make_dataset_client CLI helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    import argparse

    ns = argparse.Namespace()
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


def test_read_taihu_token_returns_none_when_no_args():
    from nexus.cli.commands.dataset import _read_taihu_token

    args = _make_args(taihu_token=None, taihu_token_file=None)
    assert _read_taihu_token(args) is None


def test_read_taihu_token_returns_stripped_inline_token():
    from nexus.cli.commands.dataset import _read_taihu_token

    args = _make_args(taihu_token='  my-token  ', taihu_token_file=None)
    assert _read_taihu_token(args) == 'my-token'


def test_read_taihu_token_reads_from_file(tmp_path):
    from nexus.cli.commands.dataset import _read_taihu_token

    token_file = tmp_path / 'token'
    token_file.write_text('file-token\n', encoding='utf-8')
    token_file.chmod(0o600)

    args = _make_args(taihu_token=None, taihu_token_file=token_file)
    assert _read_taihu_token(args) == 'file-token'


def test_read_taihu_token_rejects_both_args(tmp_path):
    from nexus.cli.commands.dataset import _read_taihu_token

    token_file = tmp_path / 'token'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o600)

    args = _make_args(taihu_token='inline', taihu_token_file=token_file)
    with pytest.raises(ValueError, match='mutually exclusive'):
        _read_taihu_token(args)


# ---------------------------------------------------------------------------
# Default Taihu token file (~/.nexus/auth/taihu-pat) fallback
# ---------------------------------------------------------------------------


def _clear_dataset_auth_env(monkeypatch):
    monkeypatch.delenv('TAI_PAT', raising=False)
    monkeypatch.delenv('WEDATA_KEY', raising=False)
    monkeypatch.delenv('WEDATA_KEY_FILE', raising=False)


def _write_token_file(path, token='default-file-token'):
    path.write_text(f'{token}\n', encoding='utf-8')
    path.chmod(0o600)
    return path


def test_default_taihu_pat_returns_none_when_missing(monkeypatch, tmp_path):
    from nexus.dataset import auth

    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', tmp_path / 'taihu')
    assert auth.default_taihu_pat() is None


def test_default_taihu_pat_reads_file(monkeypatch, tmp_path):
    from nexus.dataset import auth

    token_file = _write_token_file(tmp_path / 'taihu')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)
    assert auth.default_taihu_pat() == 'default-file-token'


def test_default_taihu_pat_rejects_loose_permissions(monkeypatch, tmp_path):
    from nexus.dataset import auth

    token_file = tmp_path / 'taihu'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o644)
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)
    with pytest.raises(DatasetConfigurationError, match='must not be readable or writable by group/other'):
        auth.default_taihu_pat()


def test_tai_pat_authenticator_uses_default_file_when_no_env(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    token_file = _write_token_file(tmp_path / 'taihu')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)

    authenticator = auth.TaiPatAuthenticator()
    assert authenticator.pat == 'default-file-token'


def test_env_tai_pat_takes_priority_over_default_file(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    monkeypatch.setenv('TAI_PAT', 'env-token')
    token_file = _write_token_file(tmp_path / 'taihu')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)

    assert auth.resolve_default_tai_pat() == 'env-token'
    assert auth.TaiPatAuthenticator().pat == 'env-token'


def test_build_default_authenticator_falls_back_to_taihu_file_when_no_wedata(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    # No Wedata key present.
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', tmp_path / 'missing-wedata-key')
    token_file = _write_token_file(tmp_path / 'taihu')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)

    authenticator = auth.build_default_authenticator()
    assert isinstance(authenticator, auth.TaiPatAuthenticator)
    assert authenticator.pat == 'default-file-token'


def test_build_default_authenticator_prefers_wedata_when_key_file_exists(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    wedata_key = tmp_path / 'wedata-key'
    wedata_key.write_text('{"subject": "u", "key": "k", "id": "1"}', encoding='utf-8')
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', wedata_key)
    # Even if a default Taihu file exists, Wedata wins to keep existing users unaffected.
    token_file = _write_token_file(tmp_path / 'taihu')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', token_file)

    authenticator = auth.build_default_authenticator()
    assert isinstance(authenticator, auth.WedataAuthenticator)


def test_build_default_authenticator_raises_taihu_hint_when_nothing_configured(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', tmp_path / 'missing-wedata-key')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', tmp_path / 'missing-taihu')

    with pytest.raises(DatasetConfigurationError, match='Taihu personal access token') as exc:
        auth.build_default_authenticator()
    # The fallback error must guide the user to Taihu, not Wedata.
    assert 'TAI_PAT' in str(exc.value)
    assert 'wedata.woa.com' not in str(exc.value)


def test_prefer_tai_pat_true_for_default_file_without_wedata(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', tmp_path / 'missing-wedata-key')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', _write_token_file(tmp_path / 'taihu'))

    assert auth.prefer_tai_pat() is True


def test_prefer_tai_pat_false_when_wedata_available(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    wedata_key = tmp_path / 'wedata-key'
    wedata_key.write_text('{"subject": "u", "key": "k", "id": "1"}', encoding='utf-8')
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', wedata_key)
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', _write_token_file(tmp_path / 'taihu'))

    assert auth.prefer_tai_pat() is False


@pytest.mark.asyncio
async def test_uc_client_uses_default_taihu_file_and_gateway_url(monkeypatch, tmp_path):
    from nexus.dataset import auth

    _clear_dataset_auth_env(monkeypatch)
    monkeypatch.setattr(auth, 'WEDATA_AUTH_KEY_FILE', tmp_path / 'missing-wedata-key')
    monkeypatch.setattr(auth, 'TAIHU_AUTH_TOKEN_FILE', _write_token_file(tmp_path / 'taihu'))

    session = FakeSession(FakeResponse(payload={'Response': {'Data': {}}}))
    client = UcMetadataClient(session=session)
    # Gateway URL auto-selected because default Taihu file makes prefer_tai_pat() true.
    assert client.uc_base_url == DEFAULT_TAIJI_UC_BASE_URL.rstrip('/')

    await client.list_assets()
    assert session.calls[0]['headers'].get('Authorization') == 'Bearer default-file-token'


# ---------------------------------------------------------------------------
# 401 error hint is tailored to the active auth mode (Taihu vs Wedata)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uc_401_reports_taihu_hint_when_using_pat():
    session = FakeSession(FakeResponse(status=401, payload={'error': 'token_inactive:AUDIENCE_MISMATCH'}))
    client = UcMetadataClient(
        uc_base_url='http://uc',
        authenticator=TaiPatAuthenticator(pat='some-pat'),
        session=session,
    )

    with pytest.raises(DatasetClientError) as exc:
        await client.list_assets()
    message = str(exc.value)
    assert 'Taihu authentication failed' in message
    assert 'wedata.woa.com' not in message


@pytest.mark.asyncio
async def test_uc_401_reports_wedata_hint_when_using_auth_token():
    session = FakeSession(FakeResponse(status=401, payload={'error': 'unauthorized'}))
    client = UcMetadataClient(uc_base_url='http://uc', auth_token='tauth.token', session=session)

    with pytest.raises(DatasetClientError) as exc:
        await client.list_assets()
    message = str(exc.value)
    assert 'Wedata authentication failed' in message


@pytest.mark.asyncio
async def test_io_401_reports_taihu_hint_when_using_pat():
    session = FakeSession(FakeResponse(status=401, payload={'message': 'token_inactive:AUDIENCE_MISMATCH'}))
    client = DatasetIoClient(
        server_url='http://io',
        authenticator=TaiPatAuthenticator(pat='some-pat'),
        session=session,
    )

    with pytest.raises(DatasetClientError) as exc:
        await client.get_dataset('ns', 'ds')
    message = str(exc.value)
    assert 'Taihu authentication failed' in message
    assert 'wedata.woa.com' not in message


def test_tai_pat_from_file_permission_error_includes_chmod_hint(tmp_path):
    from nexus.dataset.auth import tai_pat_from_file

    token_file = tmp_path / 'taihu'
    token_file.write_text('tok', encoding='utf-8')
    token_file.chmod(0o644)

    with pytest.raises(DatasetConfigurationError, match='chmod 600'):
        tai_pat_from_file(token_file)


# ---------------------------------------------------------------------------
# DatasetIoClient retry tests
# ---------------------------------------------------------------------------


class SequencedFakeSession:
    """FakeSession that cycles through a list of FakeResponse objects on each call.

    On the nth call, returns ``responses[min(n-1, len-1)]`` so the last entry
    is repeated once the sequence is exhausted.
    """

    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls: list = []

    def _next(self) -> FakeResponse:
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]

    def request(self, method, url, *, params=None, json=None, headers=None):
        self.calls.append({'method': method, 'url': url, 'params': params, 'json': json, 'headers': headers})
        return self._next()

    def post(self, url, *, json=None, headers=None):
        self.calls.append({'method': 'POST', 'url': url, 'json': json, 'headers': headers})
        return self._next()

    def get(self, url):
        self.calls.append({'method': 'GET', 'url': url})
        return self._next()

    def put(self, url, *, data=None, headers=None):
        self.calls.append({'method': 'PUT', 'url': url, 'data': data, 'headers': headers})
        return self._next()


@pytest.mark.asyncio
async def test_io_client_read_endpoint_retries_on_5xx_and_recovers():
    """A transient 503 on get_dataset is retried; the second attempt succeeds."""
    fail_resp = FakeResponse(status=503, text='Service Unavailable')
    ok_resp = FakeResponse(payload={'code': 0, 'data': {'name': 'ds'}})
    session = SequencedFakeSession([fail_resp, ok_resp])
    # Zero backoff so the test runs instantly.
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        retry_backoff_base=0,
        retry_backoff_jitter=0,
    )

    result = await client.get_dataset('ns', 'ds')

    assert result == {'name': 'ds'}
    assert len(session.calls) == 2
    assert session.calls[0]['url'] == 'http://io/api/datasets/ns/ds'
    assert session.calls[1]['url'] == 'http://io/api/datasets/ns/ds'


@pytest.mark.asyncio
async def test_io_client_sign_files_retries_on_5xx_and_recovers():
    """sign_files is idempotent (just re-signs URLs), so it is also retriable."""
    fail_resp = FakeResponse(status=502, text='Bad Gateway')
    ok_resp = FakeResponse(payload={'code': 0, 'data': {'urls': {'blob.zip': 'https://cos/blob.zip'}}})
    session = SequencedFakeSession([fail_resp, ok_resp])
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        retry_backoff_base=0,
        retry_backoff_jitter=0,
    )

    result = await client.sign_files('ns', 'ds', 'v1', ['blob.zip'])

    assert result == {'urls': {'blob.zip': 'https://cos/blob.zip'}}
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_io_client_read_endpoint_exhausts_retries_and_raises():
    """After max_retries+1 attempts all returning 503, DatasetClientError is raised."""
    fail_resp = FakeResponse(status=503, text='still broken')
    session = SequencedFakeSession([fail_resp])
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        max_retries=2,
        retry_backoff_base=0,
        retry_backoff_jitter=0,
    )

    with pytest.raises(DatasetClientError, match='503'):
        await client.get_dataset('ns', 'ds')

    # 1 initial + 2 retries = 3 total attempts
    assert len(session.calls) == 3


@pytest.mark.asyncio
async def test_io_client_write_endpoint_does_not_retry_on_5xx():
    """sign_uploads is a write endpoint; a 503 response must NOT be retried."""
    fail_resp = FakeResponse(status=503, text='Service Unavailable')
    session = SequencedFakeSession([fail_resp])
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        retry_backoff_base=0,
        retry_backoff_jitter=0,
    )

    with pytest.raises(DatasetClientError, match='503'):
        await client.sign_uploads('ns', 'ds', 'dev-1', ['file.parquet'])

    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_io_client_max_retries_zero_disables_retry_for_read_endpoints():
    """Setting max_retries=0 turns off retry even for read endpoints."""
    fail_resp = FakeResponse(status=503, text='down')
    session = SequencedFakeSession([fail_resp])
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        max_retries=0,
    )

    with pytest.raises(DatasetClientError, match='503'):
        await client.get_dataset('ns', 'ds')

    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
async def test_io_client_all_retriable_status_codes_trigger_retry(status):
    """All statuses in RETRIABLE_STATUS_CODES are retried on read endpoints."""
    fail_resp = FakeResponse(status=status, text='transient error')
    ok_resp = FakeResponse(payload={'code': 0, 'data': {'ok': True}})
    session = SequencedFakeSession([fail_resp, ok_resp])
    client = DatasetIoClient(
        server_url='http://io',
        auth_token='tok',
        session=session,
        max_retries=1,
        retry_backoff_base=0,
        retry_backoff_jitter=0,
    )

    result = await client.get_version('ns', 'ds', 'v1')

    assert result == {'ok': True}
    assert len(session.calls) == 2
