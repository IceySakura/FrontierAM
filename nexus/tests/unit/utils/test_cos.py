from __future__ import annotations

import asyncio
import json

from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import SecretStr

from nexus.utils.cos import (
    DEFAULT_MULTIPART_PART_SIZE,
    AsyncCosClient,
    CosClientError,
    CosConfig,
    CosConfigurationError,
    CosCredentials,
    CosServiceError,
    CosV5Signer,
    clear_cos_credentials_cache,
    load_cos_credentials,
    select_multipart_part_size,
    upload_presigned_file,
    upload_presigned_multipart_file,
)


def test_cos_v5_signer_matches_sdk_reference() -> None:
    authorization = CosV5Signer('AKIDEXAMPLE', 'SECRETEXAMPLE').authorization(
        'GET',
        '/folder/a b.txt',
        host='example-1250000000.cos.ap-guangzhou.myqcloud.com',
        params={'versionId': 'v 1', 'empty': ''},
        headers={'Range': 'bytes=0-99'},
        expires=600,
        timestamp=1_700_000_000,
    )

    assert authorization == (
        'q-sign-algorithm=sha1&q-ak=AKIDEXAMPLE&'
        'q-sign-time=1699999940;1700000600&q-key-time=1699999940;1700000600&'
        'q-header-list=host;range&q-url-param-list=empty;versionid&'
        'q-signature=cc40aab10d72f28ee6a23a02e49cbbff23b0d696'
    )


def test_presign_url_signs_host_params_and_temporary_token() -> None:
    url = CosV5Signer('AKIDEXAMPLE', 'SECRETEXAMPLE', token='SESSION_TOKEN').presign_url(
        'GET',
        'https://example-1250000000.cos.ap-guangzhou.myqcloud.com/folder/a%20b.txt',
        params={'response-content-type': 'text/plain'},
        expires=600,
        timestamp=1_700_000_000,
    )

    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    assert parsed.path == '/folder/a%20b.txt'
    assert query['q-sign-algorithm'] == ['sha1']
    assert query['q-header-list'] == ['host']
    assert query['q-url-param-list'] == ['response-content-type;x-cos-security-token']
    assert query['response-content-type'] == ['text/plain']
    assert query['x-cos-security-token'] == ['SESSION_TOKEN']


def test_cos_config_builds_public_internal_and_custom_origins() -> None:
    internal = CosConfig(bucket='example-1250000000', region='ap-guangzhou')
    assert internal.origin == 'http://example-1250000000.cos-internal.ap-guangzhou.tencentcos.cn'
    assert DEFAULT_MULTIPART_PART_SIZE == 256 * 1024 * 1024
    assert internal.large_file_threshold == DEFAULT_MULTIPART_PART_SIZE
    assert internal.part_size == DEFAULT_MULTIPART_PART_SIZE
    public = CosConfig(
        bucket='example-1250000000',
        region='ap-guangzhou',
        use_internal_endpoint=False,
        scheme='https',
    )
    assert public.origin == 'https://example-1250000000.cos.ap-guangzhou.myqcloud.com'
    assert CosConfig(bucket='example-1250000000', endpoint='http://127.0.0.1:9000').origin == 'http://127.0.0.1:9000'


def test_load_cos_credentials_from_env_and_redacts_repr(monkeypatch) -> None:
    clear_cos_credentials_cache()
    monkeypatch.setenv('COS_SECRET_ID', 'env-id')
    monkeypatch.setenv('COS_SECRET_KEY', 'env-key')
    monkeypatch.setenv('COS_TOKEN', 'env-token')

    credentials = load_cos_credentials()

    assert credentials.to_shared_payload() == {
        'secret_id': 'env-id',
        'secret_key': 'env-key',
        'token': 'env-token',
    }
    assert 'env-id' not in repr(credentials)
    assert 'env-key' not in repr(credentials)
    clear_cos_credentials_cache()


def test_load_cos_credentials_from_default_file(monkeypatch, tmp_path) -> None:
    key_file = tmp_path / 'cos-key.json'
    key_file.write_text(json.dumps({'secret_id': 'file-id', 'secret_key': 'file-key'}))
    monkeypatch.delenv('COS_SECRET_ID', raising=False)
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)

    credentials = load_cos_credentials(key_file=key_file)

    assert credentials.to_shared_payload() == {'secret_id': 'file-id', 'secret_key': 'file-key'}


def test_load_cos_credentials_rejects_partial_env(monkeypatch) -> None:
    monkeypatch.setenv('COS_SECRET_ID', 'env-id')
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)

    with pytest.raises(CosConfigurationError, match='must be set together'):
        load_cos_credentials(key_file='/does/not/matter')


@pytest.mark.asyncio
async def test_async_cos_client_streams_files_and_retries(tmp_path) -> None:
    stored: dict[str, bytes] = {}
    requests: list[dict[str, str]] = []
    put_attempts = 0

    async def put_object(request: web.Request) -> web.Response:
        nonlocal put_attempts
        put_attempts += 1
        requests.append(dict(request.headers))
        if put_attempts == 1:
            return web.Response(
                status=503,
                text='<Error><Code>SlowDown</Code><Message>retry</Message></Error>',
                content_type='application/xml',
            )
        stored[request.path] = await request.read()
        return web.Response(headers={'ETag': '"upload-etag"', 'x-cos-request-id': 'put-request'})

    async def get_object(request: web.Request) -> web.Response:
        requests.append(dict(request.headers))
        if request.path not in stored:
            return web.Response(
                status=404,
                text=(
                    '<Error><Code>NoSuchKey</Code><Message>missing</Message>'
                    '<RequestId>missing-request</RequestId></Error>'
                ),
                content_type='application/xml',
            )
        return web.Response(
            body=stored[request.path],
            headers={'ETag': '"download-etag"', 'x-cos-request-id': 'get-request'},
        )

    async def delete_object(request: web.Request) -> web.Response:
        requests.append(dict(request.headers))
        stored.pop(request.path, None)
        return web.Response(headers={'x-cos-request-id': 'delete-request'})

    app = web.Application()
    app.router.add_put('/{key:.*}', put_object)
    app.router.add_get('/{key:.*}', get_object)
    app.router.add_delete('/{key:.*}', delete_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='unused-for-custom-endpoint',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=1,
            retry_backoff=0,
        )
        source = tmp_path / 'source.bin'
        source.write_bytes(b'workflow output')
        target = tmp_path / 'nested' / 'target.bin'

        credentials = CosCredentials(
            secret_id=SecretStr('AKIDEXAMPLE'),
            secret_key=SecretStr('SECRETEXAMPLE'),
            token=SecretStr('SESSION_TOKEN'),
        )
        async with AsyncCosClient(config, credentials) as client:
            uploaded = await client.upload_file(source, 'folder/a b.bin', content_type='application/octet-stream')
            assert uploaded.size == len(b'workflow output')
            assert uploaded.etag == '"upload-etag"'
            assert uploaded.request_id == 'put-request'
            assert put_attempts == 2

            assert await client.download_bytes('folder/a b.bin') == b'workflow output'
            downloaded = await client.download_file('folder/a b.bin', target)
            assert downloaded.size == len(b'workflow output')
            assert downloaded.etag == '"download-etag"'
            assert target.read_bytes() == b'workflow output'

            deleted = await client.delete_object('folder/a b.bin')
            assert deleted.request_id == 'delete-request'
            assert '/folder/a b.bin' not in stored

            with pytest.raises(CosServiceError) as exc_info:
                await client.download_bytes('missing.bin')
            assert exc_info.value.status == 404
            assert exc_info.value.code == 'NoSuchKey'
            assert exc_info.value.request_id == 'missing-request'

        for headers in requests:
            assert headers['x-cos-security-token'] == 'SESSION_TOKEN'
            assert headers['Authorization'].startswith('q-sign-algorithm=sha1&')
        assert 'content-length;content-type;host;x-cos-security-token' in requests[0]['Authorization']
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_client_does_not_close_external_session() -> None:
    config = CosConfig(
        bucket='example-1250000000',
        region='ap-guangzhou',
    )
    credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
    async with aiohttp.ClientSession() as session:
        client = AsyncCosClient(config, credentials, session=session)
        await client.close()
        assert not session.closed


@pytest.mark.asyncio
async def test_cos_timeout_is_read_idle_timeout_not_total_timeout() -> None:
    async def get_object(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={'Content-Length': '10'})
        await response.prepare(request)
        for byte in b'0123456789':
            await response.write(bytes([byte]))
            await asyncio.sleep(0.02)
        return response

    app = web.Application()
    app.router.add_get('/{key:.*}', get_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            timeout=0.05,
            connect_timeout=1,
            max_retries=0,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        async with AsyncCosClient(config, credentials) as client:
            assert await client.download_bytes('slow.bin') == b'0123456789'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_large_file_multipart_upload_and_ranged_download_retry_parts(tmp_path) -> None:
    mib = 1024 * 1024
    source_content = b'a' * mib + b'b' * mib + b'final-part'
    stored: dict[str, bytes] = {}
    multipart_parts: dict[int, bytes] = {}
    upload_attempts: dict[int, int] = {}
    range_attempts: dict[str, int] = {}
    ranged_if_match: list[str | None] = []
    create_content_type: str | None = None
    upload_id = 'test-upload-id'

    async def put_object(request: web.Request) -> web.Response:
        part_number = request.query.get('partNumber')
        if part_number is None:
            return web.Response(status=500)
        number = int(part_number)
        upload_attempts[number] = upload_attempts.get(number, 0) + 1
        body = await request.read()
        if number == 1 and upload_attempts[number] == 1:
            return web.Response(
                status=503,
                text='<Error><Code>SlowDown</Code><Message>retry part</Message></Error>',
                content_type='application/xml',
            )
        assert request.query['uploadId'] == upload_id
        multipart_parts[number] = body
        return web.Response(headers={'ETag': f'"part-{number}"'})

    async def post_object(request: web.Request) -> web.Response:
        nonlocal create_content_type
        assert request.headers.get('x-cos-forbid-overwrite') == 'true'
        if 'uploads' in request.query:
            create_content_type = request.headers.get('Content-Type')
            return web.Response(
                text=f'<InitiateMultipartUploadResult><UploadId>{upload_id}</UploadId></InitiateMultipartUploadResult>',
                content_type='application/xml',
            )

        assert request.query['uploadId'] == upload_id
        root = ElementTree.fromstring(await request.read())
        completed_parts = [
            (int(part.findtext('PartNumber', '0')), part.findtext('ETag')) for part in root.findall('Part')
        ]
        assert completed_parts == [(1, '"part-1"'), (2, '"part-2"'), (3, '"part-3"')]
        stored[request.path] = b''.join(multipart_parts[number] for number, _etag in completed_parts)
        return web.Response(
            text='<CompleteMultipartUploadResult><ETag>"complete-etag"</ETag></CompleteMultipartUploadResult>',
            content_type='application/xml',
            headers={'x-cos-request-id': 'complete-request'},
        )

    async def head_object(request: web.Request) -> web.Response:
        body = stored[request.path]
        return web.Response(
            headers={
                'Content-Length': str(len(body)),
                'ETag': '"complete-etag"',
                'x-cos-request-id': 'head-request',
            }
        )

    async def get_object(request: web.Request) -> web.Response:
        body = stored[request.path]
        range_value = request.headers.get('Range')
        assert range_value is not None
        ranged_if_match.append(request.headers.get('If-Match'))
        range_attempts[range_value] = range_attempts.get(range_value, 0) + 1
        if range_value == f'bytes={mib}-{2 * mib - 1}' and range_attempts[range_value] == 1:
            return web.Response(
                status=503,
                text='<Error><Code>SlowDown</Code><Message>retry range</Message></Error>',
                content_type='application/xml',
            )
        start_text, end_text = range_value.removeprefix('bytes=').split('-', 1)
        start = int(start_text)
        end = int(end_text)
        return web.Response(
            status=206,
            body=body[start : end + 1],
            headers={'Content-Range': f'bytes {start}-{end}/{len(body)}'},
        )

    app = web.Application(client_max_size=4 * mib)
    app.router.add_put('/{key:.*}', put_object)
    app.router.add_post('/{key:.*}', post_object)
    app.router.add_head('/{key:.*}', head_object)
    app.router.add_get('/{key:.*}', get_object, allow_head=False)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=1,
            retry_backoff=0,
            large_file_threshold=mib,
            part_size=mib,
            max_concurrency=2,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        source = tmp_path / 'source.bin'
        source.write_bytes(source_content)
        target = tmp_path / 'target.bin'

        async with AsyncCosClient(config, credentials) as client:
            uploaded = await client.upload_file(
                source,
                'large.bin',
                content_type='application/octet-stream',
                headers={'x-cos-forbid-overwrite': 'true'},
            )
            downloaded = await client.download_file('large.bin', target)

        assert uploaded.size == len(source_content)
        assert uploaded.etag == '"complete-etag"'
        assert uploaded.request_id == 'complete-request'
        assert create_content_type == 'application/octet-stream'
        assert upload_attempts == {1: 2, 2: 1, 3: 1}
        assert stored['/large.bin'] == source_content
        assert downloaded.size == len(source_content)
        assert downloaded.etag == '"complete-etag"'
        assert downloaded.request_id == 'head-request'
        assert target.read_bytes() == source_content
        assert range_attempts == {
            f'bytes=0-{mib - 1}': 1,
            f'bytes={mib}-{2 * mib - 1}': 2,
            f'bytes={2 * mib}-{len(source_content) - 1}': 1,
        }
        assert ranged_if_match == ['"complete-etag"'] * 4
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_multipart_upload_aborts_after_part_failure(tmp_path) -> None:
    mib = 1024 * 1024
    aborted_upload_ids: list[str] = []

    async def post_object(request: web.Request) -> web.Response:
        assert 'uploads' in request.query
        return web.Response(
            text='<InitiateMultipartUploadResult><UploadId>failed-upload</UploadId></InitiateMultipartUploadResult>',
            content_type='application/xml',
        )

    async def put_object(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(
            status=400,
            text='<Error><Code>InvalidPart</Code><Message>failed</Message></Error>',
            content_type='application/xml',
        )

    async def delete_object(request: web.Request) -> web.Response:
        aborted_upload_ids.append(request.query['uploadId'])
        return web.Response()

    app = web.Application(client_max_size=3 * mib)
    app.router.add_post('/{key:.*}', post_object)
    app.router.add_put('/{key:.*}', put_object)
    app.router.add_delete('/{key:.*}', delete_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=0,
            large_file_threshold=mib,
            part_size=mib,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        source = tmp_path / 'source.bin'
        source.write_bytes(b'x' * (mib + 1))

        async with AsyncCosClient(config, credentials) as client:
            with pytest.raises(CosServiceError, match='InvalidPart'):
                await client.upload_file(source, 'large.bin')

        assert aborted_upload_ids == ['failed-upload']
    finally:
        await server.close()


def test_select_multipart_part_size_enforces_cos_limits() -> None:
    mib = 1024 * 1024
    gib = 1024 * mib

    assert select_multipart_part_size(0, mib) == mib
    assert select_multipart_part_size(10_001 * mib, mib) == 2 * mib
    with pytest.raises(ValueError, match='at least'):
        select_multipart_part_size(gib, mib - 1)
    with pytest.raises(CosClientError, match='requires parts larger'):
        select_multipart_part_size((5 * gib * 10_000) + 1, mib)


@pytest.mark.asyncio
async def test_presigned_file_upload_streams_retries_and_preserves_etag(tmp_path) -> None:
    attempts = 0
    received = b''

    async def put_object(request: web.Request) -> web.Response:
        nonlocal attempts, received
        attempts += 1
        if attempts == 1:
            return web.Response(status=503, text='<Error><Code>SlowDown</Code></Error>')
        received = await request.read()
        return web.Response(headers={'ETag': '"opaque-etag"', 'x-cos-request-id': 'put-request'})

    app = web.Application()
    app.router.add_put('/upload', put_object)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'source.bin'
        source.write_bytes(b'presigned upload')
        async with aiohttp.ClientSession() as session:
            result = await upload_presigned_file(
                session,
                source,
                str(server.make_url('/upload')),
                key='source.bin',
                max_retries=1,
                retry_backoff=0,
            )

        assert attempts == 2
        assert received == b'presigned upload'
        assert result.etag == '"opaque-etag"'
        assert result.request_id == 'put-request'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_presigned_file_upload_rejects_change_while_url_is_signed(tmp_path) -> None:
    source = tmp_path / 'source.bin'
    source.write_bytes(b'before')
    expected_stat = source.stat()
    # The rewrite must change the recorded size: two same-length writes can land on an
    # identical st_mtime_ns (overlayfs/tmpfs timestamp granularity), leaving the state
    # indistinguishable and letting the upload proceed to the network.
    source.write_bytes(b'after the rewrite')
    assert source.stat().st_size != expected_stat.st_size

    async with aiohttp.ClientSession() as session:
        with pytest.raises(CosClientError, match='changed before upload'):
            await upload_presigned_file(
                session,
                source,
                'https://cos.example/upload',
                key='source.bin',
                expected_stat=expected_stat,
            )


@pytest.mark.asyncio
async def test_presigned_multipart_upload_reuses_streaming_retry_and_completion(tmp_path) -> None:
    mib = 1024 * 1024
    source_content = b'a' * mib + b'b' * mib + b'last'
    parts: dict[int, bytes] = {}
    attempts: dict[int, int] = {}
    completed: list[tuple[int, str | None]] = []
    aborts = 0

    async def put_part(request: web.Request) -> web.Response:
        number = int(request.match_info['number'])
        attempts[number] = attempts.get(number, 0) + 1
        body = await request.read()
        if number == 1 and attempts[number] == 1:
            return web.Response(status=503, text='<Error><Code>SlowDown</Code></Error>')
        parts[number] = body
        return web.Response(headers={'ETag': f'"part-{number}"'})

    async def complete(request: web.Request) -> web.Response:
        root = ElementTree.fromstring(await request.read())
        completed.extend(
            (int(part.findtext('PartNumber', '0')), part.findtext('ETag')) for part in root.findall('Part')
        )
        return web.Response(
            text='<CompleteMultipartUploadResult><ETag>"multipart-etag"</ETag></CompleteMultipartUploadResult>',
            content_type='application/xml',
            headers={'x-cos-request-id': 'complete-request'},
        )

    async def abort(_request: web.Request) -> web.Response:
        nonlocal aborts
        aborts += 1
        return web.Response()

    app = web.Application(client_max_size=2 * mib)
    app.router.add_put('/part/{number}', put_part)
    app.router.add_post('/complete', complete)
    app.router.add_delete('/abort', abort)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'source.bin'
        source.write_bytes(source_content)
        upload = {
            'part_urls': [str(server.make_url(f'/part/{number}')) for number in range(1, 4)],
            'complete_url': str(server.make_url('/complete')),
            'abort_url': str(server.make_url('/abort')),
            'part_size': mib,
        }
        async with aiohttp.ClientSession() as session:
            result = await upload_presigned_multipart_file(
                session,
                source,
                upload,
                key='folder/large.bin',
                max_concurrency=2,
                max_retries=1,
                retry_backoff=0,
            )

        assert attempts == {1: 2, 2: 1, 3: 1}
        assert b''.join(parts[number] for number in sorted(parts)) == source_content
        assert completed == [(1, '"part-1"'), (2, '"part-2"'), (3, '"part-3"')]
        assert aborts == 0
        assert result.size == len(source_content)
        assert result.etag == '"multipart-etag"'
        assert result.request_id == 'complete-request'
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_stage', ['part', 'complete'])
async def test_presigned_multipart_upload_aborts_part_and_complete_failures(tmp_path, fail_stage) -> None:
    mib = 1024 * 1024
    aborted = asyncio.Event()

    async def put_part(request: web.Request) -> web.Response:
        await request.read()
        if fail_stage == 'part':
            return web.Response(status=400, text='<Error><Code>InvalidPart</Code></Error>')
        return web.Response(headers={'ETag': '"part-1"'})

    async def complete(_request: web.Request) -> web.Response:
        return web.Response(status=400, text='<Error><Code>InvalidPartOrder</Code></Error>')

    async def abort(_request: web.Request) -> web.Response:
        aborted.set()
        return web.Response()

    app = web.Application(client_max_size=2 * mib)
    app.router.add_put('/part/1', put_part)
    app.router.add_post('/complete', complete)
    app.router.add_delete('/abort', abort)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'source.bin'
        source.write_bytes(b'x' * mib)
        upload = {
            'part_urls': [str(server.make_url('/part/1'))],
            'complete_url': str(server.make_url('/complete')),
            'abort_url': str(server.make_url('/abort')),
            'part_size': mib,
        }
        async with aiohttp.ClientSession() as session:
            with pytest.raises(CosServiceError):
                await upload_presigned_multipart_file(
                    session,
                    source,
                    upload,
                    key='large.bin',
                    max_retries=0,
                )

        assert aborted.is_set()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_presigned_multipart_upload_aborts_when_source_changes(tmp_path) -> None:
    mib = 1024 * 1024
    aborted = asyncio.Event()
    source = tmp_path / 'source.bin'
    source.write_bytes(b'x' * (2 * mib))

    async def put_part(request: web.Request) -> web.Response:
        number = int(request.match_info['number'])
        await request.read()
        if number == 1:
            source.write_bytes(b'y' * (2 * mib))
        return web.Response(headers={'ETag': f'"part-{number}"'})

    async def complete(_request: web.Request) -> web.Response:
        raise AssertionError('changed source must not be completed')

    async def abort(_request: web.Request) -> web.Response:
        aborted.set()
        return web.Response()

    app = web.Application(client_max_size=2 * mib)
    app.router.add_put('/part/{number}', put_part)
    app.router.add_post('/complete', complete)
    app.router.add_delete('/abort', abort)
    server = TestServer(app)
    await server.start_server()
    try:
        upload = {
            'part_urls': [str(server.make_url('/part/1')), str(server.make_url('/part/2'))],
            'complete_url': str(server.make_url('/complete')),
            'abort_url': str(server.make_url('/abort')),
            'part_size': mib,
        }
        async with aiohttp.ClientSession() as session:
            with pytest.raises(CosClientError, match='changed during upload'):
                await upload_presigned_multipart_file(
                    session,
                    source,
                    upload,
                    key='large.bin',
                    max_concurrency=1,
                    max_retries=0,
                )

        assert aborted.is_set()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_presigned_multipart_upload_cancellation_aborts_and_releases_parts(tmp_path) -> None:
    mib = 1024 * 1024
    part_started = asyncio.Event()
    release_part = asyncio.Event()
    aborted = asyncio.Event()

    async def put_part(request: web.Request) -> web.Response:
        part_started.set()
        await release_part.wait()
        await request.read()
        return web.Response(headers={'ETag': '"part-1"'})

    async def complete(_request: web.Request) -> web.Response:
        raise AssertionError('cancelled upload must not be completed')

    async def abort(_request: web.Request) -> web.Response:
        aborted.set()
        return web.Response()

    app = web.Application(client_max_size=2 * mib)
    app.router.add_put('/part/1', put_part)
    app.router.add_post('/complete', complete)
    app.router.add_delete('/abort', abort)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'source.bin'
        source.write_bytes(b'x' * mib)
        upload = {
            'part_urls': [str(server.make_url('/part/1'))],
            'complete_url': str(server.make_url('/complete')),
            'abort_url': str(server.make_url('/abort')),
            'part_size': mib,
        }
        async with aiohttp.ClientSession() as session:
            task = asyncio.create_task(
                upload_presigned_multipart_file(session, source, upload, key='large.bin', max_retries=0)
            )
            await asyncio.wait_for(part_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(aborted.wait(), timeout=1)
            release_part.set()
    finally:
        release_part.set()
        await server.close()


@pytest.mark.asyncio
async def test_upload_retries_http_400_request_timeout(tmp_path) -> None:
    """COS returns HTTP 400 with <Code>RequestTimeout</Code> when an upload stalls.

    Its status is neither 408/429 nor 5xx, so it used to fail the whole upload on
    the first attempt. It must now be retried via the error-code allowlist.
    """
    put_attempts = 0
    stored: dict[str, bytes] = {}

    async def put_object(request: web.Request) -> web.Response:
        nonlocal put_attempts
        put_attempts += 1
        if put_attempts < 3:
            return web.Response(
                status=400,
                text='<Error><Code>RequestTimeout</Code><Message>slow</Message></Error>',
                content_type='application/xml',
            )
        stored[request.path] = await request.read()
        return web.Response(headers={'ETag': '"ok"', 'x-cos-request-id': 'put-ok'})

    app = web.Application()
    app.router.add_put('/{key:.*}', put_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=3,
            retry_backoff=0,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        async with AsyncCosClient(config, credentials) as client:
            uploaded = await client.upload_bytes(b'payload', 'folder/obj.bin')
            assert uploaded.request_id == 'put-ok'
        assert put_attempts == 3  # two RequestTimeout responses retried, third succeeds
        assert stored['/folder/obj.bin'] == b'payload'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_upload_retries_user_network_too_slow(tmp_path) -> None:
    """COS returns <Code>UserNetworkTooSlow</Code> when the observed upload rate
    dips below its minimum threshold -- common under very high concurrency where
    a single 20MB PUT is starved of egress bandwidth at the congestion peak.

    Its status is neither 408/429 nor 5xx, so before the allowlist entry it failed
    the whole upload on the first attempt (observed as attempt=0 task failures in
    the 20x2500 stress run). It must now be retried via the error-code allowlist.
    """
    put_attempts = 0
    stored: dict[str, bytes] = {}

    async def put_object(request: web.Request) -> web.Response:
        nonlocal put_attempts
        put_attempts += 1
        if put_attempts < 3:
            return web.Response(
                status=403,
                text='<Error><Code>UserNetworkTooSlow</Code><Message>too slow</Message></Error>',
                content_type='application/xml',
            )
        stored[request.path] = await request.read()
        return web.Response(headers={'ETag': '"ok"', 'x-cos-request-id': 'put-ok'})

    app = web.Application()
    app.router.add_put('/{key:.*}', put_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=3,
            retry_backoff=0,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        async with AsyncCosClient(config, credentials) as client:
            uploaded = await client.upload_bytes(b'payload', 'folder/obj.bin')
            assert uploaded.request_id == 'put-ok'
        assert put_attempts == 3  # two UserNetworkTooSlow responses retried, third succeeds
        assert stored['/folder/obj.bin'] == b'payload'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_upload_does_not_retry_non_retryable_http_400(tmp_path) -> None:
    """A non-transient HTTP 400 (e.g. AccessDenied) must still fail immediately."""
    put_attempts = 0

    async def put_object(_request: web.Request) -> web.Response:
        nonlocal put_attempts
        put_attempts += 1
        return web.Response(
            status=400,
            text='<Error><Code>AccessDenied</Code><Message>nope</Message></Error>',
            content_type='application/xml',
        )

    app = web.Application()
    app.router.add_put('/{key:.*}', put_object)
    server = TestServer(app)
    await server.start_server()
    try:
        config = CosConfig(
            bucket='local',
            endpoint=str(server.make_url('/')).rstrip('/'),
            max_retries=3,
            retry_backoff=0,
        )
        credentials = CosCredentials(secret_id=SecretStr('id'), secret_key=SecretStr('key'))
        async with AsyncCosClient(config, credentials) as client:
            with pytest.raises(CosServiceError) as exc_info:
                await client.upload_bytes(b'payload', 'folder/obj.bin')
        assert exc_info.value.status == 400
        assert exc_info.value.code == 'AccessDenied'
        assert put_attempts == 1  # not retried
    finally:
        await server.close()
