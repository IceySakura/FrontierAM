import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.utils.artifact.client import ArtifactClient
from nexus.utils.artifact.models import UploadSession
from nexus.utils.cos import upload_presigned_file


@pytest.mark.asyncio
async def test_upload_file_forwards_session_headers_to_cos(monkeypatch, tmp_path):
    # The backend returns header keys in lowercase (e.g. ``content-type``), as real
    # JSON responses do. The pre-signed COS URL is signed with those headers in its
    # ``q-header-list``, so they must be echoed back verbatim -- otherwise COS
    # returns SignatureDoesNotMatch.
    session = UploadSession.from_dict({
        'artifact_id': 'art_01KZ5CT6E6QFB7VR01N6RDWPRV',
        'upload_id': 'up_01',
        'upload': {
            'method': 'PUT',
            'url': 'https://cos.example/art_01KZ5CT6E6QFB7VR01N6RDWPRV',
            'headers': {'content-type': 'application/octet-stream'},
        },
    })

    captured: dict = {}

    async def fake_upload_presigned_file(_session, _source_path, _url, *, key, **kwargs):
        captured['key'] = key
        captured['extra_headers'] = kwargs.get('extra_headers')

    monkeypatch.setattr(
        'nexus.utils.artifact.client.upload_presigned_file',
        fake_upload_presigned_file,
    )

    client = ArtifactClient(base_url='http://unused')
    client.session = object()  # any non-None value satisfies _require_session

    source = tmp_path / 'uv.lock'
    source.write_bytes(b'lockfile-content')

    await client._put_file_to_cos(session, source)

    assert captured['key'] == 'art_01KZ5CT6E6QFB7VR01N6RDWPRV'
    # The backend-supplied lowercase content-type must reach COS.
    assert captured['extra_headers'].get('content-type') == 'application/octet-stream'


@pytest.mark.asyncio
async def test_upload_presigned_file_forwards_extra_headers(tmp_path):
    # End-to-end check that``upload_presigned_file`` actually sends caller-supplied
    # headers on the wire so that COS pre-signed signature verification succeeds.
    received_headers: dict[str, str] = {}

    async def put_object(request: web.Request) -> web.Response:
        received_headers.update(request.headers)
        await request.read()
        return web.Response(headers={'ETag': '"ok"', 'x-cos-request-id': 'r'})

    app = web.Application()
    app.router.add_put('/upload', put_object)
    server = TestServer(app)
    await server.start_server()
    try:
        source = tmp_path / 'src.bin'
        source.write_bytes(b'body')
        async with aiohttp.ClientSession() as session:
            await upload_presigned_file(
                session,
                source,
                str(server.make_url('/upload')),
                key='src.bin',
                extra_headers={'content-type': 'application/octet-stream'},
                max_retries=0,
                retry_backoff=0,
            )
    finally:
        await server.close()

    # aiohttp exposes header names case-insensitively; assert the value survived.
    assert received_headers.get('Content-Type') == 'application/octet-stream'
    assert received_headers.get('Content-Length') == '4'


@pytest.mark.asyncio
async def test_whoami_is_cached_across_calls():
    """``whoami()`` should hit the wire once per client and reuse the value."""
    calls: list[tuple] = []

    class _Client(ArtifactClient):
        async def _request(self, method, path, *, params=None, json=None, headers=None):  # noqa: ARG002
            calls.append((method, path))
            return {'employee_id': 'junlinyin'}

    client = _Client(base_url='http://unused')
    client.session = object()

    assert await client.whoami() == 'junlinyin'
    assert await client.whoami() == 'junlinyin'
    assert calls == [('GET', '/v1/whoami')]


@pytest.mark.asyncio
async def test_resolve_path_bare_uses_owned_view_only_at_top_level():
    """A bare slash-path picks the caller's own namespace (view=None) at the top level."""
    calls: list[dict] = []

    class FakePage:
        def __init__(self, items, next_cursor=None):
            self.items = items
            self.next_cursor = next_cursor

    async def fake_list_resources(*, parent_folder_id, view=None, limit=200, cursor=None, name=None):
        calls.append({'parent_folder_id': parent_folder_id, 'view': view})
        if parent_folder_id is None:
            return FakePage([{'resource_id': 'fid_test', 'name': 'test', 'resource_type': 'folder', 'owner_id': 'me'}])
        return FakePage([{'resource_id': 'fid_son', 'name': 'son', 'resource_type': 'folder', 'owner_id': 'me'}])

    client = ArtifactClient(base_url='http://unused')
    client.session = object()
    client.list_resources = fake_list_resources
    # Prime the whoami cache so bare-path resolution doesn't hit the network.
    client._whoami_cache = 'me'

    resolved = await client.resolve_resource_id('/test/son', expect='folder')

    assert resolved == 'fid_son'
    # Top-level lookup uses the owned view (view=None); the nested lookup also
    # sends no view -- the server rejects view combined with parent_folder_id.
    assert calls == [
        {'parent_folder_id': None, 'view': None},
        {'parent_folder_id': 'fid_test', 'view': None},
    ]


@pytest.mark.asyncio
async def test_resolve_path_uri_other_owner_uses_shared_view_and_owner_filter():
    """``artifact:///<other>/...`` picks the shared view at top level and filters by owner.

    This is the same-name disambiguation case: two grantors have granted a
    file called ``test.sh``; the URI names one of them, and the client-side
    owner filter picks that one row deterministically.
    """
    calls: list[dict] = []

    class FakePage:
        def __init__(self, items, next_cursor=None):
            self.items = items
            self.next_cursor = next_cursor

    async def fake_list_resources(*, parent_folder_id, view=None, limit=200, cursor=None, name=None):
        calls.append({'parent_folder_id': parent_folder_id, 'view': view})
        # Same-named entries owned by two different employees, as the shared
        # view actually returns in the wild.
        return FakePage([
            {'resource_id': 'art_a', 'name': 'test.sh', 'resource_type': 'artifact', 'owner_id': 'alice'},
            {'resource_id': 'art_b', 'name': 'test.sh', 'resource_type': 'artifact', 'owner_id': 'bob'},
        ])

    client = ArtifactClient(base_url='http://unused')
    client.session = object()
    client.list_resources = fake_list_resources
    client._whoami_cache = 'me'

    resolved = await client.resolve_resource_id('artifact:///alice/test.sh', expect='artifact')

    assert resolved == 'art_a'  # owner filter picked alice's, not bob's
    assert calls == [{'parent_folder_id': None, 'view': 'shared'}]


@pytest.mark.asyncio
async def test_resolve_path_uri_self_owner_uses_owned_view():
    """``artifact:///<self>/...`` collapses to the owned view -- no shared query."""
    calls: list[dict] = []

    class FakePage:
        def __init__(self, items, next_cursor=None):
            self.items = items
            self.next_cursor = next_cursor

    async def fake_list_resources(*, parent_folder_id, view=None, limit=200, cursor=None, name=None):
        calls.append({'parent_folder_id': parent_folder_id, 'view': view})
        return FakePage([{'resource_id': 'art_x', 'name': 'x.bin', 'resource_type': 'artifact', 'owner_id': 'me'}])

    client = ArtifactClient(base_url='http://unused')
    client.session = object()
    client.list_resources = fake_list_resources
    client._whoami_cache = 'me'

    resolved = await client.resolve_resource_id('artifact:///me/x.bin', expect='artifact')

    assert resolved == 'art_x'
    assert calls == [{'parent_folder_id': None, 'view': None}]


@pytest.mark.asyncio
async def test_resolve_path_rejects_aggregated_shared_view():
    """``artifact:///`` (empty path) is a listing alias and cannot address a resource."""
    from nexus.utils.artifact.errors import ArtifactPathError

    client = ArtifactClient(base_url='http://unused')
    client.session = object()
    client._whoami_cache = 'me'

    with pytest.raises(ArtifactPathError):
        await client.resolve_resource_id('artifact:///', expect='artifact')
