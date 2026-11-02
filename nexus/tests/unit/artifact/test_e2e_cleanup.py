"""Deterministic tests for Artifact E2E resource cleanup."""

from types import SimpleNamespace

import pytest

from tests.e2e.artifact import test_sdk_e2e as sdk_e2e
from tests.e2e.artifact.test_sdk_e2e import _TrackingArtifactClient, _cleanup_e2e_resources

from nexus.utils.artifact import ArtifactClient, ArtifactClientError, UploadSession


def _ready_upload_session() -> UploadSession:
    return UploadSession(
        artifact_id='art_1',
        upload_id='upl_1',
        status='ready',
        method='PUT',
        url='https://cos/upload',
        headers={},
    )


class _CleanupClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.folder_delete_results: list[Exception | None] = []
        self.artifact_delete_error: Exception | None = None
        self.complete_upload_error: Exception | None = None

    async def list_resources(self, *, parent_folder_id=None, view=None, limit=50, cursor=None):
        self.calls.append(('list_resources', parent_folder_id, view, limit, cursor))
        if parent_folder_id is None:
            items = [{'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'folder-name'}]
        else:
            items = [{'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'artifact-name'}]
        return SimpleNamespace(items=items, next_cursor=None)

    async def delete_grant(self, resource_id, grantee_id, *, permission):
        self.calls.append(('delete_grant', resource_id, grantee_id, permission))

    async def _put_to_cos(self, upload_session, payload):
        self.calls.append(('put_to_cos', upload_session.artifact_id, payload))

    async def complete_upload(self, upload_id):
        self.calls.append(('complete_upload', upload_id))
        if self.complete_upload_error is not None:
            raise self.complete_upload_error

    async def delete_artifact(self, artifact_id):
        self.calls.append(('delete_artifact', artifact_id))
        if self.artifact_delete_error is not None:
            raise self.artifact_delete_error

    async def delete_folder(self, folder_id):
        self.calls.append(('delete_folder', folder_id))
        result = self.folder_delete_results.pop(0) if self.folder_delete_results else None
        if result is not None:
            raise result


async def test_tracking_client_records_created_upload_session(monkeypatch):
    upload_session = _ready_upload_session()

    async def create_upload(self, **kwargs):
        return upload_session

    monkeypatch.setattr(ArtifactClient, 'create_upload', create_upload)
    client = _TrackingArtifactClient(base_url='http://artifact.test', taihu_pat='token')
    async with client:
        result = await client.create_upload(name='artifact-name', size=7)

    assert result is upload_session
    assert client.upload_session is upload_session


async def test_cleanup_recovers_ids_and_retries_only_folder_not_empty(monkeypatch):
    monkeypatch.setattr('tests.e2e.artifact.test_sdk_e2e._CLEANUP_RETRY_SECONDS', 0)
    client = _CleanupClient()
    client.folder_delete_results = [
        ArtifactClientError('not empty', code='folder_not_empty', http_status=409),
        None,
    ]

    error = await _cleanup_e2e_resources(
        client,
        folder_id=None,
        artifact_id=None,
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=False,
    )

    assert error is None
    assert client.calls == [
        ('list_resources', None, 'owned', 100, None),
        ('list_resources', 'fld_1', None, 100, None),
        ('delete_artifact', 'art_1'),
        ('delete_folder', 'fld_1'),
        ('delete_folder', 'fld_1'),
    ]


async def test_cleanup_completes_pending_upload_before_deleting():
    client = _CleanupClient()
    upload_session = UploadSession(
        artifact_id='art_1',
        upload_id='upl_1',
        status='pending',
        method='PUT',
        url='https://cos/upload',
        headers={},
    )

    error = await _cleanup_e2e_resources(
        client,
        folder_id='fld_1',
        artifact_id=None,
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=False,
        upload_session=upload_session,
        upload_payload=b'payload',
    )

    assert error is None
    assert client.calls == [
        ('put_to_cos', 'art_1', b'payload'),
        ('complete_upload', 'upl_1'),
        ('delete_artifact', 'art_1'),
        ('delete_folder', 'fld_1'),
    ]


async def test_cleanup_stops_folder_polling_when_pending_upload_cannot_complete():
    client = _CleanupClient()
    complete_error = RuntimeError('complete failed')
    client.complete_upload_error = complete_error
    client.artifact_delete_error = ArtifactClientError('pending', code='invalid_state', http_status=409)
    upload_session = UploadSession(
        artifact_id='art_1',
        upload_id='upl_1',
        status='pending',
        method='PUT',
        url='https://cos/upload',
        headers={},
    )

    error = await _cleanup_e2e_resources(
        client,
        folder_id='fld_1',
        artifact_id=None,
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=False,
        upload_session=upload_session,
        upload_payload=b'payload',
    )

    assert error is complete_error
    assert [call[0] for call in client.calls] == ['put_to_cos', 'complete_upload', 'delete_artifact']


async def test_cleanup_does_not_retry_other_conflict():
    client = _CleanupClient()
    conflict = ArtifactClientError('wrong state', code='invalid_state', http_status=409)
    client.folder_delete_results = [conflict]

    error = await _cleanup_e2e_resources(
        client,
        folder_id='fld_1',
        artifact_id='art_1',
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=False,
    )

    assert error is conflict
    assert client.calls == [('delete_artifact', 'art_1'), ('delete_folder', 'fld_1')]


async def test_cleanup_continues_to_folder_after_artifact_delete_failure():
    client = _CleanupClient()
    delete_error = RuntimeError('artifact delete failed')
    client.artifact_delete_error = delete_error

    error = await _cleanup_e2e_resources(
        client,
        folder_id='fld_1',
        artifact_id='art_1',
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=False,
    )

    assert error is delete_error
    assert client.calls == [('delete_artifact', 'art_1'), ('delete_folder', 'fld_1')]


async def test_main_preserves_business_error_when_cleanup_also_fails(monkeypatch):
    business_error = RuntimeError('create failed')
    cleanup_error = RuntimeError('cleanup failed')

    async def create_folder(self, **kwargs):
        raise business_error

    async def cleanup(*args, **kwargs):
        return cleanup_error

    monkeypatch.setattr(_TrackingArtifactClient, 'create_folder', create_folder)
    monkeypatch.setattr(sdk_e2e, '_cleanup_e2e_resources', cleanup)

    with pytest.raises(RuntimeError, match='create failed') as caught:
        await sdk_e2e.main()

    assert caught.value is business_error


async def test_cleanup_reports_timeout_without_hiding_first_error(monkeypatch):
    monkeypatch.setattr('tests.e2e.artifact.test_sdk_e2e.CLEANUP_TIMEOUT_SECONDS', 0)
    client = _CleanupClient()
    grant_error = RuntimeError('grant delete failed')

    async def delete_grant(*args, **kwargs):
        client.calls.append(('delete_grant', *args))
        raise grant_error

    client.delete_grant = delete_grant
    client.folder_delete_results = [
        ArtifactClientError('not empty', code='folder_not_empty', http_status=409),
    ]

    error = await _cleanup_e2e_resources(
        client,
        folder_id='fld_1',
        artifact_id='art_1',
        folder_name='folder-name',
        artifact_name='artifact-name',
        grant_created=True,
    )

    assert error is grant_error
    assert [call[0] for call in client.calls] == ['delete_grant', 'delete_artifact', 'delete_folder']
