"""Client SDK for the artifact service.

Programmatic access to the artifact service: folders, artifacts, sharing grants,
and high-level upload/download helpers that stream file content through
pre-signed Tencent COS URLs.

Example::

    from nexus.utils.artifact import TAIHU_GATEWAY_URL, ArtifactClient

    async with ArtifactClient(base_url=TAIHU_GATEWAY_URL, taihu_pat='tai_pat_...') as client:
        folder = await client.create_folder('models')
        artifact = await client.upload_file('model.bin', parent_folder_id=folder.folder_id)
        await client.download_to_file(artifact.artifact_id, 'restored.bin')
"""

from nexus.utils.artifact.auth import (
    ARTIFACT_SHARED_VALUE_NAME,
    ArtifactCredentials,
    clear_taihu_pat,
    load_artifact_credentials,
    load_taihu_pat,
    normalize_pat,
    save_taihu_pat,
    taihu_authorization_header,
    taihu_pat_file,
)
from nexus.utils.artifact.client import TAIHU_GATEWAY_URL, ArtifactClient
from nexus.utils.artifact.errors import (
    ArtifactClientError,
    ArtifactError,
    ArtifactPathError,
    ArtifactUploadError,
)
from nexus.utils.artifact.models import ArtifactInfo, DownloadUrl, Folder, Grant, Page, UploadSession
from nexus.utils.artifact.uploader import (
    RunFolder,
    artifact_uri_for,
    folder_uri,
    open_run_folder_from_runtime,
    prepare_run_folder,
    upload_bytes,
    upload_file,
)
from nexus.utils.artifact.uri import (
    ParsedUri,
    format_aggregated_shared_uri,
    format_artifact_uri,
    parse_artifact_uri,
)


__all__ = [
    'ARTIFACT_SHARED_VALUE_NAME',
    'ArtifactClient',
    'ArtifactCredentials',
    'ArtifactInfo',
    'ArtifactClientError',
    'ArtifactError',
    'ArtifactPathError',
    'ArtifactUploadError',
    'DownloadUrl',
    'Folder',
    'Grant',
    'Page',
    'ParsedUri',
    'RunFolder',
    'TAIHU_GATEWAY_URL',
    'UploadSession',
    'artifact_uri_for',
    'clear_taihu_pat',
    'folder_uri',
    'format_aggregated_shared_uri',
    'format_artifact_uri',
    'load_artifact_credentials',
    'load_taihu_pat',
    'normalize_pat',
    'open_run_folder_from_runtime',
    'parse_artifact_uri',
    'prepare_run_folder',
    'save_taihu_pat',
    'taihu_authorization_header',
    'taihu_pat_file',
    'upload_bytes',
    'upload_file',
]
