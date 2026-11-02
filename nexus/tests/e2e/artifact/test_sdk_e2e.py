#!/usr/bin/env python3
"""SDK 端到端测试：用 nexus.utils.artifact 客户端真实打通「上传 -> 分享 -> 下载」。

本文件站在 **客户端 SDK 视角**，只用 `nexus.utils.artifact.ArtifactClient` 的高阶
helper（upload_bytes / download_bytes / put_grant / create_download_url）走完整链路，
验证 SDK 封装对真实服务的可用性。

认证：test 环境通过太湖网关访问，使用 `ARTIFACT_TAIHU_PAT`；本地或开发环境直连时，
可改用 `ARTIFACT_DEV_EMPLOYEE`，但后端必须开启 `taihu.allow_dev_identity_header`。

运行（pytest 自动收集，无 ARTIFACT_E2E 时跳过）：
  export ARTIFACT_E2E=1
  export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
  export ARTIFACT_TAIHU_PAT=tai_pat_xxxxxxx
  uv run pytest tests/e2e/artifact/test_sdk_e2e.py -m e2e
前提：artifact-service 可访问且对应存储桶对服务可写。
"""

from __future__ import annotations

import asyncio
import os
import uuid

from datetime import datetime, timezone

import pytest

from nexus.utils.artifact import ArtifactClient, ArtifactClientError, UploadSession


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv('ARTIFACT_E2E'),
        reason='ARTIFACT_E2E not set (needs running service + identity + real COS bucket)',
    ),
]

BASE_URL = os.environ.get('ARTIFACT_SERVICE_URL', 'http://localhost:8080')
TAIHU_PAT = os.environ.get('ARTIFACT_TAIHU_PAT')
DEV_EMPLOYEE = os.environ.get('ARTIFACT_DEV_EMPLOYEE')
GRANTEE = os.environ.get('ARTIFACT_GRANTEE', 'bob')  # 演示：把 read 授权给另一名员工
CLEANUP_TIMEOUT_SECONDS = float(os.environ.get('ARTIFACT_E2E_CLEANUP_TIMEOUT', '90'))
_CLEANUP_RETRY_SECONDS = 2.0


def _banner(title: str) -> None:
    print(f'\n=== {title} ===')


class _TrackingArtifactClient(ArtifactClient):
    upload_session: UploadSession | None = None

    async def create_upload(self, *args, **kwargs) -> UploadSession:
        session = await super().create_upload(*args, **kwargs)
        self.upload_session = session
        return session


async def _find_resource_id(
    client: ArtifactClient,
    *,
    name: str,
    resource_type: str,
    parent_folder_id: str | None,
) -> str | None:
    cursor: str | None = None
    while True:
        page = await client.list_resources(
            parent_folder_id=parent_folder_id,
            view='owned' if parent_folder_id is None else None,
            limit=100,
            cursor=cursor,
        )
        for item in page.items:
            if item.get('resource_type') == resource_type and item.get('name') == name:
                resource_id = item.get('resource_id')
                return str(resource_id) if resource_id else None
        cursor = page.next_cursor
        if not cursor:
            return None


async def _cleanup_e2e_resources(
    client: ArtifactClient,
    *,
    folder_id: str | None,
    artifact_id: str | None,
    folder_name: str,
    artifact_name: str,
    grant_created: bool,
    upload_session: UploadSession | None = None,
    upload_payload: bytes | None = None,
) -> Exception | None:
    _banner('cleanup')
    first_error: Exception | None = None
    pending_artifact = False

    if folder_id is None:
        try:
            folder_id = await _find_resource_id(
                client,
                name=folder_name,
                resource_type='folder',
                parent_folder_id=None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] folder recovery failed: {exc!r}')
            first_error = exc

    if upload_session is not None:
        artifact_id = artifact_id or upload_session.artifact_id
        if not upload_session.is_ready:
            try:
                if upload_payload is None:
                    raise RuntimeError('upload payload is unavailable for cleanup recovery')
                if upload_session.upload_id is None:
                    raise RuntimeError('upload session has no upload_id for cleanup recovery')
                await client._put_to_cos(upload_session, upload_payload)
                await client.complete_upload(upload_session.upload_id)
                print('[OK] pending upload completed for cleanup')
            except Exception as exc:  # noqa: BLE001
                print(f'[WARN] pending upload recovery failed: {exc!r}')
                first_error = first_error or exc
                pending_artifact = True

    if artifact_id is None and folder_id is not None:
        try:
            artifact_id = await _find_resource_id(
                client,
                name=artifact_name,
                resource_type='artifact',
                parent_folder_id=folder_id,
            )
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] artifact recovery failed: {exc!r}')
            first_error = first_error or exc

    if grant_created and folder_id is not None:
        try:
            await client.delete_grant(folder_id, GRANTEE, permission='read')
            print('[OK] grant revoked')
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] delete_grant failed: {exc!r}')
            first_error = exc

    if artifact_id is not None:
        try:
            await client.delete_artifact(artifact_id)
            print('[OK] delete_artifact requested (async 202)')
            pending_artifact = False
        except ArtifactClientError as exc:
            print(f'[WARN] delete_artifact failed: {exc!r}')
            first_error = first_error or exc
            if exc.http_status == 409 and exc.code == 'invalid_state':
                pending_artifact = True
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] delete_artifact failed: {exc!r}')
            first_error = first_error or exc

    if folder_id is None:
        return first_error
    if pending_artifact:
        print('[WARN] pending artifact cannot be deleted; server maintenance will expire it')
        return first_error

    loop = asyncio.get_running_loop()
    deadline = loop.time() + CLEANUP_TIMEOUT_SECONDS
    while True:
        try:
            await client.delete_folder(folder_id)
            print('[OK] delete_folder ok')
            return first_error
        except ArtifactClientError as exc:
            retryable = exc.http_status == 409 and exc.code == 'folder_not_empty'
            if not retryable:
                print(f'[WARN] delete_folder failed: {exc!r}')
                return first_error or exc
            remaining = deadline - loop.time()
            if remaining <= 0:
                timeout = RuntimeError(
                    f'timed out after {CLEANUP_TIMEOUT_SECONDS:g}s waiting to delete folder {folder_id}'
                )
                print(f'[WARN] {timeout}')
                return first_error or timeout
            await asyncio.sleep(min(_CLEANUP_RETRY_SECONDS, remaining))
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] delete_folder failed: {exc!r}')
            return first_error or exc


async def main() -> int:
    pass_count = 0
    folder_id: str | None = None
    artifact_id: str | None = None
    grant_created = False
    cleanup_error: Exception | None = None
    run_id = uuid.uuid4().hex
    folder_name = f'sdk-e2e-{run_id}'
    artifact_name = f'sdk-e2e-{run_id}.txt'

    async with _TrackingArtifactClient(base_url=BASE_URL, taihu_pat=TAIHU_PAT, dev_employee=DEV_EMPLOYEE) as client:
        try:
            # 准备一个真实小文件（md5 由 SDK 内部计算，这里仅用于回环校验）
            payload = b'sdk e2e round-trip - ' + datetime.now(timezone.utc).isoformat().encode()

            # 1) 创建文件夹
            _banner('POST /v1/folders  (create_folder)')
            folder = await client.create_folder(name=folder_name)
            assert folder.folder_id
            folder_id = folder.folder_id
            print(f'[OK] folder_id={folder_id}')
            pass_count += 1

            # 2) 上传（高层 helper：create_upload -> COS PUT -> complete 一步到位）
            _banner('upload_bytes  (create_upload -> COS PUT -> complete)')
            artifact = await client.upload_bytes(
                payload,
                name=artifact_name,
                parent_folder_id=folder_id,
                content_type='text/plain',
            )
            assert artifact.artifact_id
            artifact_id = artifact.artifact_id
            assert artifact.status == 'ready', f'artifact not ready: {artifact.status}'
            print(f'[OK] artifact_id={artifact_id} status={artifact.status}')
            pass_count += 1

            # 3) 授权（所有者 -> 另一名员工 read，即「分享」）
            _banner(f'PUT /v1/resources/{folder_id}/grants/read/{GRANTEE}  (put_grant)')
            await client.put_grant(folder_id, GRANTEE, permission='read')
            grant_created = True
            grants = await client.list_grants(folder_id)
            assert any(g.grantee_id == GRANTEE and g.permission == 'read' for g in grants), 'grant not visible'
            print(f'[OK] grants={[(g.grantee_id, g.permission) for g in grants]}')
            pass_count += 1

            # 4) 获取下载链接
            _banner(f'POST /v1/artifacts/{artifact_id}/download-url  (create_download_url)')
            dl = await client.create_download_url(artifact_id)
            assert dl.url
            print(f'[OK] method={dl.method} expires_in={dl.expires_in}')
            pass_count += 1

            # 5) 通过预签名链接下载并做内容回环校验
            _banner('GET <presigned url>  (download_bytes)')
            restored = await client.download_bytes(artifact_id)
            assert restored == payload, f'content mismatch: {restored!r} != {payload!r}'
            print(f'[OK] downloaded {len(restored)} bytes, content matches')
            pass_count += 1

            # 6) 列表视图也能看到刚上传的 artifact
            _banner('GET /v1/artifacts?view=owned  (list_artifacts)')
            page = await client.list_artifacts(view='owned', limit=20)
            assert any(item.get('artifact_id') == artifact_id for item in page.items), 'artifact missing from listing'
            print(f'[OK] listed {len(page.items)} owned artifact(s)')
            pass_count += 1
        finally:
            cleanup_error = await _cleanup_e2e_resources(
                client,
                folder_id=folder_id,
                artifact_id=artifact_id,
                folder_name=folder_name,
                artifact_name=artifact_name,
                grant_created=grant_created,
                upload_session=client.upload_session if artifact_id is None else None,
                upload_payload=payload,
            )

    if cleanup_error is not None:
        raise cleanup_error
    print(f'\n=== SDK e2e 完成: PASS={pass_count} FAIL=0 ===')
    return 0


def test_artifact_sdk_e2e() -> None:
    """End-to-end upload->share->download via the real nexus.utils.artifact SDK.

    Auto-collected by pytest (marker=e2e); skipped unless ARTIFACT_E2E is set.
    """
    assert asyncio.run(main()) == 0
