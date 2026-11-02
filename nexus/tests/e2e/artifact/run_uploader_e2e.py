#!/usr/bin/env python3
"""Standalone E2E script: test the functional uploader API without a sandbox.

Validates the complete upload flow on the HOST (no sandbox needed):
  1. ``prepare_run_folder(client)`` builds the folder hierarchy
     ``/tasks/YYYY/MM/DD/<run>/``.
  2. ``upload_bytes(client, folder, ...)`` stores data in COS via the
     artifact service.
  3. The artifact is readable and downloadable.

Usage::

    # Option A – use a locally-stored PAT (run once to persist):
    #   nexus artifact login
    # Option B – pass via environment variable:
    #   export ARTIFACT_TAIHU_PAT=tai_pat_...
    # Option C – dev backend without Taihu gateway:
    #   export ARTIFACT_DEV_EMPLOYEE=<your-rtx>
    export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
    uv run python tests/e2e/artifact/run_uploader_e2e.py

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path


# ---- resolve REPO_ROOT so the script works from any cwd --------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
# ----------------------------------------------------------------------------


SERVICE_URL = os.environ.get('ARTIFACT_SERVICE_URL', 'https://artifact-nexus-agent.mcp.woa.com')
DEV_EMPLOYEE = os.environ.get('ARTIFACT_DEV_EMPLOYEE', '')


def _banner(title: str) -> None:
    print(f'\n=== {title} ===')


async def main() -> int:
    from nexus.utils.artifact import ArtifactClient
    from nexus.utils.artifact.auth import load_taihu_pat
    from nexus.utils.artifact.uploader import (
        RunFolder,
        artifact_uri_for,
        prepare_run_folder,
        upload_bytes,
    )

    taihu_pat = load_taihu_pat()  # env $ARTIFACT_TAIHU_PAT → ~/.nexus/auth/taihu
    if not taihu_pat and not DEV_EMPLOYEE:
        print(
            'ERROR: no credentials found. '
            'Run `nexus artifact login`, set $ARTIFACT_TAIHU_PAT, or set $ARTIFACT_DEV_EMPLOYEE',
            file=sys.stderr,
        )
        return 1

    suffix = uuid.uuid4().hex[:8]
    content = f'artifact_output e2e {suffix}'.encode()
    artifact_name = f'e2e-uploader-{suffix}.txt'

    artifact_id: str | None = None
    folder: RunFolder | None = None

    async with ArtifactClient(
        base_url=SERVICE_URL,
        taihu_pat=taihu_pat or None,
        dev_employee=DEV_EMPLOYEE or None,
    ) as client:
        try:
            # 1) Prepare run folder (builds /tasks/YYYY/MM/DD/<hostname-timestamp>/)
            _banner('prepare_run_folder (folder hierarchy)')
            now = datetime.now(timezone.utc)
            folder = await prepare_run_folder(client, dt=now)
            print(f'[OK] folder_id={folder.folder_id}')
            print(f'[OK] folder_path={folder.folder_path}')

            # 2) Upload bytes
            _banner('Upload bytes via upload_bytes(client, folder, ...)')
            artifact = await upload_bytes(client, folder, content, name=artifact_name, content_type='text/plain')
            artifact_id = artifact.artifact_id
            assert artifact.status == 'ready', f'artifact not ready: {artifact.status}'
            print(f'[OK] artifact_id={artifact_id} status={artifact.status}')

            full_uri = artifact_uri_for(folder, artifact_name)
            print(f'[OK] artifact_uri={full_uri}')

            # 3) Verify download
            _banner('Verify download')
            dl = await client.create_download_url(artifact_id)
            assert dl.url, 'download URL is empty'
            print(f'[OK] download url obtained (expires_in={dl.expires_in}s)')

            restored = await client.download_bytes(artifact_id)
            assert restored == content, f'content mismatch: {restored!r} != {content!r}'
            print(f'[OK] downloaded {len(restored)} bytes, content matches')

            # 4) Verify task output model
            _banner('Verify TaskRunOutput artifact fields serialisation')
            from nexus.core.task import TaskRunOutput

            class _MockOutput(TaskRunOutput):
                artifact_id: str | None = None
                artifact_uri: str | None = None

            out = _MockOutput(artifact_id=artifact_id, artifact_uri=full_uri)
            d = out.model_dump()
            assert d['artifact_id'] == artifact_id
            assert d['artifact_uri'] == full_uri
            print(f'[OK] model_dump() → {d}')

        finally:
            # cleanup
            _banner('Cleanup')
            if artifact_id:
                try:
                    await client.delete_artifact(artifact_id)
                    print(f'[OK] deleted artifact {artifact_id}')
                except Exception as exc:
                    print(f'[WARN] delete_artifact failed: {exc}')

            if folder is not None:
                try:
                    # brief wait for artifact deletion to propagate
                    await asyncio.sleep(2)
                    await client.delete_folder(folder.folder_id)
                    print(f'[OK] deleted folder {folder.folder_id}')
                except Exception as exc:
                    print(f'[WARN] delete_folder failed: {exc}')

    print('\n=== uploader functional-API E2E: ALL PASSED ===')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
