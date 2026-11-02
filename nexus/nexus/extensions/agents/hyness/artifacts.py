"""Upload caller-local npm tarballs using the existing Nexus runtime file API."""

from __future__ import annotations

import base64
import hashlib

from pathlib import Path


async def runtime_package_spec(spec: str) -> str:
    if not (spec.startswith('file:') or (spec.endswith('.tgz') and not spec.startswith(('http://', 'https://')))):
        return spec
    from nexus.core import require_context
    from nexus.extensions.harbor_bridge.upload import raise_for_upload_failures
    from nexus.runtime.executors.file import FileItem

    path = Path(spec.removeprefix('file:')).expanduser().resolve(strict=True)
    data = path.read_bytes()
    remote = f'/tmp/nexus-agent-artifacts/{hashlib.sha256(data).hexdigest()}.tgz'
    runtime = await require_context().get_runtime()
    response = await runtime.upload_file(
        files=[FileItem(path=remote, content=base64.b64encode(data).decode('ascii'), encoding='base64')],
        overwrite=True,
    )
    raise_for_upload_failures(response, 'Agent package artifact')
    return remote
