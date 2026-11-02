"""Host↔sandbox text-file transfer helpers (download / upload / clear).

Thin wrappers over the runtime client's ``download_file`` / ``upload_file`` for the
single-utf8-file case every stage uses to hand artifacts across the sandbox boundary.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from nexus.runtime.executors.models.file import FileItem


logger = logging.getLogger(__name__)


async def download_text_artifact(runtime: Any, sandbox_path: str, dest: Path) -> str | None:
    """Download a single utf8 text file from the sandbox, tolerating absence."""
    try:
        resp = await runtime.download_file([FileItem(path=sandbox_path, encoding='utf8')])
    except Exception as exc:  # noqa: BLE001 - a missing artifact is an expected outcome
        logger.warning('Failed to download %s: %s', sandbox_path, exc)
        return None
    if not resp.files:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.files[0].content or '', encoding='utf-8')
    return str(dest)


async def upload_text(runtime: Any, sandbox_path: str, content: str) -> None:
    """Upload a utf8 text file into the sandbox."""
    await runtime.upload_file([FileItem(path=sandbox_path, content=content, encoding='utf8')])


async def clear_sandbox_file(runtime: Any, sandbox_path: str) -> None:
    """Best-effort clear a per-run sandbox artifact after it has been downloaded.

    Sandbox artifacts use fixed (sgNN-free) paths reused across slices/features, so a
    failed agent write on the next iteration must not let the host read a stale copy.
    The runtime client has no delete-file API, so overwrite the path with empty
    content. Never raises — cleanup is best-effort.
    """
    try:
        # overwrite=True (the default) is essential here — the whole point is to replace
        # an existing file — so state it explicitly rather than lean on the default.
        await runtime.upload_file([FileItem(path=sandbox_path, content='', encoding='utf8')], overwrite=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.warning('swe_any: failed to clear sandbox artifact %s: %s', sandbox_path, exc)
