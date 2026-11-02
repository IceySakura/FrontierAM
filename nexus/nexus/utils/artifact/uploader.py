"""Functional artifact-upload helpers for task functions.

This module offers a small, function-first API for the "one run, one folder,
one or more artifacts" pattern:

* :class:`RunFolder` — plain immutable value object bundling ``folder_id``,
  ``folder_path`` and ``owner`` for a single run's artifact folder.
* :func:`prepare_run_folder` — Host-side constructor. Creates
  ``/tasks/YYYY/MM/DD/{hostname}-{YYYYMMDD-HHMMSS}/`` under an already-open
  :class:`~nexus.utils.artifact.ArtifactClient` and returns a
  :class:`RunFolder`.
* :func:`open_run_folder_from_runtime` — async context manager for sandbox
  worker code. Requests the Taihu PAT from the Host over the Socket.IO
  shared-value bridge, opens an :class:`ArtifactClient` session, prepares a
  run folder, and cleans up the HTTP session on exit.
* :func:`upload_file` / :func:`upload_bytes` — upload a file or in-memory
  bytes into the run folder.
* :func:`artifact_uri_for` / :func:`folder_uri` — build canonical
  ``artifact://<owner>/<path>`` URIs suitable for the task output's
  ``artifact_uri`` field.

Typical usage inside a task function running **inside a sandbox** (worker
process)::

    from nexus.utils.artifact.uploader import (
        artifact_uri_for,
        open_run_folder_from_runtime,
        upload_file,
    )

    async with open_run_folder_from_runtime() as (client, folder):
        artifact = await upload_file(client, folder, '/workspace/result.pptx')
        return MyTaskOutput(
            artifact_uri=artifact_uri_for(folder, artifact.name or 'result.pptx'),
        )

Typical usage **on the Host** (for testing / CLI tooling)::

    from nexus.utils.artifact import ArtifactClient
    from nexus.utils.artifact.uploader import prepare_run_folder, upload_file

    async with ArtifactClient(taihu_pat='tai_pat_...') as client:
        folder = await prepare_run_folder(client)
        artifact = await upload_file(client, folder, '/tmp/result.pptx')

The run folder is named ``{hostname}-{YYYYMMDD-HHMMSS}`` where ``hostname`` is
taken from the ``$HOSTNAME`` environment variable (set by the container
runtime to the sandbox identity, format ``nexus-<32hex>``) or
``socket.gethostname()`` as a fallback. No label or run-id needs to be
supplied.
"""

from __future__ import annotations

import os
import re
import socket

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from nexus.utils.artifact.models import ArtifactInfo
from nexus.utils.artifact.uri import format_artifact_uri


if TYPE_CHECKING:
    from nexus.utils.artifact.client import ArtifactClient


__all__ = [
    'RunFolder',
    'artifact_uri_for',
    'folder_uri',
    'open_run_folder_from_runtime',
    'prepare_run_folder',
    'upload_bytes',
    'upload_file',
]


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunFolder:
    """Immutable descriptor for one run's artifact folder.

    Attributes:
        folder_id:   Artifact-service folder id (use as ``parent_folder_id``).
        folder_path: Bare slash path, e.g.
                     ``/tasks/2026/08/06/nexus-090fb6f8...-20260806-085532``.
                     Kept for internal use / logging; task outputs should
                     prefer :func:`folder_uri` / :func:`artifact_uri_for`.
        owner:       Login of the employee that owns this folder (the caller
                     of :func:`prepare_run_folder`). Used to compose
                     ``artifact://`` URIs for the task output.
    """

    folder_id: str
    folder_path: str
    owner: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Canonical sandbox identity: ``nexus-<32 lowercase hex>``. Container runtimes
# (e.g. Gongfeng on top of K8s) may prepend a tenant/account prefix to the
# pod name, so a raw ``$HOSTNAME`` can look like
# ``admin123-nexus-0961a9697d8048088a2b298a1058c235``. We strip any such
# prefix so folder names stay short and stable.
_SANDBOX_ID_RE = re.compile(r'nexus-[0-9a-f]{32}')


def _sandbox_hostname() -> str:
    """Return the canonical sandbox identity (``nexus-<32hex>``) when detectable.

    Reads ``$HOSTNAME`` (set by container runtimes to the pod name) and
    extracts the ``nexus-<32hex>`` segment if present, discarding any
    tenant/account prefix the runtime may have added (e.g.
    ``admin123-nexus-090fb6f8...`` → ``nexus-090fb6f8...``).

    If neither ``$HOSTNAME`` nor ``socket.gethostname()`` contains the
    canonical pattern (bare-host / test environments), returns the raw value
    unchanged so folder names still carry *some* identity information.
    """
    raw = os.environ.get('HOSTNAME') or socket.gethostname()
    match = _SANDBOX_ID_RE.search(raw)
    return match.group(0) if match else raw


async def _ensure_date_folder(client: ArtifactClient, dt: datetime) -> str:
    """Idempotently create ``/tasks/YYYY/MM/DD`` and return the leaf folder id.

    Uses a deterministic idempotency key per segment so concurrent uploaders on
    the same day get-or-create the same shared date directories instead of
    hitting a ``name_conflict`` error.
    """
    segments = ['tasks', dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')]
    parent_id: str | None = None
    path_so_far = ''
    for seg in segments:
        path_so_far = f'{path_so_far}/{seg}'
        folder = await client.create_folder(
            seg,
            parent_folder_id=parent_id,
            idempotency_key=f'tasks-dir:{path_so_far}',
        )
        parent_id = folder.folder_id
    return parent_id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


async def prepare_run_folder(
    client: ArtifactClient,
    *,
    dt: datetime | None = None,
) -> RunFolder:
    """Create a per-run folder under an open client and return :class:`RunFolder`.

    The folder is named ``{hostname}-{YYYYMMDD-HHMMSS}`` where ``hostname``
    comes from ``$HOSTNAME`` (container sandbox identity) or
    ``socket.gethostname()`` as fallback.

    Args:
        client: An already-authenticated :class:`~nexus.utils.artifact.ArtifactClient`.
        dt:     Timestamp for the date hierarchy. Defaults to UTC now.
    """
    now = dt or datetime.now(timezone.utc)
    hostname = _sandbox_hostname()
    ts = now.strftime('%Y%m%d-%H%M%S')
    folder_name = f'{hostname}-{ts}'

    # Resolve the owner up front. ``whoami()`` on the client caches so
    # this is a one-time round-trip per client instance.
    owner = await client.whoami()

    date_folder_id = await _ensure_date_folder(client, now)
    folder = await client.create_folder(folder_name, parent_folder_id=date_folder_id)

    date_path = f'/tasks/{now.strftime("%Y/%m/%d")}'
    folder_path = f'{date_path}/{folder_name}'
    return RunFolder(folder_id=folder.folder_id, folder_path=folder_path, owner=owner)


@asynccontextmanager
async def open_run_folder_from_runtime(
    *,
    dt: datetime | None = None,
    base_url: str | None = None,
    dev_employee: str | None = None,
) -> AsyncIterator[tuple[ArtifactClient, RunFolder]]:
    """Async context manager: open a client and prepare a run folder inside a sandbox.

    This is the recommended entry point for task functions running inside a
    Gongfeng or similar sandbox. It calls
    :meth:`~nexus.utils.artifact.ArtifactClient.from_runtime` under the hood,
    which fetches the Taihu PAT from the Host via the Socket.IO shared-value
    mechanism — no credentials need to be injected into the sandbox
    environment.

    The folder is named ``{hostname}-{YYYYMMDD-HHMMSS}`` automatically; no
    label or run-id needs to be provided.

    Yields:
        A ``(client, folder)`` tuple. The client's HTTP session is closed
        when the context exits.

    Args:
        dt:           Override the UTC timestamp used for the date hierarchy.
        base_url:     Override the artifact service URL. Must be set (or
                      ``$ARTIFACT_SERVICE_URL``) when the Host-provided
                      Taihu PAT is used.
        dev_employee: Dev-only plaintext employee ID (no Taihu gateway needed).

    Example::

        async with open_run_folder_from_runtime() as (client, folder):
            art = await upload_file(client, folder, '/workspace/result.pptx')
            return artifact_uri_for(folder, art.name or 'result.pptx')
    """
    from nexus.utils.artifact.client import ArtifactClient

    client = await ArtifactClient.from_runtime(
        base_url=base_url,
        dev_employee=dev_employee,
    )
    # ``ArtifactClient.from_runtime()`` does not open the aiohttp session
    # automatically; we open it here so this context manager owns the full
    # lifecycle.
    async with client:
        folder = await prepare_run_folder(client, dt=dt)
        yield client, folder


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


async def upload_file(
    client: ArtifactClient,
    folder: RunFolder,
    local_path: str | Path,
    *,
    name: str | None = None,
    content_type: str | None = None,
) -> ArtifactInfo:
    """Upload a local file into ``folder``.

    Args:
        client:       The open :class:`~nexus.utils.artifact.ArtifactClient`.
        folder:       The :class:`RunFolder` returned by
                      :func:`prepare_run_folder` /
                      :func:`open_run_folder_from_runtime`.
        local_path:   Absolute or relative path to the file on disk.
        name:         Override the artifact name (defaults to the file's
                      basename).
        content_type: Override the MIME type (auto-detected by default).

    Returns:
        :class:`~nexus.utils.artifact.models.ArtifactInfo` with
        ``artifact_id`` and ``status == 'ready'``.
    """
    return await client.upload_file(
        str(local_path),
        parent_folder_id=folder.folder_id,
        name=name,
        content_type=content_type,
    )


async def upload_bytes(
    client: ArtifactClient,
    folder: RunFolder,
    data: bytes,
    name: str,
    *,
    content_type: str | None = None,
) -> ArtifactInfo:
    """Upload in-memory bytes into ``folder``.

    Args:
        client:       The open :class:`~nexus.utils.artifact.ArtifactClient`.
        folder:       The :class:`RunFolder` returned by
                      :func:`prepare_run_folder` /
                      :func:`open_run_folder_from_runtime`.
        data:         Raw bytes to upload.
        name:         Artifact name (required; used as the display filename).
        content_type: Override the MIME type.

    Returns:
        :class:`~nexus.utils.artifact.models.ArtifactInfo` with
        ``artifact_id`` and ``status == 'ready'``.
    """
    return await client.upload_bytes(
        data,
        name=name,
        parent_folder_id=folder.folder_id,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def folder_uri(folder: RunFolder) -> str:
    """Canonical ``artifact://<owner>/tasks/YYYY/MM/DD/<folder>/`` URI for``folder``."""
    segments = [seg for seg in folder.folder_path.split('/') if seg]
    return format_artifact_uri(folder.owner, *segments, trailing_slash=True)


def artifact_uri_for(folder: RunFolder, artifact_name: str) -> str:
    """Return the canonical ``artifact://<owner>/<...>/name`` URI for an artifact in ``folder``.

    Feed the returned string directly into ``nexus artifact download`` /
    ``info`` / ``url``: no view flag needed. Grantees the folder was shared
    with can use the same URI to reach the artifact.
    """
    segments = [seg for seg in folder.folder_path.split('/') if seg]
    return format_artifact_uri(folder.owner, *segments, artifact_name)
