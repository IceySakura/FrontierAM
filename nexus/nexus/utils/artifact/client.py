"""Async client (SDK) for the artifact service.

The artifact service stores artifacts (files) inside a folder tree and brokers
uploads/downloads through pre-signed Tencent COS URLs. This client provides:

* thin wrappers over every programmatic endpoint (folders, resources, artifacts,
  grants), and
* high-level ``upload``/``download`` helpers that perform the full COS transfer
  so callers can read and write artifacts in a single call.

Authentication depends on how the service is reached:

* Through the Taihu (太湖) unified-auth gateway (recommended): pass a Taihu
  personal access token via ``taihu_pat`` (or ``$ARTIFACT_TAIHU_PAT``) and point
  ``base_url`` at the Taihu MCP entrypoint (:data:`TAIHU_GATEWAY_URL`). The
  client sends ``Authorization: Bearer <PAT>`` and the gateway injects the
  employee identity.
* Against a local/dev backend without a gateway: pass ``dev_employee`` (or
  ``$ARTIFACT_DEV_EMPLOYEE``) to send the plaintext ``X-Dev-Employee`` header.

Example::

    async with ArtifactClient(
        base_url=TAIHU_GATEWAY_URL, taihu_pat='tai_pat_...'
    ) as client:
        artifact = await client.upload_file('model.bin', parent_folder_id=folder_id)
        await client.download_to_file(artifact.artifact_id, 'restored.bin')
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import os
import stat
import unicodedata
import uuid

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import aiohttp

from nexus.utils.artifact.auth import (
    ARTIFACT_SHARED_VALUE_NAME,
    DEV_IDENTITY_HEADER,
    ENV_DEV_EMPLOYEE,
    ArtifactCredentials,
    load_taihu_pat,
    taihu_authorization_header,
)
from nexus.utils.artifact.errors import ArtifactClientError, ArtifactPathError, ArtifactUploadError
from nexus.utils.artifact.files import FileIdentity, FileSignature
from nexus.utils.artifact.files import stat_signature as _stat_signature
from nexus.utils.artifact.models import ArtifactInfo, DownloadUrl, Folder, Grant, Page, UploadSession
from nexus.utils.artifact.uri import ParsedUri, format_artifact_uri, parse_artifact_uri
from nexus.utils.cos import (
    CosClientError,
    CosServiceError,
    CosSourceChangedError,
    iter_file_segment,
    upload_presigned_file,
    with_retries,
)


ENV_SERVER_URL = 'ARTIFACT_SERVICE_URL'
# Default Taihu MCP gateway entrypoint for this service. The gateway
# authenticates the caller (Bearer PAT or OAuth) and injects ``x-tai-identity``
# before forwarding to the backend, so clients target the gateway, not the
# backend directly. Set ``ARTIFACT_SERVICE_URL`` (or ``base_url``) to point at a
# different environment's gateway.
TAIHU_GATEWAY_URL = 'https://artifact-nexus-agent.mcp.woa.com'
DEFAULT_SERVER_URL = TAIHU_GATEWAY_URL
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 600
DEFAULT_MULTIPART_CONCURRENCY = 5
# Per-part / whole-object COS transfer retries reuse nexus.utils.cos.with_retries.
# Set well above the COS client's default (3) because under very high concurrency
# (tens of thousands of simultaneous uploads) COS returns transient RequestTimeout
# / UserNetworkTooSlow / 5xx bursts; a deep retry budget converts those into
# eventual successes instead of "fake success" task outputs that merely record an
# upload error. Backoff is exponential and capped at 5s per attempt, so 15 retries
# adds at most ~58s of wait in the worst case (0.25+0.5+1+2+4 then 10x5s) --
# still negligible against the 12h per-run timeout during stress tests. Bumped
# 8 -> 15 after the 20x2500 run still surfaced a few UserNetworkTooSlow failures
# once that code was made retryable (the extra depth rides out longer congestion
# windows at the concurrency peak).
DEFAULT_MULTIPART_MAX_RETRIES = 15
DEFAULT_MULTIPART_RETRY_BACKOFF = 0.25
# Control-plane request retries (artifact-service reached via the Taihu/MCP
# gateway). Gateways in front of the service occasionally return transient
# 502/503/504 ("socket hang up") or drop the connection under load. Every write
# this client issues either carries an Idempotency-Key or is inherently
# idempotent (complete/delete/GET), so retrying these statuses is safe. Bumped
# 5 -> 8 alongside the multipart budget for consistent deep-retry behaviour.
DEFAULT_REQUEST_MAX_RETRIES = 8
DEFAULT_REQUEST_RETRY_BACKOFF = 0.25
_RETRYABLE_REQUEST_HTTP_STATUS = frozenset({502, 503, 504})
_STREAM_CHUNK_SIZE = 1024 * 1024


class ArtifactClient:
    """Async facade for artifact service reads and writes."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        taihu_pat: str | None = None,
        dev_employee: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        transfer_timeout: int = DEFAULT_TRANSFER_TIMEOUT_SECONDS,
        _resolve_local_taihu_pat: bool = True,
    ) -> None:
        self.base_url = (base_url or os.getenv(ENV_SERVER_URL, DEFAULT_SERVER_URL)).rstrip('/')
        # Explicit credentials always outrank environment/file fallbacks. If both
        # a dev identity and a PAT are explicitly supplied, the dev identity keeps
        # precedence for local testing.
        self._has_explicit_pat = taihu_pat is not None
        self._has_explicit_dev_employee = dev_employee is not None
        # Normal Host construction resolves explicit arg > environment > local key
        # file. Runtime construction disables local resolution because the Host has
        # already provided the authoritative credential snapshot.
        self._taihu_pat = load_taihu_pat(taihu_pat) if _resolve_local_taihu_pat else taihu_pat
        resolved_dev = dev_employee if dev_employee is not None else os.getenv(ENV_DEV_EMPLOYEE)
        self._dev_employee = resolved_dev.strip() if resolved_dev and resolved_dev.strip() else None
        self.timeout = timeout
        self.transfer_timeout = transfer_timeout
        self.session: aiohttp.ClientSession | None = None
        # Cached ``GET /v1/whoami`` result -- used to decide whether an
        # ``artifact://<owner>/<path>`` URI refers to the caller's own
        # namespace (owned view) or someone else's (shared view). One HTTP
        # round-trip per client instance, fetched lazily on first use.
        self._whoami_cache: str | None = None

    @classmethod
    async def from_runtime(
        cls,
        *,
        base_url: str | None = None,
        dev_employee: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        transfer_timeout: int = DEFAULT_TRANSFER_TIMEOUT_SECONDS,
    ) -> ArtifactClient:
        """Create a client using Artifact credentials requested once from the Host."""

        from nexus.runtime.context import require_runtime_control

        credentials = await require_runtime_control().get_shared_value(
            ARTIFACT_SHARED_VALUE_NAME,
            ArtifactCredentials,
            decode=ArtifactCredentials.from_shared_payload,
        )
        has_explicit_dev_identity = bool(dev_employee and dev_employee.strip())
        if credentials.taihu_pat is not None and not has_explicit_dev_identity:
            service_url = base_url or os.getenv(ENV_SERVER_URL)
            if not service_url:
                raise ValueError(
                    f'base_url or ${ENV_SERVER_URL} must be configured when using Host-provided Artifact credentials'
                )
        taihu_pat = credentials.taihu_pat.get_secret_value() if credentials.taihu_pat is not None else None
        return cls(
            base_url=base_url,
            taihu_pat=taihu_pat,
            dev_employee=dev_employee,
            timeout=timeout,
            transfer_timeout=transfer_timeout,
            _resolve_local_taihu_pat=False,
        )

    async def __aenter__(self) -> ArtifactClient:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    # --- folders ------------------------------------------------------------

    async def create_folder(
        self,
        name: str,
        *,
        parent_folder_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Folder:
        body = _drop_none({'name': name, 'parent_folder_id': parent_folder_id})
        data = await self._request('POST', '/v1/folders', json_body=body, idempotency_key=idempotency_key)
        return Folder.from_dict(data)

    async def delete_folder(self, folder_id: str) -> None:
        await self._request('DELETE', f'/v1/folders/{folder_id}')

    # --- resources / artifacts listing --------------------------------------

    async def list_resources(
        self,
        *,
        parent_folder_id: str | None = None,
        view: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
        name: str | None = None,
    ) -> Page:
        params = _drop_none({
            'parent_folder_id': parent_folder_id,
            'view': view,
            'limit': limit,
            'cursor': cursor,
            'name': name,
        })
        data = await self._request('GET', '/v1/resources', params=params)
        return Page.from_dict(data)

    async def list_artifacts(
        self,
        *,
        view: str = 'owned',
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page:
        params = _drop_none({'view': view, 'limit': limit, 'cursor': cursor})
        data = await self._request('GET', '/v1/artifacts', params=params)
        return Page.from_dict(data)

    async def get_artifact(self, artifact_id: str) -> ArtifactInfo:
        data = await self._request('GET', f'/v1/artifacts/{artifact_id}')
        return ArtifactInfo.from_dict(data)

    # --- path resolution ----------------------------------------------------
    #
    # The service addresses every resource by id, but the CLI lets users work
    # with human-readable slash paths (for example ``/models/v1/model.bin``).
    # We resolve a path into a resource id by walking the folder tree one
    # segment at a time via ``list_resources`` -- the server guarantees names
    # are unique among visible siblings under one parent (per owner), so each
    # segment resolves to at most one resource with no ambiguity. Names are
    # NFC-normalized to match the server's ``normalize_name`` before comparison.

    async def whoami(self) -> str:
        """Return the acting employee's login (cached per client instance).

        Fetches ``GET /v1/whoami`` on first use; subsequent calls reuse the
        cached value. The login is what appears as the authority in a
        canonical ``artifact://<owner>/<path>`` URI, so we need it whenever a
        URI is resolved to decide between the owned and shared views.
        """

        if self._whoami_cache is None:
            data = await self._request('GET', '/v1/whoami')
            login = data.get('employee_id') if isinstance(data, dict) else None
            if not login:
                raise ArtifactClientError('whoami response is missing employee_id')
            self._whoami_cache = str(login)
        return self._whoami_cache

    # --- path resolution ----------------------------------------------------
    #
    # The service addresses every resource by id, but the CLI lets users work
    # with human-readable references. Two forms are accepted:
    #
    # * a bare slash-path (``/models/v1/model.bin``, leading slash optional),
    #   which always refers to the caller's own resources; and
    # * a full ``artifact:///<owner>/<path>`` URI, which explicitly names the
    #   owner and so can address a resource shared with the caller without
    #   any auxiliary ``--view`` flag.
    #
    # ``artifact:///`` (empty path) is a reserved list-only URI meaning "the
    # aggregated shared-with-me view"; single-resource resolution rejects it.
    # Path walking is server-guaranteed unambiguous: names are unique among
    # visible siblings under one parent per owner, and the top-level shared
    # view is filtered by owner id when a URI names one, so two grantors
    # sharing the same-named file remain distinguishable.

    async def _iter_children(
        self,
        parent_folder_id: str | None,
        *,
        view: str | None = None,
        owner_filter: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield every child of a folder (or every top-level resource), paging through cursors.

        ``view`` (owned vs. shared-with-me) only selects a *top-level* listing:
        the server rejects ``view`` combined with ``parent_folder_id`` and grants
        access to a folder's children through the folder's own authorization
        (owner / direct grant / inherited ancestor grant). So once we descend
        into a folder we drop ``view`` -- a folder shared with us is still
        reachable by its id.

        ``owner_filter`` scopes a top-level shared listing to entries owned by
        one specific employee (used when resolving ``artifact://<owner>/...``);
        the server does not support this filter, so it is applied client-side
        after the shared listing returns. Ignored for id-scoped listings.
        """

        effective_view = view if parent_folder_id is None else None
        apply_owner_filter = owner_filter is not None and parent_folder_id is None
        cursor: str | None = None
        while True:
            page = await self.list_resources(
                parent_folder_id=parent_folder_id, view=effective_view, limit=200, cursor=cursor
            )
            for item in page.items:
                if apply_owner_filter and str(item.get('owner_id') or '') != owner_filter:
                    continue
                yield item
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    async def _find_child(
        self,
        parent_folder_id: str | None,
        name: str,
        *,
        view: str | None = None,
        owner_filter: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the child resource whose name matches ``name`` (NFC), or None.

        Fast path: when the listing is owner-scoped -- an owned top-level lookup
        or any sub-folder (``owner_filter`` unset and not a shared/all view) --
        ask the server to resolve the single named child with ``?name=`` instead
        of streaming the whole (possibly huge) directory back to scan here. A
        name-aware server returns at most one item, so this is O(1) in the
        sibling count rather than O(N), which is what makes resolving a path
        under a directory holding tens of thousands of entries feasible.

        Backward compatibility: an older server ignores the unknown ``name``
        parameter and returns an ordinary (here length-1) listing. We therefore
        only trust a returned item when its name actually matches the target; a
        non-matching item means the server ignored ``name``, so we fall back to
        the full streaming scan. An empty page is a genuine not-found either way
        -- a name-aware miss, or an old server whose parent has no visible child
        -- so it short-circuits instead of degrading into a full scan (which is
        what lets a lookup for a missing/half-written resource fail fast).

        A cross-owner shared top-level lookup (``owner_filter`` set / a
        shared/all view) cannot be keyed server-side -- its owner is not the
        caller -- so it always uses the streaming scan with client-side owner
        filtering.
        """

        target = _normalize_name(name)
        if owner_filter is None and view not in ('shared', 'all'):
            effective_view = view if parent_folder_id is None else None
            page = await self.list_resources(parent_folder_id=parent_folder_id, view=effective_view, limit=1, name=name)
            if not page.items:
                return None
            first = page.items[0]
            if _normalize_name(str(first.get('name') or '')) == target:
                return first
            # The server ignored ``name`` (older build); fall back to a full scan.
        async for item in self._iter_children(parent_folder_id, view=view, owner_filter=owner_filter):
            if _normalize_name(str(item.get('name') or '')) == target:
                return item
        return None

    async def _resolve_view_and_owner(self, parsed: ParsedUri) -> tuple[str | None, str | None]:
        """Decide the top-level view and any owner filter for a parsed URI.

        Returns ``(view, owner_filter)`` where ``view`` is the ``?view=``
        query parameter to send to the server (``None`` = owned) and
        ``owner_filter`` is the client-side owner constraint applied to the
        top-level listing (also ``None`` when no filtering is needed).

        Rejects the aggregated ``artifact:///`` view, which is only meaningful
        for list commands and not for resolving to a single resource.
        """

        if parsed.is_aggregated_shared_view:
            raise ArtifactPathError(
                'artifact:/// addresses the aggregated shared-with-me view, not a resource; '
                'use artifact:///<owner>/<path> to name a specific artifact or folder.',
            )
        if parsed.owner is None:
            # Bare path -- the caller's own namespace.
            return None, None
        if parsed.is_self_view:
            # ``artifact:///~/...`` -- the ``~`` self-reference sentinel names
            # the caller's own namespace. The server does not understand ``~``,
            # so resolve it here rather than treating ``~`` as a literal owner
            # (which would wrongly fall through to view='shared', owner='~').
            return None, None
        self_login = await self.whoami()
        if parsed.owner == self_login:
            return None, None
        return 'shared', parsed.owner

    async def resolve_path(self, uri_or_path: str, *, expect: str | None = None) -> dict[str, Any]:
        """Resolve either a bare slash-path or an ``artifact:///`` URI to its item.

        Returns the resource item (a dict with at least ``resource_id``,
        ``name`` and ``resource_type``, as returned by ``list_resources``).

        Pass ``expect='folder'`` or ``expect='artifact'`` to require the final
        segment to be of that type. The top-level view (owned vs. shared) is
        picked automatically from the URI's owner segment.
        """

        parsed = parse_artifact_uri(uri_or_path)
        if not parsed.segments:
            raise ArtifactPathError('an empty path does not name a resource', path=uri_or_path)
        view, owner_filter = await self._resolve_view_and_owner(parsed)
        parent_id: str | None = None
        item: dict[str, Any] | None = None
        for index, segment in enumerate(parsed.segments):
            # Owner filter and view only apply to the *top-level* lookup; the
            # server rejects ``view`` combined with ``parent_folder_id`` and
            # sub-folder access is authorized by the folder itself.
            top_view = view if parent_id is None else None
            top_owner = owner_filter if parent_id is None else None
            item = await self._find_child(parent_id, segment, view=top_view, owner_filter=top_owner)
            if item is None:
                raise ArtifactPathError(f'path segment {segment!r} not found', path=uri_or_path)
            is_last = index == len(parsed.segments) - 1
            rtype = _item_type(item)
            if not is_last:
                if rtype != 'folder':
                    raise ArtifactPathError(f'path segment {segment!r} is not a folder', path=uri_or_path)
                parent_id = _item_id(item)
        assert item is not None
        if expect is not None and _item_type(item) != expect:
            raise ArtifactPathError(f'path {uri_or_path!r} does not name a {expect}', path=uri_or_path)
        return item

    async def resolve_resource_id(self, uri_or_path: str, *, expect: str | None = None) -> str:
        """Resolve a URI or bare path into a resource id (folder id / artifact id)."""

        return _item_id(await self.resolve_path(uri_or_path, expect=expect))

    async def resolve_parent_folder_id(self, uri_or_path: str) -> str | None:
        """Resolve the *parent* folder id for a URI or bare path, ignoring its final segment.

        Returns ``None`` when the reference has a single segment (its parent
        is the top level / root). Every intermediate segment must be an
        existing folder. Useful for ``mkdir`` / ``upload`` which create a new
        leaf under an existing parent.
        """

        parsed = parse_artifact_uri(uri_or_path)
        if not parsed.segments:
            raise ArtifactPathError('an empty path has no parent folder', path=uri_or_path)
        if len(parsed.segments) == 1:
            # The parent is the top level. A bare path or the ``~`` self-reference
            # both mean the caller's own root; an explicit foreign owner must not
            # silently resolve to the caller's root (which would create the leaf
            # in the wrong namespace), so reject it before returning None.
            if parsed.owner is not None and not parsed.is_self_view:
                self_login = await self.whoami()
                if parsed.owner != self_login:
                    raise ArtifactPathError(
                        f'{uri_or_path!r} names the namespace of another employee '
                        f'({parsed.owner!r}); you can only create resources under your own root.',
                        path=uri_or_path,
                    )
            return None
        if parsed.owner is not None:
            # Rebuild an owner-qualified URI so resolve_path preserves the
            # shared-view / owner filter decision when walking the parent.
            parent_ref = format_artifact_uri(parsed.owner, *parsed.segments[:-1], trailing_slash=True)
        else:
            parent_ref = '/'.join(parsed.segments[:-1])
        return _item_id(await self.resolve_path(parent_ref, expect='folder'))

    def leaf_name(self, uri_or_path: str) -> str:
        """Return the final path segment (the resource's own name)."""

        parsed = parse_artifact_uri(uri_or_path)
        if not parsed.segments:
            raise ArtifactPathError('an empty path has no name', path=uri_or_path)
        return parsed.segments[-1]

    def uri_targets_other_owner(self, uri_or_path: str, self_login: str) -> bool:
        """Return True iff ``uri_or_path`` explicitly names an owner other than the caller.

        Used by CLI write commands to short-circuit against attempts to write
        to someone else's namespace (``artifact://<other>/models``): the server
        would deny it anyway, but rejecting client-side gives a clearer error
        without a round-trip.
        """

        parsed = parse_artifact_uri(uri_or_path)
        if parsed.is_aggregated_shared_view:
            return True
        return parsed.owner is not None and parsed.owner != self_login

    async def delete_artifact(self, artifact_id: str) -> None:
        await self._request('DELETE', f'/v1/artifacts/{artifact_id}')

    # --- grants -------------------------------------------------------------

    async def list_grants(self, resource_id: str) -> list[Grant]:
        data = await self._request('GET', f'/v1/resources/{resource_id}/grants')
        items = data.get('items') if isinstance(data, dict) else None
        return [Grant.from_dict(item) for item in items] if isinstance(items, list) else []

    async def put_grant(
        self,
        resource_id: str,
        grantee_id: str,
        *,
        grantee_type: str = 'employee',
        permission: str = 'read',
        expires_at: str | datetime | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        _validate_grant(grantee_type, permission)
        body = _drop_none({'expires_at': _iso(expires_at)})
        await self._request(
            'PUT',
            f'/v1/resources/{resource_id}/grants/{permission}/{grantee_id}',
            json_body=body,
            idempotency_key=idempotency_key,
        )

    async def delete_grant(
        self,
        resource_id: str,
        grantee_id: str,
        *,
        grantee_type: str = 'employee',
        permission: str = 'read',
    ) -> None:
        _validate_grant(grantee_type, permission)
        await self._request(
            'DELETE',
            f'/v1/resources/{resource_id}/grants/{permission}/{grantee_id}',
        )

    # --- low-level upload flow ----------------------------------------------

    async def create_upload(
        self,
        *,
        filename: str,
        size_bytes: int,
        content_md5: str,
        content_type: str | None = None,
        parent_folder_id: str | None = None,
        storage_pool: str | None = None,
        multipart: bool | None = None,
        idempotency_key: str | None = None,
    ) -> UploadSession:
        body = _drop_none({
            'filename': filename,
            'size_bytes': size_bytes,
            'content_md5': content_md5,
            'content_type': content_type,
            'parent_folder_id': parent_folder_id,
            'storage_pool': storage_pool,
            'multipart': multipart,
        })
        data = await self._request('POST', '/v1/artifact-uploads', json_body=body, idempotency_key=idempotency_key)
        return UploadSession.from_dict(data)

    async def complete_upload(self, upload_id: str, parts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body = {'parts': parts} if parts is not None else None
        return await self._request('POST', f'/v1/artifact-uploads/{upload_id}/complete', json_body=body)

    # --- download url -------------------------------------------------------

    async def create_download_url(self, artifact_id: str) -> DownloadUrl:
        data = await self._request('POST', f'/v1/artifacts/{artifact_id}/download-url')
        return DownloadUrl.from_dict(data)

    async def get_download_url(self, artifact_id: str) -> str:
        return (await self.create_download_url(artifact_id)).url

    # --- high-level read/write ----------------------------------------------

    async def upload_bytes(
        self,
        data: bytes,
        name: str,
        *,
        parent_folder_id: str | None = None,
        content_type: str | None = None,
        storage_pool: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactInfo:
        """Create an upload session, PUT the bytes to COS, and mark it ready."""

        content_md5 = base64.b64encode(hashlib.md5(data).digest()).decode('ascii')  # noqa: S324
        resolved_type = content_type or _guess_content_type(name)
        session = await self.create_upload(
            filename=name,
            size_bytes=len(data),
            content_md5=content_md5,
            content_type=resolved_type,
            parent_folder_id=parent_folder_id,
            storage_pool=storage_pool,
            idempotency_key=idempotency_key,
        )
        if session.is_ready:
            return await self.get_artifact(session.artifact_id)
        if session.upload_id is None:
            raise ArtifactUploadError('upload session did not return an upload_id')
        if session.is_multipart:
            parts = await self._upload_bytes_parts_to_cos(
                session,
                data,
                max_concurrency=DEFAULT_MULTIPART_CONCURRENCY,
            )
            await self.complete_upload(session.upload_id, parts=parts)
        else:
            await self._put_to_cos(session, data)
            await self.complete_upload(session.upload_id)
        return await self.get_artifact(session.artifact_id)

    async def upload_file(
        self,
        local_path: str | Path,
        *,
        name: str | None = None,
        parent_folder_id: str | None = None,
        content_type: str | None = None,
        storage_pool: str | None = None,
        multipart: bool | None = None,
        max_concurrency: int = DEFAULT_MULTIPART_CONCURRENCY,
        idempotency_key: str | None = None,
        follow_symlinks: bool = True,
        _identity: FileIdentity | None = None,
    ) -> ArtifactInfo:
        """Upload a local file and return the ready artifact.

        When the server signs a multipart (chunked) upload session the file is
        streamed part-by-part with bounded concurrency; otherwise a single
        pre-signed PUT is used. Pass ``multipart=True`` to hint the server to
        prefer a chunked session (useful for very large files that exceed the
        5 GB single-PUT limit).

        ``follow_symlinks=False`` opens the final path component ``O_NOFOLLOW``,
        so a symlink at ``local_path`` is rejected instead of read through. Bulk
        callers that walk an untrusted tree (``publish``) use this to prevent a
        symlink from redirecting the read to a file outside that tree; the
        held descriptor is also re-checked against the path before the upload is
        completed, so a mid-flight swap is detected rather than finalized.
        """

        path = Path(local_path)
        artifact_name = name or path.name
        try:
            if follow_symlinks:
                source = open(path, 'rb')  # noqa: SIM115 -- translate open errors before entering context
            else:
                # O_NOFOLLOW makes the open fail (ELOOP) if the leaf is a
                # symlink, closing the scan-then-swap race for bulk callers.
                fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
                source = os.fdopen(fd, 'rb')
        except OSError as exc:
            raise ArtifactUploadError(f'not a readable regular file: {path}') from exc
        with source:
            source_signature = _open_file_signature(source)
            _assert_open_file_unchanged(path, source, source_signature)
            size_bytes = source_signature[2]
            if _identity is None:
                content_md5 = await _open_file_md5_base64(source)
            else:
                if _identity.signature != source_signature or _identity.size_bytes != size_bytes:
                    raise ArtifactUploadError('source file changed during upload')
                if _identity.content_md5 is None:
                    raise ValueError('precomputed upload identity requires Content-MD5')
                content_md5 = _identity.content_md5
            _assert_open_file_unchanged(path, source, source_signature)
            resolved_type = content_type or _guess_content_type(artifact_name)
            session = await self.create_upload(
                filename=artifact_name,
                size_bytes=size_bytes,
                content_md5=content_md5,
                content_type=resolved_type,
                parent_folder_id=parent_folder_id,
                storage_pool=storage_pool,
                multipart=multipart,
                idempotency_key=idempotency_key,
            )
            if session.is_ready:
                return await self.get_artifact(session.artifact_id)
            if session.upload_id is None:
                raise ArtifactUploadError('upload session did not return an upload_id')
            _assert_open_file_unchanged(path, source, source_signature)
            if session.is_multipart:
                parts = await self._upload_parts_to_cos(
                    session,
                    path,
                    max_concurrency=max_concurrency,
                    source=source,
                    file_size=size_bytes,
                )
            else:
                await self._put_file_to_cos(session, path, source=source, file_size=size_bytes)
                parts = None
            _assert_open_file_unchanged(path, source, source_signature)
            await self.complete_upload(session.upload_id, parts=parts)
            return await self.get_artifact(session.artifact_id)

    async def download_bytes(self, artifact_id: str) -> bytes:
        url = await self.get_download_url(artifact_id)
        session = self._require_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.transfer_timeout)) as response:
            if not (200 <= response.status < 300):
                text = await response.text()
                raise ArtifactUploadError(f'download failed (HTTP {response.status}): {text[:300]}')
            return await response.read()

    async def download_to_file(self, artifact_id: str, local_path: str | Path) -> Path:
        url = await self.get_download_url(artifact_id)
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        session = self._require_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.transfer_timeout)) as response:
            if not (200 <= response.status < 300):
                text = await response.text()
                raise ArtifactUploadError(f'download failed (HTTP {response.status}): {text[:300]}')
            temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
            try:
                # Open with O_NOFOLLOW | O_EXCL so a symlink (or unexpected file)
                # planted at the temp path cannot redirect the write elsewhere.
                fd = _open_new_no_follow(temporary)
                with os.fdopen(fd, 'wb') as f:
                    async for chunk in response.content.iter_chunked(_STREAM_CHUNK_SIZE):
                        f.write(chunk)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
        return target

    @dataclass(frozen=True)
    class DownloadFailure:
        """One artifact that could not be downloaded during a recursive walk.

        ``relative_path`` is the artifact's location within the walked folder
        (POSIX-style, no leading ``/``); ``local_path`` is where it would have
        landed on disk. ``error`` carries the exception's short message.
        """

        artifact_id: str
        name: str
        relative_path: str
        local_path: Path
        error: str

    async def download_folder_to_dir(
        self,
        folder_id: str,
        dest_dir: str | Path,
        *,
        concurrency: int = 4,
    ) -> tuple[int, list[DownloadFailure]]:
        """Recursively mirror a folder's artifact tree into ``dest_dir``.

        ``dest_dir`` receives the folder's *contents* (not a subfolder named
        after the source folder), matching the ``rsync -a src/ dest/`` idiom.
        Nested folders are created on demand. Empty subfolders are skipped --
        the artifact service has no first-class semantics for them, so
        reproducing them locally would just be noise.

        Downloads run concurrently up to ``concurrency`` at a time (default
        ``4``). Individual failures do not abort the walk: they are collected
        and returned so the caller can report a summary and set an exit code.

        Returns ``(success_count, failures)``.
        """

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Two-phase: first collect every (artifact_id, relative_path) via a
        # DFS over ``list_resources``, then fan out the actual downloads with
        # a bounded semaphore. Enumerating up front avoids issuing 100+
        # nested pagination requests interleaved with downloads (and gives
        # us a total count for progress reporting later if we ever add it).
        plan: list[tuple[str, str, str]] = []  # (artifact_id, name, relative_path)

        async def _walk(current_folder_id: str, prefix: str) -> None:
            async for item in self._iter_children(current_folder_id):
                name = str(item.get('name') or '')
                if not name:
                    continue
                rel = f'{prefix}{name}' if not prefix else f'{prefix}/{name}'
                rtype = _item_type(item)
                if rtype == 'folder':
                    child_id = _item_id(item)
                    if child_id:
                        await _walk(child_id, rel)
                elif rtype == 'artifact':
                    plan.append((_item_id(item), name, rel))

        await _walk(folder_id, '')

        if not plan:
            return 0, []

        successes = 0
        failures: list[ArtifactClient.DownloadFailure] = []
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(artifact_id: str, name: str, rel: str) -> None:
            nonlocal successes
            # ``rel`` is a POSIX-style forward-slash path; splitting on ``/``
            # is safe because artifact names cannot contain ``/`` (rejected
            # by both service and URI grammar).
            try:
                # Keep every write anchored inside ``dest``: reject ``..`` escapes
                # and any symlinked intermediate directory before writing, so a
                # crafted tree or a pre-planted ``dest/sub -> /elsewhere`` link
                # cannot redirect files outside the download destination.
                local_target = _resolve_within(dest, rel)
                _mkdir_within(dest, local_target.parent)
            except Exception as exc:
                failures.append(
                    ArtifactClient.DownloadFailure(
                        artifact_id=artifact_id,
                        name=name,
                        relative_path=rel,
                        local_path=dest.joinpath(*rel.split('/')),
                        error=str(exc) or type(exc).__name__,
                    )
                )
                return
            async with sem:
                try:
                    await self.download_to_file(artifact_id, local_target)
                except Exception as exc:
                    failures.append(
                        ArtifactClient.DownloadFailure(
                            artifact_id=artifact_id,
                            name=name,
                            relative_path=rel,
                            local_path=local_target,
                            error=str(exc) or type(exc).__name__,
                        )
                    )
                else:
                    successes += 1

        await asyncio.gather(*(_one(a, n, r) for a, n, r in plan))
        return successes, failures

    # --- COS transfer helpers -----------------------------------------------
    #
    # Uploads/downloads go through pre-signed COS URLs. Streaming, retries, and
    # source-integrity checks are reused from :mod:`nexus.utils.cos` instead of
    # being reimplemented here. The artifact service uses a server-completed
    # multipart model (the client PUTs each part and reports the ETags back to
    # the service via ``complete_upload``), so it cannot use
    # ``cos.upload_presigned_multipart_file`` (which drives a COS-native
    # complete/abort flow); the per-part transfer machinery is shared instead.

    async def _put_to_cos(self, session_info: UploadSession, data: bytes) -> None:
        """Upload in-memory bytes with a single pre-signed PUT (with retries)."""

        if not session_info.url:
            raise ArtifactUploadError('upload session did not return a pre-signed URL')
        headers = {**session_info.headers, 'Content-Length': str(len(data))}
        await self._put_part(session_info.url, lambda: data, headers, part_label=None)

    async def _put_file_to_cos(
        self,
        session_info: UploadSession,
        path: Path,
        *,
        source: object | None = None,  # kept for backward-compatible call sites
        file_size: int | None = None,
    ) -> None:
        """Upload a local file with a single pre-signed PUT, reusing the COS client.

        Delegates to :func:`nexus.utils.cos.upload_presigned_file`, which streams
        the file, retries transient failures, and aborts if the source changes
        mid-flight -- capabilities the artifact client previously lacked.
        """

        if not session_info.url:
            raise ArtifactUploadError('upload session did not return a pre-signed URL')
        session = self._require_session()
        expected_stat: os.stat_result | None = None
        try:
            expected_stat = path.stat()
        except OSError:
            expected_stat = None
        try:
            # Forward the backend-supplied headers verbatim (e.g. a lowercase
            # ``content-type``). This mirrors the in-memory upload path, which
            # forwards the session headers directly; doing so preserves the
            # signature-relevant headers the pre-signed URL was signed with,
            # otherwise COS returns SignatureDoesNotMatch. Content-Length is
            # managed by ``upload_presigned_file`` itself.
            await upload_presigned_file(
                session,
                path,
                session_info.url,
                key=session_info.artifact_id or path.name,
                extra_headers=session_info.headers,
                timeout=self.transfer_timeout,
                max_retries=DEFAULT_MULTIPART_MAX_RETRIES,
                retry_backoff=DEFAULT_MULTIPART_RETRY_BACKOFF,
                expected_stat=expected_stat,
            )
        except CosSourceChangedError as exc:
            raise ArtifactUploadError('source file changed during upload') from exc
        except CosClientError as exc:
            raise ArtifactUploadError(f'COS upload failed: {exc}') from exc

    async def _upload_bytes_parts_to_cos(
        self,
        session_info: UploadSession,
        data: bytes,
        *,
        max_concurrency: int = DEFAULT_MULTIPART_CONCURRENCY,
    ) -> list[dict[str, Any]]:
        """Upload in-memory bytes through a server-selected multipart session."""

        if not session_info.is_multipart:
            raise ArtifactUploadError('upload session is not a multipart session')
        part_size, part_urls = _validate_multipart_layout(session_info, len(data))

        def body_factory(offset: int, length: int) -> Any:
            return _iter_bytes_segment(data, offset, length)

        return await self._upload_multipart(
            part_urls,
            part_size,
            len(data),
            body_factory,
            max_concurrency=max_concurrency,
        )

    async def _upload_parts_to_cos(
        self,
        session_info: UploadSession,
        path: Path,
        *,
        max_concurrency: int = DEFAULT_MULTIPART_CONCURRENCY,
        source: object | None = None,  # kept for backward-compatible call sites
        file_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Upload each part to its pre-signed PUT URL and return the part ETags.

        Parts are streamed (not buffered whole) from the file at ``part_size``
        boundaries with bounded concurrency, reusing the COS client's streaming
        reader and retry helper. The returned list of ``{'part_number', 'etag'}``
        is what the service needs to perform the COS CompleteMultipartUpload. On
        failure in-flight tasks are cancelled; the server aborts the orphaned COS
        upload when the session expires (the bucket lifecycle rule is the final
        safety net).
        """

        if not session_info.is_multipart:
            raise ArtifactUploadError('upload session is not a multipart session')
        resolved_size = file_size if file_size is not None else path.stat().st_size
        part_size, part_urls = _validate_multipart_layout(session_info, resolved_size)

        def body_factory(offset: int, length: int) -> Any:
            return iter_file_segment(path, offset, length)

        return await self._upload_multipart(
            part_urls,
            part_size,
            resolved_size,
            body_factory,
            max_concurrency=max_concurrency,
        )

    async def _upload_multipart(
        self,
        part_urls: list[str],
        part_size: int,
        total_size: int,
        body_factory: Callable[[int, int], Any],
        *,
        max_concurrency: int,
    ) -> list[dict[str, Any]]:
        """Concurrently PUT every part and collect ``{'part_number', 'etag'}``.

        ``body_factory(offset, length)`` returns a fresh request body for one
        part; it is re-invoked on each retry so streamed bodies are re-read from
        the start of the part rather than resumed from a consumed iterator.
        """

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def upload_one(index: int) -> dict[str, Any]:
            offset = index * part_size
            length = min(part_size, total_size - offset)
            part_number = index + 1
            headers = {'Content-Length': str(length)}
            async with semaphore:
                etag = await self._put_part(
                    part_urls[index],
                    lambda: body_factory(offset, length),
                    headers,
                    part_label=part_number,
                )
            return {'part_number': part_number, 'etag': etag}

        tasks = [asyncio.create_task(upload_one(i)) for i in range(len(part_urls))]
        try:
            parts = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        parts.sort(key=lambda part: part['part_number'])
        return parts

    async def _put_part(
        self,
        url: str,
        body_factory: Callable[[], Any],
        headers: dict[str, str],
        *,
        part_label: int | None,
    ) -> str | None:
        """PUT one object/part to a pre-signed COS URL with shared retry logic.

        Returns the response ``ETag`` (required for multipart parts, where
        ``part_label`` is set; ``None`` is tolerated for single-object PUTs).
        """

        session = self._require_session()
        where = f'part {part_label}' if part_label is not None else 'upload'

        async def operation() -> str | None:
            body = body_factory()
            async with session.put(
                url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.transfer_timeout),
            ) as response:
                if not (200 <= response.status < 300):
                    text = await response.text()
                    # Raise a CosServiceError (not ArtifactUploadError) so the shared
                    # with_retries helper can apply its 408/429/5xx status-code retry
                    # policy; transient failures otherwise terminate the upload after a
                    # single attempt. Retry-exhausted errors are converted back to
                    # ArtifactUploadError by the caller below.
                    raise CosServiceError(
                        method='PUT',
                        key=where,
                        status=response.status,
                        message=text[:300],
                    )
                etag = response.headers.get('ETag')
            if part_label is not None and not etag:
                raise ArtifactUploadError(f'COS {where} returned no ETag')
            return etag

        try:
            # Pass ``where`` (e.g. 'part 3' / 'upload'), never the presigned
            # ``url``, as the retry key: on retry exhaustion with_retries embeds
            # this value verbatim into the raised message, so a full URL would
            # leak the COS q-signature and temporary token into CLI/task logs.
            return await with_retries(
                'PUT',
                where,
                operation,
                max_retries=DEFAULT_MULTIPART_MAX_RETRIES,
                retry_backoff=DEFAULT_MULTIPART_RETRY_BACKOFF,
            )
        except CosClientError as exc:
            raise ArtifactUploadError(f'COS {where} failed: {exc}') from exc

    # --- transport ----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        await self._apply_auth_headers(headers)
        if json_body is not None:
            headers['Content-Type'] = 'application/json'
        if idempotency_key is None and _needs_idempotency(method, path):
            idempotency_key = uuid.uuid4().hex
        if idempotency_key is not None:
            # Reuse one key across retries so create/grant mutations stay idempotent.
            headers['Idempotency-Key'] = idempotency_key
        attempts = DEFAULT_REQUEST_MAX_RETRIES + 1
        for attempt in range(attempts):
            try:
                return await self._request_once(method, path, params=params, json_body=json_body, headers=headers)
            except ArtifactClientError as exc:
                # Only transient gateway statuses (502/503/504) are retried; genuine
                # 4xx and other 5xx responses fail fast so real errors are not masked.
                if exc.http_status not in _RETRYABLE_REQUEST_HTTP_STATUS or attempt + 1 >= attempts:
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError):
                # Transport-level drops (e.g. ServerDisconnectedError / socket hang up
                # surfacing as a connection error) are transient; retry within budget.
                if attempt + 1 >= attempts:
                    raise
            await asyncio.sleep(min(DEFAULT_REQUEST_RETRY_BACKOFF * (2**attempt), 5.0))
        raise AssertionError('unreachable')

    async def _request_once(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        session = self._require_session()
        async with session.request(
            method,
            f'{self.base_url}{path}',
            params=params,
            json=json_body,
            headers=headers,
        ) as response:
            return await _parse_result(response)

    async def _apply_auth_headers(self, headers: dict[str, str]) -> None:
        """Set configured identity headers for one request onto ``headers``."""

        if self._has_explicit_dev_employee and self._dev_employee is not None:
            headers[DEV_IDENTITY_HEADER] = self._dev_employee
        elif self._has_explicit_pat and self._taihu_pat is not None:
            headers['Authorization'] = taihu_authorization_header(self._taihu_pat)
        elif self._dev_employee is not None:
            headers[DEV_IDENTITY_HEADER] = self._dev_employee
        elif self._taihu_pat is not None:
            headers['Authorization'] = taihu_authorization_header(self._taihu_pat)

    def _require_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError('ArtifactClient must be used as an async context manager')
        return self.session


def _needs_idempotency(method: str, path: str) -> bool:
    # Create and grant mutations require an Idempotency-Key; complete/delete do not.
    if method == 'POST' and path == '/v1/artifact-uploads':
        return True
    if method == 'POST' and path == '/v1/folders':
        return True
    return method == 'PUT' and '/grants/' in path


async def _parse_result(response: aiohttp.ClientResponse) -> dict[str, Any]:
    if response.status == 204:
        await response.read()
        return {}
    try:
        body = await response.json(content_type=None)
    except Exception as exc:  # noqa: BLE001
        text = await response.text()
        if not (200 <= response.status < 300):
            raise ArtifactClientError(f'HTTP {response.status}: {text[:300]}', http_status=response.status) from exc
        raise ArtifactClientError(f'non-JSON response (HTTP {response.status})', http_status=response.status) from exc

    if isinstance(body, dict) and isinstance(body.get('error'), dict):
        error = body['error']
        raise ArtifactClientError(
            error.get('message') or f'HTTP {response.status}',
            code=error.get('code'),
            request_id=error.get('request_id'),
            http_status=response.status,
        )
    if not (200 <= response.status < 300):
        message = body.get('message') if isinstance(body, dict) else str(body)
        raise ArtifactClientError(f'HTTP {response.status}: {message}', http_status=response.status)
    return body if isinstance(body, dict) else {'value': body}


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _split_path(path: str) -> list[str]:
    """Split a slash path into non-empty segments (leading/trailing slashes ignored).

    Kept for backward compatibility with a small number of internal callers
    that manipulate bare paths directly; most call sites now go through
    :mod:`nexus.utils.artifact.uri` (which also NFC-normalizes).
    """

    return [segment for segment in str(path).split('/') if segment]


def _normalize_name(name: str) -> str:
    """NFC-normalize a name to match the server's ``normalize_name``."""

    return unicodedata.normalize('NFC', name)


# ``O_NOFOLLOW`` makes ``os.open`` fail (ELOOP) if the final path component is a
# symlink, closing the symlink-following hole in local writes. It does not exist
# on Windows, where we degrade to plain opens (symlink attacks there require
# privileges and the download flow is primarily used on POSIX hosts).
_O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)


def _resolve_within(base: Path, rel: str) -> Path:
    """Return ``base``/``rel`` verified to stay lexically inside ``base``.

    ``rel`` is a POSIX-style, forward-slash relative path (artifact names cannot
    contain ``/``). Rejects any component that is ``..`` (or otherwise escapes
    ``base``) so a crafted server-side tree cannot redirect writes outside the
    download destination. Symlink-following is handled separately at write time.
    """

    base_resolved = base.resolve()
    parts = [segment for segment in rel.split('/') if segment and segment != '.']
    if any(segment == '..' for segment in parts):
        raise ArtifactPathError(f'unsafe relative path {rel!r}: parent-directory segments are not allowed', path=rel)
    target = base_resolved.joinpath(*parts)
    # Guard against any residual escape (e.g. absolute-looking components).
    if base_resolved != target and base_resolved not in target.parents:
        raise ArtifactPathError(f'unsafe relative path {rel!r}: escapes the destination directory', path=rel)
    return target


def _mkdir_within(base: Path, directory: Path) -> None:
    """Create ``directory`` under the trusted ``base``, rejecting symlinks.

    ``base`` is the caller-chosen destination root and is trusted as-is (its own
    ancestors are not re-validated). Every level strictly below ``base`` up to
    ``directory`` is created with ``os.mkdir`` if missing, and any existing level
    that is a symlink (or a non-directory) is refused -- so a pre-planted link at
    ``base/sub`` cannot redirect later writes outside ``base``.
    """

    base.mkdir(parents=True, exist_ok=True)
    if directory == base:
        return
    if base not in directory.parents:
        raise ArtifactUploadError(f'refusing to write: {directory} is outside {base}')
    # Levels strictly between base and directory (inclusive of directory),
    # ordered from shallowest to deepest.
    intermediate = [p for p in reversed(directory.parents) if base in p.parents]
    for level in (*intermediate, directory):
        try:
            info = os.lstat(level)
        except FileNotFoundError:
            os.mkdir(level)
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactUploadError(f'refusing to write through symlinked directory: {level}')
        if not stat.S_ISDIR(info.st_mode):
            raise ArtifactUploadError(f'refusing to write: {level} exists and is not a directory')


def _open_new_no_follow(path: Path) -> int:
    """Open ``path`` for writing, failing if it already exists or is a symlink.

    Uses ``O_CREAT | O_EXCL | O_NOFOLLOW`` so a pre-existing symlink at ``path``
    cannot redirect the write outside the destination (``O_EXCL`` also prevents
    clobbering an unexpected file). Returns a raw file descriptor.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    return os.open(path, flags, 0o644)


def _item_type(item: dict[str, Any]) -> str:
    """Best-effort resource type for a ``list_resources`` item."""

    rtype = item.get('resource_type')
    if rtype:
        return str(rtype)
    return 'folder' if item.get('folder_id') else 'artifact'


def _item_id(item: dict[str, Any]) -> str:
    """Resolve the resource id from a ``list_resources`` item (any id field)."""

    rid = item.get('resource_id') or item.get('folder_id') or item.get('artifact_id')
    if not rid:
        raise ArtifactPathError('resolved resource is missing an id')
    return str(rid)


def _validate_grant(grantee_type: str, permission: str) -> None:
    if grantee_type != 'employee':
        raise ValueError('Artifact grants only support employee grantees')
    if permission != 'read':
        raise ValueError('Artifact grants only support read permission')


def _iso(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _guess_content_type(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or 'application/octet-stream'


def _file_signature(path: Path) -> FileSignature:
    return _stat_signature(path.stat())


def _open_file_signature(source: BinaryIO) -> FileSignature:
    opened = os.fstat(source.fileno())
    if not stat.S_ISREG(opened.st_mode):
        raise ArtifactUploadError('source is not a regular file')
    return _stat_signature(opened)


def _assert_open_file_unchanged(path: Path, source: BinaryIO, expected: FileSignature) -> None:
    try:
        opened = _stat_signature(os.fstat(source.fileno()))
        current_path = _file_signature(path)
    except (OSError, ValueError) as exc:
        raise ArtifactUploadError('source file changed during upload') from exc
    if opened != expected or current_path != expected:
        raise ArtifactUploadError('source file changed during upload')


async def _open_file_md5_base64(source: BinaryIO) -> str:
    def _compute() -> str:
        digest = hashlib.md5()  # noqa: S324
        source.seek(0)
        for chunk in iter(lambda: source.read(_STREAM_CHUNK_SIZE), b''):
            digest.update(chunk)
        source.seek(0)
        return base64.b64encode(digest.digest()).decode('ascii')

    return await asyncio.to_thread(_compute)


def _validate_multipart_layout(
    session_info: UploadSession,
    payload_size: int,
) -> tuple[int, list[str]]:
    part_size = session_info.part_size or 0
    declared_count = session_info.part_count or 0
    part_urls = session_info.part_urls
    if part_size <= 0:
        raise ArtifactUploadError('upload session has an invalid part_size')
    expected_count = (payload_size + part_size - 1) // part_size
    if not 1 <= expected_count <= 10_000:
        raise ArtifactUploadError('multipart payload has an invalid part count')
    if declared_count != expected_count or len(part_urls) != expected_count:
        raise ArtifactUploadError('multipart session part count does not match payload size')
    return part_size, part_urls


async def _iter_bytes_segment(data: bytes, offset: int, length: int) -> AsyncIterator[memoryview]:
    view = memoryview(data)
    end = offset + length
    cursor = offset
    while cursor < end:
        next_cursor = min(cursor + _STREAM_CHUNK_SIZE, end)
        yield view[cursor:next_cursor]
        cursor = next_cursor
