"""Typed views over artifact service API responses.

The service returns plain JSON objects. These lightweight dataclasses expose the
stable fields used by the client and CLI while keeping the untouched payload on
``.raw`` so callers can read fields that are not modeled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Folder:
    folder_id: str
    name: str | None = None
    parent_folder_id: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Folder:
        return cls(
            folder_id=data.get('folder_id', ''),
            name=data.get('name'),
            parent_folder_id=data.get('parent_folder_id'),
            created_at=data.get('created_at'),
            raw=data,
        )


@dataclass(frozen=True)
class ArtifactInfo:
    artifact_id: str
    name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    status: str | None = None
    ready_at: str | None = None
    parent_folder_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactInfo:
        return cls(
            artifact_id=data.get('artifact_id', ''),
            name=data.get('name'),
            content_type=data.get('content_type'),
            size_bytes=data.get('size_bytes'),
            status=data.get('status'),
            ready_at=data.get('ready_at'),
            parent_folder_id=data.get('parent_folder_id'),
            raw=data,
        )


@dataclass(frozen=True)
class UploadSession:
    artifact_id: str
    upload_id: str | None
    status: str | None
    method: str | None
    url: str | None
    headers: dict[str, str]
    expires_at: str | None = None
    part_urls: list[str] = field(default_factory=list)
    part_size: int | None = None
    part_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadSession:
        upload = data.get('upload') or {}
        raw_parts = upload.get('part_urls')
        part_urls = [str(url) for url in raw_parts] if isinstance(raw_parts, list) else []
        return cls(
            artifact_id=data.get('artifact_id', ''),
            upload_id=data.get('upload_id'),
            status=data.get('status'),
            method=upload.get('method'),
            url=upload.get('url'),
            headers=dict(upload.get('headers') or {}),
            expires_at=upload.get('expires_at'),
            part_urls=part_urls,
            part_size=upload.get('part_size'),
            part_count=upload.get('part_count'),
            raw=data,
        )

    @property
    def is_ready(self) -> bool:
        return self.status == 'ready'

    @property
    def is_multipart(self) -> bool:
        """Whether the server signed a multipart (chunked) upload session.

        In the server-completed model the session carries per-part PUT URLs and a
        part size; the client uploads the parts and reports their ETags to the
        service, which performs the COS CompleteMultipartUpload.
        """
        return bool(self.part_urls and self.part_size)


@dataclass(frozen=True)
class DownloadUrl:
    method: str
    url: str
    expires_in: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownloadUrl:
        return cls(
            method=data.get('method', 'GET'),
            url=data.get('url') or data.get('download_url', ''),
            expires_in=data.get('expires_in'),
            raw=data,
        )


@dataclass(frozen=True)
class Grant:
    grantee_type: str
    grantee_id: str
    permission: str
    expires_at: str | None = None
    inherited_from: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grant:
        return cls(
            grantee_type=data.get('grantee_type', 'employee'),
            grantee_id=data.get('grantee_id', ''),
            permission=data.get('permission', ''),
            expires_at=data.get('expires_at'),
            inherited_from=data.get('inherited_from'),
            raw=data,
        )


@dataclass(frozen=True)
class Page:
    """A single page of a keyset-paginated listing."""

    items: list[dict[str, Any]]
    next_cursor: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        items = data.get('items')
        return cls(items=list(items) if isinstance(items, list) else [], next_cursor=data.get('next_cursor'))
