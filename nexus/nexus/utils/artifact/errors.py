"""Artifact service client exceptions."""

from __future__ import annotations


class ArtifactError(Exception):
    """Base exception for Nexus artifact service operations."""


class ArtifactClientError(ArtifactError):
    """Raised when an artifact service request returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.http_status = http_status


class ArtifactUploadError(ArtifactClientError):
    """Raised when a COS transfer against a pre-signed URL fails."""


class ArtifactPathError(ArtifactError):
    """Raised when a human-readable folder/artifact path cannot be resolved.

    The client resolves slash-separated paths (for example ``/models/v1.bin``)
    into resource ids by walking the folder tree one segment at a time. This is
    raised when a segment does not exist, or when a non-final segment names an
    artifact instead of a folder.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path
