from __future__ import annotations

from enum import IntEnum


class SandboxErrorCode(IntEnum):
    """Business error codes for sandbox operations, with HTTP status mapping."""

    PROVIDER_NOT_CONFIGURED = 50301
    ACQUIRE_FAILED = 50200
    RELEASE_FAILED = 50201
    CALL_EXECUTOR_FAILED = 50202

    @property
    def http_status(self) -> int:
        return {
            SandboxErrorCode.PROVIDER_NOT_CONFIGURED: 503,
            SandboxErrorCode.ACQUIRE_FAILED: 502,
            SandboxErrorCode.RELEASE_FAILED: 502,
            SandboxErrorCode.CALL_EXECUTOR_FAILED: 502,
        }[self]
