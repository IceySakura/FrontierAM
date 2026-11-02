"""Shared validation for Nexus runtime file uploads."""

from __future__ import annotations

from typing import Any


def raise_for_upload_failures(response: Any, operation: str) -> None:
    """Raise when a runtime upload omits results or reports a failed file."""
    results = getattr(response, 'results', None)
    if not results:
        raise RuntimeError(f'{operation} upload returned no per-file results')
    failed = [result for result in results if not result.success]
    if failed:
        details = '; '.join(f'{result.path}: {result.message or "upload failed"}' for result in failed)
        raise RuntimeError(f'Failed to upload {operation}: {details}')


__all__ = ['raise_for_upload_failures']
