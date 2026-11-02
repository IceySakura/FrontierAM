"""In-process identity of a file whose content was hashed while writing it."""

from __future__ import annotations

import os

from dataclasses import dataclass


FileSignature = tuple[int, int, int, int, int]


def stat_signature(value: os.stat_result) -> FileSignature:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    size_bytes: int
    sha256: str
    signature: FileSignature
    content_md5: str | None = None
