"""Resolve media references for the component that will consume a request.

A media reference is the ``url`` of an ``image_url`` content part. Its source kind says
where the bytes live:

- ``remote_url``: ``http://`` or ``https://``; the consumer fetches it over the network.
- ``data_url``: ``data:<mime>;base64,<payload>``; the bytes travel inline.
- ``local_file``: ``file://`` or an absolute path on the Nexus host's filesystem (the
  process running the Gateway, standalone or inside HyTuner). Only that host is
  guaranteed to read it; a remote endpoint or a sandbox may not.

Anything else (for example a relative path) is not recognised and is passed through
unchanged; the data contract rejects such references.

Each backend declares the source kinds its consumer can read
(``accepted_media_sources``). ``Processor`` resolves every reference whose kind is not
accepted into one that is, before dispatching to any backend, so a new backend gets
this behaviour without code of its own. Resolution works on a copy of the messages;
recorded trajectories and prefix matching keep the original references.

Conversions are registered per (source kind, target kind). ``choose_target`` picks the
target among registered conversions the consumer accepts; this is where target matching
grows. Today the only conversion is ``local_file -> data_url``.

A ``MediaResolver`` instance caches results by reference. One instance belongs to one
Gateway processor, i.e. one trajectory, so a multi-turn trajectory resolves each
reference once.

TODO(petersheng): sandbox consumers. Installed CLI agents (Claude Code, Codex) read media
inside their runtime, which may not mount the host's storage, and their prompts do not
pass through these messages. Their tasks must stage host files into the runtime (as
``vlm_stem_agent`` uploads images); reuse ``classify_source`` and a per-trajectory cache there.
TODO(petersheng): a ``local_file -> remote_url`` converter (upload and sign) for endpoints
where inline data URLs are too large; register it and extend ``_TARGET_PREFERENCE``.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MediaSourceKind(str, Enum):
    """Where a media reference's bytes live."""

    REMOTE_URL = 'remote_url'
    DATA_URL = 'data_url'
    LOCAL_FILE = 'local_file'


# What a consumer that only speaks the network protocol can read: no host filesystem.
REMOTE_CONSUMER_SOURCES: frozenset[MediaSourceKind] = frozenset({MediaSourceKind.REMOTE_URL, MediaSourceKind.DATA_URL})
# What a consumer on the Nexus host can read.
HOST_CONSUMER_SOURCES: frozenset[MediaSourceKind] = frozenset(MediaSourceKind)


@dataclass(frozen=True)
class MediaPart:
    """One media reference inside a message content part."""

    modality: str  # 'image' today; the content part type is f'{modality}_url'
    url: str


# A conversion turns a reference into a reference of its target kind. It runs off the event loop.
Converter = Callable[[MediaPart], str]

# Content part types that carry a media reference, by modality. Images only: the canonical
# prompt contract and the token-level path accept no other media yet.
_MEDIA_PART_TYPES = {'image': 'image_url'}


def classify_source(url: str) -> MediaSourceKind | None:
    """Classify a reference; ``None`` means it is not a reference kind this module knows."""
    if url.startswith(('http://', 'https://')):
        return MediaSourceKind.REMOTE_URL
    if url.startswith('data:'):
        return MediaSourceKind.DATA_URL
    if url.startswith('file://') or os.path.isabs(url):
        return MediaSourceKind.LOCAL_FILE
    return None


def local_file_to_data_url(part: MediaPart) -> str:
    """Read a host file and return it as a base64 data URL; the bytes are sent unchanged."""
    path = part.url[len('file://') :] if part.url.startswith('file://') else part.url
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type is None:
        # NOTE(petersheng): the data contract requires a file extension; sniffing bytes would
        # hide a malformed row and could label it with a type its consumer rejects.
        raise ValueError(
            f'local {part.modality} {path!r} has no recognised file extension, so its media type is unknown; '
            f'name it with a {part.modality} extension such as .png or .jpg'
        )
    if not mime_type.startswith(f'{part.modality}/'):
        raise ValueError(f'local {part.modality} {path!r} has type {mime_type!r}')
    with open(path, 'rb') as handle:
        data = handle.read()
    return f'data:{mime_type};base64,{base64.b64encode(data).decode("ascii")}'


_CONVERTERS: dict[tuple[MediaSourceKind, MediaSourceKind], Converter] = {
    (MediaSourceKind.LOCAL_FILE, MediaSourceKind.DATA_URL): local_file_to_data_url,
}

# Preference among targets when several registered conversions fit the consumer.
_TARGET_PREFERENCE: tuple[MediaSourceKind, ...] = (MediaSourceKind.DATA_URL, MediaSourceKind.REMOTE_URL)


def register_converter(source: MediaSourceKind, target: MediaSourceKind, converter: Converter) -> None:
    """Register how references of ``source`` kind become ``target`` kind."""
    key = (source, target)
    if key in _CONVERTERS:
        raise ValueError(f'Duplicate media converter for {source.value} -> {target.value}')
    _CONVERTERS[key] = converter


def choose_target(source: MediaSourceKind, accepts: Collection[MediaSourceKind]) -> MediaSourceKind | None:
    """Pick the kind a reference should become, or ``None`` when it can be sent as is.

    Raises when the consumer accepts neither the source kind nor any registered target.
    """
    if source in accepts:
        return None
    for target in _TARGET_PREFERENCE:
        if target in accepts and (source, target) in _CONVERTERS:
            return target
    raise ValueError(f'No media conversion from {source.value} to any of {sorted(kind.value for kind in accepts)}')


def _media_part(part: Any) -> MediaPart | None:
    if not isinstance(part, Mapping):
        return None
    for modality, part_type in _MEDIA_PART_TYPES.items():
        if part.get('type') != part_type:
            continue
        reference = part.get(part_type)
        url = reference.get('url') if isinstance(reference, Mapping) else None
        return MediaPart(modality=modality, url=url) if isinstance(url, str) else None
    return None


class MediaResolver:
    """Rewrite references the consumer cannot read, caching results per reference."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, MediaSourceKind], str] = {}

    def _plan(
        self, part: Any, accepts: Collection[MediaSourceKind]
    ) -> tuple[MediaPart, MediaSourceKind, MediaSourceKind] | None:
        media = _media_part(part)
        if media is None:
            return None
        source = classify_source(media.url)
        if source is None:
            return None
        target = choose_target(source, accepts)
        return None if target is None else (media, source, target)

    async def resolve(self, messages: list[dict], accepts: Collection[MediaSourceKind]) -> list[dict]:
        """Return messages whose references ``accepts`` can read; the input is not modified.

        Returns ``messages`` itself when nothing needs converting, and reuses every
        untouched message and part. Each reference is converted at most once per instance.
        """
        plans = [
            plan
            for message in messages
            if isinstance(message.get('content'), list)
            for part in message['content']
            if (plan := self._plan(part, accepts)) is not None
        ]
        if not plans:
            return messages
        pending: dict[tuple[str, str, MediaSourceKind], tuple[Converter, MediaPart]] = {}
        for media, source, target in plans:
            key = (media.modality, media.url, target)
            if key not in self._cache and key not in pending:
                pending[key] = (_CONVERTERS[(source, target)], media)
        if pending:
            values = await asyncio.gather(*(asyncio.to_thread(convert, media) for convert, media in pending.values()))
            self._cache.update(zip(pending, values, strict=True))
        return [self._rewrite(message, accepts) for message in messages]

    def _rewrite(self, message: dict, accepts: Collection[MediaSourceKind]) -> dict:
        content = message.get('content')
        if not isinstance(content, list):
            return message
        parts: list[Any] = []
        changed = False
        for part in content:
            plan = self._plan(part, accepts)
            if plan is None:
                parts.append(part)
                continue
            media, _, target = plan
            part_type = _MEDIA_PART_TYPES[media.modality]
            resolved = self._cache[(media.modality, media.url, target)]
            parts.append({**part, part_type: {**part[part_type], 'url': resolved}})
            changed = True
        return {**message, 'content': parts} if changed else message


__all__ = [
    'HOST_CONSUMER_SOURCES',
    'REMOTE_CONSUMER_SOURCES',
    'MediaPart',
    'MediaResolver',
    'MediaSourceKind',
    'choose_target',
    'classify_source',
    'local_file_to_data_url',
    'register_converter',
]
