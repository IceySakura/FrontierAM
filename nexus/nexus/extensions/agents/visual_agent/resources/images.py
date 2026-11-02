"""Resolve images, files and videos while preserving prompt order.

Local paths and data URIs share deterministic sandbox destinations. COS images are downloaded before resolution;
other HTTP images remain backend references.
"""

import asyncio
import base64
import hashlib
import logging
import mimetypes

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from nexus.core.context import require_context
from nexus.extensions.agents.visual_agent.resources.cos_images import download_cos_images, is_cos_image_url
from nexus.extensions.agents.visual_agent.resources.media import (
    MAX_MEDIA_BYTES,
    image_header,
    image_mime,
    pixel_size,
    read_media_file,
)
from nexus.runtime.executors import FileItem


logger = logging.getLogger(__name__)

RUNTIME_UPLOAD_DIR = '/tmp'

_IMAGE_PART_TYPES = frozenset({'image_url', 'image'})

#: Video part shapes. ``video_url`` is what OpenAI-style callers emit; ``video``
#: mirrors the Anthropic-ish ``image`` spelling accepted above.
_VIDEO_PART_TYPES = frozenset({'video_url', 'video'})

_IMAGE_SUFFIX_TO_MIME = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
}

_MIME_TO_SUFFIX = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/webm': '.webm',
    'video/x-matroska': '.mkv',
    'video/x-msvideo': '.avi',
    'video/mpeg': '.mpeg',
    'application/pdf': '.pdf',
    'application/json': '.json',
    'application/octet-stream': '.bin',
    'text/plain': '.txt',
    'text/csv': '.csv',
}


REF_LOCAL_PATH = 'local_path'
REF_DATA_URI = 'data_uri'
REF_REMOTE_URL = 'remote_url'
REF_OPAQUE = 'opaque'


def classify_ref(ref: str) -> str:
    """What kind of reference ``ref`` is — see the ``REF_*`` constants."""
    if ref.startswith('data:'):
        return REF_DATA_URI
    if ref.startswith(('http://', 'https://')):
        return REF_REMOTE_URL
    try:
        if Path(ref).is_file():
            return REF_LOCAL_PATH
    except OSError:
        pass
    return REF_OPAQUE


def encode_image(image_source: str) -> str:
    """Encode an image to a base64 string."""
    return base64.b64encode(read_source(image_source)).decode('utf-8')


def _mime_for(name: str) -> str:
    return _IMAGE_SUFFIX_TO_MIME.get(Path(name).suffix.lower(), 'image/jpeg')


def _mime_for_source(ref: str) -> str:
    """The mime type to declare when inlining ``ref`` as a data URI."""
    return _mime_of_data_uri(ref) or _mime_for(ref)


def image_data_uri(source: str) -> str:
    """Encode validated image bytes using a recognized image MIME."""
    raw = read_source(source)
    mime = image_mime(raw) or _mime_for_source(source)
    if mime not in _IMAGE_SUFFIX_TO_MIME.values():
        raise ValueError('Image data URI requires a recognized image MIME')
    if source.startswith(f'data:{mime};base64,'):
        return source
    return f'data:{mime};base64,{base64.b64encode(raw).decode("ascii")}'


def validate_reward_media(content: Any) -> None:
    """Reject file/video references that a remote reward service cannot read."""
    if not isinstance(content, list):
        return
    for part in content:
        ref = _part_file_ref(part) or _part_video_ref(part)
        if ref is not None and not ref.startswith(('data:', 'http://', 'https://')):
            raise ValueError(
                'Reward file/video inputs require data: or HTTP(S) references; '
                'local paths cannot be read by the reward service. '
                'Disable Nexus scoring or explicitly configure text-only scoring.'
            )


def _label_for(ref: str, location: str) -> str:
    """The markdown label for an image's path note."""
    if classify_ref(ref) == REF_LOCAL_PATH:
        return Path(ref).stem
    return Path(location).stem


def image_dimensions(source: str) -> tuple[int, int] | None:
    """Read raw dimensions, matching Image.open(path).size inside the sandbox."""
    try:
        return pixel_size(image_header(source))
    except (OSError, ValueError):
        logger.debug('Could not read image dimensions', exc_info=True)
        return None


def _part_image_ref(part: Any) -> str | None:
    """Return the image reference carried by a content part, if any."""
    if not isinstance(part, dict) or part.get('type') not in _IMAGE_PART_TYPES:
        return None
    image_url = part.get('image_url', part.get('image'))
    ref = image_url.get('url') if isinstance(image_url, dict) else image_url
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError('Image part requires a non-empty image_url.url or image reference')
    return ref


def _with_image_ref(part: dict[str, Any], url: str) -> dict[str, Any]:
    """Copy an image part with its reference replaced by ``url``."""
    updated = {**part, 'type': 'image_url'}
    updated.pop('image', None)
    image_url = part.get('image_url', part.get('image'))
    if isinstance(image_url, dict):
        nested = dict(image_url)
        nested['url'] = url
        updated['image_url'] = nested
    else:
        updated['image_url'] = {'url': url}
    return updated


def _part_file_ref(part: Any) -> str | None:
    """Read ``file.url``, with legacy ``file_data`` / ``file_id`` support."""
    if not isinstance(part, dict) or part.get('type') != 'file':
        return None
    file_obj = part.get('file')
    if not isinstance(file_obj, dict):
        raise ValueError('visual_agent file part requires a nested file object with file.url')
    if 'filename' in file_obj:
        filename = file_obj['filename']
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError('visual_agent file.filename must be a non-empty string when provided')
    for key in ('url', 'file_data'):
        if key in file_obj:
            ref = file_obj[key]
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f'visual_agent file.{key} must be a non-empty string')
            return ref
    file_id = file_obj.get('file_id')
    if not isinstance(file_id, str) or not file_id.strip():
        raise ValueError('visual_agent file part requires file.url (or legacy file_data / file_id)')
    return None


def _part_video_ref(part: Any) -> str | None:
    """Return the video reference carried by a content part, if any."""
    if not isinstance(part, dict) or part.get('type') not in _VIDEO_PART_TYPES:
        return None
    video_url = part.get('video_url', part.get('video'))
    ref = video_url.get('url') if isinstance(video_url, dict) else video_url
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError('Video part requires a non-empty video_url.url or video reference')
    return ref


def collect_local_files(parts: list[Any]) -> list[str]:
    """Return the uploadable references of ``file`` parts, in order, deduped."""
    found: list[str] = []
    seen: set[str] = set()
    for part in parts:
        ref = _part_file_ref(part)
        if ref is None or ref in seen or not is_uploadable(ref):
            continue
        seen.add(ref)
        found.append(ref)
    return found


def collect_local_images(parts: list[Any]) -> list[str]:
    """Return the uploadable references of image parts, in order, deduped."""
    found: list[str] = []
    seen: set[str] = set()
    for part in parts:
        ref = _part_image_ref(part)
        if ref is None or ref in seen or not is_uploadable(ref):
            continue
        seen.add(ref)
        found.append(ref)
    return found


def collect_local_videos(parts: list[Any]) -> list[str]:
    """Return the uploadable references of WIP video parts, in order, deduped."""
    found: list[str] = []
    seen: set[str] = set()
    for part in parts:
        ref = _part_video_ref(part)
        if ref is None or ref in seen or not is_uploadable(ref):
            continue
        seen.add(ref)
        found.append(ref)
    return found


def _mime_of_data_uri(ref: str) -> str | None:
    """The mime type a ``data:`` URI declares, if it declares one."""
    if not ref.startswith('data:'):
        return None
    header = ref[len('data:') :].partition(',')[0]
    mime = header.split(';', 1)[0].strip().lower()
    return mime or None


def _synthetic_name(ref: str, index: int, *, kind: str) -> str:
    """A filename for a source that has none of its own."""
    mime = _mime_of_data_uri(ref)
    default = {'video': '.mp4', 'file': '.bin'}.get(kind, '.png')
    suffix = _MIME_TO_SUFFIX.get(mime or '', default)
    return f'{kind}_{index}{suffix}'


def read_source(ref: str) -> bytes:
    """Read local bytes or decode a data URI. Remote URLs must be resolved by the caller."""
    kind = classify_ref(ref)
    if kind == REF_LOCAL_PATH:
        return read_media_file(Path(ref))
    if kind == REF_DATA_URI:
        header, separator, payload = ref.partition(',')
        if not separator or not payload or not header.endswith(';base64'):
            raise ValueError('Media data URI requires a non-empty base64 payload')
        if len(payload) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
            raise ValueError('Media data URI exceeds the 64 MiB limit')
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError:
            raise ValueError('Media data URI contains invalid base64') from None
        if len(data) > MAX_MEDIA_BYTES:
            raise ValueError('Media data URI exceeds the 64 MiB limit')
        return data
    raise ValueError('Media source must be a readable local file or a data: URI; wrap raw base64 in a data: URI')


def is_uploadable(ref: str) -> bool:
    """Whether :func:`read_source` can produce bytes for ``ref``."""
    return classify_ref(ref) in (REF_LOCAL_PATH, REF_DATA_URI)


def content_scope(sources: Sequence[str], *, length: int = 12) -> str:
    """Hash source strings into a stable upload scope. Local files use paths, not file bytes."""
    if not sources:
        return 'noinput'
    h = hashlib.sha256()
    for source in sources:
        h.update(source.encode())
        h.update(b'\x00')
    return h.hexdigest()[:length]


def _scoped_dir(scope: str) -> str:
    """``RUNTIME_UPLOAD_DIR`` plus one sanitised scope level."""
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in scope) or 'run'
    safe = safe.lstrip('.') or 'run'
    return f'{RUNTIME_UPLOAD_DIR.rstrip("/")}/{safe}'


def upload_locations(sources: Sequence[str], *, kinds: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map sources to sandbox paths, preserving filenames and disambiguating collisions.

    Data URIs receive MIME-based names numbered per media kind.
    """
    base = _scoped_dir(content_scope(sources))
    kinds = kinds or {}
    locations: dict[str, str] = {}
    used: set[str] = set()
    counters: dict[str, int] = {}
    for source in sources:
        if classify_ref(source) == REF_LOCAL_PATH:
            name = Path(source).name or 'file'
        else:
            kind = kinds.get(source, 'image')
            index = counters.get(kind, 0)
            counters[kind] = index + 1
            name = _synthetic_name(source, index, kind=kind)
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate = name
        n = 2
        while candidate in used:
            candidate = f'{stem}-{n}{suffix}'
            n += 1
        used.add(candidate)
        locations[source] = f'{base}/{candidate}'
    return locations


async def upload_images_to_runtime(
    images: list[str],
    runtime_name: str = 'main',
    *,
    kinds: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Upload sources into the sandbox. Returns {source: sandbox_path}."""
    if not images:
        return {}

    locations = upload_locations(images, kinds=kinds)
    file_items = []
    landed: dict[str, str] = {}
    for source, runtime_path in locations.items():
        data = await asyncio.to_thread(read_source, source)
        file_items.append(
            FileItem(
                path=runtime_path,
                content=base64.b64encode(data).decode('utf-8'),
                encoding='base64',
            )
        )
        landed[source] = runtime_path

    if not file_items:
        return {}

    ctx = require_context()
    runtime = await ctx.get_runtime(runtime_name)
    response = await runtime.upload_file(file_items, overwrite=True)
    results = getattr(response, 'results', None)
    if results is None or len(results) != len(file_items) or any(not result.success for result in results):
        raise RuntimeError('Media upload failed or returned incomplete results')
    logger.info(
        'Uploaded %d file(s) to runtime under %s',
        len(file_items),
        _scoped_dir(content_scope(images)),
    )
    return landed


def _size_suffix(path: str, enabled: bool) -> str:
    """Return `` (WxH px)`` for a readable image, or '' when off/unreadable."""
    if not enabled:
        return ''
    size = image_dimensions(path)
    if size is None:
        return ''
    return f' ({size[0]}x{size[1]} px)'


def rewrite_parts(
    parts: list[Any],
    runtime_locations: dict[str, str],
    *,
    input_image_paths: bool,
    inline_images: bool,
    show_image_dimensions: bool = False,
    input_image_path_position: Literal['before', 'after'] = 'before',
) -> list[dict[str, Any]]:
    """Replace media references in place, retaining caller order and optional path/size hints."""
    resolved: list[dict[str, Any]] = []
    for part in parts:
        file_ref = _part_file_ref(part)
        if file_ref is not None and is_uploadable(file_ref):
            file_obj = part['file']
            filename = file_obj.get('filename')
            label = f'File {filename}' if filename else 'File'
            location = runtime_locations.get(file_ref)
            if location:
                mime = _mime_of_data_uri(file_ref) or mimetypes.guess_type(file_ref)[0] or ''
                suffix = _size_suffix(file_ref, show_image_dimensions and mime.startswith('image/'))
                resolved.append({
                    'type': 'text',
                    'text': f'\n[{label} saved at {location}{suffix}]',
                })
            elif classify_ref(file_ref) == REF_LOCAL_PATH:
                raise RuntimeError('Local file could not be delivered to the sandbox')
            else:
                # Validate even when bypassing upload; never send unchecked inline files.
                read_source(file_ref)
                backend_file = {'file_data': file_ref}
                if filename:
                    backend_file['filename'] = filename
                resolved.append({'type': 'file', 'file': backend_file})
            continue

        video_ref = _part_video_ref(part)
        if video_ref is not None:
            # WARNING (WIP): video path notes and upload-failure fallbacks are experimental.
            location = runtime_locations.get(video_ref)
            if location:
                resolved.append({'type': 'text', 'text': f'\n[Video saved at {location}]'})
            else:
                raise RuntimeError('Video could not be delivered to the sandbox')
            continue

        ref = _part_image_ref(part)
        if ref is None or not is_uploadable(ref):
            resolved.append(part)
            continue

        location = runtime_locations.get(ref)
        path_note = None
        if input_image_paths and location:
            path_note = {
                'type': 'text',
                'text': f'\n![{_label_for(ref, location)}]({location}){_size_suffix(ref, show_image_dimensions)}',
            }
            if input_image_path_position == 'before' or not inline_images:
                resolved.append(path_note)

        if not inline_images:
            if not location or not input_image_paths:
                raise RuntimeError('Image is unavailable: enable inlining or upload with image path hints')
            continue

        resolved.append(_with_image_ref(part, image_data_uri(ref)))
        if path_note is not None and input_image_path_position == 'after':
            resolved.append(path_note)

    return resolved


def _current_context_or_none() -> Any:
    """The active :class:`ExecutionContext`, or ``None`` when there is no run."""
    try:
        return require_context()
    except Exception:
        return None


async def resolve_user_prompt(
    user_prompt: str | list[Any],
    *,
    input_image_paths: bool = True,
    input_image_path_position: Literal['before', 'after'] = 'before',
    inline_images: bool = True,
    upload_to_runtime: bool = True,
    show_image_dimensions: bool = False,
    runtime_name: str = 'main',
) -> str | list[dict[str, Any]]:
    """Resolve media in order and upload local paths/data URIs when a runtime is available.

    COS images are downloaded once per unique URL. Files and videos
    become sandbox path notes. Requested uploads fail closed.
    """
    if isinstance(user_prompt, str):
        return user_prompt

    parts = list(user_prompt)

    for index, part in enumerate(parts):
        ref = _part_image_ref(part)
        if ref is not None:
            parts[index] = _with_image_ref(part, ref)
        if ref is not None and classify_ref(ref) == REF_OPAQUE:
            raise ValueError(
                'Image source must be a readable local path, HTTP(S) URL, or data: URI; wrap raw base64 in a data: URI'
            )

    for part in parts:
        file_ref = _part_file_ref(part)
        if isinstance(part, dict) and part.get('type') == 'file':
            file_obj = part['file']
            if 'file_data' in file_obj:
                logger.warning(
                    'visual_agent file.file_data is deprecated and support may be removed in a future version; '
                    'use file.url instead.'
                )
        if file_ref is None:
            continue
        kind = classify_ref(file_ref)
        if kind == REF_REMOTE_URL:
            raise ValueError(
                'visual_agent cannot deliver a remote file URL: remote URLs are not fetched. '
                'Resolve file.url host-side to a local path or a data: URI before building the prompt.'
            )
        if kind == REF_OPAQUE:
            raise ValueError(
                'visual_agent file source must point to an existing local file or a data: URI. '
                'Resolve dataset-relative paths against the dataset resource root before calling the agent.'
            )

    for part in parts:
        ref = _part_video_ref(part)
        if ref is not None and classify_ref(ref) == REF_OPAQUE:
            raise ValueError('Video source must be a readable local path or a data: URI')
        if ref is not None and classify_ref(ref) == REF_REMOTE_URL:
            raise ValueError(
                f'visual_agent cannot deliver a remote video URL: {ref[:200]!r}. '
                'Videos reach the model through the sandbox, which needs bytes this '
                'process can read, and remote URLs are deliberately not fetched. '
                'Resolve it host-side to a local path or a data: URI before building '
                'the prompt.'
            )

    # Download COS once; ordinary HTTP images remain backend references.
    cos_refs = [ref for part in parts if (ref := _part_image_ref(part)) is not None and is_cos_image_url(ref)]
    if cos_refs:
        cos_images = await download_cos_images(cos_refs)
        parts = [
            _with_image_ref(part, cos_images[ref]) if (ref := _part_image_ref(part)) in cos_images else part
            for part in parts
        ]

    local_images = collect_local_images(parts)
    local_videos = collect_local_videos(parts)
    local_files = collect_local_files(parts)
    if not local_images and not local_videos and not local_files:
        return parts

    runtime_locations: dict[str, str] = {}
    if upload_to_runtime:
        ctx = _current_context_or_none()
        if ctx is None:
            logger.debug('No execution context: nothing can be uploaded, parts are used as-is')
        elif ctx.runtime_provider:
            kinds = {
                **dict.fromkeys(local_files, 'file'),
                **dict.fromkeys(local_videos, 'video'),
                **dict.fromkeys(local_images, 'image'),
            }
            sources = list(dict.fromkeys(local_images + local_videos + local_files))
            runtime_locations = await upload_images_to_runtime(
                sources,
                runtime_name=runtime_name,
                kinds=kinds,
            )
        elif local_files or local_videos:
            # Files and videos are only ever delivered via the sandbox, so
            # without a runtime there is no way to hand them over at all.
            logger.warning(
                'No runtime provider configured: %d file part(s) and %d video part(s) cannot be delivered',
                len(local_files),
                len(local_videos),
            )

    return rewrite_parts(
        parts,
        runtime_locations,
        input_image_paths=input_image_paths,
        input_image_path_position=input_image_path_position,
        inline_images=inline_images,
        show_image_dimensions=show_image_dimensions,
    )


__all__ = [
    'REF_DATA_URI',
    'REF_LOCAL_PATH',
    'REF_OPAQUE',
    'REF_REMOTE_URL',
    'RUNTIME_UPLOAD_DIR',
    'classify_ref',
    'collect_local_files',
    'collect_local_images',
    'collect_local_videos',
    'content_scope',
    'encode_image',
    'image_dimensions',
    'is_uploadable',
    'read_source',
    'resolve_user_prompt',
    'rewrite_parts',
    'upload_images_to_runtime',
    'upload_locations',
]
