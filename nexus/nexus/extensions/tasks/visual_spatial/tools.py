"""LOCAL spatial tools for the visual_spatial task: locate, depth, pose and segmentation.

Requests bypass environment proxies. Input coordinates use the service's 0-999 space. Images are read back from
the runtime sandbox by the path the model quotes, so files it creates there also work; explicit
spatial.image_sources and data URLs cover runs without a runtime, and only such runs read a host path directly.
HTTP 200 responses with ok=false are failures, and locate maps query to the service's phrase field.
"""

import asyncio
import base64
import logging
import os

from pathlib import Path
from typing import Annotated, Any, Literal

from annotated_types import Len
from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.extensions.agents.visual_agent.resources.media import MAX_MEDIA_BYTES, image_mime, read_media_file
from nexus.runtime.executors.models.file import FileItem
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

SPATIAL_TOOLS = ('visual/query_locate', 'visual/query_depth', 'visual/query_pose', 'visual/query_segment')
SPATIAL_CONFIG_SECTION = 'spatial'
SPATIAL_IMAGE_SOURCES_KEY = 'image_sources'

DEFAULT_BASE_URL = 'http://29.177.195.212'

#: Per-tool endpoints. The ports are the public four; locate's 9018/9089 are its
#: internal backends and are not meant to be called directly.
LOCATE_PATH = ':9028/v1/grounding/locate'
DEPTH_PATH = ':9059/v1/depth/metric-points'
POSE_PATH = ':9099/v1/pose'
SEGMENT_PATH = ':9079/v1/segment'

#: Must stay above the locate service's own queue timeout (120s as of 2026-10),
#: or a request that waits the full queue and is then shed arrives after we have
#: already given up -- we would report a timeout and never see the 429 telling us
#: to back off. The headroom is what buys the one retry below.
DEFAULT_TIMEOUT = 150.0

DEFAULT_MAX_CONCURRENCY = 8

#: Fallback wait when a 429 arrives without a usable Retry-After, and the cap we
#: honour when it carries an implausibly long one.
DEFAULT_RETRY_AFTER = 5.0
MAX_RETRY_AFTER = 30.0

#: A retry left with less request budget than this is not worth sending.
MIN_RETRY_BUDGET = 1.0

_MIME = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
}

#: Guards the service. A semaphore binds to the event loop it first waits on, so
#: each loop gets its own set, keyed by the limit so a changed config takes effect.
_semaphores: dict[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]] = {}


def _config(context: ToolContext) -> dict[str, Any]:
    """``tool_configs['spatial']``, as a plain dict."""
    return dict(context.tool_configs.get(SPATIAL_CONFIG_SECTION, {}) or {})


def _endpoint(config: dict[str, Any], key: str, env: str, suffix: str) -> str:
    """Resolve one endpoint: explicit URL, else env var, else ``base_url`` + port."""
    if url := (config.get(key) or os.environ.get(env)):
        return str(url)
    base = config.get('base_url') or os.environ.get('SPATIAL_BASE_URL') or DEFAULT_BASE_URL
    return f'{str(base).rstrip("/")}{suffix}'


def _semaphore(config: dict[str, Any]) -> asyncio.Semaphore:
    """The shared concurrency gate for this process."""
    try:
        limit = int(config.get('max_concurrency', DEFAULT_MAX_CONCURRENCY))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_CONCURRENCY
    limit = max(1, limit)
    per_loop = _semaphores.setdefault(asyncio.get_running_loop(), {})
    if limit not in per_loop:
        per_loop[limit] = asyncio.Semaphore(limit)
    return per_loop[limit]


def _image_sources(config: dict[str, Any]) -> dict[str, str]:
    """``{path_the_model_may_quote: readable_source}`` from ``spatial.image_sources``."""
    sources = config.get(SPATIAL_IMAGE_SOURCES_KEY) or {}
    if not isinstance(sources, dict):
        return {}
    return {str(k): str(v) for k, v in sources.items()}


def _sandbox_context() -> Any | None:
    """The run context when it has a runtime provider, else ``None`` (a host-only run)."""
    try:
        ctx = require_context()
    except RuntimeError:
        return None
    return ctx if getattr(ctx, 'runtime_provider', None) is not None else None


async def _read_sandbox_data_url(spec: str, config: dict[str, Any], ctx: Any | None) -> str | None:
    """Read ``spec`` from an already-acquired runtime, or ``None`` when there is none or no such file."""
    if ctx is None or not os.path.isabs(spec):
        return None
    # Only a runtime this run already holds: get_runtime would start a sandbox
    # just to look for a file. The context exposes no public lookup for that.
    runtime = getattr(ctx, '_runtimes', {}).get(str(config.get('runtime') or 'main'))
    if runtime is None:
        return None
    response = await runtime.download_file([FileItem(path=spec, encoding='base64')])
    files = [item for item in response.files or [] if item.content]
    if not files:
        return None
    content = files[0].content or ''
    # Base64 is 4/3 the raw size; reject before decoding anything that large.
    if len(content) > (MAX_MEDIA_BYTES // 3 + 1) * 4:
        raise ValueError('Image exceeds the 64 MiB limit')
    head = base64.b64decode(content[:64])
    mime = image_mime(head) or _MIME.get(Path(spec).suffix.lower(), 'image/png')
    return f'data:{mime};base64,{content}'


def _read_data_url(spec: str) -> str:
    """Blocking half of :func:`_data_url`. Runs on a worker thread."""
    path = Path(spec).expanduser()
    data = read_media_file(path)
    mime = image_mime(data[:16]) or _MIME.get(path.suffix.lower(), 'image/png')
    return f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')


async def _data_url(spec: str) -> str:
    """A base64 data URL for ``spec``, which is either one already or a path."""
    if spec.startswith('data:'):
        return spec
    return await asyncio.to_thread(_read_data_url, spec)


def _resolve_image(spec: str, sources: dict[str, str], *, allow_host: bool) -> str | None:
    """Turn what the model wrote into something this process can read.

    A host path is only honoured when ``allow_host`` is set, i.e. when there is no
    sandbox: with one, the paths the model sees live in the sandbox, and reading
    the same path on the host would send whatever file is there to the service.
    """
    if spec.startswith('data:'):
        return spec
    if source := sources.get(spec):
        return source
    if not allow_host:
        return None

    expanded = Path(spec).expanduser()
    try:
        if expanded.is_file():
            return str(expanded)
    except OSError:
        # A path long enough or odd enough to upset the OS is not one we can
        # read; fall through to the caller's "cannot find that image" message.
        logger.debug('Could not stat %r while resolving a spatial image', spec[:200])
    return None


def _fail(text: str, **meta: Any) -> ToolResponse:
    """An error the *model* can act on, rather than an exception."""
    return ToolResponse(output=[{'type': 'text', 'text': text}], meta={'success': False, **meta})


def _unreadable(spec: str, sources: dict[str, str]) -> ToolResponse:
    """The "I cannot find that image" failure, naming what would work.

    A path whose filename matches exactly one known image is named, not silently
    substituted: in multi-frame input that is how a wrong frame would slip through.
    """
    known = sorted(sources)
    target = os.path.basename(spec.rstrip('/'))
    near = [p for p in known if target and os.path.basename(p) == target]
    if len(near) == 1:
        return _fail(
            f'Could not read the image {spec!r}. Did you mean {near[0]!r}? Copy image paths verbatim '
            f'from the conversation.',
            error='image_unresolved',
        )
    if known:
        listing = '\n'.join(f'  - {p}' for p in known[:10])
        hint = f'Use one of the image paths from the conversation:\n{listing}'
    else:
        hint = (
            'No image paths are configured for this task, so pass the absolute sandbox path '
            'from the conversation, or a base64 data URL.'
        )
    return _fail(f'Could not read the image {spec!r}. {hint}', error='image_unresolved')


class _RateLimited(Exception):  # noqa: N818 - this is a signal, not a crash
    """A 429 from the service: it is up, it is shedding load. Carries the wait."""

    def __init__(self, retry_after: float, detail: str) -> None:
        super().__init__(detail)
        self.retry_after = retry_after
        self.detail = detail


class _Rejected(Exception):  # noqa: N818 - a 4xx answer, not a transport failure
    """A 4xx other than 429: the service is up and refused this request."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _error_body(resp: Any) -> tuple[Any, str | None]:
    """The decoded error body, if any, and the message it carries."""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - an error body is advisory; the status is the signal
        return None, None
    if not isinstance(body, dict):
        return body, None
    detail = body.get('error') or body.get('error_detail') or body.get('detail')
    return body, str(detail) if detail else None


def _retry_after(headers: Any, body: Any) -> float:
    """Seconds to wait before retrying, from the Retry-After header or the body.

    aiohttp's ``raise_for_status`` builds its ClientResponseError with an empty
    header set, so the only place to read this is here, while the response is
    still in hand. Clamped at both ends: a missing or malformed value becomes the
    default, and an implausibly long one is capped rather than trusted, because a
    gateway in the path -- not our service -- is the likelier source of a 600.
    """
    raw = None
    if headers is not None:
        raw = headers.get('Retry-After') or headers.get('retry-after')
    if raw is None and isinstance(body, dict):
        raw = body.get('retry_after') or body.get('retry_after_s')
    try:
        wait = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER
    if wait <= 0:
        return DEFAULT_RETRY_AFTER
    return min(wait, MAX_RETRY_AFTER)


async def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON and return the decoded body, raising on transport or HTTP error.

    429 is pulled out before ``raise_for_status`` so the Retry-After survives:
    backpressure is not the same failure as an unreachable service, and the
    difference decides whether retrying is worth anything. Other 4xx are pulled
    out too: the request was wrong, and only its body says how.
    """
    resp = await http_client.post(url, json=payload, timeout=timeout)
    if resp.status == 429:
        body, detail = _error_body(resp)
        raise _RateLimited(_retry_after(resp.headers, body), detail or resp.text[:200] or 'HTTP 429')
    if 400 <= resp.status < 500:
        _, detail = _error_body(resp)
        raise _Rejected(resp.status, (detail or resp.text or f'HTTP {resp.status}')[:300])
    resp.raise_for_status()
    body_ok: dict[str, Any] = resp.json()
    return body_ok


async def _call(
    kind: str,
    url: str,
    payload: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, ToolResponse | None]:
    """``(body, None)`` on success, ``(None, failure_response)`` otherwise."""
    try:
        timeout = float(config.get('timeout', DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT

    # One retry, and only for 429. The service sheds load by design once its
    # queue is full, so the first 429 says "too early", not "broken" -- but a
    # second one after waiting says the congestion outlasts this request, and
    # retrying past that just deepens the queue we were asked to back off from.
    loop = asyncio.get_running_loop()
    budget = timeout
    for attempt in (0, 1):
        try:
            async with _semaphore(config):
                started = loop.time()
                body = await _post(url, payload, budget)
            break
        except _RateLimited as e:
            # Both the time the refused request took and the wait come out of the
            # one budget, so the retry cannot push the total past `timeout`.
            budget -= (loop.time() - started) + e.retry_after
            if attempt == 0 and budget >= MIN_RETRY_BUDGET:
                logger.warning(
                    'spatial %s: %s is shedding load (429); waiting %.1fs then retrying once',
                    kind,
                    url,
                    e.retry_after,
                )
                await asyncio.sleep(e.retry_after)
                continue
            logger.warning(
                'spatial %s: %s rate-limited, not retrying (attempt %d): %s', kind, url, attempt + 1, e.detail
            )
            return None, _fail(
                f'The {kind} service at {url} is busy and refused the request (HTTP 429: {e.detail}). '
                f'It is up, not broken -- it is shedding load. Answer from the image itself, '
                f'or try again in about {e.retry_after:.0f}s.',
                error='rate_limited',
                retry_after=e.retry_after,
            )
        except _Rejected as e:
            logger.warning('spatial %s: %s rejected the request (HTTP %d): %s', kind, url, e.status, e.detail)
            return None, _fail(
                f'The {kind} service rejected the request (HTTP {e.status}: {e.detail}). It is up, so the '
                f'problem is the request itself: check the arguments against the tool description and call '
                f'again, or answer from the image.',
                error='rejected',
                status=e.status,
            )
        except Exception as e:  # noqa: BLE001 - any transport failure is the model's to route around
            logger.warning('spatial %s: request to %s failed: %r', kind, url, e)
            return None, _fail(
                f'The {kind} service at {url} could not be reached ({type(e).__name__}: {e}). '
                f'It may be down or restarting. Answer from the image itself, or retry once.',
                error=type(e).__name__,
            )

    if not isinstance(body, dict):
        logger.warning('spatial %s: unexpected response type %s', kind, type(body).__name__)
        return None, _fail(f'The {kind} service returned an unexpected response.', error='bad_response')

    # Default True: an endpoint that omits `ok` is reporting success. Logged
    # because it would also be the first sign of a response shape change.
    if 'ok' not in body:
        logger.debug('spatial %s: response carried no `ok` field; treating as success', kind)
    elif not body.get('ok'):
        detail = body.get('error') or body.get('error_detail') or body
        logger.warning('spatial %s: service reported not-ok: %s', kind, detail)
        return None, _fail(f'The {kind} service rejected the request: {detail}', error='not_ok')

    if warnings := body.get('warnings'):
        # Not a failure, but the model is the only one who can act on some of
        # these -- "your priority key was ignored", "that field is deprecated".
        # Logging them buried the signal in a logfile nobody reads mid-run, so
        # they are also surfaced to the caller via `_warning_note`.
        logger.warning('spatial %s: service warnings: %s', kind, warnings)
    return body, None


def _warning_note(body: dict[str, Any]) -> str:
    """The service's ``warnings`` as one line to append to a tool's output, or ``''``.

    The model cannot read our logs, and some warnings are only actionable by
    whoever wrote the request -- a mistyped field name that silently downgraded
    the request, say. Returning them inline is the only way they land anywhere
    the agent can see.
    """
    warnings = body.get('warnings')
    if not warnings or not isinstance(warnings, list):
        return ''
    notes = []
    for w in warnings:
        if isinstance(w, dict):
            text = w.get('message') or w.get('code')
            if text:
                notes.append(str(text))
        elif w:
            notes.append(str(w))
    if not notes:
        return ''
    return '\nService warnings: ' + '; '.join(notes[:5])


async def _load_image(spec: str, config: dict[str, Any]) -> tuple[str | None, ToolResponse | None]:
    """``(data_url, None)``, or ``(None, failure)``.

    Order: data URL, explicit ``image_sources`` entry, the runtime sandbox, then --
    only when there is no sandbox -- a host-local path.
    """
    sources = _image_sources(config)
    ctx = _sandbox_context()
    try:
        if spec.startswith('data:') or spec in sources:
            return await _data_url(sources.get(spec, spec)), None
        if (url := await _read_sandbox_data_url(spec, config, ctx)) is not None:
            return url, None
        if (resolved := _resolve_image(spec, sources, allow_host=ctx is None)) is not None:
            return await _data_url(resolved), None
    except Exception as e:  # noqa: BLE001 - a bad path is the model's to correct
        logger.warning('spatial: could not read %r: %r', spec[:200], e)
        return None, _fail(f'Could not read the image {spec!r} ({type(e).__name__}: {e}).', error=type(e).__name__)
    return None, _unreadable(spec, sources)


async def _prepare_image(spec: str, context: ToolContext) -> tuple[str | None, dict[str, Any], ToolResponse | None]:
    """``(data_url, config, None)``, or ``(None, config, failure)``."""
    config = _config(context)
    image, failure = await _load_image(spec, config)
    return image, config, failure


def _ok(text: str, body: dict[str, Any] | None = None, **meta: Any) -> ToolResponse:
    """A successful response carrying one block of text.

    Pass the service ``body`` to append any warnings it returned; see
    ``_warning_note`` for why those are not merely logged.
    """
    if body:
        text += _warning_note(body)
    return ToolResponse(output=[{'type': 'text', 'text': text}], meta={'success': True, **meta})


def _fields(item: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    """``['name=value', ...]`` for the fields that are actually present."""
    return [f'{name}={value}' for name in names if (value := item.get(name)) is not None]


class LocateInput(RobustBaseModel):
    # A docstring, not a comment: pydantic puts it in the JSON schema as the
    # parameter object's description, so the model reads it.
    """Locate every visible instance matching a short description."""

    query: str = Field(description='Concrete visual category/attributes, without positional or depth superlatives.')
    image: str = Field(
        description=(
            'Absolute path of the image to measure, copied verbatim from the image list '
            'in the conversation. A base64 data URL also works.'
        )
    )


@tool_registry.register(
    'visual/query_locate',
    meta={
        'name': 'query_locate',
        'description': (
            'Locate every visible instance matching a short language description. Use this '
            'when exact object positions, centers, boxes, counts, or left/right order depend '
            'on reliable grounding. Query a concrete visual category and attributes only. '
            "For 'closest', 'farthest', 'leftmost', or 'second' questions, do not put that "
            'relation in the query: locate all instances of the base category, then compare '
            'their returned centers or query their depths. All benchmark geometry and tool '
            'coordinates use the same normalized 0-999 image space. The tool returns bbox_2d '
            'and point_2d in that space; pass point_2d directly to query_depth when needed. '
            'It may return zero, one, or several instances; do not assume the first instance '
            'is the only one. Prefer this tool for answers that require explicit path points, '
            'object markers, or boxes. Answer simple categorical left/right yes-no questions '
            'directly when visually clear. It cannot infer five-point grasp-finger geometry or '
            'projected 3D cuboid corners from a 2D box, so do not treat its center or box '
            'corners as those answers. Tools are optional. In a multi-image or sampled-video '
            'task, pass the absolute image path in `image`; these tools measure exactly one '
            'image per call, so call the tool once per image you need. Copy the path verbatim '
            'from the list given above.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def locate_tool(params: LocateInput, context: ToolContext) -> ToolResponse:
    """Ground a description to bounding boxes."""
    image, config, failure = await _prepare_image(params.image, context)
    if failure is not None:
        return failure

    payload = {
        'image': image,
        'phrase': params.query,
        'prompt_type': 'box_multi',
        'mode': 'slow',
    }
    url = _endpoint(config, 'locate_url', 'SPATIAL_LOCATE_URL', LOCATE_PATH)
    body, failure = await _call('object-location', url, payload, config)
    if failure is not None:
        return failure
    assert body is not None

    items = ((body.get('result') or {}).get('items')) or []
    if not items:
        return _ok(
            f'No instance of {params.query!r} was found. Either it is absent, or the '
            f'description is too specific or too relational — try a broader category word.',
            body,
            count=0,
            query=params.query,
        )

    lines = [f'Found {len(items)} instance(s) of {params.query!r} (coordinates are normalised 0-999, not pixels):']
    normalised: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        # The service says bbox/point; the model is told bbox_2d/point_2d, which
        # is the spelling the other three tools use.
        entry = dict(item)
        entry['bbox_2d'] = item.get('bbox')
        entry['point_2d'] = item.get('point')
        normalised.append(entry)
        score = item.get('score')
        score_text = f', score {score:.3f}' if isinstance(score, (int, float)) else ''
        rendered = ' '.join(_fields(entry, ('bbox_2d', 'point_2d'))) or '(no geometry returned)'
        lines.append(f'  {i}. {rendered}{score_text}')

    status = body.get('source_status') or {}
    if status and not (status.get('locateanything') or {}).get('ok', True):
        lines.append(
            '  (note: the LocateAnything backend is currently down, so these boxes come '
            'from GroundingDINO alone and are less precise than usual.)'
        )
    lines.append('Pass a point_2d straight to query_depth to measure how far away it is.')

    return _ok(
        '\n'.join(lines),
        body,
        count=len(items),
        query=params.query,
        items=normalised,
        backend=body.get('backend'),
    )


class DepthInput(RobustBaseModel):
    """Measure how far away one or more image points are."""

    points: Annotated[list[Annotated[list[float], Len(2, 2)]], Len(1, None)] = Field(
        description='One or more [x, y] points in normalized 0-999 coordinates.'
    )
    image: str = Field(
        description=(
            'Absolute path of the image to measure, copied verbatim from the image list '
            'in the conversation. A base64 data URL also works.'
        )
    )
    window_size: int = Field(
        default=1,
        ge=1,
        le=101,
        description=(
            'Averaging window in depth-map pixels, odd, 1-101. 1 samples the exact point; '
            'raise it (3, 5) to smooth noise on a textureless or thin surface.'
        ),
    )
    reduction: Literal['center', 'mean', 'median'] = Field(
        default='center',
        description="How to reduce the window: 'center', 'mean' or 'median'. Use 'median' with window_size > 1.",
    )


@tool_registry.register(
    'visual/query_depth',
    meta={
        'name': 'query_depth',
        'description': (
            'Query depth at one or more image points. Pass every point needed for a '
            'closer/farther, front/behind, or distance comparison in one call. Point '
            'coordinates are normalized to 0-999 and the number of points is unrestricted. '
            'For named objects, first use query_locate to obtain their center points, then '
            'query all centers together. Each result contains depth_m, an estimated '
            'camera-axis distance in meters; larger means farther. Monocular depth estimates '
            'can be noisy: answer directly when the relative depth is already visually clear, '
            'and call this tool only when depth is essential and genuinely ambiguous. In a '
            'multi-image or sampled-video task, pass the absolute image path in `image`; these '
            'tools measure exactly one image per call, so call the tool once per image you '
            'need. Copy the path verbatim from the list given above.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def depth_tool(params: DepthInput, context: ToolContext) -> ToolResponse:
    """Metric depth at a set of normalised points."""
    if not params.points:
        return _fail('No points given. Pass at least one [x, y] in normalised 0-999 space.', error='invalid_arguments')
    if bad := [p for p in params.points if len(p) != 2]:
        return _fail(f'Each point must be [x, y]; got {bad[:3]}.', error='invalid_arguments')
    if params.window_size < 1 or params.window_size > 101 or params.window_size % 2 == 0:
        return _fail(
            f'window_size must be an odd number between 1 and 101; got {params.window_size}.',
            error='invalid_arguments',
        )

    image, config, failure = await _prepare_image(params.image, context)
    if failure is not None:
        return failure

    payload = {
        'image': image,
        'points_norm': [[float(x), float(y)] for x, y in params.points],
        'coord_type': 'norm',
        'window_size': params.window_size,
        'reduction': params.reduction,
    }
    url = _endpoint(config, 'depth_url', 'SPATIAL_DEPTH_URL', DEPTH_PATH)
    body, failure = await _call('depth', url, payload, config)
    if failure is not None:
        return failure
    assert body is not None

    points = body.get('points') or []
    if not points:
        return _fail('The depth service returned no measurements.', error='empty')

    lines = ['Depth in metres (depth_m is along the camera axis; larger = further away):']
    for i, p in enumerate(points, 1):
        where = f'point_2d {p["point_norm"]}' if p.get('point_norm') is not None else f'point {i}'
        depth = p.get('depth_m')
        if not isinstance(depth, (int, float)):
            lines.append(f'  {where} -> no depth returned')
            continue
        range_m = p.get('range_m')
        range_text = f' (straight-line distance {range_m:.3f} m)' if isinstance(range_m, (int, float)) else ''
        lines.append(f'  {where} -> {depth:.3f} m{range_text}')
    if len(points) > 1:
        lines.append(
            'Compare these against each other rather than treating any one as a measurement — '
            'the relative ordering is what this model is good at.'
        )
    else:
        lines.append(
            'This is a single-view estimate, so it is reliable for comparing points and much '
            'less so as an absolute measurement. Do not quote it as a measured distance.'
        )

    return _ok(
        '\n'.join(lines),
        body,
        count=len(points),
        points=points,
        model=body.get('model'),
        is_metric=body.get('is_metric'),
        image_size=body.get('image_size'),
    )


#: The literal query that switches the pose endpoint into camera-motion mode,
#: matching the tool description. Only case and surrounding whitespace are
#: forgiven; any other wording stays in object-orientation mode.
CAMERA_MOTION_QUERY = 'camera motion'


class PoseInput(RobustBaseModel):
    """An object's facing direction, or how the camera moved between two views."""

    query: str = Field(description="Object description, or 'camera motion'.")
    images: Annotated[list[str], Len(1, 2)] = Field(
        description=(
            'Absolute image paths, copied verbatim from the image list in the conversation. '
            'Order is the semantics: [from, to] for camera motion; a single path for object '
            'orientation.'
        )
    )


@tool_registry.register(
    'visual/query_pose',
    meta={
        'name': 'query_pose',
        'description': (
            "Query a named object's facing direction in one image, or camera motion across "
            'images. For an object, pass only a short visible description; the result returns '
            'azimuth_deg (horizontal facing, 0-359), elevation_deg, roll_deg and '
            'symmetry_order. symmetry_order is the rotational symmetry order: 4 means the '
            'object looks the same every 90 degrees, so its azimuth is inherently ambiguous — '
            'do not make exact orientation claims when it is greater than 1. For viewpoint '
            "change, pass exactly the literal string query='camera motion' — any other "
            "wording (for example 'camera movement', 'viewpoint change', or a whole "
            'sentence) is silently treated as an object name and returns object orientation '
            "instead, with no error. Check that the result's kind is 'camera' before reading "
            'position or view_rotation; if you get azimuth_deg or symmetry_order back, the '
            'query missed and the answer is not about camera motion. Each camera result '
            "describes to_image relative to from_image and is printed as 'moved <direction> "
            "(axes [...]), view rotated <word> (dominant axis <axis>)'. 'moved' is the "
            'direction the camera centre travelled, as a unit direction with unknown distance, '
            "in from_image's camera frame (+X right, +Y up, -Z forward); components smaller "
            "than 20% of the largest are dropped. 'view rotated' reports ONLY the largest of "
            'the three rotation components (turn, tilt, roll), so a smaller left/right turn '
            'can be missing from the output. Read it from the dominant axis: +Y means the '
            'camera turned LEFT and -Y means it turned RIGHT (for Y trust the axis sign, not '
            'the word); +X means it tilted down and -X up; Z is roll around the viewing axis, '
            "where 'clockwise'/'counterclockwise' describes how the picture appears to rotate "
            '(the camera rolled the opposite way). Roll is never a left/right turn: when the '
            'dominant axis is Z, judge left/right turning from the images. Rotations of a few '
            'degrees and small sideways movement look alike, so check the result against how '
            'the scene shifts between the images: scene content shifting right means the '
            "camera turned (or moved) left. A first-person 'front-right' style answer can mean "
            'moving forward while turning right, so for movement questions consider both '
            "'moved' and 'view rotated'. same-position means no reliable translation. Pass "
            'both absolute image paths in `images`; the order is the semantics: `images[0]` is '
            'the reference (from) and `images[1]` is the target (to). Swap them only when the '
            'question explicitly asks for the reverse direction. Use object mode only for '
            'object orientation and camera-motion mode only for actual viewpoint change; '
            'otherwise answer directly.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def pose_tool(params: PoseInput, context: ToolContext) -> ToolResponse:
    """Object orientation, or camera motion between two views."""
    camera_mode = params.query.strip().lower() == CAMERA_MOTION_QUERY

    # Two modes share one endpoint and each has its own arity. The message names
    # the *other* mode, since picking the wrong one is how a model gets here.
    if not params.images:
        return _fail('No image given.', error='invalid_arguments')
    if camera_mode and len(params.images) != 2:
        return _fail(
            f'Camera motion needs exactly two images (from, to); got {len(params.images)}. '
            f'For a single image, pass an object description as `query` instead.',
            error='invalid_arguments',
        )
    if not camera_mode and len(params.images) != 1:
        return _fail(
            f'Object orientation takes one image; got {len(params.images)}. Pass '
            f"query='camera motion' if you meant to compare two views.",
            error='invalid_arguments',
        )

    config = _config(context)
    images: list[str] = []
    for spec in params.images:
        image, failure = await _load_image(spec, config)
        if failure is not None:
            return failure
        assert image is not None
        images.append(image)

    url = _endpoint(config, 'pose_url', 'SPATIAL_POSE_URL', POSE_PATH)
    query = CAMERA_MOTION_QUERY if camera_mode else params.query
    body, failure = await _call('pose', url, {'images': images, 'query': query}, config)
    if failure is not None:
        return failure
    assert body is not None

    result = body.get('result') or []
    if not result:
        return _fail(
            f'No pose was returned for {params.query!r}. For object orientation the object '
            f'must be detectable in the image — check with query_locate first.',
            error='empty',
        )

    lines: list[str] = []
    if body.get('kind') == 'camera':
        lines.append('Camera motion (each row describes to_image relative to from_image):')
        for r in result:
            pos = r.get('position') or {}
            rot = r.get('view_rotation') or {}
            lines.append(
                f'  image {r.get("from_image")} -> {r.get("to_image")}: '
                f'moved {pos.get("direction")} (axes {pos.get("axes")}), '
                f'view rotated {rot.get("direction")} '
                f'(dominant axis {rot.get("dominant_axis")})'
            )
        lines.append('Image indices start at 1. Axes are signed right-handed: +X right, +Y up, -Z forward.')
    else:
        lines.append(f'Orientation of {params.query!r} (degrees; bbox_2d is normalised 0-999):')
        ambiguous = False
        for r in result:
            rendered = ' '.join(_fields(r, ('bbox_2d', 'azimuth_deg', 'elevation_deg', 'roll_deg', 'symmetry_order')))
            lines.append(f'  {rendered}' if rendered else '  (no orientation returned)')
            symmetry = r.get('symmetry_order')
            if isinstance(symmetry, (int, float)) and symmetry > 1:
                ambiguous = True
        lines.append(
            'azimuth_deg is the direction the object faces, elevation_deg its tilt, roll_deg '
            'its rotation in the image plane.'
        )
        if ambiguous:
            lines.append(
                'symmetry_order is greater than 1, so this object looks the same from several '
                'angles and its azimuth is inherently ambiguous — do not make exact '
                'orientation claims from it.'
            )

    return _ok(
        '\n'.join(lines),
        body,
        kind=body.get('kind'),
        query=params.query,
        result=result,
        backend=body.get('backend'),
    )


class SegmentInput(RobustBaseModel):
    """Segment every visible instance matching a short description."""

    query: str = Field(description='Short visible description of the object to segment.')
    image: str = Field(
        description=(
            'Absolute path of the image to measure, copied verbatim from the image list '
            'in the conversation. A base64 data URL also works.'
        )
    )


@tool_registry.register(
    'visual/query_segment',
    meta={
        'name': 'query_segment',
        'description': (
            'Segment every visible instance matching a short language description, returning '
            'bbox_2d [x1,y1,x2,y2], polygon_2d (the mask outline, outer list is one entry per '
            "contour), area_pixels (the segmented mask's pixel area) and score. It does not "
            'return a center point; derive one from bbox_2d if you need it. Coordinates are '
            'normalized to 0-999. Prefer this over query_locate when the question depends on '
            "an object's actual extent rather than just where it is: comparing which object "
            'is larger, how much of the view something occupies, or separating overlapping '
            'instances of the same category. Use query_locate instead when you only need '
            'positions or counts, since it is cheaper. area_pixels is the pixel area of the '
            'mask, not a real-world area, so only compare it between objects in the same image.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def segment_tool(params: SegmentInput, context: ToolContext) -> ToolResponse:
    """Segment a description into mask outlines."""
    image, config, failure = await _prepare_image(params.image, context)
    if failure is not None:
        return failure

    url = _endpoint(config, 'segment_url', 'SPATIAL_SEGMENT_URL', SEGMENT_PATH)
    body, failure = await _call('segmentation', url, {'image': image, 'query': params.query}, config)
    if failure is not None:
        return failure
    assert body is not None

    result = body.get('result') or []
    if not result:
        return _ok(
            f'Nothing matching {params.query!r} was segmented. Either it is absent, or the '
            f'description is too specific — try a broader category word.',
            body,
            count=0,
            query=params.query,
        )

    lines = [f'Segmented {len(result)} instance(s) of {params.query!r} (coordinates are normalised 0-999, not pixels):']
    for i, r in enumerate(result, 1):
        score = r.get('score')
        score_text = f', score {score:.3f}' if isinstance(score, (int, float)) else ''
        contours = r.get('polygon_2d') or []
        vertices = sum(len(c) for c in contours if isinstance(c, list))
        rendered = ' '.join(_fields(r, ('bbox_2d', 'area_pixels'))) or '(no geometry returned)'
        lines.append(f'  {i}. {rendered}{score_text}, outline: {len(contours)} contour(s), {vertices} vertices')
    image_size = body.get('image_size')
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        lines.append(f'Source image is {image_size[0]}x{image_size[1]} px.')
    lines.append(
        'area_pixels is mask area in pixels, not a real-world area — compare it only between '
        'objects in this same image.'
    )

    return _ok(
        '\n'.join(lines),
        body,
        count=len(result),
        query=params.query,
        result=result,
        image_size=body.get('image_size'),
    )


__all__ = [
    'DEFAULT_BASE_URL',
    'DEFAULT_MAX_CONCURRENCY',
    'DEFAULT_RETRY_AFTER',
    'DEFAULT_TIMEOUT',
    'MAX_RETRY_AFTER',
    'MIN_RETRY_BUDGET',
    'SPATIAL_TOOLS',
    'DepthInput',
    'LocateInput',
    'PoseInput',
    'SegmentInput',
    'depth_tool',
    'locate_tool',
    'pose_tool',
    'segment_tool',
]
