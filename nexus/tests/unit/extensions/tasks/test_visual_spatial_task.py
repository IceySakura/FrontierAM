"""Tests for the visual_spatial task and its spatial tools."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic import ValidationError

from nexus.core import tool_registry
from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.visual_agent.tools import DEFAULT_VISUAL_TOOLS
from nexus.extensions.tasks.visual_spatial import tools as spatial_tools
from nexus.extensions.tasks.visual_spatial.task import VisualSpatialTaskInput, visual_spatial_task
from nexus.extensions.tasks.visual_spatial.tools import (
    SPATIAL_CONFIG_SECTION,
    SPATIAL_IMAGE_SOURCES_KEY,
    SPATIAL_TOOLS,
    DepthInput,
    LocateInput,
    PoseInput,
    SegmentInput,
    _resolve_image,
    depth_tool,
    locate_tool,
    pose_tool,
    segment_tool,
)


SANDBOX = '/tmp/scope123/shelf.png'


@pytest.fixture
def image(tmp_path: Any) -> str:
    path = tmp_path / 'shelf.png'
    path.write_bytes(
        bytes.fromhex(
            '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4'
            '890000000a49444154789c6300010000050001'
            '0d0a2db40000000049454e44ae426082'
        )
    )
    return str(path)


@pytest.fixture
def context(image: str) -> ToolContext:
    return ToolContext(
        run_id='r',
        tool_configs={SPATIAL_CONFIG_SECTION: {SPATIAL_IMAGE_SOURCES_KEY: {SANDBOX: image}}},
    )


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    bodies: dict[str, dict[str, Any]] = {
        'locate': {'ok': True, 'result': {'items': [{'bbox': [1, 2, 3, 4], 'point': [2, 3], 'score': 0.9}]}},
        'depth': {'ok': True, 'points': [{'point_norm': [2, 3], 'depth_m': 1.5, 'range_m': 1.6}]},
        'pose': {'ok': True, 'kind': 'object', 'result': [{'bbox_2d': [1, 2, 3, 4], 'azimuth_deg': 272}]},
        'segment': {'ok': True, 'image_size': [64, 64], 'result': [{'bbox_2d': [1, 2, 3, 4], 'area_pixels': 10}]},
    }

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append({'url': url, 'payload': payload, 'timeout': timeout})
        for key, body in bodies.items():
            if key in url:
                return body
        raise AssertionError(f'unexpected url {url}')

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    return calls


def test_default_tool_set_is_unchanged() -> None:
    assert DEFAULT_VISUAL_TOOLS == ('jupyter_with_image',)
    for name in SPATIAL_TOOLS:
        assert name not in DEFAULT_VISUAL_TOOLS


@pytest.mark.parametrize(
    ('registry_name', 'schema_name'),
    [
        ('visual/query_locate', 'query_locate'),
        ('visual/query_depth', 'query_depth'),
        ('visual/query_pose', 'query_pose'),
        ('visual/query_segment', 'query_segment'),
    ],
)
def test_the_model_sees_the_unprefixed_name(registry_name: str, schema_name: str) -> None:
    meta = tool_registry.get(registry_name).__tool_meta__
    assert meta.name == schema_name
    assert meta.tool_type.value == 'local'


def test_a_data_url_passes_straight_through() -> None:
    assert _resolve_image('data:image/png;base64,AAAA', {}, allow_host=False) == 'data:image/png;base64,AAAA'


def test_the_mapping_outranks_a_host_file_at_the_same_path(tmp_path: Any) -> None:
    decoy = tmp_path / 'shelf.png'
    decoy.write_bytes(b'not the image')
    real = tmp_path / 'scope' / 'shelf.png'
    real.parent.mkdir()
    real.write_bytes(b'the real image')
    assert _resolve_image(str(decoy), {str(decoy): str(real)}, allow_host=True) == str(real)


def test_a_host_path_is_read_only_when_there_is_no_sandbox(image: str) -> None:
    """With a sandbox the model's paths live there; the same path on the host is someone else's file."""
    assert _resolve_image(image, {}, allow_host=True) == image
    assert _resolve_image(image, {}, allow_host=False) is None


def test_a_matching_filename_is_never_substituted() -> None:
    sources = {'/s1/a.png': '/h1/a.png'}
    assert _resolve_image('/elsewhere/a.png', sources, allow_host=True) is None


@pytest.mark.asyncio
async def test_a_unique_filename_match_is_named_not_used(context: ToolContext, posted: list[dict[str, Any]]) -> None:
    """In multi-frame input a silent swap is how a wrong frame would get measured."""
    result = await locate_tool(LocateInput(query='chair', image='/elsewhere/shelf.png'), context)
    assert result.meta['error'] == 'image_unresolved'
    assert f'Did you mean {SANDBOX!r}' in result.output[0]['text']
    assert posted == []


@pytest.mark.asyncio
async def test_an_unreadable_image_names_the_valid_paths(context: ToolContext) -> None:
    result = await locate_tool(LocateInput(query='chair', image='/nope/zzz.png'), context)
    assert result.meta['success'] is False
    assert result.meta['error'] == 'image_unresolved'
    assert SANDBOX in result.output[0]['text']


@pytest.mark.asyncio
async def test_locate_sends_phrase_and_inlines_the_bytes(context: ToolContext, posted: list[dict[str, Any]]) -> None:
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is True
    payload = posted[0]['payload']
    assert payload['phrase'] == 'chair'
    assert 'query' not in payload
    assert payload['image'].startswith('data:image/png;base64,')
    assert 'bbox_2d=[1, 2, 3, 4]' in result.output[0]['text']
    assert result.meta['items'][0]['point_2d'] == [2, 3]


@pytest.mark.asyncio
async def test_depth_sends_normalised_points(context: ToolContext, posted: list[dict[str, Any]]) -> None:
    result = await depth_tool(DepthInput(points=[[2, 3]], image=SANDBOX, window_size=5, reduction='median'), context)
    assert result.meta['success'] is True
    payload = posted[0]['payload']
    assert payload['points_norm'] == [[2.0, 3.0]]
    assert payload['coord_type'] == 'norm'
    assert payload['window_size'] == 5


@pytest.mark.asyncio
async def test_pose_camera_mode_needs_two_images(context: ToolContext) -> None:
    result = await pose_tool(PoseInput(query='camera motion', images=[SANDBOX]), context)
    assert result.meta['success'] is False
    assert result.meta['error'] == 'invalid_arguments'
    assert 'object description' in result.output[0]['text']


@pytest.mark.asyncio
async def test_pose_object_mode_rejects_two_images(context: ToolContext) -> None:
    result = await pose_tool(PoseInput(query='chair', images=[SANDBOX, SANDBOX]), context)
    assert result.meta['success'] is False
    assert result.meta['error'] == 'invalid_arguments'
    assert 'camera motion' in result.output[0]['text']


@pytest.mark.asyncio
async def test_camera_motion_query_is_normalised_and_other_wordings_are_not(
    context: ToolContext, posted: list[dict[str, Any]]
) -> None:
    await pose_tool(PoseInput(query=' Camera Motion ', images=[SANDBOX, SANDBOX]), context)
    assert posted[0]['payload']['query'] == 'camera motion'

    for other in ('camera', 'camera pose', 'viewpoint motion', 'camera movement'):
        result = await pose_tool(PoseInput(query=other, images=[SANDBOX, SANDBOX]), context)
        assert result.meta['success'] is False
        assert result.meta['error'] == 'invalid_arguments'
        assert "query='camera motion'" in result.output[0]['text']


@pytest.mark.asyncio
async def test_segment_summarises_the_polygon(context: ToolContext, posted: list[dict[str, Any]]) -> None:
    result = await segment_tool(SegmentInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is True
    assert 'area_pixels=10' in result.output[0]['text']


@pytest.mark.parametrize('window_size', [2, 4])
@pytest.mark.asyncio
async def test_depth_rejects_an_even_window_locally(context: ToolContext, window_size: int) -> None:
    result = await depth_tool(DepthInput(points=[[1, 2]], image=SANDBOX, window_size=window_size), context)
    assert result.meta['success'] is False
    assert result.meta['error'] == 'invalid_arguments'
    assert 'odd number' in result.output[0]['text']


@pytest.mark.parametrize('bad', [{'window_size': 0}, {'window_size': 102}, {'reduction': 'avg'}])
def test_depth_range_and_reduction_are_checked_by_the_schema(bad: dict[str, Any]) -> None:
    """Rejected before the call, so the service never sees them and answers 400."""
    with pytest.raises(ValidationError):
        DepthInput(points=[[1, 2]], image=SANDBOX, **bad)


@pytest.mark.asyncio
async def test_depth_rejects_malformed_points(context: ToolContext) -> None:
    with pytest.raises(ValidationError):
        DepthInput(points=[], image=SANDBOX)
    with pytest.raises(ValidationError):
        DepthInput(points=[[1, 2, 3]], image=SANDBOX)

    bypass = DepthInput.model_construct(points=[], image=SANDBOX, window_size=1, reduction='center')
    assert (await depth_tool(bypass, context)).meta['success'] is False


@pytest.mark.asyncio
async def test_http_200_with_ok_false_is_a_failure(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {'ok': False, 'error': 'image_not_found: no such file'}

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is False
    assert result.meta['error'] == 'not_ok'
    assert 'image_not_found' in result.output[0]['text']


@pytest.mark.asyncio
async def test_a_missing_ok_field_is_success(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {'result': {'items': [{'bbox': [1, 2, 3, 4], 'point': [2, 3]}]}}

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is True


@pytest.mark.asyncio
async def test_a_transport_failure_becomes_an_observation(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise TimeoutError('connect timed out')

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is False
    assert 'could not be reached' in result.output[0]['text']


def _fake_http(monkeypatch: pytest.MonkeyPatch, status: int, body: Any, text: str = '') -> None:
    from types import SimpleNamespace

    def raise_for_status() -> None:
        if status >= 400:
            raise RuntimeError(f'HTTP {status}')

    resp = SimpleNamespace(status=status, headers={}, text=text, json=lambda: body, raise_for_status=raise_for_status)

    async def post(url: str, json: dict[str, Any], timeout: float) -> Any:
        return resp

    monkeypatch.setattr(spatial_tools, 'http_client', SimpleNamespace(post=post))


@pytest.mark.asyncio
async def test_a_4xx_tells_the_model_to_fix_the_request(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """The service is up; calling it unreachable would hide that the arguments were wrong."""
    _fake_http(monkeypatch, 422, {'detail': 'points_norm out of range'})
    result = await depth_tool(DepthInput(points=[[1, 2]], image=SANDBOX), context)
    text = result.output[0]['text']
    assert result.meta['error'] == 'rejected'
    assert result.meta['status'] == 422
    assert 'points_norm out of range' in text
    assert 'could not be reached' not in text


@pytest.mark.asyncio
async def test_a_5xx_is_still_reported_as_unreachable(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_http(monkeypatch, 503, None, text='upstream down')
    result = await depth_tool(DepthInput(points=[[1, 2]], image=SANDBOX), context)
    assert 'could not be reached' in result.output[0]['text']


@pytest.mark.asyncio
async def test_no_detections_is_a_successful_answer(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {'ok': True, 'result': {'items': []}}

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='unicorn', image=SANDBOX), context)
    assert result.meta['success'] is True
    assert result.meta['count'] == 0


@pytest.mark.asyncio
async def test_symmetry_order_above_one_warns_about_ambiguity(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {
            'ok': True,
            'kind': 'object',
            'result': [{'bbox_2d': [1, 2, 3, 4], 'azimuth_deg': 272, 'symmetry_order': 4}],
        }

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await pose_tool(PoseInput(query='chair', images=[SANDBOX]), context)
    text = result.output[0]['text']
    assert 'symmetry_order=4' in text
    assert 'ambiguous' in text


@pytest.mark.asyncio
async def test_camera_mode_says_indices_start_at_one(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {
            'ok': True,
            'kind': 'camera',
            'result': [
                {
                    'from_image': 1,
                    'to_image': 2,
                    'position': {'direction': 'forward-left', 'axes': []},
                    'view_rotation': {'direction': 'none', 'axes': [], 'dominant_axis': 'none'},
                }
            ],
        }

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await pose_tool(PoseInput(query='camera motion', images=[SANDBOX, SANDBOX]), context)
    text = result.output[0]['text']
    assert 'forward-left' in text
    assert 'start at 1' in text


@pytest.mark.asyncio
async def test_pose_description_reads_turns_from_the_axis_sign(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pose service labels a positive yaw as ``right`` although its own
    ``+Y`` axis is a left turn, so ``view rotated right (dominant axis +Y)``
    means the camera turned left. It also reports only the largest rotation,
    and its ``clockwise`` describes the picture, not the camera. Until the
    service is fixed the description has to steer the model to the axis sign;
    once it is fixed, this guidance becomes wrong and must change with it.

    The description quotes the rendered camera line, so the rendering is
    pinned here too: changing one without the other misleads the model.
    """
    description = tool_registry.get('visual/query_pose').__tool_meta__.description
    assert '+Y means the camera turned LEFT' in description
    assert 'trust the axis sign, not the word' in description
    assert 'Roll is never a left/right turn' in description
    assert 'directly matches ordinary words' not in description
    assert "printed as 'moved <direction> (axes [...]), view rotated <word> (dominant axis <axis>)'" in description

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {
            'ok': True,
            'kind': 'camera',
            'result': [
                {
                    'from_image': 1,
                    'to_image': 2,
                    'position': {'direction': 'forward-left', 'axes': ['-X', '-Z']},
                    'view_rotation': {'direction': 'right', 'axes': ['+Y'], 'dominant_axis': '+Y'},
                }
            ],
        }

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await pose_tool(PoseInput(query='camera motion', images=[SANDBOX, SANDBOX]), context)
    assert "moved forward-left (axes ['-X', '-Z']), view rotated right (dominant axis +Y)" in result.output[0]['text']


@pytest.mark.asyncio
async def test_no_tool_ever_renders_none_to_the_model(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = {
        'locate': {'ok': True, 'result': {'items': [{}]}},
        'depth': {'ok': True, 'points': [{}]},
        'segment': {'ok': True, 'result': [{}]},
        'pose': {'ok': True, 'kind': 'object', 'result': [{}]},
    }

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        for key, body in bodies.items():
            if key in url:
                return body
        raise AssertionError(f'unexpected url {url}')

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    results = [
        await locate_tool(LocateInput(query='x', image=SANDBOX), context),
        await depth_tool(DepthInput(points=[[1, 2]], image=SANDBOX), context),
        await segment_tool(SegmentInput(query='x', image=SANDBOX), context),
        await pose_tool(PoseInput(query='x', images=[SANDBOX]), context),
    ]
    for result in results:
        assert 'None' not in result.output[0]['text']


@pytest.mark.asyncio
async def test_the_timeout_is_configurable(
    context: ToolContext, posted: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = ToolContext(
        run_id='r',
        tool_configs={
            SPATIAL_CONFIG_SECTION: {
                SPATIAL_IMAGE_SOURCES_KEY: dict(
                    context.tool_configs[SPATIAL_CONFIG_SECTION][SPATIAL_IMAGE_SOURCES_KEY]
                ),
                'timeout': 7.5,
            }
        },
    )
    await locate_tool(LocateInput(query='chair', image=SANDBOX), ctx)
    assert posted[0]['timeout'] == 7.5


@pytest.mark.parametrize('override', [None, 'http://10.0.0.2:1234/x'])
def test_endpoint_uses_base_or_explicit_override(override):
    config = {'base_url': 'http://10.0.0.1', 'locate_url': override}
    url = spatial_tools._endpoint(config, 'locate_url', 'SPATIAL_LOCATE_URL_UNSET', spatial_tools.LOCATE_PATH)
    assert url == (override or 'http://10.0.0.1:9028/v1/grounding/locate')


@pytest.mark.asyncio
async def test_a_429_is_retried_once_after_the_stated_wait(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service sheds load on purpose, so the first 429 means "too early"."""
    attempts: list[float] = []
    slept: list[float] = []

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        attempts.append(timeout)
        if len(attempts) == 1:
            raise spatial_tools._RateLimited(2.0, 'queue_full')
        return {'ok': True, 'result': {'items': [{'bbox': [1, 2, 3, 4], 'point': [2, 3]}]}}

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    monkeypatch.setattr(spatial_tools.asyncio, 'sleep', _fake_sleep)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)

    assert result.meta['success'] is True
    assert len(attempts) == 2
    assert slept == [2.0]
    # The wait comes out of this request's own budget, so the retry does not
    # push total time past what the caller allowed.
    assert attempts[0] == spatial_tools.DEFAULT_TIMEOUT
    assert attempts[1] == pytest.approx(spatial_tools.DEFAULT_TIMEOUT - 2.0, abs=0.5)


@pytest.mark.asyncio
async def test_the_refused_request_time_counts_against_the_retry(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 that arrives after a long queue wait leaves too little budget to retry."""
    import time

    attempts: list[float] = []
    ctx = ToolContext(
        run_id='r',
        tool_configs={
            SPATIAL_CONFIG_SECTION: {**context.tool_configs[SPATIAL_CONFIG_SECTION], 'timeout': 1.5},
        },
    )

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        attempts.append(timeout)
        time.sleep(0.5)
        raise spatial_tools._RateLimited(0.2, 'queue_full')

    async def _fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    monkeypatch.setattr(spatial_tools.asyncio, 'sleep', _fake_sleep)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), ctx)

    # 1.5 s - 0.5 s spent - 0.2 s wait leaves under MIN_RETRY_BUDGET.
    assert attempts == [1.5]
    assert result.meta['error'] == 'rate_limited'


@pytest.mark.asyncio
async def test_a_second_429_gives_up_instead_of_queueing_harder(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[float] = []

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        attempts.append(timeout)
        raise spatial_tools._RateLimited(3.0, 'queue_full')

    async def _fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    monkeypatch.setattr(spatial_tools.asyncio, 'sleep', _fake_sleep)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)

    assert len(attempts) == 2, 'retrying past the second 429 only deepens the queue we were told to leave'
    assert result.meta['success'] is False
    assert result.meta['error'] == 'rate_limited'
    assert result.meta['retry_after'] == 3.0
    # A 429 is not an unreachable service; saying so would send the model looking
    # for an outage that is not there.
    text = result.output[0]['text']
    assert 'busy' in text
    assert 'could not be reached' not in text


@pytest.mark.asyncio
async def test_a_429_with_no_retry_after_uses_the_default_wait(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise spatial_tools._RateLimited(spatial_tools.DEFAULT_RETRY_AFTER, 'busy')

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    monkeypatch.setattr(spatial_tools.asyncio, 'sleep', _fake_sleep)
    await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert slept == [spatial_tools.DEFAULT_RETRY_AFTER]


@pytest.mark.parametrize(
    ('header', 'expected'),
    [
        (None, spatial_tools.DEFAULT_RETRY_AFTER),
        ('7', 7.0),
        ('0', spatial_tools.DEFAULT_RETRY_AFTER),
        ('-1', spatial_tools.DEFAULT_RETRY_AFTER),
        ('soon', spatial_tools.DEFAULT_RETRY_AFTER),
        ('600', spatial_tools.MAX_RETRY_AFTER),
        ('2.5', 2.5),
    ],
)
def test_retry_after_is_clamped(header, expected) -> None:
    """An implausible wait is likelier to come from a gateway than from our service."""
    headers = {} if header is None else {'Retry-After': header}
    assert spatial_tools._retry_after(headers, None) == expected


def test_retry_after_falls_back_to_the_body() -> None:
    assert spatial_tools._retry_after({}, {'retry_after': 4}) == 4.0


@pytest.mark.asyncio
async def test_service_warnings_reach_the_model(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging alone buried these; the model is the only one who can act on some."""

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {
            'ok': True,
            'result': {'items': [{'bbox': [1, 2, 3, 4], 'point': [2, 3]}]},
            'warnings': [{'code': 'priority_key_alias', 'message': 'Key was read from the wrong field.'}],
        }

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    assert 'Service warnings' in result.output[0]['text']
    assert 'Key was read from the wrong field.' in result.output[0]['text']


@pytest.mark.asyncio
async def test_a_warning_is_also_shown_on_an_empty_result(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty result is exactly when the reason for it matters."""

    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {'ok': True, 'result': {'items': []}, 'warnings': ['locateanything degraded to dino']}

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await locate_tool(LocateInput(query='unicorn', image=SANDBOX), context)
    text = result.output[0]['text']
    assert result.meta['count'] == 0
    assert 'locateanything degraded to dino' in text


@pytest.mark.parametrize('warnings', [None, [], 'not-a-list', [{}], [{'code': None, 'message': None}]])
def test_a_body_without_usable_warnings_adds_nothing(warnings) -> None:
    body: dict[str, Any] = {'ok': True, 'warnings': warnings}
    assert spatial_tools._warning_note(body) == ''


@pytest.mark.parametrize('image_size', [[], [64], 'big', None])
@pytest.mark.asyncio
async def test_a_malformed_image_size_is_skipped(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch, image_size: Any
) -> None:
    async def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {'ok': True, 'image_size': image_size, 'result': [{'bbox_2d': [1, 2, 3, 4], 'area_pixels': 10}]}

    monkeypatch.setattr(spatial_tools, '_post', _fake_post)
    result = await segment_tool(SegmentInput(query='chair', image=SANDBOX), context)
    assert result.meta['success'] is True
    assert 'Source image is' not in result.output[0]['text']


def test_each_event_loop_gets_its_own_semaphore() -> None:
    """A semaphore waited on in one loop raises if a later loop waits on it."""
    import asyncio

    async def grab() -> tuple[Any, Any]:
        return spatial_tools._semaphore({'max_concurrency': 2}), spatial_tools._semaphore({'max_concurrency': 2})

    (a1, a2), (b1, _) = asyncio.run(grab()), asyncio.run(grab())
    assert a1 is a2
    assert a1 is not b1


PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Already-acquired runtimes whose sandbox holds ``files`` (path -> base64); records every requested path."""
    from types import SimpleNamespace

    from nexus.runtime.executors.models.file import DownloadFileResponse, FileItem

    state: dict[str, Any] = {'files': {}, 'requested': [], 'runtime_names': []}

    def runtime(name: str) -> Any:
        async def download_file(items: list[FileItem]) -> DownloadFileResponse:
            state['runtime_names'].append(name)
            state['requested'].extend(item.path for item in items)
            found = [
                FileItem(path=i.path, content=state['files'][i.path], encoding='base64')
                for i in items
                if i.path in state['files']
            ]
            return DownloadFileResponse(files=found)

        return SimpleNamespace(download_file=download_file)

    async def get_runtime(name: str = 'main') -> Any:
        raise AssertionError('reading an image must not start a sandbox')

    context = SimpleNamespace(
        runtime_provider=object(),
        _runtimes={'main': runtime('main'), 'vision': runtime('vision')},
        get_runtime=get_runtime,
    )
    monkeypatch.setattr(spatial_tools, 'require_context', lambda: context)
    return state


def test_the_context_still_tracks_acquired_runtimes_privately() -> None:
    """The sandbox read looks up ``_runtimes`` because there is no public accessor; fail loudly if it moves."""
    from nexus.core.context import ExecutionContext

    assert '_runtimes' in ExecutionContext.__private_attributes__


@pytest.mark.asyncio
async def test_a_sandbox_path_is_read_back_from_the_runtime(sandbox, posted) -> None:
    generated = '/workspace/crops/chair.png'
    sandbox['files'][generated] = PNG_B64
    result = await locate_tool(LocateInput(query='chair', image=generated), ToolContext(run_id='r'))
    assert result.meta['success'] is True
    assert sandbox['requested'] == [generated]
    assert posted[0]['payload']['image'] == f'data:image/png;base64,{PNG_B64}'


@pytest.mark.asyncio
async def test_pose_reads_both_views_from_the_sandbox(sandbox, posted) -> None:
    sandbox['files'].update({'/s/a.png': PNG_B64, '/s/b.png': PNG_B64})
    await pose_tool(PoseInput(query='camera motion', images=['/s/a.png', '/s/b.png']), ToolContext(run_id='r'))
    assert sandbox['requested'] == ['/s/a.png', '/s/b.png']
    assert len(posted[0]['payload']['images']) == 2


@pytest.mark.asyncio
async def test_the_runtime_name_is_configurable(sandbox, posted) -> None:
    sandbox['files']['/s/a.png'] = PNG_B64
    ctx = ToolContext(run_id='r', tool_configs={SPATIAL_CONFIG_SECTION: {'runtime': 'vision'}})
    await locate_tool(LocateInput(query='chair', image='/s/a.png'), ctx)
    assert sandbox['runtime_names'] == ['vision']


@pytest.mark.asyncio
async def test_explicit_sources_and_data_urls_skip_the_sandbox(sandbox, context, posted) -> None:
    await locate_tool(LocateInput(query='chair', image=SANDBOX), context)
    await locate_tool(LocateInput(query='chair', image=f'data:image/png;base64,{PNG_B64}'), context)
    assert sandbox['requested'] == []
    assert len(posted) == 2


@pytest.mark.asyncio
async def test_a_path_missing_from_the_sandbox_lists_known_paths(sandbox, context) -> None:
    result = await locate_tool(LocateInput(query='chair', image='/nope/zzz.png'), context)
    assert sandbox['requested'] == ['/nope/zzz.png']
    assert result.meta['error'] == 'image_unresolved'
    assert SANDBOX in result.output[0]['text']


@pytest.mark.asyncio
async def test_an_oversized_sandbox_image_is_rejected(sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spatial_tools, 'MAX_MEDIA_BYTES', 30)
    sandbox['files']['/s/big.png'] = PNG_B64
    result = await locate_tool(LocateInput(query='chair', image='/s/big.png'), ToolContext(run_id='r'))
    assert result.meta['success'] is False
    assert '64 MiB' in result.output[0]['text']


@pytest.mark.asyncio
async def test_with_a_sandbox_a_host_path_is_never_read(sandbox, image, posted) -> None:
    """``image`` exists on the host but not in the sandbox, so it must not be sent anywhere."""
    result = await locate_tool(LocateInput(query='chair', image=image), ToolContext(run_id='r'))
    assert sandbox['requested'] == [image]
    assert result.meta['error'] == 'image_unresolved'
    assert posted == []


@pytest.mark.asyncio
async def test_a_runtime_not_yet_acquired_is_not_started(monkeypatch: pytest.MonkeyPatch, image, posted) -> None:
    from types import SimpleNamespace

    async def get_runtime(name: str = 'main') -> Any:
        raise AssertionError('reading an image must not start a sandbox')

    context = SimpleNamespace(runtime_provider=object(), _runtimes={}, get_runtime=get_runtime)
    monkeypatch.setattr(spatial_tools, 'require_context', lambda: context)
    result = await locate_tool(LocateInput(query='chair', image=image), ToolContext(run_id='r'))
    assert result.meta['error'] == 'image_unresolved'
    assert posted == []


@pytest.mark.asyncio
async def test_without_a_runtime_the_sandbox_is_not_consulted(monkeypatch: pytest.MonkeyPatch, image, posted) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(spatial_tools, 'require_context', lambda: SimpleNamespace(runtime_provider=None))
    result = await locate_tool(LocateInput(query='chair', image=image), ToolContext(run_id='r'))
    assert result.meta['success'] is True


async def test_task_appends_spatial_tools_to_the_agent_call() -> None:
    captured: list[dict[str, Any]] = []

    async def run_agent(**kwargs):
        captured.append(kwargs)

    await visual_spatial_task(VisualSpatialTaskInput(user_prompt='Where is the chair?', run_agent=run_agent))
    assert captured[0]['user_prompt'] == 'Where is the chair?'
    assert captured[0]['tools'] == list(SPATIAL_TOOLS)


async def test_task_tools_override_any_tools_passed_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.visual_spatial import task as task_module

    captured: list[dict[str, Any]] = []

    async def run_agent(**kwargs):
        captured.append(kwargs)

    async def visual_task(task_input):
        return await task_input.run_agent(user_prompt='q', tools=['jupyter_with_image'])

    monkeypatch.setattr(task_module, 'visual_task', visual_task)
    await visual_spatial_task(VisualSpatialTaskInput(user_prompt='q', run_agent=run_agent))
    assert captured[0]['tools'] == list(SPATIAL_TOOLS)


async def test_task_tool_selection_is_configurable_and_validated() -> None:
    captured: list[dict[str, Any]] = []

    async def run_agent(**kwargs):
        captured.append(kwargs)

    task_input = VisualSpatialTaskInput(
        user_prompt='q',
        base_tools=['jupyter_with_image'],
        spatial_tools=['visual/query_depth', 'visual/query_depth'],
        run_agent=run_agent,
    )
    await visual_spatial_task(task_input)
    assert captured[0]['tools'] == ['jupyter_with_image', 'visual/query_depth']

    with pytest.raises(ValueError, match='Unknown spatial tools'):
        await visual_spatial_task(
            VisualSpatialTaskInput(user_prompt='q', spatial_tools=['visual/x'], run_agent=run_agent)
        )


async def test_task_tools_reach_visual_agent_through_the_runner(main_delegate) -> None:
    from nexus.runner.single import RunInput, run

    result = await run(
        RunInput(
            task_name='visual_spatial',
            task_config={'user_prompt': 'Where is the chair?'},
            agent_name='visual_agent',
            agent_config={'tools': ['should be replaced by the task']},
        )
    )
    assert result.success, result.error
    assert main_delegate.await_args.args[0].tools == list(SPATIAL_TOOLS)


async def test_task_inherits_visual_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.core import ExecutionContext
    from nexus.extensions.tasks.visual import reward

    client = SimpleNamespace(get_score=AsyncMock(return_value=1.0))
    monkeypatch.setattr(reward, 'RewardSystemClient', lambda **kwargs: client)
    agent = AsyncMock(return_value=SimpleNamespace(final_answer='The speaker is closer.'))
    task_input = VisualSpatialTaskInput(
        user_prompt='Which is closer?',
        answer='speaker',
        reward_system_router_id='MM-Rule-0',
        run_agent=agent,
    )
    async with ExecutionContext():
        result = await visual_spatial_task(task_input)

    assert result.score == 1.0
    assert result.evaluation_details == {'router_id': 'MM-Rule-0'}
    assert agent.await_args.kwargs['tools'] == list(SPATIAL_TOOLS)
    call = client.get_score.await_args.kwargs
    assert call['ref_answer'] == 'speaker'
    assert call['messages'][-2:] == [
        {'role': 'user', 'content': 'Which is closer?'},
        {'role': 'assistant', 'content': 'The speaker is closer.'},
    ]
    for message in call['messages']:
        assert 'tool_calls' not in message


async def test_task_without_router_skips_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.visual import reward

    monkeypatch.setattr(reward, 'RewardSystemClient', AsyncMock(side_effect=AssertionError('no scoring')))
    agent = AsyncMock(return_value=SimpleNamespace(final_answer='x'))
    result = await visual_spatial_task(VisualSpatialTaskInput(user_prompt='q', answer='y', run_agent=agent))
    assert result.model_dump() == {'score': None, 'evaluation_details': None}
