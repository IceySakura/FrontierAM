import base64
import json

from io import BytesIO
from types import SimpleNamespace

import pytest

from PIL import Image
from pydantic import ValidationError

from nexus.extensions.agents.main.agent import MainAgentOutput
from nexus.extensions.tasks.vlm_stem_agent import task as task_module
from nexus.extensions.tasks.vlm_stem_agent.task import (
    DEFAULT_MAX_IMAGE_BYTES,
    VLMStemAgentTaskInput,
    build_multimodal_user_prompt,
    upload_images_to_runtime,
    vlm_stem_agent_task,
    image_source_to_data_uri,
)


QUESTION = 'Use the supplied diagram to calculate the requested value.'
ANSWER = '73/2'
IMAGES = [
    'https://internal.example/diagram-1.png?signature=1',
    'https://internal.example/diagram-2.jpg?signature=2',
]


def _make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format='PNG')
    return buffer.getvalue()


PNG_BYTES = _make_png_bytes()
PNG_DATA_URI = 'data:image/png;base64,' + base64.b64encode(PNG_BYTES).decode('ascii')
NORMALIZED_IMAGES = [PNG_DATA_URI, PNG_DATA_URI]
SANDBOX_IMAGES = [
    '/tmp/jupyter_work/input_0.png',
    '/tmp/jupyter_work/input_1.png',
]


class _FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, _chunk_size):
        for chunk in self.chunks:
            yield chunk


class _FakeResponse:
    def __init__(
        self,
        chunks,
        *,
        status=200,
        content_length=None,
        url=IMAGES[0],
    ):
        self.status = status
        self.url = url
        self.content_length = sum(len(chunk) for chunk in chunks) if content_length is None else content_length
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def get(self, source, *, allow_redirects):
        self.requests.append((source, allow_redirects))
        return self.response


def _agent_output(
    *,
    final_answer: str | None,
    trajectory: list[dict],
    exit_status: str = 'finished',
) -> MainAgentOutput:
    return MainAgentOutput(
        main_trajectory=trajectory,
        exit_status=exit_status,
        iterations=2,
        final_answer=final_answer,
    )


@pytest.fixture(autouse=True)
def _mock_runtime_context(monkeypatch):
    class Runtime:
        async def upload_file(self, files, *, overwrite=True):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        path=item.path,
                        success=True,
                        message=None,
                    )
                    for item in files
                ]
            )

    runtime = Runtime()

    class Context:
        async def get_runtime(self, name):
            assert name == 'main'
            return runtime

    monkeypatch.setattr(
        task_module,
        'require_context',
        lambda: Context(),
    )
    return runtime


def test_build_multimodal_user_prompt_preserves_image_order():
    prompt = build_multimodal_user_prompt(
        QUESTION,
        NORMALIZED_IMAGES,
        SANDBOX_IMAGES,
    )

    assert prompt == [
        {
            'type': 'text',
            'text': (
                'Image file paths available inside the '
                'code_interpreter sandbox:\n'
                '- Image 1: /tmp/jupyter_work/input_0.png\n'
                '- Image 2: /tmp/jupyter_work/input_1.png'
            ),
        },
        {
            'type': 'image_url',
            'image_url': {'url': NORMALIZED_IMAGES[0]},
        },
        {
            'type': 'image_url',
            'image_url': {'url': NORMALIZED_IMAGES[1]},
        },
        {'type': 'text', 'text': QUESTION},
    ]


def test_build_multimodal_user_prompt_omits_an_absent_question():
    # Part of the VLM corpus asks the question inside the picture and ships an
    # empty question string; a blank text part is rejected by some endpoints.
    prompt = build_multimodal_user_prompt('', NORMALIZED_IMAGES[:1], SANDBOX_IMAGES[:1])

    assert [part['type'] for part in prompt] == ['text', 'image_url']
    assert prompt[0]['text'].startswith('Image file paths available')


def test_task_input_accepts_an_empty_question():
    task_input = VLMStemAgentTaskInput(
        question='',
        images=[IMAGES[0]],
        answer=ANSWER,
        run_agent=lambda **_kwargs: None,
    )

    assert task_input.question == ''


def test_a_placeholder_only_question_becomes_bare_images():
    # The R2 K12 rows ask the question inside the pictures: the question field is
    # nothing but one marker per image.
    prompt = build_multimodal_user_prompt('<image><image>', NORMALIZED_IMAGES, SANDBOX_IMAGES)

    assert prompt == [
        {
            'type': 'text',
            'text': (
                'Image file paths available inside the '
                'code_interpreter sandbox:\n'
                '- Image 1: /tmp/jupyter_work/input_0.png\n'
                '- Image 2: /tmp/jupyter_work/input_1.png'
            ),
        },
        {'type': 'image_url', 'image_url': {'url': NORMALIZED_IMAGES[0]}},
        {'type': 'image_url', 'image_url': {'url': NORMALIZED_IMAGES[1]}},
    ]


def test_each_placeholder_is_replaced_where_it_stands():
    # Dropping the markers instead would splice the clauses into
    # 'Compare and which is larger?' and change the question.
    prompt = build_multimodal_user_prompt(
        'Compare <image> and <image> - which is larger?',
        NORMALIZED_IMAGES,
        SANDBOX_IMAGES,
    )

    assert prompt[1:] == [
        {'type': 'text', 'text': 'Compare'},
        {'type': 'image_url', 'image_url': {'url': NORMALIZED_IMAGES[0]}},
        {'type': 'text', 'text': 'and'},
        {'type': 'image_url', 'image_url': {'url': NORMALIZED_IMAGES[1]}},
        {'type': 'text', 'text': '- which is larger?'},
    ]


def test_a_question_without_the_marker_keeps_the_images_up_front():
    # MathVision's '<image1>' is prose, not this marker, and its layout must not move.
    prompt = build_multimodal_user_prompt('See <image1> for the figure.', NORMALIZED_IMAGES, SANDBOX_IMAGES)

    assert [part['type'] for part in prompt] == ['text', 'image_url', 'image_url', 'text']
    assert prompt[-1] == {'type': 'text', 'text': 'See <image1> for the figure.'}


@pytest.mark.parametrize('question', ['<image>', '<image><image><image>'])
def test_a_marker_count_mismatch_still_sends_every_image(question):
    prompt = build_multimodal_user_prompt(question, NORMALIZED_IMAGES, SANDBOX_IMAGES)

    image_urls = [part['image_url']['url'] for part in prompt if part['type'] == 'image_url']
    assert image_urls == NORMALIZED_IMAGES


def test_build_multimodal_user_prompt_rejects_raw_url():
    with pytest.raises(ValueError, match='Base64 data URIs'):
        build_multimodal_user_prompt(
            QUESTION,
            [IMAGES[0]],
            ['/tmp/jupyter_work/input_0.png'],
        )


@pytest.mark.asyncio
async def test_http_image_is_downloaded_and_encoded_as_data_uri():
    session = _FakeSession(_FakeResponse([PNG_BYTES[:10], PNG_BYTES[10:]]))

    result = await image_source_to_data_uri(
        IMAGES[0],
        session,
        DEFAULT_MAX_IMAGE_BYTES,
    )

    assert result == PNG_DATA_URI
    assert IMAGES[0] not in result
    assert session.requests == [(IMAGES[0], True)]


@pytest.mark.asyncio
async def test_existing_data_uri_is_validated_and_mime_is_corrected():
    mislabeled = 'data:image/jpeg;base64,' + base64.b64encode(PNG_BYTES).decode('ascii')

    result = await image_source_to_data_uri(
        mislabeled,
        object(),
        DEFAULT_MAX_IMAGE_BYTES,
    )

    assert result == PNG_DATA_URI


def _png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (width, height), color='white').save(buffer, format='PNG')
    return buffer.getvalue()


def _jpeg_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (width, height), color='white').save(buffer, format='JPEG')
    return buffer.getvalue()


def _size_of(data_uri: str) -> tuple[int, int]:
    payload = base64.b64decode(data_uri.split(',', 1)[1])
    with Image.open(BytesIO(payload)) as image:
        return image.size


@pytest.mark.parametrize(
    ('width', 'height', 'expect_change'),
    [
        (800, 600, True),  # 低于下限：放大
        (1600, 1500, False),  # 已在区间内：不动
        (4000, 3000, True),  # 超过上限：缩小
    ],
)
@pytest.mark.asyncio
async def test_pixel_range_scales_input_image_into_bounds(width, height, expect_change):
    """与平台 runtime/common/image_resize 对齐：等比缩放到目标像素数。

    平台按内建 round 逐边取整，因此结果可能落在边界外不到 0.1% 的位置（每边差
    一个像素）；这里按目标像素数比对而不是硬卡区间，以保持两条链路的预处理一致。
    """
    min_pixels, max_pixels = 2257920, 8000000
    source = 'data:image/png;base64,' + base64.b64encode(_png_bytes(width, height)).decode('ascii')

    result = await image_source_to_data_uri(
        source,
        object(),
        DEFAULT_MAX_IMAGE_BYTES,
        min_pixels,
        max_pixels,
    )

    new_width, new_height = _size_of(result)
    assert ((new_width, new_height) != (width, height)) is expect_change

    pixels = new_width * new_height
    if expect_change:
        target = max_pixels if width * height > max_pixels else min_pixels
        assert abs(pixels - target) / target < 0.001
    else:
        assert min_pixels <= pixels <= max_pixels
    # 等比缩放：宽高比偏差不超过 1%
    assert abs(width / height - new_width / new_height) / (width / height) < 0.01


@pytest.mark.asyncio
async def test_max_pixels_alone_never_upscales():
    """只给上限时只封顶，不放大——与平台 max_pixels 策略一致。"""
    source = 'data:image/png;base64,' + base64.b64encode(_png_bytes(800, 600)).decode('ascii')

    result = await image_source_to_data_uri(
        source,
        object(),
        DEFAULT_MAX_IMAGE_BYTES,
        None,
        8000000,
    )

    assert _size_of(result) == (800, 600)


@pytest.mark.asyncio
async def test_input_image_is_untouched_without_pixel_bounds():
    source = 'data:image/png;base64,' + base64.b64encode(_png_bytes(800, 600)).decode('ascii')

    assert await image_source_to_data_uri(source, object(), DEFAULT_MAX_IMAGE_BYTES) == source


def test_pixel_bounds_must_not_be_inverted():
    async def fake_run_agent(**_kwargs):
        raise AssertionError('run_agent must not be called')

    with pytest.raises(ValidationError, match='min_pixels must not exceed max_pixels'):
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            min_pixels=8000000,
            max_pixels=2257920,
            run_agent=fake_run_agent,
        )


@pytest.mark.asyncio
async def test_http_image_size_limit_is_enforced_before_reading():
    session = _FakeSession(
        _FakeResponse(
            [PNG_BYTES],
            content_length=len(PNG_BYTES),
        )
    )

    with pytest.raises(ValueError, match='byte limit'):
        await image_source_to_data_uri(
            IMAGES[0],
            session,
            len(PNG_BYTES) - 1,
        )


@pytest.mark.asyncio
async def test_http_payload_must_be_a_valid_image():
    session = _FakeSession(_FakeResponse([b'not-an-image']))

    with pytest.raises(ValueError, match='not a valid image'):
        await image_source_to_data_uri(
            IMAGES[0],
            session,
            DEFAULT_MAX_IMAGE_BYTES,
        )


@pytest.mark.asyncio
async def test_remote_image_must_use_https():
    with pytest.raises(ValueError, match='Base64 data URI, HTTPS URL, or local file path'):
        await image_source_to_data_uri(
            'http://internal.example/image.png',
            object(),
            DEFAULT_MAX_IMAGE_BYTES,
        )


@pytest.mark.asyncio
async def test_local_image_path_is_encoded_as_data_uri(tmp_path):
    source = tmp_path / 'diagram.jpg'
    source.write_bytes(_jpeg_bytes(170, 106))

    result = await image_source_to_data_uri(
        str(source),
        object(),
        DEFAULT_MAX_IMAGE_BYTES,
    )

    assert result.startswith('data:image/jpeg;base64,')
    payload = base64.b64decode(result.split(',', 1)[1], validate=True)
    assert payload == source.read_bytes()


@pytest.mark.asyncio
async def test_local_image_path_honors_pixel_bounds(tmp_path):
    source = tmp_path / 'diagram.jpg'
    source.write_bytes(_jpeg_bytes(170, 106))

    result = await image_source_to_data_uri(
        str(source),
        object(),
        DEFAULT_MAX_IMAGE_BYTES,
        200704,
        None,
    )

    assert _size_of(result) == (567, 354)


@pytest.mark.asyncio
async def test_missing_local_image_path_fails(tmp_path):
    missing = tmp_path / 'absent.jpg'

    with pytest.raises(ValueError, match='does not exist'):
        await image_source_to_data_uri(
            str(missing),
            object(),
            DEFAULT_MAX_IMAGE_BYTES,
        )


@pytest.mark.asyncio
async def test_local_image_path_must_not_be_a_directory(tmp_path):
    with pytest.raises(ValueError, match='is not a file'):
        await image_source_to_data_uri(
            str(tmp_path),
            object(),
            DEFAULT_MAX_IMAGE_BYTES,
        )


@pytest.mark.asyncio
async def test_local_image_size_limit_is_enforced(tmp_path):
    source = tmp_path / 'diagram.jpg'
    source.write_bytes(_jpeg_bytes(170, 106))

    with pytest.raises(ValueError, match='byte limit'):
        await image_source_to_data_uri(
            str(source),
            object(),
            source.stat().st_size - 1,
        )


def test_task_rejects_a_row_with_neither_text_nor_picture():
    async def fake_run_agent(**_kwargs):
        raise AssertionError('run_agent must not be called')

    with pytest.raises(ValidationError, match='must not both be empty'):
        VLMStemAgentTaskInput(
            question='',
            images=[],
            answer=ANSWER,
            run_agent=fake_run_agent,
        )


def test_task_input_accepts_a_text_only_row():
    task_input = VLMStemAgentTaskInput(
        question=QUESTION,
        images=[],
        answer=ANSWER,
        run_agent=lambda **_kwargs: None,
    )

    assert task_input.images == []


def test_a_text_only_prompt_drops_the_sandbox_listing():
    # No file is staged, so the heading would promise a listing that never comes.
    assert build_multimodal_user_prompt(QUESTION, [], []) == [{'type': 'text', 'text': QUESTION}]


@pytest.mark.asyncio
async def test_a_text_only_row_still_provisions_the_sandbox(monkeypatch):
    # The agent reaches for code_interpreter whether or not the row ships a
    # picture, so the runtime has to come up on the task's critical path rather
    # than mid-rollout. Only the upload is skipped.
    calls = []

    class Runtime:
        async def upload_file(self, files, *, overwrite=True):
            raise AssertionError('a text-only row stages no file')

    class Context:
        async def get_runtime(self, name):
            calls.append(name)
            return Runtime()

    monkeypatch.setattr(task_module, 'require_context', lambda: Context())

    assert await upload_images_to_runtime([]) == []
    assert calls == ['main']


@pytest.mark.asyncio
async def test_task_handles_a_placeholder_only_row_of_local_images(tmp_path):
    """One real R2 K12 row: two local PNG paths, question is just '<image><image>'."""
    image_paths = []
    for name in ('89402912c9039789b32ddf590c5648eb.png', '8106d985728b8977bcf6e37fcdaaa07c.png'):
        path = tmp_path / name
        path.write_bytes(PNG_BYTES)
        image_paths.append(str(path))

    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return _agent_output(final_answer=ANSWER, trajectory=[{'role': 'assistant', 'content': ANSWER}])

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question='<image><image>',
            images=image_paths,
            answer=ANSWER,
            run_agent=fake_run_agent,
        )
    )

    prompt = captured['user_prompt']
    # The sandbox listing, then the two images, and no leftover marker text.
    assert [part['type'] for part in prompt] == ['text', 'image_url', 'image_url']
    assert prompt[0]['text'].startswith('Image file paths available')
    assert all(part['image_url']['url'].startswith('data:image/png;base64,') for part in prompt[1:])
    assert output.question == '<image><image>'
    assert output.exit_status == 'finished'


@pytest.mark.asyncio
async def test_task_passes_multimodal_prompt_and_skips_scoring():
    captured = {}
    history = [
        {'role': 'user', 'content': 'Earlier context'},
        {'role': 'assistant', 'content': 'Acknowledged'},
    ]

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return _agent_output(
            final_answer=('Explanation: calculation\nExact Answer: 73/2\nConfidence: 85%'),
            trajectory=[
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'python',
                                'arguments': '{"code":"6*6"}',
                            }
                        },
                        {
                            'function': {
                                'name': 'search',
                                'arguments': '{"query":"example"}',
                            }
                        },
                        {
                            'function': {
                                'name': 'python',
                                'arguments': '{"code":"7*7"}',
                            }
                        },
                    ],
                },
                {'role': 'assistant', 'content': 'final response'},
            ],
        )

    task_input = VLMStemAgentTaskInput(
        question=QUESTION,
        images=NORMALIZED_IMAGES,
        answer=ANSWER,
        ground_truth=ANSWER,
        history=history,
        run_agent=fake_run_agent,
    )
    output = await vlm_stem_agent_task(task_input)

    assert captured == {
        'user_prompt': build_multimodal_user_prompt(
            QUESTION,
            NORMALIZED_IMAGES,
            SANDBOX_IMAGES,
        ),
        'history': history,
    }
    image_parts = [
        part
        for part in captured['user_prompt']
        if part['type'] == 'image_url'
    ]
    assert all(
        part['image_url']['url'].startswith(
            'data:image/png;base64,'
        )
        for part in image_parts
    )
    assert ANSWER not in str(captured)
    assert output.score == 0.0
    assert output.prediction.endswith('Confidence: 85%')
    assert output.answer == ANSWER
    assert output.question == QUESTION
    assert output.exit_status == 'finished'
    assert output.tool_call_counts == {
        'python': 2,
        'search': 1,
    }
    assert not hasattr(output, 'trajectory')


@pytest.mark.asyncio
async def test_task_falls_back_to_last_assistant_content():
    async def fake_run_agent(**_kwargs):
        return _agent_output(
            final_answer=None,
            trajectory=[
                {'role': 'assistant', 'content': 'draft'},
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': 'Exact Answer: B'},
                    ],
                },
            ],
            exit_status='max_iterations',
        )

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer='B',
            run_agent=fake_run_agent,
        )
    )

    assert output.prediction == 'Exact Answer: B'
    assert output.exit_status == 'max_iterations'
    assert output.score == 0.0


@pytest.mark.asyncio
async def test_infrastructure_failure_gets_negative_score():
    async def fake_run_agent(**_kwargs):
        return _agent_output(
            final_answer=None,
            trajectory=[],
            exit_status='inference_error',
        )

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer='B',
            run_agent=fake_run_agent,
        )
    )

    assert output.score == -100.0
    assert json.loads(output.prediction) == {
        'exit_status': 'inference_error',
        'error': {
            'stage': 'agent',
            'type': 'inference_error',
            'code': 'inference_error',
            'message': 'Agent stopped with infrastructure status inference_error.',
        },
    }
    assert output.exit_status == 'inference_error'
    assert output.tool_call_counts == {}


class _FakeEvaluator:
    """Records what the task hands to the judge."""

    def __init__(self, score: float = 1.0):
        self.score = score
        self.calls: list[dict] = []

    async def evaluate(self, prediction, reference, question=None):
        self.calls.append(
            {
                'prediction': prediction,
                'reference': reference,
                'question': question,
            }
        )
        return self.score

    async def evaluate_with_details(self, prediction, reference, question=None):
        return await self.evaluate(prediction, reference, question), None

    def get_last_judge_result(self):
        return {'score': self.score, 'calls': len(self.calls)}


async def _finished_run_agent(**_kwargs):
    return _agent_output(
        final_answer='Exact Answer: 73/2',
        trajectory=[{'role': 'assistant', 'content': 'Exact Answer: 73/2'}],
    )


@pytest.mark.asyncio
async def test_default_evaluator_type_leaves_scoring_to_the_caller(monkeypatch):
    def unexpected_evaluator(_task_input, _images):
        raise AssertionError('evaluator_type=none must not build an evaluator')

    monkeypatch.setattr(task_module, '_build_evaluator', unexpected_evaluator)

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            run_agent=_finished_run_agent,
        )
    )

    assert output.score == 0.0
    assert output.judge_details is None


@pytest.mark.asyncio
async def test_evaluator_none_score_is_configurable():
    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            evaluator_none_score=0.5,
            run_agent=_finished_run_agent,
        )
    )

    assert output.score == 0.5


@pytest.mark.asyncio
async def test_judge_scores_prediction_against_ground_truth(monkeypatch):
    evaluator = _FakeEvaluator(score=1.0)
    monkeypatch.setattr(task_module, '_build_evaluator', lambda _task_input, _images: evaluator)

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer='display answer',
            ground_truth=ANSWER,
            evaluator_type='hle',
            run_agent=_finished_run_agent,
        )
    )

    assert output.score == 1.0
    assert output.judge_details == {'score': 1.0, 'calls': 1}
    assert evaluator.calls == [
        {
            'prediction': 'Exact Answer: 73/2',
            'reference': ANSWER,
            'question': QUESTION,
        }
    ]


@pytest.mark.asyncio
async def test_judge_falls_back_to_answer_without_ground_truth(monkeypatch):
    evaluator = _FakeEvaluator()
    monkeypatch.setattr(task_module, '_build_evaluator', lambda _task_input, _images: evaluator)

    await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            evaluator_type='general',
            run_agent=_finished_run_agent,
        )
    )

    assert evaluator.calls[0]['reference'] == ANSWER


@pytest.mark.asyncio
async def test_infrastructure_failure_outranks_the_judge(monkeypatch):
    def unexpected_evaluator(_task_input, _images):
        raise AssertionError('an infra failure must not be judged')

    monkeypatch.setattr(task_module, '_build_evaluator', unexpected_evaluator)

    async def failing_run_agent(**_kwargs):
        return _agent_output(
            final_answer=None,
            trajectory=[],
            exit_status='inference_error',
        )

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            evaluator_type='hle',
            run_agent=failing_run_agent,
        )
    )

    assert output.score == -100.0
    assert output.judge_details is None
    assert output.exit_status == 'inference_error'


def test_reward_system_evaluator_requires_a_router_id():
    task_input = VLMStemAgentTaskInput(
        question=QUESTION,
        images=[PNG_DATA_URI],
        answer=ANSWER,
        evaluator_type='reward_system',
        run_agent=_finished_run_agent,
    )

    with pytest.raises(ValueError, match='reward_system_router_id is required'):
        task_module._build_evaluator(task_input, [PNG_DATA_URI])


@pytest.mark.parametrize(
    ('evaluator_type', 'expected'),
    [
        ('hle', 'HLEEvaluator'),
        ('general', 'GeneralEvaluator'),
        ('olympiad', 'SearchStatefulEvaluator'),
        ('research', 'SearchStatefulEvaluator'),
        ('oss', 'SearchStatefulEvaluator'),
    ],
)
def test_evaluator_type_selects_the_shared_evaluator(evaluator_type, expected):
    task_input = VLMStemAgentTaskInput(
        question=QUESTION,
        images=[PNG_DATA_URI],
        answer=ANSWER,
        evaluator_type=evaluator_type,
        run_agent=_finished_run_agent,
    )

    assert type(task_module._build_evaluator(task_input, [PNG_DATA_URI])).__name__ == expected
