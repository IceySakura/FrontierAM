"""The reward system judges VLM answers against the image the policy saw."""

import base64

from io import BytesIO
from types import SimpleNamespace

import pytest

from PIL import Image

from nexus.extensions.agents.main.agent import MainAgentOutput
from nexus.extensions.tasks.search_stateful.evaluator import RewardSystemEvaluator
from nexus.extensions.tasks.vlm_stem_agent import task as task_module
from nexus.extensions.tasks.vlm_stem_agent.task import VLMStemAgentTaskInput, vlm_stem_agent_task


QUESTION = 'Use the supplied diagram to calculate the requested value.'
ANSWER = '73/2'
ROUTER_ID = 'STEMVerifierA20B--2'
TEXT_ONLY_ROUTER_ID = 'MultimodalReward-GeneralV2TextOnly-0'


def _png_data_uri() -> str:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


PNG_DATA_URI = _png_data_uri()


@pytest.fixture(autouse=True)
def _mock_runtime_context(monkeypatch):
    class Runtime:
        async def upload_file(self, files, *, overwrite=True):
            return SimpleNamespace(
                results=[SimpleNamespace(path=item.path, success=True, message=None) for item in files]
            )

    class Context:
        async def get_runtime(self, name):
            return Runtime()

    monkeypatch.setattr(task_module, 'require_context', lambda: Context())


def _build_evaluator(**kwargs) -> tuple[RewardSystemEvaluator, list[dict]]:
    """Return an evaluator whose reward call is captured instead of sent."""
    evaluator = RewardSystemEvaluator(user_token='test-token', router_id=ROUTER_ID, **kwargs)
    captured: list[dict] = []

    async def capture_get_score(**call_kwargs):
        captured.append(call_kwargs)
        return 1.0

    evaluator.client.get_score = capture_get_score
    return evaluator, captured


@pytest.mark.asyncio
async def test_images_reach_the_reward_system_before_the_question():
    evaluator, captured = _build_evaluator(images=[PNG_DATA_URI])

    score = await evaluator.evaluate(prediction='<think>hmm</think>73/2', reference=ANSWER, question=QUESTION)

    assert score == 1.0
    user_message = captured[0]['messages'][1]
    assert user_message['content'] == [
        {'type': 'image_url', 'image_url': {'url': PNG_DATA_URI}},
        {'type': 'text', 'text': QUESTION},
    ]
    assert captured[0]['messages'][2] == {'role': 'assistant', 'content': '73/2'}
    assert evaluator.get_last_judge_result()['image_count'] == 1


@pytest.mark.asyncio
async def test_text_only_router_keeps_the_plain_string_payload():
    evaluator, captured = _build_evaluator(
        images=[PNG_DATA_URI],
        strip_multimodal_content=[TEXT_ONLY_ROUTER_ID, ROUTER_ID],
    )

    await evaluator.evaluate(prediction='73/2', reference=ANSWER, question=QUESTION)

    assert captured[0]['messages'][1] == {'role': 'user', 'content': QUESTION}
    judge_result = evaluator.get_last_judge_result()
    assert judge_result['strip_multimodal'] is True
    assert judge_result['image_count'] == 0


@pytest.mark.asyncio
async def test_router_absent_from_the_strip_list_still_receives_images():
    evaluator, captured = _build_evaluator(
        images=[PNG_DATA_URI],
        strip_multimodal_content=[TEXT_ONLY_ROUTER_ID],
    )

    await evaluator.evaluate(prediction='73/2', reference=ANSWER, question=QUESTION)

    assert isinstance(captured[0]['messages'][1]['content'], list)


@pytest.mark.asyncio
async def test_strip_multimodal_content_true_drops_images_for_every_router():
    evaluator, captured = _build_evaluator(images=[PNG_DATA_URI], strip_multimodal_content=True)

    await evaluator.evaluate(prediction='73/2', reference=ANSWER, question=QUESTION)

    assert captured[0]['messages'][1] == {'role': 'user', 'content': QUESTION}


@pytest.mark.asyncio
async def test_evaluator_without_images_is_unchanged():
    evaluator, captured = _build_evaluator()

    await evaluator.evaluate(prediction='73/2', reference=ANSWER, question=QUESTION)

    assert captured[0]['messages'] == [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': QUESTION},
        {'role': 'assistant', 'content': '73/2'},
    ]


@pytest.mark.asyncio
async def test_task_hands_the_normalized_images_to_the_reward_system(monkeypatch):
    built: dict = {}

    class _Evaluator:
        async def evaluate(self, prediction, reference, question=None):
            return 1.0

        async def evaluate_with_details(self, prediction, reference, question=None):
            return 1.0, None

        def get_last_judge_result(self):
            return None

    def capture_build_evaluator(task_input, images):
        built['images'] = images
        built['strip'] = task_input.reward_system_strip_multimodal_content
        return _Evaluator()

    monkeypatch.setattr(task_module, '_build_evaluator', capture_build_evaluator)

    async def run_agent(**_kwargs):
        return MainAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': ANSWER}],
            exit_status='finished',
            iterations=1,
            final_answer=ANSWER,
        )

    output = await vlm_stem_agent_task(
        VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            evaluator_type='reward_system',
            reward_system_router_id=ROUTER_ID,
            reward_system_strip_multimodal_content=[TEXT_ONLY_ROUTER_ID],
            run_agent=run_agent,
        )
    )

    assert output.score == 1.0
    assert built['images'] == [PNG_DATA_URI]
    assert built['strip'] == [TEXT_ONLY_ROUTER_ID]


def test_task_input_accepts_both_strip_multimodal_shapes():
    for configured in (True, False, None, [TEXT_ONLY_ROUTER_ID]):
        task_input = VLMStemAgentTaskInput(
            question=QUESTION,
            images=[PNG_DATA_URI],
            answer=ANSWER,
            evaluator_type='reward_system',
            reward_system_router_id=ROUTER_ID,
            reward_system_strip_multimodal_content=configured,
            run_agent=lambda **_kwargs: None,
        )
        assert task_input.reward_system_strip_multimodal_content == configured
