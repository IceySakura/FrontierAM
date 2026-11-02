from types import SimpleNamespace

import pytest

from nexus.extensions.agents.base.evaluation import EvaluationDetails
from nexus.extensions.tasks.search import evaluator as search_evaluator_module


JUDGE_CLIENT_NAME = 'judge'
QUESTION = 'What is 1+1?'
PREDICTED = '2'
EXPECTED = '2'
CUSTOM_SYSTEM_PROMPT = 'Follow custom policy'
CUSTOM_JUDGE_PROMPT = 'Q={question}; REF={expected}; PRED={predicted}'
POSITIVE_XML = '<result><reason>same answer</reason><is_equivalent>1</is_equivalent></result>'
INVALID_XML = '<result><is_equivalent>maybe</is_equivalent></result>'
REWARD_SERVICE_URL = 'https://reward.example'
REWARD_SERVICE_TOKEN = 'token'
REWARD_SERVICE_ROUTER_ID = 'class1-class2-v1'
REWARD_SERVICE_EXP_ID = 'exp-1'
REWARD_SERVICE_TIMEOUT = '12.5'
WIDE_SEARCH_QUESTION = 'wide question'
WIDE_SEARCH_PREDICTION = 'pred'
WIDE_SEARCH_REFERENCE = 'ref'
WIDE_SEARCH_EVALUATION = {'required': ['col']}
WIDE_SEARCH_INSTANCE_ID = 'instance-1'


def test_extract_xml_tag_returns_trimmed_inner_text() -> None:
    content = '<result><reason>  because it matches  </reason></result>'

    assert search_evaluator_module._extract_xml_tag(content, 'reason') == 'because it matches'
    assert search_evaluator_module._extract_xml_tag(content, 'missing') is None


@pytest.mark.asyncio
async def test_judge_semantic_match_builds_messages_and_parses_positive_xml(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        async def create_completion(self, *, messages):
            captured['messages'] = messages
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=POSITIVE_XML
                        )
                    )
                ]
            )

    class FakeContext:
        def get_openai_client(self, name: str):
            assert name == JUDGE_CLIENT_NAME
            return FakeClient()

    monkeypatch.setattr(search_evaluator_module, 'require_context', lambda: FakeContext())

    score = await search_evaluator_module.judge_semantic_match(
        question=QUESTION,
        predicted=PREDICTED,
        expected=EXPECTED,
        judge_prompt_template=CUSTOM_JUDGE_PROMPT,
        judge_system_prompt=CUSTOM_SYSTEM_PROMPT,
    )

    assert score == 1.0
    messages = captured['messages']
    assert messages[0]['role'] == 'system'
    assert CUSTOM_SYSTEM_PROMPT in messages[0]['content']
    assert messages[1]['content'] == 'Q=What is 1+1?; REF=2; PRED=2'


@pytest.mark.asyncio
async def test_judge_semantic_match_rejects_invalid_xml_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        async def create_completion(self, *, messages):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=INVALID_XML))]
            )

    class FakeContext:
        def get_openai_client(self, name: str):
            return FakeClient()

    monkeypatch.setattr(search_evaluator_module, 'require_context', lambda: FakeContext())

    with pytest.raises(RuntimeError):
        await search_evaluator_module.judge_semantic_match(
            question='q',
            predicted='p',
            expected='e',
        )


def test_search_evaluator_resolves_reward_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('REWARD_SERVICE_URL', REWARD_SERVICE_URL)
    monkeypatch.setenv('REWARD_SERVICE_TOKEN', REWARD_SERVICE_TOKEN)
    monkeypatch.setenv('REWARD_SERVICE_ROUTER_ID', REWARD_SERVICE_ROUTER_ID)
    monkeypatch.setenv('REWARD_SERVICE_EXP_ID', REWARD_SERVICE_EXP_ID)
    monkeypatch.setenv('REWARD_SERVICE_TIMEOUT', REWARD_SERVICE_TIMEOUT)

    config = search_evaluator_module.SearchEvaluator._resolve_reward_config(search_evaluator_module.EvalMode.REWARD_SERVICE)

    assert config is not None
    assert config.url == REWARD_SERVICE_URL
    assert config.token == REWARD_SERVICE_TOKEN
    assert config.router_id == REWARD_SERVICE_ROUTER_ID
    assert config.exp_id == REWARD_SERVICE_EXP_ID
    assert config.timeout == 12.5


@pytest.mark.asyncio
async def test_search_evaluator_reward_service_requires_config() -> None:
    evaluator = search_evaluator_module.SearchEvaluator(search_evaluator_module.EvalMode.REWARD_SERVICE)

    with pytest.raises(ValueError):
        await evaluator.evaluate(prediction='p', reference='r', question='q')


@pytest.mark.asyncio
async def test_search_evaluator_evaluate_with_details_delegates_for_wide_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_wide_search_eval(question, prediction, reference, evaluation, instance_id):
        assert question == WIDE_SEARCH_QUESTION
        assert prediction == WIDE_SEARCH_PREDICTION
        assert reference == WIDE_SEARCH_REFERENCE
        assert evaluation == WIDE_SEARCH_EVALUATION
        assert instance_id == WIDE_SEARCH_INSTANCE_ID
        return 0.75, EvaluationDetails(status='success', msg='ok')

    monkeypatch.setattr(search_evaluator_module, '_wide_search_eval_with_details', fake_wide_search_eval)
    evaluator = search_evaluator_module.SearchEvaluator(
        search_evaluator_module.EvalMode.WIDESEARCH,
        widesearch_evaluation=WIDE_SEARCH_EVALUATION,
        instance_id=WIDE_SEARCH_INSTANCE_ID,
    )

    score, details = await evaluator.evaluate_with_details(
        prediction=WIDE_SEARCH_PREDICTION,
        reference=WIDE_SEARCH_REFERENCE,
        question=WIDE_SEARCH_QUESTION,
    )

    assert score == 0.75
    assert details == EvaluationDetails(status='success', msg='ok')


@pytest.mark.asyncio
async def test_judge_semantic_match_retries_on_invalid_xml_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    class FakeClient:
        async def create_completion(self, *, messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=''))]
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=POSITIVE_XML))]
            )

    class FakeContext:
        def get_openai_client(self, name: str):
            return FakeClient()

    monkeypatch.setattr(search_evaluator_module, 'require_context', lambda: FakeContext())

    score = await search_evaluator_module.judge_semantic_match(
        question='q', predicted='p', expected='e',
        max_retries=3, retry_delay=0.0,
    )

    assert score == 1.0
    assert call_count == 3


@pytest.mark.asyncio
async def test_judge_semantic_match_raises_after_all_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        async def create_completion(self, *, messages):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=''))]
            )

    class FakeContext:
        def get_openai_client(self, name: str):
            return FakeClient()

    monkeypatch.setattr(search_evaluator_module, 'require_context', lambda: FakeContext())

    with pytest.raises(RuntimeError, match='Invalid LLM judge XML response'):
        await search_evaluator_module.judge_semantic_match(
            question='q', predicted='p', expected='e',
            max_retries=2, retry_delay=0.0,
        )
