from types import SimpleNamespace

import pytest

import nexus.extensions.projects.r2_stem.evaluator.llm_judge as llm_judge_module

from nexus.extensions.projects.r2_stem.evaluator.common import parse_llm_judge_result
from nexus.extensions.projects.r2_stem.evaluator.llm_judge import LLMJudge
from nexus.extensions.projects.r2_stem.evaluator.registry import build_evaluator


class FakeOpenAIClient:
    def __init__(self, content: str):
        self.content = content
        self.messages = None

    async def create_completion(self, *, messages):
        self.messages = messages
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            model='fake-judge',
        )


class FakeContext:
    def __init__(self, client: FakeOpenAIClient):
        self.client = client

    def get_openai_client(self, name: str):
        assert name == 'judge'
        return self.client


def test_parse_llm_judge_result_requires_verdict_and_reasoning() -> None:
    score, parsed = parse_llm_judge_result('{"verdict": true, "reasoning": "Equivalent answer."}')

    assert score == 1.0
    assert parsed == {'verdict': True, 'reasoning': 'Equivalent answer.'}


@pytest.mark.parametrize(
    'response',
    [
        'not json',
        '{"verdict": 1, "reasoning": "Wrong type"}',
        '{"verdict": true}',
        '{"score": 1.0, "reasoning": "Old schema"}',
    ],
)
def test_parse_llm_judge_result_rejects_malformed_output(response: str) -> None:
    with pytest.raises(ValueError):
        parse_llm_judge_result(response)


def test_registry_builds_generic_llm_judge() -> None:
    assert isinstance(build_evaluator('llm_judge'), LLMJudge)


@pytest.mark.asyncio
async def test_llm_judge_uses_generic_prompt_and_json_contract(monkeypatch) -> None:
    client = FakeOpenAIClient('{"verdict": false, "reasoning": "The attempted answer is not equivalent."}')
    monkeypatch.setattr(llm_judge_module, 'require_context', lambda: FakeContext(client))

    score, details = await LLMJudge().evaluate_with_details(
        prediction='attempt',
        reference='reference',
        question='problem',
    )

    assert score == 0.0
    assert details is not None
    assert details.parsed_result == {
        'verdict': False,
        'reasoning': 'The attempted answer is not equivalent.',
    }
    assert client.messages[0] == {
        'role': 'system',
        'content': 'You are a strict scientific answer judge. Return only JSON.',
    }
    assert 'The problem: problem' in client.messages[1]['content']
    assert 'The reference answer: reference' in client.messages[1]['content']
    assert 'The attempted answer: attempt' in client.messages[1]['content']
    assert 'BioMystery' not in client.messages[1]['content']
