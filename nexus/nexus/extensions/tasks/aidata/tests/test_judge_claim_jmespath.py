"""``judge_claim`` JMESPath preselection coverage."""

from __future__ import annotations

from typing import Any

import pytest

from nexus.extensions.tasks.aidata.aidata import _validate_check
from nexus.extensions.tasks.aidata.engine.rubric import run_rubric_async
from nexus.extensions.tasks.aidata.rubric.primitives import judge_claim


def _check(assert_block: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': 'target_document_is_complete',
        'description': 'Judge only the requested document',
        'dimension': 'correctness',
        'weight': 1,
        'assert': assert_block,
    }


def _rubric(query: Any) -> dict[str, Any]:
    return {
        'version': '4.2',
        'checks': [
            _check({
                'type': 'judge_claim',
                'from': 'tencent_docs',
                'jmespath': query,
                'claim': 'The selected document says direct down-payment use is unavailable.',
            })
        ],
    }


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (set(), True),
        ({'evidence'}, False),
    ],
)
def test_is_empty_source_handles_sets(value: set[str], expected: bool) -> None:
    assert judge_claim._is_empty_source(value) is expected


@pytest.mark.asyncio
async def test_judge_claim_applies_jmespath_before_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {
        'tencent_docs': {
            'documents': [
                {'title': 'Unrelated', 'content': 'Direct down-payment use is allowed.'},
                {'title': 'Target', 'content': 'Direct down-payment use is unavailable; amount is 0.'},
            ]
        }
    }
    captured: dict[str, str] = {}

    def fake_judge(source: str, claim: str, **_: Any) -> tuple[str, str]:
        captured['source'] = source
        captured['claim'] = claim
        return 'Y', 'Y'

    monkeypatch.setattr(judge_claim, '_llm_judge', fake_judge)

    [row] = await run_rubric_async(state, _rubric("documents[?title=='Target'] | [0]"))

    assert row['passed'] is True
    assert row['verdict'] == 'Y'
    assert 'Target' in captured['source']
    assert 'unavailable' in captured['source']
    assert 'Unrelated' not in captured['source']
    assert 'jmespath="documents[?title==\'Target\'] | [0]"' in row['detail']


@pytest.mark.asyncio
async def test_judge_claim_empty_jmespath_result_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {'tencent_docs': {'documents': [{'title': 'Other', 'content': 'irrelevant'}]}}

    def unexpected_judge(*_: Any, **__: Any) -> tuple[str, str]:
        raise AssertionError('judge must not run for an empty selection')

    monkeypatch.setattr(judge_claim, '_llm_judge', unexpected_judge)

    [row] = await run_rubric_async(state, _rubric("documents[?title=='Missing'] | [0]"))

    assert row['passed'] is False
    assert row['verdict'] == 'EMPTY_SOURCE'
    assert 'empty_jmespath_result' in row['detail']


@pytest.mark.asyncio
async def test_judge_claim_invalid_jmespath_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {'tencent_docs': {'documents': []}}

    def unexpected_judge(*_: Any, **__: Any) -> tuple[str, str]:
        raise AssertionError('judge must not run for an invalid query')

    monkeypatch.setattr(judge_claim, '_llm_judge', unexpected_judge)

    [row] = await run_rubric_async(state, _rubric('documents[?'))

    assert row['passed'] is False
    assert row['verdict'] == 'INVALID'
    assert 'bad_jmespath' in row['detail']


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['', '   ', None, 42])
async def test_judge_claim_invalid_jmespath_value_fails_closed(query: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    state = {'tencent_docs': {'documents': []}}

    def unexpected_judge(*_: Any, **__: Any) -> tuple[str, str]:
        raise AssertionError('judge must not run for an invalid query value')

    monkeypatch.setattr(judge_claim, '_llm_judge', unexpected_judge)

    [row] = await run_rubric_async(state, _rubric(query))

    assert row['passed'] is False
    assert row['verdict'] == 'INVALID'
    assert row['detail'] == 'empty_jmespath_query'


@pytest.mark.parametrize('query', ['', '   ', None, 42])
def test_validator_rejects_invalid_judge_claim_jmespath(query: Any) -> None:
    check = _check({
        'type': 'judge_claim',
        'from': 'tencent_docs',
        'jmespath': query,
        'claim': 'The selected document is complete.',
    })

    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())

    assert any('jmespath query' in problem for problem in problems), problems
