from __future__ import annotations

import urllib.error

from email.message import Message

import pytest

from nexus.extensions.tasks.stem_mas.evaluator.lean4_verifier import Lean4VerifierError, Lean4VerifierEvaluator


@pytest.mark.asyncio
async def test_http_error_is_retryable_failure_not_zero_score(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(benchmark='minif2f', task_id='minif2f.1')

    def raise_http_error(*args, **kwargs):
        del args, kwargs
        raise urllib.error.HTTPError(evaluator.url, 503, 'unavailable', Message(), None)

    monkeypatch.setattr(evaluator, 'post_json', raise_http_error)

    with pytest.raises(Lean4VerifierError, match='HTTP 503'):
        await evaluator.evaluate('by trivial', '', 'theorem x : True := by sorry')


@pytest.mark.asyncio
async def test_request_error_is_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(benchmark='minif2f', task_id='minif2f.1')
    monkeypatch.setattr(
        evaluator,
        'post_json',
        lambda *args, **kwargs: {
            'request_ok': False,
            'correct': False,
            'score': 0.0,
            'failure_type': 'system_error',
            'error': 'runner unavailable',
        },
    )

    with pytest.raises(Lean4VerifierError, match='runner unavailable'):
        await evaluator.evaluate('by trivial', '', 'theorem x : True := by sorry')


@pytest.mark.asyncio
async def test_transport_error_is_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(benchmark='minif2f', task_id='minif2f.1')

    def raise_transport_error(*args, **kwargs):
        del args, kwargs
        raise urllib.error.URLError('connection reset')

    monkeypatch.setattr(evaluator, 'post_json', raise_transport_error)

    with pytest.raises(Lean4VerifierError, match='transport error'):
        await evaluator.evaluate('by trivial', '', 'theorem x : True := by sorry')


@pytest.mark.asyncio
async def test_manifest_version_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(
        benchmark='minif2f',
        task_id='minif2f.1',
        expected_manifest_version='expected',
    )
    monkeypatch.setattr(
        evaluator,
        'post_json',
        lambda *args, **kwargs: {
            'request_ok': True,
            'correct': True,
            'score': 1.0,
            'task': {'manifest_version': 'unexpected'},
        },
    )

    with pytest.raises(Lean4VerifierError, match='manifest mismatch'):
        await evaluator.evaluate('trivial', '', 'theorem x : True := by sorry')


@pytest.mark.asyncio
async def test_verifier_timeout_is_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(benchmark='minif2f', task_id='minif2f.1')
    monkeypatch.setattr(
        evaluator,
        'post_json',
        lambda *args, **kwargs: {
            'request_ok': True,
            'correct': False,
            'score': 0.0,
            'failure_type': 'timeout',
        },
    )

    with pytest.raises(Lean4VerifierError, match='timed out'):
        await evaluator.evaluate('trivial', '', 'theorem x : True := by sorry')


@pytest.mark.asyncio
async def test_valid_response_returns_score_and_strips_leading_by(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(
        benchmark='minif2f',
        task_id='minif2f.1',
        expected_manifest_version='minif2f-v1',
    )
    captured = {}

    def respond(path, payload):
        captured.update({'path': path, 'payload': payload})
        return {
            'request_ok': True,
            'correct': True,
            'score': 1.0,
            'task': {'manifest_version': 'minif2f-v1'},
        }

    monkeypatch.setattr(evaluator, 'post_json', respond)

    score = await evaluator.evaluate('<final_proof>by\n  trivial</final_proof>', '', 'theorem x : True := by sorry')

    assert score == 1.0
    assert captured['payload']['candidate'] == {'kind': 'proof_body', 'text': 'trivial'}


@pytest.mark.asyncio
async def test_putnam_full_code_is_bound_to_manifest_target(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(
        benchmark='putnam672',
        task_id='putnam672.putnam_1962_a1',
        candidate_policy='target_declaration_proof',
    )
    captured = {}

    def respond(path, payload):
        captured.update({'path': path, 'payload': payload})
        return {'request_ok': True, 'correct': False, 'score': 0.0}

    monkeypatch.setattr(evaluator, 'post_json', respond)
    candidate = 'import Mathlib\n\ntheorem putnam_1962_a1 : True := by\n  trivial'

    score = await evaluator.evaluate(f'```lean4\n{candidate}\n```', '', 'theorem putnam_1962_a1 : True := sorry')

    assert score == 0.0
    assert captured['payload']['candidate'] == {'kind': 'proof_body', 'text': 'by\n  trivial'}


@pytest.mark.asyncio
async def test_unrelated_full_code_is_rejected_without_calling_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(
        benchmark='putnam672',
        task_id='putnam672.putnam_1962_a1',
        candidate_policy='target_declaration_proof',
    )
    called = False

    def respond(path, payload):
        nonlocal called
        called = True
        return {'request_ok': True, 'correct': True, 'score': 1.0}

    monkeypatch.setattr(evaluator, 'post_json', respond)
    score = await evaluator.evaluate(
        '```lean4\nimport Mathlib\n\ntheorem unrelated : True := by trivial\n```',
        '',
        'theorem putnam_1962_a1 : False := sorry',
    )

    assert score == 0.0
    assert called is False
    assert evaluator.last_judge_result is not None
    assert evaluator.last_judge_result['response']['failure_type'] == 'invalid_candidate'


@pytest.mark.asyncio
async def test_fate_full_code_extracts_the_named_target_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Lean4VerifierEvaluator(
        benchmark='fatem',
        task_id='fatem.001',
        candidate_policy='target_declaration_proof',
    )
    captured = {}

    def respond(path, payload):
        captured.update({'path': path, 'payload': payload})
        return {'request_ok': True, 'correct': True, 'score': 1.0}

    monkeypatch.setattr(evaluator, 'post_json', respond)
    candidate = """import Mathlib

theorem fate_target : True := by
  trivial

theorem later_helper : True := by
  trivial"""

    score = await evaluator.evaluate(
        f'```lean4\n{candidate}\n```',
        '',
        'theorem fate_target : True := by sorry',
    )

    assert score == 1.0
    assert captured['payload']['candidate'] == {'kind': 'proof_body', 'text': 'trivial'}
