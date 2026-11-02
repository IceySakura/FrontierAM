from __future__ import annotations

import pytest

from nexus.extensions.tasks.aidata.engine.rubric import ScorePolicy, finalize_score


def test_default_score_policy_outputs_continuous_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AIDATA_SCORE_THRESHOLD', raising=False)
    monkeypatch.delenv('AIDATA_SCORE_CONTINUOUS', raising=False)
    monkeypatch.delenv('AIDATA_SCORE_USE_THRESHOLD', raising=False)

    policy = ScorePolicy.resolve({})

    assert policy.as_dict() == {
        'threshold': 1.0,
        'continuous': True,
        'use_threshold': False,
    }
    assert finalize_score(0.42, policy) == pytest.approx(0.42)


def test_score_policy_can_restore_binary_threshold_scoring() -> None:
    policy = ScorePolicy.resolve({
        'config': {
            'score': {
                'continuous': False,
                'use_threshold': True,
                'threshold': 0.8,
            }
        }
    })

    assert finalize_score(0.79, policy) == 0.0
    assert finalize_score(0.8, policy) == 1.0


def test_score_policy_env_overrides_rubric_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AIDATA_SCORE_CONTINUOUS', '0')
    monkeypatch.setenv('AIDATA_SCORE_USE_THRESHOLD', '1')
    monkeypatch.setenv('AIDATA_SCORE_THRESHOLD', '0.5')

    policy = ScorePolicy.resolve({
        'config': {
            'score': {
                'continuous': True,
                'use_threshold': False,
                'threshold': 0.8,
            }
        }
    })

    assert policy.as_dict() == {
        'threshold': 0.5,
        'continuous': False,
        'use_threshold': True,
    }
    assert finalize_score(0.49, policy) == 0.0
    assert finalize_score(0.5, policy) == 1.0
