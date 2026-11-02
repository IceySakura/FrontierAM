from __future__ import annotations

import os
import sys

import pytest

from nexus.extensions.tasks.aidata.mock_servers import llm_client
from nexus.extensions.tasks.aidata.rubric.judge import _judge_once


def test_judge_once_uses_packaged_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, 'llm_client', raising=False)
    calls: list[dict] = []

    def fake_llm_chat(**kwargs):
        calls.append(kwargs)
        return 'Y'

    monkeypatch.setattr(llm_client, 'llm_chat', fake_llm_chat)

    verdict, raw = _judge_once('source text', 'source supports claim', timeout_s=0.01)

    assert (verdict, raw) == ('Y', 'Y')
    assert len(calls) == 1
    assert calls[0]['model_env'] == 'NEXUS_JUDGE_MODEL'
    assert calls[0]['base_url_env'] == 'NEXUS_JUDGE_BASE_URL'
    assert calls[0]['api_key_env'] == 'NEXUS_JUDGE_API_KEY'


def test_judge_once_maps_public_judge_env_to_internal_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    judge_env = {
        'JUDGE_OPENAI_BASE_URL': 'http://judge.example/v1',
        'JUDGE_OPENAI_API_KEY': 'judge-key',
        'JUDGE_OPENAI_MODEL': 'judge-model',
    }
    for key, value in judge_env.items():
        monkeypatch.setenv(key, value)
    for key in ('NEXUS_JUDGE_BASE_URL', 'NEXUS_JUDGE_API_KEY', 'NEXUS_JUDGE_MODEL'):
        monkeypatch.delenv(key, raising=False)

    observed: dict[str, str] = {}

    def fake_llm_chat(**kwargs):
        observed.update({
            'base_url': os.environ[kwargs['base_url_env']],
            'api_key': os.environ[kwargs['api_key_env']],
            'model': os.environ[kwargs['model_env']],
        })
        return 'N'

    monkeypatch.setattr(llm_client, 'llm_chat', fake_llm_chat)

    verdict, raw = _judge_once('source text', 'unsupported claim', timeout_s=0.01)

    assert (verdict, raw) == ('N', 'N')
    assert observed == {
        'base_url': judge_env['JUDGE_OPENAI_BASE_URL'],
        'api_key': judge_env['JUDGE_OPENAI_API_KEY'],
        'model': judge_env['JUDGE_OPENAI_MODEL'],
    }


def test_legacy_rubric_shim_exports_text_source_selector() -> None:
    from nexus.extensions.tasks.aidata.engine import rubric as legacy_rubric

    assert legacy_rubric._text_source_selector({'from': 'trajectory'}, default='all') == 'trajectory'
