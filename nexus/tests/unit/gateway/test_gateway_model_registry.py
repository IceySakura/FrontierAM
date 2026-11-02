"""Verify model registration and explicit model-role configuration."""

from __future__ import annotations

from typing import Any

import pytest

from _fakes import FakeTokenizer, RecordingEndpoint

from nexus.core.openai_client import OpenAIConfig
from nexus.gateway.client import GatewayOpenAIClient


def build_client(**overrides: Any) -> GatewayOpenAIClient:
    tokenizer = FakeTokenizer()
    return GatewayOpenAIClient(
        OpenAIConfig(
            request_args={},
            client_args={
                'trajectory_id': 'model-routing-test',
                'models': [
                    {
                        'id': model_id,
                        'server_manager': RecordingEndpoint(tokenizer, []),
                        'tokenizer': tokenizer,
                        'max_model_len': 40 if model_id == 'main' else 80,
                        'apply_chat_template_kwargs': {'enable_thinking': model_id == 'main'},
                        'mm_plugin': object(),
                        'priority': 3 if model_id == 'main' else 7,
                        'enable_rollout_routing_replay': model_id == 'main',
                    }
                    for model_id in ['main', 'subagent']
                ],
                'default_model_id': 'main',
                **overrides,
            },
        ),
    )


def test_registry_keeps_default_and_model_dependencies() -> None:
    client = build_client()
    registry = client._model_registry
    assert registry.default_model_id == 'main', 'Keep the configured default model ID'
    assert registry.get(registry.default_model_id).max_model_len == 40, 'Keep the default model capacity'
    assert registry.get('subagent').max_model_len == 80, 'Keep each model capacity independently'
    for config in client.config.client_args['models']:
        processor = registry.get(config['id']).processor
        assert (
            processor._apply_chat_template_kwargs['enable_thinking']
            == config['apply_chat_template_kwargs']['enable_thinking']
        ), 'Keep the model-specific template parameters'
        assert processor._mm_plugin is config['mm_plugin'], 'Keep the model-specific multimodal plugin'
        assert processor._priority == config['priority'], 'Keep the model-specific default priority'
        assert processor._enable_rollout_routing_replay == config['enable_rollout_routing_replay'], (
            'Keep the model-specific R3 setting'
        )


@pytest.mark.parametrize(
    'overrides',
    [
        {field: None}
        for field in (
            'server_manager',
            'openai_client',
            'tokenizer',
            'tool_parser',
            'template_parser',
            'apply_chat_template_kwargs',
            'mm_plugin',
            'priority',
            'enable_rollout_routing_replay',
            'max_model_len',
        )
    ]
    + [
        {'apply_chat_template_kwargs': {}},
        {'mm_plugin': object()},
        {'priority': 0},
        {'enable_rollout_routing_replay': False},
        {'max_model_len': 0},
        {'max_model_len': 4096, 'enable_rollout_routing_replay': True},
    ],
)
def test_model_fields_at_top_level_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='inside models') as error:
        build_client(**overrides)
    assert all(field in str(error.value) for field in overrides), 'Report every conflicting top-level field'


@pytest.mark.parametrize('default_model_id', ['main', 'subagent'])
@pytest.mark.parametrize('overrides', [{}, {'compaction_model_id': None, 'subagent_model_id': None}])
def test_optional_model_ids_are_completed_during_init(default_model_id: str, overrides: dict[str, Any]) -> None:
    client = build_client(default_model_id=default_model_id, **overrides)
    assert client._compaction_model_id == default_model_id, 'Compaction defaults to the configured default model'
    assert client._subagent_model_id == default_model_id, 'Subagents default to the configured default model'


def test_explicit_model_ids_are_preserved() -> None:
    client = build_client(compaction_model_id='subagent', subagent_model_id='main')
    assert client._compaction_model_id == 'subagent', 'Keep the explicit compaction model'
    assert client._subagent_model_id == 'main', 'Keep the explicit subagent model'


@pytest.mark.parametrize(
    'overrides',
    [
        {'default_model_id': 'missing'},
        {'compaction_model_id': 'missing'},
        {'subagent_model_id': 'missing'},
    ],
)
def test_invalid_routing_fails_during_setup(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='default_model_id|Unknown model ID'):
        build_client(**overrides)
