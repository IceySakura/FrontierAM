import logging
import uuid

from typing import Any

import pytest

from nexus.core import (
    OpenAICompatibleClient,
    OpenAIConfig,
    TaskRunInput,
    TaskRunOutput,
    openai_client_registry,
    task_registry,
)
from nexus.core.context import require_context
from nexus.runner.single import RunInput, _resolve_env_suffix_config, run


def test_resolve_env_suffix_config_rejects_conflicting_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_TEST_API_KEY', 'secret-value')

    with pytest.raises(ValueError, match='defines both'):
        _resolve_env_suffix_config(
            {'api_key': 'literal-value', 'api_key__env': 'NEXUS_TEST_API_KEY'},
            path='openai_clients.main.config.client_args',
        )


def test_resolve_env_suffix_config_rejects_missing_env() -> None:
    with pytest.raises(ValueError, match='is not set'):
        _resolve_env_suffix_config(
            {'api_key__env': 'NEXUS_TEST_MISSING_API_KEY'},
            path='openai_clients.main.config.client_args',
        )


@pytest.mark.asyncio
async def test_run_reports_env_suffix_resolution_errors_as_initialization_failures() -> None:
    result = await run(
        RunInput(
            task_name='unused_task_name',
            task_config={'message__env': 'NEXUS_TEST_MISSING_TASK_MESSAGE'},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.type == 'ValueError'
    assert "Environment variable 'NEXUS_TEST_MISSING_TASK_MESSAGE'" in result.error.message


class EnvSuffixTaskInput(TaskRunInput):
    message: str


class EnvSuffixTaskOutput(TaskRunOutput):
    ok: bool


@pytest.mark.asyncio
async def test_openai_client_config_resolves_env_suffixes_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('NEXUS_TEST_OPENAI_API_KEY', 'secret-api-key')
    monkeypatch.setenv('NEXUS_TEST_OPENAI_MODEL', 'secret-model')
    monkeypatch.setenv('NEXUS_TEST_OPENAI_NESTED_TOKEN', 'secret-nested-token')
    monkeypatch.setenv('NEXUS_TEST_TASK_MESSAGE', 'resolved-task-message')

    task_name = f'test_env_suffix_task_{uuid.uuid4().hex}'
    client_name = f'test_env_suffix_client_{uuid.uuid4().hex}'
    seen: dict[str, Any] = {}

    @openai_client_registry.register(client_name)
    class EnvSuffixClient(OpenAICompatibleClient):
        def __init__(self, config: OpenAIConfig) -> None:
            super().__init__(config)
            seen['client_args'] = config.client_args
            seen['request_args'] = config.request_args
            seen['enforced_request_args'] = config.enforced_request_args

        async def _create_completion(self, messages, model=None, **kwargs):  # pragma: no cover
            raise AssertionError('completion should not be called')

    @task_registry.register(task_name)
    async def env_suffix_task(task_input: EnvSuffixTaskInput) -> EnvSuffixTaskOutput:
        ctx = require_context()
        seen['ctx_client'] = ctx.get_openai_client('main')
        seen['task_message'] = task_input.message
        return EnvSuffixTaskOutput(ok=True)

    run_input = RunInput(
        task_name=task_name,
        task_config={'message__env': 'NEXUS_TEST_TASK_MESSAGE'},
        openai_clients={
            'main': {
                'name': client_name,
                'config': {
                    'client_args': {
                        'api_key__env': 'NEXUS_TEST_OPENAI_API_KEY',
                        'headers': {'authorization__env': 'NEXUS_TEST_OPENAI_NESTED_TOKEN'},
                    },
                    'request_args': {'model__env': 'NEXUS_TEST_OPENAI_MODEL'},
                    'enforced_request_args': {'vendor_token__env': 'NEXUS_TEST_OPENAI_NESTED_TOKEN'},
                },
            }
        },
    )

    with caplog.at_level(logging.INFO, logger='nexus.runner.single'):
        result = await run(run_input)

    assert result.success is True
    assert seen['client_args'] == {
        'api_key': 'secret-api-key',
        'headers': {'authorization': 'secret-nested-token'},
    }
    assert seen['request_args'] == {'model': 'secret-model'}
    assert seen['enforced_request_args'] == {'vendor_token': 'secret-nested-token'}
    assert seen['ctx_client'] is not None
    assert seen['task_message'] == 'resolved-task-message'
    assert run_input.task_config == {'message__env': 'NEXUS_TEST_TASK_MESSAGE'}
    assert run_input.openai_clients['main']['config']['client_args']['api_key__env'] == 'NEXUS_TEST_OPENAI_API_KEY'
    assert 'api_key' not in run_input.openai_clients['main']['config']['client_args']

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        'run_input.task_config.message__env -> run_input.task_config.message '
        'from environment variable NEXUS_TEST_TASK_MESSAGE' in message
        for message in messages
    )
    assert any(
        'run_input.openai_clients.main.config.client_args.api_key__env -> '
        'run_input.openai_clients.main.config.client_args.api_key '
        'from environment variable NEXUS_TEST_OPENAI_API_KEY' in message
        for message in messages
    )
    assert any(
        'run_input.openai_clients.main.config.request_args.model__env -> '
        'run_input.openai_clients.main.config.request_args.model from environment variable NEXUS_TEST_OPENAI_MODEL'
        in message
        for message in messages
    )
    assert all('secret-api-key' not in message for message in messages)
    assert all('secret-model' not in message for message in messages)
    assert all('secret-nested-token' not in message for message in messages)
