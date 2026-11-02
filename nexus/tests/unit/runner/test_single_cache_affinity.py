from __future__ import annotations

import hashlib

import pytest

from nexus.runner.single import RunInput, _resolve_dynamic_client_args


def _run_input(
    run_id: str | None = 't3-harbor__task__abc',
    batch_id: str | None = 'rollout-20260914-0038',
) -> RunInput:
    return RunInput(
        run_id=run_id,
        task_name='identity',
        task_config={},
        runtime_provider_config={'batch_id': batch_id} if batch_id is not None else {},
    )


def _expected_cache_task_id(run_id: str, model: str, batch_id: str | None = None) -> str:
    identity = f'{batch_id}\0{run_id}\0{model}' if batch_id is not None else f'{run_id}\0{model}'
    return hashlib.md5(identity.encode(), usedforsecurity=False).hexdigest()


def test_cache_task_id_is_derived_from_run_and_model() -> None:
    run_input = _run_input()
    config = {
        'client_args': {
            'api_key': 'secret',
            'cache_task_id_from_run_id': True,
        },
        'request_args': {
            'model': 'bedrock/anthropic.claude-opus-4-8',
        },
    }

    client_args = _resolve_dynamic_client_args(
        run_input,
        client_name='litellm',
        config_dict=config,
        run_id_provided=True,
    )

    assert run_input.run_id is not None
    assert client_args['cache_task_id'] == _expected_cache_task_id(
        run_input.run_id,
        'bedrock/anthropic.claude-opus-4-8',
        'rollout-20260914-0038',
    )
    assert 'cache_task_id_from_run_id' not in client_args


def test_cache_task_id_is_derived_for_openai_responses() -> None:
    config = {
        'client_args': {'cache_task_id_from_run_id': True},
        'request_args': {'model': 'gpt-5.6-sol'},
    }

    cache_ids = [
        _resolve_dynamic_client_args(
            _run_input(batch_id=batch_id),
            client_name='openai_responses',
            config_dict=config,
            run_id_provided=True,
        )['cache_task_id']
        for batch_id in ('rollout-a', 'rollout-b')
    ]

    assert cache_ids[0] == _expected_cache_task_id(
        't3-harbor__task__abc',
        'gpt-5.6-sol',
        'rollout-a',
    )
    assert cache_ids[0] != cache_ids[1]


def test_nexus_trace_id_is_derived_from_run_and_model() -> None:
    run_input = _run_input()
    config = {
        'client_args': {
            'nexus_trace_id_from_run_id': True,
        },
        'request_args': {
            'model': 'openai/hy4',
        },
    }

    client_args = _resolve_dynamic_client_args(
        run_input,
        client_name='litellm',
        config_dict=config,
        run_id_provided=True,
    )

    assert run_input.run_id is not None
    assert client_args['nexus_trace_id'] == _expected_cache_task_id(run_input.run_id, 'openai/hy4')
    assert client_args['nexus_llm_lb_strategy'] == 'ConsistentHash'
    assert 'nexus_trace_id_from_run_id' not in client_args


def test_explicit_nexus_trace_routing_values_win_over_derived_defaults() -> None:
    client_args = _resolve_dynamic_client_args(
        _run_input(),
        client_name='litellm',
        config_dict={
            'client_args': {
                'nexus_trace_id_from_run_id': True,
                'nexus_trace_id': 'explicit-session',
                'nexus_llm_lb_strategy': 'RoundRobin',
            },
            'request_args': {'model': 'openai/hy4'},
        },
        run_id_provided=True,
    )

    assert client_args['nexus_trace_id'] == 'explicit-session'
    assert client_args['nexus_llm_lb_strategy'] == 'RoundRobin'


def test_cache_task_id_derivation_requires_a_supported_client_and_run_id() -> None:
    config = {
        'client_args': {'cache_task_id_from_run_id': True},
        'request_args': {'model': 'model'},
    }

    with pytest.raises(ValueError, match='gateway cache affinity is supported only by'):
        _resolve_dynamic_client_args(
            _run_input(),
            client_name='openai',
            config_dict=config,
            run_id_provided=True,
        )
    with pytest.raises(ValueError, match='requires a non-empty RunInput.run_id'):
        _resolve_dynamic_client_args(
            _run_input(''),
            client_name='litellm',
            config_dict=config,
            run_id_provided=True,
        )


def test_auto_generated_run_id_still_derives_but_warns(caplog: pytest.LogCaptureFixture) -> None:
    # ``run()`` fills a random run_id before clients are built, so affinity is per-run only.
    run_input = _run_input('a1b2c3d4e5f6')
    config = {
        'client_args': {'api_key': 'secret', 'cache_task_id_from_run_id': True},
        'request_args': {'model': 'model'},
    }

    with caplog.at_level('WARNING'):
        client_args = _resolve_dynamic_client_args(
            run_input,
            client_name='litellm',
            config_dict=config,
            run_id_provided=False,
        )

    assert client_args['cache_task_id'] == _expected_cache_task_id(
        'a1b2c3d4e5f6',
        'model',
        'rollout-20260914-0038',
    )
    assert 'auto-generated run_id a1b2c3d4e5f6' in caplog.text
    assert 'a later run of the same task will not' in caplog.text


def test_cache_task_id_derivation_can_be_disabled() -> None:
    client_args = _resolve_dynamic_client_args(
        _run_input(),
        client_name='litellm',
        config_dict={'client_args': {'api_key': 'secret'}},
        run_id_provided=True,
    )

    assert client_args == {'api_key': 'secret'}
