import argparse

import pytest

from nexus.cli.commands.run import add_run_batch_arguments, add_run_single_arguments
from nexus.runner.gateway_binding import apply_gateway_run_binding
from nexus.runner.single import ProcessorConfig, RunInput


def _run_input(**kwargs) -> RunInput:
    return RunInput(task_name='identity', **kwargs)


def test_use_gateway_wraps_main_and_fills_both_ids_from_run_id() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={
            'main': {
                'name': 'openai',
                'config': {
                    'client_args': {'base_url': 'http://example', 'scaffold_policy': 'nexus_main_agent'},
                    'request_args': {'model': 'm'},
                    'enforced_request_args': {'stream': False},
                },
            }
        },
    )
    run_input.runner.use_gateway = True

    apply_gateway_run_binding(run_input)

    main = run_input.openai_clients['main']
    assert main['name'] == 'gateway'
    assert main['config']['client_args']['trajectory_id'] == 'run-1'
    assert main['config']['client_args']['scaffold_policy'] == 'nexus_main_agent'
    inner = main['config']['client_args']['openai_client']
    assert inner['name'] == 'openai'
    assert inner['config']['client_args'] == {'base_url': 'http://example'}
    assert 'scaffold_policy' not in inner['config']['client_args']
    assert main['config']['request_args'] == {'model': 'm'}
    assert main['config']['enforced_request_args'] == {'stream': False}
    assert run_input.processors['gateway'].args['trajectory_id'] == 'run-1'


def test_existing_gateway_client_gets_run_id_without_injecting_processor() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={'main': {'name': 'gateway', 'config': {'client_args': {'openai_client': {'name': 'openai'}}}}},
    )

    apply_gateway_run_binding(run_input)

    assert run_input.openai_clients['main']['config']['client_args']['trajectory_id'] == 'run-1'
    assert run_input.processors == {}


def test_explicit_rollout_key_is_preserved() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={'main': {'name': 'gateway', 'config': {'client_args': {'trajectory_id': 'rollout-key'}}}},
    )

    apply_gateway_run_binding(run_input)

    assert run_input.openai_clients['main']['config']['client_args']['trajectory_id'] == 'rollout-key'
    assert run_input.processors == {}


def test_processor_follows_explicit_client_id() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={'main': {'name': 'gateway', 'config': {'client_args': {'trajectory_id': 'rollout-key'}}}},
    )
    run_input.runner.use_gateway = True

    apply_gateway_run_binding(run_input)

    assert run_input.processors['gateway'].args['trajectory_id'] == 'rollout-key'


def test_mismatched_explicit_ids_raise() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={'main': {'name': 'gateway', 'config': {'client_args': {'trajectory_id': 'rollout-key'}}}},
        processors={'gateway': ProcessorConfig(name='gateway_trajectory', args={'trajectory_id': 'other'})},
    )

    with pytest.raises(ValueError, match='mismatch'):
        apply_gateway_run_binding(run_input)


def test_multiple_gateway_clients_require_explicit_ids() -> None:
    run_input = _run_input(
        run_id='run-1',
        openai_clients={
            'main': {'name': 'gateway', 'config': {'client_args': {}}},
            'judge': {'name': 'gateway', 'config': {'client_args': {}}},
        },
    )

    with pytest.raises(ValueError, match='multiple gateway clients'):
        apply_gateway_run_binding(run_input)


def test_use_gateway_cli_accepts_equals_form() -> None:
    single = argparse.ArgumentParser()
    add_run_single_arguments(single)
    args = single.parse_args(['--use-gateway=True', 'cfg.yaml'])
    assert args.use_gateway is True
    assert args.config == 'cfg.yaml'
    assert single.parse_args(['cfg.yaml', '--use-gateway']).use_gateway is True
    assert single.parse_args(['--use-gateway=false', '--use', 'profile']).use_gateway is False
    assert single.parse_args(['--use', 'profile', '--use-gateway=true']).use_parts == ['profile']

    batch = argparse.ArgumentParser()
    add_run_batch_arguments(batch)
    assert batch.parse_args(['--use-gateway=true', '--generator', 'g.py']).use_gateway is True
