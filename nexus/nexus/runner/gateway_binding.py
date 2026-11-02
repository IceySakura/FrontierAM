"""Bind a standalone Gateway run to one trajectory id.

Training passes the rollout key in explicitly and finalizes the session itself.
``nexus run single --use-gateway`` (and the same flag on ``run batch``) is the
standalone path: it wraps the main client, installs ``gateway_trajectory``, and
fills any missing trajectory id from ``run_id``. An id that is already set wins.
"""

from __future__ import annotations

import copy

from typing import Any

from nexus.runner.single import ProcessorConfig, RunInput


_GATEWAY_CLIENT = 'gateway'
_GATEWAY_PROCESSOR = 'gateway_trajectory'
_GATEWAY_PROCESSOR_KEY = 'gateway'


def apply_gateway_run_binding(run_input: RunInput) -> None:
    """Align gateway client and processor trajectory ids for this run.

    With ``runner.use_gateway``, ``openai_clients.main`` is wrapped as a gateway
    client when it is not one already, and a ``gateway_trajectory`` processor is
    added when the run does not already have one. Missing trajectory ids are
    then filled from the single explicit id, or from ``run_id`` when neither
    side set one. Explicit ids that disagree raise before any client is built.
    """
    if run_input.runner.use_gateway:
        _wrap_main_client(run_input)
        _ensure_gateway_processor(run_input)
    _align_trajectory_ids(run_input)


def _wrap_main_client(run_input: RunInput) -> None:
    clients = run_input.openai_clients
    main = clients.get('main')
    if not isinstance(main, dict):
        raise ValueError('--use-gateway requires openai_clients.main')
    if main.get('instance') is not None:
        raise ValueError('--use-gateway cannot wrap a prebuilt openai_clients.main instance')
    if main.get('name') == _GATEWAY_CLIENT:
        return

    original = copy.deepcopy(main)
    original_config = dict(original.get('config') or {})
    original_client_args = dict(original_config.get('client_args') or {})
    scaffold_policy = original_client_args.pop('scaffold_policy', None)
    trajectory_id = original_client_args.pop('trajectory_id', None)
    if original.get('name') in (None, ''):
        original['name'] = 'openai'
    original_config['client_args'] = original_client_args
    original['config'] = original_config

    gateway_client_args: dict[str, Any] = {'openai_client': original}
    if scaffold_policy is not None:
        gateway_client_args['scaffold_policy'] = scaffold_policy
    if isinstance(trajectory_id, str) and trajectory_id:
        gateway_client_args['trajectory_id'] = trajectory_id
    clients['main'] = {
        'name': _GATEWAY_CLIENT,
        'config': {
            'client_args': gateway_client_args,
            'request_args': copy.deepcopy(original_config.get('request_args') or {}),
            'enforced_request_args': copy.deepcopy(original_config.get('enforced_request_args') or {}),
        },
    }


def _ensure_gateway_processor(run_input: RunInput) -> None:
    existing = [key for key, processor in run_input.processors.items() if processor.name == _GATEWAY_PROCESSOR]
    if len(existing) > 1:
        raise ValueError(
            'multiple gateway_trajectory processors '
            f'({", ".join(existing)}); keep one, or omit it and pass --use-gateway'
        )
    if existing:
        return
    run_input.processors[_GATEWAY_PROCESSOR_KEY] = ProcessorConfig(name=_GATEWAY_PROCESSOR, args={})


def _align_trajectory_ids(run_input: RunInput) -> None:
    gateway_clients = [
        (key, spec)
        for key, spec in run_input.openai_clients.items()
        if isinstance(spec, dict) and spec.get('instance') is None and spec.get('name') == _GATEWAY_CLIENT
    ]
    processors = [processor for processor in run_input.processors.values() if processor.name == _GATEWAY_PROCESSOR]
    if not gateway_clients and not processors:
        return
    if len(processors) > 1:
        keys = [key for key, processor in run_input.processors.items() if processor.name == _GATEWAY_PROCESSOR]
        raise ValueError(f'multiple gateway_trajectory processors ({", ".join(keys)})')

    explicit: list[str] = []
    missing_clients: list[tuple[str, dict[str, Any]]] = []
    for key, spec in gateway_clients:
        client_args = _client_args(spec)
        trajectory_id = _explicit_id(
            client_args.get('trajectory_id'),
            where=f'openai_clients.{key}.config.client_args.trajectory_id',
        )
        if trajectory_id is None:
            missing_clients.append((key, spec))
        else:
            explicit.append(trajectory_id)

    missing_processors: list[ProcessorConfig] = []
    for processor in processors:
        trajectory_id = _explicit_id(
            (processor.args or {}).get('trajectory_id'),
            where='gateway_trajectory args.trajectory_id',
        )
        if trajectory_id is None:
            missing_processors.append(processor)
        else:
            explicit.append(trajectory_id)

    distinct = list(dict.fromkeys(explicit))
    if len(distinct) > 1:
        raise ValueError(
            'gateway trajectory_id mismatch between the client and gateway_trajectory processor: '
            + ', '.join(distinct)
            + '. Omit both so the runner fills them from run_id, or set them to the same id.'
        )

    if not missing_clients and not missing_processors:
        return

    if len(gateway_clients) > 1 and missing_clients:
        missing = ', '.join(key for key, _spec in missing_clients)
        raise ValueError(
            f'multiple gateway clients require an explicit client_args.trajectory_id on each; missing: {missing}'
        )

    if distinct:
        chosen = distinct[0]
    else:
        if not run_input.run_id:
            raise ValueError('gateway trajectory_id requires run_id')
        chosen = run_input.run_id

    for _key, spec in missing_clients:
        _client_args(spec)['trajectory_id'] = chosen
    for processor in missing_processors:
        processor.args = {**(processor.args or {}), 'trajectory_id': chosen}


def _client_args(spec: dict[str, Any]) -> dict[str, Any]:
    config = spec.setdefault('config', {})
    client_args = config.get('client_args')
    if client_args is None:
        client_args = {}
        config['client_args'] = client_args
    return client_args


def _explicit_id(value: Any, *, where: str) -> str | None:
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError(f'{where} must be a string')
    return value
