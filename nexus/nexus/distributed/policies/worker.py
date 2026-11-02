from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

from nexus.distributed.policies.models import PolicyInput, PolicyResult
from nexus.distributed.policies.registry import load_plugins, policy_registry, signal_source_registry
from nexus.distributed.policies.workers import MAX_MESSAGE_BYTES
from nexus.distributed.utils.clock import utcnow


def evaluate(request: dict) -> dict:
    load_plugins(request['imports'])
    source = signal_source_registry.get(request['source'])()
    policy = policy_registry.get(request['policy'])()
    if policy.kind != request['kind']:
        raise ValueError('policy kind mismatch')
    if config_model := getattr(source, 'config_model', None):
        config_model.model_validate(request['source_config'])
    if params_model := getattr(policy, 'params_model', None):
        params_model.model_validate(request['params'])
    if request.get('validate_only'):
        return {'validated': True}
    observation = asyncio.run(source.read(request['source_config'], max_age_seconds=request['max_age_seconds']))
    inputs = PolicyInput(
        scope_id=request['scope_id'],
        now=utcnow(),
        params=request['params'],
        observation=observation,
    )
    result = PolicyResult.model_validate(policy.evaluate(inputs, request['previous_state']))
    result.decision = policy.decision_model.model_validate(result.decision).model_dump(mode='json')
    result.valid_until = min(result.valid_until, observation.valid_until)
    return {
        'evaluation': result.model_dump(mode='json'),
        'observation': observation.model_dump(mode='json'),
    }


def main() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    incoming, outgoing = sys.stdin.buffer, sys.stdout
    with open(os.devnull, 'w') as quiet:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            load_plugins([])
        outgoing.write('{"ready":true}\n')
        outgoing.flush()
        while line := incoming.readline(MAX_MESSAGE_BYTES + 1):
            if len(line) > MAX_MESSAGE_BYTES:
                return
            try:
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = evaluate(json.loads(line))
                encoded = json.dumps({'ok': True, 'result': result}, allow_nan=False)
                if len(encoded.encode()) + 1 > MAX_MESSAGE_BYTES:
                    raise ValueError('policy output too large')
            except Exception:
                # Provider errors may contain credentials; only return a stable error code.
                encoded = '{"ok":false}'
            outgoing.write(encoded + '\n')
            outgoing.flush()


if __name__ == '__main__':
    main()
