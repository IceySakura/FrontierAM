"""Validate runner scheduling against a configured real capacity source, without modifying its signals."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from pathlib import Path

from tests.e2e.distributed.run_tidal_scheduler import prepare_source, submit
from tests.integration.distributed.cloud_batch_api import CloudBatchApi


async def exercise(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = prepare_source(args)
    evidence = []
    token = args.token_file.read_text().strip() if args.token_file else ''

    def record(stage, **facts):
        event = {'stage': stage, 'time': time.time(), **facts}
        evidence.append(event)
        (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2))
        print(json.dumps(event), flush=True)

    async def wait(read, predicate, label, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await read()
            if predicate(value):
                return value
            await asyncio.sleep(1)
        raise TimeoutError(label)

    async with CloudBatchApi(control_url=args.control_url, application_url=args.application_url, token=token) as api:
        parent = await asyncio.to_thread(submit, args, source, batch=True, gated=True)
        record('batch_submitted', parent=parent)

        async def execution(execution_id):
            return await api.control_json('GET', f'/v1/executions/{execution_id}')

        async def policy(execution_id):
            return await api.control_json('GET', f'/v1/executions/{execution_id}/scheduler-policy')

        try:
            items = await wait(
                lambda: api.control_json('GET', f'/v1/executions/{parent}/batch/items'),
                lambda d: len(d.get('items', [])) == 3 and all(i.get('current_child_execution_id') for i in d['items']),
                'children materialized',
            )
            children = [item['current_child_execution_id'] for item in items['items']]
            first = await wait(
                lambda: policy(children[0]),
                lambda d: d['reason_code'] in {'policy_available', 'quota_exhausted', 'capacity_paused'},
                'valid live capacity decision',
            )
            assert all(scope['snapshot']['config_revision'] == args.revision for scope in first['scopes'])
            record('live_policy_snapshot', children=children, snapshot=first)
            first_version = first['scopes'][0]['snapshot']['snapshot_version']
            refreshed = await wait(
                lambda: policy(children[0]),
                lambda d: d['scopes'][0]['snapshot']['snapshot_version'] > first_version,
                'live capacity refreshed',
                timeout=120,
            )
            record('live_policy_refreshed', snapshot=refreshed)
            if not first['allow'] and not refreshed['allow']:
                for child in children:
                    state = await execution(child)
                    assert state['status'] == 'queued'
                    assert state['attempt_count'] == 0
                    assert state['current_assignment_id'] is None
                record('live_capacity_blocked', reason=refreshed['reason_code'])
            else:
                finished = await wait(
                    lambda: execution(parent),
                    lambda d: d['status'] in {'succeeded', 'failed', 'lost', 'expired'},
                    'live capacity batch completion',
                )
                assert finished['status'] == 'succeeded', finished['status']
                markers = set()
                for child in children:
                    value = await execution(child)
                    assert value['attempt_count'] == 1
                    result = await api.resolved_result(value)
                    output = result.get('task_output') or result.get('output', {}).get('task_output')
                    markers.add(output['marker'])
                    record('verified_result', child=child, marker=output['marker'], attempts=value['attempt_count'])
                assert markers == {'policy-e2e:0', 'policy-e2e:1', 'policy-e2e:2'}
                record('live_capacity_allowed')
        finally:
            state = await execution(parent)
            if state['status'] not in {'succeeded', 'failed', 'expired', 'cancelled', 'lost'}:
                await api.control_json('POST', f'/v1/executions/{parent}/stop')
            allocations = await wait(
                lambda: api.control_json('GET', f'/v1/executions/{parent}/allocations'),
                lambda d: bool(d['items']) and all(a['status'] == 'released' for a in d['items']),
                'allocation release',
            )
            record(
                'allocations_released',
                parent=parent,
                allocations=[
                    {k: a.get(k) for k in ('allocation_id', 'status', 'provider_resource_ref')}
                    for a in allocations['items']
                ],
            )
        record('passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control-url', 'application-url', 'environment', 'binding', 'scope'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    asyncio.run(exercise(parser.parse_args()))


if __name__ == '__main__':
    main()
