"""Exercise real runner-dev scheduling using a separately deployed controlled capacity API.

Credentials are read from private files. Only newly submitted executions are stopped in cleanup.
The signal server and policy binding are deployed by the operator; see the acceptance report.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import subprocess
import sys
import time
import uuid

from pathlib import Path

import aiohttp

from tests.integration.distributed.cloud_batch_api import CloudBatchApi


PROBE = """import asyncio
from nexus.core import TaskRunInput, TaskRunOutput, task_registry
class ProbeInput(TaskRunInput):
    index: int = 0
class ProbeOutput(TaskRunOutput):
    marker: str
@task_registry.register('policy_probe')
async def probe(value: ProbeInput) -> ProbeOutput:
    await asyncio.sleep(1)
    return ProbeOutput(marker=f'policy-e2e:{value.index}')
"""


def prepare_source(args):
    source = args.output / 'source'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        shutil.copyfile(args.project_root / name, source / name)
    (source / 'policy_probe.py').write_text(PROBE)
    (source / 'single.json').write_text(
        json.dumps({
            'imports': ['policy_probe'],
            'task_name': 'policy_probe',
            'task_config': {'index': 9},
        })
    )
    (source / 'generator.py').write_text(
        'from nexus.runner.batch import BatchItem\nfrom nexus.runner.single import RunInput\n'
        'def get_total_count(): return 3\n'
        'def generate_configs():\n'
        '    for i in range(3):\n'
        '        yield BatchItem(run_id=f"item-{i}", config=RunInput(imports=["policy_probe"], '
        'task_name="policy_probe", task_config={"index": i}))\n'
    )
    return source


def submit(args, source, *, batch=False, gated=False, queue_ttl=1800):
    command = [sys.executable, '-m', 'nexus.cli.main', 'submit', 'batch' if batch else 'single']
    command += ['--generator', 'generator.py'] if batch else [str(source / 'single.json')]
    command += [
        '--environment',
        args.environment,
        '--control-url',
        args.control_url,
        '--source',
        str(source),
        '--project-root',
        str(args.project_root),
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '2',
        '--manager-slots',
        '3',
        '--queue-ttl-seconds',
        str(queue_ttl),
        '--execution-ttl-seconds',
        '3600',
        '--idempotency-key',
        'policy-e2e-' + uuid.uuid4().hex,
        '--json',
    ]
    if args.token_file:
        command += ['--taihu-token-file', str(args.token_file)]
    if batch:
        command += [
            '--generation',
            'remote',
            '--manager-count',
            '1',
            '--max-inflight',
            '3',
            '--batch-manager-cpu-cores',
            '1',
            '--batch-manager-memory-gib',
            '2',
            '--child-queue-ttl-seconds',
            '1800',
        ]
    if gated:
        command += ['--scheduler-policy', args.binding, '--policy-scope', args.scope]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        (args.output / 'submit-error.log').write_text(completed.stderr)
        raise RuntimeError(f'submission failed ({completed.returncode}): {completed.stderr[-2000:]}')
    data = json.loads(completed.stdout)
    parent = data.get('execution_id') or data['execution']['execution_id']
    (args.output / f'{parent}-submission.json').write_text(json.dumps(data, indent=2))
    return parent


async def exercise(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = prepare_source(args)
    evidence, active = [], set()
    token = args.token_file.read_text().strip() if args.token_file else ''
    signal_key = args.signal_key_file.read_text().strip()

    def record(stage, **facts):
        event = {'stage': stage, 'time': time.time(), **facts}
        evidence.append(event)
        (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2))
        print(json.dumps(event), flush=True)

    async def wait(read, predicate, label, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = await read()
            if predicate(data):
                return data
            await asyncio.sleep(1)
        raise TimeoutError(label)

    async with (
        CloudBatchApi(control_url=args.control_url, application_url=args.application_url, token=token) as api,
        aiohttp.ClientSession() as signals,
    ):

        async def phase(**values):
            async with signals.post(
                args.signal_url + '/state', json=values, headers={'X-Policy-Test-Key': signal_key}
            ) as response:
                response.raise_for_status()
                data = await response.json()
            record('signal', **data)

        async def execution(parent):
            return await api.control_json('GET', f'/v1/executions/{parent}')

        async def policy(parent):
            return await api.control_json('GET', f'/v1/executions/{parent}/scheduler-policy')

        async def release(parent):
            allocations = await wait(
                lambda: api.control_json('GET', f'/v1/executions/{parent}/allocations'),
                lambda d: bool(d['items']) and all(a['status'] == 'released' for a in d['items']),
                'allocation release',
            )
            record(
                'allocations_released',
                parent=parent,
                allocations=[
                    {k: a.get(k) for k in ('allocation_id', 'status', 'environment_id', 'provider_resource_ref')}
                    for a in allocations['items']
                ],
            )
            active.discard(parent)

        try:
            await phase(remaining=0, factor=1.0, delay=0, invalid=False)
            batch = await asyncio.to_thread(submit, args, source, batch=True, gated=True)
            active.add(batch)
            record('batch_submitted', parent=batch)
            items = await wait(
                lambda: api.control_json('GET', f'/v1/executions/{batch}/batch/items'),
                lambda d: len(d.get('items', [])) == 3 and all(i.get('current_child_execution_id') for i in d['items']),
                'children materialized',
            )
            children = [i['current_child_execution_id'] for i in items['items']]
            await wait(lambda: policy(children[0]), lambda d: d['reason_code'] == 'quota_exhausted', 'quota gate')
            for child in children:
                state = await execution(child)
                assert state['status'] == 'queued'
                assert state['attempt_count'] == 0
                assert state['current_assignment_id'] is None
            record('quota_blocked', children=children, snapshot=await policy(children[0]))
            if args.restart_checkpoint:
                args.restart_checkpoint.write_text(json.dumps({'parent': batch, 'children': children}))
                record('restart_checkpoint')
                await wait(
                    lambda: asyncio.to_thread(
                        (args.restart_continue_file or Path(str(args.restart_checkpoint) + '.continue')).exists
                    ),
                    bool,
                    'scheduler restart',
                    timeout=300,
                )
                assert (await policy(children[0]))['allow'] is False
                record('restart_survived')
            await phase(remaining=600, factor=0.1)
            await wait(lambda: policy(children[0]), lambda d: d['reason_code'] == 'capacity_paused', 'congestion gate')
            record('congestion_blocked')
            await phase(invalid=True)
            await wait(
                lambda: policy(children[0]),
                lambda d: d['reason_code'] == 'policy_unavailable',
                'invalid signal expires',
            )
            record('invalid_signal_did_not_unblock')
            await phase(invalid=False, remaining=600, factor=1.2, delay=20)
            await wait(
                lambda: policy(children[0]),
                lambda d: any(s['last_error'] == 'policy_timeout' for s in d['scopes']),
                'slow source timeout',
            )
            free = await asyncio.to_thread(submit, args, source)
            active.add(free)
            finished = await wait(
                lambda: execution(free),
                lambda d: d['status'] in {'succeeded', 'failed', 'lost', 'expired'},
                'unbound task',
            )
            assert finished['status'] == 'succeeded'
            result = await api.resolved_result(finished)
            output = result.get('task_output') or result.get('output', {}).get('task_output')
            assert output['marker'] == 'policy-e2e:9'
            record('unbound_succeeded_during_slow_policy', parent=free, marker=output['marker'])
            for child in children:
                assert (await execution(child))['current_assignment_id'] is None
            await release(free)
            await phase(delay=0, remaining=600, factor=1.2)
            finished = await wait(
                lambda: execution(batch),
                lambda d: d['status'] in {'succeeded', 'failed', 'lost', 'expired'},
                'batch recovery',
            )
            assert finished['status'] == 'succeeded', finished['status']
            markers = set()
            for child in children:
                value = await execution(child)
                assert value['attempt_count'] == 1
                result = await api.resolved_result(value)
                output = result.get('task_output') or result.get('output', {}).get('task_output')
                assert output['marker'] in {'policy-e2e:0', 'policy-e2e:1', 'policy-e2e:2'}
                markers.add(output['marker'])
                record('verified_result', child=child, marker=output['marker'], attempts=value['attempt_count'])
            assert markers == {'policy-e2e:0', 'policy-e2e:1', 'policy-e2e:2'}
            await release(batch)
            await phase(remaining=0)
            await wait(
                lambda: policy(children[0]), lambda d: d['reason_code'] == 'quota_exhausted', 'quota exhausted again'
            )
            expiring = await asyncio.to_thread(submit, args, source, gated=True, queue_ttl=20)
            active.add(expiring)
            expired = await wait(
                lambda: execution(expiring), lambda d: d['status'] in {'expired', 'succeeded', 'failed'}, 'queue ttl'
            )
            assert expired['status'] == 'expired'
            assert expired['current_assignment_id'] is None
            record('queue_ttl_preserved', parent=expiring)
            await release(expiring)
            record('passed')
        finally:
            for parent in active:
                with contextlib.suppress(Exception):
                    await api.control_json('POST', f'/v1/executions/{parent}/stop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control-url', 'application-url', 'environment', 'binding', 'scope', 'signal-url'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--signal-key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).parents[3])
    parser.add_argument('--restart-checkpoint', type=Path)
    parser.add_argument('--restart-continue-file', type=Path, help='Optional operator-managed continuation marker')
    asyncio.run(exercise(parser.parse_args()))


if __name__ == '__main__':
    main()
