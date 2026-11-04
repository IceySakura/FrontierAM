"""One development attempt or one grading job, run inside its slot's systemd scope.

The parent passes the slot lease descriptor, so the slot stays leased for exactly as long as this process (and any
container work it waits for) lives, even if the parent worker dies.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from pathlib import Path

from .runtime import load_runtime
from .tasks import TASKS, Placement, write


def environment_record(placement: Placement) -> dict:
    return {
        'placement': placement.__dict__,
        'host_affinity': sorted(os.sched_getaffinity(0)),
        'cgroup': Path('/proc/self/cgroup').read_text().strip(),
        'pid': os.getpid(),
    }


async def develop(spec: dict) -> int:
    from nexus.extensions.tasks.frontieram_offline.verification import Fixture
    from nexus.runner.single import RunInput, run

    from .fixtures import fixture_command

    runtime = load_runtime()
    task = TASKS[spec['task']]
    placement = Placement(**spec['placement'])
    output = Path(spec['output'])
    write(output / 'environment.json', environment_record(placement))
    if spec.get('protocol') == 'v6':
        from .native import develop as develop_native

        await asyncio.to_thread(develop_native, spec, runtime)
        return 0
    fixture = thread = None
    base_url = spec['base_url']
    if spec['smoke']:
        fixture = Fixture('basic', fixture_command(task.name))
        thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        thread.start()
        base_url = f'http://127.0.0.1:{fixture.server_port}/v1'
    config = task.dev_config(
        runtime,
        output,
        placement,
        model=spec['model'],
        base_url=base_url,
        key_file=None if spec['smoke'] else spec['key_file'],
        smoke=spec['smoke'],
        protocol=spec.get('protocol', 'v4'),
        development_seconds=spec.get('development_seconds'),
    )
    write(output / 'input.json', config)
    try:
        result = await run(RunInput.from_dict(config))
        data = result.model_dump(mode='json')
    finally:
        if fixture is not None:
            await asyncio.to_thread(fixture.shutdown)
            fixture.server_close()
            if thread is not None:
                thread.join(timeout=2)
    write(output / 'nexus_result.json', data)
    task_result, agent = data.get('task_output') or {}, data.get('agent_output') or {}
    summary = {
        'nexus_success': data.get('success'),
        'task_status': task_result.get('status'),
        'exit_status': task_result.get('exit_status'),
        'reason': task_result.get('reason'),
        'agent_eligible': agent.get('evaluation_eligible'),
        'agent_stop_reason': agent.get('stop_reason'),
        'http_attempts': agent.get('http_attempts'),
        'error_categories': agent.get('error_categories'),
        'development_seconds': agent.get('development_seconds'),
        'source_frozen': (output / 'task/source.tar').is_file(),
        'fixture_tool_history_ok': None if fixture is None else not fixture.failures,
    }
    write(output / 'develop_result.json', summary)
    return 0


async def grade(spec: dict) -> int:
    runtime = load_runtime()
    task = TASKS[spec['task']]
    placement = Placement(**spec['placement'])
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'environment.json', environment_record(placement))
    controller = task.assets(runtime).controller
    try:
        task.stage(runtime, Path(spec['archive']), output)
    except controller.SubmissionFailure as error:
        raw = {'exit_status': 'finished', 'score': 0.0, 'reason': f'invalid_archive:{error}'}
    else:
        extra = {'paired_archive': Path(spec['paired_reference'])} if task.paired else {}
        if spec.get('case_ids'):
            extra['case_ids'] = spec['case_ids']
        raw = await task.grade(runtime, placement, output, **extra)
    write(output / 'metrics.json', task.metrics(output, raw))
    return 0


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    mode = {'develop': develop, 'grade': grade}[spec['mode']]
    raise SystemExit(asyncio.run(mode(spec)))


if __name__ == '__main__':
    main()
