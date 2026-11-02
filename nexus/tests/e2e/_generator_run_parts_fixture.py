"""Deterministic task with real process and filesystem effects for batch CLI E2E."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from pathlib import Path

from nexus.core import TaskRunInput, TaskRunOutput, task_registry


class GeneratorPartsInput(TaskRunInput):
    case: str
    state_dir: str
    values: list[int]
    factor: int
    fail_first: bool = False


class GeneratorPartsOutput(TaskRunOutput):
    case: str
    answer: int
    attempt: int
    worker_pid: int
    child_pid: int
    score: float


@task_registry.register('generator_parts_e2e')
async def generator_parts_task(task_input: GeneratorPartsInput) -> GeneratorPartsOutput:
    root = Path(task_input.state_dir)
    root.mkdir(parents=True, exist_ok=True)
    attempts = root / f'{task_input.case}.attempts'
    attempt = int(attempts.read_text()) + 1 if attempts.exists() else 1
    attempts.write_text(str(attempt))
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        'import json,os,sys; p=json.load(sys.stdin); '
        'print(json.dumps({"answer":sum(p["values"])*p["factor"],"pid":os.getpid()}))',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            child.communicate(json.dumps(task_input.model_dump()).encode()), timeout=10
        )
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    if child.returncode != 0:
        raise RuntimeError(f'child failed: {stderr.decode()}')
    result = json.loads(stdout)
    if task_input.fail_first and attempt == 1:
        raise RuntimeError('intentional first-attempt failure')
    output = GeneratorPartsOutput(
        case=task_input.case,
        answer=result['answer'],
        attempt=attempt,
        worker_pid=os.getpid(),
        child_pid=result['pid'],
        score=float(result['answer'] == sum(task_input.values) * task_input.factor),
    )
    (root / f'{task_input.case}.json').write_text(output.model_dump_json())
    return output
