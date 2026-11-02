from __future__ import annotations

import asyncio
import os
import time

from pathlib import Path

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.trace.tracer import span_context
from nexus.trace.types import Event


class ExportProbeInput(TaskRunInput):
    label: str
    marker: str
    seconds: int = 10
    fail_first: bool = False
    updates_per_second: int = 20


class ExportProbeOutput(TaskRunOutput):
    label: str
    marker: str
    steps: int
    retried: bool
    emitted_updates: int


@task_registry.register('cloud_export_probe')
async def cloud_export_probe(task_input: ExportProbeInput) -> ExportProbeOutput:
    marker_file = Path('/tmp') / f'nexus-{task_input.marker}-{task_input.label}.retry'
    first = False
    if task_input.fail_first:
        try:
            fd = os.open(marker_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
            first = True
    with span_context(
        'cloud.export.probe',
        attributes={'marker': task_input.marker, 'label': task_input.label, 'first_failure_attempt': first},
    ) as span:
        for step in range(task_input.seconds):
            for part in range(task_input.updates_per_second):
                span.update(
                    attributes={'probe.step': step, 'probe.part': part, 'probe.text': '实时轨迹'},
                    events=[
                        Event(
                            name='export.probe.step',
                            time_unix_nano=time.time_ns(),
                            attributes={'step': step, 'part': part, 'marker': task_input.marker},
                        )
                    ],
                )
            await asyncio.sleep(1)
        if first:
            raise RuntimeError('intentional export probe failure before retry')
    if task_input.fail_first:
        marker_file.unlink(missing_ok=True)
    return ExportProbeOutput(
        label=task_input.label,
        marker=task_input.marker,
        steps=task_input.seconds,
        retried=task_input.fail_first,
        emitted_updates=task_input.seconds * task_input.updates_per_second,
    )
