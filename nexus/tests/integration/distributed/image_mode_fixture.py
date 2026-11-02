from __future__ import annotations

import asyncio

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


IMAGE_ITEM_COUNT = 20


class ImageEchoInput(TaskRunInput):
    index: int
    sleep_seconds: float = Field(default=0, ge=0)


class ImageEchoOutput(TaskRunOutput):
    index: int
    marker: str


@task_registry.register('distributed_image_echo')
async def distributed_image_echo(task_input: ImageEchoInput) -> ImageEchoOutput:
    await asyncio.sleep(task_input.sleep_seconds)
    return ImageEchoOutput(index=task_input.index, marker=f'image-mode-ok:{task_input.index}')


def generate_configs():
    for index in range(IMAGE_ITEM_COUNT):
        yield BatchItem(
            config=RunInput(
                imports=['tests.integration.distributed.image_mode_fixture'],
                task_name='distributed_image_echo',
                task_config={'index': index, 'sleep_seconds': 1},
            ),
            run_id=f'image-smoke-{index:03d}',
        )
