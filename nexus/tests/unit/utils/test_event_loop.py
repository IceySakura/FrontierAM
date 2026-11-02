import asyncio

import pytest


@pytest.mark.asyncio
async def test_get_current_task_cancellation_count_returns_zero_for_running_task() -> None:
    from nexus.utils.event_loop import get_current_task_cancellation_count

    assert get_current_task_cancellation_count() == 0


@pytest.mark.asyncio
async def test_get_current_task_name_returns_asyncio_task_name() -> None:
    from nexus.utils.event_loop import get_current_task_name

    async def read_task_name() -> str | None:
        return get_current_task_name()

    task = asyncio.create_task(read_task_name(), name='cancellation-test-task')

    assert await task == 'cancellation-test-task'
