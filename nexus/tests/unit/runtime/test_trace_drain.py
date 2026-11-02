import asyncio

import pytest

from pydantic import ValidationError

from nexus.runner.single import RunInput
from nexus.runtime.trace_drain import drain_runtime_trace


@pytest.mark.parametrize('policy', ['ignore', '', None, False])
def test_trace_drain_policy_rejects_invalid_values(policy) -> None:
    with pytest.raises(ValidationError, match='trace_drain_error_policy'):
        RunInput(task_name='identity', runner={'trace_drain_error_policy': policy})


@pytest.mark.asyncio
async def test_warn_policy_preserves_task_cancellation(caplog) -> None:
    started = asyncio.Event()

    class Runtime:
        async def drain_trace(self):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        drain_runtime_trace(Runtime(), error_policy='warn', run_id='cancelled-run', runtime_name='main')
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not caplog.records
