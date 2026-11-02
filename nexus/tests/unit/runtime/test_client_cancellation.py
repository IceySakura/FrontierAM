import asyncio
import logging

import pytest


@pytest.mark.asyncio
async def test_await_remote_operation_converts_child_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nexus.runtime.client.cancellation import RemoteChildOperationCancelledError, await_remote_operation

    operation = asyncio.get_running_loop().create_future()
    operation.cancel()
    diagnostic_logger = logging.getLogger('test.remote-cancellation')

    with caplog.at_level(logging.ERROR, logger=diagnostic_logger.name):
        with pytest.raises(
            RemoteChildOperationCancelledError,
            match='phase=invoke_poll_get.*request_id=request-123',
        ):
            await await_remote_operation(
                operation,
                logger=diagnostic_logger,
                phase='invoke_poll_get',
                base_url='http://sandbox:8000',
                request_id='request-123',
            )

    assert '[nexus-cancellation]' in caplog.text
    assert 'task_cancelling=0' in caplog.text
