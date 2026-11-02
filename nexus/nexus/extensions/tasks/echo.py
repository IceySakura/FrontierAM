"""Echo task example using ExecutionContext.

This task demonstrates:
- Accessing runtime via require_context().get_runtime()
- Using WithAgentRunner for agent execution
"""

# /// script
# dependencies = []
#
# [project]
# name = "echo"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# tasks = { echo = "nexus.extensions.tasks.echo" }
# ///

import asyncio
import logging
import random

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


logger = logging.getLogger(__name__)


def _write_echo_file(message: str) -> int:
    """Write message to /tmp/echo.txt."""
    return open('/tmp/echo.txt', 'w').write(message)  # noqa: SIM115


def _read_echo_file() -> str:
    """Read echoed message from /tmp/echo.txt."""
    try:
        with open('/tmp/echo.txt') as f:
            return f.read().strip()
    except Exception:
        return ''


class EchoTaskInput(TaskRunInput, WithAgentRunner):
    """Echo task input - business parameters only."""

    message: str
    run_oracle: bool = False
    sleep_min: float = 0.0
    sleep_max: float = 0.0


class EchoTaskOutput(TaskRunOutput, WithScore):
    """Echo task output with score."""

    echoed_message: str


@task_registry.register('echo')
async def echo_task(task_input: EchoTaskInput) -> EchoTaskOutput:
    """Echo task that requires the agent to write given message into /tmp/echo.txt."""
    logger.debug(f'Running echo task with message {task_input.message}')

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('Echo task requires a runtime provider to be configured')
    ctx.runtime_provider.configure(
        'main', image='mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155'
    )

    runtime = await ctx.get_runtime()

    if task_input.run_oracle:
        sleep_duration = task_input.sleep_min + random.random() * (task_input.sleep_max - task_input.sleep_min)
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
        await runtime.invoke(_write_echo_file, task_input.message)
    else:
        await task_input.run_agent(
            user_prompt='please write message in the following <input> tag into file /tmp/echo.txt. '
            f'<input>{task_input.message}</input>',
        )

    result = await runtime.invoke(_read_echo_file)

    score = 1.0 if task_input.message == result else 0.0

    return EchoTaskOutput(
        score=score,
        echoed_message=result,
    )
