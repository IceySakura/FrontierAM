import asyncio
import shlex
import time

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, require_context, task_registry


class SleepEchoInput(TaskRunInput):
    index: int
    message: str
    sleep_seconds: float = Field(default=0, ge=0)


class SleepEchoOutput(TaskRunOutput):
    index: int
    message: str


@task_registry.register('sleep_echo')
async def sleep_echo_task(task_input: SleepEchoInput) -> SleepEchoOutput:
    await asyncio.sleep(task_input.sleep_seconds)
    return SleepEchoOutput(index=task_input.index, message=task_input.message)


class DatasetTaskFilesEchoInput(TaskRunInput):
    index: int
    expected_marker: str


class DatasetTaskFilesEchoOutput(TaskRunOutput):
    index: int
    marker: str
    paths: list[str]


@task_registry.register('dataset_task_files_echo')
async def dataset_task_files_echo_task(task_input: DatasetTaskFilesEchoInput) -> DatasetTaskFilesEchoOutput:
    ctx = require_context()
    if ctx.task_files is None:
        raise RuntimeError('dataset task files are missing')
    paths = sorted(await ctx.task_files.list())
    marker = (await ctx.task_files.read_text('marker.txt')).strip()
    if marker != task_input.expected_marker:
        raise RuntimeError(f'dataset marker mismatch: {marker!r} != {task_input.expected_marker!r}')
    return DatasetTaskFilesEchoOutput(index=task_input.index, marker=marker, paths=paths)


def _wall_time() -> float:
    return time.time()


class SandboxAgentEchoInput(TaskRunInput, WithAgentRunner):
    index: int
    marker: str = Field(min_length=1)
    hold_seconds: float = Field(default=0, ge=0)


class SandboxAgentEchoOutput(TaskRunOutput):
    index: int
    marker: str
    runtime_id: str
    sandbox_ready_at: float
    sandbox_hold_finished_at: float
    command_return_code: int
    command_stdout: str
    agent_exit_status: str


@task_registry.register('sandbox_agent_echo')
async def sandbox_agent_echo_task(task_input: SandboxAgentEchoInput) -> SandboxAgentEchoOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime('main')
    provider = ctx.runtime_provider
    runtime_id = provider.get_runtime_id('main') if provider is not None else None
    if not runtime_id:
        raise RuntimeError('sandbox_agent_echo requires a runtime provider that exposes a runtime ID')

    sandbox_ready_at = _wall_time()
    marker_path = f'/tmp/nexus-sandbox-agent-{task_input.index:012d}.txt'
    quoted_path = shlex.quote(marker_path)
    quoted_marker = shlex.quote(task_input.marker)
    cleanup = await runtime.run_command(f'rm -f {quoted_path}', timeout=30)
    if cleanup.return_code != 0:
        raise RuntimeError(f'failed to clear sandbox marker: {cleanup.stderr or cleanup.stdout}')

    agent_output = await task_input.run_agent(
        user_prompt=(
            'Use the swe_agent/bash tool exactly once to execute this exact shell command. '
            'Do not simulate the command or only describe it.\n\n'
            f"printf '%s\\n' {quoted_marker} > {quoted_path} && cat {quoted_path}\n\n"
            'After the tool succeeds, answer briefly.'
        )
    )
    verification = await runtime.run_command(f'cat {quoted_path}', timeout=30)
    command_stdout = (verification.stdout or '').strip()
    if verification.return_code != 0 or command_stdout != task_input.marker:
        raise RuntimeError(
            'main_agent did not create the expected sandbox marker: '
            f'return_code={verification.return_code}, stdout={command_stdout!r}, stderr={verification.stderr!r}'
        )

    await asyncio.sleep(task_input.hold_seconds)
    return SandboxAgentEchoOutput(
        index=task_input.index,
        marker=task_input.marker,
        runtime_id=runtime_id,
        sandbox_ready_at=sandbox_ready_at,
        sandbox_hold_finished_at=_wall_time(),
        command_return_code=int(verification.return_code),
        command_stdout=command_stdout,
        agent_exit_status=str(getattr(agent_output, 'exit_status', 'unknown')),
    )
