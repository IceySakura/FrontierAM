"""A deterministic task for dataset/file transport acceptance, without an LLM."""

import asyncio
import hashlib

from pathlib import Path

from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry


class DatasetProbeInput(TaskRunInput):
    expected: str
    file: str = 'abcd'
    relative_path: str | None = None
    runtime: str | None = None
    fail_once: bool = False


class DatasetProbeOutput(TaskRunOutput):
    content: str
    relative_content: str | None = None
    runtime_content: str | None = None
    runtime_relative_content: str | None = None


def read_relative(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


@task_registry.register('dataset_portability_probe')
async def dataset_portability_probe(value: DatasetProbeInput) -> DatasetProbeOutput:
    ctx = require_context()
    if value.fail_once:
        marker = Path('.dataset-retry-' + hashlib.sha256(value.expected.encode()).hexdigest())
        if not marker.exists():
            marker.touch(exist_ok=False)
            raise ValueError('controlled first-attempt failure')
        marker.unlink()
    if ctx.task_files is None:
        raise ValueError('Task files are missing')
    content = await ctx.task_files.read_text(value.file)
    if content != value.expected:
        raise ValueError('TaskFiles content mismatch')
    result = DatasetProbeOutput(content=content)
    if value.runtime:
        runtime = await ctx.get_runtime(value.runtime)
        raw = await runtime.invoke('nexus.dataset.task_files::runtime_read_task_file', value.file)
        result.runtime_content = raw.decode('utf-8')
        if result.runtime_content != value.expected:
            raise ValueError('Runtime TaskFiles content mismatch')
        if value.relative_path:
            result.runtime_relative_content = await runtime.invoke(
                'builtins::eval',
                f'__import__("pathlib").Path({value.relative_path!r}).read_text(encoding="utf-8")',
            )
            if result.runtime_relative_content != value.expected:
                raise ValueError('Runtime relative-path content mismatch')
    elif value.relative_path:
        result.relative_content = read_relative(value.relative_path)
        if result.relative_content != value.expected:
            raise ValueError('Relative-path content mismatch')
    return result


class CleanupProbeInput(TaskRunInput):
    paths: list[str]


class CleanupProbeOutput(TaskRunOutput):
    checked_paths: list[str]
    cwd: str


@task_registry.register('dataset_portability_cleanup_probe')
async def dataset_portability_cleanup_probe(value: CleanupProbeInput) -> CleanupProbeOutput:
    for _ in range(300):
        if not any(Path(path).exists() for path in value.paths):
            return CleanupProbeOutput(checked_paths=value.paths, cwd=str(Path.cwd()))
        await asyncio.sleep(0.1)
    raise ValueError('Task bundle directories were not cleaned')
