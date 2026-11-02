"""Repeat a single task config N times for concurrency / load testing with ``run-batch``.

Point ``nexus run-batch`` at this file as the generator and parameterise it through
environment variables::

    NEXUS_BATCH_TASK=../nexus-workspace/output/native-llm-bridge/ags-osworld.merged.yaml \
    NEXUS_BATCH_COUNT=50 \
    uv run nexus run-batch --generator nexus/utils/batch_repeat.py \
        --database /tmp/osworld-batch/batch.db --max-concurrent 20

Every copy resolves to the *same* config (``${oc.env:...}`` interpolations are expanded
once at load time) but gets a distinct ``run_id`` (``<task-stem>-00000``, ``-00001``, ...)
so the runs land in separate run directories.

``repeat_task()`` is also importable if you prefer a one-line hand-written generator::

    from nexus.utils.batch_repeat import repeat_task

    def generate_configs():
        return repeat_task('path/to/task.yaml', 100)
"""

from __future__ import annotations

import os

from pathlib import Path

from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


DEFAULT_COUNT = 10
TASK_ENV = 'NEXUS_BATCH_TASK'
COUNT_ENV = 'NEXUS_BATCH_COUNT'
RUN_PREFIX_ENV = 'NEXUS_BATCH_RUN_PREFIX'


def repeat_task(task_path: str | Path, count: int, *, run_prefix: str | None = None) -> list[BatchItem]:
    """Build *count* identical batch items from the task config at *task_path*.

    The config is loaded once (resolving ``${oc.env:...}`` interpolations) and deep-copied
    per item so runs stay independent. Each item gets a zero-padded ``run_id`` derived from
    *run_prefix* (defaults to the task file stem).
    """
    if count < 1:
        raise ValueError(f'count must be >= 1, got {count}')

    task_path = Path(task_path)
    base_config = RunInput.from_file(task_path)
    prefix = run_prefix or task_path.stem
    width = max(5, len(str(count - 1)))

    return [
        BatchItem(
            config=base_config.model_copy(deep=True),
            run_id=f'{prefix}-{index:0{width}d}',
            metadata={'source': str(task_path), 'index': index, 'repeat_total': count},
        )
        for index in range(count)
    ]


def _env_task_path() -> Path:
    raw = os.environ.get(TASK_ENV)
    if not raw:
        raise RuntimeError(f'Set {TASK_ENV} to the task YAML path to repeat (e.g. {TASK_ENV}=task.yaml).')
    return Path(raw)


def _env_count() -> int:
    raw = os.environ.get(COUNT_ENV)
    if not raw:
        return DEFAULT_COUNT
    try:
        count = int(raw)
    except ValueError as exc:
        raise RuntimeError(f'{COUNT_ENV} must be an integer, got {raw!r}.') from exc
    if count < 1:
        raise RuntimeError(f'{COUNT_ENV} must be >= 1, got {count}.')
    return count


def generate_configs() -> list[BatchItem]:
    """``run-batch`` entry point: repeat ``$NEXUS_BATCH_TASK`` ``$NEXUS_BATCH_COUNT`` times."""
    return repeat_task(_env_task_path(), _env_count(), run_prefix=os.environ.get(RUN_PREFIX_ENV))


def get_total_count() -> int:
    """Optional ``run-batch`` hook: report the batch size up front for progress display."""
    return _env_count()


__all__ = ['DEFAULT_COUNT', 'generate_configs', 'get_total_count', 'repeat_task']
