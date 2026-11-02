"""Batch runner with SQLite storage.

All batch runs use SQLite database as backend, supporting:
- Single database file for all data
- Flexible export to different formats
- In-memory mode for temporary runs
- Resume/retry support
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
import traceback

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from nexus.runner.batch.batch_config import BatchConfig, BatchItem, BatchSummary, sanitize_run_id
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.batch.loader import serialize_processor_result
from nexus.runner.batch.log_capture import log_capture_context
from nexus.runner.batch.trace_processor import batch_database_context, flush_trace_queue
from nexus.runner.single import ProcessorConfig
from nexus.runner.single import run as run_single
from nexus.utils.eventloop_monitor import EventLoopMonitor


if TYPE_CHECKING:
    from nexus.runner.batch.loader import GeneratorLoader
    from nexus.runner.compose.compose_hooks import ComposeHookSpec
    from nexus.runner.compose.model import RunPart


logger = logging.getLogger(__name__)


class BatchTimeoutError(Exception):
    """Exception raised when a batch run exceeds the configured timeout."""

    pass


@dataclass(frozen=True)
class BatchShardSpec:
    """A single worker shard database and execution config."""

    index: int
    db_path: Path
    max_concurrent: int
    run_count: int


def get_iso_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(index: int, item: BatchItem) -> str:
    """Generate run_id with priority rules.

    Args:
        index: Index in the batch (0-based)
        item: BatchItem instance

    Returns:
        Sanitized run_id string
    """
    if item.run_id is not None:
        return sanitize_run_id(item.run_id)
    return f'{index:05d}'


def _shard_root_path(db_path: str | Path) -> Path:
    db_file = Path(db_path)
    return db_file.with_name(f'{db_file.name}.shards')


def _split_concurrency(max_concurrent: int, worker_count: int) -> list[int]:
    if worker_count <= 0:
        return []
    base, remainder = divmod(max_concurrent, worker_count)
    return [base + (1 if index < remainder else 0) for index in range(worker_count)]


def _shard_spec_to_metadata(spec: BatchShardSpec) -> dict:
    return {
        'index': spec.index,
        'database': spec.db_path.name,
        'max_concurrent': spec.max_concurrent,
        'run_count': spec.run_count,
    }


def _shard_metadata_to_specs(
    *,
    db_path: str | Path,
    shards: list[dict],
    batch_config: BatchConfig,
) -> list[BatchShardSpec]:
    if shards and batch_config.max_concurrent < len(shards):
        raise ValueError(
            f'Existing sharded batch has {len(shards)} shards, so --max-concurrent must be at least {len(shards)} '
            'when resuming it. Use a new database path to create a different shard layout.'
        )
    shard_root = _shard_root_path(db_path)
    concurrencies = _split_concurrency(batch_config.max_concurrent, len(shards))
    return [
        BatchShardSpec(
            index=int(shard['index']),
            db_path=shard_root / shard['database'],
            max_concurrent=concurrencies[position],
            run_count=int(shard.get('run_count', 0)),
        )
        for position, shard in enumerate(shards)
    ]


def _worker_batch_config(spec: BatchShardSpec, batch_config: BatchConfig) -> BatchConfig:
    return batch_config.model_copy(
        update={
            'max_concurrent': spec.max_concurrent,
            'worker_processes': 1,
        }
    )


async def _is_sharded_batch_database(db_path: str | Path) -> bool:
    if str(db_path) == ':memory:' or not Path(db_path).exists():
        return False
    async with BatchDatabase(db_path) as db:
        return await db.get_batch_info('sharded_batch') is True


async def _load_existing_shard_specs(
    *,
    db_path: str | Path,
    batch_config: BatchConfig,
) -> list[BatchShardSpec] | None:
    if not Path(db_path).exists():
        return None
    async with BatchDatabase(db_path) as db:
        if await db.get_batch_info('sharded_batch') is not True:
            return None
        shards = await db.get_batch_info('shards') or []
        specs = _shard_metadata_to_specs(db_path=db_path, shards=shards, batch_config=batch_config)
        await db.set_batch_info('config', batch_config.model_dump())

    for spec in specs:
        async with BatchDatabase(spec.db_path) as shard_db:
            await shard_db.set_batch_info('config', _worker_batch_config(spec, batch_config).model_dump())

    return specs


def _iter_generator_items(
    loader: GeneratorLoader,
    overrides: list[RunPart] | None,
    limit: int | None,
    compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> Iterator[BatchItem]:
    from nexus.runner.compose.compose_hooks import postprocess_generated_items

    if limit == 0:
        return
    items = loader.generate()
    try:
        selected = islice(items, limit) if limit is not None else items
        yield from postprocess_generated_items(selected, overrides=overrides, hooks=compose_hooks)
    finally:
        close = getattr(items, 'close', None)
        if callable(close):
            close()


async def _prepare_sharded_batch(
    *,
    generator_path: Path | None,
    batch_items: list[BatchItem] | None,
    db_path: str | Path,
    batch_config: BatchConfig,
    generator_overrides: list[RunPart] | Callable[[], list[RunPart]] | None = None,
    generator_limit: int | None = None,
    generator_compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> list[BatchShardSpec]:
    if str(db_path) == ':memory:':
        raise ValueError('--worker-processes requires a file-backed database path, not :memory:')

    shard_root = _shard_root_path(db_path)

    specs = await _load_existing_shard_specs(db_path=db_path, batch_config=batch_config)
    if specs is not None:
        if generator_overrides or generator_compose_hooks or generator_limit is not None:
            logger.warning(
                'Configs are already locked; ignoring generator options '
                '(--limit/--use/--set/--set-json, including compose hooks). Use a new database to change them.'
            )
        logger.debug('Using existing sharded batch metadata from %s', db_path)
        return specs

    if shard_root.exists() and any(shard_root.iterdir()):
        raise ValueError(
            f'Shard directory exists but {db_path} is not a sharded batch database: {shard_root}. '
            'Remove it or choose a new database path before starting a sharded batch.'
        )

    if Path(db_path).exists():
        async with BatchDatabase(db_path) as db:
            if await db.get_total_runs() > 0 or await db.is_generation_complete():
                raise ValueError(
                    f'Existing database is not a sharded batch database: {db_path}. '
                    'Choose a new database path for a sharded run.'
                )

    if generator_path is None and batch_items is None:
        raise ValueError(
            'Generator path or batch items are required to create sharded batch databases. '
            f'No existing sharded metadata found in {db_path}.'
        )

    if batch_items is None:
        from nexus.runner.batch.loader import GeneratorLoader

        assert generator_path is not None
        overrides = generator_overrides() if callable(generator_overrides) else generator_overrides
        loader = GeneratorLoader(generator_path)
        batch_meta = loader.get_batch_metadata()
        item_iter = _iter_generator_items(loader, overrides, generator_limit, generator_compose_hooks)
        source_label = str(generator_path)
    else:
        batch_meta = {'source': 'compose'}
        item_iter = iter(batch_items)
        source_label = '<compose>'

    generated_items = []
    for index, item in enumerate(item_iter):
        run_id = generate_run_id(index, item)
        config_dict = item.config.model_dump()
        config_dict['run_id'] = run_id
        generated_items.append((index, run_id, config_dict, item.metadata))

    if not generated_items:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        shard_root.mkdir(parents=True, exist_ok=True)
        async with BatchDatabase(db_path) as db:
            await db.set_batch_info('metadata', batch_meta)
            await db.set_batch_info('config', batch_config.model_dump())
            await db.set_batch_info('generator_path', source_label)
            await db.set_batch_info('sharded_batch', True)
            await db.set_batch_info('shard_root', str(shard_root))
            await db.set_batch_info('shards', [])
            await db.set_batch_info('start_time', get_iso_timestamp())
            await db.mark_generation_complete()
        return []

    worker_count = min(batch_config.worker_processes, batch_config.max_concurrent, len(generated_items))
    concurrencies = _split_concurrency(batch_config.max_concurrent, worker_count)
    buckets: list[list[tuple[int, str, dict, dict]]] = [[] for _ in range(worker_count)]
    for position, item in enumerate(generated_items):
        buckets[position % worker_count].append(item)

    shard_root.mkdir(parents=True, exist_ok=True)
    specs = [
        BatchShardSpec(
            index=index,
            db_path=shard_root / f'{index:04d}.db',
            max_concurrent=concurrencies[index],
            run_count=len(bucket),
        )
        for index, bucket in enumerate(buckets)
    ]

    for spec, bucket in zip(specs, buckets, strict=True):
        async with BatchDatabase(spec.db_path) as db:
            await db.set_batch_info('metadata', batch_meta)
            await db.set_batch_info('config', _worker_batch_config(spec, batch_config).model_dump())
            await db.set_batch_info('generator_path', source_label)
            await db.set_batch_info('parent_database', str(db_path))
            await db.set_batch_info('shard_index', spec.index)
            await db.set_batch_info('shard_count', worker_count)
            await db.set_batch_info('start_time', get_iso_timestamp())
            for _index, run_id, config_dict, metadata in bucket:
                await db.insert_run(run_id, config_dict, metadata)
            await db.mark_generation_complete()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with BatchDatabase(db_path) as db:
        await db.set_batch_info('metadata', batch_meta)
        await db.set_batch_info('config', batch_config.model_dump())
        await db.set_batch_info('generator_path', source_label)
        await db.set_batch_info('sharded_batch', True)
        await db.set_batch_info('shard_root', str(shard_root))
        await db.set_batch_info('shards', [_shard_spec_to_metadata(spec) for spec in specs])
        await db.set_batch_info('start_time', get_iso_timestamp())
        await db.mark_generation_complete()

    logger.debug('Generated %s configs across %s shard databases', len(generated_items), len(specs))
    logger.debug('Sharded batch metadata stored in: %s', db_path)
    return specs


async def _run_worker_processes(specs: list[BatchShardSpec]) -> None:
    async def run_one(spec: BatchShardSpec) -> tuple[BatchShardSpec, int]:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            'nexus.runner.batch.worker',
            '--database',
            str(spec.db_path),
        )
        return spec, await process.wait()

    results = await asyncio.gather(*[run_one(spec) for spec in specs])
    failed = [(spec, return_code) for spec, return_code in results if return_code != 0]
    if failed:
        details = ', '.join(f'shard {spec.index}: exit {return_code}' for spec, return_code in failed)
        raise RuntimeError(f'Batch worker process failed: {details}')


async def _aggregate_shard_summaries(specs: list[BatchShardSpec], *, start_time_epoch: float) -> BatchSummary:
    total = 0
    pending = 0
    running = 0
    success = 0
    failed = 0
    timeout = 0
    cancelled = 0
    failed_run_ids: list[str] = []

    for spec in specs:
        async with BatchDatabase(spec.db_path) as db:
            shard_summary = await db.get_summary()
            total += shard_summary.get('total', 0)
            pending += shard_summary.get('pending', 0)
            running += shard_summary.get('running', 0)
            success += shard_summary.get('success', 0)
            failed += shard_summary.get('failed', 0)
            timeout += shard_summary.get('timeout', 0)
            cancelled += shard_summary.get('cancelled', 0)
            # Summary only needs ids; avoid materializing GB-scale output columns.
            async with contextlib.aclosing(db.iter_runs(status='failed', fields=['run_id'])) as failed_runs:
                failed_run_ids.extend([str(run['run_id']) async for run in failed_runs])

    return BatchSummary(
        total=total,
        pending=pending,
        running=running,
        success=success,
        failed=failed,
        timeout=timeout,
        cancelled=cancelled,
        failed_runs=failed_run_ids,
        duration_seconds=time.time() - start_time_epoch,
        start_time=get_iso_timestamp(),
        end_time=get_iso_timestamp(),
    )


async def _save_sharded_batch_completion(db_path: str | Path, summary: BatchSummary) -> None:
    async with BatchDatabase(db_path) as db:
        await db.set_batch_info('end_time', summary.end_time)
        await db.set_batch_info('duration_seconds', summary.duration_seconds)


async def _export_sharded_results(
    specs: list[BatchShardSpec],
    export_format: str | None,
    export_path: Path | None,
) -> None:
    if not export_format or export_path is None:
        return

    if export_format == 'jsonl':
        from nexus.cli.commands.export import _atomic_text_output, write_run_to_jsonl

        with _atomic_text_output(export_path) as export_file:
            for spec in specs:
                async with BatchDatabase(spec.db_path) as db, contextlib.aclosing(db.iter_runs()) as runs:
                    async for run in runs:
                        write_run_to_jsonl(run, export_file)
        return

    if export_format == 'csv':
        import csv

        from nexus.cli.commands.export import _CSV_FIELDS, _atomic_text_output, write_csv_header, write_run_to_csv

        with _atomic_text_output(export_path, newline='') as export_file:
            writer = csv.writer(export_file)
            write_csv_header(writer)
            for spec in specs:
                async with (
                    BatchDatabase(spec.db_path) as db,
                    contextlib.aclosing(db.iter_runs(fields=_CSV_FIELDS)) as runs,
                ):
                    async for run in runs:
                        write_run_to_csv(run, writer)
        return

    if export_format == 'full':
        from nexus.cli.commands.export import AtomicDirectoryOutput, export_run_to_full

        atomic_output = AtomicDirectoryOutput(export_path)
        staging_path = atomic_output.open()
        try:
            for spec in specs:
                async with BatchDatabase(spec.db_path) as db, contextlib.aclosing(db.iter_runs()) as runs:
                    async for run in runs:
                        export_run_to_full(run, staging_path)
            atomic_output.commit()
        except BaseException:
            atomic_output.abort()
            raise


async def _run_batch_sharded(
    generator_path: Path | None,
    batch_items: list[BatchItem] | None,
    db_path: str | Path,
    batch_config: BatchConfig,
    export_format: str | None = None,
    export_path: Path | None = None,
    generator_overrides: list[RunPart] | Callable[[], list[RunPart]] | None = None,
    generator_limit: int | None = None,
    generator_compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> BatchSummary:
    start_time_epoch = time.time()
    specs = await _prepare_sharded_batch(
        generator_path=generator_path,
        batch_items=batch_items,
        db_path=db_path,
        batch_config=batch_config,
        generator_overrides=generator_overrides,
        generator_limit=generator_limit,
        generator_compose_hooks=generator_compose_hooks,
    )
    if not specs:
        summary = BatchSummary(
            total=0,
            duration_seconds=time.time() - start_time_epoch,
            start_time=get_iso_timestamp(),
            end_time=get_iso_timestamp(),
        )
        await _export_sharded_results(specs, export_format, export_path)
        await _save_sharded_batch_completion(db_path, summary)
        return summary

    logger.info(
        'Running sharded batch with %s worker processes, total max_concurrent=%s',
        len(specs),
        batch_config.max_concurrent,
    )
    for spec in specs:
        logger.debug(
            '  shard %04d: runs=%s max_concurrent=%s db=%s',
            spec.index,
            spec.run_count,
            spec.max_concurrent,
            spec.db_path,
        )

    await _run_worker_processes(specs)
    summary = await _aggregate_shard_summaries(specs, start_time_epoch=start_time_epoch)
    await _save_sharded_batch_completion(db_path, summary)
    await _export_sharded_results(specs, export_format, export_path)
    return summary


async def run_batch(
    generator_path: Path | None,
    db_path: str | Path,
    batch_config: BatchConfig,
    export_format: str | None = None,
    export_path: Path | None = None,
    batch_items: list[BatchItem] | None = None,
    generator_overrides: list[RunPart] | Callable[[], list[RunPart]] | None = None,
    generator_limit: int | None = None,
    generator_compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> BatchSummary:
    """Run batch with SQLite storage in two phases.

    Phase 1 (Generation): Load generator and create all run configs in database.
    Phase 2 (Execution): Execute runs concurrently.

    Once Phase 1 completes, configs are locked and cannot be modified.
    Subsequent runs can skip Phase 1 by not providing generator_path.

    Args:
        generator_path: Path to config generator Python file (optional if generation complete)
        db_path: Path to SQLite database file, or ':memory:' for in-memory database
        batch_config: Batch execution configuration
        generator_overrides: Ordered RunInput overlays, or a resolver called only when generation is needed.
        generator_compose_hooks: Compose hooks applied to each generated item after the overlays.
        generator_limit: Maximum generated items to store; None consumes all, zero skips generation.
        export_format: Optional export format for incremental export (jsonl, csv, full)
        export_path: Optional export path (for jsonl/csv file or full directory)

    Returns:
        BatchSummary with execution results
    """
    start_time_epoch = time.time()

    # Convert to string for database
    db_path_str = str(db_path)
    if generator_limit is not None and generator_limit < 0:
        raise ValueError('generator_limit must be non-negative')
    if batch_config.worker_processes < 1:
        raise ValueError('worker_processes must be at least 1')
    if batch_config.max_concurrent < 1:
        raise ValueError('max_concurrent must be at least 1')
    if batch_config.worker_processes == 1 and await _is_sharded_batch_database(db_path):
        raise ValueError(
            f'Existing sharded batch found in {db_path}. '
            'Use --worker-processes > 1 to resume it, or choose a different database path for a single-worker run.'
        )
    if batch_config.worker_processes > 1:
        return await _run_batch_sharded(
            generator_path=generator_path,
            batch_items=batch_items,
            db_path=db_path,
            batch_config=batch_config,
            export_format=export_format,
            export_path=export_path,
            generator_overrides=generator_overrides,
            generator_limit=generator_limit,
            generator_compose_hooks=generator_compose_hooks,
        )

    if db_path_str == ':memory:':
        logger.debug('Using in-memory database')
    else:
        # Create parent directory if needed
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Using database: {db_path}')

    # Connect to database
    async with BatchDatabase(db_path_str) as db:
        # Check generation status
        generation_complete = await db.is_generation_complete()
        total_runs = await db.get_total_runs()

        # Phase 1: Config Generation
        if not generation_complete:
            if generator_path is None and batch_items is None:
                raise ValueError(
                    'Generator path or batch items are required for initial config generation. '
                    'The database does not have completed config generation.'
                )

            logger.debug('=' * 60)
            logger.debug('PHASE 1: Config Generation')
            logger.debug('Generator: %s', generator_path if generator_path is not None else '<compose>')
            logger.debug('=' * 60)

            if batch_items is None:
                from nexus.runner.batch.loader import GeneratorLoader

                assert generator_path is not None
                overrides = generator_overrides() if callable(generator_overrides) else generator_overrides
                loader = GeneratorLoader(generator_path)
                batch_meta = loader.get_batch_metadata()
                item_iter = _iter_generator_items(loader, overrides, generator_limit, generator_compose_hooks)
                source_label = str(generator_path)
            else:
                batch_meta = {'source': 'compose'}
                item_iter = iter(batch_items)
                source_label = '<compose>'

            # Save batch metadata
            await db.set_batch_info('metadata', batch_meta)
            await db.set_batch_info('config', batch_config.model_dump())
            await db.set_batch_info('generator_path', source_label)
            await db.set_batch_info('start_time', get_iso_timestamp())

            logger.debug('Generating configs...')
            generated_count = 0
            for index, item in enumerate(item_iter):
                run_id = generate_run_id(index, item)

                # In generation phase, only insert new runs
                if not await db.run_exists(run_id):
                    # Ensure run_id is set in the config
                    config_dict = item.config.model_dump()
                    config_dict['run_id'] = run_id
                    await db.insert_run(run_id, config_dict, item.metadata)
                    generated_count += 1

            logger.info(f'Generated {generated_count} configs')

            # Mark generation as complete
            await db.mark_generation_complete()
            total_runs = await db.get_total_runs()

            logger.info('Config generation phase completed successfully')
            logger.debug(f'Total runs in database: {total_runs}')
        else:
            # Generation already complete
            if generator_overrides or generator_compose_hooks or generator_limit is not None:
                logger.warning(
                    'Configs are already locked; ignoring generator options '
                    '(--limit/--use/--set/--set-json, including compose hooks). Use a new database to change them.'
                )
            if generator_path is not None:
                logger.warning(
                    f'Generator path provided but configs are already locked. Ignoring generator: {generator_path}'
                )
            if batch_items is not None:
                logger.warning('Batch items provided but configs are already locked. Ignoring composed items.')
            logger.debug('Config generation already complete, using existing configs')
            logger.debug(f'Total runs in database: {total_runs}')

        # Phase 2: Execution
        logger.debug('=' * 60)
        logger.debug('PHASE 2: Execution')
        logger.debug('=' * 60)

        # Filter items to run based on resume/retry settings
        items_to_run: list[tuple[int, str, BatchItem]] = []

        # Get all runs from database. Scheduling only needs the summary columns;
        # streaming with a column subset keeps GB-scale output fields on disk.
        summary_columns = ['run_id', 'status', 'config', 'metadata']
        async with contextlib.aclosing(db.iter_runs(fields=summary_columns)) as all_runs:
            async for run in all_runs:
                run_id = run['run_id']
                status = run['status']
                config = run['config']
                metadata = run.get('metadata', {})

                # Check if should skip or execute
                if batch_config.resume and status == 'success':
                    logger.debug(f'[SKIP] {run_id} already succeeded')
                    continue

                if batch_config.retry_only and status != 'failed':
                    logger.debug(f'[SKIP] {run_id} not in failed state')
                    continue

                # Create BatchItem from stored config
                from nexus.runner.single import RunInput

                # Inject run_id into config if not already present
                if 'run_id' not in config:
                    config['run_id'] = run_id
                # Use from_dict to resolve OmegaConf interpolations (e.g., ${oc.env:API_KEY})
                run_input = RunInput.from_dict(config)
                batch_item = BatchItem(config=run_input, run_id=run_id, metadata=metadata)
                items_to_run.append((0, run_id, batch_item))

        total_tasks = len(items_to_run)
        logger.info(f'Will execute {total_tasks} tasks')

        if total_tasks == 0:
            logger.info('No tasks to execute')
            summary_dict = await db.get_summary()
            return BatchSummary(**summary_dict)

        # Prepare export function and open files if needed
        export_fn = None
        export_output = None
        export_replace_on_close = False
        csv_writer = None

        if export_format and export_path:
            if export_format == 'jsonl':
                from nexus.cli.commands.export import AtomicTextOutput, write_run_to_jsonl

                # Results stream into <target>.partial and replace the target only
                # when the batch completes. The previous target is discarded up
                # front (as the old truncating open did), so an aborted batch never
                # leaves an older, complete-looking file behind.
                export_output = AtomicTextOutput(Path(export_path), discard_existing=True)
                export_file = export_output.file

                def export_fn(run: dict) -> None:
                    write_run_to_jsonl(run, export_file)

            elif export_format == 'csv':
                import csv

                from nexus.cli.commands.export import AtomicTextOutput, write_csv_header, write_run_to_csv

                # Open file and write header once
                export_output = AtomicTextOutput(Path(export_path), newline='', discard_existing=True)
                export_file = export_output.file
                csv_writer = csv.writer(export_file)
                write_csv_header(csv_writer)

                def export_fn(run: dict) -> None:
                    write_run_to_csv(run, csv_writer)
                    export_file.flush()  # Ensure data is written

            elif export_format == 'full':
                from nexus.cli.commands.export import AtomicDirectoryOutput, export_run_to_full

                export_output = AtomicDirectoryOutput(Path(export_path))
                export_root = export_output.open()

                def export_fn(run: dict) -> None:
                    export_run_to_full(run, export_root)

        # Execute tasks
        completed = 0

        async def run_single_item(run_id: str, item: BatchItem) -> None:
            """Execute a single batch item with retry support."""
            nonlocal completed

            base_config = item.config
            retry_workflow_states = base_config.workflow_states

            # Get current retry count from database
            current_retry = await db.get_retry_count(run_id)
            max_retries = batch_config.retry_failed
            attempt = 0

            # Retry loop
            while attempt <= max_retries:
                is_retry = attempt > 0
                retry_suffix = f' (retry {attempt}/{max_retries})' if is_retry else ''

                logger.debug(f'[{completed + 1}/{total_tasks}] Running {run_id}{retry_suffix}...')

                # Clear stale outputs/errors from previous attempts before re-running.
                update_fields = {
                    'end_time': None,
                    'error': None,
                    'task_output': None,
                    'agent_output': None,
                    'judge_output': None,
                    'processor_results': None,
                }
                if not is_retry:
                    update_fields['start_time'] = get_iso_timestamp()

                await db.update_run_status(run_id, 'running', **update_fields)

                # Every attempt gets an isolated input. By default retries use
                # the original input; workflow progress is opt-in.
                workflow_states = (
                    retry_workflow_states
                    if is_retry and batch_config.resume_workflow_on_retry
                    else base_config.workflow_states
                )
                config = base_config.model_copy(
                    update={
                        'processors': dict(base_config.processors),
                        'workflow_states': dict(workflow_states),
                    }
                )

                if batch_config.use_gateway:
                    config.runner.use_gateway = True

                # Add trace processor if enabled
                if batch_config.enable_trace_db:
                    if config.processors is None:
                        config.processors = {}
                    config.processors['batch_trace'] = ProcessorConfig(name='batch_trace', args={})

                # Execute with database context and log capture
                result = None
                batch_timeout = False

                with batch_database_context(db, run_id):
                    log_capture = None
                    if batch_config.enable_log_capture:
                        log_capture = log_capture_context(
                            database=db,
                            run_id=run_id,
                            dual_write=batch_config.log_dual_write,
                            chunk_size_mb=batch_config.log_chunk_size_mb,
                            chunk_time_seconds=batch_config.log_chunk_time_seconds,
                        )
                        await log_capture.__aenter__()

                    try:
                        if batch_config.timeout_per_run:
                            try:
                                result = await asyncio.wait_for(
                                    run_single(config), timeout=batch_config.timeout_per_run
                                )
                            except asyncio.TimeoutError:
                                batch_timeout = True
                        else:
                            result = await run_single(config)
                    finally:
                        if log_capture:
                            await log_capture.__aexit__(None, None, None)

                # Handle batch-level timeout
                if batch_timeout:
                    error_msg = f'Batch run exceeded timeout of {batch_config.timeout_per_run}s'
                    attempt += 1

                    if attempt > max_retries:
                        await db.update_run_status(
                            run_id,
                            'timeout',
                            end_time=get_iso_timestamp(),
                            retry_count=current_retry + attempt - 1,
                            error=error_msg,
                        )
                        logger.error(f'[TIMEOUT] {run_id}: {error_msg} (exhausted retries)')

                        if export_fn:
                            run_record = await db.get_run(run_id)
                            if run_record:
                                export_fn(run_record)

                        if batch_config.fail_fast:
                            raise BatchTimeoutError(error_msg)
                    else:
                        logger.warning(f'[TIMEOUT] {run_id}: {error_msg}, will retry ({attempt}/{max_retries})')
                        await db.update_run_status(
                            run_id, 'failed', retry_count=current_retry + attempt, error=error_msg
                        )
                    continue

                # Check result from single.run (now always returns RunOutput)
                # result is guaranteed to be non-None here (batch_timeout case handled above)
                assert result is not None

                # Serialize processor results (available even on failure)
                processor_results = {}
                for key, proc_result in result.processor_results.items():
                    processor_results[key] = serialize_processor_result(proc_result)
                task_output = serialize_processor_result(result.task_output)
                agent_output = serialize_processor_result(result.agent_output)
                judge_output = serialize_processor_result(result.judge_output)

                if result.success:
                    # Task succeeded
                    await db.update_run_status(
                        run_id,
                        'success',
                        end_time=get_iso_timestamp(),
                        retry_count=current_retry + attempt,
                        error=None,
                        task_output=task_output,
                        agent_output=agent_output,
                        judge_output=judge_output,
                        processor_results=processor_results,
                    )

                    success_msg = f'[SUCCESS] {run_id}'
                    if is_retry:
                        success_msg += f' after {attempt} retries'
                    logger.info(success_msg)

                    if export_fn:
                        run_record = await db.get_run(run_id)
                        if run_record:
                            export_fn(run_record)

                    break  # Success, exit retry loop

                # Task failed (error captured in RunOutput)
                assert result.error is not None  # error is always set when success=False
                error_msg = f'{result.error.type}: {result.error.message}'
                full_error = f'{error_msg}\n\n{result.error.traceback}' if result.error.traceback else error_msg
                attempt += 1

                if attempt > max_retries:
                    await db.update_run_status(
                        run_id,
                        'failed',
                        end_time=get_iso_timestamp(),
                        retry_count=current_retry + attempt - 1,
                        error=full_error,
                        task_output=task_output,
                        agent_output=agent_output,
                        judge_output=judge_output,
                        processor_results=processor_results,  # Save partial results
                    )
                    logger.error(f'[FAILED] {run_id}: {error_msg} (exhausted retries)')

                    if export_fn:
                        run_record = await db.get_run(run_id)
                        if run_record:
                            export_fn(run_record)

                    if batch_config.fail_fast:
                        raise RuntimeError(full_error)
                else:
                    logger.warning(f'[FAILED] {run_id}: {error_msg}, will retry ({attempt}/{max_retries})')
                    if batch_config.resume_workflow_on_retry:
                        retry_workflow_states = result.workflow_states
                    await db.update_run_status(run_id, 'failed', retry_count=current_retry + attempt, error=full_error)

            # Mark as completed
            completed += 1

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(batch_config.max_concurrent)

        # Conflict group locks: tasks sharing a group run serially
        conflict_locks: dict[str, asyncio.Lock] = {}

        def _get_conflict_locks(metadata: dict) -> list[asyncio.Lock]:
            """Get ordered locks for conflict groups to prevent deadlocks."""
            groups = metadata.get('conflict_groups', [])
            if not groups:
                return []
            locks = []
            for g in sorted(groups):  # sorted to avoid deadlocks
                if g not in conflict_locks:
                    conflict_locks[g] = asyncio.Lock()
                locks.append(conflict_locks[g])
            return locks

        async def bounded_run(run_id: str, item: BatchItem) -> None:
            # Acquire conflict group locks before the semaphore so that
            # conflicting tasks don't occupy semaphore slots while waiting.
            locks = _get_conflict_locks(item.metadata) if batch_config.enable_conflict_groups else []
            for lock in locks:
                await lock.acquire()
            try:
                async with semaphore:
                    await run_single_item(run_id, item)
            except Exception as exc:
                # gather() below collects exceptions as return values and never reads
                # them, so anything escaping run_single_item would leave the run at
                # 'running' with an empty error and counted as neither success nor
                # failure. fail_fast raises on purpose after finalizing the run, so only
                # a row still marked 'running' is an escape that needs recording here.
                logger.error(f'[ERROR] {run_id}: {type(exc).__name__}: {exc}', exc_info=True)
                run_record = await db.get_run(run_id)
                if run_record and run_record.get('status') == 'running':
                    await db.update_run_status(
                        run_id,
                        'failed',
                        end_time=get_iso_timestamp(),
                        error=f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}',
                    )
                if batch_config.fail_fast:
                    raise
            finally:
                for lock in reversed(locks):
                    lock.release()

        # Monitor event loop scheduling lag during the execution phase. The
        # watchdog runs on its own OS thread and logs a warning when the loop
        # is blocked (spike / sustained lag / stall), which is what a slow
        # synchronous call inside a task would cause.
        loop_monitor = EventLoopMonitor()
        loop_monitor.start(watchdog_interval=1.0, lag_threshold=5.0)

        try:
            # Run all tasks
            await asyncio.gather(
                *[bounded_run(rid, item) for _, rid, item in items_to_run],
                return_exceptions=not batch_config.fail_fast,
            )

            # Generate summary
            end_time_epoch = time.time()
            summary_dict = await db.get_summary()
            summary_dict['duration_seconds'] = end_time_epoch - start_time_epoch
            summary_dict['start_time'] = get_iso_timestamp()
            summary_dict['end_time'] = get_iso_timestamp()

            # Get failed run IDs
            async with contextlib.aclosing(db.iter_runs(status='failed', fields=['run_id'])) as failed_runs:
                summary_dict['failed_runs'] = [r['run_id'] async for r in failed_runs]

            # Save end time
            await db.set_batch_info('end_time', get_iso_timestamp())
            await db.set_batch_info('duration_seconds', summary_dict['duration_seconds'])

            summary = BatchSummary(**summary_dict)

            logger.info(f'Batch completed: {summary.success}/{summary.total} succeeded')
            if db_path_str != ':memory:':
                logger.info(f'Database saved to {db_path}')

            export_replace_on_close = True
            return summary

        finally:
            loop_monitor.stop()

            if batch_config.enable_trace_db:
                flush_trace_queue(timeout=5.0)

            # Publish (or keep as .partial) the incremental export file
            if export_output is not None:
                if export_replace_on_close:
                    export_output.commit()
                else:
                    export_output.abort()
                    logger.warning(
                        'Batch did not complete; partial export kept at %s (nothing published to %s)',
                        export_output.partial,
                        export_output.target,
                    )
                logger.debug(f'Closed export file: {export_path}')


__all__ = ['run_batch']
