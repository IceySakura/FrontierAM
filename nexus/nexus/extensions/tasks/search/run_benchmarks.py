#!/usr/bin/env python3
"""Search Task Benchmark Runner.

Run multiple search benchmarks concurrently with result aggregation.

Usage:
    python run_benchmarks.py                                    # Fresh run (built-in benchmarks)
    python run_benchmarks.py -c search_config.yaml              # YAML config (benchmarks + agent)
    python run_benchmarks.py -c search_config.yaml -b seal_0    # Subset of benchmarks from config
    python run_benchmarks.py -r                                 # Resume from existing database
    python run_benchmarks.py -n exp1                            # Named experiment
    python run_benchmarks.py --model gpt-4o                     # Override model
    python run_benchmarks.py -k 3                               # 3 RL rollout groups (avg@3 / pass@3)
    python run_benchmarks.py --limit 100                        # Limit items per benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import sqlite3
import sys
import tempfile

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.extensions.agents.search_agent.agent import TOOLSET_REQUIRED_TOOLS
from nexus.extensions.tasks.search.reporting import (
    build_execution_metadata,
    print_benchmark_metrics,
    print_rich_summary,
    print_summary,
    save_summary,
    save_summary_html,
)


# ============================================================================
# Constants
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_WORKSPACE = PROJECT_ROOT / 'workspace' / 'search_task'
DEFAULT_ENV_FILE = PROJECT_ROOT / '.env.search'
DEFAULT_GENERATOR = SCRIPT_DIR / 'generator.py'
DEFAULT_DB_SYNC_INTERVAL = 0.0
DEFAULT_TIMEOUT_SECONDS = 7200
PROCESS_TERMINATION_GRACE_SECONDS = 5.0

_PARENT_DEATH_WATCHDOG_SCRIPT = r"""
parent_pid=$1
child_pgid=$2
grace_seconds=$3

while kill -0 "$parent_pid" 2>/dev/null && kill -0 -- "-$child_pgid" 2>/dev/null; do
    sleep 1
done

kill -0 -- "-$child_pgid" 2>/dev/null || exit 0
kill -TERM -- "-$child_pgid" 2>/dev/null || exit 0
sleep "$grace_seconds"
kill -KILL -- "-$child_pgid" 2>/dev/null || true
"""

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        'SEARCH_BENCHMARK_DATA_DIR',
        '/apdcephfs_zwfy/share_304017095/mannaliang/data/nexus_data',
    )
)
DEFAULT_BENCHMARKS = {
    'seal_0': 'sealqa_seal_0.parquet',
    'gaia_text_only': 'gaia_text_only.parquet',
    'old_gaia_text_only': 'old_gaia_text_only.parquet',
    'browse_comp_zh': 'browse_comp_zh.parquet',
    'browse_comp_150': 'browse_comp_150.parquet',
    'browse_comp_full': 'browse_comp_full.parquet',
    'fin_search_comp_t3': 'fin_search_comp_t3.parquet',
    'fin_search_comp_t2_and_t3': 'fin_search_comp_t2_and_t3.parquet',
    'fin_search_comp_full': 'fin_search_comp_full.parquet',
    'old_wide_search': 'old_wide_search.parquet',
    'wide_search': 'wide_search.parquet',
    'facts_search': 'facts_search.parquet',
    'frames': 'frames.parquet',
    'hle_150': 'hle_150.parquet',
    'seal_0_260320': 'sealqa_seal_0_260320.parquet',
    'seal_0_260406': 'sealqa_seal_0_260406.parquet',
}

logger = logging.getLogger(__name__)


# ============================================================================
# Utilities
# ============================================================================


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f'{h}h {m}m {sec}s' if h else (f'{m}m {sec}s' if m else f'{sec}s')


def safe_copy_sqlite_db(src: Path, dst: Path) -> bool:
    """Safely copy SQLite database using backup API to ensure consistency.

    This prevents database corruption that can occur when copying a database
    file while it's being written to, especially on network filesystems like CephFS.
    """
    import sqlite3

    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    tmp_dst = dst.with_name(f'.{dst.name}.tmp.{os.getpid()}')

    try:
        # Connect to source database in read-only mode
        src_conn = sqlite3.connect(f'file:{src}?mode=ro', uri=True, timeout=30.0)
        # Copy to a temp file first, then atomically replace destination.
        # This keeps destination lock hold time minimal for external readers.
        tmp_dst.unlink(missing_ok=True)
        dst_conn = sqlite3.connect(str(tmp_dst), timeout=30.0)

        # Use SQLite backup API for consistent copy
        src_conn.backup(dst_conn)

        dst_conn.close()
        dst_conn = None
        src_conn.close()
        src_conn = None

        os.replace(tmp_dst, dst)
        return True
    except Exception as e:
        logger.warning(f'SQLite backup failed: {e}')
        return False
    finally:
        if dst_conn is not None:
            dst_conn.close()
        if src_conn is not None:
            src_conn.close()
        tmp_dst.unlink(missing_ok=True)


async def sync_db_periodically(
    src: Path,
    dst: Path,
    interval: float = 30.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Periodically sync database from temp to output directory.

    This allows real-time progress monitoring while keeping SQLite operations
    on local filesystem to avoid CephFS I/O errors.
    Uses SQLite backup API for consistent snapshots.
    """
    while not (stop_event and stop_event.is_set()):
        try:
            if src.exists():
                safe_copy_sqlite_db(src, dst)
                logger.debug(f'Synced database to {dst}')
        except Exception as e:
            logger.debug(f'Database sync failed (will retry): {e}')

        # Wait for interval or until stop event
        try:
            if stop_event:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            await asyncio.sleep(interval)
        except asyncio.TimeoutError:
            continue


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    dataset_path: Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark execution."""

    name: str
    status: str  # 'success' | 'failed' | 'skipped'
    start_time: datetime
    end_time: datetime | None = None
    db_path: Path | None = None
    metrics_path: Path | None = None
    traj_path: Path | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    # Multi-run metadata (set when k > 1)
    run_index: int | None = None  # 1-indexed run number within a k-run set
    total_runs: int | None = None  # total k in the run set

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds() if self.end_time else 0.0

    @property
    def is_multi_run_aggregate(self) -> bool:
        """True if this result aggregates multiple runs (avg@k)."""
        return self.total_runs is not None and self.total_runs > 1 and self.run_index is None


@dataclass
class RunSummary:
    """Summary of all benchmark executions."""

    results: list[BenchmarkResult] = field(default_factory=list)
    output_dir: Path | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    k: int = 1  # number of runs per benchmark

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds() if self.end_time else 0.0


# ============================================================================
# Environment Loading
# ============================================================================


def load_env_file(env_file: Path) -> dict[str, str]:
    """Load environment variables from .env file (excluding SEARCH_DATASET_PATH).

    Returns an empty dict when the file is missing — a self-contained YAML
    config is enough to run benchmarks, and requiring an env file would force
    users to maintain one even when every credential lives in YAML already.
    """
    if not env_file.exists():
        logger.info(f'Env file not found, continuing with YAML-only config: {env_file}')
        return {}

    env_vars = _parse_env_file(env_file)
    env_vars.pop('SEARCH_DATASET_PATH', None)
    return {k: v for k, v in env_vars.items() if v}


def _parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse .env file and return environment variables dict."""
    env_vars = {}
    content = env_file.read_text(encoding='utf-8')
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if line.startswith('export '):
            line = line[7:]
        if '=' not in line:
            i += 1
            continue
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()

        # Handle multi-line quoted values (double quotes spanning multiple lines)
        if value.startswith('"') and not value.endswith('"'):
            # Collect lines until closing quote
            collected = [value[1:]]  # Strip opening quote
            i += 1
            while i < len(lines):
                if lines[i].rstrip().endswith('"'):
                    collected.append(lines[i].rstrip()[:-1])  # Strip closing quote
                    break
                collected.append(lines[i].rstrip())
                i += 1
            value = '\n'.join(collected)
        elif len(value) >= 2 and value[0] in '"\'' and value[-1] == value[0]:
            # Single-line quoted value — strip quotes
            value = value[1:-1]

        env_vars[key] = value
        i += 1
    return env_vars


# ============================================================================
# Benchmark Configuration Loading
# ============================================================================


def load_benchmarks(search_config: Any | None = None) -> list[BenchmarkConfig]:
    """Load benchmark configs from a ``SearchConfig`` or fall back to built-in defaults.

    Args:
        search_config: An optional :class:`SearchConfig` whose ``benchmarks``
            mapping (``{name: parquet_path}``) takes priority over the built-in
            ``DEFAULT_BENCHMARKS``.
    """
    benchmarks = getattr(search_config, 'benchmarks', None) if search_config is not None else None
    if benchmarks:
        return [BenchmarkConfig(name=name, dataset_path=Path(path)) for name, path in benchmarks.items()]

    return [
        BenchmarkConfig(name=name, dataset_path=DEFAULT_DATA_DIR / filename)
        for name, filename in DEFAULT_BENCHMARKS.items()
    ]


@contextlib.contextmanager
def temporary_env(env_vars: dict[str, str]) -> Iterator[None]:
    """Temporarily apply environment variables while generating DB configs."""
    old_values = {key: os.environ.get(key) for key in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def reset_zero_score_runs(db_path: Path) -> int:
    """Reset successful runs with score=0 back to ``pending`` so they re-run.

    Used together with ``--resume --rerun-zero-score`` to selectively re-execute
    instances the model previously answered incorrectly without touching the
    score=1 runs.  Returns the number of rows reset.
    """
    if not db_path.exists():
        logger.info(f'Skip zero-score reset; database not found: {db_path}')
        return 0

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        cursor = conn.execute(
            """
            -- Only flip status + clear timing/error fields. Outputs (task_output,
            -- agent_output, processor_results) are left untouched: the batch runner
            -- will atomically overwrite them on the next successful run, and keeping
            -- the previous values around is useful for diff-style debugging.
            UPDATE runs
            SET status = 'pending',
                start_time = NULL,
                end_time = NULL,
                error = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'success'
              AND task_output IS NOT NULL
              AND json_valid(task_output)
              AND json_extract(task_output, '$.score') = 0
            """
        )
        reset_count = cursor.rowcount or 0
        conn.commit()
    finally:
        conn.close()

    if reset_count:
        logger.info(f'Reset {reset_count} zero-score success runs to pending: {db_path}')
    else:
        logger.info(f'No zero-score success runs to reset: {db_path}')
    return reset_count


def refresh_resume_db_configs(
    db_path: Path,
    generator_path: Path,
    env_vars: dict[str, str],
    batch_config: dict[str, Any],
) -> tuple[int, int, int]:
    """Refresh non-success run configs in an existing batch DB from the generator.

    Successful rows are intentionally left untouched because resume mode skips
    them, so changing their configs would make stored outputs look like they
    were produced by the new YAML when they were not. Generated rows missing
    from the DB are inserted as pending so they can run in the same resume pass.
    """
    if not db_path.exists():
        logger.info(f'Skip DB config refresh; database not found: {db_path}')
        return 0, 0, 0

    from nexus.runner.batch.loader import GeneratorLoader
    from nexus.runner.batch.runner import generate_run_id

    logger.info(f'Refreshing resumable DB configs from current YAML/generator: {db_path}')
    with temporary_env(env_vars):
        loader = GeneratorLoader(generator_path)
        batch_meta = loader.get_batch_metadata()
        generated_items = list(loader.generate())

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        cursor = conn.execute('SELECT version FROM schema_version')
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f'Database is missing schema_version: {db_path}')

        refreshed = 0
        skipped_success = 0
        inserted = 0

        for index, item in enumerate(generated_items):
            run_id = generate_run_id(index, item)
            config_dict = item.config.model_dump()
            config_dict['run_id'] = run_id

            cursor = conn.execute('SELECT status FROM runs WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO runs (run_id, status, config, metadata)
                    VALUES (?, 'pending', ?, ?)
                    """,
                    (run_id, json.dumps(config_dict), json.dumps(item.metadata)),
                )
                inserted += 1
                continue
            status = row[0]
            if status == 'success':
                skipped_success += 1
                continue

            conn.execute(
                """
                UPDATE runs
                SET config = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (json.dumps(config_dict), json.dumps(item.metadata), run_id),
            )
            refreshed += 1

        conn.execute(
            'INSERT OR REPLACE INTO batch_info (key, value) VALUES (?, ?)',
            ('metadata', json.dumps(batch_meta)),
        )
        conn.execute(
            'INSERT OR REPLACE INTO batch_info (key, value) VALUES (?, ?)',
            ('config', json.dumps(batch_config)),
        )
        conn.execute(
            'INSERT OR REPLACE INTO batch_info (key, value) VALUES (?, ?)',
            ('generator_path', json.dumps(str(generator_path))),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        'DB config refresh complete: refreshed=%d, skipped_success=%d, inserted=%d',
        refreshed,
        skipped_success,
        inserted,
    )
    return refreshed, skipped_success, inserted


# ============================================================================
# Core Runner
# ============================================================================


async def run_command(
    cmd: list[str], env: dict[str, str], cwd: Path, stream_output: bool = False
) -> tuple[int, str, str]:
    """Execute command and return (returncode, stdout, stderr).

    Args:
        cmd: Command to execute
        env: Environment variables
        cwd: Working directory
        stream_output: If True, stream stdout/stderr to console in real-time
    """
    if stream_output:
        async with _managed_subprocess(
            cmd,
            env,
            cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=1024 * 1024,
        ) as proc:
            stdout_chunks = []
            if proc.stdout:
                # Read in chunks instead of lines to avoid LimitOverrunError
                # when a single line is too long (>64KB)
                while True:
                    chunk = await proc.stdout.read(8192)  # Read 8KB at a time
                    if not chunk:
                        break
                    decoded = chunk.decode('utf-8', errors='replace')
                    print(decoded, end='', flush=True)
                    stdout_chunks.append(decoded)

            await proc.wait()
            stdout = ''.join(stdout_chunks)
            return proc.returncode or 0, stdout, ''

    async with _managed_subprocess(
        cmd,
        env,
        cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ) as proc:
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')


@contextlib.asynccontextmanager
async def _managed_subprocess(
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    **kwargs: Any,
) -> AsyncIterator[asyncio.subprocess.Process]:
    """Run a subprocess in its own process group and clean that group on parent exit."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env={**os.environ, **env},
        cwd=cwd,
        start_new_session=True,
        **kwargs,
    )
    watchdog = await _start_parent_death_watchdog(os.getpid(), proc.pid)

    try:
        yield proc
    except BaseException:
        await _terminate_process_group(proc)
        raise
    finally:
        await _stop_parent_death_watchdog(watchdog)


async def _terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate the child process group so nested run-batch processes do not survive."""
    if proc.returncode is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    logger.warning('Terminating child process group %s for pid %s', pgid, proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        logger.warning('Child process group %s did not exit after SIGTERM; sending SIGKILL', pgid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await proc.wait()


async def _start_parent_death_watchdog(parent_pid: int, child_pgid: int) -> asyncio.subprocess.Process | None:
    """Start an external watchdog that kills the child group if this parent is SIGKILLed."""
    if not hasattr(os, 'killpg'):
        return None

    try:
        return await asyncio.create_subprocess_exec(
            'sh',
            '-c',
            _PARENT_DEATH_WATCHDOG_SCRIPT,
            'parent-death-watchdog',
            str(parent_pid),
            str(child_pgid),
            str(PROCESS_TERMINATION_GRACE_SECONDS),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        logger.warning('Failed to start child process watchdog: %s', e)
        return None


async def _stop_parent_death_watchdog(watchdog: asyncio.subprocess.Process | None) -> None:
    if watchdog is None or watchdog.returncode is not None:
        return

    watchdog.terminate()
    try:
        await asyncio.wait_for(watchdog.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        watchdog.kill()
        await watchdog.wait()


async def run_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    env_vars: dict[str, str],
    max_concurrent: int,
    timeout: int | None,
    db_sync_interval: float,
    resume: bool = False,
    refresh_db_config: bool = False,
    rerun_zero_score: bool = False,
    run_index: int | None = None,
    total_runs: int | None = None,
    run_ids: str | None = None,
    log_trim: int | None = None,
    verbose: bool = False,
) -> BenchmarkResult:
    """Execute a single benchmark.

    Args:
        config: Benchmark configuration with name and dataset path.
        output_dir: Directory to store results (database, metrics, trajectory).
        env_vars: Environment variables for the benchmark process.
        max_concurrent: Maximum concurrent tasks within this benchmark.
        timeout: Timeout in seconds per task, or None for no timeout.
        resume: If True, continue from existing database; if False, start fresh.
        refresh_db_config: If True with resume, refresh non-success DB configs and insert missing generated rows.
        rerun_zero_score: If True with resume, also reset score=0 success runs back to pending before running.
        run_index: 1-indexed run number when part of a multi-run set.
        total_runs: Total k in the run set; set when part of a multi-run set.
        run_ids: Optional comma/space separated run IDs filter to pass to generator.

    Returns:
        BenchmarkResult with execution status and metrics.
    """
    start_time = datetime.now(timezone.utc)

    db_path = output_dir / f'search_{config.name}.db'
    metrics_path = output_dir / f'metrics_{config.name}.json'
    use_temp_db = db_sync_interval > 0
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if use_temp_db:
        temp_dir = tempfile.TemporaryDirectory(prefix=f'nexus-benchmark-{config.name}-')
        working_db_path = Path(temp_dir.name) / db_path.name
    else:
        working_db_path = db_path

    run_label = f' (run {run_index}/{total_runs})' if run_index and total_runs else ''
    logger.info('=' * 50)
    logger.info(f'Benchmark: {config.name}{run_label}')

    # Guard: dataset must exist
    if not config.dataset_path.exists():
        logger.error(f'Dataset not found: {config.dataset_path}')
        return BenchmarkResult(
            name=config.name,
            status='failed',
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=f'Dataset not found: {config.dataset_path}',
            run_index=run_index,
            total_runs=total_runs,
        )

    # Prepare output directory
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: copy existing database to temp dir for continuation
    # only when temp staging is enabled. In direct-write mode, the batch runner
    # writes to db_path itself.
    if resume and db_path.exists():
        db_size_kb = db_path.stat().st_size / 1024
        if use_temp_db:
            if safe_copy_sqlite_db(db_path, working_db_path):
                logger.info(f'Resuming from existing database ({db_size_kb:.1f} KB)')
            else:
                logger.warning(f'Failed to restore database, starting fresh (was {db_size_kb:.1f} KB)')
                working_db_path.unlink(missing_ok=True)
        else:
            logger.info(f'Resuming from existing database ({db_size_kb:.1f} KB)')
    else:
        if db_path.exists() and not resume:
            logger.info('Starting fresh run (use --resume to continue from existing database)')
            if not use_temp_db:
                db_path.unlink(missing_ok=True)
        else:
            logger.info('Starting fresh run')

    logger.info(f'Dataset: {config.dataset_path}')
    logger.info(f'Output: {db_path}')

    # Run batch - merge env file vars with current env
    # Priority: config.dataset_path > env file vars > current env SEARCH_*
    trace_file = output_dir / f'trace_{config.name}.jsonl'
    run_env = {
        # Current env SEARCH_* as base (except SEARCH_DATASET_PATH)
        **{k: v for k, v in os.environ.items() if k.startswith('SEARCH_') and k != 'SEARCH_DATASET_PATH'},
        # .env.search file overrides current env
        **env_vars,
        # Dataset path and trace file from output_dir have highest priority
        'SEARCH_DATASET_PATH': str(config.dataset_path),
        'SEARCH_TRACE_FILE': str(trace_file),
    }
    if run_ids is not None:
        run_env['SEARCH_RUN_IDS'] = run_ids

    if resume and refresh_db_config and working_db_path.exists():
        from nexus.runner.batch import BatchConfig

        refresh_resume_db_configs(
            working_db_path,
            DEFAULT_GENERATOR,
            run_env,
            BatchConfig(max_concurrent=max_concurrent, timeout_per_run=timeout, resume=True).model_dump(),
        )
    elif refresh_db_config and not resume:
        logger.warning('--refresh-db-config is only effective together with --resume')

    if resume and rerun_zero_score and working_db_path.exists():
        reset_zero_score_runs(working_db_path)
    elif rerun_zero_score and not resume:
        logger.warning('--rerun-zero-score is only effective together with --resume')

    # Optional temp-db sync: when enabled, run-batch writes to a local temp DB
    # and we snapshot it back to output_dir periodically for progress monitoring.
    stop_sync = asyncio.Event()
    sync_task: asyncio.Task[None] | None = None
    if use_temp_db:
        sync_task = asyncio.create_task(
            sync_db_periodically(working_db_path, db_path, interval=db_sync_interval, stop_event=stop_sync)
        )

    try:
        logger.info('Running batch...')
        cmd = [
            'uv',
            'run',
            'nexus',
            'run-batch',
            '-g',
            str(DEFAULT_GENERATOR),
            '-d',
            str(working_db_path),
            '--max-concurrent',
            str(max_concurrent),
        ]
        if log_trim is not None:
            cmd.extend(['--log-trim', str(log_trim)])
        if verbose:
            cmd.append('-v')
        if timeout is not None:
            cmd.extend(['--timeout', str(timeout)])
        returncode, stdout, stderr = await run_command(
            cmd,
            env=run_env,
            cwd=PROJECT_ROOT,
            stream_output=True,  # Enable real-time output streaming
        )

        if returncode != 0:
            error_output = stderr or stdout
            # Item-level failures make run-batch exit non-zero, but the DB may
            # still contain enough completed rows to compute benchmark metrics.
            logger.error(f'Batch command exited non-zero: {error_output[-500:]}')
            if not working_db_path.exists():
                logger.error('No batch database was created; cannot recover metrics')
                return BenchmarkResult(
                    name=config.name,
                    status='failed',
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    db_path=db_path,
                    error=f'Exit code {returncode}: {error_output[-1000:]}',
                    run_index=run_index,
                    total_runs=total_runs,
                )
            logger.warning('Attempting to compute metrics from partial batch database')

        # Export trajectory directly to target file
        traj_path = output_dir / f'traj_{config.name}.jsonl'
        logger.info(f'Exporting trajectory to {traj_path.name}...')
        try:
            from nexus.runner.batch import BatchDatabase

            async with BatchDatabase(working_db_path) as batch_db:
                runs = await batch_db.get_all_runs()
                with open(traj_path, 'w', encoding='utf-8') as f:
                    for run in runs:
                        task_output = run.get('task_output')
                        record = {
                            'run_id': run['run_id'],
                            'status': run['status'],
                            'start_time': run.get('start_time'),
                            'end_time': run.get('end_time'),
                            'retry_count': run.get('retry_count', 0),
                            'error': run.get('error'),
                            'task_output': task_output,
                            'agent_output': run.get('agent_output'),
                            'processor_results': run.get('processor_results'),
                            'metadata': run.get('metadata'),
                        }
                        f.write(json.dumps(record) + '\n')
                logger.info(f'Exported {len(runs)} runs to {traj_path.name}')
        except Exception as e:
            logger.warning(f'Export failed: {str(e)[:200]}')
            traj_path = None

        # Compute metrics from DB
        logger.info('Computing metrics...')
        from nexus.extensions.tasks.search.compute_metrics import compute_metrics, load_results_from_db

        results = load_results_from_db(working_db_path, include_agent_output=True)
        metrics = compute_metrics(results)

        # Save metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))

        accuracy = metrics.get('accuracy', 0)
        correct = metrics.get('correct', 0)
        total = metrics.get('total', 0)
        timeout_cnt = metrics.get('timeout', 0)
        timeout_str = f', {timeout_cnt} timeout' if timeout_cnt else ''
        logger.info(f'✓ Done: {correct}/{total} = {accuracy:.1%}{timeout_str}')

        # Persist database after all sqlite work is done locally only when using
        # a temp staging DB. In direct-write mode, working_db_path == db_path.
        if use_temp_db:
            db_path.unlink(missing_ok=True)
            shutil.copy2(working_db_path, db_path)

        result = BenchmarkResult(
            name=config.name,
            status='success',
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            db_path=db_path,
            metrics_path=metrics_path,
            traj_path=traj_path,
            metrics=metrics,
            run_index=run_index,
            total_runs=total_runs,
        )

        # Print detailed metrics immediately after benchmark completion
        print_benchmark_metrics(result)

        return result

    except (KeyboardInterrupt, SystemExit):
        raise  # Don't mask user interrupts or system exits
    except Exception as e:
        logger.exception(f'Error: {e}')
        return BenchmarkResult(
            name=config.name,
            status='failed',
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=str(e),
            run_index=run_index,
            total_runs=total_runs,
        )
    finally:
        # Stop periodic sync and do final sync
        if sync_task is not None:
            stop_sync.set()
            try:
                await asyncio.wait_for(sync_task, timeout=5.0)
            except asyncio.TimeoutError:
                sync_task.cancel()
        if temp_dir is not None:
            temp_dir.cleanup()


async def _run_rollout_group(
    group_index: int,
    k: int,
    benchmarks: list[BenchmarkConfig],
    output_dir: Path,
    env_vars: dict[str, str],
    max_concurrent: int,
    timeout: int | None,
    db_sync_interval: float,
    benchmark_parallel: int,
    resume: bool,
    refresh_db_config: bool,
    rerun_zero_score: bool,
    run_ids: str | None,
    log_trim: int | None,
    verbose: bool,
) -> dict[str, BenchmarkResult]:
    """Run every benchmark once as a single RL rollout group.

    A group is one complete pass over all benchmarks, written to
    ``{output_dir}/run_{group_index}/{benchmark}/``. Benchmarks within the group
    run concurrently bounded by ``benchmark_parallel``. Returns a mapping
    ``benchmark_name -> BenchmarkResult`` for this group.
    """
    sem = asyncio.Semaphore(benchmark_parallel)

    async def _run_one(cfg: BenchmarkConfig) -> BenchmarkResult:
        async with sem:
            run_dir = output_dir / f'run_{group_index}' / cfg.name
            run_dir.mkdir(parents=True, exist_ok=True)
            return await run_benchmark(
                cfg,
                run_dir,
                env_vars,
                max_concurrent,
                timeout,
                db_sync_interval,
                resume,
                refresh_db_config,
                rerun_zero_score=rerun_zero_score,
                run_index=group_index,
                total_runs=k,
                run_ids=run_ids,
                log_trim=log_trim,
                verbose=verbose,
            )

    tasks = [asyncio.create_task(_run_one(cfg)) for cfg in benchmarks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    group_results: dict[str, BenchmarkResult] = {}
    for cfg, result in zip(benchmarks, results, strict=True):
        if isinstance(result, BaseException):
            group_results[cfg.name] = BenchmarkResult(
                name=cfg.name,
                status='failed',
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error=str(result),
                run_index=group_index,
                total_runs=k,
            )
        else:
            group_results[cfg.name] = result
    return group_results


def _aggregate_benchmark_runs(
    config: BenchmarkConfig,
    run_results: list[BenchmarkResult],
    output_dir: Path,
    start_time: datetime,
    k: int,
) -> BenchmarkResult:
    """Aggregate a benchmark's K rollout-group results into avg@k / pass@k metrics.

    Reads each group's per-run database (via ``db_path`` on the individual
    results) and writes aggregated metrics to
    ``{output_dir}/{benchmark}/metrics_{name}.json``.
    """
    from nexus.extensions.tasks.search.compute_metrics import compute_final_multi_run_metrics

    successful = [r for r in run_results if r.status == 'success']
    if not successful:
        return BenchmarkResult(
            name=config.name,
            status='failed',
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            total_runs=k,
            error=f'All {k} runs failed',
        )

    metrics = compute_final_multi_run_metrics(run_results)

    benchmark_output_dir = output_dir / config.name
    agg_metrics_path = benchmark_output_dir / f'metrics_{config.name}.json'
    agg_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    agg_metrics_path.write_text(json.dumps(metrics, indent=2))

    result = BenchmarkResult(
        name=config.name,
        status='success',
        start_time=start_time,
        end_time=datetime.now(timezone.utc),
        metrics_path=agg_metrics_path,
        metrics=metrics,
        total_runs=k,
    )
    print_benchmark_metrics(result)
    return result


async def run_all_benchmarks(
    benchmarks: list[BenchmarkConfig],
    output_dir: Path,
    env_vars: dict[str, str],
    max_concurrent: int,
    timeout: int | None,
    db_sync_interval: float,
    benchmark_parallel: int,
    resume: bool = False,
    refresh_db_config: bool = False,
    rerun_zero_score: bool = False,
    k: int = 1,
    group_parallel: int = 1,
    run_ids: str | None = None,
    log_trim: int | None = None,
    verbose: bool = False,
) -> RunSummary:
    """Run all benchmarks with controlled parallelism.

    When ``k == 1`` each benchmark runs once concurrently (bounded by
    ``benchmark_parallel``). When ``k > 1`` the runner performs K RL rollout
    groups: each group is a full concurrent pass over all benchmarks written to
    ``run_{i}/<benchmark>/``, and per-benchmark results are aggregated into
    avg@k / pass@k metrics after all groups finish.

    ``group_parallel`` bounds how many rollout groups run concurrently (default
    1 = sequential). It is independent of ``benchmark_parallel`` (concurrency of
    benchmarks within a group), so peak concurrent benchmark runs is
    ``group_parallel * benchmark_parallel``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(output_dir=output_dir, k=k)

    logger.info('=' * 50)
    logger.info('Search Benchmark Runner')
    logger.info(f'Mode: {"resume" if resume else "fresh"}')
    logger.info(f'Output: {output_dir}')
    logger.info(f'Benchmarks: {len(benchmarks)}')
    logger.info(f'Concurrency: {max_concurrent} tasks, {benchmark_parallel} parallel')
    logger.info(f'Runs per benchmark: {k}' + (' (group rollout, avg@k / pass@k)' if k > 1 else ''))
    if k > 1 and group_parallel > 1:
        logger.info(f'Group parallelism: up to {group_parallel} rollout group(s) concurrently')
    logger.info(f'Timeout: {timeout}s per task' if timeout else 'Timeout: disabled')
    logger.info(
        f'DB sync: temp DB with snapshots every {int(db_sync_interval)}s'
        if db_sync_interval > 0
        else 'DB sync: direct write to output_dir (no temp staging)'
    )
    logger.info('=' * 50)

    # Group-rollout mode (k > 1): K complete passes over all benchmarks, one per
    # group. Handled first so the single-run path below stays at its original
    # indentation (k == 1 is left untouched).
    if k > 1:
        group_start_time = datetime.now(timezone.utc)
        results_by_benchmark: dict[str, list[BenchmarkResult]] = {b.name: [] for b in benchmarks}

        # A semaphore bounds concurrent groups; group_parallel == 1 makes this
        # sequential. gather preserves input order, so per-run breakdowns stay
        # deterministic regardless of completion order.
        group_sem = asyncio.Semaphore(group_parallel)

        async def _group(i: int) -> dict[str, BenchmarkResult]:
            async with group_sem:
                logger.info(f'Starting rollout group {i}/{k} over {len(benchmarks)} benchmark(s)')
                return await _run_rollout_group(
                    i,
                    k,
                    benchmarks,
                    output_dir,
                    env_vars,
                    max_concurrent,
                    timeout,
                    db_sync_interval,
                    benchmark_parallel,
                    resume,
                    refresh_db_config,
                    rerun_zero_score,
                    run_ids,
                    log_trim,
                    verbose,
                )

        group_dicts = await asyncio.gather(*[_group(i) for i in range(1, k + 1)])

        for group_results in group_dicts:
            for name, res in group_results.items():
                results_by_benchmark[name].append(res)

        for b in benchmarks:
            summary.results.append(
                _aggregate_benchmark_runs(
                    b,
                    results_by_benchmark[b.name],
                    output_dir,
                    group_start_time,
                    k,
                )
            )

        summary.end_time = datetime.now(timezone.utc)
        return summary

    # Single-run mode (k == 1): one concurrent pass over all benchmarks.
    sem = asyncio.Semaphore(benchmark_parallel)

    async def _run(cfg: BenchmarkConfig) -> BenchmarkResult:
        async with sem:
            return await run_benchmark(
                cfg,
                output_dir,
                env_vars,
                max_concurrent,
                timeout,
                db_sync_interval,
                resume,
                refresh_db_config,
                rerun_zero_score=rerun_zero_score,
                run_ids=run_ids,
                log_trim=log_trim,
                verbose=verbose,
            )

    tasks = [asyncio.create_task(_run(cfg)) for cfg in benchmarks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            summary.results.append(
                BenchmarkResult(
                    name=benchmarks[i].name,
                    status='failed',
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    error=str(result),
                )
            )
        else:
            summary.results.append(result)

    summary.end_time = datetime.now(timezone.utc)
    return summary


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run search benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
        '  python run_benchmarks.py                                    # Fresh run (built-in benchmarks)\n'
        '  python run_benchmarks.py -c search_config.yaml              # YAML config\n'
        '  python run_benchmarks.py -c search_config.yaml -b seal_0    # Subset of benchmarks\n'
        '  python run_benchmarks.py -r                                 # Resume\n'
        '  python run_benchmarks.py -n exp1                            # Named experiment\n'
        '  python run_benchmarks.py --model gpt-4o                     # Override model\n'
        '  python run_benchmarks.py -k 3                               # 3 RL rollout groups (avg@3 / pass@3)\n'
        '  python run_benchmarks.py --limit 100                        # Limit items\n',
    )
    parser.add_argument('-b', '--benchmarks', nargs='+', help='Benchmarks to run (subset of config)')
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        help='Path to unified YAML config (benchmarks, prompt, toolset, agent, tool_configs). '
        'See search_config.yaml.example for the full schema.',
    )
    parser.add_argument('-n', '--name', type=str, help='Experiment name (fixed folder: benchmark-run-{name})')
    parser.add_argument('-o', '--output-dir', type=Path, help='Output directory (overrides --name)')
    parser.add_argument('-m', '--max-concurrent', type=int, default=500, help='Tasks per benchmark')
    parser.add_argument(
        '-t',
        '--timeout',
        type=int,
        default=None,
        help=f'Timeout in seconds per task (default: {DEFAULT_TIMEOUT_SECONDS} when env is unset)',
    )
    parser.add_argument('-p', '--benchmark-parallel', type=int, default=1, help='Parallel benchmarks')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume incomplete tasks from existing database')
    parser.add_argument(
        '--refresh-db-config',
        action='store_true',
        help=(
            'When used with --resume, regenerate configs from the current YAML/generator and update '
            'non-success rows in the existing DB before running; missing generated rows are inserted.'
        ),
    )
    parser.add_argument(
        '--rerun-zero-score',
        action='store_true',
        help=(
            'When used with --resume, reset score=0 success runs back to pending so they re-execute. '
            'score=1 runs are left untouched.'
        ),
    )
    parser.add_argument('-e', '--env-file', type=Path, default=DEFAULT_ENV_FILE, help='Env file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without running')
    parser.add_argument(
        '-k',
        '--runs',
        type=int,
        default=1,
        metavar='K',
        help=(
            'Number of RL rollout groups; each group is one full pass over all benchmarks written to '
            'run_{i}/<benchmark>/, aggregated into avg@k / pass@k metrics per benchmark (default: 1)'
        ),
    )
    parser.add_argument(
        '--group-parallel',
        type=int,
        default=1,
        metavar='G',
        help=(
            'Number of rollout groups (-k) to run concurrently (default: 1 = sequential). '
            'Peak concurrent benchmark runs is group-parallel x benchmark-parallel, each using '
            '--max-concurrent instances; raise with care to avoid exhausting sandbox quota.'
        ),
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override SEARCH_OPENAI_MODEL; takes highest priority over env file and environment variables',
    )
    parser.add_argument(
        '--llm-max-retries',
        type=int,
        default=None,
        help='Override SEARCH_OPENAI_MAX_RETRIES; set to 1 to disable client retries',
    )
    parser.add_argument(
        '--agent-max-llm-retries',
        type=int,
        default=None,
        help='Override SEARCH_AGENT_MAX_LLM_RETRIES; max agent-level LLM retries on bad output (default: 0)',
    )
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=('exact_match', 'llm_judge', 'reward_service', 'wide_search'),
        default=None,
        help='Override SEARCH_EVAL_MODE; takes highest priority over env file and environment variables',
    )
    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Disable rich table output; use plain text summary instead',
    )
    parser.add_argument(
        '--instance-ids',
        dest='instance_ids',
        type=str,
        default=None,
        help='Comma/space separated dataset instance_ids to run (supports error IDs list)',
    )
    parser.add_argument(
        '--run-ids',
        dest='run_ids',
        type=str,
        default=None,
        help='Comma/space separated batch run_ids to run (sets SEARCH_RUN_IDS)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of dataset items per benchmark run (sets SEARCH_LIMIT)',
    )
    parser.add_argument(
        '--db-sync-interval',
        type=float,
        default=DEFAULT_DB_SYNC_INTERVAL,
        help=(
            'Interval in seconds for syncing a temp SQLite DB to the output directory. '
            'Default: 0, which writes the batch DB directly to output_dir without using /tmp. '
            'Set to > 0 to use a temp DB plus periodic sync snapshots.'
        ),
    )
    parser.add_argument(
        '--log-trim',
        type=int,
        help='Optional value forwarded as --log-trim to `nexus run-batch`.',
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    # Load environment
    try:
        env_vars = load_env_file(args.env_file)
        logger.info(f'Loaded env from: {args.env_file}')
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Apply model override with highest priority (overrides both env file and os.environ)
    if args.model:
        env_vars['SEARCH_OPENAI_MODEL'] = args.model
        logger.info(f'Model override: SEARCH_OPENAI_MODEL={args.model}')
    if args.llm_max_retries is not None:
        env_vars['SEARCH_OPENAI_MAX_RETRIES'] = str(args.llm_max_retries)
        logger.info(f'LLM retry override: SEARCH_OPENAI_MAX_RETRIES={args.llm_max_retries}')
    if args.agent_max_llm_retries is not None:
        env_vars['SEARCH_AGENT_MAX_LLM_RETRIES'] = str(args.agent_max_llm_retries)
        logger.info(f'Agent retry override: SEARCH_AGENT_MAX_LLM_RETRIES={args.agent_max_llm_retries}')
    if args.eval_mode:
        env_vars['SEARCH_EVAL_MODE'] = args.eval_mode
        logger.info(f'Eval mode override: SEARCH_EVAL_MODE={args.eval_mode}')
    if args.instance_ids:
        env_vars['SEARCH_INSTANCE_IDS'] = args.instance_ids
        logger.info('Instance ID filter enabled via --instance-ids')
    if args.run_ids:
        env_vars['SEARCH_RUN_IDS'] = args.run_ids
        logger.info('Run ID filter enabled via --run-ids')
    if args.limit is not None:
        if args.limit < 0:
            logger.error(f'--limit must be >= 0, got {args.limit}')
            return 1
        env_vars['SEARCH_LIMIT'] = str(args.limit)
        logger.info(f'Item limit enabled via --limit: {args.limit}')
    if args.db_sync_interval < 0:
        logger.error(f'--db-sync-interval must be >= 0, got {args.db_sync_interval}')
        return 1
    if args.refresh_db_config and not args.resume:
        logger.warning('--refresh-db-config is only effective together with --resume')
    if args.rerun_zero_score and not args.resume:
        logger.warning('--rerun-zero-score is only effective together with --resume')

    # Load unified YAML config and pass path to generator subprocess via env var
    yaml_config = None
    if args.config:
        if not args.config.is_file():
            logger.error(f'Config not found: {args.config}')
            return 1
        env_vars['SEARCH_CONFIG_PATH'] = str(args.config.resolve())
        logger.info(f'Config: {args.config}')
        from nexus.extensions.tasks.search.config import load_search_config

        yaml_config = load_search_config(args.config)

    # Validate toolset version against the hardcoded registry. YAML-defined custom
    # tool lists now live under agent_config.tools and bypass the toolset_version
    # lookup entirely, so there's nothing extra to merge in here.
    valid_toolset_versions = set(TOOLSET_REQUIRED_TOOLS)
    toolset_version = env_vars.get('SEARCH_TOOLSET_VERSION')
    if toolset_version is not None and toolset_version not in valid_toolset_versions:
        logger.error(
            f'Unknown SEARCH_TOOLSET_VERSION: {toolset_version!r}. Available: {sorted(valid_toolset_versions)}'
        )
        return 1

    # Validate k
    k = args.runs
    if k < 1:
        logger.error(f'--runs must be >= 1, got {k}')
        return 1

    # Validate group parallelism (only meaningful for group rollout, i.e. k > 1)
    group_parallel = args.group_parallel
    if group_parallel < 1:
        logger.error(f'--group-parallel must be >= 1, got {group_parallel}')
        return 1
    if group_parallel > 1 and k <= 1:
        logger.warning('--group-parallel is only effective with -k/--runs > 1; ignoring for single-run mode')

    # Load benchmarks from config or built-in defaults
    try:
        all_benchmarks = load_benchmarks(search_config=yaml_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f'Config error: {e}')
        return 1

    # Filter by name if specified
    if args.benchmarks:
        names = set(args.benchmarks)
        available = {b.name for b in all_benchmarks}
        if unknown := names - available:
            logger.error(f'Unknown: {unknown}. Available: {sorted(available)}')
            return 1
        all_benchmarks = [b for b in all_benchmarks if b.name in names]

    # Filter missing datasets
    all_benchmarks = [
        b
        for b in all_benchmarks
        if b.dataset_path.exists() or (logger.warning(f'Missing: {b.name} -> {b.dataset_path}') or False)
    ]

    if not all_benchmarks:
        logger.error('No valid benchmarks')
        return 1

    # Determine output directory with clear naming strategy:
    # --output-dir > --name > auto-generated.
    # YAML workspace_path changes the root used by --name and auto-generated paths.
    model_name = env_vars.get('SEARCH_OPENAI_MODEL')
    if model_name is None and yaml_config is not None:
        request_args = yaml_config.openai_clients.get('main', {}).get('config', {}).get('request_args', {})
        model_name = request_args.get('model')
    model_name = (model_name or 'unknown').replace('/', '_')
    workspace_path = (
        yaml_config.workspace_path
        if yaml_config is not None and yaml_config.workspace_path is not None
        else DEFAULT_WORKSPACE
    )

    if args.output_dir:
        output_dir = args.output_dir
    elif args.name:
        output_dir = workspace_path / f'benchmark-run-{args.name}'
    elif args.resume:
        # Resume: use fixed name to find existing data
        output_dir = workspace_path / f'benchmark-run-{model_name}'
    else:
        # Fresh run: create timestamped folder to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = workspace_path / f'benchmark-run-{model_name}-{timestamp}'

    # Guard: --resume requires existing directory (fail fast)
    if args.resume and not output_dir.exists():
        logger.error(
            f'--resume requires existing directory, but not found: {output_dir}\n'
            f'  Hint: Use -n NAME to specify experiment, or run without -r first.'
        )
        return 1

    # Dry run
    if args.dry_run:
        mode = 'resume' if args.resume else 'fresh'
        k_label = f' (group rollout x{k}, avg@{k}/pass@{k})' if k > 1 else ''
        model_label = f' [model: {args.model}]' if args.model else ''
        refresh_label = ' [refresh non-success DB configs]' if args.resume and args.refresh_db_config else ''
        print(f'\nDRY RUN ({mode}){k_label}{model_label}{refresh_label}\nOutput: {output_dir}\nBenchmarks:')
        for b in all_benchmarks:
            if k > 1:
                db_exists = any(
                    (output_dir / f'run_{i}' / b.name / f'search_{b.name}.db').exists() for i in range(1, k + 1)
                )
            else:
                db_exists = (output_dir / f'search_{b.name}.db').exists()
            status = '(will resume)' if db_exists and args.resume else ''
            print(f'  - {b.name}: {b.dataset_path} {status}')
        return 0

    # Resolve timeout priority: explicit CLI > env file / env var > module default.
    env_timeout = env_vars.get('SEARCH_TIMEOUT_PER_RUN') or os.getenv('SEARCH_TIMEOUT_PER_RUN')
    timeout = args.timeout if args.timeout is not None else int(env_timeout) if env_timeout else DEFAULT_TIMEOUT_SECONDS

    execution_metadata = build_execution_metadata(args, env_vars, all_benchmarks, timeout, output_dir)

    # Run benchmarks
    summary = await run_all_benchmarks(
        all_benchmarks,
        output_dir,
        env_vars,
        args.max_concurrent,
        timeout,
        args.db_sync_interval,
        args.benchmark_parallel,
        args.resume,
        args.refresh_db_config,
        rerun_zero_score=args.rerun_zero_score,
        k=k,
        group_parallel=group_parallel,
        run_ids=args.run_ids,
        log_trim=args.log_trim,
        verbose=args.verbose,
    )

    # Output: rich table (default) or plain text (--no-table)
    if args.no_table:
        print_summary(summary)
    else:
        print_rich_summary(summary)

    if path := save_summary(summary, execution_metadata):
        logger.info(f'Summary: {path}')
    if html_path := save_summary_html(summary, execution_metadata):
        logger.info(f'HTML: {html_path}')

    return 1 if any(r.status == 'failed' for r in summary.results) else 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info('Interrupted')
        return 130


if __name__ == '__main__':
    sys.exit(main())
