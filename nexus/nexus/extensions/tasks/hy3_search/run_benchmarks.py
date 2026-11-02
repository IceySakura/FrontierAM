#!/usr/bin/env python3
"""HY3 Search Task Benchmark Runner.

Run multiple HY3 search benchmarks concurrently with result aggregation.

Usage:
    python run_benchmarks.py                        # Fresh run (new timestamped folder)
    python run_benchmarks.py -r                     # Resume from existing database
    python run_benchmarks.py -n exp1                # Named experiment (fixed folder)
    python run_benchmarks.py -n exp1 -r             # Resume named experiment
    python run_benchmarks.py -b sealqa gaia         # Run specific benchmarks
    python run_benchmarks.py --model gpt-4o         # Override model (highest priority)
    python run_benchmarks.py -k 3                   # Run each benchmark 3 times (avg@3)
    python run_benchmarks.py -k 3 -b sealqa         # avg@3 on sealqa only
    python run_benchmarks.py --limit 100            # Limit dataset items per benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.hy3_search.reporting import (
    build_execution_metadata,
    print_benchmark_metrics,
    print_rich_summary,
    print_summary,
    save_summary,
    save_summary_excel,
)


# ============================================================================
# Constants
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_WORKSPACE = PROJECT_ROOT / 'workspace'
DEFAULT_ENV_FILE = PROJECT_ROOT / '.env.hy3_search'
DEFAULT_GENERATOR = SCRIPT_DIR / 'generator.py'

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        'SEARCH_BENCHMARK_DATA_DIR',
        '/data/workspace/dataset',
    )
)
DEFAULT_BENCHMARKS = {
    'seal_0': 'sealqa_seal_0.parquet',
    'sealqa_seal_0_260320': 'sealqa_seal_0_260320.parquet',
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
    'hy_seal_0': 'sealqa_seal_0_hy_response_api.parquet',
    'hy_browse_comp_zh': 'browse_comp_zh.parquet',
    'yuanbao': 'yuanbao_seal_0.parquet',
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
    """Load environment variables from .env file (excluding SEARCH_DATASET_PATH)."""
    if not env_file.exists():
        raise FileNotFoundError(
            f'Environment file not found: {env_file}\nPlease create it from .env.hy3_search.template'
        )

    env_vars = _parse_env_file(env_file)
    env_vars.pop('SEARCH_DATASET_PATH', None)
    return {k: v for k, v in env_vars.items() if v}


def _parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse .env file and return environment variables dict."""
    env_vars = {}
    with open(env_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            # Strip quotes
            if len(value) >= 2 and value[0] in '"\'' and value[-1] == value[0]:
                value = value[1:-1]
            env_vars[key] = value
    return env_vars


# ============================================================================
# Benchmark Configuration Loading
# ============================================================================


def load_benchmarks(config_path: Path | None) -> list[BenchmarkConfig]:
    """Load benchmark configs from YAML file or use defaults."""
    if config_path is None:
        return [
            BenchmarkConfig(name=name, dataset_path=DEFAULT_DATA_DIR / filename)
            for name, filename in DEFAULT_BENCHMARKS.items()
        ]

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or 'benchmarks' not in data:
        raise ValueError("Config must contain 'benchmarks' key")

    return [
        BenchmarkConfig(name=name, dataset_path=Path(value['dataset_path'] if isinstance(value, dict) else value))
        for name, value in data['benchmarks'].items()
    ]


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
        # Stream output in real-time
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr to stdout
            env={**os.environ, **env},
            cwd=cwd,
            limit=1024 * 1024,  # 1MB
        )

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
    # Original behavior: capture all output
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env},
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')


async def run_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    env_vars: dict[str, str],
    max_concurrent: int,
    timeout: int | None,
    resume: bool = False,
    run_index: int | None = None,
    total_runs: int | None = None,
    run_ids: str | None = None,
) -> BenchmarkResult:
    """Execute a single benchmark.

    Args:
        config: Benchmark configuration with name and dataset path.
        output_dir: Directory to store results (database, metrics, trajectory).
        env_vars: Environment variables for the benchmark process.
        max_concurrent: Maximum concurrent tasks within this benchmark.
        timeout: Timeout in seconds per task, or None for no timeout.
        resume: If True, continue from existing database; if False, start fresh.
        run_index: 1-indexed run number when part of a multi-run set.
        total_runs: Total k in the run set; set when part of a multi-run set.
        run_ids: Optional comma/space separated run IDs filter to pass to generator.

    Returns:
        BenchmarkResult with execution status and metrics.
    """
    start_time = datetime.now(timezone.utc)

    # Work directly in output directory
    db_path = output_dir / f'search_{config.name}.db'
    metrics_path = output_dir / f'metrics_{config.name}.json'
    temp_dir = tempfile.TemporaryDirectory(prefix=f'nexus-benchmark-{config.name}-')
    working_db_path = Path(temp_dir.name) / db_path.name

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
    # Only resume if explicitly requested via --resume flag
    # Use SQLite backup API to ensure database consistency (avoid CephFS corruption)
    if resume and db_path.exists():
        db_size_kb = db_path.stat().st_size / 1024
        if safe_copy_sqlite_db(db_path, working_db_path):
            logger.info(f'Resuming from existing database ({db_size_kb:.1f} KB)')
        else:
            # Backup failed, start fresh
            logger.warning(f'Failed to restore database, starting fresh (was {db_size_kb:.1f} KB)')
            working_db_path.unlink(missing_ok=True)
    else:
        if db_path.exists() and not resume:
            logger.info('Starting fresh run (use --resume to continue from existing database)')
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
        # .env.hy3_search file overrides current env
        **env_vars,
        # Dataset path and trace file from output_dir have highest priority
        'SEARCH_DATASET_PATH': str(config.dataset_path),
        'SEARCH_TRACE_FILE': str(trace_file),
    }
    if run_ids is not None:
        run_env['SEARCH_RUN_IDS'] = run_ids

    # Start periodic sync task for real-time progress monitoring
    stop_sync = asyncio.Event()
    sync_task = asyncio.create_task(sync_db_periodically(working_db_path, db_path, interval=30.0, stop_event=stop_sync))

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
            # Log last 500 chars to see the actual error
            logger.error(f'Failed: {error_output[-500:]}')
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

        # Export trajectory directly to target file
        traj_path = output_dir / f'traj_{config.name}.jsonl'
        logger.info(f'Exporting trajectory to {traj_path.name}...')
        try:
            from nexus.runner.batch import BatchDatabase

            async with BatchDatabase(working_db_path) as batch_db:
                runs = await batch_db.get_all_runs()
                with open(traj_path, 'w', encoding='utf-8') as f:
                    for run in runs:
                        record = {
                            'run_id': run['run_id'],
                            'status': run['status'],
                            'start_time': run.get('start_time'),
                            'end_time': run.get('end_time'),
                            'retry_count': run.get('retry_count', 0),
                            'error': run.get('error'),
                            'task_output': run.get('task_output'),
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
        from nexus.extensions.tasks.hy3_search.compute_metrics import compute_metrics, load_results_from_db

        results = load_results_from_db(working_db_path)
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

        # Persist database to output directory after all sqlite work is done locally
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
        stop_sync.set()
        try:
            await asyncio.wait_for(sync_task, timeout=5.0)
        except asyncio.TimeoutError:
            sync_task.cancel()
        temp_dir.cleanup()


async def run_benchmark_multi(
    config: BenchmarkConfig,
    output_dir: Path,
    env_vars: dict[str, str],
    max_concurrent: int,
    timeout: int | None,
    k: int,
    resume: bool = False,
    summary_output_dir: Path | None = None,
    execution_metadata: dict[str, Any] | None = None,
    run_ids: str | None = None,
) -> BenchmarkResult:
    """Run a benchmark k times and compute avg@k / pass@k metrics.

    Each of the k runs is stored in a separate subdirectory:
        {output_dir}/run_1/, {output_dir}/run_2/, ..., {output_dir}/run_k/

    After all runs, avg@k and pass@k metrics are computed and saved to
        {output_dir}/metrics_{name}.json

    Args:
        config: Benchmark configuration with name and dataset path.
        output_dir: Base directory; per-run subdirectories are created here.
        env_vars: Environment variables for the benchmark process.
        max_concurrent: Maximum concurrent tasks per run.
        timeout: Timeout in seconds per task, or None for no timeout.
        k: Number of independent runs.
        resume: If True, resume each run from its existing database if found.

    Returns:
        Aggregated BenchmarkResult with avg@k metrics.
    """
    start_time = datetime.now(timezone.utc)

    logger.info(f'Starting {k}-run set for benchmark: {config.name}')
    run_results: list[BenchmarkResult] = []

    from nexus.extensions.tasks.hy3_search.compute_metrics import (
        compute_final_multi_run_metrics,
        compute_interim_multi_run_metrics,
    )

    for i in range(1, k + 1):
        run_dir = output_dir / f'run_{i}'
        run_dir.mkdir(parents=True, exist_ok=True)
        result = await run_benchmark(
            config,
            run_dir,
            env_vars,
            max_concurrent,
            timeout,
            resume,
            run_index=i,
            total_runs=k,
            run_ids=run_ids,
        )
        run_results.append(result)

        # Update metrics and summaries after each completed run for real-time visibility.
        interim_metrics = compute_interim_multi_run_metrics(run_results)

        if interim_metrics:
            interim_strict_acc = interim_metrics.get(
                'strict_avg_k_accuracy',
                interim_metrics.get('accuracy', 0),
            )
            print(
                f'\n[Progress] {config.name}: completed run {i}/{k}\n'
                f'  strict_acc={interim_strict_acc:.2%}, '
                f'avg_k_acc={interim_metrics.get("avg_k_accuracy", interim_metrics.get("accuracy", 0)):.2%}, '
                f'pass_k_acc={interim_metrics.get("pass_k_accuracy", 0):.2%}'
            )
            per_run = interim_metrics.get('per_run', [])
            if per_run:
                print('  Per-Run Breakdown:')
                for idx, run_m in enumerate(per_run, 1):
                    run_total = run_m.get('total', run_m.get('evaluated', 0))
                    print(
                        f'    run {idx}: '
                        f'{run_m.get("correct", 0)}/{run_total} '
                        f'= {run_m.get("accuracy", 0):.2%} '
                        f'(timeout={run_m.get("timeout", 0)})'
                    )

            # Save checkpoint summaries after each run.
            if summary_output_dir is not None:
                # Use benchmark-scoped checkpoint dir to avoid cross-benchmark overwrite in parallel mode.
                checkpoint_summary_dir = summary_output_dir / f'_checkpoint_{config.name}'
                checkpoint_summary_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_metrics_path = output_dir / f'metrics_{config.name}.json'
                checkpoint_metrics_path.write_text(json.dumps(interim_metrics, indent=2))
                checkpoint_result = BenchmarkResult(
                    name=config.name,
                    status='success',
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    metrics_path=checkpoint_metrics_path,
                    metrics=interim_metrics,
                    total_runs=k,
                )
                checkpoint_summary = RunSummary(
                    results=[checkpoint_result],
                    output_dir=checkpoint_summary_dir,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    k=k,
                )
                checkpoint_execution = {
                    **(execution_metadata or {}),
                    'progress': {
                        'benchmark': config.name,
                        'completed_runs': i,
                        'total_runs': k,
                    },
                }
                save_summary(checkpoint_summary, checkpoint_execution)
                save_summary_excel(checkpoint_summary, checkpoint_execution)

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

    # Compute avg@k across all successful runs
    metrics = compute_final_multi_run_metrics(run_results)

    # Save aggregated metrics
    agg_metrics_path = output_dir / f'metrics_{config.name}.json'
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
    benchmark_parallel: int,
    resume: bool = False,
    k: int = 1,
    execution_metadata: dict[str, Any] | None = None,
    run_ids: str | None = None,
) -> RunSummary:
    """Run all benchmarks with controlled parallelism."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(output_dir=output_dir, k=k)

    logger.info('=' * 50)
    logger.info('Search Benchmark Runner')
    logger.info(f'Mode: {"resume" if resume else "fresh"}')
    logger.info(f'Output: {output_dir}')
    logger.info(f'Benchmarks: {len(benchmarks)}')
    logger.info(f'Concurrency: {max_concurrent} tasks, {benchmark_parallel} parallel')
    logger.info(f'Runs per benchmark: {k}' + (' (avg@k enabled)' if k > 1 else ''))
    logger.info(f'Timeout: {timeout}s per task' if timeout else 'Timeout: disabled')
    logger.info('=' * 50)

    sem = asyncio.Semaphore(benchmark_parallel)

    async def _run(cfg: BenchmarkConfig) -> BenchmarkResult:
        async with sem:
            if k > 1:
                return await run_benchmark_multi(
                    cfg,
                    output_dir / cfg.name,
                    env_vars,
                    max_concurrent,
                    timeout,
                    k,
                    resume,
                    summary_output_dir=output_dir,
                    execution_metadata=execution_metadata,
                    run_ids=run_ids,
                )
            return await run_benchmark(
                cfg,
                output_dir,
                env_vars,
                max_concurrent,
                timeout,
                resume,
                run_ids=run_ids,
            )

    results = await asyncio.gather(*[_run(cfg) for cfg in benchmarks], return_exceptions=True)

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
        '  python run_benchmarks.py                        # Fresh run\n'
        '  python run_benchmarks.py -r                     # Resume\n'
        '  python run_benchmarks.py -n exp1                # Named experiment\n'
        '  python run_benchmarks.py -n exp1 -r             # Resume named\n'
        '  python run_benchmarks.py -b sealqa gaia         # Specific benchmarks\n'
        '  python run_benchmarks.py --model gpt-4o         # Override model\n'
        '  python run_benchmarks.py -k 3                   # avg@3 (3 runs each)\n'
        '  python run_benchmarks.py -k 3 --model gpt-4o   # avg@3 with model override\n'
        '  python run_benchmarks.py --limit 100            # Limit dataset items per benchmark\n',
    )
    parser.add_argument('-b', '--benchmarks', nargs='+', help='Benchmarks to run')
    parser.add_argument('-c', '--config', type=Path, help='YAML config file')
    parser.add_argument('-n', '--name', type=str, help='Experiment name (fixed folder: benchmark-run-{name})')
    parser.add_argument('-o', '--output-dir', type=Path, help='Output directory (overrides --name)')
    parser.add_argument('-m', '--max-concurrent', type=int, default=500, help='Tasks per benchmark')
    parser.add_argument('-t', '--timeout', type=int, default=2400, help='Timeout in seconds per task')
    parser.add_argument('-p', '--benchmark-parallel', type=int, default=1, help='Parallel benchmarks')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume incomplete tasks from existing database')
    parser.add_argument('-e', '--env-file', type=Path, default=DEFAULT_ENV_FILE, help='Env file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without running')
    parser.add_argument(
        '-k',
        '--runs',
        type=int,
        default=1,
        metavar='K',
        help='Number of independent runs per benchmark; enables avg@k / pass@k metrics (default: 1)',
    )
    parser.add_argument(
        '--agent-max-retries',
        dest='agent_max_retries',
        type=int,
        default=None,
        help='Override SEARCH_AGENT_MAX_RETRIES; max agent-level retries on bad output (default: 0)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override SEARCH_OPENAI_MODEL; takes highest priority over env file and environment variables',
    )
    parser.add_argument(
        '-em',
        '--eval-mode',
        type=str,
        required=True,
        choices=('exact_match', 'llm_judge', 'reward_service', 'wide_search'),
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
    if args.agent_max_retries is not None:
        env_vars['SEARCH_AGENT_MAX_RETRIES'] = str(args.agent_max_retries)
        logger.info(f'Agent retry override: SEARCH_AGENT_MAX_RETRIES={args.agent_max_retries}')
    env_vars['SEARCH_EVAL_MODE'] = args.eval_mode
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

    # Validate k
    k = args.runs
    if k < 1:
        logger.error(f'--runs must be >= 1, got {k}')
        return 1

    # Load and filter benchmarks
    try:
        all_benchmarks = load_benchmarks(args.config)
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
    # --output-dir > --name > auto-generated (timestamped for fresh, fixed for resume)
    model_name = env_vars.get('SEARCH_OPENAI_MODEL', 'unknown').replace('/', '_')

    if args.output_dir:
        output_dir = args.output_dir
    elif args.name:
        output_dir = DEFAULT_WORKSPACE / f'benchmark-run-{args.name}'
    elif args.resume:
        # Resume: use fixed name to find existing data
        output_dir = DEFAULT_WORKSPACE / f'benchmark-run-{model_name}'
    else:
        # Fresh run: create timestamped folder to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = DEFAULT_WORKSPACE / f'benchmark-run-{model_name}-{timestamp}'

    # Guard: --resume requires existing directory (fail fast)
    if args.resume and not output_dir.exists():
        logger.error(
            f'--resume requires existing directory, but not found: {output_dir}\n'
            f'  Hint: Use -n NAME to specify experiment, or run without -r first.'
        )
        return 1

    # Dry run
    #  python nexus/extensions/tasks/hy3_search/run_benchmarks.py -m 32 -p 1 -b browse_comp_zh -k 1 -r -o /data/workspace/nexus/workspace/benchmark-run-HY3.0-A3B_MoE_SFT_256K_BUS_0212_HF-20260326_092922  > browse_comp_zh.log
    if args.dry_run:
        logger.info('start resume')
        mode = 'resume' if args.resume else 'fresh'
        k_label = f' (avg@{k})' if k > 1 else ''
        model_label = f' [model: {args.model}]' if args.model else ''
        print(f'\nDRY RUN ({mode}){k_label}{model_label}\nOutput: {output_dir}\nBenchmarks:')
        for b in all_benchmarks:
            if k > 1:
                db_exists = any(
                    (output_dir / b.name / f'run_{i}' / f'search_{b.name}.db').exists() for i in range(1, k + 1)
                )
            else:
                db_exists = (output_dir / f'search_{b.name}.db').exists()
            status = '(will resume)' if db_exists and args.resume else ''
            print(f'  - {b.name}: {b.dataset_path} {status}')
        return 0

    # Resolve timeout: CLI > env var > None
    timeout = args.timeout
    if timeout is None:
        env_timeout = os.getenv('SEARCH_TIMEOUT_PER_RUN')
        if env_timeout:
            timeout = int(env_timeout)

    execution_metadata = build_execution_metadata(args, env_vars, all_benchmarks, timeout, output_dir)

    # Run benchmarks
    summary = await run_all_benchmarks(
        all_benchmarks,
        output_dir,
        env_vars,
        args.max_concurrent,
        timeout,
        args.benchmark_parallel,
        args.resume,
        k=k,
        execution_metadata=execution_metadata,
        run_ids=args.run_ids,
    )

    # Output: rich table (default) or plain text (--no-table)
    if args.no_table:
        print_summary(summary)
    else:
        print_rich_summary(summary)

    if path := save_summary(summary, execution_metadata):
        logger.info(f'Summary: {path}')
    if excel_path := save_summary_excel(summary, execution_metadata):
        logger.info(f'Excel: {excel_path}')

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
