#!/usr/bin/env python3
"""Stress test runner and generator for sandbox platform benchmarking.

This script serves two purposes:
1. As a CLI tool to run stress tests with configurable parameters
2. As a generator module for nexus run-batch to generate BatchItems

Usage as CLI:
    # Run with default settings (10000 total sandboxes)
    python tests/stress/swe_sandbox/run_benchmark.py

    # Specify instance IDs
    python tests/stress/swe_sandbox/run_benchmark.py --instance-ids "psf__requests-2317,django__django-10097"

    # Specify total count per round (default: 10000)
    python tests/stress/swe_sandbox/run_benchmark.py --total-count-per-round 5000

    # Use production environment
    python tests/stress/swe_sandbox/run_benchmark.py --env prod

    # Multi-round stress test (each round: 10k sandboxes, 30min duration, then destroy)
    python tests/stress/swe_sandbox/run_benchmark.py --rounds 5 --round-duration 1800

    # Infinite rounds (run until interrupted)
    python tests/stress/swe_sandbox/run_benchmark.py --rounds 0

    # Generate statistics report
    python tests/stress/swe_sandbox/run_benchmark.py --generate-report

    # Multi-process mode (500 tasks per process)
    python tests/stress/swe_sandbox/run_benchmark.py --process-count 20 --tasks-per-process 500

Usage as generator (called by nexus run-batch):
    STRESS_INSTANCE_IDS="psf__requests-2317,django__django-10097" \
    STRESS_REPEAT_COUNT=2000 \
    STRESS_ROUND_PREFIX="round001" \
    SWE_DATASET_PATH=./workspace/swe-bench.jsonl \
    SWE_RUNTIME_PROVIDER_NAME=gongfeng \
    SWE_RUNTIME_BASE_URL='http://hyrl-sandbox.test.woa.com:8052' \
uv run nexus run-batch \
        -g ./tests/stress/swe_sandbox/run_benchmark.py \
        -d ./workspace/benchmark.db \
        --max-concurrent 10000 \
        --no-resume

Environment variables (for generator mode):
    - STRESS_INSTANCE_IDS: comma-separated list of instance_ids to stress test (required)
    - STRESS_REPEAT_COUNT: number of times to repeat each instance (default: 2000)
    - STRESS_ROUND_PREFIX: round prefix for run_id (e.g., 'round001', default: empty)
    - SWE_DATASET_PATH: path to JSONL dataset (default: ./workspace/swe-bench.jsonl)
    - SWE_RUN_ORACLE: "1"/"true" to use golden patch without agent (default: true for stress test)
    - SWE_RUNTIME_BASE_URL: runtime base URL (default: http://hyrl-sandbox.prod.woa.com:8052)
    - SWE_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: gongfeng)
    - SWE_RUNTIME_TAG: runtime tag to use (default: latest)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_BENCH, SWEbenchInstance
from nexus.runner.single import RunInput


# Environment configurations
ENV_CONFIGS = {
    'test': 'http://hyrl-sandbox.test.woa.com:8052',
    'prod-x3': 'http://hyrl-sandbox.prod.woa.com:8052',
    'prod-x8': 'http://hunyuan.sandbox.production.polaris:8052',
}

DEFAULT_TOTAL_COUNT_PER_ROUND = 10000
DEFAULT_DATASET_PATH = './workspace/swe-bench.jsonl'
DEFAULT_REPEAT_COUNT = 2000
DEFAULT_SWE_RUNTIME_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'

# Multi-round defaults
DEFAULT_ROUNDS = 1  # Single round by default, 0 means infinite
DEFAULT_ROUND_DURATION = 1800  # 30 minutes per round
DEFAULT_ROUND_COOLDOWN = 5  # 5 seconds between rounds

# Multi-process defaults
DEFAULT_TASKS_PER_PROCESS = 500  # Number of tasks per process

# Statistics configuration


# Override resource_spec for specific instances
INSTANCE_RESOURCE_OVERRIDE = {
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},
}

# Global flag for graceful shutdown
_shutdown_requested = False


def log_message(message: str, flush: bool = True) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {message}', flush=flush)


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f'\n[Signal {signum}] Shutdown requested, will exit after current round...')


# =============================================================================
# Statistics reporting from database
# =============================================================================


def generate_report_from_directory(work_dir: str) -> dict[str, Any]:
    """Generate comprehensive statistics report from existing stress test directory.

    This function reads all .db files in the specified directory and aggregates
    statistics across all processes.

    Args:
        work_dir: Path to the stress test directory containing .db files

    Returns:
        Dictionary containing aggregated report data
    """
    import glob
    import sqlite3

    work_path = Path(work_dir)
    if not work_path.exists():
        print(f'Error: Directory not found: {work_dir}')
        return {'error': f'Directory not found: {work_dir}'}

    # Find all .db files (excluding -shm and -wal files)
    db_files = sorted(glob.glob(str(work_path / '*.db')))
    db_files = [f for f in db_files if not f.endswith('-shm') and not f.endswith('-wal')]

    if not db_files:
        print(f'Error: No .db files found in {work_dir}')
        return {'error': f'No .db files found in {work_dir}'}

    print(f'[Report] Found {len(db_files)} database files in {work_dir}')

    # Aggregate statistics from all databases
    all_runs = []
    status_counts = defaultdict(int)
    instance_stats = defaultdict(list)
    durations = []

    for db_path in db_files:
        db_name = Path(db_path).name
        print(f'[Report] Processing {db_name}...')

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query all runs
            cursor.execute("""
                SELECT run_id, status, start_time, end_time, metadata, error
                FROM runs
            """)

            for row in cursor.fetchall():
                run_data = dict(row)
                run_data['source_db'] = db_name
                all_runs.append(run_data)

                status = run_data['status']
                status_counts[status] += 1

                # Parse instance_id from metadata
                metadata = run_data.get('metadata')
                if metadata:
                    try:
                        meta_dict = json.loads(metadata)
                        instance_id = meta_dict.get('instance_id', 'unknown')
                    except json.JSONDecodeError:
                        instance_id = 'unknown'
                else:
                    # Try to extract from run_id
                    run_id = run_data['run_id']
                    parts = run_id.rsplit('__', 1)
                    instance_id = parts[0] if len(parts) > 1 else run_id

                # Calculate duration if both times are present
                start_time = run_data.get('start_time')
                end_time = run_data.get('end_time')
                if start_time and end_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        duration = (end_dt - start_dt).total_seconds()
                        if duration >= 0:
                            durations.append(duration)
                            instance_stats[instance_id].append({
                                'duration': duration,
                                'status': status,
                            })
                    except (ValueError, TypeError):
                        pass

            conn.close()

        except sqlite3.Error as e:
            print(f'[Report] Warning: Error reading {db_name}: {e}')
            continue

    # Calculate statistics
    total_runs = len(all_runs)
    print(f'[Report] Total runs: {total_runs}')

    # Build report
    report = {
        'summary': {
            'total_runs': total_runs,
            'source_directory': str(work_dir),
            'database_files': len(db_files),
            'status_breakdown': dict(status_counts),
        },
        'timing': {},
        'instance_performance': {},
        'time_distribution': {},
    }

    # Calculate success rate
    success_count = status_counts.get('success', 0)
    failed_count = status_counts.get('failed', 0)
    timeout_count = status_counts.get('timeout', 0)
    running_count = status_counts.get('running', 0)
    pending_count = status_counts.get('pending', 0)

    completed_count = success_count + failed_count + timeout_count
    report['summary']['completed_runs'] = completed_count
    report['summary']['success_count'] = success_count
    report['summary']['failed_count'] = failed_count
    report['summary']['timeout_count'] = timeout_count
    report['summary']['running_count'] = running_count
    report['summary']['pending_count'] = pending_count
    report['summary']['success_rate'] = success_count / completed_count if completed_count > 0 else 0

    # Timing statistics
    if durations:
        report['timing'] = {
            'average_duration': statistics.mean(durations),
            'median_duration': statistics.median(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'std_dev_duration': statistics.stdev(durations) if len(durations) >= 2 else 0,
            'total_durations_collected': len(durations),
        }

        # Percentiles
        if len(durations) >= 2:
            quantiles = statistics.quantiles(durations, n=100)
            report['timing']['percentiles'] = {
                'p50': quantiles[49] if len(quantiles) > 49 else 0,
                'p90': quantiles[89] if len(quantiles) > 89 else 0,
                'p95': quantiles[94] if len(quantiles) > 94 else 0,
                'p99': quantiles[98] if len(quantiles) > 98 else 0,
            }

    # Instance-level statistics
    for instance_id, stats_list in instance_stats.items():
        instance_durations = [s['duration'] for s in stats_list]
        instance_success = sum(1 for s in stats_list if s['status'] == 'success')
        instance_failed = sum(1 for s in stats_list if s['status'] == 'failed')
        instance_timeout = sum(1 for s in stats_list if s['status'] == 'timeout')

        report['instance_performance'][instance_id] = {
            'count': len(stats_list),
            'success': instance_success,
            'failed': instance_failed,
            'timeout': instance_timeout,
            'success_rate': instance_success / len(stats_list) if stats_list else 0,
        }

        if instance_durations:
            report['instance_performance'][instance_id].update({
                'average_duration': statistics.mean(instance_durations),
                'median_duration': statistics.median(instance_durations),
                'min_duration': min(instance_durations),
                'max_duration': max(instance_durations),
            })

    # Time distribution (buckets)
    if durations:
        time_buckets = defaultdict(int)
        for duration in durations:
            if duration < 1:
                bucket = '<1s'
            elif duration < 5:
                bucket = '1-5s'
            elif duration < 10:
                bucket = '5-10s'
            elif duration < 30:
                bucket = '10-30s'
            elif duration < 60:
                bucket = '30-60s'
            elif duration < 120:
                bucket = '1-2m'
            elif duration < 300:
                bucket = '2-5m'
            else:
                bucket = '>5m'
            time_buckets[bucket] += 1

        report['time_distribution'] = dict(time_buckets)

    # Save report (in the same directory as db files)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON report
    report_path = work_path / f'report_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Human-readable summary
    summary_path = report_path.with_suffix('.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('Stress Test Report\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Source Directory: {work_dir}\n')
        f.write(f'Database Files: {len(db_files)}\n')
        f.write(f'Report Generated: {datetime.now().isoformat()}\n\n')

        f.write('-' * 60 + '\n')
        f.write('Summary\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Total Runs: {total_runs}\n')
        f.write(f'  Completed: {completed_count}\n')
        f.write(f'    Success: {success_count}\n')
        f.write(f'    Failed: {failed_count}\n')
        f.write(f'    Timeout: {timeout_count}\n')
        f.write(f'  Running: {running_count}\n')
        f.write(f'  Pending: {pending_count}\n')
        f.write(f'  Success Rate: {report["summary"]["success_rate"]:.1%}\n\n')

        if report['timing']:
            f.write('-' * 60 + '\n')
            f.write('Timing Statistics\n')
            f.write('-' * 60 + '\n')
            timing = report['timing']
            f.write(f'  Durations Collected: {timing["total_durations_collected"]}\n')
            f.write(f'  Average: {timing["average_duration"]:.2f}s\n')
            f.write(f'  Median: {timing["median_duration"]:.2f}s\n')
            f.write(f'  Min: {timing["min_duration"]:.2f}s\n')
            f.write(f'  Max: {timing["max_duration"]:.2f}s\n')

            if 'percentiles' in timing:
                p = timing['percentiles']
                f.write(f'  P50: {p["p50"]:.2f}s\n')
                f.write(f'  P90: {p["p90"]:.2f}s\n')
                f.write(f'  P95: {p["p95"]:.2f}s\n')
                f.write(f'  P99: {p["p99"]:.2f}s\n')
            f.write('\n')

        if report['time_distribution']:
            f.write('-' * 60 + '\n')
            f.write('Time Distribution\n')
            f.write('-' * 60 + '\n')
            for bucket, count in report['time_distribution'].items():
                pct = (count / len(durations)) * 100 if durations else 0
                f.write(f'  {bucket}: {count} ({pct:.1f}%)\n')
            f.write('\n')

        if report['instance_performance']:
            f.write('-' * 60 + '\n')
            f.write('Instance Performance\n')
            f.write('-' * 60 + '\n')
            for instance_id, perf in sorted(report['instance_performance'].items()):
                f.write(f'\n  {instance_id}:\n')
                f.write(f'    Count: {perf["count"]}, Success Rate: {perf["success_rate"]:.1%}\n')
                if 'average_duration' in perf:
                    f.write(
                        f'    Avg Duration: {perf["average_duration"]:.2f}s, Median: {perf["median_duration"]:.2f}s\n'
                    )

    print(f'\n[Report] JSON report saved to: {report_path}')
    print(f'[Report] Summary saved to: {summary_path}')

    # Print summary to console
    print('\n' + '=' * 60)
    print('STRESS TEST REPORT SUMMARY')
    print('=' * 60)
    print(f'Source Directory: {work_dir}')
    print(f'Database Files: {len(db_files)}')
    print('-' * 60)
    print(f'Total Runs: {total_runs}')
    print(f'  Completed: {completed_count}')
    print(f'    Success: {success_count}')
    print(f'    Failed: {failed_count}')
    print(f'    Timeout: {timeout_count}')
    print(f'  Running: {running_count}')
    print(f'  Pending: {pending_count}')
    print(f'Success Rate: {report["summary"]["success_rate"]:.1%}')

    if report['timing']:
        timing = report['timing']
        print('-' * 60)
        print(f'Durations Collected: {timing["total_durations_collected"]}')
        print(f'Average Duration: {timing["average_duration"]:.2f}s')
        print(f'Median Duration: {timing["median_duration"]:.2f}s')
        if 'percentiles' in timing:
            p = timing['percentiles']
            print(f'P95: {p["p95"]:.2f}s, P99: {p["p99"]:.2f}s')

    print('=' * 60)

    return report


# =============================================================================
# Helper functions for environment variable parsing
# =============================================================================


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Parse integer from environment variable."""
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    """Parse comma-separated list from env var, return as list."""
    v = os.getenv(name)
    if not v:
        return []
    return [item.strip() for item in v.split(',') if item.strip()]


def _load_instances_from_file(dataset_path: str, instance_ids: list[str]) -> dict[str, SWEbenchInstance]:
    """Load specified instances from a local JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')

    instance_ids_set = set(instance_ids)
    instances: dict[str, SWEbenchInstance] = {}

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            inst_id = inst.get('instance_id', '')
            if inst_id in instance_ids_set:
                instances[inst_id] = inst
                if len(instances) == len(instance_ids_set):
                    break

    # Check if all instances were found
    missing = instance_ids_set - set(instances.keys())
    if missing:
        raise ValueError(f'Instances not found in dataset: {missing}')

    return instances


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client configuration from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY', 'sk-proj-1234567890')
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    model = os.getenv('OPENAI_MODEL', 'gpt-4.1')
    return {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': {
                    'model': model,
                },
            },
        }
    }


# =============================================================================
# Generator functions (called by nexus run-batch)
# =============================================================================


def generate_configs() -> Iterator[BatchItem]:
    """Generate stress test configurations.

    Each specified instance is repeated STRESS_REPEAT_COUNT times.
    This function is called by nexus run-batch when using this file as a generator.

    In multi-process mode, each process generates only its assigned subset of tasks
    based on STRESS_START_IDX and STRESS_END_IDX environment variables.
    """
    instance_ids = _env_list('STRESS_INSTANCE_IDS')
    if not instance_ids:
        raise ValueError('STRESS_INSTANCE_IDS environment variable is required')

    repeat_count = _env_int('STRESS_REPEAT_COUNT', DEFAULT_REPEAT_COUNT) or DEFAULT_REPEAT_COUNT
    round_prefix = os.getenv('STRESS_ROUND_PREFIX', '')  # e.g., 'round001'
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_DATASET_PATH)

    # Multi-process mode: only generate tasks for the assigned index range
    start_idx = _env_int('STRESS_START_IDX', None)
    end_idx = _env_int('STRESS_END_IDX', None)
    is_multiprocess = start_idx is not None and end_idx is not None

    # Stress test defaults to oracle mode (just run the test, no agent)
    run_oracle = _env_bool('SWE_RUN_ORACLE', default=True)
    runtime_provider_name = os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')
    # runtime_provider_cluster is the resource scheduling cluster (gongfeng/suanli) inside GongfengRuntimeProvider
    # This is different from runtime_provider_name which selects the RuntimeProvider implementation
    runtime_provider_cluster = os.getenv('SWE_RUNTIME_PROVIDER_CLUSTER', None)  # None means auto selection
    runtime_base_url = os.getenv('SWE_RUNTIME_BASE_URL', DEFAULT_SWE_RUNTIME_BASE_URL)
    runtime_token = os.getenv('SWE_RUNTIME_PROVIDER_TOKEN', None)  # API token for authentication
    runtime_tag = os.getenv('SWE_RUNTIME_TAG', 'latest')
    runtime_timeout = _env_int('SWE_RUNTIME_TIMEOUT', 1800) or 1800
    batch_id = os.getenv('SWE_BATCH_ID', None)  # Batch ID for runtime start/stop (auto-generated UUID if not set)
    test_sleep = _env_int('SWE_BENCH_TEST_SLEEP', 0) or 0

    # Load instances from dataset
    instances = _load_instances_from_file(dataset_path, instance_ids)

    total_tasks = len(instances) * repeat_count
    if is_multiprocess:
        print(f'[stress_generator] Multi-process mode: generating tasks {start_idx}-{end_idx} of {total_tasks}')
    else:
        print(f'[stress_generator] Loaded {len(instances)} instances')
        print(f'[stress_generator] Each instance will be repeated {repeat_count} times')
        print(f'[stress_generator] Total sandboxes: {total_tasks}')
    if round_prefix:
        print(f'[stress_generator] Round prefix: {round_prefix}')

    # Generate configs: for each instance, create repeat_count copies
    # Task index is calculated as: instance_idx * repeat_count + repeat_idx
    task_idx = 0
    for instance_id in instance_ids:
        instance = instances[instance_id]
        inst_dataset_type = str(instance.get('dataset_type') or DATASET_SWE_BENCH)

        for i in range(repeat_count):
            # In multi-process mode, skip tasks outside the assigned range
            if is_multiprocess:
                if task_idx < start_idx:
                    task_idx += 1
                    continue
                if task_idx >= end_idx:
                    return  # No more tasks for this process
            task_idx += 1  # Create unique run_id for each copy
            # Format: {round_prefix}__{instance_id}__{index} or {instance_id}__{index}
            run_id = f'{instance_id}__{round_prefix}__{i:05d}' if round_prefix else f'{instance_id}__{i:05d}'

            task_config: dict[str, Any] = {
                'dataset_path': dataset_path,
                'instance_id': instance_id,
                'run_oracle': bool(run_oracle),
            }

            config: dict[str, Any] = {
                'imports': [
                    'nexus.extensions.tasks.swe',
                    'nexus.extensions.tasks.swe.tests.mock_swe_agent',
                    'nexus.extensions.agents.swe_agent',
                    'nexus.extensions.agents.codeact',
                ],
                'task_name': 'swe_task',
                'task_config': task_config,
                'runtime_provider_name': runtime_provider_name,
                'runtime_provider_config': {
                    'api_base_url': runtime_base_url,
                    'token': runtime_token,  # API token for authentication
                    'runtime_image': f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}',
                    'runtime_provider': runtime_provider_cluster,  # Resource scheduling cluster (gongfeng/suanli)
                    'request_timeout': runtime_timeout,
                    'batch_id': batch_id,  # Batch ID for correlating start/stop calls
                    'resource_spec': {
                        'cpu': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('cpu', 2),
                        'memory': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('memory', 10),
                    },
                },
                'runtime_configs': {
                    'eval': {
                        'environment': {
                            'SWE_BENCH_TEST_SLEEP': str(test_sleep),
                        },
                    },
                },
            }

            # Stress test runs in oracle mode by default (no agent)
            if not run_oracle:
                agent_name = os.getenv('SWE_AGENT_NAME', 'swe_agent')
                config['agent_name'] = agent_name
                config['agent_config'] = {
                    'max_iterations': int(os.getenv('SWE_MAX_ITERATIONS', '5')),
                }
                config['openai_clients'] = _build_openai_clients()
            else:
                config['agent_name'] = None
                config['openai_clients'] = {}

            yield BatchItem(
                config=RunInput.model_validate(config),
                run_id=run_id,
                metadata={
                    'instance_id': instance_id,
                    'stress_index': i,
                    'round_prefix': round_prefix,
                    'dataset_path': dataset_path,
                    'repo': instance.get('repo', ''),
                    'dataset_type': inst_dataset_type,
                },
            )


def get_total_count() -> int:
    """Return total number of sandboxes to create."""
    instance_ids = _env_list('STRESS_INSTANCE_IDS')
    if not instance_ids:
        return 0
    repeat_count = _env_int('STRESS_REPEAT_COUNT', DEFAULT_REPEAT_COUNT) or DEFAULT_REPEAT_COUNT
    return len(instance_ids) * repeat_count


def get_batch_metadata() -> dict[str, Any]:
    """Return metadata about this batch run."""
    instance_ids = _env_list('STRESS_INSTANCE_IDS')
    repeat_count = _env_int('STRESS_REPEAT_COUNT', DEFAULT_REPEAT_COUNT) or DEFAULT_REPEAT_COUNT
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_DATASET_PATH)

    round_prefix = os.getenv('STRESS_ROUND_PREFIX', '')
    return {
        'task_name': 'swe_task',
        'benchmark': True,
        'instance_ids': instance_ids,
        'repeat_count': repeat_count,
        'round_prefix': round_prefix,
        'total_sandboxes': len(instance_ids) * repeat_count,
        'run_oracle': _env_bool('SWE_RUN_ORACLE', default=True),
        'dataset_path': dataset_path,
    }


# =============================================================================
# CLI functions (for running as a standalone script)
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stress test runner for sandbox platform benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--instance-ids',
        type=str,
        required=True,
        help='Comma-separated list of instance IDs (required)',
    )

    parser.add_argument(
        '--total-count-per-round',
        type=int,
        default=DEFAULT_TOTAL_COUNT_PER_ROUND,
        help=f'Target total number of sandboxes per round (default: {DEFAULT_TOTAL_COUNT_PER_ROUND})',
    )

    parser.add_argument(
        '--env',
        type=str,
        choices=['test', 'prod-x3', 'prod-x8'],
        default='prod-x3',
        help='Environment to use (default: test)',
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f'Path to SWE-bench dataset (default: {DEFAULT_DATASET_PATH})',
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Database path (default: auto-generated with timestamp)',
    )

    parser.add_argument(
        '--log-path',
        type=str,
        default=None,
        help='Log file path (default: auto-generated with timestamp)',
    )

    parser.add_argument(
        '-y',
        '--yes',
        action='store_true',
        help='Skip confirmation prompt',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print command without executing',
    )

    # Multi-process options
    parser.add_argument(
        '--multiprocess',
        action='store_true',
        help='Enable multi-process mode (each process handles --tasks-per-process tasks)',
    )

    parser.add_argument(
        '--process-count',
        type=int,
        default=0,
        help='Number of processes to use (0 for auto-calculate based on tasks-per-process, default: 0)',
    )

    parser.add_argument(
        '--tasks-per-process',
        type=int,
        default=DEFAULT_TASKS_PER_PROCESS,
        help=f'Number of tasks per process (default: {DEFAULT_TASKS_PER_PROCESS})',
    )

    # Multi-round options
    parser.add_argument(
        '--rounds',
        type=int,
        default=DEFAULT_ROUNDS,
        help=f'Number of rounds to run (0 for infinite, default: {DEFAULT_ROUNDS})',
    )

    parser.add_argument(
        '--round-duration',
        type=int,
        default=DEFAULT_ROUND_DURATION,
        help=f'Duration of each round in seconds (default: {DEFAULT_ROUND_DURATION}, i.e., 30 minutes)',
    )

    parser.add_argument(
        '--round-cooldown',
        type=int,
        default=DEFAULT_ROUND_COOLDOWN,
        help=f'Cooldown between rounds in seconds (default: {DEFAULT_ROUND_COOLDOWN})',
    )

    parser.add_argument(
        '--separate-db',
        action='store_true',
        help='Generate separate .db file for each round (default: use single shared .db file)',
    )

    # Statistics reporting options
    parser.add_argument(
        '--report-from',
        type=str,
        default=None,
        help='Generate report from existing stress test directory (e.g., ./workspace/benchmark_20260202_114117)',
    )

    parser.add_argument(
        '--batch-id',
        type=str,
        default=None,
        help='Batch ID for correlating runtime start/stop calls (default: auto-generated UUID)',
    )

    return parser.parse_args()


def calculate_repeat_count(num_cases: int, total_count_per_round: int) -> int:
    """Calculate repeat count per case based on total count per round.


    Args:
        num_cases: Number of unique test cases
        total_count_per_round: Target total number of sandboxes per round
    Returns:
        Number of times each case should be repeated
    """
    if num_cases <= 0:
        raise ValueError('Number of cases must be positive')

    # Round up to ensure we reach or exceed the target
    return (total_count_per_round + num_cases - 1) // num_cases


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f'{minutes}m{secs}s' if secs else f'{minutes}m'
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f'{hours}h{minutes}m' if minutes else f'{hours}h'


def _reset_generation_complete(db_path: str) -> None:
    """Reset generation_complete flag in the database.

    This allows new configs to be generated for the next round
    while keeping existing runs in the database.
    """
    import sqlite3

    db_file = Path(db_path)
    if not db_file.exists():
        return  # Database doesn't exist yet, nothing to reset

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Delete the generation_complete flag from batch_info
        cursor.execute("DELETE FROM batch_info WHERE key = 'generation_complete'")
        conn.commit()
        conn.close()
        print(f'[Reset] Cleared generation_complete flag in {db_path}')
    except sqlite3.Error as e:
        print(f'[Reset] Warning: Could not reset generation_complete flag: {e}')


def _run_process_worker(
    process_id: int,
    start_idx: int,
    end_idx: int,
    instance_ids: list[str],
    repeat_count: int,
    env_dict: dict[str, str],
    db_path: str,
    log_path: str,
    round_duration: int,
) -> dict[str, Any]:
    """Worker function for multi-process mode.

    Each process handles a subset of tasks (from start_idx to end_idx).

    Args:
        process_id: Process identifier (0-indexed)
        start_idx: Start index of tasks to process
        end_idx: End index of tasks to process (exclusive)
        instance_ids: List of instance IDs
        repeat_count: Number of times each instance is repeated
        env_dict: Environment variables dictionary
        db_path: Database path for this process
        log_path: Log file path for this process
        round_duration: Maximum duration in seconds

    Returns:
        Dictionary with process results
    """

    process_start_time = time.time()
    tasks_count = end_idx - start_idx

    # Calculate which instance_ids and repeat indices this process handles
    # Total tasks = len(instance_ids) * repeat_count
    # Each task is indexed as: instance_idx * repeat_count + repeat_idx
    process_instance_ids = []
    process_repeat_start = {}
    process_repeat_end = {}

    num_instances = len(instance_ids)
    for task_idx in range(start_idx, end_idx):
        instance_idx = task_idx // repeat_count
        repeat_idx = task_idx % repeat_count

        if instance_idx < num_instances:
            instance_id = instance_ids[instance_idx]
            if instance_id not in process_instance_ids:
                process_instance_ids.append(instance_id)
                process_repeat_start[instance_id] = repeat_idx
            process_repeat_end[instance_id] = repeat_idx + 1

    # Update environment for this process's subset
    env = env_dict.copy()
    env['STRESS_PROCESS_ID'] = str(process_id)
    env['STRESS_START_IDX'] = str(start_idx)
    env['STRESS_END_IDX'] = str(end_idx)

    # Build command for this process
    cmd = [
        'uv',
        'run',
        'nexus',
        'run-batch',
        '-g',
        './tests/stress/swe_sandbox/run_benchmark.py',
        '-d',
        db_path,
        '--max-concurrent',
        str(tasks_count),
        '--no-resume',
    ]

    log_message(f'[Process {process_id}] Starting with {tasks_count} tasks (index {start_idx}-{end_idx})')
    log_message(f'[Process {process_id}] Database: {db_path}')
    log_message(f'[Process {process_id}] Log: {log_path}')

    return_code = 0
    successful = 0
    failed = 0

    try:
        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f'# Process {process_id} started at {datetime.now().isoformat()}\n')
            log_file.write(f'# Tasks: {start_idx} to {end_idx} ({tasks_count} total)\n')
            log_file.write('=' * 60 + '\n')
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            import select

            while True:
                poll_result = process.poll()
                if poll_result is not None:
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                        # Count success/failure
                        if '[SUCCESS]' in line:
                            successful += 1
                        elif '[FAILED]' in line or '[ERROR]' in line:
                            failed += 1
                    return_code = poll_result
                    break

                elapsed = time.time() - process_start_time
                if elapsed >= round_duration:
                    log_file.write('\n# Duration limit reached, terminating...\n')
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return_code = -1
                    break

                readable, _, _ = select.select([process.stdout], [], [], 1.0)
                if readable:
                    line = process.stdout.readline()
                    if line:
                        log_file.write(line)
                        log_file.flush()
                        # Count success/failure
                        if '[SUCCESS]' in line:
                            successful += 1
                        elif '[FAILED]' in line or '[ERROR]' in line:
                            failed += 1

            elapsed = time.time() - process_start_time
            log_file.write(f'\n# Process {process_id} completed in {format_duration(int(elapsed))}\n')
            log_file.write(f'# Return code: {return_code}\n')
            log_file.write(f'# Successful: {successful}, Failed: {failed}\n')

    except Exception as e:
        log_message(f'[Process {process_id}] Error: {e}')
        return_code = 1

    elapsed = time.time() - process_start_time
    log_message(
        f'[Process {process_id}] Completed in {format_duration(int(elapsed))} - Success: {successful}, Failed: {failed}'
    )

    return {
        'process_id': process_id,
        'return_code': return_code,
        'successful': successful,
        'failed': failed,
        'duration': elapsed,
        'tasks_count': tasks_count,
    }


def _collect_realtime_stats_from_dbs(db_pattern: str) -> dict[str, Any]:
    """Collect real-time statistics from database files.

    Args:
        db_pattern: Glob pattern for db files (e.g., './workspace/benchmark_xxx/*.db')

    Returns:
        Dictionary with aggregated statistics
    """
    import glob
    import sqlite3

    db_files = sorted(glob.glob(db_pattern))
    db_files = [f for f in db_files if not f.endswith('-shm') and not f.endswith('-wal')]

    if not db_files:
        return {'total': 0, 'success': 0, 'failed': 0, 'running': 0, 'pending': 0}

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'timeout': 0,
        'running': 0,
        'pending': 0,
        'durations': [],
    }

    for db_path in db_files:
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query status counts
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM runs
                GROUP BY status
            """)

            for row in cursor.fetchall():
                status = row['status']
                count = row['count']
                stats['total'] += count
                if status == 'success':
                    stats['success'] += count
                elif status == 'failed':
                    stats['failed'] += count
                elif status == 'timeout':
                    stats['timeout'] += count
                elif status == 'running':
                    stats['running'] += count
                elif status == 'pending':
                    stats['pending'] += count

            # Query recent durations for timing stats
            cursor.execute("""
                SELECT start_time, end_time
                FROM runs
                WHERE status = 'success' AND start_time IS NOT NULL AND end_time IS NOT NULL
                ORDER BY end_time DESC
                LIMIT 100
            """)

            for row in cursor.fetchall():
                try:
                    start_dt = datetime.fromisoformat(row['start_time'].replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(row['end_time'].replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    if duration >= 0:
                        stats['durations'].append(duration)
                except (ValueError, TypeError):
                    pass

            conn.close()
        except Exception:
            # Ignore errors (db might be locked)
            pass

    return stats


def _stats_monitor_thread(
    db_pattern: str,
    interval: int,
    stop_event: threading.Event,
) -> None:
    """Background thread to periodically report statistics.

    Args:
        db_pattern: Glob pattern for db files
        interval: Report interval in seconds
        stop_event: Event to signal thread to stop
    """
    last_report_time = time.time()

    while not stop_event.is_set():
        # Wait for interval or stop signal
        stop_event.wait(timeout=1.0)

        current_time = time.time()
        if current_time - last_report_time >= interval:
            stats = _collect_realtime_stats_from_dbs(db_pattern)

            if stats['total'] > 0:
                completed = stats['success'] + stats['failed'] + stats['timeout']
                success_rate = stats['success'] / completed * 100 if completed > 0 else 0

                log_message(
                    f'Progress: {completed}/{stats["total"]} completed '
                    f'(Success: {stats["success"]}, Failed: {stats["failed"]}, '
                    f'Timeout: {stats["timeout"]}, Running: {stats["running"]}), '
                    f'Success Rate: {success_rate:.1f}%'
                )

                # Show timing stats if we have duration data
                if stats['durations']:
                    avg_duration = statistics.mean(stats['durations'])
                    median_duration = statistics.median(stats['durations'])
                    log_message(
                        f'Duration: Avg={avg_duration:.1f}s, Median={median_duration:.1f}s '
                        f'(based on {len(stats["durations"])} recent samples)'
                    )

            last_report_time = current_time


def run_multiprocess_round(
    round_num: int,
    instance_ids: list[str],
    repeat_count: int,
    actual_total: int,
    tasks_per_process: int,
    env: dict[str, str],
    db_path: str,
    base_log_path: str,
    round_duration: int,
    round_prefix: str = '',
) -> int:
    """Run a single round using multi-process mode.

    Args:
        round_num: Current round number
        instance_ids: List of instance IDs
        repeat_count: Number of times each instance is repeated
        actual_total: Total number of tasks
        tasks_per_process: Number of tasks per process
        env: Environment variables
        db_path: Base database path
        base_log_path: Base log path
        round_duration: Maximum duration per process
        round_prefix: Round prefix for run_id

    Returns:
        Combined return code (0 if all processes succeeded)
    """
    # Calculate number of processes needed
    num_processes = (actual_total + tasks_per_process - 1) // tasks_per_process

    log_message(f'[Round {round_num}] Multi-process mode: {num_processes} processes, {tasks_per_process} tasks/process')
    log_message(f'[Round {round_num}] Total tasks: {actual_total}')
    print('=' * 60)

    # Update environment with round prefix
    env_copy = env.copy()
    if round_prefix:
        env_copy['STRESS_ROUND_PREFIX'] = round_prefix

    # Prepare process arguments
    process_args = []
    for proc_id in range(num_processes):
        start_idx = proc_id * tasks_per_process
        end_idx = min(start_idx + tasks_per_process, actual_total)

        # Generate per-process paths (files are already in a subdirectory)
        if round_prefix:
            proc_db_path = db_path.replace('.db', f'_{round_prefix}_proc{proc_id:03d}.db')
            proc_log_path = f'{base_log_path}_{round_prefix}_proc{proc_id:03d}.log'
        else:
            proc_db_path = db_path.replace('.db', f'_proc{proc_id:03d}.db')
            proc_log_path = f'{base_log_path}_proc{proc_id:03d}.log'

        process_args.append((
            proc_id,
            start_idx,
            end_idx,
            instance_ids,
            repeat_count,
            env_copy,
            proc_db_path,
            proc_log_path,
            round_duration,
        ))

    # Run processes in parallel
    round_start_time = time.time()
    results = []

    # Start stats monitoring thread
    db_dir = str(Path(db_path).parent)
    db_pattern = f'{db_dir}/*.db'
    stop_event = threading.Event()
    stats_thread = threading.Thread(
        target=_stats_monitor_thread,
        args=(db_pattern, 15, stop_event),  # Report every 15 seconds
        daemon=True,
    )
    stats_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(_run_process_worker, *args): args[0] for args in process_args}

            for future in as_completed(futures):
                proc_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f'[Process {proc_id}] Exception: {e}', flush=True)
                    results.append({
                        'process_id': proc_id,
                        'return_code': 1,
                        'successful': 0,
                        'failed': 0,
                        'duration': 0,
                        'tasks_count': 0,
                    })
    finally:
        # Stop the stats monitoring thread
        stop_event.set()
        stats_thread.join(timeout=5.0)

    # Aggregate results
    total_successful = sum(r['successful'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_duration = time.time() - round_start_time
    failed_processes = sum(1 for r in results if r['return_code'] != 0 and r['return_code'] != -1)

    print('\n' + '=' * 60)
    log_message(f'[Round {round_num}] Multi-process round completed!')
    log_message(f'[Round {round_num}] Duration: {format_duration(int(total_duration))}')
    log_message(f'[Round {round_num}] Processes: {num_processes} (Failed: {failed_processes})')
    log_message(f'[Round {round_num}] Tasks: Successful={total_successful}, Failed={total_failed}')
    print('=' * 60)

    # Return non-zero if any process failed (excluding timeout which is -1)
    return 1 if failed_processes > 0 else 0


def run_single_round(
    round_num: int,
    cmd: list[str],
    env: dict[str, str],
    db_path: str,
    log_path: str,
    round_duration: int,
) -> int:
    """Run a single round of stress test.

    Args:
        round_num: Current round number (1-indexed)
        cmd: Command to execute
        env: Environment variables
        db_path: Database path for this round
        log_path: Log file path for this round
        round_duration: Maximum duration for this round in seconds

    Returns:
        Return code from the process
    """
    global _shutdown_requested

    # Ensure workspace directory exists
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # Reset generation_complete flag to allow new configs for this round
    # This is necessary because we're reusing the same db file across rounds
    _reset_generation_complete(db_path)

    print(f'\n[Round {round_num}] Starting stress test...')
    print(f'[Round {round_num}] Database: {db_path}')
    print(f'[Round {round_num}] Log: {log_path}')
    print(f'[Round {round_num}] Max duration: {format_duration(round_duration)}')
    print('=' * 60)

    start_time = time.time()
    return_code = 0

    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # Log round info
            log_file.write(f'# Round {round_num} started at {datetime.now().isoformat()}\n')
            log_file.write(f'# Max duration: {format_duration(round_duration)}\n')
            log_file.write('=' * 60 + '\n')
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output to both console and log file
            assert process.stdout is not None

            # Use non-blocking read with timeout
            import select

            while True:
                # Check if process has finished
                poll_result = process.poll()
                if poll_result is not None:
                    # Process finished, read remaining output
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_file.write(line)
                        log_file.flush()
                    return_code = poll_result
                    break

                # Check time limit
                elapsed = time.time() - start_time
                if elapsed >= round_duration:
                    print(
                        f'\n[Round {round_num}] Duration limit reached ({format_duration(round_duration)}), terminating...'
                    )
                    log_file.write('\n# Duration limit reached, terminating...\n')
                    process.terminate()
                    # Give it some time to terminate gracefully
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f'[Round {round_num}] Process did not terminate, killing...')
                        process.kill()
                        process.wait()
                    return_code = -1
                    break

                # Check for shutdown request
                if _shutdown_requested:
                    print(f'\n[Round {round_num}] Shutdown requested, terminating...')
                    log_file.write('\n# Shutdown requested, terminating...\n')
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return_code = -2
                    break

                # Use select to wait for output with timeout
                readable, _, _ = select.select([process.stdout], [], [], 1.0)
                if readable:
                    line = process.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_file.write(line)
                        log_file.flush()

            # Log completion
            elapsed = time.time() - start_time
            log_file.write(f'\n# Round {round_num} completed in {format_duration(int(elapsed))}\n')
            log_file.write(f'# Return code: {return_code}\n')

    except KeyboardInterrupt:
        print(f'\n[Round {round_num}] Interrupted by user.')
        return_code = 130

    elapsed = time.time() - start_time
    print()
    print('=' * 60)
    print(f'[Round {round_num}] Completed!')
    print(f'[Round {round_num}] Duration: {format_duration(int(elapsed))}')
    print(f'[Round {round_num}] Return code: {return_code}')
    print('=' * 60)

    return return_code


def main() -> int:
    """Main entry point."""
    global _shutdown_requested

    args = parse_args()

    # Handle --report-from mode: generate report from existing directory
    if args.report_from:
        print('=' * 60)
        print('Generating Report from Existing Stress Test')
        print('=' * 60)
        report = generate_report_from_directory(args.report_from)
        return 0 if 'error' not in report else 1

    # Parse instance IDs
    if args.instance_ids:
        instance_ids = [x.strip() for x in args.instance_ids.split(',') if x.strip()]

    num_cases = len(instance_ids)
    if num_cases == 0:
        print('Error: No instance IDs specified', file=sys.stderr)
        return 1

    # Calculate repeat count dynamically
    repeat_count = calculate_repeat_count(num_cases, args.total_count_per_round)
    actual_total = num_cases * repeat_count

    # Generate paths - organize files in a timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = args.db_path or f'./workspace/benchmark_{timestamp}'
    # Remove file extension if user provided a .db path
    if work_dir.endswith('.db'):
        work_dir = work_dir[:-3]
    if work_dir.endswith('.log'):
        work_dir = work_dir[:-4]

    # Create the working directory
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Set paths within the working directory
    db_path = f'{work_dir}/benchmark.db'
    base_log_path = f'{work_dir}/benchmark'

    # Get runtime URL based on environment
    runtime_url = ENV_CONFIGS[args.env]
    runtime_tag = os.getenv('SWE_RUNTIME_TAG', 'latest')
    runtime_provider_name = os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_provider_cluster = os.getenv('SWE_RUNTIME_PROVIDER_CLUSTER', '')  # Resource scheduling cluster (gongfeng/suanli)
    batch_id = args.batch_id or os.getenv('SWE_BATCH_ID', '') or str(uuid.uuid4())  # Batch ID for runtime start/stop
    test_sleep = os.getenv('SWE_BENCH_TEST_SLEEP', '0')

    # Determine round info
    total_rounds = args.rounds
    is_infinite = total_rounds == 0
    rounds_str = 'infinite' if is_infinite else str(total_rounds)

    # Print configuration
    print('=' * 60)
    print('Sandbox Platform Stress Test')
    print('=' * 60)
    print(f'Environment:            {args.env}')
    print(f'Runtime URL:            {runtime_url}')
    print(f'Runtime Tag:            {runtime_tag}')
    print(f'Runtime Provider Name:  {runtime_provider_name}')
    print(f'Runtime Provider:       {runtime_provider_cluster or "(auto)"}')
    print(f'Batch ID:               {batch_id}')
    print(f'Test Sleep:             {test_sleep}s')
    print(f'Dataset:                {args.dataset_path}')
    print('-' * 60)
    print(f'Number of cases:        {num_cases}')
    print(f'Repeat count per case:  {repeat_count}')
    print(f'Total count per round:  {args.total_count_per_round}')
    print(f'Actual total sandboxes: {actual_total} per round')
    print('-' * 60)
    print(f'Rounds:                 {rounds_str}')
    print(f'Round duration:         {format_duration(args.round_duration)}')
    print(f'Round cooldown:         {format_duration(args.round_cooldown)}')
    print(f'Separate DB per round:  {args.separate_db}')
    if args.multiprocess:
        num_processes = (
            args.process_count
            if args.process_count > 0
            else (actual_total + args.tasks_per_process - 1) // args.tasks_per_process
        )
        print('Multi-process mode:     ENABLED')
        print(f'Tasks per process:      {args.tasks_per_process}')
        print(f'Number of processes:    {num_processes}')
    print('-' * 60)
    if args.separate_db:
        print(f'Database prefix:        {db_path.replace(".db", "")}')
    else:
        print(f'Database file:          {db_path}')
    print(f'Log prefix:             {base_log_path}')
    print('=' * 60)
    print()
    print('Instance IDs:')
    for i, instance_id in enumerate(instance_ids, 1):
        print(f'  {i}. {instance_id}')
    print()

    # Confirmation prompt
    if not args.yes and not args.dry_run:
        try:
            if is_infinite:
                prompt = f'Start infinite stress test ({actual_total} sandboxes/round, {format_duration(args.round_duration)}/round)? (y/N) '
            else:
                prompt = f'Start {total_rounds}-round stress test ({actual_total} sandboxes/round)? (y/N) '
            response = input(prompt).strip().lower()
            if response not in ('y', 'yes'):
                print('Cancelled.')
                return 0
        except (EOFError, KeyboardInterrupt):
            print('\nCancelled.')
            return 0

    # Prepare environment variables
    env = os.environ.copy()
    env.update({
        'STRESS_INSTANCE_IDS': ','.join(instance_ids),
        'STRESS_REPEAT_COUNT': str(repeat_count),
        'SWE_RUN_ORACLE': '1',
        'SWE_DATASET_PATH': args.dataset_path,
        'SWE_RUNTIME_PROVIDER_NAME': os.environ.get('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng'),
        'SWE_RUNTIME_PROVIDER_CLUSTER': os.environ.get('SWE_RUNTIME_PROVIDER_CLUSTER', ''),
        'SWE_BATCH_ID': batch_id,
        'SWE_RUNTIME_BASE_URL': runtime_url,
    })

    # Build command
    cmd = [
        'uv',
        'run',
        'nexus',
        'run-batch',
        '-g',
        './tests/stress/swe_sandbox/run_benchmark.py',
        '-d',
        db_path,
        '--max-concurrent',
        str(actual_total),
        '--no-resume',
    ]

    if args.dry_run:
        print('Dry run mode - command that would be executed:')
        print()
        if args.separate_db:
            print(f'# Database prefix: {db_path.replace(".db", "")} (separate file per round)')
        else:
            print(f'# Database file: {db_path} (shared across all rounds)')
        if is_infinite:
            print(
                f'# Infinite rounds, {format_duration(args.round_duration)} each, {format_duration(args.round_cooldown)} cooldown'
            )
        else:
            print(
                f'# {total_rounds} rounds, {format_duration(args.round_duration)} each, {format_duration(args.round_cooldown)} cooldown'
            )
        print()
        for r in range(1, min(3, total_rounds + 1) if not is_infinite else 3):
            round_prefix = f'round{r:03d}'
            env_copy = env.copy()
            env_copy['STRESS_ROUND_PREFIX'] = round_prefix
            env_str = ' \\\n'.join(
                f'{k}={v!r}' for k, v in sorted(env_copy.items()) if k.startswith('STRESS_') or k.startswith('SWE_')
            )
            if args.separate_db:
                round_db_path = db_path.replace('.db', f'_round{r:03d}.db')
                round_cmd = cmd.copy()
                round_cmd[round_cmd.index('-d') + 1] = round_db_path
                cmd_str = ' \\\n  '.join(round_cmd)
            else:
                cmd_str = ' \\\n  '.join(cmd)
            print(f'# Round {r} (run_id prefix: {round_prefix}):')
            if args.separate_db:
                print(f'# Database: {round_db_path}')
            print(f'{env_str} \\\n{cmd_str}')
            print()
        if is_infinite or total_rounds > 2:
            print('# ... (more rounds)')
        return 0

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Run rounds
    round_num = 0
    total_return_code = 0

    print('\n' + '=' * 60)
    print('Starting multi-round stress test...')
    print('Press Ctrl+C to stop after current round')
    print('=' * 60)

    while True:
        round_num += 1

        # Check if we've reached the limit
        if not is_infinite and round_num > total_rounds:
            break

        # Check for shutdown request
        if _shutdown_requested:
            print(f'\nShutdown requested, stopping before round {round_num}')
            break

        # Generate paths for this round
        if total_rounds == 1 and not is_infinite:
            # Single round, use simple names
            log_path = f'{base_log_path}.log'
            round_db_path = db_path
            round_prefix = ''  # No prefix for single round
        else:
            # Multi-round, add round suffix to log and use prefix for run_id
            log_path = f'{base_log_path}_round{round_num:03d}.log'
            round_prefix = f'round{round_num:03d}'
            # Separate db file per round if requested
            round_db_path = db_path.replace('.db', f'_round{round_num:03d}.db') if args.separate_db else db_path

        # Update environment with round prefix
        env['STRESS_ROUND_PREFIX'] = round_prefix

        # Update command with round-specific db path if using separate db files
        round_cmd = cmd.copy()
        if args.separate_db and round_db_path != db_path:
            db_index = round_cmd.index('-d') + 1
            round_cmd[db_index] = round_db_path

        # Print round header
        print('\n' + '#' * 60)
        if is_infinite:
            print(f'# ROUND {round_num} (infinite mode)')
        else:
            print(f'# ROUND {round_num} / {total_rounds}')
        if round_prefix:
            print(f'# Run ID prefix: {round_prefix}')
        print('#' * 60)

        # Run this round
        if args.multiprocess:
            # Multi-process mode
            return_code = run_multiprocess_round(
                round_num=round_num,
                instance_ids=instance_ids,
                repeat_count=repeat_count,
                actual_total=actual_total,
                tasks_per_process=args.tasks_per_process,
                env=env,
                db_path=round_db_path,
                base_log_path=base_log_path,
                round_duration=args.round_duration,
                round_prefix=round_prefix,
            )
        else:
            # Single process mode (original behavior)
            return_code = run_single_round(
                round_num=round_num,
                cmd=round_cmd,
                env=env,
                db_path=round_db_path,
                log_path=log_path,
                round_duration=args.round_duration,
            )

        if return_code != 0 and return_code != -1:  # -1 is normal timeout
            total_return_code = return_code
            if return_code == -2:  # Shutdown requested
                break

        # Check if more rounds to run
        if is_infinite or round_num < total_rounds:
            if _shutdown_requested:
                break

            # Cooldown between rounds
            print(f'\n[Cooldown] Waiting {format_duration(args.round_cooldown)} before next round...')
            print('[Cooldown] Press Ctrl+C to stop')

            cooldown_start = time.time()
            while time.time() - cooldown_start < args.round_cooldown:
                if _shutdown_requested:
                    print('\n[Cooldown] Shutdown requested, stopping.')
                    break
                time.sleep(1)

            if _shutdown_requested:
                break

    # Auto-generate report from database files (for multiprocess mode)
    if args.multiprocess:
        print('\n' + '=' * 60)
        print('Auto-generating report from multiprocess database files...')
        print('=' * 60)
        generate_report_from_directory(work_dir)

    # Final summary
    print('=' * 60)
    print('STRESS TEST COMPLETE')
    print('=' * 60)
    print(f'Total rounds executed: {round_num if not _shutdown_requested else round_num}')
    print(f'Final return code:     {total_return_code}')
    print(f'Database file:         {db_path}')
    print(f'Log prefix:            {base_log_path}')
    print('=' * 60)
    return total_return_code


if __name__ == '__main__':
    sys.exit(main())
