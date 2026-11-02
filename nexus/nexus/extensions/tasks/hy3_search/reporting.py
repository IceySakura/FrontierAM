from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f'{h}h {m}m {sec}s' if h else (f'{m}m {sec}s' if m else f'{sec}s')


def print_benchmark_metrics(result: Any) -> None:
    """Print detailed metrics for a single benchmark after completion."""
    if result.status != 'success' or result.metrics is None:
        return

    metrics = result.metrics
    name = result.name
    k = metrics.get('k', 1)

    total = metrics.get('total', 0)
    success = metrics.get('success', 0)
    failed = metrics.get('failed', 0)
    timeout = metrics.get('timeout', 0)
    accuracy = metrics.get('accuracy', 0)
    correct = metrics.get('correct', 0)
    evaluated = metrics.get('evaluated', 0)
    avg_time = metrics.get('avg_time_sec')

    run_label = f' [avg@{k}]' if k > 1 else ''
    print('\n' + '-' * 55)
    print(f'  BENCHMARK: {name.upper()}{run_label}')
    print('-' * 55)

    # Status line
    status_parts = []
    if failed:
        status_parts.append(f'{failed} failed')
    if timeout:
        status_parts.append(f'{timeout} timeout')
    status_str = f' ({", ".join(status_parts)})' if status_parts else ''
    print(f'  Runs:     {success}/{total} success{status_str}')
    if k > 1:
        print(f'  Accuracy: {correct}/{evaluated} = {accuracy:.1%}')
    else:
        print(f'  Accuracy: {correct}/{total} = {accuracy:.1%}')

    # For multi-run, also show pass@k and mean accuracy
    if k > 1:
        strict_acc = metrics.get('strict_avg_k_accuracy', accuracy)
        strict_correct = metrics.get('strict_avg_k_correct', correct)
        avg_k_acc = metrics.get('avg_k_accuracy', 0)
        avg_k_correct = metrics.get('avg_k_correct', 0)
        pass_k = metrics.get('pass_k_accuracy', 0)
        pass_correct = metrics.get('pass_k_correct', 0)
        mean_acc = metrics.get('mean_accuracy', 0)
        instances = metrics.get('instances', evaluated)
        print(f'  strict@{k}: {strict_correct}/{instances} = {strict_acc:.1%}')
        print(f'  avg@{k}:    {avg_k_correct}/{instances} = {avg_k_acc:.1%}')
        print(f'  pass@{k}:   {pass_correct}/{instances} = {pass_k:.1%}')
        print(f'  mean_acc: {mean_acc:.1%}  (simple mean of {k} runs)')

        per_run = metrics.get('per_run', [])
        if per_run:
            print('  Per-run:')
            for idx, run_m in enumerate(per_run, 1):
                ra = run_m.get('accuracy', 0)
                rc = run_m.get('correct', 0)
                rt_total = run_m.get('total', run_m.get('evaluated', 0))
                rt = run_m.get('timeout', 0)
                t_str = f', {rt}T' if rt else ''
                print(f'    run {idx}: {rc}/{rt_total} = {ra:.1%}{t_str}')

    print(f'  Duration: {format_duration(result.duration_seconds)}')
    if avg_time is not None:
        print(f'  Avg Time: {avg_time:.1f}s per task')

    # By source breakdown
    by_source = metrics.get('by_source', {})
    if by_source:
        print('  By Source:')
        for source, data in by_source.items():
            src_correct = data.get('correct', 0)
            src_total = data.get('total', 0)
            src_acc = data.get('accuracy', 0)
            print(f'    {source}: {src_correct}/{src_total} = {src_acc:.1%}')

    # Timeout run IDs
    timeout_runs = metrics.get('timeout_runs', [])
    if timeout_runs:
        ids = [tr['run_id'] if isinstance(tr, dict) else str(tr) for tr in timeout_runs[:5]]
        more = len(timeout_runs) - 5
        suffix = f' (+{more} more)' if more > 0 else ''
        print(f'  Timeouts: {", ".join(ids)}{suffix}')

    print('-' * 55 + '\n')


def print_summary(summary: Any) -> None:
    """Print execution summary with aggregated statistics (plain text fallback)."""
    if not summary.results:
        print('\n' + '=' * 50)
        print('SUMMARY: No benchmarks were run')
        print('=' * 50)
        return

    by_status: dict[str, list[Any]] = {'success': [], 'failed': [], 'skipped': []}
    for r in summary.results:
        by_status.setdefault(r.status, []).append(r)

    successful, failed, skipped = by_status['success'], by_status['failed'], by_status['skipped']

    total_correct = 0
    total_evaluated = 0
    total_tasks = 0
    total_success_tasks = 0

    for r in successful:
        if r.metrics:
            total_correct += r.metrics.get('correct', 0)
            total_evaluated += r.metrics.get('evaluated', 0)
            total_tasks += r.metrics.get('total', 0)
            total_success_tasks += r.metrics.get('success', 0)

    k = summary.k
    aggregate_denominator = total_tasks if k <= 1 else total_evaluated
    aggregate_accuracy = total_correct / aggregate_denominator if aggregate_denominator > 0 else 0.0

    print('\n' + '=' * 60)
    print('                      FINAL SUMMARY')
    print('=' * 60)

    if k > 1:
        print(f'\n  Mode: avg@{k} / pass@{k}')

    print(f'\n  Benchmarks: {len(summary.results)} total')
    print(f'    ✓ Success: {len(successful)}')
    print(f'    ✗ Failed:  {len(failed)}')
    print(f'    ○ Skipped: {len(skipped)}')

    if successful:
        print(f'\n  Aggregate Metrics (across {len(successful)} benchmarks):')
        print(f'    Total Tasks:    {total_tasks}')
        print(f'    Success Tasks:  {total_success_tasks}')
        print(f'    Correct:        {total_correct}/{aggregate_denominator}')
        print(f'    Accuracy:       {aggregate_accuracy:.2%}')

    print(f'\n  Total Duration: {format_duration(summary.duration_seconds)}')

    print('\n' + '-' * 60)
    print('  BENCHMARK DETAILS')
    print('-' * 60)

    for r in successful:
        acc = r.metrics.get('accuracy', 0) if r.metrics else 0
        correct = r.metrics.get('correct', 0) if r.metrics else 0
        evaluated = r.metrics.get('evaluated', 0) if r.metrics else 0
        total = r.metrics.get('total', evaluated) if r.metrics else evaluated
        k_label = f' [avg@{k}]' if k > 1 else ''
        correct_str = f'{correct:.2f}' if isinstance(correct, float) and not correct.is_integer() else str(int(correct))
        denominator = evaluated if k > 1 else total
        print(
            f'  ✓ {r.name:20s} {correct_str:>4s}/{denominator:<4d} = {acc:6.2%}{k_label}  ({format_duration(r.duration_seconds)})'
        )

    for r in skipped:
        print(f'  ○ {r.name:20s} skipped')

    for r in failed:
        error_msg = r.error[:60] if r.error else 'unknown error'
        print(f'  ✗ {r.name:20s} {error_msg}')

    print('-' * 60)

    if summary.output_dir:
        print(f'\n  Output Directory: {summary.output_dir}')


def print_rich_summary(summary: Any) -> None:
    """Print execution summary as a rich formatted table."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fall back to plain text if rich is not available
        print_summary(summary)
        return

    console = Console()
    k = summary.k

    if not summary.results:
        console.print('\n[bold yellow]SUMMARY: No benchmarks were run[/bold yellow]')
        return

    by_status: dict[str, list[Any]] = {'success': [], 'failed': [], 'skipped': []}
    for r in summary.results:
        by_status.setdefault(r.status, []).append(r)

    successful = by_status['success']
    failed = by_status['failed']
    skipped = by_status['skipped']

    # Compute aggregate metrics
    total_correct = 0
    total_evaluated = 0
    total_tasks = 0
    total_timeout = 0
    total_instances = 0
    sum_avg_k = 0.0
    sum_pass_k = 0.0
    sum_mean_acc = 0.0
    for r in successful:
        if r.metrics:
            total_correct += r.metrics.get('correct', 0)
            total_evaluated += r.metrics.get('evaluated', 0)
            total_tasks += r.metrics.get('total', 0)
            total_timeout += r.metrics.get('timeout', 0)
            instances = r.metrics.get('instances', r.metrics.get('evaluated', 0))
            total_instances += instances
            sum_avg_k += r.metrics.get('avg_k_accuracy', 0.0) * instances
            sum_pass_k += r.metrics.get('pass_k_accuracy', 0.0) * instances
            sum_mean_acc += r.metrics.get('mean_accuracy', 0.0) * instances
    aggregate_denominator = total_tasks if k <= 1 else total_evaluated
    aggregate_accuracy = total_correct / aggregate_denominator if aggregate_denominator > 0 else 0.0
    aggregate_avg_k = (sum_avg_k / total_instances) if total_instances > 0 else 0.0
    aggregate_pass_k = (sum_pass_k / total_instances) if total_instances > 0 else 0.0
    aggregate_mean_acc = (sum_mean_acc / total_instances) if total_instances > 0 else 0.0

    # ---- Header info ----
    mode_label = f'avg@{k} / pass@{k}' if k > 1 else 'single run'
    console.print()
    console.print(f'[bold cyan]Search Benchmark Results[/bold cyan]  [dim]mode: {mode_label}[/dim]')
    console.print(f'[dim]Output: {summary.output_dir}[/dim]')
    console.print(f'[dim]Duration: {format_duration(summary.duration_seconds)}[/dim]')
    console.print()

    # ---- Main results table ----
    table = Table(box=box.ROUNDED, show_header=True, header_style='bold white on dark_blue')
    table.add_column('Benchmark', style='bold', min_width=15)
    table.add_column('Status', justify='center', min_width=8)
    if k <= 1:
        table.add_column('Correct', justify='right', min_width=10)
        table.add_column('Accuracy', justify='right', min_width=10)

    if k > 1:
        table.add_column(f'avg@{k}', justify='right', min_width=10)
        table.add_column(f'pass@{k}', justify='right', min_width=10)
        table.add_column('mean_acc', justify='right', min_width=10)

    table.add_column('Timeout', justify='right', min_width=8)
    table.add_column('Duration', justify='right', min_width=10)

    for r in successful:
        m = r.metrics or {}
        acc = m.get('accuracy', 0)
        correct = m.get('correct', 0)
        evaluated = m.get('evaluated', 0)
        total = m.get('total', evaluated)
        timeout_cnt = m.get('timeout', 0)

        acc_str = f'[green]{acc:.1%}[/green]' if acc >= 0.5 else f'[yellow]{acc:.1%}[/yellow]'
        timeout_str = f'[red]{timeout_cnt}[/red]' if timeout_cnt else '[dim]0[/dim]'

        row: list[Any] = [r.name, '[green]✓ OK[/green]']

        if k > 1:
            avg_k = m.get('avg_k_accuracy', acc)
            pass_k = m.get('pass_k_accuracy', 0)
            mean_acc = m.get('mean_accuracy', 0)
            avg_str = f'[green]{avg_k:.1%}[/green]' if avg_k >= 0.5 else f'[yellow]{avg_k:.1%}[/yellow]'
            pass_str = f'[cyan]{pass_k:.1%}[/cyan]'
            mean_str = f'{mean_acc:.1%}'
            row += [avg_str, pass_str, mean_str]
        else:
            row += [f'{correct}/{total}', acc_str]

        row += [timeout_str, format_duration(r.duration_seconds)]
        table.add_row(*[str(x) for x in row])

    for r in skipped:
        row = [r.name, '[dim]○ skip[/dim]']
        if k > 1:
            row += ['-', '-', '-']
        else:
            row += ['-', '-']
        row += ['-', '-']
        table.add_row(*row)

    for r in failed:
        err = (r.error or 'unknown')[:40]
        row = [r.name, '[red]✗ FAIL[/red]']
        if k > 1:
            row += ['-', '-', '-']
        else:
            row += ['-', f'[red]{err}[/red]']
        row += ['-', format_duration(r.duration_seconds)]
        table.add_row(*row)

    # Separator + aggregate row
    if successful:
        sep_row: list[str] = ['[dim]─ AGGREGATE ─[/dim]', '']
        if k > 1:
            sep_row += [
                f'[bold]{aggregate_avg_k:.1%}[/bold]',
                f'[bold]{aggregate_pass_k:.1%}[/bold]',
                f'[bold]{aggregate_mean_acc:.1%}[/bold]',
            ]
        else:
            sep_row += [f'{total_correct}/{aggregate_denominator}', f'[bold]{aggregate_accuracy:.1%}[/bold]']
        sep_row += [str(total_timeout), f'[bold]{format_duration(summary.duration_seconds)}[/bold]']
        table.add_section()
        table.add_row(*sep_row)

    console.print(table)

    # ---- Timeout details table (if any timeouts exist) ----
    timeout_details: list[tuple[str, str, str]] = []
    for r in successful:
        m = r.metrics or {}
        for tr in m.get('timeout_runs', []):
            run_id = tr.get('run_id', '?') if isinstance(tr, dict) else str(tr)
            error = (tr.get('error', '') or '')[:60] if isinstance(tr, dict) else ''
            partial = str(tr.get('partial_result', '')) if isinstance(tr, dict) else ''
            timeout_details.append((r.name, run_id, error or partial or '-'))

    if timeout_details:
        console.print()
        t_table = Table(title='Timeout Instances', box=box.SIMPLE_HEAD, title_style='bold red')
        t_table.add_column('Benchmark', style='dim')
        t_table.add_column('Instance ID', style='yellow')
        t_table.add_column('Info', style='dim')
        for bench_name, run_id, info in timeout_details[:30]:
            t_table.add_row(bench_name, run_id, info)
        if len(timeout_details) > 30:
            t_table.add_row('...', f'+{len(timeout_details) - 30} more', '')
        console.print(t_table)

    # ---- Per-run breakdown for multi-run ----
    if k > 1:
        pr_rows: list[tuple[str, int, str, str, str]] = []
        for r in successful:
            m = r.metrics or {}
            for idx, run_m in enumerate(m.get('per_run', []), 1):
                ra = run_m.get('accuracy', 0)
                rc = run_m.get('correct', 0)
                re = run_m.get('evaluated', 0)
                rt = run_m.get('timeout', 0)
                pr_rows.append((r.name, idx, f'{rc}/{re}', f'{ra:.1%}', str(rt) if rt else '-'))

        if pr_rows:
            console.print()
            pr_table = Table(title=f'Per-Run Breakdown (k={k})', box=box.SIMPLE_HEAD)
            pr_table.add_column('Benchmark')
            pr_table.add_column('Run', justify='right')
            pr_table.add_column('Correct', justify='right')
            pr_table.add_column('Accuracy', justify='right')
            pr_table.add_column('Timeout', justify='right')
            for row_data in pr_rows:
                pr_table.add_row(*[str(x) for x in row_data])
            console.print(pr_table)

    console.print()


def _extract_request_args_from_env(env_vars: dict[str, str]) -> dict[str, Any]:
    """Extract model request_args-like fields from env vars for summary tracking."""
    return {
        'main': {
            'model': env_vars.get('SEARCH_OPENAI_MODEL'),
            'temperature': env_vars.get('SEARCH_TEMPERATURE'),
            'max_tokens': env_vars.get('SEARCH_MAX_TOKENS'),
            'reasoning_effort': env_vars.get('SEARCH_REASONING_EFFORT'),
            'interleaved_thinking': env_vars.get('SEARCH_INTERLEAVED_THINKING'),
            'thinking_type': env_vars.get('SEARCH_THINKING_TYPE'),
        },
        'judge': {
            'model': env_vars.get('JUDGE_OPENAI_MODEL'),
            'temperature': env_vars.get('JUDGE_TEMPERATURE'),
            'max_tokens': env_vars.get('JUDGE_MAX_TOKENS'),
        },
    }


def build_execution_metadata(
    args: argparse.Namespace,
    env_vars: dict[str, str],
    benchmarks: list[Any],
    timeout: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Build execution metadata for RUN_SUMMARY and Excel export."""
    return {
        'output_dir': str(output_dir),
        'kwargs': {
            'benchmarks': [b.name for b in benchmarks],
            'config': str(args.config) if args.config else None,
            'name': args.name,
            'max_concurrent': args.max_concurrent,
            'timeout': timeout,
            'benchmark_parallel': args.benchmark_parallel,
            'resume': args.resume,
            'runs': args.runs,
            'model_override': args.model,
            'instance_ids': getattr(args, 'instance_ids', None),
            'env_file': str(args.env_file),
            'no_table': args.no_table,
            'verbose': args.verbose,
        },
        'request_args': _extract_request_args_from_env(env_vars),
    }


def save_summary(summary: Any, execution_metadata: dict[str, Any] | None = None) -> Path | None:
    """Save summary to JSON."""
    if not summary.output_dir:
        return None

    by_status = {'success': 0, 'failed': 0, 'skipped': 0}
    for r in summary.results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    successful = [r for r in summary.results if r.status == 'success' and r.metrics]
    k = summary.k
    aggregate_metrics: dict[str, Any] = {}
    if successful:
        if k > 1:
            total_instances = 0
            sum_avg_k = 0.0
            sum_pass_k = 0.0
            sum_mean_acc = 0.0
            total_timeout = 0
            for r in successful:
                m = r.metrics or {}
                instances = m.get('instances', m.get('evaluated', 0))
                total_instances += instances
                sum_avg_k += m.get('avg_k_accuracy', 0.0) * instances
                sum_pass_k += m.get('pass_k_accuracy', 0.0) * instances
                sum_mean_acc += m.get('mean_accuracy', 0.0) * instances
                total_timeout += m.get('timeout', 0)
            aggregate_metrics = {
                'instances': total_instances,
                'avg_k_accuracy': round(sum_avg_k / total_instances, 4) if total_instances > 0 else 0.0,
                'pass_k_accuracy': round(sum_pass_k / total_instances, 4) if total_instances > 0 else 0.0,
                'mean_accuracy': round(sum_mean_acc / total_instances, 4) if total_instances > 0 else 0.0,
                'timeout': total_timeout,
            }
        else:
            total_correct = sum((r.metrics or {}).get('correct', 0) for r in successful)
            total_tasks = sum((r.metrics or {}).get('total', 0) for r in successful)
            aggregate_metrics = {
                'correct': total_correct,
                'total': total_tasks,
                'accuracy': round(total_correct / total_tasks, 4) if total_tasks > 0 else 0.0,
            }

    path = summary.output_dir / 'RUN_SUMMARY.json'
    summary_xlsx_path = summary.output_dir / 'RUN_SUMMARY.xlsx'
    data = {
        'start_time': summary.start_time.isoformat(),
        'end_time': summary.end_time.isoformat() if summary.end_time else None,
        'duration_seconds': summary.duration_seconds,
        'k': summary.k,
        'total': len(summary.results),
        **by_status,
        'aggregate_metrics': aggregate_metrics,
        'execution': execution_metadata or {},
        'benchmarks': [
            {
                'name': r.name,
                'status': r.status,
                'duration': r.duration_seconds,
                'metrics': r.metrics,
                'traj_path': str(r.traj_path) if r.traj_path else None,
                'db_path': str(r.db_path) if r.db_path else None,
                'metrics_path': str(r.metrics_path) if r.metrics_path else None,
                'summary_xlsx_path': str(summary_xlsx_path),
                'execution_kwargs': (execution_metadata or {}).get('kwargs', {}),
                'execution_request_args': (execution_metadata or {}).get('request_args', {}),
                'run_index': r.run_index,
                'total_runs': r.total_runs,
                'error': r.error,
            }
            for r in summary.results
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def save_summary_excel(summary: Any, execution_metadata: dict[str, Any] | None = None) -> Path | None:
    """Save summary to an Excel workbook with benchmark and timeout details."""
    if not summary.output_dir:
        return None

    try:
        import pandas as pd
    except Exception as e:
        logger.warning(f'Excel export skipped: pandas unavailable ({e})')
        return None

    path = summary.output_dir / 'RUN_SUMMARY.xlsx'
    execution_metadata = execution_metadata or {}

    # Sheet 1: benchmark summary
    benchmark_rows = []
    timeout_rows = []
    for r in summary.results:
        m = r.metrics or {}
        benchmark_rows.append({
            'name': r.name,
            'status': r.status,
            'duration_seconds': round(r.duration_seconds, 3),
            'total': m.get('total'),
            'success': m.get('success'),
            'failed': m.get('failed'),
            'timeout': m.get('timeout'),
            'evaluated': m.get('evaluated'),
            'correct': m.get('correct'),
            'accuracy': m.get('accuracy'),
            'avg_k_accuracy': m.get('avg_k_accuracy'),
            'pass_k_accuracy': m.get('pass_k_accuracy'),
            'mean_accuracy': m.get('mean_accuracy'),
            'avg_time_sec': m.get('avg_time_sec'),
            'correct_instance_ids': json.dumps(m.get('correct_instance_ids', []), ensure_ascii=False),
            'incorrect_instance_ids': json.dumps(m.get('incorrect_instance_ids', []), ensure_ascii=False),
            'failed_instance_ids': json.dumps(m.get('failed_instance_ids', []), ensure_ascii=False),
            'timeout_instance_ids': json.dumps(m.get('timeout_instance_ids', []), ensure_ascii=False),
            'db_path': str(r.db_path) if r.db_path else None,
            'metrics_path': str(r.metrics_path) if r.metrics_path else None,
            'traj_path': str(r.traj_path) if r.traj_path else None,
            'run_index': r.run_index,
            'total_runs': r.total_runs,
            'error': r.error,
        })
        for tr in m.get('timeout_runs', []) or []:
            timeout_rows.append({
                'benchmark': r.name,
                'run_id': tr.get('run_id') if isinstance(tr, dict) else str(tr),
                'error': tr.get('error') if isinstance(tr, dict) else None,
                'partial_result': json.dumps(tr.get('partial_result', {}), ensure_ascii=False)
                if isinstance(tr, dict)
                else None,
            })

    # Sheet 2: execution kwargs + request_args
    exec_rows = []
    for k, v in (execution_metadata.get('kwargs') or {}).items():
        exec_rows.append({'section': 'kwargs', 'key': k, 'value': json.dumps(v, ensure_ascii=False)})
    for role, role_args in (execution_metadata.get('request_args') or {}).items():
        for k, v in (role_args or {}).items():
            exec_rows.append({'section': f'request_args.{role}', 'key': k, 'value': json.dumps(v, ensure_ascii=False)})

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(benchmark_rows).to_excel(writer, sheet_name='benchmarks', index=False)
        pd.DataFrame(timeout_rows).to_excel(writer, sheet_name='timeout_runs', index=False)
        pd.DataFrame(exec_rows).to_excel(writer, sheet_name='execution', index=False)

    return path
