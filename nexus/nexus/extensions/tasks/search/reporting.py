from __future__ import annotations

import argparse
import html as html_mod
import json
import logging

from datetime import datetime
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
    judge_status = metrics.get('judge_status', {})
    if judge_status:
        print(
            '  Judge:    '
            f'{judge_status.get("success", 0)} success, '
            f'{judge_status.get("failed", 0)} failed, '
            f'{judge_status.get("missing", 0)} missing'
        )
    max_iter_reached = metrics.get('max_iter_reached', {})
    if max_iter_reached:
        print(f'  MaxIter:  {max_iter_reached.get("count", 0)}/{total} = {max_iter_reached.get("rate", 0):.1%}')
    steps = metrics.get('steps', {})
    if steps:
        print(
            f'  Steps:    mean={steps.get("mean", 0):.1f}, median={steps.get("median", 0):.1f}, '
            f'min={steps.get("min", 0)}, max={steps.get("max", 0)}'
        )
    tool_calls = metrics.get('tool_calls', {})
    if tool_calls:
        print(f'  ToolCall: total={metrics.get("total_tool_calls", 0)}, mean={tool_calls.get("mean", 0):.1f}/task')
    token_usage = metrics.get('token_usage', {})
    if token_usage:
        avg_per_inst, _ = _token_inst_and_step(token_usage, steps)
        avg_comp_inst = (token_usage.get('completion_tokens_per_instance') or {}).get('mean')
        parts = []
        if avg_comp_inst is not None:
            parts.append(f'avg_comp={float(avg_comp_inst):,.0f}')
        if avg_per_inst is not None:
            parts.append(f'avg_total={float(avg_per_inst):,.0f}')
        context_resets = token_usage.get('total_context_resets', 0)
        if context_resets:
            parts.append(f'ctx_resets={context_resets}')
        if parts:
            print(f'  Tokens:   {", ".join(parts)}')

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
    """Build execution metadata for RUN_SUMMARY exports."""
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
        'agent_config': {
            k: v
            for k, v in {
                'agent_name': env_vars.get('SEARCH_AGENT_NAME'),
                'max_iterations': env_vars.get('SEARCH_MAX_ITERATIONS'),
                'max_llm_retries': env_vars.get('SEARCH_AGENT_MAX_LLM_RETRIES'),
                'toolset_version': env_vars.get('SEARCH_TOOLSET_VERSION'),
                'keep_recent_k': env_vars.get('SEARCH_KEEP_RECENT_K'),
                'max_context_tokens': env_vars.get('SEARCH_MAX_CONTEXT_TOKENS'),
                'eval_mode': env_vars.get('SEARCH_EVAL_MODE'),
            }.items()
            if v is not None
        },
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


# ============================================================================
# HTML Report
# ============================================================================

_HTML_CSS = (
    ':root{--p:rgba(18,27,51,.86);--b:rgba(151,172,255,.18);--t:#edf2ff;--m:#a6b5d6;--a:#7c9cff}'
    '*{box-sizing:border-box}'
    'body{margin:0;padding:28px;font-family:"Segoe UI","PingFang SC",sans-serif;color:var(--t);'
    'background:radial-gradient(circle at top left,rgba(124,156,255,.22),transparent 26%),'
    'radial-gradient(circle at top right,rgba(77,226,197,.15),transparent 22%),'
    'linear-gradient(180deg,#09101d,#0f1730)}'
    '.wrap{max-width:1400px;margin:0 auto}'
    '.hero{padding:24px 28px;border:1px solid var(--b);border-radius:20px;'
    'background:linear-gradient(135deg,rgba(27,42,82,.94),rgba(14,22,43,.92));margin-bottom:20px}'
    '.hero h1{margin:0 0 6px;font-size:30px}.hero p{margin:4px 0;color:var(--m);font-size:14px}'
    '.panel{padding:20px;border:1px solid var(--b);border-radius:18px;background:var(--p);'
    'margin-bottom:18px;overflow-x:auto}'
    'table{width:100%;border-collapse:collapse;min-width:900px}'
    'th,td{padding:9px 12px;border-bottom:1px solid rgba(151,172,255,.12);text-align:left;'
    'font-size:13px;white-space:nowrap}'
    'th{color:var(--m);font-weight:600;position:sticky;top:0;background:var(--p)}'
    'tr.agg td{font-weight:700;border-top:2px solid var(--a)}'
    '.g{color:#6fdd9a}.y{color:#ffd36e}.r{color:#ff8d8d}.d{color:var(--m)}'
)


def _v(val: Any, fmt: str = '') -> str:
    if val is None or val == '':
        return '<span class="d">-</span>'
    if fmt == 'pct':
        return f'{float(val):.1%}'
    if fmt == 'int':
        return f'{int(val):,}'
    if isinstance(val, float):
        return f'{val:,.1f}' if abs(val) >= 10 else f'{val:.2f}'
    return str(val)


def _token_inst_and_step(token_usage: dict[str, Any], steps: dict[str, Any]) -> tuple[Any, Any]:
    """Return token mean per instance / step with backward-compatible fallbacks."""
    per_inst = (token_usage.get('total_tokens_per_instance') or {}).get('mean')
    if per_inst is None:
        per_inst = token_usage.get('mean_tokens_per_instance')

    per_step = token_usage.get('avg_tokens_per_step')
    if per_step is None:
        step_count = float(steps.get('count', 0) or 0)
        step_mean = float(steps.get('mean', 0) or 0)
        total_steps = step_count * step_mean
        total_tokens = float(token_usage.get('total_tokens', 0) or 0)
        if total_steps > 0:
            per_step = round(total_tokens / total_steps, 1)

    return per_inst, per_step


def _html_row(cells: list[str], tag: str = 'td', cls: str = '') -> str:
    """Build a <tr> from a list of cell contents. Guarantees column count consistency."""
    tr_attr = f' class="{cls}"' if cls else ''
    return f'<tr{tr_attr}>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'


def save_summary_html(summary: Any, execution_metadata: dict[str, Any] | None = None) -> Path | None:
    """Save benchmark summary as a standalone HTML report."""
    if not summary.output_dir:
        return None

    execution_metadata = execution_metadata or {}
    req = (execution_metadata.get('request_args') or {}).get('main', {})
    kw = execution_metadata.get('kwargs') or {}
    model = req.get('model') or '?'
    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    k = summary.k
    multi = k > 1
    successful = [r for r in summary.results if r.status == 'success' and r.metrics]

    # Table column definitions (single source of truth)
    headers: list[str] = ['Benchmark', 'Correct', 'Accuracy']
    if multi:
        headers += ['avg@k', 'pass@k']
    headers += [
        'Success',
        'Failed',
        'Timeout',
        'MaxIter',
        'Steps(mean)',
        'Steps(med)',
        'ToolCalls(mean)',
        'Avg Comp Token',
        'Avg Total Token',
        'CtxReset',
        'Judge',
        'AvgTime',
        'Duration',
    ]

    # Accumulate aggregate values
    agg_evaluated = agg_correct = agg_success = agg_failed = agg_timeout = agg_tokens = agg_ctx_resets = 0
    agg_completion_tokens = 0
    agg_instances_with_usage = 0
    agg_judge_ok = agg_judge_total = 0
    step_means: list[float] = []
    tc_means: list[float] = []

    rows: list[str] = []
    for r in successful:
        m = r.metrics or {}
        # In multi-run mode, 'evaluated' (= instances) is the correct denominator
        # for accuracy, not 'total' (which counts all k attempts).
        evaluated = m.get('evaluated', m.get('total', 0))
        correct = m.get('correct', 0)
        acc = m.get('accuracy', 0)
        failed_cnt = m.get('failed', 0)
        timeout_cnt = m.get('timeout', 0)
        steps = m.get('steps', {})
        tool_calls = m.get('tool_calls', {})
        token = m.get('token_usage', {})
        max_iter = m.get('max_iter_reached', {})
        tok_total = token.get('total_tokens', 0)
        avg_time = m.get('avg_time_sec')

        success_cnt = m.get('success', 0)

        agg_evaluated += evaluated
        agg_correct += correct
        agg_success += success_cnt
        agg_failed += failed_cnt
        agg_timeout += timeout_cnt
        agg_tokens += tok_total
        agg_completion_tokens += int(token.get('total_completion_tokens', 0) or 0)
        agg_instances_with_usage += int(token.get('instances_with_usage', 0) or 0)
        agg_ctx_resets += token.get('total_context_resets', 0)
        agg_judge_ok += (m.get('judge_status') or {}).get('success', 0)
        agg_judge_total += (m.get('judge_status') or {}).get('success', 0) + (m.get('judge_status') or {}).get(
            'failed', 0
        )
        if steps.get('mean') is not None:
            step_means.append(float(steps['mean']))
        if tool_calls.get('mean') is not None:
            tc_means.append(float(tool_calls['mean']))

        acc_cls = 'g' if acc >= 0.5 else 'y' if acc >= 0.2 else 'r'
        mi_str = f'{max_iter.get("count", 0)} ({_v(max_iter.get("rate", 0), "pct")})' if max_iter else '-'
        token_inst, _ = _token_inst_and_step(token, steps)
        comp_inst = (token.get('completion_tokens_per_instance') or {}).get('mean')
        judge = m.get('judge_status') or {}
        j_ok, j_fail = judge.get('success', 0), judge.get('failed', 0)
        judge_str = f'{j_ok}/{j_ok + j_fail}' if (j_ok + j_fail) > 0 else _v(None)

        cells: list[str] = [
            html_mod.escape(r.name),
            f'{_v(correct)}/{_v(evaluated)}',
            f'<span class="{acc_cls}">{_v(acc, "pct")}</span>',
        ]
        if multi:
            cells += [
                f'<span class="{acc_cls}">{_v(m.get("avg_k_accuracy"), "pct")}</span>',
                _v(m.get('pass_k_accuracy'), 'pct'),
            ]
        cells += [
            _v(success_cnt),
            _v(failed_cnt),
            _v(timeout_cnt),
            mi_str,
            _v(steps.get('mean')),
            _v(steps.get('median')),
            _v(tool_calls.get('mean')),
            _v(comp_inst),
            _v(token_inst),
            _v(token.get('total_context_resets', 0)),
            judge_str,
            f'{avg_time:.0f}s' if avg_time else '-',
            format_duration(r.duration_seconds),
        ]
        rows.append(_html_row(cells))

    # Aggregate row
    agg_acc = agg_correct / agg_evaluated if agg_evaluated else 0
    agg_step = round(sum(step_means) / len(step_means), 1) if step_means else None
    agg_tc = round(sum(tc_means) / len(tc_means), 1) if tc_means else None
    agg_token_inst = round(agg_tokens / agg_instances_with_usage, 1) if agg_instances_with_usage else None
    agg_comp_inst = round(agg_completion_tokens / agg_instances_with_usage, 1) if agg_instances_with_usage else None
    agg_cells: list[str] = ['AGGREGATE', f'{_v(agg_correct)}/{_v(agg_evaluated)}', _v(agg_acc, 'pct')]
    if multi:
        agg_cells += ['', '']
    agg_cells += [
        _v(agg_success),
        _v(agg_failed),
        _v(agg_timeout),
        '',
        _v(agg_step),
        '',
        _v(agg_tc),
        _v(agg_comp_inst),
        _v(agg_token_inst),
        _v(agg_ctx_resets),
        f'{agg_judge_ok}/{agg_judge_total}' if agg_judge_total else _v(None),
        '',
        format_duration(summary.duration_seconds),
    ]
    rows.append(_html_row(agg_cells, cls='agg'))

    # Failed benchmark rows (span all columns)
    ncols = len(headers)
    for r in summary.results:
        if r.status == 'failed':
            err = html_mod.escape((r.error or 'unknown')[:120])
            rows.append(f'<tr><td>{html_mod.escape(r.name)}</td><td class="r" colspan="{ncols - 1}">{err}</td></tr>')

    # Build hero config lines from execution_metadata
    def _cfg_line(label: str, pairs: list[tuple[str, Any]]) -> str:
        parts = [f'{k}={v}' for k, v in pairs if v is not None]
        return f'<p><b>{label}:</b> {html_mod.escape(", ".join(parts))}</p>' if parts else ''

    judge = (execution_metadata.get('request_args') or {}).get('judge', {})
    agent_cfg = execution_metadata.get('agent_config') or {}

    cfg_html = _cfg_line(
        'Config',
        [
            ('temp', req.get('temperature')),
            ('max_tokens', req.get('max_tokens')),
            ('reasoning', req.get('reasoning_effort')),
            ('thinking', req.get('thinking_type')),
            ('interleaved', req.get('interleaved_thinking')),
            ('timeout', f'{kw["timeout"]}s' if kw.get('timeout') else None),
            ('concurrency', kw.get('max_concurrent')),
            ('runs', kw.get('runs')),
            ('resume', kw.get('resume') or None),
        ],
    )
    judge_html = _cfg_line(
        'Judge',
        [
            ('model', judge.get('model')),
            ('temp', judge.get('temperature')),
        ],
    )
    agent_html = _cfg_line('Agent', [(k, v) for k, v in agent_cfg.items()])

    k_label = f' (k={k})' if multi else ''
    header_html = ''.join(f'<th>{h}</th>' for h in headers)
    page = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark Report</title><style>{_HTML_CSS}</style></head><body><div class="wrap">
<section class="hero"><h1>Search Benchmark Report{html_mod.escape(k_label)}</h1>
<p><b>Model:</b> {html_mod.escape(str(model))} &nbsp; <b>Generated:</b> {generated}
 &nbsp; <b>Duration:</b> {format_duration(summary.duration_seconds)}</p>
<p><b>Output:</b> {html_mod.escape(str(summary.output_dir))}</p>
{cfg_html}
{judge_html}
{agent_html}</section>
<section class="panel"><table>
<thead><tr>{header_html}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></section>
</div></body></html>"""

    path = summary.output_dir / 'RUN_SUMMARY.html'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding='utf-8')
    except Exception as e:
        logger.warning(f'HTML export failed: {e}')
        return None
    return path
