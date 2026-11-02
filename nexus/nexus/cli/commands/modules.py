"""Module management CLI commands."""

from __future__ import annotations

import argparse
import json
import sys

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from nexus.core.module_dependencies import ModuleSelector, selectors_from_run_input
from nexus.distributed.submission.input import InputNormalizationError, normalize_run_input


# Category display order (same as TUI)
CATEGORY_ORDER = [
    'Agent',
    'Task',
    'Project',
    'LLM Client',
    'Tool',
    'Executor',
    'Provider',
    'Processor',
    'Other',
]

CATEGORY_COLOR: dict[str, str] = {
    'Agent': 'dodger_blue1',
    'Task': 'green3',
    'Tool': 'dark_orange',
    'LLM Client': 'medium_purple1',
    'Executor': 'cyan3',
    'Provider': 'steel_blue1',
    'Processor': 'grey70',
    'Project': 'gold1',
    'Other': 'white',
}

_OBJ_TYPE_LABEL = {
    'agents': 'agent',
    'tasks': 'task',
    'tools': 'tool',
    'executors': 'executor',
    'openai_clients': 'client',
    'runtime_providers': 'provider',
    'span_processors': 'processor',
    'run_hooks': 'run hook',
    'run_parts': 'run part',
    'compose_hooks': 'compose hook',
    'dataset_sources': 'dataset source',
    'commands': 'command',
}


def _obj_summary(objects: dict[str, dict[str, str]]) -> str:
    """Format a module's registered objects as a compact summary."""
    if not objects:
        return ''
    parts = []
    for key, obj_map in objects.items():
        n = len(obj_map)
        label = _OBJ_TYPE_LABEL.get(key, key)
        if n > 1:
            label += 's'
        parts.append(f'{n} {label}')
    return ' + '.join(parts)


def _dep_col(total: int, missing: int) -> str:
    """Format a dependency column with color."""
    if total == 0:
        return '[green]✓[/green]'
    ok = total - missing
    text = f'{ok}/{total}'
    if missing:
        return f'[red]{text} ✗[/red]'
    return f'[green]{text} ✓[/green]'


def list_and_check(args: argparse.Namespace) -> int:
    """List all modules and check their dependency status."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from nexus.core.dep_checker import check_module
    from nexus.core.module_meta import discover_all_modules

    console = Console()
    modules = discover_all_modules()

    # Check and group by category
    statuses = [(m, check_module(m)) for m in modules]
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for m, s in statuses:
        grouped[m.category].append((m, s))

    has_failures = False

    table = Table(show_header=True, header_style='bold', box=None, pad_edge=False, show_edge=False)
    table.add_column('Module', min_width=24)
    table.add_column('Host', justify='center', min_width=10)
    table.add_column('Runtime', justify='center', min_width=10)
    table.add_column('Objects', min_width=16)

    for cat in CATEGORY_ORDER:
        if cat not in grouped:
            continue
        entries = grouped[cat]
        color = CATEGORY_COLOR.get(cat, 'white')

        # Category header row
        table.add_row()
        table.add_row(
            Text.from_markup(f'[bold {color}]{cat}[/bold {color}] [dim]({len(entries)})[/dim]'),
            '',
            '',
            '',
        )

        for m, s in entries:
            h_total = len(m.base_deps) + len(m.host_extras)
            h_missing = len(s.missing_base) + len(s.missing_host_extras)
            r_total = len(m.runtime_extras)
            r_missing = len(s.missing_runtime_extras)

            if s.missing_host:
                has_failures = True

            obj = _obj_summary(m.objects)

            table.add_row(
                f'  {m.name}',
                Text.from_markup(_dep_col(h_total, h_missing)),
                Text.from_markup(_dep_col(r_total, r_missing)),
                f'[dim]{obj}[/dim]' if obj else '',
            )

    console.print(table)
    console.print()

    # Summary
    total = len(statuses)
    ok_count = sum(1 for _, s in statuses if s.ok)
    failed_count = total - ok_count
    if failed_count:
        console.print(f'[bold]{total}[/bold] modules, [green]{ok_count} OK[/green], [red]{failed_count} missing[/red]')
    else:
        console.print(f'[bold]{total}[/bold] modules, [green]all OK[/green]')

    return 1 if has_failures else 0


def install_modules(args: argparse.Namespace) -> int:
    """Install missing dependencies for specified modules."""
    from nexus.core.dep_checker import check_module, install_packages
    from nexus.core.module_meta import discover_all_modules

    modules = discover_all_modules()

    if args.module:
        selected = [m for m in modules if m.name in args.module or m.module_path in args.module]
        if not selected:
            print(f'No modules found matching: {", ".join(args.module)}')
            return 1
    elif args.all:
        selected = modules
    else:
        print('Specify module name(s) or use --all')
        print('  nexus modules install <module> [<module>...]')
        print('  nexus modules install --all')
        return 1

    # Collect missing packages
    packages: list[str] = []
    for m in selected:
        status = check_module(m)
        if args.runtime:
            packages.extend(status.missing_runtime)
        else:
            packages.extend(status.missing_host)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in packages:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        scope = 'host + runtime' if args.runtime else 'host'
        print(f'All {scope} dependencies already installed.')
        return 0

    print(f'Installing {len(unique)} packages: {" ".join(unique)}')
    ok, msg = install_packages(unique)
    if ok:
        print('Done.')
        return 0
    print(f'Install failed: {msg}')
    return 1


def resolve_module_wheel_selectors(
    *,
    run_input_path: Path | None,
    task_list_path: Path | None,
    max_line_bytes: int,
) -> tuple[str | None, tuple[ModuleSelector, ...]]:
    if (run_input_path is None) == (task_list_path is None):
        raise ValueError('exactly one of run_input_path and task_list_path is required')
    if task_list_path is not None:
        from nexus.distributed.batch.task_list import get_local_max_batch_item_metadata_bytes
        from nexus.distributed.repositories.batch import read_representative_run_input

        representative = read_representative_run_input(
            task_list_path,
            max_line_bytes=max_line_bytes,
            max_metadata_bytes=get_local_max_batch_item_metadata_bytes(),
        )
        if representative is None:
            raise ValueError('module wheel Task List is empty')
        return representative.item_id, selectors_from_run_input(representative.run_input)

    import yaml

    assert run_input_path is not None
    payload = yaml.safe_load(run_input_path.read_text(encoding='utf-8'))
    if not isinstance(payload, Mapping):
        raise ValueError('module wheel RunInput must be a mapping')
    try:
        normalized = normalize_run_input(dict(payload))
    except InputNormalizationError as exc:
        raise ValueError(f'invalid module wheel RunInput: {exc}') from exc
    return None, selectors_from_run_input(normalized)


def wheel_modules(args: argparse.Namespace) -> int:
    from nexus.distributed.dependency_build.local import build_local_module_wheelhouse

    try:
        representative_item_id, selectors = resolve_module_wheel_selectors(
            run_input_path=Path(args.run_input) if args.run_input else None,
            task_list_path=Path(args.task_list) if args.task_list else None,
            max_line_bytes=args.max_line_bytes,
        )
        report = build_local_module_wheelhouse(
            project_root=Path(args.project_root),
            uv_lock=Path(args.lock),
            nexus_wheel=Path(args.nexus_wheel),
            output=Path(args.output),
            selectors=selectors,
            representative_item_id=representative_item_id,
            extras=args.extra,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Module wheel build failed: {exc}', file=sys.stderr)
        return 1
    print(f'Module wheelhouse written to {Path(args.output).expanduser().resolve(strict=False)}')
    print(json.dumps(report, sort_keys=True))
    return 0


def handle_modules_command(args: argparse.Namespace) -> int:
    """Dispatch modules subcommands."""
    if getattr(args, 'tui', False):
        from nexus.cli.modules_tui import run_tui

        run_tui()
        return 0

    sub = getattr(args, 'modules_command', None)
    if sub == 'install':
        return install_modules(args)
    if sub == 'wheel':
        return wheel_modules(args)

    # Default: list + check
    return list_and_check(args)


def register_modules_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the modules subcommand."""
    modules_parser = subparsers.add_parser(
        'modules',
        help='List, check, and install extension module dependencies',
    )
    modules_parser.add_argument('--tui', action='store_true', help='Launch interactive TUI')
    modules_sub = modules_parser.add_subparsers(dest='modules_command')

    # modules install
    install_parser = modules_sub.add_parser('install', help='Install missing dependencies')
    install_parser.add_argument('module', nargs='*', help='Module name(s) to install deps for')
    install_parser.add_argument('--all', action='store_true', help='Install deps for all modules')
    install_parser.add_argument('--runtime', action='store_true', help='Also install runtime dependencies')

    wheel_parser = modules_sub.add_parser('wheel', help='Build an offline wheelhouse for selected modules')
    source_group = wheel_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--run-input', help='RunInput YAML or JSON file')
    source_group.add_argument('--task-list', help='Task List JSONL file; only the first record is analyzed')
    wheel_parser.add_argument('--project-root', default='.', help='Source Package project root (default: current dir)')
    wheel_parser.add_argument('--lock', required=True, help='Existing uv.lock to preserve')
    wheel_parser.add_argument('--nexus-wheel', required=True, help='Exact nexus-agent wheel to analyze')
    wheel_parser.add_argument('--output', required=True, help='Output directory for wheels, lock, and report')
    wheel_parser.add_argument('--extra', action='append', default=[], help='Additional project extra (repeatable)')
    wheel_parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024, help=argparse.SUPPRESS)

    modules_parser.set_defaults(_parser=modules_parser)
