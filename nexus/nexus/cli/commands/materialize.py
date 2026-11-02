from __future__ import annotations

import argparse
import asyncio
import json
import sys

from contextlib import redirect_stdout
from pathlib import Path

from nexus.cli.commands.run import add_compose_arguments, has_compose_args
from nexus.cli.commands.submit import get_local_max_task_list_bytes
from nexus.dataset.item import resolve_task_files_input
from nexus.distributed.batch.materialize import MaterializedBatchArtifacts, materialize_compose, materialize_generator
from nexus.runner.compose import parse_ordered_parts
from nexus.runner.compose.compose_hooks import parse_compose_hooks
from nexus.runner.compose.model import RawRunPart, RunComposition
from nexus.runner.compose.parser import load_run_file, parse_run_composition
from nexus.utils.log import setup_logging


def register_materialize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('materialize', help='Materialize a generator or composed tasks for local review')
    parser.add_argument('config', nargs='?', type=Path, help='Compose experiment or RunInput YAML/JSON file')
    parser.add_argument('--generator', type=Path, help='Batch generator Python file')
    add_compose_arguments(parser, include_batch=True)
    parser.add_argument('--output', type=Path, required=True, help='Output Task List JSONL file')
    parser.add_argument(
        '--task-list-version',
        type=int,
        choices=(1, 2),
        default=2,
        help='Task List version (default: 2); compose upgrades to v3 when packaging local task files',
    )
    parser.add_argument(
        '--data-list-output',
        type=Path,
        help='Compose task-files container; defaults to the output path with a .data-list.bin suffix',
    )


def _validate_source(args: argparse.Namespace) -> None:
    if sum((args.generator is not None, args.config is not None, has_compose_args(args))) != 1:
        raise ValueError('provide exactly one of --generator, a config file, or --data/--use/--set/--set-json')
    if args.generator is not None and any(
        value is not None for value in (args.split, args.limit, args.data_list_output)
    ):
        raise ValueError('--split, --limit and --data-list-output require compose or a config file')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be a positive integer')
    if args.data_list_output is not None and args.task_list_version != 2:
        raise ValueError('--data-list-output requires --task-list-version 2')


def _load_composition(args: argparse.Namespace) -> RunComposition:
    if args.config is None:
        return RunComposition(parts=parse_ordered_parts(getattr(args, '_raw_argv', [])))
    data, source_path = load_run_file(args.config)
    if 'compose' in data:
        return parse_run_composition(data, source_path=source_path)
    if 'body' in data:
        raise ValueError(f'{source_path} is a RunPart profile; pass it with --use')
    if 'task_files' in data:
        data['task_files'] = resolve_task_files_input(data['task_files'], base_dir=source_path.parent)
    return RunComposition(
        parts=[
            RawRunPart(kind='input', value=data, source=str(source_path)),
        ]
    )


def _materialize(args: argparse.Namespace) -> MaterializedBatchArtifacts:
    _validate_source(args)
    output = args.output.expanduser().resolve()
    compose_hooks = parse_compose_hooks(args.hook_values)
    if args.generator is not None:
        return MaterializedBatchArtifacts(
            task_list=materialize_generator(
                args.generator.expanduser().resolve(),
                output,
                max_items=None,
                max_bytes=get_local_max_task_list_bytes(),
                max_line_bytes=None,
                task_list_version=args.task_list_version,
                compose_hooks=compose_hooks,
            )
        )
    composition = _load_composition(args)
    limit = args.limit if args.limit is not None else composition.limit
    if limit is not None and limit <= 0:
        raise ValueError('data.limit must be a positive integer')
    return asyncio.run(
        materialize_compose(
            composition.parts,
            output,
            split=args.split if args.split is not None else composition.split,
            item_id=composition.item_id,
            limit=limit,
            max_items=None,
            max_bytes=get_local_max_task_list_bytes(),
            max_line_bytes=None,
            task_list_version=args.task_list_version,
            auto_data_list=args.task_list_version == 2,
            data_list_output_path=(
                args.data_list_output.expanduser().resolve()
                if args.data_list_output is not None
                else output.with_suffix('.data-list.bin')
            ),
            reject_local_task_files=args.task_list_version == 1,
            compose_hooks=compose_hooks,
        )
    )


def materialize_command(args: argparse.Namespace) -> int:
    try:
        # Keep stdout machine-readable even when profiles/readers/generators emit diagnostics.
        with redirect_stdout(sys.stderr):
            setup_logging(compat_tqdm=False)
            materialized = _materialize(args)
    except Exception as exc:
        print(f'materialize failed: {exc}', file=sys.stderr)
        return 1

    result = materialized.task_list
    data_list = materialized.data_list
    summary = {
        'path': str(result.path),
        'record_version': result.record_version,
        'sha256': result.sha256,
        'size_bytes': result.size_bytes,
        'total_items': result.total_items,
    }
    if data_list is not None:
        summary['data_list'] = {
            'path': str(data_list.path),
            'sha256': data_list.sha256,
            'size_bytes': data_list.size_bytes,
            'frame_count': data_list.frame_count,
        }
    print(json.dumps(summary, sort_keys=True))
    return 0
