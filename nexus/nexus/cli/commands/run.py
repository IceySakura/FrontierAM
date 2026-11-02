"""Run-related CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import logging

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nexus.utils.common import pformat_for_display
from nexus.utils.log import setup_logging


if TYPE_CHECKING:
    from nexus.runner.compose import RunPart


logger = logging.getLogger(__name__)


class _OptionalBoolAction(argparse.Action):
    """Accept ``--flag``, ``--flag=true``, and ``--flag=false``.

    ``nargs='?'`` consumes the next token when it is present. Callers that put
    the flag before a config path must use ``--use-gateway=true``.
    """

    def __init__(self, option_strings: list[str], dest: str, default: bool = False, **kwargs: Any) -> None:
        kwargs.pop('type', None)
        super().__init__(option_strings, dest, nargs='?', const=True, default=default, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if values is None or values is True:
            setattr(namespace, self.dest, True)
            return
        if values is False:
            setattr(namespace, self.dest, False)
            return
        normalized = str(values).strip().lower()
        if normalized in {'1', 'true', 'yes', 'y'}:
            setattr(namespace, self.dest, True)
            return
        if normalized in {'0', 'false', 'no', 'n'}:
            setattr(namespace, self.dest, False)
            return
        parser.error(f'{option_string} expected a boolean, got {values!r}')


def _add_use_gateway_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--use-gateway',
        action=_OptionalBoolAction,
        help='Wrap openai_clients.main as a gateway client, inject gateway_trajectory, and fill a missing '
        'trajectory_id from run_id. Accepts --use-gateway or --use-gateway=true. Use the equals form when '
        'the flag appears before the config path.',
    )


def _resolve_trim_threshold(args: argparse.Namespace) -> tuple[int, int]:
    """Build the ``(head, tail)`` trim_threshold tuple from CLI args.

    ``--log-trim N`` sets both head and tail to *N*; the per-side flags then
    override either end. Unset flags fall back to 200.
    """
    both = getattr(args, 'log_trim', None)
    default = 200 if both is None else int(both)
    head_raw = getattr(args, 'log_trim_head', None)
    tail_raw = getattr(args, 'log_trim_tail', None)
    head = default if head_raw is None else int(head_raw)
    tail = default if tail_raw is None else int(tail_raw)
    return max(head, 0), max(tail, 0)


def run_single(args: argparse.Namespace) -> int:
    """Run a single task from a YAML configuration file."""
    from nexus.runner.single import run, write_run_output

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level, trim_threshold=_resolve_trim_threshold(args))

    if _has_compose_args(args):
        if args.config:
            logger.error('Config file positional argument cannot be combined with --data/--use/--set')
            return 1
        try:
            from nexus.runner.compose import compile_single, parse_ordered_parts

            run_input = compile_single(
                parse_ordered_parts(getattr(args, '_raw_argv', [])),
                item_id=args.id,
                split=args.split,
            )
        except Exception as e:
            logger.error('Invalid composed run: %s', e)
            return 1
    else:
        if not args.config:
            logger.error('Config file is required unless --data/--use/--set is provided')
            return 1
        try:
            from nexus.runner.compose import compile_single_file

            run_input = compile_single_file(args.config, item_id=args.id, split=args.split)
        except FileNotFoundError:
            logger.error('Config file not found: %s', args.config)
            return 1
        except Exception as e:
            logger.error('Invalid config file: %s', e)
            return 1

    if args.hold is not None:
        run_input.runner.hold_after_task = args.hold
        if args.output:
            run_input.runner.early_output_path = args.output
    if getattr(args, 'use_gateway', False):
        run_input.runner.use_gateway = True

    result = asyncio.run(run(run_input))
    # ``pformat_for_display`` respects the ``__pformat_exclude__`` convention on
    # task output models. Task output classes may declare
    # ``__pformat_exclude__: ClassVar[frozenset[str]] = frozenset({'artifact_id'})``
    # (or any other field name) to hide noisy internal handles from the demo
    # log; the full model is still written to ``--output`` and available to
    # programmatic consumers via ``result``.
    logger.info('Task output:\n%s', pformat_for_display(result.task_output))

    # Save output to JSON file if specified
    if args.output:
        write_run_output(args.output, result)
        logger.info('Output saved to: %s', Path(args.output))

    return 0
    # except Exception as e:
    #     logger.error('Error running task: %s', e)
    #     return 1


def _prepare_generator_overrides(args: argparse.Namespace) -> Callable[[], list[RunPart]]:
    """Validate sources and defer overlay resolution until generation is needed."""
    from nexus.runner.compose import parse_ordered_parts
    from nexus.runner.compose.resolver import resolve_non_data_part

    if args.data:
        raise ValueError('--generator cannot be combined with --data')
    parts = parse_ordered_parts(getattr(args, '_raw_argv', []))
    if any(part.kind == 'use' and (part.value == 'data' or part.value.startswith('data:')) for part in parts):
        raise ValueError('--generator cannot be combined with --use data:...')

    def resolve() -> list[RunPart]:
        try:
            return [resolve_non_data_part(part) for part in parts]
        except Exception as exc:
            raise ValueError(f'Invalid generator override: {exc}') from exc

    return resolve


def run_batch(args: argparse.Namespace) -> int:
    """Run a batch of tasks from a config generator file."""
    from nexus.runner.batch import BatchConfig
    from nexus.runner.batch import run_batch as run_batch_impl

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level, trim_threshold=_resolve_trim_threshold(args))

    use_compose = _has_compose_args(args)
    try:
        from nexus.runner.compose.compose_hooks import parse_compose_hooks

        compose_hooks = parse_compose_hooks(args.hook_values)
    except Exception as e:
        logger.error('Invalid --hook: %s', e)
        return 1

    # Validate the batch source/resume inputs.
    if not use_compose and not args.config and not args.generator and not args.database:
        logger.error('At least one config, --generator, or --database must be provided')
        return 1

    if args.config and use_compose:
        logger.error('Config file positional argument cannot be combined with --data/--use/--set')
        return 1
    if args.config and args.generator:
        logger.error('Config file cannot be combined with --generator')
        return 1

    if args.enable_conflict_groups and args.worker_processes > 1:
        logger.error('--enable-conflict-groups currently requires --worker-processes 1')
        return 1

    if args.limit is not None and args.limit < 0:
        logger.error('--limit must be non-negative')
        return 1

    # Handle generator
    generator_path = None
    generator_overrides = None
    if args.generator:
        generator_path = Path(args.generator)
        if not generator_path.exists():
            logger.error('Generator file not found: %s', generator_path)
            return 1
        if not generator_path.is_file():
            logger.error('Generator must be a Python file: %s', generator_path)
            return 1
        if generator_path.suffix != '.py':
            logger.error(f'Generator file must be a Python file (.py): {generator_path}')
            return 1
        if use_compose:
            try:
                generator_overrides = _prepare_generator_overrides(args)
            except Exception as e:
                logger.error('Invalid generator override: %s', e)
                return 1

    # Handle database
    db_path = args.database if args.database else ':memory:'
    if args.worker_processes > 1 and db_path == ':memory:':
        logger.error('--worker-processes requires --database because shard workers need file-backed databases')
        return 1

    batch_items = None
    if args.config:
        try:
            from nexus.runner.compose import compile_batch_file

            batch_items = compile_batch_file(
                args.config, split=args.split, limit=args.limit, compose_hooks=compose_hooks
            )
        except Exception as e:
            logger.error('Invalid batch config file: %s', e)
            return 1
    elif use_compose and not args.generator:
        try:
            from nexus.runner.compose import compile_batch, parse_ordered_parts

            batch_items = compile_batch(
                parse_ordered_parts(getattr(args, '_raw_argv', [])),
                split=args.split,
                limit=args.limit,
                compose_hooks=compose_hooks,
            )
        except Exception as e:
            logger.error('Invalid composed batch: %s', e)
            return 1

    # Build batch config from CLI args
    batch_config = BatchConfig(
        max_concurrent=args.max_concurrent,
        worker_processes=args.worker_processes,
        fail_fast=args.fail_fast,
        retry_failed=args.retry_failed,
        resume_workflow_on_retry=args.resume_workflow_on_retry,
        timeout_per_run=args.timeout,
        resume=args.resume,
        retry_only=args.retry_only,
        enable_trace_db='trace' not in args.no_persist,
        enable_log_capture='log' not in args.no_persist,
        enable_conflict_groups=args.enable_conflict_groups,
        use_gateway=bool(getattr(args, 'use_gateway', False)),
    )

    # Prepare export parameters
    export_format = args.export_format if hasattr(args, 'export_format') else None
    export_path = None
    if export_format:
        # Use user-specified path if provided
        if hasattr(args, 'export_path') and args.export_path:
            export_path = Path(args.export_path)
        # Otherwise, auto-generate path if not :memory:
        elif db_path != ':memory:':
            output_dir = Path(db_path).parent
            if export_format == 'jsonl':
                export_path = output_dir / 'results.jsonl'
            elif export_format == 'csv':
                export_path = output_dir / 'results.csv'
            elif export_format == 'full':
                export_path = output_dir / 'export_full'
        else:
            # :memory: database requires explicit export path
            logger.error('--export-path is required when using --export-format with :memory: database')
            return 1

    try:
        summary = asyncio.run(
            run_batch_impl(
                generator_path=generator_path,
                db_path=db_path,
                batch_config=batch_config,
                export_format=export_format,
                export_path=export_path,
                batch_items=batch_items,
                generator_overrides=generator_overrides,
                generator_compose_hooks=compose_hooks if args.generator else None,
                generator_limit=args.limit if args.generator else None,
            )
        )

        logger.info('\nBatch completed:')
        logger.info('  Total: %s', summary.total)
        logger.info('  Success: %s', summary.success)
        logger.info('  Failed: %s', summary.failed)
        if summary.timeout > 0:
            logger.info('  Timeout: %s', summary.timeout)
        if summary.failed_runs:
            logger.info('  Failed runs: %s', ', '.join(summary.failed_runs[:10]))
            if len(summary.failed_runs) > 10:
                logger.info('  ... and %s more', len(summary.failed_runs) - 10)

        if db_path != ':memory:':
            logger.info('\nDatabase: %s', db_path)
            if export_format:
                logger.info('Results exported to: %s', export_path)
            elif batch_config.worker_processes > 1:
                shard_root = Path(db_path).with_name(f'{Path(db_path).name}.shards')
                logger.info('Shard databases: %s', shard_root)
                logger.info('Use "nexus export %s" to export aggregated shard results', db_path)
            else:
                logger.info('Use "nexus export %s" to export results', db_path)

        return 0 if summary.failed == 0 else 1

    except Exception as e:
        logger.error('Error running batch: %s', e, exc_info=True)
        return 1


def add_run_single_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by ``run-single`` and ``run single``."""
    parser.add_argument(
        'config',
        nargs='?',
        help='Path to the YAML configuration file',
    )
    _add_compose_arguments(parser, include_batch=False)
    parser.add_argument(
        '-o',
        '--output',
        help='Path to save the run output as JSON file',
    )
    parser.add_argument(
        '--hold',
        type=float,
        default=None,
        metavar='SECONDS',
        help='Hold sandbox alive for N seconds after task completes, suppressing all runtime '
        'release calls. Use with "nexus sandbox exec/upload/download" to diagnose sandbox state. '
        'Send SIGINT/SIGTERM to release early.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)',
    )
    parser.add_argument(
        '--log-trim',
        type=int,
        default=None,
        help='Shorthand: set both head and tail trim to this value (default: 200). Use 0 to disable.',
    )
    parser.add_argument(
        '--log-trim-head',
        type=int,
        default=None,
        help='Head chars kept when trimming long log messages. Overrides --log-trim.',
    )
    parser.add_argument(
        '--log-trim-tail',
        type=int,
        default=None,
        help='Tail chars kept when trimming long log messages. Overrides --log-trim.',
    )
    _add_use_gateway_argument(parser)


def add_run_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by ``run-batch`` and ``run batch``."""
    parser.add_argument(
        'config',
        nargs='?',
        help='Path to a RunInput or compose YAML file',
    )
    parser.add_argument(
        '-g',
        '--generator',
        help='Generator Python file; supports --limit and ordered --use/--set/--set-json overlays',
    )
    parser.add_argument(
        '-d',
        '--database',
        help='Path to SQLite database file (default: :memory: if generator provided)',
    )
    _add_compose_arguments(parser, include_batch=True, include_generator=True)
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=1,
        help='Maximum number of concurrent tasks (default: 1)',
    )
    parser.add_argument(
        '--worker-processes',
        type=int,
        default=1,
        help='Split run-batch across N local worker processes with per-worker shard databases (default: 1)',
    )
    parser.add_argument(
        '--enable-conflict-groups',
        action='store_true',
        help='Serialize tasks sharing groups in metadata.conflict_groups (requires --worker-processes 1)',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for each task (default: no timeout)',
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop immediately on first failure',
    )
    parser.add_argument(
        '--retry-failed',
        type=int,
        default=0,
        help='Number of retries for failed tasks (default: 0)',
    )
    parser.add_argument(
        '--resume-workflow-on-retry',
        action='store_true',
        help='Resume committed workflow nodes on immediate retries (default: restart from the original input)',
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Do not skip already succeeded tasks (default: skip succeeded tasks)',
    )
    parser.add_argument(
        '--retry-only',
        action='store_true',
        help='Only retry previously failed tasks',
    )
    parser.add_argument(
        '--no-persist',
        action='append',
        choices=['trace', 'log'],
        default=[],
        help='Do not persist the selected data to the batch database; may be repeated',
    )
    parser.add_argument(
        '--export-format',
        choices=['jsonl', 'csv', 'full'],
        help='Auto-export results after batch completion (jsonl, csv, or full)',
    )
    parser.add_argument(
        '--export-path',
        help='Export file/directory path (required for :memory: database, optional otherwise)',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)',
    )
    parser.add_argument(
        '--log-trim',
        type=int,
        default=None,
        help='Shorthand: set both head and tail trim to this value (default: 200). Use 0 to disable.',
    )
    parser.add_argument(
        '--log-trim-head',
        type=int,
        default=None,
        help='Head chars kept when trimming long log messages. Overrides --log-trim.',
    )
    parser.add_argument(
        '--log-trim-tail',
        type=int,
        default=None,
        help='Tail chars kept when trimming long log messages. Overrides --log-trim.',
    )
    _add_use_gateway_argument(parser)


def _add_compose_arguments(
    parser: argparse.ArgumentParser, *, include_batch: bool, include_generator: bool = False
) -> None:
    parser.add_argument(
        '--data',
        action='append',
        default=[],
        help='Dataset URI/path run part; readable alias for --use data:<uri>',
    )
    parser.add_argument(
        '--split',
        help='Dataset split to use',
    )
    parser.add_argument(
        '--use',
        action='append',
        default=[],
        dest='use_parts',
        help='Run part name/path. Can be repeated and is merged in command-line order.',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        dest='set_values',
        help='Set a dotted RunInput field, for example runtime_configs.main.resource_spec.memory=64',
    )
    parser.add_argument(
        '--set-json',
        action='append',
        default=[],
        dest='set_json_values',
        help='Set a dotted RunInput field from a JSON value',
    )
    if include_batch:
        parser.add_argument(
            '--hook',
            action='append',
            default=[],
            dest='hook_values',
            metavar='HOOK[:key=value,...]',
            help=(
                'Compose hook (.py file or registered name) applied to every generated or dataset item after '
                'the run parts are merged. Can be repeated and runs in command-line order.'
            ),
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help=(
                'Use at most N dataset or generator items (0 creates an empty batch; default: all)'
                if include_generator
                else 'Limit the number of dataset items compiled into the batch'
            ),
        )
    else:
        parser.add_argument(
            '--id',
            help='Dataset item id for single-run compose',
        )


def _has_compose_args(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, 'data', None)
        or getattr(args, 'use_parts', None)
        or getattr(args, 'set_values', None)
        or getattr(args, 'set_json_values', None)
    )


def add_compose_arguments(parser: argparse.ArgumentParser, *, include_batch: bool) -> None:
    _add_compose_arguments(parser, include_batch=include_batch)


def has_compose_args(args: argparse.Namespace) -> bool:
    return _has_compose_args(args)


def register_run_single_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the legacy run-single subcommand."""
    run_single_parser = subparsers.add_parser(
        'run-single',
        help='Run a single task from a YAML configuration file',
    )
    add_run_single_arguments(run_single_parser)


def register_run_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the legacy run-batch subcommand."""
    run_batch_parser = subparsers.add_parser(
        'run-batch',
        help='Run a batch of tasks from a config generator file',
    )
    add_run_batch_arguments(run_batch_parser)


def register_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the grouped run command."""
    from nexus.cli.commands.live import add_live_arguments

    run_parser = subparsers.add_parser(
        'run',
        help='Run Nexus tasks and live services',
    )
    run_subparsers = run_parser.add_subparsers(dest='run_command', help='Run commands')

    single_parser = run_subparsers.add_parser(
        'single',
        help='Run a single task from a YAML configuration file',
    )
    add_run_single_arguments(single_parser)

    batch_parser = run_subparsers.add_parser(
        'batch',
        help='Run a batch of tasks from a config generator file',
    )
    add_run_batch_arguments(batch_parser)

    live_parser = run_subparsers.add_parser(
        'live',
        help='Run the live interactive runner HTTP/SSE service',
    )
    add_live_arguments(live_parser)
