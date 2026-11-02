"""Cloud export cli."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import stat
import sys

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from nexus.distributed.export.client import CloudApi
from nexus.distributed.export.follow import (
    _validate_target,
    export_trace_events,
    resolve_trace_backend,
    resolve_trace_read_scope,
)
from nexus.distributed.export.results import TdsqlTraceReader, export_execution, validate_trace_table
from nexus.distributed.export.types import ExportError, resolve_result_status
from nexus.distributed.export.writer import JsonlEventWriter


_DEFAULT_CONTROL_URL = 'https://control-runner-nexus-agent.mcp.woa.com'


def resolve_secret(
    *,
    inline: str | None,
    path: Path | None,
    inline_option: str,
    file_option: str,
    stderr: TextIO = sys.stderr,
) -> str | None:
    if inline is not None and path is not None:
        raise ExportError(f'{inline_option} and {file_option} are mutually exclusive')
    if inline is not None:
        value = inline.strip()
        if not value:
            raise ExportError(f'{inline_option} must not be empty')
        print(
            f'WARNING: {inline_option} may be exposed through shell history and process listings; prefer {file_option}.',
            file=stderr,
        )
        return value
    if path is None:
        return None
    source = path.expanduser()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ExportError(f'cannot read {file_option}: {source}') from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ExportError(f'{file_option} must be a regular file')
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ExportError(f'{file_option} must have mode 0600')
    try:
        value = source.read_text(encoding='utf-8').strip()
    except OSError as exc:
        raise ExportError(f'cannot read {file_option}: {source}') from exc
    if not value:
        raise ExportError(f'{file_option} must not be empty')
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='将终态 Nexus Cloud Batch、Single 或 Live Execution 导出为标准 JSONL。'
    )
    parser.add_argument('--control-url', default=os.environ.get('NEXUS_CONTROL_URL', _DEFAULT_CONTROL_URL))
    parser.add_argument(
        '--application-url',
        default=os.environ.get('NEXUS_APPLICATION_URL'),
        help='Application API URL；也可通过 NEXUS_APPLICATION_URL 设置',
    )
    parser.add_argument('--execution-id', action='append', required=True, help='Repeat to merge execution results')
    parser.add_argument('--item-ids', action='append', help='Comma-separated item IDs from one Batch; may be repeated')
    taihu = parser.add_mutually_exclusive_group()
    taihu.add_argument('--taihu-token')
    taihu.add_argument('--taihu-token-file', type=Path)
    parser.add_argument('--db-host')
    parser.add_argument('--db-port', type=int, default=3306)
    parser.add_argument('--db-name')
    parser.add_argument('--db-user')
    database_password = parser.add_mutually_exclusive_group()
    database_password.add_argument('--db-password')
    database_password.add_argument('--db-password-file', type=Path)
    parser.add_argument('--trace-table')
    parser.add_argument(
        '--trace-backend',
        help=(
            'Trace backend name the credentials belong to, as returned with them. '
            'Required when the Execution writes this table on several backends, '
            'because the table name alone does not identify the schema.'
        ),
    )
    parser.add_argument(
        '--parent-created-at',
        help=(
            'Partition key value for TDSQL Boundless trace tables '
            '(the parent execution created_at). Required there for partition pruning; '
            'omit for classic TDSQL tables.'
        ),
    )
    parser.add_argument(
        '--trace-db-fallback',
        choices=['api', 'fail'],
        default=os.environ.get('NEXUS_EXPORT_TRACE_DB_FALLBACK') or None,
        help=(
            'Results only: when a transient Trace Database read fails, fall back to the '
            'Application API (default: api). fail aborts instead.'
        ),
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--strict-validate',
        action='store_true',
        help=(
            'Re-validate the finished JSONL by fully parsing every line (legacy behaviour; '
            'higher peak memory). Default validation compares the file against write-time '
            'bookkeeping without parsing.'
        ),
    )
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Export terminal Cloud Batch items while others are running (default: successful items)',
    )
    parser.add_argument(
        '--result-status',
        choices=['success', 'failed', 'all'],
        default=None,
        help=(
            'Select Batch results; requires --allow-partial (default: success with partial, all otherwise). '
            'failed selects non-success terminal statuses; failed and all filter empty failure rows after export. '
            'Explicit IDs bypass Batch status selection; successful rows are retained.'
        ),
    )
    args = parser.parse_args(argv)
    if len(args.execution_id) == 1:
        args.execution_id = args.execution_id[0]
    return args


def create_trace_reader(args: argparse.Namespace, *, stderr: TextIO = sys.stderr) -> TdsqlTraceReader | None:
    optional_values = (
        args.db_host,
        args.db_name,
        args.db_user,
        args.db_password,
        args.db_password_file,
        args.trace_table,
        getattr(args, 'trace_backend', None),
    )
    if not any(value is not None for value in optional_values):
        return None
    if (
        not args.db_host
        or not args.db_name
        or not args.db_user
        or not args.trace_table
        or (args.db_password is None and args.db_password_file is None)
    ):
        raise ExportError('incomplete Trace Database arguments')
    password = resolve_secret(
        inline=args.db_password,
        path=args.db_password_file,
        inline_option='--db-password',
        file_option='--db-password-file',
        stderr=stderr,
    )
    assert password is not None
    return TdsqlTraceReader(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=password,
        database=args.db_name,
        table=args.trace_table,
        parent_created_at=getattr(args, 'parent_created_at', None),
        trace_backend=getattr(args, 'trace_backend', None),
    )


async def _run(args: argparse.Namespace) -> None:
    allow_partial = getattr(args, 'allow_partial', False)
    result_status = getattr(args, 'result_status', None)
    resolve_result_status(allow_partial=allow_partial, result_status=result_status)
    strict_validate = getattr(args, 'strict_validate', False) or os.environ.get('NEXUS_EXPORT_STRICT_VALIDATE') == '1'
    if not args.application_url:
        raise ExportError('--application-url or NEXUS_APPLICATION_URL is required')
    token = resolve_secret(
        inline=args.taihu_token,
        path=args.taihu_token_file,
        inline_option='--taihu-token',
        file_option='--taihu-token-file',
    )
    if token is None:
        raise ExportError('--taihu-token or --taihu-token-file is required')
    trace_reader = create_trace_reader(args)
    timeout = aiohttp.ClientTimeout(total=300, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        api = CloudApi(
            session,
            control_url=args.control_url,
            application_url=args.application_url,
            token=token,
        )
        await export_execution(
            api=api,
            execution_id=args.execution_id,
            output=args.output,
            force=args.force,
            trace_reader=trace_reader,
            allow_partial=allow_partial,
            result_status=result_status,
            item_ids=getattr(args, 'item_ids', None),
            strict_validate=strict_validate,
            trace_db_fallback=getattr(args, 'trace_db_fallback', None),
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        with _stderr_logging():
            return run_cloud_export(args)
    except KeyboardInterrupt:
        print('Export interrupted.', file=sys.stderr)
        return 130
    except ExportError as exc:
        print(f'Export failed: {exc}', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Export failed: {type(exc).__name__}', file=sys.stderr)
        return 1


def run_cloud_export(args: argparse.Namespace) -> int:
    """Run a validated Cloud results export from the integrated CLI."""
    with _stderr_logging(warnings_only=True):
        asyncio.run(_run(args))
    print(f'Exported JSONL to {args.output.expanduser().resolve(strict=False)}', file=sys.stderr)
    return 0


def _source_identity(args: argparse.Namespace, spec_type: str) -> dict[str, Any]:
    parsed = urlsplit(args.control_url)
    trace_backend = getattr(args, 'trace_backend', None)
    # Never save userinfo, query tokens or fragments in the checkpoint identity.
    host = parsed.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    if parsed.port is not None:
        host += f':{parsed.port}'
    identity = {
        'control_url': urlunsplit((parsed.scheme.lower(), host.lower(), parsed.path.rstrip('/'), '', '')),
        'execution_id': args.execution_id,
        'spec_type': spec_type,
        'db_host': args.db_host,
        'db_port': args.db_port,
        'db_name': args.db_name,
        'table': validate_trace_table(args.trace_table),
        'storage_scope': 'principal_v1',
        'content': 'traces',
        'schema_version': 1,
    }
    if trace_backend is not None:
        # Recorded only when named, so checkpoints written before this option remain
        # resumable. Two backends can serve one table name, and their rows are not
        # interchangeable, so a resume that switches backend must not be accepted.
        identity['trace_backend'] = trace_backend
    return identity


@contextmanager
def _stderr_logging(*, warnings_only: bool = False):
    # Scope this change to the command and restore the embedding caller's handlers.
    root = logging.getLogger()
    previous = root.handlers[:]
    previous_level = root.level
    handler = logging.StreamHandler(sys.stderr)

    def below_warning(record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING

    if warnings_only:
        # Preserve the integrated CLI's verbosity and normal output handlers.
        handler.setLevel(logging.WARNING)
        for previous_handler in previous:
            previous_handler.addFilter(below_warning)
        root.handlers = [*previous, handler]
    else:
        root.setLevel(logging.INFO)
        root.handlers = [handler]
    try:
        yield
    finally:
        root.handlers = previous
        root.setLevel(previous_level)
        if warnings_only:
            for previous_handler in previous:
                previous_handler.removeFilter(below_warning)
        handler.close()


def run_trace_export(args: argparse.Namespace) -> int:
    """Run the CLI trace exporter, restoring handlers and preserving signal codes."""
    from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at

    fallback = getattr(args, 'trace_db_fallback', None)
    if fallback == 'api':
        print(
            'WARNING: --trace-db-fallback applies only to results exports; ignoring it for traces.',
            file=sys.stderr,
        )
    elif fallback is not None:
        raise ExportError('--trace-db-fallback applies only to results exports')
    interrupted = 0

    async def run() -> None:
        nonlocal interrupted
        current = asyncio.current_task()
        previous: dict[signal.Signals, Any] = {}

        def stop(signum: int, frame: Any) -> None:
            nonlocal interrupted
            if not interrupted:
                interrupted = signum
                if current is not None:
                    current.cancel()

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, stop)
            token = resolve_secret(
                inline=args.taihu_token,
                path=args.taihu_token_file,
                inline_option='--taihu-token',
                file_option='--taihu-token-file',
                stderr=sys.stderr,
            )
            password = resolve_secret(
                inline=args.db_password,
                path=args.db_password_file,
                inline_option='--db-password',
                file_option='--db-password-file',
                stderr=sys.stderr,
            )
            if token is None or password is None:
                raise ExportError('Taihu token and database password are required')
            table = validate_trace_table(args.trace_table)
            raw_partition_key = getattr(args, 'parent_created_at', None)
            try:
                override_partition_key = (
                    normalize_parent_created_at(raw_partition_key) if raw_partition_key is not None else None
                )
            except ValueError:
                raise ExportError('--parent-created-at must be an ISO 8601 datetime') from None
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90, connect=10)) as session:
                api = CloudApi(session, control_url=args.control_url, token=token)
                execution = await api.get_execution(args.execution_id)
                spec_type = _validate_target(execution, args.execution_id, table)
                # The schema comes from the execution, not from whether a flag was passed:
                # forgetting the flag used to silently pick the classic dialect and scan
                # every day in the retention window. An explicit value still wins, for the
                # case where the envelope is unavailable. --trace-backend names which sink
                # to believe, since one table name can be published by two backends whose
                # schemas differ.
                choice = resolve_trace_backend(
                    execution,
                    spec_type,
                    table,
                    backend=getattr(args, 'trace_backend', None),
                    override_partition_key=override_partition_key,
                )
                scope = choice.scope(table=table, execution_id=args.execution_id, spec_type=spec_type)
                writer = JsonlEventWriter(
                    Path(args.output), source=_source_identity(args, spec_type), resume=args.resume, follow=args.follow
                )
                source = choice.source_class()(
                    host=args.db_host,
                    port=args.db_port,
                    db=args.db_name,
                    user=args.db_user,
                    password=password,
                    read_driver='aiomysql',
                )

                async def prepare_scope() -> dict[str, str]:
                    # Batch parents do not publish a single-run sink envelope.
                    # Probe the selected database rather than silently treating a
                    # missing envelope as classic and scanning every partition.
                    probe = TdsqlTraceReader(
                        host=args.db_host,
                        port=args.db_port,
                        database=args.db_name,
                        user=args.db_user,
                        password=password,
                        table=table,
                    )
                    try:
                        resolved = await resolve_trace_read_scope(
                            execution,
                            spec_type,
                            table,
                            backend=getattr(args, 'trace_backend', None),
                            override_partition_key=choice.partition_key,
                            probe=probe.has_partition_column,
                            fetch_execution=api.get_execution,
                        )
                    except ExportError:
                        raise
                    except Exception as exc:
                        raise ExportError('Cannot determine trace table partitioning') from exc
                    if resolved.partition_key is None:
                        return {}
                    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT

                    source.dialect = BOUNDLESS_DIALECT
                    return {'parent_created_at': resolved.partition_key}

                await export_trace_events(
                    api=api,
                    source=source,
                    writer=writer,
                    execution_id=args.execution_id,
                    scope_metadata=scope,
                    follow=args.follow,
                    poll_interval=args.poll_interval,
                    prepare_scope=prepare_scope,
                )
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    with _stderr_logging():
        try:
            asyncio.run(run())
        except asyncio.CancelledError:
            print('Trace export interrupted; committed progress preserved.', file=sys.stderr)
            return 128 + (interrupted or signal.SIGINT)
        except Exception as exc:
            message = str(exc) if isinstance(exc, ExportError) else type(exc).__name__
            print(f'Trace export failed: {message}', file=sys.stderr)
            if isinstance(exc, ExportError) and message in {
                'checkpoint source does not match requested source',
                'checkpoint output path does not match requested output',
            }:
                return 2
            return 1
    return 0
