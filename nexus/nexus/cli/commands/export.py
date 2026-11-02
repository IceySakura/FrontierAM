"""Export batch results from database to various formats."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import fcntl
import json
import logging
import math
import os
import shutil
import sys
import uuid

from pathlib import Path
from typing import Any

import yaml

from omegaconf import OmegaConf
from omegaconf.errors import GrammarParseError

from nexus.runner.batch import BatchDatabase
from nexus.utils.log import setup_logging


logger = logging.getLogger(__name__)

_DEFAULT_CONTROL_URL = 'https://control-runner-nexus-agent.mcp.woa.com'

# json.dumps uses the C encoder only for one-shot encoding; iterencode() always
# runs the pure-Python encoder. Output is byte-identical, serialization is ~2-3x
# slower, and no single string holding the whole record is ever materialized.
_JSON_ENCODER = json.JSONEncoder(ensure_ascii=False)

_CSV_FIELDS = ('run_id', 'status', 'start_time', 'end_time', 'retry_count', 'error')

# Lone surrogates (truncated LLM output) cannot be encoded as UTF-8; writing them
# as \udXXX keeps JSON valid and round-trippable instead of aborting the export.
_TEXT_ERRORS = 'backslashreplace'


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8', errors=_TEXT_ERRORS)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class AtomicTextOutput:
    """Text file written to ``<target>.partial`` and atomically published.

    Symlinks are followed so the real file is replaced (not the link), and the
    permission bits of an existing target are carried over. With
    ``discard_existing`` the old target is removed up front, matching the old
    truncate-on-open behaviour: an aborted run must not leave a previous,
    complete-looking file at the target path.
    """

    def __init__(self, target: Path, *, newline: str | None = None, discard_existing: bool = False) -> None:
        self.target = Path(target).expanduser().resolve(strict=False)
        self.partial = self.target.with_name(f'{self.target.name}.partial')
        if self.target.is_dir():
            raise IsADirectoryError(f'export target is a directory: {self.target}')
        self.target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._mode: int | None = self.target.stat().st_mode & 0o7777
        except FileNotFoundError:
            self._mode = None
        if discard_existing:
            self.target.unlink(missing_ok=True)
        self.file = open(self.partial, 'w', encoding='utf-8', errors=_TEXT_ERRORS, newline=newline)  # noqa: SIM115

    def commit(self) -> None:
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
        finally:
            self.file.close()
        if self._mode is not None:
            os.chmod(self.partial, self._mode)
        os.replace(self.partial, self.target)
        _fsync_directory(self.target.parent)

    def abort(self) -> None:
        """Close without publishing; the partial file is kept for diagnosis."""
        with contextlib.suppress(OSError):
            self.file.close()


class AtomicDirectoryOutput:
    """Stage a directory export under a unique sibling and publish it under a lock.

    Directory replacement is a two-rename swap on ordinary POSIX filesystems,
    so a tiny name-gap remains while an existing target is moved aside. The
    lock prevents concurrent exporters, and recovery restores the last known
    target after a crash during that gap.
    """

    def __init__(self, target: Path) -> None:
        requested = Path(target).expanduser()
        self.target = requested.resolve(strict=False)
        self.lock_path = self.target.with_name(f'{self.target.name}.lock')
        self.partial: Path | None = None
        self._lock_fd: int | None = None
        self._mode: int | None = None
        self._opened = False

    @staticmethod
    def _remove_entry(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)

    def _acquire_lock(self) -> None:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise RuntimeError(f'full export is already running for {self.target}') from exc
        payload = json.dumps({'pid': os.getpid()}, separators=(',', ':')).encode()
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        self._lock_fd = descriptor

    def _release_lock(self) -> None:
        descriptor = self._lock_fd
        if descriptor is None:
            return
        with contextlib.suppress(OSError):
            lock_stat = os.stat(self.lock_path)
            fd_stat = os.fstat(descriptor)
            if (lock_stat.st_dev, lock_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino):
                self.lock_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            os.close(descriptor)
        self._lock_fd = None

    def _recover(self) -> None:
        partials = sorted(self.target.parent.glob(f'{self.target.name}.partial-*'))
        backups = sorted(self.target.parent.glob(f'{self.target.name}.old-*'))
        if not self.target.exists() and not self.target.is_symlink() and backups:
            restore = max(backups, key=lambda path: path.stat().st_mtime_ns)
            os.rename(restore, self.target)
            backups.remove(restore)
        for path in (*partials, *backups):
            self._remove_entry(path)

    def open(self) -> Path:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists() and not self.target.is_dir():
            raise NotADirectoryError(f'full export target is not a directory: {self.target}')
        if self.target.exists():
            self._mode = self.target.stat().st_mode & 0o7777
        self._acquire_lock()
        try:
            self._recover()
            if self._mode is None and self.target.exists():
                self._mode = self.target.stat().st_mode & 0o7777
            self.partial = self.target.with_name(f'{self.target.name}.partial-{uuid.uuid4().hex}')
            self.partial.mkdir(parents=True, exist_ok=False)
            if self._mode is not None:
                os.chmod(self.partial, self._mode)
            self._opened = True
            return self.partial
        except BaseException:
            self._release_lock()
            raise

    def commit(self) -> None:
        if not self._opened or self.partial is None:
            raise RuntimeError('full export is not open')
        backup: Path | None = None
        try:
            if self.target.exists() or self.target.is_symlink():
                backup = self.target.with_name(f'{self.target.name}.old-{uuid.uuid4().hex}')
                os.rename(self.target, backup)
            os.rename(self.partial, self.target)
            _fsync_directory(self.target.parent)
            if backup is not None:
                self._remove_entry(backup)
            _fsync_directory(self.target.parent)
            self.partial = None
        except BaseException:
            if backup is not None and not self.target.exists() and backup.exists():
                with contextlib.suppress(OSError):
                    os.rename(backup, self.target)
            raise
        finally:
            self._opened = False
            self._release_lock()

    def abort(self) -> None:
        self._opened = False
        self._release_lock()


@contextlib.contextmanager
def _atomic_text_output(target: Path, *, newline: str | None = None):
    """Yield a text file that replaces ``target`` only if the block completes.

    On any exception the partial file is kept for diagnosis and the target is
    left untouched.
    """
    output = AtomicTextOutput(target, newline=newline)
    try:
        yield output.file
    except BaseException:
        output.abort()
        raise
    output.commit()


class _ExportUsageError(ValueError):
    pass


class _StoreExplicit(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        explicit = set(getattr(namespace, '_explicit_export_options', ()))
        explicit.add(self.dest)
        namespace._explicit_export_options = frozenset(explicit)


def _positive_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be a number') from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return result


def _require_nonblank(value: object, option: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _ExportUsageError(f'{option} must not be empty')


def _shard_root_path(db_path: Path) -> Path:
    return db_path.with_name(f'{db_path.name}.shards')


async def export_full(db: BatchDatabase, output_dir: Path) -> int:
    """Export to full directory structure with runs/ and configs/.

    The tree is built inside ``<output>.partial`` and swapped into place only
    when complete: a failed export leaves the previous target untouched plus a
    ``.partial`` directory that the next run removes first. Replacing an
    existing target briefly needs the target name free (old target is moved
    aside, then deleted after the swap).

    Args:
        db: Database instance
        output_dir: Output directory for export

    Returns:
        Exit code (0 for success, 1 for error)
    """
    atomic_output = AtomicDirectoryOutput(Path(output_dir))
    partial_dir = atomic_output.open()
    runs_dir = partial_dir / 'runs'
    configs_dir = partial_dir / 'configs'
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)

        metadata = await db.get_batch_info('metadata')
        if metadata:
            _write_text(partial_dir / 'batch_metadata.json', json.dumps(metadata, indent=2, ensure_ascii=False))

        config = await db.get_batch_info('config')
        if config:
            _write_text(partial_dir / 'batch_config.json', json.dumps(config, indent=2, ensure_ascii=False))

        logger.info('Exporting runs to full format...')
        async with contextlib.aclosing(db.iter_runs()) as runs:
            async for run in runs:
                export_run_to_full(run, partial_dir)
        atomic_output.commit()
        logger.info('Exported to %s', atomic_output.target)
        return 0
    except BaseException:
        atomic_output.abort()
        raise


def _dump_run_config_yaml(config: Any) -> str:
    """Serialise a stored run config as YAML, falling back when OmegaConf cannot parse it.

    ``OmegaConf.create`` parses every ``${...}`` eagerly, and a dataset's shell script
    can hold strings that only look like interpolations. SWE-bench Pro's NodeBB cases
    carry ``${test_files[$idx]%% | *}`` (bash's strip-longest-suffix), which raised
    ``GrammarParseError`` and aborted the whole export partway through. The stored
    config is already resolved, so for those we dump the plain mapping with OmegaConf's
    own dumper: the output format is unchanged and the script stays byte-for-byte.

    ``get_omega_conf_dumper`` is private and ``omegaconf`` carries no upper bound, so it is
    imported here rather than at module scope: if a release moves it, only this fallback
    degrades to ``yaml.safe_dump`` (which leaves keys like ``n``/``y`` unquoted) instead of
    breaking the import of this module and with it every export.
    """
    try:
        return OmegaConf.to_yaml(OmegaConf.create(config))
    except GrammarParseError:
        payload = yaml.safe_load(config) if isinstance(config, str) else config
        try:
            from omegaconf._utils import get_omega_conf_dumper
        except ImportError:
            logger.warning("omegaconf no longer exposes get_omega_conf_dumper; this config's yaml keys lose quoting")
            return yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return yaml.dump(payload, Dumper=get_omega_conf_dumper(), allow_unicode=True, sort_keys=False)


def write_run_to_jsonl(run: dict, f) -> None:
    """Write a single run to JSONL file object (incremental export).

    Args:
        run: Run record dict
        f: Open file object in write/append mode
    """
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
    extra = (run.get('config') or {}).get('extra')
    if run.get('judge_output') is not None:
        record['judge_output'] = run['judge_output']
    if extra:
        record['extra'] = extra
    # Stream token chunks instead of materializing the whole record as one str;
    # output is byte-identical to json.dumps(record, ensure_ascii=False).
    # The file must be opened with errors='backslashreplace' (AtomicTextOutput)
    # so lone surrogates become \udXXX escapes instead of raising.
    for chunk in _JSON_ENCODER.iterencode(record):
        f.write(chunk)
    f.write('\n')
    f.flush()  # Ensure data is written immediately


def write_csv_header(writer: Any) -> None:
    """Write CSV header.

    Args:
        writer: CSV writer object
    """
    writer.writerow([
        'run_id',
        'status',
        'start_time',
        'end_time',
        'retry_count',
        'error',
    ])


def write_run_to_csv(run: dict, writer: Any) -> None:
    """Write a single run to CSV writer (incremental export).

    Args:
        run: Run record dict
        writer: CSV writer object
    """
    writer.writerow([
        run['run_id'],
        run['status'],
        run.get('start_time', ''),
        run.get('end_time', ''),
        run.get('retry_count', 0),
        run.get('error', ''),
    ])


def export_run_to_full(run: dict, output_dir: Path) -> None:
    """Export a single run to full format directory structure (incremental export).

    Args:
        run: Run record dict
        output_dir: Base output directory (will create runs/ and configs/ subdirs)
    """
    run_id = run['run_id']
    runs_dir = output_dir / 'runs'
    configs_dir = output_dir / 'configs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export config as YAML
    if run.get('config'):
        _write_text(configs_dir / f'{run_id}.yaml', _dump_run_config_yaml(run['config']))

    # Export metadata
    if run.get('metadata'):
        _write_text(run_dir / 'metadata.json', json.dumps(run['metadata'], indent=2, ensure_ascii=False))

    # Export status
    status_data = {
        'status': run['status'],
        'start_time': run.get('start_time'),
        'end_time': run.get('end_time'),
        'retry_count': run.get('retry_count', 0),
        'error': run.get('error'),
    }
    _write_text(run_dir / 'status.json', json.dumps(status_data, indent=2, ensure_ascii=False))

    # Export task output
    if run.get('task_output'):
        _write_text(run_dir / 'task_output.json', json.dumps(run['task_output'], indent=2, ensure_ascii=False))

    # Export agent output
    if run.get('agent_output'):
        _write_text(run_dir / 'agent_output.json', json.dumps(run['agent_output'], indent=2, ensure_ascii=False))

    if run.get('judge_output'):
        judge_output_file = run_dir / 'judge_output.json'
        judge_output_file.write_text(json.dumps(run['judge_output'], indent=2, ensure_ascii=False))

    # Export processor results
    if run.get('processor_results'):
        processor_dir = run_dir / 'processors'
        processor_dir.mkdir(parents=True, exist_ok=True)

        for key, result in run['processor_results'].items():
            _write_text(processor_dir / f'{key}.json', json.dumps(result, indent=2, ensure_ascii=False))


async def export_jsonl(db: BatchDatabase, output_file: Path) -> int:
    """Export to JSONL format (one run per line).

    Args:
        db: Database instance
        output_file: Output JSONL file path

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info('Exporting runs to JSONL format...')

    with _atomic_text_output(output_file) as f:
        async with contextlib.aclosing(db.iter_runs()) as runs:
            async for run in runs:
                write_run_to_jsonl(run, f)

    logger.info(f'Exported to {output_file}')
    return 0


async def export_csv(db: BatchDatabase, output_file: Path) -> int:
    """Export to CSV format (summary statistics only).

    Args:
        db: Database instance
        output_file: Output CSV file path

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info('Exporting runs to CSV format...')

    with _atomic_text_output(output_file, newline='') as f:
        writer = csv.writer(f)

        # Write header
        write_csv_header(writer)

        # Write data rows (summary columns only; skipping the large payload
        # columns keeps peak memory at a handful of small rows)
        async with contextlib.aclosing(db.iter_runs(fields=_CSV_FIELDS)) as runs:
            async for run in runs:
                write_run_to_csv(run, writer)

    logger.info(f'Exported to {output_file}')
    return 0


async def _get_sharded_batch_paths(db_path: Path) -> list[Path] | None:
    if not db_path.exists() or not db_path.is_file():
        return None
    async with BatchDatabase(db_path) as db:
        if await db.get_batch_info('sharded_batch') is not True:
            return None
        shards = await db.get_batch_info('shards') or []
    shard_root = _shard_root_path(db_path)
    return [shard_root / shard['database'] for shard in shards]


async def export_sharded(db_path: Path, export_format: str, output: str | None) -> int:
    """Export an aggregated view from a sharded batch database."""
    shard_paths = await _get_sharded_batch_paths(db_path)
    if shard_paths is None:
        logger.error('Database is not a sharded batch: %s', db_path)
        return 1

    logger.info('Exporting from sharded batch database: %s', db_path)

    if export_format == 'full':
        export_dir = Path(output) if output else db_path.parent / 'export_full'
        atomic_output = AtomicDirectoryOutput(export_dir)
        staging_dir = atomic_output.open()
        try:
            for shard_path in shard_paths:
                async with BatchDatabase(shard_path) as db, contextlib.aclosing(db.iter_runs()) as runs:
                    async for run in runs:
                        export_run_to_full(run, staging_dir)
            atomic_output.commit()
            logger.info('Exported to %s', atomic_output.target)
            return 0
        except BaseException:
            atomic_output.abort()
            raise

    if export_format == 'jsonl':
        export_file = Path(output) if output else db_path.parent / 'results.jsonl'
        export_file.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_text_output(export_file) as f:
            for shard_path in shard_paths:
                async with BatchDatabase(shard_path) as db, contextlib.aclosing(db.iter_runs()) as runs:
                    async for run in runs:
                        write_run_to_jsonl(run, f)
        logger.info('Exported to %s', export_file)
        return 0

    if export_format == 'csv':
        export_file = Path(output) if output else db_path.parent / 'results.csv'
        export_file.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_text_output(export_file, newline='') as f:
            writer = csv.writer(f)
            write_csv_header(writer)
            for shard_path in shard_paths:
                async with (
                    BatchDatabase(shard_path) as db,
                    contextlib.aclosing(db.iter_runs(fields=_CSV_FIELDS)) as runs,
                ):
                    async for run in runs:
                        write_run_to_csv(run, writer)
        logger.info('Exported to %s', export_file)
        return 0

    logger.error('Unknown format: %s', export_format)
    return 1


def _validate_database_arguments(args: argparse.Namespace, *, required: bool) -> None:
    options = {
        'db_host': '--db-host',
        'db_name': '--db-name',
        'db_user': '--db-user',
        'trace_table': '--trace-table',
    }
    supplied = [getattr(args, name) is not None for name in options]
    password_supplied = args.db_password is not None or args.db_password_file is not None
    any_supplied = (
        any(supplied)
        or password_supplied
        or args.parent_created_at is not None
        or args.trace_backend is not None
        or 'db_port' in args._explicit_export_options
    )
    if args.db_password is not None and args.db_password_file is not None:
        raise _ExportUsageError('--db-password and --db-password-file are mutually exclusive')
    if required and not any_supplied:
        raise _ExportUsageError('Cloud traces require complete Trace Database arguments')
    if any_supplied and (not all(supplied) or not password_supplied):
        raise _ExportUsageError('incomplete Trace Database arguments')
    if not any_supplied:
        return
    for name, option in options.items():
        _require_nonblank(getattr(args, name), option)
    if isinstance(args.db_port, bool) or not isinstance(args.db_port, int) or not 1 <= args.db_port <= 65535:
        raise _ExportUsageError('--db-port must be between 1 and 65535')
    if args.db_password is not None:
        _require_nonblank(args.db_password, '--db-password')
    from nexus.runner.trace.tdsql.schema import PRINCIPAL_TRACE_TABLE_PATTERN

    if PRINCIPAL_TRACE_TABLE_PATTERN.fullmatch(args.trace_table) is None:
        raise _ExportUsageError('--trace-table is not a supported principal trace table')
    if args.trace_backend is not None:
        _require_nonblank(args.trace_backend, '--trace-backend')
    if args.parent_created_at is not None:
        _require_nonblank(args.parent_created_at, '--parent-created-at')
        # Delegate so this gate accepts exactly the grammar the exporter later normalizes;
        # a private parser here could pass a value the read path then rejects, or vice versa.
        from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at

        try:
            normalize_parent_created_at(args.parent_created_at)
        except ValueError:
            raise _ExportUsageError('--parent-created-at must be an ISO 8601 datetime') from None


def _validate_export_args(args: argparse.Namespace) -> None:
    database_supplied = args.database is not None
    execution_supplied = args.execution_id is not None
    if database_supplied == execution_supplied:
        raise _ExportUsageError('provide exactly one of database or --execution-id')
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        raise _ExportUsageError('--poll-interval must be a positive finite number')

    if database_supplied:
        explicit = args._explicit_export_options
        cloud_values = (
            args.taihu_token,
            args.taihu_token_file,
            args.db_host,
            args.db_name,
            args.db_user,
            args.db_password,
            args.db_password_file,
            args.trace_table,
            args.trace_backend,
            args.parent_created_at,
        )
        if (
            explicit & {'control_url', 'application_url', 'poll_interval', 'db_port'}
            or any(value is not None for value in cloud_values)
            or args.content != 'results'
            or args.follow
            or args.resume
            or args.force
            or args.allow_partial
            or args.result_status is not None
            or args.item_ids is not None
            or args.strict_validate
        ):
            raise _ExportUsageError('Cloud export options cannot be used with a local database')
        return

    execution_ids = args.execution_id if isinstance(args.execution_id, list) else [args.execution_id]
    if not execution_ids:
        raise _ExportUsageError('--execution-id must not be empty')
    for execution_id in execution_ids:
        _require_nonblank(execution_id, '--execution-id')
    execution_ids = list(dict.fromkeys(execution_ids))
    if len(execution_ids) > 1 and args.content != 'results':
        raise _ExportUsageError('multiple --execution-id values are supported only with --content results')
    args.execution_id = execution_ids[0] if len(execution_ids) == 1 else execution_ids
    if args.output is None:
        raise _ExportUsageError('--output is required for Cloud exports')
    if str(args.output) == '-':
        raise _ExportUsageError('Cloud exports require a file output; stdout is not supported')
    if args.format != 'jsonl':
        raise _ExportUsageError('Cloud exports support only --format jsonl')
    if 'poll_interval' in args._explicit_export_options and (args.content != 'traces' or not args.follow):
        raise _ExportUsageError('--poll-interval requires --content traces --follow')
    if args.taihu_token is not None and args.taihu_token_file is not None:
        raise _ExportUsageError('--taihu-token and --taihu-token-file are mutually exclusive')
    if args.taihu_token is None and args.taihu_token_file is None:
        raise _ExportUsageError('--taihu-token or --taihu-token-file is required')
    if args.taihu_token is not None:
        _require_nonblank(args.taihu_token, '--taihu-token')

    if args.content == 'results':
        from nexus.distributed.export.types import ExportError, resolve_result_status

        try:
            resolve_result_status(allow_partial=args.allow_partial, result_status=args.result_status)
        except ExportError as exc:
            raise _ExportUsageError(str(exc)) from exc
        if args.follow:
            raise _ExportUsageError('--follow is available only with --content traces')
        if args.resume:
            raise _ExportUsageError('--resume is available only with --content traces --follow')
        _validate_database_arguments(args, required=False)
    else:
        if args.item_ids is not None:
            raise _ExportUsageError('--item-ids is available only with --content results')
        if args.resume and not args.follow:
            raise _ExportUsageError('--resume requires --follow')
        if args.force:
            raise _ExportUsageError('--force is not supported for trace exports')
        if args.result_status is not None:
            raise _ExportUsageError('--result-status is available only with --content results')
        if args.allow_partial:
            raise _ExportUsageError('--allow-partial is available only with --content results')
        _validate_database_arguments(args, required=True)

    if args.control_url is None:
        args.control_url = os.environ.get('NEXUS_CONTROL_URL', _DEFAULT_CONTROL_URL)
    if args.application_url is None:
        args.application_url = os.environ.get('NEXUS_APPLICATION_URL')
    _require_nonblank(args.control_url, '--control-url')
    if args.content == 'results' and args.application_url is None:
        raise _ExportUsageError('--application-url or NEXUS_APPLICATION_URL is required')
    if args.application_url is not None:
        _require_nonblank(args.application_url, '--application-url')


def _export_local(args: argparse.Namespace) -> int:
    db_path = Path(args.database)
    if not db_path.exists():
        logger.error('Database not found: %s', db_path)
        return 1

    logger.info('Exporting from database: %s', db_path)

    async def run_export() -> int:
        shard_paths = await _get_sharded_batch_paths(db_path)
        if shard_paths is not None:
            return await export_sharded(db_path, args.format, args.output)

        async with BatchDatabase(db_path) as db:
            if args.format == 'full':
                export_dir = Path(args.output) if args.output else db_path.parent / 'export_full'
                export_dir.mkdir(parents=True, exist_ok=True)
                return await export_full(db, export_dir)
            if args.format == 'jsonl':
                export_file = Path(args.output) if args.output else db_path.parent / 'results.jsonl'
                export_file.parent.mkdir(parents=True, exist_ok=True)
                return await export_jsonl(db, export_file)
            if args.format == 'csv':
                export_file = Path(args.output) if args.output else db_path.parent / 'results.csv'
                export_file.parent.mkdir(parents=True, exist_ok=True)
                return await export_csv(db, export_file)
            logger.error('Unknown format: %s', args.format)
            return 1

    try:
        return asyncio.run(run_export())
    except Exception as exc:
        logger.error('Error exporting batch: %s', exc)
        return 1


def export_batch(args: argparse.Namespace) -> int:
    """Validate and dispatch a local or Cloud export."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.content != 'traces':
        setup_logging(log_level=log_level)
    try:
        _validate_export_args(args)
    except _ExportUsageError as exc:
        print(f'Invalid export arguments: {exc}', file=sys.stderr)
        return 2

    if args.database is not None:
        return _export_local(args)

    try:
        from nexus.distributed.export_types import ExportError

        if args.content == 'traces':
            from nexus.distributed import export_events

            return export_events.run_trace_export(args)

        from nexus.distributed import export as cloud_export

        return cloud_export.run_cloud_export(args)
    except KeyboardInterrupt:
        print('Export interrupted.', file=sys.stderr)
        return 130
    except ExportError as exc:
        print(f'Export failed: {exc}', file=sys.stderr)
        return 1
    except ImportError:
        print('Export failed: required Cloud export component is unavailable', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Export failed: {type(exc).__name__}', file=sys.stderr)
        return 1


def register_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the export subcommand.

    Args:
        subparsers: Subparser action from argparse
    """
    export_parser = subparsers.add_parser(
        'export',
        help='Export batch results from database to various formats',
    )
    export_parser.add_argument(
        'database',
        nargs='?',
        help='Path to SQLite database file',
    )
    export_parser.set_defaults(_explicit_export_options=frozenset())
    export_parser.add_argument(
        '--execution-id', action='append', help='Cloud Execution or Batch parent ID; repeat to merge results'
    )
    export_parser.add_argument(
        '--item-ids', action='append', help='Comma-separated item IDs from one Batch; may be repeated'
    )
    export_parser.add_argument(
        '--content',
        choices=['results', 'traces'],
        default='results',
        action=_StoreExplicit,
        help='Cloud content to export (default: results)',
    )
    export_parser.add_argument(
        '--format',
        choices=['full', 'jsonl', 'csv'],
        default='jsonl',
        help='Export format (default: jsonl)',
    )
    export_parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help='Output path: file path for jsonl/csv formats, directory for full format',
    )
    export_parser.add_argument(
        '--control-url',
        default=os.environ.get('NEXUS_CONTROL_URL', _DEFAULT_CONTROL_URL),
        action=_StoreExplicit,
        help='Nexus Control API URL',
    )
    export_parser.add_argument(
        '--application-url',
        default=os.environ.get('NEXUS_APPLICATION_URL'),
        action=_StoreExplicit,
        help='Nexus Application API URL',
    )
    taihu = export_parser.add_mutually_exclusive_group()
    taihu.add_argument('--taihu-token')
    taihu.add_argument('--taihu-token-file', type=Path)
    export_parser.add_argument('--db-host')
    export_parser.add_argument('--db-port', type=int, default=3306, action=_StoreExplicit)
    export_parser.add_argument('--db-name')
    export_parser.add_argument('--db-user')
    database_password = export_parser.add_mutually_exclusive_group()
    database_password.add_argument('--db-password')
    database_password.add_argument('--db-password-file', type=Path)
    export_parser.add_argument('--trace-table')
    export_parser.add_argument(
        '--trace-backend',
        help=(
            'Trace backend name these credentials belong to, as returned alongside them. '
            'Required when the Execution wrote this table on more than one backend: the '
            'table name is derived from the owner, so it cannot identify the schema.'
        ),
    )
    export_parser.add_argument(
        '--parent-created-at',
        help=(
            'Partition key of a TDSQL Boundless trace table (the parent Execution created_at). '
            'Required there so reads prune to one day partition; omit for classic TDSQL tables.'
        ),
    )
    export_parser.add_argument(
        '--trace-db-fallback',
        choices=['api', 'fail'],
        default=os.environ.get('NEXUS_EXPORT_TRACE_DB_FALLBACK') or None,
        help=(
            'Results only: when a transient Trace Database read fails, fall back to the '
            'Application API (default: api). fail aborts instead.'
        ),
    )
    export_parser.add_argument('--follow', action='store_true', help='Continue polling Cloud trace events')
    export_parser.add_argument('--resume', action='store_true', help='Resume a followed Cloud trace export')
    export_parser.add_argument(
        '--poll-interval',
        type=_positive_finite_float,
        default=1.0,
        action=_StoreExplicit,
        metavar='SECONDS',
        help='Cloud trace polling interval (default: 1)',
    )
    export_parser.add_argument('--force', action='store_true', help='Replace existing Cloud results output')
    export_parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Export terminal Cloud Batch items while others are running (default: successful items)',
    )
    export_parser.add_argument(
        '--result-status',
        choices=['success', 'failed', 'all'],
        default=None,
        help=(
            'Select Batch results; requires --allow-partial (default: success with partial, all otherwise). '
            'failed selects non-success terminal statuses; failed and all filter empty failure rows after export. '
            'Explicit IDs bypass Batch status selection; successful rows are retained.'
        ),
    )
    export_parser.add_argument(
        '--strict-validate',
        action='store_true',
        help=(
            'Re-validate the finished Cloud results JSONL by fully parsing every line '
            '(legacy behaviour; higher peak memory). Default validation compares the file '
            'against write-time bookkeeping without parsing.'
        ),
    )
    export_parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)',
    )
