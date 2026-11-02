"""Main CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from typing import Any, cast

from nexus.cli.commands.artifact import handle_artifact_command, register_artifact_parser
from nexus.cli.commands.dashboard import register_dashboard_parser, serve_dashboard
from nexus.cli.commands.dataset import handle_dataset_command, register_dataset_parser
from nexus.cli.commands.distributed import distributed_command, register_distributed_parser
from nexus.cli.commands.export import export_batch, register_export_parser
from nexus.cli.commands.inspect_batch import inspect_batch, register_inspect_batch_parser
from nexus.cli.commands.live import serve_live
from nexus.cli.commands.materialize import materialize_command, register_materialize_parser
from nexus.cli.commands.modules import handle_modules_command, register_modules_parser
from nexus.cli.commands.redis import handle_redis_command, register_redis_parser
from nexus.cli.commands.run import (
    register_run_batch_parser,
    register_run_parser,
    register_run_single_parser,
    run_batch,
    run_single,
)
from nexus.cli.commands.runtime import (
    build_runtime,
    clean_runtime,
    register_build_runtime_parser,
    register_clean_runtime_parser,
    register_runtime_parser,
    serve_runtime,
)
from nexus.cli.commands.sandbox import register_sandbox_parser, sandbox_download, sandbox_exec, sandbox_upload
from nexus.cli.commands.submit import register_submit_parser, submit_command
from nexus.cli.extension_commands import handle_module_commands, register_module_commands_parser
from nexus.utils.log import setup_logging


def main() -> int:
    """Main entry point for the nexus CLI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        prog='nexus',
        description='Nexus - AI agent orchestration framework',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register all subcommands
    register_build_runtime_parser(subparsers)
    register_clean_runtime_parser(subparsers)
    register_runtime_parser(subparsers)
    register_run_parser(subparsers)
    register_run_single_parser(subparsers)
    register_run_batch_parser(subparsers)
    register_export_parser(subparsers)
    register_inspect_batch_parser(subparsers)
    register_materialize_parser(subparsers)
    register_submit_parser(subparsers)
    register_dataset_parser(subparsers)
    register_artifact_parser(subparsers)
    register_dashboard_parser(subparsers)
    register_distributed_parser(subparsers)
    register_modules_parser(subparsers)
    register_redis_parser(subparsers)
    register_sandbox_parser(subparsers)
    register_module_commands_parser(subparsers)

    args = parser.parse_args()
    args._raw_argv = sys.argv[1:]
    if args.command in {'dataset', 'submit'} and getattr(args, 'json', False):
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                cast('logging.StreamHandler[Any]', handler).setStream(sys.stderr)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'build-runtime':
        return build_runtime(args)

    if args.command == 'clean-runtime':
        return clean_runtime(args)

    if args.command == 'runtime':
        if args.runtime_command == 'serve':
            return serve_runtime(args)
        args._parser.print_help()
        return 0

    if args.command == 'run-single':
        return run_single(args)

    if args.command == 'run-batch':
        return run_batch(args)

    if args.command == 'run':
        if args.run_command == 'single':
            return run_single(args)
        if args.run_command == 'batch':
            return run_batch(args)
        if args.run_command == 'live':
            return serve_live(args)
        return 0

    if args.command == 'export':
        return export_batch(args)

    if args.command == 'inspect-batch':
        return inspect_batch(args)

    if args.command == 'materialize':
        return materialize_command(args)

    if args.command == 'dataset':
        return handle_dataset_command(args)

    if args.command == 'artifact':
        return handle_artifact_command(args)

    if args.command == 'dashboard':
        return serve_dashboard(args)

    if args.command == 'distributed':
        return distributed_command(args)

    if args.command == 'submit':
        return submit_command(args)

    if args.command == 'modules':
        return handle_modules_command(args)

    if args.command == 'redis':
        return handle_redis_command(args)

    if args.command == 'sandbox':
        if args.sandbox_command == 'exec':
            return sandbox_exec(args)
        if args.sandbox_command == 'upload':
            return sandbox_upload(args)
        if args.sandbox_command == 'download':
            return sandbox_download(args)
        return 0

    if args.command == 'module_commands':
        return handle_module_commands(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
