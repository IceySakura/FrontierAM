"""Live runner service CLI."""

from __future__ import annotations

import argparse
import logging

from aiohttp import web

from nexus.runner.live.server import create_app
from nexus.utils.log import setup_logging


logger = logging.getLogger(__name__)
DEFAULT_HOSTS = ['0.0.0.0', '::']


def serve_live(args: argparse.Namespace) -> int:
    """Run the live runner HTTP service."""
    setup_logging(log_level=logging.DEBUG if args.verbose else logging.INFO)
    app = create_app(
        base_path=args.base_path,
        storage_path=args.database,
        terminal_run_retention_seconds=args.terminal_retention_seconds,
    )
    hosts = args.host or DEFAULT_HOSTS
    base_path = args.base_path or '/'
    logger.info('Starting Nexus live runner service on %s:%s%s', ','.join(hosts), args.port, base_path)
    web.run_app(app, host=hosts, port=args.port)
    return 0


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    """Add live runner service arguments to a parser."""
    parser.add_argument(
        '--host',
        action='append',
        default=None,
        help='Host to bind. Repeat to bind multiple hosts (default: 0.0.0.0 and ::).',
    )
    parser.add_argument('--port', type=int, default=8765, help='Port to bind (default: 8765)')
    parser.add_argument(
        '--base-path',
        default='',
        help='Public URL path prefix when serving behind a reverse proxy, for example /nexus/live.',
    )
    parser.add_argument(
        '-d',
        '--database',
        default=None,
        help='Path to a local SQLite database for live run history. Omit to keep in-memory behavior.',
    )
    parser.add_argument(
        '--terminal-retention-seconds',
        type=float,
        default=60.0,
        help='Seconds to keep closed, failed, or killed runs in memory when SQLite history is enabled. '
        'Use a negative value to disable cleanup (default: 60).',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')


def register_live_parser(subparsers: argparse._SubParsersAction, *, name: str = 'live-runner') -> None:
    """Register a legacy top-level live service subcommand."""
    parser = subparsers.add_parser(
        name,
        help='Run the live interactive runner HTTP/SSE service',
    )
    add_live_arguments(parser)


__all__ = ['add_live_arguments', 'register_live_parser', 'serve_live']
