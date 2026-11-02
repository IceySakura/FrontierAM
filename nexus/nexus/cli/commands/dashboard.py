"""CLI entry point for the offline trajectory dashboard."""

from __future__ import annotations

import argparse

from aiohttp import web


def register_dashboard_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('dashboard', help='Browse finalized Gateway trajectories (read-only)')
    parser.add_argument(
        '--path',
        required=True,
        help='Trajectory or run-result JSON/JSON.gz, batch JSONL, or local/mounted Ceph directory',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default: loopback; no authentication)')
    parser.add_argument('--port', type=int, default=8766, help='HTTP port (default: 8766)')
    parser.add_argument(
        '--max-artifact-mib',
        type=int,
        default=64,
        metavar='MIB',
        help='Read limit per JSON file (before/after gzip) or JSONL record (default: 64 MiB; positive integer). '
        'Run-result wrappers count toward the limit; the JSONL file may be larger. '
        'Whole-trajectory loading can use much more memory than this limit.',
    )


def serve_dashboard(args: argparse.Namespace) -> int:
    from nexus.dashboard.server import create_app

    if not 1 <= args.port <= 65535:
        print('Dashboard port must be between 1 and 65535')
        return 2
    if args.max_artifact_mib <= 0:
        print('Dashboard --max-artifact-mib must be a positive integer')
        return 2
    max_bytes = args.max_artifact_mib * 1024 * 1024
    try:
        web.run_app(create_app(args.path, max_bytes=max_bytes), host=args.host, port=args.port)
    except (OSError, ValueError) as exc:
        print(f'Cannot start dashboard: {exc}')
        return 2
    return 0
