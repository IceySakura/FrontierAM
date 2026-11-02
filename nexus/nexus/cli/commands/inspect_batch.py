"""Inspect batch command - TUI for browsing batch run results."""

from __future__ import annotations

import argparse
import asyncio
import logging

from pathlib import Path


logger = logging.getLogger(__name__)


def inspect_batch(args: argparse.Namespace) -> int:
    """Launch TUI to inspect batch run results."""
    from nexus.runner.batch.database import BatchDatabase
    from nexus.runner.batch.inspector import BatchInspectorApp

    db_path = Path(args.database)
    if not db_path.exists():
        logger.error(f'Database file not found: {db_path}')
        return 1

    # Check generation_complete status
    try:

        async def check_status():
            async with BatchDatabase(str(db_path)) as db:
                is_complete = await db.is_generation_complete()
                if not is_complete:
                    logger.error('Database is not in generation_complete state. Cannot inspect incomplete batch.')
                    return False
                return True

        if not asyncio.run(check_status()):
            return 1

        # Launch TUI
        app = BatchInspectorApp(db_path=str(db_path))
        app.run()
        return 0

    except Exception as e:
        logger.error(f'Error launching inspector: {e}')
        return 1


def register_inspect_batch_parser(subparsers) -> None:
    """Register inspect-batch subcommand parser."""
    parser = subparsers.add_parser(
        'inspect-batch',
        help='Launch TUI to inspect batch run results',
    )
    parser.add_argument(
        'database',
        type=str,
        help='Path to batch database file',
    )


__all__ = ['inspect_batch', 'register_inspect_batch_parser']
