"""Internal worker entrypoint for sharded batch execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from pathlib import Path

from nexus.runner.batch.batch_config import BatchConfig
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.batch.loader import GeneratorLoader
from nexus.runner.batch.runner import run_batch
from nexus.utils.log import setup_logging


logger = logging.getLogger(__name__)


async def _load_shard_context(db_path: str) -> BatchConfig:
    async with BatchDatabase(db_path) as db:
        generator_path = await db.get_batch_info('generator_path')
        config_data = await db.get_batch_info('config')

    if generator_path and generator_path != '<compose>':
        GeneratorLoader(Path(generator_path))
    if config_data is None:
        raise ValueError(f'Shard database has no batch config: {db_path}')
    return BatchConfig.model_validate(config_data).model_copy(update={'worker_processes': 1})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one pre-generated Nexus batch shard.')
    parser.add_argument('--database', required=True, help='Path to the shard SQLite database')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args(argv)

    setup_logging(log_level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        batch_config = asyncio.run(_load_shard_context(args.database))
        summary = asyncio.run(
            run_batch(
                generator_path=None,
                db_path=args.database,
                batch_config=batch_config,
            )
        )
        logger.info(
            'Shard completed: total=%s success=%s failed=%s timeout=%s',
            summary.total,
            summary.success,
            summary.failed,
            summary.timeout,
        )
        return 0
    except Exception as exc:
        logger.error('Shard worker failed: %s', exc, exc_info=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
