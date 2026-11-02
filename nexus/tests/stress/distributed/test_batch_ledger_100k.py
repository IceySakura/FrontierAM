from __future__ import annotations

import hashlib
import json
import time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence.orm import BatchItemRow, BatchRunRow, create_schema
from nexus.distributed.repositories.batch import BatchRepository, read_materialize_task_list_chunk
from nexus.distributed.utils.clock import utcnow


ITEM_COUNT = 100_000
PAGE_SIZE = 500


def _write_task_list(path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open('wb') as stream:
        for index in range(ITEM_COUNT):
            record = {
                'version': 1,
                'item_id': f'item-{index:012d}',
                'run_input': {'task_name': 'echo', 'task_config': {'index': index}},
            }
            line = json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            stream.write(line)
            digest.update(line)
            size_bytes += len(line)
    return size_bytes, digest.hexdigest()


async def test_batch_ledger_materializes_and_pages_100k_items(tmp_path):
    started = time.monotonic()
    path = tmp_path / 'tasks-100k.jsonl'
    size_bytes, sha256 = _write_task_list(path)
    ref = ArtifactRef(
        artifact_id='artifact-batch-ledger-100k',
        kind='task_list',
        sha256=sha256,
        size_bytes=size_bytes,
    )
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "batch-ledger-100k.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    try:
        async with session_factory() as session:
            session.add(
                BatchRunRow(
                    parent_execution_id='batch-ledger-100k',
                    shard_id=0,
                    generation_mode='local_task_list',
                    generation_status='materializing',
                    generation_epoch=1,
                    materialized_cursor='0',
                    materialize_cursor_bytes=0,
                    materialized_count=0,
                    task_list_ref_json=ref.model_dump(mode='json'),
                    task_list_manifest_json={
                        'version': 1,
                        'item_count': ITEM_COUNT,
                        'size_bytes': size_bytes,
                        'sha256': sha256,
                        'source': {'mode': 'stress'},
                        'datasets': [],
                    },
                    task_list_sha256=sha256,
                    owner_principal_id='owner-stress',
                    owner_tenant='tenant-stress',
                    total_items=ITEM_COUNT,
                    submitted_items=0,
                    active_items=0,
                    terminal_items=0,
                    submission_closed=False,
                    control_state='active',
                    summary_version=0,
                    last_progress_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

        chunks = 0
        while True:
            async with session_factory() as session:
                repository = BatchRepository(session, config=DistributedConfig())
                snapshot = await repository.snapshot_task_list_materialization(
                    'batch-ledger-100k',
                    expected_ref=ref,
                )
                await session.commit()
                chunk = read_materialize_task_list_chunk(
                    path,
                    cursor=snapshot.cursor_bytes,
                    materialized_count=snapshot.materialized_count,
                    expected_size_bytes=ref.size_bytes,
                    expected_record_version=snapshot.record_version,
                    chunk_size=PAGE_SIZE,
                    max_items=ITEM_COUNT,
                    max_line_bytes=16 * 1024**2,
                    max_chunk_bytes=16 * 1024**2,
                )
                result = await repository.commit_task_list_materialization_chunk(
                    'batch-ledger-100k',
                    snapshot=snapshot,
                    chunk=chunk,
                    now=utcnow(),
                )
                await session.commit()
            chunks += 1
            assert 0 <= result.inserted <= PAGE_SIZE
            if result.complete:
                break

        seen = 0
        cursor = None
        while True:
            async with session_factory() as session:
                page = await BatchRepository(session, config=DistributedConfig()).list_items(
                    'batch-ledger-100k',
                    status=None,
                    after_item_id=cursor,
                    limit=PAGE_SIZE,
                )
            if not page:
                break
            assert len(page) <= PAGE_SIZE
            assert page == sorted(page, key=lambda item: item['item_id'])
            cursor = page[-1]['item_id']
            seen += len(page)

        async with session_factory() as session:
            batch = await session.get(BatchRunRow, 'batch-ledger-100k')
            row_count = await session.scalar(
                select(func.count(BatchItemRow.item_id)).where(BatchItemRow.parent_execution_id == 'batch-ledger-100k')
            )
            first = await session.get(BatchItemRow, ('batch-ledger-100k', 'item-000000000000'))
            last = await session.get(BatchItemRow, ('batch-ledger-100k', 'item-000000099999'))

        assert chunks in {ITEM_COUNT // PAGE_SIZE, ITEM_COUNT // PAGE_SIZE + 1}
        assert seen == ITEM_COUNT
        assert row_count == ITEM_COUNT
        assert batch is not None
        assert batch.generation_status == 'complete'
        assert batch.materialized_count == ITEM_COUNT
        assert batch.materialize_cursor_bytes == size_bytes
        assert first is not None
        assert first.task_ordinal == 0
        assert last is not None
        assert last.task_ordinal == ITEM_COUNT - 1
        assert first.task_offset < last.task_offset
        print(
            json.dumps(
                {
                    'batch_ledger_100k': {
                        'items': ITEM_COUNT,
                        'bytes': size_bytes,
                        'chunks': chunks,
                        'pages': ITEM_COUNT // PAGE_SIZE,
                        'seconds': time.monotonic() - started,
                    }
                },
                sort_keys=True,
            )
        )
    finally:
        await engine.dispose()
