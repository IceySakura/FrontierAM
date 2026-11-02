"""Coordinate Parquet checkpoint export and pinned DU file copies."""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading

from pathlib import Path
from tempfile import TemporaryDirectory

from .format import ARRAY_DTYPES, CHECKPOINT_URI_PREFIX, PARQUET_BATCH_SIZE, build_schema, encode_array_column
from ..dump_protection import get_deferred_records, protect_dump_records
from ..file_storage.publication import _ENVELOPE_PREFIX, StoredNode
from ..file_storage.types import LARGE_FIELDS
from ..types import RedisClientError


logger = logging.getLogger(__name__)


def _write_parquet_batch(writer, schema, rows):
    import pyarrow as pa

    columns = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        columns.append(
            encode_array_column(values, field.name, field.type)
            if field.name in ARRAY_DTYPES
            else pa.array(values, type=field.type)
        )
    writer.write_table(pa.Table.from_arrays(columns, schema=schema), row_group_size=len(rows))


def _read_dump_batch(client, refs, first_entry_id, selected_fields):
    from ...redis import NodeRecord, NodeRef, SegmentRecord

    record_bytes_list = client.get_node_blobs([ref.record_id for ref in refs])
    missing = [index for index, record_bytes in enumerate(record_bytes_list) if record_bytes is None]
    if missing:
        retained = get_deferred_records(client, [refs[index].record_id for index in missing])
        for index, record_bytes in zip(missing, retained, strict=True):
            record_bytes_list[index] = record_bytes
    rows, export_urls, cleanup_urls_by_record = [], set(), {}
    for offset, (ref, record_bytes) in enumerate(zip(refs, record_bytes_list, strict=True)):
        if record_bytes is None:
            raise RedisClientError(f'payload not found for node_id={ref.record_id}', code='not_found')
        blob_urls = {}
        if record_bytes.startswith(_ENVELOPE_PREFIX):
            stored = StoredNode.deserialize(record_bytes)
            if stored.state != 'committed':
                raise RedisClientError('payload publication is incomplete', code='publication_incomplete')
            record_bytes, blob_urls = stored.record, stored.blob_urls
        cleanup_urls_by_record[ref.record_id] = tuple(blob_urls.values())
        payload_type = NodeRecord if isinstance(ref, NodeRef) else SegmentRecord
        payload = payload_type.deserialize(record_bytes)
        if not isinstance(payload, payload_type):
            raise TypeError(f'expected {payload_type.__name__}')
        row = {
            'entry_id': first_entry_id + offset,
            'kind': 'node' if isinstance(ref, NodeRef) else 'segment',
            'trajectory_id': payload.trajectory_id,
        }
        if isinstance(payload, NodeRecord):
            row['node_id'] = payload.node_id
            for name in (*ARRAY_DTYPES, *LARGE_FIELDS):
                if selected_fields is not None and name not in selected_fields:
                    continue
                value = getattr(payload, name)
                if name in blob_urls:
                    url = blob_urls[name]
                    row[name] = f'{CHECKPOINT_URI_PREFIX}data/{url.rsplit("/", 1)[-1]}.blob'
                    export_urls.add(url)
                elif name in LARGE_FIELDS and value is not None:
                    raise ValueError(f'Node {name} needs a hybrid blob URL; rewrite it through hybrid first')
                elif value is not None:
                    row[name] = value
        else:
            row['segment_id'] = payload.segment_id
            row['segment_fields'] = [
                (name, value)
                for name, value in payload.fields.items()
                if selected_fields is None or name in selected_fields
            ]
        rows.append(row)
    return rows, export_urls, cleanup_urls_by_record


def _drain_dump_preparation(fs, dump_id):
    """Drain failed preparation after removing Redis deletion protection."""
    while True:
        try:
            fs.finalize_dump(dump_id)
            return
        except RedisClientError as exc:
            if exc.code != 'dump_busy':
                raise
            # A timed-out prepare can still be queued/running on its DU.
            fs.get_dump_status(dump_id)
            threading.Event().wait(0.02)


def _dump_checkpoint_sync(store, refs, dump_id, directory, selected_fields, cancel_requested):
    import pyarrow.parquet as pq

    client = store._client
    publication = client._publication
    fs = publication._fs
    schema = build_schema(dump_id)
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / 'trajectory.parquet'
    if parquet_path.exists():
        raise FileExistsError(parquet_path)
    (directory / 'data').mkdir(exist_ok=True)
    temporary_parquet_path = directory / f'.trajectory-{dump_id}.parquet'
    prepare_complete = False
    prepare_started = False
    with publication.operation(), TemporaryDirectory(prefix='nexus-trajectory-') as staging:
        local_parquet_path = Path(staging) / 'trajectory.parquet'
        try:
            with protect_dump_records(client, [ref.record_id for ref in refs], dump_id) as protection:
                with pq.ParquetWriter(local_parquet_path, schema, compression='zstd', use_dictionary=False) as writer:
                    for start in range(0, len(refs), PARQUET_BATCH_SIZE):
                        if cancel_requested.is_set():
                            raise asyncio.CancelledError()
                        batch = refs[start : start + PARQUET_BATCH_SIZE]
                        rows, export_urls, cleanup_urls_by_record = _read_dump_batch(
                            client, batch, start, selected_fields
                        )
                        if export_urls:
                            prepare_started = True
                            fs.prepare_dump_sources(dump_id, sorted(export_urls))
                        # Captured metadata and pinned SSD sources no longer need deletion protection.
                        protection.release_records(
                            [ref.record_id for ref in batch], cleanup_urls_by_record=cleanup_urls_by_record
                        )
                        _write_parquet_batch(writer, schema, rows)
                        del rows
                    if cancel_requested.is_set():
                        raise asyncio.CancelledError()
                prepare_complete = True
        finally:
            if prepare_started and not prepare_complete:
                _drain_dump_preparation(fs, dump_id)
        if prepare_started:
            fs.start_dump_copy(dump_id, str(directory))
        shutil.copyfile(local_parquet_path, temporary_parquet_path)
        if prepare_started:
            while True:
                statuses = fs.get_dump_status(dump_id)
                if not any(item['state'] == 'running' for item in statuses.values()):
                    break
                threading.Event().wait(0.02)
            failures = {unit: item for unit, item in statuses.items() if item['state'] != 'complete'}
            if failures:
                # Keep source pins for explicit retry/abort. The final index stays absent.
                raise RedisClientError(
                    f'checkpoint copy failed: {failures}', code='dump_failed', body={'dump_id': dump_id}
                )
        if cancel_requested.is_set():
            if prepare_started:
                fs.finalize_dump(dump_id)
            raise asyncio.CancelledError()
        # The single coordinating dump publishes only after every blob is complete.
        temporary_parquet_path.replace(parquet_path)
        if prepare_started:
            try:
                fs.finalize_dump(dump_id)
            except (RedisClientError, OSError) as exc:
                # Published data is complete; retained source pins need explicit cleanup.
                logger.warning('Checkpoint %s is complete; source cleanup failed: %s', dump_id, exc)


async def dump_checkpoint(store, refs, dump_id, dump_dir, selected_fields):
    """Drain synchronous I/O on cancellation before releasing sources or returning."""
    cancel_requested = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            _dump_checkpoint_sync, store, refs, dump_id, Path(dump_dir).absolute(), selected_fields, cancel_requested
        )
    )
    completion = asyncio.gather(task, return_exceptions=True)
    cancellation = None
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as exc:
            cancel_requested.set()
            cancellation = cancellation or exc
    if cancellation is not None:
        if not task.cancelled() and task.exception() is not None:
            asyncio.get_running_loop().call_exception_handler({
                'message': 'Checkpoint I/O failed while draining cancellation',
                'exception': task.exception(),
                'task': task,
            })
        raise cancellation
    try:
        task.result()
    except RedisClientError as exc:
        raise RedisClientError(str(exc), code=exc.code, body={**exc.body, 'dump_id': dump_id}) from exc
    except OSError as exc:
        raise RedisClientError(str(exc), code='io_error', body={'dump_id': dump_id}) from exc
