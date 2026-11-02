"""Read and inspect Parquet checkpoints without Redis or DataUnit services."""

from __future__ import annotations

import argparse
import json
import pickle

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .format import ARRAY_DTYPES, PARQUET_BATCH_SIZE, build_schema, decode_array, parse_blob_reference
from ..file_storage.blob_codec import decode_blob
from ..file_storage.types import LARGE_FIELDS


if TYPE_CHECKING:
    from ...redis import _RedisDumpRef


def iter_payloads(
    dump_dir: str | Path,
    *,
    fields: Collection[str] | None = None,
    entry_ids: Collection[int] | None = None,
    load_blobs: bool = True,
    summaries: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream exported records without Redis/DU access; project fields before reading.

    ``summaries`` reads only numeric shape leaves and returns compact array
    descriptions. ``load_blobs=False`` returns checkpoint blob URIs without opening their files.
    """
    import pyarrow.parquet as pq

    directory = Path(dump_dir)
    parquet = pq.ParquetFile(directory / 'trajectory.parquet')
    if not parquet.schema_arrow.equals(build_schema(''), check_metadata=False):
        raise ValueError('unsupported trajectory checkpoint schema')
    if (parquet.schema_arrow.metadata or {}).get(b'nexus_format') != b'trajectory_parquet_v1':
        raise ValueError('unsupported trajectory checkpoint format')
    selected_fields = None if fields is None else set(fields)
    selected_entry_ids = None if entry_ids is None else set(entry_ids)
    field_names = [
        name for name in (*ARRAY_DTYPES, *LARGE_FIELDS) if selected_fields is None or name in selected_fields
    ]
    columns = ['entry_id', 'kind', 'trajectory_id', 'node_id', 'segment_id', 'segment_fields']
    for name in field_names:
        if summaries and name in ARRAY_DTYPES:
            columns.append(f'{name}.shape')
        else:
            columns.append(name)
    for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
        for index in range(batch.num_rows):
            row = {name: batch.column(name)[index].as_py() for name in columns[:5]}
            if selected_entry_ids is not None and row['entry_id'] not in selected_entry_ids:
                continue
            payload_fields = {}
            if row['kind'] == 'segment':
                encoded = batch.column('segment_fields')[index].as_py() or []
                payload_fields = {
                    name: pickle.loads(value)
                    for name, value in encoded
                    if selected_fields is None or name in selected_fields
                }
            else:
                for name in field_names:
                    value = batch.column(name)[index]
                    if not value.is_valid:
                        continue
                    if name in ARRAY_DTYPES:
                        payload_fields[name] = decode_array(value, name, summaries=summaries)
                    else:
                        reference = value.as_py()
                        relative = parse_blob_reference(reference)
                        payload_fields[name] = (
                            decode_blob((directory / relative).read_bytes()) if load_blobs else reference
                        )
            yield {**row, 'fields': payload_fields}


def read_dump(dump_dir: str | Path, dump_refs: Sequence[_RedisDumpRef] | None = None, *, fields=None):
    """Read records, optionally restoring persistent-ref input order and duplicates."""
    if dump_refs is None:
        return list(iter_payloads(dump_dir, fields=fields))
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(Path(dump_dir) / 'trajectory.parquet').metadata
    dump_id = metadata[b'dump_id'].decode()
    if any(ref.dump_id != dump_id for ref in dump_refs):
        raise ValueError('persistent reference belongs to a different checkpoint')
    selected_entry_ids = {ref.entry_id for ref in dump_refs}
    rows = {row['entry_id']: row for row in iter_payloads(dump_dir, fields=fields, entry_ids=selected_entry_ids)}
    return [rows[ref.entry_id] for ref in dump_refs]


def main():
    parser = argparse.ArgumentParser(description='Inspect a Redis trajectory Parquet dump offline')
    parser.add_argument('dump_dir')
    parser.add_argument('--fields', nargs='+')
    parser.add_argument('--load-blobs', action='store_true')
    args = parser.parse_args()
    for row in iter_payloads(args.dump_dir, fields=args.fields, summaries=True, load_blobs=args.load_blobs):
        print(json.dumps(row, ensure_ascii=False, default=repr))
