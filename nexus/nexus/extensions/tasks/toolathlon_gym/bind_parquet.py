"""Embed Toolathlon task/data bindings into self-contained parquet rows.

Usage:
    python -m nexus.extensions.tasks.toolathlon_gym.bind_parquet \
      INPUT.parquet OUTPUT.parquet --dataset toolathlon_synthesis
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from .binding import build_backend_mock_binding, build_toolathlon_gym_binding


def bind_nexus_payload(payload: str | dict[str, Any], *, dataset: str) -> str | dict[str, Any]:
    was_text = isinstance(payload, str)
    nexus = json.loads(payload) if isinstance(payload, str) else dict(payload)
    task_config = dict(nexus.get('task_config') or {})
    if dataset == 'toolathlon_gym':
        task_config['data_binding'] = build_toolathlon_gym_binding(task_config, dataset=dataset)
    else:
        task_config['data_binding'] = build_backend_mock_binding(task_config, dataset=dataset)
    nexus['task_config'] = task_config
    return json.dumps(nexus, ensure_ascii=False) if was_text else nexus


def bind_parquet(input_path: str | Path, output_path: str | Path, *, dataset: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('bind_parquet requires pyarrow') from exc

    table = pq.read_table(input_path)
    if 'nexus' not in table.column_names:
        raise ValueError('parquet is missing the nexus column')
    values = [bind_nexus_payload(value.as_py(), dataset=dataset) for value in table['nexus']]
    nexus_index = table.column_names.index('nexus')
    replacement = pa.array(values, type=table.schema.field('nexus').type)
    output = table.set_column(nexus_index, 'nexus', replacement)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, output_path, compression='snappy')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    bind_parquet(args.input, args.output, dataset=args.dataset)


if __name__ == '__main__':
    main()
