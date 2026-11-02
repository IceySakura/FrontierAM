"""Export Dataset-backed batch inputs to platform rows and common evaluation parameters.

Uses the same MACOSWORLD_DATASET_URI/split/filter settings as the batch generator.
The model, processors and runner are supplied by the evaluation platform.

Usage (from the repository root, with Wedata credentials configured)::

    export MACOSWORLD_DATASET_URI=macosworld/macosworld@v1.0.0
    export MACOSWORLD_DATASET_SPLIT=test
    export MACOSWORLD_CATEGORIES=all
    export MACOSWORLD_LIMIT=10
    python -m nexus.extensions.tasks.mac_os_world.export_platform_dataset \
        -o outputs/macosworld.jsonl --eval-set-out outputs/macosworld-common.json

Import the JSONL as the platform dataset. Merge the second file's
nexus_run_input into the evaluation-version parameters (NexusCommonData /
NexusCommonEvalV2). Set nexus_env.ref to a pushed commit containing this adapter
and supply DEVCLOUD_APP_ID, DEVCLOUD_APP_TOKEN and WEDATA_KEY via platform secrets.
This export does not create platform records or change published datasets.
"""

from __future__ import annotations

import argparse
import copy
import json

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.mac_os_world.generator import generate_configs


def platform_parts(items: Iterable[BatchItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    seen: set[str] = set()
    for item in items:
        config = copy.deepcopy(item.config.model_dump(exclude_unset=True))
        files = item.config.task_files
        if files is None or files.remote is None or files.type != 'zip' or not files.size or not files.sha256:
            raise ValueError('Platform export requires remote Dataset ZIP task_files with size and sha256')
        task = config['task_config']
        if task.get('task_root') or task.get('task_file') not in (None, 'task.json'):
            raise ValueError('Platform export cannot contain local task paths')
        task.pop('task_root', None)
        task.pop('task_file', None)
        per_task = {key: task.pop(key) for key in ('category', 'task_id')}
        item_id = per_task['task_id']
        if item_id in seen:
            raise ValueError(f'Duplicate platform item id: {item_id}')
        seen.add(item_id)
        config.pop('task_files')
        for key in ('processors', 'runner', 'run_id'):
            config.pop(key, None)
        # Keep only the protocol choice; platform task forms supply model/key/URL.
        clients = config.pop('openai_clients', {})
        if set(clients) - {'main'}:
            raise ValueError('macOSWorld exporter does not support additional model clients')
        config['openai_clients'] = {'main': {'name': clients['main']['name']}}
        if common is not None and common != config:
            raise ValueError('Selected tasks have different common settings; refusing a lossy platform export')
        common = config
        rows.append({
            'id': item_id,
            'meta_data_': {
                'nexus_run_input': {'task_config': per_task, 'task_files': files.model_dump(exclude_none=True)},
                'instruction': item.metadata['instruction'],
                'domain': per_task['category'],
            },
        })
    if common is None:
        raise ValueError('No tasks selected for platform export')
    return rows, {'nexus_run_input': common}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True, type=Path, help='Platform dataset JSONL')
    parser.add_argument(
        '--eval-set-out', required=True, type=Path, help='Common nexus_run_input for evaluation parameters'
    )
    args = parser.parse_args()
    if args.output.resolve() == args.eval_set_out.resolve():
        parser.error('Dataset JSONL and evaluation parameters must use different output paths')
    rows, common = platform_parts(generate_configs())
    for path in (args.output, args.eval_set_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    args.eval_set_out.write_text(json.dumps(common, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(rows)} Dataset-backed tasks to {args.output}; common parameters: {args.eval_set_out}')


if __name__ == '__main__':
    main()
