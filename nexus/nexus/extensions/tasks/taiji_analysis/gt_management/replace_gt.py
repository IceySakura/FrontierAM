#!/usr/bin/env python3
"""Replace benchmark ground truth with new model predictions from a run-batch DB.

Reads completed predictions from the DB, updates dataset.json's historical_report
field, then regenerates the parquet.

Usage:
    python3 replace_gt.py --db ./results/taiji_bench_v3.db
    python3 replace_gt.py --db ./results/taiji_bench_v3.db --dataset /path/to/episodes/dataset.json
"""

import argparse
import json
import sqlite3
import subprocess
import sys

from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'episodes' / 'dataset.json'
CONVERT_SCRIPT = Path(__file__).resolve().parent.parent / 'data_prep' / 'convert_dataset.py'


def main():
    parser = argparse.ArgumentParser(description='Replace GT with new model predictions')
    parser.add_argument('--db', required=True, help='Path to run-batch result DB (taiji_bench.db)')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET), help='Path to dataset.json')
    parser.add_argument('--dry-run', action='store_true', help='Print stats without writing')
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    predictions = {}
    for row in db.execute('SELECT run_id, task_output, status FROM runs WHERE status = "success"'):
        to = json.loads(row['task_output']) if row['task_output'] else {}
        prediction = to.get('prediction', '')
        if prediction.strip():
            predictions[str(row['run_id'])] = prediction
    db.close()
    print(f'Valid predictions from DB: {len(predictions)}')

    with open(args.dataset, encoding='utf-8') as f:
        dataset = json.load(f)

    replaced = 0
    for record in dataset:
        rid = str(record.get('run_id', ''))
        if rid in predictions:
            record['historical_report'] = predictions[rid]
            replaced += 1

    print(f'Replaced: {replaced} / {len(dataset)}')

    if args.dry_run:
        print('Dry run — no files written.')
        return

    with open(args.dataset, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f'Updated: {args.dataset}')

    # Regenerate parquet (convert_dataset.py expects cwd to contain episodes/dataset.json)
    dataset_parent = Path(args.dataset).parent
    convert_cwd = dataset_parent.parent if dataset_parent.name == 'episodes' else dataset_parent
    subprocess.run(
        [sys.executable, str(CONVERT_SCRIPT)],
        cwd=str(convert_cwd),
        check=True,
    )


if __name__ == '__main__':
    main()
