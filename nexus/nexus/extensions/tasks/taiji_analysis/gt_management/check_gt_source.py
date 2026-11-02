#!/usr/bin/env python3
"""Check if GT (reference) in result DBs matches predictions from a source DB.

Usage:
    python3 check_gt_source.py \
        --source /path/to/fable5_results/taiji_bench.db \
        --targets /path/to/hy3_nothink_v1/taiji_bench.db /path/to/hy3_high_v1/taiji_bench.db
"""

import argparse
import json
import sqlite3


def load_predictions(db_path: str) -> dict[str, str]:
    db = sqlite3.connect(db_path)
    preds = {}
    for row in db.execute('SELECT run_id, task_output FROM runs WHERE status = "success"'):
        to = json.loads(row[1]) if row[1] else {}
        pred = to.get('prediction', '')
        if pred.strip():
            preds[str(row[0])] = pred
    db.close()
    return preds


def check_target(target_path: str, source_preds: dict[str, str]) -> None:
    db = sqlite3.connect(target_path)
    rows = db.execute('SELECT run_id, config FROM runs').fetchall()
    db.close()

    match = mismatch = missing = 0
    mismatch_ids = []
    for rid, config_str in rows:
        config = json.loads(config_str) if config_str else {}
        gt = config.get('task_config', {}).get('answer', '')
        source_pred = source_preds.get(str(rid), '')
        if not source_pred:
            missing += 1
        elif gt == source_pred:
            match += 1
        else:
            mismatch += 1
            mismatch_ids.append(str(rid))

    print(f'{target_path}:')
    print(f'  total={len(rows)}, match={match}, mismatch={mismatch}, missing={missing}')
    if mismatch_ids:
        print(f'  mismatch run_ids: {mismatch_ids}')


def main():
    parser = argparse.ArgumentParser(description='Check GT source in result DBs')
    parser.add_argument('--source', required=True, help='Source DB (e.g. fable5 results)')
    parser.add_argument('--targets', nargs='+', required=True, help='Target DBs to check')
    args = parser.parse_args()

    source_preds = load_predictions(args.source)
    print(f'Source predictions: {len(source_preds)}')
    print()

    for target in args.targets:
        check_target(target, source_preds)
        print()


if __name__ == '__main__':
    main()
