#!/usr/bin/env python3
"""Reset runs with empty assistant content back to pending in the DB.

Scans results.jsonl for records where the last assistant message has empty
content, maps them back to session-level run_ids in the DB, and sets those
runs to 'pending' so they can be re-executed.

Usage:
    python reset_empty_to_pending.py <result_dir>
    python reset_empty_to_pending.py <result_dir> --dry-run

Examples:
    # Reset empty runs for step600:
    python reset_empty_to_pending.py /data/yonewu/nexus/outputs/YB_OneAgentBench_V0.3_20260423_170239/HY3.0-dev-0405-OneAgent-SFT-v0.2.4-step600

    # Dry run (only show what would be reset):
    python reset_empty_to_pending.py /path/to/result_dir --dry-run
"""

import argparse
import json
import re
import sqlite3
import sys

from pathlib import Path


def find_empty_content_run_ids(results_path: Path) -> set[str]:
    """Scan results.jsonl and return run_ids with empty last assistant content."""
    empty_run_ids: set[str] = set()
    total = 0

    with results_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            run_id = rec.get('run_id', '')
            to = rec.get('task_output', {}) or {}

            has_empty = False

            # Multi-turn: check turn_results
            turn_results = to.get('turn_results', []) or []
            if turn_results:
                for tr in turn_results:
                    answer = tr.get('answer', '')
                    if not answer or not answer.strip():
                        has_empty = True
                        break
            else:
                # Single-turn: check trajectory
                traj_str = to.get('trajectory', '')
                if not traj_str or not traj_str.strip():
                    has_empty = True
                else:
                    try:
                        traj = json.loads(traj_str)
                        last_content = ''
                        for msg in reversed(traj):
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                last_content = msg.get('content', '') or ''
                                break
                        if not last_content.strip():
                            has_empty = True
                    except (json.JSONDecodeError, TypeError):
                        pass

            if has_empty:
                empty_run_ids.add(run_id)

    print(f'Scanned {total} records in {results_path.name}')
    print(f'Found {len(empty_run_ids)} records with empty last assistant content')
    return empty_run_ids


def map_to_session_run_ids(
    empty_run_ids: set[str],
    db_run_ids: set[str],
) -> set[str]:
    """Map per-turn run_ids (e.g. yb_bench_X_t2) to session run_ids in DB."""
    session_ids: set[str] = set()
    not_found: list[str] = []

    for rid in empty_run_ids:
        if rid in db_run_ids:
            session_ids.add(rid)
        else:
            # Strip _tN suffix: yb_bench_Show-Chart-008_t2 -> yb_bench_Show-Chart-008
            task_id = rid.replace('yb_bench_', '', 1)
            session_key = re.sub(r'_t\d+$', '', task_id)
            session_rid = f'yb_bench_{session_key}'
            if session_rid in db_run_ids:
                session_ids.add(session_rid)
            else:
                not_found.append(rid)

    if not_found:
        print(f'\nWARNING: {len(not_found)} run_ids not found in DB:')
        for rid in not_found[:10]:
            print(f'  {rid}')
        if len(not_found) > 10:
            print(f'  ... and {len(not_found) - 10} more')

    return session_ids


def reset_to_pending(
    db_path: Path,
    session_run_ids: set[str],
    dry_run: bool = False,
) -> int:
    """Reset matching runs to pending in the DB. Returns count of updated rows."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print(f'\nRuns to reset ({len(session_run_ids)}):')
    for rid in sorted(session_run_ids):
        cursor.execute('SELECT status FROM runs WHERE run_id = ?', (rid,))
        row = cursor.fetchone()
        print(f'  {rid}  ({row[0] if row else "NOT_FOUND"})')

    if dry_run:
        print('\n[DRY RUN] No changes made.')
        conn.close()
        return 0

    updated = 0
    for rid in session_run_ids:
        cursor.execute(
            "UPDATE runs SET status = 'pending', updated_at = CURRENT_TIMESTAMP, "
            'task_output = NULL, agent_output = NULL, processor_results = NULL, '
            'error = NULL, end_time = NULL '
            'WHERE run_id = ?',
            (rid,),
        )
        updated += cursor.rowcount

    conn.commit()

    # Print final status distribution
    cursor.execute('SELECT status, COUNT(*) FROM runs GROUP BY status')
    print(f'\nUpdated {updated} runs to pending.')
    print('DB status after update:')
    for status, count in cursor.fetchall():
        print(f'  {status}: {count}')

    conn.close()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reset runs with empty assistant content to pending.',
    )
    parser.add_argument(
        'result_dir',
        help='Path to result directory containing results.jsonl and results.db',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show what would be reset, do not modify DB',
    )
    parser.add_argument(
        '--results',
        default='results.jsonl',
        help='Name of results JSONL file (default: results.jsonl)',
    )
    parser.add_argument(
        '--database',
        default='results.db',
        help='Name of database file (default: results.db)',
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    results_path = result_dir / args.results
    db_path = result_dir / args.database

    if not results_path.exists():
        print(f'ERROR: {results_path} not found', file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f'ERROR: {db_path} not found', file=sys.stderr)
        sys.exit(1)

    print(f'Results: {results_path}')
    print(f'Database: {db_path}')
    print()

    # Step 1: Find empty content run_ids
    empty_run_ids = find_empty_content_run_ids(results_path)
    if not empty_run_ids:
        print('No empty content records found. Nothing to do.')
        return

    # Step 2: Get DB run_ids and map
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('SELECT run_id FROM runs')
    db_run_ids = {row[0] for row in cursor.fetchall()}
    conn.close()

    session_run_ids = map_to_session_run_ids(empty_run_ids, db_run_ids)
    if not session_run_ids:
        print('No matching session run_ids found in DB. Nothing to do.')
        return

    # Step 3: Reset to pending
    reset_to_pending(db_path, session_run_ids, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
