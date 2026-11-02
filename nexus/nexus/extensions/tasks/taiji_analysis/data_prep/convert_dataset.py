#!/usr/bin/env python3
"""Convert dataset.json to dataset.parquet with optional filtering modes.

Modes:
    all        Full conversion, no filtering (default)
    rule       Rule-based filter: keep only task-failure cases + optional URL check
    model      Model-based filter: remove items all models answered correctly

Usage:
    # Full conversion (mode=all)
    python3 convert_dataset.py

    # Rule-based filter (task failure + URL check)
    python3 convert_dataset.py --mode rule --skip-url-check

    # Model-based filter (remove easy items)
    python3 convert_dataset.py --mode model --results-dir /path/to/results

Options:
    --mode MODE          Conversion mode: all, rule, model (default: all)
    --input PATH         Input path (JSON for all/rule, parquet for model)
    --output PATH        Output parquet path
    --dry-run            Print stats without writing
    --skip-url-check     [rule] Skip Taiji API connectivity check
    --url-check-sample N [rule] Only check N random URLs (0 = all)
    --results-dir DIR    [model] Directory containing model result folders
    --model-folders F    [model] Comma-separated folder names
    --exclude-all-fail   [model] Also exclude items where ALL models failed
"""

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------


def normalize_record(raw: dict) -> dict:
    """Normalize field names to snake_case used by the generator."""
    return {
        'instance_id': raw.get('run_id') or raw.get('inst_id') or raw.get('instId', ''),
        'relay_run_id': raw.get('run_id', ''),
        'session_id': raw.get('session_id', ''),
        'question': raw.get('query') or raw.get('question', ''),
        'answer': raw.get('historical_report') or raw.get('answer', ''),
        'inst_id': raw.get('inst_id') or raw.get('instId', ''),
        'task_id': raw.get('task_id') or raw.get('taskID', ''),
        'ws_id': raw.get('ws_id') or raw.get('wsId', ''),
        'url': raw.get('url', ''),
        'sender': raw.get('sender', ''),
        'started_at': raw.get('started_at', 0),
    }


def dedup_and_write(df: pd.DataFrame, output_path: Path) -> None:
    """Deduplicate by instance_id and write parquet."""
    before = len(df)
    df = df.drop_duplicates(subset='instance_id', keep='first')
    after = len(df)
    if before != after:
        print(f'Deduplicated: {before} → {after} records')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f'Written {len(df)} records → {output_path}')


# ---------------------------------------------------------------------------
# Mode: all (full conversion)
# ---------------------------------------------------------------------------


def mode_all(args) -> None:
    input_path = Path(args.input or 'episodes/dataset.json')
    output_path = Path(args.output or 'episodes/dataset.parquet')

    if not input_path.exists():
        print(f'Error: {input_path} not found.', file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding='utf-8') as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        print('Error: input must be a JSON array.', file=sys.stderr)
        sys.exit(1)

    print(f'Total records: {len(raw_data)}')

    if args.dry_run:
        print(f'[Dry run] Would write {len(raw_data)} records')
        return

    records = [normalize_record(r) for r in raw_data]
    df = pd.DataFrame(records)
    dedup_and_write(df, output_path)


# ---------------------------------------------------------------------------
# Mode: rule (task failure + URL connectivity)
# ---------------------------------------------------------------------------

MCP_SCRIPT = str(Path(__file__).resolve().parents[1] / '.agent/skills/taiji-official-skills/scripts/connect_mcp.py')
TAIJI_API_TOKEN = os.getenv('TAIJI_API_TOKEN', 'h3kcqqyYy5Uosyuyx1E5mw')

TASK_FAILURE_PATTERNS = [
    '挂了',
    '挂掉',
    '崩溃',
    '异常退出',
    '重启',
    '报错',
    '卡住',
    '卡死',
    '不达标',
    '未达标',
    '劣化',
    '掉点',
    '退化',
    'OOM',
    'CUDA out of memory',
    'cudaMalloc',
    '显存不足',
    '内存不足',
    'CUDA error',
    'cudaErrorInsufficientDriver',
    'SIGSEGV',
    'Segmentation fault',
    'core dumped',
    'abort',
    'NCCL',
    'hang',
    'Hang',
    'deadlock',
    'connection refused',
    'broken pipe',
    '超时',
    'timeout',
    'Timeout',
    'killed',
    'Killed',
    'terminated',
    'exit code 1',
    'exit code 9',
    'exit code 137',
    'exit code 139',
    'exit code 247',
    'disk quota',
    'Disk quota',
    'No space left',
    'ValueError',
    'RuntimeError',
    'Traceback',
    'AssertionError',
    'KeyError',
    'IndexError',
    'FileNotFoundError',
    'ImportError',
    'TypeError',
    'AttributeError',
    'ModuleNotFoundError',
    'NaN',
    'loss diverge',
    ' inf',
    ' nan',
    'FAILED',
]

TASK_NORMAL_PATTERNS = [
    '训练正常完成',
    '正常完成',
    '一切正常',
    '没问题',
    '✅ CI 正常通过',
    '训练正常',
    '任务正常',
    '正常结束',
    '任务跑得很好',
    '全部跑通',
    '无任何异常',
    'SUCCEEDED 准确',
    '正常跑完',
    '完全正常',
    '跑得完全正常',
    '成功完成',
    '干净退出',
    '已正常跑完',
    '正常退出',
    'CI 通过',
    '校验全部通过',
    '校验全 PASS',
    '已启动',
    '已重启成功',
    '已克隆',
    '已更新',
    '已修正',
    '已用指定 commit 启动',
    '已重跑',
    '配置已用',
]


def is_failed_task(record: dict) -> bool:
    """Determine if the Taiji task actually failed and needs root-cause analysis."""
    report = record.get('historical_report', '')
    if not report:
        return False

    check_range = report[:800]
    is_normal = any(p in check_range for p in TASK_NORMAL_PATTERNS)

    no_problem_signals = ['没有真问题', '无需处理', '无需任何处理', '可以放心']
    is_explicitly_fine = any(s in check_range for s in no_problem_signals)

    if is_normal or is_explicitly_fine:
        problem_qualifiers = [
            '但性能',
            '但吞吐',
            '劣化',
            '退化',
            '不达标',
            '未达标',
            '掉点',
            '阈值没达标',
            '阈值未达标',
            '没达到',
        ]
        if not any(q in check_range for q in problem_qualifiers):
            return False

    return any(p in report for p in TASK_FAILURE_PATTERNS)


def check_instance_accessible(inst_id: str, task_id: str, ws_id: str, timeout: int = 30) -> bool:
    """Check if a Taiji instance is still queryable via MCP API."""
    try:
        args_json = json.dumps({
            'api_token': TAIJI_API_TOKEN,
            'task_id': task_id,
            'instance_id': inst_id,
        })
        result = subprocess.run(
            ['python3', MCP_SCRIPT, 'call', 'task_detail', args_json],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False
        output = result.stdout
        if '❌' in output or 'Error executing tool' in output:
            return False
        return not ('not exist' in output or 'not found' in output.lower())
    except (subprocess.TimeoutExpired, Exception):
        return False


def mode_rule(args) -> None:
    input_path = Path(args.input or 'episodes/dataset.json')
    output_path = Path(args.output or 'episodes/dataset.parquet')

    if not input_path.exists():
        print(f'Error: {input_path} not found.', file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding='utf-8') as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        print('Error: input must be a JSON array.', file=sys.stderr)
        sys.exit(1)

    print(f'Total records: {len(raw_data)}')

    # Filter 1: Ground truth quality
    diagnosis_records = [r for r in raw_data if is_failed_task(r)]
    filtered_out_gt = len(raw_data) - len(diagnosis_records)
    print(f'After ground truth filter: {len(diagnosis_records)} (removed {filtered_out_gt} non-diagnosis records)')

    # Filter 2: URL connectivity
    if not args.skip_url_check:
        import random

        records_to_check = diagnosis_records
        if args.url_check_sample > 0:
            sample = random.sample(records_to_check, min(args.url_check_sample, len(records_to_check)))
            print(f'Checking URL connectivity for {len(sample)} sampled records...')
        else:
            sample = records_to_check
            print(f'Checking URL connectivity for {len(sample)} records...')

        accessible_inst_ids: set[str] = set()
        inaccessible_inst_ids: set[str] = set()

        for i, r in enumerate(sample):
            inst_id = r.get('instId') or r.get('inst_id', '')
            task_id = r.get('taskID') or r.get('task_id', '')
            ws_id = r.get('wsId') or r.get('ws_id', '')

            if not inst_id:
                inaccessible_inst_ids.add(inst_id)
                continue
            if inst_id in accessible_inst_ids or inst_id in inaccessible_inst_ids:
                continue

            if check_instance_accessible(inst_id, task_id, ws_id):
                accessible_inst_ids.add(inst_id)
            else:
                inaccessible_inst_ids.add(inst_id)

            if (i + 1) % 20 == 0:
                print(
                    f'  Checked {i + 1}/{len(sample)}: '
                    f'{len(accessible_inst_ids)} accessible, {len(inaccessible_inst_ids)} inaccessible'
                )
            time.sleep(0.2)

        if args.url_check_sample > 0:
            final_records = [
                r for r in diagnosis_records if (r.get('instId') or r.get('inst_id', '')) not in inaccessible_inst_ids
            ]
        else:
            final_records = [
                r for r in diagnosis_records if (r.get('instId') or r.get('inst_id', '')) in accessible_inst_ids
            ]

        filtered_out_url = len(diagnosis_records) - len(final_records)
        print(f'After URL check: {len(final_records)} (removed {filtered_out_url} inaccessible)')
    else:
        print('Skipping URL connectivity check.')
        final_records = diagnosis_records

    print(f'\nFinal dataset: {len(final_records)} records')

    if args.dry_run:
        print('[Dry run] No output written.')
        return

    records = [normalize_record(r) for r in final_records]
    df = pd.DataFrame(records)
    dedup_and_write(df, output_path)


# ---------------------------------------------------------------------------
# Mode: model (filter by model agreement)
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/nexus_results'
DEFAULT_MODEL_FOLDERS = [
    'opus4.8_xhigh_results0713',
    'opus4.8_low_results0713',
    'hy3_high_results0713',
    'hy3_nothink_results0713',
    'hy3_preview_results0713',
]


def load_model_scores(results_dir: Path, folder: str) -> dict[str, float]:
    """Load run_id -> score mapping from a model's results.jsonl."""
    scores = {}
    path = results_dir / folder / 'results.jsonl'
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            scores[str(r['run_id'])] = r.get('task_output', {}).get('score', 0) if r.get('task_output') else 0
    return scores


def mode_model(args) -> None:
    input_path = Path(args.input or 'episodes/dataset.parquet')
    output_path = Path(args.output or 'episodes/dataset.parquet')
    results_dir = Path(args.results_dir or DEFAULT_RESULTS_DIR)

    model_folders = args.model_folders.split(',') if args.model_folders else DEFAULT_MODEL_FOLDERS

    # Load scores from all models
    all_scores: dict[str, dict[str, float]] = {}
    for folder in model_folders:
        path = results_dir / folder / 'results.jsonl'
        if not path.exists():
            print(f'Warning: {path} not found, skipping')
            continue
        all_scores[folder] = load_model_scores(results_dir, folder)
        print(f'  {folder}: {len(all_scores[folder])} items')

    if not all_scores:
        print('Error: no model results found')
        return

    # Find common run_ids
    common_ids = set.intersection(*[set(s.keys()) for s in all_scores.values()])
    print(f'\nCommon items across {len(all_scores)} models: {len(common_ids)}')

    # Classify items
    all_pass_ids: set[str] = set()
    all_fail_ids: set[str] = set()
    discriminative_ids: set[str] = set()

    for rid in common_ids:
        scores = [all_scores[m][rid] >= 1.0 for m in all_scores]
        if all(scores):
            all_pass_ids.add(rid)
        elif not any(scores):
            all_fail_ids.add(rid)
        else:
            discriminative_ids.add(rid)

    print(f'All models pass (easy, to remove): {len(all_pass_ids)}')
    print(f'All models fail (hard/broken): {len(all_fail_ids)}')
    print(f'Discriminative (mixed): {len(discriminative_ids)}')

    # Check all-fail items for broken data
    inaccessible_ids: set[str] = set()
    bad_gt_ids: set[str] = set()

    inaccessible_patterns = [
        '查不到',
        '不存在',
        '已被删除',
        '已删除',
        '无法访问',
        '已过期',
        '已清理',
        'does not exist',
        'has been deleted',
        'not found',
        '无法再分析',
        '数据已随之清理',
    ]
    operational_gt_markers = ['✅']

    first_folder = model_folders[0] if model_folders else ''
    first_results_path = results_dir / first_folder / 'results.jsonl'

    if first_results_path.exists():
        # Check predictions for inaccessible data
        with open(first_results_path) as f:
            for line in f:
                r = json.loads(line)
                rid = str(r['run_id'])
                if rid in all_fail_ids:
                    pred = r.get('task_output', {}).get('prediction', '') if r.get('task_output') else ''
                    if any(p in pred[:800] for p in inaccessible_patterns):
                        inaccessible_ids.add(rid)

        # Check ground truth for operational records
        db_path = results_dir / first_folder / 'taiji_bench.db'
        if db_path.exists():
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            for rid in all_fail_ids:
                row = conn.execute('SELECT config FROM runs WHERE run_id = ?', (rid,)).fetchone()
                if row:
                    config = json.loads(row['config'])
                    gt = config.get('task_config', {}).get('answer', '')
                    gt_start = gt[:200].strip()
                    if any(gt_start.startswith(m) for m in operational_gt_markers) and (
                        '启动' in gt_start or '克隆' in gt_start or '重跑' in gt_start or '更新' in gt_start
                    ):
                        bad_gt_ids.add(rid)
            conn.close()

    print('\nAll-fail breakdown:')
    print(f'  Data inaccessible: {len(inaccessible_ids)}')
    print(f'  Bad ground truth (operational): {len(bad_gt_ids)}')
    print(f'  Genuine hard: {len(all_fail_ids) - len(inaccessible_ids) - len(bad_gt_ids)}')

    # Determine exclusions
    exclude_ids = all_pass_ids | inaccessible_ids | bad_gt_ids
    if args.exclude_all_fail:
        exclude_ids |= all_fail_ids
    print(f'\nExcluding: {len(exclude_ids)} items total')

    # Map run_id -> inst_id
    rid_to_inst: dict[str, str] = {}
    if first_results_path.exists():
        with open(first_results_path) as f:
            for line in f:
                r = json.loads(line)
                rid_to_inst[str(r['run_id'])] = r.get('metadata', {}).get('inst_id', '')

    exclude_inst_ids = {rid_to_inst[rid] for rid in exclude_ids if rid in rid_to_inst}
    keep_ids = common_ids - exclude_ids

    if args.dry_run:
        print(f'\n[Dry run] Would keep {len(keep_ids)} items')
        print(f'\n--- Pass rate on filtered subset ({len(keep_ids)} items) ---')
        for model in all_scores:
            correct = sum(1 for rid in keep_ids if all_scores[model].get(rid, 0) >= 1.0)
            print(f'  {model}: {correct}/{len(keep_ids)} = {correct / len(keep_ids):.4f}')
        return

    # Load and filter parquet
    if not input_path.exists():
        print(f'Error: {input_path} not found')
        return

    df = pd.read_parquet(input_path)
    before = len(df)
    df = df[~df['instance_id'].isin(exclude_inst_ids)]
    after = len(df)

    print(f'\nFiltered: {before} → {after} items')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f'Written → {output_path}')

    # Print pass rate on kept subset
    print(f'\n--- Pass rate on filtered subset ({len(keep_ids)} items) ---')
    for model in all_scores:
        correct = sum(1 for rid in keep_ids if all_scores[model].get(rid, 0) >= 1.0)
        print(f'  {model}: {correct}/{len(keep_ids)} = {correct / len(keep_ids):.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert/filter dataset for taiji_analysis benchmark')
    parser.add_argument(
        '--mode', choices=['all', 'rule', 'model'], default='all', help='Conversion mode (default: all)'
    )
    parser.add_argument('--input', help='Input path (JSON for all/rule, parquet for model)')
    parser.add_argument('--output', help='Output parquet path')
    parser.add_argument('--dry-run', action='store_true', help='Print stats without writing')

    # Rule mode options
    parser.add_argument('--skip-url-check', action='store_true', help='[rule] Skip URL connectivity check')
    parser.add_argument('--url-check-sample', type=int, default=0, help='[rule] Check N random URLs (0=all)')

    # Model mode options
    parser.add_argument('--results-dir', help='[model] Directory with model result folders')
    parser.add_argument('--model-folders', help='[model] Comma-separated folder names')
    parser.add_argument('--exclude-all-fail', action='store_true', help='[model] Also exclude all-fail items')

    args = parser.parse_args()

    if args.mode == 'all':
        mode_all(args)
    elif args.mode == 'rule':
        mode_rule(args)
    elif args.mode == 'model':
        mode_model(args)


if __name__ == '__main__':
    main()
