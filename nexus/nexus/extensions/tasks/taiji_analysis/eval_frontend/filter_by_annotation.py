#!/usr/bin/env python3
"""Filter dataset based on human annotations from the eval frontend.

Reads annotations.json produced by the eval frontend, then filters
the dataset parquet by annotation label and/or benchmark_quality.

Available labels: correct, data_issue, inaccessible, bad_gt, ambiguous, others
Available quality: 非常适合, 一般, 不适合

Usage examples:
    # Remove invalid data
    python3 filter_by_annotation.py --exclude-labels data_issue,inaccessible,bad_gt

    # Also exclude low quality cases
    python3 filter_by_annotation.py --exclude-labels data_issue,inaccessible,bad_gt --exclude-quality 不适合

    # Only keep high quality cases
    python3 filter_by_annotation.py --include-quality 非常适合

    # Dry run: print stats only
    python3 filter_by_annotation.py --exclude-labels data_issue --dry-run
"""

import argparse
import json

from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path('/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/nexus_results')


def load_annotations(results_dir: Path) -> dict:
    path = results_dir / 'annotations.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_exclude_reasons(results_dir: Path) -> dict:
    path = results_dir / 'exclude_reasons.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_id_col(df: pd.DataFrame) -> str:
    """Detect which column contains the run_id / case identifier."""
    for col in ['run_id', 'instance_id', 'id', 'inst_id']:
        if col in df.columns:
            return col
    raise ValueError(f'Cannot find ID column in parquet. Columns: {list(df.columns)}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter dataset by annotation labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', default='episodes/dataset.parquet', help='Input parquet path')
    parser.add_argument('--output', default=None, help='Output path (default: input with _filtered suffix)')
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'ids'],
        default='parquet',
        help='Output format: parquet file or plain text ID list',
    )
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Directory containing annotations.json')
    parser.add_argument(
        '--exclude-labels',
        default='',
        help='Comma-separated annotation labels to exclude (e.g., data_issue,inaccessible,bad_gt)',
    )
    parser.add_argument(
        '--include-labels', default='', help='Comma-separated annotation labels to keep (only keep these)'
    )
    parser.add_argument(
        '--exclude-quality',
        default='',
        help='Comma-separated benchmark_quality values to exclude (e.g., 不适合)',
    )
    parser.add_argument(
        '--include-quality',
        default='',
        help='Comma-separated benchmark_quality values to keep (e.g., 非常适合,一般)',
    )
    parser.add_argument(
        '--no-exclude-reasons', action='store_true', help='Do NOT auto-exclude cases from exclude_reasons.json'
    )
    parser.add_argument('--dry-run', action='store_true', help='Print stats without writing output')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    annotations = load_annotations(results_dir)
    exclude_reasons = load_exclude_reasons(results_dir)

    # Print annotation summary
    label_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for ann in annotations.values():
        label = ann.get('label', '')
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        q = ann.get('benchmark_quality', '')
        if q:
            quality_counts[q] = quality_counts.get(q, 0) + 1
    print(f'Annotations loaded: {len(annotations)} cases')
    print(f'  Labels: {label_counts}')
    print(f'  Benchmark quality: {quality_counts}')
    if exclude_reasons:
        print(f'  Exclude reasons: {len(exclude_reasons)} cases')

    # Load dataset
    input_path = Path(args.input)
    df = pd.read_parquet(input_path)
    id_col = get_id_col(df)
    original_count = len(df)
    print(f'Input: {input_path} ({original_count} rows, id_col={id_col})')

    # Build set of IDs to exclude
    exclude_ids: set[str] = set()

    # --- Exclude by exclude_reasons.json (inaccessible/bad_gt) ---
    if not args.no_exclude_reasons and exclude_reasons:
        for run_id in exclude_reasons:
            exclude_ids.add(str(run_id))
        print(f'Auto-excluding {len(exclude_reasons)} cases from exclude_reasons.json')

    # --- Exclude by annotation label ---
    if args.exclude_labels:
        exclude_labels = set(args.exclude_labels.split(','))
        before = len(exclude_ids)
        for run_id, ann in annotations.items():
            if ann.get('label') in exclude_labels:
                exclude_ids.add(str(run_id))
        print(f'Excluding {len(exclude_ids) - before} cases with labels: {exclude_labels}')

    # --- Include only specific annotation labels ---
    if args.include_labels:
        include_labels = set(args.include_labels.split(','))
        include_ids = {str(rid) for rid, ann in annotations.items() if ann.get('label') in include_labels}
        all_ids = set(df[id_col].astype(str))
        exclude_ids.update(all_ids - include_ids)
        print(f'Keeping only {len(include_ids)} cases with labels: {include_labels}')

    # --- Exclude by benchmark_quality ---
    if args.exclude_quality:
        exclude_qualities = set(args.exclude_quality.split(','))
        before = len(exclude_ids)
        for run_id, ann in annotations.items():
            if ann.get('benchmark_quality') in exclude_qualities:
                exclude_ids.add(str(run_id))
        print(f'Excluding {len(exclude_ids) - before} cases with quality: {exclude_qualities}')

    # --- Include only specific benchmark_quality ---
    if args.include_quality:
        include_qualities = set(args.include_quality.split(','))
        include_ids = {
            str(rid) for rid, ann in annotations.items() if ann.get('benchmark_quality') in include_qualities
        }
        all_ids = set(df[id_col].astype(str))
        exclude_ids.update(all_ids - include_ids)
        print(f'Keeping only {len(include_ids)} cases with quality: {include_qualities}')

    # Apply filter
    mask = ~df[id_col].astype(str).isin(exclude_ids)
    df_filtered = df[mask]
    final_count = len(df_filtered)

    print(f'\nResult: {original_count} → {final_count} ({original_count - final_count} removed)')

    if args.dry_run:
        print('(dry run, not writing output)')
        return

    # Determine output path
    out_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + '_filtered')

    if args.output_format == 'ids':
        if not args.output:
            out_path = out_path.with_suffix('.txt')
        with open(out_path, 'w') as f:
            for rid in sorted(df_filtered[id_col].astype(str)):
                f.write(f'{rid}\n')
        print(f'Written ID list: {out_path}')
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_parquet(out_path, index=False)
        print(f'Written parquet: {out_path}')


if __name__ == '__main__':
    main()
