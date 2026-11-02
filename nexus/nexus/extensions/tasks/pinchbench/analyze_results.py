#!/usr/bin/env python3
"""解析 PinchBench batch-run results.jsonl 并生成统计报告"""

import argparse
import json
import logging
import sys

from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_results_jsonl(jsonl_path: Path) -> list[dict]:
    """解析 results.jsonl，提取每个任务的关键信息"""
    rows = []

    if not jsonl_path.exists():
        logger.error('找不到结果文件: %s', jsonl_path)
        return rows

    with open(jsonl_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 提取基本信息
                task_id = data.get('run_id', f'unknown_{line_num}')
                status = data.get('status', 'unknown')
                success = status == 'success'
                error_msg = data.get('error')

                # 提取任务输出信息
                task_output = data.get('task_output', {})
                score = task_output.get('score')
                grading_type = task_output.get('grading_type', 'unknown')

                # 提取 metadata 信息（包含正确的 category）
                metadata = data.get('metadata', {})
                category = metadata.get('category', 'unknown')
                task_name = metadata.get('task_name', '')

                # 计算耗时
                start_time = data.get('start_time')
                end_time = data.get('end_time')
                elapsed_s = None
                if start_time and end_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        elapsed_s = (end_dt - start_dt).total_seconds()
                    except Exception as e:
                        logger.debug('Failed to parse start/end time: %s', e)
                        pass

                rows.append({
                    'task_id': task_id,
                    'task_name': task_name,
                    'category': category,
                    'grading_type': grading_type,
                    'score': score,
                    'success': success,
                    'elapsed_s': elapsed_s,
                    'error': error_msg,
                })

            except json.JSONDecodeError as e:
                logger.warning('第 %d 行 JSON 解析失败: %s', line_num, e)
            except Exception as e:
                logger.warning('第 %d 行处理失败: %s', line_num, e)

    return rows


def print_summary(rows: list[dict], total_elapsed: float | None = None) -> None:
    """打印汇总统计报告（类似 pinchbench_run_all.py 的格式）"""

    if not rows:
        logger.warning('没有找到任何任务结果')
        return

    # 按 category 分组
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r['category'], []).append(r)

    # 计算总耗时
    if total_elapsed is None:
        elapsed_times = [r['elapsed_s'] for r in rows if r['elapsed_s'] is not None]
        total_elapsed = sum(elapsed_times) if elapsed_times else 0.0

    logger.info('')
    logger.info('=' * 64)
    logger.info(
        '📊 PinchBench Batch-Run 结果汇总  (共 %d 个任务，总用时 %.0fs / %.1f min)',
        len(rows),
        total_elapsed,
        total_elapsed / 60,
    )
    logger.info('=' * 64)

    # 分类汇总
    logger.info('')
    logger.info('  %-22s  %6s  %7s  %s', 'CATEGORY', 'SCORE', 'TASKS', '')
    logger.info('  %s', '-' * 50)

    all_scores = [r['score'] for r in rows if r['score'] is not None]
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        earned = sum(r['score'] for r in cat_rows if r['score'] is not None)
        possible = len(cat_rows)
        pct = earned / possible * 100 if possible > 0 else 0
        icon = '🟢' if pct >= 90 else '🟡' if pct >= 60 else '🔴'
        logger.info('  %s %-20s  %5.1f%%  %4d 个', icon, cat.upper(), pct, possible)

    # 总分
    mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall_pct = mean * 100
    logger.info('  %s', '-' * 50)
    logger.info('  整体平均分: %.4f  (%.1f%%)', mean, overall_pct)

    # 明细表
    logger.info('')
    logger.info('  %-34s  %-12s  %6s  %7s', 'task_id', 'grading', 'score', 'elapsed')
    logger.info('  %s', '-' * 66)

    for r in rows:
        status = '✅' if r['success'] and (r['score'] or 0) > 0 else '❌'
        score_str = f'{r["score"]:.2f}' if r['score'] is not None else 'N/A '
        elapsed_str = f'{r["elapsed_s"]:.1f}s' if r['elapsed_s'] is not None else '  N/A'
        note = f'  ← {r["error"][:40]}' if r['error'] else ''
        logger.info(
            '  %s %-32s  %-12s  %6s  %7s%s',
            status,
            r['task_id'],
            r['grading_type'],
            score_str,
            elapsed_str,
            note,
        )

    logger.info('  %s', '-' * 66)

    # 成功/失败统计
    success_count = sum(1 for r in rows if r['success'])
    failed_count = len(rows) - success_count
    logger.info('')
    logger.info(
        '  成功：%d 个  |  失败：%d 个  |  成功率：%.1f%%',
        success_count,
        failed_count,
        success_count / len(rows) * 100 if rows else 0,
    )
    logger.info('')


def main() -> int:
    parser = argparse.ArgumentParser(description='解析 PinchBench batch-run results.jsonl 并生成统计报告')
    parser.add_argument('--results_file', type=Path, help='results.jsonl 文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(message)s')

    # 解析结果
    rows = parse_results_jsonl(args.results_file)

    if not rows:
        logger.error('❌ 没有解析到任何任务结果')
        return 1

    # 打印汇总
    print_summary(rows)

    return 0


if __name__ == '__main__':
    sys.exit(main())
