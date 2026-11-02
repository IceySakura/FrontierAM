#!/usr/bin/env python3
"""解析 WildClawBench batch-run results.jsonl 并生成统计报告。

用法:
    python -m nexus.extensions.tasks.wildclaw_bench.analyze_results --results_file ./outputs/results.jsonl

输出格式匹配原始 WildClawBench ``utils/grading.py`` 中的 ``print_summary`` / ``print_global_summary``。
"""

import argparse
import json
import logging
import sys

from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_results_jsonl(jsonl_path: Path) -> list[dict]:
    """解析 results.jsonl，提取每个任务的关键信息。"""
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
                scores = task_output.get('scores', {})
                overall_score = scores.get('overall_score', score)

                # 提取 metadata 信息
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
                    'score': score,
                    'overall_score': overall_score,
                    'scores': scores,
                    'success': success,
                    'elapsed_s': elapsed_s,
                    'error': error_msg,
                })

            except json.JSONDecodeError as e:
                logger.warning('第 %d 行 JSON 解析失败: %s', line_num, e)
            except Exception as e:
                logger.warning('第 %d 行处理失败: %s', line_num, e)

    return rows


def print_category_summary(rows: list[dict], category: str) -> None:
    """打印单个分类的汇总报告（匹配原始 print_summary 格式）。"""
    cat_rows = [r for r in rows if r['category'] == category]
    if not cat_rows:
        return

    print(f'\n{"#" * 60}')
    print(f'  Summary Report — {category}')
    print(f'{"#" * 60}')

    all_scores: dict[str, float] = {}
    for r in cat_rows:
        task_id = r['task_id']
        if r.get('error'):
            print(f'  ✗ {task_id}: {r["error"]}')
            all_scores[task_id] = 0.0
            continue
        scores = r.get('scores', {})
        if not scores:
            print(f'  - {task_id}: No scores → 0.0')
            all_scores[task_id] = 0.0
            continue
        if 'error' in scores:
            print(f'  ✗ {task_id}: Grading error {scores["error"]}')
            all_scores[task_id] = 0.0
            continue
        numeric_dict = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        if not numeric_dict:
            print(f'  - {task_id}: No valid numeric scores → 0.0')
            all_scores[task_id] = 0.0
            continue

        avg = sum(numeric_dict.values()) / len(numeric_dict)
        print(f'  ✓ {task_id}: avg {avg:.2f}  ({len(numeric_dict)} items)')

        final_score_val = numeric_dict.get('overall_score', avg)
        all_scores[task_id] = final_score_val

    if all_scores:
        print('\n  Final scores per task:')
        for k, score in sorted(all_scores.items()):
            bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
            print(f'    {bar} {score:.2f}  {k}')

    print('#' * 60)


def print_global_summary(rows: list[dict]) -> None:
    """打印全局汇总报告（匹配原始 print_global_summary 格式）。"""
    if not rows:
        logger.warning('没有找到任何任务结果')
        return

    # 按 category 分组
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r['category'], []).append(r)

    # 计算总耗时
    elapsed_times = [r['elapsed_s'] for r in rows if r['elapsed_s'] is not None]
    total_elapsed = sum(elapsed_times) if elapsed_times else 0.0

    logger.info('')
    logger.info('=' * 64)
    logger.info(
        '📊 WildClawBench 结果汇总  (共 %d 个任务，总用时 %.0fs / %.1f min)',
        len(rows),
        total_elapsed,
        total_elapsed / 60,
    )
    logger.info('=' * 64)

    # 分类汇总
    logger.info('')
    logger.info('  %-30s  %6s  %7s  %s', 'CATEGORY', 'SCORE', 'TASKS', '')
    logger.info('  %s', '-' * 55)

    all_final_scores: list[float] = []
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        cat_scores: list[float] = []
        for r in cat_rows:
            if r.get('error'):
                cat_scores.append(0.0)
                continue
            scores = r.get('scores', {})
            if not scores or 'error' in scores:
                cat_scores.append(0.0)
                continue
            numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
            if not numeric:
                cat_scores.append(0.0)
                continue
            final = numeric.get('overall_score', sum(numeric.values()) / len(numeric))
            cat_scores.append(final)

        if cat_scores:
            avg = sum(cat_scores) / len(cat_scores)
            pct = avg * 100
        else:
            pct = 0.0

        icon = '🟢' if pct >= 90 else '🟡' if pct >= 60 else '🔴'
        logger.info('  %s %-28s  %5.1f%%  %4d 个', icon, cat.upper(), pct, len(cat_rows))
        all_final_scores.extend(cat_scores)

    # 总分
    if all_final_scores:
        global_avg = sum(all_final_scores) / len(all_final_scores)
        bar = '█' * int(global_avg * 10) + '░' * (10 - int(global_avg * 10))
        logger.info('  %s', '-' * 55)
        logger.info('  整体平均分: %s %.4f  (%.1f%%)', bar, global_avg, global_avg * 100)
        logger.info('  有效评分任务: %d / %d', len(all_final_scores), len(rows))
    else:
        logger.info('  无有效评分数据')

    # 明细表
    logger.info('')
    logger.info('  %-40s  %6s  %7s  %s', 'task_id', 'score', 'elapsed', '')
    logger.info('  %s', '-' * 66)

    for r in rows:
        status = '✅' if r['success'] and (r.get('overall_score') or 0) > 0 else '❌'
        score_str = f'{r["overall_score"]:.2f}' if r.get('overall_score') is not None else 'N/A '
        elapsed_str = f'{r["elapsed_s"]:.1f}s' if r['elapsed_s'] is not None else '  N/A'
        note = f'  ← {r["error"][:40]}' if r.get('error') else ''
        logger.info(
            '  %s %-38s  %6s  %7s%s',
            status,
            r['task_id'],
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
        '  成功: %d 个  |  失败: %d 个  |  成功率: %.1f%%',
        success_count,
        failed_count,
        success_count / len(rows) * 100 if rows else 0,
    )
    logger.info('')


def main() -> int:
    parser = argparse.ArgumentParser(description='解析 WildClawBench batch-run results.jsonl 并生成统计报告')
    parser.add_argument('--results_file', type=Path, required=True, help='results.jsonl 文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s',
    )

    # 解析结果
    rows = parse_results_jsonl(args.results_file)

    if not rows:
        logger.error('❌ 没有解析到任何任务结果')
        return 1

    # 按分类打印
    categories = sorted({r['category'] for r in rows})
    for cat in categories:
        print_category_summary(rows, cat)

    # 打印全局汇总
    print_global_summary(rows)

    return 0


if __name__ == '__main__':
    sys.exit(main())
