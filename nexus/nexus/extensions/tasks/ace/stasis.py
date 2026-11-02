#!/usr/bin/env python3
"""按 category 统计 ACEBench 测试结果"""

import json
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def multiplt_turn_accuracy(score_list):
    end_score_list = []
    process_score_list = []
    for score in score_list:
        end_score = 0 if False in score['valid'] else 1

        process_score = score['valid'].count(True) / len(score['valid'])
        process_score = round(process_score, 3)

        end_score_list.append(end_score)
        process_score_list.append(process_score)
    end_score_total = round(sum(end_score_list) / len(end_score_list), 3)
    process_score_total = round(sum(process_score_list) / len(process_score_list), 3)
    return end_score_total, process_score_total


def agent_eval_process(model_results, possible_answers, correct_list):
    individual_accuracies = []  # Used to store the accuracy of each data point
    total_accuracy = 0  # Store the total accuracy of all data

    for index in range(len(model_results)):
        if index in correct_list:
            accuracy = 1.00
            total_accuracy += 1.00
            continue
        call_process = possible_answers[index]
        model_result = model_results[index]

        if isinstance(call_process[0], list):
            max_accuracy = -1
            for call_process_item in call_process:
                result_len = len(model_result)
                milestone_len = len(call_process_item)

                result_indices = []
                current_index = 0

                # Iterate through each element in call_process and search sequentially
                for stone in call_process_item:
                    # Start searching from the current index until the corresponding call_process element is found
                    while current_index < result_len:
                        if model_result[current_index].strip() == stone.strip():
                            result_indices.append(current_index)
                            current_index += 1
                            break
                        current_index += 1

                # Calculate call_process accuracy using floating-point division
                accuracy = 1.0 if milestone_len == 0 else len(result_indices) / milestone_len

                rounded_accuracy = round(accuracy, 3)
                if rounded_accuracy > max_accuracy:
                    max_accuracy = rounded_accuracy

            # Save the accuracy of each data point
            if accuracy != 1.00:
                individual_accuracies.append({
                    'process_accuracy': rounded_accuracy,
                    'model_output': model_result,
                    'call_process': call_process,
                })
            # Accumulate total accuracy
            total_accuracy += max_accuracy

        # For a single answer, calculate directly
        else:
            result_len = len(model_result)
            milestone_len = len(call_process)

            result_indices = []
            current_index = 0

            # Iterate through each element in call_process and search sequentially
            for stone in call_process:
                while current_index < result_len:
                    if model_result[current_index].strip() == stone.strip():
                        result_indices.append(current_index)
                        current_index += 1  # Update position and continue searching for the next stone
                        break
                    current_index += 1

            # Calculate call_process accuracy using floating-point division
            accuracy = 1.0 if milestone_len == 0 else len(result_indices) / milestone_len

            rounded_accuracy = round(accuracy, 3)

            # Save the accuracy of each data point

            if accuracy != 1.00:
                individual_accuracies.append({
                    'process_accuracy': rounded_accuracy,
                    'model_output': model_result,
                    'call_process': call_process,
                })

            # Accumulate total accuracy
            total_accuracy += accuracy

    # Calculate the overall accuracy of all entries
    overall_accuracy = total_accuracy / len(model_results)
    return round(overall_accuracy, 3)  # Keep two decimal places


def analyze_by_category(input_file, results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 category 统计结果"""

    stats = defaultdict(
        lambda: {'total': 0, 'correct': 0, 'failed': 0, 'accuracy': 0.0, 'errors': defaultdict(int), 'samples': []}
    )

    score_list_user_adjust = []
    score_list_user_switch = []
    user_adjust, user_switch = False, False
    correct_list_agent_multi_step = []
    correct_list_agent_multi_turn = []
    model_results_agent_multi_step, possible_answers_agent_multi_step = [], []
    model_results_agent_multi_turn, possible_answers_agent_multi_turn = [], []
    idx_agent_multi_turn, idx_agent_multi_step = 0, 0
    for item in results:
        # 跳过非成功状态
        if item.get('status') != 'success':
            continue

        gt = json.loads(item['task_output']['ground_truth'])['ground_truth']
        ms = json.loads(item['task_output']['ground_truth'])['mile_stone']

        item['task_output']['ground_truth'] = gt
        item['task_output']['mile_stone'] = ms

        task_output = item.get('task_output', {})
        metadata = item.get('metadata', {})

        category = metadata.get('category', 'unknown')
        correct = task_output.get('correct', False)

        model_tool_calls_format = []
        for tool_call in task_output['tool_calls']:
            model_tool_calls_format.append(list(tool_call.values())[0])

        if 'agent_multi_step' in category:
            if correct:
                correct_list_agent_multi_step.append(idx_agent_multi_step)
            model_results_agent_multi_step.append(model_tool_calls_format)
            possible_answers_agent_multi_step.append(ms)
            idx_agent_multi_step += 1
        if 'agent_multi_turn' in category:
            if correct:
                correct_list_agent_multi_turn.append(idx_agent_multi_turn)
            model_results_agent_multi_turn.append(model_tool_calls_format)
            possible_answers_agent_multi_turn.append(ms)
            idx_agent_multi_turn += 1

        # 统计
        stats[category]['total'] += 1
        if correct:
            stats[category]['correct'] += 1
        else:
            stats[category]['failed'] += 1

            # 记录错误类型
            error_type = task_output.get('checker_error_type', 'unknown')
            stats[category]['errors'][error_type] += 1

        if 'normal_multi_turn_user_adjust' in category:
            user_adjust = True
            turn = item.get('run_id', '').split('_')[-2]
            item2 = item.get('run_id', '').split('_')[-1]

            if len(score_list_user_adjust) > 0 and turn == score_list_user_adjust[-1]['turn']:
                score_list_user_adjust[-1]['valid'].append([item['task_output']['correct']])
                score_list_user_adjust[-1]['number'] = item2
            else:
                score_list_user_adjust.append({
                    'turn': turn,
                    'number': item2,
                    'valid': [item['task_output']['correct']],
                })

        if 'normal_multi_turn_user_switch' in category:
            user_switch = True
            turn = item.get('run_id', '').split('_')[-2]
            item2 = item.get('run_id', '').split('_')[-1]

            if len(score_list_user_switch) > 0 and turn == score_list_user_switch[-1]['turn']:
                score_list_user_switch[-1]['valid'].append(item['task_output']['correct'])
                score_list_user_switch[-1]['number'] = item2
            else:
                score_list_user_switch.append({
                    'turn': turn,
                    'number': item2,
                    'valid': [item['task_output']['correct']],
                })

    if 'agent' in input_file or 'test_all' in input_file:
        stats['agent_multi_step']['process_accuracy'] = agent_eval_process(
            model_results_agent_multi_step, possible_answers_agent_multi_step, correct_list_agent_multi_step
        )
        stats['agent_multi_turn']['process_accuracy'] = agent_eval_process(
            model_results_agent_multi_turn, possible_answers_agent_multi_turn, correct_list_agent_multi_turn
        )

    if user_adjust:
        if len(score_list_user_adjust) == 0:
            end_accuracy, process_accuracy = 0, 0
        else:
            end_accuracy, process_accuracy = multiplt_turn_accuracy(score_list_user_adjust)

        stats['normal_multi_turn_user_adjust']['accuracy'] = end_accuracy
        stats['normal_multi_turn_user_adjust']['process_accuracy'] = process_accuracy

    if user_switch:
        if len(score_list_user_switch) == 0:
            end_accuracy, process_accuracy = 0, 0
        else:
            end_accuracy, process_accuracy = multiplt_turn_accuracy(score_list_user_switch)

        stats['normal_multi_turn_user_switch']['accuracy'] = end_accuracy
        stats['normal_multi_turn_user_switch']['process_accuracy'] = process_accuracy

    # 计算准确率
    for category, stat in stats.items():
        if stat['total'] > 0 and 'normal_multi_turn' not in category:
            stat['accuracy'] = stat['correct'] / stat['total']

    return dict(stats)


def print_summary(stats: dict[str, dict[str, Any]]):
    """打印统计摘要"""

    print('\n' + '=' * 100)
    print(
        f'{"Category":<35} {"Total":>8} {"Correct":>8} {"Failed":>8} {"Process Accuracy":>20} {"End-to-End Accuracy":>22}'
    )

    print('=' * 100)

    # 按 category 名称排序
    sorted_categories = sorted(stats.keys())

    total_all = 0
    correct_all = 0

    for category in sorted_categories:
        stat = stats[category]
        total = stat['total']
        correct = stat['correct']
        failed = stat['failed']
        process_accuracy = stat.get('process_accuracy', 0)
        accuracy = stat['accuracy']

        total_all += total
        correct_all += correct

        # 颜色标记（高准确率=绿，低=红）
        if accuracy >= 0.9:
            marker = '✅'
        elif accuracy >= 0.7:
            marker = '⚠️'
        else:
            marker = '❌'

        print(
            f'{marker} {category:<33} '
            f'{total:>8} '
            f'{correct:>8} '
            f'{failed:>8} '
            f'{process_accuracy * 100:>19.2f}% '
            f'{accuracy * 100:>15.2f}%'
        )

    print('=' * 100)
    overall_accuracy = correct_all / total_all if total_all > 0 else 0
    print(
        f'{"OVERALL":<40} {total_all:<10} {correct_all:<10} {total_all - correct_all:<8}       {overall_accuracy * 100:>6.2f}%      '
    )
    print('=' * 100)


def export_csv(stats: dict[str, dict[str, Any]], output_file: Path):
    """导出 CSV 格式"""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Category', 'Total', 'Correct', 'Failed', 'Process Accuracy', 'End-to-End Accuracy'])

        for category in sorted(stats.keys()):
            stat = stats[category]
            writer.writerow([
                category,
                stat['total'],
                stat['correct'],
                stat['failed'],
                f'{stat.get("process_accuracy", 0) * 100:.2f}%{stat["accuracy"] * 100:.2f}%',
            ])

        # Overall
        total_all = sum(s['total'] for s in stats.values())
        correct_all = sum(s['correct'] for s in stats.values())
        accuracy_all = correct_all / total_all if total_all > 0 else 0
        writer.writerow(['OVERALL', total_all, correct_all, total_all - correct_all, f'{accuracy_all * 100:.2f}%'])

    print(f'\n✅ CSV exported to: {output_file}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python analyze_by_category.py <results.jsonl> [--csv output.csv] [--verbose]')
        print('\nExample:')
        print('  python analyze_by_category.py normal_zh.jsonl')
        print('  python analyze_by_category.py normal_zh.jsonl --csv results.csv')
        print('  python analyze_by_category.py normal_zh.jsonl --verbose')
        sys.exit(1)

    input_file = Path(sys.argv[1])

    # Parse options
    export_csv_file = None
    verbose = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--csv' and i + 1 < len(sys.argv):
            export_csv_file = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--verbose':
            verbose = True
            i += 1
        else:
            i += 1

    if not input_file.exists():
        print(f'❌ File not found: {input_file}')
        sys.exit(1)

    print(f'\n🔍 Loading results from: {input_file}')
    results = load_jsonl(input_file)
    print(f'📊 Loaded {len(results)} results')

    print('\n🧮 Analyzing by category...')
    stats = analyze_by_category(str(input_file), results)

    # Print summary
    if verbose:
        print_summary(stats)
    if export_csv_file is not None:
        export_csv(stats, export_csv_file)


if __name__ == '__main__':
    main()
