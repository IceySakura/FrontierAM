#!/usr/bin/env python3
"""从 agentd DB 提取所有太极 URL 分析任务，输出为 replay 数据集."""

import json
import re
import sqlite3

from pathlib import Path


DB_PATH = '/data/workspace/.agentd/state/agentd.sqlite'
OUTPUT_DIR = Path('/data/workspace/taiji_replay/episodes')

TAIJI_URL_RE = re.compile(r'https?://hunyuanaide\.taiji\.woa\.com/web/pre_training_inst_list\?[^\s\]）」]+')
INST_ID_RE = re.compile(r'instId=([a-f0-9]+)')
TASK_ID_RE = re.compile(r'taskID=(basic_train_[^\s&"]+)')
WSID_RE = re.compile(r'wsId=(\d+)')


def extract_url_params(url: str) -> dict:
    inst = INST_ID_RE.search(url)
    task = TASK_ID_RE.search(url)
    wsid = WSID_RE.search(url)
    return {
        'url': url,
        'instId': inst.group(1) if inst else '',
        'taskID': task.group(1) if task else '',
        'wsId': wsid.group(1) if wsid else '',
    }


def extract_user_query(prompt: str) -> str:
    """从 agentd prompt 格式中提取用户原始消息."""
    lines = prompt.strip().split('\n')
    for i, line in enumerate(lines):
        if (
            ': ' in line
            and i > 0
            and line.startswith(lines[0].split(':')[0].strip('[').strip() if '[' in lines[0] else '')
        ):
            return line.split(': ', 1)[1] if ': ' in line else line
    # fallback: 找最后一个包含 taiji URL 的行
    for line in reversed(lines):
        if 'hunyuanaide.taiji.woa.com' in line:
            # 取从 sender: 之后的内容
            if ': ' in line:
                return line.split(': ', 1)[1]
            return line
    return prompt


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, session_id, prompt, final_message_text,
               started_at, finished_at, sender_open_id
        FROM runs
        WHERE prompt LIKE '%hunyuanaide.taiji.woa.com%'
          AND prompt LIKE '%instId=%'
          AND state = 'succeeded'
          AND final_message_text != ''
        ORDER BY started_at DESC
    """).fetchall()

    print(f'找到 {len(rows)} 条成功的太极分析 run')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = []

    for row in rows:
        prompt = row['prompt']
        urls = TAIJI_URL_RE.findall(prompt)
        if not urls:
            continue

        params = extract_url_params(urls[0])
        if not params['instId']:
            continue

        episode = {
            'run_id': row['id'],
            'session_id': row['session_id'],
            'instId': params['instId'],
            'taskID': params['taskID'],
            'wsId': params['wsId'],
            'url': params['url'],
            'query': extract_user_query(prompt),
            'historical_report': row['final_message_text'],
            'started_at': row['started_at'],
            'finished_at': row['finished_at'],
            'sender': row['sender_open_id'],
        }
        episodes.append(episode)

    # 去重：同一 instId 只保留最新的一条
    seen = {}
    deduped = []
    for ep in episodes:
        key = ep['instId']
        if key not in seen:
            seen[key] = True
            deduped.append(ep)

    print(f'去重后 {len(deduped)} 条唯一实例')

    # 保存
    dataset_path = OUTPUT_DIR / 'dataset.json'
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f'已保存到 {dataset_path}')

    # 打印统计
    from collections import Counter
    from datetime import datetime

    dates = Counter()
    for ep in deduped:
        dt = datetime.fromtimestamp(ep['started_at']).strftime('%Y-%m-%d')
        dates[dt] += 1
    print('\n按日期分布（最近 10 天）：')
    for d, c in sorted(dates.items(), reverse=True)[:10]:
        print(f'  {d}: {c}')

    conn.close()


if __name__ == '__main__':
    main()
