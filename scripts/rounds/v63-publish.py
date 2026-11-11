"""Publish the v6.3 validation round (gpt-6.1-sol) into tasks/<family>/meta.json as a "协议 v6.3" result group.

Reads tasks/<family>/solutions/v6.3/<model>-<setting>-<n>/score.json (written by scripts/rounds/v63-grade.sh) and the
attempt logs for the submission IDs; writes raw metric, mapping and score per cell. Reviews (results/BASELINE_REVIEW.md)
are kept from an earlier publish. Cells without a score.json are left out.

Usage: v63-publish.py
"""

import json
import math

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'conic': 'conic', 'exponential-v3': 'exponential', 'galaxy-gravity-v3': 'galaxy-gravity',
         'polarization-demosaicking-v2': 'polarization-demosaicking', 'species-tree-v3': 'species-tree',
         'trajectory-covariance-v2': 'trajectory-covariance'}
SETTING = {'single': '单轮', 'iterative': '迭代'}
TITLE = '协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 至 11-11）'
LINEAR = '\\(E \\ge U\\) 记 0；\\(W \\le E < U\\) 为 \\(50\\,(U-E)/(U-W)\\)；\\(E < W\\) 为 \\(100 - 50\\,E/W\\)'


def submission_of(task: str, setting: str, n: int) -> str | None:
    log = ROOT / f'var/stores/store/reruns/v63-sol-{setting}-{task}-{n}.log'
    rows = [json.loads(line) for line in log.read_text().splitlines() if 'develop_finished' in line] if log.exists() else []
    return rows[-1]['submission'] if rows else None


def baseline_anchors(task: str) -> dict:
    root = ROOT / 'var/stores/store-v63/baselines' / task
    active = json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())
    return {w: active['metrics']['parts'][w]['raw_error'] for w in ('weak', 'strong')}


def cell(task: str, score: dict) -> tuple[str, str]:
    family = TASKS[task]
    if family == 'conic':
        return (f'{score["successes"]} / {score["cases"]}（\\(p = {score["p"]:.3f}\\)，参照 \\(p_B = {score["p_B"]:.3f}\\)）',
                '\\(p \\le 0.9\\,p_B\\) 记 0；否则 \\(50\\,(p-0.9p_B)/(0.1p_B)\\)，高于 \\(p_B\\) 为 \\(50 + 50\\,(p-p_B)/(1-p_B)\\)')
    if family == 'exponential':
        if score.get('status') != 'ok':
            return f'无效提交（{len(score.get("failed_cases") or {})} 例不合格）', '任一用例不合格记 0'
        return (f'74/74 合格，\\(R = {score["relative_time"]:.3f}\\)（对参照组逐例最快成员，含 Chebyshev v2）',
                '\\(100/(1+R)\\)')
    candidate = score.get('candidate') or {}
    raw = candidate.get('raw_error')
    if family == 'galaxy-gravity':
        anchors = baseline_anchors(task)
        return (f'\\(E = {raw:.3g}\\)，单次最多 {candidate.get("max_cpu_s", 0):.2f} CPU 秒（参照 r2：\\(W = {anchors["strong"]:.3g}\\)）',
                '对数刻度：\\(E < W\\) 为 \\(50 + 50\\,\\log(W/E)/\\log(W/10^{-15})\\)；\\(W \\le E < U\\) 为 \\(50\\,\\log(U/E)/\\log(U/W)\\)')
    if family == 'polarization-demosaicking':
        anchors = baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有场景失败"}）', '任一场景失败记 0'
        return f'RMSE \\(E = {raw:.4f}\\)（\\(U = {anchors["weak"]:.4f}\\)，\\(W = {anchors["strong"]:.4f}\\)）', LINEAR
    weak, strong = score['weak']['raw_error'], score['strong']['raw_error']
    name = 'nRF' if family == 'species-tree' else '白化误差'
    return f'{name} \\(E = {raw:.4f}\\)（\\(U = {weak:.4f}\\)，\\(W = {strong:.4f}\\)）', LINEAR


def main():
    for task, family in TASKS.items():
        path = ROOT / 'tasks' / family / 'meta.json'
        meta = json.loads(path.read_text())
        old = next((g for g in meta['result_groups'] if g.get('protocol') == 'v6.3'), None)
        reviews = {(r['model'], r['setting']): {k: r[k] for k in ('review', 'notes_md') if k in r}
                   for r in (old or {}).get('results', [])}
        results = []
        for setting in ('single', 'iterative'):
            folder = ROOT / 'tasks' / family / 'solutions/v6.3' / f'gpt-6.1-sol-{setting}-1'
            if not (folder / 'score.json').exists():
                continue
            score = json.loads((folder / 'score.json').read_text())
            raw, mapping = cell(task, score)
            value = score.get('score')
            results.append({'model': 'gpt-6.1-sol', 'protocol': 'v6.3', 'setting': SETTING[setting], 'attempt': 1,
                            'submission': submission_of(task, setting, 1), 'store': 'var/stores/store-v63', 'raw': raw,
                            'mapping': mapping, 'score': None if value is None or (isinstance(value, float) and math.isnan(value)) else round(value, 3),
                            **reviews.get(('gpt-6.1-sol', SETTING[setting]), {})})
        version = 'v3' if family not in ('polarization-demosaicking', 'trajectory-covariance') else 'v2'
        group = {'title': TITLE, 'version': version, 'protocol': 'v6.3', 'results': results}
        meta['result_groups'] = [group] + [g for g in meta['result_groups'] if g.get('protocol') != 'v6.3']
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'task': family, 'cells': [(r['setting'], r['score']) for r in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
