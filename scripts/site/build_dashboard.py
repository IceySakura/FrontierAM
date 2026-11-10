"""Build the FrontierAM task dashboard: a static, read-only, self-contained website.

Sources: dashboard/site.json and tasks/*/meta.json (curated text and published results), public task statements
(from the sealed public packages), the README of each graded submission (its own description of its method) and
aggregate breakdowns computed from result files. Nothing from held-out inputs, truths or seeds is read into a page;
breakdowns are per group or family only.

Usage: build_dashboard.py [--out var/dist/dashboard]
"""

import argparse
import csv
import html
import io
import json
import math
import re
import shutil
import subprocess
import tarfile

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'dashboard'
MATH = re.compile(r'(\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|(?<![\\$])\$[^$\n]+?\$)', re.S)
LINK = re.compile(r'\[([^\]]+)\]\((?!https?://)[^)]+\)')
MODEL_NAMES = {'gpt-6.1-sol': 'GPT-6.1 Sol', 'gpt-6-astra': 'GPT-6 Astra', 'claude-opus-5-5': 'Claude Opus 5.5',
               'hy4': 'Hunyuan 4', 'step-5-preview': 'Step-5 Preview', 'deepseek-v4-pro': 'DeepSeek V4 Pro',
               'minimax-m3.1-flash': 'MiniMax M3.1 Flash'}
GROUP_NAMES = {'linear': '线性', 'quadratic': '二次', 'soc': '二阶锥', 'psd': '半正定', 'exponential': '指数锥'}


def md(text: str) -> str:
    """Markdown to HTML with TeX kept verbatim for KaTeX and relative links shown as text."""
    if not text:
        return ''
    stash = []

    def keep(match):
        stash.append(match.group(0))
        return f'\x00{len(stash) - 1}\x00'

    text = LINK.sub(r'\1', MATH.sub(keep, text))
    out = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    return re.sub('\x00(\\d+)\x00', lambda m: html.escape(stash[int(m.group(1))], quote=False), out)


def statement(spec: dict) -> str:
    if spec['kind'] == 'file':
        return (ROOT / spec['path']).read_text()
    parts = []
    with tarfile.open(ROOT / spec['path']) as archive:
        for member in spec['members']:
            parts.append(archive.extractfile(member).read().decode())
    return '\n\n---\n\n'.join(parts)


def readme(store: str, submission: str) -> str | None:
    archive = ROOT / store / 'submissions' / submission / 'source.tar'
    if not archive.exists():
        return None
    with tarfile.open(archive) as tar:
        for name in ('README.md', './README.md'):
            try:
                text = tar.extractfile(name).read().decode(errors='replace')
            except KeyError:
                continue
            return text if len(text) <= 12000 else text[:12000] + '\n\n…（以下省略）'
    return None


def score_class(score) -> str:
    if score is None:
        return ''
    return 's-high' if score >= 50 else 's-mid' if score > 0 else 's-low'


def fmt_score(score) -> str:
    return '—' if score is None else f'{score:.1f}'


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


# Breakdowns (aggregates only) ---------------------------------------------------------------------------------------

def conic_counts(path: Path) -> dict:
    counts: dict = defaultdict(lambda: defaultdict(int))
    with path.open() as stream:
        for row in csv.DictReader(stream):
            group = row['group']
            counts['n'][group] += 1
            for key in row:
                if key.endswith('_solved') and row[key] == '1':
                    counts[key[:-len('_solved')]][group] += 1
    return counts


def breakdown_html(spec: dict | None, results: list[dict]) -> str:
    if not spec:
        return ''
    if spec['kind'] == 'conic_csv':
        counts = conic_counts(ROOT / spec['path'])
        keys = [('reference', 'Clarabel（参考）')] + [(r['csv_key'], model_name(r['model'])) for r in results if r.get('csv_key')]
        groups = sorted(counts['n'], key=lambda g: list(GROUP_NAMES).index(g) if g in GROUP_NAMES else 9)
        head = ''.join(f'<th class="num">{html.escape(label)}</th>' for _, label in keys)
        rows = ''.join(
            f'<tr><td>{GROUP_NAMES.get(g, g)}</td><td class="num">{counts["n"][g]}</td>'
            + ''.join(f'<td class="num">{counts[k][g]}</td>' for k, _ in keys) + '</tr>' for g in groups)
        total = f'<tr><th>合计</th><th class="num">{sum(counts["n"].values())}</th>' + ''.join(
            f'<th class="num">{sum(counts[k].values())}</th>' for k, _ in keys) + '</tr>'
        return (f'<h4>按问题类型的解出数（本机重标定时限）</h4><table><tr><th>类型</th><th class="num">实例</th>{head}</tr>'
                f'{rows}{total}</table>')
    if spec['kind'] == 'suite_report':
        sources = spec.get('paths') or [{'label': model_name(results[0]['model']), 'path': spec['path']}]
        reports = [(s['label'], json.loads((ROOT / s['path']).read_text())) for s in sources]
        reference = conic_counts(ROOT / 'tasks/conic/analysis/conic-per-case-summary.csv')
        first = reports[0][1]
        groups = sorted(first['groups'], key=lambda g: list(GROUP_NAMES).index(g) if g in GROUP_NAMES else 9)
        head = ''.join(f'<th class="num">{html.escape(label)}</th>' for label, _ in reports)
        rows = ''.join(f'<tr><td>{GROUP_NAMES.get(g, g)}</td><td class="num">{first["groups"][g]["cases"]}</td>'
                       f'<td class="num">{reference["reference"][g]}</td>'
                       + ''.join(f'<td class="num">{r["groups"][g]["successes"]}</td>' for _, r in reports) + '</tr>'
                       for g in groups)
        total = first.get('cases_graded', first.get('cases'))
        return ('<h4>按问题类型的解出数</h4><table><tr><th>类型</th><th class="num">实例</th><th class="num">Clarabel（参考）</th>'
                f'{head}</tr>{rows}'
                f'<tr><th>合计</th><th class="num">{total}</th><th class="num">{sum(reference["reference"].values())}</th>'
                + ''.join(f'<th class="num">{r["successes"]}</th>' for _, r in reports) + '</tr></table>')
    if spec['kind'] == 'exp2_score':
        cases = {c['id']: c for c in json.loads((ROOT / spec['manifest']).read_text())['cases']}
        sources = spec.get('paths') or [{'label': model_name(results[0]['model']), 'path': spec['path']}]
        columns, winners = [], defaultdict(int)
        for source in sources:
            score = json.loads((ROOT / source['path']).read_text())
            family, setting = defaultdict(list), defaultdict(list)
            for case_id, result in score['case_results'].items():
                case = cases[case_id]
                if case['role'] != 'scored' or not result.get('relative_time'):
                    continue
                family[case['family']].append(math.log(result['relative_time']))
                setting[(case['scenario'], case['precision'])].append(math.log(result['relative_time']))
            columns.append((source['label'], family, setting))
            if not winners:
                for member in (score.get('winners') or {}).values():
                    winners[member] += 1
        columns = [c for c in columns if c[1]]
        if not columns:
            return ''
        geo = lambda values: math.exp(sum(values) / len(values)) if values else None  # noqa: E731
        cell = lambda values: f'<td class="num">{geo(values):.2f}</td>' if values else '<td></td>'  # noqa: E731
        head = ''.join(f'<th class="num">{html.escape(label)}</th>' for label, _, _ in columns)
        first = columns[0]
        rows = ''.join(f'<tr><td>{html.escape(f)}</td><td class="num">{len(first[1][f])}</td>'
                       + ''.join(cell(c[1].get(f)) for c in columns) + '</tr>'
                       for f in sorted(first[1], key=lambda f: geo(first[1][f])))
        rows2 = ''.join(f'<tr><td>{s} / {p}</td><td class="num">{len(first[2][(s, p)])}</td>'
                        + ''.join(cell(c[2].get((s, p))) for c in columns) + '</tr>'
                        for s, p in sorted(first[2]))
        return ('<h4>耗时比（模型 / 参照组逐例最快者，几何平均，越小越快）</h4>'
                f'<table><tr><th>结构族</th><th class="num">会话</th>{head}</tr>{rows}</table>'
                f'<table><tr><th>场景 / 精度档</th><th class="num">会话</th>{head}</tr>{rows2}</table>'
                '<p>参照组逐例最快的成员：' + '、'.join(f'{m} {n} 例' for m, n in sorted(winners.items(), key=lambda i: -i[1]))
                + '。</p>')
    return ''


# Pages ----------------------------------------------------------------------------------------------------------------

def page(title: str, body: str, depth: int = 0) -> str:
    up = '../' * depth
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="{up}assets/style.css"><link rel="stylesheet" href="{up}assets/katex/katex.min.css">
<script defer src="{up}assets/katex/katex.min.js"></script><script defer src="{up}assets/katex/auto-render.min.js"></script>
<script>document.addEventListener("DOMContentLoaded",function(){{renderMathInElement(document.body,{{delimiters:[
{{left:"$$",right:"$$",display:true}},{{left:"\\\\[",right:"\\\\]",display:true}},{{left:"\\\\(",right:"\\\\)",display:false}},
{{left:"$",right:"$",display:false}}],throwOnError:false}});}});</script></head><body>
<header><a class="brand" href="{up}index.html">FrontierAM 任务看板</a>
<nav><a href="{up}index.html">总榜</a><a href="{up}protocol.html">评测协议</a></nav></header>
<main>{body}</main><footer>{FOOTER}</footer></body></html>'''


def anchor_of(r: dict) -> str:
    return re.sub(r'[^\w.]+', '-', f'{r["model"]}-{r["protocol"]}-{r.get("setting", "")}').strip('-')


def protocol_label(r: dict) -> str:
    return f'{r["protocol"]} {r["setting"]}' if r.get('setting') else r['protocol']


def review_tag(r: dict) -> str:
    """Above-reference scores stay 'pending review' until results/BASELINE_REVIEW.md records a conclusion."""
    if r.get('score') is None or r['score'] <= 50:
        return ''
    return f' <span class="tag warn">{html.escape(r.get("review", "待复核"))}</span>'


def process_cell(task: dict, r: dict) -> str:
    """Test submissions (iterative) and development time used, from the attempt's result.json (build_task_pages.py)."""
    setting = {'单轮': 'single', '迭代': 'iterative'}.get(r.get('setting'), 'single')
    path = ROOT / 'tasks' / task['name'] / 'solutions' / (r.get('protocol') or 'v6.2') / f'{r["model"]}-{setting}-{r.get("attempt", 1)}' / 'result.json'
    process = json.loads(path.read_text()).get('process') if path.exists() else None
    if not process:
        return ''
    return (f'{process["public_tests"]} 次提交，' if setting == 'iterative' else '') + f'{process["minutes"]:.0f} 分钟'


def result_rows(task: dict, group: dict) -> str:
    rows = []
    for r in group['results']:
        rows.append(f'<tr><td><a href="#{anchor_of(r)}">{html.escape(model_name(r["model"]))}</a></td>'
                    f'<td>{html.escape(protocol_label(r))}</td>'
                    f'<td>{md(r["raw"])[3:-4]}</td><td>{md(r["mapping"])[3:-4]}</td>'
                    f'<td class="num score {score_class(r["score"])}">{fmt_score(r["score"])}{review_tag(r)}</td>'
                    f'<td>{html.escape(process_cell(task, r))}</td></tr>')
    return ('<table><tr><th>模型</th><th>协议</th><th>原始指标</th><th>换算</th><th class="num">得分（满分 100）</th>'
            '<th>提交次数与用时</th></tr>' + ''.join(rows) + '</table>')


def task_page(task: dict) -> str:
    status = '<span class="tag stop">评测暂停</span>' if task.get('status') == 'paused' else ''
    body = [f'<h1>{html.escape(task.get("name", task["title"]))} {status}</h1><p class="lead">{html.escape(task["category"])} · '
            f'{md(task["summary"])[3:-4]}</p>',
            '<h2>题目背景</h2>', md(task['background_md']),
            f'<details><summary>公开题面原文（给模型的版本）</summary><div class="md">{md(statement(task["statement"]))}</div></details>',
            '<h2>评测依据</h2>', md(task['judge_md']),
            '<h2>Baseline：做法与本机实测结果</h2>', md(task['baseline_md']),
            '<h2>模型成绩</h2>']
    for group in task['result_groups']:
        body += [f'<h3>{html.escape(group["title"])}</h3>', result_rows(task, group),
                 breakdown_html(group.get('breakdown'), group['results'])]
    body.append('<h2>各模型的做法</h2>')
    for group in task['result_groups']:
        for r in group['results']:
            body.append(f'<h3 id="{anchor_of(r)}">{html.escape(model_name(r["model"]))}（{html.escape(protocol_label(r))}）'
                        f' <span class="score {score_class(r["score"])}">{fmt_score(r["score"])}</span>{review_tag(r)}</h3>')
            if r.get('notes_md'):
                body.append(f'<div class="note">{md(r["notes_md"])}</div>')
            text = readme(r['store'], r['submission']) if r.get('submission') else None
            if text:
                body.append(f'<details><summary>模型自己写的说明（提交中的 README.md）</summary><div class="md">{md(text)}</div></details>')
            elif not r.get('notes_md'):
                body.append('<p>没有可展示的提交。</p>')
    if task.get('notes_md'):
        body += ['<h2>观察与说明</h2>', md(task['notes_md'])]
    return page(task['title'], '\n'.join(body), depth=1)


def index_page(site: dict, tasks: list[dict]) -> str:
    models = []
    for task in tasks:
        for group in task['result_groups']:
            for r in group['results']:
                if r['model'] not in models:
                    models.append(r['model'])
    head = ''.join(f'<th>{html.escape(model_name(m))}</th>' for m in models)
    rows = []
    for task in tasks:
        for group in task['result_groups']:
            cells = []
            for m in models:
                hit = [r for r in group['results'] if r['model'] == m]
                if len(hit) == 1:
                    cells.append(f'<td class="score {score_class(hit[0]["score"])}">{fmt_score(hit[0]["score"])}{review_tag(hit[0])}</td>')
                elif hit:
                    cells.append('<td>' + '<br>'.join(
                        f'<small>{html.escape(r.get("setting", ""))}</small> '
                        f'<span class="score {score_class(r["score"])}">{fmt_score(r["score"])}</span>{review_tag(r)}' for r in hit) + '</td>')
                else:
                    cells.append('<td></td>')
            label = f'<a href="tasks/{task["id"]}.html">{html.escape(task.get("name", task["title"]))}</a>'
            rows.append(f'<tr><td>{label}<br><small>{html.escape(group["title"])}</small></td>{"".join(cells)}</tr>')
    cards = ''.join(
        f'<div class="card"><h3><a href="tasks/{t["id"]}.html">{html.escape(t.get("name", t["title"]))}</a></h3>'
        f'<p>{html.escape(t["category"])}</p><p>{md(t["summary"])[3:-4]}</p>'
        + ('<p><span class="tag stop">评测暂停</span></p>' if t.get('status') == 'paused' else '') + '</div>' for t in tasks)
    body = (f'<h1>{html.escape(site["title"])}</h1><p class="lead">{html.escape(site["subtitle"])}</p>{md(site["intro_md"])}'
            f'<h2>题目</h2><div class="cards">{cards}</div>'
            '<h2>总榜（满分 100，本机参照实现 = 50）</h2>'
            '<p>只列每道题的最新版本和最新协议（v6.2）下的成绩。超过参照（50 分）的成绩在复核结论出来前标“待复核”：复核会检查超出的原因，并寻找更强的人类方法。空格表示该模型没有在这道题上尝试；“—”表示尚未评测。点击题目查看评测依据、baseline 和各模型的做法。</p>'
            f'<table class="matrix"><tr><th>题目 / 版本</th>{head}</tr>{"".join(rows)}</table>')
    return page(site['title'], body)


FOOTER = ''


def main():
    global FOOTER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'var/dist/dashboard')
    args = parser.parse_args()
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    FOOTER = (f'FrontierAM · 构建于 {datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")} · 代码版本 {commit} · '
              '所有分数为本机实测；隐藏测试数据不在本站公开')
    site = json.loads((SRC / 'site.json').read_text())
    tasks = sorted((json.loads(p.read_text()) for p in ROOT.glob('tasks/*/meta.json')), key=lambda t: t['order'])
    staging = args.out.with_name(args.out.name + '.new')
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'tasks').mkdir(parents=True)
    shutil.copytree(SRC / 'assets', staging / 'assets')
    (staging / 'index.html').write_text(index_page(site, tasks))
    (staging / 'protocol.html').write_text(page('评测协议', md(site['protocol_md'])))
    for task in tasks:
        (staging / 'tasks' / f'{task["id"]}.html').write_text(task_page(task))
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.rename(args.out)
    print(json.dumps({'out': str(args.out), 'tasks': len(tasks), 'commit': commit}))


if __name__ == '__main__':
    main()
