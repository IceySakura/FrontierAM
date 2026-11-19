"""Build the FrontierAM task dashboard: a static, read-only, self-contained website.

Sources: dashboard/site.json and tasks/*/meta.json (curated text and published results), public task statements
(from the sealed public packages), the README of each graded submission (its own description of its method) and
aggregate breakdowns computed from result files. Nothing from held-out inputs, truths or seeds is read into a page;
breakdowns are per group or family only.

Usage: build_dashboard.py [--out var/dist/dashboard]
"""

import argparse
import csv
import hashlib
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

from board_stats import CHART_FILES, SETTINGS, averages, chart_svgs, rank_models

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'dashboard'
MATH = re.compile(r'(\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|(?<![\\$])\$[^$\n]+?\$)', re.S)
LINK = re.compile(r'\[([^\]]+)\]\((?!https?://)[^)]+\)')
MODEL_NAMES = {'gpt-6.1-sol': 'GPT-6.1 Sol', 'gpt-6-astra': 'GPT-6 Astra', 'claude-opus-5-5': 'Claude Opus 5.5',
               'hy4': 'Hunyuan 4', 'intern-s2': 'Intern S2', 'step-5-preview': 'Step-5 Preview', 'deepseek-v4-pro': 'DeepSeek V4 Pro',
               'minimax-m3.1-flash': 'MiniMax M3.1 Flash', 'qwen3.8-max-0902': 'Qwen3.8 Max'}
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

# The stylesheet URL carries its content hash, so a browser fetches it again whenever it changes.
CSS_VERSION = hashlib.sha256((SRC / 'assets/style.css').read_bytes()).hexdigest()[:10]


def page(title: str, body: str, depth: int = 0) -> str:
    up = '../' * depth
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="{up}assets/style.css?v={CSS_VERSION}"><link rel="stylesheet" href="{up}assets/katex/katex.min.css">
<script defer src="{up}assets/katex/katex.min.js"></script><script defer src="{up}assets/katex/auto-render.min.js"></script>
<script>document.addEventListener("DOMContentLoaded",function(){{renderMathInElement(document.body,{{delimiters:[
{{left:"$$",right:"$$",display:true}},{{left:"\\\\[",right:"\\\\]",display:true}},{{left:"\\\\(",right:"\\\\)",display:false}},
{{left:"$",right:"$",display:false}}],throwOnError:false}});}});</script></head><body>
<header><a class="brand" href="{up}index.html">FrontierAM 任务看板</a>
<nav><a href="{up}index.html">总榜</a><a href="{up}monitor.html">实时占用</a><a href="{up}process.html">评测流程</a><a href="{up}insights.html">结果解读</a><a href="{up}protocol.html">评测协议</a></nav></header>
<main>{body}</main><footer>{FOOTER}</footer></body></html>'''


def anchor_of(r: dict) -> str:
    return re.sub(r'[^\w.]+', '-', f'{r["model"]}-{r["protocol"]}-{r.get("folder", "")}-{r.get("setting", "")}-{r.get("attempt", 1)}').strip('-')


def protocol_label(r: dict) -> str:
    return f'{r["protocol"]} {r["setting"]} 第 {r.get("attempt", 1)} 次' if r.get('setting') else r['protocol']


SETTING_ORDER = {'单轮': 0, '迭代': 1}


def ordered(results: list[dict], models: list[str] | None = None) -> list[dict]:
    """Model, then setting (single before iterative), then attempt."""
    rank = {m: i for i, m in enumerate(models or [])}
    return sorted(results, key=lambda r: (rank.get(r['model'], len(rank)), r['model'], SETTING_ORDER.get(r.get('setting'), 2),
                                          r.get('attempt', 1)))


def group_label(group: dict) -> str:
    return f'题目版本 {group.get("version", "?")} · 协议 {group.get("protocol", "?")}'


def process_of(task: dict, r: dict) -> dict | None:
    setting = {'单轮': 'single', '迭代': 'iterative'}.get(r.get('setting'), 'single')
    folder = r.get('folder') or f'{r["model"]}-{setting}-{r.get("attempt", 1)}'
    path = ROOT / 'tasks' / task['name'] / 'solutions' / (r.get('protocol') or 'v6.2') / folder / 'result.json'
    return json.loads(path.read_text()).get('process') if path.exists() else None


def curve_svg(process: dict | None) -> str:
    """Public-set score after each test of an iterative attempt: x = minutes from the start, y = 0-100, reference 50 dashed."""
    curve = (process or {}).get('public_curve')
    if not curve:
        return ''
    width, height, pad = 520, 150, 28
    span = max(process.get('minutes') or 0, max(c['minute'] for c in curve), 1)
    x = lambda m: pad + (width - 2 * pad) * m / span
    y = lambda s: height - pad - (height - 2 * pad) * max(0.0, min(100.0, s)) / 100
    ok = [c for c in curve if c['score'] is not None]
    line = ' '.join(f'{x(c["minute"]):.1f},{y(c["score"]):.1f}' for c in ok)
    dots = ''.join(f'<circle cx="{x(c["minute"]):.1f}" cy="{y(c["score"]):.1f}" r="3"><title>{c["minute"]:.0f} 分钟：{c["score"]:.1f}</title></circle>'
                   for c in ok)
    misses = ''.join(f'<text x="{x(c["minute"]):.1f}" y="{y(0) + 4:.1f}" text-anchor="middle" fill="#999">×<title>{c["minute"]:.0f} 分钟：没有报告</title></text>'
                     for c in curve if c['score'] is None)
    axes = (f'<line x1="{pad}" y1="{y(0)}" x2="{width - pad}" y2="{y(0)}" stroke="#bbb"/>'
            f'<line x1="{pad}" y1="{y(50)}" x2="{width - pad}" y2="{y(50)}" stroke="#bbb" stroke-dasharray="4 3"/>'
            f'<text x="{pad - 4}" y="{y(50) + 4}" text-anchor="end" font-size="10" fill="#888">50</text>'
            f'<text x="{pad - 4}" y="{y(100) + 4}" text-anchor="end" font-size="10" fill="#888">100</text>'
            f'<text x="{width - pad}" y="{height - 6}" text-anchor="end" font-size="10" fill="#888">{span:.0f} 分钟</text>')
    return (f'<figure class="curve"><svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">{axes}'
            f'<polyline points="{line}" fill="none" stroke="#3366cc" stroke-width="1.5"/>{dots}{misses}</svg>'
            f'<figcaption>public 分数随测试提交（{len(curve)} 次，虚线为参照 50 分）</figcaption></figure>')


def process_cell(task: dict, r: dict) -> str:
    """Test submissions (iterative) and development time used, from the attempt's result.json (build_task_pages.py)."""
    setting = {'单轮': 'single', '迭代': 'iterative'}.get(r.get('setting'), 'single')
    folder = r.get('folder') or f'{r["model"]}-{setting}-{r.get("attempt", 1)}'
    path = ROOT / 'tasks' / task['name'] / 'solutions' / (r.get('protocol') or 'v6.2') / folder / 'result.json'
    process = json.loads(path.read_text()).get('process') if path.exists() else None
    if not process:
        return ''
    return (f'{process["public_tests"]} 次提交，' if setting == 'iterative' else '') + f'{process["minutes"]:.0f} 分钟'


def result_rows(task: dict, group: dict) -> str:
    rows = []
    for r in ordered(group['results']):
        rows.append(f'<tr><td><a href="#{anchor_of(r)}">{html.escape(model_name(r["model"]))}</a></td>'
                    f'<td>{html.escape(protocol_label(r))}</td>'
                    f'<td>{md(r["raw"])[3:-4]}</td><td>{md(r["mapping"])[3:-4]}</td>'
                    f'<td class="num score {score_class(r["score"])}">{fmt_score(r["score"])}</td>'
                    f'<td>{html.escape(process_cell(task, r))}</td></tr>')
    return ('<table><tr><th>模型</th><th>协议</th><th>原始指标</th><th>换算</th><th class="num">得分（满分 100）</th>'
            '<th>提交次数与用时</th></tr>' + ''.join(rows) + '</table>')


ATTENTION = {'author': ('需出题人处理', 'stop'), 'review': ('建议人工复核', 'warn')}


def insight_rows(task: dict) -> str:
    rows = []
    for i in task.get('insights', []):
        label, tag = ATTENTION.get(i['attention'], ('', ''))
        mark = f'<span class="tag {tag}">{label}</span>' if label else ''
        rows.append(f'<tr><td>{html.escape(i["category"])}</td><td>{md(i["claim"])[3:-4]}<br><small>证据：{md(i["evidence"])[3:-4]}</small></td>'
                    f'<td>{html.escape(i["confidence"])}</td><td>{mark}</td></tr>')
    return '<table><tr><th>类别</th><th>结论</th><th>置信度</th><th>关注</th></tr>' + ''.join(rows) + '</table>'


def insights_page(tasks: list[dict]) -> str:
    """Attention items first, then every task's result reading (meta.json insights; results/INSIGHTS.md)."""
    items = []
    for level in ('author', 'review'):
        label, tag = ATTENTION[level]
        for task in tasks:
            for i in task.get('insights', []):
                if i['attention'] == level:
                    items.append(f'<li><span class="tag {tag}">{label}</span> <a href="tasks/{task["id"]}.html">{html.escape(task["name"])}</a>'
                                 f' · {html.escape(i["category"])}：{md(i["claim"])[3:-4]}</li>')
    body = ['<h1>结果解读</h1><p class="lead">已有成绩说明了什么：参照是否偏弱、题目或数据是否有缺陷、评分映射是否合适、模型是否解决不了、'
            '超过参照靠的是工程调参还是方法创新。</p>', '<h2>需要人类关注</h2>', '<ul>' + ''.join(items) + '</ul>', '<h2>各题</h2>']
    for task in tasks:
        if task.get('insights'):
            body += [f'<h3><a href="tasks/{task["id"]}.html">{html.escape(task["name"])}</a></h3>', insight_rows(task)]
    return page('结果解读', '\n'.join(body))


MONITOR_JS = r"""
const ROLE = {housekeeping: '杂务', dev: '开发', grade: '评测', other: '其他'};
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
async function refresh() {
  let d;
  try { d = await (await fetch('status.json?t=' + Date.now(), {cache: 'no-store'})).json(); }
  catch (e) { document.getElementById('stamp').textContent = '状态暂不可用（' + e + '）'; return; }
  const age = Math.round(Date.now() / 1000 - d.generated_unix);
  document.getElementById('stamp').textContent = '更新于 ' + d.generated + (age > 30 ? '（' + age + ' 秒前，采集可能已停止）' : '');
  const holds = d.holds.filter(h => h.blocks_dev_now);
  document.getElementById('holds').innerHTML =
    (holds.length ? '<span class="tag warn">新开发暂停中</span> ' + holds.map(h => esc(h.owner)).join('；') : '新开发：正常调度') +
    ' · 运行中的迭代尝试：' + d.iterative_attempts;
  const domains = {};
  for (const c of d.cores) (domains[c.l3] = domains[c.l3] || []).push(c);
  let used = {dev: 0, grade: 0}, total = {dev: 0, grade: 0};
  const html = Object.entries(domains).map(([l3, cores]) => '<div class="l3"><div class="l3h">L3 域 · CPU ' + esc(l3) + '</div>' +
    cores.map(c => {
      const l = c.lease;
      if (c.role in total) { total[c.role]++; if (l && l.kind !== 'held') used[c.role]++; }
      const state = !l ? 'idle' : (l.kind === 'held' ? 'held' : 'busy');
      return '<div class="core ' + c.role + ' ' + state + '"><div class="ch">' + esc(c.slot || ROLE[c.role]) +
        ' <small>核 ' + c.core + ' · CPU ' + c.cpus.join(',') + '</small></div>' +
        '<div class="cb">' + (l ? esc(l.label) + (l.minutes != null ? ' <small>· ' + l.minutes + ' 分钟</small>' : '') :
                              (c.role === 'housekeeping' ? '系统与守护进程' : '空闲')) + '</div></div>';
    }).join('') + '</div>').join('');
  document.getElementById('grid').innerHTML = html;
  const KIND = {slow_slot: '评测槽变慢（负载小测）', load_gate: '负载系数超闸门'};
  const alerts = d.alerts || [];
  const open = alerts.filter(a => !a.handled).length;
  document.getElementById('alerts').innerHTML = alerts.length ?
    '<h2>异常报警（最近 ' + alerts.length + ' 条，' + (open ? open + ' 条待处理' : '均已核查') + '）</h2><ul>' +
      alerts.map(a => '<li class="' + (a.handled ? 'handled' : 'alert') + '">' + esc(a.time) + ' · ' +
      esc(KIND[a.kind] || a.kind) + ' · ' + esc(a.task || '') + ' · ' + esc(a.slot || '') + ' · ' + esc(a.job || '') +
      (a.before != null ? ' · 前 ' + a.before + ' / 后 ' + a.after : '') + (a.load_factor != null ? ' · 负载 ' + a.load_factor : '') +
      (a.handled ? ' · 已核查：' + esc(a.handled) : '') + '</li>').join('') + '</ul>' : '<p>异常报警：无</p>';
  document.getElementById('summary').textContent = '开发槽 ' + used.dev + ' / ' + total.dev + ' 占用 · 评测槽 ' + used.grade + ' / ' + total.grade + ' 占用';
}
refresh(); setInterval(refresh, 5000);
"""

MONITOR_CSS = """
.l3grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:12px 0}
.l3{border:1px solid #ccc;border-radius:6px;padding:8px;background:#fafafa}
.l3h{font-weight:600;margin-bottom:6px}
.core{border-radius:4px;padding:6px 8px;margin:6px 0;border-left:6px solid #999;background:#fff}
.core .ch{font-weight:600}.core .cb{font-size:13px;margin-top:2px}
.core.housekeeping{border-left-color:#999;background:#f0f0f0}
.core.dev{border-left-color:#3b82f6}.core.dev.busy{background:#dbeafe}
.core.grade{border-left-color:#16a34a}.core.grade.busy{background:#dcfce7}
.core.held{background:#fef3c7;border-left-color:#d97706}
.alert{color:#b91c1c}
.handled{color:#6b7280}
"""


def monitor_page() -> str:
    body = ('<h1>实时占用</h1><p class="lead">16 个物理核：2 个杂务核、7 个开发槽、7 个评测槽，按共享 L3 缓存的域分组；每 5 秒刷新。'
            '每个槽占一个物理核，第二个硬件线程留给该任务在宿主上的配套进程。</p>'
            f'<style>{MONITOR_CSS}</style><p id="stamp"></p><p id="holds"></p><p id="summary"></p>'
            '<div id="grid" class="l3grid"></div><div id="alerts"></div>'
            '<p><small>颜色：灰＝杂务核；蓝＝开发槽（深色为占用）；绿＝评测槽（深色为占用）；黄＝为隔离而空着。'
            '默认全并发；每次评分前后在该核的兄弟线程上做约 0.7 秒的内存拷贝小测，比该槽历史最佳慢 25% 以上即报警。</small></p>'
            f'<script>{MONITOR_JS}</script>')
    return page('实时占用', body)


def task_page(task: dict) -> str:
    status = '<span class="tag stop">评测暂停</span>' if task.get('status') == 'paused' else ''
    body = [f'<h1>{html.escape(task.get("name", task["title"]))} {status}</h1><p class="lead">{html.escape(task["category"])} · '
            f'{md(task["summary"])[3:-4]}</p>',
            '<h2>题目背景</h2>', md(task['background_md']),
            f'<details><summary>公开题面原文（给模型的版本）</summary><div class="md">{md(statement(task["statement"]))}</div></details>',
            '<h2>评测依据</h2>', md(task['judge_md']),
            '<h2>Baseline：做法与本机实测结果</h2>', md(task['baseline_md']),
            *(['<h2>结果解读</h2>', insight_rows(task)] if task.get('insights') else []),
            '<h2>模型成绩</h2>']
    for n, group in enumerate(task['result_groups']):
        part = [f'<h3>{html.escape(group_label(group))}</h3><p class="lead">{html.escape(group["title"])}</p>', result_rows(task, group),
                breakdown_html(group.get('breakdown'), group['results'])]
        body += part if n == 0 else [f'<details><summary>旧版本：{html.escape(group_label(group))}</summary>', *part, '</details>']
    body.append('<h2>各模型的做法</h2>')
    for n, group in enumerate(task['result_groups']):
        if n == 1:
            body.append('<details><summary>旧版本的做法</summary>')
        for r in ordered(group['results']):
            body.append(f'<h3 id="{anchor_of(r)}">{html.escape(model_name(r["model"]))}（{html.escape(protocol_label(r))}）'
                        f' <span class="score {score_class(r["score"])}">{fmt_score(r["score"])}</span></h3>')
            if r.get('notes_md'):
                body.append(f'<div class="note">{md(r["notes_md"])}</div>')
            if r.get('setting') == '迭代':
                body.append(curve_svg(process_of(task, r)))
            text = readme(r['store'], r['submission']) if r.get('submission') else None
            if text:
                body.append(f'<details><summary>模型自己写的说明（提交中的 README.md）</summary><div class="md">{md(text)}</div></details>')
            elif not r.get('notes_md'):
                body.append('<p>没有可展示的提交。</p>')
    if len(task['result_groups']) > 1:
        body.append('</details>')
    if task.get('notes_md'):
        body += ['<h2>观察与说明</h2>', md(task['notes_md'])]
    return page(task['title'], '\n'.join(body), depth=1)


BOARD_MODELS = ['gpt-6.1-sol', 'claude-opus-5-5', 'qwen3.8-max-0902', 'hy4', 'intern-s2']


def index_page(site: dict, tasks: list[dict]) -> str:
    def models_of(groups) -> list[str]:
        found = []
        for group in groups:
            for r in group['results']:
                if r['model'] not in found:
                    found.append(r['model'])
        return found

    newest_groups = [t['result_groups'][0] for t in tasks if t['result_groups']]
    older_groups = [g for t in tasks for g in t['result_groups'][1:]]
    # The main board always lists these models (empty cells where they have no result), then any other with results,
    # ordered by their iterative average.
    models = rank_models(newest_groups, [*BOARD_MODELS, *(m for m in models_of(newest_groups) if m not in BOARD_MODELS)])
    head = ''.join(f'<th>{html.escape(model_name(m))}</th>' for m in models)

    def cells_for(task: dict, group: dict, models: list[str]) -> str:
        cells = []
        for m in models:
            hit = ordered([r for r in group['results'] if r['model'] == m])
            lines = []
            for setting in ('单轮', '迭代'):
                same = [r for r in hit if r.get('setting', '单轮') == setting]
                if same:
                    scores = ' / '.join(f'<span class="score {score_class(r["score"])}">{fmt_score(r["score"])}</span>'
                                        for r in same)
                    count = f'（{len(same)} 次）' if len(same) > 1 else ''
                    process = ' / '.join(filter(None, (process_cell(task, r) for r in same)))
                    lines.append(f'<small>{setting}{count}</small> {scores}' + (f'<br><small>{process}</small>' if process else ''))
            cells.append('<td>' + '<br>'.join(lines) + '</td>')
        return ''.join(cells)

    old_models = models_of(older_groups)
    old_head = ''.join(f'<th>{html.escape(model_name(m))}</th>' for m in old_models)
    rows, history = [], []
    for task in tasks:
        if not task['result_groups']:
            continue
        newest, *older = task['result_groups']
        name = html.escape(task.get('name', task['title']))
        rows.append(f'<tr><td><a href="tasks/{task["id"]}.html">{name}</a><br><small>{html.escape(group_label(newest))}</small></td>'
                    f'{cells_for(task, newest, models)}</tr>')
        history += [f'<tr><td>{name}<br><small>{html.escape(group_label(g))}</small></td>{cells_for(task, g, old_models)}</tr>'
                    for g in older]
    means = averages(newest_groups, models)
    rows.append('<tr class="avg"><td><b>平均</b></td>' + ''.join(
        '<td>' + '<br>'.join(f'<small>{s}</small> <b>{"—" if means[(m, s)][0] is None else f"{means[(m, s)][0]:.1f}"}</b>'
                             for s in SETTINGS) + '</td>' for m in models) + '</tr>')
    svgs = chart_svgs(newest_groups, models, {m: model_name(m) for m in models})
    charts = ''.join(f'<img src="assets/{CHART_FILES[s]}?v={hashlib.sha256(svgs[CHART_FILES[s]].encode()).hexdigest()[:10]}" '
                     f'alt="{s}平均分">' for s in SETTINGS)
    cards = ''.join(
        f'<div class="card"><h3><a href="tasks/{t["id"]}.html">{html.escape(t.get("name", t["title"]))}</a></h3>'
        f'<p>{html.escape(t["category"])}</p><p>{md(t["summary"])[3:-4]}</p>'
        + ('<p><span class="tag stop">评测暂停</span></p>' if t.get('status') == 'paused' else '') + '</div>' for t in tasks)
    body = (f'<h1>{html.escape(site["title"])}</h1><p class="lead">{html.escape(site["subtitle"])}</p>{md(site["intro_md"])}'
            f'<h2>题目</h2><div class="cards">{cards}</div>'
            '<h2>总榜（满分 100，本机参照实现 = 50）</h2>'
            f'<div class="charts">{charts}</div>'
            '<p>每道题只列最新版本（题目版本 · 协议）的成绩；同一格多次尝试按次数依次列出。空格表示该模型没有在这道题上尝试；“—”表示尚未评测。'
            '列按迭代平均分排序；最后一行为各题分数的简单平均（没有成绩的题不计）。旧版本的成绩在表格下方折叠。</p>'
            f'<table class="matrix"><tr><th>题目 / 版本</th>{head}</tr>{"".join(rows)}</table>'
            + (f'<details><summary>旧版本的成绩（{len(history)} 组）</summary>'
               f'<table class="matrix"><tr><th>题目 / 版本</th>{old_head}</tr>{"".join(history)}</table></details>' if history else ''))
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
    newest = [t['result_groups'][0] for t in tasks if t['result_groups']]
    for file, svg in chart_svgs(newest, BOARD_MODELS, {m: model_name(m) for m in BOARD_MODELS}).items():
        (staging / 'assets' / file).write_text(svg)
    (staging / 'index.html').write_text(index_page(site, tasks))
    (staging / 'protocol.html').write_text(page('评测协议', md(site['protocol_md'])))
    (staging / 'insights.html').write_text(insights_page(tasks))
    (staging / 'monitor.html').write_text(monitor_page())
    (staging / 'process.html').write_text(page('评测流程', md((ROOT / 'docs/BENCH_PROCESS.md').read_text())))
    for task in tasks:
        (staging / 'tasks' / f'{task["id"]}.html').write_text(task_page(task))
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.rename(args.out)
    print(json.dumps({'out': str(args.out), 'tasks': len(tasks), 'commit': commit}))


if __name__ == '__main__':
    main()
