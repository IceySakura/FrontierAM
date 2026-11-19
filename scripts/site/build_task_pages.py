"""Build the generated parts of every task directory, tasks/<family>/, from its meta.json.

meta.json holds the curated text (background, judging, baseline, notes), the published results and, per task, the
version list and where each part lives. From it this script writes:
  README.md                               reading entry (generated; edit meta.json, not this file)
  versions/<current>/statement.md         the public statement exactly as given to the models
  versions/<current>/public/              text files of the public development package (data files listed only)
  references/r1/<member>/                 reference sources copied from the sealed vendor package, when not already here
  solutions/<protocol>/<attempt>/         frozen source of each graded submission (binaries omitted) and result.json
and tasks/README.md, the index. The same meta.json files feed the dashboard (scripts/site/build_dashboard.py).

Usage: build_task_pages.py
"""

import glob
import json
import shutil
import tarfile

from pathlib import Path

from board_stats import CHART_FILES, SETTINGS, averages, chart_svgs, rank_models

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / 'tasks'
TEXT = {'.md', '.py', '.txt', '.json', '.sh', '.c', '.cc', '.cpp', '.h', '.hpp', '.R', '.toml', '.yaml', '.yml', '.jl',
        '.m', '.cfg', ''}
MODEL = {'gpt-6.1-sol': 'GPT-6.1 Sol', 'claude-opus-5-5': 'Claude Opus 5.5', 'qwen3.8-max-0902': 'Qwen3.8 Max', 'hy4': 'Hunyuan 4', 'intern-s2': 'Intern S2'}
SETTING = {'单轮': 'single', '迭代': 'iterative'}
REFERENCE_SOURCES = {
    'galaxy-gravity': ['.runtime-ops/galaxy-gravity-v2/packages/operator/operator/baselines'],
    'polarization-demosaicking': ['.runtime-ops/polarization-demosaicking/packages/operator/'
                                  'polarization-demosaicking-maintenance-v1.2/harbor/polarization-demosaicking/references'],
    'trajectory-covariance': ['/srv/frontieram-raw/mcmc-long-run-covariance-maintainer-v1/tasks/mcmc-long-run-covariance/'
                              'pilot/submissions/obm',
                              '/srv/frontieram-raw/mcmc-long-run-covariance-maintainer-v1/tasks/mcmc-long-run-covariance/'
                              'pilot/submissions/var1'],
}


def is_binary(data: bytes) -> bool:
    return data[:4] == b'\x7fELF' or b'\0' in data[:4096]


def statement(spec: dict) -> str:
    if spec['kind'] == 'file':
        return (ROOT / spec['path']).read_text()
    with tarfile.open(ROOT / spec['path']) as archive:
        return '\n\n---\n\n'.join(archive.extractfile(m).read().decode() for m in spec['members'])


def public_texts(spec: dict, target: Path) -> list[str]:
    if spec['kind'] != 'tar':
        return []
    listed = []
    with tarfile.open(ROOT / spec['path']) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            data = archive.extractfile(member).read(200_001)
            if Path(member.name).suffix in TEXT and member.size <= 200_000 and not is_binary(data):
                path = target / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            else:
                listed.append(f'{member.name}（{member.size:,} 字节）')
    if listed:
        (target / 'DATA_FILES.md').write_text('# 公开包中的数据文件（未复制到仓库）\n\n' + '\n'.join(f'- {x}' for x in listed) + '\n')
    return listed


def copy_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.stat().st_size <= 5_000_000:
            data = path.read_bytes()
            if is_binary(data) and not path.name.endswith('.tar.gz'):
                continue
            out = target / path.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)


def extract_solution(store: str, submission: str, target: Path) -> list[str]:
    archive = ROOT / store / 'submissions' / submission / 'source.tar'
    if not archive.exists():
        return ['（提交存档不在本服务器）']
    omitted = []
    source = target / 'source'
    if source.exists():
        shutil.rmtree(source)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or '__pycache__' in member.name:
                continue
            data = tar.extractfile(member).read()
            if is_binary(data) or member.size > 2_000_000:
                omitted.append(f'{member.name}（{member.size:,} 字节）')
                continue
            path = source / member.name.lstrip('./')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
    return omitted



def run_index() -> tuple[dict, dict]:
    """submission -> development run (from the run logs) and run -> its directory."""
    sub2run = {}
    for log in glob.glob(str(ROOT / 'var/stores/store/reruns/*.log')):
        for line in open(log, errors='replace'):
            if '"develop_finished"' in line:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('submission'):
                    sub2run[event['submission']] = event['run']
    return sub2run, {Path(p).name: Path(p) for p in glob.glob(str(ROOT / 'var/stores/*/dev-runs/*'))}


def process_stats(submission: str, index: tuple[dict, dict]) -> dict | None:
    """Only what the board shows: test submissions (iterative) and the development time actually used."""
    sub2run, rundirs = index
    run = rundirs.get(sub2run.get(submission, ''))
    if not run or not (run / 'develop_result.json').exists():
        return None
    r = json.loads((run / 'develop_result.json').read_text())
    tests = r.get('test_submissions', sum(e.get('kind') == 'test' for e in r.get('evaluations', [])))
    # Public-set score of every test in submission order (None: no report), with its minute from the start.
    curve = [{'minute': round(e['requested_at_seconds'] / 60, 1), 'score': e.get('score') if e.get('status') == 'ok' else None,
              'status': e.get('status')}
             for e in r.get('evaluations', []) if e.get('kind') == 'test' and e.get('requested_at_seconds') is not None]
    return {'minutes': round(r['development_seconds'] / 60, 1), 'public_tests': tests, **({'public_curve': curve} if curve else {})}


def curve_text(p: dict | None) -> str:
    """'public 分数随测试提交：51.0（3 分钟）→ …' for an iterative attempt."""
    curve = (p or {}).get('public_curve')
    if not curve:
        return ''
    steps = [f'{"—" if c["score"] is None else format(c["score"], ".1f")}（{c["minute"]:.0f} 分钟）' for c in curve]
    return 'public 分数随测试提交：' + ' → '.join(steps) + '（—：没有拿到报告）' if any(c['score'] is None for c in curve) \
        else 'public 分数随测试提交：' + ' → '.join(steps)


def process_text(p: dict | None, iterative: bool) -> str:
    if not p:
        return ''
    return (f'{p["public_tests"]} 次提交，' if iterative else '') + f'{p["minutes"]:.0f} 分钟'


def attempt_dir(r: dict) -> str:
    """solutions/<protocol>/<folder>; a later task version within one protocol keeps its own subfolder (r['folder'])."""
    return r.get('folder') or f'{r["model"]}-{SETTING.get(r.get("setting"), "single")}-{r.get("attempt", 1)}'


ATTENTION = {'author': '**需出题人处理**', 'review': '建议人工复核'}


def insight_table(meta: dict) -> list[str]:
    """The task's result reading (meta.json insights): one row per finding with confidence, attention and evidence."""
    lines = ['| 类别 | 结论 | 置信度 | 关注 |', '| --- | --- | --- | --- |']
    for i in meta.get('insights', []):
        lines.append(f'| {i["category"]} | {i["claim"]}（证据：{i["evidence"]}） | {i["confidence"]} | {ATTENTION.get(i["attention"], "")} |')
    return lines


def write_insights(metas: list[dict]) -> None:
    """results/INSIGHTS.md: what the results so far say about each task, attention items first."""
    path = ROOT / 'results' / 'INSIGHTS.md'
    text = path.read_text() if path.exists() else ''
    head = text.split('<!-- insights:start -->')[0] if '<!-- insights:start -->' in text else text
    lines = ['<!-- insights:start -->', '## 需要人类关注', '']
    for level in ('author', 'review'):
        for meta in metas:
            for i in meta.get('insights', []):
                if i['attention'] == level:
                    lines.append(f'- {ATTENTION[level]} · [{meta["name"]}](../tasks/{meta["name"]}/README.md) · {i["category"]}：{i["claim"]}')
    lines += ['', '## 各题', '']
    for meta in metas:
        if meta.get('insights'):
            lines += [f'### [{meta["name"]}](../tasks/{meta["name"]}/README.md)', '', *insight_table(meta), '']
    lines += ['<!-- insights:end -->', '', '本节由 `scripts/site/build_task_pages.py` 从各题 `meta.json` 的 `insights` 生成。']
    path.write_text(head + '\n'.join(lines) + '\n')


def readme(meta: dict, solutions: list[tuple[dict, str, list[str]]], listed: list[str]) -> str:
    loc, cur = meta['locations'], meta['current_version']
    lines = [f'# {meta["name"]}', '', meta['summary'], '',
             '## 目录结构', '',
             '| 目录 | 内容 |', '| --- | --- |',
             f'| `versions/` | 题目的各个版本；当前版本 `{cur}/`，题面 `versions/{cur}/statement.md` |',
             '| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |',
             '| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |',
             '| `analysis/` | 这道题的专题分析与复核记录 |',
             '| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |', '',
             '## 版本', '', '| 版本 | 评测系统中的名字 | 说明 |', '| --- | --- | --- |']
    lines += [f'| `{v["dir"]}` | `{v["registry"]}` | {v["note"]} |' for v in meta['versions']]
    lines += ['', '## 题目背景', '', meta['background_md'], '',
              '## 题面、数据、评测器与参照在哪里', '', '| 内容 | 位置 |', '| --- | --- |',
              f'| 题面 | {loc["statement"]} |', f'| 公开开发材料 | {loc["public"]} |', f'| 私有测试集 | {loc["private"]} |',
              f'| 评测器 | {loc["grader"]} |', f'| 参照实现 | {loc["references"]} |', '']
    if listed:
        lines += [f'公开包中的数据文件未复制到仓库，清单见 `versions/{cur}/public/DATA_FILES.md`。', '']
    if meta.get('v63_data'):
        d = meta['v63_data']
        lines += ['## 数据划分（协议 v6.3）', '', '| 层 | 内容 |', '| --- | --- |',
                  f'| samples（给模型） | {", ".join(d["samples"])} |', f'| public（不给，反馈用） | {d["public"]} |',
                  f'| 反馈分类 | {d["groups"] or "不分类，只报总计"} |', '']
    lines += ['## 评测依据', '', meta['judge_md'], '', '## Baseline', '', meta['baseline_md'], '']
    if meta.get('insights'):
        lines += ['## 结果解读', '', *insight_table(meta), '']
    folded = False
    for group in meta['result_groups']:
        protocol = group.get('protocol')
        rows = [(r, folder, omitted) for r, folder, omitted in solutions if r.get('_group') is group]
        if not rows:
            continue
        newest = group is meta['result_groups'][0]
        if not newest and not folded:
            lines += ['<details><summary>旧版本的成绩与解法</summary>', '']
            folded = True
        lines += [f'## 模型成绩与解法（题目版本 {group.get("version")} · 协议 {protocol}）', '', group['title'], '',
                  '| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |', '| --- | --- | --- | --- | --- | --- | --- |']
        for r, folder, _ in rows:
            score = '—' if r.get('score') is None else f'{r["score"]:.1f}'
            link = f'[`{folder}/`](solutions/{protocol}/{folder}/)' if folder else ''
            process = process_text(r.get('_process'), r.get('setting') == '迭代')
            lines.append(f'| {MODEL.get(r["model"], r["model"])} | {r.get("setting", "")} | {r["raw"]} | {r["mapping"]} | {score} | {process} | {link} |')
        lines.append('')
        for r, folder, omitted in rows:
            lines += [f'### {MODEL.get(r["model"], r["model"])}（{r.get("setting", "")}，{protocol}）', '']
            if r.get('notes_md'):
                lines += [r['notes_md'], '']
            if r.get('setting') == '迭代' and curve_text(r.get('_process')):
                lines += [curve_text(r.get('_process')), '']
            if folder:
                note = f'源码 [`solutions/{protocol}/{folder}/source/`](solutions/{protocol}/{folder}/source/)（提交 `{r["submission"]}`）'
                lines += [note + ('；未复制的二进制或大文件：' + '、'.join(omitted) if omitted else '') + '。', '']
    if folded:
        lines += ['</details>', '']
    if meta.get('notes_md'):
        lines += ['## 观察与说明', '', meta['notes_md'], '']
    lines += ['---', '', '本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。', '']
    return '\n'.join(lines)


COLUMNS = [('gpt-6.1-sol', '单轮'), ('gpt-6.1-sol', '迭代'), ('claude-opus-5-5', '单轮'), ('claude-opus-5-5', '迭代'),
           ('qwen3.8-max-0902', '单轮'), ('qwen3.8-max-0902', '迭代'), ('hy4', '单轮'), ('hy4', '迭代'),
           ('intern-s2', '单轮'), ('intern-s2', '迭代')]
PROTOCOLS = ('v6.3', 'v6.2', 'v6.1', 'v6')  # newest first; the board shows the newest one that has results


def write_board(metas: list[dict], names: list[str]) -> None:
    """Scoreboard of the newest protocol with results into README.md between the board markers."""
    protocol = next(p for p in PROTOCOLS if any(g.get('protocol') == p and g['results'] for m in metas for g in m['result_groups']))
    newest = [next((g for g in meta['result_groups'] if g.get('protocol') == protocol), None) for meta in metas]  # newest task version
    groups = [g for g in newest if g]
    # every model of this round always (cells stay empty without a result), ordered by the iterative average
    models = rank_models(groups, list(dict.fromkeys(m for m, _ in COLUMNS)))
    columns = [(m, s) for m in models for s in SETTINGS]
    head = '| task | ' + ' | '.join(f'{MODEL[m]} {s}' for m, s in columns) + ' |'
    rule = '| --- |' + ' --- |' * len(columns)
    scores, process = [head, rule], [head, rule]
    for group, name in zip(newest, names):
        results = group['results'] if group else []
        row_s, row_p = [], []
        for model, setting in columns:
            hits = sorted((r for r in results if r['model'] == model and r.get('setting', '单轮') == setting),
                          key=lambda r: r.get('attempt', 1))
            row_s.append(' / '.join('—' if r.get('score') is None else f'{r["score"]:.1f}'
                                    for r in hits))
            row_p.append(' / '.join(process_text(r.get('_process'), setting == '迭代') or '—' for r in hits))
        link = f'[{name}](tasks/{name}/README.md)'
        scores.append(f'| {link} | ' + ' | '.join(row_s) + ' |')
        process.append(f'| {link} | ' + ' | '.join(row_p) + ' |')
    means = averages(groups, models)
    scores.append('| ***平均*** | ' + ' | '.join('—' if means[(m, s)][0] is None else f'***{means[(m, s)][0]:.1f}***'
                                               for m, s in columns) + ' |')
    for file, svg in chart_svgs(groups, models, MODEL).items():
        (ROOT / 'dashboard/assets' / file).write_text(svg)
    charts = ' '.join(f'![{s}平均分](dashboard/assets/{CHART_FILES[s]})' for s in SETTINGS)
    board = (f'协议 {protocol}；更早协议与题目版本的成绩见各题 README。' + '\n\n' + charts + '\n\n'
             + '\n'.join(scores) + '\n\n满分 100，参照 = 50；列按迭代平均分排序；平均为各题分数的简单平均，没有成绩的题不计。\n\n**提交次数与用时**\n\n'
             + '\n'.join(process))
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start, end = '<!-- board:start -->', '<!-- board:end -->'
    i, j = text.index(start) + len(start), text.index(end)
    readme.write_text(text[:i] + '\n' + board + '\n' + text[j:])


def main():
    index = []
    runs = run_index()
    for meta_path in sorted(TASKS.glob('*/meta.json'), key=lambda p: json.loads(p.read_text())['order']):
        family = meta_path.parent
        meta = json.loads(meta_path.read_text())
        version = family / 'versions' / meta['current_version']
        (version / 'statement.md').write_text(statement(meta['statement']))
        listed = []
        if not (version / 'package').exists() and not (version / 'data').exists():
            public = version / 'public'
            listed = public_texts(meta['statement'], public)
        for source in REFERENCE_SOURCES.get(family.name, []):
            src = Path(source) if source.startswith('/') else ROOT / source
            target = family / 'references' / 'r1' / src.name
            if src.is_dir() and not target.exists():
                copy_tree(src, target)
        solutions = []
        for group in meta['result_groups']:
            protocol = group.get('protocol')
            if protocol is None:
                continue
            for r in group['results']:
                r['_group'] = group
                if not r.get('submission'):
                    solutions.append((r, '', []))
                    continue
                folder = attempt_dir(r)
                r.setdefault('protocol', protocol)
                target = family / 'solutions' / protocol / folder
                target.mkdir(parents=True, exist_ok=True)
                omitted = extract_solution(r['store'], r['submission'], target)
                result = {k: r.get(k) for k in ('model', 'protocol', 'setting', 'submission', 'raw', 'mapping', 'score', 'review')}
                previous = json.loads((target / 'result.json').read_text()).get('process') if (target / 'result.json').exists() else None
                result['process'] = process_stats(r['submission'], runs) or previous
                r['_process'] = result['process']
                (target / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
                solutions.append((r, folder, omitted))
        (family / 'README.md').write_text(readme(meta, solutions, listed))
        index.append((meta, family.name, len(solutions)))
    lines = ['# 题目', '', '每道题一个目录，结构相同：`README.md`（阅读入口）、`versions/`（各版本的题面、公开与私有数据、评测器、生成流程）、'
             '`references/`（参照实现，分版本）、`solutions/`（模型提交与成绩）、`analysis/`（分析与复核）、`meta.json`（数据源）。', '',
             '| task | 已评提交 |', '| --- | --- |']
    lines += [f'| [{name}]({name}/README.md) | {n} |' for m, name, n in index]
    (TASKS / 'README.md').write_text('\n'.join(lines) + '\n\n本文件由 `scripts/site/build_task_pages.py` 生成。\n')
    write_board([m for m, _, _ in index], [name for _, name, _ in index])
    write_insights([m for m, _, _ in index])
    print(json.dumps({'tasks': [name for _, name, _ in index]}))


if __name__ == '__main__':
    main()
