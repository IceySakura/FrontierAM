"""Build problems/<task>/: one reading entry per current task for task authors.

Each task directory gets README.md (what the task is, where the statement, data, grader and reference live, the latest
results with their review status, and the models' solutions), statement.md (the public statement exactly as given to
the models), public/ (text files of the public development package; data files are listed, not copied), reference/
(reference implementation sources when they are small and not already in the repository) and solutions/<model>-<setting>/
(the frozen source of every graded v6.2 submission, binaries omitted). Content comes from dashboard/tasks/*.json (the
same text as the dashboard), the sealed packages and the stores; rerun after results change.

Usage: build_problem_pages.py
"""

import io
import json
import shutil
import tarfile

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'problems'
TEXT = {'.md', '.py', '.txt', '.json', '.sh', '.c', '.cc', '.cpp', '.h', '.hpp', '.R', '.toml', '.yaml', '.yml', '.jl',
        '.m', '.cfg', ''}
MODEL = {'gpt-6.1-sol': 'GPT-6.1 Sol', 'claude-opus-5-5': 'Claude Opus 5.5'}

# Where each task's pieces live in this repository (or on this server, for git-ignored sealed copies).
LOCATIONS = {
    'conic': {
        'package': 'tasks/conic/solver-20261105/（出题方 2026-11-05 修正版题包：题面 instruction.md、环境、评测器）',
        'public': 'tasks/conic/solver-20261105/environment/public-dev/（公开开发集 163 例及逐例时限）',
        'private': 'tasks/conic/solver-20261105/tests/private/ 与 tasks/conic/original/tests/private/（650 例私有集，与原版相同）',
        'grader': 'tasks/conic/solver-20261105/tests/（题包自带评测器；本机按同一规则用本机参考耗时重标定时限）',
        'reference': 'tasks/conic/reference_submission/（Clarabel 0.11.1 封装）',
        'docs': ['results/conic/CONIC_FULL.md', 'results/conic/CONIC_PER_CASE.md', 'results/conic/CONIC_REVISION_SOL.md',
                 'results/conic/CONIC_FAST_GRADING.md'],
        'history': '原版题包（协议 v6，2026-11-04/05）与修正版题包 v6.1 的成绩见 `results/LEADERBOARD.md`。',
    },
    'exponential-v2': {
        'package': 'tasks/exponential/sota_v1/（设计 DESIGN.md、数据生成 dataset_v2/、参照组 panel/、标定 calibration_v2/）',
        'public': '.runtime-exp2/public.tar（本服务器，git 忽略）：task/ 下的接口说明与公开用例；用例清单见 tasks/exponential/sota_v1/dataset_v2/cases/public*/',
        'private': 'tasks/exponential/sota_v1/dataset_v2/cases/private/（清单）；矩阵与参考解封存在 .runtime-exp2/（本服务器，git 忽略）',
        'grader': 'arena/exponential_v2.py（流式接口评测、精度检查、计时与 virtual best 计分）',
        'reference': 'tasks/exponential/sota_v1/panel/（7 个已发表方法：Expokit、KIOPS、expmv、SciPy、Chebyshev、SLEPc、ExponentialUtilities.jl）',
        'docs': ['results/exponential/EXPONENTIAL_V2.md'],
        'history': 'v1（解释器中的单个 KIOPS 参考，已取代）与 v6.1 成绩见 `results/LEADERBOARD.md`。',
    },
    'galaxy-gravity-v2': {
        'package': 'tasks/operator/galaxy-gravity-v2/（FrontierAM 重建说明）；封存副本 .runtime-ops/galaxy-gravity-v2/（本服务器，git 忽略）',
        'public': 'public/（本目录，文本文件）；10 张公开快照在 .runtime-ops/galaxy-gravity-v2/packages/public/public-dev/cases/',
        'private': '20 张私有快照（我方以私有种子用原包生成器生成，参考解用原包直接求和程序），封存在 .runtime-ops/galaxy-gravity-v2/private/',
        'grader': '原包评测器 tasks/galaxy-gravity/harbor/tests/grade.py（只改输入大小检查与架构检查）',
        'reference': 'reference/（两条 Treecode2 参照，原包 baselines/）',
        'docs': ['results/galaxy-gravity/GALAXY_SCALE.md', 'results/BASELINE_REVIEW.md'],
        'history': '原版 65,536 粒子（v6.1，sol 用直接求和得满分）见 `results/NEW_TASKS.md`。',
    },
    'polarization-demosaicking': {
        'package': 'var/raw/ 下的出题方维护包与公开包（git 忽略）；封存副本 .runtime-ops/polarization-demosaicking/',
        'public': 'public/（本目录）；公开反馈集（Wen 2019，105 景）封存在 .runtime-ops/polarization-demosaicking-feedback/',
        'private': '32 景测试集，封存在维护包内（.runtime-ops/polarization-demosaicking/）',
        'grader': '维护包 harbor/polarization-demosaicking/grade_submission.py（只改 CPU 检查；本机评分前预热镜像文件缓存）',
        'reference': 'reference/（逐方向双线性；镜像 LEIC）',
        'docs': ['results/polarization/POLARIZATION_PUBLIC_DEV_REQUEST.md', 'results/NEW_TASKS.md'],
        'history': 'v6.1 sol 成绩见 `results/NEW_TASKS.md`。',
    },
    'species-tree-v2': {
        'package': 'tasks/operator/species-tree-v2/（题面与重建说明）；模拟流程 tasks/operator/species-tree/sim/；封存副本 .runtime-ops/species-tree-v2/',
        'public': 'public/（本目录，文本文件）；50 个公开用例在 .runtime-ops/species-tree-v2/packages/public/public-dev/cases/',
        'private': '200 个私有用例（我方模拟），封存在 .runtime-ops/species-tree-v2/private/',
        'grader': '原 species-tree 1.0.1 包的批量评测器与沙箱（物种数改为 250、每例时限 180 秒）',
        'reference': 'TREE-QMC 3.0.4 不加权 / 加权（原包 operator/references/，按需物化到 .runtime-ops/species-tree-v2/references/）',
        'docs': ['results/species-tree/SPECIES_TREE_SIM.md', 'results/species-tree/SPECIES_TREE_SCALE.md'],
        'history': '原版（101 物种，出题人告知数据有误，暂停）见 `results/NEW_TASKS.md`。',
    },
    'trajectory-covariance': {
        'package': '出题方包 mcmc-long-run-covariance-maintainer-v1（/srv/frontieram-raw/，git 忽略）；封存副本 .runtime-ops/trajectory-covariance/',
        'public': 'public/（本目录）',
        'private': '3 万个私有输入（本机评测固定的 3,000 个分层子集），在出题方包 results/mcmc-long-run-covariance/formal-v1/private/',
        'grader': '出题方包的验证器（arena/operator_drivers/trajectory_batch.py 直接启动其验证容器）',
        'reference': 'reference/（OBM：R 包 mcmcse；VAR(1)）；人类候选参照见 tasks/operator/trajectory-covariance/reference_candidates/',
        'docs': ['results/NEW_TASKS.md', 'results/BASELINE_REVIEW.md'],
        'history': 'v6.1 sol 成绩见 `results/NEW_TASKS.md`。',
    },
}
REFERENCE_SOURCES = {
    'galaxy-gravity-v2': ['.runtime-ops/galaxy-gravity-v2/packages/operator/operator/baselines'],
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
    """Copy the public package's small text files; list the data files instead of copying them."""
    if spec['kind'] != 'tar':
        return []
    listed = []
    with tarfile.open(ROOT / spec['path']) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            data = archive.extractfile(member).read(200_001)
            if Path(name).suffix in TEXT and member.size <= 200_000 and not is_binary(data):
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            else:
                listed.append(f'{name}（{member.size:,} 字节）')
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
    omitted = []
    if not archive.exists():
        return ['（提交存档不在本服务器）']
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or '__pycache__' in member.name:
                continue
            data = tar.extractfile(member).read()
            if is_binary(data) or member.size > 2_000_000:
                omitted.append(f'{member.name}（{member.size:,} 字节）')
                continue
            path = target / member.name.lstrip('./')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return omitted


def latest(task: dict) -> list[dict]:
    rows = []
    for group in task['result_groups']:
        if group['title'].startswith('协议 v6.2'):
            rows += group['results']
    return rows


def page(task: dict, where: dict, listed: list[str], solutions: list[tuple[dict, str, list[str]]], refs: bool) -> str:
    lines = [f'# {task["title"]}', '', f'{task["summary"]}', '',
             '看板页面：http://9.135.91.91:18765/tasks/' + task['id'] + '.html（办公网）', '',
             '## 怎么读这道题', '',
             '1. **题目背景**：下面第一节。',
             '2. **题面原文**：[`statement.md`](statement.md)，与给模型的版本逐字相同。',
             '3. **数据**：见“数据与代码在哪里”；公开开发材料的文本文件在 [`public/`](public/)。',
             '4. **评测与计分**：“评测依据”一节；评测器位置见下表。',
             '5. **Baseline**：“Baseline”一节' + ('；参照源码在 [`reference/`](reference/)' if refs else '') + '。',
             '6. **模型解法**：“模型成绩与解法”一节；每份提交的完整源码在 [`solutions/`](solutions/)。', '',
             '## 题目背景', '', task['background_md'], '',
             '## 数据与代码在哪里', '', '| 内容 | 位置 |', '| --- | --- |',
             f'| 题包 | {where["package"]} |', f'| 公开开发材料 | {where["public"]} |',
             f'| 私有测试集 | {where["private"]} |', f'| 评测器 | {where["grader"]} |',
             f'| 参照实现 | {where["reference"]} |',
             '| 详细文档 | ' + '、'.join(f'`{d}`' for d in where['docs']) + ' |', '']
    if listed:
        lines += ['公开包中的数据文件（未复制到本目录）：' + '；'.join(listed[:12]) + ('；……' if len(listed) > 12 else '') + '。', '']
    lines += ['## 评测依据', '', task['judge_md'], '', '## Baseline', '', task['baseline_md'], '',
              '## 模型成绩与解法（协议 v6.2，最新）', '',
              '| 模型 | 设置 | 原始指标 | 换算 | 得分（满分 100，参照 50） | 复核 | 源码 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r, folder, _ in solutions:
        score = '—' if r.get('score') is None else f'{r["score"]:.1f}'
        review = r.get('review', '待复核') if (r.get('score') or 0) > 50 else ''
        src = f'[`solutions/{folder}/`](solutions/{folder}/)' if folder else ''
        lines.append(f'| {MODEL.get(r["model"], r["model"])} | {r.get("setting", "")} | {r["raw"]} | {r["mapping"]} | {score} | {review} | {src} |')
    lines += ['', '超过参照（50 分）的成绩在复核结论出来之前标“待复核”：复核会解释超出的原因、寻找更强的人类方法并做消融，见 `results/BASELINE_REVIEW.md` 与 `results/BASELINE_POLICY.md`。', '']
    for r, folder, omitted in solutions:
        lines += [f'### {MODEL.get(r["model"], r["model"])}（{r.get("setting", "")}）', '']
        if r.get('notes_md'):
            lines += [r['notes_md'], '']
        if folder:
            note = f'源码：[`solutions/{folder}/`](solutions/{folder}/)（提交 `{r["submission"]}`）'
            if omitted:
                note += '；未复制的二进制或大文件：' + '、'.join(omitted)
            lines += [note + '。', '']
    if task.get('notes_md'):
        lines += ['## 观察与说明', '', task['notes_md'], '']
    lines += ['## 历史版本', '', where['history'], '']
    return '\n'.join(lines)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    index = []
    for path in sorted((ROOT / 'dashboard/tasks').glob('*.json'), key=lambda p: json.loads(p.read_text())['order']):
        task = json.loads(path.read_text())
        where = LOCATIONS[task['id']]
        target = OUT / task['id']
        target.mkdir()
        (target / 'statement.md').write_text(statement(task['statement']))
        listed = public_texts(task['statement'], target / 'public')
        refs = False
        for source in REFERENCE_SOURCES.get(task['id'], []):
            source = Path(source) if source.startswith('/') else ROOT / source
            if source.is_dir():
                copy_tree(source, target / 'reference' / source.name)
                refs = True
        solutions = []
        seen = {}
        for r in latest(task):
            if not r.get('submission'):
                solutions.append((r, '', []))
                continue
            base = f'{r["model"]}-{"single" if r.get("setting") == "单轮" else "iterative"}'
            seen[base] = seen.get(base, 0) + 1
            folder = base if seen[base] == 1 else f'{base}-{seen[base]}'
            omitted = extract_solution(r['store'], r['submission'], target / 'solutions' / folder)
            solutions.append((r, folder, omitted))
        (target / 'README.md').write_text(page(task, where, listed, solutions, refs))
        best = [r for r in latest(task) if r.get('score') is not None]
        index.append((task, len(solutions)))
    lines = ['# 题目目录', '', '每道题一个目录，内容结构相同：', '',
             '- `README.md`：怎么读这道题，题目背景、数据与代码位置、评测依据、baseline、模型成绩与解法；',
             '- `statement.md`：给模型的题面原文；', '- `public/`：公开开发材料的文本文件；',
             '- `reference/`：参照实现源码（较小且不在仓库其他位置时）；', '- `solutions/<模型>-<设置>/`：模型冻结提交的源码。', '',
             '| 题目 | 一句话 | 已评提交 |', '| --- | --- | --- |']
    for task, n in index:
        lines.append(f'| [{task["title"]}]({task["id"]}/README.md) | {task["summary"]} | {n} |')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n\n本目录由 `scripts/build_problem_pages.py` 生成，内容与看板同源（`dashboard/tasks/*.json`）。\n')
    print(json.dumps({'tasks': [t['id'] for t, _ in index]}))


if __name__ == '__main__':
    main()
