"""ihvp-sequence-v3 replay and reference report for the author (tasks/ihvp-sequence/analysis/V3_RESULTS.md).

run:    holding new development back from the start, grade on dev_idle leases (no development running; grade slots in
        parallel, grade-to-grade interference negligible) the frozen v2 high-score programs, the two references on all 30
        public sequences and the v1 programs, with the v3 deployment (arena/operator_drivers/ihvp_sequence.py: single
        thread, 30 s per sequence, worker peak RSS 256 MiB); development stays held until gpt-6.1-sol's v3 cells are graded.
        Private references come from the v3 anchor measurement. Outputs: var/stores/store-v63/ihvp-v3-replay/.
report: per public/private split and per group (retained 15, added 15, all 30): mean loss, mean/max wall time,
        peak RSS and failures for every program; writes aggregates only (no case data) to V3_RESULTS.md.

Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... ihvp-v3-replay.py run|report
"""
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'var/stores/store-v63'
OUT = STORE / 'ihvp-v3-replay'
REPORT = ROOT / 'tasks/ihvp-sequence/analysis/V3_RESULTS.md'
PROGRAMS = [  # (label, submission, description); v2 first as the author asked
    ('v2-single', 'ihvp-sequence-v2-20261111T073210Z-66b579', 'gpt-6.1-sol v2 单轮（Galerkin 回收 CG，v2 私有集 99.7）'),
    ('v2-iterative', 'ihvp-sequence-v2-20261111T100313Z-e184d3', 'gpt-6.1-sol v2 迭代（16 解窗口投影 + CG，v2 私有集 99.9）'),
    ('v1-single', 'ihvp-sequence-20261111T051550Z-144fe2', 'gpt-6.1-sol v1 单轮（识别 DCT 结构后精确求解，v1 100）'),
    ('v1-iterative', 'ihvp-sequence-20261111T052259Z-ce34b8', 'gpt-6.1-sol v1 迭代（识别 DCT 结构，v1 100）'),
]
REFERENCES = [('cg', 'weak', '冷启动 CG（0 分锚点）'), ('lbfgs', 'strong', 'L-BFGS-PCG（50 分锚点）')]


def source_dir(submission: str) -> Path:
    target = OUT / 'sources' / submission
    if not target.exists():
        archive = next(STORE.glob(f'submissions/{submission}/source.tar'))
        target.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter='data')
    return target


def run() -> None:
    import arena.tasks  # noqa: F401
    from arena.cli import context, placement_for
    from arena.slots import Broker
    from arena.tasks import TASKS

    layout, _, _ = context()
    broker = Broker(layout, dev_drain_after=0)
    drain = broker._drain_request('ihvp v3 replay and grading')  # new development waits from now on
    task = TASKS['ihvp-sequence-v3']
    jobs = [(label, source_dir(sub), split) for label, sub, _ in PROGRAMS[:2] for split in ('private', 'public_all')]
    jobs += [(name, task.reference_dir(which), 'public_all') for name, which, _ in REFERENCES]
    jobs += [(label, source_dir(sub), split) for label, sub, _ in PROGRAMS[2:] for split in ('private', 'public_all')]
    print(json.dumps({'event': 'waiting_for_dev_idle', 'jobs': len(jobs)}), flush=True)
    def one(job):
        label, source, split = job
        out = OUT / f'{label}-{split}'
        if (out / 'summary.json').exists():
            return
        if out.exists():
            shutil.rmtree(out)
        with broker.acquire('grade', task.grade_memory, f'ihvp v3 replay {label}-{split}', isolation='dev_idle') as lease:
            result = task.run_grader(source, out, placement_for(lease, layout), split=split)
        print(json.dumps({'event': 'graded', 'job': f'{label}-{split}', 'slot': lease.id,
                          **{k: result.get(k) for k in ('status', 'raw_error', 'max_wall_s', 'max_peak_rss_mib')}}), flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:  # one grade slot stays free for gpt-6.1-sol's v3 gradings
        list(pool.map(one, jobs))
    log = ROOT / 'var/stores/store/reruns/v63-grade.log'
    while not all(f'"label": "{s}-ihvp-sequence-v3-1"' in log.read_text() for s in ('single', 'iterative')):
        time.sleep(30)
    drain[1].unlink(missing_ok=True)
    os.close(drain[0])
    print(json.dumps({'event': 'done', 'development': 'released'}), flush=True)


def records(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get('records', []) if path.exists() else []


def anchor_records() -> dict[str, list[dict]]:
    root = STORE / 'baselines/ihvp-sequence-v3'
    job = json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['job']
    return {name: records(STORE / 'results' / job / 'grade' / which / 'summary.json') for name, which, _ in REFERENCES}


def stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r['status'] == 'ok' and r.get('energy_loss') is not None]
    walls = [r['wall_s'] for r in rows if r.get('wall_s') is not None]
    rss = [r['peak_rss_mib'] for r in rows if r.get('peak_rss_mib') is not None]
    return {'n': len(rows), 'loss': sum(r['energy_loss'] for r in ok) / len(ok) if len(ok) == len(rows) and rows else None,
            'mean_wall': sum(walls) / len(walls) if walls else None, 'max_wall': max(walls) if walls else None,
            'peak_rss': max(rss) if rss else None, 'failures': len(rows) - len(ok)}


def report() -> None:
    anchors = anchor_records()
    runs = {(label, split): records(OUT / f'{label}-{split}' / 'summary.json')
            for label in [p[0] for p in PROGRAMS] + [r[0] for r in REFERENCES] for split in ('private', 'public_all')}
    for name in anchors:
        runs[(name, 'private')] = anchors[name]
    names = {**{r[0]: r[2] for r in REFERENCES}, **{p[0]: p[2] for p in PROGRAMS}}
    lines = ['# ihvp-sequence v3：参照重测与旧程序回放（自动生成）', '',
             '本机评测核，单线程、每条序列 30 秒、worker 峰值 RSS 256 MiB，整机安静（无开发、其他评测槽空闲）；v3 部署：评测器在任务镜像内以 root 运行，'
             'worker 为 UID 1000，只能读协议文件。private 的两份参照来自 v3 锚点标定（同一部署）；其余为本次回放。'
             'loss 为每条序列 50 步平均后再对序列平均（与原始指标 E 相同）；有失败的组不给平均 loss。不复用旧版锚点。', '']
    summary = {}
    for split, title in (('public_all', 'public（反馈池，30 条）'), ('private', 'private（正式集，30 条）')):
        lines += [f'## {title}', '', '| 程序 | 组 | 条数 | 平均 loss | 平均秒/序列 | 最大秒/序列 | 峰值 RSS / MiB | 失败 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for label, desc in names.items():
            rows = runs.get((label, split)) or []
            if not rows:
                lines.append(f'| {desc} | — | — | 未测 | | | | |')
                continue
            for group, title_g in (('legacy', '原有 15'), ('hard', '新增 15'), (None, '合并 30')):
                sel = [r for r in rows if group is None or r['case'].startswith(group + '_')]
                s = stats(sel)
                summary.setdefault(split, {}).setdefault(label, {})[group or 'all'] = s
                fmt = lambda v, f: '—' if v is None else format(v, f)
                lines.append(f'| {desc if group == "legacy" else ""} | {title_g} | {s["n"]} | {fmt(s["loss"], ".4g")} | '
                             f'{fmt(s["mean_wall"], ".1f")} | {fmt(s["max_wall"], ".1f")} | {fmt(s["peak_rss"], ".0f")} | {s["failures"]} |')
        lines.append('')
    REPORT.write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(REPORT)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
