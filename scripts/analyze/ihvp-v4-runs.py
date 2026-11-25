"""ihvp-sequence-v4 measurements for the v4 report (draft task: no formal score is activated).

  anchors   cold-start CG and L-BFGS-PCG on public and on private (separate anchors)
  repeat    the same three programs again (repeatability: identical frozen sources, same splits)
  replay    the frozen gpt-6.1-sol v3 iterative submission on public only (known-method regression)
  final SUBMISSION LABEL
            one frozen candidate on private, after its attempt has ended (never during development)
  report    tasks/ihvp-sequence/analysis/V4_RESULTS.md: mean loss, mean/max sequence time, max worker RSS, valid
            sequences and the score mapped on this machine's anchors of the same split (CG 0, L-BFGS-PCG 50, zero 100)

Jobs run on grade leases with the deployment of arena/operator_drivers/ihvp_sequence.py; outputs under
var/stores/store-v63/ihvp-v4/. Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... ihvp-v4-runs.py anchors|replay|final|report
"""
import json
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STORE = ROOT / 'var/stores/store-v63'
OUT = STORE / 'ihvp-v4'
REPORT = ROOT / 'tasks/ihvp-sequence/analysis/V4_RESULTS.md'
V3_ITERATIVE = 'ihvp-sequence-v3-20261112T070724Z-5b8c9b'
NAMES = {'cg': '冷启动 CG（0 分锚点）', 'lbfgs': 'L-BFGS-PCG（50 分锚点）', 'v3-iterative': 'gpt-6.1-sol v3 迭代（冻结回放）'}


def source_dir(submission: str) -> Path:
    target = OUT / 'sources' / submission
    if not target.exists():
        target.mkdir(parents=True)
        with tarfile.open(STORE / 'submissions' / submission / 'source.tar') as tar:
            tar.extractall(target, filter='data')
    return target


def grade(jobs: list[tuple[str, Path, str]]) -> None:
    import arena.tasks  # noqa: F401
    from arena.cli import context, placement_for
    from arena.slots import Broker
    from arena.tasks import TASKS

    layout, _, _ = context()
    broker, task = Broker(layout), TASKS['ihvp-sequence-v4']

    def one(job):
        label, source, split = job
        out = OUT / f'{label}-{split}'
        if (out / 'summary.json').exists():
            return
        shutil.rmtree(out, ignore_errors=True)
        with broker.acquire('grade', task.grade_memory, f'ihvp v4 {label}-{split}', isolation='shared') as lease:
            result = task.run_grader(source, out, placement_for(lease, layout), split=split)
        print(json.dumps({'event': 'graded', 'job': f'{label}-{split}', 'slot': lease.id,
                          **{k: result.get(k) for k in ('status', 'raw_error', 'max_wall_s', 'max_peak_rss_mib')}}), flush=True)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(one, jobs))


def references():
    import arena.tasks  # noqa: F401
    from arena.tasks import TASKS
    task = TASKS['ihvp-sequence-v4']
    return {'cg': task.reference_dir('weak'), 'lbfgs': task.reference_dir('strong')}


def stats(path: Path) -> dict | None:
    if not (path / 'summary.json').exists():
        return None
    rows = json.loads((path / 'summary.json').read_text()).get('records', [])
    ok = [r for r in rows if r['status'] == 'ok' and r.get('energy_loss') is not None]
    walls = [r['wall_s'] for r in rows if r.get('wall_s') is not None]
    return {'n': len(rows), 'valid': len(ok), 'loss': sum(r['energy_loss'] for r in ok) / len(ok) if ok and len(ok) == len(rows) else None,
            'mean_wall': sum(walls) / len(walls) if walls else None, 'max_wall': max(walls) if walls else None,
            'rss': max((r['peak_rss_mib'] for r in rows if r.get('peak_rss_mib') is not None), default=None)}


def score(loss, e0, e50):
    if loss is None or e0 is None or e50 is None:
        return None
    if loss >= e0:
        return 0.0
    return 50 * (e0 - loss) / (e0 - e50) if loss > e50 else 50 + 50 * (e50 - loss) / e50


def attempts() -> list[dict]:
    out = []
    for tag, model in (('sol', 'gpt-6.1-sol'), ('opus', 'claude-opus-5-5')):
        for setting in ('single', 'iterative'):
            log = ROOT / f'var/stores/store/reruns/v63-{tag}-{setting}-ihvp-sequence-v4-1.log'
            rows = [json.loads(l) for l in log.read_text().splitlines() if 'develop_finished' in l] if log.exists() else []
            if rows:
                out.append({'label': f'{tag}-{setting}', 'model': model, 'setting': setting, **rows[-1]})
    return out


def sha(path: Path) -> str:
    import hashlib
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_diagnostics() -> list[str]:
    lines = ['## 新模型的 public 诊断（开发期间的测试反馈）', '',
             '迭代设置开发期间每次测试只返回 public 15 条的整池汇总；分数按反馈基准（同一 public 划分上另一次测量的两份参照）换算。'
             '单轮设置没有测试。', '', '| 尝试 | 用时 / 分钟 | 结束方式 | 测试次数 | 各次 public 分数 |', '| --- | --- | --- | --- | --- |']
    for a in attempts():
        result = json.loads((STORE / 'dev-runs' / a['run'] / 'develop_result.json').read_text())
        tests = [e for e in result.get('evaluations') or [] if e.get('kind') == 'test']
        scores = ' → '.join(f'{e["score"]:.1f}' if e.get('score') is not None else '—' for e in tests) or '—'
        lines.append(f'| {a["label"]} | {result["development_seconds"] / 60:.0f} | {result["agent_stop_reason"]} | {len(tests)} | {scores} |')
    return lines + ['']


def environment() -> list[str]:
    import platform
    lock = json.loads((ROOT / '.runtime-ops/ihvp-sequence-v4/lock.json').read_text())
    cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), '')
    raw = ROOT / 'var/raw/ihvp-v4'
    lines = ['## 环境、数据与提交', '',
             f'- 评测机：{cpu}，Linux {platform.release()}（x86_64）；每份评测绑定一个物理核，与其他开发、评测并发（共享模式）。',
             f'- 任务镜像 `{lock["image"]["tag"]}`：`{lock["image"]["id"]}`；开发镜像 `frontieram-ihvp-sequence-v4-harness:v6`。',
             '- 代码：PR #3，提交 `50c8e12524e42256957239654ff250b8aa26db04`（按差异移植到维护端仓库）。',
             f'- 总包 `ihvp-hard-v4-with-guide.zip`：SHA-256 `{sha(raw / "ihvp-hard-v4-with-guide.zip")}`；'
             f'模型小包 `ihvp-v4-model-visible.zip`：`{sha(raw / "ihvp-v4-model-visible.zip")}`。',
             f'- 数据清单 SHA-256：public `{lock["data_manifest_sha256"]["public"]}`，private `{lock["data_manifest_sha256"]["private"]}`；'
             f'模型工作区 `public.tar` `{lock["public_tar_sha256"]}`（恰为六文件）。', '',
             '| 尝试 | 冻结提交 | 源码归档 SHA-256 |', '| --- | --- | --- |']
    for a in attempts():
        lines.append(f'| {a["label"]} | `{a["submission"]}` | `{sha(STORE / "submissions" / a["submission"] / "source.tar")}` |')
    lines.append(f'| v3-iterative（回放） | `{V3_ITERATIVE}` | `{sha(STORE / "submissions" / V3_ITERATIVE / "source.tar")}` |')
    return lines + ['']


def report() -> None:
    fmt = lambda v, f: '—' if v is None else format(v, f)
    lines = ['# ihvp-sequence v4：本机测量（自动生成，草稿版本，不是正式成绩）', '',
             f'生成时间 {datetime.now(timezone.utc).isoformat(timespec="seconds")}。每个分数都用同一划分上本机重测的两份参照作锚点'
             '（\\(E_0\\) = 冷启动 CG，\\(E_{50}\\) = L-BFGS-PCG）：\\(E \\ge E_0\\) 为 0；\\(E_{50} \\le E < E_0\\) 为 '
             '\\(50\\,(E_0-E)/(E_0-E_{50})\\)；\\(E < E_{50}\\) 为 \\(50 + 50\\,(E_{50}-E)/E_{50}\\)。重复测量列在“第二次”行。', '']
    finals = sorted(p.name[len('final-'):-len('-private')] for p in OUT.glob('final-*-private'))
    for split, title in (('public', 'public（15 条，诊断）'), ('private', 'private（30 条，最终）')):
        anchors = {k: (stats(OUT / f'{k}-{split}') or {}).get('loss') for k in ('cg', 'lbfgs')}
        lines += [f'## {title}', '', '| 程序 | 平均 loss | 平均秒/序列 | 最长秒/序列 | 最高 worker RSS / MiB | 合法序列 | 本机换算得分 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        rows = [('cg', 'cg'), ('cg-r2', 'cg'), ('lbfgs', 'lbfgs'), ('lbfgs-r2', 'lbfgs')]
        rows += [('v3-iterative', 'v3-iterative'), ('v3-iterative-r2', 'v3-iterative')] if split == 'public' else []
        rows += [(f'final-{f}', f) for f in finals] if split == 'private' else []
        for label, name in rows:
            s = stats(OUT / f'{label}-{split}')
            if s is None:
                continue
            desc = NAMES.get(name, name) + ('（第二次）' if label.endswith('-r2') else '')
            lines.append(f'| {desc} | {fmt(s["loss"], ".5g")} | {fmt(s["mean_wall"], ".1f")} | {fmt(s["max_wall"], ".1f")} | '
                         f'{fmt(s["rss"], ".0f")} | {s["valid"]}/{s["n"]} | {fmt(score(s["loss"], anchors["cg"], anchors["lbfgs"]), ".1f")} |')
        lines.append('')
    lines += public_diagnostics() + environment()
    REPORT.write_text('\n'.join(lines) + '\n')
    print(REPORT)


MODEL_OF = {'sol': 'gpt-6.1-sol', 'opus': 'claude-opus-5-5', 'qwen': 'qwen3.8-max-0902', 'hy4': 'hy4', 'intern': 'intern-s2',
            'sol62': 'gpt-6.2-sol', 'astra': 'gpt-6-astra', 'qwen4': 'qwen4-max', 'minimax': 'minimax-m3.1-flash'}


def scores() -> None:
    """score.json for every finished private final, all on the same private anchors (cg / lbfgs, first measurement)."""
    e0, e50 = (stats(OUT / 'cg-private') or {}).get('loss'), (stats(OUT / 'lbfgs-private') or {}).get('loss')
    for path in sorted(OUT.glob('final-*-private')):
        tag, setting, *attempt = path.name[len('final-'):-len('-private')].split('-')   # label tag-setting[-n]
        s = stats(path)
        if s is None or tag not in MODEL_OF:
            continue
        rows = json.loads((path / 'summary.json').read_text()).get('records', [])
        failed = s['valid'] < s['n']
        candidate = {'status': 'submission_failure' if failed else 'ok', 'raw_error': s['loss'], 'max_wall_s': s['max_wall'],
                     'max_peak_rss_mib': s['rss'], 'valid': s['valid'], 'cases': s['n'],
                     'reason': next((r.get('reason') for r in rows if r.get('status') != 'ok'), None)}
        out = ROOT / f'tasks/ihvp-sequence/solutions/v6.3/v4/{MODEL_OF[tag]}-{setting}-{attempt[0] if attempt else 1}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'score.json').write_text(json.dumps({
            'status': candidate['status'], 'score': 0.0 if failed else score(s['loss'], e0, e50), 'candidate': candidate,
            'anchors': {'weak': e0, 'strong': e50}, 'source': f'scripts/analyze/ihvp-v4-runs.py final ({path.name})'}, indent=1) + '\n')


def main() -> None:
    command = sys.argv[1]
    if command == 'scores':
        return scores()
    if command == 'anchors':
        refs = references()
        grade([(k, refs[k], split) for k in refs for split in ('public', 'private')])
    elif command == 'replay':
        grade([('v3-iterative', source_dir(V3_ITERATIVE), 'public')])
    elif command == 'repeat':
        refs = references()
        grade([(f'{k}-r2', refs[k], split) for k in refs for split in ('public', 'private')]
              + [('v3-iterative-r2', source_dir(V3_ITERATIVE), 'public')])
    elif command == 'final':
        grade([(f'final-{sys.argv[3]}', source_dir(sys.argv[2]), 'private')])
    elif command == 'report':
        report()


if __name__ == '__main__':
    main()
