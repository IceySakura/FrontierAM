"""Seal galaxy-gravity-v4: the v3 task with the author's B.4 data (tasks/galaxy-gravity/analysis/B4_PRECALIBRATION.md).

Derived from the v3 seal (hard-linked copy; replaced files are unlinked first, so v3 stays byte-identical):
- grader: the author's B.4 tests/ (per-case CPU limits passed as GALAXY_CPU_LIMIT_S, wall max(30, ceil(3T+10)) s),
  with one portability edit in worker.py (accept x86_64, as in v3);
- private data: the 80 formal snapshots and full references of the B.4 evaluation bundle; scoring-config.json with
  the per-case limits calibrated on this machine (data/private/private/local-limits.json, from
  scripts/analyze/galaxy-v4-calibrate.py) once they exist;
- references r4: Treecode2 standard tree at theta 0.8 (weak) and modified tree at theta 0.7 (strong);
- public side: the B.4 statement with the v3 scoring rule extended to the 4 scales x 5 families, the B.4 public
  development data as samples/, and the maintainer-only feedback pool (scripts/seal/make_galaxy_v4_feedback.py).

Usage: prepare_galaxy_v4.py [--force]
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / '.runtime-ops'
PARENT, SEAL = OPS / 'galaxy-gravity-v3', OPS / 'galaxy-gravity-v4'
SPEC = ROOT / 'tasks/galaxy-gravity/versions/v4'
BUNDLE = ROOT / 'var/raw/galaxy-b4-eval'
STATEMENT = ROOT / 'var/raw/galaxy-b4-0.4.0/problem.md'
SAMPLES = SPEC / 'data/samples'
PRIVATE = SPEC / 'data/private/private'
FEEDBACK = SPEC / 'data/public'
R4 = ROOT / 'tasks/galaxy-gravity/references/r4/baselines'
SCALES = (65536, 262144, 1048576, 4194304)

SCORING_OLD = ('每张快照取三次运行的 $e$ 的算术平均，并按粒子规模和分布分别报告误差、耗时、内存与有效率。只有三次均有效的快照才有完整的快照误差；'
               '缺失或失败不会被当作零误差。0/5/10的语义锚点已经冻结，但各规模与分布的评分权重及最终原始指标汇总仍待冻结，因此当前不输出正式总分。'
               '分层均值仅用于描述，不隐含各规模等权或按粒子数加权。\n\n'
               '正式评分采用：**0分对应theta=0.8经典简单树的基准水平，5分对应theta=0.7改进树的基准水平；全部预测与参考一致的零误差对应10分。** '
               '最终权重和原始指标汇总在正式比较提交之前公布并冻结。当前开发试跑主要核验源码交接、输出合法性和资源限制，返回接口有效性及原始分层结果。')
SCORING_NEW = '''每张快照取三次运行的 $e$ 的算术平均；每个“粒子规模 × 场景类别”组合（共 20 组）取其四张快照的算术平均；最后将 20 组等权平均，得到原始误差 $E$。$E$ 越低越好。方向偏差和大小偏差都会增加误差。只要 240 次全部在限时内有效，运行更快不会额外加分。

**满分 100，参照 = 50，按原始误差的对数计分。** 0 分锚点 $E_0$ 是 theta=0.8 经典简单树参照的原始误差，50 分锚点 $E_{50}$ 是 theta=0.7 改进树参照的原始误差，两者都在评测机上按相同资源、时限和汇总规则标定。$E\\le 10^{-15}$（双精度可达的误差下限）对应 100 分。

$$
S(E)=
\\begin{cases}
0,& E\\ge E_0,\\\\
50\\,\\log(E_0/E)/\\log(E_0/E_{50}),& E_{50}\\le E<E_0,\\\\
\\min\\bigl(100,\\;50+50\\,\\log(E_{50}/E)/\\log(E_{50}/10^{-15})\\bigr),& E<E_{50}.
\\end{cases}
$$'''


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f'{where}: expected exactly one occurrence of {old[:60]!r}')
    return text.replace(old, new)


def rewrite(path: Path, data: bytes) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def link_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)


def ranges(limits: dict[str, float], cases: list[dict]) -> str:
    def fmt(v):
        return f'{v:.1f}'
    parts = []
    for n in SCALES:
        values = [limits[c['case']] for c in cases if c['n'] == n]
        parts.append(f'{n:,} 粒子 **{fmt(min(values))}–{fmt(max(values))} 秒**')
    return (f'当前四档正式数据的逐例 CPU 时限范围约为：{"，".join(parts)}；整题覆盖约 **{fmt(min(limits.values()))}–'
            f'{fmt(max(limits.values()))} CPU 秒**。这是按改进树参照（theta=0.7）在评测机上每张 3 次运行的最大 CPU 时间乘以 1.25 '
            '标定得到的分档范围，具体快照的时限仍由评测器隐藏并通过环境变量提供。')


def statement(limits: dict | None, cases: list[dict]) -> tuple[str, list[str]]:
    text, edits = STATEMENT.read_text(), []
    for old, new in [
        ('评测环境为 Linux ARM64。', '评测环境为 Linux x86-64（AMD EPYC 7K62）。'),
        ('整份提交不满足完整有效性要求。当前预标定会保留其余调用的原始记录及失败状态，失败快照不能以成功的部分调用代替完整结果。** 提交无法构建或启动同样视为失败。',
         '整份提交记 0 分，不进入有效精度排名。** 提交无法构建或启动同样计 0 分。'),
        (SCORING_OLD, SCORING_NEW),
        ('[公开开发数据、文件说明与本地自检入口](public-dev/README.md)', '[样例数据、文件说明与本地自检入口](samples/README.md)'),
    ]:
        text = replace_once(text, old, new, 'problem.md')
        edits.append(old[:40])
    if limits:
        start = text.index('当前四档正式数据的逐例 CPU 时限范围约为')
        end = text.index('\n', start)
        text = text[:start] + ranges(limits, cases) + text[end:]
        edits.append('per-scale limit ranges (local calibration)')
    return text, edits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if SEAL.exists():
        if not args.force:
            raise SystemExit(f'{SEAL} exists; pass --force to reseal')
        shutil.rmtree(SEAL)
    shutil.copytree(PARENT, SEAL, symlinks=True, copy_function=os.link,
                    ignore=lambda d, names: ['public.tar'] if Path(d) == PARENT else [])
    lock = json.loads((PARENT / 'lock.json').read_text())
    operator, public = SEAL / lock['roots']['operator'], SEAL / lock['roots']['public']
    changes = []

    tests = operator / 'tasks/galaxy-gravity/harbor/tests'
    for name in ('grade.py', 'worker.py', 'bridge.py', 'test.sh', 'self_check.py'):
        rewrite(tests / name, (BUNDLE / 'task/tests' / name).read_bytes())
    worker = (tests / 'worker.py').read_text()
    worker = replace_once(worker, 'os.uname().machine != "aarch64"', 'os.uname().machine not in ("aarch64", "x86_64")', 'worker.py')
    rewrite(tests / 'worker.py', worker.encode())
    changes.append({'grader': 'B.4 tests/', 'edit': 'worker.py accepts x86_64', 'worker_sha256': sha(tests / 'worker.py')})
    for which in ('standard', 'modified'):
        for f in (R4 / which).iterdir():
            rewrite(operator / 'baselines' / which / f.name, f.read_bytes())
    changes.append({'references': 'r4 (standard theta 0.8, modified theta 0.7)'})

    link_tree(PRIVATE / 'data', operator / 'private/data')
    author = json.loads((PRIVATE / 'formal-config.json').read_text())
    local_path = PRIVATE / 'local-limits.json'
    local = json.loads(local_path.read_text())['limits'] if local_path.exists() else None
    cases = []
    for c in author['cases']:
        entry = {k: c[k] for k in ('case', 'family', 'n', 'input', 'truth', 'input_sha256', 'truth_sha256')}
        entry['author_cpu_limit_s'] = c['cpu_limit_s']
        if local:
            entry['cpu_limit_s'] = local[c['case']]
        cases.append(entry)
    rewrite(operator / 'private/scoring-config.json',
            (json.dumps({'repeats': 3, 'calibrated': bool(local), 'cases': cases}, indent=2) + '\n').encode())

    dev = public / 'public-dev'
    shutil.rmtree(dev)
    shutil.copytree(SAMPLES, dev, copy_function=os.link)
    feedback = json.loads((FEEDBACK / 'manifest.json').read_text()) if (FEEDBACK / 'manifest.json').exists() else None
    if feedback:
        link_tree(FEEDBACK / 'cases', dev / 'cases')
        fb_limits = json.loads((FEEDBACK / 'local-limits.json').read_text())['limits'] if (FEEDBACK / 'local-limits.json').exists() else {}
        for c in feedback['cases']:
            if c['case'] in fb_limits:
                c['cpu_limit_s'] = fb_limits[c['case']]
        rewrite(dev / 'manifest.json', (json.dumps(feedback, indent=1) + '\n').encode())
    text, statement_edits = statement(local, cases)
    rewrite(public / 'problem.md', text.encode())

    members = {'problem.md': text.encode()}
    for path in sorted(SAMPLES.rglob('*')):
        if path.is_file():
            members['samples/' + str(path.relative_to(SAMPLES))] = path.read_bytes()
    with tarfile.open(SEAL / 'public.tar', 'w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime, info.uid, info.gid = len(members[name]), 0o644, 0, 1000, 1000
            archive.addfile(info, io.BytesIO(members[name]))

    (SEAL / 'lock.json').unlink()
    lock.update(name='galaxy-gravity-v4', derived_from='galaxy-gravity-v3', particles=list(SCALES), cpu_limit_s=None,
                formal_cases=len(cases), public_cases=len(feedback['cases']) if feedback else 0, calibrated=bool(local),
                source_bundle={'name': 'galaxy-gravity-b4-evaluation', 'version': '0.4.0-precalibration'},
                public_tar_sha256=sha(SEAL / 'public.tar'), workspace={'files': sorted(members), 'statement_edits': statement_edits},
                v4_changes=changes, sealed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    lock.pop('edits', None)
    (SEAL / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')

    spec = json.loads((SPEC / 'task.json').read_text())
    spec.update(status='formal' if local and feedback else 'draft',
                status_reason=('v3 with the author B.4 data: 80 formal snapshots at four scales with per-case CPU limits '
                               'calibrated on this machine, Treecode2 anchors r4 (scripts/seal/prepare_galaxy_v4.py).'
                               if local and feedback else 'B.4 data sealed; local per-case limits or feedback pool pending.'),
                references={'weak': 'repo:tasks/galaxy-gravity/references/r4/baselines/standard',
                            'strong': 'repo:tasks/galaxy-gravity/references/r4/baselines/modified'})
    (SPEC / 'task.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    (SPEC / 'statement.md').write_text(text)
    print(json.dumps({'sealed': str(SEAL), 'calibrated': bool(local), 'feedback_cases': lock['public_cases'],
                      'status': spec['status'], 'workspace_files': len(members)}, indent=1))


if __name__ == '__main__':
    main()
