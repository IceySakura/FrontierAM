"""Seal the protocol v6.3 versions of the operator tasks (results/PROTOCOL_V63.md, results/EVAL_TIME_LIMITS.md).

Each new version is derived from the current sealed one: the grader, private data and references stay; the model's
workspace becomes the statement plus the samples (meta.json v63_data) under samples/, with the statement edited for
the samples, the new limits and (galaxy) the new reference and log-scale mapping; lock overrides carry the new case
limits and set sizes. Every text edit must match exactly once. The new seal is a hard-linked copy of the parent seal;
edited files are unlinked before writing, so the parent stays byte-identical.

Usage: prepare_v63.py TASK [--force]     TASK: galaxy-gravity | polarization-demosaicking | species-tree | trajectory-covariance
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

GALAXY_SCORING_OLD = '''**0 分对应经典简单方法的基准水平，5 分对应现有较先进方法的基准水平。** 两份参照程序按相同资源和汇总规则标定原始误差 $E_0>E_5>0$，分别作为 0 分与 5 分的评分锚点。全部预测与参考值一致时 $E=0$，对应 10 分。有效提交的分数按原始误差分段线性插值。

$$
S(E)=
\\begin{cases}
0,& E\\ge E_0,\\\\
5\\,(E_0-E)/(E_0-E_5),& E_5\\le E<E_0,\\\\
10-5E/E_5,& 0\\le E<E_5.
\\end{cases}
$$'''
GALAXY_SCORING_NEW = '''**满分 100，参照 = 50，按原始误差的对数计分。** 0 分锚点 $E_0$ 是经典树算法参照的原始误差，50 分锚点 $E_{50}$ 是高阶快速多极方法参照的原始误差，两者都在评测机上按相同资源和汇总规则标定。$E\\le 10^{-15}$（双精度可达的误差下限）对应 100 分。

$$
S(E)=
\\begin{cases}
0,& E\\ge E_0,\\\\
50\\,\\log(E_0/E)/\\log(E_0/E_{50}),& E_{50}\\le E<E_0,\\\\
\\min\\bigl(100,\\;50+50\\,\\log(E_{50}/E)/\\log(E_{50}/10^{-15})\\bigr),& E<E_{50}.
\\end{cases}
$$'''
GALAXY_README_PUBLIC_OLD = '''## 正式规模的公开快照

`cases/` 下有 10 张正式规模（131,072 个粒子）的公开快照，五类模型各 2 张，种子与正式评测不同。`case-NN.in` 为输入，`case-NN.ref` 为按同一直接求和程序算出的参考加速度，`manifest.json` 列出每张的模型和 SHA-256。检查自己的输出：'''
GALAXY_README_PUBLIC_NEW = '''## 正式规模的样例快照

`cases/` 下有 1 张正式规模（131,072 个粒子）的样例快照（`smooth-hernquist` 模型），种子与正式评测不同。`case-01.in` 为输入，`case-01.ref` 为按同一直接求和程序算出的参考加速度，`manifest.json` 列出它的模型和 SHA-256。其余各类模型的快照可用生成器自行生成。检查自己的输出：'''

def polarization_samples(seal: Path) -> dict[str, bytes]:
    """Two public Wen et al. scenes as samples: the mosaic input and the four-angle reference, via the grader's mosaic."""
    import sys
    import numpy as np

    lock = json.loads((seal / 'lock.json').read_text())
    sys.path.insert(0, str(seal / lock['roots']['operator'] / 'harbor/polarization-demosaicking'))
    from scoring import mosaic

    data = OPS / 'polarization-demosaicking-feedback'
    files = {}
    for scene in json.loads((data / 'manifest.json').read_text())['scenes'][:2]:
        reference = np.load(data / scene['file'], allow_pickle=False)
        for name, array in (('raw.npy', mosaic(reference)), ('reference.npy', reference)):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            files[f'public-dev/{scene["scene"]}/{name}'] = buffer.getvalue()
    return files


TASKS = {
    'polarization-demosaicking': {
        'parent': 'polarization-demosaicking', 'name': 'polarization-demosaicking-v2', 'version': 'v2',
        'keep': ['problem.md', 'public-dev/check.py'],
        'extra': polarization_samples,
        'edits': [('problem.md', '下载[公开开发自检脚本](public-dev/check.py)。', '下载[样例自检脚本](samples/check.py)。'),
                  ('problem.md', 'python3 public-dev/check.py /path/to/solution.py', 'python3 samples/check.py /path/to/solution.py'),
                  ('problem.md', '正式时限在评测方逐景启动的独立进程中执行。',
                   '正式时限在评测方逐景启动的独立进程中执行。\n\n`samples/wen-001/` 与 `samples/wen-002/` 各含一景真实场景：输入马赛克 '
                   '`raw.npy`（`uint8`，形状 `(1024, 1024)`）和四方向参考图 `reference.npy`（`uint8`，形状 `(4, 1024, 1024)`，'
                   '依次为 0°、45°、90°、135°）。它们取自公开数据集 Wen et al. 2019，与正式场景无关。')],
        'task': {'instruction': 'Read /workspace/problem.md in full and inspect /workspace/samples. Implement the '
                                'demosaicking program described there.'},
        'lock': {},
    },
    'species-tree': {
        'parent': 'species-tree-v2', 'name': 'species-tree-v3', 'version': 'v3',
        'keep': ['problem.md', 'public-dev/README.md', 'public-dev/check_rf.py', 'public-dev/manifest.json',
                 'public-dev/cases/001/genes.tre', 'public-dev/cases/001/truth.tre',
                 'public-dev/cases/002/genes.tre', 'public-dev/cases/002/truth.tre'],
        'manifest_keep': ['001', '002'],
        'edits': [('problem.md', '180 秒墙钟', '135 秒墙钟'),
                  ('problem.md', '正式集含 200 棵独立生成的物种树', '正式集含 96 棵独立生成的物种树'),
                  ('problem.md', '公开开发数据在 `public-dev/`：50 组', '样例数据在 `samples/`：2 组'),
                  ('problem.md', 'dev=/workspace/public-dev', 'dev=/workspace/samples'),
                  ('public-dev/README.md', '# 物种树推断开发数据（250 个物种）', '# 物种树推断样例数据（250 个物种）'),
                  ('public-dev/README.md', '每个 `cases/001` 至 `cases/050` 目录包含', '`cases/001` 与 `cases/002` 两个目录各包含'),
                  ('public-dev/README.md', 'dev=/workspace/public-dev', 'dev=/workspace/samples')],
        # The per-case wall limit (results/EVAL_TIME_LIMITS.md): 1.25 x the strong reference's slowest case.
        'operator_edits': [(f'packages/operator/operator/{f}', old, new) for f, old, new in (
            ('grade.py', 'n=len(rows); cap=300+180*n', 'n=len(rows); cap=300+135*n'),
            ('grade.py', 'attempt,fresh=attempt_for(cid,180,out,cap,retry,incident)', 'attempt,fresh=attempt_for(cid,135,out,cap,retry,incident)'),
            ('grade.py', "if fresh: execute('run',artifact,work,180,MEMORY,Path(row['input']))",
             "if fresh: execute('run',artifact,work,135,MEMORY,Path(row['input']))"),
            ('runtime/sandbox.py', 'if not (0 < limit <= (300 if mode == "build" else 180) and 0 < memory <= MEMORY):',
             'if not (0 < limit <= (300 if mode == "build" else 135) and 0 < memory <= MEMORY):'),
            ('runtime/worker.py', 'limit_s not in {2, 300 if mode == "build" else 180}:',
             'limit_s not in {2, 300 if mode == "build" else 135}:'))],
        # Cases are independent draws with the same parameters, so leading cases are a random subset.
        'data_manifests': {'private/manifest.json': ('case', 96), 'packages/public/public-dev/manifest.json': ('case', 5)},
        'task': {'instruction': 'Read /workspace/problem.md in full and inspect /workspace/samples. Implement the species-tree '
                                'inference program described there.'},
        'lock': {'case_limit_s': 135, 'formal_cases': 96, 'public_cases': 5},
    },
    'galaxy-gravity': {
        'parent': 'galaxy-gravity-v2', 'name': 'galaxy-gravity-v3', 'version': 'v3',
        'keep': ['problem.md', 'public-dev/README.md', 'public-dev/check_output.py', 'public-dev/generate.py',
                 'public-dev/smoke.in', 'public-dev/smoke.ref', 'public-dev/cases/case-01.in', 'public-dev/cases/case-01.ref',
                 'public-dev/manifest.json'],
        'manifest_keep': ['case-01'],
        'edits': [('problem.md', GALAXY_SCORING_OLD, GALAXY_SCORING_NEW),
                  ('problem.md', '[公开开发数据、文件说明与本地自检入口](public-dev/README.md)',
                   '[样例数据、文件说明与本地自检入口](samples/README.md)'),
                  ('public-dev/README.md', '# 星系快照引力题 · 公开开发材料', '# 星系快照引力题 · 样例'),
                  ('public-dev/README.md', GALAXY_README_PUBLIC_OLD, GALAXY_README_PUBLIC_NEW)],
        'task': {'references': {'weak': 'baselines/standard', 'strong': 'repo:tasks/galaxy-gravity/references/r2/source'},
                 'instruction': 'Read /workspace/problem.md in full and inspect /workspace/samples. Implement the '
                                'gravitational-acceleration program described there.'},
        'lock': {},
    },
}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f'{where}: expected exactly one occurrence of {old[:80]!r}')
    return text.replace(old, new)


def workspace(parent_tar: Path, out: Path, cfg: dict, seal: Path) -> dict:
    """The samples-only workspace: kept members of the parent's public package, public-dev/ renamed samples/."""
    members = {}
    with tarfile.open(parent_tar) as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name in cfg['keep']:
                members[member.name] = archive.extractfile(member).read()
    missing = set(cfg['keep']) - set(members)
    if missing:
        raise SystemExit(f'not in the parent package: {sorted(missing)}')
    if cfg.get('extra'):
        members.update(cfg['extra'](seal))
    edits = []
    for name, old, new in cfg['edits']:
        before = hashlib.sha256(members[name]).hexdigest()
        members[name] = replace_once(members[name].decode(), old, new, name).encode()
        edits.append({'file': name, 'sha256_before': before, 'sha256_after': hashlib.sha256(members[name]).hexdigest()})
    if cfg.get('manifest_keep') and 'public-dev/manifest.json' in members:
        manifest = json.loads(members['public-dev/manifest.json'])
        manifest['cases'] = [c for c in manifest['cases'] if c['case'] in cfg['manifest_keep']]
        members['public-dev/manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    with tarfile.open(out, 'w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name.replace('public-dev/', 'samples/', 1))
            info.size, info.mode, info.mtime, info.uid, info.gid = len(members[name]), 0o644, 0, 1000, 1000
            archive.addfile(info, io.BytesIO(members[name]))
    return {'files': sorted(n.replace('public-dev/', 'samples/', 1) for n in members), 'edits': edits}


def rewrite(path: Path, data: bytes) -> None:
    """Replace a hard-linked file without touching the parent seal's copy."""
    path.unlink()
    path.write_bytes(data)


def seal_changes(seal: Path, cfg: dict) -> list[dict]:
    applied = []
    for relative, old, new in cfg.get('operator_edits', []):
        path = seal / relative
        before = sha(path)
        rewrite(path, replace_once(path.read_text(), old, new, relative).encode())
        applied.append({'file': relative, 'old': old, 'new': new, 'sha256_before': before, 'sha256_after': sha(path)})
    for relative, (key, count) in cfg.get('data_manifests', {}).items():
        path = seal / relative
        manifest = json.loads(path.read_text())
        before, manifest['cases'] = len(manifest['cases']), manifest['cases'][:count]
        rewrite(path, (json.dumps(manifest, indent=1) + '\n').encode())
        applied.append({'file': relative, 'cases_before': before, 'cases_after': len(manifest['cases']),
                        'kept': [c[key] for c in manifest['cases']], 'sha256_after': sha(path)})
    return applied


def private_copy(parent: Path, seal: Path) -> None:
    """Hard-linked copy of the parent seal (symlinks kept), without its public package."""
    def link(src, dst):
        os.link(src, dst)
    shutil.copytree(parent, seal, symlinks=True, copy_function=link,
                    ignore=lambda d, names: ['public.tar'] if Path(d) == parent else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', choices=sorted(TASKS))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cfg = TASKS[args.task]
    parent, seal = OPS / cfg['parent'], OPS / cfg['name']
    if seal.exists():
        if not args.force:
            raise SystemExit(f'{seal} exists; pass --force to reseal')
        shutil.rmtree(seal)
    private_copy(parent, seal)
    record = workspace(parent / 'public.tar', seal / 'public.tar', cfg, seal)
    changes = seal_changes(seal, cfg)
    lock = json.loads((parent / 'lock.json').read_text())
    (seal / 'lock.json').unlink()
    lock.update(cfg['lock'], name=cfg['name'], derived_from=cfg['parent'], public_tar_sha256=sha(seal / 'public.tar'),
                workspace=record, v63_changes=changes, sealed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (seal / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    spec_dir = ROOT / 'tasks' / args.task / 'versions' / cfg['version']
    spec_dir.mkdir(parents=True, exist_ok=True)
    parent_spec = next(json.loads(p.read_text()) for p in (ROOT / 'tasks' / args.task / 'versions').glob('*/task.json')
                       if json.loads(p.read_text())['name'] == cfg['parent'])
    spec = {**parent_spec, **cfg['task'], 'name': cfg['name'], 'derived_from': cfg['parent'], 'status': 'formal',
            'status_reason': 'Protocol v6.3 version (results/PROTOCOL_V63.md, results/EVAL_TIME_LIMITS.md): samples-only '
                             'workspace; limits and set sizes for the evaluation time caps (scripts/seal/prepare_v63.py).'}
    (spec_dir / 'task.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    with tarfile.open(seal / 'public.tar') as archive:
        (spec_dir / 'statement.md').write_bytes(archive.extractfile('problem.md').read())
    print(json.dumps({'sealed': str(seal), 'workspace_files': record['files'], 'spec': str(spec_dir / 'task.json')}, indent=1))


if __name__ == '__main__':
    main()
