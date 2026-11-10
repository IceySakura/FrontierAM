"""Seal exponential-v3, the protocol v6.3 version of exponential-v2 (results/EVAL_TIME_LIMITS.md, results/PROTOCOL_V63.md).

Same data, panel, image, evaluator and scorer as exponential-v2, with three changes:
  - per-session limits: one_shot 60 -> 15 s, reuse 240 -> 60 s, READY startup 15 -> 10 s (evaluator, local runner,
    scorer, RULES.json and the documents that state them); every edit must match exactly once;
  - the workspace: documents, tools, SDK, starter and samples/ (meta.json v63_data samples from the smoke set);
    the full smoke set and the full-size public cases (dev_full/, part of the public-feedback set) are no longer given;
  - the public-feedback seal keeps two public cases per structure family (one one_shot, one reuse).
The new seals are hard-linked copies of the v2 seals; edited files are unlinked before writing.

Usage: prepare_exp_v3.py [--force]      (writes .runtime-exp3 and .runtime-exp3-public)
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEALS = {ROOT / '.runtime-exp2': ROOT / '.runtime-exp3', ROOT / '.runtime-exp2-public': ROOT / '.runtime-exp3-public'}
SAMPLES = ['syn_diffusion2d_p4_k1', 'syn_geometric_p4_k1', 'syn_heavytail_p4_k1', 'syn_logspectrum_p4_k1',
           'syn_signed_p4_k1', 'control_diagonal']

TOOL_EDITS = {
    'run_local.py': [
        ('READY has a separate 15-second startup deadline.', 'READY has a separate 10-second startup deadline.'),
        ('def run_session(command, case_path, repeat=1, *, startup_timeout=15.0,',
         'def run_session(command, case_path, repeat=1, *, startup_timeout=10.0,'),
        ('    cap = 60.0 if metadata["scenario"] == "one_shot" else 240.0',
         '    cap = 15.0 if metadata["scenario"] == "one_shot" else 60.0'),
        ('    for value, name, upper in ((startup_timeout, "startup_timeout", 15.0),',
         '    for value, name, upper in ((startup_timeout, "startup_timeout", 10.0),'),
    ],
    'scoring.py': [
        ('"one_shot": {"wall_seconds": 60.0, "cpu_seconds": 60.0, "requests": 1},',
         '"one_shot": {"wall_seconds": 15.0, "cpu_seconds": 15.0, "requests": 1},'),
        ('"reuse": {"wall_seconds": 240.0, "cpu_seconds": 240.0, "requests": 16},',
         '"reuse": {"wall_seconds": 60.0, "cpu_seconds": 60.0, "requests": 16},'),
    ],
}
EVALUATOR_EDITS = [
    ('startup_timeout=15, wall_timeout=None, cpu_timeout=None,', 'startup_timeout=10, wall_timeout=None, cpu_timeout=None,'),
    ("    cap = 60 if metadata['requests'] == 1 else 240", "    cap = 15 if metadata['requests'] == 1 else 60"),
    ('for val, upper in ((wall_timeout, cap), (cpu_timeout, cap), (startup_timeout, 15)):',
     'for val, upper in ((wall_timeout, cap), (cpu_timeout, cap), (startup_timeout, 10)):'),
]
DOC_EDITS = {
    'task/V2.md': [
        ('（one_shot 60 秒，reuse 240 秒）', '（one_shot 15 秒，reuse 60 秒）'),
        ('## 3. 公开开发数据', '## 3. 样例数据'),
        ('- `dev/`：缩小规模的冒烟集，56 个用例（8 个母算子 × 3 精度 × 2 场景，加 8 个解析控制例），覆盖全部结构族与场景。\n',
         '- `samples/`：缩小规模的样例，6 个用例（5 个合成结构族各 1 例，加 1 个解析控制例）。\n'),
        ('- `dev_full/`（若存在）：正式规模的公开母算子上的 one_shot 用例。reuse 用例在正式规模下每个约 30 MB，开发空间放不下，因此不随包提供；可以用同一个 one_shot 用例连续发送多个请求来估计 reuse 的表现。\n', ''),
        ('python tools/run_local.py --manifest dev/manifest.json', 'python tools/run_local.py --manifest samples/manifest.json'),
    ],
    'task/INSTRUCTION.md': [
        ('| one_shot | 数值会话墙钟60秒；整个进程CPU累计60秒，包含初始化 |', '| one_shot | 数值会话墙钟15秒；整个进程CPU累计15秒，包含初始化 |'),
        ('| reuse | 数值会话墙钟240秒；整个进程CPU累计240秒，包含初始化 |', '| reuse | 数值会话墙钟60秒；整个进程CPU累计60秒，包含初始化 |'),
        ('python tools/run_local.py --manifest dev/manifest.json', 'python tools/run_local.py --manifest samples/manifest.json'),
    ],
    'task/API.md': [('初始化墙钟最多15秒', '初始化墙钟最多10秒')],
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f'{where}: expected exactly one occurrence of {old[:70]!r}')
    return text.replace(old, new)


def rewrite(path: Path, data: bytes) -> None:
    path.unlink()
    path.write_bytes(data)


def workspace(parent_tar: Path) -> dict[str, bytes]:
    members = {}
    with tarfile.open(parent_tar) as archive:
        for member in archive.getmembers():
            if member.isfile() and not member.name.startswith(('task/dev/', 'task/dev_full/')):
                members[member.name] = archive.extractfile(member).read()
            elif member.isfile() and member.name.startswith('task/dev/'):
                stem = Path(member.name).stem
                if stem in SAMPLES or member.name == 'task/dev/manifest.json':
                    members[member.name.replace('task/dev/', 'task/samples/', 1)] = archive.extractfile(member).read()
    manifest = json.loads(members['task/samples/manifest.json'])
    cases = manifest['cases'] if isinstance(manifest, dict) else manifest
    kept = [c for c in cases if c['id'] in SAMPLES]
    if len(kept) != len(SAMPLES):
        raise SystemExit('not every sample is in the smoke manifest')
    if isinstance(manifest, dict):
        manifest['cases'] = kept
    else:
        manifest = kept
    members['task/samples/manifest.json'] = (json.dumps(manifest, indent=1) + '\n').encode()
    for name, edits in DOC_EDITS.items():
        text = members[name].decode()
        for old, new in edits:
            text = once(text, old, new, name)
        members[name] = text.encode()
    for name, edits in TOOL_EDITS.items():
        text = members[f'task/tools/{name}'].decode()
        for old, new in edits:
            text = once(text, old, new, name)
        members[f'task/tools/{name}'] = text.encode()
    rules = json.loads(members['task/RULES.json'])
    rules['scenarios']['one_shot'].update(wall_seconds=15, cpu_seconds=15)
    rules['scenarios']['reuse'].update(wall_seconds=60, cpu_seconds=60)
    rules['session']['startup_wall_seconds'] = 10
    members['task/RULES.json'] = (json.dumps(rules, indent=2) + '\n').encode()
    return members


def write_tar(members: dict[str, bytes], out: Path) -> None:
    with tarfile.open(out, 'w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime, info.uid, info.gid = len(members[name]), 0o644, 0, 1000, 1000
            archive.addfile(info, io.BytesIO(members[name]))


def public_subset(seal: Path) -> list[str]:
    """Two public cases per structure family for the feedback seal: one one_shot and one reuse."""
    path = seal / 'data/manifest.json'
    manifest = json.loads(path.read_text())
    kept, seen = [], set()
    for case in manifest['cases']:
        key = (case['family'], case['scenario'])
        if key not in seen and case['role'] == 'scored':
            seen.add(key)
            kept.append(case)
    for family in sorted({c['family'] for c in manifest['cases']} - {c['family'] for c in kept}):
        kept += [c for c in manifest['cases'] if c['family'] == family][:2]
    manifest['cases'] = kept
    rewrite(path, (json.dumps(manifest, indent=1) + '\n').encode())
    return [c['id'] for c in kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    members = workspace(ROOT / '.runtime-exp2/public.tar')
    for parent, seal in SEALS.items():
        if seal.exists():
            if not args.force:
                raise SystemExit(f'{seal} exists; pass --force to reseal')
            shutil.rmtree(seal)
        shutil.copytree(parent, seal, symlinks=True, copy_function=os.link,
                        ignore=lambda d, names: ['public.tar'] if Path(d) == parent else [])
        lock = json.loads((seal / 'lock.json').read_text())
        (seal / 'lock.json').unlink()
        for name, edits in TOOL_EDITS.items():
            path = seal / 'public/tools' / name
            text = path.read_text()
            for old, new in edits:
                text = once(text, old, new, name)
            rewrite(path, text.encode())
        path = seal / 'docker_eval/evaluator/run_docker.py'
        text = path.read_text()
        for old, new in EVALUATOR_EDITS:
            text = once(text, old, new, 'run_docker.py')
        rewrite(path, text.encode())
        write_tar(members, seal / 'public.tar')
        changed = ['public/tools/run_local.py', 'public/tools/scoring.py', 'docker_eval/evaluator/run_docker.py', 'public.tar']
        kept = None
        if seal.name.endswith('-public'):
            kept = public_subset(seal)
            changed.append('data/manifest.json')
        for relative in changed:
            if relative in lock['files'] or relative == 'data/manifest.json':
                lock['files'][relative] = sha_bytes((seal / relative).read_bytes())
        lock.update(schema=lock.get('schema'), public_tar_sha256=lock['files']['public.tar'], task='exponential-v3',
                    derived_from=parent.name, v63={'one_shot_s': 15, 'reuse_s': 60, 'startup_s': 10, 'samples': SAMPLES,
                                                   'public_cases': kept})
        (seal / 'lock.json').write_text(json.dumps(lock, indent=1) + '\n')
        print(json.dumps({'sealed': str(seal), 'public_cases': len(kept) if kept else None}))


if __name__ == '__main__':
    main()
