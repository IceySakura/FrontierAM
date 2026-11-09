"""Seal exponential-v2 (dataset v2 + human-SOTA panel) into .runtime-exp2 without touching the checked-in task.

The v1 evaluator (run_docker.py, supervisor.py), public tools and image are reused unchanged; only the manifest, the
case files and the reference change. Layout of the sealed copy (the evaluator finds tools via parents[2]):

  public/{tools,sdk}/         v1 public tools, byte-identical
  docker_eval/evaluator/      v1 evaluator, byte-identical
  data/                       graded manifest and case NPZs (truth-qualified only)
  panel/<member>/             frozen human-SOTA panel members, each an ordinary submission
  public.tar                  development package: v1 docs and tools + V2.md + v2 public cases
  lock.json                   SHA-256 of every sealed file, the image ID and the panel membership

Usage: [FRONTIERAM_EXP2_SEAL=DIR] prepare_exponential_v2.py --graded DIR [--public-dev DIR] [--public-full DIR]
       [--repeats N] [--force]
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tarfile

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'tasks/exponential'
V2 = EXP / 'sota_v1'
SEAL = Path(os.environ.get('FRONTIERAM_EXP2_SEAL') or ROOT / '.runtime-exp2')
PANEL = ('expokit', 'kiops', 'expmv', 'scipy', 'chebyshev', 'slepc', 'rational', 'exputils')
QUALIFIED = {'chebyshev_audited_qualified', 'analytic_dense_eigh_qualified'}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_cases(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text())
    for case in manifest['cases']:
        if case.get('truth_status') not in QUALIFIED:
            raise SystemExit(f'{directory}: {case["id"]} has no qualified truth ({case.get("truth_status")})')
        if sha(directory / case['file']) != case['sha256']:
            raise SystemExit(f'{directory}: {case["id"]} does not match its manifest hash')
    return manifest


def copy_cases(source: Path, target: Path) -> dict:
    manifest = checked_cases(source)
    target.mkdir(parents=True)
    for case in manifest['cases']:
        shutil.copyfile(source / case['file'], target / case['file'])
    shutil.copyfile(source / 'manifest.json', target / 'manifest.json')
    return manifest


def seal_member(source: Path, target: Path, loaded: set[str]) -> dict:
    """Copy a panel member as regular files only (the evaluator rejects links) without duplicating every alias.

    Per link group (a target file and the links resolving to it) only the names that occur as a string in some
    regular file of the member (ELF NEEDED entries, JLL library names, sysimage dlopen strings) or that the loader
    requested in a recorded run (`loaded`, e.g. Julia's dlopen("libx") resolving libx.so) are materialised; the
    target itself is kept when no name of its group is needed. Every other name is dropped.
    """
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    links = {p: Path(os.path.realpath(p)) for p in target.rglob('*') if p.is_symlink()}
    if not links:
        return {'links': 0}
    if any(not t.is_file() or not t.is_relative_to(target.resolve()) for t in links.values()):
        raise SystemExit(f'{source}: link to a directory or outside the member')
    groups = {}
    for link, resolved in links.items():
        groups.setdefault(resolved, []).append(link)
    names = {p.name for p in links} | {t.name for t in groups}
    pattern = re.compile(b'|'.join(re.escape(n.encode()) for n in sorted(names, key=len, reverse=True)))
    found = set()
    for path in target.rglob('*'):
        if path.is_file() and not path.is_symlink():
            found.update(m.group().decode() for m in pattern.finditer(path.read_bytes()))
    found |= loaded
    kept = dropped = 0
    for resolved, aliases in groups.items():
        wanted = [a for a in aliases if a.name in found]
        for alias in aliases:
            alias.unlink()
        for alias in wanted:
            shutil.copy2(resolved, alias)
        kept += len(wanted)
        if wanted and resolved.name not in found:
            resolved.unlink()
            dropped += 1
    dropped += len(links) - kept
    return {'links': len(links), 'materialised': kept, 'names_dropped': dropped,
            'bytes': sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}


def public_tar(path: Path, dev: Path, full: Path | None) -> None:
    def clean(member):
        member.uid = member.gid = 1000
        member.uname = member.gname = ''
        member.mtime = 0
        return member

    with tarfile.open(EXP / 'docker_eval/public_pilot_v2/public_pilot.tar') as v1, \
            tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as out:
        for member in v1.getmembers():
            if member.name.startswith(('task/dev/', './task/dev/')):
                continue
            out.addfile(clean(member), v1.extractfile(member) if member.isfile() else None)
        payload = (V2 / 'public/V2.md').read_bytes()
        info = tarfile.TarInfo('task/V2.md')
        info.size, info.mode = len(payload), 0o644
        out.addfile(clean(info), io.BytesIO(payload))
        for directory, arcname in ((dev, 'task/dev'), (full, 'task/dev_full')):
            if directory is None:
                continue
            manifest = checked_cases(directory)
            out.add(directory / 'manifest.json', arcname=f'{arcname}/manifest.json', filter=clean)
            for case in manifest['cases']:
                out.add(directory / case['file'], arcname=f'{arcname}/{case["file"]}', filter=clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--graded', type=Path, required=True, help='truth-qualified case directory graded by the task')
    parser.add_argument('--public-dev', type=Path, default=V2 / 'dataset_v2/cases/public_smoke')
    parser.add_argument('--public-full', type=Path, help='full-size public cases shipped as task/dev_full')
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime-suite', help='runtime holding the v1 image ID')
    parser.add_argument('--repeats', type=int, default=3, help='timed repeats per case (1 for the v6.2 public-feedback seal)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if SEAL.exists():
        if not args.force:
            raise SystemExit(f'{SEAL} exists; pass --force to replace it')
        shutil.rmtree(SEAL)
    SEAL.mkdir()
    shutil.copytree(EXP / 'public/tools', SEAL / 'public/tools', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(EXP / 'public/sdk', SEAL / 'public/sdk', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run_docker.py', 'supervisor.py'):
        (SEAL / 'docker_eval/evaluator').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXP / 'docker_eval/evaluator' / name, SEAL / 'docker_eval/evaluator' / name)
    graded = copy_cases(args.graded.resolve(), SEAL / 'data')
    panel, frozen_path = [m for m in PANEL if (V2 / 'panel' / m / 'run.sh').exists()], V2 / 'calibration_v2/FROZEN.json'
    if frozen_path.exists():
        # After public calibration only the frozen members, byte-identical to their frozen state, form the reference.
        frozen = json.loads(frozen_path.read_text())
        panel = [m for m in PANEL if m in frozen['members']]
        for relative, digest in frozen['panel_files_sha256'].items():
            if relative.split('/')[0] in panel and sha(V2 / 'panel' / relative) != digest:
                raise SystemExit(f'panel file changed since freezing: {relative}')
    loaded = json.loads((V2 / 'calibration_v2/loader_names.json').read_text())['members']
    sealing = {member: seal_member(V2 / 'panel' / member, SEAL / 'panel' / member, set(loaded.get(member, ())))
               for member in panel}
    public_tar(SEAL / 'public.tar', args.public_dev.resolve(), args.public_full.resolve() if args.public_full else None)
    image = json.loads((args.runtime / 'runtime-lock.json').read_text())['images']['exponential']
    files = {str(p.relative_to(SEAL)): sha(p) for p in sorted(SEAL.rglob('*')) if p.is_file()}
    lock = {
        'schema': 1,
        'image': image,
        'graded_source': str(args.graded.resolve().relative_to(ROOT)),
        'graded_scope': graded.get('scope'),
        'graded_cases': len(graded['cases']),
        'scored_cases': sum(c['role'] == 'scored' for c in graded['cases']),
        'panel': panel,
        'repeats': args.repeats,
        'panel_frozen': frozen_path.exists(),
        'panel_sealing': sealing,
        'public_tar_sha256': files['public.tar'],
        'files': files,
    }
    (SEAL / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({k: v for k, v in lock.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
