#!/usr/bin/env python3
"""Back up the public/private evaluation data to the private GitHub repository IceySakura/FrontierAM_databack
(user decision 2026-11-23): the hidden public-feedback and held-out sets tracked here (Git LFS on git.woa, removed from
the GitHub mirror by scripts/ops/sync_github.sh) and the raw task data kept only under var/raw.

Files under 100 MiB are plain Git objects, so they use no LFS storage or bandwidth; only files at or above GitHub's
100 MiB hard limit go to Git LFS. Archives whose every member is present extracted are not stored; EXCLUDED.md lists
them with size and SHA-256. The staging repository holds hard links (no extra disk for the working tree); commits are
cut at about 1.0 GiB of new content and pushed one at a time (GitHub refuses pushes over 2 GB). Rerunning adds or
updates changed files only. Text files are scanned for credentials before anything is committed.

Usage: databack.py [--dry-run] [--no-push]
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path('/data/workspace/FrontierAM_databack')
# HTTPS: pushes over SSH from this host stall after ~80 MiB (2026-11-23), HTTPS uploads run at full speed; the
# credential helper reads the token file at push time, so no token is written to the config or a command line
REMOTE = 'https://github.com/IceySakura/FrontierAM_databack.git'
TOKEN_FILE = '/root/.frontieram/github.token'
LFS_ABOVE = 100 * 2**20 - 1  # GitHub hard-rejects >=100 MiB; keep everything smaller off LFS
BATCH = int(1.0 * 2**30)  # stay well under GitHub's ~2 GB push cap; 1.5 GiB still tripped postBuffer
# the paths sync_github.sh removes from the GitHub mirror
TRACKED = re.compile(r'/tests/private/|/data/private/|/data/public/|/cases/private/|/docker_eval/private_pilot/'
                     r'|private_seeds\.json$|synthetic_private_manifest\.json$|suitesparse_private_provenance\.json$')
RAW = ROOT / 'var/raw'
# var/raw entries that are task data (the rest are logs, a paper, a profiling report and agent monitors)
RAW_INCLUDE = [
    'ihvp-sequence-v1', 'ihvp-sequence-v3', 'ihvp-sequence-v4', 'ihvp-v4',
    'ihvp-sequence-statement-20261117', 'ihvp-balanced-20261117', 'ihvp-neutral-20261117',
    'galaxy-b4-eval', 'galaxy-b4-0.4.0', 'pku',
    'blackbox_cubature.zip', 'ci_3d_fluorescence_deconvolution_operator.tar.gz', 'ci_3d_fluorescence_deconvolution_public.tar.gz',
    'conic-solver-20261105.tar.gz', 'continuum-estimation-1.0.1-private.tar.gz', 'continuum-estimation-1.0.1-public.tar.gz',
    'galaxy-gravity-b4-public-0.4.0-precalibration.zip', 'galaxy-gravity-operator-1.0.1(1).tar.gz', 'galaxy-gravity-public-1.0.1.zip',
    'polarization-demosaicking-maintenance-v1.2.tar.gz',
    'polarization-demosaicking-maintenance-vnext-1.3.0-dev.tar.gz', 'polarization-demosaicking-v1.2.tar.gz',
    'polarization-demosaicking-vnext-1.3.0-dev.tar.gz', 'species-tree-operator-v1.0.1.zip', 'species-tree-operator-v1.zip',
    'species-tree-public-v1(1).zip', 'trajectory-covariance-public-v1(1).tar.gz',
]
# large groups added only on request (--with): the set-aside IHVP v5 real-trace data (21 GiB, plain Git)
OPTIONAL = {'ihvp-v5': ['ihvp-sequence-v5', 'ihvp-v5-data']}
# archives whose members are all present extracted next to them (checked 2026-11-23 by name and size)
EXTRACTED = {
    'ihvp-v4/ihvp-hard-v4-with-guide.zip': 'raw/ihvp-v4, raw/ihvp-sequence-v4',
    'ihvp-sequence-v3/bundle/ihvp-hard-v3-with-guide.zip': 'raw/ihvp-sequence-v3',
    'ihvp-sequence-v1/bundle/ihvp-sequence-v1.zip': 'raw/ihvp-sequence-v1',
    'ihvp-v5-data/EVALUATOR_ONLY_ihvp_balanced_15public_30private_statement_20261117.zip': 'raw/ihvp-sequence-v5',
    'pku/physics/galaxy-gravity-b4-evaluation-0.4.0-precalibration.tar.zst': 'raw/galaxy-b4-eval (macOS ._ metadata entries omitted)',
}
SECRET = re.compile(rb'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9-]{20,}|tai_pat_[A-Za-z0-9]{10,}'
                    rb'|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----')


def git(*args, cwd=STAGE, check=True, quiet=False):
    result = subprocess.run(['git', *args], cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise SystemExit(f'git {" ".join(args[:3])} failed: {result.stderr.strip()[-600:]}')
    if not quiet and result.stdout.strip():
        print(result.stdout.strip()[-400:], flush=True)
    return result.stdout


def sources(extra: tuple[str, ...] = ()) -> list[tuple[Path, str]]:
    """(source file, destination path in the backup) for every file to keep; symlinked entries are skipped."""
    pairs = []
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True).stdout.decode().split('\0')
    pairs += [(ROOT / p, p) for p in tracked if p and TRACKED.search('/' + p)]
    for entry in [*RAW_INCLUDE, *(e for group in extra for e in OPTIONAL[group])]:
        top = RAW / entry
        if top.is_symlink():
            continue
        files = [top] if top.is_file() else sorted(p for p in top.rglob('*') if p.is_file() and not p.is_symlink())
        for p in files:
            rel = p.relative_to(RAW).as_posix()
            if rel not in EXTRACTED and not p.name.startswith('._'):
                pairs.append((p, 'raw/' + rel))
    return [(s, d) for s, d in pairs if s.exists()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def scan_secrets(pairs):
    hits = []
    for src, dest in pairs:
        if src.stat().st_size <= 4 << 20 and src.suffix not in {'.npy', '.npz', '.bin', '.in', '.zst', '.gz', '.zip', '.pt'}:
            try:
                if SECRET.search(src.read_bytes()):
                    hits.append(dest)
            except OSError:
                continue
    return hits


def place(src: Path, dest: Path) -> bool:
    """Hard-link src at dest; True when dest changed."""
    if dest.exists() and os.path.samefile(src, dest):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        import shutil
        shutil.copy2(src, dest)
    return True


def ensure_repo():
    if not (STAGE / '.git').exists():
        STAGE.mkdir(parents=True, exist_ok=True)
        git('init', '-q', '-b', 'main')
        git('remote', 'add', 'origin', REMOTE)
    git('remote', 'set-url', 'origin', REMOTE)
    helper = f'credential.{REMOTE.split("/IceySakura")[0]}.helper'
    git('config', '--unset-all', helper, check=False)
    git('config', '--add', helper, '')
    git('config', '--add', helper,
        f'!f() {{ test "$1" = get && echo username=x-access-token && echo "password=$(cat {TOKEN_FILE})"; }}; f')
    git('lfs', 'install', '--local', quiet=True)
    for key, value in (('user.name', git('config', 'user.name', cwd=ROOT, quiet=True).strip()),
                       ('user.email', git('config', 'user.email', cwd=ROOT, quiet=True).strip()),
                       ('core.compression', '1'), ('pack.window', '0'), ('core.bigFileThreshold', '50m'),
                       ('http.postBuffer', str(2 * 2**30)), ('http.version', 'HTTP/1.1')):
        git('config', key, value)


def push_pending(push: bool):
    """Push local commits the remote lacks one at a time, so no push carries more than one batch."""
    if not push:
        return
    known = git('ls-remote', 'origin', 'refs/heads/main', quiet=True).split()
    base = known[0] if known and subprocess.run(['git', 'cat-file', '-e', known[0]], cwd=STAGE).returncode == 0 else None
    commits = git('rev-list', '--reverse', 'HEAD' if base is None else f'{base}..HEAD', quiet=True).split()
    for sha in commits:
        git('push', '-q', 'origin', f'{sha}:refs/heads/main', quiet=True)
        print(json.dumps({'pushed': sha[:8]}), flush=True)


def commit_and_push(paths: list[str], message: str, push: bool, renormalize: bool = False):
    for i in range(0, len(paths), 500):
        git('add', *(['--renormalize'] if renormalize else []), '--', *paths[i:i + 500], quiet=True)
    if not git('diff', '--cached', '--name-only', quiet=True).strip():
        return
    git('commit', '-q', '-m', message, quiet=True)
    if push:
        git('push', '-q', 'origin', 'HEAD:main', quiet=True)
    print(json.dumps({'commit': message, 'files': len(paths), 'pushed': push}), flush=True)


LFS_ATTR = 'filter=lfs diff=lfs merge=lfs -text'


def attributes_text(paths) -> str:
    return ("# Git LFS only for files at or above GitHub's 100 MiB limit; everything smaller is plain Git "
            '(scripts/ops/databack.py).\n' + ''.join(f'/{p} {LFS_ATTR}\n' for p in sorted(paths)))


def write_attributes(paths):
    (STAGE / '.gitattributes').write_text(attributes_text(paths))


def set_lfs(big: list[str], sizes: dict, push: bool):
    """Make the LFS set exactly `big`. New LFS paths are declared before their files are committed. Paths that were LFS
    under an earlier, lower threshold go back to plain Git: the attribute line is dropped and the file re-added
    (--renormalize, since the stat cache does not notice an attribute change) in the same commit, about BATCH of new
    content per commit, so every commit checks out consistently."""
    attributes = STAGE / '.gitattributes'
    lines = attributes.read_text().splitlines() if attributes.exists() else []
    current = {l.split()[0].lstrip('/') for l in lines if l.endswith(LFS_ATTR)}
    if set(big) - current:
        current |= set(big)
        write_attributes(current)
        commit_and_push(['.gitattributes'], 'LFS for files >=100 MiB', push)
    dropped = sorted(p for p in current - set(big) if (STAGE / p).is_file())
    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(dropped, pool.map(lambda d: sha256(STAGE / d), dropped)))
    seen, batch, size = set(), [], 0
    for i, p in enumerate(dropped):
        batch.append(p)
        if digests[p] not in seen:
            seen.add(digests[p]); size += sizes.get(p, (STAGE / p).stat().st_size)
        if size >= BATCH or i == len(dropped) - 1:
            current -= set(batch)
            write_attributes(current)
            commit_and_push(['.gitattributes', *batch], f'data: {len(batch)} files under 100 MiB from Git LFS to plain Git',
                            push, renormalize=True)
            batch, size = [], 0
    if not attributes.exists() or attributes.read_text() != attributes_text(current):
        write_attributes(current)
        commit_and_push(['.gitattributes'], 'LFS for files >=100 MiB', push)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-push', action='store_true')
    parser.add_argument('--with', dest='extra', action='append', default=[], choices=sorted(OPTIONAL))
    args = parser.parse_args()
    pairs = sources(tuple(args.extra))
    sizes = {d: s.stat().st_size for s, d in pairs}
    big = sorted(d for d, n in sizes.items() if n > LFS_ABOVE)
    print(json.dumps({'files': len(pairs), 'GiB': round(sum(sizes.values()) / 2**30, 2), 'lfs_files': len(big),
                      'lfs_GiB': round(sum(sizes[d] for d in big) / 2**30, 2)}), flush=True)
    hits = scan_secrets(pairs)
    if hits:
        raise SystemExit(f'credential-like strings found, nothing staged: {hits[:5]}')
    if args.dry_run:
        return
    ensure_repo()
    push_pending(not args.no_push)
    for s, d in pairs:
        place(s, STAGE / d)
    set_lfs(big, sizes, not args.no_push)
    pending = set(git('ls-files', '-z', '--others', '--modified', '--exclude-standard', quiet=True).split('\0'))
    changed = [d for s, d in pairs if d in pending]
    print(json.dumps({'to_commit': len(changed)}), flush=True)
    # LFS files first (their upload does not count against the 2 GB pack limit), then plain files in ~1.0 GiB batches;
    # content seen before (by SHA-256) adds no new pack data
    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(changed, pool.map(lambda d: sha256(STAGE / d), changed)))
    bigset = set(big)
    lfs_changed = [d for d in changed if d in bigset]
    if lfs_changed:
        commit_and_push(lfs_changed, f'data: {len(lfs_changed)} files >=100 MiB (Git LFS)', not args.no_push)
    seen, batch, size, n = set(), [], 0, 0
    for d in sorted(d for d in changed if d not in bigset):
        batch.append(d)
        if digests[d] not in seen:
            seen.add(digests[d]); size += sizes[d]
        if size >= BATCH:
            n += 1
            commit_and_push(batch, f'data: batch {n} ({batch[0].split("/")[0]}/{batch[0].split("/")[1]} ... {batch[-1].split("/")[1]})', not args.no_push)
            batch, size = [], 0
    if batch:
        n += 1
        commit_and_push(batch, f'data: batch {n}', not args.no_push)
    # manifest of everything in the backup, and the archives left out
    all_digests = {}
    with ThreadPoolExecutor(8) as pool:
        for d, h in zip(sizes, pool.map(lambda d: digests.get(d) or sha256(STAGE / d), sizes)):
            all_digests[d] = h
    (STAGE / 'MANIFEST.sha256').write_text(''.join(f'{all_digests[d]}  {d}\n' for d in sorted(all_digests)))
    rows = []
    for rel, where in EXTRACTED.items():
        p = RAW / rel
        if p.exists():
            rows.append(f'| `raw/{rel}` | {p.stat().st_size / 2**30:.2f} GiB | `{sha256(p)}` | {where} |')
    (STAGE / 'EXCLUDED.md').write_text(
        '# Archives not stored\n\nEvery member of these archives is stored extracted (checked by name and size on '
        '2026-11-23), so the archives themselves are left out to keep large files out of Git LFS.\n\n'
        '| archive | size | SHA-256 | members stored under |\n| --- | --- | --- | --- |\n' + '\n'.join(rows) + '\n')
    (STAGE / 'README.md').write_text(
        '# FrontierAM data backup (private)\n\nPublic-feedback and held-out evaluation data of '
        '[FrontierAM](https://github.com/IceySakura/FrontierAM), which its GitHub mirror leaves out. Do not make this '
        'repository public and do not give it to candidate agents.\n\n'
        '- `tasks/...`: the files under `tasks/*/versions/*/data/{public,private}` (and the other private paths that '
        '`scripts/ops/sync_github.sh` removes), at their paths in the main repository.\n'
        '- `raw/...`: raw task data from `var/raw` (author packages and the IHVP real-trace data), which is not in Git.\n'
        f'- Files >=100 MiB use Git LFS (`.gitattributes`, {len(big)} files); everything else is plain Git.\n'
        '- `MANIFEST.sha256`: SHA-256 of every file (`sha256sum -c MANIFEST.sha256`). `EXCLUDED.md`: archives left out '
        'because their contents are stored extracted.\n\n'
        'Restore: copy `tasks/` over a checkout of the main repository and `raw/` to `var/raw/`, after `git lfs pull` '
        'for the LFS files. Updated by `scripts/ops/databack.py` in the main repository.\n')
    commit_and_push(['MANIFEST.sha256', 'EXCLUDED.md', 'README.md'], 'manifest, excluded archives, README', not args.no_push)


if __name__ == '__main__':
    sys.exit(main())
