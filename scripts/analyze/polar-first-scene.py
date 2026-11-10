"""Why is the first polarization scene slow? Grade the same submission three times on the grade slots:
as things stand, right after evicting only the task image's layer files from the page cache (posix_fadvise
DONTNEED on the four grade daemons' copies), and once more straight after. Each run reports the first scene's wall
time against the rest and how much of the image's numpy/python files were resident before it started.

Usage: polar-first-scene.py SUBMISSION OUT.json
"""

import json
import os
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'frontieram-polarization-demosaicking:amd64-b1-v1.2'
SLOTS = ('grade-06', 'grade-14', 'grade-22', 'grade-30')


def layer_dirs() -> list[Path]:
    dirs = []
    for slot in SLOTS:
        out = subprocess.run(['docker', '-H', f'unix:///run/frontieram-grade/{slot}/docker.sock', 'image', 'inspect',
                              IMAGE, '--format', '{{index .GraphDriver.Data "LowerDir"}}:{{index .GraphDriver.Data "UpperDir"}}'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirs += [Path(p) for p in out.split(':') if p]
    return dirs


def files(dirs):
    for d in dirs:
        for base, _, names in os.walk(d):
            for name in names:
                path = Path(base) / name
                if path.is_file() and not path.is_symlink():
                    yield path


def evict(dirs) -> int:
    count = 0
    for path in files(dirs):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        finally:
            os.close(fd)
    return count


def resident(dirs) -> dict:
    """Resident fraction of the image's numpy and libpython files (fincore from util-linux)."""
    targets = [str(p) for p in files(dirs) if 'numpy' in p.parts and p.suffix == '.so' or p.name.startswith('libpython')]
    if not targets:
        return {}
    out = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES,SIZE', *targets],
                         capture_output=True, text=True).stdout.split()
    res, size = sum(map(int, out[0::2])), sum(map(int, out[1::2]))
    return {'files': len(targets), 'resident_bytes': res, 'bytes': size, 'fraction': res / size if size else None}


def grade(submission: str) -> dict:
    env = dict(os.environ, FRONTIERAM_RUNTIME=str(ROOT / '.runtime-recal'))
    out = subprocess.run([str(ROOT / 'bench'), '--store', 'var/stores/store', 'grade', '--submission', submission,
                          '--task', 'polarization-demosaicking'], cwd=ROOT, env=env, capture_output=True, text=True)
    graded = [json.loads(l) for l in out.stdout.splitlines() if l.startswith('{') and '"graded"' in l]
    if not graded:
        raise SystemExit(out.stdout[-2000:] + out.stderr[-2000:])
    job = graded[-1]['job']
    summary = json.loads((ROOT / 'var/stores/store/results' / job / 'grade/candidate/summary.json').read_text())
    walls = [r['wall_s'] for r in summary['scene_rows']]
    slot = json.loads((ROOT / 'var/stores/store/results' / job / 'result.json').read_text())['slot']
    rest = sorted(walls[1:])
    return {'job': job, 'slot': slot, 'status': summary.get('status'), 'first_s': walls[0],
            'rest_median_s': rest[len(rest) // 2], 'rest_max_s': rest[-1], 'score': graded[-1].get('score')}


def main():
    submission, out = sys.argv[1], Path(sys.argv[2])
    dirs = layer_dirs()
    runs = []
    runs.append({'condition': 'as_is', 'resident_before': resident(dirs), **grade(submission)})
    evicted = evict(dirs)
    runs.append({'condition': 'after_evicting_image_files', 'evicted_files': evicted,
                 'resident_before': resident(dirs), **grade(submission)})
    runs.append({'condition': 'immediately_again', 'resident_before': resident(dirs), **grade(submission)})
    out.write_text(json.dumps({'submission': submission, 'runs': runs}, indent=1) + '\n')
    print(json.dumps(runs, indent=1))


if __name__ == '__main__':
    main()
