"""Verify the polarization warm-up: evict the task image's files from the page cache (as in polar-first-scene.py),
grade once with the warm-up in place, and compare the first scene with the rest. Appends to OUT.json.

Usage: polar-warmup-check.py SUBMISSION OUT.json
"""

import importlib.util
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('first_scene', ROOT / 'scripts/polar-first-scene.py')
first_scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first_scene)


def main():
    submission, out = sys.argv[1], Path(sys.argv[2])
    dirs = first_scene.layer_dirs()
    evicted = first_scene.evict(dirs)
    run = {'condition': 'after_evicting_image_files_with_warmup', 'evicted_files': evicted,
           'resident_before': first_scene.resident(dirs), **first_scene.grade(submission)}
    warmup = ROOT / 'var/stores/store/results' / run['job'] / 'grade/candidate-warmup.json'
    run['warmup'] = json.loads(warmup.read_text()) if warmup.exists() else None
    record = json.loads(out.read_text()) if out.exists() else {'submission': submission, 'runs': []}
    record['runs'].append(run)
    out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
