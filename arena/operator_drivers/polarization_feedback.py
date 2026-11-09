"""Grade a polarization-demosaicking submission on the public-feedback scenes (protocol v6.2).

Reuses the package grader's own build, sandboxed per-scene run and scorer (runner.build_submission, runner.run_scene,
scoring.score_prediction, scoring.mosaic); only the scene source differs: DATA/manifest.json lists uint8 references of
shape (4, 1024, 1024) in 0/45/90/135 order (Wen et al. 2019 CPDNet scenes, CC BY-NC, cropped and converted as in
results/POLARIZATION_PUBLIC_DEV_REQUEST.md), each checked against its sha256. Unlike the formal batch, every scene is
reported even after a failure, so feedback covers the whole set.

Usage: polarization_feedback.py GRADER_DIR SOURCE OUTPUT DATA [--cases IDS.json]
"""

import argparse
import hashlib
import json
import os
import sys

from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('grader', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--cases', type=Path)
    args = parser.parse_args()
    grader, source, out, data = (p.resolve() for p in (args.grader, args.source, args.output, args.data))
    sys.path.insert(0, str(grader))
    os.chdir(grader)
    from runner import SCENE_LIMIT_S, build_submission, run_scene
    from scoring import InvalidPrediction, mosaic, score_prediction

    manifest = json.loads((data / 'manifest.json').read_text())
    wanted = set(json.loads(args.cases.read_text())) if args.cases else None
    scenes = [s for s in manifest['scenes'] if wanted is None or s['scene'] in wanted]
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    summary = {'scope': 'public-feedback', 'scene_limit_s': SCENE_LIMIT_S, 'expected_cases': len(scenes), 'scene_rows': rows}
    build = build_submission(source, out / 'build')
    summary['build'] = build
    if build['status'] != 'ok':
        summary.update(status=build['status'], reason=build.get('reason'))
    else:
        for scene in scenes:
            path = data / scene['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != scene['sha256']:
                raise SystemExit(f'{scene["scene"]}: reference does not match the sealed manifest')
            reference = np.load(path, allow_pickle=False)
            case = out / scene['scene']
            case.mkdir()
            with (case / 'raw.npy').open('xb') as stream:
                np.save(stream, mosaic(reference), allow_pickle=False)
            row = dict(scene=scene['scene'], **run_scene(out / 'build', case / 'raw.npy', case / 'worker'))
            if row['status'] == 'ok':
                try:
                    row.update(score_prediction(case / 'worker' / 'pred.npy', reference))
                except InvalidPrediction as exc:
                    row.update(status='submission_failure', reason=str(exc))
            rows.append(row)
            if row['status'] == 'infrastructure_failure':
                summary.update(status='infrastructure_failure', reason=row.get('reason'))
                break
        else:
            ok = [r for r in rows if r['status'] == 'ok']
            summary.update(status='ok' if len(ok) == len(rows) else 'submission_failure', cases_passed=len(ok),
                           mean_rmse=sum(r['mean_rmse'] for r in ok) / len(ok) if ok else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}))


if __name__ == '__main__':
    main()
