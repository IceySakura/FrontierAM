"""Grading of fluorescence-deconvolution with the author's runtime (ci_3d_fluorescence_deconvolution 1.1.0, ported).

Formal: runtime/grade.py grade() on the 96 private cases without the author's anchors, so it reports the raw mean
nMSE; the 0- and 50-point anchors are the references rerun on this machine. Public feedback (--public PUBLIC_DEV): the
12 public examples through the same build_submission / run_case / nmse / aggregate (grade() accepts only the formal
manifest). Every case runs in the author's per-case container; the docker shim on PATH (arena/docker_shim.py) maps its
arm64 platform, CPU 0 and image ID onto the grading slot and the rebuilt amd64 image.

Usage: fluorescence.py RUNTIME DATASET SUBMISSION OUT [--public PUBLIC_DEV]
"""

import argparse
import json
import shutil
import sys
import tempfile

from pathlib import Path


def public_grade(g, submission: Path, public: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((public / 'manifest.json').read_text())
    rows = []
    build = g.build_submission(submission, out / 'build')
    if build['status'] == 'ok':
        with tempfile.TemporaryDirectory(prefix='deconv-public-build-', dir='/tmp') as temp:
            staged = Path(temp) / 'build'
            shutil.copytree(out / 'build', staged)
            for case in manifest['cases']:
                inp, truth = public / case['observed'], public / case['truth']
                if g.digest(inp) != case['observed_sha256'] or g.digest(truth) != case['truth_sha256']:
                    raise RuntimeError('public data hash mismatch')
                shape, _ = g.validate_input(inp)
                receipt = g.run_case(staged, inp, out / 'cases' / case['case'], timeout_s=g.CASE_LIMIT_S,
                                     extra_env={'TIME_BUDGET_SECONDS': str(g.CASE_LIMIT_S), 'SEED': '1'})
                row = {'case_id': case['case'], 'stratum': case['category'], **receipt}
                if row['status'] == 'ok':
                    try:
                        row['nmse'] = g.nmse(out / 'cases' / case['case'] / 'pred.npy', truth, shape)
                    except ValueError as exc:
                        row.update(status='submission_failure', reason='invalid_output', detail=str(exc))
                rows.append(row)
                if row['status'] == 'infrastructure_failure':
                    break
    summary = g.aggregate(rows, build=build, expected=len(manifest['cases']), anchors=None)
    summary.update(scope='public', build=build)
    g.write(out / 'summary.json', summary)
    g.write(out / 'records.json', rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('submission', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--public', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    import grade as g

    try:
        if args.public:
            public_grade(g, args.submission, args.public, args.out)
        else:
            summary = g.grade(args.submission, args.dataset, args.out, anchors=None)
            summary['scope'] = 'formal'
            g.write(args.out / 'summary.json', summary)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'summary.json').write_text(json.dumps({'status': 'infrastructure_failure',
                                                            'reason': f'{type(exc).__name__}: {exc}'}) + '\n')


if __name__ == '__main__':
    main()
