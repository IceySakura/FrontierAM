"""Compare a local trajectory-covariance grade with the maintainer's formal-v1 records, case by case.

The maintainer package keeps every formal case record (loss, wall and CPU time) of OBM, VAR(1) and its frozen agent
under results/.../formal-v1/runs/<method>/shard-*/attempt-*/trials/*/verifier/cases.jsonl. For the cases graded
locally this prints the largest loss difference, the distribution of wall-time ratios (local / maintainer) and the
subset totals on both machines.

Usage: trajectory-compare.py MAINTAINER METHOD LOCAL_GRADE_DIR
"""

import json
import statistics
import sys

from pathlib import Path


def maintainer_records(root: Path, method: str) -> dict:
    records = {}
    runs = root / 'results/mcmc-long-run-covariance/formal-v1/runs' / method
    for path in sorted(runs.glob('shard-*/attempt-*/trials/*/verifier/cases.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row['id']] = row  # a later attempt supersedes an earlier one
    return records


def main():
    root, method, local = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    theirs = maintainer_records(root, method)
    ours = {r['id']: r for r in json.loads((local / 'verifier/receipt.json').read_text())['cases']}
    missing = [i for i in ours if i not in theirs]
    common = [i for i in ours if i in theirs]
    diffs = [abs(ours[i]['loss'] - theirs[i]['loss']) for i in common]
    ratios = sorted(ours[i]['wall_s'] / theirs[i]['wall_s'] for i in common if theirs[i]['wall_s'] > 0)
    status = sum(ours[i]['status'] != theirs[i]['status'] for i in common)
    summary = json.loads((local / 'summary.json').read_text())
    print(json.dumps({
        'method': method, 'cases_local': len(ours), 'matched': len(common), 'missing_in_maintainer': len(missing),
        'status_disagreements': status, 'max_abs_loss_diff': max(diffs) if diffs else None,
        'mean_loss_local': statistics.fmean(ours[i]['loss'] for i in common),
        'mean_loss_maintainer': statistics.fmean(theirs[i]['loss'] for i in common),
        'wall_ratio_median': statistics.median(ratios) if ratios else None,
        'wall_ratio_p95': ratios[int(0.95 * (len(ratios) - 1))] if ratios else None,
        'subset_total_local': summary.get('raw_error'),
        'max_wall_s_local': summary.get('max_wall_s'),
    }, indent=1))


if __name__ == '__main__':
    main()
