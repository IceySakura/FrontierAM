"""Maintainer-side totals of OBM, VAR(1) and the frozen agent on the cases a local trajectory grade uses.

Uses the same case set and aggregation as arena (TrajectoryCovariance.case_ids, mean of (model, length) cell means)
on the maintainer's formal-v1 records, so local anchors and scores can be checked against them.

Usage: trajectory-expected.py   (run from the repository root with the orchestration Python)
"""

import collections
import importlib.util
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('compare', ROOT / 'scripts/trajectory-compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

from arena.tasks import TASKS  # noqa: E402
from arena.operator_tasks import score_100  # noqa: E402


def main():
    task = TASKS['trajectory-covariance']
    root = task.root()
    ids = task.case_ids()
    cases = {c['id']: c for c in json.loads((root / 'results/mcmc-long-run-covariance/formal-v1/private/manifest.json')
                                             .read_text())['cases']}
    totals = {}
    for method in ('obm', 'var1', 'agent'):
        records = compare.maintainer_records(root, method)
        cells = collections.defaultdict(list)
        for i in ids:
            cells[(cases[i]['model'], cases[i]['length'])].append(records[i]['loss'])
        totals[method] = sum(sum(v) / len(v) for v in cells.values()) / len(cells)
    print(json.dumps({'cases': len(ids), 'raw_error': totals,
                      'agent_score_100': score_100(totals['agent'], totals['obm'], totals['var1'])}, indent=1))


if __name__ == '__main__':
    main()
