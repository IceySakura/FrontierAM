"""Run the species-tree package batch grader (sealed v2 copy, 250 taxa) on species-tree-v2 cases.

DATA is a directory with manifest.json (cases with input/truth paths relative to DATA and their sha256); every file
is checked against the manifest before grading. --cases restricts the run to one shard. The package's 0/5/10 points
are ignored (dummy anchors); the caller scores the missed-split total against references measured on this machine.

Usage: species_tree_v2_batch.py OPERATOR_DIR SOURCE OUTPUT DATA [--cases IDS.json]
"""

import argparse
import hashlib
import json
import sys

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operator', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--cases', type=Path)
    args = parser.parse_args()
    operator, source, out, data = (p.resolve() for p in (args.operator, args.source, args.output, args.data))
    sys.path.insert(0, str(operator))
    import grade  # the sealed v2 copy of the package grader

    manifest = json.loads((data / 'manifest.json').read_text())
    wanted = set(json.loads(args.cases.read_text())) if args.cases else None
    rows = []
    for case in manifest['cases']:
        if wanted is not None and case['case'] not in wanted:
            continue
        row = {'case': case['case']}
        for field in ('input', 'truth'):
            path = data / case[field]
            if hashlib.sha256(path.read_bytes()).hexdigest() != case[field + '_sha256']:
                raise SystemExit(f'{case["case"]}: {field} does not match the sealed manifest')
            row[field], row[field + '_sha256'] = str(path), case[field + '_sha256']
        rows.append(row)
    if wanted is not None and len(rows) != len(wanted):
        raise SystemExit('--cases names cases outside the manifest')
    out.mkdir(parents=True, exist_ok=False)
    splits = json.loads((operator.parent.parent.parent / 'lock.json').read_text())['taxa'] - 3
    result = grade.batch(source, out / 'batch', rows, len(rows) * splits, 1)
    result = json.loads(json.dumps(result, default=str))
    result['scope'] = data.name
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}))


if __name__ == '__main__':
    main()
