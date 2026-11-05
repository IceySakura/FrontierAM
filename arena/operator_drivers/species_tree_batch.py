"""Run species-tree's own batch grader (operator/grade.py batch) on the formal cases, or on the public ones.

With PRIVATE_DATA (operator package 1.0.1: private-data/ with cases/01..50 and data-seal.json) the rows come from the
package's formal_cases loader, which checks the seal against activation.json and every case file against the seal.
Without it, the 50 public development cases are used (pipeline check only). The 0/5/10 points inside the package's
result use the original machine's anchors and are ignored; the caller scores the integer missed-split total against
references measured the same way on this machine.

Usage: species_tree_batch.py OPERATOR_DIR SOURCE OUTPUT [PRIVATE_DATA]
"""

import hashlib
import json
import sys

from pathlib import Path
from zipfile import ZipFile


def public_rows(grade, operator: Path, out: Path) -> list[dict]:
    archive = operator / 'task/environment/public-dev/species-tree-dev.zip'
    grade.verify(archive, grade.DEV_ZIP_SHA)
    rows = []
    with ZipFile(archive) as z:
        for cid in grade.IDS:
            row = {'case': cid}
            for field, name in (('input', 'genes.tre'), ('truth', 'truth.tre')):
                data = z.read(f'cases/{cid}/{name}')
                path = out / 'public-cases' / cid / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                row[field] = str(path)
                row[field + '_sha256'] = hashlib.sha256(data).hexdigest()
            rows.append(row)
    return rows


def main():
    operator, source, out = (Path(p).resolve() for p in sys.argv[1:4])
    private = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
    sys.path.insert(0, str(operator))
    import grade  # the package's grader, imported from the sealed copy

    out.mkdir(parents=True, exist_ok=False)
    activation = json.loads((operator / 'activation.json').read_text())
    if private is not None:
        rows, u, w = grade.formal_cases(private, private / 'data-seal.json', activation['seal_sha256'])
        scope = 'formal'
    else:
        rows, u, w = public_rows(grade, operator, out), activation['u_missed'], activation['w_missed']
        scope = 'public'
    result = grade.batch(source, out / 'batch', rows, u, w)
    result = json.loads(json.dumps(result, default=str))
    result['scope'] = scope
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}))


if __name__ == '__main__':
    main()
