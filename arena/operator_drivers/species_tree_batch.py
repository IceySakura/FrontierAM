"""Run species-tree's own batch grader (operator/grade.py batch) on the 50 public development cases.

The package's formal data are not shipped, so this is the public-data check: same build, worker, RF scorer and
failure rules, on cases whose truths are public. The 0/5/10 points inside the package's result use placeholder
anchors and are ignored; the caller scores the integer missed-split total against references measured the same way.

Usage: species_tree_batch.py OPERATOR_DIR SOURCE OUTPUT
"""

import hashlib
import json
import sys

from pathlib import Path
from zipfile import ZipFile


def main():
    operator, source, out = (Path(p).resolve() for p in sys.argv[1:4])
    sys.path.insert(0, str(operator))
    import grade  # the package's grader, imported from the sealed copy

    out.mkdir(parents=True, exist_ok=False)
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
    result = grade.batch(source, out / 'batch', rows, 4900, 1)
    result = json.loads(json.dumps(result, default=str))
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}))


if __name__ == '__main__':
    main()
