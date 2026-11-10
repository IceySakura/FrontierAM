"""Run sealed panel members once, in the evaluator image on a dev slot, on the public smoke cases (functional check).

Usage: exp2-container-check.py OUT MEMBER...
"""

import json
import sys

from pathlib import Path

from arena.cli import context, placement_for
from arena.exponential_v2 import SEAL
from arena.tasks import TASKS, tar_directory

SMOKE = Path('tasks/exponential/versions/v2/dataset/cases/public_smoke').resolve()


def main():
    out, members = Path(sys.argv[1]).resolve(), sys.argv[2:]
    layout, broker, runtime = context()
    task = TASKS['exponential-v2']
    task.verify()
    manifest = json.loads((SMOKE / 'manifest.json').read_text())
    manifest['cases'] = [dict(c, file=str(SMOKE / c['file'])) for c in manifest['cases']]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    c = task.assets(runtime).controller
    with broker.acquire('dev', task.dev_memory, 'exp2 panel container check', timeout=3600) as lease:
        placement = placement_for(lease, layout)
        for member in members:
            run = out / member
            run.mkdir()
            c.validate_and_extract(tar_directory(SEAL / 'panel' / member, run / 'incoming.tar'), run / 'source', run / 'source.tar')
            part = task.evaluate(runtime, placement, run, out / 'manifest.json')
            if part.get('report'):
                aggregates = json.loads(Path(part['report']).read_text())['aggregates']
                part['invalid'] = {k: a.get('reason') for k, a in aggregates.items() if not a['valid']}
            print(json.dumps({'member': member, 'slot': lease.id, **part}), flush=True)


if __name__ == '__main__':
    main()
