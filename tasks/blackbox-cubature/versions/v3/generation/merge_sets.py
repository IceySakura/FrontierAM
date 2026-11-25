"""Assemble data/<set>/instances.json from the parts launch_sets.sh built under var/raw/blackbox-cubature-v3/gen/.

Every part of a set must share its namespace; the merged set must hold exactly the instances generate.plan lists for
it (in that order), each once. A later part replaces an earlier one's instance with the same ID (a redraw).

Usage: merge_sets.py SET [PART_DIR ...]   (default parts: every gen/<SET>* directory, in name order)
"""
import json
import sys
from pathlib import Path

import generate

ROOT = Path(__file__).resolve().parents[5]
GEN = ROOT / 'var/raw/blackbox-cubature-v3/gen'
DATA = Path(__file__).resolve().parents[1] / 'data'


def main():
    which = sys.argv[1]
    parts = [Path(p) for p in sys.argv[2:]] or sorted(p for p in GEN.glob(f'{which}*') if p.is_dir())
    merged, namespace, redraws, meta = {}, None, {}, {}
    for part in parts:
        data = json.loads((part / 'instances.json').read_text())
        if data['set'] != which:
            raise SystemExit(f'{part}: set {data["set"]}, expected {which}')
        if namespace is not None and data['namespace'] != namespace:
            raise SystemExit(f'{part}: namespace differs from the other parts')
        namespace = data['namespace']
        redraws.update(data.get('redraws', {}))
        meta = {k: data[k] for k in ('generator', 'rqmc_m', 'rqmc_m_by_family')}
        merged.update({inst['id']: inst for inst in data['instances']})
    planned = [t['id'] for t in generate.plan(None, which)]
    missing, extra = set(planned) - set(merged), set(merged) - set(planned)
    if missing or extra:
        raise SystemExit(f'{which}: missing {sorted(missing)[:5]} ({len(missing)}), unexpected {sorted(extra)[:5]}')
    out = DATA / which
    out.mkdir(parents=True, exist_ok=True)
    (out / 'instances.json').write_text(json.dumps(
        {**meta, 'set': which, 'namespace': namespace, 'redraws': redraws,
         'parts': [str(p.relative_to(ROOT)) for p in parts], 'instances': [merged[i] for i in planned]}, indent=1) + '\n')
    floors = sorted(merged[i]['reference']['e_floor'] for i in planned)
    print(f'{which}: {len(planned)} instances from {len(parts)} parts -> {out / "instances.json"}; '
          f'e_floor {floors[-1]:.2f} (worst) to {floors[0]:.2f}')


if __name__ == '__main__':
    main()
