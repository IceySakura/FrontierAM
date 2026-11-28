"""Assemble versions/v3.6/data/<set>/instances.json: the v3.1 set (hard and option tracks, every instance unchanged)
followed by the curved track that generate.py --variant v3.6 built (curved.py).

The curved part must hold exactly the instances generate.plan_curved lists for the set and share the v3.1 set's
namespace (the curved streams start with 4, so they never meet the v3.1 streams). Keeping the v3.1 instances as they
are makes the two-track G of a v3.6 run comparable with the v3.1 to v3.5 runs.

Usage: merge_v36.py SET CURVED_PART_DIR
"""
import json
import sys
from pathlib import Path

import generate

VERSIONS = Path(__file__).resolve().parents[2]
ROOT = VERSIONS.parents[2]


def main():
    which, part = sys.argv[1], Path(sys.argv[2])
    base = json.loads((VERSIONS / 'v3.1/data' / which / 'instances.json').read_text())
    extra = json.loads((part / 'instances.json').read_text())
    if extra['set'] != which or extra['variant'] != 'v3.6':
        raise SystemExit(f'{part}: set {extra["set"]}, variant {extra["variant"]}; expected {which}, v3.6')
    if extra['namespace'] != base['namespace']:
        raise SystemExit(f'{part}: namespace differs from the v3.1 {which} set')
    planned = [t['id'] for t in generate.plan_curved(which)]
    planned = [f'{which}/{i}' for i in planned]
    got = [inst['id'] for inst in extra['instances']]
    if sorted(got) != sorted(planned):
        raise SystemExit(f'{which}: curved part has {len(got)} instances, plan {len(planned)}')
    by_id = {inst['id']: inst for inst in extra['instances']}
    out = VERSIONS / 'v3.6/data' / which
    out.mkdir(parents=True, exist_ok=True)
    merged = {'generator': 'tasks/blackbox-cubature/versions/v3/generation/merge_v36.py', 'variant': 'v3.6',
              'set': which, 'namespace': base['namespace'],
              'parts': {'v3.1': f'tasks/blackbox-cubature/versions/v3.1/data/{which}/instances.json',
                        'curved': str(part.resolve().relative_to(ROOT))},
              'redraws': {**base.get('redraws', {}), **extra.get('redraws', {})},
              'instances': base['instances'] + [by_id[i] for i in planned]}
    (out / 'instances.json').write_text(json.dumps(merged, indent=1) + '\n')
    floors = [by_id[i]['reference']['e_floor'] for i in planned]
    print(f'{which}: {len(base["instances"])} v3.1 + {len(planned)} curved instances -> {out / "instances.json"}; '
          f'curved e_floor {max(floors):.2f} (worst) to {min(floors):.2f}')


if __name__ == '__main__':
    main()
