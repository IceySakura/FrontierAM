"""Build the blackbox-cubature v3 prototype instance set (milestone M1): the hard synthetic track (H1-H3) and the option
track, each instance with every available reference route.

Seeds live in a namespace (`--namespace`, default 3 = prototype); the public, feedback and private sets of M3 use other
namespaces, and the private ones are kept off GitHub. Rotation seeds are drawn from the same stream.

Reference summary per instance: value and route of the primary (first) route; `uncertainty` is 3 SE for an RQMC
primary, otherwise the larger of its claimed tolerance and its distance to the most precise other deterministic route;
`e_floor` = max(-12, log10(10 * uncertainty / |I|)) is the lowest log10 relative error the grader can resolve.

Usage: generate.py OUT_DIR [--workers 6] [--rqmc-m 20] [--namespace 3] [--only FAMILY ... [--merge]]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import options  # noqa: E402
import synthetic  # noqa: E402

HARD_DIMS = (4, 8, 16, 32, 64)
OPTION_CONFIGS = (('exchange', 2, 1), ('spread', 2, 1),
                  ('rainbow_max_call', 2, 1), ('rainbow_max_call', 3, 1),
                  ('rainbow_min_call', 2, 1), ('rainbow_min_call', 3, 1),
                  ('digital_max', 2, 1), ('digital_max', 3, 1),
                  ('geo_basket_call', 4, 1), ('geo_basket_call', 8, 1),
                  ('basket_call', 2, 1), ('basket_call', 3, 1), ('basket_call', 4, 1), ('basket_call', 8, 1),
                  ('asian_call', 1, 16), ('asian_call', 1, 64),
                  ('basket_asian_call', 2, 8), ('basket_asian_call', 4, 8), ('basket_asian_call', 4, 16),
                  ('asian_spread', 2, 8), ('asian_spread', 2, 32))
SEEDS = (0, 1)


def plan(only: set | None) -> list[dict]:
    tasks = []
    for fi, fam in enumerate(synthetic.FAMILIES):
        for d in HARD_DIMS:
            for s in SEEDS:
                tasks.append({'id': f'{fam}-d{d}-s{s}', 'track': 'hard', 'family': fam, 'L': None, 'm': None, 'd': d,
                              'stream': [1, fi, d, s]})
    for fam, L, m in OPTION_CONFIGS:
        fi = options.FAMILIES.index(fam)
        for s in SEEDS:
            tasks.append({'id': f'{fam}-L{L}-m{m}-s{s}', 'track': 'options', 'family': fam, 'L': L, 'm': m,
                          'd': L * m, 'stream': [2, fi, L, m, s]})
    if only:
        tasks = [t for t in tasks if t['family'] in only]
    return tasks


def _det_route(method: str, res: dict) -> dict:
    return {'method': method, 'value': float(res['value']), 'tol': max(float(res['abs_err']), 1e-13), 'se': None,
            'derived': False}


def summarize(routes: list[dict]) -> dict:
    primary = routes[0]
    v = primary['value']
    if primary['se'] is not None:
        unc = 3.0 * primary['se']
    else:
        others = [r for r in routes[1:] if r['se'] is None and not r['derived']]
        best = min(others, key=lambda r: r['tol']) if others else None
        unc = max(primary['tol'], abs(best['value'] - v) if best else 0.0)
    rel = unc / abs(v)
    return {'value': v, 'uncertainty': unc, 'rel_uncertainty': rel,
            'e_floor': round(max(-12.0, math.log10(10.0 * rel)), 2), 'method': primary['method']}


def build(task: dict, namespace: int, rqmc_m: int) -> dict:
    warnings.filterwarnings('ignore')
    t0 = time.time()
    rng = np.random.default_rng([namespace, *task['stream']])
    if task['track'] == 'hard':
        spec = synthetic.sample(task['family'], task['d'], rng)
        spec['rotation_seed'] = int(rng.integers(2 ** 31))
        a, b = synthetic.reference(spec), synthetic.reference_swapped(spec)
        routes = [_det_route(a['method'], a), _det_route(b['method'], b)]
    else:
        spec = options.sample(task['family'], rng, L=task['L'], m=task['m'])
        spec['rotation_seed'] = int(rng.integers(2 ** 31))
        routes = options.references(spec, rqmc_m=rqmc_m)
    out = {k: task[k] for k in ('id', 'track', 'family', 'L', 'm', 'd')}
    out.update({'spec': spec, 'reference': summarize(routes), 'routes': routes,
                'seconds': round(time.time() - t0, 1)})
    return out


def cost(task: dict) -> float:
    return task['d'] * (task['m'] or 1) if task['track'] == 'options' else task['d'] / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rqmc-m', type=int, default=20)
    ap.add_argument('--namespace', type=int, default=3)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--merge', action='store_true', help='replace only the rebuilt instances in an existing set')
    args = ap.parse_args()
    tasks = sorted(plan(set(args.only) if args.only else None), key=cost, reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build, t, args.namespace, args.rqmc_m): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            ref = res['reference']
            print(f"{res['id']:28s} d={res['d']:2d} I={ref['value']:.12g} rel_unc={ref['rel_uncertainty']:.1e} "
                  f"({res['seconds']:.0f}s)", flush=True)
            results.append(res)
    path = args.out / 'instances.json'
    if args.merge and path.exists():
        rebuilt = {r['id'] for r in results}
        results += [r for r in json.loads(path.read_text())['instances'] if r['id'] not in rebuilt]
    order = {t['id']: i for i, t in enumerate(plan(None))}
    results.sort(key=lambda r: order[r['id']])
    path.write_text(json.dumps(
        {'generator': 'tasks/blackbox-cubature/versions/v3/generation/generate.py', 'namespace': args.namespace,
         'rqmc_m': args.rqmc_m, 'instances': results}, indent=1) + '\n')
    print(f'wrote {len(results)} instances to {args.out / "instances.json"}')


if __name__ == '__main__':
    main()
