"""Build blackbox-cubature v3 instance sets: the hard synthetic track (H1, H2, H3, H3t of synthetic.py) and the option
track (literature path-dependent families of options.py and paths.py, DUAL_PLAN section 9.3), each instance with every
available reference route.

Sets (`--set`):
  prototype  every configuration x SEEDS, all option families (namespace 3, the M1/M2 development set);
  samples    given to the models with their reference values: each synthetic family at d = 8 and 32, each formal
             option family in the configuration the public set does not use (namespace 4);
  public     the hidden feedback set: one synthetic instance per (family, d), one per formal option family;
  private    the formal set: every synthetic configuration and every formal option configuration x 2 seeds.
The public and private sets draw a random 62-bit namespace (recorded only in their instances.json, which stays off
GitHub), so the generator code does not reveal them. Rotation and protocol seeds come from the same stream.

Reference summary per instance: value and route of the primary (first) route; `uncertainty` is 3 SE for an RQMC
primary, otherwise the larger of its claimed tolerance and its distance to the most precise other deterministic route;
`e_floor` = max(-12, log10(10 * uncertainty / |I|)) is the lowest log10 relative error the grader can resolve.
`--redraw ID=R ...` replaces an instance by the R-th redraw of its stream (instances failing the precision rule).

Usage: generate.py OUT_DIR [--set prototype] [--workers 6] [--rqmc-m 20] [--namespace N] [--only FAMILY ...]
                   [--redraw ID=R ...] [--merge]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import options  # noqa: E402
import paths  # noqa: E402
import synthetic  # noqa: E402

HARD_DIMS = (8, 16, 32, 64)
OPTION_CONFIGS = (('asian_call', 1, 32), ('asian_call', 1, 64),
                  ('basket_asian_call', 2, 16), ('basket_asian_call', 2, 32),
                  ('digital_asian', 1, 32), ('digital_asian', 1, 64),
                  ('rainbow_asian_max', 2, 16), ('rainbow_asian_max', 2, 32),
                  ('rainbow_asian_min', 2, 16), ('rainbow_asian_min', 2, 32),
                  ('barrier_do_call', 1, 16), ('barrier_do_call', 1, 64),
                  ('barrier_dko_call', 1, 16), ('barrier_dko_call', 1, 64),
                  ('heston_asian', 1, 16), ('heston_asian', 1, 32))
OPTION_FAMILIES = ('asian_call', 'basket_asian_call') + paths.FAMILIES
# Formal option families: rainbow_asian_max and heston_asian stay out (their references resolve less than two orders
# below the strong reference, DUAL_PLAN 9.7), as does barrier_asian_do.
FORMAL_OPTION_FAMILIES = ('asian_call', 'basket_asian_call', 'digital_asian', 'rainbow_asian_min', 'barrier_do_call',
                          'barrier_dko_call')
# Families whose RQMC references need more points to resolve two orders below the strong reference.
RQMC_M = {'basket_asian_call': 23, 'rainbow_asian_min': 23}
SEEDS = (0, 1)
SAMPLE_NAMESPACE = 4


def _synthetic(fam: str, d: int, k: int, s: int | None) -> dict:
    fi = synthetic.FAMILIES.index(fam)
    suffix, stream = ('', [1, fi, d, k]) if s is None else (f'-s{s}', [1, fi, d, k, s])
    return {'id': f'{fam}-d{d}-k{k}{suffix}', 'track': 'hard', 'family': fam, 'L': None, 'm': None, 'd': d, 'k': k,
            'stream': stream}


def _option(fam: str, L: int, m: int, s: int) -> dict:
    d = 2 * m if fam == 'heston_asian' else L * m
    return {'id': f'{fam}-L{L}-m{m}-s{s}', 'track': 'options', 'family': fam, 'L': L, 'm': m, 'd': d, 'k': None,
            'stream': [2, OPTION_FAMILIES.index(fam), L, m, s]}


def plan(only: set | None, which: str = 'prototype') -> list[dict]:
    tasks = []
    configs = {fam: [(L, m) for f, L, m in OPTION_CONFIGS if f == fam] for fam in OPTION_FAMILIES}
    if which == 'prototype':
        tasks += [_synthetic(fam, d, k, None) for fam in synthetic.FAMILIES for d in HARD_DIMS
                  for k in synthetic.k_levels(fam, d)]
        tasks += [_option(fam, L, m, s) for fam, L, m in OPTION_CONFIGS for s in SEEDS]
    elif which == 'private':
        tasks += [_synthetic(fam, d, k, s) for fam in synthetic.FAMILIES for d in HARD_DIMS
                  for k in synthetic.k_levels(fam, d) for s in SEEDS]
        tasks += [_option(fam, L, m, s) for fam in FORMAL_OPTION_FAMILIES for L, m in configs[fam] for s in SEEDS]
    elif which == 'public':
        for fi, fam in enumerate(synthetic.FAMILIES):
            for di, d in enumerate(HARD_DIMS):
                levels = synthetic.k_levels(fam, d)
                tasks.append(_synthetic(fam, d, levels[(fi + di) % len(levels)], 0))
        tasks += [_option(fam, *configs[fam][fi % 2], 0) for fi, fam in enumerate(FORMAL_OPTION_FAMILIES)]
    elif which == 'samples':
        for fi, fam in enumerate(synthetic.FAMILIES):
            for di, d in enumerate((8, 32)):
                levels = synthetic.k_levels(fam, d)
                tasks.append(_synthetic(fam, d, levels[(fi + 2 * di) % len(levels)], 0))
        tasks += [_option(fam, *configs[fam][(fi + 1) % 2], 0) for fi, fam in enumerate(FORMAL_OPTION_FAMILIES)]
    else:
        raise ValueError(which)
    if which != 'prototype':  # set-qualified IDs: the samples and public sets reuse configurations and seed 0
        tasks = [{**t, 'id': f'{which}/{t["id"]}'} for t in tasks]
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


def build(task: dict, namespace: int, rqmc_m: int, redraw: int = 0) -> dict:
    warnings.filterwarnings('ignore')
    t0 = time.time()
    rng = np.random.default_rng([namespace, *task['stream'], *([redraw] if redraw else [])])
    fam = task['family']
    if task['track'] == 'hard':
        spec = synthetic.sample(fam, task['d'], task['k'], rng)
        spec['rotation_seed'] = int(rng.integers(2 ** 31))
        a, b = synthetic.reference(spec), synthetic.reference_swapped(spec)
        routes = [_det_route(a['method'], a), _det_route(b['method'], b)]
    elif fam in paths.FAMILIES:
        spec = paths.sample(fam, rng, L=task['L'], m=task['m'])
        spec['rotation_seed'] = int(rng.integers(2 ** 31))
        routes = paths.references(spec, rqmc_m=RQMC_M.get(fam, rqmc_m))
    else:
        spec = options.sample_hard(fam, rng, L=task['L'], m=task['m'])
        spec['rotation_seed'] = int(rng.integers(2 ** 31))
        routes = options.references(spec, rqmc_m=RQMC_M.get(fam, rqmc_m))
    out = {k: task[k] for k in ('id', 'track', 'family', 'L', 'm', 'd', 'k')}
    # the seed handed to the program in the protocol's init message, for its own randomization
    out.update({'seed': int(rng.integers(2 ** 31)), 'redraw': redraw, 'spec': spec, 'reference': summarize(routes),
                'routes': routes, 'seconds': round(time.time() - t0, 1)})
    return out


def cost(task: dict) -> float:
    if task['track'] != 'options':
        return task['d'] / 64
    return task['d'] * (task['m'] or 1) * 2 ** (RQMC_M.get(task['family'], 20) - 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--set', default='prototype', choices=('prototype', 'samples', 'public', 'private'))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rqmc-m', type=int, default=20)
    ap.add_argument('--namespace', type=int, default=None,
                    help='default: 3 (prototype), 4 (samples), the existing or a fresh random one (public, private)')
    ap.add_argument('--namespace-file', type=Path, default=None,
                    help='read the namespace from this file, creating it (mode 0600, random) if missing; lets parts '
                         'of one set run in parallel without the namespace on any command line')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--match', nargs='*', help='keep instances whose ID contains one of these substrings')
    ap.add_argument('--redraw', nargs='*', default=[], help='ID=R: use the R-th redraw of that instance')
    ap.add_argument('--merge', action='store_true', help='replace only the rebuilt instances in an existing set')
    args = ap.parse_args()
    path = args.out / 'instances.json'
    old = json.loads(path.read_text()) if args.merge and path.exists() else None
    namespace = args.namespace
    if args.namespace_file is not None:
        if not args.namespace_file.exists():
            args.namespace_file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(args.namespace_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, f'{secrets.randbits(62)}\n'.encode())
            os.close(fd)
        namespace = int(args.namespace_file.read_text())
    if namespace is None:
        namespace = (old['namespace'] if old else {'prototype': 3, 'samples': SAMPLE_NAMESPACE}.get(
            args.set, secrets.randbits(62)))
    if old and old['namespace'] != namespace:
        raise SystemExit('--merge into a set built in another namespace')
    redraw = {**(old or {}).get('redraws', {}), **{k: int(v) for k, v in (r.split('=') for r in args.redraw)}}
    tasks = plan(set(args.only) if args.only else None, args.set)
    if args.match:
        tasks = [t for t in tasks if any(s in t['id'] for s in args.match)]
    if args.redraw:
        tasks = [t for t in tasks if t['id'] in {r.split('=')[0] for r in args.redraw}]
    tasks.sort(key=cost, reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build, t, namespace, args.rqmc_m, redraw.get(t['id'], 0)): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            ref = res['reference']
            print(f"{res['id']:28s} d={res['d']:2d} I={ref['value']:.12g} rel_unc={ref['rel_uncertainty']:.1e} "
                  f"({res['seconds']:.0f}s)", flush=True)
            results.append(res)
    if old:
        rebuilt = {r['id'] for r in results}
        results += [r for r in old['instances'] if r['id'] not in rebuilt]
    order = {t['id']: i for i, t in enumerate(plan(None, args.set))}
    results.sort(key=lambda r: order[r['id']])
    path.write_text(json.dumps(
        {'generator': 'tasks/blackbox-cubature/versions/v3/generation/generate.py', 'set': args.set,
         'namespace': namespace, 'rqmc_m': args.rqmc_m, 'rqmc_m_by_family': RQMC_M,
         'redraws': {k: v for k, v in redraw.items() if v}, 'instances': results}, indent=1) + '\n')
    print(f'wrote {len(results)} instances to {path}')


if __name__ == '__main__':
    main()
