"""Independent checks of the v3 prototype references (milestone M1).

For every instance in DATA_DIR/instances.json:
1. route agreement: every pair of independent routes agrees within tol_i + tol_j + 4 sqrt(se_i^2 + se_j^2);
2. black-box check: replicated scrambled-Sobol' RQMC through the rotated oracle (the function contestants query)
   agrees with the reference within 4 combined standard errors plus the reference tolerance;
3. pointwise identities behind the references. Synthetic: the exact inner y1-integral against adaptive quadrature at
   random y2, with the kinks from numpy.roots (independent of the piece logic). Options: the sum-of-lognormals form
   against the payoff, and the closed-form preintegrated value against quadrature along the line, for both directions.
Once per run: the bivariate normal CDF (Owen T), used by the two-asset closed forms, against one-dimensional quadrature.

Writes DATA_DIR/verification.json and a markdown report.
Usage: verify.py DATA_DIR REPORT_MD [--workers 7] [--oracle-m 18]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

import options  # noqa: E402
import oracle  # noqa: E402
import synthetic  # noqa: E402
from gauss import bvn_cdf, bvn_cdf_quad, pdf, rqmc_estimate  # noqa: E402

LOGNORMAL = ('basket_call', 'asian_call', 'basket_asian_call', 'asian_spread')
POINTWISE_TOL = {'inner': 1e-12, 'representation': 1e-12, 'preint': 1e-11}


def agreement(routes: list[dict]) -> dict:
    indep = [r for r in routes if not r['derived']]
    worst = {'ratio': 0.0, 'pair': None, 'diff': 0.0}
    for i in range(len(indep)):
        for j in range(i + 1, len(indep)):
            a, b = indep[i], indep[j]
            allowed = a['tol'] + b['tol'] + 4.0 * math.hypot(a['se'] or 0.0, b['se'] or 0.0)
            diff = abs(a['value'] - b['value'])
            if diff / allowed >= worst['ratio']:
                worst = {'ratio': diff / allowed, 'pair': [a['method'], b['method']], 'diff': diff}
    return worst


def oracle_check(inst: dict, m: int, replicates: int = 16) -> dict:
    spec, ref, primary = inst['spec'], inst['reference'], inst['routes'][0]
    est, se = rqmc_estimate(oracle.make(spec), spec['d'], m, replicates, zlib.crc32(inst['id'].encode()))
    se_ref = primary['se'] or 0.0
    tol_ref = 0.0 if primary['se'] is not None else ref['uncertainty']
    diff = est - ref['value']
    return {'value': est, 'se': se, 'z': diff / math.hypot(se, se_ref),
            'ratio': abs(diff) / (4.0 * math.hypot(se, se_ref) + tol_ref)}


def check_inner(spec: dict, n: int = 10, seed: int = 0) -> float:
    polys = [spec['phi']] + ([spec['phi2']] if 'phi2' in spec else [])
    worst = 0.0
    for y2 in np.random.default_rng(seed).uniform(-3.0, 3.0, n):
        coefs = []
        for q in polys:
            b, c = synthetic._bc(q, y2)
            coefs.append(np.array([q['a'], float(b), float(c)]))
        if len(coefs) == 2:
            coefs.append(coefs[0] - coefs[1])
        pts = sorted(r.real for cf in coefs for r in np.roots(cf) if abs(r.imag) < 1e-12 and abs(r.real) < 12.0)
        edges = [-12.0] + pts + [12.0]
        val = sum(integrate.quad(lambda y1: float(synthetic.nonsmooth(spec, y1, y2) * pdf(y1)), lo, hi,
                                 epsabs=1e-15, epsrel=1e-13, limit=200)[0]
                  for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
        worst = max(worst, abs(val - synthetic.inner_exact(spec, y2)))
    return worst


def pointwise(inst: dict) -> dict:
    spec = inst['spec']
    if inst['track'] == 'hard':
        return {'inner': check_inner(spec)}
    if spec['family'] not in LOGNORMAL:
        return {}
    pos, neg = options.lognormal_terms(spec)
    scale = max(1.0, abs(inst['reference']['value']))
    return {'representation': options.check_representation(spec),
            'preint': max(options.check_preint(spec, th) for _, th in options.directions(spec, pos, neg)) / scale}


def check(inst: dict, oracle_m: int) -> dict:
    warnings.filterwarnings('ignore')
    t0 = time.time()
    agree = agreement(inst['routes'])
    orc = oracle_check(inst, oracle_m)
    pw = pointwise(inst)
    passed = {'agreement': agree['ratio'] <= 1.0, 'oracle': orc['ratio'] <= 1.0,
              'pointwise': all(v <= POINTWISE_TOL[k] for k, v in pw.items())}
    return {'id': inst['id'], 'track': inst['track'], 'family': inst['family'], 'd': inst['d'],
            'reference': inst['reference'], 'agreement': agree, 'oracle': orc, 'pointwise': pw, 'passed': passed,
            'seconds': round(time.time() - t0, 1)}


def bvn_check(n: int = 300, seed: int = 5) -> float:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        h, k = rng.normal(0.0, 1.5, 2)
        rho = rng.uniform(-0.95, 0.95)
        worst = max(worst, abs(bvn_cdf(h, k, rho) - bvn_cdf_quad(h, k, rho)))
    return worst


def _fmt(x: float) -> str:
    return f'{x:.1e}'


def _route_class(method: str) -> str:
    if method.startswith('A:'):
        return 'exact inner integral + adaptive outer quadrature'
    if 'closed form' in method:
        return 'closed form'
    if method.startswith('conditioning'):
        return 'conditioning + nested quadrature'
    if method.startswith('inverse-variance'):
        return 'preintegration + RQMC, two directions'
    if 'Gauss-Hermite' in method:
        return 'preintegration + tensor Gauss-Hermite'
    return method


def report(meta: dict, rows: list[dict], gen: dict, bvn: float, oracle_m: int) -> str:
    n_hard = sum(r['track'] == 'hard' for r in rows)
    fails = [r for r in rows if not all(r['passed'].values())]
    secs = {i['id']: i['seconds'] for i in gen['instances']}
    lines = [
        '# v3 prototype references: verification',
        '',
        f'Auto-generated by `verify.py` for `{meta["data"]}` (namespace {gen["namespace"]}, preintegration routes '
        f'2^{gen["rqmc_m"]} x 16 scrambles). Do not edit by hand.',
        '',
        f'- Instances: {len(rows)} ({n_hard} hard-track synthetic, {len(rows) - n_hard} option-track).',
        f'- All checks passed: {len(rows) - len(fails)} / {len(rows)}.',
        f'- Bivariate normal CDF (Owen T) against 1D quadrature, 300 random points: max difference {_fmt(bvn)}.',
        f'- Generation time {sum(secs.values()) / 60:.0f} CPU-minutes; verification {meta["minutes"]:.1f} minutes '
        f'wall-clock.',
        '',
        'Checks: **agreement** is the largest ratio |difference| / allowed over pairs of independent routes '
        '(allowed = tol_i + tol_j + 4 combined SE; pass <= 1). **Oracle** is RQMC through the rotated oracle, '
        f'2^{oracle_m} x 16 scrambles, compared with the reference (z score; pass when |difference| <= 4 combined SE '
        '+ reference tolerance). **Pointwise** is the largest error of the identities the reference rests on '
        f'(synthetic inner integral, absolute, pass <= {POINTWISE_TOL["inner"]:.0e}; option sum-of-lognormals form, '
        f'relative, <= {POINTWISE_TOL["representation"]:.0e}; closed-form preintegration against quadrature, relative '
        f'to max(1, |I|), <= {POINTWISE_TOL["preint"]:.0e}). **Reference uncertainty** is relative: 3 SE for RQMC '
        'references, otherwise the larger of the claimed tolerance and the distance to the most precise other '
        'deterministic route; '
        '`e_floor` = max(-12, log10(10 x relative uncertainty)).',
        '',
        '| Track | Family | n | d | Primary route | Max rel. uncertainty | e_floor | Max agreement | Max abs. oracle z '
        '| Pointwise max |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    fams = []
    for r in rows:
        if (r['track'], r['family']) not in fams:
            fams.append((r['track'], r['family']))
    for track, fam in fams:
        rs = [r for r in rows if r['track'] == track and r['family'] == fam]
        ds = sorted({r['d'] for r in rs})
        methods = sorted({_route_class(r['reference']['method']) for r in rs})
        floors = [r['reference']['e_floor'] for r in rs]
        pw = [v for r in rs for v in r['pointwise'].values()]
        lines.append(
            f'| {track} | {fam} | {len(rs)} | {", ".join(map(str, ds))} | {"; ".join(methods)} | '
            f'{_fmt(max(r["reference"]["rel_uncertainty"] for r in rs))} | {min(floors):.1f} to {max(floors):.1f} | '
            f'{max(r["agreement"]["ratio"] for r in rs):.2f} | {max(abs(r["oracle"]["z"]) for r in rs):.2f} | '
            f'{_fmt(max(pw)) if pw else "n/a"} |')
    lines += ['', '## Failures', '']
    if not fails:
        lines.append('None.')
    for r in fails:
        lines.append(f'- `{r["id"]}`: {", ".join(k for k, v in r["passed"].items() if not v)}; agreement '
                     f'{r["agreement"]["ratio"]:.2f} ({r["agreement"]["pair"]}), oracle z {r["oracle"]["z"]:+.2f}, '
                     f'pointwise {r["pointwise"]}.')
    lines += ['', '## Per instance', '',
              '| Instance | d | Reference I | Rel. uncertainty | Agreement | Oracle z | Generation s |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        ref = r['reference']
        lines.append(f'| `{r["id"]}` | {r["d"]} | {ref["value"]:.12g} | {_fmt(ref["rel_uncertainty"])} | '
                     f'{r["agreement"]["ratio"]:.2f} | {r["oracle"]["z"]:+.2f} | {secs[r["id"]]:.0f} |')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', type=Path)
    ap.add_argument('report', type=Path)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--oracle-m', type=int, default=18)
    args = ap.parse_args()
    t0 = time.time()
    gen = json.loads((args.data / 'instances.json').read_text())
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(check, inst, args.oracle_m) for inst in gen['instances']]
        bvn = bvn_check()
        rows = []
        for fut in futures:
            row = fut.result()
            print(f"{row['id']:28s} agree {row['agreement']['ratio']:.2f} oracle z {row['oracle']['z']:+.2f} "
                  f"pointwise {row['pointwise']} {'OK' if all(row['passed'].values()) else 'FAIL'}", flush=True)
            rows.append(row)
    repo = Path(__file__).resolve().parents[5]
    meta = {'data': os.path.relpath(args.data.resolve(), repo), 'minutes': (time.time() - t0) / 60}
    (args.data / 'verification.json').write_text(json.dumps({'bvn_max_diff': bvn, 'oracle_m': args.oracle_m,
                                                              'instances': rows}, indent=1) + '\n')
    args.report.write_text(report(meta, rows, gen, bvn, args.oracle_m))
    print(f'{sum(all(r["passed"].values()) for r in rows)} / {len(rows)} instances passed; bvn max diff {bvn:.1e}')


if __name__ == '__main__':
    main()
