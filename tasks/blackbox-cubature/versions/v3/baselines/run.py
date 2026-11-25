"""Run the M2 baselines on the prototype set and write the margin report.

Every method x budget x seed on every instance; errors against the M1 references. Per instance the error is the median
over seeds of e = log10(|I_hat - I| / |I|), clipped to [e_floor, 0]. Scores use the v3 draft rule with strong = 50 and
plain_mc = 0: per instance g = clip(e_strong - e_method, -2, 2), averaged per (family, dimension) cell, then per
track, G = mean of the two tracks; score = 50 + 25 G for G >= 0, 50 (G - G0) / (-G0) for G < 0 with G0 = G(plain_mc).

Usage: run.py DATA_DIR OUT_DIR [--workers 7] [--budgets 10 12 14 16] [--seeds 8] [--methods M ...] [--only FAMILY ...]
Writes OUT_DIR/runs.json, OUT_DIR/summary.json, OUT_DIR/REPORT.md and one error-versus-budget plot per track.
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import methods  # noqa: E402
import oracle  # noqa: E402

ORDER = ['plain_mc', 'plain_rqmc', 'as_rqmc', 'as_preint', 'strong_preint', 'strong_rqmc', 'strong', 'oracle_preint',
         'oracle_split']
PARTIAL = {'oracle_split'}
SCORED = 14


def _f(x: float) -> str:
    return f'{x:.2f}' if np.isfinite(x) else '—'


def run_cell(inst: dict, method: str, budgets: list[int], seeds: int) -> list[dict]:
    warnings.filterwarnings('ignore')
    spec = inst['spec']
    g = oracle.make(spec)
    ref = inst['reference']['value']
    out = []
    for k in budgets:
        for s in range(seeds):
            f = methods.Counted(g, spec['d'], 2 ** k)
            t0 = time.time()
            try:
                est, note = methods.METHODS[method](f, spec['d'], 1000 * s + 17, {'spec': spec})
            except methods.NotApplicable:
                return []
            out.append({'id': inst['id'], 'method': method, 'log2n': k, 'seed': s, 'estimate': est, 'used': f.used,
                        'seconds': round(time.time() - t0, 3), 'note': note, 'rel_err': abs(est - ref) / abs(ref)})
    return out


def derive_strong(runs: list[dict]) -> list[dict]:
    """strong runs exactly methods.STRONG (same code path and seeds), so its runs are copies."""
    out = [r for r in runs if r['method'] != 'strong']
    return out + [dict(r, method='strong') for r in runs if r['method'] == methods.STRONG]


def refresh_errors(runs: list[dict], insts: dict) -> list[dict]:
    """Keep the runs of the current instances and recompute their errors against the current references."""
    out = []
    for r in runs:
        if r['id'] in insts:
            ref = insts[r['id']]['reference']['value']
            out.append(dict(r, rel_err=abs(r['estimate'] - ref) / abs(ref)))
    return out


def instance_errors(runs: list[dict], insts: dict) -> dict:
    """(id, method, log2n) -> median over seeds of the clipped log10 relative error."""
    by = defaultdict(list)
    for r in runs:
        by[(r['id'], r['method'], r['log2n'])].append(r['rel_err'])
    out = {}
    for key, errs in by.items():
        e = np.log10(np.maximum(np.asarray(errs), 1e-300))
        out[key] = float(np.median(np.clip(e, insts[key[0]]['reference']['e_floor'], 0.0)))
    return out


def scores(err: dict, insts: dict, meths: list[str], k: int) -> dict:
    """G per track and overall for every method against strong at budget 2^k, and the 0-100 score."""
    out = {}
    for m in meths:
        tracks = {}
        for track in ('hard', 'options'):
            cells = defaultdict(list)
            for iid, inst in insts.items():
                if inst['track'] == track and (iid, m, k) in err and (iid, 'strong', k) in err:
                    g = err[(iid, 'strong', k)] - err[(iid, m, k)]
                    cells[(inst['family'], inst['d'])].append(float(np.clip(g, -2.0, 2.0)))
            if cells:
                tracks[track] = float(np.mean([np.mean(v) for v in cells.values()]))
        out[m] = {'tracks': tracks, 'G': float(np.mean(list(tracks.values()))) if tracks else float('nan')}
    g0 = out.get('plain_mc', {}).get('G', float('nan'))
    for m, v in out.items():
        G = v['G']
        v['score'] = 50.0 + 25.0 * G if G >= 0 else (50.0 * (G - g0) / (-g0) if g0 < 0 else float('nan'))
    return out


def family_table(err: dict, insts: dict, meths: list[str], budgets: list[int]) -> dict:
    """(track, family, method, log2n) -> mean over instances of the instance error; plus per (family, d)."""
    fam, famd = defaultdict(list), defaultdict(list)
    for (iid, m, k), e in err.items():
        inst = insts[iid]
        fam[(inst['track'], inst['family'], m, k)].append(e)
        famd[(inst['track'], inst['family'], inst['d'], m, k)].append(e)
    return ({key: float(np.mean(v)) for key, v in fam.items()}, {key: float(np.mean(v)) for key, v in famd.items()})


def plot(fam_mean: dict, floors: dict, meths: list[str], budgets: list[int], track: str, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fams = sorted({f for (t, f, _, _) in fam_mean if t == track}, key=lambda s: (len(s), s))
    cols = min(5, len(fams))
    rows = math.ceil(len(fams) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False, sharex=True)
    for ax, fam in zip(axes.flat, fams):
        for m in meths:
            ys = [fam_mean.get((track, fam, m, k)) for k in budgets]
            if all(y is not None for y in ys):
                ax.plot(budgets, ys, marker='o', ms=3, label=m)
        ax.axhline(floors[(track, fam)], color='grey', ls='--', lw=0.8)
        ax.axvline(SCORED, color='grey', ls=':', lw=0.8)
        ax.set_title(fam, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(fams):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('log2 N')
    for ax in axes[:, 0]:
        ax.set_ylabel('log10 rel. error')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f'{track} track: median log10 relative error (clipped at e_floor, dashed) vs budget', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(insts, runs, err, meths, budgets, out: Path, minutes: float) -> str:
    fam_mean, famd_mean = family_table(err, insts, meths, budgets)
    floors = defaultdict(list)
    for inst in insts.values():
        floors[(inst['track'], inst['family'])].append(inst['reference']['e_floor'])
    floors = {k: float(np.mean(v)) for k, v in floors.items()}
    for track in ('hard', 'options'):
        plot(fam_mean, floors, meths, budgets, track, out / f'{track}.png')
    sc = scores(err, insts, [m for m in meths if m not in PARTIAL], SCORED)
    lines = ['# v3 M2 baselines on the prototype set', '',
             f'Auto-generated by `run.py` ({len(insts)} instances, budgets 2^{budgets[0]}..2^{budgets[-1]}, '
             f'{len({r["seed"] for r in runs})} seeds, {len(runs)} runs, {minutes:.0f} minutes wall-clock). '
             'Error per instance: median over seeds of log10 relative error, clipped to [e_floor, 0].', '',
             f'## Scores at N = 2^{SCORED} (draft rule: strong = 50, plain_mc = 0)', '',
             'Per instance g = clip(e_strong - e_method, -2, 2); mean per (family, dimension) cell, then per track; '
             'G = mean of the two tracks; score = 50 + 25 G for G >= 0, 50 (G - G0) / (-G0) for G < 0, '
             'G0 = G(plain_mc).', '',
             '| Method | G hard | G options | G | Score |', '| --- | --- | --- | --- | --- |']
    for m, v in sc.items():
        lines.append(f'| {m} | {v["tracks"].get("hard", float("nan")):+.2f} | '
                     f'{v["tracks"].get("options", float("nan")):+.2f} | {v["G"]:+.2f} | {v["score"]:.1f} |')
    lines += ['', f'## Mean error per family at N = 2^{SCORED}', '',
              '| Track | Family | e_floor | ' + ' | '.join(meths) + ' | strong - oracle |',
              '| --- | --- | --- | ' + ' | '.join('---' for _ in meths) + ' | --- |']
    fams = sorted({(t, f) for (t, f, _, _) in fam_mean}, key=lambda x: (x[0], len(x[1]), x[1]))
    for t, f in fams:
        row = [fam_mean.get((t, f, m, SCORED), float('nan')) for m in meths]
        gap = fam_mean.get((t, f, 'strong', SCORED), float('nan')) - fam_mean.get((t, f, 'oracle_preint', SCORED),
                                                                                  float('nan'))
        lines.append(f'| {t} | {f} | {floors[(t, f)]:.1f} | ' + ' | '.join(_f(x) for x in row) + f' | {_f(gap)} |')
    lines += ['', f'## Error per (family, dimension) at N = 2^{SCORED}', '',
              '| Track | Family | d | ' + ' | '.join(meths) + ' |', '| --- | --- | --- | ' +
              ' | '.join('---' for _ in meths) + ' |']
    keys = sorted({(t, f, d) for (t, f, d, _, _) in famd_mean}, key=lambda x: (x[0], len(x[1]), x[1], x[2]))
    for t, f, d in keys:
        lines.append(f'| {t} | {f} | {d} | ' + ' | '.join(
            _f(famd_mean.get((t, f, d, m, SCORED), float('nan'))) for m in meths) + ' |')
    lines += ['', '## Convergence: mean error per track and budget', '',
              '| Track | Method | ' + ' | '.join(f'2^{k}' for k in budgets) + ' |',
              '| --- | --- | ' + ' | '.join('---' for _ in budgets) + ' |']
    for track in ('hard', 'options'):
        for m in meths:
            vals = []
            for k in budgets:
                es = [e for (iid, mm, kk), e in err.items() if mm == m and kk == k and insts[iid]['track'] == track]
                vals.append(f'{np.mean(es):.2f}' if es else 'n/a')
            lines.append(f'| {track} | {m} | ' + ' | '.join(vals) + ' |')
    lines += ['', f'## Wall time per run at N = 2^{SCORED} (limit 60 s)', '', '| Method | Median s | Max s |',
              '| --- | --- | --- |']
    for m in meths:
        secs = [r['seconds'] for r in runs if r['method'] == m and r['log2n'] == SCORED]
        if secs:
            lines.append(f'| {m} | {np.median(secs):.2f} | {np.max(secs):.2f} |')
    lines += ['', '![hard track](hard.png)', '', '![option track](options.png)', '']
    summary = {'scores': sc, 'family': {'|'.join(map(str, k)): v for k, v in fam_mean.items()},
               'family_d': {'|'.join(map(str, k)): v for k, v in famd_mean.items()}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--budgets', type=int, nargs='*', default=[10, 12, 14, 16])
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--methods', nargs='*', default=ORDER)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--report-only', action='store_true', help='rebuild the report from OUT_DIR/runs.json')
    ap.add_argument('--merge', action='store_true', help='replace only the rerun (instance, method) cells')
    args = ap.parse_args()
    t0 = time.time()
    data = json.loads((args.data / 'instances.json').read_text())['instances']
    every = {i['id']: i for i in data}
    insts = {k: v for k, v in every.items() if not args.only or v['family'] in args.only}
    args.out.mkdir(parents=True, exist_ok=True)
    derived = 'strong' in args.methods and methods.STRONG in args.methods
    path = args.out / 'runs.json'
    if args.report_only:
        runs = json.loads(path.read_text())
    else:
        runs = []
        todo = [m for m in args.methods if not (derived and m == 'strong')]
        cost = lambda iid: insts[iid]['d'] * (insts[iid]['m'] or 1)
        tasks = sorted(((iid, m) for iid in insts for m in todo), key=lambda t: -cost(t[0]))
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(run_cell, insts[iid], m, args.budgets, args.seeds): (iid, m) for iid, m in tasks}
            for n_done, fut in enumerate(as_completed(futures), 1):
                runs += fut.result()
                if n_done % 25 == 0:
                    print(f'{n_done}/{len(futures)} cells, {(time.time() - t0) / 60:.1f} min', flush=True)
        if args.merge and path.exists():
            redone = set(tasks)
            runs += [r for r in json.loads(path.read_text()) if (r['id'], r['method']) not in redone]
        path.write_text(json.dumps(runs) + '\n')
    whole = args.merge or args.report_only
    if whole:
        insts = every
    runs = refresh_errors(runs, insts)
    if derived or (whole and any(r['method'] == methods.STRONG for r in runs)):
        runs = derive_strong(runs)
    err = instance_errors(runs, insts)
    present = {r['method'] for r in runs}
    meths = [m for m in ORDER if m in present and (whole or m in args.methods)]
    text = report(insts, runs, err, meths, args.budgets, args.out, (time.time() - t0) / 60)
    (args.out / 'REPORT.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
