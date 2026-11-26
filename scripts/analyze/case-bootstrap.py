"""Case-sampling noise of the v6.3 board scores on the tasks whose top scores are close (results/OPEN_ITEMS.md, 11-26).

For each board cell (the attempt the board shows) with a valid score, the cases are resampled with replacement
(paired: the candidate and both references are recomputed on the same resample, then mapped exactly as the formal
score). The spread says how much a score would move with another draw of test cases of the same design; it does
not include attempt-to-attempt variation of the model. Cells that failed all-or-nothing are left out.

  species-tree-v3               mean normalised RF over 96 cases (suite shards), linear mapping
  polarization-demosaicking-v3  mean RMSE over 132 scenes, linear mapping
  ihvp-sequence-v4              mean energy loss over the private sequences, linear mapping (scripts/analyze/ihvp-v4-runs.py)
  trajectory-covariance-v2      mean over the 30 setting x prefix cells of the mean loss; parent chains are resampled
                                within each setting (all five prefixes of a parent together), linear mapping

Usage: case-bootstrap.py [--resamples 2000] [--seed 0]   -> var/raw/case-bootstrap.json and a table on stdout
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STORE = ROOT / 'var/stores/store-v63'
SETTING = {'单轮': 'single', '迭代': 'iterative'}


def linear(raw: float, weak: float, strong: float) -> float:
    """arena.operator_tasks.score_100: weak reference 0, strong reference 50, exact 100."""
    if raw >= weak:
        return 0.0
    if raw >= strong:
        return 50.0 * (weak - raw) / (weak - strong)
    return 100.0 - 50.0 * raw / strong


def board(family: str, version: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / 'scripts/site'))
    from board_stats import HIDDEN
    meta = json.loads((ROOT / 'tasks' / family / 'meta.json').read_text())
    group = next(g for g in meta['result_groups'] if g.get('protocol') == 'v6.3' and g.get('version') == version)
    return [r for r in group['results'] if r['model'] not in HIDDEN and r.get('score')]


def folder(family: str, sub: str, row: dict) -> Path:
    return ROOT / 'tasks' / family / 'solutions/v6.3' / sub / f'{row["model"]}-{SETTING[row["setting"]]}-{row.get("attempt", 1)}'


def suite_cases(suite: str, part: str) -> dict[str, float]:
    """Per-case normalised RF of one part (candidate, weak, strong) over every shard of a species suite."""
    out = {}
    for shard in json.loads((STORE / 'suites' / f'{suite}.json').read_text())['shards']:
        summary = STORE / 'results' / shard['job'] / 'grade' / part / 'summary.json'
        for r in json.loads(summary.read_text()).get('records', []):
            out[r['case']] = r['normalized_rf'] if r.get('status') == 'ok' else None
    return out


def species() -> tuple[dict, dict, dict]:
    cells, refs = {}, None
    for row in board('species-tree', 'v3'):
        s = json.loads((folder('species-tree', '', row) / 'score.json').read_text())
        if s.get('candidate', {}).get('status') != 'ok':
            continue
        cells[(row['model'], row['setting'])] = suite_cases(s['candidate_suite'], 'candidate')
        refs = refs or {w: suite_cases(s['reference_suite'], w) for w in ('weak', 'strong')}
    return cells, refs['weak'], refs['strong']


def polarization() -> tuple[dict, dict, dict]:
    base = STORE / 'baselines/polarization-demosaicking-v3'
    parts = json.loads((base / (base / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['metrics']['parts']
    cells = {}
    for row in board('polarization-demosaicking', 'v3'):
        c = json.loads((folder('polarization-demosaicking', 'v3', row) / 'score.json').read_text()).get('candidate') or {}
        if c.get('raw_error') is not None and len(c.get('scene_rmse') or {}) == len(parts['weak']['scene_rmse']):
            cells[(row['model'], row['setting'])] = c['scene_rmse']
    return cells, parts['weak']['scene_rmse'], parts['strong']['scene_rmse']


def ihvp() -> tuple[dict, dict, dict]:
    def losses(name: str) -> dict[str, float]:
        rows = json.loads((STORE / 'ihvp-v4' / name / 'summary.json').read_text()).get('records', [])
        return {r.get('sequence') or r.get('case') or r.get('id') or str(i): r['energy_loss']
                for i, r in enumerate(rows) if r.get('status') == 'ok' and r.get('energy_loss') is not None}
    cells = {}
    for row in board('ihvp-sequence', 'v4'):
        s = json.loads((folder('ihvp-sequence', 'v4', row) / 'score.json').read_text())
        name = s.get('source', '').split('(')[-1].rstrip(')')
        if s.get('status') == 'ok' and name:
            cells[(row['model'], row['setting'])] = losses(name)
    return cells, losses('cg-private'), losses('lbfgs-private')


def trajectory() -> tuple[dict, dict, dict]:
    """Per-case loss keyed (setting, parent, prefix length), from every shard's verifier records."""
    def cases(suite: str, part: str) -> dict:
        out = {}
        for shard in json.loads((STORE / 'suites' / f'{suite}.json').read_text())['shards']:
            path = STORE / 'results' / shard['job'] / 'grade' / part / 'verifier/cases.jsonl'
            for line in path.read_text().splitlines():
                r = json.loads(line)
                out[(r['model'], r['parent'], r['length'])] = r['loss'] if r.get('status') == 'ok' else None
        return out
    cells, refs = {}, None
    for row in board('trajectory-covariance', 'v2'):
        s = json.loads((folder('trajectory-covariance', '', row) / 'score.json').read_text())
        if (s.get('candidate') or {}).get('status') != 'ok':
            continue
        cells[(row['model'], row['setting'])] = cases(s['candidate_suite'], 'candidate')
        refs = refs or {w: cases(s['reference_suite'], w) for w in ('weak', 'strong')}
    return cells, refs['weak'], refs['strong']


def mean_of(d: dict, sample: list) -> float:
    """Plain mean over the sampled case keys."""
    return sum(d[k] for k in sample) / len(sample)


def cell_mean_of(d: dict, sample: list) -> float:
    """Trajectory: the sample is a list of (setting, parent); mean over (setting, length) cells of the mean loss."""
    lengths = {}
    for setting, _, length in d:
        lengths.setdefault(setting, set()).add(length)
    cells = {}
    for setting, parent in sample:
        for length in lengths[setting]:
            cells.setdefault((setting, length), []).append(d[(setting, parent, length)])
    return sum(sum(v) / len(v) for v in cells.values()) / len(cells)


def bootstrap(cells: dict, weak: dict, strong: dict, resamples: int, rng: random.Random, clustered: bool = False) -> dict:
    keys = sorted(weak)
    cells = {k: v for k, v in cells.items() if sorted(v) == keys and all(x is not None for x in v.values())}
    if clustered:   # parents resampled within each setting
        units = {}
        for setting, parent, _ in keys:
            units.setdefault(setting, set()).add(parent)
        units = {s: sorted(p) for s, p in units.items()}
        full = [(s, p) for s, ps in units.items() for p in ps]
        draw = lambda: [(s, rng.choice(ps)) for s, ps in units.items() for _ in ps]
        mean = cell_mean_of
    else:
        full = keys
        draw = lambda: [rng.choice(keys) for _ in keys]
        mean = mean_of
    draws = {k: [] for k in cells}
    for _ in range(resamples):
        idx = draw()
        u, w = mean(weak, idx), mean(strong, idx)
        for k, v in cells.items():
            draws[k].append(linear(mean(v, idx), u, w))
    u, w = mean(weak, full), mean(strong, full)
    out = {}
    for k, v in cells.items():
        ordered = sorted(draws[k])
        out[k] = {'score': linear(mean(v, full), u, w), 'sd': statistics.stdev(ordered),
                  'ci90': (ordered[int(0.05 * resamples)], ordered[int(0.95 * resamples) - 1])}
    for setting in SETTING:
        ranked = sorted((k for k in cells if k[1] == setting), key=lambda k: -out[k]['score'])
        for a, b in zip(ranked, ranked[1:]):
            out[a][f'p_above_next'] = sum(x > y for x, y in zip(draws[a], draws[b])) / resamples
    return {'cases': len(keys), 'cells': out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    result = {}
    for task, load in (('species-tree-v3', species), ('polarization-demosaicking-v3', polarization), ('ihvp-sequence-v4', ihvp),
                       ('trajectory-covariance-v2', trajectory)):
        result[task] = bootstrap(*load(), args.resamples, random.Random(args.seed), clustered=load is trajectory)
        print(f'\n{task} ({result[task]["cases"]} cases)')
        for (model, setting), c in sorted(result[task]['cells'].items(), key=lambda kv: (kv[0][1], -kv[1]['score'])):
            above = f'  P(> next) {c["p_above_next"]:.2f}' if 'p_above_next' in c else ''
            print(f'  {setting} {model:20s} {c["score"]:6.1f}  sd {c["sd"]:4.2f}  90% [{c["ci90"][0]:.1f}, {c["ci90"][1]:.1f}]{above}')
    out = ROOT / 'var/raw/case-bootstrap.json'
    out.write_text(json.dumps({t: {'cases': r['cases'], 'cells': {f'{m}/{s}': c for (m, s), c in r['cells'].items()}}
                               for t, r in result.items()}, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
