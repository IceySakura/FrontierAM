"""One-off migration (2026-11-10) to the per-task layout described in README.md.

Every task family gets tasks/<family>/{versions/<v>/, references/<r>/, solutions/<protocol>/<attempt>/, analysis/,
meta.json}; cross-task results stay in results/, platform notes move to docs/platform/, scripts move into purpose
subdirectories. Tracked paths move with `git mv` (history kept). Afterwards every textual reference to an old path in
the repository is rewritten (longest prefix first), and moved scripts get their repository-root depth fixed.

Usage: migrate_task_layout.py [--apply]
"""

import re
import shutil
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPLY = '--apply' in sys.argv

TASK_MOVES = [
    # conic
    ('tasks/conic/versions/original/package', 'tasks/conic/versions/original/package'),
    ('tasks/conic/versions/original/local_recal', 'tasks/conic/versions/original/local_recal'),
    ('tasks/conic/versions/original/profile_template.json', 'tasks/conic/versions/original/profile_template.json'),
    ('tasks/conic/versions/original/PILOT_BASE.md', 'tasks/conic/versions/original/PILOT_BASE.md'),
    ('tasks/conic/versions/solver-20261105/profile_template.json', 'tasks/conic/versions/solver-20261105/profile_template.json'),
    ('tasks/conic/versions/solver-20261105/package', 'tasks/conic/versions/solver-20261105/package'),
    ('tasks/conic/references/r1/clarabel', 'tasks/conic/references/r1/clarabel'),
    ('tasks/conic/references/r1/REFERENCE_PROVENANCE.json', 'tasks/conic/references/r1/REFERENCE_PROVENANCE.json'),
    # exponential
    ('tasks/exponential/versions/v1/public', 'tasks/exponential/versions/v1/public'),
    ('tasks/exponential/versions/v1/docker_eval', 'tasks/exponential/versions/v1/docker_eval'),
    ('tasks/exponential/versions/v2/dataset', 'tasks/exponential/versions/v2/dataset'),
    ('tasks/exponential/versions/v2/DESIGN.md', 'tasks/exponential/versions/v2/DESIGN.md'),
    ('tasks/exponential/versions/v2/PROBES.md', 'tasks/exponential/versions/v2/PROBES.md'),
    ('tasks/exponential/versions/v2/public', 'tasks/exponential/versions/v2/public'),
    ('tasks/exponential/references/r1', 'tasks/exponential/references/r1'),
    ('tasks/exponential/references/r1/calibration', 'tasks/exponential/references/r1/calibration'),
    ('tasks/exponential/references/r1/vendor', 'tasks/exponential/references/r1/vendor'),
    # galaxy-gravity
    ('tasks/galaxy-gravity/references/candidates', 'tasks/galaxy-gravity/references/candidates'),
    ('tasks/galaxy-gravity/versions/v2', 'tasks/galaxy-gravity/versions/v2'),
    ('tasks/galaxy-gravity/versions/v1', 'tasks/galaxy-gravity/versions/v1'),
    # polarization
    ('tasks/polarization-demosaicking/versions/v1.2', 'tasks/polarization-demosaicking/versions/v1.2'),
    # species-tree (the v2 simulation pipeline lives with v2; run_leased.py becomes a shared ops tool)
    ('scripts/ops/run_leased.py', 'scripts/ops/run_leased.py'),
    ('tasks/species-tree/versions/v2', 'tasks/species-tree/versions/v2'),
    ('tasks/species-tree/versions/v2/generation', 'tasks/species-tree/versions/v2/generation'),
    ('tasks/species-tree/versions/v1', 'tasks/species-tree/versions/v1'),
    # trajectory
    ('tasks/trajectory-covariance/references/candidates', 'tasks/trajectory-covariance/references/candidates'),
    ('tasks/trajectory-covariance/versions/v1', 'tasks/trajectory-covariance/versions/v1'),
    # shared base image of the operator tasks
    ('docker/songberg-reproduction', 'docker/songberg-reproduction'),
]

RESULT_MOVES = [
    ('tasks/conic/analysis', 'tasks/conic/analysis'),
    ('tasks/exponential/analysis', 'tasks/exponential/analysis'),
    ('tasks/galaxy-gravity/analysis', 'tasks/galaxy-gravity/analysis'),
    ('tasks/polarization-demosaicking/analysis', 'tasks/polarization-demosaicking/analysis'),
    ('tasks/species-tree/references/r1/anchors-private.json', 'tasks/species-tree/references/r1/anchors-private.json'),
    ('tasks/species-tree/references/r1/anchors-public.json', 'tasks/species-tree/references/r1/anchors-public.json'),
    ('tasks/species-tree/solutions/v6.2/gpt-6.1-sol-single-1/score.json',
     'tasks/species-tree/solutions/v6.2/gpt-6.1-sol-single-1/score.json'),
    ('tasks/species-tree/solutions/v6.2/claude-opus-5-5-single-1/score.json',
     'tasks/species-tree/solutions/v6.2/claude-opus-5-5-single-1/score.json'),
    ('tasks/species-tree/analysis', 'tasks/species-tree/analysis'),
    ('docs/platform', 'docs/platform'),
    ('tasks/conic/solutions/v6.2/gpt-6.1-sol-single-1', 'tasks/conic/solutions/v6.2/gpt-6.1-sol-single-1'),
    ('tasks/conic/solutions/v6.2/gpt-6.1-sol-iterative-1', 'tasks/conic/solutions/v6.2/gpt-6.1-sol-iterative-1'),
    ('tasks/conic/solutions/v6.2/claude-opus-5-5-single-1', 'tasks/conic/solutions/v6.2/claude-opus-5-5-single-1'),
    ('tasks/conic/solutions/v6.2/claude-opus-5-5-iterative-1', 'tasks/conic/solutions/v6.2/claude-opus-5-5-iterative-1'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-1', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-1'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-record-timing-note', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-record-timing-note'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-2', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-2'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-3', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-3'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-iterative-pilot', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-iterative-pilot'),
    ('tasks/exponential/solutions/v6.2/gpt-6.1-sol-iterative-1', 'tasks/exponential/solutions/v6.2/gpt-6.1-sol-iterative-1'),
    ('tasks/exponential/solutions/v6.2/claude-opus-5-5-single-1', 'tasks/exponential/solutions/v6.2/claude-opus-5-5-single-1'),
    ('tasks/exponential/solutions/v6.2/claude-opus-5-5-single-2', 'tasks/exponential/solutions/v6.2/claude-opus-5-5-single-2'),
    ('tasks/exponential/solutions/v6.2/claude-opus-5-5-single-3', 'tasks/exponential/solutions/v6.2/claude-opus-5-5-single-3'),
    ('tasks/exponential/solutions/v6.2/claude-opus-5-5-iterative-1', 'tasks/exponential/solutions/v6.2/claude-opus-5-5-iterative-1'),
]

META_MOVES = [(f'dashboard/tasks/{id_}.json', f'tasks/{family}/meta.json') for id_, family in [
    ('conic', 'conic'), ('exponential-v2', 'exponential'), ('galaxy-gravity-v2', 'galaxy-gravity'),
    ('polarization-demosaicking', 'polarization-demosaicking'), ('species-tree-v2', 'species-tree'),
    ('trajectory-covariance', 'trajectory-covariance')]]

SCRIPT_GROUPS = {
    'setup': ['bench.py', 'build-images.sh', 'build-operator-images.sh', 'load-grade-images.sh', 'switch-layout.sh',
              'isolation-experiment.sh', 'uid-sampler.py', 'shim-audit.py'],
    'seal': ['prepare.py', 'prepare_exponential_v2.py', 'exp2-freeze.py', 'prepare_operator_tasks.py',
             'prepare_galaxy_v2.py', 'prepare_species_v2.py', 'relock.py', 'migrate_task_layout.py'],
    'references': ['new-task-baselines.sh', 'exp2-panel-suite.sh', 'exp2-public.sh', 'exp2-private.sh',
                   'exp2-freeze-and-private.sh', 'species-v2-public-baseline.sh', 'traj-ref-screen.py'],
    'rounds': ['campaign.sh', 'serial-attempts.sh', 'concurrent-round.py', 'v62-smoke.py', 'v62-chain.sh', 'v62-final.sh',
               'v62-followup.sh', 'v62-after.sh', 'round5-singles.sh', 'round5-sol-retry.sh', 'round5-pipeline.sh'],
    'grade': ['exp2-suite-score.py', 'exp2-sol-suite.sh', 'species-v2-suite-score.py', 'conic-recal.py',
              'conic-recalibrate.py', 'conic-grade-remaining.py', 'conic2-sol-suite.sh', 'conic-skip-check.sh',
              'grade-sol-operator.sh', 'paired-validation.sh', 'exp2-container-check.py'],
    'analyze': ['conic-final-report.py', 'conic-per-case-export.py', 'conic-case-times.py', 'conic-substantive.py',
                'conic-suite-compare.py', 'trajectory-compare.py', 'trajectory-expected.py', 'polar-first-scene.py',
                'polar-warmup-check.py'],
    'site': ['build_dashboard.py', 'build_problem_pages.py'],
    'ops': ['data-maintenance.sh', 'overnight.sh', 'finish-orphan.py', 'release-after.py', 'release-when-free.py'],
}
SCRIPT_MOVES = [(f'scripts/{name}', f'scripts/{group}/{name}') for group, names in SCRIPT_GROUPS.items() for name in names]
SCRIPT_MOVES = [(a, b.replace('build_problem_pages.py', 'build_task_pages.py')) for a, b in SCRIPT_MOVES]

# Temporary links for jobs that are still running against old paths (removed once they finish).
COMPAT = [
    ('tasks/exponential/references/r1', '../references/r1'),
    ('tasks/exponential/references/r1/calibration', '../references/r1/calibration'),
    ('tasks/exponential/versions/v2/dataset', '../versions/v2/dataset'),
    ('tasks/exponential/versions/v2/DESIGN.md', '../versions/v2/DESIGN.md'),
    ('tasks/galaxy-gravity/references/candidates', '../../galaxy-gravity/references/candidates'),
    ('scripts/ops/run_leased.py', '../../../../scripts/ops/run_leased.py'),
    ('tasks/trajectory-covariance/references/candidates', '../../trajectory-covariance/references/candidates'),
]


def tracked(path: str) -> bool:
    return bool(subprocess.run(['git', 'ls-files', path], cwd=ROOT, capture_output=True, text=True).stdout.strip())


def move(old: str, new: str) -> None:
    src, dst = ROOT / old, ROOT / new
    if not src.exists() and not src.is_symlink():
        print(f'  skip (missing): {old}')
        return
    print(f'  {old} -> {new}')
    if not APPLY:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f'target exists: {new}')
    if tracked(old):
        subprocess.run(['git', 'mv', old, new], cwd=ROOT, check=True)
        if src.exists():  # untracked leftovers that git mv did not carry
            for rest in sorted(src.rglob('*')):
                if rest.is_file():
                    target = dst / rest.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(rest), str(target))
            shutil.rmtree(src)
    else:
        shutil.move(str(src), str(dst))


def fix_script_depth(path: Path) -> None:
    text = path.read_text()
    new = text.replace('Path(__file__).resolve().parents[2]', 'Path(__file__).resolve().parents[2]')
    new = new.replace('cd "$(dirname "$0")/../.."', 'cd "$(dirname "$0")/../.."')
    new = new.replace("root=$(cd -- \"$(dirname -- \"$0\")/..\" && pwd)", "root=$(cd -- \"$(dirname -- \"$0\")/../..\" && pwd)")
    if new != text:
        path.write_text(new)


def rewrite_references(mapping: list[tuple[str, str]]) -> int:
    pairs = sorted(mapping, key=lambda p: -len(p[0]))
    files = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    files = [f for f in files if not f.startswith(('nexus/', 'tasks/conic/versions/', 'tasks/exponential/references/r1/vendor'))]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'scripts').rglob('*') if p.is_file()]
    changed = 0
    for name in sorted(set(files)):
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or path.suffix not in ('.py', '.sh', '.md', '.json', '.mdc', '.toml', '') \
                or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        new = text
        for old, rep in pairs:
            new = re.sub(r'(?<![\w.-])' + re.escape(old) + r'(?![\w.-])', rep, new)
        if new != text:
            changed += 1
            if APPLY:
                path.write_text(new)
    return changed


def main():
    moves = TASK_MOVES + RESULT_MOVES + META_MOVES + SCRIPT_MOVES
    print('== moves')
    for old, new in moves:
        move(old, new)
    if APPLY:
        for old, new in SCRIPT_MOVES:
            if (ROOT / new).exists():
                fix_script_depth(ROOT / new)
    print('== references rewritten in', rewrite_references(moves), 'files')
    if APPLY:
        for link, target in COMPAT:
            path = ROOT / link
            if not path.exists() and not path.is_symlink():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(target)
        for leftover in ['tasks/exponential/sota_v1', 'tasks/operator', 'results/runs', 'dashboard/tasks', 'problems']:
            print('  leftover', leftover, sorted(p.name for p in (ROOT / leftover).iterdir()) if (ROOT / leftover).exists() else '-')


if __name__ == '__main__':
    main()
