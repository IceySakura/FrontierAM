"""Batch-evaluate EnterpriseBench (Syntara enterprise-office) on Gongfeng.

Four stages, each skippable, so a handoff can re-run the cheap ones without
repeating the expensive one::

    build      the delivery -> a Nexus dataset, one item per task
    preflight  every world image the dataset names actually exists
    run        nexus run-batch over those items
    report     aggregate the exported runs

``--dry-run`` stops after preflight, which is the cheap way to prove a
configuration before any sandbox is spent.

Usage::

    python -m nexus.extensions.tasks.context_agent.enterprisebench_native.eval \\
        --delivery /path/to/tencent_aug24_delivery \\
        --workdir /path/to/eval_run \\
        --image-tag v2 --memory 10 --max-concurrent 8

Scoring retains the delivery's RewardKit/OpenCode semantics, but Nexus owns the
normalized Judge configuration and regenerates the Runtime files before
grading. It runs inside the sandbox and arrives as ``task_output.score``, so
this module never computes a second score.

Six images, not three hundred
-----------------------------
Every task declares exactly one world in its ``environment/Dockerfile``
``FROM`` line, and :mod:`.dataset` turns that into
``<prefix>-<world>:<tag>`` on the item. Counted over the whole delivery:
suitecrm 133, appflowy 64, base 48, teable 31, suitecrm-zammad 14, zammad 10.
So the tasks of a world share one image and the run needs six, which is why
``--worlds`` is offered: it is the knob that keeps a run inside the images that
have actually been pushed.

Why the mean is not just sum(score)/len(items)
----------------------------------------------
A rollout can end without a score for two reasons that are not the model being
wrong, and folding either into the average understates the run:

``EnvironmentInitError``
    The world never booted or never passed its healthcheck, so the agent never
    saw the task. The Native task raises this specifically so it can be told
    apart from a model/code failure.

``verifier_error`` / a verifier exception
    The rollout happened but grading did not -- the historical case being
    rewardkit exiting 127 on a missing ``opencode`` and writing no reward file.

:func:`aggregate` therefore reports ``mean_score`` over graded rows only and
carries the ungraded counts beside it. Read ``graded`` as the denominator.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.context_agent.enterprisebench_native.dataset import DEFAULT_AGENT_NAME, read_world


# Task-count-per-world over the whole delivery, for the preflight summary. A
# world missing from here is not an error (a delivery may add one); it only
# means the summary cannot say how many tasks it gates.
WORLD_TASK_COUNTS: dict[str, int] = {
    'suitecrm': 133,
    'appflowy': 64,
    'base': 48,
    'teable': 31,
    'suitecrm-zammad': 14,
    'zammad': 10,
}

DEFAULT_IMAGE_PREFIX = 'mirrors.tencent.com/hunyuan_yanguan/syntara'
DEFAULT_IMAGE_TAG = 'v2'
DEFAULT_COMPOSE = Path(__file__).resolve().parent / 'base.yaml'

# 10 GB rather than the 4 GB the delivery's task.toml asks for. A world is a
# full service stack (SuiteCRM is Apache + MariaDB + PHP; zammad adds
# Elasticsearch), and the seeded database is restored at boot, so the
# task-declared figure is the per-task minimum rather than a comfortable
# ceiling. Applied as a `--set` AFTER `--data` so it overrides the value
# baked into each item; compose parts merge in command-line order.
DEFAULT_MEMORY_GB = 10
DEFAULT_CPU = 2

# Substrings that classify an ungraded run. Matched against the batch
# database's error string, which is "{type}: {message}\n\n{traceback}".
_ENV_INIT_MARKER = 'EnvironmentInitError'
_VERIFIER_MARKERS = ('RewardFileNotFoundError', 'Verifier timed out', "Reward 'reward' failed")


def log(stage: str, message: str) -> None:
    print(f'[{stage}] {message}', flush=True)


class EnterpriseBenchError(RuntimeError):
    """A stage failed with a message already fit to show a user."""


# --------------------------------------------------------------------------- build


def world_of(delivery: Path, task_id: str) -> str | None:
    """The world a task builds on, or ``None`` when it declares none.

    A directory under ``tasks/`` without a readable ``FROM syntara-<world>`` is
    skipped rather than fatal: the delivery is third-party input, and one
    malformed entry should not block a run over the other 299.
    """
    dockerfile = delivery / 'tasks' / task_id / 'environment' / 'Dockerfile'
    if not dockerfile.is_file():
        return None
    try:
        return read_world(dockerfile)
    except (OSError, ValueError):
        return None


def write_task_list(delivery: Path, worlds: list[str], destination: Path) -> int:
    """Write the ids of every task belonging to *worlds*, and return the count.

    :mod:`.dataset` selects by task id, not by world, so a world filter has
    to be resolved to ids here -- off the same ``FROM`` line the dataset
    generator itself reads, so the two cannot disagree.
    """
    wanted = set(worlds)
    tasks_root = delivery / 'tasks'
    if not tasks_root.is_dir():
        raise EnterpriseBenchError(f'no tasks/ under {delivery}')

    available: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    for path in sorted(tasks_root.iterdir()):
        if not path.is_dir():
            continue
        world = world_of(delivery, path.name)
        if world is None:
            continue
        available[world] += 1
        if world in wanted:
            selected.append(path.name)

    unknown = sorted(wanted - available.keys())
    if unknown:
        raise EnterpriseBenchError(
            f'no such world(s) {unknown} in {tasks_root}; delivery has {dict(sorted(available.items()))}'
        )
    if not selected:
        raise EnterpriseBenchError(f'no tasks found for world(s) {sorted(wanted)}')
    destination.write_text('\n'.join(sorted(selected)) + '\n', encoding='utf-8')
    return len(selected)


def stage_build(args: argparse.Namespace, dataset_dir: Path) -> None:
    delivery = Path(args.delivery).expanduser().resolve()
    if not (delivery / 'tasks').is_dir():
        raise EnterpriseBenchError(f'no tasks/ under {delivery}')

    task_list = args.tasks
    if args.worlds:
        generated = dataset_dir.parent / 'task_ids.txt'
        count = write_task_list(delivery, args.worlds, generated)
        log('build', f'world filter {args.worlds} -> {count} task(s) listed in {generated}')
        task_list = str(generated)

    command = [
        sys.executable,
        '-m',
        'nexus.extensions.tasks.context_agent.enterprisebench_native.dataset',
        '--source',
        str(delivery),
        '--out',
        str(dataset_dir),
        '--image-prefix',
        args.image_prefix,
        '--image-tag',
        args.image_tag,
        '--agent-name',
        args.agent_name,
        '--cpu',
        str(args.cpu),
        '--memory',
        str(args.memory),
        '--force',
    ]
    for tool_set in args.disable_world_tool_set:
        command += ['--disable-world-tool-set', tool_set]
    if task_list:
        command += ['--tasks', str(task_list)]
    if args.limit:
        command += ['--limit', str(args.limit)]

    log('build', f'generating dataset -> {dataset_dir}')
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise EnterpriseBenchError(f'dataset build failed (exit {result.returncode})')


# ----------------------------------------------------------------------- preflight


def dataset_images(dataset_dir: Path) -> dict[str, list[str]]:
    """Map each image the dataset references to the task ids needing it."""
    items_dir = dataset_dir / 'items'
    if not items_dir.is_dir():
        raise EnterpriseBenchError(f'no items/ under {dataset_dir}')

    by_image: dict[str, list[str]] = defaultdict(list)
    for item_path in sorted(items_dir.glob('*/item.yaml')):
        raw = yaml.safe_load(item_path.read_text(encoding='utf-8')) or {}
        body = raw.get('body') or {}
        main = ((body.get('runtime_configs') or {}).get('main')) or {}
        by_image[main.get('image') or '<no image in item>'].append(raw.get('id', item_path.parent.name))
    if not by_image:
        raise EnterpriseBenchError(f'no items found under {items_dir}')
    return dict(by_image)


def _registry_get(url: str, token: str | None = None) -> Any:
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def resolve_image(image: str) -> tuple[bool, str]:
    """Return ``(exists, detail)`` for one image reference, read-only.

    ``hunyuan_yanguan`` is a public Harbor project, so an anonymous token
    carries ``pull`` for a repository that exists and no actions at all for one
    that does not -- which is how a missing repository is told apart from a
    missing tag without any credential.
    """
    reference, _, tag = image.rpartition(':')
    if not reference or '/' not in reference:
        return False, f'cannot parse image reference: {image!r}'
    registry, _, repository = reference.partition('/')

    try:
        scope = f'repository:{repository}:pull'
        token_url = f'https://{registry}/service/token?' + urllib.parse.urlencode({
            'service': 'harbor-registry',
            'scope': scope,
        })
        token = (_registry_get(token_url) or {}).get('token')
        if not token:
            return False, 'registry returned no token'

        payload = token.split('.')[1]
        payload += '=' * (-len(payload) % 4)
        access = json.loads(base64.urlsafe_b64decode(payload)).get('access', [])
        actions = {action for entry in access for action in entry.get('actions', [])}
        if 'pull' not in actions:
            return False, 'repository does not exist (anonymous token carries no pull action)'

        tags = (_registry_get(f'https://{registry}/v2/{repository}/tags/list', token) or {}).get('tags') or []
        if tag not in tags:
            return False, f'tag {tag!r} not pushed; present: {sorted(tags)[:8]}'
        return True, f'{len(tags)} tag(s) in repository'
    except urllib.error.HTTPError as exc:
        return False, f'HTTP {exc.code}'
    except Exception as exc:  # noqa: BLE001 - a preflight must report, not crash
        return False, f'{type(exc).__name__}: {exc}'


def stage_preflight(dataset_dir: Path) -> None:
    """Fail before the batch starts if any world image is missing.

    Without this the batch begins happily and dies only when it reaches the
    first task of the unpushed world, after having already spent sandboxes.
    """
    by_image = dataset_images(dataset_dir)
    log('preflight', f'{sum(len(v) for v in by_image.values())} task(s) across {len(by_image)} image(s)')

    blocked = 0
    for image, task_ids in sorted(by_image.items(), key=lambda kv: -len(kv[1])):
        exists, detail = resolve_image(image)
        status = 'OK     ' if exists else 'MISSING'
        log('preflight', f'{status} {image}  gates {len(task_ids)} task(s) -- {detail}')
        if not exists:
            blocked += len(task_ids)
    if blocked:
        raise EnterpriseBenchError(
            f'{blocked} task(s) would fail on a missing image. Push the image(s) above, '
            'or restrict the run with --worlds to the ones that exist.'
        )
    log('preflight', 'PASS - every image the dataset names is pullable')


# ----------------------------------------------------------------------------- run


def stage_run(args: argparse.Namespace, dataset_dir: Path, results_path: Path) -> None:
    if not args.use:
        raise EnterpriseBenchError(
            '--use is required to run: the dataset item carries the task, but the '
            'agent import, the gongfeng provider and the model client come from a '
            'compose part (default: enterprisebench_native/base.yaml).'
        )
    use_path = Path(args.use).expanduser().resolve()
    if not use_path.is_file():
        raise EnterpriseBenchError(f'--use file not found: {use_path}')

    # --data BEFORE --use BEFORE --set: compose parts merge in command-line
    # order, so this is what lets the compose part add the agent import and
    # the --set overrides win over the values baked into each item.
    command = [
        sys.executable,
        '-m',
        'nexus.cli.main',
        'run-batch',
        '--data',
        str(dataset_dir),
        '--use',
        str(use_path),
        '--set',
        f'runtime_configs.main.resource_spec.memory={args.memory}',
        '--set',
        f'runtime_configs.main.resource_spec.cpu={args.cpu}',
        '--set',
        f'task_config.artifact_root={Path(args.artifact_root).expanduser().resolve()}',
        '--max-concurrent',
        str(args.max_concurrent),
        '--worker-processes',
        str(args.worker_processes),
        '--export-format',
        'jsonl',
        '--export-path',
        str(results_path),
    ]
    if args.database:
        command += ['--database', str(args.database)]
    if args.retry_failed:
        command += ['--retry-failed', str(args.retry_failed)]
    if args.task_timeout:
        command += ['--timeout', str(args.task_timeout)]

    log('run', f'memory={args.memory}GB cpu={args.cpu} concurrency={args.max_concurrent}')
    log('run', f'results -> {results_path}')
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        # Not fatal: a batch exits non-zero when individual tasks fail, and the
        # export still holds every finished row, which is what we came for.
        log('run', f'run-batch exited {result.returncode}; reporting on whatever completed')


# -------------------------------------------------------------------------- report


def classify(run: dict) -> tuple[str, float | None]:
    """Return ``(outcome, score)`` for one exported run.

    ``outcome`` is ``graded`` / ``env_init`` / ``verifier`` / ``error``. Only
    ``graded`` carries a score, and only graded rows belong in a mean.
    """
    task_output = run.get('task_output') or {}
    error = run.get('error') or ''

    if _ENV_INIT_MARKER in error:
        return 'env_init', None
    if any(marker in error for marker in _VERIFIER_MARKERS):
        return 'verifier', None
    if run.get('status') != 'success':
        return 'error', None
    if task_output.get('verifier_error'):
        return 'verifier', None

    score = task_output.get('score')
    if score is None:
        return 'verifier', None
    return 'graded', float(score)


def _bucket(n_criteria: Any) -> str:
    """Group tasks by rubric size.

    A rubric is graded all-or-nothing, so a 26-criterion task and an
    8-criterion one each contribute one 0/1 while being very different
    measurements. Reporting the buckets keeps that visible.
    """
    try:
        count = int(n_criteria)
    except (TypeError, ValueError):
        return 'unknown'
    if count <= 10:
        return '1-10'
    if count <= 20:
        return '11-20'
    return '21+'


def aggregate(results_path: Path) -> dict[str, Any]:
    if not results_path.is_file():
        raise EnterpriseBenchError(f'no results at {results_path}; run the batch first')

    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise EnterpriseBenchError(f'{results_path} is empty')

    overall: dict[str, Any] = {
        'total': len(rows),
        'graded': 0,
        'env_init': 0,
        'verifier': 0,
        'error': 0,
        'scores': [],
        'rubric_accuracies': [],
    }
    per_world: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            'total': 0,
            'graded': 0,
            'env_init': 0,
            'verifier': 0,
            'error': 0,
            'scores': [],
            'rubric_accuracies': [],
        }
    )
    per_rubric: dict[str, dict[str, Any]] = defaultdict(lambda: {'graded': 0, 'scores': [], 'rubric_accuracies': []})
    failures: list[dict[str, str]] = []

    for row in rows:
        metadata = row.get('metadata') or {}
        task_output = row.get('task_output') or {}
        world = metadata.get('world') or 'unknown'
        task_id = metadata.get('task_id') or metadata.get('dataset_item_id') or row.get('run_id') or '?'
        outcome, score = classify(row)

        overall[outcome] += 1
        bucket = per_world[world]
        bucket['total'] += 1
        bucket[outcome] += 1

        if outcome == 'graded' and score is not None:
            overall['scores'].append(score)
            bucket['scores'].append(score)
            rubric_accuracy = task_output.get('rubric_accuracy')
            if rubric_accuracy is None:
                passed = task_output.get('criteria_passed')
                total = task_output.get('criteria_total')
                if (
                    isinstance(passed, (int, float))
                    and not isinstance(passed, bool)
                    and isinstance(total, (int, float))
                    and not isinstance(total, bool)
                    and total > 0
                ):
                    rubric_accuracy = float(passed) / float(total)
            if rubric_accuracy is not None:
                overall['rubric_accuracies'].append(float(rubric_accuracy))
                bucket['rubric_accuracies'].append(float(rubric_accuracy))
            rubric = per_rubric[_bucket(metadata.get('n_criteria'))]
            rubric['graded'] += 1
            rubric['scores'].append(score)
            if rubric_accuracy is not None:
                rubric['rubric_accuracies'].append(float(rubric_accuracy))
        else:
            failures.append({
                'task_id': str(task_id),
                'world': str(world),
                'outcome': outcome,
                'detail': (row.get('error') or (row.get('task_output') or {}).get('verifier_error') or '')
                .strip()
                .splitlines()[0][:200],
            })

    def summarise(bucket: dict[str, Any]) -> dict[str, Any]:
        scores = bucket.pop('scores')
        rubric_accuracies = bucket.pop('rubric_accuracies')
        bucket['mean_score'] = round(statistics.fmean(scores), 4) if scores else None
        bucket['mean_rubric_accuracy'] = round(statistics.fmean(rubric_accuracies), 4) if rubric_accuracies else None
        bucket['solved'] = sum(1 for value in scores if value >= 1.0)
        return bucket

    return {
        'overall': summarise(overall),
        'per_world': {world: summarise(bucket) for world, bucket in sorted(per_world.items())},
        'per_rubric_size': {
            name: {
                'graded': bucket['graded'],
                'mean_score': round(statistics.fmean(bucket['scores']), 4) if bucket['scores'] else None,
                'mean_rubric_accuracy': (
                    round(statistics.fmean(bucket['rubric_accuracies']), 4) if bucket['rubric_accuracies'] else None
                ),
            }
            for name, bucket in sorted(per_rubric.items())
        },
        'ungraded': failures,
        'scoring': (
            'one-or-nothing over RewardKit criteria; rubric_accuracy is '
            'criteria_passed / criteria_total; means exclude ungraded rows'
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    overall = report['overall']
    print()
    print('=' * 78)
    print('EnterpriseBench — rewardkit score')
    print('=' * 78)
    mean = overall['mean_score']
    print(f'mean_score   : {mean if mean is not None else "n/a"}   (over {overall["graded"]} graded row(s))')
    print(f'rubric_acc   : {overall["mean_rubric_accuracy"] if overall["mean_rubric_accuracy"] is not None else "n/a"}')
    print(f'solved       : {overall["solved"]}/{overall["graded"]}')
    print(
        f'ungraded     : {overall["env_init"]} env-init, {overall["verifier"]} verifier, '
        f'{overall["error"]} other   (excluded from the mean, not counted as 0)'
    )
    print(f'total rows   : {overall["total"]}')

    print()
    print(f'{"world":<20}{"graded":>8}{"solved":>8}{"mean":>8}{"env":>6}{"vrf":>6}{"err":>6}')
    print('-' * 62)
    for world, bucket in report['per_world'].items():
        mean_text = '—' if bucket['mean_score'] is None else f'{bucket["mean_score"]:.3f}'
        print(
            f'{world:<20}{bucket["graded"]:>8}{bucket["solved"]:>8}{mean_text:>8}'
            f'{bucket["env_init"]:>6}{bucket["verifier"]:>6}{bucket["error"]:>6}'
        )

    if report['per_rubric_size']:
        print()
        print('by rubric size (all-or-nothing grading, so these are not one measurement):')
        for name, bucket in report['per_rubric_size'].items():
            mean_text = '—' if bucket['mean_score'] is None else f'{bucket["mean_score"]:.3f}'
            print(f'  {name:<10} graded={bucket["graded"]:<5} mean={mean_text}')

    if report['ungraded']:
        print()
        print(f'ungraded rows ({len(report["ungraded"])}), first 10:')
        for entry in report['ungraded'][:10]:
            print(f'  [{entry["outcome"]:<8}] {entry["world"]:<18} {entry["task_id"]}  {entry["detail"]}')
    print()


# ----------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--delivery', help='Delivery root (the directory holding tasks/ and images/).')
    parser.add_argument('--workdir', required=True, help='Where the dataset, results and report are written.')
    parser.add_argument(
        '--artifact-root',
        default=None,
        help='Where per-run evidence is stored (default: <workdir>/artifacts).',
    )
    parser.add_argument(
        '--use',
        default=str(DEFAULT_COMPOSE),
        help='Compose part supplying the agent, provider and model client (default: enterprisebench_native/base.yaml).',
    )

    parser.add_argument('--image-prefix', default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument('--image-tag', default=DEFAULT_IMAGE_TAG)
    parser.add_argument('--agent-name', default=DEFAULT_AGENT_NAME)
    parser.add_argument(
        '--disable-world-tool-set',
        action='append',
        default=[],
        help='World tool set to disable while building the dataset. Repeat for more sets. Core is always disabled.',
    )
    parser.add_argument(
        '--worlds',
        default='',
        help='Comma-separated world filter, e.g. suitecrm,appflowy. Default: every task in the delivery.',
    )
    parser.add_argument('--tasks', default=None, help='File of task ids, one per line (ignored when --worlds is set).')
    parser.add_argument('--limit', type=int, default=None, help='Keep only the first N selected tasks.')

    parser.add_argument('--memory', type=int, default=DEFAULT_MEMORY_GB, help='Sandbox memory in GB.')
    parser.add_argument('--cpu', type=int, default=DEFAULT_CPU, help='Sandbox cpu cores.')
    parser.add_argument('--max-concurrent', type=int, default=4, help='Concurrent sandboxes.')
    parser.add_argument('--worker-processes', type=int, default=1)
    parser.add_argument('--retry-failed', type=int, default=0)
    parser.add_argument('--task-timeout', type=int, default=0, help='Per-task timeout in seconds (0 = none).')
    parser.add_argument('--database', default=None, help='SQLite path, so an interrupted batch can resume.')

    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Stop after preflight: builds the dataset and checks the images without spending a sandbox.',
    )
    parser.add_argument('--report-only', action='store_true', help='Only aggregate an existing results file.')
    args = parser.parse_args(argv)
    if not args.artifact_root:
        args.artifact_root = str(Path(args.workdir).expanduser().resolve() / 'artifacts')

    args.worlds = [world.strip() for world in args.worlds.split(',') if world.strip()]

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = workdir / 'dataset'
    results_path = workdir / 'results.jsonl'
    report_path = workdir / 'report.json'

    try:
        if not args.report_only:
            if not args.skip_build:
                if not args.delivery:
                    raise EnterpriseBenchError('--delivery is required to build the dataset (or pass --skip-build)')
                stage_build(args, dataset_dir)
            if not args.skip_preflight:
                stage_preflight(dataset_dir)
            if args.dry_run:
                log('dry-run', f'stopping before the batch; dataset is ready at {dataset_dir}')
                return 0
            stage_run(args, dataset_dir, results_path)

        report = aggregate(results_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print_report(report)
        log('report', f'written to {report_path}')
    except EnterpriseBenchError as exc:
        print(f'\nFAIL: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
