#!/usr/bin/env python3
"""aidata rubric-import E2E 测试的 CI 运行器。

先用沙箱/mockserver 运行 ``test_e2e_rubrics_import.py``，随后仅对失败的
case id 以逐步降低的 xdist 并发度重试：

    round 1: -n 128  （或 --workers）
    round 2: -n 64
    round 3: -n 32
    ...

重试循环有意放在 pytest 之外。 它无需依赖 pytest 插件即可区分稳定的 case
失败与短暂的沙箱 / MCP 传输抖动，可用于 CI 或本地 shell。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


TEST_FILE = 'nexus/extensions/tasks/aidata/tests/test_e2e_rubrics_import.py'
FAILED_CASE_RE = re.compile(r'^FAILED\s+.*\[(?P<case_id>.+)](?:\s|$)')
SUMMARY_RE = re.compile(r'(?P<failed>\d+)\s+failed,\s+(?P<passed>\d+)\s+passed')


@dataclass
class RoundResult:
    returncode: int
    failed_cases: list[str]
    summary: str


def _repo_root() -> Path:
    # tests/e2e_rubrics_ci.py -> aidata -> tasks -> extensions -> nexus 包 -> 仓库根
    return Path(__file__).resolve().parents[5]


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export ') :].lstrip()
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_failed_cases(output: str) -> list[str]:
    cases: list[str] = []
    for line in output.splitlines():
        match = FAILED_CASE_RE.match(line.strip())
        if match:
            cases.append(match.group('case_id'))
    return _dedupe(cases)


def _parse_summary(output: str) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if ' passed' in stripped or ' failed' in stripped or ' skipped' in stripped:
            return stripped
    return '(no pytest summary found)'


def _run_round(
    *,
    round_index: int,
    workers: int,
    cases: list[str] | None,
    args: argparse.Namespace,
    env: dict[str, str],
) -> RoundResult:
    round_env = dict(env)
    round_env['AIDATA_CASES_ROOT'] = str(args.cases_root)
    if cases is None:
        round_env['AIDATA_E2E_ALL_RUBRICS'] = '1' if args.all else ''
        round_env.pop('AIDATA_E2E_RUBRIC_CASES', None)
    else:
        round_env.pop('AIDATA_E2E_ALL_RUBRICS', None)
        round_env['AIDATA_E2E_RUBRIC_CASES'] = ','.join(cases)

    cmd = [
        sys.executable,
        '-m',
        'pytest',
        TEST_FILE,
        '-n',
        str(workers),
        '--dist',
        'loadgroup',
        '-q',
        '--tb=short',
    ]
    if args.maxfail:
        cmd.append(f'--maxfail={args.maxfail}')
    if args.extra_pytest_args:
        cmd.extend(args.extra_pytest_args)

    target = 'all committed rubrics' if cases is None else f'{len(cases)} failed case(s)'
    print(f'\n[e2e-rubrics-ci] round {round_index}: -n {workers}, target={target}', flush=True)
    proc = subprocess.run(
        cmd,
        cwd=args.repo_root,
        env=round_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end='', flush=True)

    failed_cases = _parse_failed_cases(proc.stdout)
    summary = _parse_summary(proc.stdout)
    print(
        f'[e2e-rubrics-ci] round {round_index} done: rc={proc.returncode}, '
        f'failed_cases={len(failed_cases)}, summary={summary}',
        flush=True,
    )
    return RoundResult(proc.returncode, failed_cases, summary)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be > 0')
    return value


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=root)
    parser.add_argument(
        '--cases-root',
        type=Path,
        default=Path(os.getenv('AIDATA_CASES_ROOT', '/data/workspace/rlenv/Tasks')),
    )
    parser.add_argument('--env-file', type=Path, action='append', default=[], help='Load KEY=VALUE env file(s).')
    parser.add_argument('--workers', '-n', type=_positive_int, default=128)
    parser.add_argument('--min-workers', type=_positive_int, default=1)
    parser.add_argument('--max-rounds', type=_positive_int, default=4)
    parser.add_argument('--maxfail', type=_positive_int, default=0, help='Optional pytest --maxfail for each round.')
    parser.add_argument(
        '--all',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Round 1 runs all committed rubrics. Disable only when AIDATA_E2E_RUBRIC_CASES is already set.',
    )
    parser.add_argument('extra_pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.extra_pytest_args and args.extra_pytest_args[0] == '--':
        args.extra_pytest_args = args.extra_pytest_args[1:]

    args.repo_root = args.repo_root.resolve()
    args.cases_root = args.cases_root.resolve()

    env = os.environ.copy()
    for env_file in args.env_file:
        env_file = env_file.resolve()
        if not env_file.is_file():
            parser.error(f'--env-file not found: {env_file}')
        env.update(_parse_env_file(env_file))

    current_cases: list[str] | None
    if args.all:
        current_cases = None
    else:
        raw_cases = env.get('AIDATA_E2E_RUBRIC_CASES', '')
        current_cases = [item.strip() for item in raw_cases.split(',') if item.strip()]
        if not current_cases:
            parser.error('--no-all requires AIDATA_E2E_RUBRIC_CASES to be set')

    workers = max(args.min_workers, args.workers)
    final_result: RoundResult | None = None
    for round_index in range(1, args.max_rounds + 1):
        final_result = _run_round(
            round_index=round_index,
            workers=workers,
            cases=current_cases,
            args=args,
            env=env,
        )
        if final_result.returncode == 0:
            print('[e2e-rubrics-ci] success: no remaining failures', flush=True)
            return 0
        if not final_result.failed_cases:
            print('[e2e-rubrics-ci] pytest failed but no failed case ids were parsed; stopping', flush=True)
            return final_result.returncode
        if round_index == args.max_rounds:
            break
        current_cases = final_result.failed_cases
        workers = max(args.min_workers, workers // 2)

    assert final_result is not None
    print('\n[e2e-rubrics-ci] stable failures after retries:', flush=True)
    for case_id in final_result.failed_cases:
        print(f'  - {case_id}', flush=True)
    return final_result.returncode or 1


if __name__ == '__main__':
    raise SystemExit(main())
