"""Score a frozen STEP in a fresh AGS grader without running the agent."""

import argparse
import asyncio
import hashlib
import json
import logging
import shutil

from pathlib import Path

from dotenv import dotenv_values

from nexus.extensions.tasks.os_world_v2.adapter.cad_grader import CADGrader, CADGraderOptions
from nexus.extensions.tasks.os_world_v2.adapter.cad_guest import FILES
from nexus.extensions.tasks.os_world_v2.adapter.source import validate_task_assets, verify
from nexus.runtime.providers.ags.provider import AGSRuntimeProvider


async def replay(args):
    env = dotenv_values(args.env)
    verify(args.repo)
    validate_task_assets(args.repo, args.assets, args.task_id)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    directory = args.output / 'frozen'
    directory.mkdir(mode=0o700)
    for name in FILES:
        original = (
            args.submission
            if name == 'submission.step'
            else args.repo / f'evaluation_examples/task_class/task_{args.task_id}.py'
            if name == 'task.py'
            else args.assets / f'task_{args.task_id}' / name
        )
        shutil.copyfile(original, directory / name)
        (directory / name).chmod(0o600)
    (directory / 'manifest.json').write_text(
        json.dumps({
            'task_id': args.task_id,
            'sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES},
        })
    )
    provider = AGSRuntimeProvider(
        secret_id=env['AGS_SECRET_ID'],
        secret_key=env['AGS_SECRET_KEY'],
        region=env.get('AGS_REGION') or 'ap-guangzhou',
    )
    grader = CADGrader(
        provider,
        CADGraderOptions(
            tool_id=args.tool_id or env.get('OSWORLD_CAD_GRADER_TOOL_ID') or None,
            timeout=args.timeout,
            install_dependencies=not args.preinstalled,
        ),
        main_tool_id=env['AGS_OSWORLD_TOOL_ID'],
        password=env.get('OSWORLD_CLIENT_PASSWORD') or 'osworld-public-evaluation',
        cache=args.output,
        root=args.repo,
        assets=args.assets,
    )
    result = await grader(directory=directory, task_id=args.task_id)
    (args.output / 'report.json').write_text(json.dumps(result['report'], indent=2))
    print(
        json.dumps({
            'score': result['report']['score'],
            'submission_sha256': result['report']['submission_sha256'],
            'evaluation_execution': result['report']['evaluation_execution'],
            'report': str(args.output / 'report.json'),
        })
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--task-id', choices=('103', '104'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New private directory; refuses overwrites')
    parser.add_argument('--tool-id')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--preinstalled', action='store_true')
    args = parser.parse_args()
    if args.submission.is_symlink():
        parser.error('submission must not be a symlink')
    for name in ('repo', 'assets', 'submission', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.submission.is_file() or not 0 < args.submission.stat().st_size <= 100 * 1024**2:
        parser.error('submission must be a nonempty STEP file of at most 100 MiB')
    logging.disable(logging.CRITICAL)
    asyncio.run(replay(args))


if __name__ == '__main__':
    main()
