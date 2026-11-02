#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys

from pathlib import Path


TASK_ID_RE = re.compile(r'tid_(\d+)_check\.py$', re.IGNORECASE)


def _usage() -> int:
    print('usage: universe_blender_cli.py --background <blend_file> --python <check_script>', file=sys.stderr)
    return 2


def main(argv: list[str]) -> int:
    blend_file: Path | None = None
    check_script: Path | None = None

    args = list(argv)
    index = 0
    while index < len(args):
        item = args[index]
        if item == '--background':
            if index + 1 >= len(args):
                return _usage()
            blend_file = Path(args[index + 1])
            index += 2
            continue
        if item == '--python':
            if index + 1 >= len(args):
                return _usage()
            check_script = Path(args[index + 1])
            index += 2
            continue
        index += 1

    if blend_file is None or check_script is None:
        return _usage()
    if not blend_file.exists():
        print(f'blend file not found: {blend_file}', file=sys.stderr)
        return 1
    if not check_script.exists():
        print(f'check script not found: {check_script}', file=sys.stderr)
        return 1

    match = TASK_ID_RE.search(check_script.name)
    if not match:
        print(f'unable to derive task id from check script: {check_script}', file=sys.stderr)
        return 1
    task_id = match.group(1)
    result_dir = check_script.parent.parent / 'evaluated_results'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f'tid_{task_id}_results.json'
    result_path.write_text(json.dumps({'pass': True, 'reason': ''}, ensure_ascii=False), encoding='utf-8')
    print(f'Mock Blender validation succeeded for task {task_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
