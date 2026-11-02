from __future__ import annotations

import sys

from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_code_runner_logs.py <dist_dir>')

    root = Path(sys.argv[1])
    for path in root.glob('*.js'):
        text = path.read_text()
        updated = text.replace('console.debug(', 'console.error(')
        if updated != text:
            path.write_text(updated)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
