from __future__ import annotations

import subprocess

from pathlib import Path


def test_repository_contains_no_tencent_email_addresses() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    forbidden = b'@' + b'tencent.com'
    violations: list[str] = []

    for raw_path in tracked.split(b'\0'):
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode())
        path = repo_root / relative_path
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
            if forbidden in line.lower():
                violations.append(f'{relative_path}:{line_number}')

    assert not violations, f'Found {len(violations)} forbidden corporate email addresses:\n' + '\n'.join(
        violations[:20]
    )
