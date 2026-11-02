#!/usr/bin/env python3
"""Bulk-create Poste.io email accounts from compressed JSON.

Usage inside a Poste.io container:

    python3 toolathlon_email_user_creator.py --b64-gz "$NEXUS_EMAIL_USERS_B64"

The *b64-gz* argument is a base64-encoded gzip of a JSON array:

    [{"e": "user@mcp.com", "p": "password", "n": "Full Name"}, ...]

The script shells out to ``php /opt/admin/bin/console email:create`` to
create each account.  Failures are logged but do **not** abort the
process (Poste.io returns non-zero when a mailbox already exists).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed


PHP_CREATE = ['php', '/opt/admin/bin/console', 'email:create']
PHP_ADMIN = ['php', '/opt/admin/bin/console', 'email:admin']
PHP_DOMAIN = ['php', '/opt/admin/bin/console', 'domain:create']

ADMIN_EMAIL = 'mcpposte_admin@mcp.com'
ADMIN_PASSWORD = 'mcpposte'
ADMIN_NAME = 'System Administrator'


def _create_user(email: str, password: str, name: str) -> tuple[str, bool, str]:
    """Create a single mailbox.  Returns ``(email, ok, message)``."""
    try:
        result = subprocess.run(
            [*PHP_CREATE, email, password, name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        ok = result.returncode == 0
        msg = (result.stdout + result.stderr).strip()
        return email, ok, msg
    except Exception as exc:
        return email, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--b64-gz', required=True, help='base64-encoded gzip of users JSON array')
    parser.add_argument('--max-concurrent', type=int, default=50)
    args = parser.parse_args()

    # Decode users
    raw = gzip.decompress(base64.b64decode(args.b64_gz))
    users: list[dict[str, str]] = json.loads(raw)
    print(f'Decoded {len(users)} users to create', flush=True)

    # Ensure domain exists
    subprocess.run([*PHP_DOMAIN, 'mcp.com'], capture_output=True, check=False)

    # Create admin account
    email, ok, msg = _create_user(ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_NAME)
    if ok:
        subprocess.run([*PHP_ADMIN, ADMIN_EMAIL], capture_output=True, check=False)
        print(f'Admin created: {ADMIN_EMAIL}', flush=True)
    else:
        print(f'Admin skipped (may exist): {msg}', flush=True)

    # Bulk create regular users
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {pool.submit(_create_user, u['e'], u['p'], u['n']): u['e'] for u in users}
        for fut in as_completed(futures):
            _email, ok, _msg = fut.result()
            if ok:
                success += 1
            else:
                failed += 1

    print(f'Done: {success} created, {failed} skipped/failed (total {len(users)})', flush=True)

    if success == 0 and failed == len(users):
        print('WARNING: all users failed to create', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
