#!/usr/bin/env python3
"""Patch email_config.json files to point to an external Poste.io service.

When Email runs as a BenchmarkService in a separate container, the main
container can no longer reach ``localhost:1143`` / ``localhost:1587``.
This script rewrites all ``email_config.json`` files under ``--config-dir``
to use the actual service host and port.

Usage::

    python3 toolathlon_email_config_patcher.py \
        --config-dir /workspace/mcp-benchmark/toolathlon/api/apply-phd-email \
        --imap-endpoint host:port \
        --smtp-endpoint host:port
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _find_email_configs(root: str) -> list[str]:
    """Recursively find all email-related config JSON files.

    Matches:
    - email_config.json   (most common, used by emails-mcp)
    - emails_config.json  (used by notion-hr and similar cases)
    - receiver_config.json (used by evaluation to verify received emails)
    - example_email_config.json (credential bundle config for emails-mcp)
    """
    target_names = {
        'email_config.json',
        'emails_config.json',
        'receiver_config.json',
        'example_email_config.json',
    }
    result: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name in target_names:
                result.append(os.path.join(dirpath, name))
    return result


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse 'host:port' into (host, port)."""
    parts = endpoint.rsplit(':', 1)
    if len(parts) != 2:
        raise ValueError(f'invalid endpoint format (expected host:port): {endpoint}')
    return parts[0], int(parts[1])


def _patch_config(path: str, imap_host: str, imap_port: int, smtp_host: str, smtp_port: int) -> bool:
    """Patch a single email_config.json.  Returns True if modified."""
    with open(path, encoding='utf-8') as f:
        config = json.load(f)

    changed = False
    for field, value in [
        ('imap_server', imap_host),
        ('imap_port', imap_port),
        ('smtp_server', smtp_host),
        ('smtp_port', smtp_port),
    ]:
        if field in config and config[field] != value:
            config[field] = value
            changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            f.write('\n')

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description='Patch email_config.json for remote Poste.io')
    parser.add_argument('--config-dir', required=True, help='root dir to search for email_config.json')
    parser.add_argument('--imap-endpoint', required=True, help='IMAP endpoint as host:port')
    parser.add_argument('--smtp-endpoint', required=True, help='SMTP endpoint as host:port')
    args = parser.parse_args()

    imap_host, imap_port = _parse_endpoint(args.imap_endpoint)
    smtp_host, smtp_port = _parse_endpoint(args.smtp_endpoint)

    configs = _find_email_configs(args.config_dir)
    if not configs:
        print(f'No email config files found under {args.config_dir}', file=sys.stderr)
        return

    patched = 0
    for path in configs:
        if _patch_config(path, imap_host, imap_port, smtp_host, smtp_port):
            patched += 1
            print(f'Patched: {path}')
        else:
            print(f'Unchanged: {path}')

    print(f'Done: {patched}/{len(configs)} files patched (IMAP={imap_host}:{imap_port}, SMTP={smtp_host}:{smtp_port})')


if __name__ == '__main__':
    main()
