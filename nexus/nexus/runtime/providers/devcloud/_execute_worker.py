"""Standalone stdlib worker uploaded to a DevCloud VM (no Nexus installation needed)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from pathlib import Path


def write_state(directory: Path, state: dict) -> None:
    temporary = directory / 'state.tmp'
    temporary.write_text(json.dumps(state), encoding='utf-8')
    os.replace(temporary, directory / 'state.json')


def run(directory: Path) -> None:
    os.umask(0o077)
    spec = json.loads((directory / 'spec.json').read_text(encoding='utf-8'))
    state = {'job_id': spec['job_id'], 'state': 'running'}
    write_state(directory, state)
    try:
        # Keep OSWorld's execution/user semantics, but bypass the external HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            spec['url'],
            data=json.dumps(spec['payload']).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            response = opener.open(request, timeout=spec['timeout'])
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            state.update(state='completed', http_status=response.code, text=response.read().decode('utf-8', 'replace'))
    except Exception as exc:
        # A local HTTP failure does not prove that the underlying command was stopped.
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
    write_state(directory, state)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
