"""Load the sealed host runtime produced by ./bench prepare."""

from __future__ import annotations

import hashlib
import json
import os
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# A second sealed runtime can serve new work while jobs started on the first one are still reading it.
RUNTIME = Path(os.environ.get('FRONTIERAM_RUNTIME') or ROOT / '.runtime').resolve()


def sha(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_runtime() -> dict:
    if not (RUNTIME / 'READY').exists():
        raise RuntimeError('Run ./bench prepare first')
    config = json.loads((RUNTIME / 'runtime-lock.json').read_text())
    for relative, expected in config['code_sha256'].items():
        if sha(RUNTIME / relative) != expected:
            raise RuntimeError('Prepared integration changed: ' + relative)
    if sha(config['codex_path']) != config['codex_sha256']:
        raise RuntimeError('Codex executable changed after preparation')
    nexus = str(RUNTIME / 'nexus')
    if sys.path[0] != nexus:
        sys.path.insert(0, nexus)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    config['lock_sha256'] = sha(RUNTIME / 'runtime-lock.json')
    return config


def child_environment() -> dict:
    """Children import the sealed Nexus copy first, then this orchestration package."""
    environment = dict(os.environ)
    environment['PYTHONPATH'] = f'{RUNTIME / "nexus"}:{ROOT}'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    return environment
