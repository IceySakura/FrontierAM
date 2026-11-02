from __future__ import annotations

import os
import subprocess
import sys
import types

from pathlib import Path

import pytest


def test_count_tokens_falls_back_when_tiktoken_init_fails(monkeypatch):
    from nexus.extensions.agents.browser_gym import token_counting

    token_counting.get_cl100k_encoding.cache_clear()

    fake_tiktoken = types.SimpleNamespace(
        get_encoding=lambda _name: (_ for _ in ()).throw(TimeoutError('network timeout'))
    )
    monkeypatch.setitem(sys.modules, 'tiktoken', fake_tiktoken)

    assert token_counting.count_tokens_with_fallback('abcdefgh', chars_per_token=2) == 4

    token_counting.get_cl100k_encoding.cache_clear()


@pytest.mark.skipif(sys.executable is None, reason='Python executable is required for subprocess import test')
def test_importing_swe_does_not_initialize_tiktoken_encoding():
    repo_root = Path(__file__).resolve().parents[5]
    script = """
import importlib
import tiktoken

calls = []
orig = tiktoken.get_encoding

def wrapped(name):
    calls.append(name)
    raise AssertionError(f'unexpected get_encoding({name})')

tiktoken.get_encoding = wrapped
importlib.import_module('nexus.extensions.tasks.swe')
print(f'calls={calls!r}')
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get('PYTHONPATH', '')

    proc = subprocess.run(
        [sys.executable, '-c', script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert 'calls=[]' in proc.stdout
