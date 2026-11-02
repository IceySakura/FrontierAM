
from __future__ import annotations

import sys
import tomllib

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nexus.extensions.agents.codex import runtime_setup  # noqa: E402


def _render(server: dict) -> dict:
    return tomllib.loads('\n'.join(runtime_setup._mcp_config_lines([server])))


def test_worker_placeholders_resolve_from_this_process(monkeypatch):
    monkeypatch.setattr(runtime_setup.sys, 'executable', '/tmp/nexus/runtime/venvs/ab12cd34ef56/bin/python')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/nix/runtime/nix-env/lib')

    config = _render({
        'name': 'ppt12',
        'command': '${NEXUS_WORKER_PYTHON}',
        'args': ['-m', 'nexus.extensions.tasks.ppt_rubric.ppt_mcp_server'],
        'env': {
            'PPT12_WORKSPACE': '/workspace',
            'LD_LIBRARY_PATH': '${NEXUS_LD_LIBRARY_PATH}',
        },
    })

    server = config['mcp_servers']['ppt12']
    assert server['command'] == '/tmp/nexus/runtime/venvs/ab12cd34ef56/bin/python'
    assert server['env']['LD_LIBRARY_PATH'] == '/nix/runtime/nix-env/lib'
    assert server['env']['PPT12_WORKSPACE'] == '/workspace'


def test_placeholders_resolve_recursively_inside_args(monkeypatch):
    monkeypatch.setattr(runtime_setup.sys, 'executable', '/venv/bin/python')

    config = _render({
        'name': 'shim',
        'command': '/bin/sh',
        'args': ['-c', 'exec ${NEXUS_WORKER_PYTHON} -m mod'],
    })

    assert config['mcp_servers']['shim']['args'][1] == 'exec /venv/bin/python -m mod'


def test_entries_without_placeholders_render_unchanged():
    """Negative control: the resolver must not rewrite anything else -- unknown
    ${...} tokens survive, and non-string fields keep their JSON encoding."""
    config = _render({
        'name': 'plain.server',
        'command': 'uv',
        'args': ['run', 'x.py'],
        'env': {'NOTE': 'keep ${NEXUS_UNRELATED} as-is'},
        'startup_timeout_sec': 300,
        'required': True,
    })

    server = config['mcp_servers']['plain.server']
    assert server['command'] == 'uv'
    assert server['args'] == ['run', 'x.py']
    assert server['env'] == {'NOTE': 'keep ${NEXUS_UNRELATED} as-is'}
    assert server['startup_timeout_sec'] == 300
    assert server['required'] is True


def test_missing_ld_library_path_resolves_to_an_empty_string(monkeypatch):
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)

    config = _render({
        'name': 'p',
        'command': '${NEXUS_WORKER_PYTHON}',
        'env': {'LD_LIBRARY_PATH': '${NEXUS_LD_LIBRARY_PATH}'},
    })

    assert config['mcp_servers']['p']['env']['LD_LIBRARY_PATH'] == ''
