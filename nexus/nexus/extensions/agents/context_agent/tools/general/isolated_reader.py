"""Persistent unprivileged file-search process, separate from the user REPL."""

import json

from pathlib import Path
from typing import Any

from .state import GeneralState
from .tools.python_repl import PythonREPL


def _bootstrap(state: GeneralState) -> str:
    root = Path(__file__).parent
    sources = {
        name: (root / path).read_text()
        for name, path in {
            'reader.workspace': 'workspace.py',
            'reader.fileio': 'fileio.py',
            'reader.state': 'state.py',
            'reader.tools.read_lines': 'tools/read_lines.py',
            'reader.tools.grep': 'tools/grep.py',
            'reader.tools.file_search': 'tools/file_search.py',
        }.items()
    }
    config = {
        'root': str(state.workspace.root),
        'cwd': str(state.workspace.cwd),
        'docs_dir': str(state.workspace.docs_dir),
        'knobs': vars(state.knobs),
    }
    return f"""
import sys, types, json
from pathlib import Path
for name in ('reader', 'reader.tools'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
for name, source in {sources!r}.items():
    module = types.ModuleType(name)
    module.__file__ = '/isolated/nexus/extensions/agents/context_agent/tools/general/' + name.split('.')[-1] + '.py'
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
from reader.state import GeneralState, Knobs
from reader.workspace import Workspace
cfg = {config!r}
sys.modules['reader'].state = GeneralState(workspace=Workspace(root=Path(cfg['root']), cwd=cfg['cwd'], docs_dir=cfg['docs_dir']), knobs=Knobs(**cfg['knobs']), declared_files=[])
"""


async def dispatch_reader(state: GeneralState, name: str, arguments: dict[str, Any]) -> str:
    if state._file_reader is None:
        server = Path(__file__).parent / 'tools/repl_server.py'
        state._file_reader = PythonREPL(
            cwd=state.workspace.cwd,
            timeout=state.knobs.python_timeout,
            server_source=_bootstrap(state)
            + '\n'
            + server.read_text().replace('from __future__ import annotations', ''),
        )
    calls = {
        'read_lines': ('reader.tools.read_lines', 'read_lines'),
        'grep': ('reader.tools.grep', 'grep'),
        'file_search_msearch': ('reader.tools.file_search', 'msearch'),
        'file_search_mclick': ('reader.tools.file_search', 'mclick'),
    }
    module, function = calls[name]
    args = dict(arguments)
    if name == 'file_search_msearch' and isinstance(args.get('queries'), str):
        args['queries'] = [args['queries']]
    if name == 'file_search_mclick' and isinstance(args.get('pointers'), str):
        args['pointers'] = [args['pointers']]
    code = (
        f'import importlib, json, reader\n'
        f'print(getattr(importlib.import_module({module!r}), {function!r})(reader.state, **json.loads({json.dumps(args)!r})))'
    )
    return await state._file_reader.exec(code)
