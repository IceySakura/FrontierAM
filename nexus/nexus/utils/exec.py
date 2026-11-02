"""Built-in helpers invokable from the client via import-path.

These are regular functions — no server changes needed.  The client
calls ``invoke('nexus.runtime._builtins::exec_code', code)`` and the
existing dispatch machinery handles the rest.
"""

from __future__ import annotations

import textwrap

from typing import Any


async def exec_code(code: str) -> Any:
    """Execute a Python code string and return its result.

    The code is wrapped in an ``async def`` so ``return`` and ``await``
    work naturally::

        result = await runtime.exec('''
            import os
            return os.listdir('/')
        ''')
    """
    code = textwrap.dedent(code).strip()
    indented = textwrap.indent(code, '    ')
    wrapped = f'async def __fn__():\n{indented}'
    ns: dict[str, Any] = {}
    exec(compile(wrapped, '<exec>', 'exec'), ns)  # noqa: S102
    return await ns['__fn__']()
