"""Claude Code style tool surface for the white-box CC search agent.

This package exposes four tools, one per submodule, plus an internal
:mod:`backend` library used by the two web tools:

- :mod:`web_search` — Claude Code ``WebSearch`` (Serper-backed)
- :mod:`web_fetch` — Claude Code ``WebFetch`` (Jina + LLM extractor)
- :mod:`bash` — Minimal Anthropic ``bash`` tool. Execution is delegated
  to codeact's persistent terminal session.
- :mod:`execute_code` — Jupyter-kernel-backed Python REPL exposed to the
  LLM as ``ExecuteCode`` (one-to-one port of Cursor's
  ``mcp__ide__executeCode`` contract).
- :mod:`backend` — :class:`CCSearchBackend` library used by the two web
  tools; not registered as a Nexus tool.

Importing this package eagerly imports the four tool modules so their
``@tool_registry.register`` decorators run, which is the single entry
point listed in ``pyproject.toml``'s ``[tool.nexus.tools]`` table for
each registered tool name.

The flat re-exports below preserve the previous
``from nexus.extensions.agents.cc_search_agent_whitebox.tools import ...``
import paths so callers (e.g. :mod:`agent`) keep working unchanged.
"""

from .backend import (
    BackendError,
    CCSearchBackend,
    FetchResult,
    MissingCredentialsError,
    SearchHit,
    SearchResult,
)
from .bash import (
    BASH_DESCRIPTION,
    BASH_TOOL_NAME,
    BASH_TOOL_REGISTRY_NAME,
    BashInput,
    bash_tool,
)
from .execute_code import (
    EXECUTE_CODE_DESCRIPTION,
    EXECUTE_CODE_TOOL_NAME,
    EXECUTE_CODE_TOOL_REGISTRY_NAME,
    ExecuteCodeInput,
    execute_code_tool,
)
from .web_fetch import (
    WEB_FETCH_DESCRIPTION,
    WEB_FETCH_TOOL_NAME,
    WEB_FETCH_TOOL_REGISTRY_NAME,
    WebFetchInput,
    web_fetch_tool,
)
from .web_search import (
    WEB_SEARCH_DESCRIPTION,
    WEB_SEARCH_TOOL_NAME,
    WEB_SEARCH_TOOL_REGISTRY_NAME,
    WebSearchInput,
    web_search_tool,
)


__all__ = [
    'BASH_DESCRIPTION',
    'BASH_TOOL_NAME',
    'BASH_TOOL_REGISTRY_NAME',
    'BackendError',
    'BashInput',
    'CCSearchBackend',
    'EXECUTE_CODE_DESCRIPTION',
    'EXECUTE_CODE_TOOL_NAME',
    'EXECUTE_CODE_TOOL_REGISTRY_NAME',
    'ExecuteCodeInput',
    'FetchResult',
    'MissingCredentialsError',
    'SearchHit',
    'SearchResult',
    'WEB_FETCH_DESCRIPTION',
    'WEB_FETCH_TOOL_NAME',
    'WEB_FETCH_TOOL_REGISTRY_NAME',
    'WEB_SEARCH_DESCRIPTION',
    'WEB_SEARCH_TOOL_NAME',
    'WEB_SEARCH_TOOL_REGISTRY_NAME',
    'WebFetchInput',
    'WebSearchInput',
    'bash_tool',
    'execute_code_tool',
    'web_fetch_tool',
    'web_search_tool',
]
