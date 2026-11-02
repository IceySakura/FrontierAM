"""Path-native tool core for the ``context_agent_general`` agent.

The premise is that **disk is the source of truth**. The agent is handed nothing
but a URI — a URL or a local path — and tools take paths and read them on
demand, so there is no "loaded document" set and nothing to register before a
tool will work. What to fetch, what to convert, and how much to read are all
the agent's decisions; the system prompt supplies the discipline (check metadata
first, never read a large file whole), not the tool layer.

This is the opposite of the content-bearing design it grew out of, where
documents arrived inside a JSONL row, were rendered in memory, and every tool
resolved a ``file`` name against that in-memory list. Several comments below
still contrast the two, because the divergences are deliberate rather than
incidental.

Seven tools: ``curl``, ``bash``, ``python``, ``file_search_msearch``,
``file_search_mclick``, ``grep``, ``read_lines``. Anything derivable from a few
lines of shell or Python — listing files, outlining headings — is deliberately
left to ``bash`` / ``python`` rather than given a tool of its own.

The upstream chain carried an eighth tool, ``read_image``, which answered a
question about a PNG by calling a vision model. It is dropped here: it was the
only tool that needed a second model handle, and a harness whose entire model
contract is one ``chat_with_tools`` method has nowhere to put one. There is no
native vision call in this port; an experimental system prompt may still direct
the policy to recover image evidence with installed OCR and image-processing
packages through the ordinary bash/Python tools.
"""

from .fileio import (
    FileCache,
    FileReadError,
    iter_text_files,
    resolve_path,
)
from .registry import dispatch_tool
from .schemas import GENERAL_TOOL_ORDER, TOOL_SCHEMA_REGISTRY, build_tool_schemas
from .state import (
    REPO_ROOT,
    DeclaredFile,
    GeneralState,
    Knobs,
    format_size,
    parse_declared_files,
    stage_declared_files,
)
from .workspace import Workspace, prepare_workspace, sanitize_filename


__all__ = [
    'DeclaredFile',
    'FileCache',
    'FileReadError',
    'GENERAL_TOOL_ORDER',
    'GeneralState',
    'Knobs',
    'REPO_ROOT',
    'TOOL_SCHEMA_REGISTRY',
    'Workspace',
    'build_tool_schemas',
    'dispatch_tool',
    'format_size',
    'iter_text_files',
    'parse_declared_files',
    'prepare_workspace',
    'resolve_path',
    'sanitize_filename',
    'stage_declared_files',
]
