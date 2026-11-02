"""Assemble the ppt12 stdio MCP server entry for the codex arm of a ``ppt_topic`` row.

``agent_tools: ["ppt12"]`` is the single row-level switch both arms share: the
built-in loop expands it into registered tool names, while an installed codex
harness gets the same twelve tools mounted over MCP. This helper turns the row
into the ``mcp_servers`` entry the codex agent already accepts
(``nexus/extensions/agents/codex/agent.py``), so no yaml is touched and the same
dataset rows can move between arms unchanged.

The command must reference the worker venv python, whose path is random per
sandbox (``/tmp/nexus/runtime/venvs/<vid>/bin/python``) -- unknowable from a
static yaml. The entry therefore uses ``${NEXUS_WORKER_PYTHON}`` and
``${NEXUS_LD_LIBRARY_PATH}``, resolved at config-render time inside the worker
process by ``nexus/extensions/agents/codex/runtime_setup._mcp_config_lines``.
The LD_LIBRARY_PATH placeholder matters: codex's own subprocess env strips the
nix runtime libs (``get_clean_sandbox_env``), and the venv python needs them to
import numpy/PIL at all.
"""

from __future__ import annotations

import logging
import posixpath

from typing import TYPE_CHECKING, Any

from nexus.extensions.tasks.mm_agent.utils import TaskArtifact


if TYPE_CHECKING:
    from .task import PPTTaskInput


logger = logging.getLogger(__name__)


MCP_SERVER_NAME = 'ppt12'
MCP_SERVER_MODULE = 'nexus.extensions.tasks.ppt_rubric.ppt_mcp_server'
WORKER_PYTHON_PLACEHOLDER = '${NEXUS_WORKER_PYTHON}'
LD_LIBRARY_PATH_PLACEHOLDER = '${NEXUS_LD_LIBRARY_PATH}'

# First startup imports nexus + PIL + the vendored font machinery; one call may
# run a 180s soffice plus a 300s rasterisation.
_STARTUP_TIMEOUT_SEC = 120
_TOOL_TIMEOUT_SEC = 600


def build_ppt12_mcp_server(
    *,
    agent_name: str,
    task_input: PPTTaskInput,
    artifacts: list[TaskArtifact],
    runtime: Any,
) -> dict[str, Any] | None:
    """Build the codex ``mcp_servers`` entry serving the 12 slide tools.

    Raises ValueError for non-codex harnesses: the ppt12 switch is a codex-only
    mount today (claude_code has the same ``mcp_servers`` input but is left for a
    later change), and letting another harness run on with no tools would put a
    silently tool-less arm into the scores -- the same reasoning that makes
    task.py raise for installed-harness-with-images and unknown agent names.
    Also raises when no worker venv exists: the MCP subprocess must run the
    venv python (numpy/PIL/python-pptx), and failing loud here turns a
    silent import crash inside the sandbox into a one-line fix (configure
    ``runtime_configs.main.worker_venv`` with ``modules`` including
    ``ppt_rubric``).

    The returned entry replaces any yaml-configured ``mcp_servers`` for the row
    (runner merge order: agent_config < task kwargs), which this task family
    relies on -- no yaml configures that key today.
    """
    normalized = (agent_name or '').strip().lower()
    if normalized != 'codex':
        raise ValueError(
            'ppt_topic: agent_tools includes "ppt12" but the installed harness '
            f'{agent_name!r} does not mount it (codex only today). Configure the '
            'row for codex, or drop "ppt12" from agent_tools -- running the arm '
            'without any tools would silently skew the comparison.'
        )
    if not getattr(runtime, 'default_venv_id', None):
        raise ValueError(
            'ppt_topic: serving the ppt12 tools to codex over MCP needs the nexus worker venv, '
            'but this runtime has none. Configure runtime_configs.main.worker_venv with '
            'modules including "ppt_rubric" so the sandbox worker -- and the MCP subprocess it '
            'spawns -- can import numpy/PIL/python-pptx.'
        )

    working_dir = task_input.working_dir
    env: dict[str, str] = {
        'PPT12_WORKSPACE': working_dir,
        # Raw, not basename: the server joins it onto the workspace exactly like
        # the built-in loop's os.path.join(work, deck_filename), so sub-directory
        # and absolute deck filenames keep their loop semantics.
        'PPT12_DECK_FILENAME': task_input.deck_filename,
        'PPT12_DUMP_DIR': posixpath.join(working_dir, 'ppt12_toolimg'),
        'PPT12_MCP_LOG': posixpath.join(working_dir, 'ppt12_mcp.log'),
        'LD_LIBRARY_PATH': LD_LIBRARY_PATH_PLACEHOLDER,
    }
    reference = next((artifact.target for artifact in artifacts if artifact.is_image), '')
    if reference:
        env['PPT12_REFERENCE_PNG'] = reference

    logger.warning(
        '[ppt_rubric] mounting mcp_servers.%s for codex (agent_tools=["ppt12"]); the task-side '
        'value replaces any mcp_servers configured in yaml agent_config.',
        MCP_SERVER_NAME,
    )
    return {
        'name': MCP_SERVER_NAME,
        'command': WORKER_PYTHON_PLACEHOLDER,
        'args': ['-m', MCP_SERVER_MODULE],
        'env': env,
        'startup_timeout_sec': _STARTUP_TIMEOUT_SEC,
        'tool_timeout_sec': _TOOL_TIMEOUT_SEC,
        # A dead MCP server fails the run instead of silently degrading the arm.
        'required': True,
    }


__all__ = [
    'LD_LIBRARY_PATH_PLACEHOLDER',
    'MCP_SERVER_MODULE',
    'MCP_SERVER_NAME',
    'WORKER_PYTHON_PLACEHOLDER',
    'build_ppt12_mcp_server',
]
