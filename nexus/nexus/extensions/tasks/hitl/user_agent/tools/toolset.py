"""Assembling the tool set a run holds, and the ``tool_state`` it needs.

Registry names are namespaced (``hitl/user/read``). The schemas the model sees carry
bare names (``read``). ``CATALOGUE`` maps between them, and ``OPTIONAL`` holds the ones
a run may ask for on top.
"""

import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nexus.core import ToolExecutor, tool_registry
from nexus.extensions.tasks.hitl.user_agent.skills.loader import Skill
from nexus.extensions.tasks.hitl.user_agent.tools import acting as acting_tools
from nexus.extensions.tasks.hitl.user_agent.tools import orchestration as orchestration_tools
from nexus.extensions.tasks.hitl.user_agent.tools import skill_tool
from nexus.extensions.tasks.hitl.user_agent.tools.usage_docs import collect_usage_docs


logger = logging.getLogger(__name__)


class ToolSetError(ValueError):
    """A tool set is declared in a way the simulator cannot run."""


# Registry names, by the schema name the model calls them with.
CATALOGUE: dict[str, str] = {
    'send_message': orchestration_tools.SEND_MESSAGE_TOOL,
    'accept': orchestration_tools.ACCEPT_TOOL,
    'read': acting_tools.READ_TOOL,
    'bash': acting_tools.BASH_TOOL,
    'str_replace_editor': 'str_replace_editor',
    'invoke_skill': skill_tool.INVOKE_SKILL_TOOL,
}

# ``read`` and ``str_replace_editor`` overlap. The editor's ``view`` is what ``read``
# calls, and the editor's other four commands write. ``DEFAULT_TOOLS`` holds ``read``.
# A run asks for the editor by name.

# Tools from elsewhere in the tree that a run may ask for, by the same mapping.
OPTIONAL: dict[str, str] = {}

# Schema names that change the work rather than look at it.
EDITING_TOOLS: frozenset[str] = frozenset({'str_replace_editor'})

VERIFICATION_USAGE = 'verification'


@dataclass(frozen=True)
class ToolSet:
    """What a simulator can call this run, split by who executes it.

    ``schemas`` is captured once at assembly. ``get_tool_schemas`` reaches
    ``require_context``, which walks the Python stack to decide whether it is inside a
    runtime tool (``core/context.py:618``) and takes about 19 ms a call.
    """

    executor: ToolExecutor
    usage_docs: tuple[tuple[str, str], ...] = ()
    skills: Mapping[str, Skill] = field(default_factory=dict)
    schemas: tuple[dict[str, Any], ...] = ()

    @property
    def schema_names(self) -> tuple[str, ...]:
        return tuple(name for schema in self.schemas if (name := schema.get('function', {}).get('name')))

    @property
    def may_edit(self) -> bool:
        """Whether this run gave the user a way to change the work itself."""
        return bool(EDITING_TOOLS & set(self.schema_names))

    def describe(self) -> str:
        return f'tools={sorted(self.schema_names)} skills={sorted(self.skills)}'


def resolve(names: Sequence[str]) -> list[str]:
    """Registry names for schema names, naming the alternatives on failure."""
    known = {**CATALOGUE, **OPTIONAL}
    unknown = sorted(set(names) - set(known))
    if unknown:
        available = f'Available: {", ".join(sorted(CATALOGUE))}'
        if OPTIONAL:
            available += f'; on request: {", ".join(sorted(OPTIONAL))}'
        raise ToolSetError(f'unknown tools {unknown}. {available}')
    return [known[name] for name in names]


def build(
    names: Sequence[str],
    *,
    skills: Mapping[str, Skill] | None = None,
) -> ToolSet:
    """Build the tool set for one run.

    ``names`` is required. What a user holds is its persona's declaration.

    Refuses ``invoke_skill`` with no skills loaded, and warns on skills that no tool
    can reach.
    """
    if not names:
        raise ToolSetError('a simulator needs at least one tool')
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise ToolSetError(f'tools listed more than once: {duplicates}')

    registry_names = resolve(names)
    functions = {
        name: tool_registry.get(registry_name) for name, registry_name in zip(names, registry_names, strict=True)
    }

    loaded = dict(skills or {})
    if 'invoke_skill' in names and not loaded:
        raise ToolSetError('invoke_skill was requested but no skills were loaded')
    if loaded and 'invoke_skill' not in names:
        logger.warning('%d skills loaded but invoke_skill is not in the tool set; they are unreachable', len(loaded))

    executor = ToolExecutor(list(functions.values()))
    try:
        schemas = tuple(executor.get_tool_schemas())
    except ValueError as exc:
        # Any MCP tool already on the context is included here, and one whose name
        # matches a tool of this set collides. The names are checked against the
        # registry when the server attaches. This set registers under
        # ``hitl/user/...`` while the model calls it ``read``, and a server offering
        # a bare ``read`` passes that check and collides only here.
        raise ToolSetError(f'{exc}. An MCP server offering that name would have to declare a prefix') from exc

    return ToolSet(
        executor=executor,
        usage_docs=tuple(_docs_for(functions, schemas)),
        skills=loaded,
        schemas=schemas,
    )


def _docs_for(functions: Mapping[str, Any], schemas: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """The usage docs for this set.

    The verification one is added when no tool of this set declares that usage but one
    of them can still check something. ``str_replace_editor`` is from elsewhere in the
    tree and declares none.
    """
    offered = [name for schema in schemas if (name := schema.get('function', {}).get('name'))]
    docs = list(collect_usage_docs(functions))
    declared = {usage for usage, _ in docs}
    if VERIFICATION_USAGE not in declared and acting_tools.can_verify(offered):
        docs.append((VERIFICATION_USAGE, acting_tools.VERIFICATION_DOC))
    return docs


def tool_state(tools: ToolSet, *, working_dir: str) -> dict[str, Any]:
    """The ``ExecutionContext.tool_state`` entries this tool set needs.

    ``working_dir`` must be absolute. A relative one runs commands elsewhere without
    raising.
    """
    if not working_dir or not working_dir.startswith('/'):
        raise ToolSetError(f'working_dir must be an absolute path, got {working_dir!r}')
    state: dict[str, Any] = {acting_tools.WORKING_DIR_STATE: working_dir}
    if tools.skills:
        state.update(skill_tool.install(tools.skills))
    return state


def tool_names(declared: Sequence[str], skills: Mapping[str, Skill], extra: Sequence[str] = ()) -> list[str]:
    """``declared`` plus ``invoke_skill`` when skills were loaded, plus ``extra``.

    ``extra`` is what a run asked for beyond its persona's declaration. Names already
    declared are dropped rather than repeated.
    """
    names = list(declared)
    if skills and 'invoke_skill' not in names:
        names.append('invoke_skill')
    for name in extra:
        if name not in names:
            names.append(name)
    return names


__all__ = [
    'CATALOGUE',
    'OPTIONAL',
    'ToolSet',
    'ToolSetError',
    'build',
    'resolve',
    'tool_names',
    'tool_state',
]
