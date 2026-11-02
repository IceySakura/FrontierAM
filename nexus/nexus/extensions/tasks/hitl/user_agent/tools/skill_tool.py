"""Fetching a skill's body on demand.

The catalogue the simulator carries holds names and one-line descriptions. A body
arrives only when it asks for one.

Bodies are read at assembly and held here. This tool is LOCAL and does no IO. The
copies placed in the sandbox exist for the scripts a body refers to, which the
simulator runs with ``bash``.
"""

from collections.abc import Mapping

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tasks.hitl.user_agent.skills.loader import Skill
from nexus.extensions.tasks.hitl.user_agent.tools.usage_docs import USAGE, USAGE_DOC


INVOKE_SKILL_TOOL = 'hitl/user/invoke_skill'
SKILLS_STATE = 'user_skills'

SKILL_DOC = """\
## Skills

You can access a list of skills providing domain knowledge, procedures, and templates.
Each skill has a description clarifying how and when to use it.
When the situation matches a skill's description, use invoke_skill to get the full content of a skill.\
"""


class InvokeSkillInput(BaseModel):
    name: str = Field(description='Which skill to pull up, by name.')


def install(skills: Mapping[str, Skill]) -> dict[str, Mapping[str, Skill]]:
    """The ``tool_state`` entry that makes ``skills`` reachable by this tool."""
    return {SKILLS_STATE: dict(skills)}


def _available(state: object) -> dict[str, Skill]:
    if not isinstance(state, Mapping):
        return {}
    return {str(name): skill for name, skill in state.items() if isinstance(skill, Skill)}


@tool_registry.register(
    INVOKE_SKILL_TOOL,
    meta={
        'name': 'invoke_skill',
        'description': 'Pull up the full text of one skill by name.',
        'tool_type': ToolType.LOCAL,
        USAGE: 'skills',
        USAGE_DOC: SKILL_DOC,
    },
)
async def invoke_skill_tool(params: InvokeSkillInput, context: ToolContext) -> ToolResponse:
    """Return one skill's text, or name the ones that exist.

    An unknown name answers with the available list rather than raising.
    """
    skills = _available(context.tool_state.get(SKILLS_STATE))
    skill = skills.get(params.name)
    if skill is None:
        available = ', '.join(sorted(skills)) or '(none)'
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'<skill_error name="{params.name}">No such skill. Available: {available}</skill_error>',
                }
            ]
        )
    return ToolResponse(output=[{'type': 'text', 'text': skill.render()}], meta={'skill': skill.name})


__all__ = [
    'INVOKE_SKILL_TOOL',
    'SKILLS_STATE',
    'SKILL_DOC',
    'InvokeSkillInput',
    'install',
]
