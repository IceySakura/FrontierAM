"""Building the text a user is given.

One module per section of the system prompt, and :func:`build_system_prompt` to
assemble them. Sections are appended in order, not slotted into a template: v1 had
eleven sections and one slot, so slots never carried it.

    identity.py        who is being simulated
    harness.py         how this environment works
    <tool usage docs>  carried by the tools themselves (``user_agent/tools/``)
    <persona>          ``Persona.section()``
    memory.py          the user's own preference note
    worker_skills.py   the worker's own skills
    environment.py     the working directory and what is on the other side

``messages.py`` holds what a run appends to the history afterwards, which is a
different question from what the system prompt says.
"""

from collections.abc import Sequence

from nexus.extensions.tasks.hitl.user_agent.personas import Persona
from nexus.extensions.tasks.hitl.user_agent.prompts.environment import environment_facts
from nexus.extensions.tasks.hitl.user_agent.prompts.harness import HARNESS
from nexus.extensions.tasks.hitl.user_agent.prompts.identity import IDENTITY
from nexus.extensions.tasks.hitl.user_agent.prompts.identity import identity as identity_for
from nexus.extensions.tasks.hitl.user_agent.prompts.memory import user_memory_section
from nexus.extensions.tasks.hitl.user_agent.prompts.messages import (
    CONTINUITY_INSTRUCTION,
    HANDOVER_LOOK,
    HANDOVER_REPLY,
    HANDOVER_STATE,
    LANGUAGE_INSTRUCTION,
    NUDGE,
    REMINDER_TAG,
    append_reminder,
    build_opening_message,
    build_requirement_message,
    handover_reminder,
    language_instruction,
    reminder,
    reminder_message,
    skill_reminder,
)
from nexus.extensions.tasks.hitl.user_agent.prompts.worker_skills import (
    worker_command_lines,
    worker_skills_section,
)
from nexus.extensions.tasks.hitl.utils.text import section


def build_system_prompt(
    persona: Persona,
    *,
    usage_docs: Sequence[tuple[str, str]] = (),
    identity: str = '',
    harness: str = HARNESS,
    environment: Sequence[str] = (),
    worker_skills: str = '',
    user_memory: str = '',
    may_edit: bool = False,
) -> str:
    """Identity, harness, one doc per tool mechanism, the persona, the environment.

    The persona section is one heading, which a reminder refers back to rather than
    repeating what is under it.

    ``user_memory`` and ``worker_skills`` come after the persona: the first is this
    user's own note, the second is something to reach for, and both read against how
    this user works.

    ``may_edit`` is ``ToolSet.may_edit``: it picks how the identity words the rule
    about who writes the code. Passing ``identity`` overrides both.

    Any section may be empty; nothing depends on a slot being filled.
    """
    sections: list[str] = [(identity or identity_for(may_edit=may_edit)).strip(), harness.strip()]
    sections += [usage_doc for _, usage_doc in usage_docs if usage_doc.strip()]
    sections.append(persona.section())
    sections.append(user_memory.strip())
    sections.append(worker_skills.strip())
    facts = [fact.strip() for fact in environment if fact.strip()]
    if facts:
        sections.append(section('## Your situation', '\n'.join(f'- {fact}' for fact in facts)))
    return '\n\n'.join(part for part in sections if part)


__all__ = [
    'CONTINUITY_INSTRUCTION',
    'HANDOVER_LOOK',
    'HANDOVER_REPLY',
    'HANDOVER_STATE',
    'HARNESS',
    'IDENTITY',
    'LANGUAGE_INSTRUCTION',
    'NUDGE',
    'REMINDER_TAG',
    'append_reminder',
    'build_opening_message',
    'build_requirement_message',
    'build_system_prompt',
    'environment_facts',
    'handover_reminder',
    'identity_for',
    'language_instruction',
    'reminder',
    'reminder_message',
    'skill_reminder',
    'user_memory_section',
    'worker_command_lines',
    'worker_skills_section',
]
