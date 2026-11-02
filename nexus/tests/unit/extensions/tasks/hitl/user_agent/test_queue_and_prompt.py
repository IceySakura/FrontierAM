"""The requirement queue and the text a simulator is given.

The queue's cases come from the shape of the real dataset: 497 of 4996 records
hold one requirement and the longest holds nineteen, so both the single-requirement
path and a cap that scales are exercised. The prompt's cases are about what ends
up where -- capability facts in the system prompt, requirement-shaped content in
the first user message -- and about the requirement arriving verbatim, since
rewriting it in the user's own words is the user's first move rather than something
done to it here.
"""

import pytest

from nexus.extensions.tasks.hitl.user_agent.personas import Persona, get_persona
from nexus.extensions.tasks.hitl.user_agent.prompts import (
    build_opening_message,
    build_requirement_message,
    build_system_prompt,
    environment_facts,
    skill_reminder,
)
from nexus.extensions.tasks.hitl.user_agent.prompts.identity import identity
from nexus.extensions.tasks.hitl.user_agent.queue import QueueError, Requirements
from nexus.extensions.tasks.hitl.user_agent.tools import toolset


# -- queue ---------------------------------------------------------------


def test_a_single_requirement_is_a_queue_that_releases_once() -> None:
    queue = Requirements.build(['fix the crash'])

    assert queue.total == 1
    assert queue.current == 'fix the crash'
    assert queue.accept('done') is None, 'nothing follows, which is what ends the run'
    assert queue.exhausted


def test_accepting_releases_the_next_requirement_until_none_remain() -> None:
    queue = Requirements.build(['first', 'second', 'third'])

    assert queue.accept('a') == 'second'
    assert queue.accept('b') == 'third'
    assert queue.accept('c') is None
    assert queue.accepted == ['a', 'b', 'c']
    assert queue.exhausted


def test_accepting_an_exhausted_queue_raises_rather_than_doing_nothing() -> None:
    queue = Requirements.build(['only'])
    queue.accept()

    with pytest.raises(QueueError, match='already been accepted'):
        queue.accept()


def test_build_drops_blanks_so_the_count_is_never_inflated() -> None:
    queue = Requirements.build(['first', '   ', 'second', ''])

    assert queue.total == 2
    assert queue.segments == ('first', 'second')


def test_a_blank_requirement_reaching_the_queue_directly_is_a_data_error() -> None:
    with pytest.raises(QueueError, match='are empty'):
        Requirements(segments=('first', ''))


@pytest.mark.parametrize('segments', [[], [''], ['  ', '']])
def test_a_run_with_no_requirements_is_refused(segments: list[str]) -> None:
    with pytest.raises(QueueError):
        Requirements.build(segments)


def test_an_index_outside_the_requirements_is_refused() -> None:
    with pytest.raises(QueueError, match='outside'):
        Requirements(segments=('a', 'b'), index=5)


def test_remaining_counts_down_as_requirements_are_accepted() -> None:
    queue = Requirements.build(['a', 'b'])

    assert queue.remaining == 2
    queue.accept()
    assert queue.remaining == 1
    queue.accept()
    assert queue.remaining == 0


# -- prompt --------------------------------------------------------------


def persona() -> Persona:
    return Persona(name='impatient', prompt='You get impatient when things drag.')


def test_the_system_prompt_carries_only_the_usage_docs_the_tools_hold() -> None:
    with_verification = build_system_prompt(persona(), usage_docs=[('verification', 'check things yourself')])
    without = build_system_prompt(persona(), usage_docs=[])

    assert 'check things yourself' in with_verification
    assert 'check things yourself' not in without


def test_the_persona_prompt_is_appended_not_slotted() -> None:
    prompt = build_system_prompt(persona())

    assert 'You get impatient when things drag.' in prompt
    assert '{' not in prompt, 'an unfilled slot would leave a brace in the text'


def test_capability_facts_go_to_the_system_prompt() -> None:
    facts = environment_facts(working_dir='/testbed', workers=['backend', 'frontend'])
    prompt = build_system_prompt(persona(), environment=facts)

    assert '/testbed' in prompt
    assert 'backend' in prompt and 'frontend' in prompt


def test_several_workers_are_described_differently_from_one() -> None:
    """With one there is nobody to pick between, so the two cannot read the same.

    Which words say it is the module's business; that the branches differ at all is
    this one's -- they were a single f-string once, and the user was told to address
    somebody by name when there was only one name.
    """
    one = environment_facts(working_dir='/t', workers=['worker'])
    several = environment_facts(working_dir='/t', workers=['backend', 'frontend'])

    assert len(one) == len(several) == 2
    assert one[1] != several[1]


def test_no_workers_yields_no_worker_sentence() -> None:
    facts = environment_facts(working_dir='/t', workers=[])

    assert len(facts) == 1
    assert '/t' in facts[0]


def test_the_user_is_told_what_it_is_working_with_and_how_to_address_it() -> None:
    """``product`` is what a person calls the tool. The schema name is what a message
    is addressed to. The sentence carries both."""
    facts = environment_facts(working_dir='/testbed', workers=['worker'], product='Claude Code')
    sentence = facts[1]

    assert 'Claude Code' in sentence
    assert '`worker`' in sentence, 'the name a message is addressed to'


def test_a_backend_naming_no_product_drops_the_clause_rather_than_leaving_a_gap() -> None:
    """A manager from outside this module need not name a product."""
    sentence = environment_facts(working_dir='/testbed', workers=['worker'])[1]

    assert '`worker`' in sentence
    assert 'None' not in sentence and '  ' not in sentence


def test_the_product_name_comes_from_the_backend_rather_than_being_written_here() -> None:
    """Swapping the backend swaps the name."""
    from nexus.extensions.tasks.hitl.actor.registry import worker_manager_registry

    products = {manager.product for manager in worker_manager_registry.get_all().values()}
    assert products == {'Claude Code', 'Codex'}, products

    swapped = environment_facts(working_dir='/t', workers=['worker'], product='Codex')
    assert 'Codex' in swapped[1] and 'Claude Code' not in swapped[1]


def test_the_requirement_reaches_the_first_user_message_verbatim() -> None:
    requirement = 'ERC20DetailedWrapper applies bytes32ToString to a value that is already a string.'

    opening = build_opening_message(requirement, language='English')

    assert requirement in opening, "rephrasing it is the user's own first move, not done here"


def test_the_skill_catalogue_goes_to_the_first_user_message_not_the_system_prompt() -> None:
    catalogue = '- browser: drive a browser'

    opening = build_opening_message('do the thing', language='English', catalogue=catalogue)
    prompt = build_system_prompt(persona())

    assert catalogue in opening
    assert catalogue not in prompt


def test_the_catalogue_is_wrapped_so_its_origin_is_unambiguous() -> None:
    reminder = skill_reminder('- browser: drive a browser')

    assert reminder.startswith('<system-reminder>')
    assert reminder.endswith('</system-reminder>')
    assert 'invoke_skill' in reminder


def test_an_empty_catalogue_produces_no_reminder() -> None:
    assert skill_reminder('') == ''
    assert 'invoke_skill' not in build_opening_message('do the thing', language='English')


@pytest.mark.parametrize(
    'builder',
    [lambda text: build_opening_message(text, language='English'), build_requirement_message],
)
def test_a_message_with_no_requirement_is_refused(builder) -> None:
    with pytest.raises(ValueError, match='requirement'):
        builder('   ')


def test_a_later_requirement_arrives_under_its_own_heading() -> None:
    """Two headings, because one of them is not the first thing asked for."""
    first = build_opening_message('fix the crash', language='English')
    later = build_requirement_message('now make it work on v3 too')

    assert 'now make it work on v3 too' in later
    assert later.split('\n')[0] != first.split('\n')[0]


def test_what_the_worker_can_see_is_said_once_rather_than_at_every_task() -> None:
    """By the second task the user has been telling the worker things for rounds."""
    from nexus.extensions.tasks.hitl.user_agent.prompts.messages import (
        CONTINUITY_INSTRUCTION,
        SCOPE_INSTRUCTION,
    )

    first = build_opening_message('fix the crash', language='English')
    later = build_requirement_message('now make it work on v3 too')

    assert SCOPE_INSTRUCTION in first
    assert SCOPE_INSTRUCTION not in later
    assert CONTINUITY_INSTRUCTION in later
    assert CONTINUITY_INSTRUCTION not in first


# -- which sections reach a run ------------------------------------------


def test_a_usage_doc_reaches_the_prompt_only_when_its_tool_does() -> None:
    """A tool's cross-turn rules live in that tool's doc, so a run without the tool
    drops them. Were they in IDENTITY instead, a run with nothing to look with would
    still be told to check before accepting.

    Whole constants are compared rather than phrases from them. A phrase table was
    here before and every rewording broke it while pinning nothing: what matters is
    that assembly puts the section in, not which words the section uses.
    """
    from nexus.extensions.tasks.hitl.user_agent.tools.acting import VERIFICATION_DOC

    looking = toolset.build(('send_message', 'accept', 'read', 'bash'), skills={})
    blind = toolset.build(('send_message', 'accept'), skills={})

    with_tools = build_system_prompt(get_persona('default'), usage_docs=looking.usage_docs)
    without = build_system_prompt(get_persona('default'), usage_docs=blind.usage_docs)

    assert VERIFICATION_DOC.strip() in with_tools
    assert VERIFICATION_DOC.strip() not in without
    assert identity() in without, 'what holds for every run is not in a tool doc'


def test_every_section_the_builder_is_given_reaches_the_prompt() -> None:
    """Assembly appends; nothing here is a template slot to be silently left unfilled.

    Measured when this was not true: 15 of 16 constraints were written and absent from
    the rendered prompt, and a run spent all its turns looking and sent no message.
    """
    from nexus.extensions.tasks.hitl.actor.rendering import WORKER_REPORT_PROTOCOL
    from nexus.extensions.tasks.hitl.user_agent.prompts.harness import HARNESS

    persona = get_persona('default')
    tools = toolset.build(persona.tools, skills={})
    prompt = build_system_prompt(
        persona,
        usage_docs=(*tools.usage_docs, ('worker_report', WORKER_REPORT_PROTOCOL)),
        environment=environment_facts(working_dir='/testbed', workers=['worker'], product='Claude Code'),
        may_edit=tools.may_edit,
    )

    assert identity(may_edit=tools.may_edit) in prompt
    assert HARNESS.strip() in prompt
    assert WORKER_REPORT_PROTOCOL.strip() in prompt
    assert persona.prompt in prompt
    for _, usage_doc in tools.usage_docs:
        assert usage_doc.strip() in prompt
    assert '/testbed' in prompt
