"""The tool set a run holds, and the state its tools read.

Skills reach their tool through ``ExecutionContext.tool_state`` rather than through
the executor, so those paths run against a real context. A non-absolute
``working_dir`` is refused: a plausible one grades as a zero with nothing raised.
"""

import asyncio

from pathlib import Path
from typing import Any

import pytest

from nexus.core import ExecutionContext
from nexus.extensions.tasks.hitl.user_agent.tools.usage_docs import (
    USAGE,
    USAGE_DOC,
    UsageDocError,
    collect_usage_docs,
    usage_of,
)
from nexus.extensions.tasks.hitl.user_agent.skills import load_skills
from nexus.extensions.tasks.hitl.user_agent.tools import toolset
from nexus.extensions.tasks.hitl.user_agent.tools import acting as acting_tools


# Tools this module offers but did not write, so their schemas are not ours to hold
# to the rules below.
_UPSTREAM_TOOLS = frozenset({'str_replace_editor'})


def write_skill(parent: Path, name: str, *, files: dict[str, str] | None = None) -> Path:
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'SKILL.md').write_text(
        f'---\nname: {name}\ndescription: what {name} is for\n---\n\nthe body of {name}\n', encoding='utf-8'
    )
    for relative, content in (files or {}).items():
        (directory / relative).write_text(content, encoding='utf-8')
    return directory


@pytest.fixture
def skills(tmp_path: Path):
    write_skill(tmp_path, 'browser', files={'setup.sh': 'echo hi'})
    return load_skills(external_dirs=[tmp_path])


class Fake:
    """A stand-in carrying the meta shape registration records."""

    def __init__(self, **meta: object) -> None:
        self.nexus_meta = {'name': 'fake', 'type': 'tool', 'meta': meta}


# -- what the model is shown vs. what the prompt is built from ------------


def test_usage_docs_stay_out_of_the_schema_the_model_receives() -> None:
    tools = toolset.build(['send_message', 'accept'])

    rendered = str(list(tools.executor.get_tool_schemas()))

    assert 'usage_doc' not in rendered
    assert 'usage' not in rendered
    assert tools.usage_docs
    # The keys being absent is not the point; the doc *text* must not be there
    # either, or the model reads every usage doc twice and pays for it twice.
    for _, usage_doc in tools.usage_docs:
        excerpt = usage_doc.splitlines()[0]
        assert excerpt not in rendered, f'usage_doc text leaked into the schema: {excerpt!r}'


def test_a_schema_description_stays_short_enough_to_be_a_description() -> None:
    """Applies to the tools this module writes.

    ``str_replace_editor`` is upstream's and its description runs to 956 characters,
    which is not ours to shorten -- the rule is about where *we* would put policy and
    cross-turn usage, and those belong in ``usage_doc``.
    """
    ours = [name for name in toolset.CATALOGUE if name not in _UPSTREAM_TOOLS]
    tools = toolset.build(ours, skills={'x': None})  # type: ignore[dict-item]

    for schema in tools.executor.get_tool_schemas():
        function = schema['function']
        assert len(function['description']) < 400, (
            f'{function["name"]}: description is prose-length, which is where policy and '
            'cross-turn usage creep in; those belong in usage_doc'
        )


def test_tools_sharing_a_usage_are_explained_once() -> None:
    tools = toolset.build(['send_message', 'read', 'bash'])

    usages = [usage for usage, _ in tools.usage_docs]

    assert usages.count('verification') == 1
    assert sorted(usages) == ['delegation', 'verification']


def test_declaring_a_usage_without_its_text_is_refused() -> None:
    with pytest.raises(UsageDocError, match='usage_doc'):
        usage_of(Fake(**{USAGE: 'delegation', USAGE_DOC: '   '}))

    with pytest.raises(UsageDocError, match='usage_doc'):
        usage_of(Fake(**{USAGE: '', USAGE_DOC: 'some text'}))


def test_a_tool_declaring_no_usage_contributes_nothing() -> None:
    assert usage_of(Fake(name='plain')) == ('', '')
    assert collect_usage_docs({'plain': Fake(name='plain')}) == []


def test_two_tools_claiming_one_usage_with_different_text_is_a_conflict() -> None:
    pool = {
        'read': Fake(**{USAGE: 'verification', USAGE_DOC: 'one text'}),
        'bash': Fake(**{USAGE: 'verification', USAGE_DOC: 'another text'}),
    }

    with pytest.raises(UsageDocError, match='different usage_doc'):
        collect_usage_docs(pool)


# -- what the model sees -------------------------------------------------


def test_an_outward_tool_reports_its_arguments_for_its_handler() -> None:
    """The body changes nothing; the handler reads meta and does the work."""
    tools = toolset.build(['send_message', 'accept'])

    async def run() -> Any:
        async with ExecutionContext(tool_state=toolset.tool_state(tools, working_dir='/testbed')):
            return await tools.executor.call('send_message', {'intent': 'request', 'message': 'hi'})

    response = asyncio.run(run())

    assert response.meta['message'] == 'hi'
    assert response.meta['intent'] == 'request'


# -- assembly refusals ---------------------------------------------------


@pytest.mark.parametrize(
    ('names', 'expected'),
    [
        ([], 'at least one tool'),
        (['read', 'read'], 'more than once'),
        (['read', 'edit'], 'unknown tools'),
    ],
)
def test_a_tool_set_that_cannot_be_built_says_why(names: list[str], expected: str) -> None:
    with pytest.raises(toolset.ToolSetError, match=expected):
        toolset.build(names)


def test_offering_invoke_skill_with_no_skills_loaded_is_refused() -> None:
    with pytest.raises(toolset.ToolSetError, match='no skills were loaded'):
        toolset.build(['send_message', 'invoke_skill'])


def test_loaded_skills_with_no_tool_to_reach_them_warns(skills, caplog) -> None:
    with caplog.at_level('WARNING'):
        toolset.build(['send_message', 'read'], skills=skills)

    assert 'unreachable' in caplog.text


@pytest.mark.parametrize('working_dir', ['', 'relative/path', '~/workspace'])
def test_a_working_dir_that_is_not_absolute_is_refused(working_dir: str) -> None:
    tools = toolset.build(['read'])

    with pytest.raises(toolset.ToolSetError, match='absolute'):
        toolset.tool_state(tools, working_dir=working_dir)


def test_tool_state_carries_the_working_dir_the_commands_will_use() -> None:
    tools = toolset.build(['read', 'bash'])

    state = toolset.tool_state(tools, working_dir='/testbed')

    assert state[acting_tools.WORKING_DIR_STATE] == '/testbed'


# -- skills reaching their tool ------------------------------------------


def test_a_skill_body_arrives_through_the_execution_context(skills) -> None:
    tools = toolset.build(['send_message', 'invoke_skill'], skills=skills)

    async def run() -> str:
        async with ExecutionContext(tool_state=toolset.tool_state(tools, working_dir='/testbed')):
            response = await tools.executor.call('invoke_skill', {'name': 'browser'})
            return response.output[0]['text']

    text = asyncio.run(run())

    assert 'the body of browser' in text
    assert 'setup.sh' in text
    assert '/root/.hitl_user/skills/browser' in text


def test_an_unknown_skill_costs_a_turn_rather_than_ending_the_run(skills) -> None:
    tools = toolset.build(['send_message', 'invoke_skill'], skills=skills)

    async def run() -> str:
        async with ExecutionContext(tool_state=toolset.tool_state(tools, working_dir='/testbed')):
            response = await tools.executor.call('invoke_skill', {'name': 'absent'})
            return response.output[0]['text']

    text = asyncio.run(run())

    assert 'No such skill' in text
    assert 'browser' in text, 'the reply names what does exist so the model can correct itself'


def test_invoke_skill_without_installed_state_reports_none_available(skills) -> None:
    tools = toolset.build(['send_message', 'invoke_skill'], skills=skills)

    async def run() -> str:
        async with ExecutionContext(tool_state={}):
            response = await tools.executor.call('invoke_skill', {'name': 'browser'})
            return response.output[0]['text']

    assert '(none)' in asyncio.run(run())


# -- inspect helpers -----------------------------------------------------


def test_relative_paths_resolve_against_the_working_directory() -> None:
    from nexus.core.tool.types import ToolContext

    context = ToolContext(tool_state={acting_tools.WORKING_DIR_STATE: '/testbed'})

    assert acting_tools.resolve_path('src/app.py', context) == '/testbed/src/app.py'
    assert acting_tools.resolve_path('/etc/hosts', context) == '/etc/hosts'


def test_long_output_keeps_both_ends_because_the_middle_is_filler() -> None:
    output = 'HEAD' + ('x' * 500) + 'TAIL'

    trimmed = acting_tools.truncate(output, limit=100)

    assert trimmed.startswith('HEAD')
    assert trimmed.endswith('TAIL')
    assert 'omitted' in trimmed
    assert len(trimmed) < len(output)


def test_short_output_is_untouched() -> None:
    assert acting_tools.truncate('brief', acting_tools.OUTPUT_LIMIT) == 'brief'


def test_every_intent_is_explained_where_the_user_reads_about_them() -> None:
    """A value in the schema's enum but not in the doc is one the model picks blind."""
    import re

    from typing import get_args

    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import INTENT_DOC, Intent

    intents = set(get_args(Intent))
    explained = set(re.findall(r'`(\w+)`', INTENT_DOC)) & intents

    assert intents - explained == set()


def test_every_stop_reason_is_explained_where_the_user_reads_about_them() -> None:
    import re

    from typing import get_args

    from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import ACCEPTANCE_DOC, StopReason

    reasons = set(get_args(StopReason))
    explained = set(re.findall(r'`(\w+)`', ACCEPTANCE_DOC)) & reasons

    assert reasons - explained == set()
