"""The swe_chain_v2 recipe's wiring to the legacy task, and to the trace layer.

Nothing here tests the simulator; that is covered where it lives. These cover the
seams where a mistake is silent:

* the task passes the agent every requirement, not just the first -- the legacy
  task sends ``user_prompt=problem_statements[0]`` and drops the rest, so a
  19-segment instance would collapse to one and still score;
* the agent output declares the fields the legacy scoring path reads off it with
  ``{}`` defaults, so a missing one records a zero rather than raising;
* the user's client is labelled before it is used, or every user turn exports as
  ``_unattributed`` and no filter claims it;
* the generator configures ``branch_last``, which the agent then insists on --
  ``global_first`` merges the user and the worker into one flat list.
"""

import asyncio
import re
import textwrap

from types import SimpleNamespace
from typing import Any

import pytest

from pydantic import TypeAdapter, ValidationError

from nexus.core import task_registry
from nexus.extensions.tasks.hitl.agents.agent_with_user import USER_LABEL, AgentWithUserOutput
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2 import generator
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.task import (
    SWEChainV2TaskInput,
    SWEChainV2TaskOutput,
    _chain_instance,
    _output,
)
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.utils.generator_util import _build_worker_config
from nexus.trace.context import context_processors
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


# -- registration --------------------------------------------------------


def test_the_task_is_reachable_by_the_name_the_generator_yields() -> None:
    assert task_registry.get('hitl/swe_chain_v2_task') is not None


def test_the_generator_names_a_task_and_agent_that_exist() -> None:
    """A typo here fails the whole batch at run time, not at generation."""
    from nexus.core import agent_registry

    assert task_registry.get('hitl/swe_chain_v2_task') is not None
    assert agent_registry.get(generator.AGENT_NAME) is not None


def test_the_recipe_reaches_the_generic_agent_through_a_hook() -> None:
    """What makes it this recipe is the hook, not an agent of its own.

    The hook subscribes to events this agent fires, which ``resolve_hooks`` checks.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import EVENTS
    from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.hook import HOOK_NAME
    from nexus.extensions.tasks.hitl.user_agent.hooks import resolve_hooks

    assert generator.AGENT_NAME == 'hitl/agent_with_user'
    assert resolve_hooks([{'name': HOOK_NAME}], events=EVENTS)


# -- the record arrives on the input, not from a file ---------------------


def test_the_task_reads_a_chain_record_the_swe_instance_type_would_refuse() -> None:
    """Why the field is a plain dict.

    A chain record's ``patch`` and ``problem_statements`` are lists; the
    ``SWEbenchInstance`` TypedDict declares both as ``str``. Validating against it
    rejects the input this recipe exists to serve, which is what the dataset loader
    that used to sit in the task was written to step around.
    """
    from nexus.extensions.tasks.swe.task import SWEbenchInstanceDict

    record = {
        'instance_id': 'abc',
        'image': 'img',
        'base_commit': 'sha',
        'problem_statements': ['first', 'second'],
        'patch': ['diff one', 'diff two'],
    }

    async def run_agent(**kwargs: Any) -> Any:
        return SimpleNamespace(error=None)

    kept = _chain_instance(SWEChainV2TaskInput(instance_id='abc', instance=record, run_agent=run_agent))
    assert kept['patch'] == ['diff one', 'diff two']

    with pytest.raises(ValidationError):
        TypeAdapter(SWEbenchInstanceDict).validate_python(record)


def test_a_run_whose_record_never_arrived_says_so_instead_of_reading_a_file() -> None:
    """The task opens nothing, so a missing record is the failure, not a lookup."""

    async def run_agent(**kwargs: Any) -> Any:
        return SimpleNamespace(error=None)

    with pytest.raises(ValueError, match='no dataset record'):
        _chain_instance(SWEChainV2TaskInput(instance_id='abc', run_agent=run_agent))


@pytest.mark.parametrize('missing', ['image', 'base_commit', 'problem_statements'])
def test_a_record_missing_a_field_with_no_default_is_refused_by_name(missing: str) -> None:
    """These three have no usable default: no image means no sandbox, no base commit
    means the worker's diff is taken against nothing, no statements means nothing to
    ask for. Naming which one is missing is what the check buys over a KeyError later.
    """
    record = {
        'instance_id': 'abc',
        'image': 'img',
        'base_commit': 'sha',
        'problem_statements': ['first'],
        'patch': ['diff'],
    }
    del record[missing]

    async def run_agent(**kwargs: Any) -> Any:
        return SimpleNamespace(error=None)

    with pytest.raises(ValueError, match=missing):
        _chain_instance(SWEChainV2TaskInput(instance_id='abc', instance=record, run_agent=run_agent))


def test_the_task_gets_its_own_copy_to_write_derived_values_onto() -> None:
    """It replaces ``patch`` with the combined one and adds ``chain_patches``.

    Doing that to the input would edit the run config every other reader sees.
    """
    record = {
        'instance_id': 'abc',
        'image': 'img',
        'base_commit': 'sha',
        'problem_statements': ['first'],
        'patch': ['diff'],
    }

    async def run_agent(**kwargs: Any) -> Any:
        return SimpleNamespace(error=None)

    task_input = SWEChainV2TaskInput(instance_id='abc', instance=record, run_agent=run_agent)
    kept = _chain_instance(task_input)
    kept['patch'] = 'combined'

    assert task_input.instance['patch'] == ['diff']


# -- the output shape the legacy scoring path reads -----------------------


def output_with(**kwargs: Any) -> AgentWithUserOutput:
    return AgentWithUserOutput(success=True, output='', stderr='', exit_code=0, execution_time=0.0, **kwargs)


def test_the_output_declares_every_field_the_legacy_task_reads() -> None:
    """``recipe/swe/task.py:413-419`` reads these with {} / 0 defaults.

    An undeclared field is not an error there; it records a zero. That is the
    silent-failure shape, so the declaration is pinned here.
    """
    fields = set(AgentWithUserOutput.model_fields)

    assert {'worker_rounds', 'trajectories', 'usage_summary', 'run_meta', 'stderr'} <= fields


def test_completed_steps_travels_in_the_run_meta_where_chain_output_looks() -> None:
    """``recipe/swe_chain/task.py:367`` takes it out of the run_meta bundle."""
    output = output_with(
        accepted=[{'reason': 'satisfied', 'summary': str(index)} for index in range(3)],
        run_meta={'completed_steps': 3, 'total_steps': 5},
    )

    assert int(output.run_meta['completed_steps']) == 3
    assert len(output.accepted) == 3


def test_trajectories_are_held_in_memory_but_kept_out_of_the_dump() -> None:
    """The run record already carries them under ``processor_results.chat``.

    ``manager.traces`` reslices that same tracker snapshot by label, so dumping the field
    writes the largest part of the record twice. Readers holding the object -- the
    reward-hacking detector -- still get it.
    """
    output = output_with(trajectories={'user': {'traces': {'trajectories': [1]}}})

    assert output.trajectories['user'], 'the in-memory reader still sees it'
    assert 'trajectories' not in output.model_dump(), 'but it is not exported a second time'


# -- the tracker the labels need -----------------------------------------


def with_processors(processors: list[Any]) -> Any:
    token = context_processors.set(processors)
    return token


def test_a_run_without_a_run_level_tracker_is_refused() -> None:
    token = with_processors([])
    try:
        with pytest.raises(labels.LabelError, match='no run-level chat_tracker'):
            labels.require_labelling_tracker()
    finally:
        context_processors.reset(token)


def test_a_global_first_tracker_is_refused_because_labels_have_nowhere_to_go() -> None:
    token = with_processors([ChatTrackerProcessor(tools_mode='global_first')])
    try:
        with pytest.raises(labels.LabelError, match='branch_last'):
            labels.require_labelling_tracker()
    finally:
        context_processors.reset(token)


def test_a_branch_last_tracker_is_accepted() -> None:
    token = with_processors([ChatTrackerProcessor(tools_mode='branch_last')])
    try:
        labels.require_labelling_tracker()
    finally:
        context_processors.reset(token)


def test_the_generator_configures_the_mode_the_agent_insists_on() -> None:
    """These two are set in different files; a mismatch merges the trajectories."""
    configured = generator.CHAT_PROCESSOR['chat']['args']['tools_mode']

    token = with_processors([ChatTrackerProcessor(tools_mode=configured)])
    try:
        labels.require_labelling_tracker()
    finally:
        context_processors.reset(token)


def test_every_processor_the_generator_names_can_be_built() -> None:
    """The runner resolves these names at run time, after the sandbox is up.

    A name that is misspelled, or a class the recipe package never imports, is not
    in the registry then -- and nothing before that point says so.
    """
    from nexus.trace.processors.base import span_processor_registry

    named = {**generator.CHAT_PROCESSOR, **generator._processors('default', 'ins')}
    assert 'chat' in named

    for key, config in named.items():
        processor_cls = span_processor_registry.get(config['name'])
        assert processor_cls is not None, f'processor {key!r} names {config["name"]!r}, which is not registered'
        processor_cls(**config.get('args', {}))


def test_the_recipe_import_alone_registers_every_processor_it_names() -> None:
    """Run in a subprocess, because this test process already imported them.

    The runner resolves processor names having imported only what the run config's
    ``imports`` entry lists, which for this recipe is the package. A class the
    package does not import is absent from the registry then, and no assertion in
    this file would notice -- pytest's own collection imports it either way.
    """
    import subprocess
    import sys

    names = sorted(
        {config['name'] for config in {**generator.CHAT_PROCESSOR, **generator._processors('default', 'ins')}.values()}
    )
    script = (
        'import nexus.extensions.tasks.hitl.recipe.swe_chain_v2\n'
        'from nexus.trace.processors.base import span_processor_registry\n'
        f'for name in {names!r}:\n'
        '    span_processor_registry.get(name)\n'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# -- labelling the user's own client -------------------------------------


def test_the_user_label_is_one_the_filter_recognises() -> None:
    trajectory = {'role': USER_LABEL, 'messages': [{'role': 'assistant', 'content': 'U'}]}

    assert labels.is_wellformed(USER_LABEL)
    assert labels.filter_by_label({'trajectories': [trajectory]}, USER_LABEL)['trajectories'] == [trajectory]


def test_the_user_label_does_not_also_select_a_worker() -> None:
    result = {
        'trajectories': [
            {'role': USER_LABEL, 'messages': [{'role': 'assistant', 'content': 'U'}]},
            {'role': labels.worker_label('worker'), 'messages': [{'role': 'assistant', 'content': 'W'}]},
        ]
    }

    kept = labels.filter_by_label(result, USER_LABEL)['trajectories']

    assert [entry['messages'][-1]['content'] for entry in kept] == ['U']


def test_an_unlabelled_client_carries_no_label_until_it_is_enabled() -> None:
    """The defect this guards: a stock client's spans have no _chat_phase at all,
    so every user turn exports under UNATTRIBUTED and no filter claims it."""

    class Client:
        async def create_completion(self, **_: Any) -> Any:
            return None

    client = Client()
    assert not hasattr(client, '_chat_phase')

    labels.enable_labels(client)
    with labels.labelled(client, USER_LABEL):
        assert client._chat_phase == USER_LABEL


def test_the_label_is_cleared_once_the_run_is_over() -> None:
    class Client:
        async def create_completion(self, **_: Any) -> Any:
            return None

    client = labels.enable_labels(Client())
    with labels.labelled(client, USER_LABEL):
        pass

    assert client._chat_phase != USER_LABEL


# -- what the generator decides ------------------------------------------


def test_the_default_backend_is_one_the_factory_can_build() -> None:
    """A backend name no selector knows fails only once a run reaches setup."""
    from nexus.extensions.tasks.hitl.actor.factory import backends

    assert generator.DEFAULT_WORKER_BACKEND in backends()


def test_the_default_persona_is_registered() -> None:
    from nexus.extensions.tasks.hitl.user_agent.personas import get_persona

    assert get_persona(generator.DEFAULT_PERSONA) is not None


# -- the settings a run is configured by -----------------------------------


def test_every_environment_variable_the_generator_reads_is_documented() -> None:
    """``ENVIRONMENT`` is the list someone changing a pipeline script reads.

    A variable added to the code and not to the table is one nobody knows to set.
    The source is parsed for the names actually read, so the table cannot drift by
    being forgotten -- only by being wrong, which is visible.
    """
    import ast
    import inspect

    source = inspect.getsource(generator)
    tree = ast.parse(source)
    read: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
        if name not in {'get', '_env_bool', '_env_int', '_env_list', '_require_env'}:
            continue
        first = node.args[0]
        # An env var name is a bare uppercase literal; dict.get('usage_profile')
        # and friends are not.
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.isupper():
            read.add(first.value)

    assert read, 'the scan found no environment reads at all; the pattern list is stale'
    undocumented = sorted(read - set(generator.ENVIRONMENT))
    assert not undocumented, f'read by the generator but absent from ENVIRONMENT: {undocumented}'


def test_the_documented_worker_client_is_the_one_the_agent_asks_for() -> None:
    """The usage profile is keyed by client name.

    A key matching no client applies the profile to nothing, and the worker's
    Vertex counters are then read with the default profile -- which does not look
    for cache creation, so that counter silently reads as zero.
    """
    profiles = generator._processors('bedrock_claude')['usage']['args']['profiles']

    assert profiles == {generator.WORKER_CLIENT: 'bedrock_claude'}


# -- Langfuse upload -------------------------------------------------------


def test_langfuse_is_configured_only_when_credentials_are_present(monkeypatch) -> None:
    """Named or absent, never named-and-disabled.

    A processor the generator names is constructed, and one constructed without
    credentials logs a warning per instance about a tracker nobody asked for. So
    the decision is made here, where the config is written.
    """
    monkeypatch.delenv('HITL_DISABLE_LANGFUSE', raising=False)
    monkeypatch.delenv('LANGFUSE_PUBLIC_KEY', raising=False)
    monkeypatch.delenv('LANGFUSE_SECRET_KEY', raising=False)

    assert 'langfuse' not in generator._processors('default', 'ins')

    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    assert generator._processors('default', 'ins')['langfuse']['name'] == 'multiagent_langfuse'


def test_langfuse_can_be_turned_off_with_credentials_still_set(monkeypatch) -> None:
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')
    monkeypatch.setenv('HITL_DISABLE_LANGFUSE', '1')

    assert 'langfuse' not in generator._processors('default', 'ins')


def test_the_langfuse_session_is_the_instance_so_reruns_group_together(monkeypatch) -> None:
    """``session_id`` is what collects several traces in the Langfuse UI.

    One run uploads one trace, so the session groups the reruns of an instance
    rather than one run's several agents.
    """
    monkeypatch.delenv('HITL_DISABLE_LANGFUSE', raising=False)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    args = generator._processors('default', 'initOS__dob-lib-12')['langfuse']['args']

    assert args['session_id'] == 'initOS__dob-lib-12'
    assert args['trace_metadata']['instance_id'] == 'initOS__dob-lib-12'


def test_the_named_langfuse_processor_is_registered_by_importing_the_recipe(monkeypatch) -> None:
    """A name in the run config the registry cannot resolve fails after the sandbox is up.

    ``langfuse`` is a declared dependency of this module, so the import is
    unconditional and the registration cannot go missing.
    """
    monkeypatch.delenv('HITL_DISABLE_LANGFUSE', raising=False)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk')

    import nexus.extensions.tasks.hitl.recipe.swe_chain_v2 as recipe

    from nexus.trace.processors import span_processor_registry

    assert recipe.MultiAgentLangfuseProcessor is not None, 'importing the recipe imports it'

    config = generator._processors('default', 'ins')['langfuse']
    processor = span_processor_registry.get(config['name'])(**config['args'])

    assert processor.get_result()['by_label'] == {}, 'a fresh processor has recorded nothing'


# -- token accounting ------------------------------------------------------


def test_the_run_configures_usage_accounting_alongside_the_chat_tracker() -> None:
    """Both processors read the same completion spans; neither replaces the other.

    A processor the generator does not name is never constructed, and the run then
    reports no usage at all with nothing failing.
    """
    processors = generator._processors('bedrock_claude')

    assert processors['chat']['name'] == 'chat_tracker'
    assert processors['usage']['name'] == 'hitl_usage'


def test_the_worker_profile_applies_to_the_worker_and_not_to_the_user() -> None:
    """One profile per client, because a run has more than one provider.

    The worker may be on Vertex or Bedrock, which report cache creation separately
    from cache reads; the user is always a plain OpenAI-compatible endpoint. Reading
    one with the other's profile loses a counter that is priced on its own.
    """
    args = generator._processors('bedrock_claude')['usage']['args']

    assert args['profiles'][generator.WORKER_CLIENT] == 'bedrock_claude'
    assert args['default_profile'] == 'default', 'the user simulator is a plain endpoint'


def test_a_plain_worker_profile_reads_its_usage_with_the_default_parser() -> None:
    """``openai_basic`` and the like report one flat usage block."""
    args = generator._processors('openai_basic')['usage']['args']

    assert args['profiles'][generator.WORKER_CLIENT] == 'default'
    assert args['default_profile'] == 'default'


def test_the_named_usage_processor_is_registered_by_importing_the_recipe() -> None:
    """The runner resolves the name through the registry, and only imports the
    recipe package -- a processor registered in a module nothing imports is missing
    at run time, which is a failure at startup rather than in review."""
    import nexus.extensions.tasks.hitl.recipe.swe_chain_v2 as recipe

    from nexus.trace.processors import span_processor_registry

    del recipe
    config = generator._processors('bedrock_claude')['usage']
    processor = span_processor_registry.get(config['name'])(**config['args'])

    assert processor.profile_for(generator.WORKER_CLIENT) == 'bedrock_claude'
    assert processor.profile_for('human') == 'default'


def test_usage_comes_from_the_processor_even_when_the_manager_has_been_torn_down() -> None:
    """The defect this path removes, stated as behaviour rather than as line order.

    Run 0812g reported ``{"worker": {}}`` for 119 requests: ``shutdown()`` drops
    the handler the manager's counters live on, and the read happened afterwards.
    The processor holds its counters at run level, so a manager that has already
    forgotten everything cannot take the report down with it.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import _usage_summary
    from nexus.extensions.tasks.hitl.trace.labels import (
        LabelReadingClient,
        labelled,
        user_label,
        worker_label,
    )
    from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor
    from nexus.trace import observe, processor_context

    class Client(LabelReadingClient):
        _client_key = 'main'

        @observe('openai_completion', self_attrs=['_client_key', '_chat_phase'])
        async def create_completion(self, prompt_tokens: int):
            return {
                'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}],
                'model': 'claude-opus-5',
                'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 100},
            }

    async def drive() -> dict[str, Any]:
        processor = HITLUsageProcessor(profiles={'main': 'bedrock_claude'})
        with processor_context([processor], finalize_on_exit=False):
            client = Client()
            with labelled(client, user_label()):
                await client.create_completion(3000)
            with labelled(client, worker_label('worker')):
                await client.create_completion(9000)
            # Read with no manager at all, which is what the agent does: the counters
            # live on the processor, so a torn-down worker cannot empty the report.
            return _usage_summary()

    usage = asyncio.run(drive())

    assert usage['totals']['input_tokens'] == 12000, 'a torn-down manager must not empty the report'
    assert sorted(usage['by_label']) == [user_label(), worker_label('worker')]
    assert usage['by_label'][user_label()]['input_tokens'] == 3000, "the user's own turns are counted"


def test_a_run_with_no_usage_processor_reports_nothing_rather_than_the_worker_alone() -> None:
    """There is no manager-side fallback to fall back to.

    The manager used to keep its own counters, which covered the worker and not the
    user, and only until ``shutdown()``. The generator installs the processor
    unconditionally, so the fallback was unreachable; an empty report now says
    plainly that nothing counted the calls.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import _usage_summary
    from nexus.trace import processor_context

    with processor_context([], finalize_on_exit=False):
        assert _usage_summary() == {}


# -- the trajectories are exported once, not twice ------------------------


def test_the_record_does_not_carry_a_second_copy_of_the_trajectories() -> None:
    """Measured on run 0812c: the copy was exactly 50% of a 33.8 MB output file.

    Every trajectory the task used to copy was byte-identical to one the run-level
    tracker had already exported under ``processor_results.chat``. The agent still
    returns them -- the reward-hacking detector reads them off the agent output --
    but the task output has no field to put them in.
    """
    assert 'trajectories' not in SWEChainV2TaskOutput.model_fields
    assert 'trajectories' in AgentWithUserOutput.model_fields

    output = _output(
        output_with(
            trajectories={'worker': {'traces': {'trajectories': [{'messages': [{'role': 'user'}]}]}}},
            worker_rounds=3,
        ),
        total_steps=2,
    )

    assert 'trajectories' not in output.model_dump()
    assert output.worker_rounds == 3, 'the other agent fields must still come through'


# -- Grep and Glob reach the worker --------------------------------------


def sandbox_env(worker_config: dict[str, Any]) -> dict[str, str]:
    """The environment a worker session really gets for this config."""
    from nexus.extensions.tasks.hitl.actor.envs import session_env
    from nexus.extensions.tasks.hitl.actor.pty_claude_code.manager import (
        PtyClaudeCodeSession,
        PtyClaudeCodeSessionManager,
    )

    session = PtyClaudeCodeSession(name='worker', session_id='i', backend='pty_claude_code', config=worker_config)
    manager = PtyClaudeCodeSessionManager(default_config=worker_config)
    return session_env(worker_config, session=session, harness_env=manager.harness_env(worker_config))


def test_tool_search_is_on_so_grep_and_glob_are_disclosed() -> None:
    """Measured on run 0812e: all three workers saw 22 tools, Grep and Glob absent.

    They are only reachable through tool search, which is an experimental beta, and
    the PTY manager disables every beta by default. The workers fell back to
    ``grep`` under Bash. ``--allowedTools Grep,Glob`` was already being passed and
    changed nothing: it grants permission, not disclosure.
    """
    env = sandbox_env(_build_worker_config('default'))

    assert env['ENABLE_TOOL_SEARCH'] == 'true'
    assert env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] == '0', 'a disabled beta hides tool search'


def test_the_manager_default_would_otherwise_hide_it() -> None:
    """The override has to survive the manager's own defaults, which run first."""
    env = sandbox_env({})

    assert 'ENABLE_TOOL_SEARCH' not in env
    assert env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] == '1'


def test_tool_search_can_be_turned_off_for_one_run(monkeypatch) -> None:
    monkeypatch.setenv('HITL_WORKER_TOOL_SEARCH', '0')

    env = sandbox_env(_build_worker_config('default'))

    assert 'ENABLE_TOOL_SEARCH' not in env
    assert env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] == '1', 'off must restore the default'


def test_the_disclosure_mode_is_configurable(monkeypatch) -> None:
    """``auto:5`` discloses progressively; the plain flag discloses everything."""
    monkeypatch.setenv('HITL_WORKER_TOOL_SEARCH_MODE', 'auto:5')

    assert sandbox_env(_build_worker_config('default'))['ENABLE_TOOL_SEARCH'] == 'auto:5'



# -- the tags the user is sent are explained to it ------------------------


def emitted_tags() -> set[str]:
    """Every tag ``render_worker_turn`` can put in front of the user.

    Both branches are rendered with every field populated, so a tag added to the
    renderer joins this set without anyone updating a list here.
    """
    from nexus.extensions.tasks.hitl.actor.rendering import render_worker_turn

    fields: dict[str, Any] = {
        'final': 'f',
        'steps': ('s',),
        'note': 'n',
    }
    rendered = render_worker_turn(**fields) + render_worker_turn(**fields, question='q')
    return set(re.findall(r'<(\w+)>', rendered))


def user_system_prompt() -> str:
    """The prompt the agent really builds, with usage_docs taken from its own source.

    Building it from a hand-written docs tuple here would pass even if the agent
    stopped passing the protocol -- the test would be checking itself. So the
    agent's call site is parsed for the arguments it passes, and only those are
    used.
    """
    import ast
    import inspect

    from nexus.extensions.tasks.hitl.actor.rendering import WORKER_REPORT_PROTOCOL
    from nexus.extensions.tasks.hitl.agents import agent_with_user as agent_module
    from nexus.extensions.tasks.hitl.user_agent.personas import get_persona
    from nexus.extensions.tasks.hitl.user_agent.prompts import build_system_prompt, environment_facts
    from nexus.extensions.tasks.hitl.user_agent.tools import toolset

    source = inspect.getsource(agent_module.run_agent_with_user)
    tree = ast.parse(textwrap.dedent(source))
    passed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, 'id', '') != 'build_system_prompt':
            continue
        for keyword in node.keywords:
            if keyword.arg == 'usage_docs':
                passed = [name.id for name in ast.walk(keyword.value) if isinstance(name, ast.Name)]
    assert passed, 'the agent no longer calls build_system_prompt with usage_docs'

    tools = toolset.build(('send_message', 'accept', 'read', 'bash'), skills={})
    docs = list(tools.usage_docs) if 'tools' in passed else []
    if 'WORKER_REPORT_PROTOCOL' in passed:
        docs.append(('worker_report', WORKER_REPORT_PROTOCOL))
    return build_system_prompt(
        get_persona('default'),
        usage_docs=tuple(docs),
        environment=environment_facts(working_dir='/testbed', workers=['worker']),
    )


def test_every_tag_the_worker_report_uses_is_explained() -> None:
    """Measured on run 0812g: the user received every tag each turn and its system
    prompt explained none of them, because WORKER_REPORT_PROTOCOL was imported by
    nothing but its own test."""
    tags = emitted_tags()
    prompt = user_system_prompt()
    unexplained = sorted(tag for tag in tags if f'<{tag}>' not in prompt)

    assert len(tags) >= 4, f'the renderer emits only {sorted(tags)}; the walk stopped finding sections'
    assert not unexplained, f'the user reads {unexplained} with nothing telling it what they are'


def test_the_agent_passes_the_protocol_and_not_only_the_tool_docs() -> None:
    """The protocol is the agent's to add: user_agent does not import actor.

    Reading the source rather than running the agent, which needs a sandbox. The
    assertion is that the call site still combines both, since dropping the
    protocol half would leave the tags unexplained again with nothing failing.
    """
    import inspect

    from nexus.extensions.tasks.hitl.agents import agent_with_user as agent_module

    source = inspect.getsource(agent_module.run_agent_with_user)

    assert 'WORKER_REPORT_PROTOCOL' in source
    assert '*tools.usage_docs' in source, 'the tool docs must survive alongside the protocol'


@pytest.mark.parametrize(
    ('profile', 'enabled', 'thinking_type', 'expected'),
    [
        ('bedrock_claude', '1', 'adaptive', True),
        ('bedrock_claude', '1', 'disabled', False),
        ('bedrock_claude', '0', 'adaptive', False),
        ('gpt_responses', '1', '', True),
        ('gpt_responses', '0', '', False),
    ],
)
def test_the_worker_meta_reads_thinking_off_a_disabled_thinking_set(
    monkeypatch, profile: str, enabled: str, thinking_type: str, expected: bool
) -> None:
    """The nothink template runs HITL_WORKER_THINKING_ENABLED=true with
    HITL_WORKER_THINKING_TYPE=disabled: =0 stamps an empty set for this profile,
    dropping the effort along with the thinking block. Reading the first variable
    alone published thinking_enabled=True for every one of those runs."""
    monkeypatch.setenv('HITL_WORKER_THINKING_ENABLED', enabled)
    monkeypatch.setenv('HITL_WORKER_MODEL', 'a-model')
    if thinking_type:
        monkeypatch.setenv('HITL_WORKER_THINKING_TYPE', thinking_type)
    else:
        monkeypatch.delenv('HITL_WORKER_THINKING_TYPE', raising=False)

    meta = generator._worker_meta('pty_claude_code', {'version': '2.1.117'}, profile)

    assert meta['thinking_enabled'] is expected


def test_one_effort_reaches_the_client_the_codex_config_and_the_metadata(monkeypatch) -> None:
    """Three readers, one value. Codex builds its own requests, so its config carries
    the effort separately from the client's request args -- read apart, a run can send
    one effort and publish another."""
    import tempfile
    import tomllib

    from pathlib import Path

    from nexus.extensions.tasks.hitl.actor.print_codex.runtime import _write_config
    from nexus.extensions.tasks.hitl.model import resolve_profile
    from nexus.extensions.tasks.hitl.utils.generator_util import _worker_profile_kwargs
    from nexus.extensions.tasks.hitl.utils.generator_util.codex import _build_codex_worker_config

    monkeypatch.setenv('HITL_WORKER_REASONING_EFFORT', 'none')
    monkeypatch.setenv('HITL_WORKER_MODEL', 'gpt-5.6-sol')
    monkeypatch.setenv('HITL_CODEX_MODEL', 'gpt-5.6-sol')

    client = resolve_profile('gpt_responses')(**_worker_profile_kwargs())
    home = Path(tempfile.mkdtemp())
    _write_config(home, proxy_port=1234, config=_build_codex_worker_config('gpt_responses'))
    meta = generator._worker_meta('print_codex', {'version': '0.154.0'}, 'gpt_responses')

    assert client.think_request_args['reasoning']['effort'] == 'none'
    assert tomllib.loads((home / 'config.toml').read_text())['model_reasoning_effort'] == 'none'
    assert meta['reasoning_effort'] == 'none'
