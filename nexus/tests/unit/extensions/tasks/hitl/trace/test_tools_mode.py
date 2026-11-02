"""The tracker's export shape, which decides what survives a run.

``ToolsTrackingMode`` looks like a formatting choice and is not. ``global_first``
keeps one tool schema for the whole run -- the first one seen -- and exports each
trajectory as a bare message list, so two things are lost with no error anywhere:
the tool schema of every call after the first, and the per-request tag that says
which session a trajectory belongs to.

These tests pin the mode, the two losses that motivate it, and that asking for
the wrong mode fails instead of quietly degrading. They drive a real
``ChatTrackerProcessor`` through a real ``@observe`` span, because the losses only
appear on the export path.
"""

import asyncio

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.trace.labels import REQUIRED_TOOLS_MODE
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.tracer import observe


# The tracker dispatches spans on a background task; this is how long the tests
# wait for it to settle before reading the result.
DISPATCH_SETTLE = 0.6

BASH = {'name': 'bash'}
BROWSER = {'name': 'browser'}


class Client:
    """A client whose completion is traced by the real decorator.

    ``_chat_phase`` is in the real client's ``self_attrs`` list, so setting it is
    how a per-session tag reaches the span. ``**kwargs`` matters: the tracker
    reads ``tools`` out of ``inputs['kwargs']``.
    """

    _chat_phase: str | None = None

    @observe('openai_completion', self_attrs=['_chat_phase'])
    async def create_completion(self, messages: list[dict], model: str = 'm', **kwargs: Any) -> dict:
        return {
            'choices': [{'message': {'role': 'assistant', 'content': kwargs['reply']}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }


async def _two_calls_with_different_tools(processors: list[Any], *, tag_first: str = '', tag_second: str = '') -> None:
    """One client, two calls, each offering a different tool set."""
    client = Client()
    with processor_context(processors, finalize_on_exit=False):
        client._chat_phase = tag_first or None
        await client.create_completion([{'role': 'user', 'content': 'first'}], tools=[BASH], reply='R1')
        client._chat_phase = tag_second or None
        await client.create_completion([{'role': 'user', 'content': 'second'}], tools=[BASH, BROWSER], reply='R2')
        await asyncio.sleep(DISPATCH_SETTLE)


# --- the default -----------------------------------------------------------


def test_the_required_mode_is_branch_last():
    assert REQUIRED_TOOLS_MODE == 'branch_last'


def test_the_recipe_configures_the_required_mode():
    """The generator is the only place the mode is chosen for a run."""
    from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.generator import CHAT_PROCESSOR

    assert CHAT_PROCESSOR['chat']['args']['tools_mode'] == REQUIRED_TOOLS_MODE


# --- loss 1: the tool schema of every call after the first -----------------


@pytest.mark.asyncio
async def test_a_mid_run_tool_set_change_survives():
    """A worker that gains a tool mid-run must be described by what it had.

    An MCP server coming up, or a compaction boundary, changes the tool set
    between calls.
    """
    tracker = ChatTrackerProcessor(tools_mode=REQUIRED_TOOLS_MODE)

    await _two_calls_with_different_tools([tracker])

    exported = tracker.get_result()['trajectories']
    assert [[tool['name'] for tool in entry['tools']] for entry in exported] == [
        ['bash'],
        ['bash', 'browser'],
    ]


@pytest.mark.asyncio
async def test_global_first_would_have_kept_only_the_first_schema():
    """The loss this default prevents, measured against the upstream processor.

    Without this, "we chose branch_last" would rest on the docstring alone.
    """
    processor = ChatTrackerProcessor(tools_mode='global_first')

    await _two_calls_with_different_tools([processor])

    result = processor.snapshot_result()
    # One schema for the whole run, and it is the first one: `browser` is gone.
    assert [tool['name'] for tool in result['tools']] == ['bash']
    assert all(isinstance(entry, list) for entry in result['trajectories'])


# --- loss 2: the per-session tag ------------------------------------------


@pytest.mark.asyncio
async def test_the_per_session_tag_reaches_the_export():
    tracker = ChatTrackerProcessor(tools_mode=REQUIRED_TOOLS_MODE)

    await _two_calls_with_different_tools(
        [tracker],
        tag_first='worker/_/a',
        tag_second='worker/_/b',
    )

    exported = tracker.get_result()['trajectories']
    assert [entry.get('role') for entry in exported] == ['worker/_/a', 'worker/_/b']


@pytest.mark.asyncio
async def test_global_first_has_nowhere_to_put_the_tag():
    """A bare list cannot carry a tag, so attribution is unrecoverable."""
    processor = ChatTrackerProcessor(tools_mode='global_first')

    await _two_calls_with_different_tools([processor], tag_first='worker/_/a', tag_second='worker/_/b')

    exported = processor.snapshot_result()['trajectories']
    assert all(isinstance(entry, list) for entry in exported)
    assert all('worker/_/a' not in str(entry) for entry in exported)


# --- the shape downstream consumers see -----------------------------------


@pytest.mark.asyncio
async def test_exported_entries_are_dicts_with_messages_and_tools():
    """The shape ``recipe/swe/reward_hacking.worker_chat_messages`` reads."""
    tracker = ChatTrackerProcessor(tools_mode=REQUIRED_TOOLS_MODE)

    await _two_calls_with_different_tools([tracker])

    result = tracker.get_result()
    # branch_last carries tools per trajectory, so there is no top-level key.
    assert 'tools' not in result
    for entry in result['trajectories']:
        assert isinstance(entry, dict)
        assert isinstance(entry['messages'], list)
        assert entry['messages'][-1]['role'] == 'assistant'


# --- run_level_trajectories checks the mode it is handed -------------------


def test_run_level_trajectories_is_empty_before_a_tracker_exists():
    """No tracker installed is the before-the-run case, not a misconfiguration."""
    from nexus.extensions.tasks.hitl.trace.labels import run_level_trajectories

    assert run_level_trajectories() == {'trajectories': []}


def test_run_level_trajectories_accepts_the_required_mode():
    from nexus.extensions.tasks.hitl.trace.labels import run_level_trajectories

    tracker = ChatTrackerProcessor(tools_mode=REQUIRED_TOOLS_MODE)

    with processor_context([tracker], finalize_on_exit=False):
        assert 'trajectories' in run_level_trajectories()


def test_run_level_trajectories_refuses_global_first():
    """The bypass this closes.

    The processor is built from the recipe's generator config, so a wrong mode
    is not caught at construction. Left alone, every selector matches nothing:
    measured, ``traces(session)`` returned zero trajectories with no error and
    ``unattributed_count`` reported 2.
    """
    from nexus.extensions.tasks.hitl.trace.labels import LabelError, run_level_trajectories

    tracker = ChatTrackerProcessor(tools_mode='global_first')

    with processor_context([tracker], finalize_on_exit=False):
        with pytest.raises(LabelError, match='run-level chat_tracker'):
            run_level_trajectories()


def test_the_run_level_refusal_says_how_to_configure_it():
    from nexus.extensions.tasks.hitl.trace.labels import LabelError, run_level_trajectories

    tracker = ChatTrackerProcessor(tools_mode='global_first')

    with processor_context([tracker], finalize_on_exit=False):
        with pytest.raises(LabelError, match='tools_mode') as excinfo:
            run_level_trajectories()

    assert "'tools_mode': 'branch_last'" in str(excinfo.value)
