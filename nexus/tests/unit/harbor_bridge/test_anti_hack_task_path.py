"""Task-level tests: does a HyCode hack survive the REAL harbor_bridge_task path?

The sibling ``test_anti_hack_scaffold_parity.py`` drives the strategy directly. That
is the wrong altitude to prove the fix ships: it hand-builds the trajectory and so
cannot catch a regression in the steps BETWEEN the Agent returning and the strategy
running -- specifically

  * ``extract_messages``, which for HyCode parses CLI stdout rather than reading a
    ``main_trajectory`` field,
  * ``strip_reasoning``, which filters content BLOCKS and drops any message left
    empty -- the same block list the bridge reads, and
  * ``capture_agent_trajectory`` / ``_chat_tracker_fallback``, which decide whether a
    trajectory is even built.

These tests therefore call ``harbor_bridge_task`` itself with a real ``HyCodeOutput``
and assert on ``output.extra_fields``, which is what HyTuner consumes to decide
CODE_REWARD_HACKING. They reuse the fakes from ``test_separate_verifier`` rather than
re-deriving them, so the two files cannot drift on what "the task path" means.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pytest

import nexus.extensions.harbor_bridge.task as task_module

from nexus.extensions.agents.hycode.agent import HyCodeOutput
from tests.unit.harbor_bridge.test_separate_verifier import _FakeContext, _FakeEnvironment


CLONE = 'git clone https://github.com/foo/bar /tmp/x'
POLICY = {
    'id': 'code-anti-hack-v1',
    'delivery': 'none',
    'prompt': '',
    'trajectory_strategy': 'restricted-source-access-v1',
}


def _write_swe_task(root: Path) -> Path:
    """A shared-verifier task carrying the [other] anchors the lookup tier needs.

    ``repo`` + ``base_commit`` are what ``_is_swe_style`` requires; without both the
    answer-lookup channels stay disabled by design, so a test that omitted them
    would silently exercise only half the detector.
    """
    task_dir = root / 'task'
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'task.toml').write_text(
        """schema_version = "1.2"

[environment]
docker_image = "agent:v1"
workdir = "/app"
cpus = 2
memory_mb = 4096
network_mode = "public"

[verifier]
environment_mode = "shared"
timeout_sec = 120

[other]
repo = "foo/bar"
base_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
task_type = "SWE"
"""
    )
    (task_dir / 'instruction.md').write_text('Complete the task.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    return task_dir


def _hycode_stdout(command: str, observation: str) -> str:
    """One HyCode CLI stream: an Anthropic tool_use turn plus its tool_result.

    This is the actual wire format -- newline-delimited JSON events whose messages
    carry actions as content blocks -- so the test exercises the stdout parser
    instead of assuming its output.
    """
    events = [
        {'type': 'system', 'subtype': 'init', 'session_id': 's1'},
        {
            'type': 'assistant',
            'message': {
                'role': 'assistant',
                'content': [
                    {'type': 'thinking', 'thinking': 'I could just clone it'},
                    {'type': 'text', 'text': 'cloning'},
                    {'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'command': command}},
                ],
            },
        },
        {
            'type': 'user',
            'message': {
                'role': 'user',
                'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': observation}],
            },
        },
        {'type': 'result', 'subtype': 'success', 'num_turns': 2, 'is_error': False},
    ]
    return '\n'.join(json.dumps(event) for event in events)


async def _run_task(monkeypatch, task_dir: Path, agent_output: Any, **task_input_overrides: Any) -> Any:
    events: list[str] = []
    context = _FakeContext(events)

    async def run_agent(**_: Any):
        events.append('agent')
        return agent_output

    async def run_verifier(**_: Any):
        events.append('verify')
        return {'reward': 1.0}, {}, 'passed'

    import nexus.extensions.harbor_bridge.environment as environment_module

    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(task_module, 'update_span', lambda _span_id, *, attributes: None)
    monkeypatch.setattr(task_module, '_run_harbor_verifier', run_verifier)
    monkeypatch.setattr(environment_module, 'GongfengEnvironment', _FakeEnvironment)

    task_input_kwargs = {
        'task_dir': str(task_dir),
        'workdir': '/app',
        'verifier_mode': 'shared',
        'anti_hack_policy': POLICY,
        'run_agent': run_agent,
        **task_input_overrides,
    }
    return await task_module.harbor_bridge_task(task_module.HarborBridgeTaskInput(**task_input_kwargs))


def _hycode_output(command: str, observation: str = 'ok') -> HyCodeOutput:
    return HyCodeOutput(
        success=True,
        output=_hycode_stdout(command, observation),
        stderr='',
        exit_code=0,
        execution_time=1.0,
        termination_reason='ENV_DONE',
        result_subtype='success',
        num_turns=2,
        exit_status='env_done',
    )


@pytest.mark.asyncio
async def test_hycode_hack_is_flagged_through_the_task_path(monkeypatch, tmp_path):
    """The end-to-end claim: a HyCode clone reaches HyTuner as is_hack=True.

    Before the Anthropic-block bridge this returned is_hack=False with
    anti_hack_status='completed' -- a clean verdict over an unread trajectory, which
    is exactly what made the measured hack rate meaningless for HyCode datasets.
    """
    output = await _run_task(monkeypatch, _write_swe_task(tmp_path), _hycode_output(CLONE, 'Cloning into /tmp/x'))
    assert output.extra_fields['is_hack'] is True
    assert output.extra_fields['anti_hack_status'] == 'completed'
    # The contract stays two keys: a flag and a status, no reason taxonomy.
    assert set(output.extra_fields) == {'is_hack', 'anti_hack_status'}


@pytest.mark.asyncio
async def test_hycode_answer_lookup_is_flagged_through_the_task_path(monkeypatch, tmp_path):
    """The ported CM-2 tier must fire on the task path too, not just in isolation.

    ``installed_copy_read`` is a purely local channel, so none of the pre-existing
    network patterns can catch it -- it is only detected because the lookup tier was
    brought into harbor_bridge.
    """
    output = await _run_task(
        monkeypatch,
        _write_swe_task(tmp_path),
        _hycode_output('cat /usr/lib/python3/site-packages/bar/core.py', 'def fix(): pass'),
    )
    assert output.extra_fields['is_hack'] is True


@pytest.mark.asyncio
async def test_benign_hycode_run_is_not_flagged_through_the_task_path(monkeypatch, tmp_path):
    output = await _run_task(monkeypatch, _write_swe_task(tmp_path), _hycode_output('ls -la /app', 'core.py'))
    assert output.extra_fields['is_hack'] is False
    assert output.extra_fields['anti_hack_status'] == 'completed'


@pytest.mark.asyncio
async def test_reasoning_blocks_do_not_hide_the_tool_use_beside_them(monkeypatch, tmp_path):
    """strip_reasoning() runs before the strategy and edits the same block list.

    It drops ``thinking`` blocks and discards any message they emptied. The hack
    turn above carries thinking + text + tool_use together, so this asserts the
    stripping removes only the reasoning and leaves the action detectable.
    """
    task_dir = _write_swe_task(tmp_path)
    output = await _run_task(monkeypatch, task_dir, _hycode_output(CLONE, 'Cloning into /tmp/x'))
    assert output.extra_fields['is_hack'] is True


@pytest.mark.asyncio
async def test_bad_pattern_quality_prefers_tracker_reasoning_over_nonempty_stdout(monkeypatch, tmp_path):
    reasoning = 'Let me search.\n' * 14

    async def tracker_fallback(_runtime, *, final_answer=None):
        return ([{'role': 'assistant', 'content': '', 'reasoning_content': reasoning}], [])

    monkeypatch.setattr(task_module, '_chat_tracker_fallback', tracker_fallback)
    output = await _run_task(
        monkeypatch,
        _write_swe_task(tmp_path),
        _hycode_output('ls -la /app', 'core.py'),
        anti_hack_policy=None,
        tool_call_bad_pattern_policy={
            'id': 'tracker-reasoning-v1',
            'penalty': 0.5,
            'patterns': {
                'reasoning_repetition_loop': {'enabled': True, 'penalize': True},
                'reasoning_selftalk_flood': {'enabled': True, 'penalize': True},
            },
        },
    )

    assert output.extra_fields['tool_call_bad_pattern']['issue_counts'] == {
        'reasoning_repetition_loop': 1,
        'reasoning_selftalk_flood': 1,
    }


@pytest.mark.asyncio
async def test_reasoning_flood_reaches_task_detection_and_reward_settlement(monkeypatch, tmp_path):
    reasoning = ''.join(f'Need maybe maybe mention "item {i} is config only". Good.\n' for i in range(60))

    async def tracker_fallback(_runtime, *, final_answer=None):
        return ([{'role': 'assistant', 'content': '', 'reasoning_content': reasoning}], [])

    monkeypatch.setattr(task_module, '_chat_tracker_fallback', tracker_fallback)
    output = await _run_task(
        monkeypatch,
        _write_swe_task(tmp_path),
        _hycode_output('ls -la /app', 'core.py'),
        anti_hack_policy=None,
        tool_call_bad_pattern_policy={
            'id': 'reasoning-flood-v1',
            'penalty': 0.5,
            'reward_threshold': 0.5,
            'patterns': {'reasoning_selftalk_flood': {'enabled': True, 'penalize': True}},
        },
    )

    detection = output.extra_fields['tool_call_bad_pattern']
    assert detection['issue_counts'] == {'reasoning_selftalk_flood': 1}
    assert detection['hits'][0]['detail']['kind'] == 'template_loop'
    assert detection['penalty'] == 0.5
    assert detection['original_score'] == 1.0
    assert output.score == 0.5


@pytest.mark.asyncio
async def test_code_mt_shaped_output_keeps_working_through_the_task_path(monkeypatch, tmp_path):
    """Regression guard for the scaffold that already worked.

    A ``main_trajectory`` short-circuits the stdout parser, so this exercises the
    other branch of extract_messages and proves the bridge is additive.
    """

    class _CodeMtOutput:
        exit_status = 'submitted'
        final_answer = 'done'
        main_trajectory = [
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {'name': 'bash', 'arguments': json.dumps({'command': CLONE})},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Cloning into /tmp/x'},
        ]

    output = await _run_task(monkeypatch, _write_swe_task(tmp_path), _CodeMtOutput())
    assert output.extra_fields['is_hack'] is True
