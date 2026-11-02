import base64
import importlib
import json

from types import SimpleNamespace

import pytest


def test_code_mt_agents_declare_generic_gateway_scaffold_policy():
    from nexus.core.agent import get_agent_gateway_scaffold_policy

    importlib.import_module('nexus.extensions.projects.code_mt.agent.agent')

    assert get_agent_gateway_scaffold_policy('code_mt') == 'generic'
    assert get_agent_gateway_scaffold_policy('ch_code_mt') == 'generic'


def test_code_mt_task_imports_without_private_harbor_deliverable():
    module = importlib.import_module('nexus.extensions.projects.code_mt.task.task')

    assert hasattr(module, '_persist_code_mt_deliverable')


@pytest.mark.asyncio
async def test_code_mt_owns_repoqa_deliverable_staging():
    from nexus.extensions.projects.code_mt.task.task import _persist_code_mt_deliverable

    class FakeRuntime:
        def __init__(self):
            self.calls = []

        async def run_command(self, *, command, timeout, stdin=None):
            self.calls.append({'command': command, 'timeout': timeout, 'stdin': stdin})
            return SimpleNamespace(return_code=0, stdout='', stderr='')

    runtime = FakeRuntime()
    ctx = SimpleNamespace(
        tool_state={
            'answer': 'submitted answer',
            '_code_mt_messages': [
                {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'assistant text'},
            ],
        }
    )

    await _persist_code_mt_deliverable(runtime, ctx)

    assert len(runtime.calls) == 3
    assert base64.b64decode(runtime.calls[1]['stdin']).decode() == 'submitted answer'
    assert '[user]\nquestion' in base64.b64decode(runtime.calls[2]['stdin']).decode()


def _scan(messages: list[dict]) -> list[dict]:
    from nexus.extensions.projects.code_mt.task.task import scan_trajectory_for_anti_hack

    violations, _summary = scan_trajectory_for_anti_hack(
        messages,
        repo='owner/project',
        working_dir='/testbed',
        instance_id='owner__project-123',
        base_commit='abc1234def5678',
        history_truncated=False,
    )
    return violations


def test_code_mt_anti_hack_pairs_multiple_tool_results_by_id():
    messages = [
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'blocked',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': 'curl http://blocked.invalid'})},
                },
                {
                    'id': 'read',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': 'git show origin/main'})},
                },
            ],
        },
        {'role': 'tool', 'tool_call_id': 'blocked', 'content': 'Failed to connect to blocked.invalid'},
        {'role': 'tool', 'tool_call_id': 'read', 'content': 'commit abcdef0123456789\nremote source'},
    ]

    assert {violation['reason'] for violation in _scan(messages)} >= {'git_unverified_remote_ref'}


def test_code_mt_anti_hack_does_not_exempt_compound_command_on_blocked_network():
    messages = [
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'compound',
                    'function': {
                        'name': 'bash',
                        'arguments': json.dumps({'command': 'git show origin/main; curl http://blocked.invalid'}),
                    },
                }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'compound',
            'content': 'commit abcdef0123456789\nremote source\nFailed to connect to blocked.invalid',
        },
    ]

    assert {violation['reason'] for violation in _scan(messages)} >= {'git_unverified_remote_ref'}


def test_code_mt_anti_hack_ignores_failed_unverified_commit_lookup():
    messages = [
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'read',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': 'git show deadbeef'})},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'read', 'content': 'fatal: bad object deadbeef'},
    ]

    assert _scan(messages) == []


def test_code_mt_full_scan_keeps_base_commit_scope_for_setup():
    from nexus.extensions.projects.code_mt.task.task import scan_trajectory_for_anti_hack

    messages = [
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'read',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': 'git show deadbeef'})},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'read', 'content': 'source from another commit'},
    ]
    violations, _summary = scan_trajectory_for_anti_hack(
        messages,
        repo='owner/project',
        working_dir='/testbed',
        instance_id='owner__project-123',
        base_commit='abc1234def5678',
        history_truncated=False,
    )

    assert {violation['reason'] for violation in violations} >= {'git_unverified_commit_read'}
