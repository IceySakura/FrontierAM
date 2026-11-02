from __future__ import annotations

import json

import pytest

from nexus.core.agent import get_agent_gateway_scaffold_policy
from nexus.gateway.policies.claude_code import CLAUDE_CODE_POLICY
from nexus.gateway.policies.hycode import COMPACT_SUMMARY_MARKER, COMPACT_TRIGGER_MARKER
from nexus.gateway.scaffold_policy import (
    InvocationKind,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    build_scaffold_policy,
)
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


# Producer text below is copied from a Claude Code 2.1.286 capture through the Anthropic proxy.
SYNC_RESULT_TRAILER = (
    "agentId: {agent_id} (use SendMessage with to: '{agent_id}', summary: '<5-10 word recap>' to continue this agent)\n"
    '<usage>subagent_tokens: 16525\ntool_uses: 1\nduration_ms: 4501</usage>'
)
ASYNC_LAUNCH_RESULT = (
    'Async agent launched successfully. (This tool result is internal metadata — never quote or paste any part of '
    'it, including the agentId below, into a user-facing reply.)\n'
    "agentId: {agent_id} (internal ID - do not mention to user. Use SendMessage with to: '{agent_id}', "
    "summary: '<5-10 word recap>' to continue this agent.)\n"
    'The agent is working in the background. You will be notified automatically when it completes.'
)
TASK_NOTIFICATION = (
    '<task-notification>\n'
    '<task-id>{agent_id}</task-id>\n'
    '<tool-use-id>{tool_use_id}</tool-use-id>\n'
    '<output-file>/tmp/claude/tasks/{agent_id}.output</output-file>\n'
    '<status>completed</status>\n'
    '<summary>Agent "Run sleep and echo test" finished</summary>\n'
    '</task-notification>'
)
SYSTEM_NOTIFICATION = (
    '<system-reminder>\n[SYSTEM NOTIFICATION - NOT USER INPUT]\n'
    'This is an automated background-task event, NOT a message from the user.\n\n{body}\n</system-reminder>'
)
REMINDER = {'type': 'text', 'text': '<system-reminder>\n# gitStatus\nCurrent branch: main\n</system-reminder>\n'}


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _spawn(*calls: tuple[str, str]) -> dict:
    return {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            _tool_call(call_id, 'Agent', {'description': 'probe', 'prompt': prompt}) for call_id, prompt in calls
        ],
    }


def _request_facts(messages: list[dict], *, agent_id: str | None = None):
    request_meta = {'agent_id': agent_id} if agent_id is not None else None
    return CLAUDE_CODE_POLICY.inspect_event(RequestEvent(tuple(messages), None, request_meta))


class TestRegistration:
    def test_registry_resolves_claude_code(self) -> None:
        assert build_scaffold_policy('claude_code') is CLAUDE_CODE_POLICY

    @pytest.mark.parametrize(
        'agent_name',
        ['claude_code', 'claude_code_native_bridge', 'claude_code_multiturn', 'claude_code_interactive'],
    )
    def test_claude_code_agents_declare_the_policy(self, agent_name: str) -> None:
        import nexus.extensions.agents.claude_code.agent  # noqa: F401  (agent registration)

        assert get_agent_gateway_scaffold_policy(agent_name) == 'claude_code'

    def test_prefers_extended_responses(self) -> None:
        assert CLAUDE_CODE_POLICY.response_policy is ResponsePolicy.PREFER_EXTENDED_ELSE_LAST


class TestChainMembership:
    def test_request_without_agent_header_stays_on_main(self) -> None:
        facts = _request_facts([{'role': 'user', 'content': 'hello'}])

        assert facts.membership is None
        assert facts.context_key == ('claude_code', 'main', False)

    def test_agent_header_declares_a_subagent_chain(self) -> None:
        facts = _request_facts([{'role': 'user', 'content': 'child prompt'}], agent_id='adcc4fbdcfdb0b3f6')

        assert facts.membership is not None
        assert facts.membership.kind == 'subagent'
        assert facts.context_key == ('claude_code', 'adcc4fbdcfdb0b3f6', True)

    def test_session_title_request_is_a_utility_chain(self) -> None:
        system = {
            'role': 'system',
            'content': (
                'x-anthropic-billing-header: cc_version=2.1.286.77f; cc_entrypoint=nexus-pty;\n'
                "You are Claude Code, Anthropic's official CLI for Claude.\n"
                'You are naming a coding session so the user can pick it out of a long list of sessions. The title '
                'is a name for what the session is about.'
            ),
        }
        user = {'role': 'user', 'content': '<session>\nRead a.txt and summarize it in one sentence.\n</session>'}

        facts = _request_facts([system, user])

        assert facts.membership is not None
        assert facts.membership.kind == 'session_title'
        assert facts.context_key == ('claude_code', 'utility:session_title', True)

    def test_prompt_suggestion_forked_from_main_is_a_utility_chain(self) -> None:
        suggestion = {
            'role': 'user',
            'content': (
                '[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]\n\n'
                "FIRST: Look at the user's recent messages and original request."
            ),
        }
        history = [
            {'role': 'user', 'content': 'Read a.txt and summarize it in one sentence.'},
            {'role': 'assistant', 'content': '`a.txt` lists three lines.'},
        ]

        facts = _request_facts([*history, suggestion])

        assert facts.membership is not None
        assert facts.membership.kind == 'prompt_suggestion'
        assert facts.child_prompts == ()

    def test_suggestion_text_earlier_in_history_stays_on_main(self) -> None:
        suggestion = {
            'role': 'user',
            'content': '[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]',
        }

        facts = _request_facts([suggestion, {'role': 'user', 'content': 'real follow-up'}])

        assert facts.membership is None

    def test_legacy_hycode_title_prompt_is_not_claimed(self) -> None:
        title = {'role': 'system', 'content': 'Generate a concise, sentence-case title (3-7 words) for this session.'}

        assert _request_facts([title]).membership is None


class TestSpawnAndJoin:
    def test_agent_call_is_a_spawn_but_task_aliases_are_not(self) -> None:
        message = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                _tool_call('spawn-1', 'Agent', {'prompt': 'list   the files'}),
                _tool_call('legacy-1', 'Task', {'prompt': 'legacy spawn'}),
                _tool_call('poll-1', 'TaskOutput', {'task_id': 'child-1'}),
            ],
        }

        facts = _request_facts([message])

        assert [(intent.tool_use_id, intent.child_prompt) for intent in facts.spawn_intents] == [
            ('spawn-1', 'list the files')
        ]
        assert facts.retrieval_intents == ()

    def test_sync_result_trailer_binds_and_delivers(self) -> None:
        result = {
            'role': 'tool',
            'tool_call_id': 'spawn-1',
            'content': 'The report follows:\n  2 files\n' + SYNC_RESULT_TRAILER.format(agent_id='adcc4fbdcfdb0b3f6'),
        }

        facts = _request_facts([_spawn(('spawn-1', 'list the files')), result])

        assert [(b.tool_use_id, b.agent_id) for b in facts.spawn_bindings] == [('spawn-1', 'adcc4fbdcfdb0b3f6')]
        assert [receipt.delivered for receipt in facts.delivery_receipts] == [True]

    def test_async_launch_binds_without_delivering(self) -> None:
        result = {
            'role': 'tool',
            'tool_call_id': 'spawn-1',
            'content': ASYNC_LAUNCH_RESULT.format(agent_id='a3565e204a1e4873a'),
        }

        facts = _request_facts([_spawn(('spawn-1', 'sleep then echo')), result])

        assert [(b.tool_use_id, b.agent_id) for b in facts.spawn_bindings] == [('spawn-1', 'a3565e204a1e4873a')]
        assert [receipt.delivered for receipt in facts.delivery_receipts] == [False]

    def test_task_notification_delivers_the_background_agent(self) -> None:
        notification = {
            'role': 'user',
            'content': TASK_NOTIFICATION.format(agent_id='a3565e204a1e4873a', tool_use_id='spawn-1'),
        }

        facts = _request_facts([_spawn(('spawn-1', 'sleep then echo')), notification])

        assert [delivery.agent_id for delivery in facts.notification_deliveries] == ['a3565e204a1e4873a']
        assert [(b.tool_use_id, b.agent_id) for b in facts.spawn_bindings] == [('spawn-1', 'a3565e204a1e4873a')]

    def test_notification_appended_to_a_running_tool_result_delivers(self) -> None:
        bash = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_tool_call('bash-1', 'Bash', {'command': 'true'})],
        }
        result = {
            'role': 'tool',
            'tool_call_id': 'bash-1',
            'content': '(Bash completed with no output)\n\n'
            + SYSTEM_NOTIFICATION.format(
                body=TASK_NOTIFICATION.format(agent_id='aa080b1cf22f891cf', tool_use_id='spawn-1')
            ),
        }

        facts = _request_facts([_spawn(('spawn-1', 'compute 5 @ 6')), bash, result])

        assert [delivery.agent_id for delivery in facts.notification_deliveries] == ['aa080b1cf22f891cf']
        assert [(b.tool_use_id, b.agent_id) for b in facts.spawn_bindings] == [('spawn-1', 'aa080b1cf22f891cf')]

    @pytest.mark.parametrize(
        'content',
        [
            TASK_NOTIFICATION.format(agent_id='a1', tool_use_id='spawn-1'),
            SYSTEM_NOTIFICATION.format(body=TASK_NOTIFICATION.format(agent_id='a1', tool_use_id='bash-1')),
        ],
        ids=['bare-tag-in-tool-output', 'notification-for-a-shell-task'],
    )
    def test_tool_output_alone_is_not_an_agent_delivery(self, content: str) -> None:
        bash = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_tool_call('bash-1', 'Bash', {'command': 'cat notes.txt'})],
        }
        result = {'role': 'tool', 'tool_call_id': 'bash-1', 'content': content}

        facts = _request_facts([_spawn(('spawn-1', 'compute 5 @ 6')), bash, result])

        assert facts.notification_deliveries == ()

    def test_identical_parallel_prompts_join_their_own_chains(self) -> None:
        prompt = 'Run the Bash command date and report its output verbatim.'
        assembler = TrajectoryAssembler(scaffold_policy='claude_code')
        main_input = [{'role': 'user', 'content': 'delegate twice'}]
        spawn = _spawn(('spawn-alpha', prompt), ('spawn-beta', prompt))
        assembler.record_completion(main_input, spawn)

        child_input = [{'role': 'user', 'content': [REMINDER, {'type': 'text', 'text': prompt}]}]
        for agent_id in ('a239ab440fe93ec86', 'af46798a02ad4edfe'):
            assembler.record_completion(
                child_input, {'role': 'assistant', 'content': f'{agent_id} done'}, request_meta={'agent_id': agent_id}
            )

        results = [
            {'role': 'tool', 'tool_call_id': call_id, 'content': SYNC_RESULT_TRAILER.format(agent_id=agent_id)}
            for call_id, agent_id in (('spawn-alpha', 'a239ab440fe93ec86'), ('spawn-beta', 'af46798a02ad4edfe'))
        ]
        assembler.record_completion([*main_input, spawn, *results], {'role': 'assistant', 'content': 'both done'})

        trajectory = assembler.freeze()
        assert set(trajectory.chains) == {'main', 'a239ab440fe93ec86', 'af46798a02ad4edfe'}
        for agent_id, call_id in (('a239ab440fe93ec86', 'spawn-alpha'), ('af46798a02ad4edfe', 'spawn-beta')):
            chain = trajectory.chains[agent_id]
            assert chain.spawn is not None
            assert chain.join is not None
            assert chain.spawn.tool_use_id == call_id
            assert (chain.join.kind, chain.join.tool_use_id) == ('inline', call_id)

    def test_background_agent_joins_on_notification(self) -> None:
        assembler = TrajectoryAssembler(scaffold_policy='claude_code')
        main_input = [{'role': 'user', 'content': 'run it in the background'}]
        spawn = _spawn(('spawn-1', 'sleep then echo'))
        assembler.record_completion(main_input, spawn)
        assembler.record_completion(
            [{'role': 'user', 'content': 'sleep then echo'}],
            {'role': 'assistant', 'content': 'BG_DONE'},
            request_meta={'agent_id': 'a3565e204a1e4873a'},
        )
        launch = {
            'role': 'tool',
            'tool_call_id': 'spawn-1',
            'content': ASYNC_LAUNCH_RESULT.format(agent_id='a3565e204a1e4873a'),
        }
        notification = {
            'role': 'user',
            'content': TASK_NOTIFICATION.format(agent_id='a3565e204a1e4873a', tool_use_id='spawn-1'),
        }
        assembler.record_completion(
            [*main_input, spawn, launch, notification], {'role': 'assistant', 'content': 'received'}
        )

        chain = assembler.freeze().chains['a3565e204a1e4873a']

        assert chain.spawn is not None
        assert chain.join is not None
        assert chain.spawn.tool_use_id == 'spawn-1'
        assert chain.join.kind == 'notify'


class TestCompaction:
    def test_trigger_classifies_summary_inference(self) -> None:
        trigger = {
            'role': 'user',
            'content': f'{COMPACT_TRIGGER_MARKER}\n\n- Do NOT use Read, Bash, or ANY other tool.',
        }

        facts = _request_facts([{'role': 'user', 'content': 'read the files'}, trigger])

        assert [(fact.rule_name, fact.segment_kind) for fact in facts.boundary_facts] == [
            ('claude_code_summary_inference', 'summary_inference')
        ]
        assert facts.invocation_kind is InvocationKind.SUMMARY_INFERENCE

    def test_resume_marker_after_a_reminder_block_opens_a_segment(self) -> None:
        summary = {'type': 'text', 'text': f'{COMPACT_SUMMARY_MARKER}. The summary below covers the earlier portion.'}

        facts = _request_facts([{'role': 'user', 'content': [REMINDER, summary]}])

        assert [(fact.rule_name, fact.segment_kind) for fact in facts.boundary_facts] == [
            ('claude_code_compaction_resume', 'compaction_resume')
        ]


class TestMessageIdentity:
    def test_resent_number_spelling_matches_the_sampled_call(self) -> None:
        sampled = {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_tool_call('call-1', 'Bash', {'command': 'sleep 1', 'timeout': 5000.0})],
        }
        resent = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_tool_call('call-1', 'Bash', {'command': 'sleep 1', 'timeout': 5000})],
        }
        request = RequestEvent(({'role': 'user', 'content': 'go'},), None, None)

        sampled_identity = CLAUDE_CODE_POLICY.inspect_event(ResponseEvent(request, sampled)).message_identities[0]
        resent_identity = CLAUDE_CODE_POLICY.inspect_event(RequestEvent((resent,), None, None)).message_identities[0]

        assert sampled_identity.value == resent_identity.value
