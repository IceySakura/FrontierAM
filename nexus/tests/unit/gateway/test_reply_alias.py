from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from nexus.gateway.reply_alias import (
    LEDGER_VIEW_MAX_CHARS,
    REPLY_ALIAS_CONTENT_SIMILARITY_ENV,
    REPLY_ALIAS_ENV,
    ReplyAlias,
    content_similarity,
)
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import PromptRender


GENERATION_PROMPT = '<assistant>'
TASK = {'role': 'user', 'content': 'list the workspace'}
SUB_META = {'declared_chain': {'chain_id': 'sub', 'kind': 'worker'}}


def _call(call_id: str, name: str, **arguments) -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _assistant(*calls: dict, content: str = '') -> dict:
    message = {'role': 'assistant', 'content': content}
    if calls:
        message['tool_calls'] = list(calls)
    return message


def _tool(call_id: str, content: str) -> dict:
    return {'role': 'tool', 'tool_call_id': call_id, 'content': content}


def _span(message: dict) -> str:
    calls = ''.join(
        f'[{call["id"]}:{call["function"]["name"]}({call["function"]["arguments"]})]'
        for call in message.get('tool_calls') or []
    )
    return f'<{message["role"]}>{message.get("content") or ""}{calls}'


def _encode(text: str) -> np.ndarray:
    return np.array(list(text.encode()), dtype=np.int32)


def _record(assembler: TrajectoryAssembler, messages: list[dict], output: dict, request_meta=None):
    """Record one token-level completion the way the session does: select the frozen prefix, splice, record."""
    context = assembler.extract_context(messages, request_meta=request_meta)
    frozen = assembler.frozen_prefix(messages, context=context, request_meta=request_meta)
    count = frozen.count if frozen is not None else 0
    spans = list(frozen.templated_messages) if frozen is not None else []
    span_ids = [np.asarray(ids) for ids in frozen.message_token_ids] if frozen is not None else []
    for message in messages[count:]:
        spans.append(_span(message))
        span_ids.append(_encode(spans[-1]))
    completion = _span(output)[len(GENERATION_PROMPT) :]
    prompt_ids = np.concatenate([*span_ids, _encode(GENERATION_PROMPT)])
    render = PromptRender(
        text=''.join(spans) + GENERATION_PROMPT,
        token_ids=prompt_ids,
        templated_messages=spans,
        message_token_ids=span_ids,
        generation_prompt=GENERATION_PROMPT,
        generation_prompt_ids=_encode(GENERATION_PROMPT),
    )
    assembler.record_completion(
        messages=messages,
        output_message=output,
        token_training={
            'prompt_ids': prompt_ids,
            'raw_prompt_ids': prompt_ids,
            'response_ids': list(completion.encode()),
        },
        completion_text=completion,
        prompt_render=render,
        context=context,
        request_meta=request_meta,
        frozen_prefix=frozen,
    )
    return frozen


def _llm_node(assembler: TrajectoryAssembler, message: dict) -> int:
    return next(
        node_id
        for node_id, node in assembler.trajectory.nodes.items()
        if node.generated_by == 'llm' and node.original_message.get('tool_calls') == message.get('tool_calls')
    )


def _siblings_like(assembler: TrajectoryAssembler, node_id: int) -> list[int]:
    """Children of the node's parent with the node's role, the node included."""
    nodes = assembler.trajectory.nodes
    parent = nodes[node_id].parent
    siblings = assembler.trajectory.roots if parent is None else nodes[parent].children
    return [child for child in siblings if nodes[child].original_message.get('role') == 'assistant']


SAMPLED = _assistant(_call('call_1', 'Bash', command='cd /workspace && ls'))
REWRITTEN = _assistant(_call('call_1', 'Bash', command='ls'))
RESULT = _tool('call_1', 'a.py')
FINAL = _assistant(content='done')
FOLLOWUP = {'role': 'user', 'content': 'and then?'}


@pytest.fixture
def assembler(monkeypatch) -> TrajectoryAssembler:
    return _assembler(monkeypatch)


def _assembler(monkeypatch, **env: str) -> TrajectoryAssembler:
    """An assembler built under exactly ``env`` for the two reply alias variables."""
    for name in (REPLY_ALIAS_ENV, REPLY_ALIAS_CONTENT_SIMILARITY_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return TrajectoryAssembler(scaffold_policy='nexus_main_agent')


class TestToolCallAlias:
    def test_a_rewritten_first_resend_keeps_the_sampled_bytes(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        frozen = _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)

        sampled_id = _llm_node(assembler, SAMPLED)
        assert frozen is not None
        assert frozen.node_ids == (0, sampled_id)
        assert frozen.templated_messages[1] == _span(SAMPLED)
        assert [candidate.rejection for candidate in frozen.alias_candidates] == [None]
        assert _siblings_like(assembler, sampled_id) == [sampled_id]
        assert assembler.trajectory.nodes[sampled_id].children
        assert assembler.trajectory.metrics == {'reply_aliased': 1}

    def test_later_resends_matching_the_alias_reuse_the_path(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        frozen = _record(assembler, [TASK, REWRITTEN, RESULT, FINAL, {'role': 'user', 'content': 'again'}], FINAL)

        assert frozen is not None
        assert frozen.count == 4
        assert frozen.alias_candidates == ()
        assert assembler.trajectory.metrics == {'reply_aliased': 1}

    def test_a_resend_that_changes_again_forks(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        changed = _assistant(_call('call_1', 'Bash', command='ls -la'))
        frozen = _record(assembler, [TASK, changed, RESULT], FINAL)

        sampled_id = _llm_node(assembler, SAMPLED)
        assert frozen is not None
        assert frozen.count == 1
        assert frozen.alias_candidates == ()
        assert len(_siblings_like(assembler, sampled_id)) == 2

    def test_reordered_calls_with_the_same_ids_are_accepted(self, assembler) -> None:
        first, second = _call('call_1', 'Read', path='a'), _call('call_2', 'Read', path='b')
        _record(assembler, [TASK], _assistant(first, second))
        frozen = _record(
            assembler, [TASK, _assistant(second, first), _tool('call_1', 'x'), _tool('call_2', 'y')], FINAL
        )

        assert frozen is not None
        assert frozen.count == 2

    @pytest.mark.parametrize(
        ('resent', 'reason'),
        [
            (
                _assistant(_call('call_1', 'Bash', command='ls'), _call('call_9', 'Bash', command='pwd')),
                'tool_call_count_mismatch',
            ),
            (
                _assistant(_call('call_1', 'Bash', command='ls'), content='I will look first'),
                'content_below_similarity',
            ),
            (_assistant(_call('call_1', 'Shell', command='ls')), 'tool_name_mismatch'),
        ],
    )
    def test_a_rejected_first_resend_forks_and_is_counted(self, assembler, resent, reason) -> None:
        _record(assembler, [TASK], SAMPLED)
        frozen = _record(assembler, [TASK, resent, RESULT], FINAL)

        sampled_id = _llm_node(assembler, SAMPLED)
        assert frozen is not None
        assert frozen.count == 1
        assert [candidate.rejection for candidate in frozen.alias_candidates] == [reason]
        assert len(_siblings_like(assembler, sampled_id)) == 2
        assert assembler.trajectory.metrics == {f'reply_alias_rejected_{reason}': 1}

    def test_a_rejected_reply_is_not_aliased_by_a_later_resend(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, _assistant(_call('call_1', 'Shell', command='ls')), RESULT], FINAL)
        frozen = _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)

        assert frozen is not None
        assert frozen.alias_candidates == ()
        assert 'reply_aliased' not in assembler.trajectory.metrics

    def test_a_partially_shared_call_set_is_an_id_mismatch(self, assembler) -> None:
        sampled = _assistant(_call('call_1', 'Read', path='a'), _call('call_2', 'Read', path='b'))
        resent = _assistant(_call('call_1', 'Read', path='a'), _call('call_3', 'Read', path='b'))
        _record(assembler, [TASK], sampled)
        frozen = _record(assembler, [TASK, resent, _tool('call_1', 'x'), _tool('call_3', 'y')], FINAL)

        assert frozen is not None
        assert [candidate.rejection for candidate in frozen.alias_candidates] == ['tool_call_id_mismatch']

    def test_regenerated_call_ids_are_an_id_mismatch(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        frozen = _record(
            assembler, [TASK, _assistant(_call('call_7', 'Bash', command='ls')), _tool('call_7', 'x')], FINAL
        )

        assert frozen is not None
        assert [candidate.rejection for candidate in frozen.alias_candidates] == ['tool_call_id_mismatch']

    def test_a_tool_call_added_to_a_text_reply_is_a_count_mismatch(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(content='checking'))
        frozen = _record(assembler, [TASK, _assistant(_call('call_1', 'Bash'), content='checking'), RESULT], FINAL)

        assert frozen is not None
        assert [candidate.rejection for candidate in frozen.alias_candidates] == ['tool_call_count_mismatch']

    def test_a_non_assistant_message_is_not_a_resend(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        frozen = _record(assembler, [TASK, {'role': 'user', 'content': 'summarize the conversation'}], FINAL)

        assert frozen is not None
        assert frozen.alias_candidates == ()
        assert assembler.trajectory.metrics == {}

    def test_a_retry_after_a_recorded_resend_reuses_the_alias(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        history = [TASK, REWRITTEN, RESULT]
        _record(assembler, history, FINAL)
        frozen = _record(assembler, history, _assistant(content='done again'))

        assert frozen is not None
        assert frozen.count == 3
        assert assembler.trajectory.metrics == {'reply_aliased': 1}


class TestTextAlias:
    def test_whitespace_only_changes_keep_the_sampled_bytes(self, assembler) -> None:
        sampled = _assistant(content='The workspace has a.py.\n')
        _record(assembler, [TASK], sampled)
        frozen = _record(assembler, [TASK, _assistant(content='The workspace has a.py.'), FOLLOWUP], FINAL)

        assert frozen is not None
        assert frozen.count == 2
        assert frozen.templated_messages[1] == _span(sampled)
        assert assembler.trajectory.metrics == {'reply_aliased': 1}

    def test_a_changed_text_forks_and_counts_once(self, assembler, caplog) -> None:
        _record(assembler, [TASK], _assistant(content='first try'))
        _record(assembler, [TASK], _assistant(content='second try'))
        with caplog.at_level(logging.INFO, logger='nexus.gateway.trajectory_assembler'):
            frozen = _record(assembler, [TASK, _assistant(content='something  else\n'), FOLLOWUP], FINAL)

        assert frozen is not None
        assert frozen.count == 1
        assert [candidate.rejection for candidate in frozen.alias_candidates] == ['content_below_similarity'] * 2
        assert assembler.trajectory.metrics == {'reply_alias_rejected_content_below_similarity': 1}
        (line,) = [record.getMessage() for record in caplog.records if 'gateway reply alias' in record.getMessage()]
        assert 'node 1 content_below_similarity, message 1 resent "content \'something else\'"' in line
        assert 'where the sample has "content \'first try\'"' in line
        assert 'node 2 content_below_similarity, message 1 resent "content \'something else\'"' in line
        assert 'where the sample has "content \'second try\'"' in line

    def test_the_resent_sample_is_found_among_retried_siblings(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(content='alpha'))
        _record(assembler, [TASK], _assistant(content='beta'))
        frozen = _record(assembler, [TASK, _assistant(content='alpha '), FOLLOWUP], FINAL)

        assert frozen is not None
        assert assembler.trajectory.nodes[frozen.node_ids[1]].original_message['content'] == 'alpha'

    def test_identical_retried_siblings_resolve_to_the_newest(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(content='same'))
        _record(assembler, [TASK], _assistant(content='same'))
        frozen = _record(assembler, [TASK, _assistant(content='same '), FOLLOWUP], FINAL)

        replies = [node_id for node_id, node in assembler.trajectory.nodes.items() if node.parent == 0]
        assert frozen is not None
        assert frozen.node_ids[1] == max(replies[:2])
        assert assembler.trajectory.metrics == {'reply_aliased': 1}

    def test_a_rejected_text_resend_blocks_later_resends(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(content='the answer is 4'))
        _record(assembler, [TASK, _assistant(content='the answer is 5'), FOLLOWUP], FINAL)
        frozen = _record(assembler, [TASK, _assistant(content='the answer is 4 '), FOLLOWUP], FINAL)

        assert frozen is not None
        assert frozen.alias_candidates == ()
        assert 'reply_aliased' not in assembler.trajectory.metrics

    def test_an_older_client_assistant_sibling_does_not_block(self, assembler) -> None:
        _record(assembler, [TASK, _assistant(content='prefilled'), FOLLOWUP], FINAL)
        _record(assembler, [TASK], _assistant(content='sampled'))
        frozen = _record(assembler, [TASK, _assistant(content='sampled '), FOLLOWUP], FINAL)

        assert frozen is not None
        assert frozen.count == 2
        assert assembler.trajectory.metrics == {'reply_aliased': 1}


class TestAliasRecording:
    def test_selection_has_no_side_effects(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        sampled_id = _llm_node(assembler, SAMPLED)
        for _ in range(2):
            frozen = assembler.frozen_prefix([TASK, REWRITTEN, RESULT])
            assert frozen is not None
            assert frozen.count == 2

        assert assembler._runtime_state.facts.by_node[sampled_id].alias_identity is None
        assert assembler.trajectory.metrics == {}

    def test_the_first_recorded_alias_wins(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        other = _assistant(_call('call_1', 'Bash', command='ls .'))
        frozen_other = assembler.frozen_prefix([TASK, other, RESULT])
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        assert frozen_other is not None
        assembler._record_aliases(
            frozen_other.alias_candidates, assembler.extract_context([TASK]).context_key, [TASK, other, RESULT], [], 2
        )

        sampled_id = _llm_node(assembler, SAMPLED)
        rewritten_identity = assembler._input_identities(
            assembler._event_facts(assembler._request_event([TASK, REWRITTEN], None, None), None), 2
        )[1]
        assert assembler._runtime_state.facts.by_node[sampled_id].alias_identity == rewritten_identity
        assert assembler.trajectory.metrics == {'reply_aliased': 1}
        assert len(assembler.trajectory.extra['reply_aliases']) == 1

    def test_the_ledger_survives_freezing(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        sampled_id = _llm_node(assembler, SAMPLED)

        assert assembler.freeze(extra={'task': 't'}).extra == {
            'task': 't',
            'reply_aliases': [
                {
                    'request_sequence': 1,
                    'sampled_node_id': sampled_id,
                    'resent_node_id': sampled_id,
                    'rejection': None,
                    'content_similarity': 1.0,
                    'difference': (
                        """resent '…call_1 Bash{"command": "ls"}' """
                        """where the sample has '…call_1 Bash{"command": "cd /workspace && ls"}'"""
                    ),
                    'sampled': """content '' | call_1 Bash{"command": "cd /workspace && ls"}""",
                    'resent': """content '' | call_1 Bash{"command": "ls"}""",
                }
            ],
        }

    def test_ledger_views_are_capped(self, assembler) -> None:
        sampled = _assistant(_call('call_1', 'Write', content='x' * (LEDGER_VIEW_MAX_CHARS + 100)))
        resent = _assistant(_call('call_1', 'Write', content='y' * (LEDGER_VIEW_MAX_CHARS + 100)))
        _record(assembler, [TASK], sampled)
        _record(assembler, [TASK, resent, _tool('call_1', 'ok')], FINAL)

        (entry,) = assembler.trajectory.extra['reply_aliases']
        assert entry['rejection'] is None
        for view in (entry['sampled'], entry['resent']):
            assert len(view) == LEDGER_VIEW_MAX_CHARS
            assert view.endswith('…')

    def test_each_rejected_reply_points_at_the_fork(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(content='abcdefghij'))
        _record(assembler, [TASK], _assistant(content='klmnopqrst'))
        _record(assembler, [TASK, _assistant(content='abcdefghiX'), FOLLOWUP], FINAL)

        ledger = assembler.trajectory.extra['reply_aliases']
        (fork_id,) = {entry['resent_node_id'] for entry in ledger}
        assert assembler.trajectory.nodes[fork_id].original_message == _assistant(content='abcdefghiX')
        assert [(entry['rejection'], entry['content_similarity']) for entry in ledger] == [
            ('content_below_similarity', pytest.approx(0.9)),
            ('content_below_similarity', pytest.approx(0.0)),
        ]
        assert {entry['resent'] for entry in ledger} == {"content 'abcdefghiX'"}
        assert [entry['sampled'] for entry in ledger] == ["content 'abcdefghij'", "content 'klmnopqrst'"]

    def test_an_observation_finds_its_parent_through_the_alias(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        assembler.record_observation('late output', parent_messages=[TASK, REWRITTEN])

        observation_id = max(assembler.trajectory.nodes)
        assert assembler.trajectory.nodes[observation_id].parent == _llm_node(assembler, SAMPLED)


class TestAliasAcrossChains:
    def test_another_chain_resending_first_does_not_block_the_owner(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, {'role': 'user', 'content': 'sub task'}], FINAL, request_meta=SUB_META)
        frozen = _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)

        assert frozen is not None
        assert frozen.count == 2
        assert assembler.trajectory.metrics == {'reply_aliased': 1}

    def test_another_chain_never_registers_or_reuses_the_alias(self, assembler) -> None:
        _record(assembler, [TASK], SAMPLED)
        _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)
        frozen = _record(
            assembler, [TASK, REWRITTEN, {'role': 'user', 'content': 'sub task'}], FINAL, request_meta=SUB_META
        )

        sampled_id = _llm_node(assembler, SAMPLED)
        sub_reply = max(node_id for node_id, node in assembler.trajectory.nodes.items() if node.generated_by == 'llm')
        sub_path = assembler._node_path(sub_reply)
        assert frozen is None
        assert sampled_id not in sub_path
        assert assembler.trajectory.metrics == {'reply_aliased': 1}


class TestReplyAliasSwitch:
    def test_disabled_keeps_strict_matching(self, monkeypatch) -> None:
        assembler = _assembler(monkeypatch, **{REPLY_ALIAS_ENV: '0'})
        _record(assembler, [TASK], SAMPLED)
        frozen = _record(assembler, [TASK, REWRITTEN, RESULT], FINAL)

        assert frozen is not None
        assert frozen.count == 1
        assert frozen.alias_candidates == ()
        assert len(_siblings_like(assembler, _llm_node(assembler, SAMPLED))) == 2
        assert assembler.trajectory.metrics == {}
        assert assembler.trajectory.extra == {}

    def test_a_lower_similarity_threshold_accepts_a_small_text_change(self, monkeypatch) -> None:
        assembler = _assembler(monkeypatch, **{REPLY_ALIAS_CONTENT_SIMILARITY_ENV: '0.8'})
        _record(assembler, [TASK], _assistant(_call('call_1', 'Bash', command='ls'), content='Listing the files now.'))
        resent = _assistant(_call('call_1', 'Bash', command='ls'), content='Listing the files now')
        frozen = _record(assembler, [TASK, resent, RESULT], FINAL)

        assert frozen is not None
        assert frozen.count == 2

    def test_whitespace_only_text_changes_are_accepted_by_default(self, assembler) -> None:
        _record(assembler, [TASK], _assistant(_call('call_1', 'Bash', command='ls'), content='Listing  the files\n'))
        frozen = _record(
            assembler,
            [TASK, _assistant(_call('call_1', 'Bash', command='ls'), content='Listing the files'), RESULT],
            FINAL,
        )

        assert frozen is not None
        assert frozen.count == 2


class TestReplyAliasRule:
    @pytest.mark.parametrize('value', ['high', '-0.1', '1.5', 'nan'])
    def test_an_invalid_threshold_is_refused(self, monkeypatch, value) -> None:
        monkeypatch.setenv(REPLY_ALIAS_CONTENT_SIMILARITY_ENV, value)
        with pytest.raises(ValueError, match=REPLY_ALIAS_CONTENT_SIMILARITY_ENV):
            ReplyAlias.from_env()

    def test_arguments_are_not_compared(self) -> None:
        assert ReplyAlias().rejection(SAMPLED, REWRITTEN) is None

    def test_similarity_is_the_unchanged_share_of_the_longer_text(self) -> None:
        def similarity(sampled: object, resent: object) -> float:
            return content_similarity({'content': sampled}, {'content': resent})

        assert similarity(None, '') == 1.0
        assert similarity('abcdef', 'abXdef') == pytest.approx(5 / 6)
        assert similarity('abc', 'abcdef') == pytest.approx(3 / 6)
        assert similarity([{'type': 'text', 'text': 'a'}], 'a') == 0.0
