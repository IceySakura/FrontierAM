from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from nexus.extensions.projects.hy_live.user_simulator import (
    CerebellumResponse,
    ScriptedUserSimulatorConfig,
    UserMessage,
    UserSimulator,
    UserVisibleConversation,
    build_user_simulator,
    normalize_scripted_turn,
    parse_scripted_turn,
)


AUDIO_PART = {'type': 'audio', 'audio': {'patch': '/data/audios/turn2.wav', 'audio_length': 16000}}


class FakeContext:
    def __init__(self, clients=None) -> None:
        self.clients = clients or {}

    def get_openai_client(self, name):
        return self.clients[name]


async def _converse(
    simulator: UserSimulator,
    cerebellum_turn: Callable[[UserMessage], Awaitable[CerebellumResponse]],
    *,
    max_turns: int,
) -> tuple[list[tuple[UserMessage, CerebellumResponse]], bool]:
    """Alternate user and cerebellum turns; tell whether the user ended the conversation."""
    turns: list[tuple[UserMessage, CerebellumResponse]] = []
    visible = UserVisibleConversation()
    for _ in range(max_turns):
        user_message = await simulator.next_user_message(visible)
        if user_message is None:
            return turns, True
        visible.record_user(user_message)
        response = await cerebellum_turn(user_message)
        visible.record_cerebellum(response)
        turns.append((user_message, response))
    return turns, False


async def test_scripted_user_requests_a_then_observes_response_before_requesting_b() -> None:
    simulator = build_user_simulator(
        {
            'type': 'scripted',
            'turns': [
                '先开始任务 A，收到后先告诉我。',
                '现在再开始任务 B。',
            ],
        },
        FakeContext(),
    )
    received: list[str] = []

    async def cerebellum_turn(user_message: str) -> CerebellumResponse:
        received.append(user_message)
        if '任务 A' in user_message:
            return CerebellumResponse(content='任务 A 已受理。', metadata={'task_id': 'task-a'})
        return CerebellumResponse(content='任务 B 已受理。', metadata={'task_id': 'task-b'})

    turns, ended_by_user = await _converse(simulator, cerebellum_turn, max_turns=4)

    assert received == ['先开始任务 A，收到后先告诉我。', '现在再开始任务 B。']
    assert [response.content for _, response in turns] == ['任务 A 已受理。', '任务 B 已受理。']
    assert ended_by_user is True


async def test_scripted_user_can_delay_followup_turns() -> None:
    import time

    simulator = build_user_simulator(
        {
            'type': 'scripted',
            'turns': ['先问', '再问'],
            'turn_delay_s': 0.05,
        },
        FakeContext(),
    )
    conversation = UserVisibleConversation()
    started = time.monotonic()
    first = await simulator.next_user_message(conversation)
    after_first = time.monotonic()
    conversation.record_user(first)
    conversation.record_cerebellum(CerebellumResponse(content='收到'))
    second = await simulator.next_user_message(conversation)
    elapsed_second = time.monotonic() - after_first

    assert first == '先问'
    assert second == '再问'
    assert after_first - started < 0.05
    assert elapsed_second >= 0.05


async def test_simulator_reads_every_cerebellum_utterance_in_order() -> None:
    """A real user hears each reply, so none may be lost when several arrive back to back.

    The interactive loop used to hand over only the latest observation, so a second delivery
    landing before the user spoke overwrote the first and never reached the simulator at all.
    """
    conversation = UserVisibleConversation()
    conversation.record_user('请并行跑 A 和 B。')
    conversation.record_cerebellum(CerebellumResponse(content='A 已受理。'))
    conversation.record_cerebellum(CerebellumResponse(content='B 已受理。'))

    assert conversation.as_payload() == [
        {'role': 'user', 'content': '请并行跑 A 和 B。'},
        {'role': 'assistant', 'content': 'A 已受理。'},
        {'role': 'assistant', 'content': 'B 已受理。'},
    ]


def test_visible_conversation_drops_turns_the_user_could_not_perceive() -> None:
    conversation = UserVisibleConversation()
    # A turn that only called tools produces no speech, so the user hears nothing.
    conversation.record_cerebellum(CerebellumResponse(content='', metadata={'tool_calls': ['delegate']}))
    conversation.record_cerebellum(CerebellumResponse(content='   '))
    conversation.record_cerebellum(CerebellumResponse(content='收到。', metadata={'task_ids': ['task-a']}))

    assert conversation.as_payload() == [{'role': 'assistant', 'content': '收到。'}]


def test_visible_conversation_payload_is_an_independent_snapshot() -> None:
    conversation = UserVisibleConversation()
    conversation.record_user('开始。')
    response = CerebellumResponse(content='收到。', metadata={'internal': 'hidden'})
    conversation.record_cerebellum(response)
    snapshot = conversation.as_payload()
    snapshot[0]['content'] = 'changed'
    snapshot.clear()
    response.content = 'changed'

    assert conversation.as_payload() == [
        {'role': 'user', 'content': '开始。'},
        {'role': 'assistant', 'content': '收到。'},
    ], 'Payload and response mutations must not change recorded speech or expose metadata'


def test_audio_turn_becomes_a_single_content_part_list() -> None:
    assert normalize_scripted_turn(AUDIO_PART) == [AUDIO_PART]


def test_text_turn_stays_a_bare_string() -> None:
    assert normalize_scripted_turn('先做任务 A。') == '先做任务 A。'


def test_content_part_without_supported_type_raises() -> None:
    with pytest.raises(ValueError, match='Unsupported scripted turn content part type'):
        normalize_scripted_turn({'type': 'image', 'image': 'x'})


def test_audio_content_part_without_payload_raises() -> None:
    with pytest.raises(ValueError, match="must carry a 'audio' payload"):
        normalize_scripted_turn({'type': 'audio'})


async def test_scripted_audio_script_yields_content_parts_then_ends() -> None:
    simulator = build_user_simulator(
        {'type': 'scripted', 'turns': [AUDIO_PART, '再帮我改一下。']},
        FakeContext(),
    )

    conversation = UserVisibleConversation()
    first = await simulator.next_user_message(conversation)
    conversation.record_user(first)
    conversation.record_cerebellum(CerebellumResponse(content='已受理'))
    second = await simulator.next_user_message(conversation)
    conversation.record_user(second)
    conversation.record_cerebellum(CerebellumResponse(content='改好了'))
    third = await simulator.next_user_message(conversation)

    assert first == [AUDIO_PART]
    assert second == '再帮我改一下。'
    assert third is None


async def test_scripted_audio_turn_is_not_shared_with_the_config() -> None:
    config_turn = dict(AUDIO_PART)
    simulator = build_user_simulator({'type': 'scripted', 'turns': [config_turn]}, FakeContext())

    message = await simulator.next_user_message(UserVisibleConversation())
    message[0]['type'] = 'text'

    assert config_turn['type'] == 'audio'


async def test_empty_scripted_script_is_rejected() -> None:
    with pytest.raises(ValueError, match='at least 1 item'):
        build_user_simulator({'type': 'scripted', 'turns': []}, FakeContext())


def test_user_simulator_config_rejects_unknown_type_and_extra_fields() -> None:
    with pytest.raises(ValueError, match='Input tag'):
        build_user_simulator({'type': 'unknown'}, FakeContext())
    with pytest.raises(ValueError, match='Extra inputs'):
        build_user_simulator(
            {'type': 'scripted', 'turns': ['hello'], 'api_key': 'must-not-be-accepted'},
            FakeContext(),
        )


def test_scripted_config_accepts_the_gated_turn_shape() -> None:
    config = ScriptedUserSimulatorConfig.model_validate({
        'type': 'scripted',
        'turns': [{'content': {'type': 'audio', 'audio': {'patch': '/a.wav'}}, 'gate': 'after_delivery'}],
    })

    assert parse_scripted_turn(config.turns[0]).gate == 'after_delivery'
