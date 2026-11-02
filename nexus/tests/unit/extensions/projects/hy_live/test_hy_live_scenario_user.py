from __future__ import annotations

import asyncio
import json

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.projects.hy_live.episode_clock import EpisodeClock
from nexus.extensions.projects.hy_live.simulation_scenario import ScenarioCard
from nexus.extensions.projects.hy_live.simulation_state import SimulatorDecisionError
from nexus.extensions.projects.hy_live.simulator import system_prompt
from nexus.extensions.projects.hy_live.user_simulator import (
    CerebellumResponse,
    LLMUserSimulator,
    LLMUserSimulatorConfig,
    UserVisibleConversation,
    build_user_simulator,
)


OPENING = '我是四年级班主任，帮我做个Excel周考勤统计表模板，用公式算出每人本周缺勤次数。'
CARD: dict[str, Any] = {
    'meta': {'script_id': 'final3-001', 'topic_id': 'TP-S-1', 'expected_turns': [7, 10]},
    'topic': '小学班主任要一个能自动算缺勤次数的Excel周考勤表模板',
    'user': {'background': '高燕在大理一所小学带四年级。', 'style': 'A', 'habits': ['一句话里常带两件事']},
    'goal': '要一个Excel学生考勤统计表模板，表里用公式自动算出每个学生这周缺勤几次。',
    'required_nodes': [
        {'type': 'D1', 'when': '开场', 'action': '说自己是四年级班主任，请它做一个Excel周考勤统计表模板。'},
        {'type': 'Q2', 'when': '做完后：听到表格做好了', 'action': '问缺勤次数按什么口径算。'},
    ],
    'ending': {'normal': '听到表格已经做好，并说明了缺勤口径。'},
}
MATERIAL = '学生名单：张三、李四、王五。'


class FakeContext:
    run_id = 'run-1'

    def __init__(self, client: FakeUserModel) -> None:
        self.client = client

    def get_openai_client(self, name: str) -> FakeUserModel:
        assert name == 'user_simulator'
        return self.client


class FakeUserModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def payload(self, index: int) -> dict[str, Any]:
        return json.loads(self.calls[index]['messages'][1]['content'])


def _decision(**fields: Any) -> str:
    decision = {
        'action': 'speak',
        'message': '',
        'event': 'none',
        'required_node_index': None,
        'node_complete': False,
        'skipped_required_node_index': None,
        'optional_insertion_index': None,
        'material_chunk_id': None,
        'progress_query': False,
        'evidence_message_indices': [],
        'ending': 'none',
        'assessment': '助手已回复。',
        'reason': '按剧本推进。',
    }
    return json.dumps({**decision, **fields}, ensure_ascii=False)


def _simulator(responses: list[str], **config: Any) -> tuple[LLMUserSimulator, FakeUserModel]:
    model = FakeUserModel(responses)
    simulator = build_user_simulator({'type': 'llm', 'scenario': CARD, **config}, FakeContext(model))
    assert isinstance(simulator, LLMUserSimulator)
    return simulator, model


async def _deliver_opening(simulator: LLMUserSimulator, conversation: UserVisibleConversation) -> None:
    opening = await simulator.next_user_message(conversation)
    assert simulator.is_message_current(opening, conversation)
    conversation.record_user(opening)


def test_scenario_template_renders_like_the_data_pipeline_renderer() -> None:
    config = LLMUserSimulatorConfig.model_validate({'type': 'llm', 'scenario': CARD})

    assert config.resolved_system_prompt == system_prompt(CARD), (
        'The validated card must render byte-for-byte like the raw card the data pipeline rendered'
    )


async def test_fixed_opening_is_delivered_as_the_opening_node_without_a_model_call() -> None:
    simulator, model = _simulator([], initial_prompt=OPENING)
    conversation = UserVisibleConversation()

    opening = await simulator.next_user_message(conversation)

    assert opening == OPENING
    assert model.calls == [], 'A fixed opening must not spend a user-model call'
    assert simulator.runtime.state()['next_required_node_index'] == 0, 'Nothing advances before delivery'
    conversation.record_user(opening)
    state = simulator.runtime.state()
    assert state['next_required_node_index'] == 1
    assert state['completed_required_node_indices'] == [0]
    assert state['delivered_user_turns'] == 1


async def test_style_a_opening_that_owes_materials_stays_open_until_they_are_given() -> None:
    card = {**CARD, 'context': {'materials': [MATERIAL]}}
    model = FakeUserModel([
        _decision(
            message='名单给你。',
            required_node_index=0,
            node_complete=True,
            material_chunk_id='m0:c0',
            evidence_message_indices=[1],
        )
    ])
    simulator = build_user_simulator({'type': 'llm', 'scenario': card, 'initial_prompt': OPENING}, FakeContext(model))
    conversation = UserVisibleConversation()

    await _deliver_opening(simulator, conversation)
    state = simulator.runtime.state()
    assert state['next_required_node_index'] == 0, 'The opening owes its material, so it must stay the current node'
    assert state['current_node_started']
    conversation.record_cerebellum(CerebellumResponse(content='好的，学生名单方便给我吗？'))
    message = await simulator.next_user_message(conversation)

    assert message == f'名单给你。 {MATERIAL}', 'The program appends the material text to the short lead-in'
    conversation.record_user(message)
    state = simulator.runtime.state()
    assert state['next_required_node_index'] == 1
    assert state['delivered_material_chunk_ids'] == ['m0:c0']


async def test_only_the_decision_message_reaches_the_cerebellum_and_commits_on_delivery() -> None:
    question = '缺勤次数按什么口径算？病假算不算？'
    simulator, model = _simulator(
        [_decision(message=question, required_node_index=1, node_complete=True, evidence_message_indices=[1])],
        initial_prompt=OPENING,
    )
    conversation = UserVisibleConversation()
    await _deliver_opening(simulator, conversation)
    conversation.record_cerebellum(CerebellumResponse(content='考勤表做好了，最后一列自动算缺勤次数。'))

    message = await simulator.next_user_message(conversation)

    assert message == question
    payload = model.payload(0)
    assert [item['index'] for item in payload['conversation']] == [0, 1]
    assert payload['evidence_reference']['latest_assistant_index'] == 1
    assert payload['state']['next_required_node_index'] == 1
    assert model.calls[0]['request_meta']['invocation_kind'] == 'hy_live_user_simulator'
    assert simulator.runtime.state()['next_required_node_index'] == 1, 'A proposal must not advance the node'
    conversation.record_user(message)
    assert simulator.runtime.state()['next_required_node_index'] is None


async def test_invalid_decision_is_repaired_with_the_validation_error() -> None:
    bad = _decision(message='做好了吗？', evidence_message_indices=[7])
    simulator, model = _simulator([bad, _decision(message='好的，等你。', evidence_message_indices=[1])])
    conversation = UserVisibleConversation()
    conversation.record_user(OPENING)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))

    assert await simulator.next_user_message(conversation) == '好的，等你。'

    repair = model.payload(1)
    assert 'nonexistent message indices' in repair['validation_error']
    assert repair['invalid_decision_response'] == bad
    assert simulator.outcome()['rejected_decisions'] == 1


async def test_exhausted_repairs_fail_the_simulation_instead_of_falling_back() -> None:
    simulator, _ = _simulator(['not json', 'still not json'], decision_retries=1)
    conversation = UserVisibleConversation()
    conversation.record_user(OPENING)
    conversation.record_cerebellum(CerebellumResponse(content='收到。'))

    with pytest.raises(SimulatorDecisionError):
        await simulator.next_user_message(conversation)

    assert simulator.outcome()['ending'] == 'simulation_failed'


async def test_done_ends_only_after_the_driver_accepts_it() -> None:
    simulator, _ = _simulator(
        [
            _decision(message='口径是什么？', required_node_index=1, node_complete=True, evidence_message_indices=[1]),
            _decision(action='done', message='', ending='normal', evidence_message_indices=[3]),
        ],
        initial_prompt=OPENING,
    )
    conversation = UserVisibleConversation()
    await _deliver_opening(simulator, conversation)
    conversation.record_cerebellum(CerebellumResponse(content='表格做好了。'))
    question = await simulator.next_user_message(conversation)
    conversation.record_user(question)
    conversation.record_cerebellum(CerebellumResponse(content='填“缺”记一次，病假事假也算。'))

    assert await simulator.next_user_message(conversation) is None
    assert simulator.outcome()['ending'] is None, 'The ending is recorded only when the driver accepts it'
    assert simulator.is_message_current(None, conversation)
    assert simulator.outcome()['ending'] == 'normal'


async def test_notice_before_delivery_discards_the_decision() -> None:
    simulator, _ = _simulator([_decision(message='对了，表头要写班级名。', evidence_message_indices=[1])])
    conversation = UserVisibleConversation()
    conversation.record_user(OPENING)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))
    message = await simulator.next_user_message(conversation)

    conversation.record_cerebellum(CerebellumResponse(content='考勤表做好了。'))

    assert not simulator.is_message_current(message, conversation)
    conversation.record_user(message)
    assert simulator.runtime.state()['delivered_user_turns'] == 0, 'A discarded decision must not count as said'


async def test_wait_wakes_on_new_speech_and_reports_why() -> None:
    simulator, model = _simulator(
        [
            _decision(action='wait', message='', evidence_message_indices=[1]),
            _decision(message='好，我看看。', evidence_message_indices=[2]),
        ],
        decision_interval_s=60,
    )
    conversation = UserVisibleConversation()
    conversation.record_user(OPENING)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))
    pending = asyncio.create_task(simulator.next_user_message(conversation))
    try:
        while not model.calls:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
        assert not pending.done(), 'wait keeps the user silent'
        conversation.record_cerebellum(CerebellumResponse(content='做好了。'))
        assert await asyncio.wait_for(pending, timeout=1) == '好，我看看。'
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    assert model.payload(1)['timing']['wake_reason'] == 'new_speech'


TICK_S = 0.02


def _paused_trainer_clock() -> EpisodeClock:
    """A started clock whose trainer keeps the cerebellum parked whenever it is inside a call."""
    clock = EpisodeClock(tick_s=TICK_S)
    clock.bind(abort_event=asyncio.Event(), background_working=lambda: False)
    clock.start()
    return clock


async def _freeze(clock: EpisodeClock, seconds: float, *, after: float = 0.0) -> None:
    await asyncio.sleep(after)
    with clock.cerebellum_call():
        await asyncio.sleep(seconds)


def _clocked_simulator(responses: list[str], clock: EpisodeClock, **config: Any) -> tuple[LLMUserSimulator, Any]:
    model = FakeUserModel(responses)
    simulator = build_user_simulator(
        {'type': 'llm', 'scenario': CARD, 'initial_prompt': OPENING, **config}, FakeContext(model), clock=clock
    )
    assert isinstance(simulator, LLMUserSimulator)
    return simulator, model


async def test_frozen_stretch_is_not_time_the_user_waited() -> None:
    clock = _paused_trainer_clock()
    simulator, model = _clocked_simulator([_decision(message='好的，我等你。', evidence_message_indices=[1])], clock)
    conversation = UserVisibleConversation()
    await _deliver_opening(simulator, conversation)
    started = asyncio.get_running_loop().time()

    await _freeze(clock, 0.3)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))
    await simulator.next_user_message(conversation)
    await clock.stop()

    timing = model.payload(0)['timing']
    assert asyncio.get_running_loop().time() - started >= 0.3
    assert timing['since_last_user_message_s'] < 0.15, 'A trainer pause must not read as the assistant being slow'
    assert timing['elapsed_s'] < 0.15
    assert timing['remaining_s'] > simulator.config.max_elapsed_s - 0.15, 'The pause must not spend the user budget'
    assert simulator.outcome()['elapsed_s'] < 0.15


async def test_no_decision_starts_while_the_cerebellum_is_parked() -> None:
    clock = _paused_trainer_clock()
    simulator, model = _clocked_simulator([_decision(message='好的。', evidence_message_indices=[1])], clock)
    conversation = UserVisibleConversation()
    await _deliver_opening(simulator, conversation)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))
    resume = asyncio.Event()

    async def parked_cerebellum() -> None:
        with clock.cerebellum_call():
            await resume.wait()

    parked = asyncio.create_task(parked_cerebellum())
    await asyncio.sleep(TICK_S)
    pending = asyncio.create_task(simulator.next_user_message(conversation))
    try:
        await asyncio.sleep(0.2)
        assert model.calls == [], 'A decision taken while the cerebellum is parked would be stale on arrival'
        resume.set()
        assert await asyncio.wait_for(pending, timeout=1) == '好的。'
    finally:
        resume.set()
        await asyncio.gather(parked, return_exceptions=True)
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await clock.stop()
    assert 'decision_deferred' in [event['event'] for event in simulator.decision_events]


async def test_silent_decision_tick_is_pushed_out_by_a_frozen_stretch() -> None:
    clock = _paused_trainer_clock()
    simulator, model = _clocked_simulator(
        [
            _decision(action='wait', message='', evidence_message_indices=[1]),
            _decision(message='做好了告诉我。', evidence_message_indices=[1]),
        ],
        clock,
        decision_interval_s=0.3,
    )
    conversation = UserVisibleConversation()
    await _deliver_opening(simulator, conversation)
    conversation.record_cerebellum(CerebellumResponse(content='收到，我去做。'))
    started = asyncio.get_running_loop().time()

    freeze = asyncio.create_task(_freeze(clock, 0.2, after=0.05))
    message = await asyncio.wait_for(simulator.next_user_message(conversation), timeout=2)
    woke_after = asyncio.get_running_loop().time() - started
    await freeze
    await clock.stop()

    assert message == '做好了告诉我。'
    assert woke_after >= 0.45, 'The 0.3 s tick must wait 0.3 s of episode time, not wall time'
    second = model.payload(1)['timing']
    assert second['wake_reason'] == 'timer_tick'
    assert second['waited_s'] < 0.4, 'The user is told it waited episode time, without the frozen 0.2 s'


@pytest.mark.parametrize(
    'material',
    [
        {'name': '学生名单', 'via': '文件'},
        {'name': '学生名单', 'via': '文件', 'path': 'roster.txt'},
        {'name': '学生名单', 'via': '口述', 'content': '  '},
    ],
)
def test_card_rejects_material_without_text_or_url(material: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='content|path'):
        ScenarioCard.model_validate({**CARD, 'context': {'materials': [material]}})


def test_card_accepts_file_material_that_carries_its_text() -> None:
    card = ScenarioCard.model_validate({
        **CARD,
        'context': {'materials': [{'name': '学生名单', 'via': '文件', 'content': MATERIAL}]},
    })

    assert card.context.materials[0].content == MATERIAL


def test_card_rejects_an_opening_node_out_of_place() -> None:
    card = {**CARD, 'required_nodes': list(reversed(CARD['required_nodes']))}

    with pytest.raises(ValueError, match='opening'):
        build_user_simulator({'type': 'llm', 'scenario': card}, FakeContext(FakeUserModel([])))
