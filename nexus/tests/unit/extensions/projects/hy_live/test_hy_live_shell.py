"""One hy-live episode on hylive: the simulated user, the core and a scripted sandbox, end to end."""

from __future__ import annotations

import asyncio
import json

from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hylive.midbrain_loop import CONCLUDE_SOURCE
from hylive.replay import config_from_recording, load, replay

from nexus.extensions.agents.installed_common.host_rpc import (
    ACP_PERMISSION_REQUEST_EVENT,
    ACP_PERMISSION_RESOLVE_EVENT,
)
from nexus.extensions.agents.installed_common.multiturn import normalize_stream_json_session_id
from nexus.extensions.projects.hy_live.coordination import backend as backend_module
from nexus.extensions.projects.hy_live.coordination.backend import WorkAgentBudget
from nexus.extensions.projects.hy_live.coordination.models import TaskResult
from nexus.extensions.projects.hy_live.episode_clock import EpisodeClock
from nexus.extensions.projects.hy_live.hylive_shell import (
    LiveEpisode,
    LiveShell,
    harness_agent_tasks,
    notice_owner_events,
)
from nexus.extensions.projects.hy_live.recording import EventLog
from nexus.extensions.projects.hy_live.user_simulator import (
    ScriptedUserSimulator,
    ScriptedUserSimulatorConfig,
    UserSimulator,
)


PERMISSION = {
    'toolCall': {'title': '写入 snake.html', 'kind': 'edit', 'rawInput': {'file_path': 'snake.html'}},
    'options': [{'optionId': 'allow-once', 'kind': 'allow_once'}, {'optionId': 'reject-once', 'kind': 'reject_once'}],
}
Reply = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]
Turn = Callable[['Sandbox', str, dict[str, Any]], Awaitable[TaskResult]]


def tool(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {'id': f'call_{name}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
        ],
    }


def say(text: str) -> dict[str, Any]:
    return {'role': 'assistant', 'content': text}


class ScriptedClient:
    """A model client answering each request from ``reply(messages, kwargs)``."""

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []

    async def create_completion(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = self.reply(kwargs['messages'], kwargs)
        if isinstance(message, BaseException):
            raise message
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def cerebellum(delegate: dict[str, str] | None = None) -> ScriptedClient:
    """Hands the user's lines in ``delegate`` off, answers other lines and says what each notice tells."""
    delegate = delegate or {}

    def reply(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> dict[str, Any]:
        last = messages[-1]
        tool_choice = str(getattr(kwargs['tool_choice'], 'value', kwargs['tool_choice']))
        if tool_choice == 'auto' and last['role'] == 'user' and last['content'] in delegate:
            return tool(delegate[last['content']], text=last['content'])
        if last['role'] == 'tool':
            return say('我这就去做。')
        if str(last.get('content', '')).startswith('[通知]'):
            return say(f'通知：{last["content"].splitlines()[1]}')
        return say(f'回复：{last["content"]}')

    return ScriptedClient(reply)


def midbrain(*, permission: str = 'allow') -> ScriptedClient:
    """Creates a Task for a delegate, answers a permission with ``permission`` and sums up finished work."""

    def reply(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> dict[str, Any]:
        last = messages[-1]
        if last['role'] == 'tool':
            return tool('reply', status='accepted', say='交出去了')
        content = str(last['content'])
        if content.startswith('工作结束'):
            return tool('reply', status='ok', say='工作做完了')
        snapshot = json.loads(content.split('当前账本快照：\n', 1)[1])
        if '调用了 answer_permission' in content and snapshot['pending_permissions']:
            return tool('answer_permission', permission_id=snapshot['pending_permissions'][0]['id'], decision=permission)
        return tool('TaskCreate', subject='贪吃蛇', description=content.rsplit('小脑的原文：', 1)[-1].split('\n')[0])

    return ScriptedClient(reply)


class Sandbox:
    """The sandbox runtime: Task directories, ACP requests the turns raise, and the primitives' invokes."""

    default_venv_id = 'venv'

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self.answered = asyncio.Event()
        self.requests = 0

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        self.emitted.append((event, data))
        self.answered.set()

    async def run_command(self, command: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(return_code=0, stdout='', stderr='', status='done')

    async def invoke(self, request: Any) -> Any:
        return {'success': True}

    async def ask_permission(self, config: dict[str, Any], params: dict[str, Any]) -> str:
        """Raise one ACP permission from the turn running with ``config``; returns the option the user picked."""
        self.requests += 1
        channel = normalize_stream_json_session_id(config['session_id'], namespace='codebuddy-acp-session')
        self.answered.clear()
        await self.handlers[ACP_PERMISSION_REQUEST_EVENT](
            {'channel_id': channel, 'acp_request_id': self.requests, 'permission_token': 't', 'params': params}
        )
        while not any(
            event == ACP_PERMISSION_RESOLVE_EVENT and data['acp_request_id'] == self.requests
            for event, data in self.emitted
        ):
            await self.answered.wait()
            self.answered.clear()
        return next(
            data['option_id']
            for event, data in self.emitted
            if event == ACP_PERMISSION_RESOLVE_EVENT and data['acp_request_id'] == self.requests
        )

    def permission_answers(self) -> list[str]:
        return [data['option_id'] for event, data in self.emitted if event == ACP_PERMISSION_RESOLVE_EVENT]


async def done_at_once(sandbox: Sandbox, prompt: str, config: dict[str, Any]) -> TaskResult:
    return TaskResult(final_output=f'做好了：{prompt}')


async def never_ends(sandbox: Sandbox, prompt: str, config: dict[str, Any]) -> TaskResult:
    await asyncio.sleep(3600)
    raise AssertionError('the episode ends first')


def asks_permission_then_hangs(answers: list[str]) -> Turn:
    async def turn(sandbox: Sandbox, prompt: str, config: dict[str, Any]) -> TaskResult:
        answers.append(await sandbox.ask_permission(config, PERMISSION))
        await asyncio.sleep(3600)
        raise AssertionError('the episode ends first')

    return turn


def asks_permission_then_ends(answers: list[str]) -> Turn:
    async def turn(sandbox: Sandbox, prompt: str, config: dict[str, Any]) -> TaskResult:
        answers.append(await sandbox.ask_permission(config, PERMISSION))
        return TaskResult(final_output='用户答复后停下')

    return turn


class World:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        cerebellum_client: ScriptedClient | None = None,
        midbrain_client: ScriptedClient | None = None,
        turn: Turn = done_at_once,
        event_log: EventLog | None = None,
        clock: EpisodeClock | None = None,
        summarize_outcomes: bool = False,
    ) -> None:
        self.sandbox = Sandbox()
        self.cerebellum = cerebellum_client or cerebellum()
        self.midbrain = midbrain_client or midbrain()

        async def get_runtime() -> Sandbox:
            return self.sandbox

        async def closed(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {'success': True}

        async def preload(_executor: Any) -> None:
            return None

        monkeypatch.setattr(backend_module, 'call_runtime_agent', closed)
        monkeypatch.setattr(backend_module, 'unbind_llm_channel', lambda *_: None)
        monkeypatch.setattr(backend_module.AcpSessionExecutor, '_install_proxy_preload', preload)
        self.shell = LiveShell(
            ctx=cast(Any, SimpleNamespace(run_id='run-1', get_runtime=get_runtime, openai_clients={})),
            owner_id='owner',
            clients={'cerebellum': self.cerebellum, 'midbrain': self.midbrain},
            cerebellum_system='你是助手。',
            trusted_config={'acp_cli': 'workbuddy'},
            budget=WorkAgentBudget(),
            search_backend=None,
            clock=clock or EpisodeClock(),
            event_log=event_log,
            summarize_outcomes=summarize_outcomes,
        )
        sandbox = self.sandbox

        async def invoke(prompt: str, config: dict[str, Any]) -> TaskResult:
            return await turn(sandbox, prompt, config)

        monkeypatch.setattr(self.shell.executor, '_invoke_codebuddy', invoke)

    async def run(
        self, simulator: UserSimulator, *, max_user_turns: int = 3, task_delivery_timeout: float = 5
    ) -> LiveEpisode:
        try:
            return await asyncio.wait_for(
                self.shell.run(
                    simulator=simulator, max_user_turns=max_user_turns, task_delivery_timeout=task_delivery_timeout
                ),
                timeout=10,
            )
        finally:
            await self.shell.close()

    def cerebellum_sources(self) -> list[str]:
        return [request['request_meta']['source'] for request in self.cerebellum.requests]


def scripted(*turns: Any) -> ScriptedUserSimulator:
    return ScriptedUserSimulator(ScriptedUserSimulatorConfig(turns=list(turns)))


class StaleOnceSimulator(UserSimulator):
    """Speaks its lines in order but reports one of them stale once, as a notice racing it would."""

    def __init__(self, lines: list[str | None], stale: str | None) -> None:
        self.lines = list(lines)
        self.stale = stale
        self.reported_stale = False

    async def next_user_message(self, conversation: Any) -> str | None:
        return self.lines.pop(0) if self.lines else None

    def is_message_current(self, message: Any, conversation: Any) -> bool:
        if message == self.stale and not self.reported_stale:
            self.reported_stale = True
            return False
        return True


async def test_a_chat_episode_answers_the_user_and_ends_with_them(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch)

    live = await world.run(scripted('你好'))

    episode = live.episode
    assert episode.conversation.ended_by_user
    assert [turn.cerebellum_response.content for turn in episode.conversation.turns] == ['回复：你好']
    assert episode.summary == '回复：你好'
    assert episode.episode_termination is None
    assert world.midbrain.requests == []


async def test_a_hand_off_turn_answers_the_user_with_its_receipt_line(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}))

    live = await world.run(scripted('写游戏'))

    assert world.cerebellum_sources()[:2] == ['turn', 'acknowledge'], 'The receipt line is asked right after the turn'
    assert live.episode.conversation.turns[0].cerebellum_response.content == '我这就去做。'
    assert [task.status for task in live.episode.tasks] == ['done']
    assert [response.metadata['notice_kinds'] for response in live.episode.delivery_responses] == [['completed']]
    assert [event.kind for event in live.episode.delivery_events] == ['task_delivery']


async def test_the_after_delivery_gate_waits_for_one_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}))

    live = await world.run(scripted('写游戏', {'content': '打开看看', 'gate': 'after_delivery'}))

    assert [turn.user_message for turn in live.episode.conversation.turns] == ['写游戏', '打开看看']
    assert world.cerebellum_sources() == ['turn', 'acknowledge', 'observe', 'turn']


async def test_the_after_delivery_gate_does_not_wait_when_no_notice_can_arrive(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch)

    live = await world.run(
        scripted('先问我想要什么类型', {'content': '贪吃蛇吧', 'gate': 'after_delivery'}), task_delivery_timeout=60
    )

    assert [turn.user_message for turn in live.episode.conversation.turns] == ['先问我想要什么类型', '贪吃蛇吧']
    assert live.episode.episode_termination is None


async def test_a_stale_user_message_never_reaches_the_cerebellum(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch)

    live = await world.run(StaleOnceSimulator(['写游戏', '做到哪了？', '打开看看'], stale='做到哪了？'), max_user_turns=5)

    assert [turn.user_message for turn in live.episode.conversation.turns] == ['写游戏', '打开看看']
    sent = [request['messages'][-1]['content'] for request in world.cerebellum.requests]
    assert '做到哪了？' not in sent


async def test_a_stale_ending_keeps_the_user_in_the_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch)

    live = await world.run(StaleOnceSimulator(['写游戏', None, '谢谢'], stale=None), max_user_turns=5)

    assert [turn.user_message for turn in live.episode.conversation.turns] == ['写游戏', '谢谢']
    assert live.episode.conversation.ended_by_user


async def test_work_still_running_at_the_delivery_deadline_ends_as_task_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}), turn=never_ends)

    live = await world.run(scripted('写游戏'), task_delivery_timeout=0.3)

    assert live.episode.episode_termination == 'TASK_TIMEOUT'
    assert [task.status for task in live.episode.tasks] == ['running']


async def test_a_permission_left_open_by_a_finished_user_is_rejected_and_the_episode_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[str] = []
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}), turn=asks_permission_then_ends(answers))

    live = await world.run(scripted('写游戏'), task_delivery_timeout=1800)

    assert answers == ['reject-once'], 'Nobody is left to approve, so the permission is rejected as a user would'
    assert live.episode.episode_termination == 'USER_LEFT_EARLY'
    assert [task.status for task in live.episode.tasks] == ['done']
    assert [event.kind for event in live.episode.delivery_events] == ['permission_request', 'task_delivery']


async def test_a_permission_the_user_answers_goes_back_to_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[str] = []
    world = World(
        monkeypatch,
        cerebellum_client=cerebellum({'写游戏': 'delegate', '可以': 'answer_permission'}),
        turn=asks_permission_then_ends(answers),
    )

    live = await world.run(scripted('写游戏', {'content': '可以', 'gate': 'after_delivery'}))

    assert answers == ['allow-once']
    assert live.episode.episode_termination is None
    assert live.episode.delivery_responses[0].metadata['notice_kinds'] == ['permission']


async def test_a_permission_stays_open_while_the_user_can_still_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[str] = []
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}), turn=asks_permission_then_ends(answers))

    live = await world.run(
        scripted('写游戏', {'content': '等等', 'gate': 'after_delivery'}, {'content': '嗯', 'gate': 'after_delivery'}),
        task_delivery_timeout=0.5,
    )

    assert answers == [], 'A deadline hit before the user finished leaves the permission to the user'
    assert [request.status for request in live.state.harness.permissions.values()] == ['open']
    assert live.episode.episode_termination == 'TASK_TIMEOUT'


async def test_a_failed_midbrain_call_ends_the_episode_as_an_environment_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = ScriptedClient(lambda messages, kwargs: RuntimeError('midbrain model unavailable'))  # type: ignore[arg-type,return-value]
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}), midbrain_client=failing)

    live = await world.run(scripted('写游戏'))

    assert live.episode.episode_termination == 'ENV_ERROR'


async def test_tool_call_markup_ends_the_episode_as_a_format_error(monkeypatch: pytest.MonkeyPatch) -> None:
    markup = ScriptedClient(lambda messages, kwargs: say('<tool_call>delegate</tool_call>'))
    world = World(monkeypatch, cerebellum_client=markup)

    live = await world.run(scripted('写游戏'))

    assert live.episode.episode_termination == 'FORMAT_ERROR'
    assert live.episode.conversation.turns == []


async def test_a_recorded_episode_replays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'events.jsonl'
    log = EventLog(path)
    answers: list[str] = []
    world = World(
        monkeypatch,
        cerebellum_client=cerebellum({'写游戏': 'delegate', '可以': 'answer_permission'}),
        turn=asks_permission_then_ends(answers),
        event_log=log,
    )

    await world.run(scripted('写游戏', {'content': '可以', 'gate': 'after_delivery'}))
    log.close()

    lines = load(str(path))
    assert lines[0]['dir'] == 'config'
    report, _ = replay(lines, config_from_recording(lines))
    assert report.equal, (report.mismatches[:3], report.missing, report.extra, report.stuck_inputs)
    assert report.matched['acp_prompt'] == 1
    assert report.matched['acp_respond_permission'] == 1


def test_owner_events_cover_permissions_questions_and_deliveries() -> None:
    from hylive.harness import Harness

    notices = [
        {'kind': 'permission', 'content': '权限', 'path': ''},
        {'kind': 'input', 'content': '补充', 'path': ''},
        {'kind': 'completed', 'content': '完成', 'path': ''},
        {'kind': 'delegate', 'content': '回执', 'path': ''},
    ]

    owner_events = notice_owner_events(notices, Harness(), 'owner')

    assert [(event.sequence, event.kind) for event in owner_events] == [
        (1, 'permission_request'),
        (2, 'backend_input_request'),
        (3, 'task_delivery'),
    ]


async def test_a_completed_task_keeps_its_result(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}))

    live = await world.run(scripted('写游戏'))

    (task,) = harness_agent_tasks(live.state, 'owner')
    assert task.status == 'done'
    assert task.result is not None
    assert task.result.final_output == '做好了：写游戏'
    assert task.target_session_id == 'hy-live:owner:project:task_1'


async def test_finished_work_is_told_with_the_backends_own_result_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}))

    live = await world.run(scripted('写游戏'))

    (task,) = harness_agent_tasks(live.state, 'owner')
    assert task.status == 'done'
    sources = [request['request_meta'].get('source') for request in world.midbrain.requests]
    assert CONCLUDE_SOURCE not in sources, 'no midbrain call writes the summary'
    told = [message['content'] for message in live.state.cerebellum_messages if message.get('_injected')]
    assert any('做好了：写游戏' in content for content in told), "the cerebellum hears the backend's own result"


async def test_finished_work_is_summarized_by_the_midbrain_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(monkeypatch, cerebellum_client=cerebellum({'写游戏': 'delegate'}), summarize_outcomes=True)

    await world.run(scripted('写游戏'))

    sources = [request['request_meta'].get('source') for request in world.midbrain.requests]
    assert CONCLUDE_SOURCE in sources, 'the midbrain writes the summary before the cerebellum is told'


async def test_the_delivery_deadline_leaves_out_time_frozen_on_the_parked_cerebellum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_event = asyncio.Event()
    abort_event.set()
    clock = EpisodeClock(tick_s=0.02)
    relay = cerebellum({'写游戏': 'delegate'})
    answer = relay.reply

    async def create_completion(**kwargs: Any) -> Any:
        relay.requests.append(kwargs)
        if kwargs['request_meta']['source'] == 'observe':
            # A partial-rollout pause: the trainer parks the cerebellum inside its call.
            abort_event.clear()
            await asyncio.sleep(0.6)
            abort_event.set()
        return SimpleNamespace(choices=[SimpleNamespace(message=answer(kwargs['messages'], kwargs))])

    relay.create_completion = create_completion  # type: ignore[method-assign]
    answers: list[str] = []
    world = World(monkeypatch, cerebellum_client=relay, turn=asks_permission_then_hangs(answers), clock=clock)
    clock.bind(abort_event=abort_event, background_working=world.shell.background_working)
    timeout = 0.3

    live = await world.run(scripted('写游戏', {'content': '你好', 'gate': 'after_delivery'}), task_delivery_timeout=timeout)

    summary = live.episode.clock
    assert summary['frozen_s'] == pytest.approx(0.6, abs=0.15), (
        'Parked on the permission notice with the Run waiting for the user, the episode is frozen'
    )
    assert summary['wall_s'] - summary['frozen_s'] >= timeout - 0.05, 'The deadline grants its whole unfrozen time'
    assert live.episode.episode_termination == 'TASK_TIMEOUT'
