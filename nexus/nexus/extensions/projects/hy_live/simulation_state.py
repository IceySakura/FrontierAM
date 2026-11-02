"""Model-driven scenario progression with delivery-confirmed state and private audit events."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nexus.utils.timeout import async_timeout

from .simulation_scenario import oral_chunks
from .simulator import PROMPT_VERSION
from .trace_spans import USER_SIMULATOR_INVOCATION


if TYPE_CHECKING:
    from nexus.core import ExecutionContext

    from .episode_clock import EpisodeClock
    from .user_simulator import LLMUserSimulatorConfig, UserVisibleConversation


class UserDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    action: Literal['speak', 'wait', 'done']
    message: str = ''
    event: Literal['none', 'question', 'permission', 'failure'] = 'none'
    required_node_index: int | None = Field(default=None, ge=0)
    node_complete: bool = False
    skipped_required_node_index: int | None = Field(default=None, ge=0)
    optional_insertion_index: int | None = Field(default=None, ge=0)
    material_chunk_id: str | None = None
    progress_query: bool = False
    evidence_message_indices: list[int] = Field(default_factory=list)
    ending: Literal['none', 'normal', 'early', 'unable_to_continue', 'abandoned'] = 'none'
    assessment: str = Field(min_length=1, max_length=800)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def consistent_action(self) -> UserDecision:
        if self.action == 'speak':
            if not self.message.strip() or self.ending != 'none':
                raise ValueError('speak requires a message and ending=none')
        elif self.message or self.required_node_index is not None or self.optional_insertion_index is not None:
            raise ValueError('wait/done cannot contain a message or advance nodes')
        if self.action != 'speak' and (self.node_complete or self.material_chunk_id or self.progress_query):
            raise ValueError('only a delivered speech turn can advance materials, nodes, or progress-query time')
        if self.action != 'speak' and self.event != 'none':
            raise ValueError('event responses require an actual user message')
        if self.node_complete and self.required_node_index is None:
            raise ValueError('node_complete requires required_node_index')
        if self.required_node_index is not None and self.optional_insertion_index is not None:
            raise ValueError('one turn may address a required node or an optional insertion, not both')
        if (self.action == 'done') != (self.ending != 'none'):
            raise ValueError('only done may name an ending, and it must name one')
        if self.skipped_required_node_index is not None and self.action != 'wait':
            raise ValueError('skipping a node requires a silent wait decision, not speech or an ending')
        return self


class SimulatorDecisionError(ValueError):
    """The user model exhausted its decision repair budget; stop the batch before another case."""


def indexed_conversation(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explicit zero-based labels belong only to the private user-model input."""
    return [{**copy.deepcopy(message), 'index': index} for index, message in enumerate(history)]


def evidence_reference(history: list[dict[str, Any]]) -> dict[str, Any]:
    assistants = [index for index, message in enumerate(history) if message['role'] == 'assistant']
    return {
        'message_count': len(history),
        'valid_indices': list(range(len(history))),
        'assistant_indices': assistants,
        'latest_assistant_index': assistants[-1] if assistants else None,
        'instruction': 'Copy conversation[index].index exactly; indices start at 0. The message count is not an index.',
    }


class ScenarioUserRuntime:
    """The LLM chooses meaning; code enforces ordering, clocks, disclosure, and output boundaries.

    With an ``EpisodeClock`` every user timer runs on episode time: the budget, the decision ticks, the
    progress-query cooldown and the timing the user model reads all skip the stretches a trainer-paused
    cerebellum froze the episode, and no decision starts while the cerebellum is parked. Without one,
    as in a standalone run, they are wall time.
    """

    def __init__(
        self, config: LLMUserSimulatorConfig, ctx: ExecutionContext, *, clock: EpisodeClock | None = None
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.clock = clock
        self.client = ctx.get_openai_client(config.client_name)
        self.card = config.scenario
        self.card.assert_fresh_start()
        # Bind once so an edited template cannot silently change an in-flight episode.
        self.system_prompt = config.resolved_system_prompt
        self._started_at: float | None = None
        self._generated_turns = 0
        self._finished = False
        self._next_node = 0
        self._skipped_nodes: set[int] = set()
        self._node_started = False
        self._last_node_message_index = -1
        self._used_insertions: set[int] = set()
        self._delivered_chunks: set[str] = set()
        self._last_progress_at: float | None = None
        self._first_user_at: float | None = None
        self._last_user_at: float | None = None
        self._pending: tuple[UserDecision, str, int] | None = None
        self._pending_ending: tuple[UserDecision, int] | None = None
        self._conversation: UserVisibleConversation | None = None
        self._prepared = False
        self._prepare_task: asyncio.Task[None] | None = None
        self._materials: list[dict[str, Any]] = []
        self._decision_count = 0
        self.decision_events: list[dict[str, Any]] = []

    def _now(self) -> float:
        return self.clock.elapsed() if self.clock is not None else time.monotonic()

    def _elapsed(self) -> float:
        return 0.0 if self._started_at is None else max(0.0, self._now() - self._started_at)

    @contextmanager
    def _held(self, scope: Any) -> Iterator[None]:
        """Push an entered timeout scope out by every frozen stretch, so it measures episode time."""
        if self.clock is None:
            yield
            return
        with self.clock.hold(scope):
            yield

    async def _wait_while_cerebellum_paused(self) -> None:
        """Hold off a decision while the trainer parks the cerebellum.

        Nothing said in that stretch can be answered, and once the cerebellum resumes its speech makes
        any decision taken meanwhile stale, so the call would only burn user-model quota.
        """
        if self.clock is None or not self.clock.cerebellum_paused:
            return
        self._emit('decision_deferred', reason='cerebellum paused by the trainer')
        while self.clock.cerebellum_paused:
            await asyncio.sleep(self.clock.tick_s)

    def _emit(self, event: str, **values: Any) -> None:
        item = {
            'event': event,
            'script_id': self.card.meta.script_id,
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(self._elapsed(), 3),
            **values,
        }
        self.decision_events.append(copy.deepcopy(item))

    def state(self) -> dict[str, Any]:
        return {
            'delivered_user_turns': self._generated_turns,
            'next_required_node_index': self._next_node if self._next_node < len(self.card.required_nodes) else None,
            'current_node_started': self._node_started,
            'completed_required_node_indices': [i for i in range(self._next_node) if i not in self._skipped_nodes],
            'skipped_required_node_indices': sorted(self._skipped_nodes),
            'used_optional_insertion_indices': sorted(self._used_insertions),
            'delivered_material_chunk_ids': sorted(self._delivered_chunks),
            'remaining_material_chunk_ids': [m['id'] for m in self._materials if m['id'] not in self._delivered_chunks],
            'event_policy_counts': dict(self._event_counts),
        }

    def outcome(self) -> dict[str, Any]:
        """Summarize how the user ended and how often the user model's decisions were rejected or discarded."""
        counts = Counter(item['event'] for item in self.decision_events)
        ending = None
        for item in reversed(self.decision_events):
            if item['event'] == 'simulation_ended':
                ending = item['ending']
                break
            if item['event'] == 'simulation_failed':
                ending = 'simulation_failed'
                break
        return {
            'ending': ending,
            'elapsed_s': round(self._elapsed(), 3),
            'state': self.state(),
            'decisions': counts['decision_generated'],
            'rejected_decisions': counts['decision_rejected'],
            'discarded_decisions': counts['decision_discarded'],
        }

    @property
    def _event_counts(self) -> dict[str, int]:
        counts = {'question': 0, 'permission': 0, 'failure': 0}
        for item in self.decision_events:
            if item['event'] == 'user_message_delivered':
                event = item['decision']['event']
                if event in counts:
                    counts[event] += 1
        return counts

    async def _prepare_materials(self) -> None:
        from nexus.runtime.executors.models.file import FileItem

        prepared: list[dict[str, Any]] = []
        files: list[FileItem] = []
        identity = f'{getattr(self.ctx, "run_id", "") or id(self)}:{self.card.meta.script_id}'
        directory = '/workspace/user-inputs/' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        for index, material in enumerate(self.card.context.materials):
            if material.via == '口述':
                for part, text in enumerate(oral_chunks(material.content or '')):
                    prepared.append({
                        'id': f'm{index}:c{part}',
                        'name': material.name,
                        'via': '口述',
                        'text': text,
                        'max_message_chars': 79 - len(text),
                    })
            elif material.url:
                prepared.append({
                    'id': f'm{index}:url',
                    'name': material.name,
                    'via': '链接',
                    'text': material.url,
                    'max_message_chars': 30,
                })
            else:
                assert material.content is not None, 'ScenarioMaterial requires content for a non-URL material'
                path = f'{directory}/material-{index + 1}.txt'
                files.append(FileItem(path=path, content=material.content, encoding='utf8'))
                prepared.append({
                    'id': f'm{index}:file',
                    'name': material.name,
                    'via': '文件',
                    'text': path,
                    'max_message_chars': min(30, 79 - len(path)),
                })
        if files:
            runtime = await self.ctx.get_runtime()
            result = await runtime.upload_file(files, overwrite=False)
            succeeded = {item.path for item in result.results if item.success}
            if succeeded != {item.path for item in files}:
                raise RuntimeError('scenario material upload failed; no file reference may be sent to the assistant')
            self._emit('materials_uploaded', paths=sorted(succeeded))
        self._materials = prepared
        self._prepared = True
        self._emit('materials_prepared', material_count=len(self.card.context.materials), chunk_count=len(prepared))

    def _accept_delivery(self, message: Any, index: int) -> None:
        self._last_user_at = self._now()
        if self._pending is None:
            return
        decision, expected, expected_index = self._pending
        self._pending = None
        if message != expected or index != expected_index:
            self._emit('decision_discarded', reason='message was not delivered in the expected conversation position')
            return
        self._generated_turns += 1
        if self._first_user_at is None:
            self._first_user_at = self._last_user_at
        if decision.required_node_index is not None:
            self._node_started = True
            if decision.node_complete:
                self._next_node += 1
                self._node_started = False
                self._last_node_message_index = index
        if decision.optional_insertion_index is not None:
            self._used_insertions.add(decision.optional_insertion_index)
        if decision.material_chunk_id:
            self._delivered_chunks.add(decision.material_chunk_id)
        if decision.progress_query:
            self._last_progress_at = self._last_user_at
        self._emit(
            'user_message_delivered',
            message_index=index,
            message=message,
            decision_index=self._decision_count,
            decision=decision.model_dump(mode='json'),
        )
        if self._generated_turns >= self.config.max_turns:
            self._finished = True
            self._emit('simulation_ended', ending='runtime_budget', budget='max_turns', state=self.state())

    def is_message_current(self, message: Any, conversation: UserVisibleConversation) -> bool:
        if message is None:
            if self._pending_ending is None:
                return True
            decision, expected_index = self._pending_ending
            self._pending_ending = None
            if len(conversation.as_payload()) != expected_index:
                self._finished = False
                self._emit('decision_discarded', reason='notification arrived before the ending was accepted')
                return False
            self._emit('simulation_ended', ending=decision.ending, decision=decision.model_dump(), state=self.state())
            return True
        if self._pending is None:
            return False
        _, expected, expected_index = self._pending
        if message == expected and len(conversation.as_payload()) == expected_index:
            return True
        self._emit('decision_discarded', reason='notification arrived before the proposed user message was delivered')
        self._pending = None
        return False

    def _timing(self, conversation: UserVisibleConversation, wake: str, waited: float) -> dict[str, Any]:
        now = self._now()
        anchor = self._last_progress_at if self._last_progress_at is not None else self._first_user_at
        since_progress = None if anchor is None else max(0.0, now - anchor)
        since_user = None if self._last_user_at is None else max(0.0, now - self._last_user_at)
        return {
            'now_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(self._elapsed(), 3),
            'remaining_s': round(max(0.0, self.config.max_elapsed_s - self._elapsed()), 3),
            'wake_reason': wake,
            'waited_s': round(waited, 3),
            'since_last_user_message_s': None if since_user is None else round(since_user, 3),
            'since_last_progress_query_s': None if since_progress is None else round(since_progress, 3),
            'progress_query_allowed': since_progress is not None and since_progress >= self.config.progress_interval_s,
            'decision_interval_s': self.config.decision_interval_s,
        }

    def _validate(self, decision: UserDecision, history: list[dict], timing: dict) -> str:
        refs = decision.evidence_message_indices
        if any(index < 0 or index >= len(history) for index in refs):
            reference = evidence_reference(history)
            raise ValueError(
                f'evidence_message_indices={refs} contains nonexistent message indices. '
                f'Valid indices are {reference["valid_indices"]}; '
                f'the latest assistant index is {reference["latest_assistant_index"]}. '
                'Use the explicit index labels in conversation; do not use the message count as an index.'
            )
        if decision.event != 'none' and not any(history[i]['role'] == 'assistant' for i in refs):
            raise ValueError('an event response needs evidence of an actual assistant request or failure')
        skipped = decision.skipped_required_node_index
        if skipped is not None:
            if skipped != self._next_node or skipped >= len(self.card.required_nodes):
                raise ValueError('only the next required node may be skipped')
            if self._generated_turns == 0 or self.card.required_nodes[skipped].when.startswith('开场'):
                raise ValueError('the opening cannot be skipped')
            if not decision.reason.strip() or not any(history[i]['role'] == 'assistant' for i in refs):
                raise ValueError('skipping a node requires a reason and actual assistant evidence')
        if decision.action == 'done':
            if not refs:
                raise ValueError('ending requires visible evidence')
            if decision.ending in {'normal', 'early'} and not any(history[i]['role'] == 'assistant' for i in refs):
                raise ValueError('completed ending requires assistant delivery evidence')
            if decision.ending == 'normal' and (
                self._next_node < len(self.card.required_nodes) or len(self._delivered_chunks) < len(self._materials)
            ):
                raise ValueError('normal ending cannot skip required nodes or undisclosed materials')
            if decision.ending == 'early' and not self.card.ending.early:
                raise ValueError('this scenario does not define an early ending')
            return ''
        if decision.action == 'wait':
            if not history:
                raise ValueError('the opening must be generated before waiting')
            return ''
        text = decision.message.strip()
        if len(text) > 80 or '\n' in text or '\r' in text or re.search(r'[`|#*]|^\s*(?:[-] |\d+[.、] |[\[{<])', text):
            raise ValueError('message must be short conversational text on one line without markdown or lists')
        if any(
            token in text
            for token in (
                self.card.meta.script_id,
                self.card.meta.topic_id,
                '[WAIT]',
                '[WAIT_LONG]',
                '[DONE]',
                'required_nodes',
                'optional_insertions',
                'event_plan',
                'material_chunk_id',
            )
        ):
            raise ValueError('message must not reveal private scenario identifiers or control fields')
        if decision.event == 'none' and re.search(
            r'(?:进度|进展).{0,6}(?:怎么样|怎样|如何|到哪|多少)|做到哪|查到哪|还要多久|还差多少|'
            r'弄得怎么样|什么时候.{0,4}(?:做好|完成)',
            text,
        ):
            decision.progress_query = True
        node_index = decision.required_node_index
        if node_index is not None:
            if node_index != self._next_node or node_index >= len(self.card.required_nodes):
                raise ValueError('only the next required node may be addressed')
            node = self.card.required_nodes[node_index]
            if not node.when.startswith('开场') and (
                self._generated_turns == 0
                or not any(history[i]['role'] == 'assistant' and i > self._last_node_message_index for i in refs)
            ):
                raise ValueError('non-opening node needs new user-visible assistant evidence for its when condition')
            if node.when.startswith('等久了'):
                decision.progress_query = True
            if node.when.startswith('开场') and self.card.user.style == 'A' and decision.node_complete:
                remaining = [m['id'] for m in self._materials if m['id'] not in self._delivered_chunks]
                if remaining and remaining != [decision.material_chunk_id]:
                    raise ValueError('finish disclosing the opening materials before completing the opening node')
        if decision.progress_query and not timing['progress_query_allowed']:
            raise ValueError('progress questions are too close together; wait or address a real event')
        optional = decision.optional_insertion_index
        if optional is not None:
            if optional >= len(self.card.optional_insertions) or optional in self._used_insertions:
                raise ValueError('optional insertion is missing or already used')
            if self._generated_turns == 0 or decision.event != 'none':
                raise ValueError('optional insertion cannot replace the opening or an event response')
        if decision.material_chunk_id:
            pending = [m for m in self._materials if m['id'] not in self._delivered_chunks]
            chosen = [m for m in pending if m['id'] == decision.material_chunk_id]
            if not chosen:
                raise ValueError('material chunk is missing or has already been sent')
            chunk = chosen[0]
            same_material = [m for m in pending if m['id'].split(':')[0] == chunk['id'].split(':')[0]]
            if same_material[0]['id'] != chunk['id']:
                raise ValueError('chunks of the same material must be disclosed in order')
            if len(text) > chunk['max_message_chars']:
                raise ValueError(
                    f'use at most {chunk["max_message_chars"]} characters of introduction for this material'
                )
            text = text + ' ' + chunk['text']
        return text

    async def _decide(self, payload: dict[str, Any]) -> tuple[UserDecision, str]:
        correction = None
        invalid_response = None
        payload = {
            **payload,
            'conversation': indexed_conversation(payload['conversation']),
            'evidence_reference': evidence_reference(payload['conversation']),
        }
        for attempt in range(self.config.decision_retries + 1):
            remaining = self.config.max_elapsed_s - self._elapsed()
            if remaining <= 0:
                raise TimeoutError('user simulator runtime budget exhausted')
            current = {
                **payload,
                **(
                    {'validation_error': correction, 'invalid_decision_response': invalid_response}
                    if correction
                    else {}
                ),
            }
            self._decision_count += 1
            self._emit(
                'decision_requested', decision_index=self._decision_count, attempt=attempt, timing=payload['timing']
            )
            try:
                async with async_timeout(remaining) as budget_scope:
                    with self._held(budget_scope):
                        response = await self.client.create_completion(
                            messages=[
                                {'role': 'system', 'content': self.system_prompt},
                                {'role': 'user', 'content': json.dumps(current, ensure_ascii=False)},
                            ],
                            request_meta={
                                'invocation_kind': USER_SIMULATOR_INVOCATION,
                                'script_id': self.card.meta.script_id,
                                'decision_index': self._decision_count,
                                'user_system_prompt_source': self.config.system_prompt_source,
                                'user_system_prompt_version': PROMPT_VERSION,
                            },
                            **self.config.request_args,
                        )
                message = response.choices[0].message
                raw = str(
                    message.get('content') if isinstance(message, dict) else getattr(message, 'content', '') or ''
                )
                try:
                    decision = UserDecision.model_validate_json(raw)
                    text = self._validate(decision, payload['conversation'], payload['timing'])
                except (ValidationError, ValueError) as error:
                    correction = str(error)
                    invalid_response = raw
                    self._emit(
                        'decision_rejected', decision_index=self._decision_count, raw_response=raw, error=correction
                    )
                    continue
                if self._elapsed() >= self.config.max_elapsed_s:
                    raise TimeoutError('user simulator runtime budget exhausted')
                self._emit(
                    'decision_generated',
                    decision_index=self._decision_count,
                    decision=decision.model_dump(mode='json'),
                    proposed_message=text,
                    state=self.state(),
                    timing=payload['timing'],
                )
                return decision, text
            except asyncio.CancelledError:
                self._emit('decision_interrupted', decision_index=self._decision_count)
                raise
        error = SimulatorDecisionError(
            'user simulator could not produce a valid scenario decision after bounded repair attempts: '
            + str(correction)
        )
        self._emit('simulation_failed', error_type=type(error).__name__, error=str(error), state=self.state())
        raise error

    async def next_user_message(self, conversation: UserVisibleConversation) -> str | None:
        if self._started_at is None:
            self._started_at = self._now()
        if self._conversation is None:
            self._conversation = conversation
            conversation.add_user_delivery_listener(self._accept_delivery)
            self._emit(
                'system_prompt_bound',
                source=self.config.system_prompt_source,
                version=PROMPT_VERSION,
                sha256=hashlib.sha256(self.system_prompt.encode('utf-8')).hexdigest(),
            )
        elif self._conversation is not conversation:
            raise ValueError('one scenario simulator instance belongs to one visible conversation')
        if self._pending is not None:
            self._emit('decision_discarded', reason='previous generated speech was not delivered')
            self._pending = None
        if self._finished:
            return None
        if not self._prepared:
            if self._prepare_task is None:
                self._prepare_task = asyncio.create_task(self._prepare_materials())
            try:
                async with async_timeout(max(0.0, self.config.max_elapsed_s - self._elapsed())) as budget_scope:
                    with self._held(budget_scope):
                        await asyncio.shield(self._prepare_task)
            except TimeoutError:
                self._finished = True
                await self.close()
                self._emit(
                    'simulation_ended', ending='runtime_budget', budget='material_preparation', state=self.state()
                )
                return None
        if self._generated_turns == 0 and self.config.initial_prompt is not None:
            return self._propose_initial_prompt(conversation)
        wake, waited = 'initial' if self._generated_turns == 0 else 'new_speech', 0.0
        while True:
            await self._wait_while_cerebellum_paused()
            if self._generated_turns >= self.config.max_turns or self._elapsed() >= self.config.max_elapsed_s:
                self._finished = True
                self._emit('simulation_ended', ending='runtime_budget', state=self.state())
                return None
            history = conversation.as_payload()
            payload = {
                'scenario': self.card.model_dump(mode='json'),
                'conversation': history,
                'state': self.state(),
                'material_chunks': self._materials,
                'effective_event_plan': self.card.event_responses(),
                'timing': self._timing(conversation, wake, waited),
            }
            try:
                decision, text = await self._decide(payload)
            except TimeoutError:
                if self._elapsed() < self.config.max_elapsed_s:
                    raise
                self._finished = True
                self._emit('simulation_ended', ending='runtime_budget', state=self.state())
                return None
            if conversation.as_payload() != history:
                self._emit('decision_discarded', reason='new speech arrived while the user model was deciding')
                wake, waited = 'new_speech', 0.0
                continue
            if decision.skipped_required_node_index is not None:
                skipped = decision.skipped_required_node_index
                self._skipped_nodes.add(skipped)
                self._next_node += 1
                self._node_started = False
                # A skip produces no new user/assistant turn: the same actual
                # observation may explain why the next applicable node is due.
                self._emit(
                    'required_node_skipped',
                    required_node_index=skipped,
                    decision_index=self._decision_count,
                    decision=decision.model_dump(mode='json'),
                    state=self.state(),
                )
                wake, waited = 'node_skipped', 0.0
                continue
            if decision.action == 'done':
                self._finished = True
                self._pending_ending = (decision, len(history))
                return None
            if decision.action == 'speak':
                self._pending = (decision, text, len(history))
                return text
            interval = self.config.decision_interval_s
            elapsed = self._elapsed()
            next_tick = (math.floor(elapsed / interval) + 1) * interval
            timeout = max(0.0, min(next_tick - elapsed, self.config.max_elapsed_s - elapsed))
            self._emit('waiting', duration_limit_s=round(timeout, 3), state=self.state())
            start = self._now()
            try:
                async with async_timeout(timeout) as tick_scope:
                    with self._held(tick_scope):
                        await conversation.wait_for_change(len(history))
                wake = 'new_speech'
            except TimeoutError:
                wake = 'timer_tick'
            waited = self._now() - start
            self._emit('woken', wake_reason=wake, waited_s=round(waited, 3))

    def _propose_initial_prompt(self, conversation: UserVisibleConversation) -> str:
        """Propose the configured opening as the opening node without a user-model call.

        A style-A opening that still owes materials stays open, so the user model finishes it by
        disclosing them, exactly as the opening-node rule in ``_validate`` requires.
        """
        assert self.config.initial_prompt is not None
        text = self.config.initial_prompt
        nodes = self.card.required_nodes
        addresses_opening = bool(nodes) and nodes[0].when.startswith('开场')
        owes_materials = self.card.user.style == 'A' and bool(self._materials)
        decision = UserDecision(
            action='speak',
            message=text,
            required_node_index=0 if addresses_opening else None,
            node_complete=addresses_opening and not owes_materials,
            assessment='开场尚无助手回复。',
            reason='开场白由数据集给定。',
        )
        self._pending = (decision, text, len(conversation.as_payload()))
        self._emit(
            'decision_generated',
            decision_index=self._decision_count,
            source='initial_prompt',
            decision=decision.model_dump(mode='json'),
            proposed_message=text,
            state=self.state(),
        )
        return text

    async def close(self) -> None:
        if self._prepare_task is not None and not self._prepare_task.done():
            self._prepare_task.cancel()
            await asyncio.gather(self._prepare_task, return_exceptions=True)


__all__ = [
    'ScenarioUserRuntime',
    'SimulatorDecisionError',
    'UserDecision',
    'evidence_reference',
    'indexed_conversation',
]
