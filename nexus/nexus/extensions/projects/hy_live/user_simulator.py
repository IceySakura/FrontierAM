"""User-simulation abstractions for multi-turn hy-live cerebellum training."""

from __future__ import annotations

import asyncio
import time

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nexus.core import ExecutionContext

from .episode_clock import EpisodeClock
from .simulation_scenario import ScenarioCard
from .simulation_state import ScenarioUserRuntime
from .simulator import system_prompt


UserMessage = str | list[dict[str, Any]]
"""One user turn: plain text, or OpenAI-style content parts carrying audio."""

# Content part kinds a simulated user turn may carry. Audio parts travel to the policy as
# ``{'type': 'audio', 'audio': <patch>}``, the shape the speech dataset attaches to the
# opening turn, so a scripted turn reaches vLLM through the same multi-modal path.
_CONTENT_PART_PAYLOAD_KEYS = {'audio': 'audio', 'input_audio': 'input_audio', 'text': 'text'}


class CerebellumResponse(BaseModel):
    """The user simulator's observation after one cerebellum turn."""

    model_config = ConfigDict(extra='forbid')

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class UserVisibleConversation:
    """Keep user speech and every cerebellum utterance in delivery order."""

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _changed: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _last_user_at: float | None = field(default=None, init=False, repr=False)
    _user_delivery_listeners: list[Callable[[UserMessage, int], None]] = field(
        default_factory=list, init=False, repr=False
    )
    latest_response: CerebellumResponse | None = field(default=None, init=False)

    def record_user(self, message: UserMessage) -> None:
        """Record a user message when the conversation driver accepts it."""
        self._messages.append({'role': 'user', 'content': message})
        self._last_user_at = time.monotonic()
        self._changed.set()
        for listener in self._user_delivery_listeners:
            listener(message, len(self._messages) - 1)

    def add_user_delivery_listener(self, listener: Callable[[UserMessage, int], None]) -> None:
        """Commit private simulator state only after the driver accepts a real user message."""
        if listener not in self._user_delivery_listeners:
            self._user_delivery_listeners.append(listener)

    def seconds_since_last_user_message(self) -> float | None:
        if self._last_user_at is None:
            return None
        return time.monotonic() - self._last_user_at

    def record_cerebellum(self, response: CerebellumResponse) -> None:
        """Record audible speech without exposing internal response metadata."""
        self.latest_response = response.model_copy(deep=True)
        if response.content.strip():
            self._messages.append({'role': 'assistant', 'content': response.content})
            self._changed.set()

    async def wait_for_change(self, message_count: int) -> None:
        """Wait for speech beyond a request snapshot, including in-flight arrivals."""
        while len(self._messages) == message_count:
            self._changed.clear()
            await self._changed.wait()

    def as_payload(self) -> list[dict[str, Any]]:
        """Return an independent snapshot for one user-model request."""
        return [message.copy() for message in self._messages]


class SimulatedTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')

    index: int
    user_message: UserMessage
    cerebellum_response: CerebellumResponse


class SimulatedConversation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    turns: list[SimulatedTurn] = Field(default_factory=list)
    ended_by_user: bool = False
    reached_max_turns: bool = False


UserTurnGate = Literal['immediate', 'after_delivery', 'after_progress']
"""When a scripted turn may speak.

``immediate`` is the historical behaviour: the user speaks as soon as the policy's previous
turn returns. ``after_delivery`` holds the turn until a background delivery has been
broadcast, which is what the hy-live conversations actually did whenever the user was
reacting to a delivered result. ``after_progress`` holds the turn until an outstanding task
records a tool call it had not made when the user last spoke, so a progress question lands
while the work is running instead of during sandbox startup.
"""


class ScriptedTurn(BaseModel):
    """One scripted user turn: what to say, and what has to happen first."""

    model_config = ConfigDict(extra='forbid')

    content: str | dict[str, Any]
    gate: UserTurnGate = 'immediate'


def parse_scripted_turn(turn: str | dict[str, Any]) -> ScriptedTurn:
    """Read one ``turns`` entry, accepting the bare-content and gated forms.

    Args:
        turn (str | dict[str, Any]): Plain text, a bare content part (``{'type': ...}``),
            or the gated form ``{'content': ..., 'gate': ...}``.

    Returns:
        The turn with its gate resolved, defaulting to ``immediate``.

    Raises:
        ValueError: The dict is neither a content part nor a gated turn, so its intent is
            ambiguous and must not be guessed at.
    """
    if isinstance(turn, str):
        return ScriptedTurn(content=turn)
    if 'content' in turn:
        return ScriptedTurn.model_validate(turn)
    if 'type' in turn:
        return ScriptedTurn(content=turn)
    raise ValueError(
        f'Scripted turn dict must carry either a content part `type` or a gated-turn `content`, got keys {sorted(turn)}'
    )


class ScriptedUserSimulatorConfig(BaseModel):
    """Read user messages directly from one dataset row."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['scripted'] = 'scripted'
    turns: list[str | dict[str, Any]] = Field(min_length=1)
    """Plain text, one content part, or ``{'content': ..., 'gate': ...}``."""
    turn_delay_s: float = Field(default=0.0, ge=0.0)
    """Sleep this long before every turn after the first.

    Used so a later scripted decision can lose the race to a harness
    permission notice instead of cancelling the task first.
    """


class LLMUserSimulatorConfig(BaseModel):
    """Generate a user from a structured private scenario card and real visible feedback."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['llm'] = 'llm'
    client_name: str = 'user_simulator'
    scenario: ScenarioCard
    initial_prompt: str | None = Field(default=None, min_length=1)
    """Fixed opening delivered as the opening node; None lets the user model speak first."""
    system_prompt: str | None = Field(default=None, min_length=1)
    decision_interval_s: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    progress_interval_s: float = Field(default=90.0, gt=0, allow_inf_nan=False)
    max_elapsed_s: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    max_turns: int = Field(default=40, ge=1)
    decision_retries: int = Field(default=2, ge=0, le=3)
    request_args: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_system_prompt(self) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        return system_prompt(
            self.scenario.model_dump(mode='json'),
            decision_interval_s=self.decision_interval_s,
            progress_interval_s=self.progress_interval_s,
        )

    @property
    def system_prompt_source(self) -> str:
        return 'config-override' if self.system_prompt is not None else 'scenario-template'


UserSimulatorConfig = Annotated[
    ScriptedUserSimulatorConfig | LLMUserSimulatorConfig,
    Field(discriminator='type'),
]
_USER_SIMULATOR_CONFIG_ADAPTER = TypeAdapter(UserSimulatorConfig)


def normalize_scripted_turn(turn: str | dict[str, Any]) -> UserMessage:
    """Turn one scripted entry into the content a policy message can carry.

    Args:
        turn (str | dict[str, Any]): Plain text, or a single content part such as
            ``{'type': 'audio', 'audio': {'patch': ...}}``.

    Returns:
        The text unchanged, or a one-element content part list.

    Raises:
        ValueError: The part has no supported ``type``, or its payload key is missing. A
            malformed part must fail here rather than reach the policy as a stray dict
            that renders as literal text and silently drops the audio.
    """
    if isinstance(turn, str):
        return turn
    part_type = turn.get('type')
    payload_key = _CONTENT_PART_PAYLOAD_KEYS.get(str(part_type))
    if payload_key is None:
        raise ValueError(
            f'Unsupported scripted turn content part type {part_type!r}. '
            f'Choose one of {sorted(_CONTENT_PART_PAYLOAD_KEYS)}, or pass a plain string for a text turn.'
        )
    if turn.get(payload_key) is None:
        raise ValueError(f'Scripted {part_type!r} content part must carry a {payload_key!r} payload, got {turn!r}')
    return [dict(turn)]


class UserSimulator(ABC):
    """Generate the next user request from the complete user-visible conversation."""

    @abstractmethod
    async def next_user_message(self, conversation: UserVisibleConversation) -> UserMessage | None:
        """Return a user message without recording it, or None to end the conversation."""

    def next_turn_gate(self) -> UserTurnGate:
        """Return the gate of the turn ``next_user_message`` would produce next.

        The conversation loop, not the simulator, is what can see background notices, so it
        has to ask before deciding whether to race the user or park it.
        """
        return 'immediate'

    def is_message_current(self, message: UserMessage | None, conversation: UserVisibleConversation) -> bool:
        """Allow a model-driven simulator to discard speech made stale by a concurrent notification."""
        return True

    def outcome(self) -> dict[str, Any]:
        """Return how the simulated user ended, for episode metadata; empty when nothing is tracked."""
        return {}

    async def close(self) -> None:
        """Release any scenario material preparation still in flight."""
        return


class ScriptedUserSimulator(UserSimulator):
    def __init__(self, config: ScriptedUserSimulatorConfig) -> None:
        self.config = config
        self._index = 0

    def next_turn_gate(self) -> UserTurnGate:
        if self._index >= len(self.config.turns):
            return 'immediate'
        return parse_scripted_turn(self.config.turns[self._index]).gate

    async def next_user_message(self, conversation: UserVisibleConversation) -> UserMessage | None:
        if self._index >= len(self.config.turns):
            return None
        if self._index > 0 and self.config.turn_delay_s > 0:
            await asyncio.sleep(self.config.turn_delay_s)
        turn = parse_scripted_turn(self.config.turns[self._index])
        self._index += 1
        return normalize_scripted_turn(turn.content)


class LLMUserSimulator(UserSimulator):
    """Native simulator backed by a delivery-aware, LLM-driven scenario state machine."""

    def __init__(
        self, config: LLMUserSimulatorConfig, ctx: ExecutionContext, *, clock: EpisodeClock | None = None
    ) -> None:
        self.config = config
        self.runtime = ScenarioUserRuntime(config, ctx, clock=clock)
        self._inflight: set[asyncio.Task] = set()

    @property
    def decision_events(self) -> list[dict[str, Any]]:
        return self.runtime.decision_events

    def is_message_current(self, message: UserMessage | None, conversation: UserVisibleConversation) -> bool:
        return self.runtime.is_message_current(message, conversation)

    def outcome(self) -> dict[str, Any]:
        return self.runtime.outcome()

    async def next_user_message(self, conversation: UserVisibleConversation) -> UserMessage | None:
        current = asyncio.current_task()
        if current is not None:
            self._inflight.add(current)
        try:
            return await self.runtime.next_user_message(conversation)
        finally:
            self._inflight.discard(current)

    async def close(self) -> None:
        pending = [task for task in self._inflight if task is not asyncio.current_task() and not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.runtime.close()


def build_user_simulator(
    config: UserSimulatorConfig | dict[str, Any], ctx: ExecutionContext, *, clock: EpisodeClock | None = None
) -> UserSimulator:
    """Build the configured user; ``clock`` puts a scenario user's timers on episode time."""
    normalized = (
        config
        if isinstance(config, (ScriptedUserSimulatorConfig, LLMUserSimulatorConfig))
        else _USER_SIMULATOR_CONFIG_ADAPTER.validate_python(config)
    )
    if isinstance(normalized, ScriptedUserSimulatorConfig):
        return ScriptedUserSimulator(normalized)
    return LLMUserSimulator(normalized, ctx, clock=clock)


__all__ = [
    'CerebellumResponse',
    'LLMUserSimulator',
    'LLMUserSimulatorConfig',
    'ScriptedTurn',
    'ScriptedUserSimulator',
    'ScriptedUserSimulatorConfig',
    'SimulatedConversation',
    'SimulatedTurn',
    'UserMessage',
    'UserSimulator',
    'UserSimulatorConfig',
    'UserTurnGate',
    'UserVisibleConversation',
    'build_user_simulator',
    'normalize_scripted_turn',
    'parse_scripted_turn',
]
