"""Async orchestration loop for VitaBench: Agent <-> User <-> Environment."""

import logging
import time
import uuid

from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from nexus.extensions.tasks.vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from nexus.extensions.tasks.vita.data_model.simulation import SimulationRun, TerminationReason
from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.environment.db import DB
from nexus.extensions.tasks.vita.environment.environment import Environment
from nexus.extensions.tasks.vita.llm_utils import get_cost
from nexus.extensions.tasks.vita.user_simulator import UserSimulator, UserState, is_valid_user_history_message


logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = 'chinese'


class Role(str, Enum):
    AGENT = 'agent'
    USER = 'user'
    ENV = 'env'


def _get_now() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _format_time(dt: datetime) -> str:
    return dt.strftime('%Y%m%d_%H%M%S')


def _get_default_first_agent_message(language: str = DEFAULT_LANGUAGE) -> AssistantMessage:
    content = '你好，请问需要什么服务？' if language == 'chinese' else 'Hello, how can I help you?'
    return AssistantMessage(role='assistant', content=content, cost=0.0)


def _is_valid_agent_history_message(message: Message) -> bool:
    return (
        isinstance(message, AssistantMessage)
        or (isinstance(message, UserMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == 'assistant')
    )


class VitaLLMAgent:
    """Async wrapper around an LLM agent for VitaBench orchestration."""

    STOP_TOKEN = '###STOP###'

    def __init__(
        self,
        openai_client: Any,
        tools: list,
        system_prompt: str,
        enable_think: bool = False,
        llm_kwargs: dict | None = None,
    ):
        self.openai_client = openai_client
        self.tools = tools
        self._system_prompt = system_prompt
        self.enable_think = enable_think
        self.llm_kwargs = dict(llm_kwargs) if llm_kwargs else {}

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(self, message_history: list[Message] | None = None) -> dict:
        if message_history is None:
            message_history = []
        return {
            'system_messages': [SystemMessage(role='system', content=self._system_prompt)],
            'messages': list(message_history),
        }

    async def generate_next_message(
        self,
        message: Any,
        state: dict,
    ) -> tuple[AssistantMessage, dict]:
        from nexus.extensions.tasks.vita.llm_utils import generate

        if isinstance(message, MultiToolMessage):
            state['messages'].extend(message.tool_messages)
        else:
            state['messages'].append(message)

        messages = state['system_messages'] + state['messages']

        assistant_message = await generate(
            openai_client=self.openai_client,
            messages=messages,
            tools=self.tools,
            enable_think=self.enable_think,
            **self.llm_kwargs,
        )
        state['messages'].append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        self.llm_kwargs['seed'] = seed


class AsyncOrchestrator:
    """Async orchestrator that manages Agent <-> User <-> Environment message passing."""

    def __init__(
        self,
        domain: str,
        agent: VitaLLMAgent,
        user: UserSimulator,
        environment: Environment,
        task: Task,
        max_steps: int = 300,
        max_errors: int = 10,
        seed: int | None = None,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.language = language
        self.agent_state: dict | None = None
        self.user_state: UserState | None = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: TerminationReason | None = None
        self.num_errors = 0
        self.from_role: Role | None = None
        self.to_role: Role | None = None
        self.message: Message | None = None

    def initialize(self):
        message_history = (
            deepcopy(self.task.message_history)
            if self.task is not None and self.task.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None  # type: ignore[invalid-assignment]

        message_history = self._add_timestamps(message_history)

        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        if len(message_history) > 0:
            last_message = message_history[-1]
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                self.to_role = Role.ENV if last_message.is_tool_call() else Role.USER
                self.agent_state = self.agent.get_init_state([
                    m for m in message_history if _is_valid_agent_history_message(m)
                ])
                self.user_state = self.user.get_init_state([
                    m for m in message_history[:-1] if is_valid_user_history_message(m)
                ])
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                self.to_role = Role.ENV if last_message.is_tool_call() else Role.AGENT
                self.user_state = self.user.get_init_state([
                    m for m in message_history if is_valid_user_history_message(m)
                ])
                self.agent_state = self.agent.get_init_state([
                    m for m in message_history[:-1] if _is_valid_agent_history_message(m)
                ])
                self.message = last_message
                if UserSimulator.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.USER_STOP
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == 'assistant':
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state([
                        m for m in message_history[:-1] if _is_valid_agent_history_message(m)
                    ])
                    self.user_state = self.user.get_init_state([
                        m for m in message_history if is_valid_user_history_message(m)
                    ])
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state([
                        m for m in message_history if _is_valid_agent_history_message(m)
                    ])
                    self.user_state = self.user.get_init_state([
                        m for m in message_history[:-1] if is_valid_user_history_message(m)
                    ])
                self.message = last_message
            else:
                raise ValueError(f'Invalid last message type: {type(last_message)}')
            self.trajectory = message_history
        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            first_message = deepcopy(_get_default_first_agent_message(self.language))
            first_message.timestamp = _get_now()
            self.trajectory = [first_message]
            self.message = first_message
            self.from_role = Role.AGENT
            self.to_role = Role.USER

    async def run(self) -> SimulationRun:
        start_time = _get_now()
        start = time.perf_counter()
        self.initialize()

        logger.debug('Orchestrator run started, max_steps=%d', self.max_steps)
        while not self.done:
            await self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS

        duration = time.perf_counter() - start
        messages = self._get_trajectory()
        res = get_cost(messages)
        agent_cost, user_cost = res if res else (None, None)

        return SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=_get_now(),
            duration=duration,
            termination_reason=self.termination_reason or TerminationReason.MAX_STEPS,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
            states=self._get_states(self.environment.tools.db, self.environment.tools.db.time),  # type: ignore[arg-type]
        )

    async def step(self):
        if self.done:
            raise ValueError('Simulation is done')

        logger.debug('Step %d: %s -> %s', self.step_count, self.from_role, self.to_role)

        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            assert self.message is not None
            assert self.user_state is not None
            logger.debug('Step %d: calling user simulator LLM...', self.step_count)
            user_msg, self.user_state = await self.user.generate_next_message(self.message, self.user_state)  # type: ignore[arg-type]
            logger.debug(
                'Step %d: user simulator returned, content_len=%d', self.step_count, len(user_msg.content or '')
            )
            user_msg.validate()
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            self.to_role = Role.ENV if user_msg.is_tool_call() else Role.AGENT

        elif (self.from_role in [Role.USER, Role.ENV]) and self.to_role == Role.AGENT:
            max_retries = 3
            retry_count = 0
            agent_msg = None
            original_agent_state = deepcopy(self.agent_state)

            while retry_count < max_retries:
                current_state = deepcopy(original_agent_state)
                assert self.message is not None
                assert current_state is not None
                logger.debug(
                    'Step %d: calling agent LLM (attempt %d/%d)...', self.step_count, retry_count + 1, max_retries
                )
                agent_msg, updated_state = await self.agent.generate_next_message(self.message, current_state)
                logger.debug(
                    'Step %d: agent LLM returned, content_len=%d, tool_calls=%d',
                    self.step_count,
                    len(agent_msg.content or ''),
                    len(agent_msg.tool_calls or []),
                )
                if agent_msg.has_text_content() or agent_msg.is_tool_call():
                    self.agent_state = updated_state
                    break
                retry_count += 1
                logger.warning('Agent invalid message (attempt %d/%d): %s', retry_count, max_retries, agent_msg)

            if retry_count >= max_retries:
                self.done = True
                self.termination_reason = TerminationReason.INVALID_AGENT_MESSAGE
                return

            agent_msg.validate()
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            self.to_role = Role.ENV if agent_msg.is_tool_call() else Role.USER

        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            assert self.message is not None
            if not self.message.is_tool_call():  # type: ignore[union-attr]
                raise ValueError('Expected tool call to environment')
            tool_calls_list = self.message.tool_calls or []  # type: ignore[union-attr]
            logger.debug('Step %d: executing %d tool call(s) in environment', self.step_count, len(tool_calls_list))
            tool_msgs = []
            for tool_call in tool_calls_list:
                tool_msg = self.environment.get_response(tool_call)
                tool_msgs.append(tool_msg)
                if tool_msg.error:
                    self.num_errors += 1
            self.trajectory.extend(tool_msgs)
            if len(tool_msgs) > 1:
                self.message = MultiToolMessage(role='tool', tool_messages=tool_msgs)
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(f'Invalid role combination: {self.from_role} -> {self.to_role}')

        self.step_count += 1

    def _get_trajectory(self) -> list[Message]:
        messages = sorted(deepcopy(self.trajectory), key=lambda x: x.timestamp or '')
        for i, msg in enumerate(messages):
            msg.turn_idx = i  # type: ignore[invalid-assignment]
        return messages

    def _get_states(self, db: DB, env_time: str) -> dict[str, Any]:
        from nexus.extensions.tasks.vita.vita_utils import str_to_datetime

        states = []
        for attr in ('orders', 'books', 'reservations'):
            collection = getattr(db, attr, None)
            if collection:
                states.extend(collection.values())

        states_dict: dict[str, list] = {'old_states': [], 'new_states': []}
        for state in states:
            if str_to_datetime(state.update_time) < str_to_datetime(env_time):
                states_dict['old_states'].append(state)
            else:
                states_dict['new_states'].append(state)
        return states_dict

    def _add_timestamps(self, message_history: list[Message]) -> list[Message]:
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = _format_time(time_offset + timedelta(seconds=i))  # type: ignore[invalid-assignment]
        return message_history
