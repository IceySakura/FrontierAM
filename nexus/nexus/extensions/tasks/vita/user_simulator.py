"""Async user simulator for VitaBench using Nexus OpenAI client."""

import logging

from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from nexus.extensions.tasks.vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from nexus.extensions.tasks.vita.environment.tool import Tool
from nexus.extensions.tasks.vita.llm_utils import generate
from nexus.extensions.tasks.vita.prompts import get_prompts


logger = logging.getLogger(__name__)

STOP = '###STOP###'
TRANSFER = '###TRANSFER###'
OUT_OF_SCOPE = '###OUT-OF-SCOPE###'

ValidUserInputMessage = UserMessage | ToolMessage | MultiToolMessage


def is_valid_user_history_message(message: Message) -> bool:
    return (
        isinstance(message, UserMessage)
        or (isinstance(message, AssistantMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == 'user')
    )


class UserState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list

    def flip_roles(self) -> list:
        flipped = []
        for message in self.messages:
            if isinstance(message, UserMessage):
                flipped.append(
                    AssistantMessage(
                        role='assistant',
                        tool_calls=message.tool_calls,
                        content=message.content,
                    )
                )
            elif isinstance(message, AssistantMessage):
                if not message.is_tool_call():
                    flipped.append(UserMessage(role='user', content=message.content))
                else:
                    raise ValueError(f'Tool calls not supported in flipped messages: {message}')
            elif isinstance(message, ToolMessage):
                if message.requestor == 'user':
                    flipped.append(
                        ToolMessage(
                            id=message.id,
                            name=message.name,
                            role=message.role,
                            content=message.content,
                        )
                    )
                else:
                    raise ValueError(f'Tool messages should be sent to user: {message}')
            else:
                raise ValueError(f'Unknown message role: {getattr(message, "role", "?")}')
        return flipped


class UserSimulator:
    """Async user simulator that uses a Nexus OpenAI client."""

    def __init__(
        self,
        openai_client: Any,
        persona: str | None = None,
        instructions: str | None = None,
        tools: list[Tool] | None = None,
        language: str = 'chinese',
        llm_kwargs: dict | None = None,
    ):
        self.openai_client = openai_client
        self.persona = persona
        self.instructions = instructions
        self.tools = tools
        self.language = language
        self.llm_kwargs = deepcopy(llm_kwargs) if llm_kwargs else {}

    @property
    def system_prompt(self) -> str:
        prompts = get_prompts(self.language)
        return prompts.user_system_prompt.format(  # type: ignore[attr-defined]
            persona=self.persona,
            instructions=self.instructions,
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> UserState:
        if message_history is None:
            message_history = []
        return UserState(
            system_messages=[SystemMessage(role='system', content=self.system_prompt)],
            messages=list(message_history),
        )

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        if message.is_tool_call():
            return False
        if message.content is None:
            return False
        return STOP in message.content or TRANSFER in message.content or OUT_OF_SCOPE in message.content

    async def generate_next_message(
        self,
        message: ValidUserInputMessage,
        state: UserState,
    ) -> tuple[UserMessage, UserState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()
        assistant_message = await generate(
            openai_client=self.openai_client,
            messages=messages,
            tools=self.tools,
            **self.llm_kwargs,
        )

        user_message = UserMessage(
            role='user',
            content=assistant_message.content,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        if assistant_message.tool_calls is not None:
            user_message.tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    requestor='user',
                )
                for tc in assistant_message.tool_calls
            ]

        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int):
        self.llm_kwargs['seed'] = seed
