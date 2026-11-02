"""Tool-calling agent with think-act loop.

Uses Nexus core types directly:
- ExtendedMessage for LLM responses (no custom Message wrapper)
- ChatCompletionMessageToolCall for tool calls (no custom ToolCall wrapper)
- Plain dicts for message history (OpenAI-compatible format)
"""

import time
import traceback

from json import JSONDecodeError
from typing import Any, TypeVar, cast

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pydantic import Field

from nexus.core import (
    ContextLengthExceededError,
    OpenAICompatibleClient,
    ToolExecutor,
    ToolResponse,
    require_context,
)
from nexus.core.types import ExtendedMessage
from nexus.extensions.agents.base.base import BaseAgent
from nexus.extensions.agents.base.errors import is_non_retryable_provider_exception
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.prompt import TOOL_CALL_AGENT_SYSTEM_PROMPT, TOOL_CALL_REQUIRED
from nexus.extensions.agents.base.schema import (
    TOOL_CHOICE_TYPE,
    MessageContent,
    TerminationReason,
    ToolCallAgentInput,
    ToolCallAgentOutput,
    ToolChoice,
)
from nexus.extensions.agents.base.utils import (
    extract_text_from_content,
    parse_func_arguments,
)
from nexus.trace import get_current_span_id, start_span, update_span
from nexus.trace.types import Status


TToolCallAgent = TypeVar('TToolCallAgent', bound='ToolCallAgent')


class ToolCallAgent(BaseAgent):
    """Tool-calling agent with think-act loop. Subclass and set executor/system_prompt."""

    name: str = 'toolcall'
    description: str = 'an agent that can execute tool calls.'
    system_prompt: str = TOOL_CALL_AGENT_SYSTEM_PROMPT

    # Tool execution
    executor: ToolExecutor = Field(
        ...,
        description='The component responsible for executing tools and managing their schemas',
    )
    tool_choice: TOOL_CHOICE_TYPE = Field(
        default=ToolChoice.AUTO, description='The strategy for tool selection (auto, none, or required)'
    )
    tool_calls: list[ChatCompletionMessageToolCall] = Field(
        default_factory=list,
        exclude=True,  # transient: rebuilt each think() call, not part of agent state
    )
    finish_tool_names: list[str] = Field(
        default_factory=list,
        description='A list of tool names that will cause the agent to finish its execution when called',
    )
    tool_error_count: int = Field(default=0, description='Counter for tool errors during execution')
    is_training: bool = Field(
        default=True,
        description='When True (training), tool errors set exit_status to terminate the agent immediately. '
        'When False (inference), tool errors are fed back to LLM for self-correction.',
    )

    async def step(self) -> None:
        """One think-act cycle: decide tool calls, then execute them."""
        if await self.think():
            await self.act()

    @property
    def client(self) -> OpenAICompatibleClient:
        """Get OpenAI-compatible client from ExecutionContext."""
        ctx = require_context()
        return ctx.openai_client

    def build_messages(self) -> list[dict[str, Any]]:
        """Build messages list for LLM requests."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.extend(dict(m) for m in self.messages)
        return messages

    async def ask_llm(self, **kwargs) -> ExtendedMessage:
        """Query LLM with tool schemas injected. Returns ExtendedMessage from core."""
        tools = self.executor.get_tool_schemas()
        messages = self.build_messages()

        llm_span_attrs: dict = {
            'type': 'agent_inference',
        }
        llm_span = start_span(
            'agent.inference',
            llm_span_attrs,
        )
        try:
            response = await self.client.create_completion(
                messages=messages,
                tools=tools,
                tool_choice=self.tool_choice,
                **kwargs,
            )
            llm_span.update(status=Status(code='OK'))
            return response.choices[0].message
        except ContextLengthExceededError as e:
            # Context too long: input exceeds the model's max context window.
            # Don't append to messages — context is already full, agent must exit.
            logger.error(f'🎯 [{self.name}] Context length exceeded: {e}')
            self.exit_status = TerminationReason.TRUNCATION
            llm_span.update(status=Status(code='ERROR', message=str(e)))
            return ExtendedMessage(content=None, tool_calls=None)
        except (JSONDecodeError, KeyError) as e:
            # Tool format error: LLM response payload could not be parsed.
            logger.error(f'🎯 [{self.name}] Tool format error: {e}')
            self.tool_error_count += 1
            self.exit_status = TerminationReason.FORMAT_ERROR
            llm_span.update(status=Status(code='ERROR', message=str(e)))
            return ExtendedMessage(content=None, tool_calls=None)
        except Exception as e:
            if is_non_retryable_provider_exception(e):
                logger.error(f'🎯 [{self.name}] Provider error: {e}')
                self.tool_error_count += 1
                self.exit_status = TerminationReason.PROVIDER_ERROR
                llm_span.update(status=Status(code='ERROR', message=str(e)))
                return ExtendedMessage(content=None, tool_calls=None)
            logger.error(f'🎯 [{self.name}] LLM client error: {e}')
            self.tool_error_count += 1
            self.exit_status = TerminationReason.UNHANDLED_ERROR
            llm_span.update(status=Status(code='ERROR', message=str(e)))
            return ExtendedMessage(content=None, tool_calls=None)
        finally:
            llm_span.finish()

    async def think(self, **kwargs) -> bool:
        """Call LLM to decide tool selection. Return True if tools should be executed."""
        message = await self.ask_llm(**kwargs)

        # Early exit if error occurred during LLM call (exit_status already set)
        if self.exit_status is not None:
            return False

        content = message.content or ''
        reasoning_content = message.reasoning_content or ''
        self.tool_calls = message.tool_calls or []

        logger.debug(f'🧠 [{self.name}] Reasoning content: {reasoning_content}')
        logger.debug(f'💬 [{self.name}] Content: {content}')
        logger.debug(f'🛠️ [{self.name}] Selected tools: {[tc.function.name for tc in self.tool_calls]}')
        logger.debug(
            f'🔧 [{self.name}] Tool arguments: {", ".join(f"{c.function.name}: {c.function.arguments}" for c in self.tool_calls)}'
        )

        self.messages.append(message.model_dump(exclude_none=True))

        # Determine next action
        if self.tool_calls:
            return True

        if self.tool_choice == ToolChoice.REQUIRED:
            # Required but missing tools: feedback error and retry
            logger.warning(f'[{self.name}]: {TOOL_CALL_REQUIRED}')
            self.add_message('user', TOOL_CALL_REQUIRED)
            return True

        # No tools in AUTO/NONE: conversation finished (preserve error status if set)
        if self.exit_status is None:
            self.exit_status = TerminationReason.SUCCESS
        return False

    async def act(self) -> None:
        """Execute pending tool calls. Called only when think() returns True."""
        start_time = time.time()
        for tool_call in self.tool_calls:
            await self.execute_tool(tool_call)
        observation_time = time.time() - start_time
        span_id = get_current_span_id()
        if span_id:
            update_span(
                span_id,
                attributes={
                    'hytuner.observation_time': observation_time,
                },
            )

    async def execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> None:
        """Execute a single tool call via the tool executor."""
        if not self.executor:
            raise RuntimeError('No tool executor configured')

        name = tool_call.function.name
        try:
            self.executor._resolve_tool_name(name)
        except KeyError as e:
            self._handle_tool_error(str(e), tool_call, TerminationReason.FORMAT_ERROR)
            return

        # Handle arguments: can be str or dict
        raw_args = tool_call.function.arguments

        # Parse arguments (format already validated in ask_llm)
        try:
            args = parse_func_arguments(raw_args)
        except ValueError as e:
            # Format error: LLM returned invalid JSON arguments
            error = f'Argument parse error: {name}: {e}'
            self._handle_tool_error(error, tool_call, TerminationReason.FORMAT_ERROR)
            return

        # Execute tool: catch runtime errors
        tool_span_attrs: dict = {
            'type': 'agent_observation',
            'tool_name': name,
        }
        tool_span = start_span(
            'agent.observation',
            tool_span_attrs,
        )
        try:
            response = await self.executor.call(name, args)
            tool_span.update(status=Status(code='OK'))
        except Exception as e:
            # Execution error: tool call failed at runtime
            exception_type = type(e).__name__
            exception_message = str(e)
            if is_non_retryable_provider_exception(e):
                error = f'An unrecoverable error occurred when calling tool `{name}`:\n{exception_type}: {exception_message}'
                self._handle_tool_error(error, tool_call, TerminationReason.PROVIDER_ERROR)
                self.exit_status = TerminationReason.PROVIDER_ERROR
                tool_span.update(status=Status(code='ERROR', message=str(e)))
                return
            traceback_info = ''.join(traceback.format_tb(e.__traceback__))
            error = (
                f'An error occurred when calling tool `{name}`:\n'
                f'{exception_type}: {exception_message}\n'
                f'Traceback:\n{traceback_info}'
            )
            self._handle_tool_error(error, tool_call, TerminationReason.ENV_ERROR)
            tool_span.update(status=Status(code='ERROR', message=str(e)))
            return
        finally:
            tool_span.finish()

        # Check if this tool signals completion
        if name in self.finish_tool_names:
            logger.info(f"🏁 [{self.name}] Completed via '{name}'")
            self.exit_status = TerminationReason.SUCCESS

        # Hook: process side effects and format output
        result = self.on_tool_executed(response)
        logger.debug(f'🎯 [{self.name}] Tool response from {name}: {str(result)}')
        self.add_message('tool', result, tool_call_id=tool_call.id, name=name)

    def _handle_tool_error(
        self, error: str, tool_call: ChatCompletionMessageToolCall, status: TerminationReason
    ) -> None:
        """Record tool error and optionally terminate (training) or continue (inference)."""
        logger.error(f'🎯 [{self.name}] {error}')
        self.add_message('tool', error, tool_call_id=tool_call.id, name=tool_call.function.name)
        self.tool_error_count += 1
        if self.is_training:
            self.exit_status = status

    def on_tool_executed(self, response: ToolResponse) -> MessageContent:
        """Hook after tool execution: process side effects and return formatted output.

        Override in subclasses to handle context_updates or format output differently.
        Default implementation concatenates all text parts from response.output.
        """
        return extract_text_from_content(response.output)

    def get_final_answer(self) -> MessageContent | None:
        for msg in reversed(self.messages):
            if msg.get('role') == 'tool' and msg.get('name') in self.finish_tool_names:
                return msg.get('content')
        return None

    def get_output(self) -> ToolCallAgentOutput:
        """Get the final agent output with trajectory, final answer, and exit_status."""
        return ToolCallAgentOutput(
            main_trajectory=self.main_trajectory,
            final_answer=self.get_final_answer(),
            exit_status=self.exit_status or TerminationReason.SUCCESS,
        )

    @classmethod
    def create(cls, agent_input: ToolCallAgentInput, executor: ToolExecutor, **kw: Any) -> 'TToolCallAgent':
        """Create from agent input and executor."""
        params = agent_input.model_dump(exclude={'user_prompt'})
        params['system_prompt'] = params.get('system_prompt') or cls.model_fields['system_prompt'].default
        return cast(TToolCallAgent, cls(executor=executor, **params, **kw))
