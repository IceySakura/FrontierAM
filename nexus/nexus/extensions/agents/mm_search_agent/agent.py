"""Search agent using ToolCallAgent infrastructure.

This agent demonstrates:
- Extending ToolCallAgent for domain-specific functionality
- Data-driven configuration over complex control flow
- Clean composition: configure, create, run, return
"""

import os
import re
import traceback

from typing import Any

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pydantic import Field

from nexus.core import ToolExecutor, ToolResponse, agent_registry, tool_registry
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.schema import (
    MessageContent,
    TerminationReason,
    ToolCallAgentInput,
    ToolCallAgentOutput,
)
from nexus.extensions.agents.base.toolcall import ToolCallAgent
from nexus.extensions.agents.base.utils import parse_func_arguments


DEFAULT_SYSTEM_PROMPT_TRAINING = """You are a helpful AI assistant that conducts thorough internet research to provide accurate answers.

IMPORTANT:
- Use only the provided tools to perform actual information retrieval without fabricating answers, ensuring only one tool is called at a time.
- Explore different aspects and refine your approach until comprehensive information is gathered. Never ask for human help.
- Synthesize information from diverse, credible sources. Cross-check facts to validate accuracy and reliability.
- Use the `finish` tool to submit a detailed, objective answer based strictly on gathered evidence.
"""


DEFAULT_SYSTEM_PROMPT_INFERENCE = """You are a helpful AI assistant that conducts thorough internet research to provide accurate answers.

IMPORTANT:
- Use only the provided tools to perform actual information retrieval without fabricating answers, ensuring only one tool is called at a time. Each round is completed one at a time — there is no need to gather all information at once.
- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent. Break down and refine the problem step by step to arrive at the final answer.
- When using the "search" tool, limit the number of search query terms to 3 or fewer. If the returned search snippets are insufficient to provide a satisfactory answer or key information for the current reasoning process, select relevant web documents from the "search" results, obtain their URLs, and use the "visit" tool to extract more detailed context and information.
- Explore different aspects and refine your approach until comprehensive information is gathered. Never ask for human help.
- If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained.
- Synthesize information from diverse, credible sources. Cross-check facts to validate accuracy and reliability.
- Use the `finish` tool to submit a detailed, objective answer based strictly on gathered evidence.
"""


def get_default_system_prompt(is_training: bool = True) -> str:
    if is_training:
        return DEFAULT_SYSTEM_PROMPT_TRAINING
    return DEFAULT_SYSTEM_PROMPT_INFERENCE


def extract_text_from_content(content) -> str:
    """Extract text string from MessageContent (str or list of content parts).

    Args:
        content: MessageContent which can be str or list[dict]

    Returns:
        Extracted text as string
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                elif item.get('type') == 'image_url':
                    text_parts.append('[IMAGE_PLACEHOLDER]')
            elif isinstance(item, str):
                text_parts.append(item)
        return ' '.join(text_parts) if text_parts else ''
    return str(content)


NOT_USE_TOOL = os.environ.get('NOT_USE_TOOL', 'False').lower() == 'true'

# Core tool configuration (search engine is dynamically detected)
REQUIRED_TOOLS = [
    'base/finish',
    'google_serper_search',
    'visit',
    'image_visit',
    'text_to_image_search',
    'image_to_image_search',
    'jupyter_with_image',
]
# REQUIRED_TOOLS = os.environ.get('TRAINING_LENGTH_EXCEEDED', 'False').lower() != 'true' else ['base/finish']
# SEARCH_ENGINE_PRIORITY = ['google_serper_search', 'google_search']
TOOLCALL_MARKUP_PATTERNS = [
    re.compile(r'<tool_call\b', re.IGNORECASE),
    re.compile(r'<tool_call>', re.IGNORECASE),
    re.compile(r'<function=[^>]+>', re.IGNORECASE),
    re.compile(r'</function>', re.IGNORECASE),
    re.compile(r'<parameter=[^>]+>', re.IGNORECASE),
    re.compile(r'</parameter>', re.IGNORECASE),
]
FORCE_FINISH_PROMPT = 'You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer.'


def contains_toolcall_markup(text: str) -> bool:
    return any(pattern.search(text) for pattern in TOOLCALL_MARKUP_PATTERNS)


def get_llm_retry_reason(message: Any) -> str | None:
    content_text = extract_text_from_content(getattr(message, 'content', None) or '')
    reasoning_text = getattr(message, 'reasoning_content', None) or ''
    tool_calls = getattr(message, 'tool_calls', None) or []

    if contains_toolcall_markup(content_text) or contains_toolcall_markup(reasoning_text):
        return 'raw tool-call markup in output'
    if reasoning_text and not content_text and not tool_calls:
        return 'reasoning only, no content or tool_calls'
    return None


def _resolve_tools(required_tools: list[str] | None = None) -> list:
    """Resolve and validate all required tools. Returns tool instances (not names)."""
    # Validate required tools first
    tools = []
    if required_tools is None:
        required_tools = REQUIRED_TOOLS
    for i in range(len(required_tools)):
        if required_tools[i] == 'code_interpreter':
            required_tools[i] = 'jupyter_with_image'

    missing = [name for name in required_tools if tool_registry.get(name) is None]
    if missing:
        raise ValueError(f'Required tools not found: {missing}')

    if not NOT_USE_TOOL:
        # Add required tools
        tools.extend(tool_registry.get(name) for name in required_tools)
    else:
        print(
            '⚠️ NOT_USE_TOOL is set to True, skipping tool resolution and execution. Agent will not be able to call any tools.'
        )
    print('tool_registry: ', tools)
    return tools


class SearchAgentInput(ToolCallAgentInput):
    """Search agent input - inherits ToolCallAgent configuration with sensible defaults."""

    system_prompt: str | None = None
    is_training: bool = True
    max_iterations: int = 10
    finish_tool_names: list[str] = Field(default_factory=lambda: ['finish'])
    max_llm_retries: int = Field(default=0, description='Max LLM call retries on bad output (0 to disable)')
    required_tools: list[str] | None = None  # Optional override for tools to use (by name)


class SearchAgentOutput(ToolCallAgentOutput):
    """Search agent output - extends with exit_status for task completion tracking."""


class SearchAgent(ToolCallAgent):
    """Search agent - thin wrapper over ToolCallAgent."""

    name: str = 'search_agent'
    description: str = 'An agent that searches the web for information.'
    max_llm_retries: int = 0

    def get_output(self) -> 'SearchAgentOutput':
        return SearchAgentOutput(
            main_trajectory=self.main_trajectory,
            final_answer=self.get_final_answer(),
            exit_status=self.exit_status or TerminationReason.SUCCESS,
        )

    async def run(self, request: MessageContent | None = None) -> MessageContent | None:
        """Run agent and enforce finish-tool based termination at max iterations."""
        if self.exit_status is not None:
            raise RuntimeError(f'Cannot run agent: already exited with {self.exit_status}')

        self.current_iteration = 0
        if request is not None:
            self.add_message('user', request)

        try:
            while self.current_iteration < self.max_iterations and self.exit_status is None:
                self.current_iteration += 1
                logger.debug(f'Executing step {self.current_iteration}/{self.max_iterations}')
                await self.step()

            if self.exit_status is None and self.current_iteration >= self.max_iterations:
                if not self.is_training:
                    logger.warning(
                        f'[{self.name}] Reached max iterations ({self.max_iterations}), forcing finish tool finalization'
                    )
                    finalized = await self.force_finish_once()
                    if not finalized:
                        self.exit_status = TerminationReason.MAX_ITERATIONS
                        logger.warning(f'[{self.name}] Forced finish failed, terminated as max_iterations')
                else:
                    self.exit_status = TerminationReason.MAX_ITERATIONS
        except Exception as e:
            self.exit_status = TerminationReason.UNHANDLED_ERROR
            logger.exception(f'Agent failed with unexpected error: {e}')

        return self.get_final_answer()

    async def force_finish_once(self) -> bool:
        """Try one final constrained step that should end with finish tool."""
        self.add_message('user', FORCE_FINISH_PROMPT)
        await self.step()

        if self.exit_status == TerminationReason.SUCCESS:
            return True

        logger.warning(f'[{self.name}] Forced finish step did not reach SUCCESS')
        return False

    async def ask_llm(self, **kwargs):
        """Query LLM; in inference mode retries on malformed tool-call output."""
        if self.is_training:
            return await super().ask_llm(**kwargs)

        max_retries = max(self.max_llm_retries, 0)
        message = None
        for attempt in range(max_retries + 1):
            message = await super().ask_llm(**kwargs)
            if self.exit_status is not None:
                return message

            retry_reason = get_llm_retry_reason(message)
            if retry_reason is None or attempt >= max_retries:
                return message

            logger.warning(
                f'🎯 [{self.name}] Tool-call parse failure ({retry_reason}), retrying ({attempt + 1}/{max_retries})'
            )

        return message

    async def execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> None:
        """Execute a single tool call via the tool executor."""
        if not self.executor:
            raise RuntimeError('No tool executor configured')

        name = tool_call.function.name
        if name not in self.executor._tools and name not in self.executor._schema_name_to_tool_name():
            error = f'Tool {name} does not exists.'
            self._handle_tool_error(error, tool_call, TerminationReason.FORMAT_ERROR)
            return

        # Handle arguments: can be str or dict
        raw_args = tool_call.function.arguments

        # Parse arguments (format already validated in ask_llm)
        try:
            args = parse_func_arguments(raw_args)
        except ValueError as e:
            # Format error: LLM returned invalid JSON arguments
            error = f'An error occurred when calling tool `{name}`:\nValueError: Argument parse error: {e}\n'
            self._handle_tool_error(error, tool_call, TerminationReason.FORMAT_ERROR)
            return

        # Execute tool: catch runtime errors
        try:
            response = await self.executor.call(name, args)
        except Exception as e:
            # Execution error: tool call failed at runtime
            exception_type = type(e).__name__
            exception_message = str(e)
            traceback_info = ''.join(traceback.format_tb(e.__traceback__))
            error = (
                f'An error occurred when calling tool `{name}`:\n'
                f'{exception_type}: {exception_message}\n'
                f'Traceback:\n{traceback_info}'
            )
            self._handle_tool_error(error, tool_call, TerminationReason.ENV_ERROR)
            return

        # Check if this tool signals completion
        if name in self.finish_tool_names:
            logger.info(f"🏁 [{self.name}] Completed via '{name}'")
            self.exit_status = TerminationReason.SUCCESS

        # Hook: process side effects and format output
        result = self.on_tool_executed(response)
        logger.debug(f'🎯 [{self.name}] Tool response from {name}: {str(extract_text_from_content(response.output))}')
        self.add_message('tool', result, tool_call_id=tool_call.id, name=name)

    def on_tool_executed(self, response: ToolResponse) -> MessageContent:
        """Hook after tool execution: process side effects and return formatted output.

        Override in subclasses to:
        - Handle context_updates
        - Format output for LLM context

        Args:
            response: The tool response

        Returns:
            Formatted output to add to memory
        """
        return response.output


@agent_registry.register('mm_search_agent')
async def run_mm_search_agent(agent_input: SearchAgentInput) -> SearchAgentOutput:
    """Search agent entry point."""
    if not agent_input.system_prompt:
        agent_input.system_prompt = get_default_system_prompt(agent_input.is_training)

    executor = ToolExecutor(_resolve_tools(agent_input.required_tools))
    agent = SearchAgent.create(agent_input, executor=executor)
    await agent.run(agent_input.user_prompt)
    return agent.get_output()
