"""Agent-specific schema types.

Only defines types unique to the agent framework:
- TerminationReason, ToolChoice: agent execution control
- BaseAgentInput/Output, ToolCallAgentInput/Output: agent I/O contracts

For LLM message types, use nexus.core.types (ExtendedMessage, etc.).
For tool execution, use nexus.core (ToolExecutor, ToolResponse, etc.).
Messages in agent history are stored as plain OpenAI-compatible dicts.
"""

from enum import Enum
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import Field

from nexus.core import AgentRunInput, AgentRunOutput, WithMainTrajectory, WithUserPrompt


MessageContent = str | list[ChatCompletionContentPartParam | dict]


class TerminationReason(str, Enum):
    SUCCESS = 'SUCCESS'
    FORMAT_ERROR = 'FORMAT_ERROR'
    ENV_ERROR = 'ENV_ERROR'
    PROVIDER_ERROR = 'PROVIDER_ERROR'
    UNHANDLED_ERROR = 'UNHANDLED_ERROR'
    TRUNCATION = 'TRUNCATION'
    OUTPUT_LENGTH_EXCEEDED = 'OUTPUT_LENGTH_EXCEEDED'
    MAX_ITERATIONS = 'MAX_STEPS'  # value kept as 'MAX_STEPS' for hytuner compatibility
    TASK_TIMEOUT = 'TASK_TIMEOUT'
    ENV_DONE = 'ENV_DONE'


class ToolChoice(str, Enum):
    """Tool choice strategy for LLM function calling."""

    NONE = 'none'
    AUTO = 'auto'
    REQUIRED = 'required'


TOOL_CHOICE_TYPE = ToolChoice | Literal['none', 'auto', 'required']


class BaseAgentInput(AgentRunInput, WithUserPrompt):
    """Base input for class-based agents."""

    max_iterations: int = Field(default=30, description='Maximum iterations before termination')
    system_prompt: str | None = Field(default=None, description='Optional system prompt override')


class BaseAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Base output with conversation trajectory and final answer."""

    final_answer: MessageContent | None = Field(default=None, description='The final answer from the agent')
    exit_status: TerminationReason = Field(
        default=TerminationReason.SUCCESS, description='Why the agent stopped executing'
    )


class ToolCallAgentInput(BaseAgentInput):
    """Input for ToolCallAgent with tool-calling configuration."""

    tools: list[str] | None = Field(default=None, description='Tool names to use')
    tool_choice: TOOL_CHOICE_TYPE = Field(default=ToolChoice.AUTO, description='Tool choice mode')
    finish_tool_names: list[str] = Field(default_factory=lambda: ['finish'], description='Tools that signal completion')


class ToolCallAgentOutput(BaseAgentOutput):
    """Output for ToolCallAgent."""
