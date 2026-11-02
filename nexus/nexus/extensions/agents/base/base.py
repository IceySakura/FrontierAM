"""Base agent class using plain OpenAI-compatible message dicts.

Follows the same pattern as science_agent and demo_agent: messages are stored
as list[dict] — no custom Message/Memory wrappers. This aligns directly with
Nexus core's OpenAICompatibleClient.create_completion() interface.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.schema import (
    BaseAgentOutput,
    MessageContent,
    TerminationReason,
)


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Messages are stored as plain dicts (OpenAI-compatible format), consistent
    with Nexus core's client interface. No custom Message/Memory wrappers.

    Subclasses must implement the `step` method to define agent-specific behavior.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    # Core attributes
    name: str = Field(..., description='Unique name of the agent')
    description: str | None = Field(default=None, description='Optional agent description')
    system_prompt: str | None = Field(default=None, description='The system prompt for the agent')

    messages: list[dict] = Field(default_factory=list, description='Conversation history as OpenAI-compatible dicts')
    exit_status: TerminationReason | None = Field(default=None, description='Exit reason; None means still running')

    # Execution control
    max_iterations: int = Field(default=30, description='Maximum iterations before termination')
    current_iteration: int = Field(default=0, description='Current iteration in execution')

    def add_message(self, role: str, content: MessageContent | None = None, **kwargs: Any) -> None:
        """Append an OpenAI-compatible message dict to conversation history."""
        msg: dict[str, Any] = {'role': role}
        if content is not None:
            msg['content'] = content
        for k, v in kwargs.items():
            if v is not None:
                msg[k] = v
        self.messages.append(msg)

    async def run(self, request: MessageContent | None = None) -> MessageContent | None:
        """Execute the agent's main loop and return result."""
        if self.exit_status is not None:
            raise RuntimeError(f'Cannot run agent: already exited with {self.exit_status}')

        # Reset for this run
        self.current_iteration = 0

        if request is not None:
            self.add_message('user', request)

        try:
            # Main loop: exit when exit_status is set (by step/think) or max_iterations reached
            while self.current_iteration < self.max_iterations and self.exit_status is None:
                self.current_iteration += 1
                logger.debug(f'Executing step {self.current_iteration}/{self.max_iterations}')
                await self.step()

            # Check if we hit max iterations
            if self.exit_status is None and self.current_iteration >= self.max_iterations:
                self.exit_status = TerminationReason.MAX_ITERATIONS
                logger.warning(f'Terminated: Reached max steps ({self.max_iterations})')

        except Exception as e:
            # Catch any unexpected error and mark as unknown_error
            self.exit_status = TerminationReason.UNHANDLED_ERROR
            logger.exception(f'Agent failed with unexpected error: {e}')

        return self.get_final_answer()

    def get_final_answer(self) -> MessageContent | None:
        """Get the final result of the agent execution."""
        for msg in reversed(self.messages):
            if msg.get('role') == 'assistant' and msg.get('content') is not None:
                return msg['content']
        return None

    def get_output(self) -> BaseAgentOutput:
        """Get the final agent output with trajectory and final answer.

        Returns:
            BaseAgentOutput: Output containing main_trajectory, final_answer, and exit_status.
        """
        return BaseAgentOutput(
            main_trajectory=self.main_trajectory,
            final_answer=self.get_final_answer(),
            exit_status=self.exit_status or TerminationReason.SUCCESS,
        )

    @abstractmethod
    async def step(self) -> Any:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    @property
    def main_trajectory(self) -> list[dict]:
        """Return main trajectory with system message + agent trajectory."""
        if self.system_prompt:
            return [{'role': 'system', 'content': self.system_prompt}] + self.messages
        return list(self.messages)
