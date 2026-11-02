from pydantic import Field

from nexus.core import ToolExecutor, ToolResponse, agent_registry, tool_registry
from nexus.extensions.agents.base.schema import MessageContent, ToolCallAgentInput, ToolCallAgentOutput
from nexus.extensions.agents.base.toolcall import ToolCallAgent
from nexus.extensions.agents.rondo.prompt import (
    NEXT_STEP_NO_OUTPUT_TEMPLATE,
    NEXT_STEP_TEMPLATE,
    SYSTEM_PROMPT,
)
from nexus.extensions.agents.swe_agent.tools import bash_tool as bash_tool
from nexus.extensions.agents.swe_agent.tools import str_replace_editor_tool as str_replace_editor_tool


class RondoAgentInput(ToolCallAgentInput):
    """Input configuration for RondoAgent."""

    working_dir: str = Field(default='/workspace', description='Initial working directory')


class RondoAgentOutput(ToolCallAgentOutput):
    """Output from RondoAgent run."""

    working_dir: str = Field(default='/workspace', description='Final working directory')
    open_file: str | None = Field(default=None, description='Currently open file')
    todos: list[dict] = Field(default_factory=list, description='Final todo state')


class RondoAgent(ToolCallAgent):
    """RondoAgent - A general-purpose agent with state tracking for complex tasks."""

    name: str = 'rondo'
    system_prompt: str = SYSTEM_PROMPT

    # Environment state tracking
    working_dir: str = Field(default='/workspace', description='Current working directory')
    open_file: str | None = Field(default=None, description='Currently open file')
    todos: list[dict] = Field(default_factory=list, description='Final todo state')

    def on_tool_executed(self, response: ToolResponse) -> MessageContent:
        """
        Handle tool execution. Updates state and formats the next observation.
        """
        updates = response.context_updates or {}
        self.working_dir = updates.get('working_dir', self.working_dir)
        self.todos = updates.get('todos', self.todos)
        self.open_file = updates.get('open_file', self.open_file)
        active_todo = updates.get('active_todo', None)

        observation = response.output[0].get('text', '') if response.output else ''

        context = {
            'observation': observation,
            'working_dir': self.working_dir,
            'active_todo': active_todo,
            'open_file': self.open_file or 'None',
        }

        template = NEXT_STEP_TEMPLATE if observation else NEXT_STEP_NO_OUTPUT_TEMPLATE
        return template.format(**context)

    def get_output(self) -> RondoAgentOutput:
        """Get the final agent output with trajectory, final answer, and environment state."""
        return RondoAgentOutput(
            main_trajectory=self.main_trajectory,
            final_answer=self.get_final_answer(),
            todos=self.todos,
            working_dir=self.working_dir,
            open_file=self.open_file,
        )


@agent_registry.register('rondo_agent')
async def run_rondo_agent(agent_input: RondoAgentInput) -> RondoAgentOutput:
    """Entry point for RondoAgent."""
    tools = [
        tool_registry.get('swe_agent/bash'),
        tool_registry.get('swe_agent/str_replace_editor'),
        tool_registry.get('base/write_todos'),
        tool_registry.get('google_search'),
        tool_registry.get('python_code_interpreter'),
        tool_registry.get('visit'),
    ]
    # add finish tool if it is in the special tool names
    if 'finish' in agent_input.finish_tool_names:
        tools.append(tool_registry.get('base/finish'))

    executor = ToolExecutor(tools)

    agent = RondoAgent.create(agent_input, executor)
    await agent.run(agent_input.user_prompt)
    return agent.get_output()
