import traceback

from pydantic import Field

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.schema import (
    BaseAgentInput,
    BaseAgentOutput,
    TerminationReason,
)


class Hy3SearchAgentOutput(BaseAgentOutput):
    """hy response api Search agent output."""


class Hy3SearchAgentInput(BaseAgentInput):
    """hy response api Search agent input."""

    tools: list[str] | None = Field(default=None, description='Tool names to use')
    history: list[dict[str, str]] = Field(default_factory=list)


@agent_registry.register('hy3_search')
async def run_hy3_search(agent_input: Hy3SearchAgentInput) -> Hy3SearchAgentOutput:
    logger.debug(f'enter run_hy3_search, input: {agent_input.model_dump_json()}')
    ctx = require_context()
    client = ctx.openai_client
    input_messages = []
    input_messages.extend(agent_input.history)
    input_messages.append({'role': 'user', 'content': agent_input.user_prompt})
    tools = []
    if agent_input.tools:
        tools = [{'type': tool_name} for tool_name in agent_input.tools]
    try:
        response = await client.create_completion(messages=input_messages, tools=tools)
        response_text = response.choices[0].message.content or ''
        input_messages.append({'role': 'assistant', 'content': response_text})
        return Hy3SearchAgentOutput(
            main_trajectory=input_messages,
            final_answer=response_text,
            exit_status=TerminationReason.SUCCESS if response_text != '' else TerminationReason.UNHANDLED_ERROR,
        )
    except Exception as e:
        traceback.print_exc()
        logger.error(f'request error, messages: {input_messages}, error: {e}')
        raise e
