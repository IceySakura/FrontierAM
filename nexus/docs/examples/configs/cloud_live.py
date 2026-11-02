"""A Cloud Live example that accepts two inputs without a model or runtime."""

from nexus.core import AgentRunInput, AgentRunOutput, UserInputRequest, WithInputProvider, agent_registry


class CollectLiveInputs(AgentRunInput, WithInputProvider):
    pass


class CollectedLiveInputs(AgentRunOutput):
    messages: list[str]


@agent_registry.register('cloud_live_collect_inputs')
async def collect_live_inputs(agent_input: CollectLiveInputs) -> CollectedLiveInputs:
    messages: list[str] = []
    for turn in range(1, 3):
        text = await agent_input.get_user_input(UserInputRequest(hint=f'Input {turn}/2'))
        if text is None:
            break
        messages.append(text)
    return CollectedLiveInputs(messages=messages)
