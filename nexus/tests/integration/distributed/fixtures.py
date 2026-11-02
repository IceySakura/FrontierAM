import asyncio

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    UserInputRequest,
    WithInputProvider,
    agent_registry,
    task_registry,
)
from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


class DistributedSuccessInput(TaskRunInput):
    message: str = 'ok'


class DistributedSuccessOutput(TaskRunOutput):
    message: str


@task_registry.register('distributed_success')
async def distributed_success_task(task_input: DistributedSuccessInput) -> DistributedSuccessOutput:
    return DistributedSuccessOutput(message=task_input.message)


class DistributedSleepEchoInput(TaskRunInput):
    index: int
    message: str
    sleep_seconds: float = Field(default=20, ge=0)


class DistributedSleepEchoOutput(TaskRunOutput):
    index: int
    message: str


@task_registry.register('distributed_sleep_echo')
async def distributed_sleep_echo_task(task_input: DistributedSleepEchoInput) -> DistributedSleepEchoOutput:
    await asyncio.sleep(task_input.sleep_seconds)
    return DistributedSleepEchoOutput(index=task_input.index, message=task_input.message)


class DistributedInteractiveAgentInput(AgentRunInput, WithInputProvider):
    prompt: str = Field(default='choose')


class DistributedInteractiveAgentOutput(AgentRunOutput):
    choice: str | None


@agent_registry.register('distributed_interactive_agent')
async def distributed_interactive_agent(
    agent_input: DistributedInteractiveAgentInput,
) -> DistributedInteractiveAgentOutput:
    choice = await agent_input.get_user_input(UserInputRequest(hint=agent_input.prompt))
    return DistributedInteractiveAgentOutput(choice=choice)


@runtime_provider_registry.register('distributed_fake_runtime')
class DistributedFakeRuntimeProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        import os

        params = self.get_params(name, **params)
        base_url = (
            params.get('base_url') or self.config.get('base_url') or os.environ['NEXUS_DIST_FAKE_RUNTIME_BASE_URL']
        )
        runtime = DistributedFakeRuntime(base_url=base_url)
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self._active_runtimes.pop(name, None)


class DistributedFakeRuntime(Runtime):
    def __init__(self, *, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def _invoke(self, request):
        raise NotImplementedError('distributed fake runtime only supports proxy tests')

    async def ws_connect(self) -> None:
        return None

    def on(self, event, listener):
        return lambda: None


@runtime_provider_registry.register('distributed_remote_runtime')
class DistributedRemoteRuntimeProvider(RuntimeProvider):
    """Connect deterministic distributed E2E jobs to a real in-cluster Runtime server."""

    def __init__(self, *, base_url: str, **config):
        super().__init__(**config)
        self.base_url = base_url

    async def acquire(self, name: str, **params):
        runtime = RemoteRuntime(
            base_url=self.get_params(name, **params).get('base_url', self.base_url),
            auto_connect_ws=False,
            enable_runtime_logs=False,
        )
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        runtime = self._active_runtimes.pop(name, None)
        if isinstance(runtime, RemoteRuntime):
            await runtime.ws_disconnect()
