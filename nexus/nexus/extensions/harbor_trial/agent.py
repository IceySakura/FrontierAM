"""Harbor Agent adapter delegating each step to the configured Nexus Agent."""

from __future__ import annotations

from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from nexus.core.tool import MCPServerSpec
from nexus.extensions.agents.installed_common.execution import DirectExecutionTarget, agent_execution_target
from nexus.extensions.harbor_trial.context import require_harbor_trial_binding
from nexus.runtime.context import execution_user


_MCP_TRANSPORT_ALIASES = {
    'stdio': 'stdio',
    'sse': 'sse',
    'streamable-http': 'streamable_http',
}


def _nexus_mcp_servers(servers: list[Any], *, agent_name: str) -> list[dict[str, Any]]:
    """Translate Harbor MCP declarations into the selected Nexus Agent contract."""
    normalized: list[dict[str, Any]] = []
    for server in servers:
        raw = server.model_dump(mode='json')
        transport = _MCP_TRANSPORT_ALIASES[raw['transport']]
        spec = MCPServerSpec.model_validate({
            'name': raw['name'],
            'client': {
                'transport': transport,
                'url': raw.get('url'),
                'command': raw.get('command'),
                'args': raw.get('args', []),
            },
        })
        payload = spec.model_dump(mode='json', exclude_none=True)
        if agent_name == 'codex':
            client = payload['client']
            if client['transport'] != 'stdio':
                raise ValueError('Codex supports only stdio MCP servers in Harbor Trial tasks')
            payload = {'name': payload['name'], **client}
            payload.pop('transport', None)
        normalized.append(payload)
    return normalized


class NexusAgent(BaseAgent):
    """Expose Nexus's configured Agent runner through Harbor's Agent contract."""

    capabilities = AgentCapabilities(resume=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._binding = require_harbor_trial_binding()
        self._run_context: AgentContext | None = None
        self._agent_output: dict[str, Any] | None = None

    @staticmethod
    def name() -> str:
        return 'nexus'

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._run_context = context
        self._agent_output = None
        workdir = environment.task_env_config.workdir
        if not workdir:
            result = await environment.exec('pwd')
            if result.return_code != 0:
                raise RuntimeError(f'Failed to resolve Agent working directory: {result.stderr or result.stdout}')
            workdir = (result.stdout or '/').strip()

        nexus_context = self._binding.execution_context
        nexus_context.task_config['working_dir'] = workdir
        nexus_context.tool_state['working_dir'] = workdir

        kwargs: dict[str, Any] = {
            'user_prompt': instruction,
            'working_dir': workdir,
        }
        target_factory = getattr(environment, 'agent_execution_target', None)
        target = target_factory(environment.default_user) if target_factory is not None else None
        if target is None and environment.default_user is not None:
            target = DirectExecutionTarget(user=environment.default_user)
        agent_info = getattr(self._binding.run_agent, 'nexus_meta', {})
        agent_name = agent_info.get('name', '<unknown>')
        if target is not None and not agent_info.get('meta', {}).get('supports_execution_target', False):
            raise ValueError(f'Agent {agent_name!r} does not support the selected execution target')
        if (
            target is not None
            and target.user is not None
            and not agent_info.get('meta', {}).get('supports_execution_user', False)
        ):
            raise ValueError(f'Agent {agent_name!r} does not support complete execution-user setup and execution')
        if self.mcp_servers:
            kwargs['mcp_servers'] = _nexus_mcp_servers(self.mcp_servers, agent_name=agent_name)
        if self.skills_dir:
            if agent_name != 'claude_code':
                raise ValueError(f'Agent {agent_name!r} does not support Harbor skills_dir')
            kwargs['skills_dir'] = self.skills_dir

        kwargs.update(self._agent_run_kwargs())
        with agent_execution_target(target), execution_user(None):
            output = await self._binding.run_agent(**kwargs)
        # Harbor only invokes post-run processing for an empty AgentContext.
        self._agent_output = output.model_dump(mode='json')

    def _agent_run_kwargs(self) -> dict[str, Any]:
        return {}

    def populate_context_post_run(self, context: AgentContext) -> None:
        if context is self._run_context and self._agent_output is not None:
            context.metadata = {
                **(context.metadata or {}),
                'nexus_agent_output': self._agent_output,
            }


class NexusCodexAgent(NexusAgent):
    """Adapt Codex native logs through Harbor's post-run lifecycle."""

    def _agent_run_kwargs(self) -> dict[str, Any]:
        return {
            'logs_dir': self.environment_logs_dir.as_posix(),  # ty: ignore[unresolved-attribute]
            'codex_home_name': None,
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        from harbor.agents.installed.codex import Codex

        if context is not self._run_context:
            return
        Codex(logs_dir=self.logs_dir, model_name=self.model_name, logger=self.logger).populate_context_post_run(context)
        super().populate_context_post_run(context)


__all__ = ['NexusAgent', 'NexusCodexAgent']
