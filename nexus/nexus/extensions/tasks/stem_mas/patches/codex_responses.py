"""stem_mas-local Codex Responses converter patch."""

import json
import logging
import os

from typing import Any, TypeGuard

from nexus.core import agent_registry
from nexus.extensions.agents.codex.agent import CodexInput, CodexOutput
from nexus.extensions.agents.installed_common import call_runtime_agent, coerce_user_prompt
from nexus.runtime.server.services.responses_proxy import converter as _base_converter


logger = logging.getLogger(__name__)


class McpNamespaceToolCodec:
    """Translate Codex namespace tools through OpenAI-compatible function tools."""

    tool_call_types = frozenset({'function_call', 'custom_tool_call'})
    tool_prefix = 'mcp__'
    tool_separator = '__'
    allowed_tools_env = 'STEM_MAS_CODEX_ALLOWED_NAMESPACE_TOOLS_JSON'

    def convert_request_body(self, request_body: dict[str, Any]) -> dict[str, Any]:
        converted_body = dict(request_body)

        tools = self.flatten_namespace_tools(request_body.get('tools'))
        if tools is not request_body.get('tools'):
            converted_body['tools'] = tools

        raw_input = request_body.get('input')
        if isinstance(raw_input, list):
            converted_body['input'] = self.flatten_namespaced_input(raw_input)

        return converted_body

    def restore_response_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in events:
            item = event.get('item')
            if not self.is_tool_call_item(item):
                continue

            split_name = self.split_tool_name(item.get('name'))
            if split_name is None:
                continue

            namespace, tool_name = split_name
            item['namespace'] = self.to_responses_namespace(namespace)
            item['name'] = tool_name

        return events

    def responses_to_openai(
        self,
        request_body: dict[str, Any],
        *,
        collect_custom_tools: set[str] | None = None,
        collect_tool_name_mappings: dict[str, _base_converter.ToolIdentity] | None = None,
    ) -> dict[str, Any]:
        converted_body = self.convert_request_body(request_body)
        return _base_converter.responses_to_openai(
            converted_body,
            collect_custom_tools=collect_custom_tools,
            collect_tool_name_mappings=collect_tool_name_mappings,
        )

    def openai_to_responses_events(
        self,
        openai_response: dict[str, Any],
        *,
        custom_tool_names: set[str] | None = None,
        tool_name_mappings: dict[str, _base_converter.ToolIdentity] | None = None,
    ) -> list[dict[str, Any]]:
        events = _base_converter.openai_to_responses_events(
            openai_response,
            custom_tool_names=custom_tool_names,
            tool_name_mappings=tool_name_mappings,
        )
        return self.restore_response_events(events)

    def flatten_namespace_tools(self, tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools

        converted_tools: list[Any] = []
        changed = False
        for tool in tools:
            if not isinstance(tool, dict) or tool.get('type') != 'namespace':
                converted_tools.append(tool)
                continue

            changed = True
            namespace = str(tool.get('name') or '')
            if not namespace:
                continue
            namespace_description = str(tool.get('description') or '')
            allowed_tools = self.allowed_namespace_tools(namespace)
            for nested_tool in tool.get('tools') or []:
                if not isinstance(nested_tool, dict) or nested_tool.get('type') != 'function':
                    continue
                nested_name = str(nested_tool.get('name') or '')
                if not nested_name:
                    continue
                if allowed_tools is not None and nested_name not in allowed_tools:
                    continue

                flattened_tool = dict(nested_tool)
                flattened_tool['name'] = self.join_tool_name(namespace, nested_name)
                if namespace_description and allowed_tools is None:
                    description = str(flattened_tool.get('description') or '')
                    flattened_tool['description'] = f'{description}\n\n{namespace_description}'.strip()
                converted_tools.append(flattened_tool)

        return converted_tools if changed else tools

    def flatten_namespaced_input(self, raw_input: list[Any]) -> list[Any]:
        converted_input: list[Any] = []
        for item in raw_input:
            if not self.is_tool_call_item(item):
                converted_input.append(item)
                continue

            namespace = item.get('namespace')
            if not namespace:
                converted_input.append(item)
                continue
            if not str(namespace).startswith(self.tool_prefix):
                # The base Responses converter owns Codex-native namespaces such as
                # ``collaboration``. This codec only rewrites the MCP namespace convention.
                converted_input.append(item)
                continue

            converted_item = dict(item)
            converted_item['name'] = self.join_tool_name(str(namespace), str(item.get('name') or ''))
            converted_item.pop('namespace', None)
            converted_input.append(converted_item)

        return converted_input

    def join_tool_name(self, namespace: str, tool_name: str) -> str:
        namespace = self.normalize_namespace(namespace)
        return f'{self.tool_prefix}{namespace}{self.tool_separator}{tool_name}'

    def split_tool_name(self, name: Any) -> tuple[str, str] | None:
        if not isinstance(name, str) or not name.startswith(self.tool_prefix):
            return None

        namespace, separator, tool_name = name.removeprefix(self.tool_prefix).partition(self.tool_separator)
        if not separator or not namespace or not tool_name:
            return None
        return namespace, tool_name

    def normalize_namespace(self, namespace: str) -> str:
        return namespace.removeprefix(self.tool_prefix).removesuffix(self.tool_separator)

    def to_responses_namespace(self, namespace: str) -> str:
        normalized = self.normalize_namespace(namespace)
        return f'{self.tool_prefix}{normalized}{self.tool_separator}'

    def is_tool_call_item(self, item: Any) -> TypeGuard[dict[str, Any]]:
        return isinstance(item, dict) and item.get('type') in self.tool_call_types

    def allowed_namespace_tools(self, namespace: str) -> set[str] | None:
        raw_config = os.environ.get(self.allowed_tools_env)
        if not raw_config:
            return None

        try:
            parsed = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{self.allowed_tools_env} must contain valid JSON.') from exc

        if not isinstance(parsed, dict):
            raise ValueError(f'{self.allowed_tools_env} must contain a JSON object.')

        normalized_namespace = self.normalize_namespace(namespace)
        raw_tools = parsed.get(normalized_namespace)
        if raw_tools is None:
            return set()
        if not isinstance(raw_tools, list):
            raise ValueError(f'{self.allowed_tools_env}[{normalized_namespace!r}] must be a list.')
        if not all(isinstance(tool_name, str) and tool_name.strip() for tool_name in raw_tools):
            raise ValueError(f'{self.allowed_tools_env}[{normalized_namespace!r}] must contain non-empty strings.')
        return {tool_name.strip() for tool_name in raw_tools}


_NAMESPACE_CODEC = McpNamespaceToolCodec()


async def setup_and_run_stem_mas_codex(*args, **kwargs):
    """Run the shared Codex runtime with the stem_mas Responses converter."""
    from nexus.extensions.agents.codex.runtime_setup import setup_and_run_codex

    kwargs['responses_to_openai_converter'] = _NAMESPACE_CODEC.responses_to_openai
    kwargs['openai_to_responses_events_converter'] = _NAMESPACE_CODEC.openai_to_responses_events
    result = await setup_and_run_codex(*args, **kwargs)
    if result.get('success') and not str(result.get('output') or '').strip():
        result = dict(result)
        result['success'] = False
        result['failure_kind'] = 'agent'
        result['failure_stage'] = 'process'
        stderr = str(result.get('stderr') or '').rstrip()
        result['stderr'] = f'{stderr}\nCodex produced no final message.'.strip()
    return result


@agent_registry.register('stem_mas_codex')
async def run_stem_mas_codex(input_data: CodexInput) -> CodexOutput:
    """Run Codex with the stem_mas-only Responses converter patch."""
    task = coerce_user_prompt(input_data.user_prompt)

    logger.debug(f'Starting stem_mas Codex agent for task: {task[:100]}...')

    result = await call_runtime_agent(
        setup_and_run_stem_mas_codex,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        model=input_data.model,
        npm_registry=input_data.npm_registry,
        extra_cli_args=list(input_data.extra_cli_args),
        developer_instructions=input_data.developer_instructions,
        extra_config_toml=input_data.extra_config_toml,
        use_azure_provider=input_data.use_azure_provider,
        auto_compact_token_limit=input_data.auto_compact_token_limit,
    )

    logger.info(f'stem_mas Codex execution completed: success={result.get("success")}')
    success = bool(result.get('success', False))
    timed_out = bool(result.get('timed_out', False))
    output = result.get('output', '')
    exit_status = 'timeout' if timed_out else ('completed' if success else 'error')
    return CodexOutput(
        success=success,
        output=output,
        final_answer=output.strip()
        if exit_status == 'completed' and isinstance(output, str) and output.strip()
        else None,
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=result.get('failure_kind'),
        failure_stage=result.get('failure_stage'),
        exit_status=exit_status,
    )


__all__ = [
    'McpNamespaceToolCodec',
    'run_stem_mas_codex',
    'setup_and_run_stem_mas_codex',
]
