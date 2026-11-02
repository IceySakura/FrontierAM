"""Explicit macOSWorld client adapters; never modify shared SDK configuration."""

from __future__ import annotations

import copy

from dataclasses import replace
from typing import Any

from nexus.core import OpenAIConfig, openai_client_registry
from nexus.extensions.openai_clients.anthropic_messages import AnthropicMessagesClient
from nexus.extensions.openai_clients.infer_proxy import InferProxyClient
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.extensions.openai_clients.vertex import _build_payload, convert_tools_to_anthropic


def _platform_config(config: OpenAIConfig) -> OpenAIConfig:
    return replace(config, client_args={k: v for k, v in config.client_args.items() if k != 'request_id_key'})


@openai_client_registry.register('macosworld_openai')
class MacOSWorldOpenAIClient(OpenAIClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(_platform_config(config))


@openai_client_registry.register('macosworld_openai_responses')
class MacOSWorldResponsesClient(OpenAIResponsesClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(_platform_config(config))


@openai_client_registry.register('macosworld_infer_proxy')
class MacOSWorldInferProxyClient(InferProxyClient):
    def __init__(self, config: OpenAIConfig):
        self.normalize_image_detail = config.client_args.get('normalize_image_detail', False)
        config = _platform_config(config)
        super().__init__(
            replace(config, client_args={k: v for k, v in config.client_args.items() if k != 'normalize_image_detail'})
        )

    def _apply_enforced_request_args(self, request_args) -> dict[str, Any]:
        result = super()._apply_enforced_request_args(request_args)
        if self.normalize_image_detail and 'messages' in result:
            result = copy.deepcopy(result)
            for message in result['messages']:
                content = message.get('content')
                for part in content if isinstance(content, list) else []:
                    image = part.get('image_url') if isinstance(part, dict) else None
                    if isinstance(image, dict) and image.get('detail') == 'original':
                        image.pop('detail')
        return result


@openai_client_registry.register('macosworld_anthropic_messages')
class MacOSWorldAnthropicClient(AnthropicMessagesClient):
    def _build_request_payload(self, *, messages, request_args):
        payload = _build_payload(
            messages=messages, request_args={k: v for k, v in request_args.items() if k != 'tools'}
        )
        tools = []
        for tool in request_args.get('tools') or []:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get('type', '')
            if not tool_type or tool_type == 'function':
                tools.extend(convert_tools_to_anthropic([tool]) or [])
                continue
            if tool_type not in {
                'bash_20250124',
                'computer_20250124',
                'computer_20251124',
                'computer_toolset_20260801',
            }:
                raise ValueError(f'Unsupported Anthropic native tool type: {tool_type}')
            flattened = copy.deepcopy(tool)
            if 'function' in flattened:
                function = flattened.pop('function')
                if not isinstance(function, dict):
                    raise ValueError(f'Anthropic native tool {tool_type} requires a function object')
                if 'name' not in flattened and function.get('name'):
                    flattened['name'] = function['name']
                for key, value in (function.get('parameters') or {}).items():
                    flattened.setdefault(key, value)
            tools.append(flattened)
        if tools:
            payload['tools'] = tools
        return payload
